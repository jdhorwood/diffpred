import typer
from pathlib import Path
from typing import Annotated, Optional
import numpy as np
import pandas as pd
from functools import partial
from diffpred.utils.ops import is_array_1d
from diffpred.uq.predictive_dist import get_test_samples, get_mc_dropout_samples, get_baseline_preds_with_uncertainty
from diffpred.trainers import (
    DiffusionTrainer,
    MVETrainer,
    MCDropoutTrainer,
    DeepEnsembleTrainer,
)
from diffpred.utils.checkpoint_reload import RunReconstructor
from diffpred.eval_ckpt import _discover_run_dirs
from diffpred.utils.reloader import resolve_uncertainty_method, extract_tag_from_run_path

app = typer.Typer()

def _check_input_shapes(y_true: np.ndarray, y_pred_samples: np.ndarray):
    if not is_array_1d(y_true):
        raise ValueError("y_true must be a 1D array.")
    if y_pred_samples.ndim not in [2, 3]:
        raise ValueError("y_pred_samples must be a 2D or 3D array.")
    if y_true.shape[0] != y_pred_samples.shape[1]:
        raise ValueError("The number of data samples in y_true and y_pred_samples must match.")
    
    if y_pred_samples.ndim == 3 and not y_pred_samples.shape[2] == 1:
        raise ValueError("If y_pred_samples is a 3D array, the last dimension must be 1 (target_dim=1).")
    if y_pred_samples.ndim == 3:
        y_pred_samples = y_pred_samples.squeeze(-1)
    if y_true.ndim == 2:
        y_true = y_true.squeeze()

    return y_true, y_pred_samples
    
def picp(y_true: np.ndarray, y_pred_samples: np.ndarray, lower_quantile: float, upper_quantile: float) -> np.floating:
    """
    Prediction Interval Coverage Probability (PICP)

    Parameters
    ----------
    y_true : np.ndarray of shape (N,) or (N, 1)
        True target values.
    y_pred_samples : np.ndarray of shape (prediction_samples, N) or (prediction_samples, N, 1)
        Predicted samples from the predictive distribution.
    lower_quantile : float
        The lower quantile for the prediction interval.
    upper_quantile : float
        The upper quantile for the prediction interval.

    Returns
    -------
    picp : np.floating
        The prediction interval coverage probability.
    """
    y_true, y_pred_samples = _check_input_shapes(y_true, y_pred_samples)
    if not (0 <= lower_quantile < upper_quantile <= 1):
        raise ValueError("Quantiles must satisfy 0 <= lower_quantile < upper_quantile <= 1.")
    
    y_pred_lower = np.quantile(y_pred_samples, lower_quantile, axis=0)
    y_pred_upper = np.quantile(y_pred_samples, upper_quantile, axis=0)
    picp_val = np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))
    return picp_val

def qice(y_true: np.ndarray, y_pred_samples: np.ndarray, num_quantiles: int = 10) -> np.floating:
    """
    Quantile Interval Coverage Error (QICE)

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,) or (n_samples, 1)
        True target values.
    y_pred_samples : np.ndarray of shape (n_samples, prediction_samples) or (n_samples, prediction_samples, 1)
        Predicted samples from the predictive distribution.

    Returns
    -------
    qice : np.floating
        The quantile interval coverage error.
    """

    if num_quantiles < 2:
        raise ValueError("num_quantiles must be at least 2 to compute QICE.")
    y_true, y_pred_samples = _check_input_shapes(y_true, y_pred_samples)

    quantiles = np.linspace(0, 1, num_quantiles + 1)
    coverage_errors = []
    for i in range(num_quantiles):
        lower_quantile = quantiles[i]
        upper_quantile = quantiles[i + 1]
        picp_value = picp(y_true, y_pred_samples, lower_quantile, upper_quantile)
        coverage_error = np.abs(picp_value - (upper_quantile - lower_quantile))
        coverage_errors.append(coverage_error)

    qice_val = np.mean(coverage_errors)
    return qice_val

class PredIntervalEvaluator:
    def __init__(self, model: DiffusionTrainer | MVETrainer | MCDropoutTrainer, num_quantiles: int = 10, lower_quantile: float = 0.025, upper_quantile: float = 0.975):
        self.model = model
        self.num_quantiles = num_quantiles
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def _get_samples_fn(self):
        if isinstance(self.model, MCDropoutTrainer):
            return get_mc_dropout_samples
        elif isinstance(self.model, (MVETrainer, DeepEnsembleTrainer)):
            return partial(get_baseline_preds_with_uncertainty, num_samples=1000, return_samples=True)
        else:
            return get_test_samples

    def get_pred_samples(self, datamodule, split: str = "test"):
        samples_fn = self._get_samples_fn()
        samples, targets = samples_fn(model=self.model, datamodule=datamodule, split=split)
        return samples, targets
    
    def evaluate(self, datamodule, split: str = "test") -> tuple[np.floating, np.floating]:
        samples, targets = self.get_pred_samples(datamodule, split=split)
        qice_val = qice(targets.numpy(), samples.numpy(), num_quantiles=self.num_quantiles)
        picp_val = picp(targets.numpy(), samples.numpy(), self.lower_quantile, self.upper_quantile)
        return picp_val, qice_val
    
def evaluate_prediction_intervals(run_dir):
    run_path = Path(run_dir)
    if not (run_path / ".hydra" / "config.yaml").exists():
            raise ValueError(f"Invalid --run-dir: {run_dir} (missing .hydra/config.yaml)")
    
    reconstructor = RunReconstructor(run_dir=run_path)
    datamodule = reconstructor.build_datamodule()
    model, _ = reconstructor.load_checkpointed_model(datamodule=datamodule, ckpt_path=None, use_ema=True)
    evaluator = PredIntervalEvaluator(model, num_quantiles=10, lower_quantile=0.025, upper_quantile=0.975)
    picp_val, qice_val = evaluator.evaluate(datamodule)
    return picp_val,qice_val

def save_prediction_metrics(output_dir, output_name, df):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / output_name, index=True)
    else:
        df.to_csv(output_name, index=True)


@app.command()
def main(
    run_dir: Annotated[Path, typer.Argument(help="Path to a single Hydra run directory containing .hydra/config.yaml.")],
    output_dir: Annotated[Optional[Path], typer.Option(help="Path to output directory.")] = None,
    ):

    picp_val, qice_val = evaluate_prediction_intervals(run_dir)
    print(f"PICP: {picp_val}")
    print(f"QICE: {qice_val}")
    df = pd.DataFrame({"PICP": [picp_val], "QICE": [qice_val]})
    save_prediction_metrics(output_dir, "prediction_interval_metrics.csv", df)


@app.command()
def eval_multi(multirun_dir: Annotated[Path, typer.Argument(help="Path to a multirun Hydra directory containing runs")],
    output_dir: Annotated[Optional[Path], typer.Option(help="Path to output directory.")] = None,
):
    def mean_with_std(x):
        return f"{np.mean(x):.3f} ± {np.std(x):.2f}"
    
    run_dirs = _discover_run_dirs(multirun_dir)
    results = []
    if not run_dirs:
        raise ValueError(f"No Hydra runs found under multirun directory: {multirun_dir}")
    for run_dir in run_dirs:
        picp_val, qice_val = evaluate_prediction_intervals(run_dir)
        uq_method = resolve_uncertainty_method(run_dir)
        dataset = extract_tag_from_run_path(run_dir, "data.dataset")
        seed = extract_tag_from_run_path(run_dir, "seed")
        results.append({"uq_method": uq_method, "dataset": dataset, "seed": seed, "PICP": picp_val, "QICE": qice_val})
    
    df = pd.DataFrame(results)
    picp_table = pd.pivot_table(df, values="PICP", index="uq_method", columns="dataset", aggfunc=mean_with_std)
    qice_table = pd.pivot_table(df, values="QICE", index="uq_method", columns="dataset", aggfunc=mean_with_std)
    save_prediction_metrics(output_dir, "prediction_interval_metrics.csv", df)
    save_prediction_metrics(output_dir, "picp_table.csv", picp_table)
    save_prediction_metrics(output_dir, "qice_table.csv", qice_table)


if __name__ == "__main__":
    app()



