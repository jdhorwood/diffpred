from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from joblib import Parallel, delayed, parallel
from torch_geometric.data import Data
from tqdm import tqdm
from diffpred.utils.checkpoint_reload import RunReconstructor
from diffpred.viz.multitask_imputation import plot_test_rmse_summary

app = typer.Typer(add_completion=False)


def infer_model_label(cfg) -> str:
    base_name = str(cfg.model.name)
    return base_name.upper()


def _resolve_dataset_name(cfg) -> str:
    data_cfg = cfg.data
    dataset_name = data_cfg.get("dataset_key", None)
    if dataset_name is None:
        dataset_name = data_cfg.get("dataset", None)
    if dataset_name is None:
        raise KeyError("Missing both data.dataset_key and data.dataset")
    return str(dataset_name)


def compute_prefixed_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_prefix: str,
) -> dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if y_true.size == 0:
        return {
            f"{metric_prefix}_rmse": float("nan"),
            f"{metric_prefix}_mae": float("nan"),
            f"n_{metric_prefix}_cells": 0,
        }

    abs_err = np.abs(y_pred - y_true)
    sq_err = np.square(y_pred - y_true)
    return {
        f"{metric_prefix}_rmse": float(np.sqrt(np.mean(sq_err))),
        f"{metric_prefix}_mae": float(np.mean(abs_err)),
        f"n_{metric_prefix}_cells": int(y_true.size),
    }


def summarize_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=["dataset", "ratio", "model", "n_seeds"])

    id_cols = {"dataset", "ratio", "model", "seed", "split_policy"}
    metric_cols = [
        c
        for c in raw_df.columns
        if c not in id_cols and pd.api.types.is_numeric_dtype(raw_df[c])
    ]

    agg_kwargs: dict[str, tuple[str, str]] = {"n_seeds": ("seed", "nunique")}
    for metric_col in metric_cols:
        agg_kwargs[f"{metric_col}_mean"] = (metric_col, "mean")
        agg_kwargs[f"{metric_col}_std"] = (metric_col, "std")

    grouped = (
        raw_df.groupby(["dataset", "ratio", "model"], as_index=False)
        .agg(**agg_kwargs)
        .sort_values(["dataset", "ratio", "model"])
        .reset_index(drop=True)
    )

    std_cols = [c for c in grouped.columns if c.endswith("_std")]
    if std_cols:
        grouped.loc[:, std_cols] = grouped.loc[:, std_cols].fillna(0.0)

    return grouped


def _predict_split_tensor(model, datamodule, split: str) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        import torch

        model_device = torch.device("cpu")

    import torch
    with torch.no_grad():
        dataloader = getattr(datamodule, f"{split}_dataloader")()
        for x, _ in dataloader:
            if isinstance(x, (torch.Tensor, Data)):
                x = x.to(model_device)
            if hasattr(model, "predict_samples"):
                out = model.predict_samples(x)
            elif hasattr(model, "predict"):
                out = model.predict(x)
                if isinstance(out, tuple):
                    out = out[0]
            else:
                raise ValueError("Loaded model does not expose predict or predict_samples")
            preds.append(out.detach().cpu().numpy())

    if len(preds) == 0:
        raise ValueError(f"No predictions generated for {split} split")

    return np.concatenate(preds, axis=0)


def evaluate_run(
    run_dir: Path,
    use_ema: bool | None = None,
    disable_guidance: bool = False,
) -> dict[str, Any] | None:
    reconstructor = RunReconstructor(run_dir=run_dir)
    cfg = reconstructor.cfg

    if cfg.data.name != "dataframe_masked_imputation":
        return None

    datamodule = reconstructor.build_datamodule()
    model, _ = reconstructor.load_checkpointed_model(datamodule=datamodule, use_ema=use_ema)
    model = RunReconstructor.apply_inference_overrides(
        model,
        disable_guidance=disable_guidance,
    )

    target_cols = list(datamodule.target_columns)
    metrics: dict[str, float] = {}

    split_predictions: dict[str, np.ndarray] = {}

    for split in ["val", "test"]:
        y_pred_split = _predict_split_tensor(model=model, datamodule=datamodule, split=split)
        split_predictions[split] = y_pred_split
        y_true_split = datamodule.get_original_targets(split).detach().cpu().numpy()
        split_frame = getattr(datamodule, f"{split}_df")
        split_values = split_frame.loc[:, target_cols].to_numpy()
        masked_mask = np.isnan(split_values)
        observed_mask = ~masked_mask

        metrics.update(
            compute_prefixed_metrics(
                y_true=y_true_split[masked_mask],
                y_pred=y_pred_split[masked_mask],
                metric_prefix=f"{split}_masked",
            )
        )
        metrics.update(
            compute_prefixed_metrics(
                y_true=y_true_split[observed_mask],
                y_pred=y_pred_split[observed_mask],
                metric_prefix=f"{split}_observed",
            )
        )

    # Keep aggregate test metrics for backwards-compatible summaries and plotting.
    metrics.update(
        compute_prefixed_metrics(
            y_true=datamodule.get_original_targets("test").detach().cpu().numpy(),
            y_pred=split_predictions["test"],
            metric_prefix="test",
        )
    )
    manifest = getattr(datamodule, "manifest", None)
    split_policy = (
        str(manifest.get("split_policy", "unknown"))
        if isinstance(manifest, dict)
        else "unknown"
    )

    return {
        "dataset": _resolve_dataset_name(cfg),
        "ratio": float(cfg.data.masking.ratio),
        "seed": int(cfg.experiment.seed),
        "model": infer_model_label(cfg),
        "split_policy": split_policy,
        **metrics,
    }


def _evaluate_run_safe(
    run_dir: Path,
    use_ema: bool | None,
    disable_guidance: bool,
) -> tuple[Path, dict[str, Any] | None, str | None]:
    try:
        row = evaluate_run(
            run_dir=run_dir,
            use_ema=use_ema,
            disable_guidance=disable_guidance,
        )
        return run_dir, row, None
    except Exception as exc:
        # Keep worker return values fully serializable for process-based joblib.
        return run_dir, None, str(exc)


@contextmanager
def _tqdm_joblib(tqdm_object):
    class _TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = _TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


@app.command("evaluate")
def evaluate(
    results_root: str = typer.Option("artifacts/results_raw/multitask_imputation", help="Root directory containing multirun outputs"),
    output_dir: str = typer.Option("artifacts/figures/multitask_imputation", help="Output directory for CSVs and plots"),
    use_ema: bool = typer.Option(
        True,
        "--use-ema/--no-use-ema",
        help="Prefer EMA checkpoints where available",
    ),
    n_jobs: int = typer.Option(
        1,
        "--n-jobs",
        help="Number of runs to evaluate concurrently",
    ),
    disable_guidance: bool = typer.Option(
        False,
        "--disable-guidance",
        help="Disable guidance at inference time for diffusion models",
    ),
) -> None:
    run_roots = sorted((Path(results_root)).glob("**/.hydra/config.yaml"))
    run_dirs = [p.parent.parent for p in run_roots]

    rows: list[dict[str, Any]] = []

    if n_jobs <= 1:
        eval_results = [
            _evaluate_run_safe(
                run_dir=run_dir,
                use_ema=use_ema,
                disable_guidance=disable_guidance,
            )
            for run_dir in tqdm(run_dirs, desc="Evaluating runs")
        ]
    else:
        with _tqdm_joblib(tqdm(total=len(run_dirs), desc="Evaluating runs")):
            eval_results = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_evaluate_run_safe)(run_dir, use_ema, disable_guidance)
                for run_dir in run_dirs
            )

    for run_path, row, error_msg in eval_results:
        if error_msg is not None:
            # Report skipped runs in the parent process to avoid interleaved worker logs.
            typer.echo(f"Skipping {run_path}: {error_msg}")
            continue
        if row is not None:
            rows.append(row)

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    raw_df = pd.DataFrame(rows).sort_values(["dataset", "ratio", "model", "seed"]) if rows else pd.DataFrame()
    raw_path = out_root / "raw_metrics.csv"
    raw_df.to_csv(raw_path, index=False)

    summary_df = summarize_metrics(raw_df)
    summary_path = out_root / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)

    plots_dir = out_root / "plots"
    plot_test_rmse_summary(summary_df=summary_df, output_dir=plots_dir)

    typer.echo(f"Wrote raw test metrics: {raw_path}")
    typer.echo(f"Wrote summary test metrics: {summary_path}")
    typer.echo(f"Wrote plots under: {plots_dir}")


if __name__ == "__main__":
    app()
