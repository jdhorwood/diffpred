#!/usr/bin/env python3
import re
from pathlib import Path
from typing import Annotated, List, Optional, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import typer

from diffpred.trainers import DiffusionTrainer
from diffpred.constants.metrics import compute_metrics
from diffpred.eval_ckpt import _discover_run_dirs
from diffpred.utils.checkpoint_reload import RunReconstructor
from diffpred.utils.ops import seed_everything
from diffpred.uq.predictive_dist import _get_split_dataloader
from diffpred.viz.utils import set_mpl_settings
from diffpred.utils.reloader import resolve_uncertainty_method, extract_tag_from_run_path


app = typer.Typer(add_completion=False)

METHOD_ORDER = ["DIFFPRED", "MVE", "MCDropout", "DeepEnsemble"]
DIFFPRED_ENSEMBLE_LABEL = "DIFFPRED Ensemble"


def _normalize_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor.unsqueeze(-1)
    if tensor.ndim == 2:
        return tensor
    raise ValueError(f"Expected 1D or 2D tensor, got shape {tuple(tensor.shape)}")


def _resolve_metric_fn(model, metric_name: str):
    if not hasattr(model, "metric_fns"):
        raise ValueError("Model does not expose metric_fns for metric lookup")
    if metric_name not in model.metric_fns:
        available = sorted(model.metric_fns.keys())
        raise ValueError(f"Unknown metric '{metric_name}'. Available metrics: {available}")
    return model.metric_fns[metric_name]


def _predict_with_uncertainty_for_split(
    model,
    datamodule,
    split: str,
    method: Literal[
        "DIFFPRED",
        "MVE",
        "MCDropout",
        "DeepEnsemble",
        ]
):
    model.eval()
    dataloader = _get_split_dataloader(datamodule, split=split)

    preds = []
    targets = []
    uncertainties = []
    samples_list = []

    for batch in dataloader:
        x, y = batch
        with torch.no_grad():
            if method == 'DIFFPRED':
                samples = model.predict_samples(x.to(model.device), reduce=False)
                pred = model.pred_mean_from_samples(samples)
                uncertainty = model.get_uncertainty(samples)
            else:
                pred_out = model.compute_pred_with_uncertainty(x.to(model.device))
                if len(pred_out) != 2:
                    raise ValueError(
                        "Unsupported return format from compute_pred_with_uncertainty; "
                        "expected (pred, std)"
                    )
                pred, uncertainty = pred_out
                samples = None

        uncertainty = uncertainty.abs()
        y_inv = datamodule.inverse_target_transform(y)

        preds.append(pred.detach().cpu())
        targets.append(y_inv.detach().cpu())
        uncertainties.append(uncertainty.detach().cpu())
        if samples is not None:
            samples_list.append(samples.detach().cpu())


    y_pred = torch.cat(preds, dim=0)
    y_true = torch.cat(targets, dim=0)
    uncertainty = torch.cat(uncertainties, dim=0)
    y_pred = _normalize_2d(y_pred)
    y_true = _normalize_2d(y_true)
    uncertainty = _normalize_2d(uncertainty)
    # Single confidence score per sample for sorting/rejection - lower is better
    confidence_score = uncertainty.mean(dim=1)

    if samples_list:
        samples = torch.cat(samples_list, dim=1) # Batch dim 

    return y_pred, y_true, confidence_score, samples


def diffpred_ensemble_pred_with_unc(all_samples: torch.FloatTensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred = DiffusionTrainer.pred_mean_from_samples(all_samples)
    uncertainty = DiffusionTrainer.get_uncertainty(all_samples)
    return pred, uncertainty

def _build_run_curve(
    metric_name: str,
    metric_fn,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    confidence_score: torch.Tensor,
    step_pct: int = 5,
):
    n_samples = y_true.shape[0]
    sorted_idx = torch.argsort(confidence_score, dim=0, descending=False)
    retain_pcts = list(range(100, 0, -step_pct))

    rows = []
    for retain_pct in retain_pcts:
        retain_n = int(np.floor((retain_pct / 100.0) * n_samples))

        if retain_n <= 0:
            metric_value = np.nan
        else:
            keep_idx = sorted_idx[:retain_n].squeeze(-1)
            y_true_cut = _normalize_2d(y_true[keep_idx])
            y_pred_cut = _normalize_2d(y_pred[keep_idx])
            y_proba = y_pred_cut if metric_name in ("roc_auc", "pr_auc", "bce") else None

            metric_res = compute_metrics(
                y_true=y_true_cut,
                y_pred=y_pred_cut,
                y_proba=y_proba,
                metric_fns={metric_name: metric_fn},
            )
            metric_value = metric_res[metric_name]
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.item()
            metric_value = float(metric_value)

        rows.append(
            {
                "retain_pct": retain_pct,
                "num_retained": retain_n,
                "num_total": n_samples,
                "metric": metric_value,
            }
        )

    return pd.DataFrame(rows)


def _format_uncertainty_name(model_name: str) -> str:
    if model_name.startswith("simple_diff"):
        return model_name.replace("simple_diff", "DIFFPRED", 1)
    return model_name



def _plot_rejection_curve(
    summary_df: pd.DataFrame,
    metric_name: str,
    out_path: Path,
    uncertainty_name: str,
):
    set_mpl_settings()
    fig, ax = plt.subplots(figsize=(8, 5))

    x = 100.0 - summary_df["retain_pct"].to_numpy(dtype=float)
    y = summary_df["metric_mean"].to_numpy(dtype=float)
    y_std = summary_df["metric_std"].to_numpy(dtype=float)

    ax.plot(x, y, label=uncertainty_name)
    ax.fill_between(x, y - y_std, y + y_std, alpha=0.2)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence Percentile")
    ax.set_ylabel(metric_name.upper())
    ax.legend(loc="best", fontsize=14, handlelength=1.8, borderpad=0.35)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_rejection_curve_multi(
    summary_df: pd.DataFrame,
    metric_name: str,
    out_path: Path,
):
    set_mpl_settings()
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = sorted(
        summary_df["uncertainty_method"].unique().tolist(),
        key=lambda name: (
            METHOD_ORDER.index(name)
            if name in METHOD_ORDER
            else (len(METHOD_ORDER) if name == DIFFPRED_ENSEMBLE_LABEL else len(METHOD_ORDER) + 1),
            name,
        ),
    )

    for method_name in methods:
        method_df = summary_df[summary_df["uncertainty_method"] == method_name]
        x = 100.0 - method_df["retain_pct"].to_numpy(dtype=float)
        y = method_df["metric_mean"].to_numpy(dtype=float)
        y_std = method_df["metric_std"].to_numpy(dtype=float)

        is_diffpred_ensemble = method_name == DIFFPRED_ENSEMBLE_LABEL
        ax.plot(
            x,
            y,
            label=method_name,
            linestyle=(":" if is_diffpred_ensemble else "-"),
            linewidth=(2.5 if is_diffpred_ensemble else 2.0),
            color=('red' if is_diffpred_ensemble else None),
        )
        if not is_diffpred_ensemble:
            ax.fill_between(x, y - y_std, y + y_std, alpha=0.15)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence Percentile")
    ax.set_ylabel(metric_name.upper())
    ax.legend(loc="best", fontsize=14, handlelength=1.8, borderpad=0.35)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _get_run_metadata(run_dir: Path) -> tuple[str, str, int]:
    dataset = extract_tag_from_run_path(run_dir, "data.dataset")
    model_name = extract_tag_from_run_path(run_dir, "model")
    seed_str = extract_tag_from_run_path(run_dir, "seed")
    if dataset is None:
        raise ValueError(f"Could not parse 'data.dataset=' from run path: {run_dir}")
    if model_name is None:
        raise ValueError(f"Could not parse 'model=' from run path: {run_dir}")
    if seed_str is None:
        raise ValueError(f"Could not parse 'seed=' from run path: {run_dir}")
    try:
        seed = int(seed_str)
    except ValueError as exc:
        raise ValueError(
            f"Parsed non-integer seed '{seed_str}' from run path: {run_dir}"
        ) from exc

    return dataset, model_name, seed


def _resolve_filtered_run_dirs(
    multirun_dir: Path,
    dataset_name: str,
    model_name: str,
    seeds: list[int],
) -> list[Path]:
    run_dirs = _discover_run_dirs(multirun_dir)
    if not run_dirs:
        raise ValueError(f"No Hydra runs found under multirun directory: {multirun_dir}")

    seed_to_run_dir: dict[int, Path] = {}
    recovered_seeds: list[int] = []

    for run_dir in run_dirs:
        run_dataset, run_model, run_seed = _get_run_metadata(run_dir)
        if run_dataset != dataset_name or run_model != model_name:
            continue

        recovered_seeds.append(run_seed)
        if run_seed in seed_to_run_dir:
            raise ValueError(
                "Multiple runs matched the same dataset/model/seed combination: "
                f"dataset={dataset_name}, model={model_name}, seed={run_seed}. "
                f"Conflicting runs: {seed_to_run_dir[run_seed]} and {run_dir}"
            )
        seed_to_run_dir[run_seed] = run_dir

    requested_seed_set = set(seeds)
    recovered_seed_set = set(recovered_seeds)
    if recovered_seed_set != requested_seed_set:
        missing = sorted(requested_seed_set - recovered_seed_set)
        extra = sorted(recovered_seed_set - requested_seed_set)
        raise ValueError(
            "Filtered multirun seed mismatch for "
            f"dataset={dataset_name}, model={model_name}. "
            f"Requested seeds={sorted(requested_seed_set)}, "
            f"recovered seeds={sorted(recovered_seed_set)}, "
            f"missing={missing}, extra={extra}"
        )

    return [seed_to_run_dir[seed] for seed in seeds]


def _resolve_dataset_method_run_dirs(
    multirun_dir: Path,
    dataset_name: str,
    seeds: list[int],
) -> dict[str, list[Path]]:
    run_dirs = _discover_run_dirs(multirun_dir)
    if not run_dirs:
        raise ValueError(f"No Hydra runs found under multirun directory: {multirun_dir}")

    per_method_seed_dir: dict[str, dict[int, Path]] = {}
    for run_dir in run_dirs:
        run_dataset, _, run_seed = _get_run_metadata(run_dir)
        if run_dataset != dataset_name:
            continue

        method = resolve_uncertainty_method(run_dir)
        if method is None:
            continue

        if method not in per_method_seed_dir:
            per_method_seed_dir[method] = {}
        if run_seed in per_method_seed_dir[method]:
            raise ValueError(
                "Multiple runs matched the same dataset/method/seed combination: "
                f"dataset={dataset_name}, method={method}, seed={run_seed}. "
                f"Conflicting runs: {per_method_seed_dir[method][run_seed]} and {run_dir}"
            )
        per_method_seed_dir[method][run_seed] = run_dir

    if not per_method_seed_dir:
        raise ValueError(
            f"No uncertainty-method runs found for dataset={dataset_name} under {multirun_dir}"
        )

    requested_seed_set = set(seeds)
    resolved: dict[str, list[Path]] = {}
    for method, seed_dir in per_method_seed_dir.items():
        recovered_seed_set = set(seed_dir.keys())
        if recovered_seed_set != requested_seed_set:
            missing = sorted(requested_seed_set - recovered_seed_set)
            extra = sorted(recovered_seed_set - requested_seed_set)
            raise ValueError(
                "Seed mismatch for uncertainty method in dataset-only mode: "
                f"dataset={dataset_name}, method={method}. "
                f"Requested seeds={sorted(requested_seed_set)}, "
                f"recovered seeds={sorted(recovered_seed_set)}, "
                f"missing={missing}, extra={extra}"
            )
        resolved[method] = [seed_dir[seed] for seed in seeds]

    return resolved


def _parse_seed_list(raw_seeds: list[str] | None) -> list[int] | None:
    if raw_seeds is None:
        return None

    merged = " ".join(raw_seeds).strip()
    if merged == "":
        return []

    # Accept forms like: "1 2 3", "1,2,3", "[1,2,3]", "(1, 2, 3)".
    cleaned = re.sub(r"[\[\]\(\)]", " ", merged)
    tokens = [tok for tok in re.split(r"[\s,]+", cleaned) if tok]
    try:
        return [int(tok) for tok in tokens]
    except ValueError as exc:
        raise ValueError(
            "Invalid --seeds format. Use integers, e.g. '--seeds 1 2 3' or "
            "'--seeds [1,2,3]'"
        ) from exc

def _check_matches(split, diffpred_seed_preds, new_seed, new_y_true):
    if diffpred_seed_preds:
        y_true_ref = diffpred_seed_preds[0][2]
        if not torch.allclose(
                        y_true_ref,
                        new_y_true,
                        atol=1e-6,
                        rtol=1e-5,
                        equal_nan=True,
                    ):
            max_abs_diff = float((y_true_ref - new_y_true).abs().max().item())
            raise ValueError(
                            "Cannot build DIFFPRED ensemble curve: y_true mismatch across seeds "
                            f"for split='{split}'. Mismatching seed: {new_seed}. "
                            f"max_abs_diff={max_abs_diff:.3e}"
                        )
        
@app.command()
def main(
    run_dir: Annotated[
        Optional[Path],
        typer.Option(help="Path to a single Hydra run directory containing .hydra/config.yaml."),
    ] = None,
    multirun_dir: Annotated[
        Optional[Path],
        typer.Option(help="Path to multirun root containing per-seed run directories."),
    ] = None,
    metric: Annotated[
        Optional[str],
        typer.Option(help="Metric name to evaluate at each rejection cutoff (e.g., rmse, mae, r2_score)."),
    ] = None,
    split: Annotated[
        str,
        typer.Option(help="Dataset split to evaluate."),
    ] = "test",
    step_pct: Annotated[
        int,
        typer.Option(help="Retention percentage step size. Default: 5."),
    ] = 5,
    ckpt_path: Annotated[
        Optional[str],
        typer.Option(help="Optional checkpoint path (absolute or relative to each run dir)."),
    ] = None,
    use_ema: Annotated[
        bool,
        typer.Option(help="Prefer -EMA checkpoint when auto-resolving."),
    ] = False,
    no_ema: Annotated[
        bool,
        typer.Option(help="Force non-EMA checkpoint selection when auto-resolving."),
    ] = False,
    output_dir_opt: Annotated[
        Optional[Path],
        typer.Option(
            help=(
                "Output directory for tables/figures "
                "(default: <run-dir or multirun-dir>/rejection_curves_<metric>_<split>)."
            )
        ),
    ] = None,
    dataset_name: Annotated[
        Optional[str],
        typer.Option(help="Dataset name for filtered multirun mode."),
    ] = None,
    model_name: Annotated[
        Optional[str],
        typer.Option(help="Model name (net_type) for filtered multirun mode."),
    ] = None,
    seeds: Annotated[
        Optional[List[str]],
        typer.Option(
            help=(
                "Explicit seed list for filtered multirun mode. "
                "Supported forms: '--seeds 1 --seeds 2 --seeds 3', "
                "'--seeds 1,2,3', '--seeds [1,2,3]'."
            )
        ),
    ] = None,
    filtered_output_dir: Annotated[
        Optional[Path],
        typer.Option(help="Required output directory for filtered multirun mode."),
    ] = None,
    num_refinement_steps: Annotated[
        Optional[int],
        typer.Option(
            help="Optional override for data.diffusion_params.refinement_steps on DIFFPRED runs."
        ),
    ] = None,
    num_pred_samples: Annotated[
        Optional[int],
        typer.Option(
            help="Optional override for model.training.num_pred_samples on DIFFPRED runs."
        ),
    ] = None,
    disable_guidance: Annotated[
        bool,
        typer.Option(
            help="Disable guidance during DIFFPRED sampling (sets diffusion guidance to 0)."
        ),
    ] = True,
    deterministic_sampling: Annotated[
        Optional[bool],
        typer.Option(
            help=(
                "Optional override for EDM deterministic sampling. "
                "If set, forces diffuser.sample(..., deterministic=<value>) at runtime."
            )
        ),
    ] = None,
    ensemble_epistemic_lambda: Annotated[
        float,
        typer.Option(
            help=(
                "Scale factor lambda for DIFFPRED ensemble epistemic uncertainty. "
                "Used as: total_uncertainty = mean_uncertainty + lambda * epistemic_std"
            )
        ),
    ] = 1.0,
    guidance_ablation: Annotated[
        bool,
        typer.Option(
            help=(
                "Generate a separate DIFFPRED-only rejection-curves ablation by "
                "evaluating each DIFFPRED run with guidance enabled and disabled. "
                "In this mode, DIFFPRED ensemble is not computed."
            )
        ),
    ] = False,
):
    parsed_seeds, use_ema_choice, filtered_multirun_mode, dataset_only_multirun_mode = _parse_check_args(run_dir, multirun_dir, metric, split, step_pct, use_ema, no_ema, dataset_name, model_name, seeds, filtered_output_dir, num_refinement_steps, num_pred_samples, ensemble_epistemic_lambda)

    single_run_mode = run_dir is not None
    run_method_map: dict[str, str] = {}
    ckpt_path_arg = ckpt_path
    if single_run_mode:
        if not (run_dir / ".hydra" / "config.yaml").exists():
            raise ValueError(f"Invalid --run-dir: {run_dir} (missing .hydra/config.yaml)")
        run_dirs = [run_dir]
    elif filtered_multirun_mode:
        assert multirun_dir is not None
        assert dataset_name is not None
        assert model_name is not None
        assert parsed_seeds is not None
        run_dirs = _resolve_filtered_run_dirs(
            multirun_dir=multirun_dir,
            dataset_name=dataset_name,
            model_name=model_name,
            seeds=parsed_seeds,
        )
    elif dataset_only_multirun_mode:
        assert multirun_dir is not None
        assert dataset_name is not None
        assert parsed_seeds is not None
        method_dirs = _resolve_dataset_method_run_dirs(
            multirun_dir=multirun_dir,
            dataset_name=dataset_name,
            seeds=parsed_seeds,
        )

        if guidance_ablation:
            if "DIFFPRED" not in method_dirs:
                raise ValueError(
                    "--guidance-ablation requested but no DIFFPRED runs were found for "
                    f"dataset={dataset_name} under {multirun_dir}"
                )
            run_dirs = list(method_dirs["DIFFPRED"])
            for path in run_dirs:
                run_method_map[str(path)] = "DIFFPRED"
        else:
            run_dirs = []
            for method_name, method_run_dirs in method_dirs.items():
                for path in method_run_dirs:
                    run_dirs.append(path)
                    run_method_map[str(path)] = method_name
    else:
        assert multirun_dir is not None
        if guidance_ablation:
            raise ValueError(
                "--guidance-ablation is supported only in dataset-only filtered multirun mode "
                "(--multirun-dir + --dataset-name + --seeds, without --model-name)."
            )
        run_dirs = _discover_run_dirs(multirun_dir)

    if not run_dirs:
        raise ValueError(f"No Hydra runs found under multirun directory: {multirun_dir}")

    if single_run_mode:
        assert run_dir is not None
        default_base_dir = run_dir
    else:
        assert multirun_dir is not None
        default_base_dir = multirun_dir

    run_metadata = [_get_run_metadata(path) for path in run_dirs]
    dataset_names = sorted({dataset for dataset, _, _ in run_metadata})
    model_names = sorted({model for _, model, _ in run_metadata})
    dataset_tag = dataset_names[0] if len(dataset_names) == 1 else "mixed_dataset"
    uncertainty_name = (
        _format_uncertainty_name(model_names[0])
        if len(model_names) == 1
        else "MULTI_MODEL"
    )

    base_output_dir = (
        filtered_output_dir
        if (filtered_multirun_mode or dataset_only_multirun_mode)
        else (output_dir_opt if output_dir_opt is not None else default_base_dir)
    )
    assert base_output_dir is not None
    output_dir = base_output_dir / f"rejection_curves_{dataset_tag}_{metric}_{split}"
    if guidance_ablation:
        output_dir = base_output_dir / f"rejection_curves_{dataset_tag}_{metric}_{split}_guidance_ablation"
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    curve_frames = []
    diffpred_seed_preds: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
    diffpred_metric_fn = None
    for run_dir in run_dirs:
        try:
            _, run_model_name, _ = _get_run_metadata(run_dir)
            method_name = run_method_map.get(str(run_dir), resolve_uncertainty_method(run_dir))
            if method_name is None:
                method_name = _format_uncertainty_name(run_model_name)
            reconstructor = RunReconstructor(run_dir=run_dir)
            cfg = reconstructor.cfg
            seed_everything(cfg.experiment.seed)

            datamodule = reconstructor.build_datamodule()
            model, checkpoint = reconstructor.load_checkpointed_model(
                datamodule=datamodule,
                ckpt_path=ckpt_path_arg,
                use_ema=use_ema_choice,
            )
            metric_fn = _resolve_metric_fn(model, metric)

            if guidance_ablation and method_name == "DIFFPRED":
                regime_configs = [
                    ("DIFFPRED (Guidance)", False),
                    ("DIFFPRED (Guidance Off)", True),
                ]
            else:
                regime_configs = [(method_name, disable_guidance)]

            for regime_name, regime_disable_guidance in regime_configs:
                model = _param_overrides(
                    num_refinement_steps,
                    num_pred_samples,
                    regime_disable_guidance,
                    deterministic_sampling,
                    run_model_name,
                    model,
                )

                y_pred, y_true, uncertainty, samples = _predict_with_uncertainty_for_split(
                    model=model,
                    datamodule=datamodule,
                    split=split,
                    method=method_name,
                )

                if (
                    dataset_only_multirun_mode
                    and method_name == "DIFFPRED"
                    and not guidance_ablation
                ):
                    new_seed = int(cfg.experiment.seed)
                    new_y_pred = y_pred.detach().cpu()
                    new_y_true = y_true.detach().cpu()
                    new_uncertainty = uncertainty.detach().cpu()
                    # Early check: if not the first, compare y_true to reference
                    _check_matches(split, diffpred_seed_preds, new_seed, new_y_true)
                    diffpred_seed_preds.append((new_seed, new_y_pred, new_y_true, new_uncertainty, samples))
                    if diffpred_metric_fn is None:
                        diffpred_metric_fn = metric_fn

                run_curve = _build_run_curve(
                    metric_name=metric,
                    metric_fn=metric_fn,
                    y_true=y_true,
                    y_pred=y_pred,
                    confidence_score=uncertainty,
                    step_pct=step_pct,
                )
                run_curve["run_dir"] = str(run_dir)
                run_curve["seed"] = int(cfg.experiment.seed)
                run_curve["checkpoint"] = str(checkpoint)
                run_curve["uncertainty_method"] = regime_name
                curve_frames.append(run_curve)
        except Exception as exc:
            failures.append({"run_dir": str(run_dir), "error": str(exc)})
            print(f"FAILED: {run_dir} -> {exc}")

    if not curve_frames:
        raise RuntimeError("All runs failed; no rejection curves were computed.")

    if dataset_only_multirun_mode and (not guidance_ablation) and len(diffpred_seed_preds) > 1:
        diffpred_seed_preds.sort(key=lambda item: item[0])
        y_pred_stack = torch.cat([item[-1] for item in diffpred_seed_preds], dim=0) # pred samples dim is 0
        y_true_ref = diffpred_seed_preds[0][2]

        # Alignment already checked during collection; proceed to ensemble computation
        ensemble_y_pred, ensemble_confidence_score = diffpred_ensemble_pred_with_unc(all_samples=y_pred_stack)
        ensemble_curve = _build_run_curve(
            metric_name=metric,
            metric_fn=diffpred_metric_fn,
            y_true=y_true_ref,
            y_pred=ensemble_y_pred,
            confidence_score=ensemble_confidence_score,
            step_pct=step_pct,
        )
        ensemble_curve["run_dir"] = "DIFFPRED_ENSEMBLE"
        ensemble_curve["seed"] = -1
        ensemble_curve["checkpoint"] = "ensemble"
        ensemble_curve["uncertainty_method"] = DIFFPRED_ENSEMBLE_LABEL
        curve_frames.append(ensemble_curve)

    per_run_df = pd.concat(curve_frames, ignore_index=True)
    if single_run_mode:
        summary_df = (
            per_run_df[["retain_pct", "num_retained", "metric"]]
            .rename(
                columns={
                    "num_retained": "num_retained_mean",
                    "metric": "metric_mean",
                }
            )
            .sort_values("retain_pct", ascending=False)
        )
        summary_df["metric_std"] = 0.0
        summary_df["n_seeds"] = 1
    elif dataset_only_multirun_mode:
        summary_df = (
            per_run_df.groupby(["uncertainty_method", "retain_pct"], as_index=False)
            .agg(
                metric_mean=("metric", "mean"),
                metric_std=("metric", "std"),
                num_retained_mean=("num_retained", "mean"),
                n_seeds=("metric", "count"),
            )
            .sort_values(["uncertainty_method", "retain_pct"], ascending=[True, False])
        )
        summary_df["metric_std"] = summary_df["metric_std"].fillna(0.0)
    else:
        summary_df = (
            per_run_df.groupby("retain_pct", as_index=False)
            .agg(
                metric_mean=("metric", "mean"),
                metric_std=("metric", "std"),
                num_retained_mean=("num_retained", "mean"),
                n_seeds=("metric", "count"),
            )
            .sort_values("retain_pct", ascending=False)
        )
        summary_df["metric_std"] = summary_df["metric_std"].fillna(0.0)

    per_run_path = output_dir / "rejection_curve_per_run.csv"
    summary_path = output_dir / "rejection_curve_summary.csv"
    failures_path = output_dir / "rejection_curve_failures.csv"
    fig_path = output_dir / "rejection_curve.png"

    per_run_df.to_csv(per_run_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pd.DataFrame(failures).to_csv(failures_path, index=False)

    if dataset_only_multirun_mode:
        _plot_rejection_curve_multi(
            summary_df=summary_df,
            metric_name=metric,
            out_path=fig_path,
        )
    else:
        _plot_rejection_curve(
            summary_df=summary_df,
            metric_name=metric,
            out_path=fig_path,
            uncertainty_name=uncertainty_name,
        )

    typer.echo(f"Evaluated runs: {len(curve_frames)} / {len(run_dirs)}")
    typer.echo(f"Per-run results: {per_run_path}")
    typer.echo(f"Summary table: {summary_path}")
    typer.echo(f"Failures: {failures_path}")
    typer.echo(f"Figure: {fig_path}")

def _parse_check_args(run_dir, multirun_dir, metric, split, step_pct, use_ema, no_ema, dataset_name, model_name, seeds, filtered_output_dir, num_refinement_steps, num_pred_samples, ensemble_epistemic_lambda):
    if metric is None:
        raise ValueError("--metric is required")

    if step_pct <= 0 or step_pct > 100:
        raise ValueError("--step-pct must be in [1, 100]")

    if split not in ("train", "val", "test"):
        raise ValueError("--split must be one of: train, val, test")

    if (run_dir is None) == (multirun_dir is None):
        raise ValueError("Exactly one of --run-dir or --multirun-dir must be provided")

    if use_ema and no_ema:
        raise ValueError("Cannot pass both --use-ema and --no-ema")

    if num_refinement_steps is not None and num_refinement_steps < 0:
        raise ValueError("--num-refinement-steps must be >= 0")

    if num_pred_samples is not None and num_pred_samples < 1:
        raise ValueError("--num-pred-samples must be >= 1")

    if ensemble_epistemic_lambda < 0:
        raise ValueError("--ensemble-epistemic-lambda must be >= 0")

    parsed_seeds = _parse_seed_list(seeds)

    use_ema_choice: bool | None
    if use_ema:
        use_ema_choice = True
    elif no_ema:
        use_ema_choice = False
    else:
        use_ema_choice = None

    has_filter_args = any(
        v is not None for v in (dataset_name, model_name, seeds)
    )
    filtered_multirun_mode = (
        multirun_dir is not None
        and dataset_name is not None
        and model_name is not None
        and parsed_seeds is not None
    )
    dataset_only_multirun_mode = (
        multirun_dir is not None
        and dataset_name is not None
        and model_name is None
        and parsed_seeds is not None
    )
    if has_filter_args and not (filtered_multirun_mode or dataset_only_multirun_mode):
        raise ValueError(
            "Filtered multirun mode requires either: "
            "(1) --multirun-dir + --dataset-name + --model-name + --seeds, or "
            "(2) --multirun-dir + --dataset-name + --seeds (all uncertainty methods)"
        )
    if (filtered_multirun_mode or dataset_only_multirun_mode) and filtered_output_dir is None:
        raise ValueError(
            "--filtered-output-dir is required in filtered multirun mode"
        )
    if (not filtered_multirun_mode) and (not dataset_only_multirun_mode) and filtered_output_dir is not None:
        raise ValueError(
            "--filtered-output-dir is only valid in filtered multirun mode"
        )
    if filtered_multirun_mode or dataset_only_multirun_mode:
        if parsed_seeds is None:
            raise ValueError("--seeds is required in filtered multirun mode")
        if len(parsed_seeds) == 0:
            raise ValueError("--seeds cannot be empty in filtered multirun mode")
    return parsed_seeds,use_ema_choice,filtered_multirun_mode,dataset_only_multirun_mode

def _param_overrides(num_refinement_steps, num_pred_samples, disable_guidance, deterministic_sampling, run_model_name, model):
    if (
                num_refinement_steps is not None
                and run_model_name.startswith("simple_diff")
                and hasattr(model, "diffuser")
            ):
        model.diffuser.refinement_steps = int(num_refinement_steps)
    if num_pred_samples is not None and run_model_name.startswith("simple_diff"):
        model.num_pred_samples = int(num_pred_samples)
    if disable_guidance and run_model_name.startswith("simple_diff") and hasattr(model, "diffuser"):
        model.diffuser.guidance = 0.0
    if (
                deterministic_sampling is not None
                and run_model_name.startswith("simple_diff")
                and hasattr(model, "diffuser")
                and hasattr(model.diffuser, "sample")
            ):
        model.diffuser.deterministic_sampling = deterministic_sampling
    return model




if __name__ == "__main__":
    app()
