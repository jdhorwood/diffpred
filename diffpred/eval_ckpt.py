#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import yaml

from diffpred.utils.ops import seed_everything
from diffpred.utils.checkpoint_reload import ConfigAlignment, RunReconstructor


def recompute_metrics(
    run_dir: Path,
    ckpt_path: str | None,
    use_ema: bool | None,
    include_train: bool,
    output_path: Path,
    num_refinement_steps: int | None = None,
    num_pred_samples: int | None = None,
    disable_guidance: bool = False,
):
    if num_refinement_steps is not None and num_refinement_steps < 0:
        raise ValueError("--num-refinement-steps must be >= 0")
    if num_pred_samples is not None and num_pred_samples < 1:
        raise ValueError("--num-pred-samples must be >= 1")

    reconstructor = RunReconstructor(run_dir=run_dir)
    cfg = reconstructor.cfg
    if num_refinement_steps is not None:
        cfg.data.diffusion_params.refinement_steps = num_refinement_steps
        print(f"Overriding refinement_steps to {num_refinement_steps}")
    if num_pred_samples is not None:
        cfg.model.training.num_pred_samples = num_pred_samples
        print(f"Overriding num_pred_samples to {num_pred_samples}")
    seed_everything(cfg.experiment.seed)

    datamodule = reconstructor.build_datamodule()
    model, checkpoint = reconstructor.load_checkpointed_model(
        datamodule=datamodule,
        ckpt_path=ckpt_path,
        use_ema=use_ema,
    )
    if num_pred_samples is not None and hasattr(model, "num_pred_samples"):
        model.num_pred_samples = int(num_pred_samples)
    if disable_guidance and hasattr(model, "diffuser"):
        model.diffuser.guidance = 0.0
        print("Overriding sampling guidance: disabled (guidance=0.0)")

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        accelerator=(cfg.experiment.device if cfg.experiment.device is not None else "auto"),
        enable_model_summary=False,
        enable_progress_bar=True,
    )

    results = {}
    if include_train:
        ConfigAlignment.set_eval_split(model, "train")
        results["train"] = trainer.validate(
            model=model,
            dataloaders=datamodule.train_dataloader(),
            verbose=False,
        )

    ConfigAlignment.set_eval_split(model, "val")
    results["val"] = trainer.validate(model=model, datamodule=datamodule, verbose=False)
    results["test"] = trainer.test(model=model, datamodule=datamodule, verbose=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(results, f, sort_keys=False)

    print(f"Checkpoint: {checkpoint}")
    print(f"Wrote recomputed metrics to: {output_path}")
    return results


def _aggregate_results(results_by_run: list[dict]) -> dict:
    values_by_metric: dict[tuple[str, int, str], list[float]] = defaultdict(list)

    for run_result in results_by_run:
        for split_name in ("train", "val", "test"):
            split_entries = run_result.get(split_name, [])
            if not isinstance(split_entries, list):
                continue

            for idx, entry in enumerate(split_entries):
                if not isinstance(entry, dict):
                    continue
                for metric_name, metric_value in entry.items():
                    if isinstance(metric_value, (int, float)):
                        values_by_metric[(split_name, idx, metric_name)].append(
                            float(metric_value)
                        )

    aggregated: dict[str, list[dict]] = {}
    for split_name in ("train", "val", "test"):
        split_keys = [key for key in values_by_metric.keys() if key[0] == split_name]
        if not split_keys:
            continue

        max_idx = max(idx for _, idx, _ in split_keys)
        split_summary: list[dict] = [{} for _ in range(max_idx + 1)]
        for _, idx, metric_name in sorted(split_keys, key=lambda key: (key[1], key[2])):
            vals = np.asarray(values_by_metric[(split_name, idx, metric_name)], dtype=np.float64)
            metric_std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            split_summary[idx][f"{metric_name}/mean"] = float(np.mean(vals))
            split_summary[idx][f"{metric_name}/std"] = metric_std
            split_summary[idx][f"{metric_name}/n"] = int(vals.size)

        aggregated[split_name] = split_summary

    return aggregated


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute train/val/test metrics from a saved checkpoint."
    )
    run_scope = parser.add_mutually_exclusive_group(required=True)
    run_scope.add_argument(
        "--run-dir",
        type=Path,
        help="Path to one Hydra run directory containing .hydra/config.yaml",
    )
    run_scope.add_argument(
        "--multirun-dir",
        type=Path,
        help=(
            "Path to a multirun root. All subdirectories containing .hydra/config.yaml "
            "will be evaluated."
        ),
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Optional checkpoint path (absolute or relative to --run-dir)",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Prefer -EMA checkpoint when auto-resolving checkpoint.",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Force non-EMA checkpoint selection when auto-resolving checkpoint.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip recomputing train split metrics.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output YAML path for single-run mode only "
            "(default: <run-dir>/recomputed_results.yaml)"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help=(
            "Output YAML path for multirun aggregate mean/std summary "
            "(default: <multirun-dir>/recomputed_summary.yaml)"
        ),
    )
    parser.add_argument(
        "--num-refinement-steps",
        type=int,
        default=None,
        help=(
            "Optional override for data.diffusion_params.refinement_steps during "
            "checkpoint evaluation."
        ),
    )
    parser.add_argument(
        "--num-pred-samples",
        type=int,
        default=None,
        help=(
            "Optional override for model.training.num_pred_samples during "
            "checkpoint evaluation."
        ),
    )
    parser.add_argument(
        "--disable-guidance",
        action="store_true",
        help=(
            "Disable guidance during sampling at evaluation time by setting "
            "model.diffuser.guidance=0.0 on the loaded model."
        ),
    )
    return parser.parse_args()


def _guidance_suffix(disable_guidance: bool) -> str:
    return "_guidance-off" if disable_guidance else ""


def _discover_run_dirs(multirun_dir: Path) -> list[Path]:
    config_paths = sorted(multirun_dir.glob("**/.hydra/config.yaml"))
    run_dirs = [p.parent.parent for p in config_paths]
    return run_dirs


def main():
    args = parse_args()
    if args.use_ema and args.no_ema:
        raise ValueError("Cannot pass both --use-ema and --no-ema")

    use_ema: bool | None
    if args.use_ema:
        use_ema = True
    elif args.no_ema:
        use_ema = False
    else:
        use_ema = None

    if args.run_dir is not None:
        if args.summary_output is not None:
            raise ValueError("--summary-output is only supported with --multirun-dir")
        output_path = (
            args.output
            if args.output is not None
            else args.run_dir
            / f"recomputed_results{_guidance_suffix(args.disable_guidance)}.yaml"
        )
        recompute_metrics(
            run_dir=args.run_dir,
            ckpt_path=args.ckpt_path,
            use_ema=use_ema,
            include_train=(not args.skip_train),
            output_path=output_path,
            num_refinement_steps=args.num_refinement_steps,
            num_pred_samples=args.num_pred_samples,
            disable_guidance=args.disable_guidance,
        )
        return

    if args.output is not None:
        raise ValueError("--output is only supported with --run-dir")

    run_dirs = _discover_run_dirs(args.multirun_dir)
    if not run_dirs:
        raise ValueError(
            f"No Hydra runs found under multirun directory: {args.multirun_dir}"
        )

    failures = []
    successful_runs: list[Path] = []
    run_results: list[dict] = []
    for run_dir in run_dirs:
        try:
            results = recompute_metrics(
                run_dir=run_dir,
                ckpt_path=args.ckpt_path,
                use_ema=use_ema,
                include_train=(not args.skip_train),
                output_path=run_dir
                / f"recomputed_results{_guidance_suffix(args.disable_guidance)}.yaml",
                num_refinement_steps=args.num_refinement_steps,
                num_pred_samples=args.num_pred_samples,
                disable_guidance=args.disable_guidance,
            )
            successful_runs.append(run_dir)
            run_results.append(results)
        except Exception as exc:
            failures.append((run_dir, str(exc)))
            print(f"FAILED: {run_dir} -> {exc}")

    if run_results:
        summary_output = (
            args.summary_output
            if args.summary_output is not None
            else args.multirun_dir
            / f"recomputed_summary{_guidance_suffix(args.disable_guidance)}.yaml"
        )
        summary_payload = {
            "num_runs_total": len(run_dirs),
            "num_runs_success": len(successful_runs),
            "num_runs_failed": len(failures),
            "runs_success": [str(path) for path in successful_runs],
            "runs_failed": [{"run_dir": str(path), "error": err} for path, err in failures],
            "aggregated": _aggregate_results(run_results),
        }
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        with summary_output.open("w") as f:
            yaml.safe_dump(summary_payload, f, sort_keys=False)
        print(f"Wrote aggregate summary to: {summary_output}")

    if failures:
        print("\nCompleted with failures:")
        for run_dir, err in failures:
            print(f"- {run_dir}: {err}")
    else:
        print(f"\nSuccessfully recomputed metrics for {len(run_dirs)} runs.")


if __name__ == "__main__":
    main()
