#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import typer
import yaml

from diffpred.uq.predictive_dist import calibration_plots
from diffpred.utils.checkpoint_reload import RunReconstructor


app = typer.Typer(add_completion=False)


def _resolve_method(config: dict) -> str:
    model_cfg = config["model"]
    if bool(model_cfg.get("use_mve", False)):
        return "MVE"
    if bool(model_cfg.get("use_mcdropout", False)):
        return "MCDropout"
    if bool(model_cfg.get("use_deepensemble", False)) or bool(
        model_cfg.get("use_deep_ensemble", False)
    ):
        return "DeepEnsemble"

    hparams = model_cfg.get("hparams", {})
    model_name = str(model_cfg.get("name", ""))
    diffusion_type = (
        config.get("data", {})
        .get("diffusion_params", {})
        .get("diffusion_type", None)
    )

    # Match rejection-curves semantics: non-baseline diffusion runs are DIFFPRED.
    if (
        hparams.get("arch_type", None) is not None
        or diffusion_type is not None
        or model_name.startswith("simple_diff")
        or model_name.endswith("_diff")
    ):
        return "DIFFPRED"

    raise ValueError(
        "Could not resolve uncertainty method from config. "
        "Expected one of: use_mve/use_mcdropout/use_deepensemble for baselines, "
        "or diffusion config for DIFFPRED."
    )


def _load_ecdf_values(csv_path: Path) -> tuple[float, float]:
    df = pd.read_csv(csv_path, index_col=0)
    if "val" not in df.index or "test" not in df.index:
        raise ValueError(f"Expected 'val' and 'test' rows in {csv_path}")
    # Files are written as a single-column table where row names are split labels.
    first_col = df.columns[0]
    val = float(np.asarray(df.at["val", first_col], dtype=np.float64))
    test = float(np.asarray(df.at["test", first_col], dtype=np.float64))
    return val, test


def _extract_multirun(
    multirun_dir: Path,
) -> tuple[tuple[str, str], dict[int, tuple[float, float]], dict[int, Path]]:
    csv_paths = sorted(multirun_dir.glob("**/viz/ecdf_mmd_values.csv"))
    if not csv_paths:
        raise ValueError(f"No ecdf_mmd_values.csv files found under {multirun_dir}")

    combos: set[tuple[str, str]] = set()
    per_seed: dict[int, tuple[float, float]] = {}
    per_seed_run_dir: dict[int, Path] = {}

    for csv_path in csv_paths:
        run_dir = csv_path.parent.parent
        cfg_path = run_dir / ".hydra" / "config.yaml"
        if not cfg_path.exists():
            raise ValueError(f"Missing config file for run: {run_dir}")

        with cfg_path.open("r") as f:
            cfg = yaml.safe_load(f)

        dataset = cfg["data"]["dataset"]
        method = _resolve_method(cfg)
        seed = int(cfg["experiment"]["seed"])

        combos.add((dataset, method))
        if seed in per_seed:
            raise ValueError(
                f"Duplicate seed {seed} in multirun {multirun_dir} for combo {(dataset, method)}"
            )

        per_seed[seed] = _load_ecdf_values(csv_path)
        per_seed_run_dir[seed] = run_dir

    if len(combos) != 1:
        raise ValueError(
            f"Expected exactly one dataset/method combo per multirun at {multirun_dir}, got {sorted(combos)}"
        )

    combo = next(iter(combos))
    return combo, per_seed, per_seed_run_dir


def _merge_seed_dict(
    target: dict[int, tuple[float, float]],
    incoming: dict[int, tuple[float, float]],
    combo: tuple[str, str],
) -> None:
    overlap = set(target.keys()) & set(incoming.keys())
    if overlap:
        raise ValueError(
            "Duplicate dataset/method/seed combo found across multiruns: "
            f"{combo}. Duplicate seeds: {sorted(overlap)}"
        )
    target.update(incoming)


def _merge_run_dir_dict(
    target: dict[int, Path],
    incoming: dict[int, Path],
    combo: tuple[str, str],
) -> None:
    overlap = set(target.keys()) & set(incoming.keys())
    if overlap:
        raise ValueError(
            "Duplicate run-dir mapping for dataset/method/seed across multiruns: "
            f"{combo}. Duplicate seeds: {sorted(overlap)}"
        )
    target.update(incoming)


def _check_seed_consistency(per_combo: dict[tuple[str, str], dict[int, tuple[float, float]]]) -> None:
    expected: set[int] | None = None
    for combo, seed_vals in sorted(per_combo.items()):
        seeds = set(seed_vals.keys())
        if expected is None:
            expected = seeds
            continue
        if seeds != expected:
            raise ValueError(
                "Seed mismatch across dataset/method combos. "
                f"Expected {sorted(expected)}, got {sorted(seeds)} for combo {combo}"
            )


def _build_tables(per_combo: dict[tuple[str, str], dict[int, tuple[float, float]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets = sorted({dataset for dataset, _ in per_combo.keys()})
    methods = sorted({method for _, method in per_combo.keys()})

    val_df = pd.DataFrame(index=methods, columns=datasets, dtype=object)
    test_df = pd.DataFrame(index=methods, columns=datasets, dtype=object)

    for (dataset, method), seed_vals in per_combo.items():
        seed_items = [seed_vals[s] for s in sorted(seed_vals.keys())]
        vals = np.array([item[0] for item in seed_items], dtype=float)
        tests = np.array([item[1] for item in seed_items], dtype=float)

        val_df.loc[method, dataset] = f"{vals.mean():.4f} +- {vals.std():.4f}"
        test_df.loc[method, dataset] = f"{tests.mean():.4f} +- {tests.std():.4f}"

    val_df.index.name = "method"
    test_df.index.name = "method"
    return val_df, test_df


def _compute_guidance_off_ecdf_values(run_dir: Path) -> tuple[float, float]:
    output_dir = run_dir / "viz_guidance_off"
    output_csv = output_dir / "ecdf_mmd_values.csv"

    if output_csv.exists():
        return _load_ecdf_values(output_csv)

    reconstructor = RunReconstructor(run_dir=run_dir)
    datamodule = reconstructor.build_datamodule()
    model, _ = reconstructor.load_checkpointed_model(datamodule=datamodule)
    model = RunReconstructor.apply_inference_overrides(model, disable_guidance=True)

    # Reuse existing full calibration evaluation path; it writes ecdf_mmd_values.csv.
    calibration_plots(model=model, datamodule=datamodule, output_dir=output_dir.as_posix())
    return _load_ecdf_values(output_csv)


def _build_diffpred_guidance_ablation_tables(
    per_combo: dict[tuple[str, str], dict[int, tuple[float, float]]],
    per_combo_run_dirs: dict[tuple[str, str], dict[int, Path]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diffpred_keys = sorted(
        combo for combo in per_combo.keys() if combo[1] == "DIFFPRED"
    )
    if not diffpred_keys:
        raise ValueError("No DIFFPRED runs found; cannot build guidance ablation tables.")

    datasets = [dataset for dataset, _ in diffpred_keys]
    regime_labels = ["DIFFPRED (Guidance)", "DIFFPRED (Guidance Off)"]
    val_df = pd.DataFrame(index=regime_labels, columns=datasets, dtype=object)
    test_df = pd.DataFrame(index=regime_labels, columns=datasets, dtype=object)

    for combo in diffpred_keys:
        dataset, _ = combo
        seed_vals = per_combo[combo]
        run_dirs = per_combo_run_dirs.get(combo, {})

        if set(seed_vals.keys()) != set(run_dirs.keys()):
            raise ValueError(
                "Run directory mapping mismatch for DIFFPRED guidance ablation. "
                f"dataset={dataset}, seeds={sorted(seed_vals.keys())}, "
                f"run_dir_seeds={sorted(run_dirs.keys())}"
            )

        # Existing guidance-on values from training-time calibration callback.
        seed_items_on = [seed_vals[s] for s in sorted(seed_vals.keys())]
        vals_on = np.array([item[0] for item in seed_items_on], dtype=float)
        tests_on = np.array([item[1] for item in seed_items_on], dtype=float)

        # Recomputed guidance-off values using checkpoint reload + inference overrides.
        seed_items_off = [
            _compute_guidance_off_ecdf_values(run_dirs[s])
            for s in sorted(run_dirs.keys())
        ]
        vals_off = np.array([item[0] for item in seed_items_off], dtype=float)
        tests_off = np.array([item[1] for item in seed_items_off], dtype=float)

        val_df.loc["DIFFPRED (Guidance)", dataset] = f"{vals_on.mean():.4f} +- {vals_on.std():.4f}"
        test_df.loc["DIFFPRED (Guidance)", dataset] = f"{tests_on.mean():.4f} +- {tests_on.std():.4f}"

        val_df.loc["DIFFPRED (Guidance Off)", dataset] = f"{vals_off.mean():.4f} +- {vals_off.std():.4f}"
        test_df.loc["DIFFPRED (Guidance Off)", dataset] = f"{tests_off.mean():.4f} +- {tests_off.std():.4f}"

    val_df.index.name = "regime"
    test_df.index.name = "regime"
    return val_df, test_df


@app.command()
def main(
    root_dir: Annotated[
        Path,
        typer.Argument(help="Higher-level directory containing multirun folders, or top-level multirun directory."),
    ],
    output_dir: Annotated[
        Optional[Path],
        typer.Option(help="Optional output directory (default: root_dir)."),
    ] = None,
    compute_guidance_ablation: Annotated[
        bool,
        typer.Option(
            "--compute-guidance-ablation/--no-compute-guidance-ablation",
            help=(
                "Recompute DIFFPRED calibration values with guidance disabled at "
                "inference time and write separate DIFFPRED guidance ablation tables."
            ),
        ),
    ] = False,
):
    root_dir = root_dir.resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        raise ValueError(f"Invalid root directory: {root_dir}")

    multirun_dirs = sorted(
        path for path in root_dir.iterdir() if path.is_dir() and (path / "multirun.yaml").exists()
    )
    if not multirun_dirs:
        if (root_dir / "multirun.yaml").exists():
            # Top-level multirun mode: only include Hydra run directories.
            multirun_dirs = sorted(
                path
                for path in root_dir.iterdir()
                if path.is_dir() and (path / ".hydra" / "config.yaml").exists()
            )
        else:
            raise ValueError(f"No multirun directories found under {root_dir}")

    per_combo: dict[tuple[str, str], dict[int, tuple[float, float]]] = {}
    per_combo_run_dirs: dict[tuple[str, str], dict[int, Path]] = {}
    for multirun_dir in multirun_dirs:
        combo, seed_vals, seed_run_dirs = _extract_multirun(multirun_dir)
        if combo in per_combo:
            _merge_seed_dict(per_combo[combo], seed_vals, combo)
            _merge_run_dir_dict(per_combo_run_dirs[combo], seed_run_dirs, combo)
        else:
            per_combo[combo] = dict(seed_vals)
            per_combo_run_dirs[combo] = dict(seed_run_dirs)

    _check_seed_consistency(per_combo)
    val_df, test_df = _build_tables(per_combo)

    out_dir = (output_dir or root_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    val_path = out_dir / "ecdf_mmd_summary_val.csv"
    test_path = out_dir / "ecdf_mmd_summary_test.csv"
    val_df.to_csv(val_path)
    test_df.to_csv(test_path)

    guidance_ablation_val_path: Path | None = None
    guidance_ablation_test_path: Path | None = None
    if compute_guidance_ablation:
        guidance_val_df, guidance_test_df = _build_diffpred_guidance_ablation_tables(
            per_combo=per_combo,
            per_combo_run_dirs=per_combo_run_dirs,
        )
        guidance_ablation_val_path = out_dir / "ecdf_mmd_summary_val_diffpred_guidance_ablation.csv"
        guidance_ablation_test_path = out_dir / "ecdf_mmd_summary_test_diffpred_guidance_ablation.csv"
        guidance_val_df.to_csv(guidance_ablation_val_path)
        guidance_test_df.to_csv(guidance_ablation_test_path)

    typer.echo(f"Processed multiruns: {len(multirun_dirs)}")
    typer.echo(f"Validation summary: {val_path}")
    typer.echo(f"Test summary: {test_path}")
    if guidance_ablation_val_path is not None and guidance_ablation_test_path is not None:
        typer.echo(f"Guidance ablation validation summary: {guidance_ablation_val_path}")
        typer.echo(f"Guidance ablation test summary: {guidance_ablation_test_path}")


if __name__ == "__main__":
    app()
