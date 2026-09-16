from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import typer
import yaml
import requests

from diffpred.constants.multitask_imputation import MANIFEST_SCHEMA_VERSION
from diffpred.data.masked_imputation_artifacts import apply_masking_from_manifest
from diffpred.data.dockstring import DockstringDatamodule
from diffpred.data.masked_dataframe_datamodule import DOCKSTRING_TARGETS

app = typer.Typer(add_completion=False)


def _build_manifest_from_explicit_splits(
    target_columns: list[str],
    seed: int,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    split_policy: str,
) -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "seed": seed,
        "split_policy": split_policy,
        "target_columns": list(target_columns),
        "splits": {
            "train": np.asarray(train_ids).tolist(),
            "val": np.asarray(val_ids).tolist(),
            "test": np.asarray(test_ids).tolist(),
        },
    }


def _load_dockstring_panel(seed: int) -> tuple[pd.DataFrame, dict, str, list[str]]:
    dataset_path = Path(DockstringDatamodule.DATA_PATH) / DockstringDatamodule.DATASET_NAME
    if not dataset_path.exists():
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Downloading Dockstring dataset from {DockstringDatamodule.DATASET_URL}...")
        response = requests.get(DockstringDatamodule.DATASET_URL, timeout=120)
        response.raise_for_status()
        dataset_path.write_bytes(response.content)

    table = pd.read_csv(dataset_path, sep="\t")
    required_cols = ["smiles", *DOCKSTRING_TARGETS]
    missing_cols = [c for c in required_cols if c not in table.columns]
    if missing_cols:
        raise ValueError(
            "Dockstring dataset missing required panel columns: "
            f"{missing_cols}"
        )

    table = table.loc[:, required_cols].copy()
    duplicate_mask = table["smiles"].duplicated(keep="first")
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        typer.echo(
            f"Dropping {duplicate_count} duplicate SMILES rows from Dockstring panel during prep."
        )
        table = table.loc[~duplicate_mask].copy()

    # Canonicalize row order before assigning integer IDs so IDs are deterministic
    table = table.sort_values("smiles").reset_index(drop=True)
    table.index = np.arange(len(table), dtype=np.int64)
    table.index.name = "row_id"

    target_columns = list(DOCKSTRING_TARGETS)

    # Mirror DockstringDatamodule behavior: deterministic test split, seed-dependent val split.
    train_val_ids, test_ids = train_test_split(
        table.index.to_numpy(),
        test_size=0.2,
        random_state=0,
    )
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=0.1,
        random_state=seed,
    )

    split_policy = "dockstring_random_fixed_test"
    return table, {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }, split_policy, target_columns


def _get_numeric_target_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"smiles", "ORIGINAL_SMILES", "UNIQUE_ID", "sample_id", "row_id"}
    cols: list[str] = []
    for c in df.columns:
        if c in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _derive_mask_seed(dataset_key: str, ratio: float, seed: int) -> int:
    token = f"{dataset_key}|{ratio:.6f}|{seed}"
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _write_artifacts(
    out_dir: Path,
    manifest: dict,
    masked_df: pd.DataFrame,
    mask_map: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "manifest.yaml").open("w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    for split, ids in manifest["splits"].items():
        frame = masked_df.loc[ids].copy()
        frame["row_id"] = frame.index
        frame.to_parquet(out_dir / f"{split}.parquet", index=False)

    mask_map.to_parquet(out_dir / "mask_map.parquet", index=False)


@app.command("prepare")
def prepare(
    dataset: str = typer.Option(..., help="Dataset key (dockstring)"),
    ratio: float = typer.Option(..., help="Mask ratio from 0.0 to 1.0"),
    seed: int = typer.Option(..., help="Experiment seed used for split/masking"),
    output_root: str = typer.Option(
        "artifacts/data/multitask_imputation",
        help="Root output directory for generated artifacts",
    ),
) -> None:
    if dataset != "dockstring":
        raise ValueError(
            f"Unsupported dataset key for multitask imputation prep: {dataset}. Expected 'dockstring'."
        )
    dataset_key = "dockstring"

    raw_df, split_indices, split_policy, target_columns = _load_dockstring_panel(seed=seed)

    if "sample_id" not in raw_df.columns:
        raw_df["sample_id"] = raw_df.index.map(lambda x: f"{dataset_key}_{x}")

    if len(target_columns) == 0:
        target_columns = _get_numeric_target_columns(raw_df)
    manifest = _build_manifest_from_explicit_splits(
        target_columns=target_columns,
        seed=seed,
        train_ids=np.asarray(split_indices["train"]),
        val_ids=np.asarray(split_indices["val"]),
        test_ids=np.asarray(split_indices["test"]),
        split_policy=split_policy,
    )

    mask_seed = _derive_mask_seed(dataset_key=dataset_key, ratio=ratio, seed=seed)
    masked_df, mask_map = apply_masking_from_manifest(
        df=raw_df,
        manifest=manifest,
        mask_ratio=ratio,
        seed=mask_seed,
        sample_id_col="sample_id",
        mask_splits=("train", "val", "test"),
    )

    out_dir = Path(output_root) / dataset_key / f"ratio={ratio:.1f}" / f"seed={seed}"
    _write_artifacts(out_dir, manifest, masked_df, mask_map)

    typer.echo(
        f"Prepared artifacts for dataset={dataset_key}, ratio={ratio:.1f}, seed={seed} at {out_dir}"
    )
    typer.echo(
        f"Split policy={split_policy}. Rows: train={len(manifest['splits']['train'])}, val={len(manifest['splits']['val'])}, test={len(manifest['splits']['test'])}; masked_cells={len(mask_map)}"
    )


if __name__ == "__main__":
    app()
