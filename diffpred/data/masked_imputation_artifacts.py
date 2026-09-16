from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from diffpred.constants.multitask_imputation import (
    MANIFEST_REQUIRED_SPLITS,
    MANIFEST_REQUIRED_TOP_LEVEL_KEYS,
    MASK_MAP_IS_MASKED_COL,
    MASK_MAP_RATIO_COL,
    MASK_MAP_REQUIRED_COLUMNS,
    MASK_MAP_ROW_ID_COL,
    MASK_MAP_SAMPLE_ID_COL,
    MASK_MAP_SEED_COL,
    MASK_MAP_SPLIT_COL,
    MASK_MAP_TARGET_COL,
    MASK_MAP_TRUE_VALUE_COL,
)


def apply_masking_from_manifest(
    df: pd.DataFrame,
    manifest: dict[str, Any],
    mask_ratio: float,
    seed: int,
    sample_id_col: str = "sample_id",
    mask_splits: tuple[str, ...] = ("train", "val", "test"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mask_ratio < 0.0 or mask_ratio > 1.0:
        raise ValueError("mask_ratio must be between 0.0 and 1.0")

    validate_manifest_schema(manifest)

    target_columns = list(manifest["target_columns"])
    split_to_ids = manifest["splits"]

    if sample_id_col not in df.columns:
        working_df = df.copy()
        working_df[sample_id_col] = working_df.index
    else:
        working_df = df.copy()

    masked_df = working_df.copy()
    rng = np.random.default_rng(seed)

    mask_rows: list[dict[str, Any]] = []
    mask_splits_set = set(mask_splits)

    for split_name, split_ids in split_to_ids.items():
        if split_name not in mask_splits_set:
            continue

        split_view = masked_df.loc[split_ids, target_columns]
        split_values = split_view.to_numpy(copy=True)

        for row_offset, row_id in enumerate(split_ids):
            row_values = split_values[row_offset]
            observed_target_ix = np.flatnonzero(~pd.isna(row_values))
            if observed_target_ix.size == 0:
                continue

            candidate_count = observed_target_ix.size
            if candidate_count <= 1:
                continue

            # Sample each observed target independently so low ratios (e.g. 0.1)
            # do not collapse to zero masks under integer rounding.
            chosen_flags = rng.random(candidate_count) < mask_ratio
            if not np.any(chosen_flags):
                continue

            # Keep at least one observed target per row.
            if np.all(chosen_flags):
                keep_idx = int(rng.integers(0, candidate_count))
                chosen_flags[keep_idx] = False

            chosen_ix = observed_target_ix[chosen_flags]
            for target_ix in chosen_ix:
                target_col = target_columns[int(target_ix)]
                y_true = row_values[target_ix]
                masked_df.at[row_id, target_col] = np.nan
                mask_rows.append(
                    {
                        MASK_MAP_SPLIT_COL: split_name,
                        sample_id_col: masked_df.at[row_id, sample_id_col],
                        MASK_MAP_ROW_ID_COL: row_id,
                        MASK_MAP_TARGET_COL: target_col,
                        MASK_MAP_IS_MASKED_COL: True,
                        MASK_MAP_TRUE_VALUE_COL: y_true,
                        MASK_MAP_RATIO_COL: float(mask_ratio),
                        MASK_MAP_SEED_COL: int(seed),
                    }
                )

    mask_map = pd.DataFrame(mask_rows)
    if len(mask_map) == 0:
        mask_map = pd.DataFrame(
            columns=[
                MASK_MAP_SPLIT_COL,
                sample_id_col,
                MASK_MAP_ROW_ID_COL,
                MASK_MAP_TARGET_COL,
                MASK_MAP_IS_MASKED_COL,
                MASK_MAP_TRUE_VALUE_COL,
                MASK_MAP_RATIO_COL,
                MASK_MAP_SEED_COL,
            ]
        )

    validate_mask_map(mask_map, sample_id_col=sample_id_col)
    return masked_df, mask_map


def validate_manifest_schema(manifest: dict[str, Any]) -> None:
    missing_top_level = [
        k for k in MANIFEST_REQUIRED_TOP_LEVEL_KEYS if k not in manifest
    ]
    if missing_top_level:
        raise ValueError(f"manifest missing required keys: {missing_top_level}")

    splits = manifest["splits"]
    missing_splits = [split for split in MANIFEST_REQUIRED_SPLITS if split not in splits]
    if missing_splits:
        raise ValueError(f"manifest missing required splits: {missing_splits}")


def validate_mask_map(mask_map: pd.DataFrame, sample_id_col: str = "sample_id") -> None:
    required_columns = [
        c if c != MASK_MAP_SAMPLE_ID_COL else sample_id_col
        for c in MASK_MAP_REQUIRED_COLUMNS
    ]
    missing = [c for c in required_columns if c not in mask_map.columns]
    if missing:
        raise ValueError(f"mask_map missing required columns: {missing}")
