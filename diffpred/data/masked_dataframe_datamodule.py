from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from diffpred.constants.multitask_imputation import (
    MASK_MAP_ROW_ID_COL,
    MASK_MAP_SPLIT_COL,
    MASK_MAP_TARGET_COL,
    MASK_MAP_TRUE_VALUE_COL,
)
from diffpred.data.masked_imputation_artifacts import (
    validate_manifest_schema,
    validate_mask_map,
)
from diffpred.data.mpp_datamodule import MPPDatamodule

DOCKSTRING_TARGETS = [
    "IGF1R", "JAK2", "KIT", "LCK", "MAPK14",
    "MAPKAPK2", "MET", "PTK2", "PTPN1", "SRC",
]


class MaskedDataframeImputationDatamodule(MPPDatamodule):
    def __init__(
        self,
        dataset: str,
        seed: int,
        batch_size: int = 128,
        featurizer=None,
        standardize_targets: bool = False,
        log_transform: bool = False,
        use_benchmark: bool = False,
        masking: dict[str, Any] | None = None,
        sample_id_col: str = "sample_id",
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            featurizer=featurizer,
            standardize_targets=standardize_targets,
            log_transform=log_transform,
            seed=seed,
            use_benchmark=use_benchmark,
        )
        self.masking_info = masking or {}
        self.sample_id_col = sample_id_col
        self.use_benchmark = use_benchmark
        self.manifest: dict[str, Any] | None = None
        self.mask_map: pd.DataFrame | None = None
        self.masked_index_map: dict[str, list[tuple[Any, str]]] = {}
        self.original_targets_by_split: dict[str, torch.Tensor] = {}
        self.original_frames_by_split: dict[str, pd.DataFrame] = {}

    def setup(self, stage=None) -> None:
        if self.is_setup:
            return
        self.is_setup = True

        artifact_dir = self._resolve_artifact_dir()

        self.manifest = self._load_manifest(artifact_dir)
        validate_manifest_schema(self.manifest)
        self._validate_split_config()
        self.target_columns = list(self.manifest["target_columns"])

        split_frames = {
            split: self._load_split_table(artifact_dir, split)
            for split in ["train", "val", "test"]
        }

        split_frames = self._normalize_smiles_columns(split_frames)

        split_frames = self._align_indices_with_manifest(split_frames)
        self.mask_map = self._load_mask_map(artifact_dir)
        validate_mask_map(self.mask_map, sample_id_col=self.sample_id_col)

        self.masked_index_map = self._build_masked_index_map(self.mask_map)
        self.original_frames_by_split = self._build_original_split_frames(split_frames)
        self.original_targets_by_split = {
            split: torch.as_tensor(df.loc[:, self.target_columns].values)
            for split, df in self.original_frames_by_split.items()
        }

        self._assert_at_least_one_observed_target(split_frames)

        self.train_df = split_frames["train"]
        self.val_df = split_frames["val"]
        self.test_df = split_frames["test"]

        train_targets = self.train_df.loc[:, self.target_columns].values
        self.standardization_stats(train_targets)
        transform = self.get_transform(self.train_df["smiles"].values)

        self.train_dt, self.val_dt, self.test_dt = (
            self.df_to_dt(self.train_df, transform_op=transform, n_jobs=1),
            self.df_to_dt(self.val_df, transform_op=transform, n_jobs=1),
            self.df_to_dt(self.test_df, transform_op=transform, n_jobs=1),
        )

    def _extract_smiles_targets(self, df):
        smiles = df["smiles"].values
        targets = df.loc[:, self.target_columns].values
        return smiles, targets

    def _normalize_smiles_columns(
        self, split_frames: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        normalized: dict[str, pd.DataFrame] = {}
        for split, frame in split_frames.items():
            if "smiles" in frame.columns:
                normalized[split] = frame
                continue

            candidate_cols = [
                c
                for c in frame.columns
                if c.lower() in {"smiles", "original_smiles", "drug"}
            ]
            if len(candidate_cols) == 0:
                raise ValueError(
                    f"No SMILES-like input column found in {split} split. Expected one of smiles/ORIGINAL_SMILES/Drug."
                )

            selected = candidate_cols[0]
            normalized[split] = frame.rename(columns={selected: "smiles"})

        return normalized

    def get_original_targets(self, split: str) -> torch.Tensor:
        if split not in self.original_targets_by_split:
            raise ValueError(f"Unknown split {split}. Expected one of train/val/test")
        return self.original_targets_by_split[split]

    def _resolve_artifact_dir(self) -> Path:
        artifact_root = self.masking_info.get("artifact_root")
        if artifact_root is None:
            raise ValueError("masking.artifact_root must be provided")

        ratio = float(self.masking_info.get("ratio", 0.0))
        explicit_mask_seed = self.masking_info.get("seed", None)
        mask_seed = int(self.seed) if explicit_mask_seed is None else int(explicit_mask_seed)

        root = Path(artifact_root)
        direct_dir = root
        if direct_dir.exists() and (direct_dir / "manifest.yaml").exists():
            return direct_dir

        ratio_dir = root / f"ratio={ratio:.1f}" / f"seed={mask_seed}"
        if ratio_dir.exists():
            return ratio_dir

        raise FileNotFoundError(
            f"Could not find masking artifacts under {root} for ratio={ratio:.1f}, seed={mask_seed}"
        )

    def _load_manifest(self, artifact_dir: Path) -> dict[str, Any]:
        yaml_path = artifact_dir / "manifest.yaml"

        if yaml_path.exists():
            with yaml_path.open("r") as f:
                return yaml.safe_load(f)

        raise FileNotFoundError(f"No manifest found in {artifact_dir}")

    def _load_split_table(self, artifact_dir: Path, split: str) -> pd.DataFrame:
        parquet_path = artifact_dir / f"{split}.parquet"
        csv_path = artifact_dir / f"{split}.csv"

        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        if csv_path.exists():
            return pd.read_csv(csv_path)

        raise FileNotFoundError(f"Missing split table for {split} in {artifact_dir}")

    def _load_mask_map(self, artifact_dir: Path) -> pd.DataFrame:
        parquet_path = artifact_dir / "mask_map.parquet"
        csv_path = artifact_dir / "mask_map.csv"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(f"Missing mask_map table in {artifact_dir}")

    def _align_indices_with_manifest(
        self, split_frames: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        assert self.manifest is not None
        aligned: dict[str, pd.DataFrame] = {}

        for split, frame in split_frames.items():
            expected_ids = list(self.manifest["splits"][split])

            if MASK_MAP_ROW_ID_COL in frame.columns:
                frame = frame.set_index(MASK_MAP_ROW_ID_COL, drop=False)

            frame_ids = frame.index.tolist()
            if set(frame_ids) != set(expected_ids):
                raise ValueError(
                    f"{split} split rows do not match manifest: expected {len(expected_ids)} ids, got {len(frame_ids)}"
                )

            aligned[split] = frame.loc[expected_ids].copy()

        return aligned

    def _build_masked_index_map(
        self, mask_map: pd.DataFrame
    ) -> dict[str, list[tuple[Any, str]]]:
        index_map: dict[str, list[tuple[Any, str]]] = {"train": [], "val": [], "test": []}
        for split in index_map:
            split_df = mask_map.loc[
                mask_map[MASK_MAP_SPLIT_COL] == split,
                [MASK_MAP_ROW_ID_COL, MASK_MAP_TARGET_COL],
            ]
            index_map[split] = list(
                zip(
                    split_df[MASK_MAP_ROW_ID_COL].tolist(),
                    split_df[MASK_MAP_TARGET_COL].tolist(),
                )
            )
        return index_map

    def _build_original_split_frames(
        self, split_frames: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        originals = {split: frame.copy() for split, frame in split_frames.items()}
        assert self.mask_map is not None

        if len(self.mask_map) == 0:
            return originals

        for split, frame in originals.items():
            split_mask_map = self.mask_map.loc[
                self.mask_map[MASK_MAP_SPLIT_COL] == split,
                [MASK_MAP_ROW_ID_COL, MASK_MAP_TARGET_COL, MASK_MAP_TRUE_VALUE_COL],
            ]
            if split_mask_map.empty:
                continue

            row_pos = frame.index.get_indexer(split_mask_map[MASK_MAP_ROW_ID_COL])
            target_pos = pd.Index(self.target_columns).get_indexer(
                split_mask_map[MASK_MAP_TARGET_COL]
            )
            valid = (row_pos >= 0) & (target_pos >= 0)
            if not np.all(valid):
                raise ValueError(
                    f"mask_map contains unknown row_id/target references for split={split}"
                )

            restored_targets = frame.loc[:, self.target_columns].to_numpy(copy=True)
            restored_targets[row_pos, target_pos] = split_mask_map[
                MASK_MAP_TRUE_VALUE_COL
            ].to_numpy()
            frame.loc[:, self.target_columns] = restored_targets

        return originals

    def _assert_at_least_one_observed_target(
        self, split_frames: dict[str, pd.DataFrame]
    ) -> None:
        for split, frame in split_frames.items():
            observed_counts = frame.loc[:, self.target_columns].notna().sum(axis=1)
            if (observed_counts < 1).any():
                raise ValueError(
                    f"Found fully masked rows in {split} split. Each row must keep at least one observed target."
                )

    def _validate_split_config(self) -> None:
        if self.use_benchmark:
            raise ValueError("Masked dataframe imputation requires use_benchmark=false")
        if self.manifest is None:
            raise ValueError("Manifest must be loaded before validating masking constraints")
