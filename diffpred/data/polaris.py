import os

import torch
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import cast
try:
    from polaris.dataset._subset import Subset # import if available - not all users will have polaris installed
except ImportError:
    Subset = None
from diffpred.data.mpp_datamodule import MPPDatamodule
from sklearn.model_selection import train_test_split
from diffpred.utils.ops import is_tensor_binary, convert_to_symmetric_binary


@dataclass
class _LocalTableDataset:
    table: pd.DataFrame


class _LocalSubset:
    def __init__(
        self,
        dataset,
        indices,
        input_cols,
        target_cols,
        adapters=None,
        featurization_fn=None,
    ):
        self.dataset = dataset
        self.indices = list(indices)
        self.input_cols = list(input_cols)
        self.target_cols = list(target_cols)
        self._adapters = adapters
        self._featurization_fn = featurization_fn

    @property
    def inputs(self):
        return self.dataset.table.loc[self.indices, self.input_cols[0]].tolist()


class PolarisDatamodule(MPPDatamodule):
    DATA_PATH = Path.joinpath(Path.home(), Path("data/Datasets/polaris/"))
    BENCHMARK_SPLIT_FILE = "benchmark_split.csv"
    BENCHMARK_TARGET_COLUMNS_FILE = "target_columns.txt"

    def __init__(
        self,
        dataset,
        seed,
        batch_size=128,
        featurizer=None,
        standardize_targets: bool = False,
        log_transform: bool = False,
        use_benchmark: bool = True,
        symmetric_binary_targets: bool = True,
        save_benchmark_dataset: bool = True,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            featurizer=featurizer,
            standardize_targets=standardize_targets,
            log_transform=log_transform,
            seed=seed,
        )
        self.use_benchmark = use_benchmark
        self.symmetric_binary_targets = symmetric_binary_targets  # if True, convert binary targets to symmetric binary (for diffusion models)
        self.save_benchmark_dataset = save_benchmark_dataset
        os.makedirs(PolarisDatamodule.DATA_PATH, exist_ok=True)

    def setup(self, stage=None) -> None:
        if not self.is_setup:
            self.is_setup = True

            # download data
            if self.use_benchmark:
                benchmark_subsets = self._prepare_benchmark_data(load_subsets=True)
                if benchmark_subsets is None:
                    raise RuntimeError(
                        "Expected benchmark subsets to be prepared during setup"
                    )
                train, val, test = benchmark_subsets

            else:
                # lazy import for speed
                import polaris as po

                self.data = po.load_dataset(self.dataset)
                train_ix, val_ix, test_ix = self.random_split_index(self.data.table)
                if self.symmetric_binary_targets:
                    self.data = self._handle_possibly_binary_targets(self.data)
                train = self.create_subset(train_ix, self.data)
                val = self.create_subset(val_ix, self.data)
                test = self.create_subset(test_ix, self.data)

            self.target_columns = train.target_cols
            train_targets = self._extract_targets_as_tensor(
                dataframe=train.dataset.table.loc[list(train.indices), :]
            )
            self.standardization_stats(train_targets)
            transform = self.get_transform(cast(list, train.inputs))
            self.train_dt, self.val_dt, self.test_dt = (
                self.df_to_dt(train, transform_op=transform),
                self.df_to_dt(val, transform_op=transform),
                self.df_to_dt(test, transform_op=transform),
            )

    def prepare_benchmark_split_cache(self) -> None:
        if not self.use_benchmark:
            raise ValueError("prepare_benchmark_split_cache requires use_benchmark=True")
        self._prepare_benchmark_data(load_subsets=False)

    def _prepare_benchmark_data(self, load_subsets: bool = True):
        cached_benchmark_data = self._load_saved_benchmark_split()
        if cached_benchmark_data is not None:
            self.benchmark = None
            if not load_subsets:
                return None

            table, target_cols = cached_benchmark_data
            return self._create_benchmark_subsets_from_saved_table(
                table=table,
                target_cols=target_cols,
            )

        # lazy import for speed and optional dependency behavior
        import polaris as po

        self.benchmark = po.load_benchmark(self.dataset)
        train, val, test = self._train_val_test_benchmark_split(self.benchmark)
        if not load_subsets:
            return None
        return train, val, test

    def _benchmark_seed_dir(self) -> Path:
        return PolarisDatamodule.DATA_PATH / self.dataset / str(self.seed)

    def _benchmark_split_cache_path(self) -> Path:
        return self._benchmark_seed_dir() / PolarisDatamodule.BENCHMARK_SPLIT_FILE

    def _benchmark_target_columns_path(self) -> Path:
        return (
            self._benchmark_seed_dir()
            / PolarisDatamodule.BENCHMARK_TARGET_COLUMNS_FILE
        )

    def _load_saved_benchmark_split(self):
        split_path = self._benchmark_split_cache_path()
        target_cols_path = self._benchmark_target_columns_path()
        if not split_path.exists() or not target_cols_path.exists():
            return None

        table = pd.read_csv(split_path, index_col=0)
        with open(target_cols_path, "r", encoding="utf-8") as f:
            target_cols = [line.strip() for line in f if line.strip()]

        return table, target_cols

    def _create_benchmark_subsets_from_saved_table(
        self, table: pd.DataFrame, target_cols: list
    ):
        table = table.copy()
        if "split" not in table.columns:
            raise ValueError("Cached benchmark split table is missing the 'split' column")

        dataset = _LocalTableDataset(table=table)
        train_ix = table.index[table["split"] == "train"].tolist()
        val_ix = table.index[table["split"] == "val"].tolist()
        test_ix = table.index[table["split"] == "test"].tolist()
        if not train_ix or not val_ix or not test_ix:
            raise ValueError("Cached benchmark split table must contain train/val/test rows")

        return (
            self.create_subset(
                train_ix,
                dataset,
                input_cols=["smiles"],
                target_cols=target_cols,
            ),
            self.create_subset(
                val_ix,
                dataset,
                input_cols=["smiles"],
                target_cols=target_cols,
            ),
            self.create_subset(
                test_ix,
                dataset,
                input_cols=["smiles"],
                target_cols=target_cols,
            ),
        )

    def _maybe_save_benchmark_split(
        self,
        benchmark,
        train_ix,
        val_ix,
        test_ix,
        target_cols: list,
    ) -> None:
        if not self.save_benchmark_dataset:
            return

        split_path = self._benchmark_split_cache_path()
        target_cols_path = self._benchmark_target_columns_path()
        if split_path.exists() and target_cols_path.exists():
            return

        seed_dir = self._benchmark_seed_dir()
        seed_dir.mkdir(parents=True, exist_ok=True)

        df = benchmark.dataset.table.copy()
        # Ensure SMILES column is named 'smiles' (lowercase)
        smiles_col = next((c for c in df.columns if c.lower() == "smiles"), None)
        if smiles_col is not None and smiles_col != "smiles":
            df.rename(columns={smiles_col: "smiles"}, inplace=True)

        df.loc[:, "split"] = ""
        df.loc[list(train_ix), "split"] = "train"
        df.loc[list(val_ix), "split"] = "val"
        df.loc[list(test_ix), "split"] = "test"
        df.to_csv(split_path)
        with open(target_cols_path, "w", encoding="utf-8") as f:
            f.write("\n".join(target_cols))

    def _handle_possibly_binary_targets(self, dataset):
        target_cols = self._get_target_cols(dataset)
        target_values = torch.as_tensor(dataset.table.loc[:, target_cols].values)
        if is_tensor_binary(target_values):
            dataset.table.loc[:, target_cols] = convert_to_symmetric_binary(
                target_values
            ).numpy()

        return dataset

    def _extract_smiles_targets(self, df):
        smiles = df.inputs
        # access table directly as highly inefficient to call point-wise polaris accessor
        targets = df.dataset.table.loc[df.indices, df.target_cols]
        targets = torch.as_tensor(targets.values)
        return smiles, targets

    def _get_target_cols(self, dataset) -> list:
        target_cols = [
            c
            for c in dataset.table.columns
            if c not in set(["SMILES", "ORIGINAL_SMILES", "UNIQUE_ID"])
        ]

        return target_cols

    @staticmethod
    def _get_dataset_targets(df: pd.DataFrame, target_cols: list) -> torch.Tensor:
        return torch.as_tensor(df.loc[:, target_cols].values)

    @staticmethod
    def _set_dataset_targets(
        df: pd.DataFrame, targets: torch.Tensor, target_cols: list
    ) -> pd.DataFrame:
        df.loc[:, target_cols] = targets.numpy()
        return df

    def _train_val_test_benchmark_split(self, benchmark):
        if Subset is None:
            raise ImportError(
                "polaris is required for benchmark split creation when cache files are absent"
            )

        train, test = self._get_benchmark_train_test_split(benchmark)
        train_ix, val_ix = train_test_split(
            train.indices, test_size=0.1, random_state=self.seed
        )
        self._maybe_save_benchmark_split(
            benchmark=benchmark,
            train_ix=train_ix,
            val_ix=val_ix,
            test_ix=test.indices,
            target_cols=train.target_cols,
        )
        train = self.create_subset(
            train_ix,
            dataset=benchmark.dataset,
            input_cols=train.input_cols,
            target_cols=train.target_cols,
            adapters=train._adapters,
            featurization_fn=train._featurization_fn,
        )
        val = self.create_subset(
            val_ix,
            dataset=benchmark.dataset,
            input_cols=train.input_cols,
            target_cols=train.target_cols,
            adapters=train._adapters,
            featurization_fn=train._featurization_fn,
        )

        assert len(set(train.indices).intersection(set(val.indices))) == 0
        assert len(set(train.indices).intersection(set(test.indices))) == 0

        return train, val, test

    def _get_benchmark_train_test_split(self, benchmark):
        split = benchmark.get_train_test_split()
        if isinstance(split, dict):
            return split["train"], split["test"]
        return split

    def _extract_targets_as_tensor(self, dataframe: pd.DataFrame):
        targets = dataframe.loc[:, self.target_columns]
        return torch.as_tensor(targets.values)

    def benchmark_eval(self, preds):
        if self.benchmark is None:
            import polaris as po

            self.benchmark = po.load_benchmark(self.dataset)
        prediction_dict = {
            col: preds[:, i].cpu() for i, col in enumerate(self.target_columns)
        }
        try:
            results = self.benchmark.evaluate(prediction_dict)
        except ValueError as e:
            print("Polaris evaluation failed. Error message:")
            print(e)
            results = {}
        return results

    def random_split_index(self, dataframe: pd.DataFrame, test_size=0.1):
        train_ix, test_ix = train_test_split(
            dataframe.index, test_size=test_size, random_state=self.seed
        )
        train_ix, val_ix = train_test_split(
            train_ix, test_size=test_size, random_state=self.seed
        )

        return train_ix, val_ix, test_ix

    def create_subset(
        self,
        indices,
        dataset,
        input_cols: list | None = None,
        target_cols: list | None = None,
        adapters=None,
        featurization_fn=None,
    ):
        if input_cols is None:
            input_cols = ["smiles"]
        if target_cols is None:
            target_cols = self._get_target_cols(dataset)

        # Use Polaris Subset for native Polaris datasets and a local equivalent for cached tables.
        if Subset is not None and not isinstance(dataset, _LocalTableDataset):
            return Subset(
                dataset=dataset,
                indices=indices,
                input_cols=input_cols,
                target_cols=target_cols,
                adapters=adapters,
                featurization_fn=featurization_fn,
            )

        return _LocalSubset(
            dataset=dataset,
            indices=indices,
            input_cols=input_cols,
            target_cols=target_cols,
            adapters=adapters,
            featurization_fn=featurization_fn,
        )
