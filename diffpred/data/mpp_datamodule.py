import os
from functools import partial
from pathlib import Path
import torch
import hashlib
import tqdm
import fsspec
import numpy as np
from joblib import Parallel, delayed
import math
import datamol as dm
from torch.utils.data import TensorDataset
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader as PyGLoader
import torch_geometric.transforms as T
from torch_geometric.transforms.to_sparse_tensor import ToSparseTensor
import lightning.pytorch as pl
from sklearn.model_selection import train_test_split
from molfeat.trans.fp import FPVecTransformer
from molfeat.calc.bond import EdgeMatCalculator
from molfeat.trans.graph import PYGGraphTransformer
from molfeat.trans.concat import FeatConcat
from diffpred.constants.literals import GRAPH_FEATURIZERS
from diffpred.utils.ops import nanstd, is_tensor_dicrete
from diffpred.models.networks.graph.graphormer_helpers.data import (
    GraphormerPreprocessTransform,
    collate_graphormer_items,
)
from diffpred.models.networks.graph.graphormer_helpers.data import OGBGraphormerFeaturizer

scratch_output_root = os.getenv("SCRATCH_OUTPUT_ROOT")
if not scratch_output_root:
    raise RuntimeError("SCRATCH_OUTPUT_ROOT is not set")
DATA_CACHE = Path(scratch_output_root) / "data" / "data_cache"


class MPPDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        featurizer,
        seed,
        batch_size: int = 512,
        standardize_targets: bool = True,
        log_transform: bool = False,
        smiles=None,
        targets=None,
        upsample_targets: bool = False,
        use_benchmark: bool = False,
        net_type: str | None = None,
    ):
        super().__init__()
        self.use_benchmark = use_benchmark
        self.seed = seed
        self.dataset = dataset
        self.batch_size = batch_size
        self.featurizer_name = featurizer
        self.standardize_targets = standardize_targets
        self.log_transform = log_transform
        self.is_setup = False
        self.data_mean, self.data_std = None, None
        self.log_target_offset = 1
        self.std_factor = 2  # standardize the data to 0.5 to match EDM
        self.upsample_targets = upsample_targets
        self._degrees = None
        self.net_type = net_type
        if smiles is not None and targets is not None:
            self.smiles, self.targets = smiles, targets

    def _get_featurizer(self, featurizer: str):
        if featurizer is None:
            return None
        if isinstance(featurizer, str) and featurizer == "graph":
            if self.net_type == "graphormer":
                return OGBGraphormerFeaturizer()
            return PYGGraphTransformer(
                self_loop=True,
                canonical_atom_order=True,
                bond_featurizer=EdgeMatCalculator(),
            )
        if isinstance(featurizer, str):
            return FPVecTransformer(kind=featurizer)
        elif hasattr(featurizer, "__iter__"):
            return FeatConcat(iterable=featurizer)
        else:
            raise ValueError(
                "Featurizer must be a string or an iterable of strings, not {}".format(
                    type(featurizer)
                )
            )

    @property
    def is_graph_model(self):
        return self.featurizer_name == "graph"

    def _min_num_nodes(self, graphs: list[Data | str]):
        if not self.is_graph_model:
            raise ValueError("Only graph models have a max number of nodes")
        if isinstance(graphs[0], str):
            return min([dm.to_mol(g).GetNumAtoms() for g in graphs])
        return min([g.num_nodes for g in graphs])  # type: ignore

    def get_featurizer(self):
        return self._get_featurizer(self.featurizer_name)

    def get_dataloader_cls(self):
        if self.featurizer_name in GRAPH_FEATURIZERS:
            if self.net_type == "graphormer":
                return partial(
                    torch.utils.data.DataLoader,
                    collate_fn=collate_graphormer_items,
                )
            return PyGLoader
        return torch.utils.data.DataLoader

    def get_batch_size(self) -> int:
        return len(self.train_dt) if self.batch_size is None else self.batch_size

    def setup(self, stage: str | None = None) -> None:
        """
        Download and/or set up the data and create the splits for training, validation, and testing.
        """
        if not self.is_setup:
            self.is_setup = True
            if self._has_cache():
                self._load_cache()
                return
            # split the data
            train_smiles, test_smiles, train_targets, test_targets = train_test_split(
                self.smiles,
                self.targets,
                test_size=0.1,
                random_state=self.seed,
            )
            train_smiles, val_smiles, train_targets, val_targets = train_test_split(
                train_smiles,
                train_targets,
                test_size=0.1,
                random_state=self.seed,
            )
            self.train_df = {"smiles": train_smiles, "targets": train_targets}
            self.val_df = {"smiles": val_smiles, "targets": val_targets}
            self.test_df = {"smiles": test_smiles, "targets": test_targets}

            self.target_dtype = train_targets.dtype
            if self.log_transform:
                self.log_min = torch.as_tensor(self.targets.min())

            self.standardization_stats(train_targets)
            transform_op = self.get_transform(self.smiles)
            self.train_dt, self.val_dt, self.test_dt = (
                self.df_to_dt(self.train_df, transform_op),
                self.df_to_dt(self.val_df, transform_op),
                self.df_to_dt(self.test_df, transform_op),
            )

            self._save_cache()

    def _extract_smiles_targets(self, df):
        smiles, targets = df["smiles"], df["targets"]
        return smiles, targets

    @property
    def feature_dim(self):
        sample = self.train_dt[0][0]
        if isinstance(sample, (torch.Tensor, np.ndarray)):
            return sample.shape[-1]
        else:  # consider it as a pyg graph
            return sample.x.shape[-1]

    @property
    def edge_feat_dim(self):
        sample = self.train_dt[0][0]
        if isinstance(sample, (torch.Tensor, np.ndarray)):
            return None
        else:  # consider it as a pyg graph
            return sample.edge_attr.shape[-1]

    @property
    def target_dim(self):
        return self.train_dt[0][1].shape[-1]

    @property
    def degree_histogram(self):
        """
        Compute histogram of in-degrees for the training dataset - used in PNA.
        Adapted from molfeat's example at:
        https://molfeat-docs.datamol.io/stable/tutorials/pyg_integration.html#dataset:~:text=as%20input%20to-,generate,-the%20data%20point
        """
        if not self.is_graph_model:
            return None

        if self._degrees is None:
            max_degree = -1
            for ix in range(len(self.train_dt)):
                data = self.train_dt[ix][0]
                d = degree(
                    data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long
                )
                max_degree = max(max_degree, int(d.max()))

            # Compute the in-degree histogram tensor
            deg = torch.zeros(max_degree + 1, dtype=torch.long)
            for ix in range(len(self.train_dt)):
                data = self.train_dt[ix][0]
                d = degree(
                    data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long
                )
                deg += torch.bincount(d, minlength=deg.numel())
            self._degrees = deg
        return self._degrees

    @property
    def shuffle_train(self):
        return False if (len(self.train_dt) == self.get_batch_size()) else True

    def _log_transform(self, targets):
        targets = targets - self.log_min  # positive values for log transform
        targets = torch.log(targets + self.log_target_offset)
        return targets

    def standardization_stats(self, targets):
        targets = torch.as_tensor(targets)
        vector = self._log_transform(targets) if self.log_transform else targets
        self.data_mean = vector.float().nanmean(0)
        self.data_std = nanstd(vector.float(), dim=0)

    def target_transform(self, targets):
        targets = torch.as_tensor(targets)
        self.orig_target_dtype = targets.dtype
        self.orig_targets_dicrete = is_tensor_dicrete(targets)
        if self.log_transform:
            targets = self._log_transform(targets)
        if self.standardize_targets:
            # divide by 2 to make the std 0.5 as in EDM
            targets = (targets - self.data_mean) / (self.std_factor * self.data_std)
        return targets

    def inverse_target_transform(self, targets):
        if targets.device != self.data_mean.device:
            self.data_mean = self.data_mean.to(targets.device)
            self.data_std = self.data_std.to(targets.device)

        if self.standardize_targets:
            # multiply by 2 to revert the standardization procedure
            targets = (targets * self.std_factor * self.data_std) + self.data_mean
        if self.log_transform:
            targets = torch.exp(targets) - self.log_target_offset
            targets = targets + self.log_min  # revert min

        if self.orig_targets_dicrete:
            targets = targets.round()

        targets = targets.to(dtype=self.orig_target_dtype)
        return targets

    def df_to_dt(self, df, transform_op, n_jobs: int = 1):
        smiles, targets = self._extract_smiles_targets(df)
        features, targets = self._featurize_format(smiles, targets, n_jobs=n_jobs)
        dataset = self._convert_to_dataset(features, targets, transform_op)

        return dataset

    def get_transform(self, graphs: list[Data]):
        if self.is_graph_model:
            self.min_num_nodes = self._min_num_nodes(graphs)
            if self.net_type == "graphormer":
                return T.Compose(
                    [
                        GraphormerPreprocessTransform(precompute_structural=True)
                    ]
                )
            return T.Compose(
                [
                    T.AddRandomWalkPE(walk_length=10, attr_name="rwse"),
                    ToSparseTensor(remove_edge_index=False),
                ]
            )
        else:
            self.min_num_nodes = None
            return None

    def _convert_to_dataset(self, features, targets, transform_op=None):
        if isinstance(features, (list, tuple)):
            if transform_op is not None:
                transforms = getattr(transform_op, "transforms", None)
                supports_progress = bool(getattr(transform_op, "supports_progress", False))
                if transforms is not None:
                    supports_progress = supports_progress or any(
                        bool(getattr(t, "supports_progress", False)) for t in transforms
                    )

                if supports_progress:
                    dataset = [
                        (transform_op(f), t)
                        for (f, t) in tqdm.tqdm(
                            zip(features, targets),
                            total=len(features),
                            desc="Applying graph transforms",
                            unit="graph",
                        )
                    ]
                else:
                    dataset = [(transform_op(f), t) for (f, t) in zip(features, targets)]
            else:
                dataset = [(f, t) for (f, t) in zip(features, targets)]
        else:
            dataset = TensorDataset(
                torch.as_tensor(features, dtype=torch.float32),
                torch.as_tensor(targets, dtype=torch.float32),
            )

        return dataset

    def _featurize_format(self, smiles, targets, n_jobs: int = 1):
        featurizer = self.get_featurizer()
        targets = self.target_transform(targets)
        # Use parallel processing if list is large enough
        if featurizer is not None and len(smiles) > 12000:  # Configurable threshold
            features = self._parallel_featurize(smiles, featurizer, n_jobs=n_jobs)
        else:
            features = featurizer(smiles) if featurizer is not None else smiles
        targets = (
            torch.as_tensor(targets, dtype=torch.float32).unsqueeze(-1)
            if targets.ndim == 1
            else torch.tensor(targets, dtype=torch.float32)
        )

        return features, targets

    def _parallel_featurize(
        self, smiles, featurizer, n_jobs: int = 1, batch_size=1024
    ):
        """Process SMILES strings in parallel using joblib.

        Args:
            smiles: List of SMILES strings to featurize
            featurizer: The featurizer object to use
            n_jobs: Number of parallel jobs (default: 1, single-process)
            batch_size: Size of batches for processing (default: 100)

        Returns:
            List of featurized molecules
        """

        # Define a processing function for a batch of SMILES
        def process_batch(batch):
            return featurizer(batch)

        # Split into batches
        n_batches = math.ceil(len(smiles) / batch_size)
        batches = [
            smiles[i * batch_size : (i + 1) * batch_size] for i in range(n_batches)
        ]

        print(
            f"Featurizing {len(smiles)} molecules in {n_batches} batches with {n_jobs} workers"
        )

        # Process batches in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_batch)(batch)
            for batch in tqdm.tqdm(
                batches, desc="Featurizing molecules", total=n_batches, unit="batch"
            )
        )

        # Combine results - handling different return types from featurizers
        if isinstance(results[0], list):
            # For featurizers that return lists (like PYGGraphTransformer)
            return [item for sublist in results for item in sublist]
        else:
            try:
                return np.vstack(results)
            except ValueError:
                # Fallback for other types
                return results

    def train_dataloader(self):
        dataloader_cls = self.get_dataloader_cls()
        if self.upsample_targets:
            train_targets = self.target_transform(
                torch.as_tensor(self.train_df["targets"])
            )
            hist, bins = torch.histogram(train_targets, bins=20)
            total_count = torch.sum(hist)
            bin_weights = total_count / hist
            weights = []
            for t in train_targets:
                for i in range(1, len(bins)):
                    if bins[i - 1] < t <= bins[i]:
                        weights.append(bin_weights[i - 1])
                        break
            weights = torch.tensor(weights)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, len(weights), replacement=True
            )

            return dataloader_cls(
                self.train_dt,
                batch_size=self.batch_size,
                shuffle=(False if self.upsample_targets else self.shuffle_train),
                sampler=sampler,
            )

        return dataloader_cls(
            self.train_dt, batch_size=self.get_batch_size(), shuffle=self.shuffle_train
        )

    def val_dataloader(self):
        dataloader_cls = self.get_dataloader_cls()
        return dataloader_cls(
            self.val_dt, batch_size=self.get_batch_size(), shuffle=False
        )

    def test_dataloader(self):
        dataloader_cls = self.get_dataloader_cls()
        return dataloader_cls(
            self.test_dt, batch_size=self.get_batch_size(), shuffle=False
        )

    def _get_cache_key(self):
        """
        Generate a unique cache key based on dataset, featurizer, seed, and standardize_targets.
        """
        key_str = f"{self.dataset}_{self.featurizer_name}_{self.seed}_{self.standardize_targets}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self):
        cache_dir = DATA_CACHE
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self._get_cache_key()
        return cache_dir / f"dt_cache_{cache_key}.pkl"

    def _has_cache(self):
        return self._get_cache_path().exists()

    def _load_cache(self):
        path = self._get_cache_path()
        print(f"Found cache at {path}, loading...")
        with fsspec.open(path, "rb") as f:
            cache = torch.load(f, weights_only=False)
        self.train_dt = cache["train_dt"]
        self.val_dt = cache["val_dt"]
        self.test_dt = cache["test_dt"]
        self.min_num_nodes = cache["min_num_nodes"]
        self.log_min = cache["log_min"]
        self.data_mean = cache["data_mean"]
        self.data_std = cache["data_std"]
        self.orig_target_dtype = cache["orig_target_dtype"]
        self.orig_targets_dicrete = cache["orig_targets_dicrete"]
        self.target_columns = cache["target_columns"]
        if "benchmark" in cache:
            self.benchmark = cache["benchmark"]
        return

    def _save_cache(self):
        cache = {
            "train_dt": self.train_dt,
            "val_dt": self.val_dt,
            "test_dt": self.test_dt,
            "min_num_nodes": self.min_num_nodes,
            "log_min": (self.log_min if hasattr(self, "log_min") else None),
            "data_mean": self.data_mean,
            "data_std": self.data_std,
            "orig_target_dtype": self.orig_target_dtype,
            "orig_targets_dicrete": self.orig_targets_dicrete,
            "target_columns": (
                self.target_columns if hasattr(self, "target_columns") else None
            ),
            "benchmark": (self.benchmark if hasattr(self, "benchmark") else None),
        }
        with fsspec.open(self._get_cache_path(), "wb") as f:
            torch.save(cache, f)
