import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from diffpred.data.mpp_datamodule import MPPDatamodule
from pathlib import Path


class DockstringDatamodule(MPPDatamodule):
    DATA_PATH = Path.joinpath(Path.home(), Path("data/Datasets/dockstring/"))
    DATASET_NAME = "dockstring-dataset.tsv"
    DATASET_URL = "https://figshare.com/ndownloader/files/35948138"

    def __init__(
        self,
        dataset,
        seed,
        batch_size=128,
        featurizer=None,
        standardize_targets: bool = False,
        log_transform: bool = False,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            featurizer=featurizer,
            standardize_targets=standardize_targets,
            log_transform=log_transform,
            seed=seed,
        )
        os.makedirs(DockstringDatamodule.DATA_PATH, exist_ok=True)

    def setup(self, stage=None) -> None:
        if not self.is_setup:
            self.is_setup = True
            if self._has_cache():
                self._load_cache()
                return

            dataset_path = os.path.join(
                DockstringDatamodule.DATA_PATH, DockstringDatamodule.DATASET_NAME
            )
            if not os.path.isfile(dataset_path):
                import requests
                print(f"Downloading Dockstring dataset from {DockstringDatamodule.DATASET_URL}...")
                response = requests.get(DockstringDatamodule.DATASET_URL)
                response.raise_for_status()
                with open(dataset_path, "wb") as f:
                    f.write(response.content)

            self.data = pd.read_csv(
                dataset_path,
                sep="\t",
            )

            # keep only the relevant target data
            self.data = self.data.loc[:, ["smiles", self.dataset]]

            # Split the data into train, validation, and test sets
            train_val, test = train_test_split(
                self.data, test_size=0.2, random_state=0, shuffle=False # Ensure the test split is deterministic across runs
            )
            test = test.sort_values("smiles").reset_index(drop=True)
            train, val = train_test_split(
                train_val, test_size=0.1, random_state=self.seed
            )

            if self.log_transform:
                self.log_min = np.min(self.get_targets(self.data))

            self.target_dtype = torch.as_tensor(self.get_targets(train)).dtype
            transform_op = self.get_transform(self.get_smiles(train))
            if self.standardize_targets:
                self.standardization_stats(self.get_targets(train))

            self.train_dt = self.df_to_dt(train, transform_op)
            self.val_dt = self.df_to_dt(val, transform_op)
            self.test_dt = self.df_to_dt(test, transform_op)

            self._save_cache()

    def get_smiles(self, df):
        return df["smiles"].values

    def get_targets(self, df):
        return df[self.dataset].values

    def _extract_smiles_targets(self, df):
        return self.get_smiles(df), self.get_targets(df)
