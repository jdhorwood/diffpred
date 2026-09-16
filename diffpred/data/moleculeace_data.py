import torch
from sklearn.model_selection import train_test_split
import numpy as np
from MoleculeACE import Data
from MoleculeACE import datasets as AVAILABLE_DATASETS
from diffpred.data.mpp_datamodule import MPPDatamodule


class MoleculeAceDatamodule(MPPDatamodule):
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

    def setup(self, stage=None) -> None:
        if not self.is_setup:
            self.is_setup = True
            # download data
            if self.dataset in AVAILABLE_DATASETS:
                self.data = Data(self.dataset)
            else:
                raise ValueError(
                    f"Dataset {self.dataset} is not available. Please choose from {AVAILABLE_DATASETS}."
                )

            # split the data
            cliff_index = self._get_cliff_index()
            train_smiles, val_smiles, train_targets, val_targets = train_test_split(
                self.data.smiles_train,
                self.data.y_train,
                test_size=0.1,
                random_state=self.seed,
            )
            self.train = {"smiles": train_smiles, "targets": train_targets}
            self.val = {"smiles": val_smiles, "targets": val_targets}
            self.test = {"smiles": self.data.smiles_test, "targets": self.data.y_test}
            self.cliff_test = {
                "smiles": np.array(self.data.smiles_test)[cliff_index],
                "targets": np.array(self.data.y_test)[cliff_index],
            }

            if self.log_transform:
                self.log_min = np.min(
                    np.min(self.train["targets"]),
                    np.min(self.val["targets"]),
                    np.min(self.test["targets"]),
                )

            self.target_dtype = torch.as_tensor(self.train["targets"]).dtype
            transform_op = self.get_transform(self.train["smiles"])
            if self.standardize_targets:
                self.standardization_stats(np.array(self.train["targets"]))
            self.train_dt, self.val_dt, self.test_dt, self.cliff_test_dt = (
                self.df_to_dt(self.train, transform_op),
                self.df_to_dt(self.val, transform_op),
                self.df_to_dt(self.test, transform_op),
                self.df_to_dt(self.cliff_test, transform_op),
            )

    def _extract_smiles_targets(self, df):
        """
        Extracts the smiles and targets from a dataframe. Dependent on specific formatting from data source.
        """
        return np.array(df["smiles"]), np.array(df["targets"])

    def _get_cliff_index(self):
        if not self.is_setup:
            raise ValueError("Data has not been setup yet. Please call setup() first.")

        assert len(self.data.cliff_mols_test) == len(
            self.data.smiles_test
        ), "Cliff mols and test smiles are not the same length. Something went wrong."
        cliff_index = [i for i, cliff in enumerate(self.data.cliff_mols_test) if cliff]
        return cliff_index

    def test_dataloader(self):
        dataloader_cls = self.get_dataloader_cls()
        test_loader = dataloader_cls(
            self.test_dt, batch_size=self.batch_size, shuffle=False
        )
        cliff_test_loader = dataloader_cls(
            self.cliff_test_dt, batch_size=self.batch_size, shuffle=False
        )
        return [test_loader, cliff_test_loader]
