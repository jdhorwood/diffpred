import os
from pathlib import Path
from tdc.single_pred import ADME, Tox
from tdc.metadata import adme_dataset_names, toxicity_dataset_names
from tdc.benchmark_group import admet_group
from diffpred.data.mpp_datamodule import MPPDatamodule
import pandas as pd
from sklearn.model_selection import train_test_split


class TDCDatamodule(MPPDatamodule):
    DATA_PATH = Path.joinpath(Path.home(), Path("data/Datasets/tdc/"))

    def __init__(
        self,
        dataset,
        seed,
        batch_size=128,
        featurizer=None,
        standardize_targets: bool = False,
        log_transform: bool = False,
        use_benchmark: bool = False,
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
        os.makedirs(TDCDatamodule.DATA_PATH, exist_ok=True)

    def setup(self, stage=None) -> None:
        if not self.is_setup:
            self.is_setup = True
            # download data
            if (
                self.dataset in adme_dataset_names
                or self.dataset in toxicity_dataset_names
            ):
                cls_name = ADME if self.dataset in adme_dataset_names else Tox
                if not self.use_benchmark:
                    self.data = cls_name(
                        name=self.dataset, path=TDCDatamodule.DATA_PATH.as_posix()
                    )

                    self.split = self.data.get_split(
                        frac=[0.8, 0.1, 0.1], seed=self.seed
                    )
                else:
                    self.benchmark_group = admet_group(
                        path=TDCDatamodule.DATA_PATH.as_posix()
                    )
                    benchmark = self.benchmark_group.get(self.dataset)
                    train_val, test = benchmark["train_val"], benchmark["test"]
                    train, val = train_test_split(
                        train_val, test_size=0.1, random_state=self.seed
                    )
                    self.split = {"train": train, "valid": val, "test": test}

            self.train_df, self.val_df, self.test_df = (
                self.split["train"],
                self.split["valid"],
                self.split["test"],
            )
            self.data = pd.concat([self.train_df, self.val_df, self.test_df])

            self.target_dtype = self.train_df.Y.dtype  # type: ignore
            if self.log_transform:
                self.log_min = self.train_df.Y.min()  # type: ignore

            transform_op = self.get_transform(self.train_df["Drug"].values)
            self.standardization_stats(self.train_df["Y"].values)
            self.train_dt, self.val_dt, self.test_dt = (
                self.df_to_dt(self.train_df, transform_op),
                self.df_to_dt(self.val_df, transform_op),
                self.df_to_dt(self.test_df, transform_op),
            )

    def _extract_smiles_targets(self, df):
        smiles, targets = df["Drug"].values, df["Y"].values
        return smiles, targets

    def benchmark_eval(self, preds):
        predictions = {self.dataset: preds.cpu().numpy()}
        res = self.benchmark_group.evaluate(predictions)
        return res
