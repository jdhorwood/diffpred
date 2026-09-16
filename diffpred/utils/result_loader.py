import os
import os.path as osp
import tqdm
import fsspec
import numpy as np
import pandas as pd
import glob
import yaml
import typer
import re
import torch
from pathlib import Path
from typing import List
from diffpred.constants.metrics import (
    REGRESSION_METRICS,
    CLASSIFICATION_METRICS,
    compute_metrics,
)


class ResultLoader:
    def __init__(self, path: str, is_multirun: bool = False):
        self.path = path
        self.is_multirun = is_multirun

    @staticmethod
    def get_results_path(run_path: str) -> str | None:
        paths = glob.glob(
            "**/results.yaml", root_dir=run_path, recursive=True, include_hidden=True
        )
        if len(paths) == 0:
            # raise a warning
            print(f"WARNING: No results file found in {run_path}")
            return None
        if len(paths) > 1:
            raise ValueError(f"Multiple results files found in {run_path}")

        return osp.join(run_path, paths[0])

    def _get_multirun_paths(self) -> list[str]:
        if not self.is_multirun:
            raise ValueError(
                "This is not a multirun directory according to init settings."
            )

        all_paths = glob.glob(f"{self.path}/*", recursive=True)
        run_paths = [path for path in all_paths if osp.isdir(path)]
        return run_paths

    def load_run_result(self, run_path: str) -> dict | None:
        """
        Load the results yaml file from the given directory path.

        Args:
            dir_path (str): The directory where the results file is located.

        Returns:
            dict: The loaded results.
        """
        results_path = self.get_results_path(run_path)
        if results_path is None:
            return None

        with open(results_path, "r") as f:
            results = yaml.safe_load(f)

        return results

    def get_multirun_results(self) -> dict:
        """
        Load the results from all runs in the multirun directory.

        Returns:
            dict: The loaded results.
        """
        if not self.is_multirun:
            raise ValueError(
                "This is not a multirun directory according to init settings."
            )

        results = {}
        run_paths = self._get_multirun_paths()
        for run_path in run_paths:
            dataset, method = self.get_run_info(run_path)
            # keep only last part of path
            path_key = run_path.split("/")[-1]
            results[path_key] = self.load_run_result(run_path)
            if results[path_key] is not None:
                results[path_key]["dataset"] = dataset  # type: ignore
                results[path_key]["method"] = method  # type: ignore
            else:
                print(f"WARNING: No results found for run at {run_path}")

        return results

    def get_uncertainty_method(self, run_path) -> str:
        with fsspec.open(f"{run_path}/.hydra/config.yaml", "r") as f:
            config = yaml.safe_load(f)
            arch_type = config["model"]["hparams"].get("arch_type", None)

            if config["model"].get("use_mve", False):
                method = "MVE"
            elif config["model"].get("use_mcdropout", False):
                method = "MC-Dropout"
            elif config["model"].get("use_deepensemble", False):
                method = "Deep Ensemble"
            else:
                if arch_type is not None:
                    method = "DIFFPRED"
                else:
                    raise ValueError(
                        "No uncertainty method found in the config file. "
                        "Please check the model configuration."
                    )

        return method

    def get_run_info(self, run_path) -> tuple[str, str]:
        with fsspec.open(f"{run_path}/.hydra/config.yaml", "r") as f:
            config = yaml.safe_load(f)
            dataset = config["data"]["dataset"]
            net_type, arch_type, name = (
                config["model"]["hparams"].get("net_type", None),
                config["model"]["hparams"].get("arch_type", None),
                config["model"]["name"],
            )
            if arch_type is not None:
                method = net_type + "_diff" if net_type is not None else name + "_diff"
            else:
                method = net_type if net_type is not None else name
        return dataset, method

    def get_score_table(self, score_col: str) -> pd.DataFrame:
        """
        Create a DataFrame where the index is the method, the columns are the dataset,
        and the entry is the value for the unique column containing score_col.

        Args:
            score_col (str): The column name containing the score.

        Returns:
            pd.DataFrame: The score matrix.
        """
        if not self.is_multirun:
            raise ValueError(
                "This is not a multirun directory according to init settings."
            )

        results = self.get_multirun_results()
        records = []
        for val in results.values():
            dataset = val.get("dataset")
            method = val.get("method")
            # Find the unique column containing score_col
            score = None
            test_val = val.get("test")
            if len(test_val) > 1:
                raise ValueError(
                    f"Multiple test results found for method {method}, dataset {dataset}"
                )
            test_val = test_val[0]  # there can be multiple test dataloaders
            for k, v in test_val.items():
                if score_col == k:
                    score = v
                    break
            if score is None:
                raise ValueError(
                    f"No column containing '{score_col}' found for method {method}, dataset {dataset}"
                )
            records.append({"method": method, "dataset": dataset, "score": score})

        df = pd.DataFrame(records)
        score_matrix = df.pivot(index="method", columns="dataset", values="score")
        return score_matrix


class MultiSeedAggregator:
    def __init__(self, multiseed_results: dict, metrics: list[str]):
        """
        Class for processing the results from multiple seeds into a cleaned format for analysis and export.
        Args:
            multiseed_results (dict): The results from multiple seeds, where each key is a run id and the value is the results dict (for val and test).

        """
        self.metrics = metrics
        self.multiseed_results = multiseed_results

        self.dataset = self._check_dataset()
        self.method = self._check_method()

    def _check_dataset(self) -> str:
        """
        Check that all runs have the same dataset.
        Returns:
            str: The dataset name.
        """
        datasets = set(
            [result["dataset"] for result in self.multiseed_results.values()]
        )
        if len(datasets) > 1:
            raise ValueError("Multiple datasets found in results.")

        return datasets.pop()

    def _check_method(self) -> str:
        """
        Check that all runs have the same method.
        Returns:
            str: The method name.
        """
        methods = set([result["method"] for result in self.multiseed_results.values()])
        if len(methods) > 1:
            raise ValueError("Multiple methods found in results.")

        return methods.pop()

    def _get_phase_metric(
        self, phase_results: dict | list, metric: str
    ) -> float | dict:
        """
        Get a specific metric from the results for a given phase.
        Args:
            phase_results (dict): The results for the phase.
            metric (str): The metric to extract.
        Returns:
            float: The metric value.
        """
        if isinstance(phase_results, dict):
            for key, val in phase_results.items():
                if metric in key:
                    return val
            raise ValueError(f"Metric {metric} not found in phase results.")
        elif isinstance(phase_results, list):
            return self._get_phase_metric(phase_results[0], metric)
        else:
            raise ValueError("Phase results should be a dict or list.")

    def _get_metric(self, metric: str, phase: str) -> list[float | dict]:
        """
        Get a specific metric from the results for a given phase.
        Args:
            metric (str): The metric to extract.
            phase (str): The phase to extract the metric from.
        Returns:
            list[float]: The list of metric values.
        """

        return [
            self._get_phase_metric(result[phase], metric)
            for result in self.multiseed_results.values()
        ]

    def get_metric(self, metric: str) -> dict:
        """
        Get a specific metric from the results for both val and test phases.
        Args:
            metric (str): The metric to extract.
        Returns:
            dict: The dictionary of metric values for val and test phases.
        """
        return {
            "val": self._get_metric(metric, "val"),
            "test": self._get_metric(metric, "test"),
        }

    def get_mean_metric(self, metric: str, round: bool = True) -> dict:
        """
        Get the mean of a specific metric from the results for both val and test phases.
        Args:
            metric (str): The metric to extract.
        Returns:
            dict: The dictionary of mean metric values for val and test phases.
        """
        val_mean = np.mean(self._get_metric(metric, "val"))
        test_mean = np.mean(self._get_metric(metric, "test"))
        if round:
            val_mean = np.round(val_mean, 3)
            test_mean = np.round(test_mean, 3)

        return {"val": val_mean, "test": test_mean}

    def get_std_metric(self, metric: str, round: bool = True) -> dict:
        """
        Get the standard deviation of a specific metric from the results for both val and test phases.
        Args:
            metric (str): The metric to extract.
        Returns:

            dict: The dictionary of standard deviation metric values for val and test phases.
        """
        val_std = np.std(self._get_metric(metric, "val"))
        test_std = np.std(self._get_metric(metric, "test"))
        if round:
            val_std = np.round(val_std, 3)
            test_std = np.round(test_std, 3)

        return {"val": val_std, "test": test_std}

    def collect_metrics(self) -> dict:
        """
        Collect all metrics for both val and test phases.
        Returns:
            dict: The dictionary of all metrics for val and test phases.
        """
        output = {}
        for metric in self.metrics:
            output[metric] = self.get_mean_metric(metric)
            output[f"{metric}_std"] = self.get_std_metric(metric)

        output["dataset"] = self.dataset  # type: ignore
        output["method"] = self.method  # type: ignore
        return output

    def collect_metrics_as_df(self, process_std=True) -> pd.DataFrame:
        """
        Convert the aggregated results to a pandas DataFrame.
        Args:
            process_std: bool
                If True, combine the mean and std into a single column for each metric.
        Returns:
            pd.DataFrame: The DataFrame of the aggregated results.
        """
        if not process_std:
            return pd.DataFrame(self.collect_metrics())

        output = {}
        for metric in self.metrics:
            output[metric] = {
                "val": f"{self.get_mean_metric(metric)['val']} ± {self.get_std_metric(metric)['val']}",
                "test": f"{self.get_mean_metric(metric)['test']} ± {self.get_std_metric(metric)['test']}",
            }

        output["dataset"] = self.dataset  # type: ignore
        output["method"] = self.method  # type: ignore

        return pd.DataFrame(output)


def aggregate_multirun_results(
    multirun_paths: list[str], metrics: list[str]
) -> pd.DataFrame:
    """
    Aggregate the results from multiple multiseed jobs into a single DataFrame.
    Args:
        multirun_paths (list[str]): The list of paths to the multirun directories.
        final_metric (str): The metric to use for the final aggregation.
    Returns:
        pd.DataFrame: The DataFrame of the aggregated results.
    """
    results = pd.DataFrame()
    for path in tqdm.tqdm(multirun_paths):
        loader = ResultLoader(path, is_multirun=True)
        job_results = loader.get_multirun_results()
        if not job_results:
            continue
        aggregator = MultiSeedAggregator(job_results, metrics=metrics)
        aggregate_df = aggregator.collect_metrics_as_df()
        for metric in metrics:
            results.loc[
                aggregate_df.method.test, aggregate_df.dataset.test + "_" + metric
            ] = aggregate_df.loc["test"][metric]

    return results


class MultirunResultLoader:
    def __init__(
        self,
        root_dir: str,
        output_dir: str,
        metrics: list[str],
        ranking_metric: str,
        chemprop_root: str | None = None,
    ):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.metrics = metrics
        self.ranking_metric = ranking_metric
        self.chemprop_root = chemprop_root
        if self.ranking_metric not in self.metrics:
            raise ValueError(
                f"Ranking metric {self.ranking_metric} not in metrics {self.metrics}"
            )

    def get_aggregated_table(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns a pivot table of Mean ± Std for a specific metric and phase.
        """
        all_paths = list(os.listdir(self.root_dir))
        all_paths = [os.path.join(self.root_dir, path) for path in all_paths]
        table = aggregate_multirun_results(all_paths, metrics=self.metrics)
        table = self._add_chemprop_results(table)
        return self._format_table(table)

    def _dataset_name_from_table(self, table: pd.DataFrame) -> str | None:
        if table.empty or len(table.columns) == 0:
            return None
        first_col = str(table.columns[0])

        # Column names are formatted as "{dataset}_{metric}".
        for metric in sorted(self.metrics, key=len, reverse=True):
            suffix = f"_{metric}"
            if first_col.endswith(suffix):
                return first_col[: -len(suffix)]

        if "_" not in first_col:
            return None
        return first_col.rsplit("_", 1)[0]

    def _dataset_tokens(self, dataset_name: str) -> list[str]:
        mapping = {
            "biogen/adme-fang-reg-v1": ["adme_fang", "adme-fang", "adme-fang-reg-v1"],
            "graphium/tox21-v1": ["tox21", "tox21-v1"],
            "graphium/zinc12k-v1": ["zinc12k", "zinc12k-v1", "zinc"],
        }
        default_token = dataset_name.split("/")[-1].replace("-", "_")
        return mapping.get(dataset_name, [default_token, dataset_name.split("/")[-1]])

    def _find_chemprop_prediction_files(self, dataset_name: str) -> list[Path]:
        roots = [self.chemprop_root]
        tokens = self._dataset_tokens(dataset_name)
        files: set[Path] = set()
        for root in roots:
            if root is None:
                continue
            root_path = Path(root)
            if not root_path.exists():
                continue
            for path in root_path.rglob("test_predictions.csv"):
                path_lower = str(path).lower()
                if any(token.lower() in path_lower for token in tokens):
                    files.add(path)
        return sorted(files)

    def _seed_from_path_or_config(self, pred_path: Path) -> int | None:
        seed_match = re.search(r"seed=(\d+)", str(pred_path))
        if seed_match:
            return int(seed_match.group(1))

        config_path = pred_path.parents[1] / "config.toml"
        if not config_path.exists():
            return None

        config_text = config_path.read_text(encoding="utf-8")
        data_seed_match = re.search(r"data-seed\s*=\s*(\d+)", config_text)
        if data_seed_match:
            return int(data_seed_match.group(1))
        return None

    def _split_csv_from_config_or_default(
        self, dataset_name: str, seed: int | None
    ) -> Path | None:
        if seed is None:
            return None
        default_split = (
            Path.home()
            / "data"
            / "Datasets"
            / "polaris"
            / dataset_name
            / str(seed)
            / "benchmark_split.csv"
        )
        if default_split.exists():
            return default_split
        return None

    def _target_columns_from_cached_split(self, split_csv: Path) -> list[str]:
        target_cols_path = split_csv.parent / "target_columns.txt"
        if not target_cols_path.exists():
            return []
        with open(target_cols_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _metric_fns_for_names(self, metric_names: list[str]) -> dict:
        metric_fns = {}
        for metric in metric_names:
            metric_key = "r2_score" if metric == "r2" else metric
            if metric_key in CLASSIFICATION_METRICS:
                metric_fns[metric] = CLASSIFICATION_METRICS[metric_key]
            elif metric_key in REGRESSION_METRICS:
                metric_fns[metric] = REGRESSION_METRICS[metric_key]
        return metric_fns

    def _compute_chemprop_metrics_for_file(
        self, pred_path: Path, split_csv: Path, metric_names: list[str]
    ) -> dict[str, float] | None:
        pred_df = pd.read_csv(pred_path)
        split_df = pd.read_csv(split_csv, index_col=0)
        if "split" not in split_df.columns or "smiles" not in split_df.columns:
            return None

        test_df = split_df.loc[split_df["split"] == "test", :].copy()
        target_cols = self._target_columns_from_cached_split(split_csv)
        target_cols = [
            col for col in target_cols if col in pred_df.columns and col in test_df.columns
        ]
        if len(target_cols) == 0:
            return None

        merged = test_df[["smiles", *target_cols]].merge(
            pred_df[["smiles", *target_cols]],
            on="smiles",
            suffixes=("_true", "_pred"),
            how="inner",
        )
        if merged.empty:
            return None

        y_true_np = merged[[f"{col}_true" for col in target_cols]].to_numpy(dtype=float)
        y_pred_np = merged[[f"{col}_pred" for col in target_cols]].to_numpy(dtype=float)

        # Align NaN masks for robust metric handling.
        invalid_mask = np.isnan(y_true_np) | np.isnan(y_pred_np)
        y_true_np[invalid_mask] = np.nan
        y_pred_np[invalid_mask] = np.nan

        metric_fns = self._metric_fns_for_names(metric_names)
        if len(metric_fns) == 0:
            return None

        metric_vals = compute_metrics(
            y_true=torch.as_tensor(y_true_np, dtype=torch.float32),
            y_pred=torch.as_tensor(y_pred_np, dtype=torch.float32),
            metric_fns=metric_fns,
            y_proba=torch.as_tensor(y_pred_np, dtype=torch.float32),
            split_name="",
        )

        out: dict[str, float] = {}
        for metric in metric_names:
            if metric in metric_vals:
                out[metric] = float(metric_vals[metric])
        return out if out else None

    def _collect_chemprop_seed_metrics(self, dataset_name: str) -> dict[int, dict[str, float]]:
        seed_metrics: dict[int, dict[str, float]] = {}
        files = self._find_chemprop_prediction_files(dataset_name)
        for pred_path in files:
            seed = self._seed_from_path_or_config(pred_path)
            split_csv = self._split_csv_from_config_or_default(
                dataset_name=dataset_name,
                seed=seed,
            )
            if seed is None or split_csv is None:
                continue
            metrics = self._compute_chemprop_metrics_for_file(
                pred_path=pred_path,
                split_csv=split_csv,
                metric_names=self.metrics,
            )
            if metrics is None:
                continue
            seed_metrics[seed] = metrics
        return seed_metrics

    def _add_chemprop_results(self, table: pd.DataFrame) -> pd.DataFrame:
        dataset_name = self._dataset_name_from_table(table)
        if dataset_name is None:
            return table

        seed_metrics = self._collect_chemprop_seed_metrics(dataset_name)
        if not seed_metrics:
            return table

        chemprop_row: dict[str, str] = {}
        for metric in self.metrics:
            values = [seed_metrics[s][metric] for s in sorted(seed_metrics) if metric in seed_metrics[s]]
            if len(values) == 0:
                continue
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            chemprop_row[f"{dataset_name}_{metric}"] = f"{mean_val:.3f} ± {std_val:.3f}"

        if not chemprop_row:
            return table

        table.loc["chemprop", list(chemprop_row.keys())] = list(chemprop_row.values())
        return table

    def _determine_ranking_order(self):
        if self.ranking_metric in ('rmse', 'mae', 'mse'):
            return 'ascending'
        elif self.ranking_metric in ('pearsonr', 'spearmanr', 'pr_auc', 'roc_auc', 'r2', 'f1_score', 'mcc'):
            return 'descending'
        else:
            raise ValueError(f"Unknown ranking metric {self.ranking_metric}")


    def _get_diffpred_index(self, table: pd.DataFrame) -> list:
        return [idx for idx in table.index if str(idx).lower().endswith('_diff')]

    def _format_table(self, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Capitalize rows and identify DIFFPRED result"""
        
        raw_table = table.copy()
        table.index = table.index.str.upper()
        diffpred_indices = self._get_diffpred_index(table)

        # sort index by ranking metric
        ranking_cols = [col for col in table.columns if col.endswith(f"_{self.ranking_metric}")]
        
        if ranking_cols:
            def extract_score(x):
                if isinstance(x, str) and '±' in x:
                    return float(x.split('±')[0].strip())
                try:
                    return float(x)
                except ValueError:
                    return np.nan
            
            # calculate mean score across datasets for the ranking metric
            scores = table[ranking_cols].apply(lambda col: col.map(extract_score)).mean(axis=1)
            ascending = self._determine_ranking_order() == 'ascending'
            
            # identify the best row amongst diffpred according to the ranking metric and sort direction
            if diffpred_indices:
                diffpred_scores = scores.loc[diffpred_indices]
                best_diffpred_idx = diffpred_scores.idxmin() if ascending else diffpred_scores.idxmax()
                
                # drop other diffpred variants and rename the best one
                other_diffpreds = [idx for idx in diffpred_indices if idx != best_diffpred_idx]
                table = table.drop(index=other_diffpreds)
                scores = scores.drop(index=other_diffpreds)
                
                table = table.rename(index={best_diffpred_idx: 'DIFFPRED'})
                scores = scores.rename(index={best_diffpred_idx: 'DIFFPRED'})

            # sort the table based on the best scores
            table = table.loc[scores.sort_values(ascending=ascending).index]
            
        return raw_table, table
    
    def extract(self):
        raw_table, table = self.get_aggregated_table()
        # save to CSV
        raw_table.to_csv(os.path.join(self.output_dir, 'raw_table.csv'))
        table.to_csv(os.path.join(self.output_dir, 'table.csv'))

app = typer.Typer()

@app.command()
def extract_multitask(
    root_dir: str = typer.Option(..., help="Root directory of the multirun results"),
    output_dir: str = typer.Option(..., help="Output directory to save the tables"),
    metrics: str = typer.Option(
        "pearsonr,r2,rmse,mae", help="Comma-separated list of metrics to extract"
    ),
    ranking_metric: str = typer.Option("pearsonr", help="Metric to use for ranking"),
    chemprop_root: str | None = typer.Option(
        None,
        help="Root directory containing Chemprop outputs",
    ),
):
    """
    Extracts and aggregates multirun results.
    """
    metrics_list = [m.strip() for m in metrics.split(",")]
    loader = MultirunResultLoader(
        root_dir=root_dir,
        output_dir=output_dir,
        metrics=metrics_list,
        ranking_metric=ranking_metric,
        chemprop_root=chemprop_root,
    )
    loader.extract()
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    app()