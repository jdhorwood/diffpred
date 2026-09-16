import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import seaborn as sns

from diffpred.constants.literals import (
    CHEMBL_DATASETS,
    CHEMBL_DEFAULT_DATASET,
    TDC_REGRESSION_DATASETS,
)
from diffpred.utils.reloader import batch_extract_run_results


class TrainTestGapVisualizer:
    """Visualize train-vs-test performance gaps for diffusion and baseline models."""

    def __init__(self, path: str, metrics: list[str] | None = None, viz_dir_name: str = "figures"):
        """
        Initialize the visualizer with a benchmark result directory.

        Args:
            path: Path to benchmark experiment outputs.
            metrics: Metrics to plot (default: ["mae", "pearsonr", "spearmanr", "rmse"]).
            viz_dir_name: Relative output directory where figures are saved.
        """
        self.path = path
        self.metrics = metrics or ["mae", "pearsonr", "spearmanr", "rmse"]
        self.viz_dir = viz_dir_name
        os.makedirs(self.viz_dir, exist_ok=True)

        self.results_df = batch_extract_run_results(path)

    def visualize_results(self, datasets: list[str] | None = None):
        """
        Generate scatter plots for all architectures and available dataloaders.

        Each plot contains one architecture and one metric. Points are datasets,
        with x-axis = test metric and y-axis = train metric, and color encoding model type.

        Args:
            datasets: Optional list of dataset names to include.
        """
        self._set_matplotlib_defaults()

        total_plots = 0
        for dataloader_idx, df in self.results_df.items():
            curr_df = self._filter_datasets(df, datasets)
            if curr_df.empty:
                if datasets is not None:
                    print(
                        f"Warning: no requested datasets found for dataloader {dataloader_idx}."
                    )
                continue

            for net_type in sorted(curr_df["net_type"].dropna().unique()):
                total_plots += self._plot_net_type(
                    dataloader_idx=dataloader_idx,
                    df=curr_df,
                    net_type=net_type,
                )

        if total_plots == 0:
            print(
                "Warning: no plots were generated. Check dataset names and metric/train-column availability."
            )

    def _filter_datasets(self, df: pd.DataFrame, datasets: list[str] | None) -> pd.DataFrame:
        """Return all datasets or the requested dataset subset."""
        if datasets is None:
            return df

        available = set(df["dataset"].dropna().unique())
        valid_datasets = [dataset for dataset in datasets if dataset in available]
        if not valid_datasets:
            return pd.DataFrame()

        return df[df["dataset"].isin(valid_datasets)]

    def _plot_net_type(self, dataloader_idx: int, df: pd.DataFrame, net_type: str):
        """Generate train-vs-test scatter plots for one architecture across metrics."""
        net_df = df[df["net_type"] == net_type]
        if net_df.empty:
            return 0

        if 'arch_type' not in net_df.columns:
            print(f"Warning: 'arch_type' column not found for net_type {net_type} in dataloader {dataloader_idx}. Skipping this architecture.")
            return 0
        diffusion_df = net_df[net_df["arch_type"] == "baseline"]
        baseline_df = net_df[net_df["arch_type"].isna()]

        test_prefix = f"test_{dataloader_idx}/"

        n_plots = 0
        for metric in self.metrics:
            test_col = f"{test_prefix}{metric}"
            reference_specs = self._resolve_reference_columns(
                net_df=net_df,
                metric=metric,
            )

            if test_col not in net_df.columns or not reference_specs:
                continue

            for reference_col, transform_reference, reference_split in reference_specs:
                diff_points = self._build_plot_df(
                    diffusion_df,
                    train_col=reference_col,
                    test_col=test_col,
                    model_label="Diffusion",
                    transform_train=transform_reference,
                )
                base_points = self._build_plot_df(
                    baseline_df,
                    train_col=reference_col,
                    test_col=test_col,
                    model_label="Baseline",
                    transform_train=transform_reference,
                )

                plot_df = pd.concat([diff_points, base_points], ignore_index=True)
                if plot_df.empty:
                    continue

                self._create_scatter_plot(
                    dataloader_idx=dataloader_idx,
                    net_type=net_type,
                    metric=metric,
                    plot_df=plot_df,
                    reference_split=reference_split,
                )
                n_plots += 1

        return n_plots

    def _resolve_reference_columns(
        self,
        net_df: pd.DataFrame,
        metric: str,
    ) -> list[tuple[str, bool, str]]:
        """
        Find available train and validation columns that match a test metric.

        Returns:
            List of (column_name, apply_sqrt, split_name)
        """
        train_candidates = [
            f"train_0/{metric}",
            f"train_0_{metric}",
            f"train/{metric}",
            f"train_{metric}",
        ]
        val_candidates = [
            f"val_0/{metric}",
            f"val_0_{metric}",
            f"val/{metric}",
            f"val_{metric}",
        ]

        refs: list[tuple[str, bool, str]] = []

        # Train/validation references are shared across all test dataloaders.
        for col in train_candidates:
            if col in net_df.columns:
                refs.append((col, False, "train"))
                break

        for col in val_candidates:
            if col in net_df.columns:
                refs.append((col, False, "validation"))
                break

        if refs:
            return refs

        # Fallbacks observed in logged outputs.
        if metric in {"mae", "rmse"} and "train_loss" in net_df.columns:
            return [("train_loss", False, "train")]

        # If only train MSE is available and RMSE is requested, convert by sqrt.
        if metric == "rmse" and "train_mse" in net_df.columns:
            return [("train_mse", True, "train")]

        return []

    def _build_plot_df(
        self,
        model_df: pd.DataFrame,
        train_col: str,
        test_col: str,
        model_label: str,
        transform_train: bool,
    ) -> pd.DataFrame:
        """Create dataset-level mean points for one model type."""
        if model_df.empty or train_col not in model_df.columns or test_col not in model_df.columns:
            return pd.DataFrame(columns=["dataset", "train", "test", "model"])

        working_df = model_df[["dataset", train_col, test_col]].copy()
        working_df = working_df.dropna(subset=["dataset", train_col, test_col])
        if working_df.empty:
            return pd.DataFrame(columns=["dataset", "train", "test", "model"])

        grouped = working_df.groupby("dataset", as_index=False).mean(numeric_only=True)
        grouped = grouped.rename(columns={train_col: "train", test_col: "test"})

        if transform_train:
            grouped["train"] = np.sqrt(grouped["train"].clip(lower=0.0))

        grouped["model"] = model_label
        return grouped

    def _create_scatter_plot(
        self,
        dataloader_idx: int,
        net_type: str,
        metric: str,
        plot_df: pd.DataFrame,
        reference_split: str,
    ):
        """Create and save the train-vs-test scatter figure for one metric and architecture."""
        fig, ax = plt.subplots(figsize=(10, 8))

        self._draw_paired_connectors(ax=ax, plot_df=plot_df, metric=metric)

        palette = {
            "Diffusion": sns.color_palette("colorblind")[0],
            "Baseline": sns.color_palette("colorblind")[1],
        }
        markers = {"Diffusion": "o", "Baseline": "s"}

        for model_name in ["Diffusion", "Baseline"]:
            model_df = plot_df[plot_df["model"] == model_name]
            if model_df.empty:
                continue

            ax.scatter(
                model_df["test"],
                model_df["train"],
                s=110,
                alpha=0.85,
                color=palette[model_name],
                marker=markers[model_name],
                label=model_name,
                zorder=2,
            )

            # Fit y = m*x + b and display slope m in the legend.
            x = model_df["test"].to_numpy(dtype=float)
            y = model_df["train"].to_numpy(dtype=float)
            if len(model_df) >= 2 and np.unique(x).size >= 2:
                slope, intercept = np.polyfit(x, y, deg=1)
                x_fit = np.linspace(np.min(x), np.max(x), 100)
                y_fit = slope * x_fit + intercept
                ax.plot(
                    x_fit,
                    y_fit,
                    linestyle="--",
                    color=palette[model_name],
                    linewidth=2,
                    label=f"{model_name} fit (m={slope:.2f})",
                    zorder=3,
                )

        ax.set_xlabel(f"Test {metric.upper()}")
        y_split_label = "Validation" if reference_split == "validation" else "Train"
        ax.set_ylabel(f"{y_split_label} {metric.upper()}")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        fig.tight_layout()
        split_name = "val" if reference_split == "validation" else "train"
        out_name = f"train_test_gap_{split_name}_{metric}_{net_type}_{dataloader_idx}"
        fig.savefig(os.path.join(self.viz_dir, f"{out_name}.png"))
        fig.savefig(os.path.join(self.viz_dir, f"{out_name}.pdf"))
        plt.close(fig)

    def _draw_paired_connectors(self, ax: Axes, plot_df: pd.DataFrame, metric: str):
        """Draw connectors between paired points colored by diffusion improvement direction."""
        paired = plot_df.pivot_table(
            index="dataset",
            columns="model",
            values=["test", "train"],
            aggfunc="first",
        )

        required_cols = [
            ("test", "Baseline"),
            ("train", "Baseline"),
            ("test", "Diffusion"),
            ("train", "Diffusion"),
        ]
        if not all(col in paired.columns for col in required_cols):
            return

        # Metrics where larger values indicate better performance.
        higher_is_better = {"pearsonr", "spearmanr", "r2_score", "r2"}
        metric_key = metric.lower()
        is_higher_better = metric_key in higher_is_better

        improvement_color = "#2ca02c"  # green
        degradation_color = "#d62728"  # red

        for _, row in paired.iterrows():
            x_base = row[("test", "Baseline")]
            y_base = row[("train", "Baseline")]
            x_diff = row[("test", "Diffusion")]
            y_diff = row[("train", "Diffusion")]

            if np.isnan([x_base, y_base, x_diff, y_diff]).any():
                continue

            if is_higher_better:
                improved = x_diff >= x_base
            else:
                improved = x_diff <= x_base

            connector_color = improvement_color if improved else degradation_color

            ax.plot(
                [x_base, x_diff],
                [y_base, y_diff],
                color=connector_color,
                alpha=0.35,
                linewidth=1.0,
                zorder=1,
            )

    def _set_matplotlib_defaults(self):
        """Set plotting defaults consistent with the other visualizers."""
        sns.set_palette("colorblind")
        sns.set_context("paper")
        sns.set_style("whitegrid")
        plt.style.use("seaborn-v0_8-colorblind")

        plt.rcParams.update({"font.size": 16})
        plt.rcParams.update({"xtick.labelsize": 14})
        plt.rcParams.update({"ytick.labelsize": 14})
        plt.rcParams.update({"axes.labelsize": 16})
        plt.rcParams.update({"axes.titlesize": 18})
        plt.rcParams.update({"legend.fontsize": 14})
        plt.rcParams.update({"lines.linewidth": 2.0})


def build_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Plot train-vs-test performance gaps for diffusion vs baseline models."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Root directory containing benchmark run folders.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["mae", "pearsonr", "spearmanr", "rmse"],
        help="Metrics to visualize.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional list of datasets to include.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        choices=["CHEMBL", "TDC", "chembl", "tdc"],
        help=(
            "Benchmark dataset preset to use. "
            "CHEMBL uses CHEMBL datasets except CHEMBL_DEFAULT_DATASET. "
            "TDC uses the six TDC regression datasets."
        ),
    )
    parser.add_argument(
        "--viz-dir-name",
        type=str,
        default="figures",
        help="Relative output directory where plots are saved.",
    )
    return parser


def _resolve_datasets(
    benchmark: str | None,
    datasets: list[str] | None,
) -> list[str] | None:
    """Resolve the dataset list from benchmark preset and optional user override."""
    if benchmark is None:
        return datasets

    benchmark_upper = benchmark.upper()
    if benchmark_upper == "CHEMBL":
        resolved = [ds for ds in CHEMBL_DATASETS if ds != CHEMBL_DEFAULT_DATASET]
    elif benchmark_upper == "TDC":
        resolved = list(TDC_REGRESSION_DATASETS)
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")

    # If explicit datasets are provided, intersect with benchmark datasets.
    if datasets is not None:
        dataset_set = set(datasets)
        resolved = [dataset for dataset in resolved if dataset in dataset_set]

    return resolved


def main():
    """CLI entrypoint."""
    args = build_parser().parse_args()
    visualizer = TrainTestGapVisualizer(
        path=args.path,
        metrics=args.metrics,
        viz_dir_name=args.viz_dir_name,
    )
    resolved_datasets = _resolve_datasets(args.benchmark, args.datasets)
    visualizer.visualize_results(datasets=resolved_datasets)


if __name__ == "__main__":
    main()
