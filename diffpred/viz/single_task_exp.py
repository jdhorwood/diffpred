from diffpred.utils.reloader import batch_extract_run_results
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


class ModelComparisonVisualizer:
    """
    A class to visualize comparisons between diffusion and baseline models across datasets.
    The class encapsulates data loading, processing, and visualization.
    """

    def __init__(self, path, metrics=None, viz_dir_name: str = "viz", viz_dir: str | None = None):
        """
        Initialize the visualizer with the path to experiment results.

        Args:
            path (str): Path to the experiment results directory
            metrics (list): List of metrics to visualize (default: ["mae", "pearsonr", "spearmanr", "rmse"])
        """
        self.path = path
        self.metrics = metrics or ["mae", "pearsonr", "spearmanr", "rmse"]
        self.viz_dir = viz_dir if viz_dir is not None else os.path.join(path, viz_dir_name)
        os.makedirs(self.viz_dir, exist_ok=True)

        self.results_df = batch_extract_run_results(path)

        self.processed_data: dict = {}

    def visualize_results(self, datasets=None):
        """
        Create visualizations comparing diffusion and baseline models.

        Args:
            datasets (list, optional): List of specific dataset names to process.
                                    If None, all datasets will be processed.
        """
        self._set_matplotlib_defaults()
        for dataloader_idx, df in self.results_df.items():
            if datasets is not None:
                available_datasets = df.dataset.unique()
                valid_datasets = [d for d in datasets if d in available_datasets]

                if not valid_datasets:
                    print(
                        f"Warning: None of the requested datasets found in dataloader {dataloader_idx}"
                    )
                    continue

                df = df[df.dataset.isin(valid_datasets)]

            self._process_dataloader(dataloader_idx, df)

    def _process_dataloader(self, dataloader_idx, df):
        """Process results for a specific dataloader."""
        if 'arch_type' not in df.columns:
            print(f"Warning: 'arch_type' column not found in dataloader {dataloader_idx}. Skipping this dataloader.")
            return
        
        diffusion_df = df[df["arch_type"] == "baseline"] 
        baseline_df = df[df["arch_type"].isna()]
        test_prefix = f"test_{dataloader_idx}/"

        for net_type in df["net_type"].unique():
            self._process_net_type(
                dataloader_idx, net_type, diffusion_df, baseline_df, test_prefix
            )

    def _process_net_type(
        self, dataloader_idx, net_type, diffusion_df, baseline_df, test_prefix
    ):
        """Process results for a specific net type within a dataloader."""
        cache_key = f"{dataloader_idx}_{net_type}"

        if cache_key in self.processed_data:
            (
                diffusion_test,
                baseline_test,
                common_datasets,
                difference_std,
            ) = self.processed_data[cache_key]
        else:
            diffusion_net_df = diffusion_df[diffusion_df["net_type"] == net_type]
            baseline_net_df = baseline_df[baseline_df["net_type"] == net_type]

            if diffusion_net_df.empty or baseline_net_df.empty:
                return

            try:
                diffusion_mean = self._compute_dataset_means(diffusion_net_df)
                baseline_mean = self._compute_dataset_means(baseline_net_df)
            except Exception as e:
                print(
                    f"Error in groupby for net_type {net_type}, dataloader {dataloader_idx}: {e}"
                )
                return

            if diffusion_mean.empty or baseline_mean.empty:
                return

            try:
                diffusion_test = self._extract_metrics(diffusion_mean, test_prefix)
                baseline_test = self._extract_metrics(baseline_mean, test_prefix)
            except Exception as e:
                print(
                    f"Error extracting metrics for net_type {net_type}, dataloader {dataloader_idx}: {e}"
                )
                return

            if diffusion_test.empty or baseline_test.empty:
                return

            common_datasets = diffusion_test.index.intersection(baseline_test.index)
            if len(common_datasets) == 0:
                return

            difference_std = self._compute_difference_se(
                diffusion_net_df, baseline_net_df, common_datasets, test_prefix
            )

            self.processed_data[cache_key] = (
                diffusion_test,
                baseline_test,
                common_datasets,
                difference_std,
            )

        self._create_visualizations(
            dataloader_idx,
            net_type,
            diffusion_test,
            baseline_test,
            common_datasets,
            difference_std,
        )

    def _compute_dataset_means(self, df):
        """Calculate mean and standard deviation values across seeds for each dataset."""
        numeric_cols = df.select_dtypes(include=np.number).columns

        means = df.groupby("dataset")[numeric_cols].mean()

        return means

    def _extract_metrics(self, df_mean, test_prefix):
        """Extract and rename metric columns for the specified test dataset."""
        metric_cols = [col for col in df_mean.columns if col.startswith(test_prefix)]

        if not metric_cols:
            return pd.DataFrame()

        result_df = df_mean[metric_cols].copy()

        metric_mapping = {col: col.split("/")[1] for col in metric_cols}

        result_df.rename(columns=metric_mapping, inplace=True)

        available_metrics = [m for m in self.metrics if m in result_df.columns]
        if not available_metrics:
            return pd.DataFrame()

        return result_df[available_metrics]

    def _create_visualizations(
        self,
        dataloader_idx,
        net_type,
        diffusion_test,
        baseline_test,
        common_datasets,
        difference_std,
    ):
        """Create visualizations for all metrics with standard deviation error bars."""
        difference_df = self._compute_differences(
            diffusion_test, baseline_test, common_datasets
        )

        for metric in diffusion_test.columns:
            if self._has_valid_data(difference_df, metric, common_datasets):
                self._create_comparison_plot(
                    dataloader_idx,
                    net_type,
                    difference_df,
                    metric,
                    common_datasets,
                    difference_std,
                )

    def _compute_differences(self, diffusion_test, baseline_test, common_datasets):
        """Compute differences between diffusion and baseline metrics."""
        return diffusion_test.loc[common_datasets].subtract(
            baseline_test.loc[common_datasets], fill_value=np.nan
        )

    def _compute_difference_se(
        self, diffusion_df, baseline_df, common_datasets, test_prefix
    ):
        """
        Compute standard error of differences directly from individual seed results.
        This captures the actual variation in the difference between models.
        """
        difference_se = pd.DataFrame(index=common_datasets, columns=self.metrics)

        for dataset in common_datasets:
            diffusion_seeds = diffusion_df[
                diffusion_df.dataset == dataset
            ].seed.unique()
            baseline_seeds = baseline_df[baseline_df.dataset == dataset].seed.unique()

            common_seeds = np.intersect1d(diffusion_seeds, baseline_seeds)

            if len(common_seeds) <= 1:
                for metric in self.metrics:
                    difference_se.loc[dataset, metric] = 0
                continue

            for metric in self.metrics:
                metric_col = f"{test_prefix}{metric}"

                if (
                    metric_col not in diffusion_df.columns
                    or metric_col not in baseline_df.columns
                ):
                    difference_se.loc[dataset, metric] = np.nan
                    continue

                differences = []
                for seed in common_seeds:
                    diff_value = diffusion_df[
                        (diffusion_df["dataset"] == dataset)
                        & (diffusion_df["seed"] == seed)
                    ][metric_col].values[0]
                    base_value = baseline_df[
                        (baseline_df["dataset"] == dataset)
                        & (baseline_df["seed"] == seed)
                    ][metric_col].values[0]
                    differences.append(diff_value - base_value)

                if len(differences) > 1:
                    difference_se.loc[dataset, metric] = np.std(
                        differences, ddof=1
                    ) / np.sqrt(len(differences))
                else:
                    difference_se.loc[dataset, metric] = 0

        return difference_se

    def _has_valid_data(self, difference_df, metric, common_datasets):
        """Check if there's valid (non-NaN) data to plot for a metric."""
        if difference_df[metric].isna().all():
            return False

        for dataset_name in common_datasets:
            if not np.isnan(difference_df.loc[dataset_name, metric]):
                return True

        return False

    def _create_comparison_plot(
        self,
        dataloader_idx,
        net_type,
        difference_df,
        metric,
        common_datasets,
        std_combined,
    ):
        """Create and save a horizontal bar plot showing metric differences with error bars."""

        fig, ax = plt.subplots(figsize=(10, 10))

        valid_datasets = []
        diffs = []
        stds = []

        for dataset_name in common_datasets:
            diff = difference_df.loc[dataset_name, metric]
            std = std_combined.loc[dataset_name, metric]

            if np.isnan(diff):
                continue

            valid_datasets.append(dataset_name)
            diffs.append(diff)
            stds.append(std)

        sorted_indices = np.argsort(diffs)
        valid_datasets = [valid_datasets[i] for i in sorted_indices]
        diffs = [diffs[i] for i in sorted_indices]
        stds = [stds[i] for i in sorted_indices]

        y_pos = np.arange(len(valid_datasets))
        ax.barh(y_pos, diffs, align="center", alpha=0.5, color="b")
        ax.errorbar(diffs, y_pos, xerr=stds, fmt="o", color="black", capsize=5)

        ax.tick_params(axis="y", labelsize=16)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(valid_datasets)

        ax.axvline(0, color="k", linestyle="dotted")
        ax.tick_params(axis="both", which="major")
        extra_y_label = "-Activity Cliffs" if dataloader_idx == 1 else ""
        ax.set_ylabel(f"Dataset{extra_y_label}")
        ax.set_xlabel(f"Diffusion - Baseline {metric.upper()}")

        fig.tight_layout()
        fig.savefig(f"{self.viz_dir}/results_{metric}_{net_type}_{dataloader_idx}.png")
        fig.savefig(f"{self.viz_dir}/results_{metric}_{net_type}_{dataloader_idx}.pdf")
        plt.close(fig)

    def visualize_aggregate_differences(self, plot_type="box", datasets=None):
        """
        Create visualizations showing aggregate performance differences across datasets for each metric.
        All net types and dataloaders will be shown in a single plot per metric.

        Args:
            plot_type (str): Type of plot to generate - "box" or "violin"
            datasets (list, optional): List of specific dataset names to process.
                                    If None, all datasets will be processed.
        """
        self._set_matplotlib_defaults()
        if not self.processed_data:
            self.visualize_results(datasets=datasets)

        # Structure: {metric: {(net_type, dataloader_idx): [differences]}}
        aggregated_by_metric = {}

        for cache_key, (
            diffusion_test,
            baseline_test,
            common_datasets,
            _,
        ) in self.processed_data.items():
            dataloader_idx, net_type = cache_key.split("_", 1)

            difference_df = self._compute_differences(
                diffusion_test, baseline_test, common_datasets
            )

            for metric in difference_df.columns:
                if not self._has_valid_data(difference_df, metric, common_datasets):
                    continue

                if metric not in aggregated_by_metric:
                    aggregated_by_metric[metric] = {}

                valid_diffs = [
                    diff for diff in difference_df[metric].values if not np.isnan(diff)
                ]
                if valid_diffs:
                    aggregated_by_metric[metric][(net_type, dataloader_idx)] = (
                        valid_diffs
                    )

        for metric, combinations in aggregated_by_metric.items():
            if combinations:
                self._plot_consolidated_comparison(metric, combinations, plot_type)

    def _plot_consolidated_comparison(self, metric, combinations, plot_type):
        """
        Create a single consolidated plot for a metric showing all net types and dataloaders.
        Boxes for different dataloaders appear side by side for each net type.
        """
        net_types = sorted(set(nt for nt, _ in combinations.keys()))
        dataloader_ids = sorted(set(dl for _, dl in combinations.keys()))

        positions = []
        data = []
        colors = []
        dataloader_to_color = {}

        for i, dl in enumerate(dataloader_ids):
            color_idx = int(dl) % 10 if dl.isdigit() else 0
            dataloader_to_color[dl] = plt.cm.Set2(color_idx)

        group_spacing = 0.5

        net_type_middle_positions = {}

        current_pos = 1
        for net_type in net_types:
            net_type_dataloaders = [
                (nt, dl) for (nt, dl) in combinations.keys() if nt == net_type
            ]

            if net_type_dataloaders:
                net_type_dataloaders.sort(key=lambda x: x[1])

                net_type_start_pos = current_pos

                for nt, dl in net_type_dataloaders:
                    positions.append(current_pos)
                    data.append(combinations[(nt, dl)])
                    colors.append(dataloader_to_color[dl])
                    current_pos += 1

                middle_pos = net_type_start_pos + (len(net_type_dataloaders) - 1) / 2
                net_type_middle_positions[net_type] = middle_pos

                current_pos += group_spacing

        n_boxes = len(positions)
        fig_width = max(10, n_boxes * 0.8)
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        if plot_type.lower() == "violin":
            parts = ax.violinplot(data, positions, showmeans=True, showmedians=True)
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)
        else:
            # For boxplot with custom positions, we need to create one box at a time
            for i, (pos, d, color) in enumerate(zip(positions, data, colors)):
                box = ax.boxplot([d], positions=[pos], patch_artist=True, widths=0.6)
                for patch in box["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

        ax.axhline(y=0, color="r", linestyle="--", alpha=0.7)

        xtick_positions = list(net_type_middle_positions.values())
        xtick_labels = [f"{nt.upper()}" for nt in net_type_middle_positions.keys()]
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

        net_type_positions = []
        pos = 1
        for net_type in net_types:
            dataloaders = [dl for nt, dl in combinations.keys() if nt == net_type]
            if dataloaders:
                pos += len(dataloaders)
                net_type_positions.append(pos - 0.5 + (group_spacing / 2))
                pos += group_spacing

        for pos in net_type_positions[:-1]:  # Don't add line after the last group
            ax.axvline(x=pos, color="gray", linestyle="-", alpha=0.3)

        legend_patches = []
        for dl in dataloader_ids:
            if dl == "1":
                legend_label = "Activity Cliffs Test"
            else:
                legend_label = "Test Set"
            patch = plt.Rectangle((0, 0), 1, 1, fc=dataloader_to_color[dl], alpha=0.7)
            legend_patches.append((patch, legend_label))

        ax.legend(*zip(*legend_patches), loc="upper left")

        ax.set_ylabel(f"Diffusion - Baseline {metric.upper()}")

        ax.grid(axis="y", linestyle="--", alpha=0.7)

        fig.tight_layout()
        plot_type_str = plot_type.lower()
        fig.savefig(f"{self.viz_dir}/consolidated_{metric}_{plot_type_str}.png")
        fig.savefig(f"{self.viz_dir}/consolidated_{metric}_{plot_type_str}.pdf")
        plt.close(fig)

    def _set_matplotlib_defaults(self):
        """
        Set default matplotlib parameters for consistent styling.
        """

        sns.set_palette("colorblind")
        sns.set_context("paper")
        sns.set_style("whitegrid")
        plt.style.use("seaborn-v0_8-colorblind")

        plt.rcParams.update({"font.size": 20})
        plt.rcParams.update({"xtick.labelsize": 20})
        plt.rcParams.update({"ytick.labelsize": 20})
        plt.rcParams.update({"axes.labelsize": 20})
        plt.rcParams.update({"axes.titlesize": 22})
        plt.rcParams.update({"legend.fontsize": 20})
        plt.rcParams.update({"lines.linewidth": 2.5})
