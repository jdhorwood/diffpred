#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

from diffpred.constants.literals import CHEMBL_DATASETS, CHEMBL_DEFAULT_DATASET
from diffpred.viz.single_task_exp import ModelComparisonVisualizer


def _parse_csv(raw_value: str) -> list[str]:
    values = [v.strip() for v in raw_value.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def _parse_metrics(raw_metrics: str) -> list[str]:
    metrics = _parse_csv(raw_metrics)
    if not metrics:
        raise ValueError("At least one metric must be provided")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate MoleculeACE and TDC single-task figures using "
            "ModelComparisonVisualizer."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root directory containing raw single-task multirun outputs.",
    )
    parser.add_argument(
        "--command",
        type=str,
        default="figures",
        choices=["figures", "chembl-improvement-summary"],
        help=(
            "Command to run. 'figures' generates CHEMBL/TDC figures; "
            "'chembl-improvement-summary' computes CHEMBL improvement summary tables."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory where figure artifacts should be written.",
    )
    parser.add_argument("--chembl-metrics", type=str, default="pearsonr")
    parser.add_argument(
        "--chembl-plot-type",
        type=str,
        default="violin",
        choices=["box", "violin"],
    )
    parser.add_argument("--chembl-viz-dir-name", type=str, default="chembl_viz")
    parser.add_argument(
        "--chembl-summary-dir-name",
        type=str,
        default="chembl_improvement_summary",
    )
    parser.add_argument("--tdc-metrics", type=str, default="rmse,pearsonr,r2_score")
    parser.add_argument(
        "--tdc-plot-type",
        type=str,
        default="box",
        choices=["box", "violin"],
    )
    parser.add_argument("--tdc-viz-dir-name", type=str, default="adme_viz")
    parser.add_argument(
        "--tdc-datasets",
        type=str,
        default=(
            "lipophilicity_astrazeneca,solubility_aqsoldb,clearance_hepatocyte_az,"
            "clearance_microsome_az,vdss_lombardo,ld50_zhu"
        ),
    )
    return parser.parse_args()


def _get_chembl_datasets() -> list[str]:
    datasets = CHEMBL_DATASETS.copy()
    if CHEMBL_DEFAULT_DATASET in datasets:
        datasets.remove(CHEMBL_DEFAULT_DATASET)
    return datasets


def generate_figures(args: argparse.Namespace) -> None:

    chembl_output_dir = args.output_root / args.chembl_viz_dir_name
    chembl_output_dir.mkdir(parents=True, exist_ok=True)
    chembl_metrics = _parse_metrics(args.chembl_metrics)

    chembl_visualizer = ModelComparisonVisualizer(
        path=str(args.results_root),
        metrics=chembl_metrics,
        viz_dir=str(chembl_output_dir),
    )

    datasets = _get_chembl_datasets()

    chembl_visualizer.visualize_results(datasets=datasets)
    chembl_visualizer.visualize_aggregate_differences(
        datasets=datasets,
        plot_type=args.chembl_plot_type,
    )

    tdc_output_dir = args.output_root / args.tdc_viz_dir_name
    tdc_output_dir.mkdir(parents=True, exist_ok=True)
    tdc_metrics = _parse_metrics(args.tdc_metrics)
    tdc_datasets = _parse_csv(args.tdc_datasets)

    tdc_visualizer = ModelComparisonVisualizer(
        path=str(args.results_root),
        metrics=tdc_metrics,
        viz_dir=str(tdc_output_dir),
    )
    tdc_visualizer.visualize_results(datasets=tdc_datasets)
    tdc_visualizer.visualize_aggregate_differences(
        datasets=tdc_datasets,
        plot_type=args.tdc_plot_type,
    )

    print(f"Generated CHEMBL figures in: {chembl_output_dir}")
    print(f"Generated TDC figures in: {tdc_output_dir}")


def compute_chembl_improvement_summary(args: argparse.Namespace) -> None:
    summary_output_dir = args.output_root / args.chembl_summary_dir_name
    summary_output_dir.mkdir(parents=True, exist_ok=True)

    chembl_metrics = _parse_metrics(args.chembl_metrics)
    datasets = _get_chembl_datasets()

    visualizer = ModelComparisonVisualizer(
        path=str(args.results_root),
        metrics=chembl_metrics,
        viz_dir=str(summary_output_dir),
    )
    visualizer.visualize_results(datasets=datasets)

    split_labels = {
        "0": "test",
        "1": "activity_cliff_test",
    }
    stat_rows = ["MEAN", "MIN", "MAX", "STD"]

    # (split_name, metric) -> {NET_TYPE: {stat: value}}
    aggregated: dict[tuple[str, str], dict[str, dict[str, float]]] = {}

    for cache_key, (
        diffusion_test,
        baseline_test,
        common_datasets,
        _,
    ) in visualizer.processed_data.items():
        dataloader_idx, net_type = cache_key.split("_", 1)
        if dataloader_idx not in split_labels:
            continue

        difference_df = visualizer._compute_differences(
            diffusion_test, baseline_test, common_datasets
        )

        for metric in chembl_metrics:
            if metric not in difference_df.columns:
                continue

            improvements = difference_df[metric].dropna()
            if improvements.empty:
                continue

            stats = {
                "MEAN": float(improvements.mean()),
                "MIN": float(improvements.min()),
                "MAX": float(improvements.max()),
                "STD": float(improvements.std(ddof=0)),
            }
            key = (split_labels[dataloader_idx], metric)
            aggregated.setdefault(key, {})[net_type.upper()] = stats

    if not aggregated:
        raise RuntimeError(
            "No CHEMBL improvement data found for test/activity-cliff dataloaders."
        )

    for (split_name, metric), net_type_stats in sorted(aggregated.items()):
        table_df = pd.DataFrame(net_type_stats).reindex(index=stat_rows)
        table_df = table_df.reindex(sorted(table_df.columns), axis=1)

        output_path = (
            summary_output_dir
            / f"chembl_improvement_summary_{split_name}_{metric}.csv"
        )
        table_df.to_csv(output_path, float_format="%.6f")

        print(f"\nCHEMBL improvement summary ({split_name}, metric={metric}):")
        print(table_df.to_string(float_format=lambda x: f"{x:.6f}"))
        print(f"Saved summary table to: {output_path}")


def main() -> None:
    args = parse_args()

    if not args.results_root.exists():
        raise FileNotFoundError(f"Results root not found: {args.results_root}")

    if args.command == "chembl-improvement-summary":
        compute_chembl_improvement_summary(args)
        return

    generate_figures(args)


if __name__ == "__main__":
    main()
