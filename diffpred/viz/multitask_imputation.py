from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_CONFIG = [
    ("test_rmse_mean", "test_rmse_std", "Test RMSE"),
]


def _plot_metric_panels(summary_df: pd.DataFrame, metric_col: str, std_col: str, y_label: str, output_path: Path) -> None:
    datasets = sorted(summary_df["dataset"].unique())
    if len(datasets) == 0:
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4), squeeze=False)

    for i, dataset in enumerate(datasets):
        ax = axes[0, i]
        data_ds = summary_df[summary_df["dataset"] == dataset]
        for model in sorted(data_ds["model"].unique()):
            d = data_ds[data_ds["model"] == model].sort_values("ratio")
            x = d["ratio"].to_numpy()
            y = d[metric_col].to_numpy()
            y_std = d[std_col].to_numpy()

            ax.plot(x, y, marker="o", label=model)
            ax.fill_between(x, y - y_std, y + y_std, alpha=0.2)

        ax.set_xlabel("Mask ratio")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_test_rmse_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return

    for metric_col, std_col, y_label in METRIC_CONFIG:
        _plot_metric_panels(
            summary_df=summary_df,
            metric_col=metric_col,
            std_col=std_col,
            y_label=y_label,
            output_path=output_dir / metric_col,
        )
