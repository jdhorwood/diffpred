import os
from typing import Dict, Tuple
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from diffpred.viz.utils import set_mpl_settings


def viz_sigma_losses(
    initial_losses: Dict[float, Tuple[float, float, float]],
    post_losses: Dict[float, Tuple[float, float, float]],
    sampled_sigmas: torch.FloatTensor | None,
    output_dir: str,
    split: str = "train",
    plot_std: bool = True,
    plot_density: bool = True,
    plot_weights: bool = True,
):
    """
    Plots the initial and post losses (mean, std) for the EDM model across different sigma values.

    """

    set_mpl_settings()
    # plot the losses across different sigma values for the start and end of the training on same plot
    fig, ax = plt.subplots(figsize=(8, 6))
    initial_means, initial_stds, initial_weights = zip(*initial_losses.values())
    post_means, post_stds, post_weights = zip(*post_losses.values())

    sigmas = list(initial_losses.keys())

    # plot line plots with standard deviations shaded on log scale of sigma
    means1 = ax.plot(sigmas, initial_means, label="Loss at init")
    means2 = ax.plot(sigmas, post_means, label="Loss after training")
    ax.set_xscale("log")

    if plot_std:
        ax.fill_between(
            list(initial_losses.keys()),
            [mean - std for mean, std in zip(initial_means, initial_stds)],
            [mean + std for mean, std in zip(initial_means, initial_stds)],
            alpha=0.3,
        )
        ax.fill_between(
            list(post_losses.keys()),
            [mean - std for mean, std in zip(post_means, post_stds)],
            [mean + std for mean, std in zip(post_means, post_stds)],
            alpha=0.3,
        )

    ax.set_xlabel("Sigma")
    ax.set_ylabel("Loss")

    if plot_weights:
        ax2 = ax.twinx()

        ax2_color = "#CC79A7"
        weights_ln = ax2.plot(
            sigmas,
            initial_weights,
            color=ax2_color,
            label="Loss weight",
            linestyle="--",
        )
        ax2.set_ylabel("Loss weight (log)")
        ax2.set_yscale("log")
        ax2.get_yaxis().set_tick_params(which="minor", size=0)

    if plot_density and sampled_sigmas is not None:
        # Finally plot the sampling density for the sigma values across noise levels
        ax3_color = "#D55E00"
        ax3 = ax.twinx()
        ax3 = sns.kdeplot(
            sampled_sigmas.cpu(),
            ax=ax3,
            cut=0,
            color=ax3_color,
            legend=False,
            linestyle="--",
        )
        ax3.set_yticks([])
        ax3.set_ylabel(None)
        density = ax3.lines
        density[0].set_label("Sigma sampling density")

    legend_lines = means1 + means2
    if plot_weights:
        legend_lines = legend_lines + weights_ln
    if plot_density:
        legend_lines = legend_lines + density

    legend_labels = [line.get_label() for line in legend_lines]
    ax.legend(legend_lines, legend_labels)

    # save the figure
    os.makedirs(f"{output_dir}/viz", exist_ok=True)
    save_path = f"{output_dir}/viz/sigma_losses_{split}.svg"
    save_path = f"{output_dir}/viz/sigma_losses_{split}.png"
    fig.savefig(save_path)
    plt.close()

    return fig
