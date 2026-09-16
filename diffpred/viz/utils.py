import matplotlib.pyplot as plt
import seaborn as sns


def set_mpl_settings():
    sns.set_palette("colorblind")
    sns.set_context("paper")
    sns.set_style("whitegrid")

    # set a tight layout
    plt.tight_layout()

    # set the text size
    plt.rcParams.update({"font.size": 22})
    plt.rcParams.update({"xtick.labelsize": 16})
    plt.rcParams.update({"ytick.labelsize": 16})
    plt.rcParams.update({"axes.labelsize": 18})
    plt.rcParams.update({"axes.titlesize": 20})
    plt.rcParams.update({"legend.fontsize": 20})
    plt.rcParams.update({"legend.title_fontsize": 20})
    # set line thickness
    plt.rcParams.update({"lines.linewidth": 2.5})
    plt.style.use("seaborn-v0_8-colorblind")
