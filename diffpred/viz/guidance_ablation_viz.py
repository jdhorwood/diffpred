import argparse
import yaml
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from diffpred.utils.reloader import batch_extract_run_results, extract_tag_from_run_path
from diffpred.utils.result_loader import ResultLoader
from diffpred.viz.utils import set_mpl_settings

def load_and_categorize_results(base_dir: Path) -> pd.DataFrame:
    """Loads all results from the unified directory and categorizes them into regimes."""
    print(f"--- Loading and categorizing results from: {base_dir} ---")
    if not base_dir.is_dir():
        print(f"ERROR: Base results directory not found at {base_dir}")
        return pd.DataFrame()

    config_paths = list(base_dir.glob("**/config.yaml"))
    print(f"Found {len(config_paths)} total config files.")
    if not config_paths:
        return pd.DataFrame()

    records = []
    for config_path in tqdm(config_paths, desc="Processing all runs"):
        run_dir = config_path.parent.parent
        
        # Common metadata
        dataset = extract_tag_from_run_path(run_dir, "data.dataset")
        model = extract_tag_from_run_path(run_dir, "model")
        seed = extract_tag_from_run_path(run_dir, "experiment.seed")
        guidance_val = extract_tag_from_run_path(run_dir, "data.diffusion_params.guidance")

        # --- Main results from results.yaml ---
        results_path_str = ResultLoader.get_results_path(run_dir.as_posix())
        if results_path_str is None:
            continue

        results_path = Path(results_path_str)
        if results_path.exists():
            with open(results_path, "r") as f:
                results = yaml.safe_load(f)
            
            test_metrics = results.get("test", [{}])[0]
            
            record = {
                "dataset": str(dataset).split('_')[0].upper(), # Simplify dataset name for better plot labels
                "model": model,
                "seed": int(seed) if seed else -1,
                "Regime": "Guidance" if guidance_val != "0.0" else "No Guidance",
            }
            for key, value in test_metrics.items():
                record[key.split("/")[-1]] = value
            records.append(record)

    print(f"Successfully processed {len(records)} total records across all regimes.")
    return pd.DataFrame(records)


def plot_guidance_ablation(
    results_dir: str,
    output_dir: str,
    metrics: list[str],
    save_formats: list[str] = ["pdf", "png"]
):
    set_mpl_settings()
    plt.rcParams.update({"legend.fontsize": 14})
    plt.rcParams.update({"legend.title_fontsize": 16})
    base_results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load results from both regimes (Guidance vs No Guidance) ---
    full_df = load_and_categorize_results(base_results_path)

    if full_df.empty:
        print("No results found to plot. Aborting.")
        return

    # Save the dataframe for debugging
    df_path = output_path / "debug_full_dataframe.csv"
    print(f"Saving full dataframe to: {df_path}")
    full_df.to_csv(df_path, index=False)

    # --- Generate Plots ---
    for metric in metrics:
        if metric not in full_df.columns:
            print(f"Metric '{metric}' not found in results dataframe. Skipping.")
            continue

        g = sns.catplot(
            data=full_df,
            kind="bar",
            x="dataset",
            y=metric,
            hue="Regime",
            col="model",
            errorbar="sd",
            capsize=.05,
            height=5,
            aspect=1.5,
            legend_out=False
        )
        
        g.set_axis_labels("Dataset", metric.upper())
        g.set_titles("{col_name}")

        for k, v in g.axes_dict.items():
            # set titles from column names
            v.set_title(k.replace("simple_diff_", "").upper(), fontsize=16)
            
            

        # Save the plot
        for fmt in save_formats:
            plot_filename = output_path / f"guidance_ablation_grid_{metric}.{fmt}"
            g.savefig(plot_filename)
            print(f"Saved plot: {plot_filename}")
        
        plt.close('all')

    print(f"Guidance ablation plots saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot guidance modes ablation results.")
    parser.add_argument("--results-dir", type=str, required=True, help="Path to the unified results directory for the ablation study.")
    parser.add_argument("--output-dir", type=str, help="Output directory for plots.")
    parser.add_argument("--metrics", type=str, default="pearsonr,rmse", help="Comma-separated list of metrics to plot.")
    args = parser.parse_args()
    
    metrics_list = [m.strip() for m in args.metrics.split(",")]
    
    # If output_dir is not provided, use a default based on the results dir name
    results_path = Path(args.results_dir)
    output_dir = args.output_dir or f"artifacts/figures/ablations/{results_path.name}"

    plot_guidance_ablation(args.results_dir, output_dir, metrics_list)

if __name__ == "__main__":
    main()
