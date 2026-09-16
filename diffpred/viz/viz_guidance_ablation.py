import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.lines as mlines

from diffpred.diffusers.edm import GuidanceScheduler
from diffpred.utils.reloader import load_config
from diffpred.viz.utils import set_mpl_settings


@dataclass
class RunEntry:
    run_dir: Path
    metadata: dict[str, Any]
    pkl_by_split: dict[str, Path]


class GuidanceAblationViz:
    def __init__(
        self,
        run_dir: str,
        dataset: str | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.dataset = dataset
        if not self.run_dir.exists() or not self.run_dir.is_dir():
            raise ValueError(f"run_dir is not a valid directory: {self.run_dir}")

    @staticmethod
    def _coerce_value(value: str) -> Any:
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def parse_run_metadata(self, run_path: Path) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for token in run_path.name.split(","):
            if "=" not in token:
                continue
            key, raw_value = token.split("=", 1)
            short_key = key.split(".")[-1]
            metadata[short_key] = self._coerce_value(raw_value)

        required = {"dataset", "guidance_schedule", "guidance_dropout"}
        if required.issubset(metadata.keys()):
            return metadata

        # Fallback to Hydra config for robustness when path tokens are incomplete.
        try:
            cfg = load_config(str(run_path))
            metadata.setdefault("dataset", cfg.data.dataset)
            metadata.setdefault(
                "guidance_schedule", cfg.data.diffusion_params.guidance_schedule
            )
            metadata.setdefault(
                "guidance_dropout", cfg.data.diffusion_params.guidance_dropout
            )
            metadata.setdefault("guidance", cfg.data.diffusion_params.guidance)
            metadata.setdefault("model", cfg.model.name)
        except Exception:
            pass

        return metadata

    @staticmethod
    def _resolve_split_pkl(run_path: Path, split: str) -> Path | None:
        candidates = sorted(run_path.glob(f"*sigma_losses_after_{split}.pkl"))
        if not candidates:
            candidates = sorted(run_path.rglob(f"*sigma_losses_after_{split}.pkl"))
        if not candidates:
            return None
        if len(candidates) > 1:
            print(
                f"WARNING: Multiple sigma loss pickles for split={split} in {run_path}; using {candidates[0]}"
            )
        return candidates[0]

    def discover_paths(self) -> tuple[list[RunEntry], list[RunEntry]]:
        ablation_entries: list[RunEntry] = []
        cosine_entries: list[RunEntry] = []

        for child in sorted(self.run_dir.iterdir()):
            if not child.is_dir():
                continue

            metadata = self.parse_run_metadata(child)
            dataset = metadata.get("dataset")
            if self.dataset is not None and dataset != self.dataset:
                continue

            split_paths = {
                split: p
                for split in ("train", "val")
                if (p := self._resolve_split_pkl(child, split)) is not None
            }
            if not split_paths:
                continue

            schedule = metadata.get("guidance_schedule")
            entry = RunEntry(run_dir=child, metadata=metadata, pkl_by_split=split_paths)
            if schedule == "constant":
                ablation_entries.append(entry)
            elif schedule == "cosine":
                cosine_entries.append(entry)

        # Sort ablation runs by dropout for consistent plotting order.
        ablation_entries.sort(key=lambda x: float(x.metadata.get("guidance_dropout", 1e9)))
        return ablation_entries, cosine_entries

    @staticmethod
    def load_sigma_losses(pkl_path: Path) -> tuple[list[float], list[float]]:
        data = pd.read_pickle(pkl_path)
        if not isinstance(data, dict) or len(data) == 0:
            raise ValueError(f"Unexpected sigma loss payload in {pkl_path}")

        sorted_items = sorted(data.items(), key=lambda kv: float(kv[0]))
        sigmas: list[float] = []
        means: list[float] = []
        for sigma, value in sorted_items:
            sigma_f = float(sigma)
            if isinstance(value, (list, tuple)) and len(value) > 0:
                mean = float(value[0])
            elif isinstance(value, (list, tuple)):
                raise ValueError(f"Empty sequence value for sigma={sigma_f} in {pkl_path}")
            else:
                mean = float(value)
            sigmas.append(sigma_f)
            means.append(mean)

        return sigmas, means

    def plot_curves(
        self,
        split: str,
        ablation_entries: list[RunEntry],
        cosine_entries: list[RunEntry],
    ) -> tuple[Figure, Axes]:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        valid_ablation = [entry for entry in ablation_entries if split in entry.pkl_by_split]
        if not valid_ablation:
            raise ValueError(f"No ablation runs with split='{split}' found under {self.run_dir}")

        for entry in valid_ablation:
            pkl_path = entry.pkl_by_split[split]
            sigmas, means = self.load_sigma_losses(pkl_path)
            dropout = entry.metadata.get("guidance_dropout")
            label = rf"$d_{{\mathrm{{max}}}}$: {dropout}"
            ax.plot(sigmas, means, label=label, alpha=0.7)

        valid_cosine = [entry for entry in cosine_entries if split in entry.pkl_by_split]
        if len(valid_cosine) > 1:
            print(
                f"WARNING: Found {len(valid_cosine)} cosine runs for split={split}; plotting all of them"
            )
        for ix, entry in enumerate(valid_cosine):
            pkl_path = entry.pkl_by_split[split]
            sigmas, means = self.load_sigma_losses(pkl_path)
            label = "cosine" if ix == 0 else f"cosine_{ix + 1}"
            ax.plot(sigmas, means, label=label, alpha=0.8, color="red")

        split_label = "Train" if split == "train" else "Validation"
        ax.set_xscale("log")
        ax.set_xlabel("Sigma")
        ax.set_ylabel(f"{split_label} Loss")
        ax.legend()
        return fig, ax

    def run(
        self,
        split: str,
        output_dir: str,
        save_formats: list[str],
        show: bool = False,
    ):
        set_mpl_settings()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ablation_entries, cosine_entries = self.discover_paths()
        if not ablation_entries:
            raise ValueError(
                f"No constant-schedule ablation runs found under {self.run_dir}"
            )

        splits = ["train", "val"] if split == "both" else [split]
        for curr_split in splits:
            fig, _ = self.plot_curves(
                split=curr_split,
                ablation_entries=ablation_entries,
                cosine_entries=cosine_entries,
            )
            for fmt in save_formats:
                out_path = out_dir / f"guidance_dropout_ablation_{curr_split}.{fmt}"
                fig.savefig(str(out_path), bbox_inches="tight")
                print(f"Saved {out_path}")
            if show:
                plt.show()
            plt.close(fig)

        self.viz_guidance_schedulers(
            output_dir=out_dir,
            save_formats=save_formats,
            show=show,
        )

    def viz_guidance_schedulers(
        self,
        output_dir: Path,
        save_formats: list[str],
        show: bool = False,
    ) -> None:

        schedules = ["log_linear", "cosine", "exponential"]
        sigma_min = 0.0001
        sigma_max = 5
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        set_mpl_settings()

        for schedule in schedules:
            scheduler = GuidanceScheduler(
                schedule=schedule,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                constant_guidance=0.5,
            )
            num_samples = 1000
            sigmas = torch.linspace(sigma_min, sigma_max, num_samples)
            guidance_values = [float(scheduler.get_guidance(sigma)) for sigma in sigmas]
            dropout_rates = [float(scheduler.get_dropout_rate(sigma)) for sigma in sigmas]
            (guidance_line,) = ax.plot(
                sigmas.numpy(), guidance_values, label=schedule, alpha=0.8
            )
            (dropout_line,) = ax.plot(
                sigmas.numpy(),
                dropout_rates,
                linestyle="--",
                alpha=0.5,
                color=guidance_line.get_color(),
                label=None,
            )

        dropout_handle = mlines.Line2D(
            [], [], color="grey", alpha=0.5, linestyle="--", label="Dropout rate"
        )
        handles, labels = ax.get_legend_handles_labels()
        handles.append(dropout_handle)
        labels.append("Dropout rate")
        ax.set_xlabel("Noise Level (Sigma)")
        ax.set_ylabel("Guidance")
        ax.legend(handles, labels)
        fig.tight_layout()

        for fmt in save_formats:
            out_path = output_dir / f"guidance_schedulers.{fmt}"
            fig.savefig(str(out_path), bbox_inches="tight")
            print(f"Saved {out_path}")

        if show:
            plt.show()
        plt.close(fig)

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot guidance-dropout ablation sigma-loss curves from a run parent directory."
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Parent directory containing experiment subdirectories.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "both"],
        default="both",
        help="Which split(s) to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory where plot files are saved.",
    )
    parser.add_argument(
        "--save-format",
        nargs="+",
        default=["pdf"],
        help="One or more output formats, e.g. pdf png.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset filter (e.g., CHEMBL228_Ki).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively.",
    )
    return parser


def main():
    args = _build_parser().parse_args()
    viz = GuidanceAblationViz(run_dir=args.run_dir, dataset=args.dataset)
    viz.run(
        split=args.split,
        output_dir=args.output_dir,
        save_formats=args.save_format,
        show=args.show,
    )


if __name__ == "__main__":
    main()