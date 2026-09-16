#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from matplotlib.lines import Line2D

import datamol as dm
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde

from MoleculeACE.benchmark.cliffs import ActivityCliffs

from diffpred.utils.checkpoint_reload import RunReconstructor
from diffpred.utils.ops import seed_everything
from diffpred.viz.utils import set_mpl_settings


@dataclass
class CliffMolecule:
    split: str
    smiles: str
    y_true: float
    x: object


@dataclass
class CliffGroup:
    test_molecule: CliffMolecule
    train_molecules: list[CliffMolecule]


def _to_feature_device(x, device: str):
    if hasattr(x, "to"):
        x = x.to(device)

    # For vector features, ensure a singleton batch dimension so len(x) reflects batch size (=1).
    if isinstance(x, torch.Tensor) and x.ndim == 1:
        x = x.unsqueeze(0)

    return x


def _debug_group_pool_stats(datamodule) -> dict[str, int]:
    train_smiles = list(datamodule.train["smiles"])
    test_cliff_smiles = list(datamodule.cliff_test["smiles"])

    overlap = set(train_smiles) & set(test_cliff_smiles)
    return {
        "len_train_smiles": len(train_smiles),
        "len_cliff_test_smiles": len(test_cliff_smiles),
        "len_train_candidates": len(train_smiles),
        "len_train_test_overlap": len(overlap),
    }


def _build_cliff_groups(
    datamodule,
    n_groups: int,
    min_train_per_group: int = 1,
    in_log10: bool = False,
    similarity: float = 0.9,
    potency_fold: float = 10.0,
) -> list[CliffGroup]:
    train_smiles = list(datamodule.train["smiles"])
    train_targets = np.asarray(datamodule.train["targets"]).reshape(-1)
    test_cliff_smiles = list(datamodule.cliff_test["smiles"])
    test_cliff_targets = np.asarray(datamodule.cliff_test["targets"]).reshape(-1)

    overlap = set(train_smiles) & set(test_cliff_smiles)
    if overlap:
        example = sorted(overlap)[0]
        raise ValueError(f"Identical SMILES found in test and train: {example}")

    combined_smiles = train_smiles + test_cliff_smiles
    combined_targets = np.concatenate([train_targets, test_cliff_targets]).astype(np.float64)
    if in_log10:
        combined_targets = np.power(10.0, combined_targets)

    cliffs = ActivityCliffs(combined_smiles, combined_targets).find_cliffs(
        similarity=float(similarity),
        potency_fold=float(potency_fold),
    )
    n_train = len(train_smiles)
    package_candidate_map: dict[int, set[int]] = {}
    for cliff_ix in range(len(test_cliff_smiles)):
        row = cliffs[n_train + cliff_ix, :n_train]
        package_candidate_map[cliff_ix] = set(np.where(row == 1)[0].tolist())

    min_train_per_group = max(1, int(min_train_per_group))
    groups: list[CliffGroup] = []
    used_test_smiles: set[str] = set()
    used_train_smiles: set[str] = set()
    used_group_train_signatures: set[tuple[str, ...]] = set()
    for cliff_ix, (test_smi, test_y) in enumerate(zip(test_cliff_smiles, test_cliff_targets)):
        if test_smi in used_test_smiles:
            continue
        matched_train: list[CliffMolecule] = []

        # Deterministic package-driven candidate order: ascending train indices marked as cliff pairs.
        for train_ix in sorted(package_candidate_map.get(cliff_ix, set())):
            train_smi = train_smiles[train_ix]
            if train_smi in used_train_smiles:
                continue
            if any(m.smiles == train_smi for m in matched_train):
                continue
            is_pair_cliff = train_ix in package_candidate_map.get(cliff_ix, set())

            if is_pair_cliff:
                train_y = float(train_targets[train_ix])
                matched_train.append(
                    CliffMolecule(
                        split="train",
                        smiles=train_smi,
                        y_true=train_y,
                        x=datamodule.train_dt[train_ix][0],
                    )
                )
            if len(matched_train) >= min_train_per_group:
                break

        if len(matched_train) < min_train_per_group:
            continue

        signature = tuple(sorted(m.smiles for m in matched_train))
        if signature in used_group_train_signatures:
            continue

        group = CliffGroup(
            test_molecule=CliffMolecule(
                split="test",
                smiles=test_smi,
                y_true=float(test_y),
                x=datamodule.cliff_test_dt[cliff_ix][0],
            ),
            train_molecules=matched_train,
        )
        groups.append(group)
        used_test_smiles.add(test_smi)
        used_train_smiles.update(m.smiles for m in matched_train)
        used_group_train_signatures.add(signature)

        if len(groups) >= n_groups:
            break

    return groups


def _extract_trajectory(model, molecule_x, num_samples: int):
    with torch.no_grad():
        preds, extras = model.sample_pred(
            x=molecule_x,
            y_dim=model.network.target_dim,
            num_samples=num_samples,
            device=model.device,
            keep_intermediates=True,
            reduce=False,
        )

    _ = preds  # Preds are equivalent to the final intermediates.
    intermediates = extras["intermediates"]
    step_ids = sorted(intermediates.keys())

    # shape: [num_steps, num_samples]
    values = []
    for step in step_ids:
        z = intermediates[step]
        if hasattr(model, "datamodule") and hasattr(model.datamodule, "inverse_target_transform"):
            z = model.datamodule.inverse_target_transform(z)
        z = z.detach().cpu()
        if z.ndim == 3:
            z = z[:, 0, 0]
        elif z.ndim == 2:
            z = z[:, 0]
        else:
            z = z.reshape(-1)
        values.append(z.numpy())

    return np.asarray(step_ids, dtype=np.int64), np.asarray(values, dtype=np.float64)


def _draw_endpoint_kde(ax, endpoint_x: float, samples: np.ndarray, color: str, side: str, width: float = 0.18):
    if len(samples) < 2 or np.allclose(samples.std(), 0):
        return

    grid = np.linspace(samples.min(), samples.max(), 200)
    kde = gaussian_kde(samples)
    density = kde(grid)
    density = density / (density.max() + 1e-12)
    if side == "left":
        x_curve = endpoint_x - density * width
    else:
        x_curve = endpoint_x + density * width

    ax.fill_betweenx(grid, endpoint_x, x_curve, color=color, alpha=0.18, linewidth=0)


def _step_ids_to_time_values(model, step_ids: np.ndarray) -> np.ndarray:
    # Intermediates are keyed by zero-based sampler iteration index.
    # For EDM sampling with num_steps=N, we store keys 0..N-1, so len(step_ids) == N.
    num_steps = int(len(step_ids))
    if num_steps == 0:
        return np.asarray([], dtype=np.float64)
    timesteps = model.diffuser.timesteps_discretization(num_steps)
    if int(np.max(step_ids)) >= num_steps:
        raise ValueError(
            f"Step index out of bounds for recovered schedule: max_step={int(np.max(step_ids))}, num_steps={num_steps}"
        )
    time_vals = timesteps[step_ids].detach().cpu().numpy().astype(np.float64)
    return time_vals


def _format_time_tick(value: float) -> str:
    # Use compact adaptive formatting so small sigmas remain distinguishable
    # without producing long labels that overlap.
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    if value >= 0.1:
        return f"{value:.2f}"
    if value >= 0.01:
        return f"{value:.2f}"
    return f"{value:.0e}"


def _render_group_molecule_strip(group: CliffGroup) -> tuple[np.ndarray | None, list[str], list[str]]:
    molecules = [group.test_molecule] + group.train_molecules
    mol_objs = []
    legends = []
    splits = []
    for idx, mol in enumerate(molecules):
        mol_obj = dm.to_mol(mol.smiles)
        if mol_obj is None:
            continue
        mol_objs.append(mol_obj)
        if idx == 0:
            label = "test"
        else:
            label = "train" if len(group.train_molecules) == 1 else f"train{idx}"
        legends.append(label)
        splits.append(mol.split)

    if not mol_objs:
        return None, [], []

    img = dm.viz.to_image(
        mol_objs,
        legends=None,
        n_cols=len(mol_objs),
        use_svg=False,
        # Larger per-molecule rendering so structures are visibly bigger without stretching.
        mol_size=(150, 150),
        legend_fontsize=12,
    )
    return np.asarray(img), legends, splits


def _add_group_molecule_strip(ax, group: CliffGroup):
    colors = {"test": "#d62728", "train": "#1f77b4"}
    strip, labels, splits = _render_group_molecule_strip(group)
    if strip is None:
        return

    # Pin strip to the top-left, legend-style.
    inset = ax.inset_axes([0.02, 0.74, 0.96, 0.25], zorder=10)
    inset.set_facecolor("white")
    inset.patch.set_alpha(0.92)
    # Preserve aspect ratio so molecule drawings are not distorted.
    inset.set_aspect("equal", adjustable="box", anchor="NW")
    inset.imshow(strip, interpolation="lanczos", aspect="equal")
    inset.set_xticks([])
    inset.set_yticks([])
    if labels:
        n = len(labels)
        for idx, (label, split) in enumerate(zip(labels, splits, strict=False)):
            x = (idx + 0.5) / n
            y = 0.08
            inset.scatter([x - 0.05], [y], s=28, color=colors.get(split, "#444444"), transform=inset.transAxes, clip_on=False)
    for spine in inset.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#BBBBBB")


def _build_split_feature_maps(datamodule) -> dict[str, dict[str, object]]:
    train_map: dict[str, object] = {}
    test_map: dict[str, object] = {}

    for idx, smi in enumerate(datamodule.train["smiles"]):
        train_map[str(smi)] = datamodule.train_dt[idx][0]

    for idx, smi in enumerate(datamodule.cliff_test["smiles"]):
        test_map[str(smi)] = datamodule.cliff_test_dt[idx][0]

    return {"train": train_map, "test": test_map}


def _predict_baseline_endpoint(model, molecule_x) -> float:
    with torch.no_grad():
        pred = model.predict(molecule_x)
        pred_t = pred.detach().cpu() if torch.is_tensor(pred) else torch.as_tensor(pred)
        flat = pred_t.reshape(-1)
        if flat.numel() != 1:
            raise ValueError(
                f"Expected scalar baseline point prediction, got shape={tuple(pred_t.shape)}"
            )
        return float(flat.item())


def _plot_group(
    ax,
    model,
    group: CliffGroup,
    num_samples: int,
    baseline_model=None,
    baseline_feature_maps: dict[str, dict[str, object]] | None = None,
):
    colors = {"test": "#d62728", "train": "#1f77b4"}
    mean_values: list[np.ndarray] = []
    target_endpoint_values: list[np.ndarray] = []
    true_values: list[float] = []
    step_axis: np.ndarray | None = None
    time_axis: np.ndarray | None = None

    molecules = [group.test_molecule] + group.train_molecules
    for mol in molecules:
        x = _to_feature_device(mol.x, model.device)
        step_ids, trajectories = _extract_trajectory(model, x, num_samples=num_samples)
        steps = step_ids.astype(np.float64)
        step_axis = step_ids
        time_axis = _step_ids_to_time_values(model, step_ids)

        max_bg = min(60, trajectories.shape[1])
        for sample_ix in range(max_bg):
            ax.plot(
                steps,
                trajectories[:, sample_ix],
                color=colors[mol.split],
                alpha=0.06,
                linewidth=1.0,
            )
        mean_traj = trajectories.mean(axis=1)
        ax.plot(
            steps,
            mean_traj,
            color=colors[mol.split],
            alpha=0.95,
            linewidth=2.6,
        )
        mean_values.append(mean_traj)

        final_mean = trajectories[-1].mean()
        ax.scatter([steps[-1]], [final_mean], color=colors[mol.split], s=24, zorder=4)

        if baseline_model is not None:
            baseline_x = None
            if baseline_feature_maps is not None:
                baseline_x = baseline_feature_maps.get(mol.split, {}).get(mol.smiles)
            if baseline_x is not None:
                baseline_x = _to_feature_device(baseline_x, baseline_model.device)
                baseline_pred = _predict_baseline_endpoint(baseline_model, baseline_x)
                ax.scatter(
                    [steps[-1]],
                    [baseline_pred],
                    marker="o",
                    facecolors="none",
                    edgecolors=colors[mol.split],
                    s=80,
                    linewidths=1.8,
                    zorder=6,
                )

        # Overlay true target value at the prediction endpoint with an x-marker.
        ax.scatter(
            [steps[-1]],
            [float(mol.y_true)],
            marker="x",
            color=colors[mol.split],
            s=56,
            linewidths=1.6,
            zorder=5,
        )
        true_values.append(float(mol.y_true))

        x_span = max(1.0, float(steps[-1] - steps[0]))
        # Keep endpoint densities visually legible even for short step ranges (e.g., 0..9).
        kde_width = max(0.8, 0.12 * x_span)
        _draw_endpoint_kde(
            ax,
            float(steps[-1]),
            trajectories[-1],
            colors[mol.split],
            side="right",
            width=kde_width,
        )
        target_endpoint_values.append(trajectories[-1])

    if step_axis is not None and len(step_axis) > 0:
        x0 = float(step_axis[0])
        x1 = float(step_axis[-1])
        x_span = max(1.0, x1 - x0)
        kde_width = max(0.8, 0.12 * x_span)
        ax.set_xlim(x0, x1 + kde_width * 1.2)
        # Keep trajectories at full step resolution, but cap displayed tick labels
        # to avoid clutter for long schedules (e.g., 50 refinement steps).
        all_ticks = np.arange(int(step_axis[0]), int(step_axis[-1]) + 1, dtype=int)
        max_tick_labels = 10
        if len(all_ticks) > max_tick_labels:
            tick_idx = np.linspace(0, len(all_ticks) - 1, num=max_tick_labels, dtype=int)
            tick_vals = all_ticks[tick_idx]
        else:
            tick_vals = all_ticks
        ax.set_xticks(tick_vals.astype(float).tolist())
        if time_axis is not None and len(time_axis) == len(step_axis):
            label_vals = []
            for tick in tick_vals.tolist():
                idx = int(tick - int(step_axis[0]))
                if 0 <= idx < len(time_axis):
                    label_vals.append(_format_time_tick(float(time_axis[idx])))
                else:
                    label_vals.append(str(int(tick)))
            ax.set_xticklabels(label_vals)
        else:
            ax.set_xticklabels([str(int(t)) for t in tick_vals.tolist()])
    ax.set_xlabel("Time")
    ax.set_ylabel("Activity")
    ax.grid(alpha=0.25)

    if mean_values:
        mean_stack = np.concatenate(mean_values)
        lo = float(np.min(mean_stack))
        hi = float(np.max(mean_stack))

        if target_endpoint_values:
            endpoint_stack = np.concatenate(target_endpoint_values)
            lo = min(lo, float(np.quantile(endpoint_stack, 0.01)))
            hi = max(hi, float(np.quantile(endpoint_stack, 0.99)))

        if true_values:
            lo = min(lo, float(np.min(np.asarray(true_values, dtype=np.float64))))
            hi = max(hi, float(np.max(np.asarray(true_values, dtype=np.float64))))

        if hi > lo:
            pad = 0.08 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)

    _add_group_molecule_strip(ax, group)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot reverse diffusion trajectories for MoleculeACE activity cliff groups."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing .hydra/config.yaml and checkpoints.",
    )
    parser.add_argument(
        "--baseline-run-dir",
        type=Path,
        default=None,
        help="Optional run directory for baseline (non-diffusion) model; plots hollow-circle endpoint predictions.",
    )
    parser.add_argument("--n-groups", type=int, default=2, help="Number of cliff groups to plot.")
    parser.add_argument(
        "--min-train-per-group",
        type=int,
        default=1,
        help="Minimum number of unique train cliff matches required per selected test group.",
    )
    parser.add_argument(
        "--in-log10",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether dataset targets are in log10 space (converted back to linear before fold-check).",
    )
    parser.add_argument("--similarity", type=float, default=0.9, help="MoleculeACE structural similarity threshold.")
    parser.add_argument("--potency-fold", type=float, default=10.0, help="MoleculeACE potency fold threshold.")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of reverse samples per molecule.")
    parser.add_argument(
        "--num-refinement-steps",
        type=int,
        default=None,
        help="Optional override for diffusion sampling refinement steps.",
    )
    parser.add_argument(
        "--dry-run-groups",
        action="store_true",
        help="Only compute and print selected cliff groups; skip model loading and plotting.",
    )
    parser.add_argument(
        "--debug-groups",
        action="store_true",
        help="Print group-construction pool stats before selection.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic behavior.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("figures") / "ac_trajectory",
        help="Output directory for PNG/PDF files.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    run_dir = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    reconstructor = RunReconstructor(run_dir=run_dir)
    cfg = reconstructor.cfg
    if cfg.data.name != "molecule_ace":
        raise ValueError(
            f"Run directory is not MoleculeACE data (cfg.data.name={cfg.data.name})."
        )
    dataset_name = str(cfg.data.dataset)

    seed_everything(args.seed)
    datamodule = reconstructor.build_datamodule()
    if args.debug_groups:
        stats = _debug_group_pool_stats(datamodule)
        print("Group pool stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    groups = _build_cliff_groups(
        datamodule,
        n_groups=args.n_groups,
        min_train_per_group=args.min_train_per_group,
        in_log10=args.in_log10,
        similarity=args.similarity,
        potency_fold=args.potency_fold,
    )
    if len(groups) == 0:
        raise RuntimeError(
            f"No train-test cliff groups found for dataset {dataset_name}."
        )

    n_groups = min(args.n_groups, len(groups))
    print(f"Run dir: {run_dir}")
    print(f"Dataset: {dataset_name}")
    print(f"Selected groups: {n_groups}")
    for ix, group in enumerate(groups[:n_groups]):
        train_smiles = ", ".join([mol.smiles for mol in group.train_molecules])
        print(f"  Group {ix + 1}: test={group.test_molecule.smiles} | train={train_smiles}")

    if args.dry_run_groups:
        return

    model, resolved_ckpt = reconstructor.load_checkpointed_model(
        datamodule=datamodule,
        ckpt_path=None,
        use_ema=None,
    )
    model.eval()

    if args.num_refinement_steps is not None:
        if args.num_refinement_steps < 1:
            raise ValueError("--num-refinement-steps must be >= 1")
        if not hasattr(model, "diffuser") or not hasattr(model.diffuser, "refinement_steps"):
            raise TypeError("Loaded main model does not expose diffuser.refinement_steps")
        model.diffuser.refinement_steps = int(args.num_refinement_steps)
        print(f"Overriding refinement_steps to {model.diffuser.refinement_steps}")

    baseline_model = None
    baseline_ckpt = None
    baseline_feature_maps = None
    if args.baseline_run_dir is not None:
        baseline_run_dir = args.baseline_run_dir
        if not baseline_run_dir.exists():
            raise FileNotFoundError(f"Baseline run directory not found: {baseline_run_dir}")

        baseline_reconstructor = RunReconstructor(run_dir=baseline_run_dir)
        if not baseline_reconstructor.is_baseline:
            raise ValueError(
                "--baseline-run-dir must point to a baseline (non-diffusion) run."
            )
        baseline_cfg = baseline_reconstructor.cfg
        if baseline_cfg.data.name != "molecule_ace":
            raise ValueError(
                f"Baseline run directory is not MoleculeACE data (cfg.data.name={baseline_cfg.data.name})."
            )
        baseline_dataset_name = str(baseline_cfg.data.dataset)
        if baseline_dataset_name != dataset_name:
            raise ValueError(
                f"Baseline dataset mismatch: main={dataset_name}, baseline={baseline_dataset_name}."
            )

        baseline_datamodule = baseline_reconstructor.build_datamodule()
        baseline_model, baseline_ckpt = baseline_reconstructor.load_checkpointed_model(
            datamodule=baseline_datamodule,
            ckpt_path=None,
            use_ema=None,
        )
        baseline_model.eval()
        baseline_feature_maps = _build_split_feature_maps(baseline_datamodule)

        print(f"Baseline run dir: {baseline_run_dir}")
        print(f"Baseline checkpoint: {baseline_ckpt}")

    set_mpl_settings()
    fig, axes = plt.subplots(1, n_groups, figsize=(8 * n_groups, 6), sharey=False)
    if isinstance(axes, np.ndarray):
        axes = axes.ravel().tolist()
    else:
        axes = [axes]

    for ix in range(n_groups):
        _plot_group(
            axes[ix],
            model,
            groups[ix],
            num_samples=args.n_samples,
            baseline_model=baseline_model,
            baseline_feature_maps=baseline_feature_maps,
        )

    for ix, ax in enumerate(axes):
        if ix == 0:
            ax.set_ylabel("Activity")
        else:
            ax.set_ylabel("")

    legend_handles = [
        Line2D([0], [0], color="#d62728", linewidth=2.6, label="test"),
        Line2D([0], [0], color="#1f77b4", linewidth=2.6, label="train"),
    ]
    if baseline_model is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker="o",
                markerfacecolor="none",
                markeredgecolor="#333333",
                markeredgewidth=1.6,
                markersize=7,
                label="baseline",
            )
        )
    legend_handles.append(
        Line2D(
            [0],            
            [0],
            linestyle="None",
            marker="x",
            color="#333333",
            markersize=7,
            label="ground truth",
        )
    )
    axes[0].legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        framealpha=0.9,
        fontsize=14,
        title="Trajectories",
        title_fontsize=14,
        borderpad=0.9,
        labelspacing=0.6,
        handlelength=2.6,
    )

    fig.tight_layout()

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"ac_trajectory_{dataset_name}_{resolved_ckpt.stem}"
    if baseline_ckpt is not None:
        stem = f"{stem}_vs_{baseline_ckpt.stem}"
    png_path = args.outdir / f"{stem}.png"
    pdf_path = args.outdir / f"{stem}.pdf"

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Checkpoint: {resolved_ckpt}")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
