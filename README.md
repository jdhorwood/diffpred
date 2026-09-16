# Diffpred

## Installation
0. We use uv for installation. If not already installed, this needs to by installed first via `curl -LsSf https://astral.sh/uv/install.sh | sh`.
1. Build the project environment with `uv sync`.
2. Install dvc : https://doc.dvc.org/install
3. Create the `data/` and `artifacts/` directories and set `SCRATCH_OUTPUT_ROOT` (required for every run, not just the DVC pipelines) — see [Setup and Storage](#setup-and-storage).

### Platform notes

This environment was set up and run on Linux with NVIDIA GPUs. `torch` is pinned
to the CUDA 12.8 wheel index in [pyproject.toml](pyproject.toml):

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
```

Change the index URL to match your CUDA version (for example
`https://download.pytorch.org/whl/cu121`), or to `https://download.pytorch.org/whl/cpu`
for a CPU-only install. See https://pytorch.org/get-started/locally/ for the
right URL for your setup. macOS is not covered by these indexes; drop the
`[[tool.uv.index]]` and `[tool.uv.sources]` blocks entirely to install the
default CPU/MPS build.

The project requires Python 3.11.

`experiment.device` is passed through as the Lightning `accelerator` and
defaults to `auto`, so direct runs detect the device themselves and fall back to
CPU. The DVC pipeline parameters in [params.yaml](params.yaml) instead request
`cuda` explicitly (`single_task.device` and
`single_task_additional_figures.activity_cliff_device`), which will fail without
a GPU — set those to `cpu` if you are running on one.

## DVC Pipelines

Experiments are organized into DVC pipelines for maximum reproducibility and tracking.

### Setup and Storage

Two directories must exist at the repository root before anything runs:
- `data/` — raw datasets and the featurization cache
- `artifacts/` — results and figures

`SCRATCH_OUTPUT_ROOT` must also be set.

The simplest setup keeps both directories inside the repository (they are
gitignored):

```bash
export SCRATCH_OUTPUT_ROOT="$PWD"
mkdir -p data artifacts

uv run dvc init # if not already initialized
```

Rooting the storage outside the repository is recommended for real runs. Point `SCRATCH_OUTPUT_ROOT` at the external
location and symlink both paths back into the repository, so that the
repo-relative output paths in `dvc.yaml` work as expected:

```bash
export SCRATCH_OUTPUT_ROOT=<path_to_scratch_output>
mkdir -p "$SCRATCH_OUTPUT_ROOT/data" "$SCRATCH_OUTPUT_ROOT/artifacts"
ln -s "$SCRATCH_OUTPUT_ROOT/data" data
ln -s "$SCRATCH_OUTPUT_ROOT/artifacts" artifacts

uv run dvc init # if not already initialized
```

### Running Pipelines

All pipeline parameters live in [params.yaml](params.yaml).

#### Single Task
- `single_task.raw_results`: Generate training outputs.
- `single_task.figures`: Generate standard figures (CHEMBL, TDC).
- `single_task.generalization_figures`: Generate generalization gap figures.
- `single_task.activity_cliff_figures`: Generate activity-cliff trajectories.

```bash
uv run dvc repro <stage_name>
```

#### Multi-Task
- `multitask.prepare_data`: Prepare Polaris benchmark splits.
- `multitask.raw_results`: Run multi-task training (ADME-Fang, Tox21, Zinc).
- `multitask.tables`: Generate performance summary tables.

```bash
uv run dvc repro multitask.tables
```

#### Multi-Task Imputation
- `multitask_imputation.data_prep`: Prepare datasets with varying masking ratios.
- `multitask_imputation.raw_results`: Run imputation training.
- `multitask_imputation.evaluation`: Evaluate imputation accuracy.

```bash
uv run dvc repro multitask_imputation.evaluation
```

#### Ablations and Calibration
- `ablations.guidance.raw_results` / `ablations.guidance.figures`: Diffusion guidance ablations.
- `ablations.guidance_modes.raw_results` / `ablations.guidance_modes.figures`: Guidance mode comparisons.
- `calibration`: Stages for uncertainty quantification (rejection curves, predictive intervals).

## Launching Experiments Directly

We use Hydra's CLI tools for dispatching experiments.
For example, to run a DiffPred run with an MLP featurization on a dataset from Molecule ACE, run

```bash
uv run python diffpred/run.py data=molecule_ace model=simple_diff_mlp \
data.dataset=CHEMBL233_Ki model.training.max_epochs=500 experiment.device=cpu
```

## Configuring Runs

Configuration files are found in diffpred/configs.
We have subconfigs for the model, data and general experiment components.

## Organization of the repository

High‑level layout:
- diffpred/run.py: Hydra entrypoint.
- diffpred/configs/: Hydra config tree.
  - data/: dataset configs.
  - model/: model architecture / training configs.
  - experiment/: logging, optimization, callbacks, device.
- diffpred/** (modules): core implementation (models, diffusion logic, utils).
- pyproject.toml / uv.lock: dependency specification.

## Typical workflow

1. Dry run with a short epoch count:

  ```bash
  uv run python diffpred/run.py data=<cfg> model=<cfg> model.training.max_epochs=5
  ```
2. Launch a full run; outputs land in outputs/YYYY-MM-DD/<job_name>/ created by Hydra by default.
3. Inspect outputs:
   - .hydra/config.yaml: exact resolved config (store this for reproducibility).
   - metrics (e.g. JSON / CSV, depending on logger).
   - checkpoints/ for model weights.

## Using / overriding configs

Hydra style overrides:
- Dot notation: model.training.lr=1e-4
- Add new grouped choice: data=<new_dataset_cfg>
- Multi overrides:
  ```bash
  uv run python diffpred/run.py data=molecule_ace model=simple_diff_mlp \
  model.training.batch_size=256 seed=42 experiment.device=cuda
  ```

To list available options:
```bash
uv run python diffpred/run.py --config-path diffpred/configs --config-name default --help
```

## Sweeps (multi-run)

Use Hydra multirun:
```bash
uv run python diffpred/run.py -m model.training.lr=1e-3,1e-4 experiment.seed=1,2
```
Outputs: multirun/…


