#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

RESULTS_ROOT="artifacts/results_raw/calibration"
N_JOBS_SMALL="5"
N_JOBS_LARGE="5"
mkdir -p "$RESULTS_ROOT"

COMMON_ARGS=(
  "experiment.seed=0,1,2,3,4"
  "experiment.callbacks=[calibration_plot]"
  "experiment.wandb_mode=disabled"
  "hydra/launcher=joblib"
  "hydra.sweep.dir=$RESULTS_ROOT"
)

run_multirun() {
  uv run python diffpred/run.py \
    "$@" \
    "${COMMON_ARGS[@]}" \
    -m
}

BASELINES_UNCS=(
  "model.use_mve=True"
  "model.use_deepensemble=True"
  "model.use_mcdropout=True"
)

# ------------------------------------ Baselines ------------------------------------
for unc_type in "${BASELINES_UNCS[@]}"; do
  # MA
  run_multirun \
    data=molecule_ace \
    data.dataset=CHEMBL228_Ki \
    model=mlp \
    model.training.max_epochs=600 \
    hydra.launcher.n_jobs="$N_JOBS_SMALL" \
    "$unc_type"

  # CACO2_WANG
  run_multirun \
    data=tdc \
    data.dataset=caco2_wang \
    model=gin \
    model.hparams.output_mlp_type=linear \
    model.training.max_epochs=800 \
    hydra.launcher.n_jobs="$N_JOBS_SMALL" \
    "$unc_type"

  # Solubility
  run_multirun \
    data=tdc \
    data.dataset=solubility_aqsoldb \
    model=gcn \
    model.hparams.output_mlp_type=linear \
    model.training.max_epochs=800 \
    hydra.launcher.n_jobs="$N_JOBS_SMALL" \
    "$unc_type"

  # Dockstring 1
  run_multirun \
    data=dockstring \
    data.dataset=ABL1 \
    model=mlp \
    model.training.max_epochs=100 \
    hydra.launcher.n_jobs="$N_JOBS_LARGE" \
    "$unc_type"

  # Dockstring 2
  run_multirun \
    data=dockstring \
    data.dataset=CASP3 \
    model=mlp \
    model.training.max_epochs=100 \
    hydra.launcher.n_jobs="$N_JOBS_LARGE" \
    "$unc_type"
done

# ------------------------------------ DiffPred ------------------------------------
DIFF_UNC_ARGS=(
  "data.diffusion_params.guidance_dropout=0.2"
  "data.diffusion_params.refinement_steps=50"
)

# MA
run_multirun \
  data=molecule_ace \
  data.dataset=CHEMBL228_Ki \
  model=simple_diff_mlp \
  model.training.max_epochs=600 \
  hydra.launcher.n_jobs="$N_JOBS_SMALL" \
  "${DIFF_UNC_ARGS[@]}"

# CACO2_WANG
run_multirun \
  data=tdc \
  data.dataset=caco2_wang \
  model=simple_diff_gin \
  model.training.max_epochs=800 \
  hydra.launcher.n_jobs="$N_JOBS_SMALL" \
  "${DIFF_UNC_ARGS[@]}"

# Solubility
run_multirun \
  data=tdc \
  data.dataset=solubility_aqsoldb \
  model=simple_diff_gcn \
  model.training.max_epochs=800 \
  hydra.launcher.n_jobs="$N_JOBS_SMALL" \
  "${DIFF_UNC_ARGS[@]}"

# Dockstring 1
run_multirun \
  data=dockstring \
  data.dataset=ABL1 \
  model=simple_diff_mlp \
  model.training.max_epochs=100 \
  hydra.launcher.n_jobs="$N_JOBS_LARGE" \
  "${DIFF_UNC_ARGS[@]}"

# Dockstring 2
run_multirun \
  data=dockstring \
  data.dataset=CASP3 \
  model=simple_diff_mlp \
  model.training.max_epochs=100 \
  hydra.launcher.n_jobs="$N_JOBS_LARGE" \
  "${DIFF_UNC_ARGS[@]}"

printf 'Calibration raw results complete: %s\n' "$RESULTS_ROOT"
