#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

GUIDANCE="0.9"
GUIDANCE_DROPOUTS="0,0.2,0.4,0.6,0.8,1.0"
EPOCHS="500"
N_JOBS="10"
CALLBACKS='[sigma_losses]'
RESULTS_ROOT="artifacts/results_raw/ablations/guidance"

mkdir -p "$RESULTS_ROOT"

uv run python diffpred/run.py \
  data=molecule_ace \
  data.dataset=CHEMBL228_Ki \
  data.diffusion_params.guidance="$GUIDANCE" \
  data.diffusion_params.guidance_dropout="$GUIDANCE_DROPOUTS" \
  experiment.callbacks="$CALLBACKS" \
  data.diffusion_params.guidance_schedule=constant \
  model=simple_diff_mlp \
  hydra.launcher.n_jobs="$N_JOBS" \
  model.training.max_epochs="$EPOCHS" \
  hydra.sweep.dir="$RESULTS_ROOT" \
  experiment.save_last_ckpt=True \
  -m

uv run python diffpred/run.py \
  data=molecule_ace \
  data.dataset=CHEMBL228_Ki \
  data.diffusion_params.guidance=1 \
  data.diffusion_params.guidance_dropout=0.5 \
  experiment.callbacks="$CALLBACKS" \
  data.diffusion_params.guidance_schedule=cosine \
  model=simple_diff_mlp \
  hydra.launcher.n_jobs=1 \
  model.training.max_epochs="$EPOCHS" \
  hydra.sweep.dir="$RESULTS_ROOT" \
  experiment.save_last_ckpt=True \
  -m

printf 'Guidance ablation raw results complete: %s\n' "$RESULTS_ROOT"
