#!/usr/bin/env bash
set -euo pipefail

# Find project root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

# Arguments from DVC/CLI
EXPERIMENT_ID="${1}"
MODELS="${2}"
SEEDS="${3}"
GUIDANCE_VALS="${4}"
GUIDANCE_DROPOUT="${5}"
GUIDANCE_SCHEDULE="${6}"
DEFAULT_EPOCHS="${7}"
DEFAULT_N_JOBS="${8}"
DOCKSTRING_EPOCHS="${9}"
DOCKSTRING_N_JOBS="${10}"
MOLECULE_ACE_DATASETS="${11}"
TDC_DATASETS="${12}"
DOCKSTRING_DATASETS="${13}"

RESULTS_ROOT="artifacts/results_raw/ablations/${EXPERIMENT_ID}"

# --- Function to run Hydra with provided args ---
run_hydra() {
    local max_epochs=$1
    local n_jobs=$2
    shift 2

    echo "Running Hydra for: $@ (epochs=${max_epochs}, jobs=${n_jobs})"

    uv run python diffpred/run.py \
        "$@" \
        "experiment.seed=$SEEDS" \
        "model.training.max_epochs=$max_epochs" \
        "hydra.launcher.n_jobs=$n_jobs" \
        "hydra.sweep.dir=$RESULTS_ROOT" \
        "data.diffusion_params.guidance_dropout=$GUIDANCE_DROPOUT" \
        "data.diffusion_params.guidance_schedule=$GUIDANCE_SCHEDULE" \
        "data.diffusion_params.guidance=$GUIDANCE_VALS" \
        "experiment.wandb_mode=disabled" \
        "-m"
}

# --- Execute runs for each dataset group ---
if [[ -n "$MOLECULE_ACE_DATASETS" ]]; then
    run_hydra "$DEFAULT_EPOCHS" "$DEFAULT_N_JOBS" data=molecule_ace "data.dataset=${MOLECULE_ACE_DATASETS}" "model=${MODELS}"
fi
if [[ -n "$TDC_DATASETS" ]]; then
    run_hydra "$DEFAULT_EPOCHS" "$DEFAULT_N_JOBS" data=tdc "data.dataset=${TDC_DATASETS}" "model=${MODELS}"
fi
if [[ -n "$DOCKSTRING_DATASETS" ]]; then
    run_hydra "$DOCKSTRING_EPOCHS" "$DOCKSTRING_N_JOBS" data=dockstring "data.dataset=${DOCKSTRING_DATASETS}" "model=${MODELS}"
fi

printf 'Guidance modes ablation run complete: %s\n' "$RESULTS_ROOT"
