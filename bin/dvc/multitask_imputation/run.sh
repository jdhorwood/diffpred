#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

dataset="dockstring"
RATIOS="0.0,0.1,0.2,0.3,0.4,0.5"
SEEDS="1,2,3"
MODELS="mlp,mlp_diff"
N_JOBS="6"
EPOCH_OVERRIDE=""
WANDB_MODE="disabled"
RESULTS_ROOT="artifacts/results_raw/multitask_imputation"
DATA_ROOT="artifacts/data/multitask_imputation"
CHECK_VAL_FREQ=""
DEVICE_OVERRIDE=""
NUM_PRED_SAMPLES_OVERRIDE=""
REFINEMENT_STEPS_OVERRIDE=""

usage() {
  cat <<'USAGE'
Usage: bash bin/dvc/multitask_imputation/run.sh [options]

Options:
  --dataset <csv>        Dataset keys. Default: dockstring
  --ratios <csv>         Mask ratios. Default: 0.0,0.1,0.2,0.3,0.4,0.5
  --seeds <csv>          Seeds. Default: 1,2,3,4,5
  --models <csv>         Models. Default: mlp,mlp_diff
  --n-jobs <int>         Hydra launcher parallel jobs
  --epochs <int>         Optional max-epochs override
  --wandb-mode <mode>    WandB mode (default: disabled)
  --check-val-freq <int> Optional validation frequency override
  --device <cpu|cuda>    Optional experiment.device override
  --num-pred-samples <int> Optional model.training.num_pred_samples override
  --refinement-steps <int> Optional data.diffusion_params.refinement_steps override
  --results-root <path>  Root output for raw run results
  --data-root <path>     Root path for Stage A artifacts
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      dataset="$2"
      shift 2
      ;;
    --ratios)
      RATIOS="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --models)
      MODELS="$2"
      shift 2
      ;;
    --n-jobs)
      N_JOBS="$2"
      shift 2
      ;;
    --epochs)
      EPOCH_OVERRIDE="$2"
      shift 2
      ;;
    --wandb-mode)
      WANDB_MODE="$2"
      shift 2
      ;;
    --check-val-freq)
      CHECK_VAL_FREQ="$2"
      shift 2
      ;;
    --device)
      DEVICE_OVERRIDE="$2"
      shift 2
      ;;
    --num-pred-samples)
      NUM_PRED_SAMPLES_OVERRIDE="$2"
      shift 2
      ;;
    --refinement-steps)
      REFINEMENT_STEPS_OVERRIDE="$2"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

mkdir -p "$RESULTS_ROOT"
DATA_ROOT_ABS="$(cd "$ROOT_DIR" && realpath "$DATA_ROOT")"


SWEEP_DIR="$RESULTS_ROOT/$dataset"
HYDRA_SUBDIR="ratio_\${data.masking.ratio}/\${model.name}/\${hydra.job.num}"

EXTRA_ARGS=()
if [[ -n "$EPOCH_OVERRIDE" ]]; then
  EXTRA_ARGS+=("model.training.max_epochs=$EPOCH_OVERRIDE")
fi
if [[ -n "$CHECK_VAL_FREQ" ]]; then
  EXTRA_ARGS+=("experiment.check_val_freq=$CHECK_VAL_FREQ")
fi
if [[ -n "$DEVICE_OVERRIDE" ]]; then
  EXTRA_ARGS+=("experiment.device=$DEVICE_OVERRIDE")
fi
if [[ -n "$NUM_PRED_SAMPLES_OVERRIDE" ]]; then
  EXTRA_ARGS+=("model.training.num_pred_samples=$NUM_PRED_SAMPLES_OVERRIDE")
fi
if [[ -n "$REFINEMENT_STEPS_OVERRIDE" ]]; then
  EXTRA_ARGS+=("data.diffusion_params.refinement_steps=$REFINEMENT_STEPS_OVERRIDE")
fi

uv run python diffpred/run.py \
  data=dataframe_masked_imputation \
  model="$MODELS" \
  experiment.seed="$SEEDS" \
  data.dataset="$dataset" \
  data.masking.ratio="$RATIOS" \
  data.masking.artifact_root="$DATA_ROOT_ABS/$dataset" \
  hydra.sweep.dir="$SWEEP_DIR" \
  hydra.sweep.subdir="$HYDRA_SUBDIR" \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs="$N_JOBS" \
  experiment.wandb_mode="$WANDB_MODE" \
  "${EXTRA_ARGS[@]}" \
  -m

echo "Multitask imputation raw runs complete: $RESULTS_ROOT"
