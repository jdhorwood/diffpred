#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

EXPERIMENT_ID="single_task_full"
DEVICE="cuda"
N_JOBS="8"
EPOCHS="800"
SEEDS="1,2,3,4,5"
TDC_DATASETS="lipophilicity_astrazeneca,solubility_aqsoldb,clearance_hepatocyte_az,clearance_microsome_az,vdss_lombardo,ld50_zhu"
CHEMBL_DATASETS="CHEMBL4203_Ki,CHEMBL2034_Ki,CHEMBL233_Ki,CHEMBL4616_EC50,CHEMBL287_Ki,CHEMBL218_EC50,CHEMBL264_Ki,CHEMBL219_Ki,CHEMBL2835_Ki,CHEMBL2147_Ki,CHEMBL231_Ki,CHEMBL3979_EC50,CHEMBL237_EC50,CHEMBL244_Ki,CHEMBL4792_Ki,CHEMBL1871_Ki,CHEMBL237_Ki,CHEMBL262_Ki,CHEMBL2047_EC50,CHEMBL239_EC50,CHEMBL2971_Ki,CHEMBL204_Ki,CHEMBL214_Ki,CHEMBL1862_Ki,CHEMBL234_Ki,CHEMBL238_Ki,CHEMBL235_EC50,CHEMBL4005_Ki,CHEMBL236_Ki"
BASELINE_MODELS="mlp,gcn,gin,gine,pna,chemprop"
DIFF_MODELS="simple_diff_mlp,simple_diff_gcn,simple_diff_gin,simple_diff_gine,simple_diff_pna,simple_diff_chemprop"

usage() {
  cat <<'USAGE'
Usage: bash bin/dvc/single_task/run.sh [options]

Options:
  --experiment-id <id>
  --device <cpu|cuda>
  --n-jobs <int>
  --epochs <int>
  --seeds <csv>
  --tdc-datasets <csv>
  --chembl-datasets <csv>
  --baseline-models <csv>
  --diff-models <csv>
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-id)
      EXPERIMENT_ID="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --n-jobs)
      N_JOBS="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --tdc-datasets)
      TDC_DATASETS="$2"
      shift 2
      ;;
    --chembl-datasets)
      CHEMBL_DATASETS="$2"
      shift 2
      ;;
    --baseline-models)
      BASELINE_MODELS="$2"
      shift 2
      ;;
    --diff-models)
      DIFF_MODELS="$2"
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

mkdir -p "$ROOT_DIR/data" "$ROOT_DIR/artifacts"

if [[ -z "${SCRATCH_OUTPUT_ROOT:-}" ]]; then
  echo "SCRATCH_OUTPUT_ROOT must be set (for example via .envrc)." >&2
  exit 1
fi

RESULTS_ROOT="$SCRATCH_OUTPUT_ROOT/artifacts/results_raw/single_task/$EXPERIMENT_ID"
mkdir -p "$RESULTS_ROOT"

run_hydra_multirun() {
  uv run python diffpred/run.py "$@" -m
}

run_hydra_multirun \
  data=tdc \
  data.dataset="$TDC_DATASETS" \
  model="$BASELINE_MODELS" \
  model.training.max_epochs="$EPOCHS" \
  experiment.check_val_freq=1 \
  hydra.launcher.n_jobs="$N_JOBS" \
  hydra.sweep.dir="$RESULTS_ROOT/tdc_baseline" \
  experiment.seed="$SEEDS" \
  experiment.device="$DEVICE" \
  experiment.wandb_mode=disabled

run_hydra_multirun \
  data=tdc \
  data.dataset="$TDC_DATASETS" \
  model="$DIFF_MODELS" \
  model.training.max_epochs="$EPOCHS" \
  hydra.launcher.n_jobs="$N_JOBS" \
  hydra.sweep.dir="$RESULTS_ROOT/tdc_diffpred" \
  experiment.seed="$SEEDS" \
  experiment.device="$DEVICE" \
  experiment.wandb_mode=disabled

run_hydra_multirun \
  data=molecule_ace \
  data.dataset="$CHEMBL_DATASETS" \
  model="$BASELINE_MODELS" \
  model.training.max_epochs="$EPOCHS" \
  experiment.check_val_freq=1 \
  hydra.launcher.n_jobs="$N_JOBS" \
  hydra.sweep.dir="$RESULTS_ROOT/molecule_ace_baseline" \
  experiment.seed="$SEEDS" \
  experiment.device="$DEVICE" \
  experiment.wandb_mode=disabled

run_hydra_multirun \
  data=molecule_ace \
  data.dataset="$CHEMBL_DATASETS" \
  model="$DIFF_MODELS" \
  model.training.max_epochs="$EPOCHS" \
  hydra.launcher.n_jobs="$N_JOBS" \
  hydra.sweep.dir="$RESULTS_ROOT/molecule_ace_diffpred" \
  experiment.seed="$SEEDS" \
  experiment.device="$DEVICE" \
  experiment.wandb_mode=disabled

echo "Raw results complete: $RESULTS_ROOT"
