#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p "$ROOT_DIR/data" "$ROOT_DIR/artifacts"

RESULTS_ROOT="artifacts/results_raw/multitask"
mkdir -p "$RESULTS_ROOT"

ADME_DIR_PATH="$ROOT_DIR/diffpred/exp_configs/adme-fang"
TOX21_DIR_PATH="$ROOT_DIR/diffpred/exp_configs/tox21"
ZINC_DIR_PATH="$ROOT_DIR/diffpred/exp_configs/zinc"
graphormer_models=('graphormer' 'graphormer_diff')
other_models=('gcn' 'gcn_diff' 'gin' 'gin_diff' 'gine' 'gine_diff' 'gps' 'gps_diff' 'pna' 'pna_diff' 'mlp' 'mlp_diff' 'chemprop' 'chemprop_diff')
n_jobs=5
seeds="1,2,3,4,5"
TOX21_ARGS="data.classification=True experiment.monitor_metric=val/pr_auc experiment.monitor_mode=max"
LAUNCHER="experiment.wandb_mode=disabled hydra/launcher=joblib hydra.launcher.n_jobs=$n_jobs -m"

for cfg_dir in "$ADME_DIR_PATH" "$TOX21_DIR_PATH" "$ZINC_DIR_PATH"; do
    if [[ ! -d "$cfg_dir" ]]; then
        echo "Config directory not found: $cfg_dir" >&2
        exit 1
    fi
done

run_dataset_model() {
    local cfg_dir="$1"
    local dataset_name="$2"
    local model="$3"
    local extra_args="${4:-}"
    local sweep_dir="$RESULTS_ROOT/$dataset_name/$model"
    echo "Running $dataset_name/$model"
    uv run python diffpred/run.py --config-path "$cfg_dir" --config-name "$model" experiment.seed=$seeds $extra_args hydra.sweep.dir=$sweep_dir $LAUNCHER
}

# Fail-fast preflight: run Graphormer variants first across all multitask datasets.
for model in "${graphormer_models[@]}"; do
    run_dataset_model "$ADME_DIR_PATH" "adme-fang" "$model"
    run_dataset_model "$TOX21_DIR_PATH" "tox21" "$model" "$TOX21_ARGS"
    run_dataset_model "$ZINC_DIR_PATH" "zinc" "$model"
done

for model in "${other_models[@]}"; do
    run_dataset_model "$ADME_DIR_PATH" "adme-fang" "$model"
    run_dataset_model "$TOX21_DIR_PATH" "tox21" "$model" "$TOX21_ARGS"
    run_dataset_model "$ZINC_DIR_PATH" "zinc" "$model"
done
