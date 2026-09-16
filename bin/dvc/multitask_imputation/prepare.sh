#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

DATASETS="dockstring"
RATIOS="0.0,0.1,0.2,0.3,0.4,0.5"
SEEDS="1,2,3,4,5"
OUTPUT_ROOT="artifacts/data/multitask_imputation"

usage() {
  cat <<'USAGE'
Usage: bash bin/dvc/multitask_imputation/prepare.sh [options]

Options:
  --datasets <csv>      Dataset aliases. Default: dockstring
  --ratios <csv>        Mask ratios. Default: 0.0,0.1,0.2,0.3,0.4,0.5
  --seeds <csv>         Seeds. Default: 1,2,3,4,5
  --output-root <path>  Output root directory
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets)
      DATASETS="$2"
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
    --output-root)
      OUTPUT_ROOT="$2"
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

mkdir -p "$OUTPUT_ROOT"

IFS=',' read -r -a dataset_arr <<< "$DATASETS"
IFS=',' read -r -a ratio_arr <<< "$RATIOS"
IFS=',' read -r -a seed_arr <<< "$SEEDS"

for dataset in "${dataset_arr[@]}"; do
  for ratio in "${ratio_arr[@]}"; do
    for seed in "${seed_arr[@]}"; do
      echo "[prepare] dataset=$dataset ratio=$ratio seed=$seed"
      uv run python -m diffpred.data.multitask_imputation_prep \
        --dataset "$dataset" \
        --ratio "$ratio" \
        --seed "$seed" \
        --output-root "$OUTPUT_ROOT"
    done
  done
done

echo "Masking artifact generation complete: $OUTPUT_ROOT"
