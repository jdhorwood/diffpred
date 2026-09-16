#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

RESULTS_ROOT="artifacts/results_raw/calibration"
OUT_DIR="artifacts/figures/calibration/rejection_curves"
SEEDS="0,1,2,3,4"
METRIC="rmse"

mkdir -p "$OUT_DIR"

DATASETS=(
  "CHEMBL228_Ki"
  "caco2_wang"
  "solubility_aqsoldb"
  "ABL1"
  "CASP3"
)

for dataset in "${DATASETS[@]}"; do
  uv run python -m diffpred.uq.rejection_curves \
    --multirun-dir "$RESULTS_ROOT" \
    --dataset-name "$dataset" \
    --seeds "$SEEDS" \
    --metric "$METRIC" \
    --filtered-output-dir "$OUT_DIR"

  uv run python -m diffpred.uq.rejection_curves \
    --multirun-dir "$RESULTS_ROOT" \
    --dataset-name "$dataset" \
    --seeds "$SEEDS" \
    --metric "$METRIC" \
    --filtered-output-dir "$OUT_DIR" \
    --guidance-ablation
done

printf 'Calibration rejection curves complete: %s\n' "$OUT_DIR"
