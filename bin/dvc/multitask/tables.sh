#!/usr/bin/env bash
set -euo pipefail

# This script is for generating multitask result tables.
# It is called by DVC.

if [ "$#" -ne 4 ]; then
    echo "Illegal number of parameters. Usage: bash bin/dvc/multitask/tables.sh <reg_metrics> <reg_ranking_metric> <clf_metrics> <clf_ranking_metric>"
    exit 1
fi

REG_METRICS=$1
REG_RANKING_METRIC=$2
CLF_METRICS=$3
CLF_RANKING_METRIC=$4

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

# ADME-Fang
OUT_DIR_ADME="artifacts/figures/multitask/adme-fang"
mkdir -p "$OUT_DIR_ADME"
uv run python -m diffpred.utils.result_loader \
    --root-dir "artifacts/results_raw/multitask/adme-fang" \
    --output-dir "$OUT_DIR_ADME" \
    --metrics "$REG_METRICS" \
    --ranking-metric "$REG_RANKING_METRIC"

# Tox21
OUT_DIR_TOX21="artifacts/figures/multitask/tox21"
mkdir -p "$OUT_DIR_TOX21"
uv run python -m diffpred.utils.result_loader \
    --root-dir "artifacts/results_raw/multitask/tox21" \
    --output-dir "$OUT_DIR_TOX21" \
    --metrics "$CLF_METRICS" \
    --ranking-metric "$CLF_RANKING_METRIC"

# Zinc
OUT_DIR_ZINC="artifacts/figures/multitask/zinc"
mkdir -p "$OUT_DIR_ZINC"
uv run python -m diffpred.utils.result_loader \
    --root-dir "artifacts/results_raw/multitask/zinc" \
    --output-dir "$OUT_DIR_ZINC" \
    --metrics "$REG_METRICS" \
    --ranking-metric "$REG_RANKING_METRIC"

echo "Multitask tables generated."