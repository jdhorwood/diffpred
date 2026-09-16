#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

RESULTS_ROOT="artifacts/results_raw/calibration"
OUT_DIR="artifacts/figures/calibration/tables"

mkdir -p "$OUT_DIR"

uv run python -m diffpred.utils.calibration_result_loader \
  "$RESULTS_ROOT" \
  --output-dir "$OUT_DIR" \
  --compute-guidance-ablation

printf 'Calibration summary tables complete: %s\n' "$OUT_DIR"
