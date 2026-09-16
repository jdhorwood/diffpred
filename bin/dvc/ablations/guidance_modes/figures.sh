#!/usr/bin/env bash
set -euo pipefail

# Find project root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

# Arguments from DVC/CLI
EXPERIMENT_ID="${1}"
METRICS="${2:-pearsonr,rmse}"

FIGURES_ROOT="artifacts/figures/ablations/${EXPERIMENT_ID}"
RESULTS_ROOT="artifacts/results_raw/ablations/${EXPERIMENT_ID}"

mkdir -p "$FIGURES_ROOT"

uv run python -m diffpred.viz.guidance_ablation_viz \
    --results-dir "$RESULTS_ROOT" \
    --output-dir "$FIGURES_ROOT" \
    --metrics "$METRICS"

printf 'Guidance modes ablation figures complete: %s
' "$FIGURES_ROOT"
