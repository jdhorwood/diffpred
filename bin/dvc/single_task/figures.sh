#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

EXPERIMENT_ID="single_task_full"
CHEMBL_METRICS="pearsonr"
CHEMBL_PLOT_TYPE="violin"
CHEMBL_VIZ_DIR_NAME="chembl_viz"
TDC_METRICS="rmse,pearsonr,r2_score"
TDC_PLOT_TYPE="violin"
TDC_VIZ_DIR_NAME="adme_viz"
TDC_DATASETS="lipophilicity_astrazeneca,solubility_aqsoldb,clearance_hepatocyte_az,clearance_microsome_az,vdss_lombardo,ld50_zhu"

usage() {
  cat <<'USAGE'
Usage: bash bin/dvc/single_task/figures.sh [options]

Options:
  --experiment-id <id>
  --chembl-metrics <csv>
  --chembl-plot-type <box|violin>
  --chembl-viz-dir-name <name>
  --tdc-metrics <csv>
  --tdc-plot-type <box|violin>
  --tdc-viz-dir-name <name>
  --tdc-datasets <csv>
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-id)
      EXPERIMENT_ID="$2"
      shift 2
      ;;
    --chembl-metrics)
      CHEMBL_METRICS="$2"
      shift 2
      ;;
    --chembl-plot-type)
      CHEMBL_PLOT_TYPE="$2"
      shift 2
      ;;
    --chembl-viz-dir-name)
      CHEMBL_VIZ_DIR_NAME="$2"
      shift 2
      ;;
    --tdc-metrics)
      TDC_METRICS="$2"
      shift 2
      ;;
    --tdc-plot-type)
      TDC_PLOT_TYPE="$2"
      shift 2
      ;;
    --tdc-viz-dir-name)
      TDC_VIZ_DIR_NAME="$2"
      shift 2
      ;;
    --tdc-datasets)
      TDC_DATASETS="$2"
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

if [[ -z "${SCRATCH_OUTPUT_ROOT:-}" ]]; then
  echo "SCRATCH_OUTPUT_ROOT must be set (for example via .envrc)." >&2
  exit 1
fi

RESULTS_ROOT="$SCRATCH_OUTPUT_ROOT/artifacts/results_raw/single_task/$EXPERIMENT_ID"
FIGURE_ROOT="$SCRATCH_OUTPUT_ROOT/artifacts/figures/single_task/$EXPERIMENT_ID"
mkdir -p "$FIGURE_ROOT"

if [[ ! -d "$RESULTS_ROOT" ]]; then
  echo "Missing raw results directory: $RESULTS_ROOT" >&2
  exit 1
fi

uv run python -m diffpred.viz.single_task_chembl_figures \
  --results-root "$RESULTS_ROOT" \
  --output-root "$FIGURE_ROOT" \
  --chembl-metrics "$CHEMBL_METRICS" \
  --chembl-plot-type "$CHEMBL_PLOT_TYPE" \
  --chembl-viz-dir-name "$CHEMBL_VIZ_DIR_NAME" \
  --tdc-metrics "$TDC_METRICS" \
  --tdc-plot-type "$TDC_PLOT_TYPE" \
  --tdc-viz-dir-name "$TDC_VIZ_DIR_NAME" \
  --tdc-datasets "$TDC_DATASETS"

echo "Figure generation complete: $FIGURE_ROOT"
