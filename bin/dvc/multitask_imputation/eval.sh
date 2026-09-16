#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

RESULTS_ROOT="artifacts/results_raw/multitask_imputation"
OUTPUT_DIR="artifacts/figures/multitask_imputation"
USE_EMA_FLAG="--use-ema"
N_JOBS="1"
DISABLE_GUIDANCE="false"
DISABLE_GUIDANCE_FLAG=""

usage() {
  cat <<'USAGE'
Usage: bash bin/dvc/multitask_imputation/eval.sh [options]

Options:
  --results-root <path>   Root of multitask imputation raw runs
  --output-dir <path>     Output directory for CSVs and plots
  --n-jobs <int>          Number of runs to evaluate concurrently
  --no-ema                Disable EMA checkpoint preference
  --disable-guidance      Disable guidance at inference time for diffusion models
  --no-disable-guidance   Keep guidance enabled (default)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --n-jobs)
      N_JOBS="$2"
      shift 2
      ;;
    --no-ema)
      USE_EMA_FLAG="--no-use-ema"
      shift 1
      ;;
    --disable-guidance)
      DISABLE_GUIDANCE="true"
      shift 1
      ;;
    --no-disable-guidance)
      DISABLE_GUIDANCE="false"
      shift 1
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

mkdir -p "$OUTPUT_DIR"

if [[ "$DISABLE_GUIDANCE" == "true" ]]; then
  DISABLE_GUIDANCE_FLAG="--disable-guidance"
fi

uv run python -m diffpred.utils.multitask_imputation_eval \
  --results-root "$RESULTS_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --n-jobs "$N_JOBS" \
  "$USE_EMA_FLAG" \
  $DISABLE_GUIDANCE_FLAG

echo "Evaluation complete: $OUTPUT_DIR"
