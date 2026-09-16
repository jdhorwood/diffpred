#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

DATASETS="biogen/adme-fang-reg-v1,graphium/tox21-v1,graphium/zinc12k-v1"
SEEDS="1,2,3,4,5"
STAMP_PATH="artifacts/data/multitask/polaris_benchmark_splits_prepared.txt"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --stamp-path)
            STAMP_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$(dirname "$STAMP_PATH")"

uv run python bin/dvc/multitask/prepare.py \
    --datasets "$DATASETS" \
    --seeds "$SEEDS" \
    --stamp-path "$STAMP_PATH"
