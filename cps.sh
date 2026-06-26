#!/usr/bin/env bash
set -euo pipefail

# cps.sh
# Run the full conformal-prediction pipeline for a single backbone.
#
# Supported backbones (see cps_config.BACKBONES):
#   efficientnet_b3   (default)
#   vit_b_16
#   inceptionnext_base
#
# Usage:
#   ./cps.sh
#   ./cps.sh --backbone vit_b_16
#   ./cps.sh --backbone inceptionnext_base --env rsna-boneage --gpu 1
#   ./cps.sh --backbone efficientnet_b3 --env rsna-boneage --gpu 1 --compare

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/gpu_env.sh
source "$SCRIPT_DIR/scripts/gpu_env.sh"

CONDA_ENV=""
# Default to the env var if already set, else let cps_config pick its default.
BACKBONE="${BACKBONE:-}"
GPU=""
COMPARE=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --backbone) BACKBONE="$2"; shift 2 ;;
        --env) CONDA_ENV="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --compare) COMPARE=1; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

if [[ -n "$CONDA_ENV" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

apply_gpu_selection "$GPU"

# Export so every `python -m ...` call below sees the same backbone.
if [[ -n "$BACKBONE" ]]; then
    export BACKBONE
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cps}"
mkdir -p "$MPLCONFIGDIR"

echo "Running conformal prediction pipeline"
python - <<'PY'
import os
import cps_config as C

print("BACKBONE:", C.BACKBONE)
print("RUN_MODE:", C.RUN_MODE)
print("RUN_TAG:", C.RUN_TAG)
print("SEEDS:", C.SEEDS)
print("DATA_FRACTION:", C.DATA_FRACTION)
print("EPOCHS:", C.EPOCHS)
print("OUTPUT:", C.RUN_OUTPUT_DIR)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<all>"))
PY

python -m cps.run_all
python -m conformal.evaluate
python -m conformal.tables
python -m conformal.plots
python -m conformal.report

if [[ "$COMPARE" -eq 1 ]]; then
    echo "==================================================================="
    echo "Building backbone comparison report"
    echo "==================================================================="
    python compare_backbones.py
fi

echo "Done."
