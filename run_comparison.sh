#!/usr/bin/env bash
set -euo pipefail

# run_comparison.sh
# Run the full conformal-prediction pipeline once per backbone, then build the
# side-by-side PICP/PINAW comparison report.
#
# Usage:
#   ./run_comparison.sh
#   ./run_comparison.sh --env rsna-boneage
#   ./run_comparison.sh --gpu 1
#   ./run_comparison.sh --backbones "efficientnet_b3 vit_b_16" --gpu 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/gpu_env.sh
source "$SCRIPT_DIR/scripts/gpu_env.sh"

CONDA_ENV=""
BACKBONES=""
GPU=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env) CONDA_ENV="$2"; shift 2 ;;
        --backbones) BACKBONES="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

if [[ -n "$CONDA_ENV" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

apply_gpu_selection "$GPU"

# Default to the backbones declared in cps_config.COMPARE_BACKBONES.
if [[ -z "$BACKBONES" ]]; then
    BACKBONES="$(python -c 'import cps_config as C; print(" ".join(C.COMPARE_BACKBONES))')"
fi

for bb in $BACKBONES; do
    echo "==================================================================="
    echo "Backbone: $bb"
    echo "==================================================================="
    cps_args=(--backbone "$bb")
    if [[ -n "$GPU" ]]; then
        cps_args+=(--gpu "$GPU")
    fi
    "$SCRIPT_DIR/cps.sh" "${cps_args[@]}"
done

echo "==================================================================="
echo "Building backbone comparison report"
echo "==================================================================="
python compare_backbones.py

echo "Comparison complete."
