#!/bin/bash
# run_training.sh
# Script to easily launch the training process.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/gpu_env.sh
source "$SCRIPT_DIR/scripts/gpu_env.sh"

# Default values
CONDA_ENV="rsna-boneage"
QUICK_TEST=0
OUTPUT_NAME="run"
SEED=123
BACKBONE="efficientnet_b3"
GPU=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --quick-test) QUICK_TEST=1; shift ;;
        --output-name) OUTPUT_NAME="$2"; shift 2 ;;
        --env) CONDA_ENV="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --backbone) BACKBONE="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# Initialize conda if needed, and activate environment
# This allows using "conda activate" inside the shell script
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Check if environment activation was successful
if [ $? -ne 0 ]; then
    echo "Failed to activate Conda environment '$CONDA_ENV'. Please verify it exists."
    exit 1
fi

apply_gpu_selection "$GPU"

# Build python command
PYTHON_CMD="python train.py --output-name $OUTPUT_NAME --seed $SEED --backbone $BACKBONE"
if [ "$QUICK_TEST" -eq 1 ]; then
    PYTHON_CMD="$PYTHON_CMD --quick-test"
fi

echo "Running: $PYTHON_CMD"
eval "$PYTHON_CMD"
