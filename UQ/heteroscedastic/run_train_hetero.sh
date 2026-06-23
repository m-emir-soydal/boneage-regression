#!/bin/bash
# UQ/heteroscedastic/run_train_hetero.sh
# Launch heteroscedastic UQ training (train_hetero module only).

# Default values
CONDA_ENV="rsna-boneage"
QUICK_TEST=0
OUTPUT_NAME="hetero"
SEED=42
EPOCHS=50
GPU_ID=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --quick-test) QUICK_TEST=1; shift ;;
        --output-name) OUTPUT_NAME="$2"; shift 2 ;;
        --env) CONDA_ENV="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# Initialize conda if needed, and activate environment
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

if [ $? -ne 0 ]; then
    echo "Failed to activate Conda environment '$CONDA_ENV'. Please verify it exists."
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &> /dev/null && pwd)"
cd "$PROJECT_ROOT"

PYTHON_CMD="python -m UQ.heteroscedastic.train_hetero --output-name $OUTPUT_NAME --seed $SEED --epochs $EPOCHS"
if [ "$QUICK_TEST" -eq 1 ]; then
    PYTHON_CMD="$PYTHON_CMD --quick-test"
fi

if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"
fi

echo "Running: $PYTHON_CMD"
eval "$PYTHON_CMD"
