#!/bin/bash
# run_training.sh
# Script to easily launch the training process.

# Default values
CONDA_ENV="uq"
QUICK_TEST=0
OUTPUT_NAME="run"
SEED=123

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --quick-test) QUICK_TEST=1; shift ;;
        --output-name) OUTPUT_NAME="$2"; shift 2 ;;
        --env) CONDA_ENV="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
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

# Build python command
PYTHON_CMD="python train.py --output-name $OUTPUT_NAME --seed $SEED"
if [ "$QUICK_TEST" -eq 1 ]; then
    PYTHON_CMD="$PYTHON_CMD --quick-test"
fi

echo "Running: $PYTHON_CMD"
eval "$PYTHON_CMD"
