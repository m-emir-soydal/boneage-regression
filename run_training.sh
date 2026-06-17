#!/bin/bash
# run_training.sh
# Script to easily launch the training process.
# Trains (and tests) once per seed to check run-to-run stability.

# Default values
CONDA_ENV="uq"
QUICK_TEST=0
OUTPUT_NAME="run"
#SEEDS="42 123 7"
SEEDS="42"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --quick-test) QUICK_TEST=1; shift ;;
        --output-name) OUTPUT_NAME="$2"; shift 2 ;;
        --env) CONDA_ENV="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
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

# Suffix train.py appends to run_name (must match train.py)
SUFFIX=""
if [ "$QUICK_TEST" -eq 1 ]; then
    SUFFIX="_quicktest"
fi

for SEED in $SEEDS; do
    RUN_NAME="${OUTPUT_NAME}_seed${SEED}${SUFFIX}"
    echo "=============================================="
    echo "Seed $SEED -> run: $RUN_NAME"
    echo "=============================================="

    TRAIN_CMD="python train.py --output-name $OUTPUT_NAME --seed $SEED"
    if [ "$QUICK_TEST" -eq 1 ]; then
        TRAIN_CMD="$TRAIN_CMD --quick-test"
    fi
    echo "Running: $TRAIN_CMD"
    eval "$TRAIN_CMD"

    # Pick the most recent checkpoint from this run's folder and evaluate on TEST set
    CKPT="$(ls -t outputs/${RUN_NAME}/best_*.pth 2>/dev/null | head -1)"
    if [ -z "$CKPT" ]; then
        echo "No checkpoint found in outputs/${RUN_NAME}/, skipping test."
        continue
    fi
    echo "Testing checkpoint: $CKPT"
    python test.py --checkpoint "$CKPT" --output-name "$RUN_NAME"
done
