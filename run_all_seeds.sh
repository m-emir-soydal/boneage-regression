#!/bin/bash
# run_all_seeds.sh
# Train + conformalize the CQR model over a range of seeds (default 0..19,
# matching FULL_SEEDS = range(20)). Each seed uses its own per-seed data split.

set -u

# Defaults
CONDA_ENV="uq"
OUTPUT_NAME="cqr"
START=0
END=19
EPOCHS=50
QUICK_TEST=0

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env) CONDA_ENV="$2"; shift 2 ;;
        --output-name) OUTPUT_NAME="$2"; shift 2 ;;
        --start) START="$2"; shift 2 ;;
        --end) END="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --quick-test) QUICK_TEST=1; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# Activate conda env once for the whole sweep
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
if [ $? -ne 0 ]; then
    echo "Failed to activate Conda environment '$CONDA_ENV'. Please verify it exists."
    exit 1
fi

echo "Sweeping seeds $START..$END | env=$CONDA_ENV | output-name=$OUTPUT_NAME | epochs=$EPOCHS | quick_test=$QUICK_TEST"

for SEED in $(seq "$START" "$END"); do
    echo ""
    echo "==================== SEED $SEED ===================="
    CMD="python train.py --output-name $OUTPUT_NAME --seed $SEED --epochs $EPOCHS"
    if [ "$QUICK_TEST" -eq 1 ]; then
        CMD="$CMD --quick-test"
    fi
    echo "Running: $CMD"
    eval "$CMD"
    if [ $? -ne 0 ]; then
        echo "Seed $SEED failed; continuing with next seed."
    fi
done

echo ""
echo "All seeds done."
