#!/bin/bash
# UQ/heteroscedastic/run_hetero.sh
# Convenience wrapper around the heteroscedastic train + evaluate workflow.
# Activates the project conda environment then trains, fits calibration on
# the calib split, and evaluates the test split with that calibration.

set -e

CONDA_ENV="rsna-boneage"
OUTPUT_NAME="hetero"
SEED=42
EPOCHS=50
QUICK_TEST=0
TRAIN_ONLY=0
SKIP_TRAIN=0

usage() {
    cat <<USAGE
Usage: $(basename "$0") [--env NAME] [--output-name NAME] [--seed N]
                              [--epochs N] [--quick-test]
                              [--train-only] [--skip-train]

  --env NAME          Conda env to activate (default: rsna-boneage)
  --output-name NAME  Run name prefix (default: hetero)
  --seed N            Torch seed (default: 42)
  --epochs N          Training epochs (default: 50)
  --quick-test        1%% data, 2 epochs (overrides --epochs)
  --train-only        Stop after training; skip calib/test eval
  --skip-train        Reuse an existing outputs/<run_name>/ checkpoint and
                      just run calib + test eval
USAGE
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env) CONDA_ENV="$2"; shift 2 ;;
        --output-name) OUTPUT_NAME="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --quick-test) QUICK_TEST=1; shift ;;
        --train-only) TRAIN_ONLY=1; shift ;;
        --skip-train) SKIP_TRAIN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown parameter: $1" >&2; usage; exit 1 ;;
    esac
done

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
if [ $? -ne 0 ]; then
    echo "Failed to activate Conda environment '$CONDA_ENV'." >&2
    exit 1
fi

RUN_NAME="${OUTPUT_NAME}_seed${SEED}"
if [ "$QUICK_TEST" -eq 1 ]; then
    RUN_NAME="${RUN_NAME}_quicktest"
fi

# Project root is two levels up from UQ/heteroscedastic/.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &> /dev/null && pwd)"
cd "$PROJECT_ROOT"

if [ "$SKIP_TRAIN" -eq 0 ]; then
    TRAIN_CMD="python -m UQ.heteroscedastic.train_hetero --output-name $OUTPUT_NAME --seed $SEED --epochs $EPOCHS"
    if [ "$QUICK_TEST" -eq 1 ]; then
        TRAIN_CMD="$TRAIN_CMD --quick-test"
    fi
    echo "Running: $TRAIN_CMD"
    eval "$TRAIN_CMD"
fi

if [ "$TRAIN_ONLY" -eq 1 ]; then
    exit 0
fi

CKPT_DIR="${OUTPUT_DIR:-./outputs}/$RUN_NAME"
# Pick the most recently modified best_* checkpoint in the run directory.
CKPT=$(ls -t "$CKPT_DIR"/best_*.pth 2>/dev/null | head -n 1 || true)
if [ -z "$CKPT" ]; then
    echo "No best_*.pth checkpoint found under $CKPT_DIR; aborting eval." >&2
    exit 1
fi
echo "Using checkpoint: $CKPT"

echo "Fitting calibration on calib split..."
python -m UQ.heteroscedastic.run_hetero \
    --checkpoint "$CKPT" --split calib --fit-calibration

echo "Evaluating test split with auto calibration..."
python -m UQ.heteroscedastic.run_hetero \
    --checkpoint "$CKPT" --split test --calibration auto

echo "Done. Compare against other UQ methods:"
echo "  python -m UQ.compare --split test"
