#!/bin/bash
# run_uq_pipeline.sh
# Train BOTH UQ methods (BNN + MC-Dropout) across all configured seeds, evaluate
# on the test set at every confidence, then generate the comparison tables and
# plots. One invocation does train -> eval -> report for both methods.
#
# Usage:
#   ./run_uq_pipeline.sh                  # full run (10 seeds, 50 epochs) on GPU 0
#   ./run_uq_pipeline.sh --gpu 1          # pick a GPU
#   ./run_uq_pipeline.sh --debug          # fast smoke test (2 seeds, 2 epochs, 1% data)
#   ./run_uq_pipeline.sh --report-only    # rebuild tables/plots from existing metrics
#   ./run_uq_pipeline.sh --seeds "0 1 2" --epochs 30
set -euo pipefail

CONDA_ENV="uq"
GPU_ID="0"
DEBUG=0
REPORT_ONLY=0
SEEDS=""
EPOCHS=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env)         CONDA_ENV="$2"; shift 2 ;;
        --gpu)         GPU_ID="$2"; shift 2 ;;
        --debug)       DEBUG=1; shift ;;
        --report-only) REPORT_ONLY=1; shift ;;
        --seeds)       SEEDS="$2"; shift 2 ;;
        --epochs)      EPOCHS="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Activate the conda environment that has torch / pandas / scipy / matplotlib.
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV" || { echo "Failed to activate conda env '$CONDA_ENV'."; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1   # stream per-epoch prints to the terminal in real time
echo "Using GPU: $CUDA_VISIBLE_DEVICES | conda env: $CONDA_ENV"

# Always runs both methods: the default METHODS in pconfig.py is [bnn, mc_dropout].
# 'python -u' = unbuffered so each epoch line shows immediately.
CMD=(python -u -m UQ.pipeline.run_all --methods bnn mc_dropout)
[[ "$DEBUG" -eq 1 ]]        && CMD+=(--debug)
[[ "$REPORT_ONLY" -eq 1 ]] && CMD+=(--report-only)
[[ -n "$SEEDS" ]]          && CMD+=(--seeds $SEEDS)
[[ -n "$EPOCHS" ]]         && CMD+=(--epochs "$EPOCHS")

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Done. Outputs under UQ/comparison/ (per-seed folders, tables/, figures/)."
