#!/usr/bin/env bash
set -euo pipefail

# Runs the full calibration/scale sweep, then builds tables and figures.
# This reuses existing predictions/embeddings from results/cps/full_2; it does
# not retrain any model.

PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-results/cps/full_2}"
OUTPUT_DIR="${OUTPUT_DIR:-sweep_test/results/full_2_sweep}"
SAVE_INTERVALS="${SAVE_INTERVALS:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cps}"

RUN_ARGS=(
  --source-run-dir "$SOURCE_RUN_DIR"
  --output-dir "$OUTPUT_DIR"
)

if [[ "$SAVE_INTERVALS" != "1" ]]; then
  RUN_ARGS+=(--no-save-intervals)
fi

echo "Running sweep_test experiment"
echo "SOURCE_RUN_DIR: $SOURCE_RUN_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "SAVE_INTERVALS: $SAVE_INTERVALS"
echo "PYTHON_BIN: $PYTHON_BIN"
echo ""

"$PYTHON_BIN" -m sweep_test.run "${RUN_ARGS[@]}"
"$PYTHON_BIN" -m sweep_test.tables --input-dir "$OUTPUT_DIR"
"$PYTHON_BIN" -m sweep_test.plots --input-dir "$OUTPUT_DIR"

echo ""
echo "Done. Results saved to: $OUTPUT_DIR"
echo "Tables:  $OUTPUT_DIR/tables"
echo "Figures: $OUTPUT_DIR/figures"
