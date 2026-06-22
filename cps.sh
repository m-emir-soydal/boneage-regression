#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cps}"
mkdir -p "$MPLCONFIGDIR"

echo "Running conformal prediction pipeline"
python - <<'PY'
import cps_config as C

print("RUN_MODE:", C.RUN_MODE)
print("RUN_TAG:", C.RUN_TAG)
print("SEEDS:", C.SEEDS)
print("DATA_FRACTION:", C.DATA_FRACTION)
print("EPOCHS:", C.EPOCHS)
print("OUTPUT:", C.RUN_OUTPUT_DIR)
PY

python -m cps.run_all
python -m conformal.evaluate
python -m conformal.tables
python -m conformal.plots
python -m conformal.report

echo "Done."
