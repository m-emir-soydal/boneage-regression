#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper. Prefer:
#   ./cps.sh --backbone inceptionnext_base [--env rsna-boneage] [--gpu N] [--compare]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ENV="rsna-boneage"
GPU=""
COMPARE=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env) CONDA_ENV="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --compare) COMPARE=1; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

cps_args=(--backbone inceptionnext_base --env "$CONDA_ENV")
if [[ -n "$GPU" ]]; then
    cps_args+=(--gpu "$GPU")
fi
if [[ "$COMPARE" -eq 1 ]]; then
    cps_args+=(--compare)
fi

exec "$SCRIPT_DIR/cps.sh" "${cps_args[@]}"
