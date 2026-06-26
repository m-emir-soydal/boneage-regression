# Shared GPU selection for project shell scripts.
#
# Usage (from another script):
#   source "$(dirname "${BASH_SOURCE[0]}")/scripts/gpu_env.sh"
#   apply_gpu_selection "$GPU"
#
# --gpu accepts a single index (e.g. 1) or a comma-separated list (e.g. 0,1).
# The selected physical GPU(s) are exposed via CUDA_VISIBLE_DEVICES before any
# Python process starts, so existing code can keep using cuda / cuda:0.

apply_gpu_selection() {
    local gpu="${1:-}"

    if [[ -n "$gpu" ]]; then
        export CUDA_VISIBLE_DEVICES="$gpu"
        echo "Using GPU(s): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "Using GPU(s): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (from environment)"
    else
        echo "Using GPU(s): default (all visible devices)"
    fi
}
