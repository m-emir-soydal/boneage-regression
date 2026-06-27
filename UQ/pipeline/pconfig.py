"""Configuration for the fair BNN / MC-Dropout pipeline.

Mirrors the conformal-prediction config the project already uses so the two
families of methods are trained and evaluated under the same protocol:

  * per-seed stratified 50/25/25 split of ``train.csv`` (random_state = seed),
  * external ``test.csv`` as the held-out TEST set,
  * 50 training epochs in full mode,
  * post-hoc temperature (std-scale) calibration fitted on the calib split,
  * interval metrics reported at the same confidences (0.85 / 0.90 / 0.95).

Flip ``RUN_MODE`` to ``"debug"`` for a fast 1%-data / 2-epoch smoke test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# pipeline -> UQ -> project root
PIPELINE_DIR = Path(__file__).resolve().parent
UQ_DIR = PIPELINE_DIR.parent
PROJECT_ROOT = UQ_DIR.parent
for _p in (str(PROJECT_ROOT), str(UQ_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --------------------------------------------------------------------------- #
# Run mode
# --------------------------------------------------------------------------- #
RUN_MODE = "full"          # "full" or "debug"

# All pipeline outputs live here:  UQ/comparison/<Method>/seed_XX/outputs/ ...
OUTPUT_ROOT = UQ_DIR / "comparison"

# --------------------------------------------------------------------------- #
# Target space (months) — same bounds as the conformal config.
# --------------------------------------------------------------------------- #
Y_MIN = 0.0
Y_MAX = 240.0

# Confidence levels and the matching two-sided Gaussian z-scores. 90/95 match
# the project's UQ.common.inference.Z_SCORES; 85 added here.
CONFIDENCES = [0.85, 0.90, 0.95]
Z_SCORES = {0.85: 1.440, 0.90: 1.645, 0.95: 1.960}
PLOT_CONFIDENCE = 0.95

# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #
METHODS = ["bnn", "mc_dropout"]
METHOD_DISPLAY_NAMES = {"bnn": "BNN", "mc_dropout": "MC-Dropout"}

# --------------------------------------------------------------------------- #
# Seeds / data / epochs
# --------------------------------------------------------------------------- #
FULL_SEEDS = list(range(10))
DEBUG_SEEDS = [0, 1]

FULL_DATA_FRACTION = 1.0
DEBUG_DATA_FRACTION = 0.01

FULL_EPOCHS = 50
DEBUG_EPOCHS = 2

# Per-seed stratified split, identical recipe to the conformal pipeline.
PER_SEED_SPLITS = True
SPLIT_RANDOM_STATE = 42      # used only if PER_SEED_SPLITS is False
SPLIT_STRATIFY_COL = "male"
SPLIT_NOTE = (
    "per-seed stratified 50/25/25 split of train.csv (random_state=seed); "
    "external test.csv is TEST — identical partition to the conformal run"
)

# --------------------------------------------------------------------------- #
# UQ training / calibration knobs (same defaults as the existing UQ trainers).
# --------------------------------------------------------------------------- #
DROPOUT = 0.5
LR = 1e-4
BNN_PRIOR_SIGMA = 1.0
BNN_KL_WEIGHT = 1.0
SELECT_BY = "combo"          # UQ-aware checkpoint selection
MAE_WEIGHT = 0.01
MC_EVAL_SAMPLES = 10         # MC passes for per-epoch calib scoring
MC_EVAL_EVERY = 1
FINAL_SAMPLES = 30           # MC passes for the definitive calib fit + test eval

SAVE_MODELS = True           # keep model_best.pt per seed
SAVE_INTERVALS = True        # keep per-sample test intervals per seed

# Debug-only: cap the TEST set size (None = use the full 1425-case test set).
# Lets the smoke test finish quickly on CPU without touching the full-run path.
MAX_TEST = None

# --------------------------------------------------------------------------- #
# Resolved active settings
# --------------------------------------------------------------------------- #
if RUN_MODE == "debug":
    SEEDS = DEBUG_SEEDS
    DATA_FRACTION = DEBUG_DATA_FRACTION
    EPOCHS = DEBUG_EPOCHS
else:
    SEEDS = FULL_SEEDS
    DATA_FRACTION = FULL_DATA_FRACTION
    EPOCHS = FULL_EPOCHS


def split_random_state(seed: int) -> int:
    """sklearn random_state for this seed's partitions (mirrors conformal)."""
    return seed if PER_SEED_SPLITS else SPLIT_RANDOM_STATE


def active_config_dict() -> dict:
    return {
        "RUN_MODE": RUN_MODE,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "OUTPUT_ROOT": str(OUTPUT_ROOT),
        "Y_MIN": Y_MIN,
        "Y_MAX": Y_MAX,
        "CONFIDENCES": CONFIDENCES,
        "Z_SCORES": Z_SCORES,
        "METHODS": METHODS,
        "METHOD_DISPLAY_NAMES": METHOD_DISPLAY_NAMES,
        "SEEDS": SEEDS,
        "DATA_FRACTION": DATA_FRACTION,
        "EPOCHS": EPOCHS,
        "PER_SEED_SPLITS": PER_SEED_SPLITS,
        "SPLIT_RANDOM_STATE": SPLIT_RANDOM_STATE,
        "SPLIT_STRATIFY_COL": SPLIT_STRATIFY_COL,
        "SPLIT_NOTE": SPLIT_NOTE,
        "DROPOUT": DROPOUT,
        "LR": LR,
        "BNN_PRIOR_SIGMA": BNN_PRIOR_SIGMA,
        "BNN_KL_WEIGHT": BNN_KL_WEIGHT,
        "SELECT_BY": SELECT_BY,
        "MAE_WEIGHT": MAE_WEIGHT,
        "MC_EVAL_SAMPLES": MC_EVAL_SAMPLES,
        "MC_EVAL_EVERY": MC_EVAL_EVERY,
        "FINAL_SAMPLES": FINAL_SAMPLES,
        "SAVE_MODELS": SAVE_MODELS,
        "SAVE_INTERVALS": SAVE_INTERVALS,
    }


def save_active_config(path: Path | None = None) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = path or (OUTPUT_ROOT / "pipeline_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(active_config_dict(), f, indent=2)
    return path
