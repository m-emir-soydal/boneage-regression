import json
import os
from pathlib import Path


# Change this to "full" for the final experiment.
RUN_MODE = "full"

# Optional output folder name under results/cps/.
# Leave as None to use the default folder name: "debug" or "full".
RUN_TAG = "full_2"

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("DATA_DIR", "./data"))
if not DATA_ROOT.is_absolute():
    DATA_ROOT = (PROJECT_ROOT / DATA_ROOT).resolve()
OUTPUT_ROOT = PROJECT_ROOT / "results" / "cps"

# Image feature backbone used for both the point predictor and the embeddings
# consumed by the conformal-prediction pipeline. The fusion head is identical
# across backbones, so the 256-dim embedding stays comparable.
# Override at runtime with the BACKBONE env var (used by cps.sh / run_comparison.sh)
# so a single config file can drive every backbone without manual edits.
BACKBONES = {
    "efficientnet_b3": (300, 300),
    "vit_b_16": (224, 224),
    "inceptionnext_base": (224, 224),
}
BACKBONE = os.getenv("BACKBONE", "efficientnet_b3")
if BACKBONE not in BACKBONES:
    raise ValueError(f"Unknown BACKBONE '{BACKBONE}'. Known: {list(BACKBONES)}")
# Backbones compared side by side by compare_backbones.py.
COMPARE_BACKBONES = ["efficientnet_b3", "vit_b_16", "inceptionnext_base"]

Y_MIN = 0.0
Y_MAX = 240.0

CONFIDENCES = [0.40, 0.50, 0.60, 0.70, 0.85, 0.90, 0.95]
TABLE_CONFIDENCES = [0.85, 0.90, 0.95]
PLOT_CONFIDENCE = 0.95

METHODS = ["scp", "knn_ncp", "as_mcp", "bcp"]
METHOD_DISPLAY_NAMES = {
    "scp": "SCP",
    "knn_ncp": "KNN-NCP",
    "as_mcp": "AS-MCP",
    "bcp": "BCP",
}

FULL_SEEDS = list(range(20))
# FULL_SEEDS = [0, 1, 2]
DEBUG_SEEDS = [0, 1]

FULL_DATA_FRACTION = 1.0
DEBUG_DATA_FRACTION = 0.01

# Keep the existing project default for full training.
FULL_EPOCHS = 50
# FULL_EPOCHS = 10
DEBUG_EPOCHS = 2

# Debug defaults to CPU because it is meant to catch pipeline errors even when
# the shared GPU is occupied. Full mode uses CUDA automatically when available.
DEBUG_DEVICE = "cpu"
FULL_DEVICE = "auto"

# The CP pipeline reuses the existing project split from data_loader.load_data:
# train/val/calibration/test. The calibration part is divided internally into
# scale/cal for KNN-NCP and conformal calibration.
SPLIT_RATIOS = {
    "train": 0.50,
    "val": 0.25,
    "scale": 0.125,
    "cal": 0.125,
}
# When True, each seed gets its own stratified train/val/cal/scale split
# (random_state = seed). The official source validation set remains TEST.
# When False, all seeds share the fixed split from SPLIT_RANDOM_STATE (full_1).
PER_SEED_SPLITS = True
SPLIT_NOTE = (
    "per-seed stratified 50/25/12.5/12.5 split from train.csv; "
    "official source validation set is TEST"
    if PER_SEED_SPLITS
    else "fixed split for all seeds; test uses the existing source validation set"
)
SPLIT_RANDOM_STATE = 42

KNN_K = 50
KNN_METRIC = "cosine"
KNN_EPS = 1e-6
KNN_STANDARDIZE = True

N_AGE_QUANTILE_BINS = 5
AGE_QUANTILES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
MIN_MONDRIAN_BIN_SIZE = 30

# Backward conformal prediction controls interval size first and estimates the
# resulting coverage. This default is a clinically interpretable ±12 month band.
BCP_INTERVAL_WIDTH_MONTHS = 24.0

USE_CREPES = False

SAVE_INTERVALS = True
SAVE_EMBEDDINGS = True

if RUN_MODE == "debug":
    SEEDS = DEBUG_SEEDS
    DATA_FRACTION = DEBUG_DATA_FRACTION
    EPOCHS = DEBUG_EPOCHS
    RUN_TAG_DIR = OUTPUT_ROOT / (RUN_TAG or "debug")
    DEVICE = DEBUG_DEVICE
else:
    SEEDS = FULL_SEEDS
    DATA_FRACTION = FULL_DATA_FRACTION
    EPOCHS = FULL_EPOCHS
    RUN_TAG_DIR = OUTPUT_ROOT / (RUN_TAG or "full")
    DEVICE = FULL_DEVICE

# Outputs are scoped per backbone so EfficientNet and ViT runs do not collide.
RUN_OUTPUT_DIR = RUN_TAG_DIR / BACKBONE


def backbone_img_size(backbone=None):
    """Return the (H, W) input size expected by the given backbone."""
    name = BACKBONE if backbone is None else backbone
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Known: {list(BACKBONES)}")
    return BACKBONES[name]


def comparison_dir():
    """Directory for cross-backbone comparison artifacts (shared run tag)."""
    return RUN_TAG_DIR / "comparison"


def active_config_dict():
    return {
        "RUN_MODE": RUN_MODE,
        "RUN_TAG": RUN_TAG,
        "BACKBONE": BACKBONE,
        "BACKBONES": BACKBONES,
        "COMPARE_BACKBONES": COMPARE_BACKBONES,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "DATA_ROOT": str(DATA_ROOT),
        "OUTPUT_ROOT": str(OUTPUT_ROOT),
        "RUN_OUTPUT_DIR": str(RUN_OUTPUT_DIR),
        "Y_MIN": Y_MIN,
        "Y_MAX": Y_MAX,
        "CONFIDENCES": CONFIDENCES,
        "TABLE_CONFIDENCES": TABLE_CONFIDENCES,
        "PLOT_CONFIDENCE": selected_plot_confidence(),
        "METHODS": METHODS,
        "FULL_SEEDS": FULL_SEEDS,
        "DEBUG_SEEDS": DEBUG_SEEDS,
        "SEEDS": SEEDS,
        "FULL_DATA_FRACTION": FULL_DATA_FRACTION,
        "DEBUG_DATA_FRACTION": DEBUG_DATA_FRACTION,
        "DATA_FRACTION": DATA_FRACTION,
        "FULL_EPOCHS": FULL_EPOCHS,
        "DEBUG_EPOCHS": DEBUG_EPOCHS,
        "EPOCHS": EPOCHS,
        "DEBUG_DEVICE": DEBUG_DEVICE,
        "FULL_DEVICE": FULL_DEVICE,
        "DEVICE": DEVICE,
        "SPLIT_RATIOS": SPLIT_RATIOS,
        "SPLIT_NOTE": SPLIT_NOTE,
        "PER_SEED_SPLITS": PER_SEED_SPLITS,
        "SPLIT_RANDOM_STATE": SPLIT_RANDOM_STATE,
        "KNN_K": KNN_K,
        "KNN_METRIC": KNN_METRIC,
        "KNN_EPS": KNN_EPS,
        "KNN_STANDARDIZE": KNN_STANDARDIZE,
        "N_AGE_QUANTILE_BINS": N_AGE_QUANTILE_BINS,
        "AGE_QUANTILES": AGE_QUANTILES,
        "MIN_MONDRIAN_BIN_SIZE": MIN_MONDRIAN_BIN_SIZE,
        "BCP_INTERVAL_WIDTH_MONTHS": BCP_INTERVAL_WIDTH_MONTHS,
        "USE_CREPES": USE_CREPES,
        "SAVE_INTERVALS": SAVE_INTERVALS,
        "SAVE_EMBEDDINGS": SAVE_EMBEDDINGS,
    }


def save_active_config():
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = RUN_OUTPUT_DIR / "config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(active_config_dict(), f, indent=2)
    return path


def selected_plot_confidence():
    if any(abs(conf - PLOT_CONFIDENCE) < 1e-9 for conf in CONFIDENCES):
        return PLOT_CONFIDENCE
    return max(CONFIDENCES)
