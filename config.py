import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training configurations
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
RANDOM_STATE = 42

# Ensure we deal with absolute paths if they are relative
if not DATA_DIR.is_absolute():
    DATA_DIR = (BASE_DIR / DATA_DIR).resolve()
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = (BASE_DIR / OUTPUT_DIR).resolve()

# ---------------------------------------------------------------------------
# Conformalized Quantile Regression (CQR) settings
# ---------------------------------------------------------------------------
# Target bone-age range in months, used to clip de-normalized intervals.
Y_MIN = 0.0
Y_MAX = 240.0

# Confidence levels the quantile head is trained/calibrated for.
CONFIDENCES = [0.85, 0.90, 0.95]

# Per-seed stratified split of the source train.csv into
# train / val / scale / cal. The source val.csv is the held-out TEST set.
# `scale` is unused by CQR but kept so the split matches the cps pipeline.
SPLIT_RATIOS = {
    "train": 0.50,
    "val": 0.25,
    "scale": 0.125,
    "cal": 0.125,
}
# When True each seed gets its own stratified split (random_state = seed).
PER_SEED_SPLITS = True


def _build_quantiles(confidences):
    """Map each confidence level to its (lower, upper) quantile pair and
    return the sorted unique list of quantiles the head must predict
    (always including the 0.5 median)."""
    pairs = {}
    qs = {0.5}
    for c in confidences:
        alpha = 1.0 - c
        lo = round(alpha / 2.0, 6)
        hi = round(1.0 - alpha / 2.0, 6)
        pairs[c] = (lo, hi)
        qs.add(lo)
        qs.add(hi)
    return pairs, sorted(qs)


# QUANTILE_PAIRS: {confidence: (lower_q, upper_q)}
# QUANTILES: sorted unique quantile levels the model outputs.
QUANTILE_PAIRS, QUANTILES = _build_quantiles(CONFIDENCES)
N_QUANTILES = len(QUANTILES)
MEDIAN_IDX = QUANTILES.index(0.5)


def quantile_index(q):
    """Index of quantile level `q` inside the model's output vector."""
    for i, qq in enumerate(QUANTILES):
        if abs(qq - q) < 1e-9:
            return i
    raise ValueError(f"Quantile {q} not in QUANTILES={QUANTILES}")
