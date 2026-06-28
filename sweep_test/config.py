from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Existing trained EfficientNet CP run. This experiment only reads artifacts
# from here; it does not retrain models.
SOURCE_RUN_DIR = PROJECT_ROOT / "results" / "cps" / "full_2"

# Keep the new experiment self-contained.
OUTPUT_DIR = PROJECT_ROOT / "sweep_test" / "results" / "full_2"

SEEDS = list(range(20))

CONFIDENCES = [0.40, 0.50, 0.60, 0.70, 0.85, 0.90, 0.95]
TABLE_CONFIDENCES = [0.85, 0.90, 0.95]
PLOT_CONFIDENCE = 0.95
PLOT_CONFIDENCES = [0.85, 0.95]

# Fractions are relative to the original source train.csv pool.
# 0.075 means 7.5% of the original training source rows.
CAL_FRACTIONS = [0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25]

# The available CP pool in full_2 is scale + cal = 12.5% + 12.5%.
# KNN-NCP uses cal_fraction for conformal scores and the remaining part of this
# fixed budget for scale/difficulty estimation.
AVAILABLE_CP_FRACTION = 0.25
KNN_TOTAL_FRACTION = 0.25

SAVE_INTERVALS = True
