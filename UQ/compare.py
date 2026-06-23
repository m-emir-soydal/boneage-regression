"""Aggregate every UQ method's metrics into one comparison table.

Scans ``UQ/results/<method>/*_metrics_*.csv`` (the shared schema emitted by
``common.uq_metrics.summarize``), keeps the most recent row per
``(method, split)``, and prints/saves a ranked comparison so different
uncertainty quantification approaches can be judged on the same metrics
(PICP@90/95, MPIW@90/95) and point accuracy.

Example:
    python -m UQ.compare                 # compare the latest run of every method
    python -m UQ.compare --split test    # restrict to one split
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_THIS = Path(__file__).resolve()
_UQ_DIR = _THIS.parent
if str(_UQ_DIR) not in sys.path:
    sys.path.insert(0, str(_UQ_DIR))

from common.paths import RESULTS_DIR  # noqa: E402

# Columns shared by every method (method-specific extras are dropped here so
# the comparison stays uniform).
SHARED_COLS = [
    "method", "split", "MAE", "RMSE", "R2",
    "pct_within_6mo", "pct_within_12mo",
    "PICP_90", "PICP_95", "MPIW_90", "MPIW_95", "n", "timestamp",
]

NOMINAL = {"PICP_90": 0.90, "PICP_95": 0.95}


def collect_metrics():
    """Read every method's metrics CSVs into one DataFrame."""
    paths = sorted(RESULTS_DIR.glob("*/*_metrics_*.csv"))
    if not paths:
        return pd.DataFrame()
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception as exc:  # noqa: BLE001
            print(f"  warning: could not read {p}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def latest_per_method_split(df):
    """Keep the most recent row (by timestamp) for each (method, split)."""
    df = df.sort_values("timestamp")
    return df.groupby(["method", "split"], as_index=False).tail(1)


def main():
    parser = argparse.ArgumentParser(description="Compare UQ methods on shared metrics")
    parser.add_argument("--split", type=str, default=None,
                        help="Restrict the comparison to a single split (e.g. test)")
    parser.add_argument("--output-name", type=str, default="comparison")
    args = parser.parse_args()

    df = collect_metrics()
    if df.empty:
        print(f"No metrics CSVs found under {RESULTS_DIR}/*/. Run a UQ method first.")
        return

    df = latest_per_method_split(df)
    if args.split:
        df = df[df["split"] == args.split]
        if df.empty:
            print(f"No results for split '{args.split}'.")
            return

    for col in SHARED_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    table = df[SHARED_COLS].copy()

    # Calibration error: how far each PICP lands from its nominal level.
    table["cal_err_90"] = (table["PICP_90"] - NOMINAL["PICP_90"]).abs()
    table["cal_err_95"] = (table["PICP_95"] - NOMINAL["PICP_95"]).abs()

    # Rank: best calibrated first, then tightest intervals.
    table = table.sort_values(
        ["split", "cal_err_90", "MPIW_90"]
    ).reset_index(drop=True)

    show_cols = [
        "method", "split", "MAE", "RMSE", "R2",
        "PICP_90", "PICP_95", "MPIW_90", "MPIW_95",
        "cal_err_90", "cal_err_95", "n",
    ]
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}",
                           "display.max_columns", None, "display.width", 200):
        print("\n=== UQ method comparison (latest run per method/split) ===")
        print(table[show_cols].to_string(index=False))
        print()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{args.output_name}_{ts}.csv"
    table.to_csv(out_path, index=False)
    print(f"Saved comparison table to: {out_path}")


if __name__ == "__main__":
    main()
