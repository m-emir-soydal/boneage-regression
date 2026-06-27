"""Interval + point metrics for the UQ pipeline.

The core metric functions are exactly those specified for the conformal
comparison so the BNN / MC-Dropout numbers are directly comparable to the
conformal ``metrics_long.csv`` columns.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Metric functions (verbatim spec)
# --------------------------------------------------------------------------- #
def picp(y_true, lower, upper):
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def mpiw(lower, upper):
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def pinaw(lower, upper, y_min, y_max):
    denom = y_max - y_min
    return float(mpiw(lower, upper) / denom) if denom > 0 else np.nan


def coverage_gap_pp(y_true, lower, upper, confidence):
    return float(100.0 * (confidence - picp(y_true, lower, upper)))


def miscoverage_rate(y_true, lower, upper):
    return float(1.0 - picp(y_true, lower, upper))


def interval_score(y_true, lower, upper, alpha):
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    width = upper - lower
    below = np.maximum(lower - y_true, 0.0)
    above = np.maximum(y_true - upper, 0.0)
    return float(np.mean(width + (2.0 / alpha) * below + (2.0 / alpha) * above))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    err = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(err ** 2)))


def median_absolute_error(y_true, y_pred):
    return float(np.median(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


# Column order kept consistent with the conformal metrics_long.csv so the two
# tables concatenate cleanly.
METRIC_COLUMNS = [
    "seed", "method", "confidence",
    "picp", "picp_percent", "mpiw", "pinaw",
    "coverage_gap_pp", "miscoverage_rate", "interval_score",
    "mae", "rmse", "medae",
    "mean_width_months", "median_width_months", "n_test",
]


def gaussian_bounds(pred_mean, std_eff, z):
    """``mean +/- z * std_eff`` (std already temperature-scaled)."""
    pred_mean = np.asarray(pred_mean)
    std_eff = np.asarray(std_eff)
    return pred_mean - z * std_eff, pred_mean + z * std_eff


def metrics_at_confidence(seed, method, confidence, z, y_true,
                          pred_mean, std_eff, y_min, y_max):
    """Return ``(row_dict, (lower, upper))`` for one Gaussian interval."""
    lower, upper = gaussian_bounds(pred_mean, std_eff, z)
    alpha = 1.0 - confidence
    p = picp(y_true, lower, upper)
    width = np.asarray(upper) - np.asarray(lower)
    row = {
        "seed": int(seed),
        "method": method,
        "confidence": round(float(confidence), 2),
        "picp": p,
        "picp_percent": 100.0 * p,
        "mpiw": mpiw(lower, upper),
        "pinaw": pinaw(lower, upper, y_min, y_max),
        "coverage_gap_pp": coverage_gap_pp(y_true, lower, upper, confidence),
        "miscoverage_rate": miscoverage_rate(y_true, lower, upper),
        "interval_score": interval_score(y_true, lower, upper, alpha),
        "mae": mae(y_true, pred_mean),
        "rmse": rmse(y_true, pred_mean),
        "medae": median_absolute_error(y_true, pred_mean),
        "mean_width_months": float(np.mean(width)),
        "median_width_months": float(np.median(width)),
        "n_test": int(len(np.asarray(y_true))),
    }
    return row, (lower, upper)
