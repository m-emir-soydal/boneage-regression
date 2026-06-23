"""Shared UQ + point metrics used by every UQ method.

All functions operate in de-normalized month space so results are directly
comparable across methods and against the base point-regression model
(see ``metrics.py`` in the project root).
"""
from datetime import datetime

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def picp(y_true, lower, upper):
    """Prediction Interval Coverage Probability.

    Fraction of true values that fall within ``[lower, upper]`` (in [0, 1]).
    A well-calibrated 90% interval should yield ~0.90.
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def mpiw(lower, upper):
    """Mean Prediction Interval Width (in months). Lower is better, given coverage."""
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    return float(np.mean(upper - lower))


def point_metrics(y_true, y_pred_mean):
    """Standard regression metrics on the mean prediction.

    Mirrors the metrics reported in ``metrics.py`` so UQ runs can be compared
    against the base model on accuracy as well as on uncertainty.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred_mean)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    acc6 = float(np.mean(np.abs(y_pred - y_true) <= 6.0) * 100)
    acc12 = float(np.mean(np.abs(y_pred - y_true) <= 12.0) * 100)

    return {
        "MAE": float(mae),
        "RMSE": rmse,
        "MSE": float(mse),
        "R2": float(r2),
        "pct_within_6mo": acc6,
        "pct_within_12mo": acc12,
    }


def summarize(method_name, y_true, y_pred_mean,
              lower90, upper90, lower95, upper95,
              split="test", extra=None):
    """Assemble a single comparison row shared by all UQ methods.

    The fixed schema (method, point metrics, PICP@90/95, MPIW@90/95, n,
    timestamp) lets a future ``UQ/compare.py`` concatenate every method's
    metrics CSV into one comparison table.
    """
    row = {
        "method": method_name,
        "split": split,
    }
    row.update(point_metrics(y_true, y_pred_mean))
    row["PICP_90"] = picp(y_true, lower90, upper90)
    row["PICP_95"] = picp(y_true, lower95, upper95)
    row["MPIW_90"] = mpiw(lower90, upper90)
    row["MPIW_95"] = mpiw(lower95, upper95)
    row["n"] = int(len(np.asarray(y_true)))
    if extra:
        row.update(extra)
    row["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    return row
