import numpy as np


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
