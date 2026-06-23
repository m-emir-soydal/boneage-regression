"""Post-hoc uncertainty calibration helpers shared by UQ methods.

The MC Dropout intervals are built as ``mean +/- z * std`` assuming the
per-sample standard deviation is a well-scaled Gaussian sigma. In practice the
raw dropout ``std`` is usually *too small*, so the intervals under-cover (low
PICP). The standard fix is **temperature scaling**: learn a single scalar
``s`` on a held-out calibration split and report ``mean +/- z * (s * std)``.

``s`` is the maximum-likelihood scale of a zero-mean Gaussian fit to the
residuals, i.e. the root-mean-square of the standardized z-scores::

    z_i = (y_i - mean_i) / std_i
    s   = sqrt( mean( z_i^2 ) )

If the raw std already matched the residual spread, ``s`` would be ~1.0.
``s > 1`` widens the intervals (the common case here), ``s < 1`` tightens them.
A single scalar keeps the 90% and 95% intervals consistent with one another.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np

CALIBRATION_GLOB = "calibration_*.json"


def fit_temperature(y_true, mean, std, eps=1e-6):
    """Fit the scalar std multiplier that calibrates Gaussian intervals.

    Returns ``s = sqrt(mean(((y_true - mean) / std) ** 2))`` computed in
    de-normalized month space. ``std`` is floored by ``eps`` to avoid division
    by zero for samples with (near) zero dropout variance.
    """
    y_true = np.asarray(y_true, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    std_safe = np.clip(std, eps, None)
    z = (y_true - mean) / std_safe
    scale = float(np.sqrt(np.mean(z ** 2)))
    return scale


def apply_temperature(std, scale):
    """Scale a std array (or scalar) by the calibration temperature."""
    return np.asarray(std, dtype=float) * float(scale)


def save_calibration(out_dir, std_scale, split, n, extra=None, prefix="calibration"):
    """Write a calibration JSON to ``out_dir`` and return its path.

    The payload is intentionally small and self-describing so it can be
    reloaded later (e.g. to calibrate the test split) and audited.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "std_scale": float(std_scale),
        "split": split,
        "n": int(n),
        "timestamp": ts,
    }
    if extra:
        payload.update(extra)
    path = out_dir / f"{prefix}_{ts}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_calibration(path):
    """Load a calibration JSON payload from ``path``."""
    with open(path) as f:
        return json.load(f)


def latest_calibration(search_dir):
    """Return the newest ``calibration_*.json`` in ``search_dir`` or ``None``."""
    search_dir = Path(search_dir)
    matches = sorted(search_dir.glob(CALIBRATION_GLOB))
    return matches[-1] if matches else None


def resolve_calibration(spec, search_dir):
    """Resolve a ``--calibration`` argument to a payload.

    ``spec`` is either the literal string ``"auto"`` (pick the most recent
    calibration JSON under ``search_dir``) or a path to a specific JSON file.
    Returns ``(payload, path)``; raises ``FileNotFoundError`` if nothing is
    found.
    """
    if spec == "auto":
        path = latest_calibration(search_dir)
        if path is None:
            raise FileNotFoundError(
                f"No {CALIBRATION_GLOB} found in {search_dir}. "
                "Fit one first with --fit-calibration on the calib split."
            )
    else:
        path = Path(spec)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
    return load_calibration(path), path
