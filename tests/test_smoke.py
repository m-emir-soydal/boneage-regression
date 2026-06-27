import sys
from pathlib import Path

import numpy as np
import torch

# Ensure the parent directory is in sys.path so we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import N_QUANTILES, QUANTILES, CONFIDENCES, QUANTILE_PAIRS
from model import build_multi_input_model
from cqr_utils import pinball_loss, calibrate, build_intervals


def test_model_outputs_quantiles():
    """The model should emit one value per quantile level."""
    model = build_multi_input_model()
    model.eval()

    img = torch.randn(2, 3, 300, 300)
    sex = torch.tensor([[0.0], [1.0]])
    with torch.no_grad():
        out = model(img, sex)

    assert out.shape == (2, N_QUANTILES), f"expected (2, {N_QUANTILES}), got {tuple(out.shape)}"


def test_pinball_loss_scalar():
    preds = torch.zeros(4, N_QUANTILES)
    target = torch.ones(4, 1)
    loss = pinball_loss(preds, target)
    assert loss.ndim == 0 and loss.item() > 0


def test_calibration_and_intervals():
    """CQR calibration produces a correction per confidence and ordered bounds."""
    rng = np.random.default_rng(0)
    n = 200
    cal_preds = np.sort(rng.normal(0.5, 0.1, size=(n, N_QUANTILES)), axis=1)
    cal_true = rng.normal(0.5, 0.1, size=n)

    q_hat = calibrate(cal_preds, cal_true)
    assert set(q_hat.keys()) == set(CONFIDENCES)
    assert all(v >= 0 for v in q_hat.values())

    intervals = build_intervals(cal_preds, q_hat, max_age=240.0, y_min=0.0, y_max=240.0)
    for conf in CONFIDENCES:
        lo, hi = intervals[conf]
        assert np.all(hi >= lo), f"upper < lower for conf={conf}"
