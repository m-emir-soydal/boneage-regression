"""Conformalized Quantile Regression (CQR) helpers.

Pipeline:
  1. Train a multi-quantile head with the pinball (quantile) loss.
  2. On the calibration split, compute conformity scores per confidence level
     and derive the additive correction q_hat (Romano et al., 2019).
  3. On the test split, build calibrated intervals [lo - q_hat, hi + q_hat]
     and report empirical coverage and mean interval width.

All quantile regression happens in normalized target space (boneage / max_age);
intervals are de-normalized to months for reporting.
"""
import numpy as np
import torch

from config import QUANTILES, QUANTILE_PAIRS, MEDIAN_IDX, quantile_index


def pinball_loss(preds, target, quantiles=QUANTILES):
    """Average pinball loss over all quantiles.

    preds:  (B, n_quantiles) predicted quantiles.
    target: (B, 1) or (B,) ground truth.
    """
    if target.dim() == 1:
        target = target.unsqueeze(1)
    q = torch.tensor(quantiles, dtype=preds.dtype, device=preds.device).view(1, -1)
    errors = target - preds  # (B, n_quantiles)
    loss = torch.maximum(q * errors, (q - 1.0) * errors)
    return loss.mean()


@torch.no_grad()
def predict_quantiles(model, loader, device):
    """Run the model over a loader and return (preds, y_true) as numpy arrays.

    preds: (N, n_quantiles) in normalized space, sorted ascending per row to
    enforce non-crossing quantiles. y_true: (N,) in normalized space.
    """
    model.eval()
    all_preds, all_true = [], []
    for batch in loader:
        inputs, targets = batch
        img = inputs["image_input"].to(device)
        sex = inputs["sex_input"].to(device)
        out = model(img, sex)
        all_preds.append(out.cpu().numpy())
        all_true.append(targets.cpu().numpy())

    preds = np.vstack(all_preds)
    preds = np.sort(preds, axis=1)  # prevent quantile crossing
    y_true = np.concatenate(all_true).flatten()
    return preds, y_true


def calibrate(cal_preds, cal_true, confidences=None):
    """Compute the CQR correction q_hat for each confidence level.

    cal_preds: (N, n_quantiles) normalized predictions on the cal split.
    cal_true:  (N,) normalized targets.
    Returns {confidence: q_hat} in normalized space.
    """
    pairs = QUANTILE_PAIRS if confidences is None else {
        c: QUANTILE_PAIRS[c] for c in confidences
    }
    n = len(cal_true)
    q_hat = {}
    for conf, (lo_q, hi_q) in pairs.items():
        lo = cal_preds[:, quantile_index(lo_q)]
        hi = cal_preds[:, quantile_index(hi_q)]
        # Conformity score: signed distance outside the interval.
        scores = np.maximum(lo - cal_true, cal_true - hi)
        # (1-alpha)(1 + 1/n) empirical quantile, clipped to [0, 1].
        level = min(1.0, (1.0 - (1.0 - conf)) * (1.0 + 1.0 / n))
        q_hat[conf] = float(np.quantile(scores, level, method="higher"))
    return q_hat


def build_intervals(preds, q_hat, max_age, y_min=None, y_max=None):
    """De-normalize calibrated intervals to months for each confidence level.

    Returns {confidence: (lo_months, hi_months)} arrays of shape (N,).
    """
    out = {}
    for conf, qh in q_hat.items():
        lo_q, hi_q = QUANTILE_PAIRS[conf]
        lo = (preds[:, quantile_index(lo_q)] - qh) * max_age
        hi = (preds[:, quantile_index(hi_q)] + qh) * max_age
        if y_min is not None:
            lo = np.clip(lo, y_min, None)
        if y_max is not None:
            hi = np.clip(hi, None, y_max)
        out[conf] = (lo, hi)
    return out


def median_prediction(preds, max_age):
    """De-normalized median (0.5 quantile) point prediction in months."""
    return preds[:, MEDIAN_IDX] * max_age
