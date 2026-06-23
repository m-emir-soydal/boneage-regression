"""Loss functions for the heteroscedastic regression model.

The Gaussian negative log-likelihood (NLL) for a per-sample Gaussian
``N(mu(x), sigma^2(x))`` is, up to a constant,

    L = 0.5 * ( exp(-log_var) * (y - mu)^2 + log_var )

Predicting ``log_var`` (instead of ``sigma`` or ``sigma^2`` directly) keeps
the parameter unconstrained and avoids divisions in the loss.

An optional ``smooth_l1`` term on the mean can be mixed in via
``hetero_loss(..., mae_weight=...)``. With ``mae_weight = 0`` the loss is
pure Gaussian NLL; a small positive weight stabilizes the mean head while
still letting NLL drive the variance head, which empirically preserves point
accuracy close to the base SmoothL1-trained model.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# log_var range clamp: prevents variance collapse to ~0 (huge inv-var causes
# exploding gradients) and runaway log_var values. The bounds correspond to
# sigma in roughly [1.8e-3, 7.4] in normalized age space.
LOG_VAR_MIN = -12.0
LOG_VAR_MAX = 4.0


def clamp_log_var(log_var: torch.Tensor) -> torch.Tensor:
    """Clamp ``log_var`` into a numerically safe range (see module docstring)."""
    return torch.clamp(log_var, min=LOG_VAR_MIN, max=LOG_VAR_MAX)


def gaussian_nll_loss(y: torch.Tensor, mean: torch.Tensor,
                      log_var: torch.Tensor) -> torch.Tensor:
    """Mean Gaussian NLL for a heteroscedastic regression head.

    All tensors must broadcast to the same shape (typically ``(B, 1)``).
    ``log_var`` is clamped before use; see :data:`LOG_VAR_MIN` / :data:`LOG_VAR_MAX`.
    """
    log_var = clamp_log_var(log_var)
    inv_var = torch.exp(-log_var)
    return 0.5 * (inv_var * (y - mean) ** 2 + log_var).mean()


def hetero_loss(y: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor,
                mae_weight: float = 0.0) -> torch.Tensor:
    """Gaussian NLL with an optional SmoothL1 penalty on the mean.

    ``mae_weight = 0`` recovers the standard heteroscedastic NLL. A small
    positive value (e.g. 0.01) keeps the mean head close to a SmoothL1
    optimum, which protects point accuracy when NLL alone would let MAE
    drift in favour of a better-calibrated variance.
    """
    nll = gaussian_nll_loss(y, mean, log_var)
    if mae_weight <= 0:
        return nll
    return nll + mae_weight * F.smooth_l1_loss(mean, y)
