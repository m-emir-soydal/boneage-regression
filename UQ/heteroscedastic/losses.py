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
    """Clamp ``log_var`` into a numerically safe range (see module docstring).

    Clamping is crucial because:
    - If log_var becomes too negative (variance -> 0), the inverse variance `exp(-log_var)`
      approaches infinity, leading to exploding gradients during backpropagation.
    - If log_var becomes too positive (variance -> infinity), training can collapse
      as the loss function is minimized simply by predicting massive uncertainty.
    """
    return torch.clamp(log_var, min=LOG_VAR_MIN, max=LOG_VAR_MAX)


def gaussian_nll_loss(y: torch.Tensor, mean: torch.Tensor,
                      log_var: torch.Tensor) -> torch.Tensor:
    """Mean Gaussian NLL for a heteroscedastic regression head.

    Calculates the loss:
        L = 0.5 * ( exp(-log_var) * (y - mean)^2 + log_var )
    
    This loss function achieves two goals:
    1. `exp(-log_var) * (y - mean)^2`: Divides the squared residual error by the predicted
       variance. For inputs where the model is highly uncertain, it predicts a larger variance
       (larger log_var), which attenuates/reduces the impact of large prediction errors.
    2. `log_var`: Serves as a penalty term. Without this term, the model could minimize
       the loss by predicting infinite variance for every sample.
    """
    log_var = clamp_log_var(log_var)
    inv_var = torch.exp(-log_var)  # exp(-log_var) = 1 / var
    # Up to a constant additive term, this is the negative log of a Gaussian density:
    return 0.5 * (inv_var * (y - mean) ** 2 + log_var).mean()


def hetero_loss(y: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor,
                mae_weight: float = 0.0) -> torch.Tensor:
    """Gaussian NLL with an optional SmoothL1 penalty on the mean.

    ``mae_weight = 0`` recovers the standard heteroscedastic NLL. A small
    positive value (e.g. 0.01) keeps the mean head close to a SmoothL1
    optimum, which protects point accuracy when NLL alone would let MAE
    drift in favour of a better-calibrated variance.
    
    Args:
        y (Tensor): Ground truth normalized targets.
        mean (Tensor): Predicted normalized means.
        log_var (Tensor): Predicted log-variances.
        mae_weight (float): Weight of the auxiliary Smooth L1 loss.
        
    Returns:
        Tensor: Combined scalar loss.
    """
    nll = gaussian_nll_loss(y, mean, log_var)
    if mae_weight <= 0:
        return nll
    # Add a point-prediction optimization penalty (Smooth L1) to stabilize the mean predictor
    return nll + mae_weight * F.smooth_l1_loss(mean, y)
