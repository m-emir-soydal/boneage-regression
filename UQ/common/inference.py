"""Shared inference helpers for UQ methods."""
import numpy as np
import torch
import torch.nn as nn

# z-scores for two-sided Gaussian prediction intervals.
Z_SCORES = {
    90: 1.645,
    95: 1.960,
}


def enable_dropout(model):
    """Put the model in eval mode but keep dropout layers active.

    This is the core Monte Carlo Dropout trick: BatchNorm and other layers
    stay in eval mode, while every ``nn.Dropout`` module is switched back to
    train mode so it keeps sampling at inference time.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    return model


def gaussian_intervals(mean, std, confidence):
    """Build ``mean +/- z * std`` intervals for a given confidence level.

    ``confidence`` is an int percentage (90 or 95). Returns ``(lower, upper)``
    as numpy arrays.
    """
    if confidence not in Z_SCORES:
        raise ValueError(f"Unsupported confidence {confidence}; expected one of {sorted(Z_SCORES)}")
    z = Z_SCORES[confidence]
    mean = np.asarray(mean)
    std = np.asarray(std)
    lower = mean - z * std
    upper = mean + z * std
    return lower, upper


def denormalize(arr, max_age):
    """Undo the training-time target normalization (divide-by-max_age)."""
    return np.asarray(arr) * max_age


@torch.no_grad()
def mc_forward_passes(model, loader, device, samples):
    """Run ``samples`` stochastic forward passes over ``loader``.

    Returns ``preds`` of shape ``(samples, N)`` in normalized space and the
    aligned ground-truth targets ``y_norm`` of shape ``(N,)``.

    Dropout must already be enabled via :func:`enable_dropout`. The data loader
    must not shuffle so that targets stay aligned across passes.
    """
    per_pass = []
    targets_norm = None

    for t in range(samples):
        batch_preds = []
        batch_targets = [] if targets_norm is None else None
        for batch in loader:
            inputs, targets = batch
            img = inputs["image_input"].to(device)
            sex = inputs["sex_input"].to(device)
            out = model(img, sex)
            batch_preds.append(out.cpu().numpy().reshape(-1))
            if batch_targets is not None:
                batch_targets.append(targets.cpu().numpy().reshape(-1))
        per_pass.append(np.concatenate(batch_preds))
        if batch_targets is not None:
            targets_norm = np.concatenate(batch_targets)

    preds = np.vstack(per_pass)  # (samples, N)
    return preds, targets_norm
