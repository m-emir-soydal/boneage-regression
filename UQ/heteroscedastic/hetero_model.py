"""Heteroscedastic multi-input model for bone age regression.

Mirrors the architecture of ``model.MultiInputModel`` (EfficientNet-B3 image
backbone + sex branch + shared fully-connected head) but exposes **two**
output heads:

- ``mean_out``: predicts the normalized bone age (same target as the base
  model, ``y / max_age``).
- ``log_var_out``: predicts the log-variance of the Gaussian likelihood for
  that sample. Using ``log_var`` (instead of raw sigma) keeps the parameter
  unconstrained and numerically stable for the Gaussian NLL loss.

The backbone is duplicated rather than imported from ``model.py`` on purpose:
the UQ layer is meant to be additive, so this method must not modify or
depend on the production point-regression model definition.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3


class HeteroscedasticMultiInputModel(nn.Module):
    """EfficientNet-B3 + sex backbone with mean and log-variance heads."""

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()

        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        self.base_model = efficientnet_b3(weights=weights)

        num_ftrs = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Identity()

        self.sex_fc = nn.Linear(1, 32)

        self.fc1 = nn.Linear(num_ftrs + 32, 256)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Two heads sharing the same 256-d feature representation.
        self.mean_out = nn.Linear(256, 1)
        self.log_var_out = nn.Linear(256, 1)

    def forward(self, img: torch.Tensor, sex: torch.Tensor):
        feat = self.base_model(img)
        sex = self.relu(self.sex_fc(sex))
        x = torch.cat((feat, sex), dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        mean = self.mean_out(x)
        log_var = self.log_var_out(x)
        return mean, log_var


def build_hetero_model(dropout: float = 0.5) -> HeteroscedasticMultiInputModel:
    """Factory mirroring ``model.build_multi_input_model`` signature."""
    return HeteroscedasticMultiInputModel(dropout=dropout)
