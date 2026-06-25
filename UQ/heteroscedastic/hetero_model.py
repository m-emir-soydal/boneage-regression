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
    """EfficientNet-B3 + sex backbone with mean and log-variance heads.

    This model modifies a standard point regression model by adding a second
    prediction head. Instead of just outputting the predicted value (the mean),
    it outputs both the mean (point prediction) and the log-variance (uncertainty
    estimate) of a Gaussian distribution for each individual input sample.
    """

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()

        # Image feature extractor: pre-trained EfficientNet-B3
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        self.base_model = efficientnet_b3(weights=weights)

        num_ftrs = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Identity()  # Expose the raw pooling features

        # Sex feature extractor: linear projection of sex metadata (0 or 1)
        self.sex_fc = nn.Linear(1, 32)

        # Joint fully connected layers fusing image and tabular metadata
        self.fc1 = nn.Linear(num_ftrs + 32, 256)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Two heads sharing the same 256-d joint feature representation:
        # 1. mean_out: Predicts the expected target value (bone age, normalized).
        # 2. log_var_out: Predicts the log of target variance (log(sigma^2)).
        # Predicting log-variance keeps outputs unconstrained (range (-inf, inf)),
        # avoiding negative variance and numerical instability during NLL loss calculation.
        self.mean_out = nn.Linear(256, 1)
        self.log_var_out = nn.Linear(256, 1)

    def forward(self, img: torch.Tensor, sex: torch.Tensor):
        # Extract features from the image and sex metadata
        feat = self.base_model(img)
        sex = self.relu(self.sex_fc(sex))
        
        # Concatenate features and pass through the joint FC layers
        x = torch.cat((feat, sex), dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Predict both mean and log-variance
        mean = self.mean_out(x)
        log_var = self.log_var_out(x)
        return mean, log_var


def build_hetero_model(dropout: float = 0.5) -> HeteroscedasticMultiInputModel:
    """Factory function mirroring ``model.build_multi_input_model`` signature."""
    return HeteroscedasticMultiInputModel(dropout=dropout)
