"""Bayesian multi-input model for bone age regression.

Mirrors the architecture of ``model.MultiInputModel`` (EfficientNet-B3 image
backbone + sex branch + shared fully-connected head) but replaces the fully-connected
head layers with ``VariationalLinear`` layers (Bayes by Backprop, Blundell et al. 2015).

This allows capturing epistemic uncertainty in the regression head while keeping
the pre-trained EfficientNet backbone deterministic for tractable training.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3


class VariationalLinear(nn.Module):
    """Linear layer with variational weights and biases (Bayes by Backprop).

    Maintains mean (mu) and unconstrained scale (rho) parameters for weights
    and biases. During forward passes, weights are sampled from the Gaussian
    posterior using the reparameterization trick.
    """

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Weight parameters
        self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.rho_w = nn.Parameter(torch.Tensor(out_features, in_features))

        # Bias parameters
        self.mu_b = nn.Parameter(torch.Tensor(out_features))
        self.rho_b = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize mu with standard Kaiming / He uniform initialization
        stdv = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.mu_w, -stdv, stdv)
        nn.init.uniform_(self.mu_b, -stdv, stdv)

        # Initialize rho to yield a small initial standard deviation (e.g., sigma ~ 0.05)
        # sigma = softplus(rho) => rho = log(exp(sigma) - 1)
        init_rho = math.log(math.exp(0.05) - 1.0)
        nn.init.constant_(self.rho_w, init_rho)
        nn.init.constant_(self.rho_b, init_rho)

    @property
    def sigma_w(self) -> torch.Tensor:
        return F.softplus(self.rho_w)

    @property
    def sigma_b(self) -> torch.Tensor:
        return F.softplus(self.rho_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In training mode or when sampling is explicitly desired at inference,
        # sample weights using the reparameterization trick.
        # Note: BNN evaluation sets model.train() or keeps sampling enabled to get MC spread.
        if self.training:
            eps_w = torch.randn_like(self.mu_w)
            eps_b = torch.randn_like(self.mu_b)
            weight = self.mu_w + self.sigma_w * eps_w
            bias = self.mu_b + self.sigma_b * eps_b
        else:
            # Deterministic mean forward pass
            weight = self.mu_w
            bias = self.mu_b

        return F.linear(x, weight, bias)

    def kl_loss(self) -> torch.Tensor:
        """Compute the analytical KL divergence between posterior and prior.

        D_KL( N(mu, sigma^2) || N(0, prior_sigma^2) ) =
            log(prior_sigma / sigma) + (sigma^2 + mu^2) / (2 * prior_sigma^2) - 0.5
        """
        sigma_w = self.sigma_w
        kl_w = (
            math.log(self.prior_sigma)
            - torch.log(sigma_w)
            + (sigma_w**2 + self.mu_w**2) / (2.0 * self.prior_sigma**2)
            - 0.5
        )

        sigma_b = self.sigma_b
        kl_b = (
            math.log(self.prior_sigma)
            - torch.log(sigma_b)
            + (sigma_b**2 + self.mu_b**2) / (2.0 * self.prior_sigma**2)
            - 0.5
        )

        return kl_w.sum() + kl_b.sum()


class BayesianMultiInputModel(nn.Module):
    """EfficientNet-B3 + sex backbone with a variational Bayesian head."""

    def __init__(self, dropout: float = 0.5, prior_sigma: float = 1.0) -> None:
        super().__init__()

        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        self.base_model = efficientnet_b3(weights=weights)

        num_ftrs = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Identity()

        self.sex_fc = nn.Linear(1, 32)

        # Variational fully-connected layers
        self.fc1 = VariationalLinear(num_ftrs + 32, 256, prior_sigma=prior_sigma)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.out = VariationalLinear(256, 1, prior_sigma=prior_sigma)

    def forward(self, img: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        feat = self.base_model(img)
        sex = self.relu(self.sex_fc(sex))
        x = torch.cat((feat, sex), dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.out(x)
        return out

    def kl_loss(self) -> torch.Tensor:
        """Accumulate KL divergence from all variational layers."""
        return self.fc1.kl_loss() + self.out.kl_loss()


def build_bnn_model(dropout: float = 0.5, prior_sigma: float = 1.0) -> BayesianMultiInputModel:
    """Factory mirroring ``model.build_multi_input_model`` signature."""
    return BayesianMultiInputModel(dropout=dropout, prior_sigma=prior_sigma)


def enable_bnn_sampling(model: nn.Module) -> nn.Module:
    """Put the model in eval mode but keep variational and dropout layers active.

    Ensures BatchNorm layers use running statistics while VariationalLinear
    and Dropout layers sample stochastic weights/activations for Monte Carlo evaluation.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, (VariationalLinear, nn.Dropout)):
            module.train()
    return model
