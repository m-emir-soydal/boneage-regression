"""Heteroscedastic regression UQ method.

Trains the same EfficientNet-B3 + sex backbone as the base model, but replaces
the single regression head with two heads (mean + log-variance) optimized
with a Gaussian negative log-likelihood. At inference the model returns
``mean +/- z * sigma(x)`` intervals reusing the shared UQ helpers.

This package is fully self-contained: removing ``UQ/heteroscedastic/`` and
``UQ/results/heteroscedastic/`` removes the method without touching the core
training pipeline or other UQ methods.
"""
