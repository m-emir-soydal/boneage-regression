import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import build_multi_input_model


def test_model_architecture():
    """Smoke test to check if the multi-input model builds and runs a forward pass."""
    model = build_multi_input_model()
    assert isinstance(model, torch.nn.Module)

    img = torch.randn(2, 3, 300, 300)
    sex = torch.randn(2, 1)
    with torch.no_grad():
        out = model(img, sex)

    assert out.shape == (2, 1)
