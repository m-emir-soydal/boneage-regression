"""Smoke tests for the multi-backbone PyTorch bone-age model and pipeline wiring.

These tests are dataset-independent: they exercise model construction, the
fused-embedding forward path, backbone-aware image sizing, and that the
conformal-prediction config scopes its outputs per backbone. Pretrained weights
are loaded from the local torch hub cache when available.
"""

import sys
from pathlib import Path

import pytest
import torch

# Ensure the project root is importable when running `pytest` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import (
    DEFAULT_BACKBONE,
    BACKBONE_IMG_SIZE,
    MultiInputModel,
    backbone_img_size,
    build_multi_input_model,
)

BACKBONES = ["efficientnet_b3", "vit_b_16"]


@pytest.mark.parametrize("backbone", BACKBONES)
def test_backbone_img_size(backbone):
    h, w = backbone_img_size(backbone)
    assert (h, w) == BACKBONE_IMG_SIZE[backbone]
    assert h > 0 and w > 0


def test_backbone_img_size_unknown_raises():
    with pytest.raises(ValueError):
        backbone_img_size("not_a_backbone")


def test_unknown_backbone_raises():
    with pytest.raises(ValueError):
        build_multi_input_model(backbone="not_a_backbone")


def test_default_backbone():
    model = build_multi_input_model()
    assert isinstance(model, MultiInputModel)
    assert model.backbone == DEFAULT_BACKBONE


@pytest.mark.parametrize("backbone", BACKBONES)
def test_forward_and_embedding_shapes(backbone):
    """Model builds, predicts a scalar, and exposes a 256-dim fused embedding."""
    model = build_multi_input_model(backbone=backbone)
    model.eval()

    batch = 2
    h, w = backbone_img_size(backbone)
    img = torch.randn(batch, 3, h, w)
    sex = torch.randn(batch, 1)

    with torch.no_grad():
        out, emb = model.forward_with_embedding(img, sex)

    assert out.shape == (batch, 1), "Model should output one regression value per sample"
    assert emb.shape == (batch, 256), "Fused embedding must be 256-dim for all backbones"


@pytest.mark.parametrize("backbone", BACKBONES)
def test_forward_matches_embedding_path(backbone):
    """`forward` must return the same prediction as `forward_with_embedding`."""
    model = build_multi_input_model(backbone=backbone)
    model.eval()

    h, w = backbone_img_size(backbone)
    img = torch.randn(2, 3, h, w)
    sex = torch.randn(2, 1)

    with torch.no_grad():
        out_a = model(img, sex)
        out_b, _ = model.forward_with_embedding(img, sex)

    assert torch.allclose(out_a, out_b)


def test_cps_config_scopes_outputs_per_backbone():
    import cps_config as C

    assert C.BACKBONE in C.BACKBONES
    # Outputs live under results/cps/<tag>/<backbone>/ so backbones never collide.
    assert C.RUN_OUTPUT_DIR.name == C.BACKBONE
    assert C.backbone_img_size() == C.BACKBONES[C.BACKBONE]
    # Comparison artifacts are shared across backbones (one level up).
    assert C.comparison_dir().parent == C.RUN_OUTPUT_DIR.parent


def test_data_loader_helpers_importable():
    """Loader builders import and accept an img_size without touching the dataset."""
    from data_loader import build_datasets, build_val_or_test_loader

    assert callable(build_datasets)
    assert callable(build_val_or_test_loader)
