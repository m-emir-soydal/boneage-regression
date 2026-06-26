import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b3,
    EfficientNet_B3_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)

DEFAULT_BACKBONE = "efficientnet_b3"

# Input (H, W) each backbone expects. EfficientNet-B3 keeps the project default
# 300x300; torchvision ViT-B/16 and timm InceptionNeXt-Base require 224x224.
BACKBONE_IMG_SIZE = {
    "efficientnet_b3": (300, 300),
    "vit_b_16": (224, 224),
    "inceptionnext_base": (224, 224),
}


def backbone_img_size(backbone=DEFAULT_BACKBONE):
    """Return the (H, W) input size expected by the given backbone."""
    if backbone not in BACKBONE_IMG_SIZE:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Known: {list(BACKBONE_IMG_SIZE)}"
        )
    return BACKBONE_IMG_SIZE[backbone]


class _TimmPooledBackbone(nn.Module):
    """Wrap a timm model so it returns a (B, num_features) pooled vector.

    timm's ``num_classes=0`` path does not yield clean pooled features for
    InceptionNeXt (its MLP head makes a direct call return 0-dim output), so we
    take the convolutional feature map from ``forward_features`` and global
    average pool it to match the (B, num_features) contract used by the other
    backbones.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feats = self.model.forward_features(x)
        feats = self.pool(feats)
        return torch.flatten(feats, 1)


def _build_backbone(backbone):
    """Return (feature_extractor, num_features) for the requested backbone.

    Every backbone exposes a callable that maps an image batch to a
    (B, num_features) vector so the downstream fusion head stays identical.
    """
    if backbone == "efficientnet_b3":
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        base_model = efficientnet_b3(weights=weights)
        num_ftrs = base_model.classifier[1].in_features
        base_model.classifier = nn.Identity()
        return base_model, num_ftrs

    if backbone == "vit_b_16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        base_model = vit_b_16(weights=weights)
        num_ftrs = base_model.heads.head.in_features
        # Drop the classification head so the model returns the (B, hidden_dim)
        # class-token embedding.
        base_model.heads = nn.Identity()
        return base_model, num_ftrs

    if backbone == "inceptionnext_base":
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "inceptionnext_base requires the 'timm' package. Install it with "
                "`pip install timm`."
            ) from e
        # timm registers this architecture as "inception_next_base". We wrap it
        # to global-average-pool forward_features into a (B, num_features) vector.
        timm_model = timm.create_model(
            "inception_next_base", pretrained=True, num_classes=0
        )
        num_ftrs = timm_model.num_features
        return _TimmPooledBackbone(timm_model), num_ftrs

    raise ValueError(
        f"Unknown backbone '{backbone}'. Expected one of: "
        "'efficientnet_b3', 'vit_b_16', 'inceptionnext_base'."
    )


class MultiInputModel(nn.Module):
    def __init__(self, dropout=0.5, backbone=DEFAULT_BACKBONE):
        super().__init__()

        self.backbone = backbone

        # Image feature extractor (backbone-specific, returns (B, num_ftrs)).
        self.base_model, num_ftrs = _build_backbone(backbone)

        # Dense layer for sex input
        self.sex_fc = nn.Linear(1, 32)

        # Build the new head: feature + 32 (for processed sex_input)
        self.fc1 = nn.Linear(num_ftrs + 32, 256)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.out = nn.Linear(256, 1)

    def forward_with_embedding(self, img, sex):
        """Return (prediction, penultimate_embedding).

        The penultimate 256-dim embedding is backbone-agnostic and is what the
        conformal-prediction pipeline consumes for KNN-NCP.
        """
        feat = self.base_model(img)
        sex = self.relu(self.sex_fc(sex))
        x = torch.cat((feat, sex), dim=1)

        penultimate = self.relu(self.fc1(x))
        out = self.out(self.dropout(penultimate))
        return out, penultimate

    def forward(self, img, sex):
        out, _ = self.forward_with_embedding(img, sex)
        return out


def build_multi_input_model(dropout=0.5, learning_rate=1e-4, backbone=None):
    """
    Returns the PyTorch multi-input model.
    Note: learning_rate is handled in the optimizer in PyTorch,
    but we keep the signature compatible.
    """
    if backbone is None:
        backbone = DEFAULT_BACKBONE
    model = MultiInputModel(dropout=dropout, backbone=backbone)
    return model
