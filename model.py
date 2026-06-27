import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from config import N_QUANTILES

class MultiInputModel(nn.Module):
    def __init__(self, dropout=0.5, n_quantiles=N_QUANTILES):
        super().__init__()

        self.n_quantiles = n_quantiles

        # Load pre-trained EfficientNetB3
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        self.base_model = efficientnet_b3(weights=weights)

        # Extract the in_features of the classifier
        num_ftrs = self.base_model.classifier[1].in_features

        # Remove the classifier so base_model returns features
        self.base_model.classifier = nn.Identity()

        # Dense layer for sex input
        self.sex_fc = nn.Linear(1, 32)

        # Build the new head: feature + 32 (for processed sex_input)
        self.fc1 = nn.Linear(num_ftrs + 32, 256)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # One output per quantile level (CQR quantile-regression head).
        self.out = nn.Linear(256, n_quantiles)

    def forward(self, img, sex):
        # Extract features (B, num_ftrs)
        feat = self.base_model(img)

        # Process sex through dense layer
        sex = self.relu(self.sex_fc(sex))

        # Concatenate features with processed sex
        x = torch.cat((feat, sex), dim=1)

        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.out(x)  # (B, n_quantiles)

        return out

def build_multi_input_model(dropout=0.5, learning_rate=1e-4, n_quantiles=N_QUANTILES):
    """
    Returns the PyTorch multi-input quantile-regression model.
    Note: learning_rate is handled in the optimizer in PyTorch,
    but we keep the signature compatible.
    """
    model = MultiInputModel(dropout=dropout, n_quantiles=n_quantiles)
    return model
