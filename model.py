import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class MultiInputModel(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        
        # Load pre-trained EfficientNetB3
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        self.base_model = efficientnet_b3(weights=weights)
        
        # Extract the in_features of the classifier
        num_ftrs = self.base_model.classifier[1].in_features
        
        # Remove the classifier so base_model returns features
        self.base_model.classifier = nn.Identity()
        
        # Build the new head: feature + 1 (for sex_input)
        self.fc1 = nn.Linear(num_ftrs + 1, 256)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.out = nn.Linear(256, 1)

    def forward(self, img, sex):
        # Extract features (B, num_ftrs)
        feat = self.base_model(img)
        
        # Concatenate features with sex
        x = torch.cat((feat, sex), dim=1)
        
        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.out(x)
        
        return out

def build_multi_input_model(dropout=0.5, learning_rate=1e-4):
    """
    Returns the PyTorch multi-input model.
    Note: learning_rate is handled in the optimizer in PyTorch,
    but we keep the signature compatible.
    """
    model = MultiInputModel(dropout=dropout)
    return model
