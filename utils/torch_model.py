import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2

# Resolution must match training
RESOLUTION = 260
DROPOUT = 0.3


class SimpleClassifier(nn.Module):
    """
    Simple classifier matching the training architecture.
    Uses EfficientNet-B2 encoder with a custom classification head.
    """

    def __init__(self, encoder, num_classes, dropout=DROPOUT):
        super().__init__()
        self.encoder = encoder

        # Compute output features from encoder
        with torch.no_grad():
            dummy = torch.zeros(1, 3, RESOLUTION, RESOLUTION)
            out_features = self.encoder(dummy).shape[1]

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)


def load_torch_model(path, num_classes, device):
    """
    Load a trained PyTorch model.

    Args:
        path: Path to the .pth state dict file
        num_classes: Number of output classes
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Loaded model in eval mode
    """
    # Create backbone (EfficientNet-B2)
    backbone = efficientnet_b2(weights=None)

    # Extract encoder (features only, no classifier)
    encoder = nn.Sequential(*backbone.features)

    # Create the classifier
    model = SimpleClassifier(encoder, num_classes, dropout=DROPOUT)

    # Load weights
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model
