import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SMPClassifier(nn.Module):
    def __init__(self, encoder_name="efficientnet-b2", num_classes=8):
        super().__init__()

        # Encoder (this matches encoder.* keys)
        self.encoder = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=3,
            depth=5,
            weights=None,
        )

        # Classification head (this matches head.* keys)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = smp.base.ClassificationHead(
            in_channels=self.encoder.out_channels[-1],
            classes=num_classes,
            pooling="avg",
            dropout=0.2,
            activation=None,
        )

    def forward(self, x):
        features = self.encoder(x)
        x = features[-1]
        x = self.head(x)
        return x


def load_torch_model(path, num_classes, device):
    model = SMPClassifier(
        encoder_name="efficientnet-b2",
        num_classes=num_classes
    )

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model
