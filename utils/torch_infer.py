import torch
import numpy as np
from torchvision import transforms

def predict_torch(model, image, device):
    if not isinstance(model, torch.nn.Module):
        raise TypeError("predict_torch() received a non-PyTorch model")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    return probs.cpu().numpy().squeeze()
