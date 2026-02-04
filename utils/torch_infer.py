import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Resolution must match training
RESOLUTION = 260


class ResizeAndPad:
    """
    Resize image while preserving aspect ratio and pad to square.
    This matches the exact preprocessing used during training.
    """

    def __init__(self, output_size, fill=0):
        self.output_size = output_size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        ratio = min(self.output_size / w, self.output_size / h)
        new_w, new_h = int(w * ratio), int(h * ratio)

        img = transforms.functional.resize(img, (new_h, new_w), antialias=True)

        # Create padded canvas
        canvas = Image.new("RGB", (self.output_size, self.output_size), self.fill)
        canvas.paste(
            img,
            ((self.output_size - new_w) // 2,
             (self.output_size - new_h) // 2)
        )
        return canvas


def get_inference_transform():
    """Get the inference transform pipeline matching training."""
    return transforms.Compose([
        ResizeAndPad(RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict_torch(model, image, device):
    """
    Run inference on a PIL image using a PyTorch model.

    Args:
        model: PyTorch model in eval mode
        image: PIL Image (RGB)
        device: Device string ('cuda' or 'cpu')

    Returns:
        numpy array of class probabilities
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("predict_torch() received a non-PyTorch model")

    transform = get_inference_transform()
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    return probs.cpu().numpy().squeeze()


def predict_torch_with_gradcam(model, image, device, target_class=None):
    """
    Run inference and return both predictions and data needed for GradCAM.

    Args:
        model: PyTorch model
        image: PIL Image (RGB)
        device: Device string
        target_class: Optional target class index for GradCAM

    Returns:
        tuple: (probs, input_tensor, logits)
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("predict_torch_with_gradcam() received a non-PyTorch model")

    transform = get_inference_transform()
    x = transform(image).unsqueeze(0).to(device)
    x.requires_grad = True

    logits = model(x)
    probs = torch.softmax(logits, dim=1)

    return probs.detach().cpu().numpy().squeeze(), x, logits
