"""
GradCAM++ implementation for PyTorch models.

GradCAM++ provides improved visual explanations by using weighted gradients
to identify which parts of the image influenced the model's prediction.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class GradCAMPlusPlus:
    """
    GradCAM++ implementation for PyTorch models.

    This implementation hooks into the last convolutional layer of the encoder
    and computes weighted gradient-based class activation maps.
    """

    def __init__(self, model, target_layer=None):
        """
        Initialize GradCAM++.

        Args:
            model: PyTorch model (SimpleClassifier)
            target_layer: Target layer for activation maps (default: last encoder layer)
        """
        self.model = model
        self.model.eval()

        # Get the last layer of the encoder
        if target_layer is None:
            # For SimpleClassifier, encoder is a Sequential of EfficientNet features
            self.target_layer = model.encoder[-1]
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        Generate GradCAM++ heatmap.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (default: predicted class)

        Returns:
            numpy array: Heatmap of shape (H, W) with values in [0, 1]
        """
        # Enable gradients
        input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)

        # GradCAM++ weights computation
        # Second derivative (approximated)
        grad_2 = gradients ** 2
        grad_3 = grad_2 * gradients

        # Sum over spatial dimensions for denominator
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        alpha_denom = 2 * grad_2 + sum_activations * grad_3
        alpha_denom = torch.where(
            alpha_denom != 0,
            alpha_denom,
            torch.ones_like(alpha_denom)
        )

        alpha = grad_2 / alpha_denom

        # ReLU on gradients (only positive influence)
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)

        # Weighted combination of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()

        return cam


def apply_colormap(heatmap, colormap=cv2.COLORMAP_JET):
    """
    Apply colormap to heatmap.

    Args:
        heatmap: numpy array (H, W) with values in [0, 1]
        colormap: OpenCV colormap

    Returns:
        numpy array: RGB image (H, W, 3) with values in [0, 255]
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    # Convert BGR to RGB
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored


def overlay_heatmap(image, heatmap, alpha=0.4):
    """
    Overlay heatmap on original image.

    Args:
        image: PIL Image (original)
        heatmap: numpy array (H, W) with values in [0, 1]
        alpha: Transparency of heatmap overlay

    Returns:
        PIL Image: Image with heatmap overlay
    """
    # Convert image to numpy
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply colormap
    heatmap_colored = apply_colormap(heatmap_resized)

    # Overlay
    overlay = (1 - alpha) * img_array + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


def generate_gradcam(model, input_tensor, original_image, target_class=None):
    """
    Generate GradCAM++ visualization.

    Args:
        model: PyTorch model (SimpleClassifier)
        input_tensor: Preprocessed input tensor (1, C, H, W)
        original_image: Original PIL Image for overlay
        target_class: Optional target class index

    Returns:
        tuple: (heatmap_image, overlay_image)
            - heatmap_image: PIL Image of the colored heatmap
            - overlay_image: PIL Image with heatmap overlaid on original
    """
    # Create GradCAM++ instance
    gradcam = GradCAMPlusPlus(model)

    # Generate heatmap
    heatmap = gradcam.generate(input_tensor, target_class)

    # Create colored heatmap image
    heatmap_colored = apply_colormap(heatmap)
    heatmap_resized = cv2.resize(
        heatmap_colored,
        (original_image.width, original_image.height)
    )
    heatmap_image = Image.fromarray(heatmap_resized)

    # Create overlay
    overlay_image = overlay_heatmap(original_image, heatmap)

    return heatmap_image, overlay_image
