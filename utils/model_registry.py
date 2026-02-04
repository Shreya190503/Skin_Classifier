import os
import json

# Base directory for models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "selected_models")
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")


def load_classes():
    """Load class names from classes.json"""
    with open(CLASSES_PATH, "r") as f:
        classes = json.load(f)
    # Return uppercase for display consistency
    return [c.upper() for c in classes]


# Class descriptions for skin conditions
CLASS_DESCRIPTIONS = {
    "AKIEC": "Actinic Keratoses / Intraepithelial Carcinoma - Precancerous lesion",
    "BCC": "Basal Cell Carcinoma - Common skin cancer",
    "BKL": "Benign Keratosis - Non-cancerous growth",
    "DF": "Dermatofibroma - Benign skin nodule",
    "MEL": "Melanoma - Serious skin cancer",
    "NV": "Melanocytic Nevus - Common mole (benign)",
    "SCC": "Squamous Cell Carcinoma - Skin cancer",
    "VASC": "Vascular Lesion - Blood vessel abnormality"
}


def get_available_models():
    """
    Get dictionary of available models with their configurations.

    Returns:
        dict: Model name -> config dict with keys:
            - framework: 'torch' or 'keras'
            - path: Path to model file
            - classes: List of class names
            - supports_gradcam: Whether GradCAM is supported
    """
    classes = load_classes()

    return {
        "EfficientNet-B2 (Combo Balanced)": {
            "framework": "torch",
            "path": os.path.join(MODELS_DIR, "best_model_confeetune_combo_balanced.pth"),
            "classes": classes,
            "supports_gradcam": True
        },

        "EfficientNet-B2 (Golden Combo)": {
            "framework": "torch",
            "path": os.path.join(MODELS_DIR, "best_model_golden_combo.pth"),
            "classes": classes,
            "supports_gradcam": True
        },

        "EfficientNet-B2 (Smart Combo v2)": {
            "framework": "torch",
            "path": os.path.join(MODELS_DIR, "best_model_smart_combo_v2.pth"),
            "classes": classes,
            "supports_gradcam": True
        }
    }


def get_class_description(class_name):
    """Get description for a class name."""
    return CLASS_DESCRIPTIONS.get(class_name.upper(), "Unknown condition")
