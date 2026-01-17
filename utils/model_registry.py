def get_available_models():
    return {
        "Keras HAM10000 (7 classes)": {
            "framework": "keras",
            "path": "models/sc_70.h5",
            "classes": [
                "AKIEC", "BCC", "BKL",
                "DF", "MEL", "NV", "VASC"
            ],
            "supports_gradcam": True
        },

        "PyTorch EfficientNet-B2 (8 classes)": {
            "framework": "torch",
            "path": "models/best_model_confeetune_combo_balanced.pth",
            "classes": [
                "AKIEC", "BCC", "BKL", "DF",
                "MEL", "NV", "SCC", "VASC"
            ],
            "supports_gradcam": True
        }
    }
