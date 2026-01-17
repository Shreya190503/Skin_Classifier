import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils.model_registry import get_available_models
from utils.tf_infer import load_tf_model, predict_tf
from utils.torch_infer import predict_torch

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="Skin Cancer Detection & Analysis",
    page_icon="🧬",
    layout="wide"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# MODEL REGISTRY
# ======================================================
MODELS = get_available_models()

st.sidebar.title("🧠 Model Selection")
model_name = st.sidebar.selectbox(
    "Choose a model",
    list(MODELS.keys())
)

model_info = MODELS[model_name]
CLASS_NAMES = model_info["classes"]

# ======================================================
# LOAD MODEL
# ======================================================
from utils.torch_model import load_torch_model

@st.cache_resource
def load_model(model_name):
    model_info = MODELS[model_name]

    if model_info["framework"] == "keras":
        return load_tf_model(model_info["path"])

    elif model_info["framework"] == "torch":
        return load_torch_model(
            path=model_info["path"],
            num_classes=len(model_info["classes"]),
            device=DEVICE
        )

    else:
        st.error("Unsupported model framework")
        st.stop()

model = load_model(model_name)

st.sidebar.write("Framework:", model_info["framework"])
st.sidebar.write("Model type:", type(model))

# ======================================================
# UI
# ======================================================
st.title("🔬 Skin Cancer Detection, Interpretation & Analysis")

uploaded_file = st.file_uploader(
    "Upload dermoscopic image",
    type=["jpg", "jpeg", "png"]
)

# ======================================================
# PREDICTION PIPELINE
# ======================================================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    st.subheader("🧾 Prediction")

    # -------- Framework-safe inference --------
    if model_info["framework"] == "keras":
        probs = predict_tf(model, image)

    elif model_info["framework"] == "torch":
        probs = predict_torch(model, image, DEVICE)

    else:
        st.error("Unsupported model framework")
        st.stop()

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    st.metric("Predicted Class", pred_class)
    st.metric("Confidence", f"{confidence * 100:.2f}%")

    # -------- Probability Plot --------
    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, probs)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Class Probability Distribution")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # -------- Grad-CAM Section --------
    st.subheader("🔍 Model Interpretation (Grad-CAM)")
    if model_info["supports_gradcam"]:
        st.info("Grad-CAM supported for this model (can be enabled).")
    else:
        st.warning(
            "Grad-CAM not available for this model "
            "(PyTorch model was saved without full architecture)."
        )

# ======================================================
# ABOUT
# ======================================================
st.markdown("---")
st.markdown("""
### ℹ️ About
- Supports **multiple models**
- Handles **different class counts safely**
- Designed for **research & IEEE-style demos**

⚠️ Not a medical diagnostic tool.
""")
