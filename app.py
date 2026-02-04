import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob

from utils.model_registry import get_available_models, get_class_description
from utils.torch_infer import predict_torch, get_inference_transform
from utils.torch_model import load_torch_model
from utils.gradcam_torch import generate_gradcam

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="Skin Cancer Detection & Analysis",
    page_icon="🧬",
    layout="wide"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sample data directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DATA_DIR = os.path.join(BASE_DIR, "sample_data", "test")

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
@st.cache_resource
def load_model(model_name):
    model_info = MODELS[model_name]

    if model_info["framework"] == "torch":
        return load_torch_model(
            path=model_info["path"],
            num_classes=len(model_info["classes"]),
            device=DEVICE
        )
    else:
        st.error("Unsupported model framework")
        st.stop()

model = load_model(model_name)

st.sidebar.write("**Framework:**", model_info["framework"])
st.sidebar.write("**Device:**", DEVICE)
st.sidebar.write("**Classes:**", len(CLASS_NAMES))

# ======================================================
# SAMPLE DATA BROWSER
# ======================================================
st.sidebar.markdown("---")
st.sidebar.title("📁 Sample Data Browser")

# Get available class folders
sample_classes = sorted([d for d in os.listdir(SAMPLE_DATA_DIR) 
                         if os.path.isdir(os.path.join(SAMPLE_DATA_DIR, d))])

selected_class = st.sidebar.selectbox(
    "Select class folder",
    sample_classes,
    format_func=lambda x: f"{x.upper()} - {get_class_description(x).split(' - ')[0]}"
)

# Get images in selected folder
class_folder = os.path.join(SAMPLE_DATA_DIR, selected_class)
sample_images = sorted(glob.glob(os.path.join(class_folder, "*.jpg")))

use_sample = st.sidebar.checkbox("Use sample image", value=False)

selected_sample = None
if use_sample and sample_images:
    sample_names = [os.path.basename(p) for p in sample_images]
    selected_sample_name = st.sidebar.selectbox(
        "Select image",
        sample_names[:50]  # Limit to first 50
    )
    selected_sample = os.path.join(class_folder, selected_sample_name)

# ======================================================
# GRADCAM SETTINGS
# ======================================================
st.sidebar.markdown("---")
st.sidebar.title("🔍 GradCAM++ Settings")
enable_gradcam = st.sidebar.checkbox("Enable GradCAM++ Visualization", value=True)

# ======================================================
# UI
# ======================================================
st.title("🔬 Skin Cancer Detection, Interpretation & Analysis")

st.markdown("""
This app uses **EfficientNet-B2** deep learning models to classify dermoscopic images 
into **8 skin condition categories**. Grad-CAM++ visualization shows which regions 
the model focuses on for its predictions.
""")

# ======================================================
# IMAGE INPUT
# ======================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Upload dermoscopic image",
        type=["jpg", "jpeg", "png"]
    )

# Determine which image to use
image = None
image_source = None

if use_sample and selected_sample:
    image = Image.open(selected_sample).convert("RGB")
    image_source = f"Sample: {os.path.basename(selected_sample)}"
elif uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_source = f"Uploaded: {uploaded_file.name}"

# ======================================================
# PREDICTION PIPELINE
# ======================================================
if image:
    with col1:
        st.image(image, caption=image_source, use_column_width=True)

    st.markdown("---")
    st.subheader("🧾 Prediction Results")

    # Run inference
    probs = predict_torch(model, image, DEVICE)

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    # Display results
    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric("🎯 Predicted Class", pred_class)

    with result_col2:
        st.metric("📊 Confidence", f"{confidence * 100:.1f}%")

    with result_col3:
        st.metric("📍 Device", DEVICE.upper())

    # Class description
    st.info(f"**{pred_class}**: {get_class_description(pred_class)}")

    # Probability distribution
    st.subheader("📈 Class Probability Distribution")

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#ff6b6b' if i == pred_idx else '#4ecdc4' for i in range(len(CLASS_NAMES))]
    bars = ax.bar(CLASS_NAMES, probs, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Class Probabilities")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    # Top-3 predictions
    st.subheader("🏆 Top-3 Predictions")
    top_k = 3
    top_indices = np.argsort(probs)[::-1][:top_k]

    for i, idx in enumerate(top_indices):
        medal = ["🥇", "🥈", "🥉"][i]
        st.write(f"{medal} **{CLASS_NAMES[idx]}** - {probs[idx]*100:.2f}% - {get_class_description(CLASS_NAMES[idx])}")

    # ======================================================
    # GRADCAM++ VISUALIZATION
    # ======================================================
    if enable_gradcam and model_info["supports_gradcam"]:
        st.markdown("---")
        st.subheader("🔍 GradCAM++ Model Interpretation")

        with st.spinner("Generating GradCAM++ visualization..."):
            # Get input tensor
            transform = get_inference_transform()
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
            input_tensor.requires_grad = True

            # Generate GradCAM++
            heatmap_img, overlay_img = generate_gradcam(
                model, input_tensor, image, target_class=pred_idx
            )

        # Display
        gcam_col1, gcam_col2, gcam_col3 = st.columns(3)

        with gcam_col1:
            st.image(image, caption="Original Image", use_column_width=True)

        with gcam_col2:
            st.image(heatmap_img, caption="GradCAM++ Heatmap", use_column_width=True)

        with gcam_col3:
            st.image(overlay_img, caption="Overlay", use_column_width=True)

        st.caption("""
        **GradCAM++ Interpretation**: The heatmap shows which regions of the image 
        the model focused on when making its prediction. Red/yellow areas indicate 
        high importance, while blue areas indicate low importance.
        """)

# ======================================================
# CLASS REFERENCE
# ======================================================
with st.expander("📚 Class Reference Guide"):
    st.markdown("""
    | Code | Condition | Description |
    |------|-----------|-------------|
    | AKIEC | Actinic Keratoses | Precancerous scaly patches from sun damage |
    | BCC | Basal Cell Carcinoma | Most common skin cancer, rarely spreads |
    | BKL | Benign Keratosis | Non-cancerous skin growths |
    | DF | Dermatofibroma | Harmless firm skin nodules |
    | MEL | Melanoma | ⚠️ Serious skin cancer, needs immediate attention |
    | NV | Melanocytic Nevus | Common moles, usually benign |
    | SCC | Squamous Cell Carcinoma | Second most common skin cancer |
    | VASC | Vascular Lesion | Blood vessel abnormalities |
    """)

# ======================================================
# ABOUT
# ======================================================
st.markdown("---")
st.markdown("""
### ℹ️ About
- Uses **EfficientNet-B2** architecture trained on skin lesion datasets
- Supports **3 model variants** with different training strategies
- **GradCAM++** visualization for model interpretability
- Classifies into **8 skin condition categories**

⚠️ **Disclaimer**: This is a research tool and NOT a medical diagnostic device. 
Always consult a qualified dermatologist for medical advice.
""")
