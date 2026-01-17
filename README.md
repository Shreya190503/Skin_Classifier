## 🧬 Skin Cancer Detection, Interpretation, and Analysis System

This project presents an **end-to-end interactive skin cancer classification system** designed to support **research, academic demonstrations, and clinical AI interpretability studies**. The system integrates **deep learning–based image classification**, **model explainability**, and **performance analysis** within a unified web interface.

The application supports **multiple deep learning frameworks** and datasets, enabling seamless comparison between:

* A **Keras (.h5) model** trained on the **HAM10000 dataset (7 classes)**
* A **PyTorch (.pth) model** trained on a **custom dermoscopic dataset (8 classes)** using an **EfficientNet-B2 encoder with a classification head**

---

### 🔍 Key Features

**1. Multi-Model Selection Interface**
Users can dynamically select between different trained models via a sidebar, with automatic handling of:

* Framework differences (TensorFlow vs PyTorch)
* Varying class counts
* Dataset-specific label mappings

**2. Image-Based Skin Lesion Classification**
The system accepts dermoscopic images and performs real-time inference, displaying:

* Predicted skin lesion class
* Confidence score
* Full class-wise probability distribution

**3. Model Interpretability (Grad-CAM Ready)**
The PyTorch model architecture is reconstructed to enable **gradient-based visual explanations**, allowing:

* Localization of discriminative lesion regions
* Visual insight into model decision-making
* Transparent AI behavior for medical imaging tasks

**4. Quantitative Performance Analysis**
The application is designed to support:

* Accuracy reporting
* Confusion matrix visualization
* Comparative evaluation across models

**5. Robust Deployment-Oriented Design**
The system addresses real-world challenges such as:

* Loading raw PyTorch `state_dict` files
* Reconstructing model architecture programmatically
* Safe inference routing based on model framework
* Error handling for incompatible inputs

---

### 🧠 Technical Highlights

* **EfficientNet-B2 encoder** with SMP-based classification head (PyTorch)
* Framework-agnostic inference pipeline
* Modular architecture for easy extension to additional diseases or models
* Streamlit-based UI for rapid experimentation and demonstrations

---

### 🎯 Use Cases

* Academic research and IEEE conference demos
* Medical AI explainability studies
* Comparative evaluation of deep learning models
* Educational demonstrations in biomedical AI

---

### ⚠️ Disclaimer

This system is intended **strictly for research and educational purposes** and **must not be used for clinical diagnosis or treatment decisions**.
