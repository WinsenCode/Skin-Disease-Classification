import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Config

NUM_CLASSES = 10
IMAGE_SIZE = 224
IN_CHANNELS = 3

CLASS_NAMES = [
    "Eczema",
    "Warts / Molluscum",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma (BCC)",
    "Melanocytic Nevi (NV)",
    "Benign Keratosis (BKL)",
    "Psoriasis / Lichen Planus",
    "Seborrheic Keratoses",
    "Tinea / Fungal Infections"
]

# Model Definition

def load_model(weights_path, device):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Preprocessing

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.6620, 0.4988, 0.4817],
                        [0.1619, 0.1485, 0.1517])   
])

# Streamlit App

st.set_page_config(
    page_title="DermaScan — Skin Disease Classifier",
    page_icon="🔬",
    layout="wide"
)

st.title("DermaScan — Skin Disease Classifier")
st.caption("Upload a skin lesion image for AI-powered classification using deep learning models.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sidebar
with st.sidebar:
    st.header("Settings")

    model_choice = "EfficientNet-B0"

    # Weight paths — same directory as app.py
    weight_paths = {
        "EfficientNet-B0": "efficientnet_best.pt"
    }

    top_k = st.slider("Show top-K predictions", min_value=3, max_value=10, value=5)

    st.divider()
    st.caption(f"Running on: **{device}**")

#  Main Area 
uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Supported formats: JPG, PNG, BMP"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Load model
    try:
        model = load_model(weight_paths[model_choice], device)
    except FileNotFoundError:
        st.error(f"Model weights not found at `{weight_paths[model_choice]}`. "
                 f"Make sure you've saved your trained weights first.")
        st.stop()

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()

    pred_idx = np.argmax(probs)
    pred_class = CLASS_NAMES[pred_idx]
    pred_conf = probs[pred_idx]

    # Layout: Two columns
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")

        st.metric(
            label="Predicted class",
            value=pred_class,
            delta=f"{pred_conf * 100:.1f}% confidence"
        )

        st.divider()

        # Top-K bar chart
        st.write(f"**Top {top_k} predictions**")

        sorted_indices = np.argsort(probs)[::-1][:top_k]
        top_classes = [CLASS_NAMES[i] for i in sorted_indices]
        top_probs = [probs[i] for i in sorted_indices]

        fig, ax = plt.subplots(figsize=(6, 0.4 * top_k))
        bars = ax.barh(
            range(len(top_classes)),
            top_probs,
            color=["#378ADD" if i == 0 else "#B4B2A9" for i in range(len(top_classes))]
        )
        ax.set_yticks(range(len(top_classes)))
        ax.set_yticklabels(top_classes, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Confidence")
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for i, (bar, prob) in enumerate(zip(bars, top_probs)):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{prob * 100:.1f}%", va="center", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Disclaimer
    st.divider()
    st.warning(
        "⚠️ This tool is for **educational purposes only**. "
        "It is not a substitute for professional medical diagnosis. "
        "Always consult a dermatologist for skin concerns."
    )

else:
    st.info("Upload an image from the file uploader above to get started.")

    st.divider()
    st.subheader("Supported conditions")

    cols = st.columns(5)
    for i, name in enumerate(CLASS_NAMES):
        with cols[i % 5]:
            st.write(f"• {name}")