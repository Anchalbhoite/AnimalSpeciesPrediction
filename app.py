import streamlit as st
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import gdown

# -------------------------------
# 1. Download model from Google Drive if not exists
# -------------------------------
MODEL_PATH = "animal_species_model.pth"
DRIVE_ID = "1SzvGyDls3p8qoNOJIfSLwJz_NpGlTimb"

@st.cache_resource
def load_model(arch="resnet18"):
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    # Choose architecture
    if arch == "resnet18":
        model = models.resnet18(weights=None)
    elif arch == "resnet34":
        model = models.resnet34(weights=None)
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError("Invalid architecture")

    # Replace classifier
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

    # Load weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# -------------------------------
# 2. Define transforms
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------------------
# 3. Class labels (update with your dataset labels)
# -------------------------------
class_labels = [
    "butterfly", "cat", "chicken", "cow", "dog",
    "elephant", "horse", "sheep", "spider", "squirrel"
]


# -------------------------------
# 4. Prediction function
# -------------------------------
def predict_image(model, image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    return predicted.item(), probs[0]

# -------------------------------
# 5. Streamlit App UI
# -------------------------------
st.title("🐾 Animal Species Prediction App")
st.write("Upload an image, and the model will predict the animal species!")

# Architecture selector
arch_choice = st.selectbox("Choose model architecture:", ["resnet18", "resnet34", "resnet50"])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("🔍 Predicting...")
    model = load_model(arch_choice)
    idx, probs = predict_image(model, image)

    # Predicted label
    label = class_labels[idx]
    st.success(f"✅ Predicted Animal: **{label}**")

    # Show raw probabilities
    st.subheader("📊 Class Probabilities")
    for i, cls in enumerate(class_labels):
        st.write(f"{cls}: {probs[i].item()*100:.2f}%")

    # Debug: raw tensor
    st.subheader("🛠 Raw Model Output")
    st.write(probs.tolist())
