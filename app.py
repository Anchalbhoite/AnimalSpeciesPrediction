import streamlit as st
import torch
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
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    # Load pre-trained VGG16 and replace classifier
    model = models.vgg16(pretrained=False)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 10)  # Animals-10 dataset has 10 classes
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
    "Dog", "Cat", "Horse", "Elephant", "Butterfly",
    "Chicken", "Sheep", "Spider", "Squirrel", "Cow"
]

# -------------------------------
# 4. Prediction function
# -------------------------------
def predict_image(model, image):
    img = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    return class_labels[predicted.item()]

# -------------------------------
# 5. Streamlit App UI
# -------------------------------
st.title("🐾 Animal Species Prediction App")
st.write("Upload an image, and the model will predict the animal species!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("🔍 Predicting...")
    model = load_model()
    label = predict_image(model, image)

    st.success(f"✅ Predicted Animal: **{label}**")
