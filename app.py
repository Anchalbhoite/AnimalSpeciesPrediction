import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# Define the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the model
@st.cache_resource
def load_model():
    model = CustomCNN()
    state_dict = torch.load("animal_species_model.pth", map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# Class labels
class_labels = ["Dog", "Cat", "Horse", "Elephant", "Butterfly", "Chicken", "Sheep", "Spider", "Squirrel", "Cow"]

# Prediction function
def predict_image(image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    return class_labels[predicted.item()]

# Streamlit UI
st.title("🐾 Animal Species Prediction App")
st.write("Upload an image, and the model will predict the animal species!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("🔍 Predicting...")
    label = predict_image(image)
    st.success(f"✅ Predicted Animal: **{label}**")
