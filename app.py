import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
from flask import Flask, request, render_template, jsonify
import gdown
import os

# =======================
# CONFIG
# =======================
app = Flask(__name__)

MODEL_PATH = "animal_species_model.pth"
GOOGLE_DRIVE_ID = "1SzvGyDls3p8qoNOJIfSLwJz_NpGlTimb"   # Replace with your file ID

# =======================
# DOWNLOAD MODEL FROM DRIVE
# =======================
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}", MODEL_PATH, quiet=False)

# =======================
# DEFINE MODEL
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 10  # Animals-10 dataset has 10 classes
model = models.vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# =======================
# IMAGE TRANSFORM
# =======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Class labels for Animals-10 dataset
classes = [
    "dog", "cat", "horse", "elephant", "butterfly",
    "chicken", "spider", "sheep", "cow", "squirrel"
]

# =======================
# ROUTES
# =======================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    try:
        # Read image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            label = classes[predicted.item()]

        return jsonify({"prediction": label})

    except Exception as e:
        return jsonify({"error": str(e)})

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    app.run(debug=True)
