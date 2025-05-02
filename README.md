# AnimalSpeciesPrediction
# 🐾 Animal Species Prediction using Deep Learning

This project leverages transfer learning (VGG16) to build an image classification model that can identify **10 different animal species** from images. The model is trained using a large-scale image dataset and deployed in a way that enables real-time or uploaded image prediction.

---

## 📌 Project Overview

- 📚 **Model Architecture:** VGG16 with custom classification layers
- 🐶 **Classes:** 10 species (dog, cat, lion, elephant, etc.)
- 🧠 **Technique:** Transfer Learning (Keras, TensorFlow)
- 💾 **Dataset:** Raw image folders organized by species
- 🖥️ **Platform:** Google Colab (Python)
- 🗂️ **Deployment:** Local & Colab prediction supported

---

## 🏗️ Project Structure

```
AnimalSpeciesPrediction/
│
├── dataset/
│   └── raw-img/              # Subfolders for each animal class
│
├── main.py                   # Model training script
├── predict.py                # Script for image prediction (optional enhancement)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── saved_models/
    └── animal_species_model.h5  # Saved trained model
```

---

## 🚀 How to Run the Project

### 📌 Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/AnimalSpeciesPrediction.git
cd AnimalSpeciesPrediction
```

### 📌 Step 2: Install Requirements

```bash
pip install -r requirements.txt
```

### 📌 Step 3: Train the Model (or Load Pretrained)

```bash
python main.py
```

---

## 🧪 Dataset Format

The dataset must be structured like this:

```
dataset/
└── raw-img/
    ├── cat/
    ├── dog/
    ├── elephant/
    └── ...
```

- Each folder should contain relevant images of that species.

---

## 📈 Model Training Details

- ✅ Base Model: VGG16 (without top layers)
- 🔒 Frozen Layers: All layers except the custom classifier
- 📊 Accuracy & Loss: Plotted using Matplotlib
- ⏱️ Epochs: 5+ (configurable)
- 📂 Validation Split: 20%

---

## 📷 Predict on New Image

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("saved_models/animal_species_model.h5")
img = image.load_img("test.jpg", target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
pred = model.predict(x)
print("Predicted class:", np.argmax(pred))
```

---

## 🌟 Future Enhancements

- ✅ Real-time prediction via webcam
- ✅ Deployment via Flask or Streamlit
- ✅ Model improvement using ResNet/EfficientNet
- ✅ Grad-CAM visualization for interpretability

---

## 🤝 Contributors

- **Anchal Bhoite** – Developer, Trainer, Documenter

---

## 📃 License

This project is open source and available under the [MIT License](LICENSE).
