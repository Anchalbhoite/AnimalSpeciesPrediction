import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys
import os

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
CLASS_LABELS = [
    'butterfly', 'cat', 'chicken', 'cow', 'dog',
    'elephant', 'horse', 'sheep', 'spider', 'squirrel'
]

# =========================
# FUNCTION FOR PREDICTION
# =========================
def predict_species(model_path, image_path):
    # Check paths
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image file not found: {image_path}")

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Load & preprocess image
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)
    predicted_idx = np.argmax(prediction)
    predicted_label = CLASS_LABELS[predicted_idx]
    confidence = float(np.max(prediction))

    return predicted_label, confidence


# =========================
# COMMAND LINE EXECUTION
# =========================
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <path_to_model.h5> <path_to_image>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    try:
        label, conf = predict_species(model_path, image_path)
        print(f"✅ Predicted Class: {label} (Confidence: {conf:.2f})")
    except Exception as e:
        print(f"❌ Error: {e}")
