import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sys

# Ensure the model and image paths are passed as arguments
if len(sys.argv) < 3:
    print("Usage: python predict.py <path_to_model.h5> <path_to_image>")
    sys.exit(1)

model_path = sys.argv[1]
image_path = sys.argv[2]

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Define image size
IMG_SIZE = 224

# Load and preprocess the image
img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize

# Make prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# Class labels (adjust these based on your dataset folders)
class_labels = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

print(f"Predicted Class: {class_labels[predicted_class]}")
