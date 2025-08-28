import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model

# ========================
# Configurations
# ========================
BASE_DIR = "dataset/raw-img"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = "models/animal_species_model.h5"

os.makedirs("models", exist_ok=True)

# ========================
# Build Model
# ========================
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = Flatten()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation="softmax")(x)  # Adjust class count if needed

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ========================
# Data Generators
# ========================
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    BASE_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    BASE_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# ========================
# Train Model
# ========================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# Save model
model.save(MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")

# ========================
# Evaluate Performance
# ========================
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

# Save metrics as numpy for later use
np.save("models/train_acc.npy", acc)
np.save("models/val_acc.npy", val_acc)
np.save("models/train_loss.npy", loss)
np.save("models/val_loss.npy", val_loss)

# ========================
# Plot Curves
# ========================
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(acc, label="Train Accuracy", marker="o")
plt.plot(val_acc, label="Val Accuracy", marker="o")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(loss, label="Train Loss", marker="o")
plt.plot(val_loss, label="Val Loss", marker="o")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig("models/training_curves.png")
plt.show()
