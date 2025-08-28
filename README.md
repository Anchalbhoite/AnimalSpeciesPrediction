# 🐾 Animal Species Prediction using Deep Learning  

[![Live Demo](https://img.shields.io/badge/Live%20App-Streamlit-brightgreen?logo=streamlit)](https://animalspeciesprediction-hd9rgdle5bsmzoq6fas8dt.streamlit.app/)  

This project leverages **Transfer Learning (VGG16)** to build an image classification model that can identify **10 different animal species** from images.  
The trained model is deployed with **Streamlit** to allow real-time predictions on uploaded images.  

---

## 📌 Project Overview  

- 📚 **Model Architecture:** VGG16 with custom classification layers  
- 🐶 **Classes:** 10 species (butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel)  
- 🧠 **Technique:** Transfer Learning (PyTorch)  
- 💾 **Dataset:** Raw images organized by species  
- 🖥️ **Platform:** Trained in Colab / Local (Python)  
- 🌐 **Deployment:** Streamlit App  

---

## 🏗️ Project Structure  

AnimalSpeciesPrediction/
│
├── dataset/
│ ├── raw-img/ # Original dataset
│ └── split/ # Train/Val/Test splits
│
├── models/ # Saved model checkpoints
│
├── app.py # Streamlit app for prediction
├── train.py # Training script
├── requirements.txt # Dependencies
├── README.md # Documentation

yaml
Copy code

---

## 🚀 How to Run the Project  

### 📌 Step 1: Clone the Repository  
```bash
git clone https://github.com/your-username/AnimalSpeciesPrediction.git
cd AnimalSpeciesPrediction
📌 Step 2: Install Requirements
bash
Copy code
pip install -r requirements.txt
📌 Step 3: Train the Model (Optional if you use pre-trained)
bash
Copy code
python train.py
📌 Step 4: Run the Streamlit App
bash
Copy code
streamlit run app.py
🧪 Dataset Format
Your dataset must be structured like this:

bash
Copy code
dataset/
└── split/
    ├── train/
    │   ├── cat/
    │   ├── dog/
    │   ├── elephant/
    │   └── ...
    ├── val/
    └── test/
📈 Model Training Details
✅ Base Model: VGG16 (without top layers)

🔒 Frozen Layers: All except custom classifier

📊 Metrics: Accuracy & Loss plotted

⏱️ Epochs: 5+ (configurable)

📂 Validation split handled in split/

📷 Prediction Example
Upload an image in the Streamlit app to get predictions with probabilities.

Example (Elephant image):

vbnet
Copy code
✅ Predicted Animal: Elephant  

📊 Class Probabilities  
butterfly: 0.00%  
cat: 0.00%  
chicken: 0.00%  
cow: 0.00%  
dog: 0.00%  
elephant: 100.00%  
...
🌟 Future Enhancements
🔮 Top-3 predictions display

🎥 Real-time webcam predictions

📊 Grad-CAM visualization for explainability

⚡ Deployment on Hugging Face Spaces / Docker

🤝 Contributors
Anchal Bhoite – Developer, Trainer, Documenter

📃 License
This project is open source and available under the MIT License.








ChatGPT can make mistakes. Check important info. 
