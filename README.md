# 🫁 AI Pneumonia Detection System

A professional Flask-based web application for detecting pneumonia in chest X-ray images using deep learning (DenseNet/CNN) with **Grad-CAM explainability**.

---

## 🚀 Quick Start

### 1. Clone / copy the project
```bash
cd pneumonia_web_app
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your pretrained model (optional)
Place your `model.h5` file in the root of the project.  
The model must accept **224×224 RGB images** and output a **single sigmoid neuron**  
(>0.5 = Pneumonia, ≤0.5 = Normal).

> **No model?** The app runs in **demo mode** with a realistic synthetic prediction
> and auto-generated Grad-CAM heatmap so you can explore all features immediately.

### 5. Run the app
```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🏗 Project Structure

```
pneumonia_web_app/
│
├── app.py                  # Flask routes & application logic
├── utils.py                # Model loading, preprocessing, Grad-CAM
├── model.h5                # ← Place your pretrained model here
├── detection_history.json  # Auto-created – stores scan history
├── requirements.txt
│
├── templates/
│   ├── dashboard.html      # Landing dashboard with statistics
│   ├── detect.html         # Upload page with drag-and-drop
│   ├── result.html         # Prediction results + heatmap
│   └── history.html        # Detection history table
│
└── static/
    ├── css/
    │   └── main.css        # Complete design system
    ├── uploads/            # Saved X-ray images
    └── heatmaps/           # Generated Grad-CAM heatmaps
```

---

## 🧠 Training Your Own Model

Use the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.

```python
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, Model

base = DenseNet121(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dense(256, activation='relu')(x)
out = layers.Dense(1, activation='sigmoid')(x)

model = Model(base.input, out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=10, validation_data=val_data)
model.save('model.h5')
```

---

## 🎯 Features

| Feature | Detail |
|---|---|
| **Upload** | Drag-and-drop or browse; JPG/PNG up to 16 MB |
| **AI Prediction** | DenseNet-based binary classification |
| **Grad-CAM** | Heatmap overlay showing influential lung regions |
| **Confidence Score** | Animated ring showing model certainty |
| **AI Explanation** | Plain-English clinical explanation |
| **History** | Filterable table with thumbnails and delete option |
| **Demo Mode** | Full functionality without a real model |

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**.  
It is **not** a substitute for professional medical advice, diagnosis, or treatment.  
Always consult a qualified radiologist for clinical decisions.
