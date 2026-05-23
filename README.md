# crack-detection
AI-powered structural crack detection using U-Net deep learning model trained on the DeepCrack dataset

# 🔍 CrackScan — AI-Powered Crack Detection System

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)

> A deep learning web application that detects cracks in walls, roads,
> and structures using U-Net segmentation with ResNet-34 backbone.
> Upload any photo and get instant crack analysis with severity assessment.

---

## 📸 Demo

| Input Image | Probability Map | Crack Overlay |
|---|---|---|
| Original photo | Heatmap of crack probability | Red highlighted cracks |

---

## ✅ Model Performance

| Metric | Score |
|---|---|
| **IoU** | 0.71 |
| **Dice Score** | 0.83 |
| **Pixel Accuracy** | 90%+ |
| **Precision** | 0.70+ |
| **Recall** | 0.70+ |

---

## 🚀 Features

- 📤 Upload any crack photo — JPG, PNG, BMP supported
- 🧠 Deep learning model detects cracks in seconds
- 🗺️ Shows probability heatmap of crack regions
- 🔴 Red overlay highlighting detected cracks
- 📊 Crack coverage percentage and severity level
- ⚠️ Severity assessment — Low / Medium / High risk
- 🌐 Clean web interface — runs in your browser

---

## 🏗️ Model Architecture

| Component | Detail |
|---|---|
| **Architecture** | U-Net |
| **Backbone** | ResNet-34 (ImageNet pretrained) |
| **Loss Function** | BCE + Dice Loss (50/50) |
| **Optimizer** | Adam (lr = 0.0001) |
| **Scheduler** | ReduceLROnPlateau |
| **Epochs** | 50 |
| **Input Size** | 256 × 256 |
| **Training Device** | Google Colab T4 GPU |

---

## 📁 Project Structure
crack-detection/
│
├── app.py                  ← Flask web application
├── requirements.txt        ← Python dependencies
├── crack_unet_best.pth     ← Trained model weights
│
├── templates/
│   └── index.html          ← Web UI (upload + results)
│
└── static/
└── uploads/            ← Temporary image storage

---

## ⚙️ Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/ompreet-s/crack-detection.git
cd crack-detection
```

### 2. Create conda environment
```bash
conda create -n crack_env python=3.10 -y
conda activate crack_env
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the trained model
Download `crack_unet_best.pth` and place it in the root folder next to `app.py`.

📥 **[Download Model from Google Drive](#)** ← paste your Drive link here

### 5. Run the app
```bash
python app.py
```

### 6. Open in browser
---

## 📦 Requirements
flask==3.0.0
torch
torchvision
segmentation-models-pytorch==0.3.3
albumentations
opencv-python
numpy
Pillow
---

## 🗄️ Dataset

**DeepCrack Dataset**
- 300 training images with pixel-level crack masks
- 237 test images with ground truth masks
- Source: [github.com/yhlleo/DeepCrack](https://github.com/yhlleo/DeepCrack)

---

## 🧪 Training Pipeline
Raw Images + Masks
↓
Augmentation (flip, rotate, brightness, blur)
↓
U-Net with ResNet-34 encoder
↓
BCE + Dice Loss
↓
Adam Optimizer + ReduceLROnPlateau
↓
Best model saved automatically
↓
Evaluation on 237 test images
---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Deep Learning** | PyTorch |
| **Model Library** | segmentation-models-pytorch |
| **Augmentation** | Albumentations |
| **Web Backend** | Flask |
| **Frontend** | HTML + CSS + JavaScript |
| **Training** | Google Colab T4 GPU |
| **Version Control** | GitHub |

---

## 👤 Author

**Ompreet Mohapatra**

Built as part of AI Internship Project.

---

## 📄 License

This project is licensed under the MIT License.
