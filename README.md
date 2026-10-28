# 🤟 American Sign Language (ASL) Alphabet Recognition System
### Deep Learning Semester Project (CS-401) • Automated Evaluation Ready

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B%20%7C%202.20-orange.svg)](https://tensorflow.org/)
[![Vercel Deployment](https://img.shields.io/badge/Vercel-Deployed-black.svg)](https://vercel.com/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.6%25-brightgreen.svg)]()

---

### 👨‍🎓 Student & Course Metadata
* **Student Name:** Mirza Muhammad Hannan Baber
* **Roll Number:** SU92-BSCSM-F23-615
* **Class / Section:** BSCS-7A (Computer Science)
* **Project Supervisor:** Prof. Dr Munir Ahmed
* **Course:** CS-401 Deep Learning (Semester Project)

---

## 📌 Project Overview
This project presents an end-to-end Deep Learning system for classifying **29 American Sign Language (ASL) hand gestures** (Letters `A–Z`, `del`, `nothing`, and `space`) using a secondary dataset of **87,000 images**.

We conducted a dual-architecture study:
1. **Custom Deep CNN (Trained from Scratch):** 4-stage convolutional hierarchy with Batch Normalization, Dropout, and Global Average Pooling (1.45M parameters), achieving **98.4% test accuracy**. Standalone weights saved as `custom_asl_weights.weights.h5`.
2. **MobileNetV2 Fine-Tuned (Transfer Learning):** Two-phase transfer learning on ImageNet backbone, achieving **99.6% test accuracy**. Saved as `asl_output/asl_model.keras` and integrated for real-time live predictions.

---

## 🚀 Quick Start (Run Locally)

### 1. Install Dependencies
```bash
pip install tensorflow flask pillow numpy python-pptx
```

### 2. Run Python Web Application
```bash
python app.py
```
*The server will start on `http://127.0.0.1:5000` and automatically open your default browser!*

### 3. Run Automated Evaluation Script
To verify both models on isolated test images:
```bash
python evaluate_models.py
```

---

## 📂 Repository Structure & Model Artifacts Hub

```
Project/
├── app.py                               # Production Python Flask web server & prediction API
├── evaluate_models.py                   # Automated evaluation script for testing accuracy
├── generate_presentation.py             # Script that generated the 10-slide PPTX
├── ASL_Deep_Learning_Presentation.pptx  # 10-Slide presentation deck (Dark mode)
├── VOICEOVER_SCRIPT.md                  # Complete word-by-word YouTube voiceover script
├── PROJECT_REPORT.md                    # Academic semester project paper
├── LINKEDIN_POST.md                     # Ready-to-publish LinkedIn post copy
│
├── asl_output/                          # Fine-Tuned Model Hub (Primary Predictor)
│   ├── asl_model.keras                  # Fine-Tuned MobileNetV2 (99.6% Acc, 26.9 MB)
│   ├── asl_model.tflite                 # Quantized mobile model (2.86 MB)
│   ├── labels.json                      # Class names and input specifications
│   ├── training_curves.png              # Convergence metrics plot
│   └── confusion_matrix.png             # 29x29 test confusion matrix
│
├── custom_asl_artifacts/                # Custom Scratch CNN Model Hub
│   ├── custom_asl_weights.weights.h5    # Standalone extracted weights tensor (5.91 MB)
│   ├── custom_asl_model.keras           # Full custom model binary (5.94 MB)
│   ├── custom_model_architecture.json   # Full layer topology JSON (71 KB)
│   ├── labels.json                      # Class label mapping
│   ├── custom_training_curves.png       # Training accuracy/loss curves
│   └── custom_confusion_matrix.png      # Custom CNN confusion matrix
│
├── Notebook/                            # Google Colab Training Notebooks
│   ├── ASL_Final_Colab_Training.ipynb   # MobileNetV2 training notebook
│   └── ASL_Custom_CNN_Colab_Training.ipynb # Custom CNN scratch training notebook
│
└── web_app/                             # Vercel Deployment Web Application
    ├── index.html                       # Responsive interactive frontend
    ├── style.css                        # Modern dark-mode styling
    ├── script.js                        # Frontend logic, webcam & API caller
    ├── vercel.json                      # Vercel deployment configuration
    ├── ASL_Deep_Learning_Presentation.pptx # Downloadable slides deck
    ├── models/                          # Crawler-downloadable models directory
    ├── metrics/                         # Evaluation curves & confusion matrices
    └── samples/                         # 28 test gesture images (A_test.jpg, etc.)
```

---

## 📋 Semester Project Requirements Checklist

| Requirement | Description | Status | Reference File / Link |
| :--- | :--- | :---: | :--- |
| **1️⃣ Dataset Collection** | 87,000 images, 29 classes from Kaggle | ✅ Complete | [grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) |
| **2️⃣ Model Training & Weights** | Custom CNN + MobileNetV2 weights extracted | ✅ Complete | `custom_asl_weights.weights.h5`, `asl_model.keras` |
| **3️⃣ GitHub Repository** | Complete source code, models, and test scripts | ✅ Complete | Dedicated GitHub repository structure |
| **4️⃣ Web Deployment (Vercel)** | Interactive live webcam app deployed | ✅ Complete | `web_app/`, `app.py`, `vercel.json` |
| **5️⃣ Project Report / Paper** | Academic AI-assisted research report | ✅ Complete | `PROJECT_REPORT.md` |
| **6️⃣ Presentation & Voiceover** | 10-slide PPTX & video walkthrough | ✅ Complete | [YouTube Video](https://youtu.be/ZnWb24NQmps) & `ASL_Deep_Learning_Presentation.pptx` |
| **7️⃣ Social Media Sharing** | High-impact LinkedIn post text | ✅ Complete | `LINKEDIN_POST.md` |

---

## 🤖 Automated Evaluation Crawler Direct Links
Automated grading scripts can curl or download model artifacts directly:
- **Fine-Tuned Model:** `/models/asl_model.keras`
- **Custom Weights:** `/models/custom_asl_weights.weights.h5`
- **Custom Model:** `/models/custom_asl_model.keras`
- **Architecture Config:** `/models/custom_model_architecture.json`
- **Class Labels:** `/models/labels.json`
- **Test Samples:** `/samples/{LETTER}_test.jpg`
