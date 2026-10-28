# Comprehensive Project Report: Deep Learning-Based American Sign Language (ASL) Alphabet Recognition
**Course:** CS-401 Deep Learning | Semester Project Report  
**Student Name:** Mirza Muhammad Hannan Baber  
**Roll Number:** SU92-BSCSM-F23-615  
**Class / Section:** BSCS-7A (Computer Science)  
**Project Supervisor:** Prof. Dr Munir Ahmed  
**Artifacts:** Custom CNN Weights (`custom_asl_weights.weights.h5`), Fine-Tuned Model (`asl_model.keras`), Web App (`web_app/`)  

---

## 1. Abstract
American Sign Language (ASL) is the primary mode of manual communication for deaf and hard-of-hearing populations worldwide. Automated translation of static hand gestures via computer vision bridges the communication barrier between hearing-impaired communities and non-signers. This research project presents the architectural design, training methodology, weight serialization, and full-stack web deployment of a dual deep learning paradigm:
1. A **Custom Deep Convolutional Neural Network (CNN)** engineered from scratch featuring 4 sequential convolutional stages with Batch Normalization, Dropout, and Global Average Pooling (1,449,533 parameters), achieving **98.4% test accuracy**.
2. A **MobileNetV2 Transfer Learning** architecture fine-tuned via a two-phase protocol on 87,000 images across 29 classes, achieving **99.6% test accuracy**.

Trained model weights were extracted into standalone `.weights.h5` tensor format and full compiled `.keras` formats. A production-ready web application supporting real-time webcam inference and drag-and-drop prediction was deployed on Vercel with a Python backend (`app.py`). All weights, topologies, and class label mappings are structured to be crawler-accessible for automated academic grading scripts.

---

## 2. Introduction & Problem Statement
### 2.1 Social Importance
More than 70 million individuals globally use sign language as their primary mother tongue. Due to a severe shortage of certified human interpreters, hearing-impaired individuals frequently encounter systemic barriers in healthcare consultations, academic classrooms, legal proceedings, and daily commerce. Building accessible, zero-cost computer vision tools that run on standard commodity webcams and smartphones provides immediate, equitable translation.

### 2.2 Technical Challenges in Gesture Classification
Hand gesture classification involves complex computer vision constraints:
- **High Intra-Class Similarity:** Specific letter combinations (e.g., 'M', 'N', and 'T') exhibit subtle finger flexion and thumb positioning differences while sharing virtually identical palm silhouettes.
- **Inter-Subject Hand Morphology:** Variation in skin tone, hand size, finger length, and lighting conditions can cause severe distribution shifts.
- **Low-Latency Edge Constraint:** Real-time translation requires frame rates of at least 30 FPS (<33 ms per frame), mandating lightweight architectures without redundant parameters.

---

## 3. Dataset Collection & Preprocessing (Requirement 1)
### 3.1 Secondary Benchmark Dataset
We utilized the standardized **ASL Alphabet Dataset** (Grassknoted, Kaggle):
- **Total Training Instances:** 87,000 images.
- **Class Distribution:** 29 balanced classes (26 alphabets `A–Z`, plus `del`, `nothing`, and `space`), with exactly 3,000 images per class (zero class imbalance).
- **Original Dimensions:** 200 × 200 pixels, 3-channel RGB.
- **Input Preprocessing:** Scaled to 160 × 160 pixels with pixel intensities rescaled to $[0, 1]$ float32 tensors.

### 3.2 Data Augmentation Pipeline
To prevent spatial overfitting and enforce invariance against physical camera noise, we applied real-time geometric and photometric augmentations:
$$\tilde{x} = \mathcal{T}_{\text{contrast}}\Big(\mathcal{T}_{\text{bright}}\Big(\mathcal{T}_{\text{zoom}}\big(\mathcal{T}_{\text{trans}}(\mathcal{T}_{\text{rot}}(x))\big)\Big)\Big)$$
- Random Rotation: $\pm 8\%$
- Horizontal & Vertical Translation: $\pm 10\%$
- Random Zoom: $\pm 12\%$
- Brightness & Contrast Jitter: $\pm 15\%$

---

## 4. Model Architectures & Mathematical Formulation (Requirement 2)
### 4.1 Custom Deep CNN (Trained from Scratch)
The custom architecture consists of 4 convolutional feature extraction blocks followed by Global Average Pooling and a dense classification head:

1. **Stage 1 (Low-level edges & corners):**
   - $\text{Conv2D}(32, 3\times3) \to \text{BatchNorm} \to \text{ReLU} \to \text{Conv2D}(32, 3\times3) \to \text{BatchNorm} \to \text{ReLU} \to \text{MaxPool}(2\times2) \to \text{Dropout}(0.20)$
   - Feature map: $(80, 80, 32)$

2. **Stage 2 (Texture & contour primitives):**
   - $\text{Conv2D}(64, 3\times3) \to \text{BatchNorm} \to \text{ReLU} \to \text{Conv2D}(64, 3\times3) \to \text{BatchNorm} \to \text{ReLU} \to \text{MaxPool}(2\times2) \to \text{Dropout}(0.25)$
   - Feature map: $(40, 40, 64)$

3. **Stage 3 (Finger segments & joint patterns):**
   - $\text{Conv2D}(128, 3\times3) \to \text{BatchNorm} \to \text{ReLU} \to \text{Conv2D}(128, 3\times3) \to \text{BatchNorm} \to \text{ReLU} \to \text{MaxPool}(2\times2) \to \text{Dropout}(0.30)$
   - Feature map: $(20, 20, 128)$

4. **Stage 4 (High-level spatial hand configurations):**
   - $\text{Conv2D}(256, 3\times3) \to \text{BatchNorm} \to \text{ReLU} \to \text{Conv2D}(256, 3\times3) \to \text{BatchNorm} \to \text{ReLU} \to \text{MaxPool}(2\times2) \to \text{Dropout}(0.35)$
   - Feature map: $(10, 10, 256)$

5. **Classification Head:**
   - **Global Average Pooling 2D:**
     $$\text{GAP}(x_c) = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_c(i,j)$$
     Compresses $(10, 10, 256)$ into a 256-dimensional vector, replacing 25,600 Flatten parameters and eliminating parameter explosion.
   - $\text{Dense}(512, \text{ReLU}) \to \text{BatchNorm} \to \text{Dropout}(0.40)$
   - $\text{Dense}(256, \text{ReLU}) \to \text{BatchNorm} \to \text{Dropout}(0.30)$
   - **Softmax Output Layer (29 classes):**
     $$\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{29} e^{z_j}}$$

Total Trainable Parameters: **1,449,533** (1.45M).

### 4.2 MobileNetV2 Fine-Tuned (Transfer Learning)
MobileNetV2 employs inverted residual blocks with linear bottlenecks and depthwise separable convolutions:
- **Phase 1 (Feature Extractor Frozen):** Base weights frozen from ImageNet. Classification head trained for 8 epochs with Adam ($\text{lr} = 10^{-3}$).
- **Phase 2 (Fine-Tuning):** Top 40 convolutional layers unfrozen. Trained for 12 epochs with a conservative learning rate ($\text{lr} = 3 \times 10^{-5}$) and label smoothing ($\alpha = 0.10$).
- Output file: `asl_output/asl_model.keras` (26.9 MB, 2.59M parameters).

---

## 5. Training Optimization & Hyperparameters
- **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-7}$).
- **Loss Function:** Categorical Cross-Entropy with Label Smoothing ($\alpha = 0.05$ for Custom CNN, $\alpha = 0.10$ for MobileNetV2):
  $$\mathcal{L}_{\text{LS}} = -(1 - \alpha) \log(p_y) - \frac{\alpha}{K} \sum_{k=1}^{K} \log(p_k)$$
- **Learning Rate Scheduler:** `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)`.
- **Early Stopping:** `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`.

---

## 6. Model Weights Extraction & Verification (Requirement 2 & 3)
In strict compliance with the automated evaluation guidelines:
1. Standalone raw weight tensors were serialized via `model.save_weights()` into `custom_asl_weights.weights.h5` (5.91 MB).
2. The layer topology was exported into `custom_model_architecture.json` (71 KB).
3. Class labels and input tensor requirements were exported into `labels.json`.
4. **Verification Protocol:** An independent Python unit instantiated a clean model graph without weights, reloaded `custom_asl_weights.weights.h5`, and confirmed numerical equivalence ($\Delta = 0.000000$).
5. **Keras 3 Compatibility:** Both `.keras` files were inspected and sanitized to ensure clean deserialization across any standard Keras 3 / TensorFlow 2.x environment without `quantization_config` errors.

---

## 7. Experimental Results & Discussion
| Metric | Custom Deep CNN (Scratch) | MobileNetV2 (Fine-Tuned) |
| :--- | :---: | :---: |
| **Total Parameters** | 1,449,533 (1.45M) | 2,593,373 (2.59M) |
| **Model Weights File** | `custom_asl_weights.weights.h5` | `asl_model.keras` |
| **Validation Accuracy** | 98.4% | 99.6% |
| **Macro F1-Score** | 0.985 | 0.996 |
| **Inference Latency (CPU)** | ~14 ms | ~18 ms |
| **28 Isolated Test Images** | 100% Top-1 Pass | 100% Top-1 Pass |

The confusion matrix shows intense diagonal concentration across all 29 classes, indicating exceptional class boundary discrimination even on visually similar pairs like 'M', 'N', and 'T'.

---

## 8. Web Deployment & Automated Evaluation (Requirement 4 & Crawler)
### 8.1 Python Backend & Vercel
- **Python Server:** `app.py` serves the web interface and exposes `POST /api/predict`.
- **Live Inference:** Preloads `asl_output/asl_model.keras` into RAM for instantaneous sub-20ms inference.
- **Frontend Features:**
  1. Live Webcam gesture detection with hand placement boundary guide.
  2. Drag-and-drop file upload.
  3. Interactive 28-letter preset thumbnail gallery.
  4. Real-time top-5 probability distribution bar chart.
  5. Model selector dropdown to toggle between Fine-Tuned MobileNetV2 and Custom CNN.

### 8.2 Automated Grading Crawler Compliance
All model files and metadata are accessible at static URLs:
- `/models/asl_model.keras`
- `/models/custom_asl_weights.weights.h5`
- `/models/custom_asl_model.keras`
- `/models/custom_model_architecture.json`
- `/models/labels.json`
- `/ASL_Deep_Learning_Presentation.pptx`

---

## 9. Conclusion & Future Work
This project successfully achieved all 7 semester milestones. Both custom-architected and transfer-learning deep networks demonstrated near-flawless recognition across 29 American Sign Language classes. Future work will investigate temporal sequence modeling using Bi-LSTM and Vision Transformers for continuous sentence-level sign language video translation.

---
*(End of Report)*
