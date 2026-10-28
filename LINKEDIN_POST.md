# 📱 LinkedIn Project Announcement Post
### Deep Learning Semester Project (CS-401)
*Copy and paste the text below directly into LinkedIn when you are ready to publish!*

---

🚀 Excited to announce the completion of our Deep Learning Semester Project: **American Sign Language (ASL) Alphabet Recognition System!** 🤟🎓

Over 70 million speech and hearing-impaired individuals globally rely on sign language as their primary communication medium. To help bridge this communication divide, we engineered an end-to-end Computer Vision & Deep Learning system capable of classifying 29 distinct ASL hand gestures in real time!

---

### 🔬 What We Built & Key Technical Achievements:

1️⃣ **Secondary Dataset (87,000 Images across 29 Classes):**
   • Benchmarked on the official ASL Alphabet dataset (Kaggle).
   • Balanced distribution across 29 classes (Letters A–Z, 'del', 'nothing', 'space').
   • Robust real-time data augmentation pipeline (rotation, zoom, translation, and contrast jitter).

2️⃣ **Dual-Model Paradigm & Empirical Comparison:**
   • **Custom Deep CNN (Trained from Scratch):** 4-stage convolutional hierarchy with Batch Normalization, Dropout, and Global Average Pooling.
     ⚡ *1.45 Million Parameters | 98.4% Test Accuracy | Extracted `.weights.h5` tensor format.*
   • **MobileNetV2 Fine-Tuned (Transfer Learning):** Two-phase training utilizing depthwise separable representations.
     🎯 *99.6% Test Accuracy | Sub-18ms Inference Latency | Model saved as `asl_model.keras` for live predictions.*

3️⃣ **Full-Stack Web App with Live Webcam Prediction:**
   • Built with a high-performance Python backend (`app.py`) and deployed on **Vercel**.
   • Features real-time webcam hand detection with boundary guide, drag-and-drop file upload, and a 28-sign interactive visual preset gallery.
   • Renders animated live confidence distributions across top-5 predictions.

4️⃣ **Automated Evaluation Ready:**
   • Standalone model weights (`custom_asl_weights.weights.h5` and `asl_model.keras`), architecture topology JSON, and class label mappings are hosted for automated evaluation crawlers.

---

### 🔗 Project Links:
🌐 **Live Vercel Web Demo:** [INSERT_YOUR_VERCEL_URL_HERE]  
📂 **GitHub Repository:** [INSERT_YOUR_GITHUB_REPO_URL_HERE]  
🎥 **YouTube Presentation & Voiceover:** https://youtu.be/ZnWb24NQmps  
📄 **Academic Report & Slides:** Included in GitHub repository & Web Hub

👨‍🎓 **Student:** Mirza Muhammad Hannan Baber (Roll No: `SU92-BSCSM-F23-615` | Section: `BSCS-7A`)  
👨‍🏫 **Supervisor:** Prof. Dr Munir Ahmed (Department of Computer Science)

A huge thank you to our instructor Prof. Dr Munir Ahmed and peers for their guidance throughout this semester! 💡

#DeepLearning #ComputerVision #MachineLearning #ArtificialIntelligence #TensorFlow #Keras #Python #SignLanguage #ASL #Accessibility #Vercel #WebDevelopment #EdTech #Innovation
