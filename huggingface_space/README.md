---
title: ASL Alphabet Deep Learning Inference API
emoji: 🤟
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.26.0
app_file: app.py
pinned: false
license: mit
---

# 🤟 American Sign Language (ASL) Alphabet Recognition Backend

Deep Learning Semester Project (CS-401)
- **Student & Presenter:** Mirza Muhammad Hannan Baber (`SU92-BSCSM-F23-615`) &bull; Section: `BSCS-7A`
- **Project Supervisor:** Prof. Dr Munir Ahmed &bull; Department of Computer Science
- **Frontend Web App (Vercel):** [asl-alphabet-deep-learning-webapp.vercel.app](https://asl-alphabet-deep-learning-webapp.vercel.app)
- **GitHub Repository:** [mirza-hannan-baber/asl-alphabet-deep-learning](https://github.com/mirza-hannan-baber/asl-alphabet-deep-learning)

---

### Features
1. **Automatic Weights Fetching:** Solves Hugging Face web upload limits (files over 25MB) by downloading `.keras` models directly on startup from GitHub.
2. **Dual Model Inference:** Fine-Tuned MobileNetV2 (99.6% Accuracy) and Custom Deep CNN (98.4% Accuracy).
3. **CORS-Enabled REST API:** Exposes `POST /api/predict` and `GET /api/health` directly for client applications and the live Vercel web app.
