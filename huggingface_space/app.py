"""
ASL Alphabet Recognition - Hugging Face Space Production Backend
Student: Mirza Muhammad Hannan Baber (SU92-BSCSM-F23-615) - BSCS-7A
Supervisor: Prof. Dr Munir Ahmed (Department of Computer Science)

Features:
- Dual Engine: Fine-Tuned MobileNetV2 (99.6%) & Custom Deep CNN (98.4%)
- Auto-downloads model weights from GitHub (solves Hugging Face 25MB file upload limit)
- Dual Interface:
  1. Interactive Gradio UI at "/"
  2. High-Speed REST API for Vercel at "/api/predict" and "/api/health"
"""

import os
import io
import time
import json
import base64
import urllib.request
import numpy as np
from PIL import Image
import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

# Suppress verbose TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FINE_TUNED_PATH = os.path.join(MODELS_DIR, "asl_model.keras")
CUSTOM_PATH = os.path.join(MODELS_DIR, "custom_asl_model.keras")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/mirza-hannan-baber/asl-alphabet-deep-learning/main"

DEFAULT_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"
]

def ensure_file(local_path, rel_url):
    """Auto-downloads model weights if missing (avoids Hugging Face browser upload limit)."""
    if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
        url = f"{GITHUB_RAW_BASE}/{rel_url}"
        print(f"[INFO] Auto-downloading {rel_url} from GitHub...")
        urllib.request.urlretrieve(url, local_path)
        print(f"[SUCCESS] Downloaded {local_path} ({os.path.getsize(local_path)} bytes)")

# Ensure labels
ensure_file(LABELS_PATH, "asl_output/labels.json")
CLASS_NAMES = DEFAULT_CLASSES
if os.path.exists(LABELS_PATH):
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            CLASS_NAMES = json.load(f).get("class_names", DEFAULT_CLASSES)
    except Exception as e:
        print(f"[WARN] Error loading labels.json: {e}")

# Global model cache
models_cache = {}

def load_models():
    """Load both Keras models into memory."""
    global models_cache
    print("[INFO] Pre-loading Deep Learning models...")

    # 1. Fine-Tuned MobileNetV2
    ensure_file(FINE_TUNED_PATH, "asl_output/asl_model.keras")
    if os.path.exists(FINE_TUNED_PATH):
        try:
            print("[INFO] Loading MobileNetV2...")
            m = tf.keras.models.load_model(FINE_TUNED_PATH)
            m.predict(np.zeros((1, 160, 160, 3), dtype=np.float32), verbose=0)
            models_cache["fine_tuned"] = m
            print("[SUCCESS] MobileNetV2 (99.6%) loaded!")
        except Exception as e:
            print(f"[ERROR] Loading fine_tuned: {e}")

    # 2. Custom CNN
    ensure_file(CUSTOM_PATH, "custom_asl_artifacts/custom_asl_model.keras")
    if os.path.exists(CUSTOM_PATH):
        try:
            print("[INFO] Loading Custom CNN...")
            c = tf.keras.models.load_model(CUSTOM_PATH)
            c.predict(np.zeros((1, 160, 160, 3), dtype=np.float32), verbose=0)
            models_cache["custom_cnn"] = c
            print("[SUCCESS] Custom CNN (98.4%) loaded!")
        except Exception as e:
            print(f"[ERROR] Loading custom_cnn: {e}")

load_models()

def predict_tensor(img_pil, model_choice="fine_tuned"):
    """Runs inference on a PIL image."""
    start_time = time.time()
    img_resized = img_pil.convert("RGB").resize((160, 160), Image.Resampling.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32)
    tensor = np.expand_dims(arr, axis=0)

    selected_model = models_cache.get(model_choice) or models_cache.get("fine_tuned") or list(models_cache.values())[0]
    preds = selected_model.predict(tensor, verbose=0)[0]

    top_idx = int(np.argmax(preds))
    top_class = CLASS_NAMES[top_idx]
    top_conf = float(preds[top_idx])

    top5_indices = np.argsort(preds)[::-1][:5]
    top5_list = [{"class": CLASS_NAMES[i], "confidence": round(float(preds[i]), 4)} for i in top5_indices]
    latency = max(int((time.time() - start_time) * 1000), 12)

    return top_class, top_conf, top5_list, latency

# ------------------------------------------------------------------------------
# 1. FastAPI REST API Engine (For Vercel Web App)
# ------------------------------------------------------------------------------
api_app = FastAPI(title="ASL Alphabet Deep Learning Inference API")

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api_app.get("/api/health")
def health_endpoint():
    return {
        "status": "healthy",
        "service": "ASL Alphabet Recognition API",
        "author": "Mirza Muhammad Hannan Baber (SU92-BSCSM-F23-615)",
        "supervisor": "Prof. Dr Munir Ahmed",
        "loaded_models": list(models_cache.keys()),
        "classes_count": len(CLASS_NAMES)
    }

@api_app.post("/api/predict")
async def predict_endpoint(request: Request):
    try:
        data = await request.json()
        if not data or "image" not in data:
            return JSONResponse(status_code=400, content={"error": "Missing image in request body"})

        img_str = data["image"]
        if "," in img_str:
            img_str = img_str.split(",")[1]
        img_bytes = base64.b64decode(img_str)
        img = Image.open(io.BytesIO(img_bytes))

        req_model = data.get("model", "fine_tuned")
        top_class, top_conf, top5, latency = predict_tensor(img, req_model)

        engine_name = (
            "Fine-Tuned MobileNetV2 (asl_model.keras - 99.6% Accuracy)"
            if req_model == "fine_tuned"
            else "Custom Deep CNN (1.45M Params - 98.4% Accuracy)"
        )

        return JSONResponse(content={
            "success": True,
            "top_prediction": top_class,
            "confidence": round(top_conf, 4),
            "top5": top5,
            "latency_ms": latency,
            "model_used": engine_name
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

# ------------------------------------------------------------------------------
# 2. Gradio Web Interface (Interactive Testing on Hugging Face)
# ------------------------------------------------------------------------------
def gradio_fn(img, model_name):
    if img is None:
        return {}, "Please provide an image."
    key = "fine_tuned" if "MobileNet" in model_name else "custom_cnn"
    top_c, top_conf, top5, latency = predict_tensor(img, key)
    conf_dict = {item["class"]: item["confidence"] for item in top5}
    info = f"Top Prediction: {top_c} ({top_conf*100:.1f}%)\nLatency: {latency} ms\nModel: {key}"
    return conf_dict, info

demo = gr.Interface(
    fn=gradio_fn,
    inputs=[
        gr.Image(type="pil", label="Upload Hand Gesture Image or Webcam"),
        gr.Radio(["MobileNetV2 Fine-Tuned (99.6%)", "Custom Deep CNN (98.4%)"], value="MobileNetV2 Fine-Tuned (99.6%)", label="Model Architecture")
    ],
    outputs=[
        gr.Label(num_top_classes=5, label="Top 5 Confidence Distribution"),
        gr.Textbox(label="Inference Details")
    ],
    title="🤟 American Sign Language (ASL) Alphabet Recognition",
    description="Dual Deep Learning Models trained on 87,000 images across 29 classes.\nDeveloped by **Mirza Muhammad Hannan Baber** (`SU92-BSCSM-F23-615`, `BSCS-7A`) under supervision of **Prof. Dr Munir Ahmed**."
)

# Mount Gradio onto FastAPI root
app = gr.mount_gradio_app(api_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    PORT = int(os.environ.get("PORT", 7860))
    print(f"[INFO] Server starting on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
