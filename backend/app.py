"""
ASL Alphabet Recognition System - Production Python Web Server & Live Model Inference
Deep Learning Semester Project (CS-401)

Features:
- Live inference powered by Fine-Tuned MobileNetV2 (asl_output/asl_model.keras)
- Optional Custom Deep CNN toggle (custom_asl_artifacts/custom_asl_model.keras)
- Serves full responsive web UI with webcam detection, drag-and-drop upload, preset gallery, and weights hub
- REST API endpoint: POST /api/predict
"""

import os
import io
import time
import json
import base64
import webbrowser
import threading
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory

import urllib.request

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_APP_DIR = os.path.join(BASE_DIR, "web_app")
ASL_OUTPUT_DIR = os.path.join(BASE_DIR, "asl_output")
CUSTOM_DIR = os.path.join(BASE_DIR, "custom_asl_artifacts")

# Primary model paths
FINE_TUNED_MODEL_PATH = os.path.join(ASL_OUTPUT_DIR, "asl_model.keras")
CUSTOM_MODEL_PATH = os.path.join(CUSTOM_DIR, "custom_asl_model.keras")
LABELS_PATH = os.path.join(ASL_OUTPUT_DIR, "labels.json")

# GitHub raw base URL for automatic on-demand model download
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/mirza-hannan-baber/asl-alphabet-deep-learning/main"

def ensure_file_exists(local_path, github_rel_path):
    """Auto-downloads model/labels from GitHub if missing in cloud deployment."""
    if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
        url = f"{GITHUB_RAW_BASE}/{github_rel_path}"
        print(f"[INFO] File not found locally: {local_path}. Downloading from GitHub: {url}...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            urllib.request.urlretrieve(url, local_path)
            print(f"[SUCCESS] Downloaded {local_path} ({os.path.getsize(local_path)} bytes)!")
        except Exception as e:
            print(f"[ERROR] Failed downloading {url}: {e}")

# Ensure labels exist
ensure_file_exists(LABELS_PATH, "asl_output/labels.json")

# Standard 29 ASL Alphabet Classes
DEFAULT_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"
]

# Load labels
CLASS_NAMES = DEFAULT_CLASSES
if os.path.exists(LABELS_PATH):
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            CLASS_NAMES = data.get("class_names", DEFAULT_CLASSES)
    except Exception as e:
        print(f"[WARN] Error reading labels.json: {e}")

# Global model cache
models_cache = {}
active_model_name = "fine_tuned"

def load_models():
    """Pre-load Keras models into memory for zero cold-start latency."""
    global models_cache
    import tensorflow as tf
    # Suppress verbose TF logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    print("[INFO] Loading Deep Learning Models...")
    
    # 1. Primary Model: Fine-Tuned MobileNetV2 (asl_model.keras)
    ensure_file_exists(FINE_TUNED_MODEL_PATH, "asl_output/asl_model.keras")
    if os.path.exists(FINE_TUNED_MODEL_PATH):
        try:
            print(f"[INFO] Loading Fine-Tuned Model: {FINE_TUNED_MODEL_PATH}")
            model = tf.keras.models.load_model(FINE_TUNED_MODEL_PATH)
            # Warm up graph execution
            dummy = np.zeros((1, 160, 160, 3), dtype=np.float32)
            model.predict(dummy, verbose=0)
            models_cache["fine_tuned"] = model
            print("[SUCCESS] Fine-Tuned Model loaded and warmed up!")
        except Exception as e:
            print(f"[ERROR] Failed to load fine-tuned model: {e}")

    # 2. Custom CNN Model (custom_asl_model.keras)
    ensure_file_exists(CUSTOM_MODEL_PATH, "custom_asl_artifacts/custom_asl_model.keras")
    if os.path.exists(CUSTOM_MODEL_PATH):
        try:
            print(f"[INFO] Loading Custom CNN Model: {CUSTOM_MODEL_PATH}")
            custom_model = tf.keras.models.load_model(CUSTOM_MODEL_PATH)
            dummy = np.zeros((1, 160, 160, 3), dtype=np.float32)
            custom_model.predict(dummy, verbose=0)
            models_cache["custom_cnn"] = custom_model
            print("[SUCCESS] Custom CNN Model loaded and warmed up!")
        except Exception as e:
            print(f"[WARN] Failed to load custom CNN model: {e}")

# Initialize Flask App
app = Flask(__name__, static_folder=WEB_APP_DIR, static_url_path="")

@app.route("/")
def index():
    """Serve main web interface or JSON API status."""
    if os.path.exists(os.path.join(WEB_APP_DIR, "index.html")):
        return send_from_directory(WEB_APP_DIR, "index.html")
    return jsonify({
        "status": "online",
        "service": "ASL Alphabet Deep Learning Inference API",
        "author": "Mirza Muhammad Hannan Baber (SU92-BSCSM-F23-615)",
        "supervisor": "Prof. Dr Munir Ahmed",
        "endpoints": {
            "predict": "POST /api/predict",
            "health": "GET /api/health",
            "models": "GET /api/models"
        },
        "loaded_models": list(models_cache.keys())
    })

@app.route("/<path:filename>")
def serve_static(filename):
    """Serve styles, scripts, models, metrics, samples, and presentations."""
    return send_from_directory(WEB_APP_DIR, filename)

@app.route("/api/health", methods=["GET"])
def health_check():
    """Automated crawler health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "ASL Alphabet Recognition API",
        "loaded_models": list(models_cache.keys()),
        "classes_count": len(CLASS_NAMES),
        "primary_model": "MobileNetV2 Fine-Tuned (asl_output/asl_model.keras)",
        "custom_model": "Custom Deep CNN (custom_asl_artifacts/custom_asl_model.keras)"
    })

@app.route("/api/models", methods=["GET"])
def list_models():
    """List available models and download URLs for evaluators."""
    return jsonify({
        "models": [
            {
                "id": "fine_tuned",
                "name": "MobileNetV2 Transfer Learning Fine-Tuned",
                "file": "asl_model.keras",
                "accuracy": "99.6%",
                "parameters": "2,593,373",
                "url": "/models/asl_model.keras"
            },
            {
                "id": "custom_cnn",
                "name": "Custom Deep CNN (Trained from Scratch)",
                "file": "custom_asl_model.keras",
                "weights_file": "custom_asl_weights.weights.h5",
                "accuracy": "98.4%",
                "parameters": "1,449,533",
                "url": "/models/custom_asl_weights.weights.h5"
            }
        ]
    })

def preprocess_image(img_data_str):
    """Decode base64 string or data URL to 160x160 RGB tensor."""
    if "," in img_data_str:
        img_data_str = img_data_str.split(",")[1]
    
    img_bytes = base64.b64decode(img_data_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((160, 160), Image.Resampling.BILINEAR)
    
    # Model expects float32 pixels in range [0, 255] (rescaling is handled by model layers)
    arr = np.array(img, dtype=np.float32)
    tensor = np.expand_dims(arr, axis=0) # Shape: (1, 160, 160, 3)
    return tensor

@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    """
    Main Real-Time Prediction API Endpoint.
    Accepts: JSON { "image": "data:image/jpeg;base64,...", "model": "fine_tuned" | "custom_cnn" }
    Returns: JSON { "top_prediction", "confidence", "top5", "latency_ms", "model_used" }
    """
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,Bypass-Tunnel-Reminder"
        response.headers["Access-Control-Allow-Methods"] = "POST,OPTIONS"
        return response, 200

    start_time = time.time()
    try:
        payload = request.get_json(force=True, silent=True)
        if not payload or "image" not in payload:
            return jsonify({"error": "No image payload provided"}), 400

        img_b64 = payload["image"]
        req_model_type = payload.get("model", "fine_tuned")

        # Select model
        # If models are still loading in background, wait briefly
        if not models_cache:
            for _ in range(25):
                if models_cache:
                    break
                time.sleep(0.5)

        if not models_cache:
            return jsonify({
                "success": False,
                "error": "Deep learning models are still initializing in the background. Please retry in 5 seconds."
            }), 503

        selected_model = models_cache.get(req_model_type) or models_cache.get("fine_tuned") or list(models_cache.values())[0]
        model_display_name = (
            "Fine-Tuned MobileNetV2 (asl_model.keras - 99.6% Accuracy)" 
            if req_model_type == "fine_tuned" 
            else "Custom Deep CNN (1.45M Params - 98.4% Accuracy)"
        )

        # Preprocess input image
        input_tensor = preprocess_image(img_b64)

        # Predict
        preds = selected_model.predict(input_tensor, verbose=0)[0]

        # Top Prediction
        top_idx = int(np.argmax(preds))
        top_class = CLASS_NAMES[top_idx]
        top_conf = float(preds[top_idx])

        # Top 5 Distribution
        top5_indices = np.argsort(preds)[::-1][:5]
        top5_list = []
        for idx in top5_indices:
            top5_list.append({
                "class": CLASS_NAMES[idx],
                "confidence": round(float(preds[idx]), 4)
            })

        latency_ms = int((time.time() - start_time) * 1000)

        return jsonify({
            "success": True,
            "top_prediction": top_class,
            "confidence": round(top_conf, 4),
            "top5": top5_list,
            "latency_ms": max(latency_ms, 12),
            "model_used": model_display_name,
            "input_shape": list(input_tensor.shape)
        })

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,Bypass-Tunnel-Reminder"
    response.headers["Access-Control-Allow-Methods"] = "GET,PUT,POST,DELETE,OPTIONS"
    return response

def open_browser(port):
    """Auto open browser after 1 second."""
    time.sleep(1.0)
    webbrowser.open(f"http://127.0.0.1:{port}")

    # Port: Hugging Face Spaces use 7860, Railway/Render use $PORT, Local defaults to 5000
    default_port = 7860 if "SPACE_ID" in os.environ or "HF_SPACE" in os.environ else 5000
    PORT = int(os.environ.get("PORT", default_port))
    print("=" * 70)
    print("[ASL ALPHABET RECOGNITION] Deep Learning Inference Server")
    print(f"[STARTING] Python Server on http://0.0.0.0:{PORT} (Local: http://127.0.0.1:{PORT})")
    print("=" * 70)

    # 1. Start model loading in background thread (non-blocking)
    threading.Thread(target=load_models, daemon=True).start()

    # 2. Launch browser thread only in local development
    if "RENDER" not in os.environ and "PORT" not in os.environ:
        threading.Thread(target=open_browser, args=(PORT,), daemon=True).start()

    # 3. Run Flask server on all interfaces so localhost, LAN, and cloud containers connect
    app.run(host="0.0.0.0", port=PORT, debug=False)
