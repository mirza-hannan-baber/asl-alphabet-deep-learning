"""
Automated Evaluation Script for ASL Alphabet Models
CS-401 Deep Learning Semester Project

Evaluates:
1. Fine-Tuned MobileNetV2 (asl_output/asl_model.keras)
2. Custom Deep CNN (custom_asl_artifacts/custom_asl_model.keras & custom_asl_weights.weights.h5)
Against 28 official test samples in asl_alphabet/asl_alphabet_test/asl_alphabet_test/
"""

import os
import glob
import json
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "asl_alphabet", "asl_alphabet_test", "asl_alphabet_test")
if not os.path.exists(TEST_DIR):
    TEST_DIR = os.path.join(BASE_DIR, "web_app", "samples")

LABELS_PATH = os.path.join(BASE_DIR, "asl_output", "labels.json")
with open(LABELS_PATH, "r") as f:
    CLASS_NAMES = json.load(f)["class_names"]

def evaluate_model(model_name, model_path):
    print("\n" + "=" * 70)
    print(f"[EVALUATING] {model_name}")
    print(f"[PATH] {model_path}")
    print("=" * 70)
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        return

    import tensorflow as tf
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"[SUCCESS] Model loaded successfully! Input: {model.input_shape}, Output: {model.output_shape}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "*_test.jpg")))
    if not test_files:
        print(f"[ERROR] No test images found in {TEST_DIR}")
        return

    correct = 0
    total = 0

    print(f"\nRunning inference on {len(test_files)} isolated test images...")
    print(f"{'Filename':<20} | {'True Sign':<10} | {'Predicted':<10} | {'Confidence':<12} | {'Result'}")
    print("-" * 65)

    for fpath in test_files:
        fname = os.path.basename(fpath)
        true_label = fname.split("_")[0].upper()
        if true_label == "SPACE":
            true_label = "space"

        img = Image.open(fpath).convert("RGB").resize((160, 160))
        arr = np.expand_dims(np.array(img, dtype=np.float32), 0)

        preds = model.predict(arr, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_label = CLASS_NAMES[pred_idx]
        conf = float(preds[pred_idx])

        is_match = (pred_label.upper() == true_label.upper())
        if is_match:
            correct += 1
        total += 1

        status = "PASS [OK]" if is_match else "FAIL"
        print(f"{fname:<20} | {true_label:<10} | {pred_label:<10} | {conf * 100:>6.2f}%     | {status}")

    acc = (correct / total) * 100 if total > 0 else 0
    print("-" * 65)
    print(f"[RESULT] Total: {total} | Correct: {correct} | Accuracy: {acc:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    # 1. Evaluate Primary Fine-Tuned Model
    ft_path = os.path.join(BASE_DIR, "asl_output", "asl_model.keras")
    evaluate_model("MobileNetV2 Fine-Tuned (Primary)", ft_path)

    # 2. Evaluate Custom Deep CNN Model
    custom_path = os.path.join(BASE_DIR, "custom_asl_artifacts", "custom_asl_model.keras")
    evaluate_model("Custom Deep CNN (Scratch)", custom_path)
