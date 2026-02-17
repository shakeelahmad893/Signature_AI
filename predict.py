"""
predict.py
==========
Signature Forgery Prediction Script.

Compare a reference (genuine) signature against a test signature
and predict whether the test signature is GENUINE or FORGED.

Usage
-----
    python predict.py --genuine path/to/genuine.png --test path/to/test.png

    Or run without arguments to use interactive prompts.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

import sys
import argparse
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from model_architecture import build_siamese_model

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

IMG_SHAPE = (105, 105)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "my_signature_ai.weights.h5")
THRESHOLD = 0.5  # Above = genuine, Below = forged


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(image_path):
    """Load an image, convert to grayscale, resize, and normalise."""
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        sys.exit(1)

    img = cv2.resize(img, IMG_SHAPE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)   # (105, 105, 1)
    img = np.expand_dims(img, axis=0)    # (1, 105, 105, 1) — batch dim
    return img


def predict(model, genuine_path, test_path):
    """
    Compare two signatures and return prediction.

    Returns
    -------
    score : float
        Similarity score (0-1). Higher = more likely genuine.
    verdict : str
        'GENUINE' or 'FORGED'
    """
    img_genuine = load_and_preprocess(genuine_path)
    img_test = load_and_preprocess(test_path)

    score = model.predict([img_genuine, img_test], verbose=0)[0][0]

    verdict = "GENUINE ✅" if score >= THRESHOLD else "FORGED ❌"

    return float(score), verdict


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Signature Forgery Detector - Predict if a signature is genuine or forged."
    )
    parser.add_argument("--genuine", type=str, help="Path to the genuine (reference) signature image")
    parser.add_argument("--test", type=str, help="Path to the test signature image to verify")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold (default: 0.5). Higher = stricter.")

    args = parser.parse_args()

    # Interactive mode if no arguments provided
    if not args.genuine:
        args.genuine = input("Enter path to GENUINE (reference) signature: ").strip().strip('"')
    if not args.test:
        args.test = input("Enter path to TEST signature to verify: ").strip().strip('"')

    global THRESHOLD
    THRESHOLD = args.threshold

    # Load model
    print(f"\n{'='*60}")
    print("  Signature Forgery Detector — Prediction")
    print(f"{'='*60}")

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at: {MODEL_PATH}")
        print("        Please train the model first: python train_engine.py")
        sys.exit(1)

    # Rebuild architecture and load trained weights
    print(f"\n[INFO] Loading model ...")
    model = build_siamese_model(input_shape=(IMG_SHAPE[0], IMG_SHAPE[1], 1))
    model.load_weights(MODEL_PATH)
    print("[INFO] Model loaded successfully!")

    # Predict
    print(f"\n[INFO] Comparing signatures ...")
    print(f"       Genuine : {os.path.basename(args.genuine)}")
    print(f"       Test    : {os.path.basename(args.test)}")

    score, verdict = predict(model, args.genuine, args.test)

    print(f"\n{'─'*60}")
    print(f"  RESULT:  {verdict}")
    print(f"  Score :  {score:.4f}  (threshold: {THRESHOLD})")
    print(f"  {'Signatures MATCH' if score >= THRESHOLD else 'Signatures DO NOT match'}")
    print(f"{'─'*60}\n")

    return score, verdict


if __name__ == "__main__":
    main()
