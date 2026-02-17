"""
train_engine.py
===============
Training pipeline for the Siamese Signature Forgery Detector.

This script:
  1. Loads genuine & forged signature images grouped by person identity.
  2. Creates person-aware pairs: same-person genuine (positive),
     same-person forgery (negative), different-person (negative).
  3. Trains the Siamese model with augmentation, EarlyStopping, and LR scheduling.
  4. Saves the trained weights to  models/my_signature_ai.weights.h5.

Usage
-----
    python train_engine.py

All execution is guarded by  ``if __name__ == "__main__"``  so that
importing this module elsewhere will NOT trigger training.
"""

import os
import random
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from model_architecture import build_siamese_model
import tensorflow as tf

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

IMG_SHAPE = (105, 105)           # H × W expected by the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENUINE_DIR = os.path.join(BASE_DIR, "data", "genuine")
FORGED_DIR  = os.path.join(BASE_DIR, "data", "forged")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "my_signature_ai.weights.h5")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "checkpoint.weights.h5")
EPOCHS = 50
BATCH_SIZE = 32


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def extract_person_id(filename):
    """
    Extract person ID from CEDAR filename.
    e.g. 'original_10_3.png' → 10, 'forgeries_5_12.png' → 5
    """
    parts = filename.replace(".png", "").replace(".jpg", "").split("_")
    # Format: original_{person}_{sample} or forgeries_{person}_{sample}
    if len(parts) >= 3:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def load_images_by_person(folder_path):
    """
    Read images from folder and group them by person ID.

    Returns
    -------
    dict[int, list[np.ndarray]]
        Dictionary mapping person ID → list of preprocessed images.
    """
    person_images = {}
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(valid_extensions):
            continue

        person_id = extract_person_id(filename)
        if person_id is None:
            continue

        filepath = os.path.join(folder_path, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Could not read: {filepath}")
            continue
        img = cv2.resize(img, IMG_SHAPE)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)  # (105, 105, 1)

        if person_id not in person_images:
            person_images[person_id] = []
        person_images[person_id].append(img)

    return person_images


def create_pairs(genuine_by_person, forged_by_person, max_pairs=12000):
    """
    Build PERSON-AWARE training pairs:

    * **Positive pairs** (label = 1): Two genuine signatures from the
      SAME person  →  these should MATCH.
    * **Negative Type A** (label = 0): Genuine + forged of the SAME
      person  →  forgery detection.
    * **Negative Type B** (label = 0): Genuine signatures from
      DIFFERENT people  →  inter-person variation.

    Parameters
    ----------
    genuine_by_person : dict[int, list[np.ndarray]]
    forged_by_person  : dict[int, list[np.ndarray]]
    max_pairs         : int

    Returns
    -------
    pair_a, pair_b, labels
    """
    third = max_pairs // 3
    person_ids = sorted(genuine_by_person.keys())

    pair_a_list, pair_b_list, labels = [], [], []

    # ---- Positive: Same person, genuine + genuine (50% of pairs) ----
    n_positive = max_pairs // 2
    for _ in range(n_positive):
        pid = random.choice(person_ids)
        imgs = genuine_by_person[pid]
        if len(imgs) < 2:
            continue
        i, j = random.sample(range(len(imgs)), 2)
        pair_a_list.append(imgs[i])
        pair_b_list.append(imgs[j])
        labels.append(1)

    # ---- Negative A: Same person, genuine + forged (25% of pairs) ----
    n_neg_forgery = max_pairs // 4
    forged_ids = [p for p in person_ids if p in forged_by_person]
    for _ in range(n_neg_forgery):
        pid = random.choice(forged_ids)
        g_imgs = genuine_by_person[pid]
        f_imgs = forged_by_person[pid]
        gi = random.randint(0, len(g_imgs) - 1)
        fi = random.randint(0, len(f_imgs) - 1)
        pair_a_list.append(g_imgs[gi])
        pair_b_list.append(f_imgs[fi])
        labels.append(0)

    # ---- Negative B: Different people, genuine + genuine (25% of pairs) ----
    n_neg_diff = max_pairs // 4
    for _ in range(n_neg_diff):
        pid1, pid2 = random.sample(person_ids, 2)
        img1 = random.choice(genuine_by_person[pid1])
        img2 = random.choice(genuine_by_person[pid2])
        pair_a_list.append(img1)
        pair_b_list.append(img2)
        labels.append(0)

    pair_a = np.array(pair_a_list)
    pair_b = np.array(pair_b_list)
    labels = np.array(labels, dtype="float32")

    # Shuffle the pairs
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    pair_a = pair_a[indices]
    pair_b = pair_b[indices]
    labels = labels[indices]

    return pair_a, pair_b, labels


def augment_image(img):
    """
    Apply random augmentations to a single image to increase data variety.
    Helps prevent overfitting by creating slightly different versions.
    """
    # Random rotation (-10 to +10 degrees)
    angle = np.random.uniform(-10, 10)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=1.0)

    # Random shift (up to 5% of image size)
    tx = np.random.uniform(-0.05, 0.05) * w
    ty = np.random.uniform(-0.05, 0.05) * h
    M_shift = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M_shift, (w, h), borderValue=1.0)

    # Random brightness adjustment
    brightness = np.random.uniform(0.85, 1.15)
    img = np.clip(img * brightness, 0, 1).astype("float32")

    # Restore channel dimension if lost by OpenCV
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    return img


def augment_pairs(pair_a, pair_b, labels):
    """
    Create augmented copies of all pairs, doubling the dataset size.
    """
    aug_a, aug_b = [], []
    for i in range(len(pair_a)):
        aug_a.append(augment_image(pair_a[i].copy()))
        aug_b.append(augment_image(pair_b[i].copy()))

    # Combine original + augmented
    all_a = np.concatenate([pair_a, np.array(aug_a)], axis=0)
    all_b = np.concatenate([pair_b, np.array(aug_b)], axis=0)
    all_labels = np.concatenate([labels, labels], axis=0)

    # Shuffle again
    indices = np.arange(len(all_labels))
    np.random.shuffle(indices)
    return all_a[indices], all_b[indices], all_labels[indices]


def plot_training_history(history, save_path="training_history.png"):
    """
    Save accuracy and loss curves for the training run.

    Parameters
    ----------
    history : keras.callbacks.History
    save_path : str
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history.history["accuracy"], label="Train Accuracy")
    if "val_accuracy" in history.history:
        ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Routine
# ──────────────────────────────────────────────────────────────────────────────

def train():
    """
    Full training pipeline:
      1. Load images.
      2. Create pairs.
      3. Split into train / validation sets.
      4. Build & train the Siamese model.
      5. Save model and training curves.
    """
    print("=" * 60)
    print("  Siamese Signature Forgery Detector — Training Engine")
    print("=" * 60)

    # ---- 1. Load images grouped by person ----
    print(f"\n[INFO] Loading genuine images from  {GENUINE_DIR} ...")
    genuine_by_person = load_images_by_person(GENUINE_DIR)
    total_genuine = sum(len(v) for v in genuine_by_person.values())
    print(f"       → Found {total_genuine} genuine images from {len(genuine_by_person)} people.")

    print(f"[INFO] Loading forged images from   {FORGED_DIR} ...")
    forged_by_person = load_images_by_person(FORGED_DIR)
    total_forged = sum(len(v) for v in forged_by_person.values())
    print(f"       → Found {total_forged} forged images from {len(forged_by_person)} people.")

    if len(genuine_by_person) < 2:
        raise ValueError(
            "Need at least 2 people with genuine images. "
            f"Found {len(genuine_by_person)} in '{GENUINE_DIR}'."
        )

    # ---- 2. Create person-aware pairs ----
    print("\n[INFO] Creating PERSON-AWARE training pairs ...")
    print("       → Positive:   same person, genuine + genuine")
    print("       → Negative A: same person, genuine + forged")
    print("       → Negative B: different people, genuine + genuine")
    pair_a, pair_b, labels = create_pairs(genuine_by_person, forged_by_person,
                                          max_pairs=12000)
    print(f"       → Total pairs : {len(labels)}")
    print(f"       → Positive    : {int(labels.sum())}")
    print(f"       → Negative    : {int(len(labels) - labels.sum())}")

    # ---- 3. Train / Validation split ----
    (train_a, val_a,
     train_b, val_b,
     train_labels, val_labels) = train_test_split(
        pair_a, pair_b, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    print(f"\n[INFO] Training samples   : {len(train_labels)}")
    print(f"[INFO] Validation samples : {len(val_labels)}")

    # ---- 4. Build model ----
    print("\n[INFO] Building Siamese model ...")
    model = build_siamese_model(input_shape=(IMG_SHAPE[0], IMG_SHAPE[1], 1))

    # ---- 4b. Resume from checkpoint if available ----
    initial_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n[INFO] Checkpoint found! Loading weights from {CHECKPOINT_PATH} ...")
        model.load_weights(CHECKPOINT_PATH)
        print("[INFO] Best checkpoint loaded. Training will continue to find even better weights.")
    else:
        print("[INFO] No checkpoint found — starting fresh.")

    model.summary()

    # ---- 5. Augment training data ----
    print("\n[INFO] Augmenting training data ...")
    train_a, train_b, train_labels = augment_pairs(train_a, train_b, train_labels)
    print(f"[INFO] Training samples after augmentation: {len(train_labels)}")

    # ---- 6. Train with callbacks ----
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_weights_only=True,
        save_best_only=True,       # save only best val_loss
        monitor="val_loss",
        verbose=1,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,                # stop if val_loss doesn't improve for 8 epochs
        restore_best_weights=True,
        verbose=1,
    )

    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,               # halve the LR when stuck
        patience=4,
        min_lr=1e-6,
        verbose=1,
    )

    print(f"\n[INFO] Training for up to {EPOCHS} epochs ...")
    print("[INFO] EarlyStopping enabled (patience=8) — training will stop if val_loss doesn't improve.")
    history = model.fit(
        [train_a, train_b], train_labels,
        validation_data=([val_a, val_b], val_labels),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, early_stop, lr_reduce],
        shuffle=True,
        verbose=1,
    )

    # ---- 7. Save final model weights ----
    model.save_weights(MODEL_SAVE_PATH)
    print(f"\n[SUCCESS] Model weights saved to → {MODEL_SAVE_PATH}")

    # Clean up checkpoint files after successful training
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    print("[INFO] Checkpoint files cleaned up.")

    plot_training_history(history)

    return model, history


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point — only runs when invoked directly
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
