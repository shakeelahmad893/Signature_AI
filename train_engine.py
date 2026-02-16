"""
train_engine.py
===============
Training pipeline for the Siamese Signature Forgery Detector.

This script:
  1. Loads genuine & forged signature images from  data/genuine  and  data/forged.
  2. Creates balanced positive (genuine–genuine) and negative (genuine–forged) pairs.
  3. Trains the Siamese model for 20 epochs.
  4. Saves the trained model to  models/my_signature_ai.h5.

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
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "my_signature_ai.h5")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "checkpoint.weights.h5")
EPOCHS = 50
BATCH_SIZE = 32


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def load_images_from_folder(folder_path):
    """
    Read every image in *folder_path*, convert to grayscale, resize to
    IMG_SHAPE, and normalise pixel values to [0, 1].

    Parameters
    ----------
    folder_path : str
        Path to a directory containing image files.

    Returns
    -------
    list[np.ndarray]
        List of preprocessed images, each of shape (*IMG_SHAPE, 1).
    """
    images = []
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(valid_extensions):
            continue
        filepath = os.path.join(folder_path, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Could not read: {filepath}")
            continue
        img = cv2.resize(img, IMG_SHAPE)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)          # (105, 105, 1)
        images.append(img)

    return images


def create_pairs(genuine_images, forged_images, max_pairs=10000):
    """
    Build balanced training pairs using **random sampling** to avoid
    memory issues.

    * **Positive pairs** (label = 1): randomly sampled pairs of two
      different genuine signatures.
    * **Negative pairs** (label = 0): randomly sampled pairs of one
      genuine + one forged signature.

    Parameters
    ----------
    genuine_images : list[np.ndarray]
    forged_images  : list[np.ndarray]
    max_pairs      : int
        Maximum TOTAL number of pairs (half positive, half negative).
        Default 10,000 keeps memory around ~2 GB.

    Returns
    -------
    pair_a : np.ndarray of shape (N, 105, 105, 1)
    pair_b : np.ndarray of shape (N, 105, 105, 1)
    labels : np.ndarray of shape (N,)
    """
    half = max_pairs // 2
    num_genuine = len(genuine_images)
    num_forged  = len(forged_images)

    pair_a_list, pair_b_list, labels = [], [], []

    # ---- Positive pairs (genuine – genuine) ----
    for _ in range(half):
        i, j = random.sample(range(num_genuine), 2)
        pair_a_list.append(genuine_images[i])
        pair_b_list.append(genuine_images[j])
        labels.append(1)

    # ---- Negative pairs (genuine – forged) ----
    for _ in range(half):
        gi = random.randint(0, num_genuine - 1)
        fi = random.randint(0, num_forged - 1)
        pair_a_list.append(genuine_images[gi])
        pair_b_list.append(forged_images[fi])
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
    img = np.clip(img * brightness, 0, 1)

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

    # ---- 1. Load images ----
    print(f"\n[INFO] Loading genuine images from  {GENUINE_DIR} ...")
    genuine_images = load_images_from_folder(GENUINE_DIR)
    print(f"       → Found {len(genuine_images)} genuine images.")

    print(f"[INFO] Loading forged images from   {FORGED_DIR} ...")
    forged_images = load_images_from_folder(FORGED_DIR)
    print(f"       → Found {len(forged_images)} forged images.")

    if len(genuine_images) < 2:
        raise ValueError(
            "Need at least 2 genuine images to form positive pairs. "
            f"Found {len(genuine_images)} in '{GENUINE_DIR}'."
        )
    if len(forged_images) < 1:
        raise ValueError(
            "Need at least 1 forged image to form negative pairs. "
            f"Found {len(forged_images)} in '{FORGED_DIR}'."
        )

    # ---- 2. Create pairs ----
    print("\n[INFO] Creating training pairs (randomly sampled) ...")
    pair_a, pair_b, labels = create_pairs(genuine_images, forged_images,
                                          max_pairs=10000)
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
        # Read the epoch number from the checkpoint filename metadata
        # We save epoch info in a small text file alongside the checkpoint
        epoch_file = CHECKPOINT_PATH + ".epoch"
        if os.path.exists(epoch_file):
            with open(epoch_file, "r") as f:
                initial_epoch = int(f.read().strip())
            print(f"[INFO] Resuming from epoch {initial_epoch + 1}/{EPOCHS}")
        else:
            print("[INFO] Resuming from checkpoint (epoch unknown, restarting count)")
    else:
        print("[INFO] No checkpoint found — starting fresh.")

    model.summary()

    # ---- 5. Augment training data ----
    print("\n[INFO] Augmenting training data ...")
    train_a, train_b, train_labels = augment_pairs(train_a, train_b, train_labels)
    print(f"[INFO] Training samples after augmentation: {len(train_labels)}")

    # ---- 6. Train with callbacks ----
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    class SaveEpochCallback(tf.keras.callbacks.Callback):
        """Saves the current epoch number alongside the checkpoint."""
        def on_epoch_end(self, epoch, logs=None):
            with open(CHECKPOINT_PATH + ".epoch", "w") as f:
                f.write(str(epoch + 1))

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

    remaining = EPOCHS - initial_epoch
    print(f"\n[INFO] Training for up to {remaining} remaining epochs ({initial_epoch + 1} → {EPOCHS}) ...")
    print("[INFO] EarlyStopping enabled (patience=8) — training will stop if val_loss doesn't improve.")
    history = model.fit(
        [train_a, train_b], train_labels,
        validation_data=([val_a, val_b], val_labels),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=[checkpoint_cb, SaveEpochCallback(), early_stop, lr_reduce],
        shuffle=True,
        verbose=1,
    )

    # ---- 6. Save final model ----
    model.save(MODEL_SAVE_PATH)
    print(f"\n[SUCCESS] Model saved to → {MODEL_SAVE_PATH}")

    # Clean up checkpoint files after successful training
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    if os.path.exists(CHECKPOINT_PATH + ".epoch"):
        os.remove(CHECKPOINT_PATH + ".epoch")
    print("[INFO] Checkpoint files cleaned up.")

    plot_training_history(history)

    return model, history


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point — only runs when invoked directly
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
