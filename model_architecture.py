"""
model_architecture.py
=====================
Siamese Network Architecture for Signature Forgery Detection.

This module defines the CNN-based Siamese model that learns to distinguish
between genuine and forged signatures by comparing their learned feature
embeddings.

Author : User
Framework: TensorFlow / Keras
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K


# ──────────────────────────────────────────────────────────────────────────────
# Base CNN (Shared Weights)
# ──────────────────────────────────────────────────────────────────────────────

def _build_base_network(input_shape):
    """
    Build the shared CNN that extracts a 128-D feature vector from an image.

    Architecture
    ------------
    Block 1 : Conv2D(64, 10x10) → ReLU → BatchNorm → MaxPool(2x2) → Dropout(0.3)
    Block 2 : Conv2D(128, 7x7)  → ReLU → BatchNorm → MaxPool(2x2) → Dropout(0.3)
    Block 3 : Conv2D(256, 4x4)  → ReLU → BatchNorm → MaxPool(2x2) → Dropout(0.3)
    FC      : Flatten → Dense(512) → ReLU → Dense(128)
    """
    inp = layers.Input(shape=input_shape, name="base_input")

    # ---------- Block 1 ----------
    x = layers.Conv2D(64, (10, 10), activation="relu",
                      kernel_initializer="he_uniform",
                      padding="same", name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(0.3, name="drop1")(x)

    # ---------- Block 2 ----------
    x = layers.Conv2D(128, (7, 7), activation="relu",
                      kernel_initializer="he_uniform",
                      padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(0.3, name="drop2")(x)

    # ---------- Block 3 ----------
    x = layers.Conv2D(256, (4, 4), activation="relu",
                      kernel_initializer="he_uniform",
                      padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(0.3, name="drop3")(x)

    # ---------- Feature Vector ----------
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                     name="fc1")(x)
    x = layers.Dropout(0.3, name="drop_fc")(x)
    x = layers.Dense(128, activation="relu", name="embedding")(x)

    return Model(inp, x, name="BaseCNN")


# ──────────────────────────────────────────────────────────────────────────────
# Siamese Model Builder
# ──────────────────────────────────────────────────────────────────────────────

def build_siamese_model(input_shape=(105, 105, 1)):
    """
    Assemble the full Siamese network.

    Two identical CNN sub-networks (shared weights) each process one input
    image. Their 128-D embeddings are compared using:
      - Absolute difference |A - B|
      - Element-wise product A * B
    These are concatenated and fed into Dense classification layers
    to produce a similarity score in [0, 1].

    Label: 1 = genuine pair (same person), 0 = forged pair.
    """
    # Shared base network (same weights for both branches)
    base_network = _build_base_network(input_shape)

    # Twin inputs
    input_a = layers.Input(shape=input_shape, name="input_a")
    input_b = layers.Input(shape=input_shape, name="input_b")

    # Forward pass through the *same* base network
    embedding_a = base_network(input_a)
    embedding_b = base_network(input_b)

    # Compare embeddings using multiple similarity measures
    abs_diff = layers.Lambda(
        lambda x: tf.abs(x[0] - x[1]), name="abs_difference"
    )([embedding_a, embedding_b])

    product = layers.Multiply(name="element_product")(
        [embedding_a, embedding_b]
    )

    # Concatenate all comparison features
    merged = layers.Concatenate(name="merged")([abs_diff, product])

    # Classification head
    x = layers.Dense(128, activation="relu", name="cls_fc1")(merged)
    x = layers.Dropout(0.3, name="cls_drop1")(x)
    x = layers.Dense(64, activation="relu", name="cls_fc2")(x)
    x = layers.Dropout(0.2, name="cls_drop2")(x)
    output = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=[input_a, input_b], outputs=output,
                  name="SiameseNetwork")

    # Compile with binary crossentropy (proper loss for sigmoid output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check (prints model summary when run directly)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_siamese_model()
    model.summary()
