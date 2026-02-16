# 📝 Signature Forgery Detection — Project Notes

## 🎯 Objective

Build a **Siamese Neural Network** that can determine whether two signature
images belong to the same person (genuine) or if one is a forgery.  The model
outputs a single probability score between **0** (forged) and **1** (genuine).

---

## 🏗️ Project Structure

```
Signature_AI/
├── data/
│   ├── genuine/          ← Place genuine signature images here
│   └── forged/           ← Place forged signature images here
├── models/               ← Trained model weights will be saved here
├── model_architecture.py ← Siamese CNN definition + contrastive loss
├── train_engine.py       ← Data loading, pair creation, training loop
├── requirements.txt      ← Python dependencies
└── NOTES.md              ← This file
```

---

## 🧠 Architecture Deep-Dive

### What is a Siamese Network?

A **Siamese Network** consists of **two identical sub-networks** (twins) that
share the exact same weights.  Each sub-network processes one input image and
produces a fixed-length **feature embedding**.  The two embeddings are then
compared — if they are close together in the embedding space the inputs are
considered similar; if far apart, they are considered different.

```
  Image A ──► [Shared CNN] ──► Embedding A ─┐
                                              ├─ Euclidean Distance ─► σ ─► Score
  Image B ──► [Shared CNN] ──► Embedding B ─┘
```

### Why Siamese over a Standard Classifier?

| Feature              | Standard CNN Classifier       | Siamese Network                  |
|----------------------|-------------------------------|----------------------------------|
| Training data needed | Thousands per class           | Just a few per person            |
| New person added     | Must retrain entire model     | No retraining — just add image   |
| Output               | Class label                   | Similarity score (more flexible) |
| Best for             | Fixed, large-class problems   | Verification / one-shot learning |

Signature verification is a classic **one-shot learning** problem: you may have
only 3–5 reference signatures per person, making Siamese networks the ideal
architectural choice.

### Base CNN Architecture

The shared convolutional backbone extracts features through three blocks:

| Layer        | Filters | Kernel  | Output Shape (approx.) | Purpose                       |
|-------------|---------|---------|------------------------|-------------------------------|
| Conv2D      | 64      | 10 × 10| 96 × 96 × 64          | Capture broad stroke patterns |
| MaxPool2D   | –       | 2 × 2  | 48 × 48 × 64          | Downsample                    |
| Dropout     | –       | 25 %   | 48 × 48 × 64          | Regularization                |
| Conv2D      | 128     | 7 × 7  | 42 × 42 × 128         | Finer details (curves, loops) |
| MaxPool2D   | –       | 2 × 2  | 21 × 21 × 128         | Downsample                    |
| Dropout     | –       | 25 %   | 21 × 21 × 128         | Regularization                |
| Conv2D      | 128     | 4 × 4  | 18 × 18 × 128         | Micro-features (pressure)     |
| MaxPool2D   | –       | 2 × 2  | 9 × 9 × 128           | Downsample                    |
| Dropout     | –       | 25 %   | 9 × 9 × 128           | Regularization                |
| Flatten     | –       | –      | 10 368                 | Reshape for Dense             |
| Dense       | 256     | –      | 256                    | High-level feature mixing     |
| Dense       | 128     | –      | 128                    | Final embedding vector        |

> **Kernel Weight Initialisation:** `he_uniform` — best practice for ReLU
> activations, ensuring gradients remain healthy early in training.
>
> **L2 Regularisation:** Applied to the first Dense layer (λ = 1e-4) to
> discourage overly large weights and reduce overfitting.

### Euclidean Distance Layer

A custom Keras layer computes:

```
d(a, b) = √( Σ (aᵢ - bᵢ)² )
```

A small epsilon (1e-7) is added inside the square root to avoid numerical
instability when the two vectors are nearly identical.

### Contrastive Loss Function

Instead of standard cross-entropy, we use **contrastive loss** (Hadsell et al.,
2006):

```
L = y · d²  +  (1 - y) · max(0, margin - d)²
```

- **y = 1** (genuine pair) → the loss penalises large distances, pulling
  similar signatures *together*.
- **y = 0** (forged pair) → the loss penalises distances smaller than the
  `margin`, pushing dissimilar signatures *apart*.
- **margin = 1.0** — the minimum separation we demand between genuine and
  forged pairs in the embedding space.

### Optimizer: RMSprop

RMSprop adapts the learning rate *per parameter* using a moving average of
squared gradients.  It is particularly effective for:

- Recurrent and Siamese architectures with shared weights.
- Training where the gradient magnitudes vary widely across parameters.

Learning rate is set to **1e-4** — a conservative value that avoids overshooting
the loss surface.

---

## ⚙️ Training Pipeline (train_engine.py)

### Step-by-step Flow

1. **Image Loading**
   - Read from `data/genuine/` and `data/forged/`.
   - Convert to grayscale, resize to **105 × 105**, normalise to [0, 1].

2. **Pair Creation**
   - **Positive pairs** (label 1): every unique combination of two genuine images.
   - **Negative pairs** (label 0): every genuine image paired with every forged
     image, then randomly sub-sampled to match the number of positive pairs
     (balancing).

3. **Train / Validation Split**
   - 80 % training, 20 % validation (stratified to maintain label distribution).

4. **Training**
   - 20 epochs, batch size 32.
   - Validation metrics are computed at the end of each epoch.

5. **Model Saving**
   - Saved as `models/my_signature_ai.h5` (HDF5 format, full model).

6. **Visualisation**
   - Accuracy & loss curves saved to `training_history.png`.

### How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place images
#    - Put genuine signature images in   data/genuine/
#    - Put forged  signature images in   data/forged/

# 3. Train
python train_engine.py
```

> ⚠️ Training will **not** start if you simply import `train_engine` from
> another module — it is protected by `if __name__ == "__main__":`.

---

## 🛠️ Technology Stack

| Tool / Library      | Version   | Role                                            |
|---------------------|-----------|-------------------------------------------------|
| **Python**          | 3.9+      | Core language                                   |
| **TensorFlow**      | ≥ 2.12    | Deep learning framework (model, training, GPU)  |
| **Keras**           | (bundled) | High-level API inside TensorFlow                |
| **NumPy**           | ≥ 1.23    | Numerical operations & array handling           |
| **OpenCV**          | ≥ 4.7     | Image I/O, grayscale conversion, resizing       |
| **scikit-learn**    | ≥ 1.2     | `train_test_split` with stratified sampling     |
| **Matplotlib**      | ≥ 3.7     | Plotting training accuracy & loss curves        |

---

## 📐 Design Decisions & Rationale

### Why 105 × 105 × 1?
- Follows the original **SigNet** and **Omniglot one-shot** literature.
- Grayscale channel is sufficient — colour provides no useful signal for
  handwritten signatures.
- Small enough for fast training on a laptop GPU; large enough to preserve
  stroke details.

### Why Dropout (25 %) after every Conv block?
- Signatures are high-variance data (everyone writes differently even across
  their own samples).
- Dropout prevents the CNN from memorising training signatures and forces it to
  learn robust general features like *pressure variation*, *slant angle*, and
  *stroke curvature*.

### Why Balanced Pairs?
- If we generated all possible negative pairs we would have far more negatives
  than positives, biasing the model toward predicting "forged" for everything.
- Random sub-sampling of negatives to match positives keeps the dataset balanced.

### Why Sigmoid on the Final Output?
- Maps the raw Euclidean distance to a clean [0, 1] probability, making it easy
  to set a decision threshold (e.g., > 0.5 → genuine).

---

## 🚀 Next Steps (After Training)

1. **Evaluate** — Compute precision, recall, F1-score, and ROC-AUC on a held-out
   test set.
2. **Inference script** — Load `my_signature_ai.h5`, accept two images, return
   the similarity score.
3. **Threshold tuning** — Use the ROC curve to find the optimal decision
   boundary instead of the default 0.5.
4. **Data augmentation** — Add random rotations, shearing, and elastic
   deformations to make the model more robust.
5. **Deploy** — Wrap the inference in a Flask / FastAPI endpoint or a Gradio
   demo for real-time verification.

---

## 📚 References

- Bromley, J. et al. (1993). *Signature Verification using a Siamese Time
  Delay Neural Network.* NIPS.
- Hadsell, R. et al. (2006). *Dimensionality Reduction by Learning an
  Invariant Mapping* (Contrastive Loss). CVPR.
- Koch, G. et al. (2015). *Siamese Neural Networks for One-shot Image
  Recognition.* ICML Workshop.
- Dey, S. et al. (2017). *SigNet: Convolutional Siamese Network for Writer
  Independent Offline Signature Verification.* arXiv:1707.02131.
