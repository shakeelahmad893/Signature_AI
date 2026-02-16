"""
prepare_data.py
===============
Copies images from the CEDAR dataset into the flat structure
expected by train_engine.py.

CEDAR structure (input):
    CEDAR/
    ├── 1/
    │   ├── original_1_1.png  ... original_1_24.png
    │   └── forgeries_1_1.png ... forgeries_1_24.png
    ├── 2/
    │   └── ...
    └── 55/

Project structure (output):
    Signature_AI/data/genuine/   ← all original_*.png files
    Signature_AI/data/forged/    ← all forgeries_*.png files

Usage:
    python prepare_data.py
"""

import os
import shutil
import glob

# ── Paths ─────────────────────────────────────────────────────────────────────
# CEDAR folder is at the same level as Signature_AI
CEDAR_DIR   = os.path.join(os.path.dirname(__file__), "..", "CEDAR")
GENUINE_DIR = os.path.join(os.path.dirname(__file__), "data", "genuine")
FORGED_DIR  = os.path.join(os.path.dirname(__file__), "data", "forged")


def prepare():
    cedar_path = os.path.abspath(CEDAR_DIR)
    genuine_path = os.path.abspath(GENUINE_DIR)
    forged_path  = os.path.abspath(FORGED_DIR)

    print("=" * 60)
    print("  CEDAR → Signature_AI  Data Preparation")
    print("=" * 60)
    print(f"\n  Source  : {cedar_path}")
    print(f"  Genuine : {genuine_path}")
    print(f"  Forged  : {forged_path}\n")

    if not os.path.isdir(cedar_path):
        raise FileNotFoundError(
            f"CEDAR folder not found at: {cedar_path}\n"
            "Make sure the CEDAR folder is next to the Signature_AI folder."
        )

    os.makedirs(genuine_path, exist_ok=True)
    os.makedirs(forged_path, exist_ok=True)

    genuine_count = 0
    forged_count  = 0

    # Walk through each person folder (1, 2, 3, ... 55)
    for person_folder in sorted(os.listdir(cedar_path)):
        person_path = os.path.join(cedar_path, person_folder)
        if not os.path.isdir(person_path):
            continue

        for filename in os.listdir(person_path):
            filepath = os.path.join(person_path, filename)
            if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue

            if filename.startswith("original"):
                dest = os.path.join(genuine_path, filename)
                shutil.copy2(filepath, dest)
                genuine_count += 1
            elif filename.startswith("forgeries"):
                dest = os.path.join(forged_path, filename)
                shutil.copy2(filepath, dest)
                forged_count += 1

    print(f"  ✅ Copied {genuine_count} genuine images  → data/genuine/")
    print(f"  ✅ Copied {forged_count} forged  images  → data/forged/")
    print(f"\n  Total: {genuine_count + forged_count} images ready for training!")
    print("=" * 60)
    print("\n  Next step:  python train_engine.py\n")


if __name__ == "__main__":
    prepare()
