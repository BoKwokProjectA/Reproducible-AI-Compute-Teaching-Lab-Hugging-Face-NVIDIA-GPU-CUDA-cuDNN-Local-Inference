"""Shared constants.

These live in one module so train.py, inference.py and the API can't drift
apart on model paths or dataset IDs. Anything that a user might reasonably
want to change per-run (epochs, batch size) is a CLI argument instead.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ID = "AI-Lab-Makerere/beans"
BASE_MODEL_ID = "google/vit-base-patch16-224-in21k"

# Pin these to real commit SHAs once you've run:
#   python -m src.dataset --show-revisions
# Leaving them as None means "whatever is on main today", which will silently
# change under you. See docs/REPRODUCIBILITY.md.
DATASET_REVISION = "27aa014ce09b193e1a6f58112d4a66e0eddb69c5"
BASE_MODEL_REVISION = "b4569560a39a0f1af58e3ddaf17facf20ab919b0"

# Where train.py writes the fine-tuned model, and where inference.py and the
# API look for it. Kept out of Git (see .gitignore) - model weights are large
# and are rebuilt by rerunning training.
MODEL_DIR = PROJECT_ROOT / "models" / "vit-beans"

# The beans dataset names its target column "labels", not the "label" that
# most HF image datasets use. Named here so the mismatch only bites once.
LABEL_COLUMN = "labels"
IMAGE_COLUMN = "image"
