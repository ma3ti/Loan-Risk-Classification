"""
config.py
---------
Project-wide constants: paths, random seed, scoring metrics, device selection.
"""

import os
import random
import numpy as np
import torch

# Paths
DATA_DIR = os.path.join(".", "data")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
MODEL_DIR = os.path.join(".", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Reproducibility
SEED = 42


def fix_random(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # Use manual_seed only (it already seeds CUDA internally when safe).
    # Calling torch.cuda.manual_seed_all directly can trigger a lazy CUDA
    # init that surfaces *earlier* asynchronous CUDA errors — the classic
    # "device-side assert triggered inside fix_random" on Colab.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Synchronize first to flush any pending CUDA errors from
        # previous operations so they don't surface here misleadingly.
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass  # no CUDA context yet — that's fine
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()

# Scoring (used by ML GridSearchCV)
SCORING_METRICS = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_weighted": "f1_weighted",
    "f1_macro": "f1_macro",
}
