from __future__ import annotations

import os
import random

# Avoid OpenMP duplicate-runtime crashes on some Windows Anaconda installs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
