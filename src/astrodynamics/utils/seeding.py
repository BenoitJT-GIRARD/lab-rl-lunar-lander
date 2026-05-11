"""Utility helpers for deterministic seeding across `random`, NumPy and PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed every PRNG that affects training reproducibility.

    The function purposefully covers Python's ``random`` module, NumPy and
    PyTorch (CPU + CUDA) so that experiments can be replayed when only the
    seed is known.  ``PYTHONHASHSEED`` is also set to mitigate hash-based
    non-determinism in dictionary ordering.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
