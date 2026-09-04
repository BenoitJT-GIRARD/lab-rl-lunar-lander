"""Deterministic seeding across `random`, NumPy and PyTorch."""

from __future__ import annotations

import os
import random
import warnings

import numpy as np
import torch

#: Whether the "too late for this process" notice has already been given. Once is enough:
#: the condition cannot change while the process runs, and repeating it on every call would
#: make it noise, which is how a true warning stops being read.
_HASH_SEED_REPORTED = False


def set_global_seed(seed: int, deterministic_cudnn: bool = True) -> None:
    """Seed every generator that affects a training run.

    Covers Python's ``random``, NumPy, PyTorch on CPU and **every** visible CUDA device --
    ``manual_seed`` alone leaves the second GPU onwards unseeded, which is silent until the
    day the machine has two.

    With ``deterministic_cudnn`` the cuDNN autotuner is switched off. Left on, it picks its
    algorithm from a timing benchmark, so the same seed on the same machine can take a
    different code path depending on what else is running.

    What this cannot do: ``PYTHONHASHSEED`` fixes string hashing, and Python reads it once,
    at interpreter start. Setting it here affects processes launched afterwards and not
    this one. When the variable was absent at start -- the case where hashing really is
    randomised and nothing here can fix it -- that is said once, rather than left as a
    comment claiming a guarantee the function does not give.
    """
    global _HASH_SEED_REPORTED
    if os.environ.get("PYTHONHASHSEED") is None and not _HASH_SEED_REPORTED:
        _HASH_SEED_REPORTED = True
        warnings.warn(
            "PYTHONHASHSEED was not set when this process started, so string hashing is "
            f"randomised and set_global_seed({seed}) cannot change it -- it only applies to "
            f"subprocesses. Launch with PYTHONHASHSEED={seed} if a result depends on "
            "dictionary or set ordering.",
            RuntimeWarning,
            stacklevel=2,
        )
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
