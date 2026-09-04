"""Deterministic seeding across `random`, NumPy and PyTorch."""

from __future__ import annotations

import os
import random
import warnings

import numpy as np
import torch


def set_global_seed(seed: int, deterministic_cudnn: bool = True) -> None:
    """Seed every generator that affects a training run.

    Covers Python's ``random``, NumPy, PyTorch on CPU and **every** visible CUDA device --
    ``manual_seed`` alone leaves the second GPU onwards unseeded, which is silent until the
    day the machine has two.

    With ``deterministic_cudnn`` the cuDNN autotuner is switched off. Left on, it picks a
    convolution algorithm from a timing benchmark, so the same seed on the same machine can
    take a different code path depending on what else is running.

    What this cannot do: ``PYTHONHASHSEED`` fixes string hashing, and Python reads it once,
    at interpreter start. Setting it here only affects processes launched afterwards. It is
    set for that reason, and a mismatch is reported rather than hidden -- an experiment
    whose reproducibility depends on hash ordering has to be launched with the variable
    already in the environment.
    """
    existing = os.environ.get("PYTHONHASHSEED")
    if existing is not None and existing != str(seed):
        warnings.warn(
            f"PYTHONHASHSEED is {existing!r} in this process and cannot be changed after "
            f"start; set_global_seed({seed}) applies to subprocesses only.",
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
