"""What `set_global_seed` seeds, and what it honestly cannot."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from astrodynamics.utils import seeding
from astrodynamics.utils.seeding import set_global_seed


def test_it_seeds_python_numpy_and_torch_together() -> None:
    set_global_seed(123)
    first = (random.random(), float(np.random.random()), float(torch.rand(1)))
    set_global_seed(123)
    assert (random.random(), float(np.random.random()), float(torch.rand(1))) == first


def test_it_pins_cudnn_so_the_autotuner_cannot_change_the_code_path() -> None:
    """Left benchmarking, cuDNN picks its algorithm by timing -- so the same seed on the
    same machine can take a different path depending on what else is running."""
    set_global_seed(1)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_it_says_once_that_it_cannot_fix_hashing_for_this_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PYTHONHASHSEED is read at interpreter start; setting it here reaches subprocesses.

    The original docstring claimed it was set "to mitigate hash-based non-determinism",
    which is a guarantee the function does not give. It now says so -- once, because the
    condition cannot change mid-process and a repeated warning stops being read.
    """
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.setattr(seeding, "_HASH_SEED_REPORTED", False)

    with pytest.warns(RuntimeWarning, match="only applies to subprocesses"):
        set_global_seed(5)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        set_global_seed(6)  # second call, same process: silent


def test_it_still_exports_the_seed_for_child_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seeding, "_HASH_SEED_REPORTED", True)
    set_global_seed(4321)
    import os

    assert os.environ["PYTHONHASHSEED"] == "4321"
