"""Fixtures shared across the suite."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import pytest
from stable_baselines3 import PPO

from rl_lander.training.environments import LUNAR_LANDER_ID


@pytest.fixture(scope="session")
def untrained_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A saved but untrained LunarLander policy.

    Enough for anything that needs *a* model rather than a good one -- the API surface, the
    recorder's metadata, the loading contract. It lands nothing, which is exactly what the
    tests of the no-landing paths need.
    """
    path = tmp_path_factory.mktemp("checkpoints") / "untrained.zip"
    env = gym.make(LUNAR_LANDER_ID)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, seed=0, device="cpu")
    model.save(path)
    env.close()
    return path


@pytest.fixture(scope="session", autouse=True)
def _hash_seed_notice_is_not_for_the_suite() -> None:
    """Silence the PYTHONHASHSEED notice for the suite, and only for the suite.

    `set_global_seed` warns once per process when the variable was absent at interpreter
    start, because it can then only affect subprocesses. That is a true and useful notice
    for someone launching a training run; the test suite is not a launcher, and pytest does
    not set the variable. Marking it as already given here keeps the warning honest -- it is
    still raised, still an error, and `tests/test_seeding.py` asserts exactly that -- without
    every test that seeds anything failing on a message aimed at a person.
    """
    from rl_lander.utils import seeding

    seeding._HASH_SEED_REPORTED = True
