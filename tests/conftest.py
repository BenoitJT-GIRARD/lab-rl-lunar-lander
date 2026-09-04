"""Fixtures shared across the suite."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import pytest
from stable_baselines3 import PPO

from astrodynamics.training.environments import LUNAR_LANDER_ID


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
