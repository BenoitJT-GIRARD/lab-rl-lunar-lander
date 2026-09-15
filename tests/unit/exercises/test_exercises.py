"""Smoke / correctness tests for the foundational exercises."""

from __future__ import annotations

import numpy as np
import pytest

from rl_lander.exercises.qlearning_frozenlake import (
    QLearningConfig,
    evaluate,
    train,
)
from rl_lander.exercises.random_policy_cartpole import (
    describe_spaces,
    run_random_policy,
)


def test_describe_spaces_contains_expected_keys() -> None:
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    try:
        spaces = describe_spaces(env)
    finally:
        env.close()
    assert {"observation_space", "action_space"} <= spaces.keys()
    assert "Box" in spaces["observation_space"]
    assert "Discrete" in spaces["action_space"]


def test_random_policy_produces_expected_episode_count() -> None:
    history = run_random_policy(n_episodes=3)
    assert len(history) == 3
    assert all(record.steps > 0 for record in history)


@pytest.mark.parametrize("n_episodes", [2_000])
def test_qlearning_solves_deterministic_frozen_lake(n_episodes: int) -> None:
    cfg = QLearningConfig(n_episodes=n_episodes)
    artefacts = train(cfg)
    metrics = evaluate(artefacts["q_table"], n_episodes=50)
    # Deterministic FrozenLake is straightforward; we expect at least 90%
    # success rate within the training budget.
    assert metrics["success_rate"] >= 0.9
    assert isinstance(artefacts["q_table"], np.ndarray)
    assert artefacts["q_table"].shape == (16, 4)
