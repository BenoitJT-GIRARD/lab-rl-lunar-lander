"""Tests for the LunarLander environment factories."""

from __future__ import annotations

from rl_lander.training.environments import (
    LUNAR_LANDER_ID,
    make_eval_env,
    make_train_env,
)


def test_make_eval_env_yields_expected_shapes() -> None:
    env = make_eval_env(seed=0)
    try:
        obs, _info = env.reset(seed=0)
        assert env.spec is not None and env.spec.id == LUNAR_LANDER_ID
        assert obs.shape == (8,)
        assert env.action_space.n == 4
    finally:
        env.close()


def test_make_train_env_returns_vector_env() -> None:
    vec = make_train_env(n_envs=2, seed=0)
    try:
        assert vec.num_envs == 2
        observations = vec.reset()
        assert observations.shape == (2, 8)
    finally:
        vec.close()
