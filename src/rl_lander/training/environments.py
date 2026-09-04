"""Helpers to build vectorised LunarLander-v3 environments."""

from __future__ import annotations

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

LUNAR_LANDER_ID: str = "LunarLander-v3"


def make_eval_env(env_id: str = LUNAR_LANDER_ID, seed: int | None = None) -> gym.Env:
    """Single-environment factory used at evaluation / inference time.

    The environment is wrapped in :class:`Monitor` so episode statistics
    are exposed via ``info["episode"]``.
    """
    env = gym.make(env_id)
    env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_train_env(
    env_id: str = LUNAR_LANDER_ID,
    n_envs: int = 8,
    seed: int = 0,
    use_subproc: bool = False,
) -> VecEnv:
    """Vectorised training environment.

    On Windows ``SubprocVecEnv`` requires ``__main__`` guards which is not
    always practical inside notebooks; we therefore default to
    :class:`DummyVecEnv` and let the caller opt-in to subprocesses.
    """
    cls = SubprocVecEnv if use_subproc else DummyVecEnv
    return make_vec_env(env_id, n_envs=n_envs, seed=seed, vec_env_cls=cls)
