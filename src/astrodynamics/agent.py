"""Thin abstraction wrapping a Stable-Baselines3 policy for inference.

The class is intentionally model-agnostic: it accepts both PPO and DQN
checkpoints saved via :py:meth:`stable_baselines3.common.base_class.BaseAlgorithm.save`
and exposes a stable Python API that the FastAPI service, the GUI and
the notebook all share.  Keeping the inference logic in a single place
guarantees consistent behaviour across the deliverables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from astrodynamics.training.environments import LUNAR_LANDER_ID, make_eval_env

ACTION_LABELS: dict[int, str] = {
    0: "noop",
    1: "left_engine",
    2: "main_engine",
    3: "right_engine",
}


@dataclass(slots=True)
class EpisodeResult:
    """Summary of a complete episode rollout."""

    total_reward: float
    length: int
    #: The lander came to rest, read from the environment's terminal reward. Not the same
    #: question as whether the score cleared 200, and the two are kept apart.
    landed: bool
    #: The episode ended by itself rather than hitting the step limit. A truncated episode
    #: neither landed nor crashed: it ran out of time still flying.
    terminated: bool
    final_observation: list[float]
    actions: list[int]
    rewards: list[float]


class LunarLanderAgent:
    """Deterministic inference wrapper around an SB3 LunarLander policy."""

    def __init__(
        self,
        model_path: str | Path,
        algorithm: Literal["ppo", "dqn"] = "ppo",
        env_id: str = LUNAR_LANDER_ID,
        device: str = "auto",
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file '{self.model_path}' not found. "
                "Train the agent first via `python -m astrodynamics.training.train_lunarlander`."
            )
        loader = PPO if algorithm == "ppo" else DQN
        self.model: BaseAlgorithm = loader.load(str(self.model_path), device=device)
        self.algorithm = algorithm
        self.env_id = env_id

    def predict(self, observation: np.ndarray | list[float]) -> int:
        """Return the deterministic action for a single observation."""
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        action, _state = self.model.predict(obs, deterministic=True)
        return int(action[0]) if action.ndim else int(action)

    def play_episode(
        self,
        seed: int | None = None,
        render_mode: str | None = None,
        max_steps: int = 1_000,
    ) -> tuple[EpisodeResult, list[np.ndarray]]:
        """Play a single episode and return the result + RGB frames if any.

        ``render_mode='rgb_array'`` makes the environment yield image
        arrays that the GUI / video pipeline can collect; otherwise an
        empty list is returned.
        """
        env: gym.Env = gym.make(self.env_id, render_mode=render_mode)
        frames: list[np.ndarray] = []
        try:
            obs, _info = env.reset(seed=seed)
            if render_mode == "rgb_array":
                frames.append(env.render())
            actions: list[int] = []
            rewards: list[float] = []
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated) and steps < max_steps:
                action = self.predict(obs)
                obs, reward, terminated, truncated, _info = env.step(action)
                actions.append(action)
                rewards.append(float(reward))
                if render_mode == "rgb_array":
                    frames.append(env.render())
                steps += 1
            total_reward = float(sum(rewards))
            # LunarLander assigns exactly +100 when the lander comes to rest and -100 when
            # it crashes or leaves the frame, so the last reward answers the question
            # directly. `total_reward >= 200` answered a different one -- it is the mean
            # at which the task counts as solved, and it says nothing about one episode.
            landed = bool(terminated and rewards and rewards[-1] > 0.0)
            return (
                EpisodeResult(
                    total_reward=total_reward,
                    length=steps,
                    landed=landed,
                    terminated=bool(terminated),
                    final_observation=obs.tolist(),
                    actions=actions,
                    rewards=rewards,
                ),
                frames,
            )
        finally:
            env.close()

    @staticmethod
    def reset_environment(seed: int | None = None) -> np.ndarray:
        """Convenience helper used by the API to expose a fresh state."""
        env = make_eval_env(seed=seed)
        try:
            obs, _info = env.reset(seed=seed)
            return obs
        finally:
            env.close()
