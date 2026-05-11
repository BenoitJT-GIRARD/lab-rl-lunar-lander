"""Curated hyper-parameter sets for the Eagle-1 mission.

The default values track the well-known Stable-Baselines3 RL Zoo recipe
for LunarLander-v3 (PPO) and provide a sensible DQN baseline as a
secondary reference.  The two configurations are kept small enough to be
exhaustively listed in the notebook for the hyper-parameter discussion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class PPOHyperParameters:
    """PPO hyper-parameters tuned for LunarLander-v3."""

    n_envs: int = 16
    n_steps: int = 1024
    batch_size: int = 64
    n_epochs: int = 4
    gamma: float = 0.999
    gae_lambda: float = 0.98
    learning_rate: float = 3e-4
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 1_000_000
    seed: int = 42

    def to_kwargs(self) -> dict[str, Any]:
        """Return SB3-compatible keyword arguments for :class:`PPO`."""
        return {
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "learning_rate": self.learning_rate,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class DQNHyperParameters:
    """DQN hyper-parameters used as a discrete-action baseline."""

    n_envs: int = 1
    learning_rate: float = 6.3e-4
    buffer_size: int = 50_000
    learning_starts: int = 0
    batch_size: int = 128
    gamma: float = 0.99
    target_update_interval: int = 250
    train_freq: int = 4
    gradient_steps: int = -1
    exploration_fraction: float = 0.12
    exploration_final_eps: float = 0.10
    total_timesteps: int = 200_000
    seed: int = 42
    policy_kwargs: dict[str, Any] = field(default_factory=lambda: {"net_arch": [256, 256]})

    def to_kwargs(self) -> dict[str, Any]:
        """Return SB3-compatible keyword arguments for :class:`DQN`."""
        return {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "target_update_interval": self.target_update_interval,
            "train_freq": self.train_freq,
            "gradient_steps": self.gradient_steps,
            "exploration_fraction": self.exploration_fraction,
            "exploration_final_eps": self.exploration_final_eps,
            "policy_kwargs": dict(self.policy_kwargs),
            "seed": self.seed,
        }
