"""Curated hyper-parameter sets for the mission.

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
    #: CPU, deliberately. Stable-Baselines3 warns that PPO with an MLP policy belongs on
    #: the CPU, and on this project it is measurable: 50k steps take 14.9 s on the CPU
    #: against 20.3 s on an RTX 4060 Ti, so a full run is 5 minutes rather than 7. The
    #: policy is three small dense layers; the GPU spends its time on transfers, and the
    #: rollout loop -- sixteen Box2D simulations -- is CPU-bound either way.
    device: str = "cpu"

    def to_kwargs(self) -> dict[str, Any]:
        """Return SB3-compatible keyword arguments for :class:`PPO`."""
        return {
            "device": self.device,
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

    #: DQN learns off-policy from a replay buffer, so collecting from several environments
    #: changes only how fast transitions arrive, not what is learned from them. One is the
    #: honest default here; the field exists because the training code reads it.
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
    #: The opposite choice from PPO, for the same reason: measurement. DQN replays batches
    #: of 128 through two 256-unit layers with `gradient_steps=-1`, which is enough work to
    #: pay for the transfers -- 20k steps take 41.8 s on the GPU against 64.0 s on the CPU.
    device: str = "auto"

    def to_kwargs(self) -> dict[str, Any]:
        """Return SB3-compatible keyword arguments for :class:`DQN`."""
        return {
            "device": self.device,
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
