"""Exercice 3 — Deep Q-Network on CartPole-v1 (manual + Stable-Baselines3).

The module exposes:

* :class:`DQN` — a small fully-connected Q-network in PyTorch.
* :class:`ReplayBuffer` — a fixed-capacity experience replay buffer.
* :func:`train_manual_dqn` — implements the full DQN loop with target
  network and ε-greedy exploration.
* :func:`train_sb3_dqn` — equivalent setup using Stable-Baselines3 to
  highlight the productivity boost of high-level libraries.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN as SB3DQN
from stable_baselines3.common.evaluation import evaluate_policy
from torch import nn, optim


@dataclass(slots=True)
class Transition:
    """Single experience tuple stored in the replay buffer."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class DQN(nn.Module):
    """Two-hidden-layer MLP that maps observations to Q-values."""

    def __init__(self, n_observations: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.layer1 = nn.Linear(n_observations, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class ReplayBuffer:
    """Fixed-capacity replay buffer backed by a :class:`collections.deque`."""

    def __init__(self, capacity: int) -> None:
        self._memory: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self._memory.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self._memory, batch_size)

    def __len__(self) -> int:
        return len(self._memory)


@dataclass(slots=True)
class DQNConfig:
    """Hyper-parameters of the manual DQN training loop."""

    n_episodes: int = 600
    batch_size: int = 64
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995
    learning_rate: float = 5e-4
    target_update: int = 20
    buffer_capacity: int = 10_000
    learning_starts: int = 1_000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


@dataclass(slots=True)
class DQNTrainingResult:
    """Result of :func:`train_manual_dqn`."""

    rewards: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    final_epsilon: float = 0.0


def _epsilon_greedy(
    policy_net: DQN,
    state: torch.Tensor,
    epsilon: float,
    n_actions: int,
    device: str,
) -> int:
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        return int(torch.argmax(policy_net(state.to(device))).item())


def _optimise(
    policy_net: DQN,
    target_net: DQN,
    optimizer: optim.Optimizer,
    buffer: ReplayBuffer,
    cfg: DQNConfig,
) -> float | None:
    """Perform one gradient step on a batch sampled from the replay buffer.

    Returns the scalar loss for diagnostics, or ``None`` if the buffer is
    not yet populated enough to draw a batch.
    """
    if len(buffer) < cfg.batch_size:
        return None

    batch = buffer.sample(cfg.batch_size)
    states = torch.from_numpy(np.stack([t.state for t in batch])).float().to(cfg.device)
    actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=cfg.device)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=cfg.device)
    next_states = torch.from_numpy(np.stack([t.next_state for t in batch])).float().to(cfg.device)
    dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=cfg.device)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0]
        targets = rewards + cfg.gamma * next_q * (1.0 - dones)
    loss = F.smooth_l1_loss(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(policy_net.parameters(), 100.0)
    optimizer.step()
    return float(loss.item())


def train_manual_dqn(
    env_id: str = "CartPole-v1",
    config: DQNConfig | None = None,
) -> tuple[DQN, DQNTrainingResult]:
    """Train a DQN agent from scratch using only PyTorch primitives."""
    cfg = config or DQNConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = gym.make(env_id)
    n_obs = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    policy_net = DQN(n_obs, n_actions).to(cfg.device)
    target_net = DQN(n_obs, n_actions).to(cfg.device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=cfg.learning_rate, amsgrad=True)
    buffer = ReplayBuffer(cfg.buffer_capacity)

    result = DQNTrainingResult()
    epsilon = cfg.eps_start
    try:
        for episode in range(cfg.n_episodes):
            state, _info = env.reset(seed=cfg.seed + episode)
            state_t = torch.from_numpy(state).float()
            episode_reward = 0.0
            done = False
            while not done:
                action = _epsilon_greedy(policy_net, state_t, epsilon, n_actions, cfg.device)
                next_state, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)
                buffer.push(Transition(state, action, float(reward), next_state, terminated))
                state = next_state
                state_t = torch.from_numpy(state).float()
                episode_reward += float(reward)

                loss = _optimise(policy_net, target_net, optimizer, buffer, cfg)
                if loss is not None:
                    result.losses.append(loss)

            result.rewards.append(episode_reward)
            epsilon = max(cfg.eps_end, epsilon * cfg.eps_decay)
            if (episode + 1) % cfg.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
    finally:
        env.close()

    result.final_epsilon = epsilon
    return policy_net, result


@dataclass(slots=True)
class SB3DQNConfig:
    """Hyper-parameters for the Stable-Baselines3 DQN agent."""

    total_timesteps: int = 25_000
    learning_rate: float = 6.3e-4
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    batch_size: int = 128
    gamma: float = 0.99
    target_update_interval: int = 250
    exploration_fraction: float = 0.16
    exploration_final_eps: float = 0.04
    seed: int = 42


def train_sb3_dqn(
    env_id: str = "CartPole-v1",
    config: SB3DQNConfig | None = None,
    save_path: str | Path | None = None,
    tensorboard_log: str | Path | None = None,
) -> tuple[SB3DQN, dict]:
    """Train CartPole DQN with Stable-Baselines3.

    The function mirrors the manual loop above but offloads the training
    plumbing to SB3, evaluates the trained policy on 100 episodes and
    optionally persists the model to ``save_path``.
    """
    cfg = config or SB3DQNConfig()
    env = gym.make(env_id)
    model = SB3DQN(
        "MlpPolicy",
        env,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        target_update_interval=cfg.target_update_interval,
        exploration_fraction=cfg.exploration_fraction,
        exploration_final_eps=cfg.exploration_final_eps,
        seed=cfg.seed,
        verbose=0,
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
        device="auto",
    )
    model.learn(total_timesteps=cfg.total_timesteps, progress_bar=False)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
    env.close()
    return model, {"mean_reward": float(mean_reward), "std_reward": float(std_reward)}


def evaluate_manual_dqn(
    policy_net: DQN,
    env_id: str = "CartPole-v1",
    n_episodes: int = 20,
    seed: int = 1234,
) -> dict[str, float]:
    """Greedy evaluation of a manually-trained policy."""
    env = gym.make(env_id)
    device = next(policy_net.parameters()).device
    policy_net.eval()
    rewards: list[float] = []
    try:
        for episode in range(n_episodes):
            state, _info = env.reset(seed=seed + episode)
            done = False
            total = 0.0
            while not done:
                with torch.no_grad():
                    action = int(
                        torch.argmax(policy_net(torch.from_numpy(state).float().to(device))).item()
                    )
                state, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)
                total += float(reward)
            rewards.append(total)
    finally:
        env.close()
    rewards_array: Sequence[float] = rewards
    return {
        "mean_reward": float(np.mean(rewards_array)),
        "std_reward": float(np.std(rewards_array)),
        "n_episodes": float(n_episodes),
    }
