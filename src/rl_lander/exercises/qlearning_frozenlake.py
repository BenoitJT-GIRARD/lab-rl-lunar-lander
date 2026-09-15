"""Tabular Q-learning on FrozenLake-v1.

Implements the canonical Q-learning algorithm with an
:math:`\\varepsilon`-greedy policy.  The Bellman update follows

.. math::

    Q(s, a) \\leftarrow Q(s, a) +
        \\alpha\\,\\bigl(r + \\gamma \\max_{a'} Q(s', a') - Q(s, a)\\bigr).

FrozenLake is the reference environment because its state space is
discrete (16 cells on the default 4x4 layout) which makes the tabular
representation tractable and the algorithm easy to follow.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass(slots=True)
class QLearningConfig:
    """Hyper-parameters of the tabular Q-learning loop."""

    n_episodes: int = 20_000
    max_steps: int = 100
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.0005
    seed: int = 42


def build_env(slippery: bool = False) -> gym.Env:
    """Return a freshly initialised FrozenLake-v1 environment.

    The deterministic version (``is_slippery=False``) is used for the
    pedagogical setup so the agent can converge to 100 % success.
    """
    return gym.make("FrozenLake-v1", is_slippery=slippery, map_name="4x4")


def _epsilon_greedy(
    q_table: np.ndarray, state: int, epsilon: float, rng: np.random.Generator
) -> int:
    """Select an action with the :math:`\\varepsilon`-greedy strategy."""
    if rng.random() < epsilon:
        return int(rng.integers(q_table.shape[1]))
    return int(np.argmax(q_table[state, :]))


def train(config: QLearningConfig | None = None, slippery: bool = False) -> dict:
    """Train the agent and return the learned Q-table along with diagnostics.

    Returns
    -------
    dict with the following keys:
        * ``q_table``: ``np.ndarray`` of shape ``(n_states, n_actions)``
        * ``rewards``: list of cumulative episode rewards over training
        * ``epsilons``: list of epsilon values per episode
        * ``config``: the :class:`QLearningConfig` used
    """
    cfg = config or QLearningConfig()
    rng = np.random.default_rng(cfg.seed)
    env = build_env(slippery=slippery)
    try:
        q_table = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)

        epsilon = cfg.epsilon_start
        rewards: list[float] = []
        epsilons: list[float] = []
        for episode in range(cfg.n_episodes):
            state, _info = env.reset(seed=cfg.seed + episode)
            total_reward = 0.0
            for _step in range(cfg.max_steps):
                action = _epsilon_greedy(q_table, int(state), epsilon, rng)
                new_state, reward, terminated, truncated, _info = env.step(action)
                old_value = q_table[state, action]
                # No bootstrap through a terminal state: there is no next action to take,
                # so the target is the reward alone. Truncation is different -- the episode
                # was cut short and the value of where it stopped still counts.
                future_max = 0.0 if terminated else float(np.max(q_table[new_state, :]))
                q_table[state, action] = old_value + cfg.learning_rate * (
                    reward + cfg.discount_factor * future_max - old_value
                )
                total_reward += float(reward)
                state = new_state
                if terminated or truncated:
                    break
            rewards.append(total_reward)
            epsilons.append(epsilon)
            # Decay from the number of episodes *finished*. Indexing on `episode` made the
            # first two episodes share epsilon = epsilon_start, so the schedule ran one
            # episode behind its own definition for the whole of training.
            epsilon = max(
                cfg.epsilon_min,
                cfg.epsilon_start * np.exp(-cfg.epsilon_decay * (episode + 1)),
            )
    finally:
        env.close()

    return {
        "q_table": q_table,
        "rewards": rewards,
        "epsilons": epsilons,
        "config": cfg,
    }


def evaluate(
    q_table: np.ndarray,
    n_episodes: int = 100,
    max_steps: int = 100,
    slippery: bool = False,
    seed: int = 1234,
) -> dict:
    """Greedy evaluation of a trained Q-table.

    Returns ``{"success_rate": float, "average_steps": float}``.
    """
    env = build_env(slippery=slippery)
    try:
        wins = 0
        steps_collected: list[int] = []
        for episode in range(n_episodes):
            state, _info = env.reset(seed=seed + episode)
            for step in range(max_steps):
                action = int(np.argmax(q_table[int(state), :]))
                state, reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated:
                    if reward >= 1.0:
                        wins += 1
                    steps_collected.append(step + 1)
                    break
            else:
                steps_collected.append(max_steps)
    finally:
        env.close()

    return {
        "success_rate": wins / n_episodes,
        "average_steps": float(np.mean(steps_collected)),
        "total_episodes": n_episodes,
    }


def main() -> None:  # pragma: no cover - CLI helper
    print("Training tabular Q-learning on FrozenLake-v1 ...")
    artefacts = train()
    metrics = evaluate(artefacts["q_table"])
    print(
        f"Success rate over {metrics['total_episodes']} episodes: "
        f"{metrics['success_rate']:.0%} | average steps: {metrics['average_steps']:.1f}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
