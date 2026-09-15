"""The loop, the two spaces, and a policy that chooses at random, on CartPole-v1.

The goal is to familiarise ourselves with the canonical
``observation -> action -> reward`` loop and with the two fundamental
spaces exposed by every Gymnasium environment:

* ``observation_space`` (continuous ``Box`` for CartPole)
* ``action_space`` (``Discrete`` for CartPole)

We then implement a *random-policy* agent and run it for ten episodes,
printing the cumulative reward of each rollout.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym


@dataclass(slots=True)
class EpisodeStats:
    """Result of a single episode driven by the random policy."""

    episode: int
    steps: int
    total_reward: float


def describe_spaces(env: gym.Env) -> dict[str, str]:
    """Return a human-readable description of an environment's spaces.

    The dictionary is friendly for both ``print`` and notebook ``display``.
    """
    return {
        "observation_space": str(env.observation_space),
        "observation_sample": str(env.observation_space.sample()),
        "action_space": str(env.action_space),
        "action_sample": str(env.action_space.sample()),
    }


def run_random_policy(
    env_id: str = "CartPole-v1",
    n_episodes: int = 10,
    seed: int | None = 42,
) -> list[EpisodeStats]:
    """Run ``n_episodes`` episodes with a uniformly random policy.

    Each episode resets the environment, then loops while the trajectory is
    neither terminated nor truncated.  Cumulative reward is reported per
    episode and the environment is closed cleanly at the end.

    Parameters
    ----------
    env_id:
        Identifier of the Gymnasium environment.
    n_episodes:
        Number of full episodes to run.
    seed:
        Optional seed forwarded to ``env.reset`` to make results comparable.
    """
    env = gym.make(env_id)
    # `action_space.sample()` draws from the space's own generator, not from the one
    # `reset(seed=...)` touches. Without this line the seed made the *initial states*
    # reproducible and left the policy itself random, so two runs of a "seeded" experiment
    # returned different rewards.
    if seed is not None:
        env.action_space.seed(seed)
    history: list[EpisodeStats] = []
    try:
        for episode_idx in range(1, n_episodes + 1):
            episode_seed = None if seed is None else seed + episode_idx
            _obs, _info = env.reset(seed=episode_seed)
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0
            while not (terminated or truncated):
                action = env.action_space.sample()
                _obs, reward, terminated, truncated, _info = env.step(action)
                total_reward += float(reward)
                steps += 1
            history.append(
                EpisodeStats(episode=episode_idx, steps=steps, total_reward=total_reward)
            )
    finally:
        env.close()
    return history


def main() -> None:  # pragma: no cover - CLI helper
    env = gym.make("CartPole-v1")
    print("=== CartPole-v1 spaces ===")
    for key, value in describe_spaces(env).items():
        print(f"{key}: {value}")
    env.close()

    print("\n=== Random-policy rollouts ===")
    for stats in run_random_policy():
        print(
            f"Episode {stats.episode:>2}: steps={stats.steps:>3}, "
            f"total_reward={stats.total_reward:.1f}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
