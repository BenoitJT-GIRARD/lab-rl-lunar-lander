"""Re-run a recorded action sequence, and check that the replay is the same episode.

The cockpit asks the API for a trajectory and rebuilds the pictures locally, because
shipping a thousand 600x400 frames over HTTP to draw them is not a design, it is a bill.
That is only legitimate if the local replay *is* the episode the service ran, so the check
lives here rather than in the interface: it is domain logic, it has no Streamlit in it, and
the test suite can reach it without installing a web framework.
"""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np

from astrodynamics.training.environments import LUNAR_LANDER_ID

#: How far a replayed reward may drift from the reported one before the two are called
#: different episodes. Box2D is deterministic given a seed, so anything above float noise is
#: a real divergence, not a tolerance to be widened.
REPLAY_TOLERANCE = 1e-6


def replay_actions(
    actions: Sequence[int], seed: int | None, env_id: str = LUNAR_LANDER_ID
) -> tuple[list[np.ndarray], list[float]]:
    """Re-run a fixed action sequence, returning the rendered frames and the rewards seen."""
    env: gym.Env = gym.make(env_id, render_mode="rgb_array")
    frames: list[np.ndarray] = []
    rewards: list[float] = []
    try:
        env.reset(seed=seed)
        frames.append(env.render())
        for action in actions:
            _obs, reward, terminated, truncated, _info = env.step(int(action))
            rewards.append(float(reward))
            frames.append(env.render())
            if terminated or truncated:
                break
    finally:
        env.close()
    return frames, rewards


def replay_matches(replayed: Sequence[float], reported: Sequence[float]) -> bool:
    """Whether the replay reproduced the episode that was reported."""
    if len(replayed) != len(reported):
        return False
    # rtol=0: `allclose` defaults to a *relative* tolerance of 1e-5, which at a reward of
    # 250 is a quarter of a point per step -- enough to let a genuinely different trajectory
    # pass as identical. The tolerance here is meant to cover float noise and nothing else.
    return bool(np.allclose(replayed, reported, rtol=0.0, atol=REPLAY_TOLERANCE))
