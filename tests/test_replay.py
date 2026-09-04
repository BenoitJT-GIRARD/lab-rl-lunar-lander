"""Replaying a recorded action sequence.

The cockpit shows an animation next to metrics it did not compute. The two describe the
same episode only if the replay is faithful, and that is checked rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_lander.agent import LunarLanderAgent
from rl_lander.replay import replay_actions, replay_matches


def test_a_replay_reproduces_the_episode_it_replays(untrained_checkpoint: Path) -> None:
    agent = LunarLanderAgent(untrained_checkpoint)
    result, _frames = agent.play_episode(seed=11, max_steps=120)

    frames, rewards = replay_actions(result.actions, seed=11)

    assert replay_matches(rewards, result.rewards)
    assert len(frames) == len(rewards) + 1, "one frame before the first action, then one each"
    assert frames[0].ndim == 3


def test_a_replay_under_another_seed_is_not_the_same_episode(untrained_checkpoint: Path) -> None:
    """This is the case the cockpit has to catch: same actions, different world."""
    agent = LunarLanderAgent(untrained_checkpoint)
    result, _frames = agent.play_episode(seed=11, max_steps=120)

    _frames, rewards = replay_actions(result.actions, seed=12)

    assert not replay_matches(rewards, result.rewards)


def test_matching_is_exact_rather_than_approximate() -> None:
    assert replay_matches([1.0, 2.0], [1.0, 2.0])
    assert not replay_matches([1.0, 2.0], [1.0, 2.0, 3.0])
    assert not replay_matches([1.0, 2.000_01], [1.0, 2.0])
    assert replay_matches(np.array([1.0, 2.0]), [1.0, 2.0])
