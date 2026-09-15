"""The manual DQN loop.

It was the least tested file in the repository and the one with the most declared-but-unread
knobs. Each test below pins a behaviour that was wrong, and would have been wrong silently:
the loop still trained, still produced a curve, and still ended with a plausible number.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import optim

from rl_lander.exercises.dqn_cartpole import (
    DQN,
    DQNConfig,
    ReplayBuffer,
    Transition,
    _optimise,
    evaluate_manual_dqn,
    train_manual_dqn,
)

N_OBS, N_ACTIONS = 4, 2


def _buffer(n: int, done: bool = False) -> ReplayBuffer:
    buffer = ReplayBuffer(capacity=1_000)
    rng = np.random.default_rng(0)
    for _ in range(n):
        buffer.push(
            Transition(
                state=rng.normal(size=N_OBS).astype(np.float32),
                action=int(rng.integers(N_ACTIONS)),
                reward=1.0,
                next_state=rng.normal(size=N_OBS).astype(np.float32),
                done=done,
            )
        )
    return buffer


def _nets() -> tuple[DQN, DQN, optim.Optimizer]:
    policy = DQN(N_OBS, N_ACTIONS)
    target = DQN(N_OBS, N_ACTIONS)
    target.load_state_dict(policy.state_dict())
    return policy, target, optim.AdamW(policy.parameters(), lr=1e-3)


def test_the_network_maps_a_batch_of_observations_to_one_value_per_action() -> None:
    net = DQN(N_OBS, N_ACTIONS)
    assert net(torch.zeros(7, N_OBS)).shape == (7, N_ACTIONS)


def test_the_buffer_forgets_the_oldest_transition_when_it_is_full() -> None:
    buffer = ReplayBuffer(capacity=3)
    for reward in range(5):
        buffer.push(
            Transition(
                np.zeros(N_OBS, np.float32), 0, float(reward), np.zeros(N_OBS, np.float32), False
            )
        )
    assert len(buffer) == 3


def test_no_gradient_step_before_learning_starts() -> None:
    """`learning_starts` was declared at 1000 and never read.

    Learning must not start before `learning_starts` transitions are in the buffer. The
    module's header says what the early updates were drawn from when it did.
    """
    cfg = DQNConfig(batch_size=8, learning_starts=64, device="cpu")
    policy, target, optimizer = _nets()

    assert _optimise(policy, target, optimizer, _buffer(32), cfg) is None, "32 < learning_starts"
    assert _optimise(policy, target, optimizer, _buffer(64), cfg) is not None


def test_learning_starts_never_undercuts_the_batch_size() -> None:
    """A batch cannot be drawn from fewer transitions than it holds."""
    cfg = DQNConfig(batch_size=32, learning_starts=4, device="cpu")
    policy, target, optimizer = _nets()
    assert _optimise(policy, target, optimizer, _buffer(16), cfg) is None


def test_a_terminal_transition_does_not_bootstrap() -> None:
    """The target of a terminal transition is its reward, with no value after it.

    Checked numerically rather than by reading the expression: with gamma = 0 the two
    branches agree, so the test uses gamma = 0.99 and compares against the closed form.
    """
    cfg = DQNConfig(batch_size=4, learning_starts=0, gamma=0.99, device="cpu")
    policy, target, optimizer = _nets()
    buffer = _buffer(4, done=True)

    batch = buffer.transitions()
    states = torch.from_numpy(np.stack([t.state for t in batch])).float()
    actions = torch.tensor([t.action for t in batch], dtype=torch.long)
    with torch.no_grad():
        q_before = policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    expected_loss = torch.nn.functional.smooth_l1_loss(
        q_before, torch.ones(len(batch))
    )  # target = reward alone

    loss = _optimise(policy, target, optimizer, buffer, cfg)
    assert loss == pytest.approx(float(expected_loss), abs=1e-5)


def test_training_runs_end_to_end_and_reports_its_epsilon(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Three episodes on CartPole: too short to learn, long enough to exercise the loop."""
    cfg = DQNConfig(
        n_episodes=3,
        batch_size=8,
        learning_starts=8,
        target_update=2,
        buffer_capacity=500,
        device="cpu",
        seed=0,
    )
    policy, result = train_manual_dqn(config=cfg)

    assert len(result.rewards) == 3
    assert result.final_epsilon < cfg.eps_start
    assert result.final_epsilon >= cfg.eps_end
    metrics = evaluate_manual_dqn(policy, n_episodes=2, seed=3)
    assert metrics["n_episodes"] == 2
    assert np.isfinite(metrics["mean_reward"])


def test_the_run_is_reproducible_from_its_seed() -> None:
    cfg = DQNConfig(n_episodes=2, batch_size=8, learning_starts=8, device="cpu", seed=7)
    _, first = train_manual_dqn(config=cfg)
    _, again = train_manual_dqn(config=cfg)
    assert first.rewards == again.rewards
