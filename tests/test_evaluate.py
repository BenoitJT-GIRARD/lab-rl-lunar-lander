"""Tests for the evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from stable_baselines3 import PPO

from astrodynamics.training.environments import make_eval_env
from astrodynamics.training.evaluate import (
    EpisodeRecord,
    run_episodes,
    summarise,
    write_csv,
)


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory):
    env = make_eval_env(seed=0)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, seed=0, verbose=0)
    model.learn(total_timesteps=128, progress_bar=False)
    env.close()
    return model


def test_run_episodes_returns_records(tiny_model) -> None:  # type: ignore[no-untyped-def]
    records = run_episodes(tiny_model, n_episodes=3, seed=0)
    assert len(records) == 3
    for record in records:
        assert isinstance(record, EpisodeRecord)
        assert record.length > 0


def test_summarise_handles_empty_input() -> None:
    metrics = summarise([])
    assert metrics["n_episodes"] == 0
    assert metrics["mean_reward"] == 0.0


def test_write_csv_creates_expected_columns(tmp_path: Path, tiny_model) -> None:  # type: ignore[no-untyped-def]
    records = run_episodes(tiny_model, n_episodes=2, seed=0)
    output = tmp_path / "eval.csv"
    write_csv(records, output=output)
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    header = text.splitlines()[0].split(",")
    assert header == [
        "episode",
        "total_reward",
        "length",
        "landed",
        "final_x",
        "final_y",
        "fuel_used",
    ]
