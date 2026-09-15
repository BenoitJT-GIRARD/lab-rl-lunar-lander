"""Tests for the evaluation helpers.

Two of these guard distinctions the first version collapsed: a landing is not a score, and
a seed grid is not a measurement.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from stable_baselines3 import PPO

from rl_lander.training.environments import make_eval_env
from rl_lander.training.evaluate import (
    FIELDS,
    EpisodeRecord,
    evaluate_seeds,
    run_episodes,
    summarise,
    write_csv,
    write_manifest,
)


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory):
    env = make_eval_env(seed=0)
    model = PPO(
        "MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, seed=0, verbose=0, device="cpu"
    )
    model.learn(total_timesteps=128, progress_bar=False)
    env.close()
    return model


def _record(**overrides) -> EpisodeRecord:
    base = {
        "episode": 0,
        "seed": 42,
        "total_reward": 250.0,
        "length": 300,
        "landed": True,
        "meets_threshold": True,
        "final_x": 0.0,
        "final_y": 0.0,
        "main_engine_firings": 100,
        "side_engine_firings": 20,
    }
    return EpisodeRecord(**{**base, **overrides})


def test_run_episodes_returns_records(tiny_model) -> None:  # type: ignore[no-untyped-def]
    records = run_episodes(tiny_model, n_episodes=3, seed=0)
    assert len(records) == 3
    for record in records:
        assert isinstance(record, EpisodeRecord)
        assert record.length > 0


def test_every_episode_records_the_seed_it_replays_from(tiny_model) -> None:  # type: ignore[no-untyped-def]
    """Without it a published row cannot be reproduced, which is what a row is for."""
    records = run_episodes(tiny_model, n_episodes=4, seed=77)
    assert [r.seed for r in records] == [77, 78, 79, 80]


def test_a_collection_replays_exactly(tiny_model) -> None:  # type: ignore[no-untyped-def]
    first = run_episodes(tiny_model, n_episodes=3, seed=5)
    again = run_episodes(tiny_model, n_episodes=3, seed=5)
    assert [r.total_reward for r in first] == [r.total_reward for r in again]


def test_landing_and_score_are_two_different_columns() -> None:
    """A score of 250 with a truncated episode is not a landing, and the record says so."""
    flying = _record(landed=False, meets_threshold=True)
    metrics = summarise([flying, _record()])

    assert metrics["landing_rate"] == 0.5
    assert metrics["threshold_rate"] == 1.0


def test_side_engine_firings_are_counted_too() -> None:
    """`fuel_used` used to mean the main engine only, under a name that said all of it."""
    metrics = summarise([_record(main_engine_firings=10, side_engine_firings=30)])
    assert metrics["mean_main_engine_firings"] == 10
    assert metrics["mean_side_engine_firings"] == 30


def test_the_standard_deviation_is_the_sample_one() -> None:
    """ddof=1, so the dashboard's pandas default and this module cannot disagree."""
    records = [_record(total_reward=r) for r in (100.0, 200.0, 300.0)]
    assert summarise(records)["std_reward"] == pytest.approx(100.0)


def test_summarising_nothing_is_refused_rather_than_answered_with_zeros() -> None:
    """Zero episodes gave `mean_reward: 0.0`, which reads as a measurement and is not one."""
    with pytest.raises(ValueError, match="empty"):
        summarise([])


def test_summarise_accepts_a_sequence_it_can_walk_twice() -> None:
    records = [_record(episode=i) for i in range(3)]
    assert summarise(records)["n_episodes"] == 3
    assert summarise(records)["n_episodes"] == 3, "the input survived being read"


def test_zero_episodes_is_refused_at_the_boundary(tiny_model) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="at least 1"):
        run_episodes(tiny_model, n_episodes=0)


def test_write_csv_carries_the_seed_and_both_engine_counts(tmp_path: Path, tiny_model) -> None:  # type: ignore[no-untyped-def]
    records = run_episodes(tiny_model, n_episodes=2, seed=0)
    output = write_csv(records, output=tmp_path / "eval.csv")

    header = output.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == list(FIELDS)
    assert "seed" in header
    assert "main_engine_firings" in header and "side_engine_firings" in header


def test_evaluate_seeds_reports_the_spread_not_just_the_mean(tiny_model) -> None:  # type: ignore[no-untyped-def]
    out = evaluate_seeds(tiny_model, seeds=[1, 2, 3], n_episodes=2)

    assert out["seeds"] == [1, 2, 3]
    assert len(out["per_seed"]) == 3
    assert out["worst_seed_mean"] <= out["mean_of_means"]
    assert out["worst_episode"] <= out["worst_seed_mean"]
    assert 0.0 <= out["lowest_landing_rate"] <= 1.0


def test_a_single_seed_reports_no_spread(tiny_model) -> None:  # type: ignore[no-untyped-def]
    out = evaluate_seeds(tiny_model, seeds=[1], n_episodes=2)
    assert out["spread_of_means"] == 0.0


def test_the_manifest_names_the_artefact_and_not_the_machine(tmp_path: Path) -> None:
    """A manifest naming a checkpoint by a path outside the repository names nothing."""
    output = write_manifest(
        tmp_path / "manifest.json",
        model_path=Path("models/ppo/best.zip"),
        algorithm="PPO",
        seeds=[2024, 7],
        n_episodes=100,
        deterministic=True,
    )

    import json

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert not Path(payload["model"]).is_absolute()
    assert payload["seeds"] == [2024, 7]
    assert payload["versions"]["gymnasium"]
    assert payload["evaluated_at"].endswith("+00:00")
