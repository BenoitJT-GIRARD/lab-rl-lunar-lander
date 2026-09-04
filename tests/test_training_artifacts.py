"""What a training run leaves on disk, and which of it gets published.

Every test here guards a defect that shipped: two algorithms writing to one path, the
final policy saved under the name `best`, metrics printed and lost, and a published
figure with no way back to the run that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astrodynamics.training.hyperparameters import DQNHyperParameters, PPOHyperParameters
from astrodynamics.training.train_lunarlander import _parse_args, train_ppo
from astrodynamics.utils import run_dir


def test_each_algorithm_and_seed_gets_its_own_directory() -> None:
    """PPO and DQN pointed EvalCallback at models/, so the second run overwrote the first."""
    assert run_dir("ppo", 42) != run_dir("dqn", 42)
    assert run_dir("ppo", 42) != run_dir("ppo", 43)


def test_the_output_default_is_decided_after_the_algorithm_is_read() -> None:
    """`--output` defaulted to the PPO path whatever `--algo` said.

    A DQN run without `--output` therefore overwrote the shipped PPO artefact, and nothing
    in the process said so.
    """
    assert _parse_args(["--algo", "dqn"]).output is None
    assert _parse_args(["--algo", "ppo"]).output is None


def test_hyperparameter_seeds_reach_the_run_directory() -> None:
    assert run_dir("ppo", PPOHyperParameters().seed).name.startswith("seed-")
    assert run_dir("dqn", DQNHyperParameters().seed).parent.name == "dqn"


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real, very short PPO run. Too short to learn, long enough to write its artefacts."""
    run = tmp_path_factory.mktemp("run")
    train_ppo(PPOHyperParameters(seed=3, total_timesteps=2048, n_envs=2), output=run)
    return run


def test_a_run_leaves_a_checkpoint_and_the_metrics_it_scored(tiny_run: Path) -> None:
    assert (tiny_run / "best.zip").exists()
    manifest = json.loads((tiny_run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["algorithm"] == "PPO"
    assert manifest["seed"] == 3
    assert manifest["n_episodes"] == 100
    assert "landing_rate" in manifest and "mean_reward" in manifest


def test_a_run_that_never_improved_says_so_instead_of_shipping_its_final_state(
    tiny_run: Path,
) -> None:
    """2048 timesteps is shorter than one evaluation interval, so no best checkpoint exists.

    The first version silently saved the final policy under the name `best` and evaluated
    it. Here the run records that the two are the same file.
    """
    manifest = json.loads((tiny_run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["best_is_final"] is True


def test_publishing_refuses_a_run_that_never_improved(tiny_run: Path) -> None:
    """Otherwise an untrained policy ends up behind the README's figures."""
    from scripts.publish_run import publish

    with pytest.raises(SystemExit, match="never improved"):
        publish(tiny_run)


def test_publishing_refuses_a_directory_that_is_not_a_finished_run(tmp_path: Path) -> None:
    from scripts.publish_run import publish

    with pytest.raises(SystemExit, match="not a finished run"):
        publish(tmp_path)
