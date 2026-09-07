"""What a training run leaves on disk, and which of it gets published.

Every test here guards a defect that shipped: two algorithms writing to one path, the
final policy saved under the name `best`, metrics printed and lost, and a published
figure with no way back to the run that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl_lander.training.hyperparameters import DQNHyperParameters, PPOHyperParameters
from rl_lander.training.train_lunarlander import _parse_args, train_dqn, train_ppo
from rl_lander.utils import run_dir


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


def test_dqn_writes_the_same_artefacts_under_the_same_names(tmp_path: Path) -> None:
    """Both algorithms have to leave the same shape of run behind them.

    They used to leave two: `EvalCallback` wrote `models/best_model.zip` for both, so a DQN
    run silently replaced a PPO one, and the two CLIs disagreed about where the final model
    went. The names are the contract `scripts/publish_run.py` reads.
    """
    run = tmp_path / "dqn"
    hp = DQNHyperParameters(seed=5, total_timesteps=1_500, learning_starts=100)
    _, metrics = train_dqn(hp, output=run)

    assert (run / "best.zip").exists()
    assert (run / "training_curves.csv").exists()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["algorithm"] == "DQN"
    assert manifest["seed"] == 5
    assert manifest["hyperparameters"]["total_timesteps"] == 1_500
    assert metrics["n_episodes"] == 100


def test_a_slotted_dataclass_default_is_read_from_an_instance() -> None:
    """`PPOHyperParameters.total_timesteps` is a slot descriptor, not the default.

    The CLI used it as a fallback when `--timesteps` was absent. The descriptor passed
    through the constructor without complaint and failed much later, inside SB3, on
    `while self.num_timesteps < total_timesteps` -- a TypeError between an int and a
    `member_descriptor`, thrown after the environments were built and the run had started.
    """
    assert isinstance(type(PPOHyperParameters.total_timesteps), type)
    assert not isinstance(PPOHyperParameters.total_timesteps, int), (
        "if this ever becomes an int, slots=True was dropped -- check why"
    )
    assert isinstance(PPOHyperParameters().total_timesteps, int)
    assert isinstance(DQNHyperParameters().total_timesteps, int)


def test_the_cli_builds_hyperparameters_that_sb3_can_actually_use() -> None:
    """Every field the trainers pass to SB3 has to be a number by the time they do."""
    from rl_lander.training import train_lunarlander as module

    for argv, cls in (
        (["--algo", "ppo"], PPOHyperParameters),
        (["--algo", "dqn"], DQNHyperParameters),
        (["--algo", "ppo", "--timesteps", "5000"], PPOHyperParameters),
    ):
        args = module._parse_args(argv)
        overrides = {} if args.timesteps is None else {"total_timesteps": args.timesteps}
        hp = cls(seed=args.seed, **overrides)
        assert isinstance(hp.total_timesteps, int) and hp.total_timesteps > 0
        assert isinstance(hp.n_envs, int) and hp.n_envs > 0
