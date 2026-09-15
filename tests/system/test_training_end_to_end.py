"""A training run, started the way the README starts one, in a checkout of its own.

Everything else about training is tested in pieces: the hyper-parameters, the callback that
writes the curve, the manifest, the promotion rule. None of that runs the command, and the
command is what a reader types. What breaks there breaks nowhere else — an entry point that
no longer resolves, a run directory computed from the wrong algorithm, a checkpoint saved
under a name the next step does not look for.

So this one runs it: one PPO policy, a budget small enough to be worth a suite's time, in a
root of its own so nothing it writes touches the published checkpoint. What is asserted is
what the next step needs from it. The policy is untrained at this budget and its score is
terrible, which is not the point: the point is that the run leaves behind an artefact that
names itself.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rl_lander.utils.paths import ROOT_DIR, ROOT_ENV

pytestmark = pytest.mark.system

#: Two rollouts of eight environments. Enough for the callback to fire and for both
#: checkpoints to be decided; far too little to land anything.
TIMESTEPS = 2048
SEED = 7
TIMEOUT = 600


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the training command once, in a throwaway root, and hand back that root."""
    elsewhere = tmp_path_factory.mktemp("training-root")
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "rl_lander.training.train_lunarlander",
            "--algo",
            "ppo",
            "--timesteps",
            str(TIMESTEPS),
            "--seed",
            str(SEED),
        ],
        cwd=str(ROOT_DIR),
        env={**os.environ, "PYTHONIOENCODING": "utf-8", ROOT_ENV: str(elsewhere)},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT,
        check=False,
    )
    if done.returncode != 0:
        pytest.fail(f"the training command failed ({done.returncode}):\n{done.stderr[-4000:]}")
    return elsewhere


def test_the_run_writes_where_the_algorithm_and_the_seed_say(trained: Path) -> None:
    """`var/runs/ppo/seed-7/`, and nothing under `var/runs/dqn/`.

    The path is built from `--algo` and `--seed`, and the defect that rule replaced had a
    DQN run overwrite the PPO artefact.
    """
    run = trained / "var" / "runs" / "ppo" / f"seed-{SEED}"
    assert run.is_dir(), sorted(p.name for p in (trained / "var" / "runs").glob("*"))
    assert not (trained / "var" / "runs" / "dqn").exists()


def test_the_run_leaves_a_checkpoint_the_next_step_can_load(trained: Path) -> None:
    """`best.zip` exists, and it is a Stable-Baselines3 archive rather than an empty file."""
    checkpoint = trained / "var" / "runs" / "ppo" / f"seed-{SEED}" / "best.zip"
    assert checkpoint.is_file()
    assert checkpoint.stat().st_size > 10_000


def test_the_run_records_what_it_did_beside_its_checkpoint(trained: Path) -> None:
    """The manifest carries the budget, the seed, the evaluation grid and the score."""
    manifest = trained / "var" / "runs" / "ppo" / f"seed-{SEED}" / "manifest.json"
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    assert recorded["algorithm"] == "PPO"
    assert recorded["seed"] == SEED
    assert recorded["total_timesteps"] == TIMESTEPS
    assert recorded["n_episodes"] == 100
    assert "mean_reward" in recorded and "landing_rate" in recorded
    assert recorded["hyperparameters"]["seed"] == SEED


def test_a_run_that_never_improved_says_so_instead_of_leaving_two_names(trained: Path) -> None:
    """At this budget no evaluation beats the start, and the manifest is the one that says it.

    `best_is_final` is the flag downstream reads. When it is true there is no `final.zip`
    beside `best.zip`, because two names for one file is what
    `scripts/compare_checkpoints.py` exists to keep apart.
    """
    run = trained / "var" / "runs" / "ppo" / f"seed-{SEED}"
    recorded = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if recorded["best_is_final"]:
        assert not (run / "final.zip").exists()
    else:
        assert (run / "final.zip").is_file()


def test_the_run_writes_a_curve_the_dashboard_could_read(trained: Path) -> None:
    """The curve carries at least the columns `rl_lander.artifacts` requires of it.

    The writer records more than that — the wall clock and the episode length — and the
    reader asks only for what it draws. Equality here would fail the day a column is added
    for a reason the dashboard has no opinion on.
    """
    from rl_lander.artifacts import CURVE_COLUMNS

    run = trained / "var" / "runs" / "ppo" / f"seed-{SEED}"
    curves = sorted(run.glob("*.csv"))
    assert curves, sorted(path.name for path in run.iterdir())
    header = curves[0].read_text(encoding="utf-8").splitlines()[0].split(",")
    assert set(CURVE_COLUMNS) <= set(header), header


def test_nothing_of_the_published_checkout_moved(trained: Path) -> None:
    """The run wrote in its own root, and the repository's `var/runs/` is untouched by it.

    `RL_LANDER_ROOT` is the only thing keeping a training started from a test out of the
    working tree, and a regression in `paths` would put it back there silently.
    """
    assert (trained / "var").is_dir()
    published = ROOT_DIR / "var" / "runs" / "ppo" / f"seed-{SEED}"
    assert not published.exists(), f"the run landed in the checkout at {published}"
