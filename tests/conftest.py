"""What every tier of this suite shares: where the repository is, and how a test is skipped.

The suite is read by tier — ``unit/``, ``integration/``, ``system/`` — and each tier states
in its own ``conftest.py`` what it forbids. This file holds only what all three need.

**A skip names the command that would run the test.** A skip whose reason is a condition —
``"needs the database"``, ``"no model on disk"`` — teaches a reader that the test is
unrunnable. A skip that says ``run: docker compose up -d postgres`` teaches them how to run
it. Use :func:`skip_unless` and the message writes itself.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import gymnasium as gym
import pytest
from stable_baselines3 import PPO

from rl_lander.training.environments import LUNAR_LANDER_ID
from rl_lander.utils.paths import ROOT_DIR as ROOT

# The root is NOT recomputed here: the import above takes it from the package, which already
# decides where the repository is. A second answer to that question is a second answer.


def skip_unless(condition: bool, *, command: str) -> pytest.MarkDecorator:
    """Skip the test unless the condition holds, naming the command that makes it hold.

    @skip_unless(port_is_open(5432), command="docker compose up -d postgres")
    def test_the_api_writes_its_prediction_to_the_database(): ...
    """
    return pytest.mark.skipif(not condition, reason=f"run: {command}")


def port_is_open(port: int, host: str = "127.0.0.1", timeout: float = 0.25) -> bool:
    """Is something listening? Asked once at collection, never retried in a loop."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def env_is_set(name: str) -> bool:
    """Is this environment variable set to something?

    A test gated on a variable that no workflow and no documented command ever sets skips on
    every checkout, and the count of tests it belongs to is a count of tests nobody runs.
    """
    return bool(os.environ.get(name))


@pytest.fixture(scope="session")
def root() -> Path:
    """The repository root, for a test that must open a published artefact."""
    return ROOT


# --- What this repository's tiers share -------------------------------------


@pytest.fixture(scope="session")
def untrained_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A saved but untrained LunarLander policy.

    Enough for anything that needs *a* model rather than a good one -- the API surface, the
    recorder's metadata, the loading contract. It lands nothing, which is exactly what the
    tests of the no-landing paths need.
    """
    path = tmp_path_factory.mktemp("checkpoints") / "untrained.zip"
    env = gym.make(LUNAR_LANDER_ID)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, seed=0, device="cpu")
    model.save(path)
    env.close()
    return path


@pytest.fixture(scope="session", autouse=True)
def _hash_seed_notice_is_not_for_the_suite() -> None:
    """Silence the PYTHONHASHSEED notice for the suite, and only for the suite.

    `set_global_seed` warns once per process when the variable was absent at interpreter
    start, because it can then only affect subprocesses. That is a true and useful notice
    for someone launching a training run; the test suite is not a launcher, and pytest does
    not set the variable. Marking it as already given here keeps the warning honest -- it is
    still raised, still an error, and `tests/test_seeding.py` asserts exactly that -- without
    every test that seeds anything failing on a message aimed at a person.
    """
    from rl_lander.utils import seeding

    seeding._HASH_SEED_REPORTED = True
