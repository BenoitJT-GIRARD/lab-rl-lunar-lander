"""A real uvicorn, a real port, and the shipped policy asked to fly.

The tier below this one drives the FastAPI application object in-process, which settles the
routing and the validation and settles nothing about the command in ``## Running it``. An entry
point that no longer resolves, a checkpoint the process cannot find, a port already held: the
test client meets none of them, and a reader meets all three.

What is asserted here is what a caller sees. A trajectory comes back; it is the trajectory of
the seed that was asked for and not of a fresh episode; and the policy that is shipped lands
it.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from rl_lander.training.evaluate import SOLVED_THRESHOLD
from rl_lander.utils import DEFAULT_MODEL_PATH, ROOT_DIR

pytestmark = pytest.mark.system

BOOT_TIMEOUT = 120

#: A seed whose episode the shipped policy lands. Any seed of the published grid would do;
#: this one is fixed so two runs of the suite compare the same trajectory.
SEED = 2024


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _get(url: str, timeout: float = 30.0) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=timeout) as answer:
        return answer.status, json.loads(answer.read().decode("utf-8"))


def _post(url: str, payload: dict, timeout: float = 60.0) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as answer:
        return answer.status, json.loads(answer.read().decode("utf-8"))


@pytest.fixture(scope="module")
def service(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Start the API the way ``## Running it`` says to, on a port of its own."""
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip(
            f"no policy at {DEFAULT_MODEL_PATH.name}: run: uv run python scripts/publish_run.py"
        )

    port = _free_port()
    log = tmp_path_factory.mktemp("service") / "uvicorn.log"
    # Output goes to a file. Nothing in this process reads the pipe a Popen would create, so
    # uvicorn would fill it, block on a write, and stop answering.
    with log.open("w", encoding="utf-8") as handle:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "rl_lander.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=str(ROOT_DIR),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + BOOT_TIMEOUT
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    pytest.fail(
                        "uvicorn exited before serving:\n"
                        + Path(log).read_text(encoding="utf-8", errors="replace")
                    )
                try:
                    if _get(base + "/ready", timeout=2)[0] == 200:
                        break
                except (urllib.error.URLError, OSError, TimeoutError):
                    time.sleep(0.5)
            else:
                pytest.fail(
                    f"the service did not answer /ready within {BOOT_TIMEOUT} s:\n"
                    + Path(log).read_text(encoding="utf-8", errors="replace")
                )
            yield base
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()


def test_the_service_reports_itself_healthy_and_ready(service: str) -> None:
    """Two routes, two questions: restart the process, or stop routing to it."""
    status, health = _get(service + "/health")
    assert status == 200
    assert health["status"] == "ok"

    status, ready = _get(service + "/ready")
    assert status == 200
    assert ready["status"] == "ready"
    assert Path(ready["model_path"]).name == DEFAULT_MODEL_PATH.name


def test_the_shipped_policy_lands_the_episode_it_is_asked_for(service: str) -> None:
    status, run = _post(service + "/run", {"seed": SEED})
    assert status == 200

    assert run["total_reward"] > SOLVED_THRESHOLD
    assert run["landed"] is True
    assert len(run["actions"]) == run["length"]
    assert len(run["rewards"]) == run["length"]


def test_the_same_seed_gives_the_same_trajectory(service: str) -> None:
    """A seed that does not pin the episode makes every published score a coincidence."""
    _status, first = _post(service + "/run", {"seed": SEED})
    _status, second = _post(service + "/run", {"seed": SEED})

    assert first["actions"] == second["actions"]
    assert first["total_reward"] == pytest.approx(second["total_reward"])


def test_one_action_is_asked_for_and_answered(service: str) -> None:
    status, answer = _post(service + "/play", {"state": [0.0, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})
    assert status == 200
    assert answer["action"] in {0, 1, 2, 3}
    assert answer["action_label"]


def test_a_state_outside_the_environment_is_refused_with_the_reason(service: str) -> None:
    """422 and the index that is wrong, not a 500 and not an answer that means nothing.

    The policy is a function: hand it an altitude of 900 and it returns an action. The
    refusal has to happen before that, and has to say which coordinate is impossible.
    """
    request = urllib.request.Request(
        service + "/play",
        data=json.dumps({"state": [0.0, 900.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(request, timeout=30)

    assert refused.value.code == 422
    body = refused.value.read().decode("utf-8")
    assert "state[1]" in body


def test_the_service_names_the_policy_it_loaded(service: str) -> None:
    """A score published against an unnamed checkpoint is a score against nothing."""
    status, info = _get(service + "/info")
    assert status == 200
    assert info["algorithm"].upper() == "PPO"
    assert info["env_id"] == "LunarLander-v3"
    assert Path(info["model_path"]).name == DEFAULT_MODEL_PATH.name
