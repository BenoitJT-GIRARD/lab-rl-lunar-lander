"""Integration tests for the FastAPI service.

The fixtures build policies but never train them. A service test asks whether the routing,
the validation and the failure modes are right; the quality of the policy has nothing to do
with any of it, and the previous version spent a training run per session to answer a
question it never asked.
"""

from __future__ import annotations

import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from fastapi.testclient import TestClient
from stable_baselines3 import PPO

from rl_lander.agent import LunarLanderAgent, observation_bounds
from rl_lander.training.environments import LUNAR_LANDER_ID, make_eval_env

#: A state the environment could actually produce: hovering, upright, legs free.
VALID_STATE = [0.0, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _untrained(env_id: str, path: Path) -> Path:
    """Save an untrained policy for ``env_id``. Enough for every test in this file."""
    env = gym.make(env_id)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, seed=0, device="cpu")
    model.save(path)
    env.close()
    return path


@pytest.fixture(scope="module")
def lander_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _untrained(LUNAR_LANDER_ID, tmp_path_factory.mktemp("api") / "lander.zip")


@pytest.fixture(scope="module")
def client(lander_checkpoint: Path):
    os.environ["RL_LANDER_MODEL_PATH"] = str(lander_checkpoint)
    os.environ["RL_LANDER_ALGO"] = "ppo"
    from rl_lander.api import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The service booted with no checkpoint to load.

    Its own fixture, because the interesting behaviour of this API is what it does when the
    model is missing.
    """
    monkeypatch.setenv("RL_LANDER_MODEL_PATH", str(tmp_path / "absent.zip"))
    monkeypatch.setenv("RL_LANDER_ALGO", "ppo")
    from rl_lander.api import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_stays_200_without_a_model(client_without_model: TestClient) -> None:
    """Liveness is about the process. A missing model is not a reason to restart it."""
    response = client_without_model.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_ready_is_503_without_a_model(client_without_model: TestClient) -> None:
    """Readiness is about traffic, and this is the status code a load balancer reads.

    With only ``/health``, a service that answers 503 to every prediction still advertised
    itself as healthy, and kept receiving requests.
    """
    response = client_without_model.get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"


def test_ready_is_200_with_a_model(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_info_endpoint(client: TestClient) -> None:
    payload = client.get("/info").json()
    assert payload["env_id"] == LUNAR_LANDER_ID
    assert payload["algorithm"] == "ppo"
    assert payload["action_space"]["2"] == "main_engine"


def test_endpoints_that_need_the_model_answer_503(client_without_model: TestClient) -> None:
    assert client_without_model.get("/info").status_code == 503
    assert client_without_model.post("/play", json={"state": VALID_STATE}).status_code == 503
    assert client_without_model.post("/run", json={"seed": 0}).status_code == 503


def test_play_endpoint_returns_action(client: TestClient) -> None:
    payload = client.post("/play", json={"state": VALID_STATE}).json()
    assert 0 <= payload["action"] <= 3
    assert payload["action_label"] in {"noop", "left_engine", "main_engine", "right_engine"}
    assert payload["state"] == VALID_STATE


def test_play_rejects_a_state_of_the_wrong_length(client: TestClient) -> None:
    assert client.post("/play", json={"state": [0.0, 1.0]}).status_code == 422


def test_play_rejects_a_state_that_is_not_finite(client: TestClient) -> None:
    # `json=` cannot encode NaN, and a client that cannot send the payload cannot
    # test the server against it. Python's decoder accepts the bare literal, which is
    # what a hand-written or non-Python client would put on the wire.
    body = '{"state": [NaN, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}'
    response = client.post("/play", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_play_rejects_a_state_outside_the_observation_box(client: TestClient) -> None:
    """An altitude of 900 is not a LunarLander state, and the policy would answer anyway.

    Validating only for NaN let physically impossible inputs through and returned an action
    for them, which reads as a prediction and is not one.
    """
    state = list(VALID_STATE)
    state[1] = 900.0
    response = client.post("/play", json={"state": state})
    assert response.status_code == 422
    assert "observation space" in response.text


def test_play_rejects_a_fractional_leg_contact(client: TestClient) -> None:
    """The legs are booleans in the observation. 1.5 is not a partial touchdown."""
    state = list(VALID_STATE)
    state[6] = 1.5
    assert client.post("/play", json={"state": state}).status_code == 422


def test_run_endpoint_returns_episode(client: TestClient) -> None:
    payload = client.post("/run", json={"seed": 0, "max_steps": 200}).json()
    assert len(payload["actions"]) == len(payload["rewards"]) == payload["length"]
    assert payload["length"] <= 200
    assert np.isfinite(payload["total_reward"])
    assert isinstance(payload["landed"], bool)


def test_run_refuses_a_step_budget_outside_its_declared_range(client: TestClient) -> None:
    assert client.post("/run", json={"max_steps": 0}).status_code == 422
    assert client.post("/run", json={"max_steps": 10_000}).status_code == 422


def test_reset_returns_a_state_the_environment_could_produce(client: TestClient) -> None:
    state = client.post("/reset", json={"seed": 7}).json()["state"]
    low, high = observation_bounds()
    assert np.all(np.asarray(state) >= low - 1e-4)
    assert np.all(np.asarray(state) <= high + 1e-4)


def test_reset_is_the_same_draw_as_a_plain_env_reset(client: TestClient) -> None:
    """The API's reset must be the same draw as ``env.reset(seed=n)`` everywhere else.

    It used to seed the environment at construction *and* reset it again, returning the
    second draw of the sequence -- a state no other code path in the project produces.
    """
    from_api = client.post("/reset", json={"seed": 7}).json()["state"]
    env = make_eval_env()
    try:
        expected, _info = env.reset(seed=7)
    finally:
        env.close()
    assert np.allclose(from_api, expected)


def test_an_unknown_algorithm_is_refused_rather_than_read_as_dqn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rl_lander.api import _resolve_algorithm

    monkeypatch.setenv("RL_LANDER_ALGO", "xgboost")
    with pytest.raises(ValueError, match="not one of"):
        _resolve_algorithm()


def test_a_checkpoint_from_another_environment_is_refused_at_load(tmp_path: Path) -> None:
    """A CartPole policy used to load cleanly and answer every request with nonsense."""
    wrong = _untrained("CartPole-v1", tmp_path / "cartpole.zip")
    with pytest.raises(ValueError, match="Wrong checkpoint"):
        LunarLanderAgent(wrong)
