"""Integration tests for the FastAPI service.

The tests instantiate the application with a freshly trained tiny model
when no checkpoint is available, so the API surface can be exercised
without a long training run.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from stable_baselines3 import PPO

from astrodynamics.training.environments import LUNAR_LANDER_ID, make_eval_env


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a tiny PPO for a handful of steps so the API has something to load."""
    tmp_dir = tmp_path_factory.mktemp("ppo_smoke")
    env = make_eval_env(seed=0)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, seed=0, verbose=0)
    model.learn(total_timesteps=128, progress_bar=False)
    path = tmp_dir / "ppo_smoke.zip"
    model.save(path)
    env.close()
    return path


@pytest.fixture(scope="module")
def client(trained_model: Path):
    os.environ["ASTRODYNAMICS_MODEL_PATH"] = str(trained_model)
    os.environ["ASTRODYNAMICS_ALGO"] = "ppo"
    from astrodynamics.api import app  # imported after env vars are set

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True


def test_info_endpoint(client: TestClient) -> None:
    response = client.get("/info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["env_id"] == LUNAR_LANDER_ID
    assert payload["algorithm"] == "ppo"


def test_play_endpoint_returns_action(client: TestClient) -> None:
    state = [0.0, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    response = client.post("/play", json={"state": state})
    assert response.status_code == 200
    payload = response.json()
    assert 0 <= payload["action"] <= 3
    assert payload["state"] == state


def test_run_endpoint_returns_episode(client: TestClient) -> None:
    response = client.post("/run", json={"seed": 0, "max_steps": 200})
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["actions"], list)
    assert isinstance(payload["rewards"], list)
    assert len(payload["actions"]) == len(payload["rewards"])
    assert np.isfinite(payload["total_reward"])


def test_play_rejects_invalid_state(client: TestClient) -> None:
    response = client.post("/play", json={"state": [0.0, 1.0]})
    assert response.status_code == 422
