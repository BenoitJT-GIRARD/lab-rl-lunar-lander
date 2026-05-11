"""FastAPI service exposing the trained Eagle-1 autopilot.

Endpoints
---------
``GET /health``
    Simple liveness check.

``GET /info``
    Metadata about the loaded model and environment.

``POST /play``
    Predict a single action for a given observation.

``POST /run``
    Roll out a full episode server-side and return aggregated metrics.

``POST /reset``
    Sample a fresh starting observation (no action returned).

The RL inference logic lives entirely on the backend so frontends
(GUI / dashboard) only deal with JSON payloads.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from astrodynamics import __version__
from astrodynamics.agent import ACTION_LABELS, LunarLanderAgent
from astrodynamics.utils import DEFAULT_MODEL_PATH

OBSERVATION_DIM: int = 8


class Observation(BaseModel):
    """Eight-dimensional LunarLander observation vector."""

    state: list[float] = Field(
        ...,
        description=(
            "8-D state vector: [pos_x, pos_y, vel_x, vel_y, angle, "
            "angular_velocity, leg_left_contact, leg_right_contact]."
        ),
        min_length=OBSERVATION_DIM,
        max_length=OBSERVATION_DIM,
    )

    @field_validator("state")
    @classmethod
    def _validate_finite(cls, value: list[float]) -> list[float]:
        if any(np.isnan(v) or np.isinf(v) for v in value):
            raise ValueError("state must contain finite floats only")
        return value


class ActionResponse(BaseModel):
    """Response payload for :func:`predict_action`."""

    action: int = Field(..., ge=0, le=3)
    action_label: str
    state: list[float]


class RunRequest(BaseModel):
    """Request payload for :func:`run_episode`."""

    seed: int | None = Field(None, description="Optional seed for reproducibility.")
    max_steps: int = Field(1_000, ge=1, le=2_000)


class RunResponse(BaseModel):
    """Aggregated metrics for a full episode rollout."""

    total_reward: float
    length: int
    landed: bool
    final_state: list[float]
    actions: list[int]
    rewards: list[float]


class ResetRequest(BaseModel):
    seed: int | None = None


class InfoResponse(BaseModel):
    model_version: str
    algorithm: str
    env_id: str
    model_path: str
    action_space: dict[int, str]


def _resolve_model_path() -> Path:
    """Locate the model file from the environment or fall back to default."""
    return Path(os.environ.get("ASTRODYNAMICS_MODEL_PATH", DEFAULT_MODEL_PATH))


def _resolve_algorithm() -> str:
    return os.environ.get("ASTRODYNAMICS_ALGO", "ppo").lower()


def get_agent() -> LunarLanderAgent:
    """Cached agent dependency (FastAPI handles app-state caching for us)."""
    return _agent_singleton


_agent_singleton: LunarLanderAgent | None = None  # populated in lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Eagerly load the policy at boot so the first request is fast."""
    global _agent_singleton
    model_path = _resolve_model_path()
    algorithm = _resolve_algorithm()
    if model_path.exists():
        _agent_singleton = LunarLanderAgent(model_path, algorithm=algorithm)  # type: ignore[arg-type]
    else:
        _agent_singleton = None
    try:
        yield
    finally:
        _agent_singleton = None


app = FastAPI(
    title="AstroDynamics Eagle-1 — RL autopilot API",
    version=__version__,
    summary=(
        "Inference service for the LunarLander-v3 autopilot trained for the "
        "reinforcement-learning mission."
    ),
    lifespan=lifespan,
)


def _ensure_loaded(agent: LunarLanderAgent | None) -> LunarLanderAgent:
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No trained model is available. Set ASTRODYNAMICS_MODEL_PATH "
                "to a valid '.zip' checkpoint."
            ),
        )
    return agent


@app.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    """Liveness probe."""
    return {"status": "ok", "model_loaded": _agent_singleton is not None}


@app.get("/info", response_model=InfoResponse, tags=["meta"])
def info(agent: Annotated[LunarLanderAgent | None, Depends(get_agent)]) -> InfoResponse:
    """Metadata about the running service."""
    agent = _ensure_loaded(agent)
    return InfoResponse(
        model_version=__version__,
        algorithm=agent.algorithm,
        env_id=agent.env_id,
        model_path=str(agent.model_path),
        action_space=ACTION_LABELS,
    )


@app.post("/play", response_model=ActionResponse, tags=["agent"])
def predict_action(
    observation: Observation,
    agent: Annotated[LunarLanderAgent | None, Depends(get_agent)],
) -> ActionResponse:
    """Return the deterministic action for a single observation."""
    agent = _ensure_loaded(agent)
    action = agent.predict(observation.state)
    return ActionResponse(
        action=action,
        action_label=ACTION_LABELS[action],
        state=observation.state,
    )


@app.post("/run", response_model=RunResponse, tags=["agent"])
def run_episode(
    request: RunRequest,
    agent: Annotated[LunarLanderAgent | None, Depends(get_agent)],
) -> RunResponse:
    """Run a full episode server-side and return the trajectory."""
    agent = _ensure_loaded(agent)
    result, _frames = agent.play_episode(seed=request.seed, max_steps=request.max_steps)
    return RunResponse(
        total_reward=result.total_reward,
        length=result.length,
        landed=result.landed,
        final_state=result.final_observation,
        actions=result.actions,
        rewards=result.rewards,
    )


@app.post("/reset", response_model=Observation, tags=["agent"])
def reset_environment(request: ResetRequest) -> Observation:
    """Sample a fresh starting observation from the environment."""
    obs = LunarLanderAgent.reset_environment(seed=request.seed)
    return Observation(state=obs.tolist())
