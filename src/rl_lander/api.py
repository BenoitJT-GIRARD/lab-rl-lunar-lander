"""FastAPI service exposing the trained autopilot.

Endpoints
---------
``GET /health``
    Liveness. The process is up. Always 200 while that is true.

``GET /ready``
    Readiness. A policy is loaded and predictions can be served. 503 otherwise.

``GET /info``
    What is loaded: version, algorithm, environment, checkpoint, action space.

``POST /play``
    The deterministic action for one observation.

``POST /run``
    A full episode rolled out server-side, with its trajectory.

``POST /reset``
    A fresh starting observation.

The inference logic lives in :mod:`rl_lander.agent`, so the GUI, the notebook and this
service all predict through the same code. What is here is the boundary: validation, the
loading contract, and the status codes.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from rl_lander import __version__
from rl_lander.agent import ACTION_LABELS, ALGORITHMS, LunarLanderAgent
from rl_lander.agent import observation_bounds as _observation_bounds
from rl_lander.utils import DEFAULT_MODEL_PATH

#: Read from ``LunarLander-v3`` itself rather than restated here, so a change of
#: environment cannot leave the service validating against numbers nobody updated.
OBSERVATION_LOW, OBSERVATION_HIGH = _observation_bounds()
OBSERVATION_DIM: int = len(OBSERVATION_LOW)

#: Floating-point slack on those bounds. The environment can return a value a hair outside
#: its own box after integration, and a service that rejects the environment's own output
#: is worse than one that accepts a rounding error.
_TOLERANCE: float = 1e-4

router = APIRouter()


class Observation(BaseModel):
    """One LunarLander observation vector."""

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
    def _validate_within_the_environment(cls, value: list[float]) -> list[float]:
        """Finite, and inside the box the environment actually produces.

        Checking only for NaN let a caller post an altitude of 900 or a leg-contact flag of
        1.5. The policy answers anyway -- it is a function, it always answers -- and the
        action returned means nothing, because no such state exists. The bounds come from
        the environment's own ``observation_space``.
        """
        state = np.asarray(value, dtype=np.float64)
        if not np.isfinite(state).all():
            raise ValueError("state must contain finite floats only")
        outside = np.flatnonzero(
            (state < OBSERVATION_LOW - _TOLERANCE) | (state > OBSERVATION_HIGH + _TOLERANCE)
        )
        if outside.size:
            details = ", ".join(
                f"state[{i}]={state[i]:g} outside [{OBSERVATION_LOW[i]:g}, {OBSERVATION_HIGH[i]:g}]"
                for i in outside
            )
            raise ValueError(f"state is outside the LunarLander-v3 observation space: {details}")
        return value


class ActionResponse(BaseModel):
    """One deterministic action, with the label that names it."""

    action: int = Field(..., ge=0, le=3)
    action_label: str
    state: list[float]


class RunRequest(BaseModel):
    """What to roll out."""

    seed: int | None = Field(None, description="Optional seed for reproducibility.")
    max_steps: int = Field(1_000, ge=1, le=2_000)


class RunResponse(BaseModel):
    """A complete episode.

    ``landed`` is read from the environment's terminal reward, not from the score: the
    +200 threshold is a property of the task averaged over episodes, and says nothing about
    any single one.
    """

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
    """The checkpoint to serve: ``RL_LANDER_MODEL_PATH``, or the shipped policy."""
    return Path(os.environ.get("RL_LANDER_MODEL_PATH", DEFAULT_MODEL_PATH))


def _resolve_algorithm() -> str:
    """The algorithm to load with, refused rather than guessed when it is unknown.

    ``RL_LANDER_ALGO=xgboost`` used to load the checkpoint as a DQN -- anything that
    was not exactly ``ppo`` fell through to the other branch -- and then failed somewhere
    deeper, on a message about tensor shapes.
    """
    algorithm = os.environ.get("RL_LANDER_ALGO", "ppo").lower()
    if algorithm not in ALGORITHMS:
        raise ValueError(f"RL_LANDER_ALGO={algorithm!r} is not one of {sorted(ALGORITHMS)}.")
    return algorithm


def get_agent(request: Request) -> LunarLanderAgent | None:
    """The policy this application loaded, or ``None`` when there is none.

    Two things were wrong before. The annotation promised a ``LunarLanderAgent`` while the
    body returned a module-level singleton that is ``None`` until a model is found -- the
    case every endpoint has to handle. And that singleton was module state, so two
    applications in one process shared it, and the second to shut down cleared the policy
    of the first.
    """
    return getattr(request.app.state, "agent", None)


def _ensure_loaded(agent: LunarLanderAgent | None) -> LunarLanderAgent:
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No trained model is available. Set RL_LANDER_MODEL_PATH "
                "to a valid '.zip' checkpoint."
            ),
        )
    return agent


@router.get("/health", tags=["meta"])
def health(request: Request) -> dict[str, Any]:
    """Liveness: the process is up and serving."""
    return {"status": "ok", "model_loaded": getattr(request.app.state, "agent", None) is not None}


@router.get("/ready", tags=["meta"])
def ready(request: Request, response: Response) -> dict[str, Any]:
    """Readiness: a policy is loaded and a prediction can be answered.

    Separate from ``/health`` because an orchestrator acts differently on each: liveness
    failing means restart the process, readiness failing means stop sending it traffic. One
    endpoint returning 200 with ``model_loaded: false`` conflates them, and a load balancer
    reading the status code kept routing requests to a service answering 503 to all of them.
    """
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unavailable", "reason": "no model loaded"}
    return {"status": "ready", "model_path": str(agent.model_path)}


@router.get("/info", response_model=InfoResponse, tags=["meta"])
def info(agent: Annotated[LunarLanderAgent | None, Depends(get_agent)]) -> InfoResponse:
    """What the running service has loaded."""
    agent = _ensure_loaded(agent)
    return InfoResponse(
        model_version=__version__,
        algorithm=agent.algorithm,
        env_id=agent.env_id,
        model_path=str(agent.model_path),
        action_space=ACTION_LABELS,
    )


@router.post("/play", response_model=ActionResponse, tags=["agent"])
def predict_action(
    observation: Observation,
    agent: Annotated[LunarLanderAgent | None, Depends(get_agent)],
) -> ActionResponse:
    """The deterministic action for a single observation."""
    agent = _ensure_loaded(agent)
    action = agent.predict(observation.state)
    return ActionResponse(
        action=action,
        action_label=ACTION_LABELS[action],
        state=observation.state,
    )


@router.post("/run", response_model=RunResponse, tags=["agent"])
def run_episode(
    request: RunRequest,
    agent: Annotated[LunarLanderAgent | None, Depends(get_agent)],
) -> RunResponse:
    """Roll out a full episode server-side and return its trajectory."""
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


@router.post("/reset", response_model=Observation, tags=["agent"])
def reset_environment(request: ResetRequest) -> Observation:
    """A fresh starting observation, the same draw as ``env.reset(seed=n)``."""
    obs = LunarLanderAgent.reset_environment(seed=request.seed)
    return Observation(state=obs.tolist())


def _json_safe(value: Any) -> Any:
    """Replace values JSON cannot carry, recursively.

    Rejecting a NaN observation used to crash the service. The validator raised, FastAPI
    built its 422, and the 422 quotes the offending input -- which was ``nan``, which the
    JSON encoder refuses. The client got a 500 and no explanation, for an input the service
    had correctly identified as invalid.
    """
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


async def _on_invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 with the reason, even when the reason is a value JSON cannot represent."""
    return JSONResponse(
        # 422 spelled out: Starlette renamed the constant, and this handler exists
        # precisely so a validation failure never becomes a 500.
        status_code=422,
        content={"detail": _json_safe(jsonable_encoder(exc.errors()))},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Load the policy at boot, so the first request does not pay for it."""
    model_path = _resolve_model_path()
    algorithm = _resolve_algorithm()
    app.state.agent = (
        LunarLanderAgent(model_path, algorithm=algorithm)  # type: ignore[arg-type]
        if model_path.exists()
        else None
    )
    try:
        yield
    finally:
        app.state.agent = None


def create_app() -> FastAPI:
    """Build a service instance.

    A factory rather than one module-level object, because the environment is read at boot:
    a test that wants a service with no model and one that wants a service with a model
    need two applications, not one global mutated between them. The routes live on a
    router for the same reason -- decorators bound to a single instance make the factory a
    lie, and every route on the second instance a 404.
    """
    application = FastAPI(
        title="LunarLander autopilot API",
        version=__version__,
        summary="Inference service for the LunarLander-v3 autopilot trained in this repository.",
        lifespan=lifespan,
    )
    application.include_router(router)
    application.add_exception_handler(RequestValidationError, _on_invalid_request)
    return application


#: The instance uvicorn serves: ``uv run uvicorn rl_lander.api:app``.
app = create_app()
