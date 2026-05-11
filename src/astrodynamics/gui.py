"""Streamlit GUI animating an Eagle-1 episode driven by the FastAPI service.

The interface deliberately keeps RL inference on the API side: the GUI
queries ``POST /run`` to obtain the trajectory, then re-plays it locally
for visualisation.  Frames are produced by the Gymnasium ``rgb_array``
renderer instead of being shipped over the wire (which would be costly
for ~150 frames of 600x400 RGB data).
"""

from __future__ import annotations

import io
import os
import time
from typing import Any

import gymnasium as gym
import httpx
import numpy as np
import streamlit as st
from PIL import Image

from astrodynamics.agent import ACTION_LABELS, LunarLanderAgent
from astrodynamics.training.environments import LUNAR_LANDER_ID
from astrodynamics.utils import DEFAULT_MODEL_PATH

st.set_page_config(
    page_title="Eagle-1 — Cockpit",
    page_icon=":rocket:",
    layout="wide",
)

DEFAULT_API_URL = os.environ.get("ASTRODYNAMICS_API_URL", "http://127.0.0.1:8000")


def _replay_actions(actions: list[int], seed: int | None) -> list[np.ndarray]:
    """Re-run an episode locally with a fixed action sequence to grab frames."""
    env: gym.Env = gym.make(LUNAR_LANDER_ID, render_mode="rgb_array")
    frames: list[np.ndarray] = []
    try:
        env.reset(seed=seed)
        frames.append(env.render())
        for action in actions:
            _obs, _r, terminated, truncated, _info = env.step(int(action))
            frames.append(env.render())
            if terminated or truncated:
                break
    finally:
        env.close()
    return frames


def _to_png_bytes(frame: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    return buffer.getvalue()


def _render_episode(frames: list[np.ndarray], fps: int = 30) -> None:
    """Replay the captured frames inside a single Streamlit slot."""
    placeholder = st.empty()
    delay = 1.0 / max(fps, 1)
    for frame in frames:
        placeholder.image(_to_png_bytes(frame), caption=None)
        time.sleep(delay)


def _run_via_api(api_url: str, seed: int | None) -> dict[str, Any]:
    response = httpx.post(
        f"{api_url}/run",
        json={"seed": seed, "max_steps": 1_000},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def _run_locally(seed: int | None) -> dict[str, Any]:
    agent = LunarLanderAgent(DEFAULT_MODEL_PATH)
    result, _frames = agent.play_episode(seed=seed)
    return {
        "total_reward": result.total_reward,
        "length": result.length,
        "landed": result.landed,
        "final_state": result.final_observation,
        "actions": result.actions,
        "rewards": result.rewards,
    }


def main() -> None:
    st.title(":rocket: Eagle-1 — Lunar Landing Cockpit")
    st.caption(
        "Visualisation of an episode driven by the AstroDynamics RL agent.  "
        "Inference is delegated to the FastAPI service when reachable."
    )

    with st.sidebar:
        st.subheader("Mission control")
        api_url = st.text_input("API base URL", value=DEFAULT_API_URL)
        seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42, step=1)
        use_api = st.toggle("Use API backend", value=True)
        fps = st.slider("Replay FPS", min_value=10, max_value=60, value=30)
        launch = st.button(":satellite: Run episode", use_container_width=True)
        st.divider()
        st.markdown(
            "**Action space**\n\n"
            + "\n".join(f"- `{idx}` → {label}" for idx, label in ACTION_LABELS.items())
        )

    if not launch:
        st.info("Configure the run on the left, then press *Run episode*.")
        return

    try:
        payload = _run_via_api(api_url, int(seed)) if use_api else _run_locally(int(seed))
    except (httpx.HTTPError, FileNotFoundError) as exc:
        st.error(f"Failed to query the agent: {exc}")
        return

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("Replay")
        frames = _replay_actions(payload["actions"], int(seed))
        _render_episode(frames, fps=fps)

    with col_right:
        st.subheader("Episode metrics")
        st.metric(
            "Total reward",
            f"{payload['total_reward']:.1f}",
            delta="landed" if payload["landed"] else "missed",
        )
        st.metric("Episode length", payload["length"])
        st.metric("Final altitude", f"{payload['final_state'][1]:.2f}")
        st.divider()
        st.markdown("**Action distribution**")
        action_counts = {
            ACTION_LABELS[i]: int(np.sum(np.array(payload["actions"]) == i)) for i in range(4)
        }
        st.bar_chart(action_counts)
        st.markdown("**Reward per step**")
        st.line_chart(payload["rewards"])


if __name__ == "__main__":  # pragma: no cover
    main()
else:
    main()
