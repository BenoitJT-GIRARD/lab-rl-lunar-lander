"""Streamlit cockpit: one episode of the autopilot, played and explained.

Inference stays on the API side when the service is reachable; the GUI asks ``POST /run``
for the trajectory and rebuilds the pictures locally, because shipping a thousand frames of
600x400 RGB over HTTP to draw them is not a design, it is a bill.

Rebuilding the pictures locally is only legitimate if the local replay *is* the episode the
service ran. It is checked rather than assumed: the replay compares its own reward sequence
against the one the API returned, and says so when they differ. A silent divergence would
put an animation of one episode next to the metrics of another.
"""

from __future__ import annotations

import io
import os
from typing import Any

import httpx
import imageio.v2 as imageio
import numpy as np
import streamlit as st

from rl_lander.agent import ACTION_LABELS, LunarLanderAgent
from rl_lander.replay import replay_actions, replay_matches
from rl_lander.utils import DEFAULT_MODEL_PATH

DEFAULT_API_URL = os.environ.get("RL_LANDER_API_URL", "http://127.0.0.1:8000")


@st.cache_resource(show_spinner=False)
def _local_agent(model_path: str) -> LunarLanderAgent:
    """The policy, loaded once per session rather than once per click.

    Unpacking a checkpoint takes a noticeable moment, and the previous version paid it on
    every run of the same unchanged model.
    """
    return LunarLanderAgent(model_path)


def _animation(frames: list[np.ndarray], fps: int) -> bytes:
    """Encode the frames once, as an animated GIF the browser plays on its own.

    The first version drew the frames one by one into a placeholder with a ``time.sleep``
    between them. That blocks the Streamlit script for the whole animation, so the metrics
    -- which were already computed -- appeared only once the landing had finished playing,
    and any interaction during it was queued behind the sleep.
    """
    buffer = io.BytesIO()
    imageio.mimwrite(buffer, frames, format="GIF", fps=fps, loop=0)
    return buffer.getvalue()


def _run_via_api(api_url: str, seed: int | None) -> dict[str, Any]:
    response = httpx.post(
        f"{api_url}/run",
        json={"seed": seed, "max_steps": 1_000},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def _run_locally(seed: int | None) -> dict[str, Any]:
    agent = _local_agent(str(DEFAULT_MODEL_PATH))
    result, _frames = agent.play_episode(seed=seed)
    return {
        "total_reward": result.total_reward,
        "length": result.length,
        "landed": result.landed,
        "final_state": result.final_observation,
        "actions": result.actions,
        "rewards": result.rewards,
    }


def _obtain_episode(api_url: str, seed: int, prefer_api: bool) -> tuple[dict[str, Any], str]:
    """The episode, and where it came from.

    When the API is preferred but unreachable, this falls back to the local policy and says
    so. The previous version showed a red error and stopped, with a perfectly usable model
    sitting on disk -- the toggle asked the user to diagnose a connection problem.
    """
    if prefer_api:
        try:
            return _run_via_api(api_url, seed), "API"
        except httpx.HTTPError as exc:
            st.warning(
                f"The API at {api_url} did not answer ({exc}). Falling back to the local model."
            )
    return _run_locally(seed), "local model"


def _metrics_panel(payload: dict[str, Any], source: str) -> None:
    st.subheader("Episode metrics")
    st.caption(f"Computed by the {source}.")
    st.metric(
        "Total reward",
        f"{payload['total_reward']:.1f}",
        delta="landed" if payload["landed"] else "did not land",
        delta_color="normal" if payload["landed"] else "inverse",
    )
    st.metric("Episode length", payload["length"])
    st.metric("Final altitude", f"{payload['final_state'][1]:.2f}")
    st.divider()
    st.markdown("**Action distribution**")
    actions = np.asarray(payload["actions"])
    st.bar_chart({ACTION_LABELS[i]: int(np.sum(actions == i)) for i in range(4)})
    st.markdown("**Reward per step**")
    st.line_chart(payload["rewards"])


def main() -> None:
    st.set_page_config(page_title="Cockpit", page_icon=":rocket:", layout="wide")
    st.title(":rocket: Lunar landing cockpit")
    st.caption(
        "One episode of the trained autopilot. Inference runs on the FastAPI service when "
        "it is reachable, and on the local checkpoint otherwise."
    )

    with st.sidebar:
        st.subheader("Mission control")
        api_url = st.text_input("API base URL", value=DEFAULT_API_URL)
        seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42, step=1)
        prefer_api = st.toggle("Prefer the API backend", value=True)
        fps = st.slider("Replay FPS", min_value=10, max_value=60, value=30)
        launch = st.button(":satellite: Run episode", use_container_width=True)
        st.divider()
        st.markdown(
            "**Action space**\n\n"
            + "\n".join(f"- `{idx}` → {label}" for idx, label in ACTION_LABELS.items())
        )

    # The last episode survives a rerun -- moving the FPS slider used to discard the result
    # and leave an empty page until the user pressed the button again.
    if launch:
        payload, source = _obtain_episode(api_url, int(seed), prefer_api)
        frames, replayed = replay_actions(payload["actions"], int(seed))
        st.session_state["episode"] = {
            "payload": payload,
            "source": source,
            "seed": int(seed),
            "frames": frames,
            "faithful": replay_matches(replayed, payload["rewards"]),
        }

    episode = st.session_state.get("episode")
    if episode is None:
        st.info("Configure the run on the left, then press *Run episode*.")
        return

    col_left, col_right = st.columns([3, 2])
    with col_right:
        _metrics_panel(episode["payload"], episode["source"])

    with col_left:
        st.subheader(f"Replay — seed {episode['seed']}")
        if not episode["faithful"]:
            st.error(
                "The local replay diverged from the episode the service ran, so the "
                "animation below is a different trajectory from the metrics on the right. "
                "This means the two are not running the same environment version."
            )
        st.image(_animation(episode["frames"], fps=fps))


if __name__ == "__main__":
    main()
