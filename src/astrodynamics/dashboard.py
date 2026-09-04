"""Streamlit dashboard over the evaluation run.

Reads `data/evaluation_episodes.csv` and `data/training_curves.csv`, and shows the training
curve, the aggregate metrics, the per-episode telemetry behind filters, and how the reward
relates to engine use.

One rule holds the page together: **every section sees the same filtered frame.** The first
version filtered inside the episode section and then handed the whole, unfiltered frame to
the engine analysis, so two panels on one screen described two different populations.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from astrodynamics.artifacts import CURVE_COLUMNS, EVALUATION_COLUMNS, read_table
from astrodynamics.utils import EVALUATION_CSV, TRAINING_CURVES_CSV

OUTCOME_COLOURS = {"landed": "#1f9d55", "did not land": "#c81e1e"}


def _with_outcome(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a text outcome column, so the colour scale stays categorical.

    Plotly reads an integer column as continuous and paints a gradient over two values,
    which is unreadable and implies an ordering that does not exist.
    """
    return frame.assign(
        outcome=frame["landed"].map({1: "landed", 0: "did not land"}).astype("string")
    )


def _kpi_row(frame: pd.DataFrame) -> None:
    cols = st.columns(5)
    cols[0].metric("Episodes", len(frame))
    cols[1].metric(
        "Mean reward",
        f"{frame['total_reward'].mean():.1f}",
        delta=f"std {frame['total_reward'].std():.1f}",
    )
    # Two rates, not one. Landing is read from the environment's terminal reward; the
    # threshold is the score. They answer different questions and a weaker policy separates
    # them.
    cols[2].metric("Landing rate", f"{frame['landed'].mean() * 100:.1f}%")
    cols[3].metric("Above solved threshold", f"{frame['meets_threshold'].mean() * 100:.1f}%")
    cols[4].metric(
        "Mean engine firings",
        f"{(frame['main_engine_firings'] + frame['side_engine_firings']).mean():.1f}",
        delta=f"main {frame['main_engine_firings'].mean():.0f}",
    )


def _training_section(curves: pd.DataFrame) -> None:
    st.subheader(":chart_with_upwards_trend: Training progress")
    st.caption(
        "`ep_rew_mean` is a rolling mean over episodes finished during training, under a "
        "stochastic policy. It is a progress signal, not the evaluation result."
    )
    fig = px.line(curves, x="timesteps", y="ep_rew_mean", title="Rolling mean reward")
    for sign, name in ((1, "+1 std"), (-1, "-1 std")):
        fig.add_scatter(
            x=curves["timesteps"],
            y=curves["ep_rew_mean"] + sign * curves["ep_rew_std"],
            mode="lines",
            name=name,
            line={"dash": "dot"},
        )
    fig.update_layout(yaxis_title="Reward", xaxis_title="Timesteps")
    st.plotly_chart(fig, use_container_width=True)


def _filters(frame: pd.DataFrame) -> pd.DataFrame:
    """Collect the sidebar filters and return the frame every section will use."""
    with st.sidebar:
        st.header(":mag: Filters")
        outcome = st.selectbox(
            "Outcome", options=["All episodes", "Landed only", "Did not land"], index=0
        )
        low, high = st.slider(
            "Reward range",
            float(frame["total_reward"].min()),
            float(frame["total_reward"].max()),
            (float(frame["total_reward"].min()), float(frame["total_reward"].max())),
        )
        max_length = st.slider(
            "Max episode length",
            int(frame["length"].min()),
            int(frame["length"].max()),
            int(frame["length"].max()),
        )

    filtered = frame
    if outcome == "Landed only":
        filtered = filtered[filtered["landed"] == 1]
    elif outcome == "Did not land":
        filtered = filtered[filtered["landed"] == 0]
    return filtered[
        (filtered["total_reward"] >= low)
        & (filtered["total_reward"] <= high)
        & (filtered["length"] <= max_length)
    ]


def _episode_section(frame: pd.DataFrame) -> None:
    st.subheader(":telescope: Per-episode telemetry")
    coloured = _with_outcome(frame)

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(
            coloured,
            x="total_reward",
            nbins=20,
            title="Reward distribution",
            color="outcome",
            color_discrete_map=OUTCOME_COLOURS,
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.scatter(
            coloured,
            x="final_x",
            y="final_y",
            color="outcome",
            size="main_engine_firings",
            hover_data=["episode", "seed", "total_reward", "length"],
            title="Where the lander came to rest (size = main-engine firings)",
            color_discrete_map=OUTCOME_COLOURS,
        )
        fig.add_vline(x=-0.1, line_dash="dot", line_color="grey")
        fig.add_vline(x=0.1, line_dash="dot", line_color="grey")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(frame, use_container_width=True, height=320)


def _engine_section(frame: pd.DataFrame) -> None:
    st.subheader(":joystick: Engine use")
    st.caption(
        "Both engines, not only the main one. The side thrusters cost fuel and the reward "
        "function charges for them."
    )

    total = frame["main_engine_firings"] + frame["side_engine_firings"]
    if total.nunique() < 2:
        st.info("Every episode used the same number of firings — nothing to bucket.")
        return

    # Quartiles of the observed distribution rather than fixed cut points. Fixed bands of
    # 10 / 30 / 60 put every episode of this run in the last bucket, which is a chart that
    # cannot say anything.
    buckets = pd.qcut(total, q=4, duplicates="drop")
    breakdown = (
        frame.assign(bucket=buckets.astype(str))
        .groupby("bucket", observed=True)["total_reward"]
        .agg(["mean", "count"])
        .reset_index()
    )
    fig = px.bar(
        breakdown,
        x="bucket",
        y="mean",
        text=breakdown["count"].apply(lambda value: f"n={value}"),
        title="Mean reward by total engine firings (quartiles)",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="Total firings", yaxis_title="Mean reward")
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Eagle-1 — Performance dashboard",
        page_icon=":bar_chart:",
        layout="wide",
    )
    st.title(":bar_chart: Eagle-1 — Performance dashboard")
    st.caption("Training and evaluation of the autopilot. The sidebar filters every panel.")

    curves, curves_problem = read_table(TRAINING_CURVES_CSV, CURVE_COLUMNS)
    episodes, episodes_problem = read_table(EVALUATION_CSV, EVALUATION_COLUMNS)

    if curves is None and episodes is None:
        st.error(
            f"Nothing to show. {curves_problem} {episodes_problem}\n\n"
            "Train with `uv run python -m astrodynamics.training.train_lunarlander`, "
            "then export with `uv run python scripts/evaluate_and_export.py`."
        )
        return

    if curves is not None:
        _training_section(curves)
    else:
        st.info(curves_problem)

    if episodes is None:
        st.info(episodes_problem)
        return

    filtered = _filters(episodes)
    if filtered.empty:
        st.warning("No episode matches the current filters.")
        return

    _kpi_row(filtered)
    _episode_section(filtered)
    _engine_section(filtered)


if __name__ == "__main__":  # pragma: no cover
    main()
