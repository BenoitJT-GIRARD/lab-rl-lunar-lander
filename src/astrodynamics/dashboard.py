"""Streamlit dashboard summarising the Eagle-1 evaluation run.

Displays:

* training reward curve (from ``data/training_curves.csv``),
* aggregated evaluation metrics (mean / std / success rate / fuel use),
* per-episode telemetry with interactive filters,
* action distribution and reward histogram.

Filters honour the review requirement of having at least one
dynamic chart / filter on the dashboard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from astrodynamics.utils import (
    EVALUATION_CSV,
    TRAINING_CURVES_CSV,
)

st.set_page_config(
    page_title="Eagle-1 — Performance dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)


def _read_csv_safely(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _kpi_row(df: pd.DataFrame) -> None:
    cols = st.columns(4)
    cols[0].metric("Episodes", len(df))
    cols[1].metric(
        "Mean reward",
        f"{df['total_reward'].mean():.1f}",
        delta=f"std {df['total_reward'].std():.1f}",
    )
    cols[2].metric("Success rate", f"{(df['landed']).mean() * 100:.1f}%")
    cols[3].metric("Mean fuel firings", f"{df['fuel_used'].mean():.1f}")


def _training_section(curves: pd.DataFrame) -> None:
    st.subheader(":chart_with_upwards_trend: Training progress")
    fig = px.line(
        curves,
        x="timesteps",
        y="ep_rew_mean",
        title="Rolling mean reward during training",
        markers=False,
    )
    fig.add_scatter(
        x=curves["timesteps"],
        y=curves["ep_rew_mean"] + curves["ep_rew_std"],
        mode="lines",
        name="+1 std",
        line={"dash": "dot"},
    )
    fig.add_scatter(
        x=curves["timesteps"],
        y=curves["ep_rew_mean"] - curves["ep_rew_std"],
        mode="lines",
        name="-1 std",
        line={"dash": "dot"},
    )
    fig.update_layout(yaxis_title="Reward", xaxis_title="Timesteps")
    st.plotly_chart(fig, use_container_width=True)


def _episode_section(df: pd.DataFrame) -> None:
    st.subheader(":telescope: Per-episode telemetry")
    with st.sidebar:
        st.header(":mag: Filters")
        landed_filter = st.selectbox(
            "Outcome", options=["All episodes", "Landed only", "Crashed only"], index=0
        )
        min_reward, max_reward = st.slider(
            "Reward range",
            float(df["total_reward"].min()),
            float(df["total_reward"].max()),
            (float(df["total_reward"].min()), float(df["total_reward"].max())),
        )
        max_length = st.slider(
            "Max episode length",
            int(df["length"].min()),
            int(df["length"].max()),
            int(df["length"].max()),
        )

    filtered = df.copy()
    if landed_filter == "Landed only":
        filtered = filtered[filtered["landed"] == 1]
    elif landed_filter == "Crashed only":
        filtered = filtered[filtered["landed"] == 0]
    filtered = filtered[
        (filtered["total_reward"] >= min_reward)
        & (filtered["total_reward"] <= max_reward)
        & (filtered["length"] <= max_length)
    ]

    if filtered.empty:
        st.warning("No episode matches the current filters.")
        return

    _kpi_row(filtered)
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(
            filtered,
            x="total_reward",
            nbins=20,
            title="Reward distribution",
            color="landed",
            color_discrete_map={1: "#1f9d55", 0: "#c81e1e"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.scatter(
            filtered,
            x="final_x",
            y="final_y",
            color="landed",
            size="fuel_used",
            hover_data=["episode", "total_reward", "length"],
            title="Final position vs. reward (size = fuel firings)",
            color_discrete_map={1: "#1f9d55", 0: "#c81e1e"},
        )
        fig.add_vline(x=-0.1, line_dash="dot", line_color="grey")
        fig.add_vline(x=0.1, line_dash="dot", line_color="grey")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(filtered, use_container_width=True, height=320)


def _action_section(df: pd.DataFrame) -> None:
    st.subheader(":joystick: Action analysis")
    fuel_buckets = pd.cut(
        df["fuel_used"],
        bins=[-1, 10, 30, 60, np.inf],
        labels=["≤10", "11-30", "31-60", "60+"],
    )
    breakdown = (
        df.assign(fuel_bucket=fuel_buckets)
        .groupby("fuel_bucket", observed=True)["total_reward"]
        .agg(["mean", "count"])
        .reset_index()
    )
    fig = px.bar(
        breakdown,
        x="fuel_bucket",
        y="mean",
        text=breakdown["count"].apply(lambda v: f"n={v}"),
        title="Mean reward per fuel-firing bucket",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title(":bar_chart: Eagle-1 — Performance dashboard")
    st.caption(
        "Interactive review of the autopilot's training and evaluation runs. "
        "Use the sidebar filters to slice the per-episode data."
    )

    eval_df = _read_csv_safely(EVALUATION_CSV)
    curves_df = _read_csv_safely(TRAINING_CURVES_CSV)

    if eval_df is None and curves_df is None:
        st.error(
            "No evaluation or training data available yet. Run "
            "`python -m astrodynamics.training.train_lunarlander` and the "
            "evaluation pipeline before launching the dashboard."
        )
        return

    if curves_df is not None and not curves_df.empty:
        _training_section(curves_df)
    else:
        st.info("Training curve CSV not found — the training section is hidden.")

    if eval_df is not None and not eval_df.empty:
        _episode_section(eval_df)
        _action_section(eval_df)
    else:
        st.info("Evaluation CSV not found — the per-episode section is hidden.")


if __name__ == "__main__":  # pragma: no cover
    main()
else:
    main()
