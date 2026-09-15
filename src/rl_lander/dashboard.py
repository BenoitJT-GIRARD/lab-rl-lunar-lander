"""Streamlit dashboard over the evaluation run.

Reads `reports/evaluation_episodes.csv` and `reports/training_curves.csv`, and shows the training
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

from rl_lander.artifacts import CURVE_COLUMNS, EVALUATION_COLUMNS, read_table
from rl_lander.figure_style import PALETTE, STATE
from rl_lander.utils import EVALUATION_CSV, TRAINING_CURVES_CSV

#: Two outcomes, two state colours, taken by name from the palette the figures read. The page
#: used a green and a red of its own, which meant nothing shared between this dashboard and
#: the figures of the README.
OUTCOME_COLOURS = {"landed": STATE["ok"], "did not land": STATE["danger"]}

#: What every chart of this page is drawn with. Plotly takes nothing from the Streamlit theme,
#: so a figure left to itself arrives in the library's defaults on a page painted otherwise.
LAYOUT = {
    "paper_bgcolor": PALETTE["paper"],
    "plot_bgcolor": PALETTE["paper"],
    "font": {"color": PALETTE["ink"], "size": 13},
    "margin": {"t": 50, "b": 40, "l": 10, "r": 10},
}
AXIS = {
    "gridcolor": PALETTE["grid"],
    "zerolinecolor": PALETTE["grid"],
    "linecolor": PALETTE["muted"],
}


def _styled(figure, x_title: str, y_title: str):
    """One place where a chart of this page gets its colours and its axis titles."""
    figure.update_layout(**LAYOUT)
    figure.update_xaxes(title_text=x_title, **AXIS)
    figure.update_yaxes(title_text=y_title, **AXIS)
    return figure


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
    st.subheader("Training progress")
    st.caption(
        "`ep_rew_mean` is a rolling mean over episodes finished during training, under a "
        "stochastic policy. It is a progress signal, not the evaluation result."
    )
    fig = px.line(
        curves,
        x="timesteps",
        y="ep_rew_mean",
        title="Rolling mean reward",
        color_discrete_sequence=[PALETTE["primary"]],
    )
    for sign, name in ((1, "+1 std"), (-1, "-1 std")):
        fig.add_scatter(
            x=curves["timesteps"],
            y=curves["ep_rew_mean"] + sign * curves["ep_rew_std"],
            mode="lines",
            name=name,
            line={"dash": "dot"},
        )
    _styled(fig, "Timesteps of training", "Mean reward over the last 100 episodes")
    st.plotly_chart(fig, use_container_width=True)


def _filters(frame: pd.DataFrame) -> pd.DataFrame:
    """Collect the sidebar filters and return the frame every section will use."""
    with st.sidebar:
        st.header("Filters")
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
    st.subheader("Per-episode telemetry")
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
        _styled(fig, "Total reward of the episode", "Episodes")
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
        # The pad: the band of horizontal positions that count as on target. Dashed, because
        # it is a reference and not a series, and the palette reserves the dash for exactly
        # that.
        fig.add_vline(x=-0.1, line_dash="dot", line_color=PALETTE["reference"])
        fig.add_vline(x=0.1, line_dash="dot", line_color=PALETTE["reference"])
        _styled(fig, "Horizontal position at rest", "Altitude at rest")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(frame, use_container_width=True, height=320)


def _engine_section(frame: pd.DataFrame) -> None:
    st.subheader("Engine use")
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
    _styled(fig, "Engine firings in the episode", "Mean reward")
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Performance dashboard", layout="wide")
    st.title("Performance dashboard")
    st.caption("Training and evaluation of the autopilot. The sidebar filters every panel.")

    curves, curves_problem = read_table(TRAINING_CURVES_CSV, CURVE_COLUMNS)
    episodes, episodes_problem = read_table(EVALUATION_CSV, EVALUATION_COLUMNS)

    if curves is None and episodes is None:
        st.error(
            f"Nothing to show. {curves_problem} {episodes_problem}\n\n"
            "Train with `uv run python -m rl_lander.training.train_lunarlander`, "
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
