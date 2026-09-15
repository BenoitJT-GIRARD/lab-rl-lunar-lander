"""Draw the three figures the README publishes, from the artefacts of the study.

Each one reads a tracked file under `reports/`, so a picture and a number on the same page
come from the same run. None is edited by hand, and regenerating all three is one command.

    uv run python scripts/build_figures.py

Colours come from `rl_lander.figure_style`, and the writer is the same module: it refuses a
figure whose axes say nothing, or one that draws a band without naming what the band covers.
That refusal is why the three dispersions of this repository can be drawn at all. Every one
of them is a spread of rewards in the same unit, and a reader told only « ± » would have no
way to tell which of the three a bar is.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rl_lander.figure_style import (
    PALETTE,
    apply_style,
    close,
    reference_line,
    save_figure,
)
from rl_lander.training.evaluate import SOLVED_THRESHOLD
from rl_lander.utils import FIGURES_DIR, REPORTS_DIR, ensure_dirs

SOURCE = "scripts/build_figures.py"

SEED_STUDY = REPORTS_DIR / "seed_study.json"
LEARNING_BAND = REPORTS_DIR / "learning_curve_band.csv"
EVALUATION_SUMMARY = REPORTS_DIR / "evaluation_summary.json"


def _breathe(fig) -> None:
    """Leave the stamp its line.

    `save_figure` writes the effective and the dispersion at the very bottom of the figure,
    and `bbox_inches="tight"` then crops to whatever is drawn. Without this, that line lands
    on top of the x-axis label.
    """
    fig.tight_layout()
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + 0.08)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"{path} is missing — run scripts/aggregate_study.py first")
    return json.loads(path.read_text(encoding="utf-8"))


# --- 1. The five runs, and the two spreads that are not the same ------------


def figure_seed_spread(study: dict) -> Path:
    """Five trainings, each with its own episode spread, against the spread between them.

    The figure exists because those two are read as one. A bar is one run's dispersion over
    its hundred evaluation episodes; the band behind them is the dispersion of the five run
    means, which is the quantity a reader is actually being told about when a repository
    publishes « the model scores X ± Y ».
    """
    runs = sorted(study["per_run"], key=lambda run: run["seed"])
    seeds = [run["seed"] for run in runs]
    means = [run["mean_reward"] for run in runs]
    spreads = [run["std_reward"] for run in runs]
    centre = study["mean_of_runs"]
    between = study["spread_between_runs"]

    fig, ax = plt.subplots()
    positions = range(len(runs))
    ax.axvspan(
        centre - between,
        centre + between,
        color=PALETTE["tertiary"],
        alpha=0.18,
        label=f"±{between:.2f} between runs",
    )
    ax.axvline(centre, color=PALETTE["primary"], linewidth=1.2, label=f"mean of runs {centre:.2f}")
    ax.errorbar(
        means,
        list(positions),
        xerr=spreads,
        fmt="o",
        color=PALETTE["primary"],
        ecolor=PALETTE["muted"],
        elinewidth=1.4,
        capsize=4,
        markersize=6,
        linestyle="none",
        label="run mean, ±1 SD across episodes",
    )
    reference_line(ax, x=SOLVED_THRESHOLD, label=f"solved, {SOLVED_THRESHOLD:.0f}")

    ax.set_yticks(list(positions))
    ax.set_yticklabels([f"seed {seed}" for seed in seeds])
    ax.invert_yaxis()
    ax.set_xlabel("Mean reward per episode")
    ax.set_ylabel("Training run")
    ax.set_title("Five identical trainings, and the two dispersions they carry")
    # Outside the axes: five rows of error bars leave no corner a four-entry legend fits in.
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    _breathe(fig)

    path = save_figure(
        fig,
        FIGURES_DIR / "seed_spread.png",
        n={"runs": study["n_runs"], "episodes per run": study["evaluation"]["n_episodes"]},
        dispersion=(
            f"bars: ±1 SD across {study['evaluation']['n_episodes']} evaluation episodes; "
            f"band: ±{between:.2f}, the SD of the {study['n_runs']} run means"
        ),
        source=SOURCE,
    )
    close(fig)
    return path


# --- 2. What the five runs did on the way there -----------------------------


def figure_learning_curve(band: pd.DataFrame, study: dict) -> Path:
    """The median of the five runs, with the interquartile band around it.

    A single run's curve is the one figure every reinforcement-learning README carries, and
    it is the one a reader cannot check. Five drawn as a band say how much of the shape is
    the method and how much is the seed.
    """
    fig, ax = plt.subplots()
    steps = band["timesteps"] / 1e6
    ax.fill_between(
        steps,
        band["q25"],
        band["q75"],
        color=PALETTE["primary"],
        alpha=0.20,
        label="interquartile range",
    )
    ax.plot(
        steps,
        band["median"],
        color=PALETTE["primary"],
        linewidth=1.8,
        label="median of the runs",
    )
    reference_line(ax, y=SOLVED_THRESHOLD, label=f"solved, {SOLVED_THRESHOLD:.0f}")

    ax.set_xlabel("Environment steps, in millions")
    ax.set_ylabel("Episode reward, rolling mean")
    ax.set_title("Training, read across five seeds rather than one")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    _breathe(fig)

    path = save_figure(
        fig,
        FIGURES_DIR / "learning_curve.png",
        n={"runs": int(band["n_runs"].min()), "points": len(band)},
        dispersion=(
            f"band: 25th to 75th percentile across the {study['n_runs']} runs, at each checkpoint"
        ),
        source=SOURCE,
        note="training episodes, actions sampled; the published score is a deterministic run",
    )
    close(fig)
    return path


# --- 3. The three dispersions, side by side ---------------------------------


def figure_dispersions(study: dict, summary: dict) -> Path:
    """The same « ± » computed three ways, on the same axis, in the same unit.

    Nothing about the shipped policy changes between the three bars. What changes is what was
    allowed to vary while it was measured: the episode, the evaluation grid, or the training
    itself. The tallest bar is the one a single-run report never shows.
    """
    across = summary["across_seeds"]
    bars = [
        (
            "Between episodes",
            summary["metrics"]["std_reward"],
            f"n = {int(summary['metrics']['n_episodes'])} episodes, one grid",
        ),
        (
            "Between evaluation grids",
            across["spread_of_means"],
            f"n = {len(across['seeds'])} grids, one policy",
        ),
        (
            "Between trainings",
            study["spread_between_runs"],
            f"n = {study['n_runs']} runs, one configuration",
        ),
    ]

    fig, ax = plt.subplots()
    labels = [f"{label}\n{detail}" for label, _, detail in bars]
    values = [value for _, value, _ in bars]
    ax.bar(
        labels,
        values,
        color=[PALETTE["control"], PALETTE["control"], PALETTE["secondary"]],
        width=0.55,
    )
    # The number on the bar, because a reader comparing three heights should not have to
    # walk back to the axis for the one that matters.
    for index, value in enumerate(values):
        ax.text(index, value + 0.6, f"{value:.2f}", ha="center", fontsize=10, color=PALETTE["ink"])

    ax.set_xlabel("What was allowed to vary")
    ax.set_ylabel("Standard deviation of the mean reward")
    ax.set_title("Three dispersions a ± can describe, on one policy")
    ax.tick_params(axis="x", labelsize=8)
    _breathe(fig)

    path = save_figure(
        fig,
        FIGURES_DIR / "dispersions.png",
        n={
            "episodes": int(summary["metrics"]["n_episodes"]),
            "grids": len(across["seeds"]),
            "runs": study["n_runs"],
        },
        dispersion="each bar is itself a standard deviation; none is an error bar on another",
        source=SOURCE,
        note="sample SD, ddof = 1, in all three",
    )
    close(fig)
    return path


def main() -> int:
    ensure_dirs()
    apply_style()

    study = _read_json(SEED_STUDY)
    summary = _read_json(EVALUATION_SUMMARY)
    if not LEARNING_BAND.is_file():
        raise SystemExit(f"{LEARNING_BAND} is missing — run scripts/aggregate_study.py first")
    band = pd.read_csv(LEARNING_BAND)

    written = [
        figure_seed_spread(study),
        figure_learning_curve(band, study),
        figure_dispersions(study, summary),
    ]
    for path in written:
        print(f"[ok] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
