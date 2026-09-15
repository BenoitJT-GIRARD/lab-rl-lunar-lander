"""Score both checkpoints of a run, and publish what the difference costs.

Every training run keeps two policies: `best.zip`, the one `EvalCallback` scored highest
during training, and `final.zip`, the state training ended on. They are not the same policy,
and the first version of this repository shipped the second under the first's name.

The claim that the confusion is expensive rests on four numbers, so those four numbers are
measured here and written to `reports/checkpoint_comparison.csv` rather than quoted from a
run nobody can open.

Usage:
    uv run python scripts/compare_checkpoints.py
    uv run python scripts/compare_checkpoints.py --runs var/runs/ppo/seed-45 var/runs/dqn/seed-42
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rl_lander.agent import LunarLanderAgent
from rl_lander.artifacts import LINE_TERMINATOR
from rl_lander.training.evaluate import run_episodes, summarise
from rl_lander.utils import REPORTS_DIR, ROOT_DIR, RUNS_DIR

OUTPUT = REPORTS_DIR / "checkpoint_comparison.csv"

#: The grid the exported evaluation uses, so the two checkpoints are compared on the same
#: episodes as everything else this repository publishes.
GRID_SEED = 2024
N_EPISODES = 100

DEFAULT_RUNS = (RUNS_DIR / "ppo" / "seed-45", RUNS_DIR / "dqn" / "seed-42")


def score(checkpoint: Path, algorithm: str) -> dict:
    """The metrics of one checkpoint on the shared grid."""
    agent = LunarLanderAgent(checkpoint, algorithm=algorithm)
    records = run_episodes(agent.model, seed=GRID_SEED, n_episodes=N_EPISODES)
    return summarise(records)


def compare(runs: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for run in runs:
        algorithm = run.parent.name
        for name in ("best", "final"):
            checkpoint = run / f"{name}.zip"
            if not checkpoint.exists():
                raise SystemExit(
                    f"{checkpoint} is missing. Train the study first: "
                    "`uv run python scripts/train_study.py`."
                )
            metrics = score(checkpoint, algorithm)
            rows.append(
                {
                    "run": run.relative_to(ROOT_DIR).as_posix(),
                    "algorithm": algorithm.upper(),
                    "checkpoint": name,
                    "grid_seed": GRID_SEED,
                    "n_episodes": N_EPISODES,
                    "mean_reward": round(metrics["mean_reward"], 2),
                    "std_reward": round(metrics["std_reward"], 2),
                    "min_reward": round(metrics["min_reward"], 2),
                    "landing_rate": round(metrics["landing_rate"], 3),
                }
            )
            print(
                f"  {run.name} {name}.zip: {rows[-1]['mean_reward']}, "
                f"landing rate {rows[-1]['landing_rate']}"
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="*", default=list(DEFAULT_RUNS))
    arguments = parser.parse_args(argv)

    rows = compare(list(arguments.runs))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator=LINE_TERMINATOR)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] {OUTPUT.relative_to(ROOT_DIR).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
