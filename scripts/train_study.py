"""Train the whole study: five baseline seeds, three single-parameter trials, one DQN.

Three questions, one grid, because they share the same machinery and only mean anything
read together.

**How much of the published number is the training seed?** Five PPO runs, identical
hyper-parameters, seeds 42 to 46. The repository used to publish one run's score as if it
were the method's.

**Are the notebook's hyper-parameter trials real?** They were quoted as "~280", "< 200",
"~270" with nothing behind them. Each trial changes one parameter from the baseline, runs
the same budget, and is scored by the same protocol. One seed each, so a trial is only
readable against the seed-to-seed spread the five baselines measure -- which is why they
belong in the same grid.

**Is the DQN baseline real?** The README claimed one and the repository had none. It is
trained here at an *equal step budget*: a comparison at unequal budgets measures the
budget.

Runs that already carry a manifest are skipped, so the grid resumes after an interruption.

    uv run python scripts/train_study.py
    uv run python scripts/train_study.py --only ppo-seed-42 dqn-seed-42
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from astrodynamics.training.hyperparameters import (  # noqa: E402
    DQNHyperParameters,
    PPOHyperParameters,
)
from astrodynamics.training.train_lunarlander import train_dqn, train_ppo  # noqa: E402
from astrodynamics.utils import run_dir  # noqa: E402

#: The seeds whose spread is published as the method's variability.
BASELINE_SEEDS = (42, 43, 44, 45, 46)

#: One parameter changed at a time, against the baseline, at the same budget. The keys are
#: the rows of the notebook's table, so its unsourced numbers can be replaced one for one.
TRIALS = {
    "lr1e-3": {"learning_rate": 1e-3},
    "nsteps2048": {"n_steps": 2048},
    "gamma099": {"gamma": 0.99},
}

#: Equal step budget for DQN. Its own default is 200_000, which would turn the comparison
#: into a statement about budget rather than about algorithm.
DQN_TIMESTEPS = PPOHyperParameters.total_timesteps


def _plan() -> list[tuple[str, Path, object]]:
    """The grid, as ``(name, run directory, thunk)``. Nothing trains until a thunk runs."""
    jobs: list[tuple[str, Path, object]] = []
    for seed in BASELINE_SEEDS:
        hp = PPOHyperParameters(seed=seed)
        run = run_dir("ppo", seed)
        jobs.append((f"ppo-seed-{seed}", run, lambda hp=hp, run=run: train_ppo(hp, output=run)))
    for variant, change in TRIALS.items():
        hp = replace(PPOHyperParameters(seed=BASELINE_SEEDS[0]), **change)
        run = run_dir("ppo", hp.seed, variant)
        jobs.append((f"ppo-{variant}", run, lambda hp=hp, run=run: train_ppo(hp, output=run)))
    hp_dqn = DQNHyperParameters(seed=BASELINE_SEEDS[0], total_timesteps=DQN_TIMESTEPS)
    run = run_dir("dqn", hp_dqn.seed)
    jobs.append(("dqn-seed-42", run, lambda hp=hp_dqn, run=run: train_dqn(hp, output=run)))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", default=None, help="run just these job names")
    parser.add_argument("--list", action="store_true", help="print the grid and stop")
    parser.add_argument(
        "--force", action="store_true", help="retrain runs that already have a manifest"
    )
    args = parser.parse_args()

    jobs = _plan()
    if args.list:
        for name, run, _ in jobs:
            print(f"{name}\t{run.relative_to(ROOT)}")
        return
    if args.only:
        known = {name for name, _, _ in jobs}
        unknown = set(args.only) - known
        if unknown:
            raise SystemExit(f"unknown job(s): {', '.join(sorted(unknown))}")
        jobs = [job for job in jobs if job[0] in args.only]

    for index, (name, run, thunk) in enumerate(jobs, start=1):
        if not args.force and (run / "manifest.json").exists():
            print(f"[{index}/{len(jobs)}] {name}: already trained, skipping", flush=True)
            continue
        print(f"\n=== [{index}/{len(jobs)}] {name} ===", flush=True)
        started = time.monotonic()
        _, metrics = thunk()
        elapsed = time.monotonic() - started
        print(
            f"{name}: mean_reward {metrics['mean_reward']:.2f} "
            f"+/- {metrics['std_reward']:.2f}, landing rate {metrics['landing_rate']:.0%}, "
            f"{elapsed / 60:.1f} min",
            flush=True,
        )


if __name__ == "__main__":
    main()
