"""Run the evaluation of the trained policy, once, and export what it produced.

One collection feeds everything: the printed number, the CSV the dashboard reads, the
summary, and the manifest that says where they came from. The first version ran two
evaluations with different reset semantics and published both — 261.4 at the root of the
JSON, 262.2 under `metrics`, standard deviations 44% apart — without designating either as
the result.

    uv run python scripts/evaluate_and_export.py
    uv run python scripts/evaluate_and_export.py --seeds 2024 7 99 123 555

Passing several seeds runs the same collection on each grid and reports the spread. A
single grid gives one draw; the mean is stable across draws and the dispersion, the worst
episode and the landing rate are not.

Writes `reports/evaluation_episodes.csv`, `reports/evaluation_summary.json` and
`reports/evaluation_manifest.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from rl_lander.training.evaluate import (
    SOLVED_THRESHOLD,
    evaluate_seeds,
    run_episodes,
    summarise,
    write_csv,
    write_manifest,
)
from rl_lander.utils import DEFAULT_MODEL_PATH, REPORTS_DIR, ensure_dirs

#: The grid whose episodes are exported. The first entry is the canonical collection: the
#: CSV, the summary and the dashboard all read it.
DEFAULT_SEEDS = (2024, 7, 99, 123, 555, 31337)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="evaluation seed grids; the first is the one whose episodes are exported",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(
            f"Model '{args.model}' not found. Train it first via "
            "`uv run python -m rl_lander.training.train_lunarlander`."
        )

    ensure_dirs()
    # CPU: this is a three-layer MLP asked for one action at a time, and the measured
    # rollout is faster there. See PPOHyperParameters.device.
    model = PPO.load(str(args.model), device="cpu")

    # The canonical collection. Everything published comes from these episodes.
    canonical_seed = args.seeds[0]
    records = run_episodes(model, n_episodes=args.episodes, seed=canonical_seed)
    metrics = summarise(records)
    csv_path = write_csv(records)

    across = (
        evaluate_seeds(model, args.seeds, n_episodes=args.episodes) if len(args.seeds) > 1 else None
    )

    summary_path = REPORTS_DIR / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "canonical_seed": canonical_seed,
                "metrics": metrics,
                "across_seeds": across,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest_path = write_manifest(
        REPORTS_DIR / "evaluation_manifest.json",
        model_path=args.model,
        algorithm="PPO",
        seeds=args.seeds,
        n_episodes=args.episodes,
        deterministic=True,
    )

    print(
        f"PPO LunarLander-v3 — mean_reward={metrics['mean_reward']:.2f} "
        f"± {metrics['std_reward']:.2f} over {args.episodes} episodes, seed {canonical_seed}"
    )
    print(
        f"  landing rate {metrics['landing_rate']:.2%}, "
        f"score above the solved threshold {metrics['threshold_rate']:.2%}"
    )
    if across:
        print(
            f"  across {len(args.seeds)} seed grids: "
            f"{across['mean_of_means']:.2f} ± {across['spread_of_means']:.2f}, "
            f"worst grid {across['worst_seed_mean']:.2f}, "
            f"worst episode {across['worst_episode']:.2f}, "
            f"lowest landing rate {across['lowest_landing_rate']:.2%}"
        )
    print(f"\n[ok] {csv_path}\n[ok] {summary_path}\n[ok] {manifest_path}")

    # The verdict, checked rather than left to a reader. The repository publishes "solves
    # LunarLander-v3", and that sentence is about a mean over 100 episodes -- on every grid
    # that was run, not only on the one whose episodes happen to be exported.
    grids = across["per_seed"] if across else [{"seed": canonical_seed, **metrics}]
    failing = [grid for grid in grids if grid["mean_reward"] < SOLVED_THRESHOLD]
    if not failing:
        print(
            f"\nSolved: every seed grid clears the {SOLVED_THRESHOLD:.0f} mean-reward "
            "threshold over 100 episodes."
        )
        return
    detail = ", ".join(f"seed {grid['seed']} at {grid['mean_reward']:.1f}" for grid in failing)
    message = f"Below the {SOLVED_THRESHOLD:.0f} threshold on {len(failing)} grid(s): {detail}."
    if args.allow_below_threshold:
        print(f"\n{message} Exported anyway, as asked.")
        return
    raise SystemExit(
        f"\n{message}\nThe exports above were written. Re-run with "
        "--allow-below-threshold to accept them, and say so wherever the number is published."
    )


if __name__ == "__main__":
    main()
