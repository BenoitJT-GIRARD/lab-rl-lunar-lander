"""Run the final evaluation of the trained Eagle-1 policy.

Invoked once the training script completes; produces:

* ``data/evaluation_episodes.csv``   — per-episode telemetry,
* ``data/evaluation_summary.json``   — aggregated metrics,
* a printout of the official 100-episode mean reward.
"""

from __future__ import annotations

import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from astrodynamics.training.environments import make_eval_env
from astrodynamics.training.evaluate import run_episodes, summarise, write_csv
from astrodynamics.utils import DATA_DIR, DEFAULT_MODEL_PATH, ensure_dirs


def main(model_path: Path = DEFAULT_MODEL_PATH, n_episodes: int = 100) -> None:
    if not model_path.exists():
        raise SystemExit(
            f"Model '{model_path}' not found. Train it first via "
            "`python -m astrodynamics.training.train_lunarlander`."
        )

    ensure_dirs()
    model = PPO.load(str(model_path), device="auto")
    eval_env = make_eval_env(seed=2024)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=n_episodes, deterministic=True
    )
    eval_env.close()

    print(
        f"PPO LunarLander-v3 — mean_reward={mean_reward:.2f} ± {std_reward:.2f} "
        f"(n_episodes={n_episodes})"
    )

    records = run_episodes(model, n_episodes=n_episodes, seed=2024)
    metrics = summarise(records)
    csv_path = write_csv(records)

    summary_path = DATA_DIR / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model": str(model_path),
                "n_episodes": n_episodes,
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Records saved to {csv_path}")
    print(f"Summary saved to {summary_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
