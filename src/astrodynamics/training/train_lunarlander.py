"""Eagle-1 mission — training pipeline for LunarLander-v3.

The script supports both PPO and DQN through a single CLI.  The default
configuration follows the SB3 RL Zoo recipe for ``LunarLander-v3`` and is
expected to comfortably exceed the +200 reward threshold on the 100
evaluation-episode horizon set by the project brief.

Usage
-----
.. code-block:: powershell

    uv run python -m astrodynamics.training.train_lunarlander \
        --algo ppo --timesteps 1_000_000 --output models/ppo_lunarlander_best.zip
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy

from astrodynamics.training.callbacks import CsvProgressCallback
from astrodynamics.training.environments import (
    make_eval_env,
    make_train_env,
)
from astrodynamics.training.hyperparameters import (
    DQNHyperParameters,
    PPOHyperParameters,
)
from astrodynamics.utils import (
    DEFAULT_MODEL_PATH,
    LOGS_DIR,
    MODELS_DIR,
    TENSORBOARD_DIR,
    TRAINING_CURVES_CSV,
    ensure_dirs,
    set_global_seed,
)


def train_ppo(
    hp: PPOHyperParameters | None = None,
    output: Path = DEFAULT_MODEL_PATH,
    tensorboard_run_name: str = "ppo_lunarlander",
) -> tuple[PPO, dict]:
    """Train a PPO agent end-to-end and return the policy + final metrics."""
    hp = hp or PPOHyperParameters()
    ensure_dirs()
    set_global_seed(hp.seed)

    train_env = make_train_env(n_envs=hp.n_envs, seed=hp.seed)
    eval_env = make_train_env(n_envs=4, seed=hp.seed + 999)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=str(TENSORBOARD_DIR),
        device="auto",
        **hp.to_kwargs(),
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(output.parent),
        log_path=str(LOGS_DIR / "eval"),
        eval_freq=max(10_000 // hp.n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // hp.n_envs, 1),
        save_path=str(MODELS_DIR / "checkpoints" / "ppo"),
        name_prefix="ppo_lunarlander",
    )
    progress_callback = CsvProgressCallback(output_path=TRAINING_CURVES_CSV)

    model.learn(
        total_timesteps=hp.total_timesteps,
        callback=CallbackList([eval_callback, checkpoint_callback, progress_callback]),
        tb_log_name=tensorboard_run_name,
        progress_bar=False,
    )

    model.save(output)
    eval_single = make_eval_env(seed=hp.seed + 1)
    mean_reward, std_reward = evaluate_policy(model, eval_single, n_eval_episodes=100)
    eval_single.close()
    train_env.close()
    eval_env.close()

    metrics = {
        "algorithm": "PPO",
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "total_timesteps": hp.total_timesteps,
        "hyperparameters": asdict(hp),
    }
    return model, metrics


def train_dqn(
    hp: DQNHyperParameters | None = None,
    output: Path = MODELS_DIR / "dqn_lunarlander.zip",
    tensorboard_run_name: str = "dqn_lunarlander",
) -> tuple[DQN, dict]:
    """Train a DQN agent on LunarLander-v3."""
    hp = hp or DQNHyperParameters()
    ensure_dirs()
    set_global_seed(hp.seed)

    train_env = make_eval_env(seed=hp.seed)
    eval_env = make_eval_env(seed=hp.seed + 1)

    model = DQN(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=str(TENSORBOARD_DIR),
        device="auto",
        **hp.to_kwargs(),
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(output.parent),
        log_path=str(LOGS_DIR / "eval_dqn"),
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    progress_callback = CsvProgressCallback(
        output_path=TRAINING_CURVES_CSV.with_name("training_curves_dqn.csv")
    )

    model.learn(
        total_timesteps=hp.total_timesteps,
        callback=CallbackList([eval_callback, progress_callback]),
        tb_log_name=tensorboard_run_name,
        progress_bar=False,
    )

    model.save(output)
    eval_single = make_eval_env(seed=hp.seed + 2)
    mean_reward, std_reward = evaluate_policy(model, eval_single, n_eval_episodes=100)
    eval_single.close()
    train_env.close()
    eval_env.close()

    metrics = {
        "algorithm": "DQN",
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "total_timesteps": hp.total_timesteps,
        "hyperparameters": asdict(hp),
    }
    return model, metrics


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eagle-1 LunarLander training")
    parser.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the saved model (zip).",
    )
    parser.add_argument("--n-envs", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI helper
    args = _parse_args(argv)
    if args.algo == "ppo":
        hp = PPOHyperParameters(
            seed=args.seed,
            total_timesteps=args.timesteps or PPOHyperParameters.total_timesteps,
            n_envs=args.n_envs or PPOHyperParameters.n_envs,
        )
        _, metrics = train_ppo(hp, output=args.output)
    else:
        hp = DQNHyperParameters(
            seed=args.seed,
            total_timesteps=args.timesteps or DQNHyperParameters.total_timesteps,
        )
        _, metrics = train_dqn(hp, output=args.output)

    print(
        f"\n[{metrics['algorithm']}] mean_reward={metrics['mean_reward']:.2f} "
        f"+/- {metrics['std_reward']:.2f} on 100 eval episodes."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
