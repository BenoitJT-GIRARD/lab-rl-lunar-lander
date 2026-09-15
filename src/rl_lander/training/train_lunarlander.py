"""Training pipeline for LunarLander-v3, PPO and DQN behind one CLI.

Each run gets its own directory, ``models/<algo>/seed-<n>/``, holding the checkpoint
``EvalCallback`` kept (``best.zip``), the state training ended on (``final.zip``) and a
manifest with the metrics that checkpoint scored. Three things follow from that layout,
and all three were wrong in the first version.

**Two algorithms cannot overwrite each other.** Both used to point ``EvalCallback`` at
``models/``, so whichever ran second replaced the other's ``best_model.zip``.

**The best checkpoint is what gets shipped and scored.** The first version evaluated the
object left in memory when training stopped, and saved it under a name that said ``best``.
In reinforcement learning the two differ: performance oscillates late in training, and the
gap is not small. When no evaluation ever improved on the start there is no best
checkpoint, and the manifest says which of the two it is.

**A model on disk carries the metrics it scored.** They used to be printed and lost.

Usage
-----
.. code-block:: powershell

    uv run python -m rl_lander.training.train_lunarlander --algo ppo --seed 42
    uv run python -m rl_lander.training.train_lunarlander --algo dqn --timesteps 500_000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from rl_lander.training.callbacks import CsvProgressCallback
from rl_lander.training.environments import (
    make_eval_env,
    make_train_env,
)
from rl_lander.training.evaluate import run_episodes, summarise
from rl_lander.training.hyperparameters import (
    DQNHyperParameters,
    PPOHyperParameters,
)
from rl_lander.utils import (
    LOGS_DIR,
    TENSORBOARD_DIR,
    ensure_dirs,
    run_dir,
    set_global_seed,
)

#: The evaluation grid every training run is scored on, so two runs are comparable.
CANONICAL_EVAL_SEED = 2024


def _score(model, *, algorithm: str, hp, best_is_final: bool) -> dict:
    """Evaluate a trained policy the same way for every algorithm.

    Uses the project's own collection, and never `evaluate_policy`, so a training metric
    and a published metric are computed by the same code on the same seed grid. The first
    version used one loop here and another in the exporter, and the two disagreed.
    """
    records = run_episodes(model, n_episodes=100, seed=CANONICAL_EVAL_SEED)
    summary = summarise(records)
    return {
        "algorithm": algorithm,
        "seed": hp.seed,
        "total_timesteps": hp.total_timesteps,
        "evaluation_seed": CANONICAL_EVAL_SEED,
        # True when no evaluation ever improved on the start, so `best` and `final` are the
        # same file. Worth recording: it means the callback never fired usefully.
        "best_is_final": best_is_final,
        **summary,
        "hyperparameters": asdict(hp),
    }


def _write_run_manifest(run: Path, metrics: dict) -> Path:
    """Persist a run's metrics beside its checkpoints.

    The CLI used to print them and stop there, so a model on disk was attached to nothing.
    """
    target = run / "manifest.json"
    target.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return target


def train_ppo(
    hp: PPOHyperParameters | None = None,
    output: Path | None = None,
    tensorboard_run_name: str = "ppo_lunarlander",
) -> tuple[PPO, dict]:
    """Train PPO, keep both checkpoints, and return the **best** one with its metrics.

    ``output`` is the run directory. It holds ``best.zip`` -- the checkpoint
    ``EvalCallback`` kept -- and ``final.zip``, the state training ended on. The two are
    not the same policy, and returning the second under the first's name is what the
    original version did. When no evaluation ever improved on the start there is no best
    checkpoint: the final state becomes ``best.zip``, no ``final.zip`` is left beside it,
    and the manifest records ``best_is_final`` so nothing downstream has to guess.
    """
    hp = hp or PPOHyperParameters()
    ensure_dirs()
    set_global_seed(hp.seed)
    run = output or run_dir("ppo", hp.seed)
    run.mkdir(parents=True, exist_ok=True)

    train_env = make_train_env(n_envs=hp.n_envs, seed=hp.seed)
    eval_env = make_train_env(n_envs=4, seed=hp.seed + 999)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=str(TENSORBOARD_DIR),
        **hp.to_kwargs(),
    )

    # Its own directory, so a DQN run cannot overwrite a PPO run's best checkpoint.
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(run),
        log_path=str(LOGS_DIR / "eval" / f"ppo-seed-{hp.seed}"),
        eval_freq=max(10_000 // hp.n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )
    # In the run's own directory, like every other artefact it produces. The published
    # curve under data/ is the retained run's, copied there by scripts/publish_run.py.
    progress_callback = CsvProgressCallback(output_path=run / "training_curves.csv")

    model.learn(
        total_timesteps=hp.total_timesteps,
        callback=CallbackList([eval_callback, progress_callback]),
        tb_log_name=tensorboard_run_name,
        progress_bar=False,
    )

    model.save(run / "final.zip")
    train_env.close()
    eval_env.close()

    # Evaluate the checkpoint that will be shipped, not the object left in memory.
    best_path = run / "best_model.zip"
    if best_path.exists():
        best_path.replace(run / "best.zip")
    kept = run / "best.zip"
    if not kept.exists():
        # No evaluation ever improved on the start, so there is no best checkpoint. Say so
        # where the earlier version shipped the final model under the other name in silence.
        (run / "final.zip").replace(kept)
        best_is_final = True
    else:
        best_is_final = False

    evaluated = PPO.load(str(kept), device=hp.device)
    metrics = _score(evaluated, algorithm="PPO", hp=hp, best_is_final=best_is_final)
    _write_run_manifest(run, metrics)
    return evaluated, metrics


def train_dqn(
    hp: DQNHyperParameters | None = None,
    output: Path | None = None,
    tensorboard_run_name: str = "dqn_lunarlander",
) -> tuple[DQN, dict]:
    """Train DQN, under the same convention as :func:`train_ppo`."""
    hp = hp or DQNHyperParameters()
    ensure_dirs()
    set_global_seed(hp.seed)
    run = output or run_dir("dqn", hp.seed)
    run.mkdir(parents=True, exist_ok=True)

    # `n_envs` was a declared field nothing read: DQN trained on a bare single environment
    # while the dataclass said otherwise. Same factory as PPO, so both algorithms see the
    # same wrappers and the field means what it says.
    train_env = make_train_env(n_envs=hp.n_envs, seed=hp.seed)
    eval_env = make_eval_env(seed=hp.seed + 1)

    model = DQN(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=str(TENSORBOARD_DIR),
        **hp.to_kwargs(),
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(run),
        log_path=str(LOGS_DIR / "eval" / f"dqn-seed-{hp.seed}"),
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    progress_callback = CsvProgressCallback(output_path=run / "training_curves.csv")

    model.learn(
        total_timesteps=hp.total_timesteps,
        callback=CallbackList([eval_callback, progress_callback]),
        tb_log_name=tensorboard_run_name,
        progress_bar=False,
    )

    model.save(run / "final.zip")
    train_env.close()
    eval_env.close()

    best_path = run / "best_model.zip"
    if best_path.exists():
        best_path.replace(run / "best.zip")
    kept = run / "best.zip"
    if not kept.exists():
        (run / "final.zip").replace(kept)
        best_is_final = True
    else:
        best_is_final = False

    evaluated = DQN.load(str(kept), device=hp.device)
    metrics = _score(evaluated, algorithm="DQN", hp=hp, best_is_final=best_is_final)
    _write_run_manifest(run, metrics)
    return evaluated, metrics


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LunarLander training")
    parser.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Run directory. Defaults to models/<algo>/seed-<seed>/, chosen after --algo is "
            "read -- the default used to be the PPO path whatever the algorithm, so a DQN "
            "run without --output overwrote the PPO artefact."
        ),
    )
    parser.add_argument("--n-envs", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI helper
    args = _parse_args(argv)
    # Only the flags actually given are passed, so each dataclass supplies its own
    # defaults. Writing `args.timesteps or PPOHyperParameters.total_timesteps` looks like
    # the same thing and is not: these dataclasses use `slots=True`, and on a slotted
    # dataclass the class attribute is the slot descriptor and not the default. The
    # fallback silently produced a `member_descriptor`, which SB3 then compared against an
    # integer, deep inside its training loop.
    overrides = {}
    if args.timesteps is not None:
        overrides["total_timesteps"] = args.timesteps
    if args.algo == "ppo":
        if args.n_envs is not None:
            overrides["n_envs"] = args.n_envs
        _, metrics = train_ppo(PPOHyperParameters(seed=args.seed, **overrides), output=args.output)
    else:
        _, metrics = train_dqn(DQNHyperParameters(seed=args.seed, **overrides), output=args.output)

    run = args.output or run_dir(args.algo, args.seed)
    print(
        f"\n[{metrics['algorithm']}] mean_reward={metrics['mean_reward']:.2f} "
        f"+/- {metrics['std_reward']:.2f}, landing rate {metrics['landing_rate']:.0%}, "
        f"over {int(metrics['n_episodes'])} episodes on evaluation seed "
        f"{metrics['evaluation_seed']}."
    )
    if metrics["best_is_final"]:
        print(
            "  No evaluation improved on the starting policy, so best.zip is the "
            "final state. Train longer before reading anything into the score."
        )
    print(f"  {run / 'best.zip'}\n  {run / 'manifest.json'}")


if __name__ == "__main__":  # pragma: no cover
    main()
