"""Evaluation: one canonical collection, names that say what they measure.

Three things this module is careful about, because the first version was not.

**One collection.** Everything published — the headline number, the CSV, the summary, the
dashboard — comes from the same episodes. Running two loops and reporting both, as the
first version did, leaves a reader with two numbers and no way to tell which is the result.

**Landing is detected, not inferred from the score.** `landed` used to mean
``total_reward >= 200``, which is the threshold at which Gymnasium considers the
*environment solved on average* — not a statement about any single episode. The real
signal is unambiguous and comes from the environment itself: on termination it assigns
exactly ``+100`` when the lander comes to rest and exactly ``-100`` when it crashes or
leaves the frame. That is what is read here.

**One seed is not a measurement.** A single evaluation seed produces a mean, a spread and
a success rate that look like properties of the model and are properties of the draw.
:func:`evaluate_seeds` runs the same collection over several seed grids so the dispersion
can be published instead of implied.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess  # nosec B404 - one call, on a constant argv
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from astrodynamics.training.environments import make_eval_env
from astrodynamics.utils import EVALUATION_CSV

#: Discrete actions of ``LunarLander-v3``: 0 idle, 1 left engine, 2 main engine, 3 right.
MAIN_ENGINE = 2
SIDE_ENGINES = (1, 3)

#: Mean reward at which Gymnasium considers the environment solved. It is a property of the
#: task, not of an episode, and it is only ever compared against a *mean*.
SOLVED_THRESHOLD = 200.0


@dataclass(slots=True)
class EpisodeRecord:
    """One evaluated episode, with everything needed to trace it back."""

    episode: int
    #: The seed this episode was reset with. Without it a row cannot be replayed.
    seed: int
    total_reward: float
    length: int
    #: The lander came to rest. Read from the environment's terminal reward, not guessed
    #: from the score.
    landed: bool
    #: The score cleared the solved threshold. A different question, kept separate because
    #: conflating the two is what made a score look like a landing.
    meets_threshold: bool
    final_x: float
    final_y: float
    main_engine_firings: int
    side_engine_firings: int


def run_episodes(
    model: BaseAlgorithm,
    n_episodes: int = 100,
    seed: int = 1234,
    deterministic: bool = True,
) -> list[EpisodeRecord]:
    """Run ``n_episodes``, each reset with its own seed, and capture the telemetry.

    Seeding per episode rather than once means the whole collection replays exactly: the
    grid is ``seed`` through ``seed + n_episodes - 1``, and every row records which one it
    came from.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be at least 1, got {n_episodes}")

    env = make_eval_env(seed=seed)
    records: list[EpisodeRecord] = []
    try:
        for episode_idx in range(n_episodes):
            episode_seed = seed + episode_idx
            obs, _ = env.reset(seed=episode_seed)
            terminated = truncated = False
            total_reward = 0.0
            length = 0
            main = side = 0
            reward = 0.0
            final_obs = obs
            while not (terminated or truncated):
                action, _state = model.predict(obs, deterministic=deterministic)
                action_int = int(action)
                if action_int == MAIN_ENGINE:
                    main += 1
                elif action_int in SIDE_ENGINES:
                    side += 1
                obs, reward, terminated, truncated, _info = env.step(action_int)
                total_reward += float(reward)
                length += 1
                final_obs = obs

            # LunarLander assigns exactly +100 on coming to rest and -100 on crashing or
            # leaving the frame. A truncated episode ran out of time and landed nothing.
            landed = bool(terminated and float(reward) > 0.0)

            records.append(
                EpisodeRecord(
                    episode=episode_idx,
                    seed=episode_seed,
                    total_reward=total_reward,
                    length=length,
                    landed=landed,
                    meets_threshold=bool(total_reward >= SOLVED_THRESHOLD),
                    final_x=float(final_obs[0]),
                    final_y=float(final_obs[1]),
                    main_engine_firings=main,
                    side_engine_firings=side,
                )
            )
    finally:
        env.close()
    return records


def summarise(records: Sequence[EpisodeRecord]) -> dict[str, float]:
    """Aggregate a collection.

    Takes a ``Sequence`` rather than an ``Iterable`` on purpose: the body walks it several
    times, and a generator would be silently empty after the first pass.

    ``landing_rate`` and ``threshold_rate`` are both reported because they answer different
    questions and, on a good policy, do not agree.
    """
    records = list(records)
    if not records:
        raise ValueError("nothing to summarise: the collection is empty")

    rewards = np.array([r.total_reward for r in records])
    lengths = np.array([r.length for r in records])
    main = np.array([r.main_engine_firings for r in records])
    side = np.array([r.side_engine_firings for r in records])

    return {
        "n_episodes": float(len(records)),
        "mean_reward": float(rewards.mean()),
        # Sample standard deviation, ddof=1. The hundred episodes are a sample of the
        # initial conditions, not the population of them, and the dashboard's pandas
        # default is the same — two views of one collection must not disagree. One episode
        # has no dispersion to estimate, and nan says that rather than pretending to zero.
        "std_reward": float(rewards.std(ddof=1)) if len(rewards) > 1 else float("nan"),
        "min_reward": float(rewards.min()),
        "max_reward": float(rewards.max()),
        "median_reward": float(np.median(rewards)),
        "mean_length": float(lengths.mean()),
        "mean_main_engine_firings": float(main.mean()),
        "mean_side_engine_firings": float(side.mean()),
        "landing_rate": float(np.mean([r.landed for r in records])),
        "threshold_rate": float(np.mean([r.meets_threshold for r in records])),
    }


def evaluate_seeds(
    model: BaseAlgorithm,
    seeds: Sequence[int],
    n_episodes: int = 100,
    deterministic: bool = True,
) -> dict[str, object]:
    """Run the same collection on several seed grids, and report the spread.

    What a single grid gives is one draw. The mean is usually stable across grids; the
    dispersion, the worst episode and the success rate are not, and publishing the first
    without the others is what makes an evaluation look tighter than it is.
    """
    per_seed = []
    for seed in seeds:
        summary = summarise(run_episodes(model, n_episodes, seed, deterministic))
        per_seed.append({"seed": int(seed), **summary})

    means = np.array([s["mean_reward"] for s in per_seed])
    return {
        "seeds": [int(s) for s in seeds],
        "n_episodes": n_episodes,
        "per_seed": per_seed,
        "mean_of_means": float(means.mean()),
        "spread_of_means": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
        "worst_seed_mean": float(means.min()),
        "worst_episode": float(min(s["min_reward"] for s in per_seed)),
        "lowest_landing_rate": float(min(s["landing_rate"] for s in per_seed)),
    }


FIELDS = (
    "episode",
    "seed",
    "total_reward",
    "length",
    "landed",
    "meets_threshold",
    "final_x",
    "final_y",
    "main_engine_firings",
    "side_engine_firings",
)


def write_csv(records: Sequence[EpisodeRecord], output: Path = EVALUATION_CSV) -> Path:
    """Persist a collection. Every row carries the seed that produced it."""
    records = list(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for record in records:
            row = asdict(record)
            writer.writerow(
                [
                    row["episode"],
                    row["seed"],
                    f"{row['total_reward']:.4f}",
                    row["length"],
                    int(row["landed"]),
                    int(row["meets_threshold"]),
                    f"{row['final_x']:.4f}",
                    f"{row['final_y']:.4f}",
                    row["main_engine_firings"],
                    row["side_engine_firings"],
                ]
            )
    return output


def _git_revision() -> str:
    """The commit an evaluation was produced from, or "unknown" outside a checkout."""
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        return subprocess.check_output(  # nosec B603 - absolute path, constant argv
            [git, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()[:12]
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def write_manifest(
    output: Path,
    *,
    model_path: Path,
    algorithm: str,
    seeds: Sequence[int],
    n_episodes: int,
    deterministic: bool,
) -> Path:
    """Write what a published number needs in order to be traced back.

    The first version published an absolute path to the author's own working folder and
    nothing else — no seed, no version, no date. A figure without a manifest is a figure
    nobody can check.
    """
    import gymnasium
    import stable_baselines3

    # Relative to the repository, so the manifest says which artefact rather than which
    # machine.
    try:
        model = model_path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        model = Path(model_path.name)

    payload = {
        "model": model.as_posix(),
        "algorithm": algorithm,
        "n_episodes": n_episodes,
        "seeds": [int(s) for s in seeds],
        "deterministic": deterministic,
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": _git_revision(),
        "versions": {
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "numpy": np.__version__,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


__all__ = [
    "FIELDS",
    "MAIN_ENGINE",
    "SIDE_ENGINES",
    "SOLVED_THRESHOLD",
    "EpisodeRecord",
    "evaluate_seeds",
    "run_episodes",
    "summarise",
    "write_csv",
    "write_manifest",
]
