"""Evaluation utilities — used by tests, the dashboard and the CLI."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from astrodynamics.training.environments import make_eval_env
from astrodynamics.utils import EVALUATION_CSV


@dataclass(slots=True)
class EpisodeRecord:
    """One row of evaluation telemetry persisted to CSV."""

    episode: int
    total_reward: float
    length: int
    landed: bool
    final_x: float
    final_y: float
    fuel_used: int  # number of main-engine firings (action == 2)


def run_episodes(
    model: BaseAlgorithm,
    n_episodes: int = 100,
    seed: int = 1234,
    deterministic: bool = True,
) -> list[EpisodeRecord]:
    """Run ``n_episodes`` and capture detailed per-episode telemetry."""
    env = make_eval_env(seed=seed)
    records: list[EpisodeRecord] = []
    try:
        for episode_idx in range(n_episodes):
            obs, _ = env.reset(seed=seed + episode_idx)
            terminated = truncated = False
            total_reward = 0.0
            length = 0
            fuel_used = 0
            final_obs = obs
            while not (terminated or truncated):
                action, _state = model.predict(obs, deterministic=deterministic)
                action_int = int(action)
                if action_int == 2:  # main engine
                    fuel_used += 1
                obs, reward, terminated, truncated, _info = env.step(action_int)
                total_reward += float(reward)
                length += 1
                final_obs = obs
            landed = bool(total_reward >= 200.0)
            records.append(
                EpisodeRecord(
                    episode=episode_idx,
                    total_reward=total_reward,
                    length=length,
                    landed=landed,
                    final_x=float(final_obs[0]),
                    final_y=float(final_obs[1]),
                    fuel_used=fuel_used,
                )
            )
    finally:
        env.close()
    return records


def summarise(records: Iterable[EpisodeRecord]) -> dict[str, float]:
    """Aggregate evaluation records into a small metrics dictionary."""
    rewards = np.array([r.total_reward for r in records])
    lengths = np.array([r.length for r in records])
    fuel = np.array([r.fuel_used for r in records])
    landed = np.array([r.landed for r in records], dtype=float)
    return {
        "n_episodes": float(len(rewards)),
        "mean_reward": float(rewards.mean()) if len(rewards) else 0.0,
        "std_reward": float(rewards.std()) if len(rewards) else 0.0,
        "min_reward": float(rewards.min()) if len(rewards) else 0.0,
        "max_reward": float(rewards.max()) if len(rewards) else 0.0,
        "median_reward": float(np.median(rewards)) if len(rewards) else 0.0,
        "mean_length": float(lengths.mean()) if len(lengths) else 0.0,
        "mean_fuel_used": float(fuel.mean()) if len(fuel) else 0.0,
        "success_rate": float(landed.mean()) if len(landed) else 0.0,
    }


def write_csv(records: Iterable[EpisodeRecord], output: Path = EVALUATION_CSV) -> Path:
    """Persist episode records to CSV (used by the dashboard)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "episode",
                "total_reward",
                "length",
                "landed",
                "final_x",
                "final_y",
                "fuel_used",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.episode,
                    f"{record.total_reward:.4f}",
                    record.length,
                    int(record.landed),
                    f"{record.final_x:.4f}",
                    f"{record.final_y:.4f}",
                    record.fuel_used,
                ]
            )
    return output
