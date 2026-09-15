"""Project-specific Stable-Baselines3 callbacks."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rl_lander.artifacts import LINE_TERMINATOR


class CsvProgressCallback(BaseCallback):
    """Persist training progress to a CSV the dashboard can plot.

    Each row is the rolling mean and standard deviation over the episodes SB3 keeps in
    ``ep_info_buffer`` (the last 100 finished episodes), the cumulative timestep count and
    the seconds elapsed since training started.

    The sampling interval is a *distance*, not a divisibility test. ``num_timesteps``
    advances by ``n_envs`` on every step, so ``num_timesteps % 1000 == 0`` never fired at
    all for 16 environments -- rows landed every 2000 steps, the first common multiple, and
    the interval silently depended on a parameter that has nothing to do with logging.
    """

    def __init__(
        self,
        output_path: Path | str,
        check_every: int = 1_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.output_path = Path(output_path)
        self.check_every = check_every
        self._writer: csv.DictWriter | None = None
        self._fh = None
        self._next_at = 0
        self._started_at = 0.0
        self._fields = ("timesteps", "elapsed_seconds", "ep_rew_mean", "ep_rew_std", "ep_len_mean")

    def _on_training_start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._fh, fieldnames=self._fields, lineterminator=LINE_TERMINATOR
        )
        self._writer.writeheader()
        self._next_at = self.check_every
        self._started_at = time.monotonic()

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_at:
            return True
        # Skip whole intervals rather than one, so a large n_envs cannot leave the schedule
        # permanently behind the run.
        self._next_at = (self.num_timesteps // self.check_every + 1) * self.check_every

        ep_buffer = self.model.ep_info_buffer
        if ep_buffer is None or len(ep_buffer) == 0:
            return True
        rewards = np.array([info["r"] for info in ep_buffer])
        lengths = np.array([info["l"] for info in ep_buffer])
        assert self._writer is not None
        self._writer.writerow(
            {
                "timesteps": int(self.num_timesteps),
                "elapsed_seconds": round(time.monotonic() - self._started_at, 3),
                "ep_rew_mean": float(rewards.mean()),
                "ep_rew_std": float(rewards.std()),
                "ep_len_mean": float(lengths.mean()),
            }
        )
        if self._fh is not None:
            self._fh.flush()
        return True

    def _on_training_end(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None
