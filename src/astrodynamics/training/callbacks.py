"""Project-specific Stable-Baselines3 callbacks."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CsvProgressCallback(BaseCallback):
    """Persist training progress to a CSV file readable by the dashboard.

    Each row stores the rolling mean / std of the latest 100 episodes,
    the cumulated number of timesteps and the wall-clock time elapsed
    since the start of training.  The columns line up with the headers
    produced by :class:`stable_baselines3.common.logger.Logger`.
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
        self._fields = ("timesteps", "ep_rew_mean", "ep_rew_std", "ep_len_mean")

    def _on_training_start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self._fields)
        self._writer.writeheader()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_every != 0:
            return True
        ep_buffer = self.model.ep_info_buffer
        if ep_buffer is None or len(ep_buffer) == 0:
            return True
        rewards = np.array([info["r"] for info in ep_buffer])
        lengths = np.array([info["l"] for info in ep_buffer])
        assert self._writer is not None
        self._writer.writerow(
            {
                "timesteps": int(self.num_timesteps),
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
