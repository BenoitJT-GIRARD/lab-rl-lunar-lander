"""Centralised filesystem paths for the project."""

from __future__ import annotations

from pathlib import Path

# Project anchored at the parent of the ``src`` directory.
ROOT_DIR: Path = Path(__file__).resolve().parents[3]
SRC_DIR: Path = ROOT_DIR / "src" / "astrodynamics"

MODELS_DIR: Path = ROOT_DIR / "models"
DATA_DIR: Path = ROOT_DIR / "data"
LOGS_DIR: Path = ROOT_DIR / "logs"
VIDEOS_DIR: Path = ROOT_DIR / "videos"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"
TENSORBOARD_DIR: Path = LOGS_DIR / "tensorboard"

DEFAULT_MODEL_PATH: Path = MODELS_DIR / "ppo_lunarlander_best.zip"
EVALUATION_CSV: Path = DATA_DIR / "evaluation_episodes.csv"
TRAINING_CURVES_CSV: Path = DATA_DIR / "training_curves.csv"


def ensure_dirs() -> None:
    """Create every runtime directory if it does not exist yet."""
    for path in (
        MODELS_DIR,
        DATA_DIR,
        LOGS_DIR,
        VIDEOS_DIR,
        TENSORBOARD_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
