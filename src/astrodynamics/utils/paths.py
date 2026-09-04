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

#: The shipped policy: the best checkpoint of the retained PPO run.
#:
#: One directory per algorithm, and `best` and `final` under different names. The first
#: version pointed `EvalCallback` at `models/` for both algorithms, so each overwrote the
#: other's `best_model.zip`, then saved the *last* model under a filename that said `best`
#: -- and evaluated that. In reinforcement learning the two differ, because performance
#: oscillates at the end of training.
DEFAULT_MODEL_PATH: Path = MODELS_DIR / "ppo" / "best.zip"


def run_dir(algorithm: str, seed: int) -> Path:
    """Where one training run keeps its checkpoints and its manifest."""
    return MODELS_DIR / algorithm.lower() / f"seed-{seed}"


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
