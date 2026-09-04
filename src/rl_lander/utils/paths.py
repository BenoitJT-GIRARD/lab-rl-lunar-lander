"""Centralised filesystem paths for the project.

Every path here hangs off :data:`ROOT_DIR`, which is found rather than assumed. Counting
three directories up from this file only works from an editable ``src/`` checkout: installed
as a wheel it lands somewhere inside ``site-packages``, and the whole tree -- models, data,
logs -- would be written there without a word.
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_root() -> Path:
    """The repository root: an explicit override, the marker file, or the process's cwd."""
    override = os.environ.get("RL_LANDER_ROOT")
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    # Installed outside a checkout. The cwd is the only defensible guess, and it keeps the
    # artefacts where the user is working instead of inside site-packages.
    return Path.cwd().resolve()


#: Set ``RL_LANDER_ROOT`` to write artefacts somewhere other than the checkout.
ROOT_DIR: Path = _find_root()
SRC_DIR: Path = ROOT_DIR / "src" / "rl_lander"

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


def run_dir(algorithm: str, seed: int, variant: str | None = None) -> Path:
    """Where one training run keeps its checkpoints, its curve and its manifest.

    ``variant`` names a single-hyper-parameter change tried against the baseline. It is part
    of the directory name rather than a separate tree, so a trial and a baseline seed are
    the same kind of object and the same aggregation reads both.
    """
    name = f"seed-{seed}" if variant is None else f"seed-{seed}-{variant}"
    return MODELS_DIR / algorithm.lower() / name


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
