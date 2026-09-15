"""Where this project's files are. One module answers, and nothing else computes a root.

Every path hangs off :data:`ROOT_DIR`, which is **found**, never assumed. Counting
directories up from a file only works from an editable ``src/`` checkout: installed as a
wheel, the package lands inside ``site-packages``, and the whole tree — models, reports,
figures — would be written there without a word. Finding the marker file instead means the
same code works from a checkout, from a wheel and from a container.

Nothing else in the project resolves a root. No ``sys.path.insert``, no ``Path(__file__)``
walked back three times in a script, no artefact read or written through a path relative to
the working directory: each of those is a second answer to a question that already has one,
and they disagree the day someone runs a script from another directory.

The directory names are the closed vocabulary shared by every repository of the portfolio.
A project adds its own *named artefacts* below — the served model, the published table — and
never a new root directory: a directory outside the vocabulary exists only when
``targets.yaml`` declares it with the technical reason that imposes it.
"""

from __future__ import annotations

import os
from pathlib import Path


def _package_name() -> str:
    """The distribution package this module belongs to, read from the import system.

    The same file is copied into every project of the portfolio, so it must not name one.
    """
    if __package__:
        return __package__.split(".")[0]
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if candidate.parent.name == "src":
            return candidate.name
    return here.parent.name


#: The package name, and the environment variable that overrides the root: ``<PACKAGE>_ROOT``.
PACKAGE: str = _package_name()
ROOT_ENV: str = f"{PACKAGE.upper()}_ROOT"


def _find_root() -> Path:
    """An explicit override, the marker file, or — installed outside a checkout — the cwd."""
    override = os.environ.get(ROOT_ENV)
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    # Installed outside a checkout. The working directory is the only defensible guess, and
    # it keeps what a run writes where the user is working, and out of site-packages.
    return Path.cwd().resolve()


ROOT_DIR: Path = _find_root()

SRC_DIR: Path = ROOT_DIR / "src" / PACKAGE
TESTS_DIR: Path = ROOT_DIR / "tests"
SCRIPTS_DIR: Path = ROOT_DIR / "scripts"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"
DOCS_DIR: Path = ROOT_DIR / "docs"
IMAGES_DIR: Path = DOCS_DIR / "images"
REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"
INFRA_DIR: Path = ROOT_DIR / "infra"

#: The only ignored directory: logs, checkpoints, caches, coverage reports, anything a run
#: leaves behind that no reader is meant to open.
VAR_DIR: Path = ROOT_DIR / "var"


# --- What this project writes and reads, by name ----------------------------
#
# The directory vocabulary above is shared by every repository of the portfolio. Below are
# this one's own artefacts, and the rule that decides where each goes: `models/` and
# `reports/` hold what is PUBLISHED, `var/` holds what a run leaves behind. A training run
# writes checkpoints, a TensorBoard trace and a video of the policy flying; none of the three
# is published, and the first version kept all three beside the shipped policy.

#: The shipped policy: the best checkpoint of the retained PPO run.
#:
#: One directory per algorithm, and `best` and `final` under different names. The first
#: version pointed `EvalCallback` at `models/` for both algorithms, so each overwrote the
#: other's `best_model.zip`, then saved the *last* model under a filename that said `best`
#: -- and evaluated that. In reinforcement learning the two differ, because performance
#: oscillates at the end of training.
DEFAULT_MODEL_PATH: Path = MODELS_DIR / "ppo" / "best.zip"

#: Where a training run keeps what only it needs: checkpoints, curves, TensorBoard, clips.
RUNS_DIR: Path = VAR_DIR / "runs"
TENSORBOARD_DIR: Path = VAR_DIR / "tensorboard"
LOGS_DIR: Path = VAR_DIR / "logs"
VIDEOS_DIR: Path = VAR_DIR / "videos"

#: The two published tables the dashboard and the README read.
EVALUATION_CSV: Path = REPORTS_DIR / "evaluation_episodes.csv"
TRAINING_CURVES_CSV: Path = REPORTS_DIR / "training_curves.csv"


def run_dir(algorithm: str, seed: int, variant: str | None = None) -> Path:
    """Where one training run keeps its checkpoints, its curve and its manifest.

    ``variant`` names a single-hyper-parameter change tried against the baseline. It is part
    of the directory name rather than a separate tree, so a trial and a baseline seed are
    the same kind of object and the same aggregation reads both.
    """
    name = f"seed-{seed}" if variant is None else f"seed-{seed}-{variant}"
    return RUNS_DIR / algorithm.lower() / name


def ensure_dirs() -> None:
    """Create the directories a run writes into.

    Only what a run writes into. An input that is absent has to stop the run where it is
    read, and never become an empty directory that looks like an answer.
    """
    for path in (REPORTS_DIR, FIGURES_DIR, MODELS_DIR, RUNS_DIR, TENSORBOARD_DIR,
                 LOGS_DIR, VIDEOS_DIR):
        path.mkdir(parents=True, exist_ok=True)
