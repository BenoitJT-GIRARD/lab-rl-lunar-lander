"""Promote one training run to the artefacts the repository publishes.

Training writes to `models/<algo>/seed-<n>/`, one directory per run, none of them tracked.
Exactly one run is the one the README quotes, the API serves and the dashboard plots, and
this script is the step that says which. Nothing else in the pipeline writes to the
published paths, so a figure in the README can always be traced to the run it came from.

    uv run python scripts/publish_run.py var/runs/ppo/seed-42
    uv run python scripts/publish_run.py --from-study

`--from-study` picks the run whose mean is nearest the **median** of the study, rather than
the best of them. Publishing the best of five and printing the mean of five beside it would
be a third way of choosing the number after having seen it; the median run is the one whose
score the published dispersion actually describes.

Copies `best.zip` and `manifest.json` to `models/<algo>/`, and the run's training curve to
`reports/training_curves.csv`. Then re-run `scripts/evaluate_and_export.py`, which re-scores
the published model and rewrites the evaluation exports from it.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from rl_lander.utils import MODELS_DIR, REPORTS_DIR, TRAINING_CURVES_CSV
from rl_lander.utils import ROOT_DIR as ROOT

REQUIRED = ("best.zip", "manifest.json")


def median_run(study_path: Path) -> Path:
    """The baseline run whose mean is nearest the median of the study.

    Ties go to the lower seed, so the choice is a function of the artefact and not of the
    order the filesystem happened to return.
    """
    if not study_path.exists():
        raise SystemExit(
            f"{study_path} is missing. Run `uv run python scripts/aggregate_study.py` first."
        )
    runs = json.loads(study_path.read_text(encoding="utf-8"))["per_run"]
    if not runs:
        raise SystemExit(f"{study_path} lists no baseline run.")
    means = sorted(run["mean_reward"] for run in runs)
    middle = means[len(means) // 2] if len(means) % 2 else sum(means[len(means) // 2 - 1 :][:2]) / 2
    chosen = min(runs, key=lambda run: (abs(run["mean_reward"] - middle), run["seed"]))
    print(
        f"Median of {len(runs)} runs is {middle:.2f}; nearest is seed {chosen['seed']} "
        f"at {chosen['mean_reward']:.2f}."
    )
    return ROOT / chosen["run"]


def publish(run: Path) -> list[Path]:
    """Copy a run's artefacts to the published paths. Returns what was written."""
    missing = [name for name in REQUIRED if not (run / name).exists()]
    if missing:
        raise SystemExit(
            f"'{run}' is not a finished run: {', '.join(missing)} missing. Train it with "
            "`uv run python -m rl_lander.training.train_lunarlander`."
        )

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("best_is_final"):
        raise SystemExit(
            f"'{run}' never improved on its starting policy, so its best.zip is just the "
            "final state. Publishing it would put an untrained policy behind the README's "
            "figures. Train longer, or pass another run."
        )

    target = MODELS_DIR / str(manifest["algorithm"]).lower()
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for name in REQUIRED:
        shutil.copy2(run / name, target / name)
        written.append(target / name)

    curve = run / "training_curves.csv"
    if curve.exists():
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(curve, TRAINING_CURVES_CSV)
        written.append(TRAINING_CURVES_CSV)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run", type=Path, nargs="?", help="the run directory, e.g. var/runs/ppo/seed-42"
    )
    parser.add_argument(
        "--from-study",
        action="store_true",
        help="pick the run nearest the median of data/seed_study.json",
    )
    args = parser.parse_args()

    if args.from_study == bool(args.run):
        raise SystemExit("Pass a run directory, or --from-study. Not both, and not neither.")
    run = median_run(REPORTS_DIR / "seed_study.json") if args.from_study else args.run

    written = publish(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    print(
        f"Published {run} — mean_reward {manifest['mean_reward']:.2f}, "
        f"landing rate {manifest['landing_rate']:.0%}."
    )
    for path in written:
        print(f"  [ok] {path.relative_to(ROOT)}")
    print("\nNow run: uv run python scripts/evaluate_and_export.py")


if __name__ == "__main__":
    main()
