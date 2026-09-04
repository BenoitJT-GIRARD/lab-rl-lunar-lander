"""Promote one training run to the artefacts the repository publishes.

Training writes to `models/<algo>/seed-<n>/`, one directory per run, none of them tracked.
Exactly one run is the one the README quotes, the API serves and the dashboard plots, and
this script is the step that says which. Nothing else in the pipeline writes to the
published paths, so a figure in the README can always be traced to the run it came from.

    uv run python scripts/publish_run.py models/ppo/seed-42

Copies `best.zip` and `manifest.json` to `models/<algo>/`, and the run's training curve to
`data/training_curves.csv`. Then re-run `scripts/evaluate_and_export.py`, which re-scores
the published model and rewrites the evaluation exports from it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from astrodynamics.utils import DATA_DIR, MODELS_DIR, TRAINING_CURVES_CSV  # noqa: E402

REQUIRED = ("best.zip", "manifest.json")


def publish(run: Path) -> list[Path]:
    """Copy a run's artefacts to the published paths. Returns what was written."""
    missing = [name for name in REQUIRED if not (run / name).exists()]
    if missing:
        raise SystemExit(
            f"'{run}' is not a finished run: {', '.join(missing)} missing. Train it with "
            "`uv run python -m astrodynamics.training.train_lunarlander`."
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
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(curve, TRAINING_CURVES_CSV)
        written.append(TRAINING_CURVES_CSV)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="the run directory, e.g. models/ppo/seed-42")
    args = parser.parse_args()

    written = publish(args.run)
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    print(
        f"Published {args.run} — mean_reward {manifest['mean_reward']:.2f}, "
        f"landing rate {manifest['landing_rate']:.0%}."
    )
    for path in written:
        print(f"  [ok] {path.relative_to(ROOT)}")
    print("\nNow run: uv run python scripts/evaluate_and_export.py")


if __name__ == "__main__":
    main()
