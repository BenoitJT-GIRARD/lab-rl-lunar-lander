"""Turn the study's runs into the three artefacts the README and the notebook quote.

Reads every manifest under ``models/`` and writes:

``data/seed_study.json``
    The five baseline seeds, their spread, and the DQN run at the same step budget. This is
    the transportable number: a single run's score is a draw, and the repository used to
    publish one as if it were the method's.

``data/hyperparameter_trials.csv``
    One row per single-parameter trial, scored by the same protocol as the baseline. It
    replaces a table of "~280", "< 200", "~270" that nothing produced. The column that
    matters is the last one: whether the difference from the baseline is larger than the
    spread the baselines themselves show. Below that, a trial says nothing.

``data/learning_curve_band.csv``
    The median training curve across the five seeds with its interquartile band, on a
    common timestep grid. A single trajectory is not a learning curve.

    uv run python scripts/aggregate_study.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_lander.utils import DATA_DIR, MODELS_DIR  # noqa: E402

#: The grid the curves are resampled onto, so five runs that logged at slightly different
#: timesteps can be summarised column by column.
CURVE_GRID = np.arange(10_000, 1_000_001, 10_000)


def _runs() -> list[dict]:
    """Every finished run, with its manifest and where it lives."""
    found = []
    for manifest in sorted(MODELS_DIR.glob("*/seed-*/manifest.json")):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["run"] = manifest.parent.relative_to(ROOT).as_posix()
        payload["variant"] = manifest.parent.name.partition("-")[2].partition("-")[2] or None
        found.append(payload)
    return found


def _curve(run: Path) -> pd.Series | None:
    """One run's reward curve, resampled onto the common grid."""
    path = run / "training_curves.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path).drop_duplicates(subset="timesteps", keep="last")
    if frame.empty:
        return None
    return pd.Series(
        np.interp(CURVE_GRID, frame["timesteps"], frame["ep_rew_mean"]), index=CURVE_GRID
    )


def _seed_study(baselines: list[dict], dqn: list[dict]) -> dict:
    means = np.array([run["mean_reward"] for run in baselines])
    return {
        "algorithm": "PPO",
        "n_runs": len(baselines),
        "total_timesteps": baselines[0]["total_timesteps"] if baselines else None,
        "evaluation": {
            "n_episodes": baselines[0]["n_episodes"] if baselines else None,
            "seed": baselines[0]["evaluation_seed"] if baselines else None,
        },
        "per_run": [
            {
                "seed": run["seed"],
                "mean_reward": round(run["mean_reward"], 2),
                "std_reward": round(run["std_reward"], 2),
                "landing_rate": run["landing_rate"],
                "threshold_rate": run["threshold_rate"],
                "min_reward": round(run["min_reward"], 2),
                "run": run["run"],
            }
            for run in baselines
        ],
        "mean_of_runs": round(float(means.mean()), 2) if len(means) else None,
        # Between *training* runs. The dispersion inside one run, across evaluation
        # episodes, is a different quantity and lives in evaluation_summary.json. Reporting
        # one as the other is how a method looks more stable than it is.
        "spread_between_runs": round(float(means.std(ddof=1)), 2) if len(means) > 1 else None,
        "worst_run": round(float(means.min()), 2) if len(means) else None,
        "best_run": round(float(means.max()), 2) if len(means) else None,
        "runs_clearing_200": int((means >= 200).sum()) if len(means) else 0,
        "dqn_at_equal_budget": [
            {
                "seed": run["seed"],
                "total_timesteps": run["total_timesteps"],
                "mean_reward": round(run["mean_reward"], 2),
                "std_reward": round(run["std_reward"], 2),
                "landing_rate": run["landing_rate"],
                "run": run["run"],
            }
            for run in dqn
        ],
    }


def _trials_csv(baselines: list[dict], trials: list[dict], output: Path) -> Path:
    means = np.array([run["mean_reward"] for run in baselines])
    baseline_mean = float(means.mean())
    spread = float(means.std(ddof=1)) if len(means) > 1 else float("nan")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "trial",
                "changed",
                "value",
                "seed",
                "total_timesteps",
                "mean_reward",
                "std_reward",
                "landing_rate",
                "delta_vs_baseline",
                "larger_than_seed_spread",
                "run",
            ]
        )
        writer.writerow(
            [
                "baseline",
                "",
                "",
                f"{len(baselines)} seeds",
                baselines[0]["total_timesteps"] if baselines else "",
                round(baseline_mean, 2),
                round(spread, 2),
                round(float(np.mean([r["landing_rate"] for r in baselines])), 3),
                0.0,
                "",
                "models/ppo/seed-*",
            ]
        )
        for run in trials:
            hp = run["hyperparameters"]
            changed, value = _what_changed(hp, baselines[0]["hyperparameters"] if baselines else hp)
            delta = run["mean_reward"] - baseline_mean
            writer.writerow(
                [
                    run["variant"],
                    changed,
                    value,
                    run["seed"],
                    run["total_timesteps"],
                    round(run["mean_reward"], 2),
                    round(run["std_reward"], 2),
                    run["landing_rate"],
                    round(delta, 2),
                    "yes" if abs(delta) > spread else "no",
                    run["run"],
                ]
            )
    return output


def _what_changed(trial: dict, baseline: dict) -> tuple[str, str]:
    """Which single hyper-parameter this trial moved, and to what."""
    differences = {
        key: trial[key] for key in trial if key in baseline and trial[key] != baseline[key]
    }
    differences.pop("seed", None)
    if not differences:
        return "", ""
    key, value = next(iter(differences.items()))
    return key, str(value)


def _curve_band(baselines: list[dict], output: Path) -> Path | None:
    curves = [c for c in (_curve(ROOT / run["run"]) for run in baselines) if c is not None]
    if not curves:
        return None
    frame = pd.concat(curves, axis=1)
    band = pd.DataFrame(
        {
            "timesteps": CURVE_GRID,
            "median": frame.median(axis=1).to_numpy(),
            "q25": frame.quantile(0.25, axis=1).to_numpy(),
            "q75": frame.quantile(0.75, axis=1).to_numpy(),
            "min": frame.min(axis=1).to_numpy(),
            "max": frame.max(axis=1).to_numpy(),
            "n_runs": frame.notna().sum(axis=1).to_numpy(),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    band.to_csv(output, index=False, float_format="%.4f")
    return output


def main() -> None:
    runs = _runs()
    if not runs:
        raise SystemExit(
            "No finished run under models/. Train the grid first: "
            "`uv run python scripts/train_study.py`."
        )

    baselines = sorted(
        (r for r in runs if r["algorithm"] == "PPO" and r["variant"] is None),
        key=lambda r: r["seed"],
    )
    trials = sorted(
        (r for r in runs if r["algorithm"] == "PPO" and r["variant"]), key=lambda r: r["variant"]
    )
    dqn = sorted((r for r in runs if r["algorithm"] == "DQN"), key=lambda r: r["seed"])

    study = _seed_study(baselines, dqn)
    study_path = DATA_DIR / "seed_study.json"
    study_path.parent.mkdir(parents=True, exist_ok=True)
    study_path.write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")

    written = [study_path]
    if trials and baselines:
        written.append(_trials_csv(baselines, trials, DATA_DIR / "hyperparameter_trials.csv"))
    band = _curve_band(baselines, DATA_DIR / "learning_curve_band.csv")
    if band is not None:
        written.append(band)

    print(
        f"{len(baselines)} baseline run(s): {study['mean_of_runs']} "
        f"+/- {study['spread_between_runs']} between runs, "
        f"worst {study['worst_run']}, {study['runs_clearing_200']}/{len(baselines)} above 200."
    )
    for run in dqn:
        print(
            f"DQN seed {run['seed']} at {run['total_timesteps']:,} steps: "
            f"{run['mean_reward']:.2f}, landing rate {run['landing_rate']:.0%}."
        )
    for path in written:
        print(f"  [ok] {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
