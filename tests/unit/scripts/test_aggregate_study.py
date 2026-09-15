"""What the study aggregation publishes, and the distinctions it must not collapse.

`scripts/` is on the test path, so the functions are imported and called rather than shelled
out to. These three are where the repository's central claim is computed: the spread between
training runs, and the column that says whether a trial's difference clears it.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytestmark = pytest.mark.claim


def _run(seed: int, mean: float, *, variant: str | None = None, **hyper) -> dict:
    """One finished run, in the shape `aggregate_study` reads from a manifest."""
    return {
        "seed": seed,
        "variant": variant,
        "mean_reward": mean,
        "std_reward": 20.0,
        "min_reward": mean - 100,
        "landing_rate": 0.98,
        "threshold_rate": 0.97,
        "n_episodes": 100,
        "evaluation_seed": 2024,
        "total_timesteps": 1_000_000,
        "run": f"var/runs/ppo/seed-{seed}",
        "hyperparameters": {"learning_rate": 3e-4, "n_steps": 1024, "gamma": 0.999, **hyper},
    }


def test_the_published_spread_is_between_runs_and_not_between_episodes() -> None:
    """The two are different quantities, and reporting one as the other is the whole subject.

    Every run below has a within-run standard deviation of 20. The study's own spread is the
    dispersion of the five means, which is a different number entirely; a version that read
    `std_reward` here would have published 20.0 and called it the method's uncertainty.
    """
    from aggregate_study import _seed_study

    study = _seed_study([_run(42 + i, mean) for i, mean in enumerate([240.0, 250.0, 260.0])], [])

    assert study["mean_of_runs"] == 250.0
    assert study["spread_between_runs"] == 10.0  # sample sd of 240, 250, 260
    assert study["worst_run"] == 240.0
    assert study["runs_clearing_200"] == 3


def test_a_single_run_publishes_no_spread_rather_than_a_zero() -> None:
    """One point has no dispersion. Writing 0.0 would read as a method that never varies."""
    from aggregate_study import _seed_study

    study = _seed_study([_run(42, 254.0)], [])

    assert study["spread_between_runs"] is None
    assert study["mean_of_runs"] == 254.0


def test_a_run_below_the_threshold_is_counted_as_such() -> None:
    from aggregate_study import _seed_study

    study = _seed_study([_run(42, 254.0), _run(43, 173.0)], [])

    assert study["runs_clearing_200"] == 1


def test_the_dqn_is_reported_beside_the_study_and_never_inside_it() -> None:
    """One run of another algorithm is not a sixth seed of this one."""
    from aggregate_study import _seed_study

    study = _seed_study([_run(42, 254.0), _run(43, 260.0)], [_run(42, 271.13)])

    assert study["n_runs"] == 2
    assert study["mean_of_runs"] == 257.0
    assert len(study["dqn_at_equal_budget"]) == 1
    assert study["dqn_at_equal_budget"][0]["mean_reward"] == 271.13


def test_a_trial_is_judged_against_the_spread_between_seeds(tmp_path: Path) -> None:
    """The last column is what makes the table readable instead of a ranking of four numbers.

    The baseline below spreads by 10 between its seeds. A trial 4 points above it is one draw
    from the same distribution; a trial 50 points below it is not, and only the second is
    reported as larger than the spread.
    """
    from aggregate_study import _trials_csv

    baselines = [_run(42 + i, mean) for i, mean in enumerate([240.0, 250.0, 260.0])]
    trials = [
        _run(42, 254.0, variant="lr1e-3", learning_rate=1e-3),
        _run(42, 200.0, variant="gamma099", gamma=0.99),
    ]

    rows = list(csv.DictReader(_trials_csv(baselines, trials, tmp_path / "trials.csv")
                               .read_text(encoding="utf-8").splitlines()))

    assert rows[0]["trial"] == "baseline"
    assert rows[0]["seed"] == "3 seeds"
    assert rows[0]["std_reward"] == "10.0"

    lr, gamma = rows[1], rows[2]
    assert lr["changed"] == "learning_rate"
    assert lr["delta_vs_baseline"] == "4.0"
    assert lr["larger_than_seed_spread"] == "no"
    assert gamma["changed"] == "gamma"
    assert gamma["delta_vs_baseline"] == "-50.0"
    assert gamma["larger_than_seed_spread"] == "yes"


def test_the_changed_parameter_is_read_from_the_run_and_not_from_its_name() -> None:
    """A directory called `seed-42-lr1e-3` is a label; the manifest is the evidence."""
    from aggregate_study import _what_changed

    baseline = {"learning_rate": 3e-4, "n_steps": 1024, "gamma": 0.999}

    assert _what_changed({**baseline, "gamma": 0.99}, baseline) == ("gamma", "0.99")
    assert _what_changed(baseline, baseline) == ("", "")


def test_the_seed_is_never_reported_as_the_changed_parameter() -> None:
    """Two runs at different seeds differ in their seed, which is not a hyper-parameter."""
    from aggregate_study import _what_changed

    baseline = {"learning_rate": 3e-4, "seed": 42}

    assert _what_changed({"learning_rate": 3e-4, "seed": 43}, baseline) == ("", "")
