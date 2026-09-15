"""The video recorder, the cockpit's replay check, and the published artefacts.

These are the three places where the repository makes a claim to someone who will not read
the code: a clip, a metrics panel, and the numbers in the README. Each test here is a
statement that was once false.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from rl_lander.record_video import Take, _assemble, _crossfade, record_landing
from rl_lander.replay import replay_matches
from rl_lander.training.evaluate import SOLVED_THRESHOLD
from rl_lander.utils import DEFAULT_MODEL_PATH, REPORTS_DIR, ROOT_DIR

pytestmark = [pytest.mark.integration, pytest.mark.claim]

FRAME_SHAPE = (4, 6, 3)


def _take(seed: int, reward: float, landed: bool, n_frames: int = 5) -> Take:
    frames = [
        np.full(FRAME_SHAPE, fill_value=seed * 10 + i, dtype=np.uint8) for i in range(n_frames)
    ]
    return Take(seed=seed, total_reward=reward, landed=landed, frames=frames)


# --- the clip and what it claims -------------------------------------------------------


def test_a_crossfade_actually_fades() -> None:
    """The function it replaces returned black frames under the name `fade`."""
    black = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    white = np.full(FRAME_SHAPE, 255, dtype=np.uint8)
    fade = _crossfade(black, white, length=3)

    assert len(fade) == 3
    means = [float(frame.mean()) for frame in fade]
    assert means == sorted(means), "the transition is monotonic"
    assert means[0] > 0 and means[-1] < 255, "neither end is a hard cut"


def test_a_crossfade_of_zero_length_is_no_frames_rather_than_an_error() -> None:
    assert _crossfade(np.zeros(FRAME_SHAPE, np.uint8), np.zeros(FRAME_SHAPE, np.uint8), 0) == []


def test_assembly_keeps_the_order_it_was_given() -> None:
    """The clip is the takes, in the order the manifest will name them."""
    takes = [_take(1, 100.0, True), _take(2, 300.0, True)]
    sequence = _assemble(takes, fade_length=2)

    assert len(sequence) == 5 + 2 + 5
    assert np.array_equal(sequence[0], takes[0].frames[0])
    assert np.array_equal(sequence[-1], takes[1].frames[-1])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_attempts": 0}, "max_attempts"),
        ({"min_landings": 0}, "min_landings"),
        ({"fps": 0}, "fps"),
        ({"target_min_seconds": 30.0, "target_max_seconds": 10.0}, "target window"),
        ({"target_min_seconds": -1.0}, "target window"),
    ],
)
def test_impossible_recording_parameters_are_refused(
    kwargs: dict, message: str, tmp_path: Path
) -> None:
    """They used to be accepted and produce an empty file, or divide by zero."""
    with pytest.raises(ValueError, match=message):
        record_landing(output=tmp_path / "clip.mp4", gif=tmp_path / "clip.gif", **kwargs)


def test_a_clip_that_shows_no_landing_says_so(tmp_path: Path, untrained_checkpoint: Path) -> None:
    """The old fallback reported `n_landings: 1` for a crash.

    An untrained policy lands nothing, so this exercises exactly that path.
    """
    info = record_landing(
        model_path=untrained_checkpoint,
        output=tmp_path / "clip.mp4",
        gif=tmp_path / "clip.gif",
        max_attempts=2,
        target_min_seconds=0.1,
        target_max_seconds=60.0,
        min_landings=1,
        fps=50,
    )

    assert info["shows_a_landing"] is False
    assert info["n_landings"] == 0
    assert info["episodes"] and all(not ep["landed"] for ep in info["episodes"])
    assert Path(info["output"]).exists()


def test_the_manifest_names_the_episodes_that_were_encoded(
    tmp_path: Path, untrained_checkpoint: Path
) -> None:
    """The seeds reported and the frames encoded came from two different orderings.

    Episodes were merged after sorting by reward, and the manifest listed the first ones
    accepted -- chronological order. With more than one landing the two lists differed, and
    nothing in the pipeline could have caught it.
    """
    info = record_landing(
        model_path=untrained_checkpoint,
        output=tmp_path / "clip.mp4",
        gif=tmp_path / "clip.gif",
        max_attempts=3,
        target_min_seconds=0.1,
        target_max_seconds=60.0,
        min_landings=1,
        fps=50,
    )

    assert info["seeds"] == [ep["seed"] for ep in info["episodes"]]
    assert info["mean_reward"] == pytest.approx(
        float(np.mean([ep["total_reward"] for ep in info["episodes"]])), abs=0.01
    )
    assert info["landing_rate"] == 0.0
    assert info["attempts"] == 3


def test_a_rehearsal_does_not_replace_the_published_animation(
    tmp_path: Path, untrained_checkpoint: Path
) -> None:
    """Recording somewhere else leaves `docs/images/` exactly as it was.

    This is the guard on a defect that shipped silently: `record_landing` wrote its GIF to
    the published path whatever `output` said, and declared it in the manifest. Running the
    suite therefore replaced the README's landing clip with eighty-two frames of an
    untrained policy crashing, and rewrote the manifest entry to match.
    """
    published = ROOT_DIR / "docs" / "images" / "landing.gif"
    manifest = ROOT_DIR / "docs" / "images" / "MANIFEST.json"
    before = (published.read_bytes(), manifest.read_bytes())

    info = record_landing(
        model_path=untrained_checkpoint,
        output=tmp_path / "clip.mp4",
        gif=tmp_path / "clip.gif",
        max_attempts=1,
        target_min_seconds=0.1,
        target_max_seconds=60.0,
        min_landings=1,
        fps=50,
    )

    assert Path(info["gif"]) == tmp_path / "clip.gif"
    assert Path(info["gif"]).exists()
    assert (published.read_bytes(), manifest.read_bytes()) == before


# --- the cockpit's replay --------------------------------------------------------------


def test_the_cockpit_detects_a_replay_that_is_not_the_same_episode() -> None:
    """The animation and the metrics have to describe one episode, and it is checked."""
    assert replay_matches([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert not replay_matches([1.0, 2.0], [1.0, 2.0, 3.0])
    assert not replay_matches([1.0, 2.0, 3.5], [1.0, 2.0, 3.0])


# --- what the repository publishes ------------------------------------------------------


@pytest.fixture(scope="module")
def published_summary() -> dict:
    path = REPORTS_DIR / "evaluation_summary.json"
    if not path.exists():
        pytest.skip("no evaluation export in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


def _published_episodes() -> list[dict[str, str]]:
    """The exported episodes, with the handle closed.

    `csv.DictReader(path.open())` leaks the file until the garbage collector runs, and with
    warnings as errors the ResourceWarning fails the test that reads it -- which is the
    rule doing its job on this suite rather than on a dependency.
    """
    with (REPORTS_DIR / "evaluation_episodes.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_the_shipped_policy_is_where_everything_expects_it() -> None:
    assert DEFAULT_MODEL_PATH.exists(), f"{DEFAULT_MODEL_PATH} is the policy the API serves"


def test_the_published_csv_and_the_published_summary_are_one_collection(
    published_summary: dict,
) -> None:
    """Two numbers used to be published from two different evaluation loops."""
    rows = _published_episodes()
    metrics = published_summary["metrics"]

    assert len(rows) == int(metrics["n_episodes"])
    rewards = [float(row["total_reward"]) for row in rows]
    assert float(np.mean(rewards)) == pytest.approx(metrics["mean_reward"], abs=1e-3)
    assert float(np.std(rewards, ddof=1)) == pytest.approx(metrics["std_reward"], abs=1e-3)
    landed = [int(row["landed"]) for row in rows]
    assert float(np.mean(landed)) == pytest.approx(metrics["landing_rate"], abs=1e-6)


def test_every_published_episode_records_the_seed_that_replays_it(published_summary: dict) -> None:
    rows = _published_episodes()
    seeds = [int(row["seed"]) for row in rows]
    canonical = int(published_summary["canonical_seed"])
    assert seeds == list(range(canonical, canonical + len(rows)))


def test_the_manifest_points_at_the_policy_that_was_evaluated() -> None:
    path = REPORTS_DIR / "evaluation_manifest.json"
    if not path.exists():
        pytest.skip("no evaluation export in this checkout")
    manifest = json.loads(path.read_text(encoding="utf-8"))

    model = ROOT_DIR / manifest["model"]
    assert model.exists(), f"the manifest names {manifest['model']}, which is not in the repository"
    assert not Path(manifest["model"]).is_absolute()
    assert manifest["versions"]["gymnasium"]


def test_the_headline_claim_holds_on_every_evaluation_seed(published_summary: dict) -> None:
    """The README says the policy solves the environment. This is that sentence, as a test.

    The claim is about the *mean* over 100 episodes, and it is checked on every seed grid
    the exporter scored, and never on the exported one alone, which is the most
    favourable of them.
    """
    across = published_summary.get("across_seeds")
    if across is None:
        pytest.skip("the export was produced from a single seed grid")

    below = [s for s in across["per_seed"] if s["mean_reward"] < SOLVED_THRESHOLD]
    assert not below, "the solved threshold is not cleared on seed(s) " + ", ".join(
        f"{s['seed']} ({s['mean_reward']:.1f})" for s in below
    )


# --- which run becomes the shipped policy -----------------------------------------------


def test_the_published_run_is_the_median_not_the_best(tmp_path: Path) -> None:
    """Given five means, the promoted run is the median one and never the highest.

    The rule, and the reason it is a rule rather than a judgement, are in the docstring of
    `median_run`; what is asserted here is that the function obeys it."""
    from publish_run import median_run

    study = tmp_path / "seed_study.json"
    study.write_text(
        json.dumps(
            {
                "per_run": [
                    {"seed": 42, "mean_reward": 264.0, "run": "models/ppo/seed-42"},
                    {"seed": 43, "mean_reward": 242.0, "run": "models/ppo/seed-43"},
                    {"seed": 44, "mean_reward": 251.0, "run": "models/ppo/seed-44"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert median_run(study).name == "seed-44"


def test_a_tie_between_two_runs_is_broken_by_the_seed(tmp_path: Path) -> None:
    """So the choice is a function of the artefact, not of filesystem ordering."""
    from publish_run import median_run

    study = tmp_path / "seed_study.json"
    study.write_text(
        json.dumps(
            {
                "per_run": [
                    {"seed": 44, "mean_reward": 250.0, "run": "models/ppo/seed-44"},
                    {"seed": 42, "mean_reward": 250.0, "run": "models/ppo/seed-42"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert median_run(study).name == "seed-42"


def test_publishing_from_a_study_that_does_not_exist_says_which_command_makes_it(
    tmp_path: Path,
) -> None:

    from publish_run import median_run

    with pytest.raises(SystemExit, match="aggregate_study"):
        median_run(tmp_path / "absent.json")
