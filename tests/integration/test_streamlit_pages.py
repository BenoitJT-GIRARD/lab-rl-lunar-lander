"""The two pages, run as Streamlit runs them.

Nothing else in the suite executes a page. The modules import cleanly, `test_theme.py` reads
the colours back, and both would keep passing with a page that raises on its first widget:
a Streamlit script is only exercised by a Streamlit runtime, and the failures that matter
live there. A column indexed by a name the export no longer carries, a `st.columns` list
unpacked one short, a chart handed a frame the filters emptied — each throws where a reader
sees a red traceback and none throws at import.

`AppTest` is that runtime, in-process and without a browser. It runs the script, it collects
what the script rendered, and it surfaces the exception the page would have shown. The
cockpit here flies on the local checkpoint: the service has its own tier, and what is under
test on this page is the page.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rl_lander.utils.paths import SRC_DIR

pytestmark = pytest.mark.integration

COCKPIT = SRC_DIR / "gui.py"
DASHBOARD = SRC_DIR / "dashboard.py"

#: Loading a checkpoint and flying an episode takes seconds. AppTest's default of three
#: would time the page out on the speed of the machine.
TIMEOUT = 180


def _page(path: Path) -> AppTest:
    app = AppTest.from_file(str(path), default_timeout=TIMEOUT)
    app.run()
    assert not app.exception, [str(e) for e in app.exception]
    return app


# --- The cockpit ------------------------------------------------------------


def test_the_cockpit_opens_on_an_instruction_and_not_on_an_empty_page() -> None:
    """Before the first run there is nothing to draw, and the page says what to press."""
    app = _page(COCKPIT)
    assert app.title[0].value == "Lunar landing cockpit"
    assert any("Run episode" in message.value for message in app.info)


def test_the_cockpit_flies_an_episode_on_the_local_checkpoint() -> None:
    """The toggle off, the button pressed: metrics, an outcome, and no traceback.

    This is the path a reader takes with no service running. It loads
    `models/ppo/best.zip`, plays one seeded episode and replays it locally, which is the
    whole of what the page does.
    """
    app = _page(COCKPIT)
    app.toggle[0].set_value(False)
    app.button[0].click().run()
    assert not app.exception, [str(e) for e in app.exception]

    labels = [metric.label for metric in app.metric]
    assert labels == ["Total reward", "Episode length", "Final altitude"]
    assert any("local model" in caption.value for caption in app.caption)
    # The shipped policy lands seed 42, so the outcome is the green one. A page that showed
    # neither badge would still have all three metrics.
    assert app.success or app.error


def test_the_cockpit_does_not_present_a_replay_it_could_not_verify() -> None:
    """The red banner is wired to the replay check, and to nothing else.

    `replay_matches` is unit-tested on its own; what this asserts is that the page reads its
    answer. Forced to false, the banner appears; left alone, it does not.
    """
    app = AppTest.from_file(str(COCKPIT), default_timeout=TIMEOUT)
    app.run()
    app.toggle[0].set_value(False)
    app.button[0].click().run()
    assert not app.error or all("diverged" not in message.value for message in app.error)

    app.session_state["episode"]["faithful"] = False
    app.run()
    assert any("diverged" in message.value for message in app.error)


# --- The dashboard ----------------------------------------------------------


def test_the_dashboard_reads_the_published_exports_and_draws_every_panel() -> None:
    """The tracked CSVs, the five cards, and the four sections, in one run."""
    app = _page(DASHBOARD)
    assert app.title[0].value == "Performance dashboard"

    labels = [metric.label for metric in app.metric]
    assert labels == [
        "Episodes",
        "Mean reward",
        "Landing rate",
        "Above solved threshold",
        "Mean engine firings",
    ]
    headings = [heading.value for heading in app.subheader]
    assert "Training progress" in headings
    assert "Per-episode telemetry" in headings
    assert "Engine use" in headings


def test_the_dashboard_says_what_is_missing_and_does_not_raise(tmp_path: Path) -> None:
    """With neither export on disk, the page answers with a sentence and a command.

    `rl_lander.artifacts.read_table` is what turns a missing file into that sentence, and
    this is the only test that shows a reader meeting it.

    The two paths are patched on `rl_lander.utils`. `RL_LANDER_ROOT` would change nothing
    here: the root is resolved once, when `rl_lander.utils.paths` is first imported, and this
    process imported it long ago. The page re-imports the two names on every run, which is
    what makes patching them there work at all.
    """
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr("rl_lander.utils.EVALUATION_CSV", tmp_path / "absent_episodes.csv")
        patched.setattr("rl_lander.utils.TRAINING_CURVES_CSV", tmp_path / "absent_curves.csv")
        empty = AppTest.from_file(str(DASHBOARD), default_timeout=TIMEOUT)
        empty.run()

    assert not empty.exception, [str(e) for e in empty.exception]
    assert empty.error, "a page with nothing to read has to say so"
    assert "not found" in empty.error[0].value
    assert "train_lunarlander" in empty.error[0].value
