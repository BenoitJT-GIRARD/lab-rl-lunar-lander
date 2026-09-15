"""One palette, two consumers, and nothing to stop them drifting apart but this file.

Streamlit reads ``.streamlit/config.toml`` before the process starts, so the file can import
nothing and its four colours are typed in by hand. The charts read them from
:mod:`rl_lander.figure_style` instead. These tests are the only thing that holds the two to the
same values.
"""

from __future__ import annotations

import tomllib

from rl_lander.figure_style import PALETTE
from rl_lander.utils.paths import ROOT_DIR

THEME = tomllib.loads((ROOT_DIR / ".streamlit" / "config.toml").read_text(encoding="utf-8"))


def test_every_theme_colour_is_a_token_of_the_palette() -> None:
    expected = {
        "primaryColor": PALETTE["primary"],
        "backgroundColor": PALETTE["paper"],
        "secondaryBackgroundColor": PALETTE["surface"],
        "textColor": PALETTE["ink"],
    }
    assert {key: THEME["theme"][key] for key in expected} == expected


def test_the_editor_toolbar_is_hidden() -> None:
    """Deploy button and three-dot menu: the editor's furniture, in a reader's screenshot."""
    assert THEME["client"]["toolbarMode"] == "minimal"


def test_the_page_accepts_no_upload() -> None:
    """Two pages that read versioned artefacts and ask an API have nothing to receive."""
    assert THEME["server"]["maxUploadSize"] == 1
    assert THEME["server"]["enableXsrfProtection"] is True
