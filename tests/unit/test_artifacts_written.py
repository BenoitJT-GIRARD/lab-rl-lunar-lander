"""Published artefacts end their lines the same way on every platform.

The defect this pins was invisible for as long as nobody compared a run against what was
committed. `.gitattributes` normalises text to LF on the way in, so a CSV written with CRLF
on Windows is stored as LF; the checkout then holds one and the blob the other, and `git
status` reports a file modified that `git diff` cannot show. `scripts/smoke.py` runs the
exporter and compares, which is where it surfaced: two artefacts « differing from the
committed version » with every number identical.
"""

from __future__ import annotations

import csv
from pathlib import Path

from rl_lander.artifacts import LINE_TERMINATOR, write_json
from rl_lander.training.evaluate import EpisodeRecord, write_csv


def _record(seed: int) -> EpisodeRecord:
    return EpisodeRecord(
        episode=0,
        seed=seed,
        total_reward=254.1,
        length=310,
        landed=True,
        meets_threshold=True,
        final_x=0.01,
        final_y=-0.001,
        main_engine_firings=140,
        side_engine_firings=100,
    )


def test_an_exported_table_carries_no_carriage_return(tmp_path: Path) -> None:
    """Three rows written on Windows, read back as bytes: not a CR among them."""
    output = write_csv([_record(2024), _record(2025), _record(2026)], tmp_path / "episodes.csv")
    raw = output.read_bytes()
    assert b"\r" not in raw
    assert raw.count(b"\n") == 4


def test_an_exported_table_is_still_a_csv_a_reader_can_parse(tmp_path: Path) -> None:
    """The terminator is fixed, and nothing else about the format moves."""
    output = write_csv([_record(2024)], tmp_path / "episodes.csv")
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["seed"] for row in rows] == ["2024"]


def test_a_published_json_ends_with_one_newline_and_no_carriage_return(tmp_path: Path) -> None:
    """The final newline is part of the file: without it every diff shows a last line."""
    output = write_json(tmp_path / "summary.json", {"mean_reward": 254.14, "seeds": [42, 43]})
    raw = output.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(LINE_TERMINATOR.encode())
    assert not raw.endswith((LINE_TERMINATOR * 2).encode())


def test_a_published_json_creates_the_directory_it_is_asked_for(tmp_path: Path) -> None:
    """The exporters write into `reports/`, which a clean clone does not have."""
    output = write_json(tmp_path / "reports" / "nested" / "summary.json", {"n": 1})
    assert output.exists()
