"""Reading a published export, and refusing one that predates its schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from astrodynamics.artifacts import EVALUATION_COLUMNS, read_table


def _write(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False)
    return path


def test_a_missing_file_is_a_sentence_not_an_exception(tmp_path: Path) -> None:
    frame, problem = read_table(tmp_path / "absent.csv", EVALUATION_COLUMNS)
    assert frame is None
    assert "not found" in problem


def test_an_empty_file_is_reported_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    frame, problem = read_table(path, EVALUATION_COLUMNS)
    assert frame is None and "could not be read" in problem


def test_a_header_with_no_rows_is_reported_as_empty(tmp_path: Path) -> None:
    path = _write(tmp_path / "head.csv", pd.DataFrame(columns=list(EVALUATION_COLUMNS)))
    frame, problem = read_table(path, EVALUATION_COLUMNS)
    assert frame is None and "is empty" in problem


def test_an_export_from_an_older_version_names_the_columns_it_lacks(tmp_path: Path) -> None:
    """The dashboard used to fail with a KeyError halfway down a rendered page.

    By then the reader had already seen three panels drawn from a fourth of the data.
    """
    old = pd.DataFrame({"episode": [0], "total_reward": [1.0], "length": [2], "landed": [1]})
    frame, problem = read_table(_write(tmp_path / "old.csv", old), EVALUATION_COLUMNS)

    assert frame is None
    assert "seed" in problem and "meets_threshold" in problem
    assert "re-run it" in problem


def test_a_current_export_is_read(tmp_path: Path) -> None:
    current = pd.DataFrame({column: [0] for column in EVALUATION_COLUMNS})
    frame, problem = read_table(_write(tmp_path / "ok.csv", current), EVALUATION_COLUMNS)
    assert problem is None
    assert list(frame.columns) == list(EVALUATION_COLUMNS)
