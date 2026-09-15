"""Reading the published exports, and refusing the ones that predate their schema.

The dashboard used to read `reports/evaluation_episodes.csv` and index columns as it went, so
a CSV written by an older exporter failed with a `KeyError` halfway down a rendered page --
after the reader had already seen three panels of a fourth of the data. The schema is
declared here, checked once, and reported as a sentence.

It lives outside the Streamlit module on purpose: this is what the export *is*, and both
the dashboard and the test suite need it. Streamlit is an optional extra, and a schema that
can only be read with a web framework installed is not a schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

#: How every published table and manifest ends its lines, on any platform.
#:
#: `.gitattributes` normalises text to LF when it is committed, so a CSV written with CRLF
#: is stored as LF and the two never look different. They differ the day a run is compared
#: against what was committed, which is what `scripts/smoke.py` does: `git` then reports a
#: file modified that no diff can show. Both writers have to be told, because `csv.writer`
#: ends its rows with CRLF by default and `Path.write_text` translates on Windows.
LINE_TERMINATOR = "\n"

#: What `scripts/evaluate_and_export.py` writes, per episode.
EVALUATION_COLUMNS = (
    "episode",
    "seed",
    "total_reward",
    "length",
    "landed",
    "meets_threshold",
    "final_x",
    "final_y",
    "main_engine_firings",
    "side_engine_firings",
)

#: What `CsvProgressCallback` writes during training.
CURVE_COLUMNS = ("timesteps", "ep_rew_mean", "ep_rew_std")

#: What `scripts/aggregate_study.py` writes for the band across seeds.
BAND_COLUMNS = ("timesteps", "median", "q25", "q75")


def write_json(path: Path, payload: object) -> Path:
    """Write a published JSON artefact: two-space indent, LF, one final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + LINE_TERMINATOR,
        encoding="utf-8",
        newline=LINE_TERMINATOR,
    )
    return path


def read_table(path: Path, required: tuple[str, ...]) -> tuple[pd.DataFrame | None, str | None]:
    """Read a published CSV and check its columns.

    Returns ``(frame, problem)``: exactly one of the two is ``None``. The problem is a
    sentence meant to be shown to whoever opened the page, naming the file and what to run.
    """
    if not path.exists():
        return None, f"`{path.name}` not found."
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        return None, f"`{path.name}` could not be read: {exc}"
    if frame.empty:
        return None, f"`{path.name}` is empty."
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return None, (
            f"`{path.name}` is missing {', '.join(missing)}. It was probably written by an "
            "older version of `scripts/evaluate_and_export.py`; re-run it."
        )
    return frame, None
