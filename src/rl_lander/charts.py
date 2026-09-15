"""How a chart of either page is painted. One module, two pages, one answer.

Streamlit's own `st.bar_chart` and `st.line_chart` are one line each and paint in the
library's default colours, next to a page painted in this project's. The cockpit used them,
and that is also how `tests/integration/test_streamlit_pages.py` came to crash the
interpreter instead of failing: their Altair path converts the frame through pyarrow, and in
a process already holding Box2D and torch that conversion ended in an access violation, with
no Python traceback to read.

So both pages draw with Plotly, and both take their colours from
:mod:`rl_lander.figure_style` through this module. A chart here never picks a colour of its
own, and the two pages cannot drift apart into two palettes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import plotly.graph_objects as go

from rl_lander.figure_style import PALETTE, STATE

#: Two outcomes, two state colours, taken by name from the palette the figures read.
OUTCOME_COLOURS = {"landed": STATE["ok"], "did not land": STATE["danger"]}

#: What every chart of both pages is drawn with. Plotly takes nothing from the Streamlit
#: theme, so a figure left to itself arrives in the library's defaults.
LAYOUT = {
    "paper_bgcolor": PALETTE["paper"],
    "plot_bgcolor": PALETTE["paper"],
    "font": {"color": PALETTE["ink"], "size": 13},
    "margin": {"t": 50, "b": 40, "l": 10, "r": 10},
}
AXIS = {
    "gridcolor": PALETTE["grid"],
    "zerolinecolor": PALETTE["grid"],
    "linecolor": PALETTE["muted"],
}


def styled(figure: Any, x_title: str, y_title: str) -> Any:
    """One place where a chart gets its colours and its axis titles."""
    figure.update_layout(**LAYOUT)
    figure.update_xaxes(title_text=x_title, **AXIS)
    figure.update_yaxes(title_text=y_title, **AXIS)
    return figure


def bars(labels: Sequence[str], values: Sequence[float], *, x_title: str, y_title: str) -> Any:
    """A categorical bar chart in the palette's primary colour."""
    figure = go.Figure(go.Bar(x=list(labels), y=list(values), marker_color=PALETTE["primary"]))
    figure.update_layout(showlegend=False)
    return styled(figure, x_title, y_title)


def line(values: Sequence[float], *, x_title: str, y_title: str) -> Any:
    """A single series against its own index, for a sequence with no other x."""
    figure = go.Figure(
        go.Scatter(
            x=list(range(len(values))),
            y=list(values),
            mode="lines",
            line={"color": PALETTE["primary"], "width": 1.6},
        )
    )
    figure.update_layout(showlegend=False)
    return styled(figure, x_title, y_title)
