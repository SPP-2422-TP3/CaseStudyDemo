"""Where the withheld centre state lands, across stroke windows and real-data budgets."""

from __future__ import annotations

from typing import NamedTuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..calibration import BUDGETS, WINDOWS, Calibration
from ..theme import MUTED, OKABE_ITO


class Series(NamedTuple):
    """How one variant is drawn. Named rather than a bare tuple: the page reads this too,
    and positional unpacking silently breaks the moment a field is added."""

    label: str
    color: str
    symbol: str


# Fixed, never cycled: the result, its control, and the simulation-free baseline. Each
# carries a marker shape as well as a hue -- the control and the real-only baseline land
# on top of each other at most budgets, which is itself the finding, and colour alone
# would leave the overlap unreadable.
SERIES = {
    "mix": Series("Sweep + real endpoints", OKABE_ITO[0], "circle"),
    "shuffled-sim": Series("Shuffled sweep (control)", OKABE_ITO[1], "square"),
    "real-only": Series("Real endpoints only", OKABE_ITO[2], "diamond-open"),
}


# "Window" and "budget" are both stroke counts, so name them explicitly wherever they
# appear -- otherwise the two axes read as the same number repeated.
def _window_titles() -> list[str]:
    return [
        f"Window: first {w} strokes" if w < 500 else "Window: whole run (500 strokes)"
        for w in WINDOWS
    ]


def placement_figure(calibration: Calibration) -> go.Figure:
    """One panel per window: placement against budget, with both controls.

    0 is the pristine anchor and 1 the worn one, so a sweep that carries the wear
    ordering should put the withheld state near 0.5. The controls say whether any
    given panel means anything.
    """
    titles = _window_titles()
    figure = make_subplots(rows=1, cols=len(WINDOWS), shared_yaxes=True, subplot_titles=titles)

    for column, window in enumerate(WINDOWS, start=1):
        # A budget of zero has no real strokes to anchor on, so its placement is the bare
        # prior and its control scatters over several times the axis. It stays in the
        # table, where a number can be read as degenerate; on the plot it only wrecks the
        # scale for every other point.
        budgets = [b for b in BUDGETS[window] if b > 0]
        for variant, series in SERIES.items():
            points = [(b, calibration.at(window, b, variant)) for b in budgets]
            points = [(b, p) for b, p in points if p is not None]
            if not points:
                continue
            figure.add_trace(
                go.Scatter(
                    x=[b for b, _ in points],
                    y=[p.position for _, p in points],
                    error_y={
                        "type": "data",
                        "array": [p.spread for _, p in points],
                        "color": series.color,
                        "thickness": 1,
                        "width": 3,
                    },
                    mode="lines+markers",
                    name=series.label,
                    legendgroup=variant,
                    showlegend=column == 1,
                    line={"color": series.color, "width": 2},
                    marker={"color": series.color, "size": 9, "symbol": series.symbol},
                    hovertemplate=(
                        f"<b>{series.label}</b><br>Budget: %{{x}} real strokes per endpoint"
                        "<br>position %{y:.3f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )

        guides = (
            (1.0, "worn anchor", "top right"),
            (0.5, "exactly centred", "top right"),
            (0.0, "pristine anchor", "bottom right"),
        )
        for y, text, position in guides:
            label = {
                "annotation": {"text": text, "font": {"size": 10, "color": MUTED}},
                "annotation_position": position,
            }
            figure.add_hline(
                y=y,
                line={"color": MUTED, "width": 1, "dash": "dash" if y == 0.5 else "dot"},
                row=1,
                col=column,
                **label,
            )
        # Ticks only where a budget was actually run, so the axis never implies a
        # measurement between two of them.
        figure.update_xaxes(
            title="Budget: real strokes per endpoint",
            tickmode="array",
            tickvals=budgets,
            row=1,
            col=column,
        )

    figure.update_yaxes(range=[-0.25, 1.2], row=1, col=1, title="Placement of the withheld state")
    # Top margin and legend height keep the legend clear of the subplot titles.
    figure.update_layout(
        height=350,
        margin={"l": 70, "r": 20, "t": 90, "b": 45},
        legend={"y": 1.24, "yanchor": "bottom"},
    )
    for annotation in figure.layout.annotations[: len(WINDOWS)]:
        annotation.font.size = 12
        annotation.font.color = MUTED
    return figure


# (metric field on Placement, y-axis title, subplot row)
_QUALITY_METRICS = (("accuracy", "Accuracy", 1), ("f1", "Macro F1", 2))


def quality_figure(calibration: Calibration) -> go.Figure:
    """Accuracy and macro-F1 against budget, one column per window, same variants as
    `placement_figure`.

    This is a harsher, discrete read of the same fitted mean: cut at the sweep's tercile
    edges and scored as a three-way level call, including on the withheld centre state
    that the placement figure never asks to be classified correctly, only positioned.
    1/3 is chance on three roughly balanced levels.
    """
    titles = [*_window_titles(), *([""] * len(WINDOWS))]
    figure = make_subplots(
        rows=2,
        cols=len(WINDOWS),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=titles,
        vertical_spacing=0.16,
    )

    for metric, metric_label, row in _QUALITY_METRICS:
        for column, window in enumerate(WINDOWS, start=1):
            budgets = [b for b in BUDGETS[window] if b > 0]
            for variant, series in SERIES.items():
                points = [(b, calibration.at(window, b, variant)) for b in budgets]
                points = [(b, p) for b, p in points if p is not None]
                if not points:
                    continue
                figure.add_trace(
                    go.Scatter(
                        x=[b for b, _ in points],
                        y=[getattr(p, metric) for _, p in points],
                        mode="lines+markers",
                        name=series.label,
                        legendgroup=variant,
                        showlegend=row == 1 and column == 1,
                        line={"color": series.color, "width": 2},
                        marker={"color": series.color, "size": 8, "symbol": series.symbol},
                        hovertemplate=(
                            f"<b>{series.label}</b><br>Budget: %{{x}} real strokes per endpoint"
                            f"<br>{metric_label}: %{{y:.3f}}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=column,
                )
            figure.add_hline(
                y=1 / 3,
                line={"color": MUTED, "width": 1, "dash": "dot"},
                row=row,
                col=column,
            )
            figure.update_xaxes(
                title="Budget: real strokes per endpoint" if row == 2 else None,
                tickmode="array",
                tickvals=budgets,
                row=row,
                col=column,
            )
        figure.update_yaxes(range=[0, 1], row=row, col=1, title=metric_label)

    figure.update_layout(
        height=520,
        margin={"l": 70, "r": 20, "t": 80, "b": 45},
        legend={"y": 1.1, "yanchor": "bottom"},
    )
    for annotation in figure.layout.annotations[: len(WINDOWS)]:
        annotation.font.size = 12
        annotation.font.color = MUTED
    return figure
