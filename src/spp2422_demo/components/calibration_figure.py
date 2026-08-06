"""Where the withheld centre state lands, across stroke windows and real-data budgets."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..calibration import BUDGETS, WINDOWS, Calibration
from ..theme import MUTED, OKABE_ITO

# Fixed, never cycled: the result, its control, and the simulation-free baseline. Each
# carries a marker shape as well as a hue -- the control and the real-only baseline land
# on top of each other at most budgets, which is itself the finding, and colour alone
# would leave the overlap unreadable.
SERIES = {
    "mix": ("Sweep + real endpoints", OKABE_ITO[0], "circle"),
    "shuffled-sim": ("Shuffled sweep (control)", OKABE_ITO[1], "square"),
    "real-only": ("Real endpoints only", OKABE_ITO[2], "diamond-open"),
}


def placement_figure(calibration: Calibration) -> go.Figure:
    """One panel per window: placement against budget, with both controls.

    0 is the pristine anchor and 1 the worn one, so a sweep that carries the wear
    ordering should put the withheld state near 0.5. The controls say whether any
    given panel means anything.
    """
    titles = [f"first {w} strokes" if w < 500 else "whole 500-stroke run" for w in WINDOWS]
    figure = make_subplots(rows=1, cols=len(WINDOWS), shared_yaxes=True, subplot_titles=titles)
    last = len(WINDOWS)

    for column, window in enumerate(WINDOWS, start=1):
        # A budget of zero has no real strokes to anchor on, so its placement is the bare
        # prior and its control scatters over several times the axis. It stays in the
        # table, where a number can be read as degenerate; on the plot it only wrecks the
        # scale for every other point.
        budgets = [b for b in BUDGETS[window] if b > 0]
        for variant, (label, color, symbol) in SERIES.items():
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
                        "color": color,
                        "thickness": 1,
                        "width": 3,
                    },
                    mode="lines+markers",
                    name=label,
                    legendgroup=variant,
                    showlegend=column == 1,
                    line={"color": color, "width": 2},
                    marker={"color": color, "size": 9, "symbol": symbol},
                    hovertemplate=(
                        f"<b>{label}</b><br>%{{x}} real strokes per endpoint"
                        "<br>position %{y:.3f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )

        # Labelled once, on the last panel, where the space above the data is empty --
        # repeating them per panel puts text straight through the marks.
        guides = (
            (1.0, "worn anchor", "top right"),
            (0.5, "exactly centred", "top right"),
            (0.0, "pristine anchor", "bottom right"),
        )
        for y, text, position in guides:
            # The annotation arguments have to be absent, not None: add_hline reads a
            # None as "annotate with the default", which prints a literal "new text".
            label = (
                {
                    "annotation": {"text": text, "font": {"size": 10, "color": MUTED}},
                    "annotation_position": position,
                }
                if column == last
                else {}
            )
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
            title="real strokes per endpoint",
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
