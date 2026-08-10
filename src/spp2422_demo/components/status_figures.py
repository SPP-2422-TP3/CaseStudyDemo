"""Figures for the status board: two trends and the stroke log behind them.

Both trends plot the same stretch of strokes on the same x axis, so the eye can carry a
moment from one to the other -- a wear step and an alignment step at the same stroke are
two faults, not one. Per-stroke readings are drawn faintly behind the trailing mean the
board actually reads, because the gap between them *is* the argument for smoothing.

State is never carried by colour alone: the bands are labelled on the axis, and every
caller renders the state's word beside its colour.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import html

from ..data import STATIONS
from ..health import (
    COLOR,
    CRITICAL,
    GOOD,
    WATCH,
    WATCH_FRACTION,
    alignment_state,
    majority_level,
)
from ..scenario import ALIGNMENT_WINDOW, WEAR_WINDOW, Run
from ..theme import GRID, INK, LEVEL_COLORS, MUTED

STROKE_TITLE = "Stroke"
LOG_ROWS = 12  # recent strokes listed in a detail window


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def trailing(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean at every position, short at the start rather than absent."""
    sums = np.cumsum(np.insert(values, 0, 0.0))
    starts = np.maximum(np.arange(len(values)) - window + 1, 0)
    return (sums[1:] - sums[starts]) / (np.arange(len(values)) - starts + 1)


def _band(figure: go.Figure, low: float, high: float, state: str) -> None:
    figure.add_hrect(
        y0=low, y1=high, fillcolor=_rgba(COLOR[state], 0.09), line_width=0, layer="below"
    )


def _frame(figure: go.Figure, strokes: np.ndarray, title: str) -> go.Figure:
    figure.update_layout(
        height=280,
        margin={"l": 60, "r": 20, "t": 30, "b": 45},
        yaxis={"title": title},
        xaxis={"title": STROKE_TITLE, "range": [strokes[0], strokes[-1]]},
        showlegend=True,
    )
    return figure


def _flag_bands(figure: go.Figure, flags: list[tuple[int, int]]) -> None:
    """Shade the stretches an operator reported as bad parts."""
    for start, end in flags:
        figure.add_vrect(
            x0=start,
            x1=end,
            fillcolor=_rgba(INK, 0.07),
            line_width=0,
            layer="below",
            annotation_text="reported",
            annotation_position="top left",
            annotation={"font": {"size": 9, "color": MUTED}},
        )


def wear_trend_figure(
    run: Run, strokes: np.ndarray, flags: list[tuple[int, int]] | None = None
) -> go.Figure:
    """Both stations on the friction axis over the recent stretch of the run."""
    figure = go.Figure()
    _flag_bands(figure, flags or [])
    _band(figure, 0.0, 0.35, GOOD)
    _band(figure, 0.35, 0.7, WATCH)
    _band(figure, 0.7, 1.0, CRITICAL)

    for index, key in enumerate(STATIONS):
        values = run.position[key][strokes]
        colour = [INK, MUTED][index]
        figure.add_trace(
            go.Scatter(
                x=strokes,
                y=values,
                mode="markers",
                marker={"size": 4, "color": _rgba(colour, 0.28)},
                name=f"{STATIONS[key].name}, per stroke",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=strokes,
                y=trailing(run.position[key], WEAR_WINDOW)[strokes],
                mode="lines",
                line={"color": colour, "width": 2.5, "dash": ["solid", "dot"][index]},
                name=STATIONS[key].name,
                hovertemplate="stroke %{x}: %{y:.0%}<extra></extra>",
            )
        )

    figure.update_yaxes(range=[-0.05, 1.05], tickformat=".0%")
    return _frame(figure, strokes, "Pristine tool → worn tool")


def alignment_trend_figure(
    run: Run, strokes: np.ndarray, tolerance_mm: float, flags: list[tuple[int, int]] | None = None
) -> go.Figure:
    """Predicted offset at the cup, per stroke and smoothed, against the tolerance."""
    figure = go.Figure()
    _flag_bands(figure, flags or [])
    ceiling = max(tolerance_mm * 1.35, float(run.alignment_mm[strokes].max()) * 1.15)
    _band(figure, 0.0, 0.75 * tolerance_mm, GOOD)
    _band(figure, 0.75 * tolerance_mm, tolerance_mm, WATCH)
    _band(figure, tolerance_mm, ceiling, CRITICAL)

    figure.add_trace(
        go.Scatter(
            x=strokes,
            y=run.alignment_true_mm[strokes],
            mode="lines",
            line={"color": MUTED, "width": 1.5, "dash": "dash", "shape": "hv"},
            name="Actually fed",
            hovertemplate="stroke %{x}: %{y:.2f} mm<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=strokes,
            y=run.alignment_mm[strokes],
            mode="markers",
            marker={"size": 4, "color": _rgba(INK, 0.28)},
            name="Predicted, per stroke",
            hovertemplate="stroke %{x}: %{y:.2f} mm<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=strokes,
            y=trailing(run.alignment_mm, ALIGNMENT_WINDOW)[strokes],
            mode="lines",
            line={"color": INK, "width": 2.5},
            name=f"Predicted, last {ALIGNMENT_WINDOW}",
            hovertemplate="stroke %{x}: %{y:.2f} mm<extra></extra>",
        )
    )

    figure.update_yaxes(range=[0, ceiling])
    return _frame(figure, strokes, "Off-centre at the cup (mm)")


def alignment_dots(
    values: np.ndarray, ceiling: float, limit: float, smoothed: np.ndarray
) -> go.Figure:
    """One dot per stroke on a ruled grid -- a control chart, read at a glance.

    A dot per stroke rather than a line because each stroke really is an independent
    reading: the scatter is the measurement's own, and a line implies a continuity between
    consecutive strokes that the data does not have. Each dot carries its own state
    colour, so a handful of excursions is visibly different from a settled drift upward,
    which one colour for the whole trace would hide.

    The trailing mean the alarm actually watches is drawn over the top, so the card shows
    both the reading and the thing being judged.
    """
    figure = go.Figure()
    figure.add_hline(y=limit, line={"color": COLOR[CRITICAL], "width": 1, "dash": "dash"})
    figure.add_hline(
        y=WATCH_FRACTION * limit, line={"color": COLOR[WATCH], "width": 1, "dash": "dot"}
    )
    figure.add_trace(
        go.Scatter(
            x=np.arange(len(values)),
            y=values,
            mode="markers",
            marker={
                "size": 5,
                "color": [COLOR[alignment_state(value, limit)] for value in values],
            },
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=np.arange(len(smoothed)),
            y=smoothed,
            mode="lines",
            line={"color": INK, "width": 1.5},
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        height=104,
        margin={"l": 34, "r": 6, "t": 6, "b": 6},
        xaxis={
            "showgrid": True,
            "gridcolor": GRID,
            "showticklabels": False,
            "ticks": "",
            "zeroline": False,
            "fixedrange": True,
        },
        yaxis={
            "range": (0.0, ceiling),
            "showgrid": True,
            "gridcolor": GRID,
            "tickfont": {"size": 9, "color": MUTED},
            "ticks": "",
            "zeroline": False,
            "fixedrange": True,
            "nticks": 4,
        },
        showlegend=False,
    )
    return figure


def stroke_log(run: Run, stroke: int, tolerance_mm: float) -> dbc.Table:
    """The most recent strokes, newest first, with what each model made of them.

    The per-stroke classification is shown raw, unsmoothed -- this is the table someone
    opens to see whether a single stroke drove the state, and a smoothed table could not
    answer that.
    """
    header = html.Thead(
        html.Tr(
            [
                html.Th("Stroke"),
                *(html.Th(STATIONS[key].name) for key in STATIONS),
                html.Th("Off-centre"),
            ]
        )
    )

    rows = []
    for index in reversed(run.window(stroke, LOG_ROWS)):
        cells = [html.Td(f"{index + 1}", className="log-stroke")]
        for key in STATIONS:
            level = run.level(key, index)
            cells.append(
                html.Td(
                    [
                        html.Span(
                            STATIONS[key].level_name(level),
                            className="log-level",
                            style={"background": LEVEL_COLORS[level]},
                        ),
                        html.Span(f"{run.confidence(key, index):.0%}", className="log-confidence"),
                    ]
                )
            )
        offset = run.alignment_mm[index]
        cells.append(
            html.Td(
                f"{offset:.2f} mm",
                style={"color": COLOR[CRITICAL]} if offset >= tolerance_mm else None,
            )
        )
        rows.append(html.Tr(cells, className="log-current" if index == stroke else None))

    return dbc.Table([header, html.Tbody(rows)], borderless=True, size="sm", className="stroke-log")


def confidence_bars(run: Run, station_key: str, stroke: int) -> html.Div:
    """How the classifier split its probability on the current stroke."""
    station = STATIONS[station_key]
    called = majority_level(run, station_key, stroke)
    rows = []
    for level, share in enumerate(run.proba[station_key][stroke], start=1):
        rows.append(
            html.Div(
                [
                    html.Div(station.level_name(level), className="proba-name"),
                    html.Div(
                        html.Div(
                            style={
                                "width": f"{max(share, 0.005):.1%}",
                                "background": LEVEL_COLORS[level],
                            },
                            className="proba-fill",
                        ),
                        className="proba-track",
                    ),
                    html.Div(f"{share:.0%}", className="proba-value"),
                ],
                className="proba-row" + (" proba-called" if level == called else ""),
            )
        )
    return html.Div(rows, className="proba-block")
