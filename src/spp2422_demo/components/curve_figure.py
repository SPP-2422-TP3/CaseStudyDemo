"""The figures: force curves, wear-level references, and attribution overlays."""

from __future__ import annotations

from functools import cache

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..data import EVENT_TIME, LEVELS, StationData, load_station
from ..explain import Attribution
from ..theme import GRID, INK, LEVEL_COLORS, MUTED, OKABE_ITO, SIMULATED
from .layout import percent

AXIS_TITLE = "Event time (0 = onset, 1 = end of event)"
FORCE_TITLE = "Normalized force"


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


@cache
def _force_range(station_key: str) -> tuple[float, float]:
    """Y limits wide enough for every stroke of one stage, with a little headroom.

    Fixed rather than autoscaled: stepping through a run rescales the axis on every
    stroke otherwise, which flattens the very thing the view is for. The ironing burst in
    particular varies by more than the rest of the curve, and an axis that follows it
    makes every stroke look the same height.
    """
    curves = load_station(station_key).curves
    low, high = float(curves.min()), float(curves.max())
    pad = 0.05 * (high - low)
    return low - pad, high + pad


def level_reference(data: StationData, level: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean curve and +/- one standard deviation band for one wear level."""
    curves = data.curves[data.labels == level]
    mean, std = curves.mean(axis=0), curves.std(axis=0)
    return mean, mean - std, mean + std


def stroke_figure(
    data: StationData, index: int, show_references: bool = True, label: str | None = None
) -> go.Figure:
    """One measured stroke against the mean curve of each wear level.

    `label` overrides the trace name. The station pages walk a production run and want its
    own stroke number; the status board walks an assembled run and would otherwise print a
    number that matches nothing the operator can see on screen.
    """
    figure = go.Figure()

    if show_references:
        for level in LEVELS:
            mean, low, high = level_reference(data, level)
            color = LEVEL_COLORS[level]
            figure.add_trace(
                go.Scatter(
                    x=np.concatenate([EVENT_TIME, EVENT_TIME[::-1]]),
                    y=np.concatenate([high, low[::-1]]),
                    fill="toself",
                    fillcolor=_rgba(color, 0.10),
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=EVENT_TIME,
                    y=mean,
                    name=f"{data.station.level_name(level)} reference",
                    line={"color": color, "width": 1.6, "dash": "dot"},
                    hovertemplate="%{y:.3f}<extra></extra>",
                )
            )

    figure.add_trace(
        go.Scatter(
            x=EVENT_TIME,
            y=data.curves[index],
            name=label or f"Stroke {int(data.stroke_index[index])}",
            line={"color": INK, "width": 2.2},
            hovertemplate="x = %{x:.3f}<br>force = %{y:.3f}<extra></extra>",
        )
    )

    figure.update_layout(
        xaxis={"title": AXIS_TITLE, "range": [0, 1]},
        yaxis={"title": FORCE_TITLE, "range": _force_range(data.station.key)},
        margin={"l": 60, "r": 20, "t": 10, "b": 45},
        height=380,
    )
    return figure


def attribution_figure(data: StationData, index: int, attribution: Attribution) -> go.Figure:
    """The stroke coloured by evidence, above the evidence profile itself.

    Both panels share an x-axis and the focus region is marked on both, so the region
    the model relied on can be read straight off the force curve.
    """
    curve = data.curves[index]
    values = attribution.values
    start, end = attribution.focus

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.06,
    )

    figure.add_trace(
        go.Scatter(
            x=EVENT_TIME,
            y=curve,
            line={"color": GRID, "width": 1.5},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=EVENT_TIME,
            y=curve,
            mode="markers",
            marker={
                "color": values,
                "colorscale": "RdBu",
                "reversescale": True,
                "cmid": 0,
                "size": 5,
                "colorbar": {
                    "title": {"text": "Evidence", "font": {"size": 11, "color": MUTED}},
                    "thickness": 10,
                    "len": 0.55,
                    "y": 0.78,
                    "tickfont": {"size": 10},
                },
            },
            hovertemplate="x = %{x:.3f}<br>force = %{y:.3f}<br>evidence = %{marker.color:.2f}"
            "<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=EVENT_TIME,
            y=values,
            fill="tozeroy",
            fillcolor=_rgba(LEVEL_COLORS[attribution.level], 0.28),
            line={"color": LEVEL_COLORS[attribution.level], "width": 1.4},
            hovertemplate="x = %{x:.3f}<br>evidence = %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    for row in (1, 2):
        figure.add_vrect(
            x0=start,
            x1=end,
            fillcolor=_rgba(LEVEL_COLORS[attribution.level], 0.09),
            line_width=0,
            layer="below",
            row=row,
            col=1,
        )

    figure.update_layout(
        margin={"l": 60, "r": 20, "t": 10, "b": 45},
        height=430,
        title={"text": ""},
    )
    figure.update_xaxes(range=[0, 1], row=1, col=1)
    figure.update_xaxes(title=AXIS_TITLE, range=[0, 1], row=2, col=1)
    # Same limits as the stroke figure above it, so the two panels read as one curve.
    figure.update_yaxes(title=FORCE_TITLE, range=_force_range(data.station.key), row=1, col=1)
    figure.update_yaxes(title="Evidence", row=2, col=1)
    return figure


def level_means_figure(data: StationData) -> go.Figure:
    """Mean measured curve per wear level -- how much the levels differ at all."""
    figure = go.Figure()
    for level in LEVELS:
        mean, low, high = level_reference(data, level)
        color = LEVEL_COLORS[level]
        figure.add_trace(
            go.Scatter(
                x=np.concatenate([EVENT_TIME, EVENT_TIME[::-1]]),
                y=np.concatenate([high, low[::-1]]),
                fill="toself",
                fillcolor=_rgba(color, 0.12),
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=EVENT_TIME,
                y=mean,
                name=data.station.level_name(level),
                line={"color": color, "width": 2},
                hovertemplate="%{y:.3f}<extra></extra>",
            )
        )
    figure.update_layout(
        xaxis={"title": AXIS_TITLE, "range": [0, 1]},
        yaxis={"title": FORCE_TITLE},
        margin={"l": 60, "r": 20, "t": 10, "b": 45},
        height=290,
    )
    return figure


def measured_vs_simulated_figure(data: StationData) -> go.Figure:
    """Measured mean curves beside the FE-simulated ones.

    Amplitudes are only comparable because both sides were mapped onto normalized force;
    the measured signals are uncalibrated volts and the simulated ones kN.
    """
    figure = go.Figure()
    for level in LEVELS:
        mean, _, _ = level_reference(data, level)
        figure.add_trace(
            go.Scatter(
                x=EVENT_TIME,
                y=mean,
                name=f"Measured {data.station.level_name(level)}",
                line={"color": LEVEL_COLORS[level], "width": 2},
                hovertemplate="%{y:.3f}<extra></extra>",
            )
        )
    for row, curve in enumerate(data.sim_curves):
        figure.add_trace(
            go.Scatter(
                x=EVENT_TIME,
                y=curve,
                name="FE simulation",
                legendgroup="sim",
                showlegend=row == 0,
                line={"color": _rgba(SIMULATED, 0.5), "width": 1},
                hovertemplate=f"mu = {data.sim_mu[row]:.2f}<br>%{{y:.3f}}<extra></extra>",
            )
        )
    figure.update_layout(
        xaxis={"title": AXIS_TITLE, "range": [0, 1]},
        yaxis={"title": FORCE_TITLE},
        margin={"l": 60, "r": 20, "t": 10, "b": 45},
        height=290,
    )
    return figure


def accuracy_figure(models: dict, accuracy: dict) -> go.Figure:
    """Held-out-stroke accuracy per model. One series, so the title carries the identity."""
    names = [models[key].name for key in models]
    figure = go.Figure(
        go.Bar(
            x=names,
            y=[accuracy[key] for key in models],
            marker={"color": OKABE_ITO[0]},
            text=[percent(accuracy[key]) for key in models],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>%{y:.2%} of held-out strokes<extra></extra>",
        )
    )
    figure.add_hline(
        y=1 / len(LEVELS),
        line={"color": MUTED, "width": 1, "dash": "dash"},
        annotation={"text": "chance", "font": {"size": 10, "color": MUTED}},
        annotation_position="top left",
    )
    figure.update_layout(
        showlegend=False,
        yaxis={"range": [0, 1.12], "tickformat": ".0%", "title": "Accuracy"},
        margin={"l": 60, "r": 20, "t": 10, "b": 40},
        height=280,
    )
    return figure
