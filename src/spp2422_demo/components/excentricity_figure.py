"""Figures for the strip-misalignment page.

Three views of the same stroke, at three zoom levels: the whole punch stroke, the plateau
the line is fitted to, and the two-number feature space every stroke is reduced to.

The measured stroke keeps the ink colour it has on the wear pages, so a stroke reads as a
stroke everywhere in the dashboard. The fitted line and the centred reference are told
apart by dash pattern as well as hue, which is what carries them for a colourblind reader
-- the two neutrals are deliberately outside the categorical palette.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..excentricity import Excentricity, excentricity_mm
from ..theme import GRID, INK, MUTED, OKABE_ITO

FIT_COLOR = OKABE_ITO[5]  # vermillion
REFERENCE_COLOR = MUTED
PLATEAU_FILL = "rgba(0,114,178,0.08)"

TIME_TITLE = "Time on the press cycle (s)"
FORCE_TITLE = "Punch force (kN)"
# The stroke is flat outside this; showing it all would squeeze the plateau to nothing.
TIME_RANGE = (0.225, 0.350)

GOOD, WATCH, CRITICAL = "#1B9E73", "#E6A400", "#D6453D"


def status_of(value_mm: float, threshold_mm: float) -> tuple[str, str, str]:
    """Colour, label and icon for a predicted eccentricity against the alarm threshold.

    Status is never carried by colour alone -- every caller renders the label beside it.
    """
    if value_mm >= threshold_mm:
        return CRITICAL, "Out of tolerance", "⚠"
    if value_mm >= 0.75 * threshold_mm:
        return WATCH, "Approaching the limit", "▲"
    return GOOD, "Within tolerance", "✓"


def _force_range(data: Excentricity) -> tuple[float, float]:
    """Y limits over every stroke, so heights compare from one stroke to the next."""
    forces = data.curves * data.force_scale
    low, high = float(forces.min()), float(forces.max())
    pad = 0.06 * (high - low)
    return low - pad, high + pad


def stroke_figure(data: Excentricity, index: int) -> go.Figure:
    """One stroke, the window the features come from, and the line fitted across it."""
    start, end = data.plateau
    figure = go.Figure()

    figure.add_vrect(
        x0=float(data.time[start]),
        x1=float(data.time[end]),
        fillcolor=PLATEAU_FILL,
        line_width=0,
        layer="below",
        annotation={"text": "plateau window", "font": {"size": 10, "color": MUTED}},
        annotation_position="top left",
    )

    figure.add_trace(
        go.Scatter(
            x=data.time,
            y=data.mean_curve_kn(0),
            name="Centred reference (60.00 mm)",
            line={"color": REFERENCE_COLOR, "width": 1.6, "dash": "dot"},
            hovertemplate="%{y:.3f} kN<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=data.time,
            y=data.force_kn(index),
            name="This stroke",
            line={"color": INK, "width": 2.2},
            hovertemplate="t = %{x:.4f} s<br>%{y:.3f} kN<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=data.time[start:end],
            y=data.fitted_line_kn(index),
            name=f"Plateau fit · {data.slope_kn_per_s(index):+.2f} kN/s",
            line={"color": FIT_COLOR, "width": 2.6, "dash": "dash"},
            hovertemplate="fit: %{y:.3f} kN<extra></extra>",
        )
    )

    figure.update_layout(
        xaxis={"title": TIME_TITLE, "range": list(TIME_RANGE)},
        yaxis={"title": FORCE_TITLE, "range": _force_range(data)},
        margin={"l": 60, "r": 20, "t": 30, "b": 45},
        height=390,
        hovermode="x unified",
    )
    return figure


def plateau_figure(data: Excentricity, index: int) -> go.Figure:
    """The plateau alone, where the tilt the model reads is actually visible.

    On the full-stroke axis a 2 kN/s tilt across 39 ms is a fraction of a millimetre of
    ink. Zoomed to the window and autoscaled, it is the whole picture.
    """
    start, end = data.plateau
    time = data.time[start:end]
    measured = data.force_kn(index)[start:end]
    fitted = data.fitted_line_kn(index)
    slope = data.slope_kn_per_s(index)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=time,
            y=measured,
            name="Measured",
            mode="lines",
            line={"color": INK, "width": 1.6},
            hovertemplate="t = %{x:.4f} s<br>%{y:.3f} kN<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=time,
            y=fitted,
            name=f"Fit · slope {slope:+.2f} kN/s",
            line={"color": FIT_COLOR, "width": 2.6, "dash": "dash"},
            hovertemplate="fit: %{y:.3f} kN<extra></extra>",
        )
    )

    # The drop the slope describes, stated in kN rather than left to be read off the axis.
    drop = fitted[-1] - fitted[0]
    figure.add_annotation(
        x=float(time[-1]),
        y=float(fitted[-1]),
        text=f"{drop:+.3f} kN across the window",
        showarrow=True,
        arrowhead=0,
        arrowcolor=FIT_COLOR,
        ax=-70,
        ay=-28,
        font={"size": 11, "color": FIT_COLOR},
    )

    figure.update_layout(
        xaxis={"title": TIME_TITLE},
        yaxis={"title": FORCE_TITLE},
        margin={"l": 60, "r": 20, "t": 30, "b": 45},
        height=300,
    )
    return figure


def feature_space_figure(data: Excentricity, index: int) -> go.Figure:
    """Every stroke as the model sees it: slope against intercept, shaded by true infeed.

    Magnitude, so one hue light to dark rather than seven categorical colours -- the
    levels are ordered, and reading the gradient is the point.
    """
    slopes = data.features[:, 0] * data.force_scale / data.dt
    intercepts = (data.features[:, 1] + data.features[:, 0] * data.plateau[0]) * data.force_scale

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=slopes,
            y=intercepts,
            mode="markers",
            marker={
                "size": 8,
                "color": [excentricity_mm(label) for label in data.labels],
                "colorscale": "Blues",
                "cmin": 0,
                "cmax": excentricity_mm(30),
                "line": {"color": "rgba(255,255,255,0.85)", "width": 1},
                "colorbar": {
                    "title": {"text": "True<br>eccentricity<br>(mm)", "font": {"size": 10}},
                    "thickness": 10,
                    "len": 0.85,
                    "tickfont": {"size": 10},
                },
            },
            customdata=np.stack(
                [data.labels / 100 * 3, data.stroke_index, data.fit_quality], axis=1
            ),
            hovertemplate=(
                "slope %{x:+.2f} kN/s<br>intercept %{y:.3f} kN"
                "<br>true %{customdata[0]:.2f} mm · stroke %{customdata[1]}"
                "<br>fit R² %{customdata[2]:.2f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[slopes[index]],
            y=[intercepts[index]],
            mode="markers",
            marker={
                "size": 15,
                "color": "rgba(0,0,0,0)",
                "line": {"color": FIT_COLOR, "width": 2.5},
            },
            name="This stroke",
            hovertemplate="this stroke<extra></extra>",
            showlegend=False,
        )
    )

    figure.update_layout(
        xaxis={"title": "Plateau slope (kN/s)"},
        yaxis={"title": "Plateau force at window start (kN)"},
        margin={"l": 60, "r": 20, "t": 30, "b": 45},
        height=300,
    )
    return figure


def indicator_figure(
    predicted_mm: float, running_mm: float, true_mm: float, threshold_mm: float
) -> go.Figure:
    """Predicted eccentricity on the scale the press actually spans, against the limit.

    A bar rather than a dial: position on a common scale is read more accurately than
    angle, and the threshold has to sit on the same scale to be comparable at a glance.
    Three marks on one axis -- this stroke, the last ten averaged, and the measured
    truth -- so the scatter of a single stroke is visible rather than implied.
    """
    color, _, _ = status_of(predicted_mm, threshold_mm)
    axis_max = excentricity_mm(30) + 0.05

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=[predicted_mm],
            y=["predicted"],
            orientation="h",
            marker={"color": color},
            width=0.5,
            hovertemplate="predicted %{x:.3f} mm<extra></extra>",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[running_mm],
            y=["predicted"],
            mode="markers",
            marker={"symbol": "diamond", "size": 11, "color": MUTED},
            name="last 10 strokes",
            hovertemplate="last 10 strokes: %{x:.3f} mm<extra></extra>",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[true_mm],
            y=["predicted"],
            mode="markers",
            marker={
                "symbol": "line-ns",
                "size": 26,
                "line": {"color": INK, "width": 2.5},
            },
            name="true",
            hovertemplate="true %{x:.3f} mm<extra></extra>",
            showlegend=False,
        )
    )
    figure.add_vline(
        x=threshold_mm,
        line={"color": CRITICAL, "width": 2, "dash": "dash"},
        annotation={"text": "limit", "font": {"size": 10, "color": CRITICAL}},
        annotation_position="top",
    )

    figure.update_layout(
        xaxis={
            "range": [0, axis_max],
            "title": "Eccentricity at the cup (mm)",
            "tickvals": [excentricity_mm(level) for level in (0, 10, 20, 30)],
            "gridcolor": GRID,
        },
        yaxis={"showticklabels": False, "showgrid": False},
        margin={"l": 10, "r": 20, "t": 26, "b": 40},
        height=120,
        bargap=0.4,
    )
    return figure
