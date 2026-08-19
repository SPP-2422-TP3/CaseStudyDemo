"""Strip misalignment: stepping through measured strokes and reading the infeed off them.

Only measured strokes appear here. The simulated sweep is what the underlying paper used
to establish that the plateau slope carries the infeed at all; this page is the shop-floor
half of that argument, so everything on screen is a real stroke off the press.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, State, callback, dcc, html, no_update

from spp2422_demo.components.excentricity_alert import alert_facts, excentricity_alert
from spp2422_demo.components.excentricity_figure import (
    feature_space_figure,
    indicator_figure,
    plateau_figure,
    status_of,
    stroke_figure,
)
from spp2422_demo.components.layout import caveat, page_header, panel, stat_card
from spp2422_demo.excentricity import (
    INFEED_LEVELS,
    REFERENCE_FEED_MM,
    excentricity_mm,
    load_excentricity,
    overfeed_mm,
)

dash.register_page(__name__, path="/excentricity", name="Excentricity", order=5)

STREAM_INTERVAL_MS = 700
STROKES_PER_SERIES = 49
RUNNING_WINDOW = 10  # strokes averaged alongside the stroke-by-stroke prediction
RE_ARM_FRACTION = 0.75  # the alarm re-arms only below this share of the limit
# Default alarm limit. Chosen for the demo, not taken from the paper -- no scrap
# tolerance was published with the trials -- and adjustable on the page for that reason.
DEFAULT_THRESHOLD_MM = 0.60


def series_label(level: int) -> str:
    return f"{REFERENCE_FEED_MM + overfeed_mm(level):.2f} mm"


def _series_option(level: int) -> dict:
    return {
        "label": f"{series_label(level)}  ·  +{overfeed_mm(level):.2f} overfeed "
        f"→ {excentricity_mm(level):.2f} mm off-centre",
        "value": level,
    }


def layout(**_kwargs):
    data = load_excentricity()

    controls = dbc.Row(
        [
            dbc.Col(
                [
                    html.Div("Measured series (tool infeed)", className="form-label"),
                    dcc.Dropdown(
                        id="exc-series",
                        options=[_series_option(level) for level in INFEED_LEVELS],
                        value=INFEED_LEVELS[0],
                        clearable=False,
                    ),
                ],
                md=4,
            ),
            dbc.Col(
                [
                    html.Div("Stroke", className="form-label"),
                    dcc.Slider(
                        id="exc-stroke",
                        min=1,
                        max=STROKES_PER_SERIES,
                        step=1,
                        value=1,
                        marks={1: "1", 25: "25", STROKES_PER_SERIES: str(STROKES_PER_SERIES)},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ],
                md=3,
            ),
            dbc.Col(
                [
                    html.Div("Alarm limit (mm at the cup)", className="form-label"),
                    dcc.Slider(
                        id="exc-threshold",
                        min=0.15,
                        max=0.90,
                        step=0.15,
                        value=DEFAULT_THRESHOLD_MM,
                        marks={
                            value: f"{value:.2f}" for value in (0.15, 0.30, 0.45, 0.60, 0.75, 0.90)
                        },
                    ),
                ],
                md=3,
            ),
            dbc.Col(
                [
                    html.Div("Alarm watches", className="form-label"),
                    dbc.RadioItems(
                        id="exc-alarm-source",
                        options=[
                            {"label": f"Last {RUNNING_WINDOW}", "value": "running"},
                            {"label": "This stroke", "value": "stroke"},
                        ],
                        value="running",
                        inline=True,
                    ),
                ],
                md=2,
            ),
            dbc.Col(
                [
                    html.Div("Live", className="form-label"),
                    dbc.Button(
                        "▶ Stream",
                        id="exc-stream-toggle",
                        color="primary",
                        outline=True,
                        className="w-100",
                    ),
                ],
                md=2,
            ),
        ],
        className="g-3 align-items-end",
    )

    return html.Div(
        [
            page_header(
                "Strip Misalignment",
                "Reading how far off-centre the blank sits from the shape of the "
                "deep-drawing punch force.",
            ),
            dbc.Card(dbc.CardBody(controls), className="mb-4"),
            dbc.Row(
                [
                    dbc.Col(
                        panel(
                            "Punch Force of the Selected Stroke",
                            dcc.Graph(id="exc-stroke-graph", config={"displayModeBar": False}),
                            note=(
                                "The dotted line is the mean stroke of the centred series, so "
                                "the tilt and the drop in peak force can be read against a "
                                "reference. The force axis is fixed across every stroke."
                            ),
                        ),
                        lg=8,
                    ),
                    dbc.Col(
                        panel(
                            "Predicted Misalignment",
                            html.Div(id="exc-prediction", className="mb-2"),
                            dcc.Graph(id="exc-indicator", config={"displayModeBar": False}),
                            html.Div(id="exc-features", className="mt-2"),
                        ),
                        lg=4,
                    ),
                ],
                className="g-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        panel(
                            "The Plateau, and the Line Fitted across It",
                            dcc.Graph(id="exc-plateau-graph", config={"displayModeBar": False}),
                            note=(
                                "Zoomed to the fitting window. The tilt is a fraction of a "
                                "percent of peak force, which is why it is invisible at full "
                                "scale and why the model measures it rather than looking at it."
                            ),
                        ),
                        lg=6,
                    ),
                    dbc.Col(
                        panel(
                            "Every Measured Stroke, as the Model Sees It",
                            dcc.Graph(id="exc-feature-graph", config={"displayModeBar": False}),
                            note=(
                                "The whole 912-sample trace reduced to the two numbers the "
                                "forest is given. Darker points sit further off-centre; the "
                                "ringed point is the stroke on screen."
                            ),
                        ),
                        lg=6,
                    ),
                ],
                className="g-4 mt-0",
            ),
            html.Div(
                caveat(
                    [
                        html.Strong("One production run per infeed level. "),
                        html.Span(
                            "Each of the seven series is a single uninterrupted run of 49 "
                            "strokes, so strokes that share a tool temperature, a lubrication "
                            "state and one setup end up on both sides of the train/test split. "
                            "The error below therefore measures recognising a stroke from a run "
                            "the model has already seen — holding a whole run out is impossible "
                            "here, because that would remove its infeed level from training "
                            "entirely. Repeated runs per level are what an honest "
                            "generalisation estimate would need."
                        ),
                    ]
                ),
                className="mt-4",
            ),
            html.Div(panel("About This Model", _about(data)), className="mt-4"),
            excentricity_alert(),
            dcc.Interval(id="exc-interval", interval=STREAM_INTERVAL_MS, disabled=True),
            dcc.Store(id="exc-current"),
            dcc.Store(id="exc-alarm-latched", data=False),
            dcc.Store(id="exc-auto-stroke", data={"stroke": 1, "series": INFEED_LEVELS[0]}),
        ]
    )


def _snapped(data) -> np.ndarray:
    """Each prediction rounded to the nearest infeed level actually run on the press."""
    levels = np.array(INFEED_LEVELS)
    return levels[np.abs(data.predicted[:, None] - levels[None, :]).argmin(axis=1)]


def _exact_level_rate(data) -> float:
    return float((_snapped(data) == data.labels).mean())


def _near_level_rate(data) -> float:
    return float((np.abs(_snapped(data) - data.labels) <= 5).mean())


def _about(data) -> html.Div:
    def item(label: str, *children) -> dbc.Col:
        return dbc.Col(
            [html.Div(label, className="about-label"), html.Div(children, className="about-text")],
            md=4,
        )

    return html.Div(
        [
            html.Div("Random forest on two plateau features", style={"fontWeight": 600}),
            html.Div(
                "20 trees, depth 4, following Moske et al. (NUMISHEET 2025).",
                className="section-note mb-3",
            ),
            dbc.Row(
                [
                    item(
                        "Reads",
                        "Two numbers, and nothing else: the slope and the height of a straight "
                        "line fitted across the force plateau. As the blank goes off-centre the "
                        "plateau tilts downward and peak force falls, and that is the whole "
                        "signal — no spectrum, no shape descriptors, no second sensor.",
                    ),
                    item(
                        "Trained on",
                        f"{len(data.curves)} strokes: {len(INFEED_LEVELS)} measured series of "
                        f"{STROKES_PER_SERIES}, one per infeed level from "
                        f"{series_label(0)} to {series_label(30)}. The first stroke of each "
                        "series is dropped as a warm-up transient.",
                    ),
                    item(
                        "Gets right",
                        html.Div(
                            f"{data.mae_mm * 1000:.1f} µm mean absolute error "
                            f"(± {data.mae_std_mm * 1000:.1f}) on held-out strokes, in infeed "
                            "terms. That average hides a tail: a single stroke lands on the "
                            f"exact infeed level {_exact_level_rate(data):.0%} of the time and "
                            f"within one level {_near_level_rate(data):.0%} of the time, so the "
                            "stroke-by-stroke number is worth watching as a trend rather than "
                            "read one stroke at a time.",
                        ),
                        html.Div(
                            "Predictions on this page are out-of-fold: every stroke shown is "
                            "predicted by a forest that did not train on it.",
                            className="mt-1",
                        ),
                    ),
                ],
                className="g-4",
            ),
            html.Div(
                [
                    "Background: ",
                    html.A(
                        "Simulation Driven Modeling of Strip Misalignment",
                        href="https://doi.org/10.1088/1742-6596/3104/1/012058",
                        target="_blank",
                    ),
                    " (Moske, Schumann, Wüst, Kersting and Groche, 2025). The paper establishes "
                    "the slope–infeed relation on an FE friction sweep and transfers it to the "
                    "press; this page shows the measured half of that.",
                ],
                className="section-note mt-3",
            ),
        ]
    )


@callback(
    Output("exc-stroke-graph", "figure"),
    Output("exc-plateau-graph", "figure"),
    Output("exc-feature-graph", "figure"),
    Output("exc-indicator", "figure"),
    Output("exc-prediction", "children"),
    Output("exc-features", "children"),
    Output("exc-current", "data"),
    Input("exc-series", "value"),
    Input("exc-stroke", "value"),
    Input("exc-threshold", "value"),
)
def update_view(level, stroke, threshold):
    """Predict the selected stroke and draw it at all three zoom levels."""
    data = load_excentricity()
    index = data.row(int(level), int(stroke) - 1)

    predicted = excentricity_mm(data.predicted[index])
    running = excentricity_mm(data.running_mean(int(level), int(stroke) - 1, RUNNING_WINDOW))
    true = excentricity_mm(data.labels[index])
    color, status, icon = status_of(predicted, threshold)

    prediction = html.Div(
        [
            html.Div(
                f"{predicted:.2f} mm",
                style={"fontSize": "2.1rem", "fontWeight": 600, "color": color, "lineHeight": 1.1},
            ),
            html.Div("off-centre at the cup", className="section-note"),
            html.Div(
                f"{icon}  {status}",
                style={"color": color, "fontWeight": 600},
                className="mt-2",
            ),
            html.Div(
                (
                    f"Last {min(RUNNING_WINDOW, int(stroke))} strokes averaged: {running:.2f} mm"
                    if stroke > 1
                    else "First stroke of the run — no average yet"
                )
                + f" · measured {series_label(int(level))} ({true:.2f} mm)",
                className="section-note mt-1",
            ),
        ]
    )

    features = dbc.Row(
        [
            dbc.Col(
                stat_card(
                    "Plateau Slope",
                    f"{data.slope_kn_per_s(index):+.2f}",
                    "kN/s — tilts down as the blank goes off-centre",
                )
            ),
            dbc.Col(
                stat_card(
                    "Plateau Force",
                    f"{data.intercept_kn(index):.2f}",
                    "kN at the start of the window — falls with misalignment",
                )
            ),
        ],
        className="g-2",
    )

    return (
        stroke_figure(data, index),
        plateau_figure(data, index),
        feature_space_figure(data, index),
        indicator_figure(predicted, running, true, threshold),
        prediction,
        features,
        {
            "predicted": predicted,
            "running": running,
            "running_n": min(RUNNING_WINDOW, int(stroke)),
            "true": true,
            "series": series_label(int(level)),
            "stroke": int(stroke),
        },
    )


@callback(
    Output("exc-stroke", "value"),
    Output("exc-series", "value"),
    Output("exc-auto-stroke", "data"),
    Input("exc-interval", "n_intervals"),
    State("exc-stroke", "value"),
    State("exc-series", "value"),
    prevent_initial_call=True,
)
def advance_stream(_, stroke, level):
    """Walk the strokes in production order, rolling on into the next infeed series.

    Running the series in order is what makes the stream worth watching: the misalignment
    climbs as it goes, so the limit is crossed on the way rather than being set up. The
    landing spot is echoed into `exc-auto-stroke` in the same response, so
    `pause_on_manual_stroke` below can tell its own writes apart from a hand on the
    controls.
    """
    if stroke < STROKES_PER_SERIES:
        next_stroke = stroke + 1
        return next_stroke, no_update, {"stroke": next_stroke, "series": level}
    position = INFEED_LEVELS.index(level)
    next_level = INFEED_LEVELS[(position + 1) % len(INFEED_LEVELS)]
    return 1, next_level, {"stroke": 1, "series": next_level}


@callback(
    Output("exc-interval", "disabled"),
    Output("exc-stream-toggle", "children"),
    Output("exc-stream-toggle", "color"),
    Input("exc-stream-toggle", "n_clicks"),
    State("exc-interval", "disabled"),
    prevent_initial_call=True,
)
def toggle_stream(_, disabled):
    running = disabled  # the click flips it
    return (not running, "⏸ Pause" if running else "▶ Stream", "warning" if running else "primary")


@callback(
    Output("exc-interval", "disabled", allow_duplicate=True),
    Output("exc-stream-toggle", "children", allow_duplicate=True),
    Output("exc-stream-toggle", "color", allow_duplicate=True),
    Input("exc-stroke", "value"),
    Input("exc-series", "value"),
    State("exc-auto-stroke", "data"),
    State("exc-interval", "disabled"),
    prevent_initial_call=True,
)
def pause_on_manual_stroke(stroke, level, auto, disabled):
    """A hand on the stroke slider or the series picker always wins: stop the stream rather
    than race it for the value.

    `advance_stream` writes these from a `State` read that can go stale in flight, so a tick
    landing just after a manual move can silently snap the controls back. Stopping the
    stream the moment either control shows something other than its own last write closes
    that window instead of trying to win the race.
    """
    if disabled or auto is None or (stroke == auto["stroke"] and level == auto["series"]):
        return no_update, no_update, no_update
    return True, "▶ Stream", "primary"


@callback(
    Output("exc-alert-modal", "is_open"),
    Output("exc-alert-facts", "children"),
    Output("exc-interval", "disabled", allow_duplicate=True),
    Output("exc-stream-toggle", "children", allow_duplicate=True),
    Output("exc-stream-toggle", "color", allow_duplicate=True),
    Output("exc-alarm-latched", "data"),
    Input("exc-current", "data"),
    State("exc-threshold", "value"),
    State("exc-alarm-source", "value"),
    State("exc-alarm-latched", "data"),
    prevent_initial_call=True,
)
def raise_alarm(current, threshold, source, latched):
    """Stop the stream and interrupt on the crossing into out-of-tolerance.

    Latched on the transition, not on the state: once the operator has been told, walking
    further through an already-bad series must not reopen the modal on every stroke. The
    latch clears only once a stroke comes back well under the limit rather than at the
    limit itself -- a single stroke carries enough scatter to cross the line and recross
    it repeatedly, and an alarm that re-fires on that is an alarm nobody reads.
    """
    watched = current["running"] if source == "running" else current["predicted"]
    # A part-filled window is one noisy stroke wearing an average's name. Hold the alarm
    # off until it has the strokes it claims to have.
    if source == "running" and current["running_n"] < RUNNING_WINDOW:
        return no_update, no_update, no_update, no_update, no_update, latched
    if latched:
        cleared = watched < RE_ARM_FRACTION * threshold
        return no_update, no_update, no_update, no_update, no_update, not cleared
    if watched < threshold:
        return no_update, no_update, no_update, no_update, no_update, False

    facts = alert_facts(watched, current["true"], threshold, current["series"], current["stroke"])
    return True, facts, True, "▶ Stream", "primary", True


@callback(
    Output("exc-alert-modal", "is_open", allow_duplicate=True),
    Input("exc-alert-dismiss", "n_clicks"),
    prevent_initial_call=True,
)
def dismiss_alert(_):
    return False


@callback(
    Output("exc-alert-modal", "is_open", allow_duplicate=True),
    Output("exc-interval", "disabled", allow_duplicate=True),
    Output("exc-stream-toggle", "children", allow_duplicate=True),
    Output("exc-stream-toggle", "color", allow_duplicate=True),
    Input("exc-alert-resume", "n_clicks"),
    prevent_initial_call=True,
)
def resume_stream(_):
    return False, False, "⏸ Pause", "warning"
