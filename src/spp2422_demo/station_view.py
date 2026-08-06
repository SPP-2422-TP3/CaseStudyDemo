"""The wear-state page, shared by both forming stages.

Deep drawing and ironing differ only in which data they read, so one layout builder and
one set of pattern-matching callbacks serve both. Every component id carries its station
key, and the callbacks use `MATCH` to stay on their own page.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import numpy as np
from dash import MATCH, Input, Output, State, callback, ctx, dcc, html, no_update

from .artifacts import load_artifacts
from .components.curve_figure import confidence_figure, stroke_figure
from .components.layout import caveat, level_badge, page_header, panel
from .components.wear_alert import alert_facts, alert_id, wear_alert
from .data import LEVELS, STATIONS
from .explain import explain
from .theme import LEVEL_NAMES

CRITICAL = 3
STREAM_INTERVAL_MS = 900
STREAM_STEP = 7  # strokes per tick -- fast enough that drift is visible within a talk


def _id(station_key: str, part: str) -> dict[str, str]:
    return {"type": part, "station": station_key}


def layout(station_key: str) -> html.Div:
    station = STATIONS[station_key]
    trained = load_artifacts(station_key)
    data = trained.data
    runs = data.runs()
    default_run = runs.index((1, 1))

    controls = dbc.Row(
        [
            dbc.Col(
                [
                    html.Div("Model", className="form-label"),
                    dcc.Dropdown(
                        id=_id(station_key, "model-select"),
                        options=[
                            {"label": trained.models[key].name, "value": key}
                            for key in trained.models
                        ],
                        value=trained.default_model,
                        clearable=False,
                    ),
                ],
                md=3,
            ),
            dbc.Col(
                [
                    html.Div("Production run", className="form-label"),
                    dcc.Dropdown(
                        id=_id(station_key, "run-select"),
                        options=[
                            {"label": data.run_label(own, other), "value": f"{own}-{other}"}
                            for own, other in runs
                        ],
                        value=f"{runs[default_run][0]}-{runs[default_run][1]}",
                        clearable=False,
                    ),
                ],
                md=3,
            ),
            dbc.Col(
                [
                    html.Div("Stroke", className="form-label"),
                    dcc.Slider(
                        id=_id(station_key, "stroke-slider"),
                        min=0,
                        max=499,
                        step=1,
                        value=0,
                        marks={0: "0", 250: "250", 499: "499"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ],
                md=4,
            ),
            dbc.Col(
                [
                    html.Div("Live", className="form-label"),
                    dbc.Button(
                        "▶ Stream",
                        id=_id(station_key, "stream-toggle"),
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
                f"{station.name} — {station.german}",
                station.description,
            ),
            dbc.Card(dbc.CardBody(controls), className="mb-4"),
            dbc.Row(
                [
                    dbc.Col(
                        panel(
                            "Force signature of the selected stroke",
                            dcc.Graph(
                                id=_id(station_key, "stroke-graph"),
                                config={"displayModeBar": False},
                            ),
                            note=(
                                "Dotted lines are the mean curve of each wear level across the "
                                "whole dataset, shaded by one standard deviation."
                            ),
                        ),
                        lg=8,
                    ),
                    dbc.Col(
                        panel(
                            "Predicted tool state",
                            html.Div(id=_id(station_key, "prediction-panel"), className="mb-3"),
                            dcc.Graph(
                                id=_id(station_key, "confidence-graph"),
                                config={"displayModeBar": False},
                            ),
                            dbc.Button(
                                "Explain this prediction",
                                id=_id(station_key, "explain-button"),
                                color="primary",
                                outline=True,
                                size="sm",
                                className="mt-2 w-100",
                            ),
                        ),
                        lg=4,
                    ),
                ],
                className="g-4",
            ),
            dbc.Collapse(
                panel(
                    "Where the model found its evidence",
                    dcc.Graph(
                        id=_id(station_key, "explain-graph"), config={"displayModeBar": False}
                    ),
                    html.Div(id=_id(station_key, "explain-summary"), className="mt-2"),
                ),
                id=_id(station_key, "explain-collapse"),
                is_open=False,
                className="mt-4",
            ),
            html.Div(
                caveat(
                    [
                        html.Strong("Read the levels as tool condition, not as a measurement. "),
                        f"{station.level_prefix}1 to {station.level_prefix}3 are the roughness "
                        "classes recorded with the trials; no numerical roughness value was "
                        "logged alongside them.",
                    ]
                ),
                className="mt-4",
            ),
            wear_alert(station),
            dcc.Interval(
                id=_id(station_key, "stream-interval"), interval=STREAM_INTERVAL_MS, disabled=True
            ),
            dcc.Store(id=_id(station_key, "last-level"), data=None),
        ]
    )


def _row_index(data, run_value: str, stroke: int) -> int:
    own, other = (int(part) for part in run_value.split("-"))
    strokes = data.run_strokes(own, other)
    return int(strokes[min(stroke, len(strokes) - 1)])


@callback(
    Output(_id(MATCH, "stroke-graph"), "figure"),
    Output(_id(MATCH, "confidence-graph"), "figure"),
    Output(_id(MATCH, "prediction-panel"), "children"),
    Output(alert_id(MATCH, "modal"), "is_open"),
    Output(alert_id(MATCH, "facts"), "children"),
    Output(_id(MATCH, "last-level"), "data"),
    Input(_id(MATCH, "model-select"), "value"),
    Input(_id(MATCH, "run-select"), "value"),
    Input(_id(MATCH, "stroke-slider"), "value"),
    State(_id(MATCH, "last-level"), "data"),
)
def update_view(model_key, run_value, stroke, last_level):
    """Classify the selected stroke and raise the alert when it turns critical."""
    station_key = ctx.outputs_list[0]["id"]["station"]
    trained = load_artifacts(station_key)
    data = trained.data
    station = data.station

    own, other = (int(part) for part in run_value.split("-"))
    index = _row_index(data, run_value, stroke)
    peak_ref = data.peak_ref[index][None, :] if data.peak_ref is not None else None
    probabilities = trained.models[model_key].predict_proba(data.curves[index][None, :], peak_ref)[
        0
    ]
    level = LEVELS[int(np.argmax(probabilities))]
    confidence = float(probabilities.max())
    truth = station.level_name(int(data.labels[index]))

    panel_children = html.Div(
        [
            level_badge(level, station.level_name(level)),
            html.Div(
                f"{confidence:.0%} confidence · true label {truth}",
                className="section-note mt-2",
            ),
        ]
    )

    # Alert on the transition into the critical state, not on every stroke that stays there.
    became_critical = level == CRITICAL and last_level != CRITICAL
    facts = (
        alert_facts(station, level, confidence, data.run_label(own, other), stroke)
        if became_critical
        else no_update
    )

    return (
        stroke_figure(data, index),
        confidence_figure(station.level_prefix, probabilities),
        panel_children,
        True if became_critical else no_update,
        facts,
        level,
    )


@callback(
    Output(_id(MATCH, "stroke-slider"), "value"),
    Input(_id(MATCH, "stream-interval"), "n_intervals"),
    State(_id(MATCH, "stroke-slider"), "value"),
    prevent_initial_call=True,
)
def advance_stream(_, stroke):
    """Walk the run in production order, wrapping around at the end."""
    return ((stroke or 0) + STREAM_STEP) % 500


@callback(
    Output(_id(MATCH, "stream-interval"), "disabled"),
    Output(_id(MATCH, "stream-toggle"), "children"),
    Output(_id(MATCH, "stream-toggle"), "color"),
    Input(_id(MATCH, "stream-toggle"), "n_clicks"),
    State(_id(MATCH, "stream-interval"), "disabled"),
    prevent_initial_call=True,
)
def toggle_stream(_, disabled):
    running = disabled  # the click flips it
    return (not running, "■ Stop" if running else "▶ Stream", "danger" if running else "primary")


@callback(
    Output(_id(MATCH, "explain-collapse"), "is_open"),
    Output(_id(MATCH, "explain-graph"), "figure"),
    Output(_id(MATCH, "explain-summary"), "children"),
    Output(alert_id(MATCH, "modal"), "is_open", allow_duplicate=True),
    Input(_id(MATCH, "explain-button"), "n_clicks"),
    Input(alert_id(MATCH, "explain"), "n_clicks"),
    State(_id(MATCH, "model-select"), "value"),
    State(_id(MATCH, "run-select"), "value"),
    State(_id(MATCH, "stroke-slider"), "value"),
    prevent_initial_call=True,
)
def show_explanation(_page_click, _modal_click, model_key, run_value, stroke):
    """Attribute the current prediction over event time, and close the alert behind it."""
    # Imported here so the figure module is not a hard dependency of the layout import.
    from .components.curve_figure import attribution_figure

    station_key = ctx.outputs_list[0]["id"]["station"]
    trained = load_artifacts(station_key)
    data = trained.data
    model = trained.models[model_key]

    index = _row_index(data, run_value, stroke)
    peak_ref = data.peak_ref[index] if data.peak_ref is not None else None
    probabilities = model.predict_proba(
        data.curves[index][None, :], peak_ref[None, :] if peak_ref is not None else None
    )[0]
    level = LEVELS[int(np.argmax(probabilities))]

    attribution = explain(model, data.curves[index], peak_ref, level)
    summary = html.Div(
        [
            html.Div(attribution.summary(data.station.name)),
            html.Div(
                f"Method: {attribution.method} · explaining the case for "
                f"{data.station.level_name(level)} ({LEVEL_NAMES[level].lower()}).",
                className="section-note mt-1",
            ),
        ]
    )
    return True, attribution_figure(data, index, attribution), summary, False


@callback(
    Output(alert_id(MATCH, "modal"), "is_open", allow_duplicate=True),
    Input(alert_id(MATCH, "dismiss"), "n_clicks"),
    prevent_initial_call=True,
)
def dismiss_alert(_):
    return False
