"""The status board: three cards that say whether the press can keep running.

The rest of the dashboard argues about models. This page assumes the argument was won and
shows what the models are for -- one glance, three answers, and the evidence one click
behind each of them. It is laid out as press-side equipment rather than as a report: a
machine bar carrying the stroke count, the cards, and the controls last.

What the board is assembled from, and what it may not be read as, is in the Help glossary
rather than on the board itself -- a shop-floor screen is not where a caveat gets read.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update

from spp2422_demo.components.status_cards import board, detail
from spp2422_demo.health import DEFAULT_TOLERANCE_MM
from spp2422_demo.scenario import N_STROKES, SCENARIOS, WEAR_WINDOW, load_run

dash.register_page(__name__, path="/", name="Status", order=0, top_level=True)

STREAM_INTERVAL_MS = 600
# The board opens with its trailing windows already full, so the first frame reads the
# same way every later one does rather than averaging a single stroke.
FIRST_STROKE = WEAR_WINDOW - 1


def _machine_bar() -> html.Div:
    """The equipment strip: what this is, how many strokes it has run, and the run control."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Progressive die · press cell", className="hmi-label"),
                    html.Div(
                        "Deep drawing · Ironing · Strip feed",
                        className="hmi-line",
                    ),
                ]
            ),
            html.Div(
                [
                    html.Div("Strokes", className="hmi-label"),
                    html.Div(id="status-counter", className="hmi-counter"),
                ],
                className="hmi-count",
            ),
            html.Div(
                [
                    html.Span(id="status-live-dot", className="hmi-dot"),
                    dbc.Button("Run", id="status-play", className="hmi-button"),
                ],
                className="hmi-run",
            ),
        ],
        className="hmi-bar",
    )


def _controls() -> html.Div:
    return html.Div(
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div("Press run", className="form-label"),
                        dcc.Dropdown(
                            id="status-scenario",
                            options=[
                                {"label": scenario.name, "value": key}
                                for key, scenario in SCENARIOS.items()
                            ],
                            value=next(iter(SCENARIOS)),
                            clearable=False,
                        ),
                    ],
                    lg=4,
                ),
                dbc.Col(
                    [
                        html.Div("Jump to stroke", className="form-label"),
                        dcc.Slider(
                            id="status-stroke",
                            min=FIRST_STROKE,
                            max=N_STROKES - 1,
                            step=1,
                            value=FIRST_STROKE,
                            marks={
                                value: str(value + 1)
                                for value in (FIRST_STROKE, 99, 199, N_STROKES - 1)
                            },
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                    lg=5,
                ),
                dbc.Col(
                    [
                        html.Div("Alignment tolerance (mm)", className="form-label"),
                        dcc.Slider(
                            id="status-tolerance",
                            min=0.30,
                            max=0.90,
                            step=0.15,
                            value=DEFAULT_TOLERANCE_MM,
                            marks={
                                value: f"{value:.2f}" for value in (0.30, 0.45, 0.60, 0.75, 0.90)
                            },
                        ),
                    ],
                    lg=3,
                ),
            ],
            className="g-4 align-items-end",
        ),
        className="hmi-controls",
    )


def layout(**_kwargs):
    return html.Div(
        [
            _machine_bar(),
            html.Div(id="status-board", className="hmi-board"),
            _controls(),
            dcc.Interval(id="status-interval", interval=STREAM_INTERVAL_MS, disabled=True),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle(id="status-modal-title")),
                    dbc.ModalBody(id="status-modal-body"),
                ],
                id="status-modal",
                size="lg",
                is_open=False,
                scrollable=True,
            ),
        ],
        className="hmi",
    )


@callback(
    Output("status-board", "children"),
    Output("status-counter", "children"),
    Input("status-scenario", "value"),
    Input("status-stroke", "value"),
    Input("status-tolerance", "value"),
)
def _board(scenario_key, stroke, tolerance):
    counter = [
        html.Span(f"{stroke + 1:,}".replace(",", " "), className="hmi-counter-value"),
        html.Span(f"/ {N_STROKES}", className="hmi-counter-total"),
    ]
    return board(load_run(scenario_key), stroke, tolerance), counter


@callback(
    Output("status-stroke", "value"),
    Input("status-interval", "n_intervals"),
    State("status-stroke", "value"),
    prevent_initial_call=True,
)
def _advance(_ticks, stroke):
    return FIRST_STROKE if stroke >= N_STROKES - 1 else stroke + 1


@callback(
    Output("status-interval", "disabled"),
    Output("status-play", "children"),
    Output("status-live-dot", "className"),
    Input("status-play", "n_clicks"),
    State("status-interval", "disabled"),
    prevent_initial_call=True,
)
def _toggle(_clicks, disabled):
    if disabled:
        return False, "Pause", "hmi-dot hmi-dot-live"
    return True, "Run", "hmi-dot"


@callback(
    Output("status-modal", "is_open"),
    Output("status-modal-title", "children"),
    Output("status-modal-body", "children"),
    Input({"type": "status-card", "card": ALL}, "n_clicks"),
    State("status-scenario", "value"),
    State("status-stroke", "value"),
    State("status-tolerance", "value"),
    prevent_initial_call=True,
)
def _open_detail(clicks, scenario_key, stroke, tolerance):
    # Dash fires this when the board is rebuilt as well as when a card is clicked; only
    # an actual click carries a count on the card that triggered it.
    if not ctx.triggered_id or not any(clicks or []):
        return no_update, no_update, no_update
    title, body = detail(ctx.triggered_id["card"], load_run(scenario_key), stroke, tolerance)
    return True, title, body
