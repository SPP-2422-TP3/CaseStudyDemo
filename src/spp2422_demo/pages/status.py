"""The status board: three cards that say whether the press can keep running.

The rest of the dashboard argues about models. This page assumes the argument was won and
shows what the models are for -- one glance, three answers, and the evidence one click
behind each of them.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update

from spp2422_demo.components.layout import caveat, page_header
from spp2422_demo.components.status_cards import board, detail
from spp2422_demo.health import DEFAULT_TOLERANCE_MM
from spp2422_demo.scenario import N_STROKES, SCENARIOS, WEAR_WINDOW, load_run

dash.register_page(__name__, path="/", name="Status", order=0, top_level=True)

STREAM_INTERVAL_MS = 600
# The board opens with its trailing windows already full, so the first frame reads the
# same way every later one does rather than averaging a single stroke.
FIRST_STROKE = WEAR_WINDOW - 1


def _controls() -> dbc.Row:
    return dbc.Row(
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
                lg=3,
                className="mb-2",
            ),
            dbc.Col(
                [
                    html.Div("Stroke", className="form-label"),
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
                className="mb-2",
            ),
            dbc.Col(
                [
                    html.Div("Tolerance at the cup (mm)", className="form-label"),
                    dcc.Slider(
                        id="status-tolerance",
                        min=0.30,
                        max=0.90,
                        step=0.15,
                        value=DEFAULT_TOLERANCE_MM,
                        marks={value: f"{value:.2f}" for value in (0.30, 0.45, 0.60, 0.75, 0.90)},
                    ),
                ],
                lg=2,
                className="mb-2",
            ),
            dbc.Col(
                [
                    html.Div("Live", className="form-label"),
                    dbc.Button("▶ Run", id="status-play", color="primary", outline=True),
                ],
                lg=2,
                className="mb-2",
            ),
        ],
        className="g-3 align-items-end mb-4",
    )


def layout(**_kwargs):
    return html.Div(
        [
            page_header(
                "Machine status",
                "Two forming stations and the strip feed, read off the press's own force "
                "signals, stroke by stroke.",
            ),
            _controls(),
            html.Div(id="status-board"),
            html.Div(id="status-summary", className="section-note mt-2"),
            caveat(
                [
                    html.Strong("This is an assembled run, not a recording. "),
                    "Every stroke on screen is a real measured stroke shown with its own "
                    "model's prediction. What is authored is the order they arrive in: the "
                    "data holds nine production runs at fixed wear levels and seven feed "
                    "series at fixed infeed, never a transition between them. Wear and "
                    "misalignment also come from two separate measurement campaigns on "
                    "separate tooling, so presenting them as one machine is a composition. ",
                    dcc.Link("The research pages", href="/overview"),
                    " show the same models without the staging.",
                ]
            ),
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
        ]
    )


@callback(
    Output("status-board", "children"),
    Output("status-summary", "children"),
    Input("status-scenario", "value"),
    Input("status-stroke", "value"),
    Input("status-tolerance", "value"),
)
def _board(scenario_key, stroke, tolerance):
    run = load_run(scenario_key)
    summary = (
        f"{run.scenario.summary} Wear read by "
        + ", ".join(f"{name.lower()}" for name in sorted(set(run.model_name.values())))
        + f"; stroke {stroke + 1} of {N_STROKES}."
    )
    return board(run, stroke, tolerance), summary


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
    Input("status-play", "n_clicks"),
    State("status-interval", "disabled"),
    prevent_initial_call=True,
)
def _toggle(_clicks, disabled):
    return (False, "❚❚ Pause") if disabled else (True, "▶ Run")


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
