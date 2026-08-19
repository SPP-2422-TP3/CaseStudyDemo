"""The status board: three cards that say whether the press can keep running.

The rest of the dashboard argues about models. This page assumes the argument was won and
shows what the models are for -- one glance, three answers, and the evidence one click
behind each of them. It is laid out as press-side equipment rather than as a report: a
machine bar carrying the scenario and the stroke count, the cards, and the controls last.

The two scenarios are separate faults deliberately: one press run where the tools wear at
their own pace and the strip stays put, one where the strip walks off centre and the tools
do not. A board worth having has to say *which* of the two is happening, and it can only
be seen to do that if the two are shown apart.

What the board is assembled from, and what it may not be read as, is on the details page
rather than on the board itself -- a shop-floor screen is not where a caveat gets read.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update

from spp2422_demo.components.status_cards import (
    ALIGNMENT,
    MACHINE,
    WEAR,
    board,
    card_id,
    card_slots,
    detail,
)
from spp2422_demo.feedback import (
    FEEDBACK_STROKES,
    ISSUES,
    WINDOW_CHOICES,
    from_store,
    report,
    to_store,
)
from spp2422_demo.health import DEFAULT_TOLERANCE_MM
from spp2422_demo.scenario import (
    DEFAULT_SCENARIO,
    N_STROKES,
    SCENARIOS,
    WEAR_WINDOW,
    load_run,
)

dash.register_page(__name__, path="/", name="Status", order=0, top_level=True)

STREAM_INTERVAL_MS = 600
# The single page behind the board; see `_details_link`.
DETAILS_PATH = "/details"
# The board opens with its trailing windows already full, so the first frame reads the
# same way every later one does rather than averaging a single stroke.
FIRST_STROKE = WEAR_WINDOW - 1


def _scenario_picker() -> dbc.RadioItems:
    """Which of the two authored runs the board is watching.

    It sits in the machine bar rather than down with the sliders: the scenario decides
    what every card is saying, so it belongs where the press identifies itself, not among
    the controls that only change the view. The buttons carry their own names, so they
    get neither a label above nor a caption below.
    """
    return dbc.RadioItems(
        id="status-scenario",
        options=[
            {"label": f"Data {scenario.name}", "value": key}
            for key, scenario in SCENARIOS.items()
        ],
        value=DEFAULT_SCENARIO,
        className="btn-group hmi-choice",
        inputClassName="btn-check",
        labelClassName="btn hmi-choice-button",
        labelCheckedClassName="active",
    )


def _machine_bar() -> html.Div:
    """The equipment strip: what this is, what it is running, and how far into it."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Progressive Die · Press Cell", className="hmi-label"),
                    html.Div(
                        "Deep Drawing · Ironing · Strip Feed",
                        className="hmi-line",
                    ),
                ]
            ),
            _scenario_picker(),
            html.Div(
                [
                    html.Div("Stroke", className="hmi-label"),
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


def _feedback_bar() -> html.Div:
    """The operator's way into the board, given the weight that deserves.

    Everything else on this page is the press talking. This is the one control that
    carries information the other direction, so it is a full-width action rather than a
    button tucked into a toolbar.
    """
    return html.Div(
        [
            dbc.Button(
                [html.Span("⚑", className="feedback-icon"), "Report bad parts"],
                id="status-report",
                className="feedback-button",
            ),
            html.Div(
                [
                    html.Div(
                        "Something wrong with the parts coming off the die?",
                        className="feedback-line",
                    ),
                    html.Div(
                        "Report it against the strokes that produced them. The monitor's "
                        "reading of the same window is recorded alongside.",
                        className="feedback-note",
                    ),
                ]
            ),
        ],
        className="feedback-bar",
    )


def _feedback_form() -> dbc.Modal:
    """What the operator is asked, and nothing more: when, what, and anything else."""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Report Bad Parts")),
            dbc.ModalBody(
                [
                    html.Div(id="status-feedback-when", className="feedback-anchor"),
                    html.Div("How far back were the parts bad?", className="form-label mt-3"),
                    dbc.RadioItems(
                        id="status-feedback-window",
                        options=[
                            {"label": f"Last {count} strokes", "value": count}
                            for count in WINDOW_CHOICES
                        ],
                        value=FEEDBACK_STROKES,
                        inline=True,
                    ),
                    html.Div(
                        "Stated in strokes rather than minutes: press rate was never "
                        "recorded with these trials, so a time would be invented.",
                        className="section-note mt-1",
                    ),
                    html.Div("What was wrong with them?", className="form-label mt-3"),
                    dbc.RadioItems(
                        id="status-feedback-issue",
                        options=[{"label": label, "value": key} for key, label in ISSUES],
                        value=ISSUES[0][0],
                    ),
                    html.Div("Anything to add?", className="form-label mt-3"),
                    dbc.Textarea(
                        id="status-feedback-note",
                        placeholder="Optional — in your own words",
                        rows=2,
                    ),
                ]
            ),
            dbc.ModalFooter(
                [
                    html.Div(
                        "Recorded against these strokes. Nothing is retrained.",
                        className="section-note me-auto",
                    ),
                    dbc.Button("Cancel", id="status-feedback-cancel", color="link"),
                    dbc.Button("Submit report", id="status-feedback-submit", color="dark"),
                ]
            ),
        ],
        id="status-feedback-modal",
        is_open=False,
    )


def _details_link() -> html.Div:
    """The way off the board.

    The status page renders without the site's top bar, so this is the only route to
    everything else. It is one link rather than a row of them: the board is a shop-floor
    screen, and the page it points at carries its own way on to the per-station pages.
    """
    return html.Div(
        dcc.Link("More Details", href=DETAILS_PATH, className="hmi-link"),
        className="hmi-links",
    )


def _controls() -> html.Div:
    return html.Div(
        [
            dbc.Row(
                [
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
                        lg=8,
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
                                    value: f"{value:.2f}"
                                    for value in (0.30, 0.45, 0.60, 0.75, 0.90)
                                },
                            ),
                        ],
                        lg=4,
                    ),
                ],
                className="g-4 align-items-end",
            ),
            _details_link(),
        ],
        className="hmi-controls",
    )


def layout(**_kwargs):
    return html.Div(
        [
            _machine_bar(),
            html.Div(card_slots(), className="hmi-board"),
            html.Div(id="status-reports-strip"),
            _feedback_bar(),
            _controls(),
            dcc.Interval(id="status-interval", interval=STREAM_INTERVAL_MS, disabled=True),
            dcc.Store(id="status-auto-stroke", data=FIRST_STROKE),
            dcc.Store(id="status-reports", data=[]),
            dcc.Store(id="status-feedback-anchor"),
            _feedback_form(),
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
    Output(card_id(MACHINE), "children"),
    Output(card_id(WEAR), "children"),
    Output(card_id(ALIGNMENT), "children"),
    Output("status-counter", "children"),
    Input("status-stroke", "value"),
    Input("status-tolerance", "value"),
    Input("status-scenario", "value"),
)
def _board(stroke, tolerance, scenario_key):
    """Redraw the three card faces. The slots holding them are never replaced."""
    run = load_run(scenario_key)
    counter = html.Span(f"{stroke + 1:,}".replace(",", " "), className="hmi-counter-value")
    return (*board(run, stroke, tolerance), counter)


@callback(
    Output("status-stroke", "value", allow_duplicate=True),
    Output("status-auto-stroke", "data", allow_duplicate=True),
    Output("status-reports", "data", allow_duplicate=True),
    Input("status-scenario", "value"),
    prevent_initial_call=True,
)
def _restart(_scenario_key):
    """A different scenario is a different press run: rewind, and drop the old reports.

    The reports are pinned to stroke numbers, and stroke 200 of one run has nothing to do
    with stroke 200 of the other. Carrying them across would put an operator's verdict
    against strokes they never saw. `status-auto-stroke` is rewound in the same response so
    a run in progress does not read as a manual jump and pause itself; see `_advance`.
    """
    return FIRST_STROKE, FIRST_STROKE, []


@callback(
    Output("status-feedback-modal", "is_open", allow_duplicate=True),
    Output("status-feedback-anchor", "data"),
    Output("status-feedback-when", "children"),
    Output("status-feedback-note", "value"),
    Input("status-report", "n_clicks"),
    State("status-stroke", "value"),
    prevent_initial_call=True,
)
def _open_feedback(_clicks, stroke):
    """Open the form and pin it to the stroke on screen now, not when it is submitted."""
    when = f"Reporting from stroke {stroke + 1}, the last stroke the board has shown."
    return True, stroke, when, ""


@callback(
    Output("status-reports", "data"),
    Output("status-feedback-modal", "is_open"),
    Input("status-feedback-submit", "n_clicks"),
    Input("status-feedback-cancel", "n_clicks"),
    State("status-reports", "data"),
    State("status-feedback-anchor", "data"),
    State("status-feedback-window", "value"),
    State("status-feedback-issue", "value"),
    State("status-feedback-note", "value"),
    State("status-scenario", "value"),
    State("status-tolerance", "value"),
    prevent_initial_call=True,
)
def _record_report(_submit, _cancel, stored, anchor, window, issue, note, scenario_key, tolerance):
    """Take the operator's word for it and record what the monitor said at the time."""
    if ctx.triggered_id == "status-feedback-cancel" or anchor is None:
        return no_update, False
    reports = from_store(stored)
    reports.append(report(load_run(scenario_key), anchor, window, issue, note or "", tolerance))
    return to_store(reports), False


@callback(
    Output("status-reports-strip", "children"),
    Input("status-reports", "data"),
)
def _reports_strip(stored):
    """A one-line acknowledgement on the board; the detail window carries the rest."""
    reports = from_store(stored)
    if not reports:
        return None
    latest = reports[-1]
    return html.Div(
        [
            html.Span("⚑", className="strip-icon"),
            html.Span(f"{len(reports)} operator report(s)", className="strip-count"),
            html.Span(
                f"latest: {latest.issue_label.lower()} at {latest.label}",
                className="strip-detail",
            ),
            html.Span(latest.disagreement(), className="strip-verdict"),
        ],
        className="report-strip",
    )


@callback(
    Output("status-stroke", "value"),
    Output("status-auto-stroke", "data"),
    Input("status-interval", "n_intervals"),
    State("status-stroke", "value"),
    prevent_initial_call=True,
)
def _advance(_ticks, stroke):
    """Step the run forward. The value is echoed into `status-auto-stroke` in the same
    response, so `_pause_on_manual_stroke` below can tell its own writes apart from a hand
    on the slider."""
    next_stroke = FIRST_STROKE if stroke >= N_STROKES - 1 else stroke + 1
    return next_stroke, next_stroke


@callback(
    Output("status-interval", "disabled", allow_duplicate=True),
    Output("status-play", "children", allow_duplicate=True),
    Output("status-live-dot", "className", allow_duplicate=True),
    Input("status-stroke", "value"),
    State("status-auto-stroke", "data"),
    State("status-interval", "disabled"),
    prevent_initial_call=True,
)
def _pause_on_manual_stroke(stroke, auto_stroke, disabled):
    """A hand on the slider always wins: stop the run rather than race it for the value.

    `_advance` writes the slider from a `State` read that can go stale in flight, so a tick
    landing just after a manual jump can silently snap the slider back. Stopping the run the
    moment the slider shows something other than its own last write closes that window
    instead of trying to win the race.
    """
    if disabled or stroke == auto_stroke:
        return no_update, no_update, no_update
    return True, "Run", "hmi-dot"


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
    State("status-stroke", "value"),
    State("status-tolerance", "value"),
    State("status-reports", "data"),
    State("status-scenario", "value"),
    prevent_initial_call=True,
)
def _open_detail(clicks, stroke, tolerance, stored, scenario_key):
    # The slots are permanent, so a stroke no longer retriggers this -- but returning to
    # the board from another page mounts them afresh, and only a real click carries a
    # count on the card that triggered it.
    if not ctx.triggered_id or not any(clicks or []):
        return no_update, no_update, no_update
    title, body = detail(
        ctx.triggered_id["card"], load_run(scenario_key), stroke, tolerance, from_store(stored)
    )
    return True, title, body
