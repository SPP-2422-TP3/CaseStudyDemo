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

import math

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, clientside_callback, ctx, dcc, html, no_update

from spp2422_demo.artifacts import load_artifacts
from spp2422_demo.components.layout import percent
from spp2422_demo.components.status_cards import (
    ALIGNMENT,
    MACHINE,
    WEAR,
    board,
    card_id,
    card_slots,
    detail,
)
from spp2422_demo.data import STATIONS
from spp2422_demo.feedback import (
    FEEDBACK_STROKES,
    ISSUES,
    WINDOW_CHOICES,
    from_store,
    report,
    to_store,
)
from spp2422_demo.health import (
    CRITICAL,
    DEFAULT_TOLERANCE_MM,
    ICON,
    MACHINE_HEADLINE,
    Signal,
    machine_state,
)
from spp2422_demo.scenario import (
    DEFAULT_MODEL,
    DEFAULT_SCENARIO,
    N_STROKES,
    SCENARIOS,
    WEAR_WINDOW,
    Run,
    load_run,
)

dash.register_page(__name__, path="/", name="Status", order=0, top_level=True)

# Operator-facing speed range for the run, in strokes per minute.
MIN_SPEED = 10
MAX_SPEED = 1000
DEFAULT_SPEED = 100
# The interval never ticks faster than this. It is not a fresh guess: it is the fixed
# cadence the board's redraw -- three cards, one of them a re-mounted Plotly graph, plus
# the separate stop-alert check, both round trips to the server on every tick -- already
# ran at before speed became adjustable, and it visibly kept up. A round trip measurably
# faster than this (down to ~150ms) still lags in practice, so rather than re-guess a
# lower floor, the ceiling stays pinned to the one cadence already proven safe. Past this
# point, a higher stroke rate is reached by advancing more strokes per tick rather than
# ticking faster -- see `_interval_and_step` and the clientside callbacks below, which
# mirror it in JS since the run itself is stepped in the browser.
MIN_TICK_MS = 600
# The single page behind the board; see the help link in `_machine_bar`.
DETAILS_PATH = "/details"
# The board opens with its trailing windows already full, so the first frame reads the
# same way every later one does rather than averaging a single stroke.
FIRST_STROKE = WEAR_WINDOW - 1


def _interval_and_step(speed_per_min: int) -> tuple[int, int]:
    """How often the interval should tick, and how many strokes it should advance each
    tick, to run at `speed_per_min` without ticking faster than `MIN_TICK_MS`.

    Below `MIN_TICK_MS`, redraws would be requested faster than a round trip to the
    server can clear them, so the extra speed is spent skipping strokes between redraws
    instead. Only used here to set the initial interval and step; the running clientside
    callbacks below repeat this same arithmetic in JS, since the run itself is stepped in
    the browser -- keep the two in step if either changes.
    """
    ms_per_stroke = 60_000 / speed_per_min
    strokes_per_tick = max(1, math.ceil(MIN_TICK_MS / ms_per_stroke))
    return round(ms_per_stroke * strokes_per_tick), strokes_per_tick


_INITIAL_INTERVAL_MS, _INITIAL_STEP = _interval_and_step(DEFAULT_SPEED)


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
            {"label": f"Data {scenario.name}", "value": key} for key, scenario in SCENARIOS.items()
        ],
        value=DEFAULT_SCENARIO,
        className="btn-group hmi-choice",
        inputClassName="btn-check",
        labelClassName="btn hmi-choice-button",
        labelCheckedClassName="active",
    )


def _machine_bar() -> html.Div:
    """The equipment strip: what this is, what it is running, how far into it, and -- since
    the board renders without the site's top bar -- the one way off it, in the far corner
    an operator expects a help icon to sit."""
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
            dcc.Link(
                html.I(className="fa fa-info-circle hmi-help-icon"),
                href=DETAILS_PATH,
                className="hmi-help",
                title="About & Help",
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


def _stop_alert() -> dbc.Modal:
    """The popup that interrupts the operator when the board's verdict turns critical.

    An alert head, a body a callback fills in, three buttons -- but the buttons are the
    board's own next moves rather than a per-station explanation: see where the board can
    go from a stopped press.
    """
    return dbc.Modal(
        [
            html.Div(
                dbc.Row(
                    [
                        dbc.Col(html.Div(ICON[CRITICAL], className="alert-icon"), width="auto"),
                        dbc.Col(
                            [
                                html.Div(MACHINE_HEADLINE[CRITICAL], className="alert-title"),
                                html.Div("Progressive die · press cell", className="alert-sub"),
                            ]
                        ),
                    ],
                    align="center",
                ),
                className="alert-head",
            ),
            dbc.ModalBody(html.Div(id="status-stop-body")),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "View details", id="status-stop-view", color="secondary", outline=True
                    ),
                    dbc.Button(
                        "Report bad part", id="status-stop-report", color="secondary", outline=True
                    ),
                    dbc.Button("Acknowledge", id="status-stop-ack", color="dark"),
                ]
            ),
        ],
        id="status-stop-modal",
        is_open=False,
        centered=True,
        size="lg",
        className="alert-modal",
    )


def _signal_detail(signal: Signal, run: Run, stroke: int) -> str:
    """The signal's own detail line, with extra context appended per signal kind."""
    if signal.key in STATIONS:
        return f"{signal.detail} · {run.confidence(signal.key, stroke):.0%} confidence"
    if signal.key == ALIGNMENT:
        return f"{signal.detail} · measured {run.alignment_true_mm[stroke]:.2f} mm at the cup"
    return signal.detail


# Practical guidance for the cause that tripped the stop, ported from the old
# `excentricity_alert` -- only alignment has an entry, since there was no wear-side
# equivalent to carry over when `wear_alert` was retired.
STOP_GUIDANCE = {
    ALIGNMENT: (
        "An off-centre blank draws one flange wide and thins the opposite wall. Check the "
        "feed before continuing."
    ),
}


def _stop_body(critical: list[Signal], stroke: int, run: Run) -> html.Div:
    """What tripped the stop, read the same way the machine card's own detail window reads it."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(ICON[signal.state], className="signal-icon"),
                            html.Span(signal.name, className="signal-name"),
                            html.Span(signal.value, className="signal-value"),
                            html.Div(
                                _signal_detail(signal, run, stroke), className="signal-detail"
                            ),
                        ],
                        className=f"signal signal-{signal.state}",
                    )
                    for signal in critical
                ],
                className="signal-list",
            ),
            *(
                html.Div(STOP_GUIDANCE[signal.key], className="section-note mt-2")
                for signal in critical
                if signal.key in STOP_GUIDANCE
            ),
            html.Div(f"Stroke {stroke + 1}", className="section-note mt-2"),
        ]
    )


def _stop_card(critical: list[Signal]) -> str:
    """Which detail view `View details` opens: the one signal that tripped it, if there is
    exactly one -- otherwise there is no single view to send the operator to, so it opens
    strip alignment.
    """
    if len(critical) == 1 and critical[0].key in STATIONS:
        return WEAR
    return ALIGNMENT


def _overall_accuracy(model_key: str) -> float:
    """One model's held-out accuracy pooled across both stations' test strokes.

    A per-station accuracy would favour whichever station happens to be easier, and the
    dropdown now offers one model for both stations at once, so the number next to it has
    to answer for both.
    """
    correct = total = 0
    for station_key in STATIONS:
        trained = load_artifacts(station_key)
        n_test = int((~trained.data.train_mask).sum())
        correct += round(trained.accuracy[model_key] * n_test)
        total += n_test
    return correct / total


def _model_control() -> dbc.Col:
    trained = load_artifacts(next(iter(STATIONS)))
    return dbc.Col(
        [
            html.Div("Tool Wear Model", className="form-label"),
            dcc.Dropdown(
                id="status-model",
                options=[
                    {
                        "label": (
                            f"{trained.models[key].name} "
                            f"({percent(_overall_accuracy(key))} accuracy)"
                        ),
                        "value": key,
                    }
                    for key in trained.models
                ],
                value=DEFAULT_MODEL,
                clearable=False,
            ),
        ],
        lg=6,
    )


def _speed_control() -> dbc.Col:
    return dbc.Col(
        [
            html.Div("Speed (strokes/min)", className="form-label"),
            dcc.Slider(
                id="status-speed",
                min=MIN_SPEED,
                max=MAX_SPEED,
                step=10,
                value=DEFAULT_SPEED,
                marks={value: str(value) for value in (10, 250, 500, 750, 1000)},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
        lg=6,
    )


def _visualization_controls() -> html.Div:
    return html.Div(
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
                    lg=6,
                ),
                _speed_control(),
            ],
            className="g-4 align-items-end",
        ),
        className="hmi-controls",
    )


def _model_settings_controls() -> html.Div:
    return html.Div(
        dbc.Row(
            [
                _model_control(),
                dbc.Col(
                    [
                        html.Div("Strip Alignment Tolerance (mm)", className="form-label"),
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
                    lg=6,
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
            html.Div(card_slots(), className="hmi-board"),
            html.Div(id="status-reports-strip"),
            _feedback_bar(),
            _visualization_controls(),
            _model_settings_controls(),
            dcc.Interval(id="status-interval", interval=_INITIAL_INTERVAL_MS, disabled=True),
            dcc.Store(id="status-step", data=_INITIAL_STEP),
            dcc.Store(id="status-running", data=False),
            dcc.Store(id="status-reports", data=[]),
            dcc.Store(id="status-feedback-anchor"),
            dcc.Store(id="status-last-machine-state"),
            _feedback_form(),
            _stop_alert(),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle(id="status-modal-title")),
                    dbc.ModalBody(id="status-modal-body"),
                ],
                id="status-modal",
                size="xl",
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
    Input("status-model", "value"),
)
def _board(stroke, tolerance, scenario_key, model_key):
    """Redraw the three card faces. The slots holding them are never replaced."""
    run = load_run(scenario_key, model_key)
    counter = html.Span(f"{stroke + 1:,}".replace(",", " "), className="hmi-counter-value")
    return (*board(run, stroke, tolerance), counter)


@callback(
    Output("status-stroke", "value", allow_duplicate=True),
    Output("status-reports", "data", allow_duplicate=True),
    Input("status-scenario", "value"),
    prevent_initial_call=True,
)
def _restart(_scenario_key):
    """A different scenario is a different press run: rewind, and drop the old reports.

    The reports are pinned to stroke numbers, and stroke 200 of one run has nothing to do
    with stroke 200 of the other. Carrying them across would put an operator's verdict
    against strokes they never saw.
    """
    return FIRST_STROKE, []


@callback(
    Output("status-feedback-modal", "is_open", allow_duplicate=True),
    Output("status-feedback-anchor", "data"),
    Output("status-feedback-when", "children"),
    Output("status-feedback-note", "value"),
    Output("status-stop-modal", "is_open", allow_duplicate=True),
    Input("status-report", "n_clicks"),
    Input("status-stop-report", "n_clicks"),
    State("status-stroke", "value"),
    prevent_initial_call=True,
)
def _open_feedback(_clicks, _stop_clicks, stroke):
    """Open the form and pin it to the stroke on screen now, not when it is submitted.

    Reachable from the board's own report button and from the stop popup's; either way the
    popup behind it closes.
    """
    when = f"Reporting from stroke {stroke + 1}, the last stroke the board has shown."
    return True, stroke, when, "", False


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
    State("status-model", "value"),
    prevent_initial_call=True,
)
def _record_report(
    _submit, _cancel, stored, anchor, window, issue, note, scenario_key, tolerance, model_key
):
    """Take the operator's word for it and record what the monitor said at the time."""
    if ctx.triggered_id == "status-feedback-cancel" or anchor is None:
        return no_update, False
    reports = from_store(stored)
    run = load_run(scenario_key, model_key)
    reports.append(report(run, anchor, window, issue, note or "", tolerance))
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


# The speed slider only ever picks strokes per minute; turning that into an interval and a
# per-tick step is `_interval_and_step` above, mirrored here in JS since the run itself is
# stepped in the browser. Once one-stroke-per-tick would mean ticking faster than
# MIN_TICK_MS, the extra speed instead shows up as a bigger step at that same tick rate, so
# the board skips frames cleanly instead of piling up redraws it can't clear in time.
clientside_callback(
    f"""
    function(speed) {{
        const msPerStroke = 60000 / speed;
        const strokesPerTick = Math.max(1, Math.ceil({MIN_TICK_MS} / msPerStroke));
        return [Math.round(msPerStroke * strokesPerTick), strokesPerTick];
    }}
    """,
    Output("status-interval", "interval"),
    Output("status-step", "data"),
    Input("status-speed", "value"),
)


# Stepping the run forward runs in the browser rather than on the server. A manual jump on
# the same slider is also just a browser-side write to its `value`, so doing the tick here
# too means the two can never race a network round trip against each other: whichever the
# browser processes last is what stays on screen, and a jump is never silently overwritten
# by a tick that was already in flight. It also means a jump no longer needs to pause the
# run to stay safe -- the next tick simply continues from wherever the slider now points.
clientside_callback(
    f"""
    function(n_intervals, stroke, step) {{
        const next = stroke + (step || 1);
        return (next > {N_STROKES - 1}) ? {FIRST_STROKE} : next;
    }}
    """,
    Output("status-stroke", "value"),
    Input("status-interval", "n_intervals"),
    State("status-stroke", "value"),
    State("status-step", "data"),
    prevent_initial_call=True,
)


@callback(
    Output("status-running", "data"),
    Input("status-play", "n_clicks"),
    State("status-running", "data"),
    prevent_initial_call=True,
)
def _toggle(_clicks, running):
    return not running


@callback(
    Output("status-interval", "disabled"),
    Output("status-play", "children"),
    Output("status-live-dot", "className"),
    Input("status-running", "data"),
)
def _render_run_controls(running):
    """The one place that turns "is it running" into the interval, the button and the dot.

    `_toggle` and `_raise_stop_alert` each have their own reason to start or stop the run,
    and each used to write the interval's `disabled`, the button's label and the dot's class
    directly -- independent writers racing for the same three outputs, which is exactly how
    the button and the dot could end up disagreeing with what was actually running. Now they
    only ever set `status-running`, and this callback is the single, deterministic view of
    it, so the controls can never show a state that store does not hold.
    """
    if running:
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
    State("status-model", "value"),
    prevent_initial_call=True,
)
def _open_detail(clicks, stroke, tolerance, stored, scenario_key, model_key):
    # The slots are permanent, so a stroke no longer retriggers this -- but returning to
    # the board from another page mounts them afresh, and only a real click carries a
    # count on the card that triggered it.
    if not ctx.triggered_id or not any(clicks or []):
        return no_update, no_update, no_update
    run = load_run(scenario_key, model_key)
    title, body = detail(ctx.triggered_id["card"], run, stroke, tolerance, from_store(stored))
    return True, title, body


@callback(
    Output("status-stop-modal", "is_open"),
    Output("status-stop-body", "children"),
    Output("status-running", "data", allow_duplicate=True),
    Output("status-last-machine-state", "data"),
    Input("status-stroke", "value"),
    Input("status-tolerance", "value"),
    Input("status-scenario", "value"),
    Input("status-model", "value"),
    State("status-last-machine-state", "data"),
    prevent_initial_call=True,
)
def _raise_stop_alert(stroke, tolerance, scenario_key, model_key, last_state):
    """Interrupt the operator on the transition into a stopped machine, not on every stroke
    that stays there -- the same rule the per-station wear alert used to apply, read here
    off the board's combined verdict instead of one station's level.
    """
    run = load_run(scenario_key, model_key)
    state, current_signals = machine_state(run, stroke, tolerance)
    if state == CRITICAL and last_state != CRITICAL:
        critical = [signal for signal in current_signals if signal.state == CRITICAL]
        return True, _stop_body(critical, stroke, run), False, state
    return no_update, no_update, no_update, state


@callback(
    Output("status-modal", "is_open", allow_duplicate=True),
    Output("status-modal-title", "children", allow_duplicate=True),
    Output("status-modal-body", "children", allow_duplicate=True),
    Output("status-stop-modal", "is_open", allow_duplicate=True),
    Input("status-stop-view", "n_clicks"),
    State("status-stroke", "value"),
    State("status-tolerance", "value"),
    State("status-reports", "data"),
    State("status-scenario", "value"),
    State("status-model", "value"),
    prevent_initial_call=True,
)
def _view_stop_cause(_clicks, stroke, tolerance, stored, scenario_key, model_key):
    """Send `View details` to the card that explains the stop; see `_stop_card`."""
    run = load_run(scenario_key, model_key)
    _state, current_signals = machine_state(run, stroke, tolerance)
    critical = [signal for signal in current_signals if signal.state == CRITICAL]
    title, body = detail(_stop_card(critical), run, stroke, tolerance, from_store(stored))
    return True, title, body, False


@callback(
    Output("status-stop-modal", "is_open", allow_duplicate=True),
    Input("status-stop-ack", "n_clicks"),
    prevent_initial_call=True,
)
def _ack_stop(_clicks):
    return False
