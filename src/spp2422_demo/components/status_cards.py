"""The three cards the status board is, and the detail windows behind them.

Each card answers one question at a glance and opens into the evidence for its answer.
The cards are the whole interface; everything else on the page is the press controls that
drive them.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import numpy as np
from dash import dcc, html

from ..data import STATIONS
from ..health import (
    COLOR,
    GOOD,
    ICON,
    MACHINE_HEADLINE,
    MACHINE_LABEL,
    Signal,
    machine_state,
    worst,
)
from ..scenario import ALIGNMENT_WINDOW, WEAR_WINDOW, Run
from .status_figures import (
    LOG_ROWS,
    alignment_trend_figure,
    confidence_bars,
    sparkline,
    stroke_log,
    wear_trend_figure,
)

MACHINE, WEAR, ALIGNMENT = "machine", "wear", "alignment"
CARDS = (MACHINE, WEAR, ALIGNMENT)

# Strokes drawn on a card face and in its detail window.
FACE_STROKES = 60
DETAIL_STROKES = 100

CARD_TITLE = {
    MACHINE: "Machine status",
    WEAR: "Tool wear",
    ALIGNMENT: "Strip alignment",
}


def card_id(card: str) -> dict[str, str]:
    return {"type": "status-card", "card": card}


def _shell(card: str, state: str, *children, solid: bool = False) -> html.Div:
    """A clickable card, coloured by its state and named by its title.

    `solid` floods the whole card with the state colour instead of accenting its edge.
    The machine verdict uses it: a card that is green or red across its full area is
    legible from the other side of a press hall, where a coloured edge is not.

    The click lives on a wrapping div rather than the card: `dbc.Card` has no `n_clicks`,
    so a callback bound to the card itself would never fire.
    """
    fill = " status-solid" if solid else ""
    return html.Div(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.Span(CARD_TITLE[card], className="card-title"),
                            html.Span("Details ›", className="card-open"),
                        ],
                        className="card-head",
                    ),
                    *children,
                ]
            ),
            className=f"status-card status-{state}{fill} h-100",
        ),
        id=card_id(card),
        n_clicks=0,
        className="h-100",
    )


def _bar(name: str, fraction: float, state: str, note: str) -> html.Div:
    """One labelled progress bar: a station's position between the two wear anchors."""
    return html.Div(
        [
            html.Div(
                [
                    html.Span(name, className="bar-name"),
                    html.Span(f"{fraction:.0%}", className="bar-value"),
                ],
                className="bar-head",
            ),
            html.Div(
                html.Div(
                    style={
                        "width": f"{np.clip(fraction, 0.01, 1.0):.1%}",
                        "background": COLOR[state],
                    },
                    className="bar-fill",
                ),
                className="bar-track",
            ),
            html.Div(note, className="bar-note"),
        ],
        className="wear-bar",
    )


def machine_card(state: str, signals: list[Signal]) -> html.Div:
    """The verdict, what drove it, and every reading behind it on one line."""
    driving = [signal.name for signal in signals if signal.state == state]
    reason = (
        f"all {len(signals)} signals within limits"
        if state == GOOD
        else f"raised by {', '.join(driving)}"
    )
    return _shell(
        MACHINE,
        state,
        html.Div(
            [
                html.Div(ICON[state], className="machine-icon"),
                html.Div(MACHINE_LABEL[state], className="machine-word"),
            ],
            className="machine-state",
        ),
        html.Div(reason, className="machine-driver"),
        html.Div(
            [html.Span(f"{signal.name} {signal.value}", className="chip") for signal in signals],
            className="machine-chips",
        ),
        solid=True,
    )


def wear_card(signals: list[Signal]) -> html.Div:
    wear_signals = [signal for signal in signals if signal.key in STATIONS]
    return _shell(
        WEAR,
        worst(signal.state for signal in wear_signals),
        html.Div(
            [
                _bar(signal.name, signal.amount, signal.state, signal.detail)
                for signal in wear_signals
            ],
            className="wear-bars",
        ),
        html.Div(
            f"0% is a pristine tool, 100% a worn-out one · mean of {WEAR_WINDOW} strokes",
            className="card-foot",
        ),
    )


def alignment_card(run: Run, stroke: int, signal: Signal, tolerance_mm: float) -> html.Div:
    strokes = run.window(stroke, FACE_STROKES)
    return _shell(
        ALIGNMENT,
        signal.state,
        html.Div(
            [
                html.Span(signal.value, className="stat-value"),
                html.Span("off centre", className="alignment-unit"),
            ],
            className="alignment-head",
        ),
        html.Div(signal.detail, className="alignment-detail"),
        dcc.Graph(
            figure=sparkline(
                run.alignment_mm[strokes],
                COLOR[signal.state],
                ceiling=max(tolerance_mm * 1.2, float(run.alignment_mm[strokes].max()) * 1.1),
                limit=tolerance_mm,
            ),
            config={"displayModeBar": False, "staticPlot": True},
        ),
        html.Div(
            f"Feed direction only · last {len(strokes)} strokes, dotted line is the tolerance",
            className="card-foot",
        ),
    )


def board(run: Run, stroke: int, tolerance_mm: float) -> dbc.Row:
    state, signals = machine_state(run, stroke, tolerance_mm)
    alignment = next(signal for signal in signals if signal.key == ALIGNMENT)
    return dbc.Row(
        [
            dbc.Col(machine_card(state, signals), lg=4, className="mb-3"),
            dbc.Col(wear_card(signals), lg=4, className="mb-3"),
            dbc.Col(alignment_card(run, stroke, alignment, tolerance_mm), lg=4, className="mb-3"),
        ],
        className="g-3",
    )


def _graph(figure) -> dcc.Graph:
    return dcc.Graph(figure=figure, config={"displayModeBar": False})


def detail(card: str, run: Run, stroke: int, tolerance_mm: float) -> tuple[str, html.Div]:
    """Title and body of the window a card opens into."""
    strokes = run.window(stroke, DETAIL_STROKES)
    state, signals = machine_state(run, stroke, tolerance_mm)
    log = html.Div(
        [
            html.Div(f"Last {LOG_ROWS} strokes", className="card-title"),
            stroke_log(run, stroke, tolerance_mm),
        ],
        className="mt-3",
    )

    if card == WEAR:
        return "Tool wear", html.Div(
            [
                _graph(wear_trend_figure(run, strokes)),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(STATIONS[key].name, className="card-title"),
                                confidence_bars(run, key, stroke),
                            ],
                            md=6,
                        )
                        for key in STATIONS
                    ],
                    className="mt-2",
                ),
                html.Div(
                    "The bars are the classifier on this one stroke; the card reads the "
                    f"majority over {WEAR_WINDOW}. The trend is the friction axis, which is "
                    "continuous but far noisier -- the two can disagree, and the classifier "
                    "is the accurate one.",
                    className="section-note mt-2",
                ),
                log,
            ]
        )

    if card == ALIGNMENT:
        return "Strip alignment", html.Div(
            [
                _graph(alignment_trend_figure(run, strokes, tolerance_mm)),
                html.Div(
                    "Only the feed direction is measured -- the campaign varied strip "
                    "overfeed along one axis, so there is no second axis to predict. "
                    f"The alarm reads the mean of {ALIGNMENT_WINDOW} strokes, which a "
                    "single stroke's scatter would otherwise trip on its own.",
                    className="section-note mt-2",
                ),
                log,
            ]
        )

    return MACHINE_HEADLINE[state], html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(ICON[signal.state], className="signal-icon"),
                            html.Span(signal.name, className="signal-name"),
                            html.Span(signal.value, className="signal-value"),
                            html.Div(signal.detail, className="signal-detail"),
                        ],
                        className=f"signal signal-{signal.state}",
                    )
                    for signal in signals
                ],
                className="signal-list",
            ),
            log,
        ]
    )
