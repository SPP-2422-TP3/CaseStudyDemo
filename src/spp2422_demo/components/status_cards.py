"""The three cards the status board is, and the detail windows behind them.

Each card answers one question at a glance and opens into the evidence for its answer.
The cards are the whole interface; everything else on the page is the press controls that
drive them.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import numpy as np
from dash import dcc, html

from ..artifacts import TrainedStation, load_artifacts
from ..data import LEVELS, STATIONS
from ..excentricity import INFEED_LEVELS, load_excentricity
from ..explain import explain
from ..features import curve_features
from ..feedback import FEEDBACK_STROKES, Report
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
from ..models import CnnModel, FeatureModel, HybridModel
from ..scenario import ALIGNMENT_WINDOW, WEAR_WINDOW, Run
from ..theme import LEVEL_NAMES
from .curve_figure import attribution_figure, stroke_figure
from .excentricity_figure import feature_space_figure, indicator_figure, plateau_figure, status_of
from .excentricity_figure import stroke_figure as excentricity_stroke_figure
from .layout import caveat, panel, percent, stat_card
from .status_figures import (
    LOG_ROWS,
    alignment_dots,
    alignment_trend_figure,
    confidence_bars,
    stroke_log,
    trailing,
    wear_trend_figure,
)

MACHINE, WEAR, ALIGNMENT = "machine", "wear", "alignment"
CARDS = (MACHINE, ALIGNMENT, WEAR)

# Strokes drawn on a card face and in its detail window.
FACE_STROKES = 60
DETAIL_STROKES = 100

STAGE_NAMES = [LEVEL_NAMES[level] for level in LEVELS]

CARD_TITLE = {
    MACHINE: "Machine Status",
    WEAR: "Tool Wear",
    ALIGNMENT: "Strip Alignment",
}


def card_id(card: str) -> dict[str, str]:
    return {"type": "status-card", "card": card}


def _shell(card: str, state: str, *children, solid: bool = False) -> dbc.Card:
    """A card's face, coloured by its state and named by its title.

    `solid` floods the whole card with the state colour instead of accenting its edge.
    The machine verdict uses it: a card that is green or red across its full area is
    legible from the other side of a press hall, where a coloured edge is not.

    This is only the face. What carries the click is the slot it is placed into, which
    outlives it -- see `card_slots`.
    """
    fill = " status-solid" if solid else ""
    return dbc.Card(
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
    )


def card_slots() -> dbc.Row:
    """The three clickable frames the cards are drawn into, mounted once.

    They are permanent on purpose, and the faces inside them are what the stroke
    callback replaces. Rebuilding a card wholesale takes its `n_clicks` back to zero and
    destroys the component that raised any click still in flight -- so a click landing
    while the wear card's attribution was still being computed used to be discarded along
    with it, and the detail window never opened. That is invisible on a laptop, where the
    attribution beats the next stroke, and reliable on a small host, where it does not.

    `dbc.Card` has no `n_clicks`, which is why the click lives on a wrapping div.
    """
    return dbc.Row(
        [
            dbc.Col(
                html.Div(id=card_id(card), n_clicks=0, className="h-100"),
                lg=4,
                className="mb-3",
            )
            for card in CARDS
        ],
        className="g-3",
    )


def _stage_bar(name: str, stage: float, level: str, state: str, note: str) -> html.Div:
    """How far a tool has worn down, on the three stages the shop floor already names.

    The track is the whole journey, fresh to critical, divided where the stages divide.
    The fill says how far down it the tool has come; the stage badge says which stage that
    lands in. No percentage: a tool is in stage 2, not at 47%, and the stages are ordinal
    classes with no measured roughness behind them -- a number would imply a precision
    the labels never had.
    """
    fraction = float(np.clip((stage - LEVELS[0]) / (LEVELS[-1] - LEVELS[0]), 0.0, 1.0))
    return html.Div(
        [
            html.Div(
                [
                    html.Span(name, className="bar-name"),
                    html.Span(level, className="stage-badge", style={"background": COLOR[state]}),
                ],
                className="bar-head",
            ),
            html.Div(
                [
                    html.Div(
                        style={"width": f"{fraction:.1%}", "background": COLOR[state]},
                        className="bar-fill",
                    ),
                    *(
                        html.Div(
                            className="stage-divider",
                            style={"left": f"{index / (len(LEVELS) - 1):.2%}"},
                        )
                        # Only the boundaries between stages; the two ends are the track.
                        for index in range(1, len(LEVELS) - 1)
                    ),
                    html.Div(className="stage-marker", style={"left": f"{fraction:.1%}"}),
                ],
                className="bar-track stage-track",
            ),
            html.Div(
                [html.Span(name, className="stage-tick") for name in STAGE_NAMES],
                className="stage-scale",
            ),
            html.Div(note, className="bar-note"),
        ],
        className="wear-bar",
    )


def machine_card(state: str, signals: list[Signal]) -> dbc.Card:
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


def wear_card(signals: list[Signal]) -> dbc.Card:
    wear_signals = [signal for signal in signals if signal.key in STATIONS]
    return _shell(
        WEAR,
        worst(signal.state for signal in wear_signals),
        html.Div(
            [
                _stage_bar(signal.name, signal.amount, signal.value, signal.state, signal.detail)
                for signal in wear_signals
            ],
            className="wear-bars",
        ),
        html.Div(
            f"How far the tools have worn down, over the last {WEAR_WINDOW} strokes",
            className="card-foot",
        ),
    )


def alignment_card(run: Run, stroke: int, signal: Signal, tolerance_mm: float) -> dbc.Card:
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
            figure=alignment_dots(
                run.alignment_mm[strokes],
                ceiling=max(tolerance_mm * 1.2, float(run.alignment_mm[strokes].max()) * 1.1),
                limit=tolerance_mm,
                smoothed=trailing(run.alignment_mm, ALIGNMENT_WINDOW)[strokes],
            ),
            config={"displayModeBar": False, "staticPlot": True},
        ),
        html.Div(
            f"One dot per stroke, feed direction only · last {len(strokes)} strokes, "
            "dashed line is the tolerance",
            className="card-foot",
        ),
    )


def board(run: Run, stroke: int, tolerance_mm: float) -> tuple[dbc.Card, dbc.Card, dbc.Card]:
    """The three card faces for one stroke, in the order `card_slots` mounts them."""
    state, signals = machine_state(run, stroke, tolerance_mm)
    alignment = next(signal for signal in signals if signal.key == ALIGNMENT)
    return (
        machine_card(state, signals),
        alignment_card(run, stroke, alignment, tolerance_mm),
        wear_card(signals),
    )


def _graph(figure) -> dcc.Graph:
    return dcc.Graph(figure=figure, config={"displayModeBar": False})


def _about_item(label: str, *children) -> dbc.Col:
    return dbc.Col(
        [html.Div(label, className="about-label"), html.Div(children, className="about-text")],
        md=4,
    )


def _about_model(trained: TrainedStation, model_key: str) -> html.Div:
    """Plain-language card on the model currently selected: what it reads, what it was
    trained on, how well it does, and how it is explained."""
    data = trained.data
    model = trained.models[model_key]
    burst = data.peak_ref is not None

    n_features = len(
        curve_features(data.curves[0], burst=burst, peak_ref=data.peak_ref[0] if burst else None)
    )
    descriptors = f"{n_features} shape descriptors measured off the curve"
    in_full = (
        f"{descriptors} — how high the peak is and when it falls, how long the rise and the fall "
        "take, how straight each segment runs and how steady the force is within each tenth of "
        "the stroke"
        + (", plus the burst as the tool takes contact and the dip that follows." if burst else ".")
    )
    raw = (
        "The raw 500-sample curve, which the network reads for itself rather than being told what "
        "to measure."
    )
    reads = {
        FeatureModel: in_full,
        CnnModel: raw,
        HybridModel: f"{raw[:-1]}, and the {descriptors} alongside it — the two are joined "
        "below a single head, so each can cover what the other misses.",
    }[type(model)]

    method = (
        "Integrated gradients: the sensitivity of the prediction accumulated along a path from "
        "the average stroke to this one."
        if isinstance(model, CnnModel)
        else "Occlusion sensitivity: every stretch of the stroke is flattened in turn and the "
        "confidence it was carrying is recorded."
    )

    # What this number is not: every wear level appears in training here, so it measures
    # monitoring an already characterised tool. The Wear Threshold page asks the harder
    # question, about a state the model was never shown.
    scope = (
        "Every wear level appears in training, so this measures watching a tool that has "
        "already been characterised — not recognising a state it has never been shown."
    )

    n_train = int(data.train_mask.sum())
    return html.Div(
        [
            html.Div(model.name, style={"fontWeight": 600}),
            html.Div(model.description.replace("--", "—"), className="section-note mb-3"),
            dbc.Row(
                [
                    _about_item("Reads", reads),
                    _about_item(
                        "Trained on",
                        f"{n_train:,}".replace(",", " ") + " strokes: the first 400 of each of "
                        f"the {len(data.runs())} production runs, every wear combination "
                        "included.",
                    ),
                    _about_item(
                        "Gets right",
                        html.Div(
                            f"{percent(trained.accuracy[model_key])} of later strokes from "
                            "those same runs."
                        ),
                        html.Div(scope, className="mt-1"),
                    ),
                ],
                className="g-4",
            ),
            html.Div(method, className="section-note mt-3"),
        ]
    )


def _snapped(data) -> np.ndarray:
    """Each alignment prediction rounded to the nearest infeed level actually run on the press."""
    levels = np.array(INFEED_LEVELS)
    return levels[np.abs(data.predicted[:, None] - levels[None, :]).argmin(axis=1)]


def _exact_level_rate(data) -> float:
    return float((_snapped(data) == data.labels).mean())


def _near_level_rate(data) -> float:
    return float((np.abs(_snapped(data) - data.labels) <= 5).mean())


def _about_alignment(data) -> html.Div:
    """The misalignment forest's own reads/trained-on/gets-right panel -- the same shape
    `_about_model` gives the wear stations' models."""
    return html.Div(
        [
            html.Div("Random forest on two plateau features", style={"fontWeight": 600}),
            html.Div(
                "20 trees, depth 4, following Moske et al. (NUMISHEET 2025).",
                className="section-note mb-3",
            ),
            dbc.Row(
                [
                    _about_item(
                        "Reads",
                        "Two numbers, and nothing else: the slope and the height of a straight "
                        "line fitted across the force plateau. As the blank goes off-centre the "
                        "plateau tilts downward and peak force falls, and that is the whole "
                        "signal — no spectrum, no shape descriptors, no second sensor.",
                    ),
                    _about_item(
                        "Trained on",
                        f"{len(data.curves)} strokes: {len(INFEED_LEVELS)} measured series of "
                        "49, one per infeed level. The first stroke of each series is dropped "
                        "as a warm-up transient.",
                    ),
                    _about_item(
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
                            "Predictions here are out-of-fold: every stroke shown is predicted "
                            "by a forest that did not train on it.",
                            className="mt-1",
                        ),
                    ),
                ],
                className="g-4",
            ),
        ]
    )


def _station_stroke(run: Run, station_key: str, stroke: int) -> html.Div:
    """One stroke at one station, read the way the deep drawing and ironing explorers used
    to read it: the same force curve, the same time-resolved attribution, the same
    confidence split, plus the model's own technical panel below.
    """
    trained = load_artifacts(station_key)
    data = trained.data
    row = run.rows[stroke]
    level = run.level(station_key, stroke)
    model_key = run.model_key
    peak_ref = data.peak_ref[row] if data.peak_ref is not None else None
    attribution = explain(trained.models[model_key], data.curves[row], peak_ref, level)

    return html.Div(
        [
            html.Div(
                [
                    html.Span(f"Stroke {stroke + 1}", className="card-title"),
                    html.Span(run.model_name, className="card-model"),
                ],
                className="card-head mt-3",
            ),
            _graph(stroke_figure(data, row, label=f"Stroke {stroke + 1}")),
            html.Div(
                "The measured force curve for this stroke, against the mean curve of each "
                "wear level. Below: the same stroke coloured by where the model found its "
                "evidence.",
                className="section-note",
            ),
            _graph(attribution_figure(data, row, attribution)),
            html.Div(attribution.summary(STATIONS[station_key].name), className="section-note"),
            confidence_bars(run, station_key, stroke),
            html.Div(panel("About This Model", _about_model(trained, model_key)), className="mt-4"),
        ]
    )


def _alignment_stroke(run: Run, stroke: int, tolerance_mm: float) -> html.Div:
    """One stroke's punch force and the two plateau features the forest reads off it, at
    the three zoom levels and with the prediction readout the Excentricity page gave a
    whole control panel to.
    """
    data = load_excentricity()
    index = run.alignment_rows[stroke]
    predicted = float(run.alignment_mm[stroke])
    running = run.smoothed_alignment(stroke)
    true = float(run.alignment_true_mm[stroke])
    color, state, icon = status_of(predicted, tolerance_mm)
    window_n = min(ALIGNMENT_WINDOW, stroke + 1)

    prediction = html.Div(
        [
            html.Div(
                f"{predicted:.2f} mm",
                style={"fontSize": "2.1rem", "fontWeight": 600, "color": color, "lineHeight": 1.1},
            ),
            html.Div("off-centre at the cup", className="section-note"),
            html.Div(
                f"{icon}  {state}", style={"color": color, "fontWeight": 600}, className="mt-2"
            ),
            html.Div(
                f"Last {window_n} strokes averaged: {running:.2f} mm · measured {true:.2f} mm",
                className="section-note mt-1",
            ),
            _graph(indicator_figure(predicted, running, true, tolerance_mm)),
            dbc.Row(
                [
                    dbc.Col(
                        stat_card(
                            "Plateau Slope",
                            f"{data.slope_kn_per_s(index):+.2f}",
                            "kN/s — tilts down as the blank goes off-centre",
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        stat_card(
                            "Plateau Force",
                            f"{data.intercept_kn(index):.2f}",
                            "kN at the start of the window — falls with misalignment",
                        ),
                        width=6,
                    ),
                ],
                className="g-2 mt-2 alignment-stats",
            ),
        ]
    )

    return html.Div(
        [
            html.Div(f"Stroke {stroke + 1}", className="card-title mt-3"),
            dbc.Row(
                [
                    dbc.Col(_graph(excentricity_stroke_figure(data, index)), lg=8),
                    dbc.Col(prediction, lg=4),
                ],
                className="g-3",
            ),
            dbc.Row(
                [
                    dbc.Col(_graph(plateau_figure(data, index)), lg=6),
                    dbc.Col(_graph(feature_space_figure(data, index)), lg=6),
                ],
                className="g-3 mt-0",
            ),
            html.Div(panel("About This Model", _about_alignment(data)), className="mt-4"),
            html.Div(
                caveat(
                    [
                        html.Strong("One production run per infeed level. "),
                        html.Span(
                            "Each of the seven series is a single uninterrupted run of 49 "
                            "strokes, so strokes that share a tool temperature, a lubrication "
                            "state and one setup end up on both sides of the train/test split. "
                            "The error above therefore measures recognising a stroke from a "
                            "run the model has already seen — holding a whole run out is "
                            "impossible here, because that would remove its infeed level from "
                            "training entirely. Repeated runs per level are what an honest "
                            "generalisation estimate would need."
                        ),
                    ]
                ),
                className="mt-3",
            ),
        ]
    )


def _reports_panel(reports: list[Report]) -> html.Div:
    """What the operator has reported, against what the monitor was saying at the time."""
    if not reports:
        return html.Div(
            f"No operator reports yet. *Report Bad Parts* asks how far back the parts were "
            f"bad -- {FEEDBACK_STROKES} strokes by default -- and what was wrong with them, "
            "then records what the monitor said over the same window. That pair is what a "
            "label-collection loop needs. It does not retrain anything; the models are fixed.",
            className="section-note mt-3",
        )
    return html.Div(
        [
            html.Div("Operator Reports", className="card-title mt-3"),
            *(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(item.issue_label, className="report-tag"),
                                html.Span(item.label, className="report-range"),
                            ],
                            className="report-head",
                        ),
                        *(
                            [html.Div(f"“{item.note}”", className="report-note")]
                            if item.note
                            else []
                        ),
                        html.Div(item.readings(), className="report-readings"),
                        html.Div(item.disagreement(), className="report-verdict"),
                    ],
                    className="report",
                )
                for item in reversed(reports)
            ),
            html.Div(
                "Captured, not learned from: one operator's verdict on one window is not a "
                "training set, and the models on this board are fixed. The step being shown "
                "is the one missing from every dataset in this project -- which is why the "
                "intermediate wear state has no labels to train on in the first place.",
                className="section-note mt-2",
            ),
        ]
    )


def detail(
    card: str, run: Run, stroke: int, tolerance_mm: float, reports: list[Report]
) -> tuple[str, html.Div]:
    """Title and body of the window a card opens into."""
    strokes = run.window(stroke, DETAIL_STROKES)
    flags = [(item.start, item.end) for item in reports]
    state, signals = machine_state(run, stroke, tolerance_mm)
    log = html.Div(
        [
            html.Div(f"Last {LOG_ROWS} Strokes", className="card-title"),
            stroke_log(run, stroke, tolerance_mm),
        ],
        className="mt-3",
    )

    if card == WEAR:
        return "Tool Wear", html.Div(
            [
                _graph(wear_trend_figure(run, strokes, flags)),
                html.Div(
                    "Where the last strokes place each tool between the two anchors. "
                    "Continuous, and far noisier than the stage the card reports -- the "
                    "two can disagree, and the classifier is the accurate one.",
                    className="section-note mt-1 mb-3",
                ),
                dbc.Tabs(
                    [
                        dbc.Tab(
                            _station_stroke(run, key, stroke),
                            label=STATIONS[key].name,
                            tab_id=key,
                        )
                        for key in STATIONS
                    ],
                    active_tab=next(iter(STATIONS)),
                ),
                log,
            ]
        )

    if card == ALIGNMENT:
        return "Strip Alignment", html.Div(
            [
                _graph(alignment_trend_figure(run, strokes, tolerance_mm, flags)),
                html.Div(
                    "Only the feed direction is measured -- the campaign varied strip "
                    "overfeed along one axis, so there is no second axis to predict. "
                    f"The alarm reads the mean of {ALIGNMENT_WINDOW} strokes, which a "
                    "single stroke's scatter would otherwise trip on its own.",
                    className="section-note mt-1 mb-3",
                ),
                _alignment_stroke(run, stroke, tolerance_mm),
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
            _reports_panel(reports),
            log,
        ]
    )
