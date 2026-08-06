"""Locating the wear state that cannot be labelled, by anchoring on the simulated sweep."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, callback, dcc, html

from spp2422_demo.artifacts import load_artifacts
from spp2422_demo.calibration import BUDGETS, CENTRE
from spp2422_demo.components.calibration_figure import SERIES, placement_figure
from spp2422_demo.components.layout import caveat, page_header, panel, stat_card
from spp2422_demo.data import STATIONS

dash.register_page(__name__, path="/wear-threshold", name="Wear threshold", order=3)

STATION_SELECT = "threshold-station"
BODY = "threshold-body"


def layout(**_kwargs):
    return html.Div(
        [
            page_header(
                "Locating the wear threshold",
                "The state that matters is the one nobody can label. So withhold it, and ask "
                "the simulated friction sweep to put it back.",
            ),
            panel(
                "The problem with the middle state",
                html.P(
                    [
                        "A tool does not step from good to scrap. It passes through a threshold "
                        "where the parts are still within tolerance but the surface is going, and "
                        "that is the state worth catching. It is also the one state nobody can "
                        "hand a model: wear crosses it uncontrolled, and the press cannot be held "
                        "there long enough to collect labelled strokes.",
                    ],
                    className="mb-2",
                ),
                html.P(
                    [
                        "So it is taken out of training entirely. The model sees a pristine tool, "
                        "a heavily worn one, and a finite-element friction sweep that spans the "
                        "middle continuously. The sweep supplies the ordering; a handful of real "
                        "strokes at each end tie it to the actual press. If that works, the "
                        "withheld state should land midway between the two anchors.",
                    ],
                    className="mb-0",
                ),
            ),
            html.Div(
                [
                    html.Div("Forming stage", className="form-label mt-4"),
                    dcc.RadioItems(
                        id=STATION_SELECT,
                        options=[
                            {"label": station.name, "value": key}
                            for key, station in STATIONS.items()
                        ],
                        value="ironing",
                        inline=True,
                        className="radio-inline",
                    ),
                ]
            ),
            html.Div(id=BODY, className="mt-3"),
        ]
    )


def _headline(station_key: str, calibration) -> list:
    station = STATIONS[station_key]
    withheld = station.level_name(CENTRE)
    best = calibration.best()
    if best is None:
        return [
            caveat(
                f"For {station.name.lower()}, no window and budget placed the withheld "
                f"{withheld} state closer to the centre than the shuffled-sweep control. "
                "Nothing here is evidence of recovery."
            )
        ]

    window, budget, p = best
    mix = calibration.at(window, budget, "mix")
    control = calibration.at(window, budget, "shuffled-sim")
    real_only = calibration.at(window, budget, "real-only")
    window_text = f"first {window} strokes" if window < 500 else "the whole run"

    cards = dbc.Row(
        [
            dbc.Col(
                stat_card(
                    f"{withheld} placed at",
                    f"{mix.position:.3f}",
                    f"0.5 is exactly between the two anchors ({window_text}, "
                    f"{budget} real strokes per endpoint)",
                ),
                md=4,
            ),
            dbc.Col(
                stat_card(
                    "Shuffled-sweep control",
                    f"{control.position:.3f}",
                    "the same curves with their friction ordering destroyed",
                ),
                md=4,
            ),
            dbc.Col(
                stat_card(
                    "Against that control",
                    f"p = {p:.4f}" if p >= 0.0001 else "p < 0.0001",
                    "paired over eight seeds",
                ),
                md=4,
            ),
        ],
        className="g-3",
    )

    reading = (
        f"The sweep is doing the work: with the same real strokes but a scrambled friction "
        f"ordering the state lands at {control.position:.3f}, and with no sweep at all at "
        f"{real_only.position:.3f}."
        if real_only is not None
        else ""
    )
    return [cards, html.P(reading, className="section-note mt-3 mb-0")]


def _table(calibration, window: int) -> dbc.Table:
    """The numbers behind the figure -- the orange control line is deliberately low
    contrast, so the values it encodes are also available as text."""
    header = html.Thead(
        html.Tr(
            [html.Th("Real strokes per endpoint")]
            + [html.Th(label) for label, _ in SERIES.values()]
            + [html.Th("p vs. control")]
        )
    )
    rows = []
    for budget in BUDGETS[window]:
        cells = [html.Td(str(budget))]
        for variant in SERIES:
            placement = calibration.at(window, budget, variant)
            cells.append(html.Td(f"{placement.position:.3f}" if placement else "—"))
        p = calibration.p_values.get((window, budget), np.nan)
        cells.append(html.Td(f"{p:.4f}" if np.isfinite(p) else "—"))
        rows.append(html.Tr(cells))
    return dbc.Table([header, html.Tbody(rows)], bordered=False, hover=True, size="sm")


@callback(Output(BODY, "children"), Input(STATION_SELECT, "value"))
def _body(station_key: str):
    calibration = load_artifacts(station_key).calibration
    best = calibration.best()
    window = best[0] if best else min(BUDGETS)
    scope = f"first {window} strokes" if window < 500 else "whole run"

    return html.Div(
        [
            *_headline(station_key, calibration),
            html.Div(
                panel(
                    "Placement against the real-data budget",
                    dcc.Graph(
                        figure=placement_figure(calibration), config={"displayModeBar": False}
                    ),
                    note=(
                        "Bars are the spread over eight seeds. A control that leaves the axis "
                        "has no stable placement at all, which is itself the point."
                    ),
                ),
                className="mt-4",
            ),
            html.Div(
                panel(
                    f"The same numbers, {scope}",
                    _table(calibration, window),
                    note=f"Calibrated on {', '.join(calibration.features)}.",
                ),
                className="mt-4",
            ),
            caveat(
                "Read this narrowly. Most window and budget combinations do not separate from "
                "the control, and this is a placement on a friction axis rather than a wear "
                "label — it says the withheld state sits between the two anchors, not that any "
                "individual stroke can be classified."
            ),
        ]
    )
