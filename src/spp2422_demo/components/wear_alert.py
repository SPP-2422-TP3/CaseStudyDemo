"""The critical-wear alert that interrupts the operator."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from ..data import Station


def alert_id(station_key: str, part: str) -> dict[str, str]:
    return {"type": f"alert-{part}", "station": station_key}


def _fact(label: str, value: str) -> dbc.Col:
    return dbc.Col(
        [html.Div(label, className="alert-fact"), html.Div(value, className="alert-fact-value")]
    )


def wear_alert(station: Station) -> dbc.Modal:
    """The modal itself. Its body is filled in by a callback when it opens."""
    return dbc.Modal(
        [
            html.Div(
                dbc.Row(
                    [
                        dbc.Col(html.Div("⚠", className="alert-icon"), width="auto"),
                        dbc.Col(
                            [
                                html.Div("Critical tool wear detected", className="alert-title"),
                                html.Div(
                                    f"{station.name} station — {station.german}",
                                    className="alert-sub",
                                ),
                            ]
                        ),
                    ],
                    align="center",
                ),
                className="alert-head",
            ),
            dbc.ModalBody(
                [
                    dbc.Row(id=alert_id(station.key, "facts"), className="mb-3"),
                    html.Div(
                        "The force signature of this stroke matches the most worn tool "
                        "condition in the reference data. Inspect the tool before the next "
                        "production run.",
                        className="text-muted",
                    ),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Acknowledge",
                        id=alert_id(station.key, "dismiss"),
                        color="secondary",
                        outline=True,
                    ),
                    dbc.Button(
                        "Explain this prediction",
                        id=alert_id(station.key, "explain"),
                        color="danger",
                    ),
                ]
            ),
        ],
        id=alert_id(station.key, "modal"),
        is_open=False,
        centered=True,
        size="lg",  # so the four facts stay on one line
        className="alert-modal",
    )


def alert_facts(station: Station, level: int, confidence: float, run: str, stroke: int) -> list:
    """The four numbers shown in the alert body."""
    return [
        _fact("Predicted state", f"{station.level_name(level)} · critical"),
        _fact("Confidence", f"{confidence:.0%}"),
        _fact("Production run", run),
        _fact("Stroke", str(stroke)),
    ]
