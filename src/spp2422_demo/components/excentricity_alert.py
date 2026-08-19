"""The out-of-tolerance alert that stops the stream.

Deliberately not shared with `wear_alert`: that one reports a categorical tool state and
offers to explain it, this one reports a measured distance against a limit the operator
set, and its job is to say why the line stopped.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def _fact(label: str, value: str) -> dbc.Col:
    return dbc.Col(
        [html.Div(label, className="alert-fact"), html.Div(value, className="alert-fact-value")]
    )


def excentricity_alert() -> dbc.Modal:
    """The modal itself. Its body is filled in by a callback when it opens."""
    return dbc.Modal(
        [
            html.Div(
                dbc.Row(
                    [
                        dbc.Col(html.Div("⚠", className="alert-icon"), width="auto"),
                        dbc.Col(
                            [
                                html.Div(
                                    "Strip Misalignment Out of Tolerance",
                                    className="alert-title",
                                ),
                                html.Div("Deep drawing station", className="alert-sub"),
                            ]
                        ),
                    ],
                    align="center",
                ),
                className="alert-head",
            ),
            dbc.ModalBody(
                [
                    dbc.Row(id="exc-alert-facts", className="mb-3"),
                    html.Div(
                        "The plateau of this stroke tilts far enough that the model places the "
                        "blank beyond the limit set for this run. The stream has stopped here. "
                        "Check the feed before continuing -- an off-centre blank draws one "
                        "flange wide and thins the opposite wall.",
                        className="text-muted",
                    ),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Acknowledge and stay here",
                        id="exc-alert-dismiss",
                        color="secondary",
                        outline=True,
                    ),
                    dbc.Button("Resume the stream", id="exc-alert-resume", color="danger"),
                ]
            ),
        ],
        id="exc-alert-modal",
        is_open=False,
        centered=True,
        size="lg",
        className="alert-modal",
    )


def alert_facts(predicted_mm: float, true_mm: float, threshold_mm: float, series: str, stroke: int):
    """The four numbers shown in the alert body."""
    return [
        _fact("Predicted eccentricity", f"{predicted_mm:.2f} mm"),
        _fact("Limit", f"{threshold_mm:.2f} mm"),
        _fact("Measured infeed", f"{series} · {true_mm:.2f} mm at the cup"),
        _fact("Stroke", str(stroke)),
    ]
