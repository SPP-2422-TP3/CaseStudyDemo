"""Product quality prediction — placeholder for the strip-misalignment work."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import html

from spp2422_demo.components.layout import page_header, panel

dash.register_page(__name__, path="/quality", name="Product quality", order=4)


def layout(**_kwargs):
    return html.Div(
        [
            page_header(
                "Product quality prediction",
                "Predicting part defects from the position of the strip in the progressive die.",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        panel(
                            "Planned",
                            html.Div(
                                [
                                    html.P(
                                        "Deviations in how the sheet strip is positioned between "
                                        "stations propagate into the drawn part: an off-centre "
                                        "blank draws unevenly, thins on one side and can tear.",
                                    ),
                                    html.P(
                                        "This page will show the same loop as the tool wear "
                                        "pages — a signal comes in, a model predicts whether the "
                                        "resulting part is within tolerance, and the prediction "
                                        "can be interrogated — but for misalignment rather than "
                                        "wear.",
                                        className="mb-0",
                                    ),
                                ]
                            ),
                        ),
                        lg=7,
                    ),
                    dbc.Col(
                        panel(
                            "Status",
                            html.Div(
                                [
                                    html.Div("🔧", style={"fontSize": "2.2rem"}),
                                    html.Div(
                                        "Under construction",
                                        className="mt-2",
                                        style={"fontWeight": 600},
                                    ),
                                    html.Div(
                                        "Contributed separately.",
                                        className="section-note",
                                    ),
                                ],
                                className="placeholder",
                            ),
                        ),
                        lg=5,
                    ),
                ],
                className="g-4",
            ),
            html.Div(
                panel(
                    "Background",
                    html.Div(
                        [
                            "The approach builds on ",
                            html.A(
                                "Simulation Driven Modeling of Strip Misalignment: Enhancing "
                                "Process Insight and Failure Prediction in Sheet Metal Forming",
                                href="https://doi.org/10.1088/1742-6596/3104/1/012058",
                                target="_blank",
                            ),
                            " (Moske, Schumann, Wüst, Kersting and Groche, 2025), which uses "
                            "finite element simulation to link strip position to part failure "
                            "and detects the resulting anomalies in the force signals.",
                        ]
                    ),
                ),
                className="mt-4",
            ),
        ]
    )
