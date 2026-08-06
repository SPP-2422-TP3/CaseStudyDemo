"""Landing dashboard: what the data is, how well the models read it, where to go next."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import MATCH, Input, Output, callback, ctx, dcc, html

from spp2422_demo.artifacts import load_artifacts
from spp2422_demo.components.curve_figure import (
    accuracy_figure,
    level_means_figure,
    measured_vs_simulated_figure,
)
from spp2422_demo.components.layout import caveat, page_header, panel, percent, stat_card
from spp2422_demo.components.process import process_diagram
from spp2422_demo.data import STATIONS

dash.register_page(__name__, path="/", name="Overview", order=0)


MEASURED = "measured"
SIMULATED = "simulated"


def _id(station_key: str, part: str) -> dict[str, str]:
    return {"type": part, "station": station_key}


@callback(
    Output(_id(MATCH, "curves-graph"), "figure"),
    Input(_id(MATCH, "curves-tabs"), "active_tab"),
)
def _curves(active_tab):
    data = load_artifacts(ctx.outputs_list["id"]["station"]).data
    return (
        level_means_figure(data) if active_tab == MEASURED else measured_vs_simulated_figure(data)
    )


def layout(**_kwargs):
    stations = {key: load_artifacts(key) for key in STATIONS}
    deep_drawing, ironing = stations["deep_drawing"], stations["ironing"]
    total_strokes = len(deep_drawing.data.curves)
    n_runs = len(deep_drawing.data.runs())
    n_simulated = len(deep_drawing.data.sim_curves) + len(ironing.data.sim_curves)

    best = {key: max(trained.run_accuracy.values()) for key, trained in stations.items()}

    return html.Div(
        [
            page_header(
                "Tool wear from forming force signals",
                "SPP 2422 · Teilprojekt 3 — reading the condition of a progressive die "
                "from the forces it produces.",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        stat_card(
                            "Measured strokes",
                            f"{total_strokes:,}".replace(",", " "),
                            f"{n_runs} production runs, 500 strokes each",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        stat_card(
                            "Simulated curves",
                            str(n_simulated),
                            "finite element exports across friction coefficients",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        stat_card(
                            "Deep drawing",
                            percent(best["deep_drawing"]),
                            "accuracy on a production run never seen in training",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        stat_card(
                            "Ironing",
                            percent(best["ironing"]),
                            "accuracy on a production run never seen in training",
                        ),
                        md=3,
                    ),
                ],
                className="g-4 mb-4",
            ),
            dbc.Row(
                dbc.Col(
                    panel(
                        "The process",
                        process_diagram(),
                        note=(
                            "One press stroke drives all three stations. Each forming station "
                            "carries its own force sensor; both signals are aligned on the cut."
                        ),
                    )
                ),
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        panel(
                            f"{trained.data.station.name}: mean force per wear level",
                            dbc.Tabs(
                                [
                                    dbc.Tab(label="Measured", tab_id=MEASURED),
                                    dbc.Tab(label="vs. simulation", tab_id=SIMULATED),
                                ],
                                id=_id(trained.data.station.key, "curves-tabs"),
                                active_tab=MEASURED,
                                className="mb-2",
                            ),
                            # Filled by callback rather than inline: a Plotly figure built
                            # inside a tab panel is laid out while the panel still has no
                            # size, and renders blank.
                            dcc.Graph(
                                id=_id(trained.data.station.key, "curves-graph"),
                                config={"displayModeBar": False},
                            ),
                            note=(
                                "Normalized force on a shared event-time axis: 0 is the onset of "
                                "the forming event, 1 its end."
                            ),
                        ),
                        lg=6,
                    )
                    for trained in stations.values()
                ],
                className="g-4 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        panel(
                            f"{trained.data.station.name}: how well the models read it",
                            dcc.Graph(
                                figure=accuracy_figure(
                                    trained.models, trained.accuracy, trained.run_accuracy
                                ),
                                config={"displayModeBar": False},
                            ),
                        ),
                        lg=6,
                    )
                    for trained in stations.values()
                ],
                className="g-4 mb-4",
            ),
            caveat(
                [
                    html.Strong("The two bars answer different questions. "),
                    html.Span(
                        "“Held-out strokes” keeps later strokes of runs the model already "
                        "trained on — it measures monitoring a tool that has been characterised. "
                        "“Unseen run” withholds a whole production run, which is the honest test "
                        "of whether the wear state itself is being recognised. Deep drawing "
                        "passes both. "
                    ),
                    html.Strong("Ironing passes only the first: "),
                    html.Span(
                        f"on an unseen run it drops to {percent(best['ironing'])}, at or below the "
                        "33% chance level, so its high held-out score reflects run identity "
                        "rather than wear. That matches the project's own finding that the "
                        "ironing signal is substantially harder than the deep drawing one."
                    ),
                ]
            ),
        ]
    )
