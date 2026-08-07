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
from spp2422_demo.components.layout import caveat, page_header, panel, stat_card
from spp2422_demo.data import STATIONS

dash.register_page(__name__, path="/", name="Overview", order=0)


MEASURED = "measured"
SIMULATED = "simulated"
TOOL_VIDEO = "tool_cad_animated.mp4"


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


def _placement(calibration) -> str:
    """The withheld state's best placement, or a dash where nothing beat the control."""
    best = calibration.best()
    if best is None:
        return "—"
    window, budget, _ = best
    return f"{calibration.at(window, budget, 'mix').position:.3f}"


def layout(**_kwargs):
    stations = {key: load_artifacts(key) for key in STATIONS}
    deep_drawing, ironing = stations["deep_drawing"], stations["ironing"]
    total_strokes = len(deep_drawing.data.curves)
    # One run per wear-level combination, so name the grid rather than its size.
    runs = deep_drawing.data.runs()
    n_own = len({own for own, _ in runs})
    n_other = len({other for _, other in runs})
    n_simulated = len(deep_drawing.data.sim_curves) + len(ironing.data.sim_curves)

    return html.Div(
        [
            page_header(
                "Tool Wear from Forming Force Signals",
                "SPP 2422 · Teilprojekt 3 — reading the condition of a progressive die "
                "from the forces it produces.",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        stat_card(
                            "Measured strokes",
                            f"{total_strokes:,}".replace(",", " "),
                            f"{n_own} × {n_other} production runs, 500 strokes each",
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
                    *(
                        dbc.Col(
                            stat_card(
                                trained.data.station.name,
                                _placement(trained.calibration),
                                "where the withheld wear state lands between the two "
                                "measured extremes; 0.5 is centred",
                            ),
                            md=3,
                        )
                        for trained in stations.values()
                    ),
                ],
                className="g-4 mb-4",
            ),
            dbc.Row(
                dbc.Col(
                    panel(
                        "The process",
                        # Muted and inline are what let a browser autoplay this at all;
                        # the controls are there for pausing it mid-talk.
                        html.Video(
                            src=dash.get_asset_url(TOOL_VIDEO),
                            autoPlay=True,
                            loop=True,
                            muted=True,
                            controls=True,
                            playsInline=True,
                            className="tool-video",
                        ),
                        note=(
                            "The progressive die in motion. One press stroke drives all three "
                            "stations; each forming station carries its own force sensor, and "
                            "both signals are aligned on the cut."
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
                                figure=accuracy_figure(trained.models, trained.accuracy),
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
                    html.Strong("Read these as monitoring, not as recognition. "),
                    html.Span(
                        "Every wear level appears in training here, and later strokes of the "
                        "same runs are held back to score against — which is what a deployed "
                        "model actually faces once its tool has been characterised. It is not "
                        "evidence that a state the model has never been shown would be placed "
                        "correctly. "
                    ),
                    html.Strong("That question has its own page: "),
                    html.Span(
                        "Wear Threshold withholds the intermediate state altogether and asks "
                        f"whether the {n_simulated} simulated curves can put it back between "
                        "the two measured extremes."
                    ),
                ]
            ),
        ]
    )
