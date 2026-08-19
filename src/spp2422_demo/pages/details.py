"""The one page behind the board: what the data is, how to read the dashboard, and the
published work it all stands on.

The board is the demo; this is everything someone asks about it afterwards. It is one page
rather than several because the questions arrive together -- what am I looking at, should I
believe it, who did the work -- and a visitor who has to find three pages to answer them
usually finds none.
"""

from __future__ import annotations

from typing import NamedTuple

import dash
import dash_bootstrap_components as dbc
from dash import MATCH, Input, Output, callback, ctx, dcc, html

from spp2422_demo.artifacts import load_artifacts
from spp2422_demo.components.curve_figure import level_means_figure, measured_vs_simulated_figure
from spp2422_demo.components.layout import caveat, page_header, panel, stat_card
from spp2422_demo.data import STATIONS

dash.register_page(__name__, path="/details", name="About & Help", order=1, top_level=True)

MEASURED = "measured"
SIMULATED = "simulated"
TOOL_VIDEO = "tool_cad_animated.mp4"

PROJECT_URL = "https://www.ifu.uni-stuttgart.de/spp-2422/teilprojekte/teilprojekt-3/"
PUBLICATIONS_URL = "https://www.ifu.uni-stuttgart.de/spp-2422/publikationen/"
REPO_URL = "https://github.com/SPP-2422-TP3/CaseStudyDemo"
LICENSE_URL = f"{REPO_URL}/blob/main/LICENSE"
AUTHORS = "Felix Divo, Antonia Wüst, Jonas Moske and Markus Schumann"

# The people carrying Teilprojekt 3; their names are highlighted in the author lists below.
TEAM = ["Moske", "Schumann", "Wüst", "Divo", "Kersting", "Groche"]

PEOPLE = [
    (
        "Peter Groche",
        "PtU · Lead",
        "https://www.ptu.tu-darmstadt.de/institut_3/mitarbeiterinnen_3/details_3853.en.jsp",
    ),
    ("Kristian Kersting", "AI/ML Lab · Lead", "https://ml-research.github.io/people/kkersting/"),
    ("Jonas Moske", "PtU", "https://www.researchgate.net/profile/Jonas-Moske"),
    ("Markus Schumann", "PtU", "https://www.researchgate.net/profile/Markus_Schumann"),
    ("Antonia Wüst", "AI/ML Lab", "https://scholar.google.com/citations?user=BltylusAAAAJ"),
    ("Felix Divo", "AI/ML Lab", "https://scholar.google.de/citations?user=TOu-cpQAAAAJ"),
]


class Paper(NamedTuple):
    """One publication. A NamedTuple rather than a dataclass: Dash imports page modules
    from their file path, so they never land in `sys.modules`, and `@dataclass` needs that
    to resolve the postponed annotations this file uses."""

    title: str
    authors: str
    venue: str
    doi: str
    why: str

    @property
    def url(self) -> str:
        return f"https://doi.org/{self.doi}"


GROUPS: list[tuple[str, str, list[Paper]]] = [
    (
        "What This Dashboard Shows",
        "The two studies the pages are built on.",
        [
            Paper(
                "Information Content Analysis of Direct and Indirect Force Measurements for "
                "Machine Learning-Based Process State Classification in Multi-stage Sheet "
                "Metal Forming",
                "M. Schumann, J. Moske, F. Divo, A. Wüst, K. Kersting, P. Groche",
                "Transactions of the Indian Institute of Metals 79(6), 2026",
                "10.1007/s12666-026-03839-4",
                "The measurement campaign behind how deep drawing and ironing wear is read on "
                "the dashboard: which force signals carry enough information to tell the "
                "process state apart, and how much is lost when the sensor sits away from the "
                "forming zone.",
            ),
            Paper(
                "Simulation Driven Modeling of Strip Misalignment: Enhancing Process Insight "
                "and Failure Prediction in Sheet Metal Forming",
                "J. Moske, M. Schumann, A. Wüst, K. Kersting, P. Groche",
                "Journal of Physics: Conference Series 3104, 012058 — NUMISHEET 2025",
                "10.1088/1742-6596/3104/1/012058",
                "The method behind reading strip alignment on the status board: finite "
                "element simulation links how the strip is positioned to the defects that "
                "follow from it, and the plateau slope of the punch force is what carries "
                "that position back out.",
            ),
        ],
    ),
    (
        "Where the Signals Come From",
        "Sensing the press, and turning what it records into data a model can learn from.",
        [
            Paper(
                "Sensorgestützte Kraftüberwachung in der Umformtechnik",
                "J. Moske, M. Schumann, P. Groche",
                "Zeitschrift für wirtschaftlichen Fabrikbetrieb 120(10), 687–694, 2025 — German",
                "10.1515/zwf-2025-1117",
                "Direct against indirect force sensor concepts in a progressive die — the "
                "instrumentation that produces the curves plotted here.",
            ),
            Paper(
                "Camera-based feature extraction and uncertainty analysis in deep drawing in "
                "progressive dies",
                "M. Schumann, J. Moske, A. Wüst, K. Kersting, P. Groche",
                "Production Engineering 20(2), 2026",
                "10.1007/s11740-026-01434-6",
                "A second, optical view of the same press, and a candid look at how noisy the "
                "labels are that any of these models is trained against.",
            ),
            Paper(
                "Requirements for numeric models as sources of synthetic data for predicting "
                "real-world data sets",
                "M. Schumann, J. Moske, A. Wüst, K. Kersting, P. Groche",
                "Preprint, 2025",
                "10.21203/rs.3.rs-8164519/v1",
                "When a simulated curve may stand in for a measured one — the question behind "
                "the “vs. simulation” tab on the overview.",
            ),
            Paper(
                "Structured representation of simulation and annotation data for machine "
                "learning in forming technologies",
                "M. Schumann, J. Moske, A. Wüst, F. Divo, D. Gelbich, P. Niemietz, K. Kersting, "
                "T. Bergs, P. Groche",
                "Production Engineering 20(2), 2026",
                "10.1007/s11740-026-01441-7",
                "One format for force-time series and the expert annotations that go with them, "
                "so data from different presses and simulations can be pooled.",
            ),
        ],
    ),
    (
        "Seeing Wear Directly",
        "The force signal only sees the effect of wear. These look at the tool surface itself.",
        [
            Paper(
                "Photometric stereo for tool wear monitoring: addressing challenges of specular "
                "surfaces in sheet metal forming",
                "J. Moske, H. Kutlu, P. Groenewold, P. Santos, A. Weinmann, A. Kuijper, P. Groche",
                "Manufacturing Review 12, 29, 2025",
                "10.1051/mfreview/2025026",
                "Reconstructing the worn surface optically, despite the mirror finish that makes "
                "a forming tool hard to photograph.",
            ),
            Paper(
                "Inline Wear Detection in High-Speed Progressive Dies Using Photometric Stereo",
                "J. Moske, H. Kutlu, A. Steinmeier, P. Groenewold, P. Santos, A. Kuijper, "
                "A. Weinmann, P. Groche",
                "MATEC Web of Conferences 408, 01031 — IDDRG 2025",
                "10.1051/matecconf/202540801031",
                "The same measurement running inline, between strokes, at production speed.",
            ),
        ],
    ),
    (
        "The Machine Learning Side",
        "Work from the AI/ML Lab: how this demo explains and validates itself, and where the "
        "group's time series modelling goes next.",
        [
            Paper(
                "Right on Time: Revising Time Series Models by Constraining Their Explanations",
                "M. Kraus, D. Steinmann, A. Wüst, A. Kokozinski, K. Kersting",
                "ECML PKDD 2025, LNCS, 490–507",
                "10.1007/978-3-032-06109-6_28",
                "Time-resolved explanations of the kind this demo overlays on a stroke — and what "
                "to do when a model turns out to be reading the wrong part of the curve.",
            ),
            Paper(
                "Navigating Shortcuts, Spurious Correlations, and Confounders: From Origins via "
                "Detection to Mitigation",
                "D. Steinmann, F. Divo, M. Kraus, A. Wüst, L. Struppek, F. Friedrich, K. Kersting",
                "arXiv:2412.05152, 2024",
                "10.48550/arXiv.2412.05152",
                "Why the accuracies on the overview page are reported twice: a model that keys on "
                "which production run a stroke came from, rather than on wear, is exactly the "
                "shortcut catalogued here.",
            ),
            Paper(
                "xLSTM-Mixer: Multivariate Time Series Forecasting by Mixing via Scalar Memories",
                "M. Kraus, F. Divo, D. S. Dhami, K. Kersting",
                "NeurIPS 2025 · arXiv:2410.16928",
                "10.48550/arXiv.2410.16928",
                "Forecasting rather than classification, but the same underlying question: how to "
                "carry a long signal through a model without flattening the time axis. The CNN "
                "behind these pages is a deliberately small answer to it.",
            ),
            Paper(
                "Exploring Neural Granger Causality with xLSTMs: Unveiling Temporal Dependencies "
                "in Complex Data",
                "H. Poonia, F. Divo, K. Kersting, D. S. Dhami",
                "NeurIPS 2025 · arXiv:2502.09981",
                "10.48550/arXiv.2502.09981",
                "Recovering which signal drives which, over time. Deep drawing and ironing happen "
                "in the same stroke on the same strip, so their force traces are anything but "
                "independent.",
            ),
        ],
    ),
]

GLOSSARY = [
    (
        "The status board",
        "The landing page, and an assembled press run rather than a recording. Every stroke on "
        "it is a real measured stroke shown with its own model's prediction; the order they "
        "arrive in is authored, because the data holds fixed wear levels and fixed infeeds but "
        "no transition between them. Wear and misalignment also come from two separate "
        "campaigns on separate tooling, so showing them as one machine is a composition.",
    ),
    (
        "Data Scenario 1 and Data Scenario 2",
        "The two authored runs the machine bar switches between, each carrying one fault. In "
        "Data Scenario 1 the tools wear and the strip stays centred — ironing goes off first "
        "and ends critical, deep drawing follows much later and over twice the span, because "
        "two tools on one press do not go off together and a board worth having has to say "
        "which one went. In Data Scenario 2 both tools stay fresh while the strip walks off "
        "centre through every overfeed level the campaign ran. One fault at a time is the "
        "point: shown together, nothing would demonstrate that the board can tell them apart.",
    ),
    (
        "Good, watch, stop",
        "The state of a signal, and of the machine as its worst signal. Wear states come from "
        "the classifier as the majority call over the last 20 strokes — the accurate "
        "instrument, and one odd stroke should not stop a press. Alignment reads the running "
        "mean of 10 against a tolerance that is set for the demo, not taken from the trials.",
    ),
    (
        "The wear stage track",
        "How far a tool has worn down, on the three stages the trials already name. The badge "
        "is the classifier's majority call over the last 20 strokes; the marker's position is "
        "the mean of that same classifier's probabilities over the window, so a tool drifting "
        "toward the next stage sits between the two rather than jumping when the majority "
        "tips. No percentage: the levels are ordinal classes with no measured roughness "
        "behind them, and nothing here records when a tool was retired, so no fraction of "
        "life consumed exists to report.",
    ),
    (
        "Report bad parts",
        "The operator knows what the parts look like, which the force signals do not. Pressing "
        "it marks the last 60 strokes and records what the monitor was saying over the same "
        "window — the pair a label-collection loop needs. It retrains nothing; the models here "
        "are fixed. It demonstrates the capture step, which is exactly what is missing from "
        "every dataset in this project, and why the intermediate wear state has no labels.",
    ),
    (
        "T1 – T3 and A1 – A3",
        "The wear state of the deep drawing and ironing tools, recorded with the trials as "
        "three roughness classes per station. They are ordered — 1 fresh, 3 critical — but no "
        "roughness value was measured alongside them, so they are classes, not a scale.",
    ),
    (
        "Normalized force, event time",
        "Every curve is scaled to a common amplitude and stretched onto the same 0 → 1 axis, "
        "where 0 is the onset of the forming event and 1 its end. This makes strokes comparable; "
        "it also means the y-axis carries no kN.",
    ),
    (
        "Held-out strokes",
        "The split behind every accuracy on the station pages: train on the first 400 strokes "
        "of each production run, score on the rest. Every wear level is in training, so this "
        "measures monitoring a tool that has already been characterised.",
    ),
    (
        "The withheld centre state",
        "The harder question, on the Wear Threshold page. The intermediate wear level is taken "
        "out of training entirely — as it is in production, where a tool crosses that threshold "
        "uncontrolled — and the simulated friction sweep has to place it between the two "
        "measured extremes. 0.5 would be exactly centred.",
    ),
    (
        "Shuffled sweep",
        "The control that decides whether a placement means anything: the same simulated curves "
        "with their friction ordering scrambled. Beating it is what separates a physically "
        "grounded result from a model that merely interpolates between two anchors.",
    ),
    (
        "The explanation panel",
        "Colour along the curve is how much each moment of the stroke pushed the model toward "
        "the state it predicted. Red supports the prediction, blue argues against it. The "
        "shaded band marks where the evidence concentrates.",
    ),
]


def _authors(text: str) -> html.Div:
    """Author list with the Teilprojekt 3 members in bold."""
    parts: list = []
    for i, name in enumerate(text.split(", ")):
        if i:
            parts.append(", ")
        parts.append(html.Strong(name) if name.split()[-1] in TEAM else html.Span(name))
    return html.Div(parts, className="paper-authors")


def _paper(paper: Paper) -> html.Div:
    return html.Div(
        [
            html.A(paper.title, href=paper.url, target="_blank", className="paper-title"),
            _authors(paper.authors),
            html.Div(paper.venue, className="paper-venue"),
            html.Div(paper.why, className="paper-why"),
        ],
        className="paper",
    )


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


def _person(name: str, institute: str, url: str) -> dbc.Col:
    return dbc.Col(
        html.A(
            [
                html.Div(name, className="person-name"),
                html.Div(institute, className="section-note"),
            ],
            href=url,
            target="_blank",
            className="person",
        ),
        md=12,
    )


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
                            "Measured Strokes",
                            f"{total_strokes:,}".replace(",", " "),
                            f"{n_own} × {n_other} production runs, 500 strokes each",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        stat_card(
                            "Simulated Curves",
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
                        "The Process",
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
                            f"{trained.data.station.name}: Mean Force per Wear Level",
                            dbc.Tabs(
                                [
                                    dbc.Tab(label="Measured", tab_id=MEASURED),
                                    dbc.Tab(label="vs. Simulation", tab_id=SIMULATED),
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
                                "Data Scenario 1: the centred-strip, tool-wear campaign. "
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
            page_header(
                "Help and Background",
                "How to read this dashboard, and the published work it stands on.",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        panel(
                            "Reading the Dashboard",
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(term, className="glossary-term"),
                                            html.Div(text, className="glossary-text"),
                                        ],
                                        className="glossary-item",
                                    )
                                    for term, text in GLOSSARY
                                ]
                            ),
                        ),
                        lg=8,
                    ),
                    dbc.Col(
                        panel(
                            "The Project",
                            html.Div(
                                [
                                    html.P(
                                        [
                                            "Teilprojekt 3 of DFG priority programme ",
                                            html.A(
                                                "SPP 2422",
                                                href=PROJECT_URL,
                                                target="_blank",
                                            ),
                                            " — data-driven process modelling in metal forming. "
                                            "It pairs the Institute for Production Engineering "
                                            "and Forming Machines (PtU) with the Artificial "
                                            "Intelligence and Machine Learning Lab, both at "
                                            "TU Darmstadt.",
                                        ]
                                    ),
                                    dbc.Row(
                                        [_person(*person) for person in PEOPLE],
                                        className="g-2",
                                    ),
                                ]
                            ),
                        ),
                        lg=4,
                    ),
                ],
                className="g-4 mb-4",
            ),
            *[
                html.Div(
                    panel(
                        title,
                        html.Div(subtitle, className="section-note mb-3"),
                        html.Div([_paper(paper) for paper in papers]),
                    ),
                    className="mb-4",
                )
                for title, subtitle, papers in GROUPS
            ],
            caveat(
                [
                    html.Span("A curated selection, not the full output of the project. "),
                    html.A(
                        "The complete SPP 2422 publication list",
                        href=PUBLICATIONS_URL,
                        target="_blank",
                    ),
                    html.Span(" covers every Teilprojekt in the programme."),
                ]
            ),
            html.Div(
                [
                    html.A("Source on GitHub", href=REPO_URL, target="_blank"),
                    " · ",
                    html.A("MIT licence", href=LICENSE_URL, target="_blank"),
                    " · © 2026 " + AUTHORS,
                ],
                className="colophon",
            ),
        ]
    )
