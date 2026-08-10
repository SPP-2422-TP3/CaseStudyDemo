"""Help page: how to read the dashboard, and the published work behind it."""

from __future__ import annotations

from typing import NamedTuple

import dash
import dash_bootstrap_components as dbc
from dash import html

from spp2422_demo.components.layout import caveat, page_header, panel

dash.register_page(__name__, path="/help", name="Help", order=6)

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
        "What this dashboard shows",
        "The two studies the pages are built on.",
        [
            Paper(
                "Information Content Analysis of Direct and Indirect Force Measurements for "
                "Machine Learning-Based Process State Classification in Multi-stage Sheet "
                "Metal Forming",
                "M. Schumann, J. Moske, F. Divo, A. Wüst, K. Kersting, P. Groche",
                "Transactions of the Indian Institute of Metals 79(6), 2026",
                "10.1007/s12666-026-03839-4",
                "The measurement campaign behind the deep drawing and ironing pages: which "
                "force signals carry enough information to tell the process state apart, and "
                "how much is lost when the sensor sits away from the forming zone.",
            ),
            Paper(
                "Simulation Driven Modeling of Strip Misalignment: Enhancing Process Insight "
                "and Failure Prediction in Sheet Metal Forming",
                "J. Moske, M. Schumann, A. Wüst, K. Kersting, P. Groche",
                "Journal of Physics: Conference Series 3104, 012058 — NUMISHEET 2025",
                "10.1088/1742-6596/3104/1/012058",
                "The method behind the Excentricity page: finite element simulation links how "
                "the strip is positioned to the defects that follow from it, and the plateau "
                "slope of the punch force is what carries that position back out.",
            ),
        ],
    ),
    (
        "Where the signals come from",
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
        "Seeing wear directly",
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
        "The machine learning side",
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
        "campaigns on separate tooling.",
    ),
    (
        "Good, watch, stop",
        "The state of a signal, and of the machine as its worst signal. Wear states come from "
        "the classifier as the majority call over the last 20 strokes — the accurate "
        "instrument, and one odd stroke should not stop a press. Alignment reads the running "
        "mean of 10 against a tolerance that is set for the demo, not taken from the trials.",
    ),
    (
        "The wear percentage",
        "Where a stroke sits between a pristine tool and a worn-out one on the friction axis: "
        "0 at one anchor, 1 at the other. It is continuous, so it can show a tool approaching "
        "a level rather than only arriving at one — and it is far noisier than the classifier, "
        "so it never raises an alarm on its own. It is not a fraction of life consumed; nothing "
        "in this data records when a tool was retired.",
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
        md=6,
    )


def layout(**_kwargs):
    return html.Div(
        [
            page_header(
                "Help and Background",
                "How to read this dashboard, and the published work it stands on.",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        panel(
                            "Reading the dashboard",
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
                        lg=7,
                    ),
                    dbc.Col(
                        panel(
                            "The project",
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
                        lg=5,
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
