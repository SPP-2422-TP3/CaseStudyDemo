"""Shared visual language: palette and the Plotly template every figure uses."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# Okabe-Ito, colourblind-safe. Same palette the research plots use.
OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # purple
    "#56B4E9",  # sky
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
]

INK = "#1B2430"
MUTED = "#6B7785"
GRID = "#E3E8EE"
SURFACE = "#FFFFFF"
CANVAS = "#F5F7FA"

# Wear severity always reads green -> amber -> red, on every page and every figure.
LEVEL_COLORS = {1: "#1B9E77", 2: "#E6A400", 3: "#D6453D"}
LEVEL_NAMES = {1: "Fresh", 2: "Worn", 3: "Critical"}

SIMULATED = "#8C6BB1"  # FE-simulated curves, kept visually distinct from measured ones

TEMPLATE = "spp2422"


def register_template() -> None:
    """Install the shared template and make it the default for every figure."""
    pio.templates[TEMPLATE] = go.layout.Template(
        layout=go.Layout(
            colorway=OKABE_ITO,
            font={"family": "Inter, Segoe UI, system-ui, sans-serif", "size": 13, "color": INK},
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            margin={"l": 60, "r": 20, "t": 40, "b": 45},
            xaxis={
                "gridcolor": GRID,
                "zerolinecolor": GRID,
                "linecolor": GRID,
                "ticks": "outside",
                "tickcolor": GRID,
                "title": {"font": {"size": 12, "color": MUTED}},
            },
            yaxis={
                "gridcolor": GRID,
                "zerolinecolor": GRID,
                "linecolor": GRID,
                "ticks": "outside",
                "tickcolor": GRID,
                "title": {"font": {"size": 12, "color": MUTED}},
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "x": 0,
                "bgcolor": "rgba(0,0,0,0)",
            },
            title={"font": {"size": 15, "color": INK}, "x": 0, "xanchor": "left"},
            hoverlabel={"font": {"family": "Inter, Segoe UI, system-ui, sans-serif", "size": 12}},
        )
    )
    pio.templates.default = TEMPLATE
