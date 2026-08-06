"""Deep drawing (Tiefziehen) wear state, T1 to T3."""

from __future__ import annotations

import dash

from spp2422_demo.station_view import layout as station_layout

dash.register_page(__name__, path="/deep-drawing", name="Deep drawing", order=1)


def layout(**_kwargs):
    return station_layout("deep_drawing")
