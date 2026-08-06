"""Ironing (Abstreckgleitziehen) wear state, A1 to A3."""

from __future__ import annotations

import dash

from spp2422_demo.station_view import layout as station_layout

dash.register_page(__name__, path="/ironing", name="Ironing", order=2)


def layout(**_kwargs):
    return station_layout("ironing")
