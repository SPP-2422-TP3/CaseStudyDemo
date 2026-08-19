"""The Dash application shell: top bar, navigation, page container."""

from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html

from .theme import register_template

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
# Dash discovers and imports these itself, loading them from their file path rather than as
# package members, which is why they import the rest of the package absolutely.
PAGES = HERE / "pages"

TITLE = "SPP 2422 · TP3 — Tool Wear from Forming Forces"
# Font Awesome 4: font is OFL-1.1, CSS is MIT -- both permissive, neither requires
# attribution, so pulling in the one icon the status board needs (see `_machine_bar` in
# `pages/status.py`) costs nothing to license-track.
FONT_AWESOME = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"

# The one page that renders without any site chrome around it.
STATUS_PATH = "/"


def _nav() -> list:
    """One nav link per registered page, in their declared order."""
    pages = sorted(dash.page_registry.values(), key=lambda page: page.get("order", 99))
    return [dbc.NavLink(page["name"], href=page["relative_path"], active="exact") for page in pages]


def _topbar() -> html.Div:
    links = _nav()
    return html.Div(
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(
                        html.A(
                            [
                                html.Div("SPP 2422 · Teilprojekt 3", className="brand"),
                                html.Div(
                                    "Tool Wear from Forming Force Signals", className="brand-sub"
                                ),
                            ],
                            href="/",
                            style={"textDecoration": "none", "color": "inherit"},
                        ),
                        width="auto",
                    ),
                    dbc.Col(dbc.Nav(links, className="justify-content-end"), className="ms-auto"),
                ],
                align="center",
                className="g-2",
            ),
            fluid="xl",
        ),
        id="topbar",
        className="topbar",
    )


def _build() -> dash.Dash:
    """Construct the single Dash instance.

    Called exactly once, at import. Dash hands the callbacks registered by `@callback`
    to the first app that is constructed, so a second instance would come up with an
    empty page and no working controls.
    """
    register_template()
    app = dash.Dash(
        __name__,
        use_pages=True,
        pages_folder=str(PAGES),
        assets_folder=str(ASSETS),
        external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME],
        title=TITLE,
        update_title=None,
        suppress_callback_exceptions=True,
    )
    app.layout = html.Div(
        [
            dcc.Location(id="url"),
            _topbar(),
            dbc.Container(dash.page_container, id="page", fluid="xl", className="py-4"),
        ]
    )
    return app


app = _build()


@callback(
    Output("topbar", "className"),
    Output("page", "className"),
    Input("url", "pathname"),
)
def _chrome(pathname: str | None):
    """Strip the site chrome off the status board.

    The board is meant to sit on a press-side screen showing one thing, so it gets the
    whole viewport and no navigation. Every other page keeps the top bar, and the board's
    own control panel carries the way back to them.
    """
    if pathname == STATUS_PATH:
        return "topbar topbar-hidden", "py-3"
    return "topbar", "py-4"


server = app.server  # for a WSGI host: `gunicorn spp2422_demo.app:server`
