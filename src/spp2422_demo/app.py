"""The Dash application shell: top bar, navigation, page container."""

from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import html

from .theme import register_template

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
# Dash discovers and imports these itself. It loads them from their file path rather than
# as package members, which is why they import the rest of the package absolutely and why
# the shared station view lives outside this folder -- a module in here would be imported
# twice and register its callbacks twice.
PAGES = HERE / "pages"

TITLE = "SPP 2422 · TP3 — Tool Wear from Forming Forces"

# The status board is the page the dashboard opens on; everything that argues about how
# the models work sits one menu behind it. A page opts in by setting `top_level=True`.
DETAILS = "Details"


def _nav() -> list:
    """Top-level links, then everything else folded into one menu."""
    pages = sorted(dash.page_registry.values(), key=lambda page: page.get("order", 99))
    top = [page for page in pages if page.get("top_level")]
    rest = [page for page in pages if not page.get("top_level")]
    return [
        *(dbc.NavLink(page["name"], href=page["relative_path"], active="exact") for page in top),
        dbc.DropdownMenu(
            label=DETAILS,
            nav=True,
            in_navbar=True,
            align_end=True,
            children=[
                dbc.DropdownMenuItem(page["name"], href=page["relative_path"]) for page in rest
            ],
        ),
    ]


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
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title=TITLE,
        update_title=None,
        suppress_callback_exceptions=True,
    )
    app.layout = html.Div(
        [
            _topbar(),
            dbc.Container(dash.page_container, fluid="xl", className="py-4"),
        ]
    )
    return app


app = _build()
server = app.server  # for a WSGI host: `gunicorn spp2422_demo.app:server`
