"""Small building blocks shared by every page."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from ..theme import LEVEL_NAMES


def stat_card(title: str, value: str, unit: str = "") -> dbc.Card:
    """A headline number with a label above and a caption below."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="card-title"),
                html.Div(value, className="stat-value"),
                html.Div(unit, className="stat-unit"),
            ]
        )
    )


def panel(title: str, *children, note: str | None = None) -> dbc.Card:
    """A titled card wrapping arbitrary content, with an optional footnote."""
    body: list = [html.Div(title, className="card-title"), *children]
    if note:
        body.append(html.Div(note, className="section-note mt-2"))
    return dbc.Card(dbc.CardBody(body))


def level_badge(level: int, code: str) -> html.Span:
    """The wear level, colour-coded the same way everywhere: green, amber, red."""
    return html.Span(
        [
            html.Span(code, className="level-code"),
            html.Span(LEVEL_NAMES[level]),
        ],
        className=f"level-badge level-{level}",
    )


def page_header(title: str, subtitle: str) -> html.Div:
    return html.Div(
        [html.Div(title, className="page-title"), html.Div(subtitle, className="page-subtitle")],
        className="mb-4",
    )


def caveat(children) -> html.Div:
    """A visually distinct note for things the audience should not over-read.

    Takes children directly rather than as *args: wrapping a list in a tuple nests the
    array one level deeper than dash-renderer follows, and the components inside reach
    React unconverted.
    """
    return html.Div(children, className="caveat")
