"""Inline SVG schematic of the progressive die.

Three stations act within one press stroke: the strip is cut free, drawn into a cup, and
its wall is thinned. The demo covers the two forming stations; the cut is shown for
context because it is what the force signals are aligned on.
"""

from __future__ import annotations

from dash import html

from ..theme import GRID, INK, MUTED, OKABE_ITO

_STAGES = [
    ("Shear cutting", "Scherschneiden", "#9AA7B5", False),
    ("Deep drawing", "Tiefziehen", OKABE_ITO[0], True),
    ("Ironing", "Abstreckgleitziehen", OKABE_ITO[1], True),
]


def _svg(tag: str, **attrs) -> str:
    body = attrs.pop("_body", "")
    rendered = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f"<{tag} {rendered}>{body}</{tag}>"


def process_diagram() -> html.Div:
    """A left-to-right strip through three stations, the two modelled ones highlighted."""
    width, height = 720, 172
    parts = [
        # The sheet strip running through every station.
        _svg("rect", x=0, y=64, width=width, height=22, fill="#EDF1F6", rx=3),
        # Below the stations, so it never collides with a station box.
        _svg(
            "text",
            x=width / 2,
            y=166,
            fill=MUTED,
            font_size=11,
            text_anchor="middle",
            _body="Strip feed direction →",
        ),
    ]

    slot = width / len(_STAGES)
    for i, (name, german, color, modelled) in enumerate(_STAGES):
        cx = slot * i + slot / 2
        parts += [
            _svg(
                "rect",
                x=cx - 52,
                y=22,
                width=104,
                height=106,
                rx=8,
                fill="#fff",
                stroke=color if modelled else GRID,
                stroke_width=2 if modelled else 1,
            ),
            _svg("rect", x=cx - 26, y=38, width=52, height=16, rx=2, fill=color),
            _svg("rect", x=cx - 3, y=54, width=6, height=14, fill=color),
            _svg("rect", x=cx - 26, y=88, width=52, height=16, rx=2, fill=color, opacity=0.55),
            _svg(
                "text",
                x=cx,
                y=118,
                fill=INK,
                font_size=11.5,
                font_weight=600,
                text_anchor="middle",
                _body=name,
            ),
            _svg(
                "text",
                x=cx,
                y=17,
                fill=MUTED,
                font_size=10,
                text_anchor="middle",
                _body=german,
            ),
        ]
        if i < len(_STAGES) - 1:
            arrow_x = slot * (i + 1)
            parts.append(
                _svg(
                    "path",
                    d=f"M {arrow_x - 6} 75 l 12 0 m -5 -4 l 5 4 l -5 4",
                    stroke=MUTED,
                    stroke_width=1.4,
                    fill="none",
                )
            )

    svg = (
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'style="max-width:{width}px" xmlns="http://www.w3.org/2000/svg">'
        + "".join(parts)
        + "</svg>"
    )
    return html.Div(
        html.Iframe(
            srcDoc=f'<body style="margin:0;font-family:Inter,system-ui,sans-serif">{svg}</body>',
            style={"width": "100%", "height": f"{height + 8}px", "border": "none"},
        )
    )
