"""Operator feedback: the parts were bad, and here is when.

A press operator knows something the force signals do not: what the parts coming off the
die actually look like. That knowledge arrives late, in a lump and in the operator's own
terms -- *the last minute or so was no good* -- never as a label on a single stroke.

The board takes it in exactly that shape. Pressing the button marks the trailing
`FEEDBACK_STROKES` strokes as bad parts and records what the monitor was saying at the
time, which is the pair a label-collection loop actually needs: the operator's verdict on
a window, against the models' reading of the same window.

**Nothing here retrains anything.** The models on this board are fixed, and one flag from
one operator is not a training set. What it demonstrates is the capture step -- the part
that has to exist before any of the retraining questions can even be asked, and the part
that is missing from every dataset in this project, which is why the intermediate wear
state has no labels in the first place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .data import LEVELS, STATIONS
from .health import majority_level
from .scenario import Run

# Strokes one report covers. A progressive die runs on the order of a stroke a second, so
# this is roughly the "last minute" an operator would mean, and the window is stated on
# the button rather than inferred -- press rate was never recorded with these trials.
FEEDBACK_STROKES = 60

FRESH = LEVELS[0]  # a station reading this is not one the monitor had raised


@dataclass(frozen=True)
class Report:
    """One operator report: which strokes, and what the monitor said about them."""

    start: int  # first flagged stroke, 0-based
    end: int  # last flagged stroke, inclusive
    called: dict[str, int]  # station key -> the wear level the monitor held over the window
    alignment_mm: float  # the monitor's mean misalignment over the window

    @property
    def label(self) -> str:
        return f"strokes {self.start + 1}–{self.end + 1}"

    def readings(self) -> str:
        """What the monitor held over the window, in the names the stations use."""
        stages = ", ".join(
            f"{STATIONS[key].name} {STATIONS[key].level_name(level)}"
            for key, level in sorted(self.called.items())
        )
        return f"{stages}, strip {self.alignment_mm:.2f} mm"

    def disagreement(self) -> str:
        """Whether the monitor had already flagged what the operator is reporting.

        The interesting report is the one where every signal read fine and the parts did
        not: that is a gap in what the force signals carry, and it is the case worth
        collecting. Agreement is a confirmation, which is worth much less.
        """
        flagged = [STATIONS[key].name for key, level in self.called.items() if level > FRESH]
        if flagged:
            return "the monitor had already raised " + ", ".join(sorted(flagged))
        return "the monitor read every signal as normal over this window"


def report(run: Run, stroke: int, tolerance_mm: float) -> Report:
    """Capture what the monitor was saying across the window the operator just flagged."""
    del tolerance_mm  # the report records readings, not whether they crossed a limit
    start = max(0, stroke - FEEDBACK_STROKES + 1)
    return Report(
        start=start,
        end=stroke,
        called={key: majority_level(run, key, stroke) for key in STATIONS},
        alignment_mm=run.smoothed_alignment(stroke),
    )


def to_store(reports: list[Report]) -> list[dict]:
    """Reports as plain JSON, which is all a `dcc.Store` can hold."""
    return [asdict(item) for item in reports]


def from_store(payload: list[dict] | None) -> list[Report]:
    return [Report(**item) for item in payload or []]
