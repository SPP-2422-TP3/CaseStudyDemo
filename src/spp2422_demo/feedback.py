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
from .health import GOOD, alignment_state, majority_level
from .scenario import Run

# How far back a report can reach, in strokes. A progressive die runs on the order of a
# stroke a second, so these are roughly the last half minute, minute and two minutes -- but
# the choices are stated in strokes, because press rate was never recorded with these
# trials and a minute figure would be a number this demo invented.
WINDOW_CHOICES = (30, 60, 120)
FEEDBACK_STROKES = WINDOW_CHOICES[1]

# What the operator saw on the parts. These are the defects this process actually produces,
# and they are asked for because *which* defect is what decides whether a force signal
# could ever have carried it: a burr belongs to the cutting station, a thin wall to
# ironing, an off-centre cup to the strip feed.
ISSUES = (
    ("thin_wall", "Wall too thin or torn"),
    ("off_centre", "Cup off centre"),
    ("burr", "Burr or poor cut edge"),
    ("surface", "Surface marks or scoring"),
    ("dimension", "Out of dimensional tolerance"),
    ("other", "Something else"),
)
ISSUE_LABEL = dict(ISSUES)

FRESH = LEVELS[0]  # a station reading this is not one the monitor had raised


@dataclass(frozen=True)
class Report:
    """One operator report: which strokes, and what the monitor said about them."""

    start: int  # first flagged stroke, 0-based
    end: int  # last flagged stroke, inclusive
    issue: str  # which defect the operator saw, from ISSUES
    note: str  # whatever else they wanted to say, in their own words
    called: dict[str, int]  # station key -> the wear level the monitor held over the window
    alignment_mm: float  # the monitor's mean misalignment over the window
    tolerance_mm: float  # the tolerance in force at the time, which decides what that meant

    @property
    def label(self) -> str:
        return f"strokes {self.start + 1}–{self.end + 1}"

    @property
    def issue_label(self) -> str:
        return ISSUE_LABEL.get(self.issue, self.issue)

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
        if alignment_state(self.alignment_mm, self.tolerance_mm) != GOOD:
            flagged.append("strip alignment")
        if flagged:
            return "the monitor had already raised " + ", ".join(sorted(flagged))
        return "the monitor read every signal as normal over this window"


def report(
    run: Run, stroke: int, window: int, issue: str, note: str, tolerance_mm: float
) -> Report:
    """Capture what the monitor was saying across the window the operator just flagged.

    `stroke` is where the press was when the operator opened the form, not when they
    submitted it: they are reporting on what they had just seen, and a form left open
    while the press runs on would otherwise slide the window off the parts they meant.
    """
    return Report(
        start=max(0, stroke - window + 1),
        end=stroke,
        issue=issue,
        note=note.strip(),
        called={key: majority_level(run, key, stroke) for key in STATIONS},
        alignment_mm=run.smoothed_alignment(stroke),
        tolerance_mm=tolerance_mm,
    )


def to_store(reports: list[Report]) -> list[dict]:
    """Reports as plain JSON, which is all a `dcc.Store` can hold."""
    return [asdict(item) for item in reports]


def from_store(payload: list[dict] | None) -> list[Report]:
    return [Report(**item) for item in payload or []]
