"""Turning four predictions into the one sentence a status board exists to say.

The board answers "can this press keep running", and everything else on it is the
justification. Three signals feed that answer -- deep-drawing wear, ironing wear and strip
alignment -- and the machine is only as good as its worst one.

Which instrument decides is deliberate, and the two are not interchangeable:

- The **state** comes from the classifier, taken as the majority call over the last
  `WEAR_WINDOW` strokes. That is the accurate instrument (100% and 99% on held-out
  strokes), and the intermediate level is itself the state worth stopping for.
- The **placement inside the three stages** comes from the same classifier, read as the
  mean of its probabilities over that window rather than as a count of its calls. That
  says how firmly it holds its answer, so a tool drifting toward the next stage sits
  between the two instead of jumping when the majority tips.

Both therefore come from one instrument, on one scale the shop floor already uses. The
continuous friction axis is a different question and stays in the detail window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import LEVELS, STATIONS
from .scenario import ALIGNMENT_WINDOW, WEAR_WINDOW, Run
from .theme import LEVEL_COLORS

GOOD, WATCH, CRITICAL = "good", "watch", "critical"

RANK = {GOOD: 0, WATCH: 1, CRITICAL: 2}
COLOR = {GOOD: LEVEL_COLORS[1], WATCH: LEVEL_COLORS[2], CRITICAL: LEVEL_COLORS[3]}
ICON = {GOOD: "✓", WATCH: "▲", CRITICAL: "⚠"}

# What the whole machine is doing, as opposed to what one signal says. The board shows the
# short word, sized to be read across a press hall; the detail window shows the sentence.
MACHINE_LABEL = {GOOD: "OK", WATCH: "WATCH", CRITICAL: "STOP"}
MACHINE_HEADLINE = {
    GOOD: "Running Normally",
    WATCH: "Watch This Press",
    CRITICAL: "Stop the Press",
}

# A wear level maps straight onto a state: the intermediate one is the threshold this
# whole project is about, so it warns rather than passing.
WEAR_STATE = {1: GOOD, 2: WATCH, 3: CRITICAL}

# Default tolerance on strip misalignment at the cup, in mm. Chosen for the demo -- no
# scrap tolerance was published with the trials -- and adjustable on the page for that
# reason. The watch band opens at three quarters of it, as on the Excentricity page.
DEFAULT_TOLERANCE_MM = 0.60
WATCH_FRACTION = 0.75


@dataclass(frozen=True)
class Signal:
    """One thing the board watches, reduced to a state and two lines of text."""

    key: str
    name: str
    state: str
    amount: float  # the reading itself, for a bar or an axis
    value: str  # the same reading formatted for display
    detail: str  # what it means, in a phrase


def worst(states) -> str:
    return max(states, key=lambda state: RANK[state])


def wear_signal(run: Run, station_key: str, stroke: int) -> Signal:
    """Wear at one station: which of the three stages, and where inside it."""
    level = majority_level(run, station_key, stroke)
    station = STATIONS[station_key]
    return Signal(
        key=station_key,
        name=station.name,
        state=WEAR_STATE[level],
        amount=wear_stage(run, station_key, stroke),
        value=station.level_name(level),
        detail=f"over the last {WEAR_WINDOW} strokes",
    )


def majority_level(run: Run, station_key: str, stroke: int) -> int:
    """The classifier's call over the trailing window, so one odd stroke cannot stop a press."""
    window = run.window(stroke, WEAR_WINDOW)
    calls = run.proba[station_key][window].argmax(axis=1) + 1
    return int(np.bincount(calls, minlength=4)[1:].argmax()) + 1


def wear_stage(run: Run, station_key: str, stroke: int) -> float:
    """Where the trailing window places the tool on the 1..3 stage scale, between stages.

    The classifier already says which stage a stroke is in. Averaging its probabilities
    over the window -- rather than counting its calls -- says how firmly, so a tool whose
    strokes are starting to read as the next stage sits visibly between the two rather
    than jumping when the majority tips. It is the same instrument as the badge, read
    more finely; nothing new is claimed by it.
    """
    proba = run.proba[station_key][run.window(stroke, WEAR_WINDOW)]
    return float((proba * np.array(LEVELS)).sum(axis=1).mean())


def alignment_state(value_mm: float, tolerance_mm: float) -> str:
    """State of one misalignment reading, whether it is a single stroke or a mean."""
    if value_mm >= tolerance_mm:
        return CRITICAL
    return WATCH if value_mm >= WATCH_FRACTION * tolerance_mm else GOOD


def alignment_signal(run: Run, stroke: int, tolerance_mm: float) -> Signal:
    """Strip alignment, read off the running mean rather than a single stroke.

    A single stroke carries several hundredths of a millimetre of scatter -- enough to
    cross a tolerance on its own -- so the alarm watches the trailing mean, which is the
    same choice the Excentricity page defends at length.
    """
    value = run.smoothed_alignment(stroke)
    state = alignment_state(value, tolerance_mm)
    detail = {
        CRITICAL: f"past the {tolerance_mm:.2f} mm tolerance at the cup",
        WATCH: f"approaching the {tolerance_mm:.2f} mm tolerance",
        GOOD: f"inside the {tolerance_mm:.2f} mm tolerance",
    }[state]
    return Signal(
        key="alignment",
        name="Strip alignment",
        state=state,
        amount=value,
        value=f"{value:.2f} mm",
        detail=f"{detail}, averaged over {ALIGNMENT_WINDOW} strokes",
    )


def signals(run: Run, stroke: int, tolerance_mm: float) -> list[Signal]:
    """Every signal the board watches, in the order the cards read them."""
    return [
        *(wear_signal(run, key, stroke) for key in STATIONS),
        alignment_signal(run, stroke, tolerance_mm),
    ]


def machine_state(run: Run, stroke: int, tolerance_mm: float) -> tuple[str, list[Signal]]:
    """The machine's state and the signals behind it -- the machine is its worst signal."""
    current = signals(run, stroke, tolerance_mm)
    return worst(signal.state for signal in current), current
