"""The press runs the status board watches, assembled from measured strokes.

The board wants something the underlying data does not contain: a press that runs for
hundreds of strokes while something gradually goes off. What was actually recorded is nine
production runs at *fixed* wear levels and seven misalignment series at *fixed* infeed --
snapshots of states, never a transition between them.

So a run is a **schedule**, and only the schedule is authored. Two are offered, because a
board that showed both faults at once could never demonstrate telling them apart:

    Scenario 1 -- tool wear, and the two tools do not go together
      ironing        A1 ==\\__ A2 ______________________/== A3
      deep drawing   T1 ==============\\____ T2 (slowly) ____
      strip          60.00 mm, centred throughout

    Scenario 2 -- the strip walks off centre while the tools stay fresh
      ironing        A1 ---------------------------------------
      deep drawing   T1 ---------------------------------------
      strip          60.00 mm ---- 0.05 mm steps ---- 60.30 mm

Every stroke on screen is a real measured stroke, shown with the prediction its own model
made for it -- out of fold for misalignment, from strokes the wear axis never trained on.
What is invented is the *order*: which recorded state each stroke is drawn from, and a
crossfade across the wear transitions so the rolling means drift rather than step. Nothing
interpolates between two strokes, and no curve is synthesised.

The stations wear independently because the recorded runs allow it: all nine T x A
combinations were measured, so a stroke at T1/A2 is as real as one at T1/A1.

Two further things the board says out loud rather than hiding:

- Wear and misalignment come from **separate measurement campaigns** on separate tooling.
  Presenting them as one machine is a composition, not a recording.
- Deep drawing and ironing strokes at the same index *are* the same physical stroke, since
  each production run is one T x A combination measured at both stations at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from .artifacts import load_artifacts
from .data import STATIONS
from .excentricity import INFEED_LEVELS, excentricity_mm, load_excentricity
from .features import feature_matrix

N_STROKES = 300
SEED = 7
# Strokes skipped at the head of every production run. The first strokes of a block are a
# real press transient -- cold die, lubricant not yet distributed -- and they read as a
# briefly worn tool. Genuine, but not what a status board should open on.
SETTLE = 5

# Strokes averaged before anything is shown. A single stroke carries a large share of the
# span between the wear anchors, and several hundredths of a millimetre of misalignment
# scatter; neither underlying quantity moves that fast.
WEAR_WINDOW = 20
ALIGNMENT_WINDOW = 10

# The strip drift of scenario 2: a stretch of good alignment, then a walk up through the
# recorded infeed series in the order they were run.
INFEED_START = 60  # strokes before the drift sets in
INFEED_HOLD = 40  # strokes spent at each recorded level on the way up

CENTRED = ((INFEED_LEVELS[0], 0),)
DRIFTING = tuple(
    (level, 0 if index == 0 else INFEED_START + INFEED_HOLD * (index - 1))
    for index, level in enumerate(INFEED_LEVELS)
)


@dataclass(frozen=True)
class Scenario:
    """One authored press run: how each tool wears, and where the strip sits.

    `wear` gives each station `(level, stroke it takes over, strokes to cross)`, and the
    levels crossfade over that span -- strokes are drawn from both recorded runs with a
    ramping probability, so a rolling mean drifts across a transition instead of stepping.
    A long span is a tool going off slowly; a short one is a tool going off quickly.

    `infeed` steps rather than crossfades: the strip is fed where it is fed, and the
    recorded series are 0.05 mm apart, which is finer than the model can resolve anyway.
    """

    key: str
    name: str
    headline: str
    summary: str
    wear: dict[str, tuple[tuple[int, int, int], ...]]
    infeed: tuple[tuple[int, int], ...]


WEAR_SCENARIO = Scenario(
    key="wear",
    name="Scenario 1",
    headline="Tool wear",
    summary="Tools wear at their own pace · strip stays centred",
    wear={
        # Ironing goes first -- friction acts directly on the forming force there -- and
        # it is the tool that reaches the critical stage at the end of the run.
        "ironing": ((1, 0, 0), (2, 70, 50), (3, 245, 45)),
        # Deep drawing follows much later and over twice the span, so the board shows two
        # tools that are plainly not on the same clock.
        "deep_drawing": ((1, 0, 0), (2, 170, 110)),
    },
    infeed=CENTRED,
)

ALIGNMENT_SCENARIO = Scenario(
    key="alignment",
    name="Scenario 2",
    headline="Strip misalignment",
    summary="Strip walks off centre · both tools stay fresh",
    wear={key: ((1, 0, 0),) for key in STATIONS},
    infeed=DRIFTING,
)

SCENARIOS = {scenario.key: scenario for scenario in (WEAR_SCENARIO, ALIGNMENT_SCENARIO)}
DEFAULT_SCENARIO = WEAR_SCENARIO.key


@dataclass(frozen=True)
class Run:
    """Everything the board reads, one entry per stroke of the scenario."""

    scenario: Scenario
    wear_level: dict[str, np.ndarray]  # station -> (N,) the recorded run it was drawn from
    position: dict[str, np.ndarray]  # station -> (N,) 0 = pristine anchor, 1 = worn
    proba: dict[str, np.ndarray]  # station -> (N, 3) classifier probabilities
    model_name: dict[str, str]  # station -> the classifier reading it
    rows: np.ndarray  # (N,) row into the measured extract; the same stroke at both stations
    alignment_mm: np.ndarray  # (N,) predicted offset at the cup, out of fold
    alignment_true_mm: np.ndarray  # (N,) the offset its series was actually run at
    alignment_rows: np.ndarray  # (N,) row into the misalignment dataset, for the stroke view

    def level(self, station_key: str, stroke: int) -> int:
        """The classifier's call for one stroke: wear level 1..3."""
        return int(self.proba[station_key][stroke].argmax()) + 1

    def confidence(self, station_key: str, stroke: int) -> float:
        return float(self.proba[station_key][stroke].max())

    def smoothed_position(self, station_key: str, stroke: int) -> float:
        return _trailing_mean(self.position[station_key], stroke, WEAR_WINDOW)

    def smoothed_alignment(self, stroke: int) -> float:
        return _trailing_mean(self.alignment_mm, stroke, ALIGNMENT_WINDOW)

    def window(self, stroke: int, length: int) -> np.ndarray:
        """Indices of the last `length` strokes up to and including `stroke`."""
        return np.arange(max(0, stroke - length + 1), stroke + 1)


def _trailing_mean(values: np.ndarray, stroke: int, window: int) -> float:
    return float(np.mean(values[max(0, stroke - window + 1) : stroke + 1]))


def _draw_order(rows: np.ndarray, rng: np.random.Generator, count: int) -> np.ndarray:
    """`count` rows drawn from `rows`, reshuffled each time the pool is exhausted.

    Used for the misalignment series, where 49 recorded strokes have to cover a longer
    stretch of the board. Reshuffling avoids a visible 49-stroke period; the scatter it
    draws from is the real scatter of that series.
    """
    passes = [rng.permutation(rows) for _ in range(count // len(rows) + 1)]
    return np.concatenate(passes)[:count]


def _wear_schedule(stages: tuple[tuple[int, int, int], ...], rng: np.random.Generator):
    """Which recorded wear level each stroke is drawn from, with crossfaded transitions."""
    strokes = np.arange(N_STROKES)
    level = np.full(N_STROKES, stages[0][0], dtype=int)
    for new_level, takeover, span in stages[1:]:
        ramp = np.clip((strokes - (takeover - span / 2)) / span, 0.0, 1.0)
        level = np.where(rng.random(N_STROKES) < ramp, new_level, level)
    return level


def _infeed_schedule(stages: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Which recorded infeed series each stroke is drawn from."""
    level = np.full(N_STROKES, stages[0][0], dtype=int)
    for value, takeover in stages[1:]:
        level[takeover:] = value
    return level


def _stroke_rows(wear_level: dict[str, np.ndarray]) -> np.ndarray:
    """A row of the right production run for every stroke, in the order the press made them.

    One row serves both stations. A production run is one T x A combination measured at
    both at once, so `run_strokes(t, a)` returns the same physical strokes whether it is
    asked of deep drawing or of ironing -- which is what lets the board show two stations
    of the same press rather than two unrelated presses, and what lets the two stations
    sit at different wear levels at the same time.

    Each combination keeps its own place in its run, so a crossfade interleaves two runs
    while both continue to advance through their own real strokes. Strokes either
    station's wear axis was fitted on are dropped: their position is an in-sample fit that
    would flatter the method.
    """
    fitted = np.concatenate([load_artifacts(key).wear.fitted_rows for key in STATIONS])
    reference_key, other_key = STATIONS
    reference = load_artifacts(reference_key).data

    combinations = [
        (int(own), int(other))
        for own, other in zip(wear_level[reference_key], wear_level[other_key], strict=True)
    ]
    pools = {}
    for combination in sorted(set(combinations)):
        run = reference.run_strokes(*combination)[SETTLE:]
        pools[combination] = run[~np.isin(run, fitted)]

    taken = dict.fromkeys(pools, 0)
    rows = np.empty(N_STROKES, dtype=int)
    for stroke, combination in enumerate(combinations):
        rows[stroke] = pools[combination][taken[combination]]
        taken[combination] += 1
    return rows


def _classify(station_key: str, rows: np.ndarray) -> tuple[np.ndarray, str]:
    """The station's best classifier applied to the scenario's strokes."""
    trained = load_artifacts(station_key)
    data = trained.data
    model = trained.models[trained.default_model]
    burst = data.peak_ref is not None
    peak_ref = data.peak_ref[rows] if burst else None
    features, _ = feature_matrix(data.curves[rows], burst=burst, peak_ref=peak_ref)
    return model.predict_proba_matrix(data.curves[rows], features), model.name


@cache
def load_run(scenario_key: str = DEFAULT_SCENARIO) -> Run:
    """Assemble one scenario's run. Cached -- the schedule is deterministic."""
    scenario = SCENARIOS[scenario_key]
    rng = np.random.default_rng(SEED)

    wear_level = {key: _wear_schedule(scenario.wear[key], rng) for key in STATIONS}
    rows = _stroke_rows(wear_level)
    classified = {key: _classify(key, rows) for key in STATIONS}

    alignment = load_excentricity()
    infeed_level = _infeed_schedule(scenario.infeed)
    alignment_rows = np.empty(N_STROKES, dtype=int)
    for level in np.unique(infeed_level):
        strokes = np.flatnonzero(infeed_level == level)
        alignment_rows[strokes] = _draw_order(alignment.series_rows(level), rng, len(strokes))

    return Run(
        scenario=scenario,
        wear_level=wear_level,
        position={key: load_artifacts(key).wear.display(rows) for key in STATIONS},
        proba={key: classified[key][0] for key in STATIONS},
        model_name={key: classified[key][1] for key in STATIONS},
        rows=rows,
        alignment_mm=np.array([excentricity_mm(v) for v in alignment.predicted[alignment_rows]]),
        alignment_true_mm=np.array([excentricity_mm(v) for v in infeed_level]),
        alignment_rows=alignment_rows,
    )
