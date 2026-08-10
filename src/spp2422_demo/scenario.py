"""The press run the status board watches, assembled from measured strokes.

The board wants something the underlying data does not contain: a press that runs for
hundreds of strokes while its tools gradually go off. What was actually recorded is nine
production runs at *fixed* wear levels and seven misalignment series at *fixed* infeed --
snapshots of states, never a transition between them.

So the run is a **schedule**, and only the schedule is authored:

    stroke 0 ............................................. stroke 300
    wear      T1/A1 ====\\____ T2/A2 ____/==== T3/A3
    infeed    60.00 mm ------- ramping in 0.05 mm steps ------- 60.30 mm

Every stroke on screen is a real measured stroke, shown with the prediction its own model
made for it -- out of fold for misalignment, from strokes the wear axis never trained on.
What is invented is the *order*: which recorded state each stroke is drawn from, and a
crossfade across the transitions so the rolling means drift rather than step. Nothing
interpolates between two strokes, and no curve is synthesised.

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

# How the recorded wear levels are laid out along the run: (level, stroke it takes over).
# Levels crossfade over TRANSITION strokes, drawing from both runs with a ramping
# probability, so the rolling mean drifts across a transition instead of stepping.
WEAR_STAGES = ((1, 0), (2, 110), (3, 215))
TRANSITION = 60

# The strip drifts off centre independently of the tools: a stretch of good alignment,
# then a walk up through the recorded infeed series in the order they were run.
INFEED_START = 60  # strokes before the drift sets in
INFEED_HOLD = 40  # strokes spent at each recorded level on the way up


@dataclass(frozen=True)
class Run:
    """Everything the board reads, one entry per stroke of the scenario."""

    wear_level: np.ndarray  # (N,) the recorded run each stroke was drawn from, 1..3
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


def _wear_schedule(rng: np.random.Generator) -> np.ndarray:
    """Which recorded wear level each stroke is drawn from, with crossfaded transitions."""
    level = np.full(N_STROKES, WEAR_STAGES[0][0], dtype=int)
    for (_, _), (new_level, start) in zip(WEAR_STAGES, WEAR_STAGES[1:], strict=False):
        half = TRANSITION // 2
        ramp = np.clip((np.arange(N_STROKES) - (start - half)) / TRANSITION, 0.0, 1.0)
        level = np.where(rng.random(N_STROKES) < ramp, new_level, level)
    return level


def _infeed_schedule() -> np.ndarray:
    """Which recorded infeed series each stroke is drawn from."""
    step = (np.arange(N_STROKES) - INFEED_START) // INFEED_HOLD + 1
    return np.array(INFEED_LEVELS)[np.clip(step, 0, len(INFEED_LEVELS) - 1)]


def _stroke_rows(wear_level: np.ndarray) -> np.ndarray:
    """A row of the right production run for every stroke, in the order the press made them.

    One row serves both stations. A production run is one T x A combination measured at
    both at once, so `run_strokes(level, level)` returns the same physical strokes whether
    it is asked of deep drawing or of ironing -- which is what lets the board show two
    stations of the same press rather than two unrelated presses.

    Each level keeps its own place in its run, so a crossfade interleaves two runs while
    both continue to advance through their own real strokes. Strokes either station's wear
    axis was fitted on are dropped: their position is an in-sample fit that would flatter
    the method.
    """
    fitted = np.concatenate([load_artifacts(key).wear.fitted_rows for key in STATIONS])
    reference = load_artifacts(next(iter(STATIONS))).data

    pools = {}
    for level in np.unique(wear_level):
        run = reference.run_strokes(int(level), int(level))[SETTLE:]
        pools[int(level)] = run[~np.isin(run, fitted)]

    taken = dict.fromkeys(pools, 0)
    rows = np.empty(N_STROKES, dtype=int)
    for stroke, level in enumerate(wear_level):
        rows[stroke] = pools[int(level)][taken[int(level)]]
        taken[int(level)] += 1
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
def load_run() -> Run:
    """Assemble the run. Cached -- the schedule is deterministic."""
    rng = np.random.default_rng(SEED)

    wear_level = _wear_schedule(rng)
    rows = _stroke_rows(wear_level)
    classified = {key: _classify(key, rows) for key in STATIONS}

    alignment = load_excentricity()
    infeed_level = _infeed_schedule()
    alignment_rows = np.empty(N_STROKES, dtype=int)
    for level in INFEED_LEVELS:
        strokes = np.flatnonzero(infeed_level == level)
        alignment_rows[strokes] = _draw_order(alignment.series_rows(level), rng, len(strokes))

    return Run(
        wear_level=wear_level,
        position={key: load_artifacts(key).wear.display(rows) for key in STATIONS},
        proba={key: classified[key][0] for key in STATIONS},
        model_name={key: classified[key][1] for key in STATIONS},
        rows=rows,
        alignment_mm=np.array([excentricity_mm(v) for v in alignment.predicted[alignment_rows]]),
        alignment_true_mm=np.array([excentricity_mm(v) for v in infeed_level]),
        alignment_rows=alignment_rows,
    )
