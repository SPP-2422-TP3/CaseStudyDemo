"""Where a stroke sits between a pristine tool and a worn-out one, as one number.

`calibration.py` asks a research question -- can the simulated friction sweep put a
withheld wear state back between two real anchors -- and answers it by averaging over
seeds, windows and budgets. This module takes the *one* configuration that answered it
best and turns it into a reading a shop floor can act on: every measured stroke gets a
position on the friction axis, **0 at the pristine anchor and 1 at the worn one**.

Three things it is not:

- Not a fraction of life consumed. Nothing in this data records when a tool was retired,
  so 0.5 means "halfway between these two tools", not "half worn out".
- Not reliable stroke by stroke. The scatter of a single stroke is a large share of the
  span between the anchors, which is why every display averages a run of strokes.
- Not measured on the intermediate state. `CENTRE` is withheld from the fit exactly as it
  is in the research sweep, so a stroke of the intermediate tool is placed by a model that
  has never seen one. That is the point, and it is also the reason to read the number
  loosely.

The `budget` strokes per endpoint that the discrepancy GP is fitted on are recorded in
`fitted_rows` and excluded from anything the dashboard displays -- their position is an
in-sample fit and would read better than the method deserves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import (
    BUDGETS,
    CENTRE,
    ENDPOINTS,
    N_FEATURES,
    WINDOWS,
    Calibration,
    apply_zscore,
    combine,
    fit_gp,
    select_features,
    zscore_stats,
)
from .data import StationData

# The board streams whole production runs, so the axis is fitted on the window that spans
# one -- anything narrower leaves most strokes outside the domain the anchors came from.
# Within that window the sweep picks the budget; this is the fallback if none separates.
WINDOW = WINDOWS[-1]
FALLBACK_BUDGET = BUDGETS[WINDOW][-1]
SEED = 0


@dataclass(frozen=True)
class WearPosition:
    """The friction-axis position of every measured stroke of one forming stage."""

    station_key: str
    window: int  # strokes from the start of each run the anchors were taken from
    budget: int  # real strokes per endpoint spent on the discrepancy fit
    position: np.ndarray  # (n,) 0 = pristine anchor, 1 = worn anchor; not clipped
    fitted_rows: np.ndarray  # rows the discrepancy GP saw; never displayed

    def display(self, rows: np.ndarray | int) -> np.ndarray:
        """Positions clipped to 0..1, which is what a bar or a percentage can show."""
        return np.clip(self.position[rows], 0.0, 1.0)


def fit_wear_position(data: StationData, calibration: Calibration) -> WearPosition:
    """Fit the best-performing calibration once and score every measured stroke with it."""
    best = calibration.best(only_window=WINDOW)
    window, budget = (WINDOW, best[1] if best else FALLBACK_BUDGET)

    selected = select_features(data)
    inside = np.flatnonzero(data.stroke_index < window)

    # The anchors define the axis, so the z-scoring that feeds the GP is fitted on the
    # same window they come from and then applied unchanged to every other stroke.
    stats = zscore_stats(selected.real[inside])
    x_all = apply_zscore(selected.real, stats)

    sim = apply_zscore(selected.sim, zscore_stats(selected.sim))
    prior = fit_gp(sim, data.sim_mu, ard=sim.shape[1] <= N_FEATURES)

    # Same draw as the research sweep's seed 0: shuffle each endpoint level inside the
    # window, spend the first `budget` strokes on the fit, keep the rest for the anchors.
    rng = np.random.default_rng(SEED)
    drawn = {level: rng.permutation(inside[data.labels[inside] == level]) for level in ENDPOINTS}
    fitted_rows = np.concatenate([drawn[level][:budget] for level in ENDPOINTS])

    correction = None
    if budget:
        targets = np.concatenate(
            [np.full(budget, data.sim_mu.min()), np.full(budget, data.sim_mu.max())]
        )
        x_fit = x_all[fitted_rows]
        correction = fit_gp(x_fit, targets - prior.predict(x_fit), ard=False)

    mu_hat, _ = combine(prior, correction, x_all)

    low, high = (np.mean(mu_hat[drawn[level][budget:]]) for level in ENDPOINTS)
    span = high - low
    position = (mu_hat - low) / span if span else np.zeros_like(mu_hat)

    return WearPosition(
        station_key=data.station.key,
        window=window,
        budget=budget,
        position=position,
        fitted_rows=np.sort(fitted_rows),
    )


def centre_placement(data: StationData, wear: WearPosition) -> float:
    """Mean position of the withheld intermediate state -- 0.5 would be exactly centred.

    The single-configuration counterpart of the sweep's headline number, reported on the
    dashboard so the board and the research page cannot quietly disagree.
    """
    rows = np.flatnonzero((data.labels == CENTRE) & (data.stroke_index < wear.window))
    return float(np.mean(wear.position[rows]))
