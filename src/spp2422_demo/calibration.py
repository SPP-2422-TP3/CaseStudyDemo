"""Locating a wear state that was never labelled, by anchoring on the simulated sweep.

The intermediate wear state is the one that matters and the one nobody can measure: a
tool passes through it uncontrolled on its way to producing scrap, and the press cannot
be held there long enough to collect labelled strokes. So it is withheld from training
entirely -- only the pristine and the heavily worn endpoints are given as real data --
and the question is whether the FE friction sweep, which *does* span the middle
continuously, can put the withheld state back where it belongs.

The mechanism is Bayesian calibration rather than classification, because the useful
output is a position on the wear axis rather than a label:

1. A GP prior maps curve descriptors to the friction coefficient, fit on the simulated
   sweep alone -- physics, with no measurement in it.
2. A second GP fits the *discrepancy* between that prior and a handful of real endpoint
   strokes, correcting the sweep for everything the model does not capture.
3. Their sum places any real stroke on the friction axis, and the withheld centre state
   should land midway between the two endpoints. `mid_position` reports where it lands:
   0 at the low anchor, 1 at the high one, 0.5 exactly centred.

Two controls decide whether a placement means anything. **shuffled-sim** refits the prior
on permuted friction labels, so the sweep keeps its curves and loses its physical
ordering; beating it is what separates a real result from "any GP prior interpolates
between two anchors". **real-only** drops the sweep and keeps the endpoint strokes, which
isolates what the simulation itself contributes.

Everything is restricted to an early window of each production run. The die beds in as a
block runs, its surface smoothing back towards the fresh state, so late strokes carry a
wear label their force curve no longer supports -- taking them into training is actively
harmful, not merely uninformative.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.stats import ttest_rel
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import f1_score, roc_auc_score

from .data import LEVELS, StationData
from .features import feature_matrix

# The state withheld from training: the transient threshold between good and scrap.
CENTRE = 2
ENDPOINTS = (1, 3)

SEEDS = tuple(range(8))
# Windows are strokes from the start of each run; 500 is the whole run, i.e. no window.
# Ordered narrowest first, so the panels read left-to-right as "less early data" to
# "the whole run" -- matching the budget axis, which also grows within each panel.
WINDOWS = (20, 50, 500)
BUDGETS = {500: (0, 5, 10, 25, 50), 50: (0, 5, 10, 25), 20: (0, 5, 10)}
# One length-scale per feature needs points to estimate it from, and the sweep supplies
# 11 (deep drawing) or 33 (ironing). Six keeps the kernel well inside that budget.
N_FEATURES = 6

VARIANTS = ("mix", "shuffled-sim", "real-only")


@dataclass(frozen=True)
class Placement:
    """Where the withheld centre state landed, for one window/budget/variant."""

    window: int
    budget: int
    variant: str
    position: float  # mean over seeds; 0 = low anchor, 1 = high anchor
    spread: float  # standard deviation over seeds
    alarm_auc: float  # centre-or-worse vs. confidently-good, threshold-free
    accuracy: float  # mean over seeds; the tercile call from the same discretised mean
    f1: float  # macro-F1 over seeds, same tercile call; 1/3 is chance on three levels


@dataclass(frozen=True)
class Calibration:
    """The whole sweep for one forming stage."""

    station_key: str
    features: list[str]
    placements: list[Placement]
    # Paired across seeds, mix vs. shuffled-sim, per (window, budget).
    p_values: dict[tuple[int, int], float]

    def at(self, window: int, budget: int, variant: str) -> Placement | None:
        for placement in self.placements:
            if (placement.window, placement.budget, placement.variant) == (
                window,
                budget,
                variant,
            ):
                return placement
        return None

    def best(self) -> tuple[int, int, float] | None:
        """The (window, budget) whose placement separates most convincingly, if any does.

        Ranked by p-value and required to sit nearer the centre than the control, so a
        configuration that is significantly *wrong* is never reported as the headline.
        """
        candidates = []
        for (window, budget), p in self.p_values.items():
            mix = self.at(window, budget, "mix")
            control = self.at(window, budget, "shuffled-sim")
            if mix is None or control is None or not np.isfinite(p):
                continue
            if abs(mix.position - 0.5) < abs(control.position - 0.5):
                candidates.append((p, window, budget))
        if not candidates:
            return None
        p, window, budget = min(candidates)
        return window, budget, p


def _kernel(n_features: int, *, ard: bool):
    """Constant * RBF + white noise.

    ARD -- a length-scale per feature -- for the prior, which has the whole sweep to fit
    it. The discrepancy GP can see as few as five strokes, far too little to identify
    per-feature scales, so it gets one shared length-scale.
    """
    length_scale = np.ones(n_features) if ard else 1.0
    return ConstantKernel(1.0, (1e-2, 1e2)) * RBF(
        length_scale=length_scale, length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))


def _fit_gp(x: np.ndarray, y: np.ndarray, *, ard: bool) -> GaussianProcessRegressor:
    gp = GaussianProcessRegressor(
        kernel=_kernel(x.shape[1], ard=ard),
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=0,
    )
    with warnings.catch_warnings():
        # The simulated curves are noise-free averages, so the fitted noise term really
        # does want to sit on its lower bound. That is the correct answer here, not a
        # convergence problem, and it would otherwise print once per fit.
        warnings.simplefilter("ignore", ConvergenceWarning)
        return gp.fit(x, y)


def _standardize(x: np.ndarray) -> np.ndarray:
    """Z-score each column against its own domain.

    Fit separately on the simulated and the measured matrix, never shared. The two are
    normalized against different references upstream -- measured against a percentile of
    per-stroke maxima, simulated against the population max of noise-free curves -- so
    their raw magnitudes are not comparable, but "how extreme within its own population"
    is.
    """
    spread = x.std(axis=0)
    return (x - x.mean(axis=0)) / np.where(spread > 0, spread, 1.0)


def _correlate(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """|Pearson r| of every column of `x` against `y`, with constant columns scored 0.

    A descriptor can be flat across the sweep -- the burst features are, since the
    simulation has no contact ring -- and correlating against it is a divide by zero.
    """
    out = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        column = x[:, i]
        if column.std() > 0 and y.std() > 0:
            out[i] = abs(np.corrcoef(y, column)[0, 1])
    return out


def select_features(data: StationData) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Descriptors this stage can be calibrated on: simulated matrix, real matrix, names.

    Two cuts. For ironing only, an *axis-consistency* filter: the trace there superimposes
    the ironing die's response on the upstream deep-drawing die's, so a descriptor may
    correlate with ironing friction in simulation while its real variation tracks the
    upstream state instead. Training on that relationship learns the wrong axis. A
    descriptor is kept only when simulation and measurement agree on which axis it
    follows. Deep drawing has no competing axis and skips this.

    Then the strongest `N_FEATURES` by correlation with the sweep's own friction axis,
    to keep the GP's kernel smaller than the sweep that fits it.
    """
    burst = data.peak_ref is not None
    x_sim, names = feature_matrix(data.sim_curves, burst=burst, peak_ref=None)
    x_real, _ = feature_matrix(data.curves, burst=burst, peak_ref=data.peak_ref)

    keep = list(range(len(names)))
    if data.sim_mu_other is not None:
        own = _correlate(data.sim_mu, x_sim)
        upstream = _correlate(data.sim_mu_other, x_sim)
        keep = []
        for i in range(len(names)):
            tracks_own_in_sim = own[i] >= upstream[i]
            own_spread = _level_spread(x_real[:, i], data.labels)
            upstream_spread = _level_spread(x_real[:, i], data.other_labels)
            if tracks_own_in_sim and own_spread >= upstream_spread:
                keep.append(i)

    strongest = np.argsort(-_correlate(data.sim_mu, x_sim[:, keep]))[:N_FEATURES]
    chosen = [keep[i] for i in strongest]
    return x_sim[:, chosen], x_real[:, chosen], [names[i] for i in chosen]


def _level_spread(column: np.ndarray, labels: np.ndarray) -> float:
    """How far a descriptor's per-level means spread out -- its variation along one axis."""
    means = [column[labels == level].mean() for level in LEVELS]
    return float(max(means) - min(means))


def _score(
    mean: np.ndarray, std: np.ndarray, levels: np.ndarray, edges: np.ndarray
) -> tuple[float, float, float, float]:
    """Placement of the centre state, alarm separability, and tercile classification quality."""
    low, high = mean[levels == ENDPOINTS[0]], mean[levels == ENDPOINTS[1]]
    centre = mean[levels == CENTRE]
    span = high.mean() - low.mean() if len(low) and len(high) else np.nan
    position = float((centre.mean() - low.mean()) / span) if span and len(centre) else np.nan

    # An alarm only has to flag elevated risk, not name the level, so it is scored
    # threshold-free on "centre or worse" against "confidently good". Uncertainty counts
    # as much as severity here: a model that does not know is itself a reason to look.
    def z(values: np.ndarray) -> np.ndarray:
        spread = values.std()
        return (values - values.mean()) / spread if spread > 0 else np.zeros_like(values)

    risky = (levels >= CENTRE).astype(int)
    mixed = 0 < risky.sum() < len(risky)
    alarm = float(roc_auc_score(risky, z(mean) + z(std))) if mixed else np.nan

    # The same continuous mean, cut at the sweep's own tercile edges and named a level --
    # a harsher, discrete read of the same placement, including on the centre state that
    # the position and alarm scores above never ask to be classified correctly.
    predicted = np.digitize(mean, edges) + 1
    accuracy = float(np.mean(predicted == levels))
    f1 = float(f1_score(levels, predicted, labels=list(LEVELS), average="macro", zero_division=0))
    return position, alarm, accuracy, f1


def calibrate(data: StationData) -> Calibration:
    """Run the whole window/budget sweep for one stage, over every seed and control."""
    x_sim_raw, x_real_raw, names = select_features(data)
    x_sim = _standardize(x_sim_raw)
    ard = x_sim.shape[1] <= N_FEATURES
    edges = np.quantile(data.sim_mu, [1 / 3, 2 / 3])
    low_target, high_target = data.sim_mu.min(), data.sim_mu.max()

    # The prior depends on the sweep and the seed only, never on the window or the
    # budget, so it is fit once per seed instead of once per cell.
    priors = {}
    for seed in SEEDS:
        shuffled = np.random.default_rng(seed).permutation(data.sim_mu)
        priors[seed] = (
            _fit_gp(x_sim, data.sim_mu, ard=ard),
            _fit_gp(x_sim, shuffled, ard=ard),
        )

    raw: dict[tuple[int, int, str], list[tuple[float, float, float, float]]] = {}
    for window in WINDOWS:
        inside = data.stroke_index < window
        x_window = _standardize(x_real_raw[inside])
        levels_window = data.labels[inside]
        centre_rows = np.flatnonzero(levels_window == CENTRE)

        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            order = {
                level: rng.permutation(np.flatnonzero(levels_window == level))
                for level in ENDPOINTS
            }
            prior, prior_shuffled = priors[seed]

            for budget in BUDGETS[window]:
                # Endpoint strokes spent on training cannot also be scored.
                train_rows = np.concatenate([order[level][:budget] for level in ENDPOINTS])
                holdout = np.concatenate(
                    [*(order[level][budget:] for level in ENDPOINTS), centre_rows]
                )
                x_holdout, levels_holdout = x_window[holdout], levels_window[holdout]

                fitted: dict[str, tuple] = {}
                if budget:
                    x_train = x_window[train_rows]
                    y_train = np.concatenate(
                        [np.full(budget, low_target), np.full(budget, high_target)]
                    )
                    for variant, base in (("mix", prior), ("shuffled-sim", prior_shuffled)):
                        residual = y_train - base.predict(x_train)
                        fitted[variant] = (base, _fit_gp(x_train, residual, ard=False))
                    fitted["real-only"] = (None, _fit_gp(x_train, y_train, ard=False))
                else:
                    fitted = {"mix": (prior, None), "shuffled-sim": (prior_shuffled, None)}

                for variant in VARIANTS:
                    if variant not in fitted:
                        continue
                    base, correction = fitted[variant]
                    mean, std = _combine(base, correction, x_holdout)
                    key = (window, budget, variant)
                    raw.setdefault(key, []).append(_score(mean, std, levels_holdout, edges))

    return _summarise(data.station.key, names, raw)


def _combine(
    prior: GaussianProcessRegressor | None,
    correction: GaussianProcessRegressor | None,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Prior plus discrepancy, variances added.

    The two GPs are fit on disjoint data -- simulated curves and measured strokes -- so
    treating them as independent is a fair approximation rather than an exact posterior.
    """
    if prior is None:  # real-only: the correction is the whole model
        return correction.predict(x, return_std=True)
    mean, std = prior.predict(x, return_std=True)
    if correction is None:
        return mean, std
    mean_correction, std_correction = correction.predict(x, return_std=True)
    return mean + mean_correction, np.sqrt(std**2 + std_correction**2)


def _summarise(
    station_key: str,
    names: list[str],
    raw: dict[tuple[int, int, str], list[tuple[float, float, float, float]]],
) -> Calibration:
    placements = []
    for (window, budget, variant), scores in sorted(raw.items()):
        positions = np.array([s[0] for s in scores])
        alarms = np.array([s[1] for s in scores])
        accuracies = np.array([s[2] for s in scores])
        f1s = np.array([s[3] for s in scores])
        placements.append(
            Placement(
                window=window,
                budget=budget,
                variant=variant,
                position=float(np.nanmean(positions)),
                spread=float(np.nanstd(positions)),
                alarm_auc=float(np.nanmean(alarms)),
                accuracy=float(np.nanmean(accuracies)),
                f1=float(np.nanmean(f1s)),
            )
        )

    p_values = {}
    for window, budget, variant in raw:
        if variant != "mix" or (window, budget) in p_values:
            continue
        mix = np.array([s[0] for s in raw[(window, budget, "mix")]])
        control = np.array([s[0] for s in raw[(window, budget, "shuffled-sim")]])
        usable = np.isfinite(mix) & np.isfinite(control)
        p = np.nan
        if usable.sum() > 1 and not np.allclose(mix[usable], control[usable]):
            p = float(ttest_rel(mix[usable], control[usable]).pvalue)
        p_values[(window, budget)] = p

    return Calibration(
        station_key=station_key, features=names, placements=placements, p_values=p_values
    )
