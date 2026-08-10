"""Strip misalignment: reading tool infeed off the deep-drawing punch force.

The progressive die is fed 60 mm per stroke. Feeding it further pushes the blank
off-centre under the punch, and because three progressive stages precede deep drawing the
overfeed accumulates roughly threefold: 0.30 mm of overfeed puts the cup ~0.9 mm
off-centre, which draws one flange wide and the opposite wall thin.

The whole representation is two numbers. As the blank goes off-centre the force plateau
tilts downward and peak force drops, so each stroke is collapsed to the **slope and
intercept** of a straight line fit across the plateau window, and a random forest maps
that pair back to the infeed. This follows Moske et al., NUMISHEET 2025
(doi:10.1088/1742-6596/3104/1/012058).

Labels live in hundredths of a millimetre throughout -- `0, 5, ... 30` -- because that is
what the source folder names encode. Everything user-facing converts with `overfeed_mm`
or `excentricity_mm`; nothing displays a raw label.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

DATA = Path(__file__).resolve().parents[2] / "data" / "excentricity.npz"

REFERENCE_FEED_MM = 60.0
# Three progressive stages accumulate ahead of the deep-drawing station.
AMPLIFICATION = 3
# Overfeed levels actually run on the press, in hundredths of a millimetre.
INFEED_LEVELS = (0, 5, 10, 15, 20, 25, 30)
SEEDS = range(10)  # the paper validates across 10 random splits

FOREST = dict(n_estimators=20, max_depth=4)


def overfeed_mm(label: int | float) -> float:
    """Hundredths of a millimetre of overfeed -> millimetres."""
    return float(label) / 100


def excentricity_mm(label: int | float) -> float:
    """Overfeed as it arrives at the cup, after three stages have compounded it."""
    return overfeed_mm(label) * AMPLIFICATION


@dataclass
class Excentricity:
    """Every stroke of the seven experimental series, with its features and prediction."""

    curves: np.ndarray  # (343, 912) normalized force
    labels: np.ndarray  # (343,) hundredths of a mm of overfeed
    stroke_index: np.ndarray  # (343,) position within its own 49-stroke run
    time: np.ndarray  # (912,) seconds on the press cycle
    plateau: tuple[int, int]  # index window the line is fitted over
    force_scale: float  # kN per unit of normalized force
    features: np.ndarray  # (343, 2) slope and intercept, normalized units per sample
    fit_quality: np.ndarray  # (343,) R^2 of each stroke's plateau fit
    predicted: np.ndarray  # (343,) out-of-fold prediction, hundredths of a mm
    mae_mm: float  # mean over seeds of the held-out MAE
    mae_std_mm: float

    @property
    def dt(self) -> float:
        """Seconds per sample of the resampled trace."""
        return float(self.time[1] - self.time[0])

    def slope_kn_per_s(self, index: int) -> float:
        """The fitted slope as a physical rate, which is how the paper plots it."""
        return float(self.features[index, 0]) * self.force_scale / self.dt

    def intercept_kn(self, index: int) -> float:
        """Where the fitted line crosses the start of the plateau window, in kN.

        Reported at the window edge rather than at t=0: the raw intercept extrapolates
        hundreds of samples back to a point the stroke never visits, so it reads as a
        number with no physical referent.
        """
        start = self.plateau[0]
        value = self.features[index, 1] + self.features[index, 0] * start
        return float(value) * self.force_scale

    def force_kn(self, index: int) -> np.ndarray:
        return self.curves[index] * self.force_scale

    def fitted_line_kn(self, index: int) -> np.ndarray:
        """The regression line across the plateau, in kN, ready to plot."""
        start, end = self.plateau
        slope, intercept = self.features[index]
        return (intercept + slope * np.arange(start, end)) * self.force_scale

    def series_rows(self, label: int) -> np.ndarray:
        """Row indices of one experimental series, in production order."""
        return np.flatnonzero(self.labels == label)

    def running_mean(self, label: int, stroke: int, window: int = 10) -> float:
        """Prediction averaged over this stroke and the ones before it in the same run.

        A single stroke carries several hundredths of a millimetre of scatter, so the
        stroke-by-stroke number bounces in a way the underlying misalignment does not.
        Averaging a handful of consecutive strokes cuts the error by about a third.
        """
        rows = self.series_rows(label)[max(0, stroke - window + 1) : stroke + 1]
        return float(np.mean(self.predicted[rows]))

    def row(self, label: int, stroke: int) -> int:
        rows = self.series_rows(label)
        return int(rows[min(stroke, len(rows) - 1)])

    def mean_curve_kn(self, label: int) -> np.ndarray:
        return self.curves[self.labels == label].mean(axis=0) * self.force_scale


def _plateau_features(curves: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Slope, intercept and fit quality of a straight line across the plateau."""
    x = np.arange(start, end)
    fits = [linregress(x, curve[start:end]) for curve in curves]
    features = np.array([(fit.slope, fit.intercept) for fit in fits])
    return features, np.array([fit.rvalue**2 for fit in fits])


@cache
def load_excentricity() -> Excentricity:
    """The dataset, its features and the forest's predictions. Cached for the session."""
    if not DATA.exists():
        raise FileNotFoundError(
            f"{DATA} is missing -- run `uv run python scripts/extract_excentricity_data.py`"
        )
    payload = np.load(DATA)
    curves = payload["curves"]
    labels = payload["labels"].astype(int)
    start, end = (int(value) for value in payload["plateau"])

    features, fit_quality = _plateau_features(curves, start, end)

    # Held-out error, exactly as the paper reports it: 10 random 80/20 splits.
    scores = []
    for seed in SEEDS:
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=seed
        )
        forest = RandomForestRegressor(random_state=seed, **FOREST).fit(x_train, y_train)
        scores.append(mean_absolute_error(y_test, forest.predict(x_test)) / 100)

    # Every stroke shown in the dashboard is predicted by a forest that did not train on
    # it. Fitting once on everything and displaying that would show the training fit, and
    # a depth-4 forest over 343 points would look far better on screen than it is.
    predicted = cross_val_predict(
        RandomForestRegressor(random_state=0, **FOREST),
        features,
        labels,
        cv=KFold(n_splits=10, shuffle=True, random_state=0),
    )

    return Excentricity(
        curves=curves,
        labels=labels,
        stroke_index=payload["stroke_index"].astype(int),
        time=payload["time"],
        plateau=(start, end),
        force_scale=float(payload["force_scale"]),
        features=features,
        fit_quality=fit_quality,
        predicted=predicted,
        mae_mm=float(np.mean(scores)),
        mae_std_mm=float(np.std(scores)),
    )
