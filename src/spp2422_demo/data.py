"""Loading and slicing the committed curve extract.

Everything downstream sees normalized force on the shared 0..1 event-time axis, 500
samples per curve. Amplitudes are comparable *within* a dataset but never across the
measured and simulated ones -- the measured signals are uncalibrated volts and the
simulated ones kN, and no conversion between them exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "curves.npz"

N_SAMPLES = 500
EVENT_TIME = np.linspace(0.0, 1.0, N_SAMPLES)

LEVELS = (1, 2, 3)
# Strokes 0..TRAIN_STROKES-1 of each production run train, the rest is held out. Adjacent
# strokes within a run are near-duplicates, so a random split would leak; a contiguous
# block also keeps the held-out strokes on the far side of the tool's bedding-in drift.
TRAIN_STROKES = 400


@dataclass(frozen=True)
class Station:
    """One forming stage, and how its curves and labels are named in the extract."""

    key: str
    name: str
    label_key: str
    level_prefix: str
    curve_key: str
    peak_ref_key: str | None
    sim_key: str
    sim_mu_key: str
    description: str

    def level_name(self, level: int) -> str:
        return f"{self.level_prefix}{level}"


DEEP_DRAWING = Station(
    key="deep_drawing",
    name="Deep drawing",
    label_key="T",
    level_prefix="T",
    curve_key="real_deep_drawing",
    peak_ref_key=None,
    sim_key="sim_deep_drawing",
    sim_mu_key="sim_dd_mu",
    description=(
        "The punch draws the blank through the die. Force rises as the blank flows over "
        "the draw radius, then declines as the flange is consumed."
    ),
)

IRONING = Station(
    key="ironing",
    name="Ironing",
    label_key="A",
    level_prefix="A",
    curve_key="real_ironing",
    peak_ref_key="real_ironing_burst_suppressed",
    sim_key="sim_ironing",
    sim_mu_key="sim_ir_mu",
    description=(
        "The wall is thinned between punch and ironing ring. Friction acts directly on "
        "the forming force here, which makes this stage the more wear-sensitive of the two."
    ),
)

STATIONS = {station.key: station for station in (DEEP_DRAWING, IRONING)}


@dataclass(frozen=True)
class StationData:
    """Everything the app needs about one forming stage."""

    station: Station
    curves: np.ndarray  # (n, 500) measured, normalized force
    peak_ref: np.ndarray | None  # (n, 500) peak-location reference, ironing only
    labels: np.ndarray  # (n,) wear level, 1..3
    other_labels: np.ndarray  # (n,) the *other* stage's level, identifying the run
    stroke_index: np.ndarray  # (n,) position within the 500-stroke run
    sim_curves: np.ndarray  # (m, 500) FE-simulated, normalized force
    sim_levels: np.ndarray  # (m,) wear level from the mu tercile
    sim_mu: np.ndarray  # (m,) friction coefficient

    @property
    def train_mask(self) -> np.ndarray:
        return self.stroke_index < TRAIN_STROKES

    def run_mask(self, level: int, other_level: int) -> np.ndarray:
        return (self.labels == level) & (self.other_labels == other_level)

    def runs(self) -> list[tuple[int, int]]:
        """The (own level, other stage's level) pairs present, in a stable order."""
        pairs = {(int(a), int(b)) for a, b in zip(self.labels, self.other_labels, strict=True)}
        return sorted(pairs)

    def run_strokes(self, level: int, other_level: int) -> np.ndarray:
        """Row indices of one production run, ordered as the press produced them."""
        rows = np.flatnonzero(self.run_mask(level, other_level))
        return rows[np.argsort(self.stroke_index[rows])]

    def run_label(self, level: int, other_level: int) -> str:
        other_prefix = "A" if self.station.level_prefix == "T" else "T"
        return f"{self.station.level_prefix}{level} · {other_prefix}{other_level}"


def mu_terciles(mu: np.ndarray) -> np.ndarray:
    """Map friction coefficients onto wear levels 1..3 by tercile.

    No literal level-to-mu mapping was ever recorded for this process, so terciles of mu
    stand in for the measured roughness classes -- the same proxy the research code uses
    when it puts simulated and measured curves side by side. It orders the simulated
    curves correctly by severity; it is not a calibration, and nothing downstream treats
    a simulated level as interchangeable with a measured one.
    """
    edges = np.quantile(mu, [1 / 3, 2 / 3])
    return (np.digitize(mu, edges) + 1).astype(np.int8)


@cache
def _load_raw() -> dict[str, np.ndarray]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} is missing. Rebuild it with scripts/extract_data.py (see README)."
        )
    with np.load(DATA_FILE) as archive:
        return {key: archive[key] for key in archive.files}


@cache
def load_station(key: str) -> StationData:
    """Load one forming stage. Cached -- the extract is read once per process."""
    station = STATIONS[key]
    raw = _load_raw()
    other_key = "A" if station.label_key == "T" else "T"
    sim_mu = raw[station.sim_mu_key]

    return StationData(
        station=station,
        curves=raw[station.curve_key].astype(np.float32),
        peak_ref=(raw[station.peak_ref_key].astype(np.float32) if station.peak_ref_key else None),
        labels=raw[station.label_key].astype(int),
        other_labels=raw[other_key].astype(int),
        stroke_index=raw["stroke_index"].astype(int),
        sim_curves=raw[station.sim_key].astype(np.float32),
        sim_levels=mu_terciles(sim_mu),
        sim_mu=sim_mu,
    )
