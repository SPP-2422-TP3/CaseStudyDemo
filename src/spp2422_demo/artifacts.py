"""Training the models once, calibrating against the sweep, and keeping both on disk.

The dashboard should start instantly, so results are cached. Delete `data/models/` (or
run `spp2422-demo prepare --force`) to rebuild.

Two questions are answered here, and they are not the same question:

- **Held-out strokes** -- train on strokes 0..399 of each production run, test on the
  rest. This is the split the deployed classifiers use, and it measures monitoring a tool
  that has already been characterised.
- **The withheld centre state** -- take the intermediate wear level out of training
  altogether and ask whether the simulated friction sweep can put it back between the two
  real endpoints. That is `calibration.py`, and it is the question worth asking about a
  state nobody can label in production.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from .calibration import Calibration, calibrate
from .data import STATIONS, StationData, load_station
from .features import feature_matrix
from .models import WearModel, build_models
from .wear_position import WearPosition, centre_placement, fit_wear_position

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
# Bump when anything that changes a trained model changes, so stale caches are ignored.
CACHE_VERSION = 7


@dataclass
class TrainedStation:
    """Models for one forming stage, its centre-state calibration and its wear axis."""

    data: StationData
    models: dict[str, WearModel]
    accuracy: dict[str, float]  # held-out strokes of known runs
    calibration: Calibration
    wear: WearPosition  # per-stroke position between the pristine and the worn anchor

    @property
    def default_model(self) -> str:
        """The model to select first."""
        return max(self.models, key=lambda key: self.accuracy.get(key, 0.0))


def _train(key: str) -> TrainedStation:
    data = load_station(key)
    burst = data.peak_ref is not None  # only ironing rings on contact
    features, _ = feature_matrix(data.curves, burst=burst, peak_ref=data.peak_ref)
    curves = np.asarray(data.curves, dtype=np.float32)

    train, test = data.train_mask, ~data.train_mask
    models: dict[str, WearModel] = {}
    accuracy: dict[str, float] = {}
    for model in build_models(burst=burst):
        model.fit_matrix(curves[train], features[train], data.labels[train])
        predicted = model.predict_proba_matrix(curves[test], features[test]).argmax(axis=1) + 1
        models[model.key] = model
        accuracy[model.key] = float(np.mean(predicted == data.labels[test]))

    calibration = calibrate(data)
    return TrainedStation(
        data=data,
        models=models,
        accuracy=accuracy,
        calibration=calibration,
        wear=fit_wear_position(data, calibration),
    )


@cache
def load_artifacts(key: str) -> TrainedStation:
    """Trained models for one stage, from cache when available."""
    path = CACHE_DIR / f"{key}.pkl"
    if path.exists():
        payload = pickle.loads(path.read_bytes())
        if payload.get("version") == CACHE_VERSION:
            return TrainedStation(
                data=load_station(key),
                models=payload["models"],
                accuracy=payload["accuracy"],
                calibration=payload["calibration"],
                wear=payload["wear"],
            )

    trained = _train(key)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        pickle.dumps(
            {
                "version": CACHE_VERSION,
                "models": trained.models,
                "accuracy": trained.accuracy,
                "calibration": trained.calibration,
                "wear": trained.wear,
            }
        )
    )
    return trained


def prepare(force: bool = False) -> None:
    """Train and cache every stage, printing what each one is worth."""
    if force:
        for path in CACHE_DIR.glob("*.pkl"):
            path.unlink()
        load_artifacts.cache_clear()

    for key, station in STATIONS.items():
        trained = load_artifacts(key)
        held_out = int((~trained.data.train_mask).sum())
        print(f"\n==> {station.name}, {held_out} held-out strokes")
        for model_key in trained.models:
            print(f"    {trained.models[model_key].name:22s} {trained.accuracy[model_key]:9.1%}")

        calibration = trained.calibration
        print(f"    calibrated on {', '.join(calibration.features)}")
        wear = trained.wear
        print(
            f"    wear axis from first {wear.window} strokes, {wear.budget} real/endpoint: "
            f"the withheld state sits at {centre_placement(trained.data, wear):.3f}"
        )
        best = calibration.best()
        if best is None:
            print("    ! the withheld centre state is never placed better than the control")
            continue
        window, budget, p = best
        mix = calibration.at(window, budget, "mix")
        control = calibration.at(window, budget, "shuffled-sim")
        print(
            f"    centre state best placed at first {window} strokes, {budget} real/endpoint: "
            f"{mix.position:.3f} vs {control.position:.3f} shuffled (p = {p:.4f}); "
            f"0.5 would be exactly centred"
        )
