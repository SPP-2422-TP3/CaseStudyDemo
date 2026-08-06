"""Training the models once, validating them two ways, and keeping both on disk.

The dashboard should start instantly, so results are cached. Delete `data/models/` (or
run `spp2422-demo prepare --force`) to rebuild.

Two accuracies are reported for every model, because they answer different questions and
they disagree sharply for one of the two stages:

- **Held-out strokes** -- train on strokes 0..399 of each production run, test on the
  rest. This is the split the deployed model uses. It measures monitoring an already
  characterised tool.
- **Unseen run** -- hold out a whole T x A production run, train on the other eight.
  This measures whether the model recognises the wear state itself rather than the run
  it came from, and it is the number to trust when asking "would this work on a tool we
  have not seen before".
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np

from .data import STATIONS, StationData, load_station
from .features import feature_matrix
from .models import FeatureModel, WearModel, build_models

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
# Bump when anything that changes a trained model changes, so stale caches are ignored.
CACHE_VERSION = 2


@dataclass
class TrainedStation:
    """Models for one forming stage, with both validation scores."""

    data: StationData
    models: dict[str, WearModel]
    accuracy: dict[str, float]  # held-out strokes of known runs
    run_accuracy: dict[str, float] = field(default_factory=dict)  # unseen run

    @property
    def default_model(self) -> str:
        """The model to select first: best on the honest split, ties broken by the other."""
        return max(
            self.models,
            key=lambda key: (self.run_accuracy.get(key, 0.0), self.accuracy.get(key, 0.0)),
        )

    def generalises(self, threshold: float = 0.5) -> bool:
        """Whether any model beats coin-flip territory on a run it has never seen."""
        return bool(self.run_accuracy) and max(self.run_accuracy.values()) >= threshold


def _fit(model: WearModel, data: StationData, mask: np.ndarray, features: np.ndarray) -> None:
    if isinstance(model, FeatureModel):
        model.fit_matrix(features[mask], data.labels[mask])
    else:
        model.fit(data.curves[mask], data.labels[mask], None)


def _predict(model: WearModel, data: StationData, mask: np.ndarray, features: np.ndarray):
    if isinstance(model, FeatureModel):
        return model.predict_proba_matrix(features[mask])
    return model.predict_proba(data.curves[mask], None)


def _train(key: str) -> TrainedStation:
    data = load_station(key)
    burst = data.peak_ref is not None  # only ironing rings on contact
    features, _ = feature_matrix(data.curves, burst=burst, peak_ref=data.peak_ref)

    train, test = data.train_mask, ~data.train_mask
    models: dict[str, WearModel] = {}
    accuracy: dict[str, float] = {}
    for model in build_models(burst=burst):
        _fit(model, data, train, features)
        predicted = _predict(model, data, test, features).argmax(axis=1) + 1
        models[model.key] = model
        accuracy[model.key] = float(np.mean(predicted == data.labels[test]))

    # Leave-one-run-out, on freshly built models so the deployed ones stay untouched.
    folds: dict[str, list[float]] = {}
    for own, other in data.runs():
        held = data.run_mask(own, other)
        for model in build_models(burst=burst):
            _fit(model, data, ~held, features)
            predicted = _predict(model, data, held, features).argmax(axis=1) + 1
            folds.setdefault(model.key, []).append(float(np.mean(predicted == own)))
    run_accuracy = {key: float(np.mean(scores)) for key, scores in folds.items()}

    return TrainedStation(data=data, models=models, accuracy=accuracy, run_accuracy=run_accuracy)


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
                run_accuracy=payload["run_accuracy"],
            )

    trained = _train(key)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        pickle.dumps(
            {
                "version": CACHE_VERSION,
                "models": trained.models,
                "accuracy": trained.accuracy,
                "run_accuracy": trained.run_accuracy,
            }
        )
    )
    return trained


def prepare(force: bool = False) -> None:
    """Train and cache every stage, printing both validation scores."""
    if force:
        for path in CACHE_DIR.glob("*.pkl"):
            path.unlink()
        load_artifacts.cache_clear()

    for key, station in STATIONS.items():
        trained = load_artifacts(key)
        held_out = int((~trained.data.train_mask).sum())
        print(f"\n==> {station.name} ({station.german}), {held_out} held-out strokes")
        print(f"    {'':22s} {'held-out':>10s} {'unseen run':>12s}")
        for model_key in trained.models:
            print(
                f"    {trained.models[model_key].name:22s}"
                f" {trained.accuracy[model_key]:9.1%} {trained.run_accuracy[model_key]:11.1%}"
            )
        if not trained.generalises():
            print("    ! no model beats chance on an unseen run -- see README")
