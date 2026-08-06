"""Sanity checks on the data, the features and the trained models."""

from __future__ import annotations

import numpy as np
import pytest

from spp2422_demo.artifacts import load_artifacts
from spp2422_demo.data import LEVELS, N_SAMPLES, STATIONS, load_station, mu_terciles
from spp2422_demo.explain import explain
from spp2422_demo.features import curve_features

STATION_KEYS = list(STATIONS)


@pytest.mark.parametrize("key", STATION_KEYS)
def test_extract_has_expected_shape(key):
    data = load_station(key)
    n = len(data.curves)
    assert data.curves.shape == (n, N_SAMPLES)
    assert data.labels.shape == data.other_labels.shape == data.stroke_index.shape == (n,)
    assert set(np.unique(data.labels)) == set(LEVELS)
    assert data.sim_curves.shape[1] == N_SAMPLES
    if data.peak_ref is not None:
        assert data.peak_ref.shape == data.curves.shape


@pytest.mark.parametrize("key", STATION_KEYS)
def test_runs_are_complete_production_runs(key):
    data = load_station(key)
    runs = data.runs()
    assert len(runs) == 9  # every T x A combination
    for own, other in runs:
        strokes = data.run_strokes(own, other)
        assert len(strokes) == 500
        # Returned in production order, which is what the live stream walks.
        assert np.all(np.diff(data.stroke_index[strokes]) > 0)


def test_mu_terciles_split_into_three_levels():
    levels = mu_terciles(np.linspace(0.02, 0.20, 12))
    assert set(np.unique(levels)) == set(LEVELS)
    # Higher friction never maps to a lower wear level.
    assert np.all(np.diff(levels) >= 0)


@pytest.mark.parametrize("key", STATION_KEYS)
def test_features_are_finite_on_measured_and_simulated_curves(key):
    data = load_station(key)
    burst = data.peak_ref is not None
    for index in (0, len(data.curves) // 2, len(data.curves) - 1):
        ref = data.peak_ref[index] if burst else None
        values = curve_features(data.curves[index], burst=burst, peak_ref=ref)
        assert np.all(np.isfinite(list(values.values())))
    for curve in data.sim_curves:
        assert np.all(np.isfinite(list(curve_features(curve, burst=burst).values())))


@pytest.mark.parametrize("key", STATION_KEYS)
def test_models_beat_chance_on_held_out_strokes(key):
    trained = load_artifacts(key)
    assert set(trained.accuracy) == set(trained.models)
    for model_key, score in trained.accuracy.items():
        assert score > 1 / len(LEVELS), f"{model_key} is at or below chance"


def test_deep_drawing_generalises_to_an_unseen_run():
    """The claim the demo actually makes for deep drawing -- guard it."""
    assert load_artifacts("deep_drawing").generalises(threshold=0.8)


def test_ironing_does_not_generalise_to_an_unseen_run():
    """The counterpart, kept as a test so the caveat on the overview stays truthful.

    If a future change makes ironing generalise, this fails loudly and the wording on the
    overview page needs revisiting -- that is the point.
    """
    assert not load_artifacts("ironing").generalises(threshold=0.5)


@pytest.mark.parametrize("key", STATION_KEYS)
def test_attributions_cover_the_whole_stroke(key):
    trained = load_artifacts(key)
    data = trained.data
    index = 0
    peak_ref = data.peak_ref[index] if data.peak_ref is not None else None
    for model in trained.models.values():
        attribution = explain(model, data.curves[index], peak_ref, level=3)
        assert attribution.values.shape == (N_SAMPLES,)
        assert np.all(np.isfinite(attribution.values))
        assert np.isclose(np.abs(attribution.values).max(), 1.0)
        start, end = attribution.focus
        assert 0.0 <= start <= end <= 1.0
