"""Sanity checks on the data, the features and the trained models."""

from __future__ import annotations

import numpy as np
import pytest

from spp2422_demo.artifacts import load_artifacts
from spp2422_demo.calibration import (
    BUDGETS,
    CENTRE,
    ENDPOINTS,
    N_FEATURES,
    VARIANTS,
    WINDOWS,
)
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


@pytest.mark.parametrize("key", STATION_KEYS)
def test_the_centre_state_is_never_trained_on(key):
    """The whole claim rests on the withheld state being genuinely withheld."""
    data = load_station(key)
    inside = data.stroke_index < min(WINDOWS)
    trainable = set(np.unique(data.labels[inside])) - {CENTRE}
    assert trainable == set(ENDPOINTS), "only the two endpoints may anchor the calibration"


@pytest.mark.parametrize("key", STATION_KEYS)
def test_calibration_reports_every_variant_against_its_control(key):
    calibration = load_artifacts(key).calibration
    assert len(calibration.features) == N_FEATURES
    for window in WINDOWS:
        for budget in BUDGETS[window]:
            # A budget of zero has no real strokes to fit a real-only model from.
            expected = set(VARIANTS) if budget else {"mix", "shuffled-sim"}
            present = {v for v in VARIANTS if calibration.at(window, budget, v) is not None}
            assert present == expected, f"{key} {window}/{budget}"


def test_ironing_places_the_withheld_state_better_than_a_scrambled_sweep():
    """The claim the Wear threshold page makes -- guard it, including the direction.

    If a future change breaks the result, this fails loudly and the page's wording has to
    be revisited rather than quietly becoming untrue.
    """
    calibration = load_artifacts("ironing").calibration
    best = calibration.best()
    assert best is not None
    window, budget, p = best
    assert p < 0.05
    mix = calibration.at(window, budget, "mix")
    control = calibration.at(window, budget, "shuffled-sim")
    assert abs(mix.position - 0.5) < abs(control.position - 0.5)


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


@pytest.mark.parametrize("key", STATION_KEYS)
def test_the_force_axis_is_fixed_across_strokes(key):
    """A stroke-dependent y range would rescale the curve on every tick of the stream,
    hiding the amplitude differences the view exists to show."""
    from spp2422_demo.components.curve_figure import stroke_figure

    data = load_station(key)
    ranges = {tuple(stroke_figure(data, i).layout.yaxis.range) for i in (0, 250, 4499)}
    assert len(ranges) == 1, "the y range follows the selected stroke"

    low, high = ranges.pop()
    assert low < data.curves.min() and high > data.curves.max()


def test_every_page_imports_and_has_a_layout():
    """Dash imports page modules from their file path, which breaks constructs that expect
    the module in `sys.modules`. It does so lazily at server start, so without this the
    first symptom is a stack trace instead of a dashboard."""
    import dash

    from spp2422_demo.app import app  # noqa: F401  -- importing builds the app

    pages = dash.page_registry
    assert {page["path"] for page in pages.values()} == {
        "/",
        "/overview",
        "/deep-drawing",
        "/ironing",
        "/wear-threshold",
        "/excentricity",
        "/help",
    }
    for page in pages.values():
        assert callable(page["layout"])
        page["layout"]()

    # The status board is the landing page; everything else sits behind the Details menu.
    top_level = {page["path"] for page in pages.values() if page.get("top_level")}
    assert top_level == {"/"}


@pytest.mark.parametrize("key", STATION_KEYS)
def test_the_wear_threshold_body_renders(key):
    """Building the layout is not enough -- the page's content comes from a callback, and
    a mismatch between it and the figure module only shows up when that callback runs."""
    import sys

    from spp2422_demo.app import app  # noqa: F401  -- importing registers the pages

    # Dash imports page modules by file path, so they land under "pages.", not the package.
    page = sys.modules["pages.wear_threshold"]
    body = page._body(key)
    assert body.children, f"{key} rendered an empty body"
