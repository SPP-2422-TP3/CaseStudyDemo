"""Checks on the status board: the wear axis, the assembled runs and the verdict."""

from __future__ import annotations

import numpy as np
import pytest

from spp2422_demo.artifacts import load_artifacts
from spp2422_demo.calibration import CENTRE, ENDPOINTS
from spp2422_demo.components.status_cards import CARDS, board, detail
from spp2422_demo.data import LEVELS, STATIONS
from spp2422_demo.health import (
    CRITICAL,
    DEFAULT_TOLERANCE_MM,
    GOOD,
    RANK,
    machine_state,
    majority_level,
)
from spp2422_demo.scenario import N_STROKES, SCENARIOS, WEAR_WINDOW, load_run
from spp2422_demo.wear_position import centre_placement

STATION_KEYS = list(STATIONS)
SCENARIO_KEYS = list(SCENARIOS)
TOLERANCE = DEFAULT_TOLERANCE_MM
# Every stroke the board can actually be viewed at -- before this the trailing means are
# still filling, and the page's slider does not go there.
FIRST_STROKE = WEAR_WINDOW - 1


@pytest.mark.parametrize("key", STATION_KEYS)
def test_the_wear_axis_orders_the_two_anchors(key):
    """0 is the pristine tool and 1 the worn one, or the percentage means nothing."""
    trained = load_artifacts(key)
    data, wear = trained.data, trained.wear
    scored = np.setdiff1d(np.flatnonzero(data.stroke_index < wear.window), wear.fitted_rows)

    low, high = (
        wear.position[scored[data.labels[scored] == level]].mean() for level in ENDPOINTS
    )
    assert low == pytest.approx(0.0, abs=0.1)
    assert high == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("key", STATION_KEYS)
def test_the_withheld_state_lands_between_the_anchors(key):
    """The single deployed configuration has to reproduce the sweep's result, not just
    the sweep's average over configurations -- the board shows the former."""
    trained = load_artifacts(key)
    assert 0.1 < centre_placement(trained.data, trained.wear) < 0.9


@pytest.mark.parametrize("key", STATION_KEYS)
def test_the_wear_axis_never_scores_a_stroke_it_was_fitted_on(key):
    """Fitted strokes are an in-sample fit; no scenario may put one on screen."""
    fitted = set(load_artifacts(key).wear.fitted_rows.tolist())
    for scenario_key in SCENARIO_KEYS:
        assert not fitted & set(load_run(scenario_key).rows.tolist())


@pytest.mark.parametrize("scenario_key", SCENARIO_KEYS)
def test_a_scenario_is_built_from_real_strokes_of_real_runs(scenario_key):
    run = load_run(scenario_key)
    assert len(run.rows) == N_STROKES

    # One row, two stations: the pair has to be the same physical stroke.
    for key in STATIONS:
        data = load_artifacts(key).data
        own = data.labels[run.rows]
        assert np.array_equal(own, data.other_labels[run.rows])
        assert np.array_equal(own, run.wear_level)


@pytest.mark.parametrize("scenario_key", SCENARIO_KEYS)
def test_a_scenario_never_repeats_a_stroke(scenario_key):
    """Each level advances through its own run, so a repeat would mean a stalled pointer."""
    rows = load_run(scenario_key).rows
    assert len(set(rows.tolist())) == len(rows)


def test_the_stable_run_stays_good_the_whole_way():
    run = load_run("stable")
    for stroke in range(FIRST_STROKE, N_STROKES):
        state, _ = machine_state(run, stroke, TOLERANCE)
        assert state == GOOD, f"stroke {stroke} of a healthy press is {state}"


def test_the_degrading_run_starts_good_and_ends_stopped():
    run = load_run("degrading")
    assert machine_state(run, FIRST_STROKE, TOLERANCE)[0] == GOOD
    assert machine_state(run, N_STROKES - 1, TOLERANCE)[0] == CRITICAL


def test_the_degrading_run_visits_every_wear_level_in_order():
    """The scenario is meant to walk *through* the intermediate state, not skip it."""
    levels = load_run("degrading").wear_level
    first_seen = [int(np.flatnonzero(levels == level)[0]) for level in LEVELS]
    assert first_seen == sorted(first_seen)
    assert CENTRE in set(levels.tolist())


def test_the_machine_is_never_better_than_its_worst_signal():
    run = load_run("degrading")
    for stroke in range(FIRST_STROKE, N_STROKES, 7):
        state, signals = machine_state(run, stroke, TOLERANCE)
        assert RANK[state] == max(RANK[signal.state] for signal in signals)


@pytest.mark.parametrize("key", STATION_KEYS)
def test_one_odd_stroke_cannot_stop_the_press(key):
    """The state reads a majority, so it may only ever be a level the window contains."""
    run = load_run("degrading")
    for stroke in range(FIRST_STROKE, N_STROKES, 11):
        window = run.window(stroke, WEAR_WINDOW)
        called = run.proba[key][window].argmax(axis=1) + 1
        assert majority_level(run, key, stroke) in set(called.tolist())


@pytest.mark.parametrize("scenario_key", SCENARIO_KEYS)
def test_every_card_and_detail_window_renders(scenario_key):
    run = load_run(scenario_key)
    for stroke in (FIRST_STROKE, N_STROKES // 2, N_STROKES - 1):
        assert board(run, stroke, TOLERANCE) is not None
        for card in CARDS:
            title, body = detail(card, run, stroke, TOLERANCE)
            assert title and body is not None
