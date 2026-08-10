"""Checks on the status board: the wear axis, the assembled runs and the verdict."""

from __future__ import annotations

import numpy as np
import pytest

from spp2422_demo.artifacts import load_artifacts
from spp2422_demo.calibration import CENTRE, ENDPOINTS
from spp2422_demo.components.status_cards import CARDS, board, detail
from spp2422_demo.data import LEVELS, STATIONS
from spp2422_demo.feedback import (
    FEEDBACK_STROKES,
    ISSUES,
    WINDOW_CHOICES,
    from_store,
    report,
    to_store,
)
from spp2422_demo.health import (
    CRITICAL,
    DEFAULT_TOLERANCE_MM,
    GOOD,
    RANK,
    machine_state,
    majority_level,
)
from spp2422_demo.scenario import N_STROKES, WEAR_WINDOW, load_run
from spp2422_demo.wear_position import centre_placement

STATION_KEYS = list(STATIONS)
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
    assert not fitted & set(load_run().rows.tolist())


def test_the_run_is_built_from_real_strokes_of_real_runs():
    run = load_run()
    assert len(run.rows) == N_STROKES

    # One row, two stations: the pair has to be the same physical stroke.
    for key in STATIONS:
        data = load_artifacts(key).data
        own = data.labels[run.rows]
        assert np.array_equal(own, data.other_labels[run.rows])
        assert np.array_equal(own, run.wear_level)


def test_the_run_never_repeats_a_stroke():
    """Each level advances through its own run, so a repeat would mean a stalled pointer."""
    rows = load_run().rows
    assert len(set(rows.tolist())) == len(rows)


def test_the_run_starts_good_and_ends_stopped():
    run = load_run()
    assert machine_state(run, FIRST_STROKE, TOLERANCE)[0] == GOOD
    assert machine_state(run, N_STROKES - 1, TOLERANCE)[0] == CRITICAL


def test_the_run_visits_every_wear_level_in_order():
    """The run is meant to walk *through* the intermediate state, not skip it."""
    levels = load_run().wear_level
    first_seen = [int(np.flatnonzero(levels == level)[0]) for level in LEVELS]
    assert first_seen == sorted(first_seen)
    assert CENTRE in set(levels.tolist())


def test_the_machine_is_never_better_than_its_worst_signal():
    run = load_run()
    for stroke in range(FIRST_STROKE, N_STROKES, 7):
        state, signals = machine_state(run, stroke, TOLERANCE)
        assert RANK[state] == max(RANK[signal.state] for signal in signals)


@pytest.mark.parametrize("key", STATION_KEYS)
def test_one_odd_stroke_cannot_stop_the_press(key):
    """The state reads a majority, so it may only ever be a level the window contains."""
    run = load_run()
    for stroke in range(FIRST_STROKE, N_STROKES, 11):
        window = run.window(stroke, WEAR_WINDOW)
        called = run.proba[key][window].argmax(axis=1) + 1
        assert majority_level(run, key, stroke) in set(called.tolist())


def test_every_card_and_detail_window_renders():
    run = load_run()
    for stroke in (FIRST_STROKE, N_STROKES // 2, N_STROKES - 1):
        assert board(run, stroke, TOLERANCE) is not None
        for card in CARDS:
            title, body = detail(card, run, stroke, TOLERANCE, [])
            assert title and body is not None


def test_a_detail_window_renders_with_operator_reports():
    run = load_run()
    stroke = N_STROKES - 1
    reports = [report(run, stroke, FEEDBACK_STROKES, ISSUES[0][0], "left wall tore")]
    for card in CARDS:
        title, body = detail(card, run, stroke, TOLERANCE, reports)
        assert title and body is not None


def test_a_report_covers_the_window_the_operator_chose():
    run = load_run()
    stroke = N_STROKES - 1
    for window in WINDOW_CHOICES:
        item = report(run, stroke, window, ISSUES[0][0], "")
        assert item.end == stroke
        assert item.end - item.start + 1 == window
    assert set(item.called) == set(STATIONS)


def test_a_report_survives_the_store_round_trip():
    """The store is JSON, so the dataclass has to come back out unchanged."""
    item = report(load_run(), N_STROKES - 1, FEEDBACK_STROKES, ISSUES[1][0], "  cup off  ")
    assert item.note == "cup off"  # whitespace is the operator's, not the record's
    assert from_store(to_store([item])) == [item]


def test_a_report_names_the_defect_it_was_given():
    item = report(load_run(), N_STROKES - 1, FEEDBACK_STROKES, ISSUES[2][0], "")
    assert item.issue_label == dict(ISSUES)[ISSUES[2][0]]


def test_a_report_says_whether_the_monitor_had_already_raised_it():
    run = load_run()
    early = report(run, FIRST_STROKE, FEEDBACK_STROKES, ISSUES[0][0], "")
    late = report(run, N_STROKES - 1, FEEDBACK_STROKES, ISSUES[0][0], "")
    assert "read every signal as normal" in early.disagreement()
    assert "already raised" in late.disagreement()
