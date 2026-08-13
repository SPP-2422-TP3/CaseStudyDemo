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
from spp2422_demo.scenario import (
    ALIGNMENT_SCENARIO,
    N_STROKES,
    SCENARIOS,
    WEAR_SCENARIO,
    WEAR_WINDOW,
    load_run,
)
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

    low, high = (wear.position[scored[data.labels[scored] == level]].mean() for level in ENDPOINTS)
    assert low == pytest.approx(0.0, abs=0.1)
    assert high == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("key", STATION_KEYS)
def test_the_withheld_state_lands_between_the_anchors(key):
    """The single deployed configuration has to reproduce the sweep's result, not just
    the sweep's average over configurations -- the board shows the former."""
    trained = load_artifacts(key)
    assert 0.1 < centre_placement(trained.data, trained.wear) < 0.9


@pytest.mark.parametrize("scenario", SCENARIO_KEYS)
@pytest.mark.parametrize("key", STATION_KEYS)
def test_the_wear_axis_never_scores_a_stroke_it_was_fitted_on(key, scenario):
    """Fitted strokes are an in-sample fit; no scenario may put one on screen."""
    fitted = set(load_artifacts(key).wear.fitted_rows.tolist())
    assert not fitted & set(load_run(scenario).rows.tolist())


@pytest.mark.parametrize("scenario", SCENARIO_KEYS)
def test_the_run_is_built_from_real_strokes_of_real_runs(scenario):
    run = load_run(scenario)
    assert len(run.rows) == N_STROKES

    # One row, two stations: the row has to be the same physical stroke at both, carrying
    # each station's own scheduled level and the other station's as its run partner.
    for key, other_key in (STATION_KEYS, STATION_KEYS[::-1]):
        data = load_artifacts(key).data
        assert np.array_equal(data.labels[run.rows], run.wear_level[key])
        assert np.array_equal(data.other_labels[run.rows], run.wear_level[other_key])


@pytest.mark.parametrize("scenario", SCENARIO_KEYS)
def test_the_run_never_repeats_a_stroke(scenario):
    """Each run advances through its own strokes, so a repeat would mean a stalled pointer."""
    rows = load_run(scenario).rows
    assert len(set(rows.tolist())) == len(rows)


@pytest.mark.parametrize("scenario", SCENARIO_KEYS)
def test_a_run_starts_good_and_ends_stopped(scenario):
    """Both scenarios are worth watching end to end, whichever fault they carry."""
    run = load_run(scenario)
    assert machine_state(run, FIRST_STROKE, TOLERANCE)[0] == GOOD
    assert machine_state(run, N_STROKES - 1, TOLERANCE)[0] == CRITICAL


def test_the_wear_scenario_walks_the_tools_through_every_level_out_of_step():
    """One tool reaches the intermediate state well before the other, and only one ends
    critical -- a board that showed them moving together would prove nothing about
    telling the stations apart."""
    run = load_run(WEAR_SCENARIO.key)
    reached = {
        key: {level: int(np.flatnonzero(levels == level)[0]) for level in LEVELS if level in levels}
        for key, levels in run.wear_level.items()
    }
    for key, first_seen in reached.items():
        assert list(first_seen) == sorted(first_seen), f"{key} skips or reorders a level"
    assert CENTRE in reached["ironing"]
    assert CENTRE in reached["deep_drawing"]
    assert reached["ironing"][CENTRE] < reached["deep_drawing"][CENTRE]
    assert LEVELS[-1] in reached["ironing"]
    assert LEVELS[-1] not in reached["deep_drawing"]


def test_the_wear_scenario_leaves_the_strip_centred():
    """Its whole point is one fault at a time; a drifting strip would confound it."""
    run = load_run(WEAR_SCENARIO.key)
    assert len(set(run.alignment_true_mm.tolist())) == 1


def test_the_alignment_scenario_drifts_the_strip_past_fresh_tools():
    run = load_run(ALIGNMENT_SCENARIO.key)
    assert all(set(levels.tolist()) == {LEVELS[0]} for levels in run.wear_level.values())
    assert run.alignment_true_mm[0] == 0.0
    assert run.alignment_true_mm[-1] > TOLERANCE
    assert np.all(np.diff(run.alignment_true_mm) >= 0)


@pytest.mark.parametrize("scenario", SCENARIO_KEYS)
def test_the_machine_is_never_better_than_its_worst_signal(scenario):
    run = load_run(scenario)
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


@pytest.mark.parametrize("scenario", SCENARIO_KEYS)
def test_every_card_and_detail_window_renders(scenario):
    run = load_run(scenario)
    for stroke in (FIRST_STROKE, N_STROKES // 2, N_STROKES - 1):
        assert board(run, stroke, TOLERANCE) is not None
        for card in CARDS:
            title, body = detail(card, run, stroke, TOLERANCE, [])
            assert title and body is not None


def test_a_detail_window_renders_with_operator_reports():
    run = load_run()
    stroke = N_STROKES - 1
    reports = [report(run, stroke, FEEDBACK_STROKES, ISSUES[0][0], "left wall tore", TOLERANCE)]
    for card in CARDS:
        title, body = detail(card, run, stroke, TOLERANCE, reports)
        assert title and body is not None


def test_a_report_covers_the_window_the_operator_chose():
    run = load_run()
    stroke = N_STROKES - 1
    for window in WINDOW_CHOICES:
        item = report(run, stroke, window, ISSUES[0][0], "", TOLERANCE)
        assert item.end == stroke
        assert item.end - item.start + 1 == window
    assert set(item.called) == set(STATIONS)


def test_a_report_survives_the_store_round_trip():
    """The store is JSON, so the dataclass has to come back out unchanged."""
    item = report(
        load_run(), N_STROKES - 1, FEEDBACK_STROKES, ISSUES[1][0], "  cup off  ", TOLERANCE
    )
    assert item.note == "cup off"  # whitespace is the operator's, not the record's
    assert from_store(to_store([item])) == [item]


def test_a_report_names_the_defect_it_was_given():
    item = report(load_run(), N_STROKES - 1, FEEDBACK_STROKES, ISSUES[2][0], "", TOLERANCE)
    assert item.issue_label == dict(ISSUES)[ISSUES[2][0]]


@pytest.mark.parametrize("scenario", SCENARIO_KEYS)
def test_a_report_says_whether_the_monitor_had_already_raised_it(scenario):
    """Either fault counts as already raised -- a drifting strip is not a silent monitor."""
    run = load_run(scenario)
    early = report(run, FIRST_STROKE, FEEDBACK_STROKES, ISSUES[0][0], "", TOLERANCE)
    late = report(run, N_STROKES - 1, FEEDBACK_STROKES, ISSUES[0][0], "", TOLERANCE)
    assert "read every signal as normal" in early.disagreement()
    assert "already raised" in late.disagreement()
