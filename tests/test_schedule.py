import numpy as np
import pytest

from theatre_sched import (
    WorldConfig,
    add_weights,
    book_to_average,
    chance_constrained,
    fit_plugin,
    list_total_minutes,
    make_world,
)


@pytest.fixture(scope="module")
def setup():
    world = make_world(WorldConfig(n_surgeons=3, patients_per_surgeon=40), seed=7)
    patients = add_weights(world.patients, lam=1.0)
    fit = fit_plugin(world.history)
    plan = fit.model.sample(patients, 1000, np.random.default_rng(0))
    return world, patients, fit, plan


def _check_structure(schedule, world, patients):
    booked = [i for members in schedule.values() for i in members]
    assert len(booked) == len(set(booked)), "a patient was booked twice"
    surgeon_of = dict(zip(world.sessions.session_id, world.sessions.surgeon))
    for sid, members in schedule.items():
        assert (patients.loc[members, "surgeon"] == surgeon_of[sid]).all(), "patient on another surgeon's list"


@pytest.mark.parametrize("alpha", [0.02, 0.1, 0.3])
def test_chance_constraint_holds_on_planning_samples(setup, alpha):
    world, patients, _fit, plan = setup
    turnaround = world.config.turnaround_minutes
    sched = chance_constrained(patients, world.sessions, plan, turnaround, alpha)
    _check_structure(sched, world, patients)
    minutes = dict(zip(world.sessions.session_id, world.sessions.minutes))
    for sid, members in sched.items():
        total = list_total_minutes(plan, members, turnaround)
        assert np.mean(total > minutes[sid]) <= alpha + 1e-12


@pytest.mark.parametrize("fill", [0.8, 1.0])
def test_book_to_average_respects_buffer(setup, fill):
    world, patients, fit, _plan = setup
    turnaround = world.config.turnaround_minutes
    expected = fit.historical_mean(patients)
    sched = book_to_average(patients, world.sessions, expected, turnaround, fill)
    _check_structure(sched, world, patients)
    minutes = dict(zip(world.sessions.session_id, world.sessions.minutes))
    for sid, members in sched.items():
        planned = expected[members].sum() + turnaround * max(len(members) - 1, 0)
        assert planned <= fill * minutes[sid] + 1e-9


def test_looser_limit_never_books_fewer(setup):
    # A sanity check rather than a theorem: a greedy heuristic isn't guaranteed to be
    # monotone in alpha, but on this fixed world it should be.
    world, patients, _fit, plan = setup
    t = world.config.turnaround_minutes
    counts = [sum(len(m) for m in chance_constrained(patients, world.sessions, plan, t, a).values())
              for a in (0.02, 0.1, 0.3)]
    assert counts == sorted(counts)


def test_world_is_reproducible_and_history_size_changes_only_history():
    a = make_world(seed=3)
    b = make_world(seed=3)
    c = make_world(WorldConfig(history_median_cases=200), seed=3)
    assert a.patients.equals(b.patients) and a.history.equals(b.history)
    assert a.patients.equals(c.patients) and a.true_params.equals(c.true_params)
    assert len(c.history) > len(a.history)
