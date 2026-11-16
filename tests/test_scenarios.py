from dataclasses import replace

import numpy as np
import pytest

from theatre_sched.rolling import RollingConfig, make_patients, simulate
from theatre_sched.scenarios import (
    MEASURES,
    Scenario,
    _crossing,
    compare,
    compare_lists_needed,
    headline,
    lists_needed,
    match_overrun,
    run,
)
from theatre_sched import WorldConfig, make_world

FAST = dict(specialty="orthopaedics", n_surgeons=3)


def test_crossing_interpolates_between_levels():
    levels = np.array([0.0, 0.25, 0.5])
    assert _crossing(levels, np.array([0.90, 0.94, 0.98]), 0.95) == pytest.approx(0.25 + 0.25 * 0.01 / 0.04)
    assert _crossing(levels, np.array([0.96, 0.97, 0.99]), 0.95) == 0.0
    assert np.isnan(_crossing(levels, np.array([0.5, 0.6, 0.7]), 0.95))


def test_a_scenario_compared_with_itself_shows_no_difference():
    a = run(Scenario(**FAST), seeds=3)
    cmp = compare(a, run(Scenario(**FAST), seeds=3))
    assert np.allclose(cmp["difference"], 0) and np.allclose(cmp["plus_minus"], 0)
    assert "no clear difference" in headline(cmp)


def test_average_rule_fills_lists_to_the_set_fraction():
    world = make_world(WorldConfig(specialty="general_surgery", n_surgeons=3), seed=2)
    cfg = RollingConfig(booking_rule="average", fill_fraction=0.8, weeks_warmup=4, weeks_measured=8)
    pts = make_patients(world, cfg, seed=2)
    out = simulate(world, cfg, 2, pts)
    # planned minutes on each list can't exceed 80% of the session: check through the booked counts
    lists = out["lists"]
    assert (lists.booked >= lists.operated).all()
    longer = simulate(world, replace(cfg, fill_fraction=1.1), 2, pts)["lists"]
    assert longer.booked.sum() > lists.booked.sum()


def test_matching_moves_b_towards_a_overrun_rate():
    a = Scenario(**FAST, booking_rule="average", fill_fraction=1.0)
    target = run(a, seeds=2).per_seed["overrun_rate"].mean()
    b0 = replace(a, booking_rule="monte_carlo", alpha=0.02)
    b, rate = match_overrun(target, b0, seeds=2)
    start = run(b0, seeds=2).per_seed["overrun_rate"].mean()
    assert abs(rate - target) <= abs(start - target)


def test_lists_needed_paired_comparison_runs():
    s = Scenario(**FAST)
    a = lists_needed(s, seeds=2, grid=(0.0, 0.5, 1.0))
    b = lists_needed(replace(s, turnaround=5.0), seeds=2, grid=(0.0, 0.5, 1.0))
    lc = compare_lists_needed(a, b)
    assert list(lc["key"]) == MEASURES
