import numpy as np
import pytest

from theatre_sched import SPECIALTIES, WorldConfig, make_world, touch_minutes
from theatre_sched.rolling import (
    RollingConfig,
    arrival_rates,
    make_leave,
    make_patients,
    simulate,
    summarise_run,
)

SHORT = dict(weeks_warmup=8, weeks_measured=20)


@pytest.fixture(scope="module")
def world():
    return make_world(WorldConfig(specialty="orthopaedics", n_surgeons=4), seed=11)


def test_touch_time_excludes_changeovers_and_overrun():
    d = np.array([[100.0], [100.0], [100.0]])  # three 100-minute cases, 20-minute changeovers
    assert touch_minutes(d, 20, 480)[0] == pytest.approx(300)
    # the third case starts at 240 and runs to 340; with a 300-minute session only 60 of it counts
    assert touch_minutes(d, 20, 300)[0] == pytest.approx(260)


def test_every_specialty_builds():
    for sp in SPECIALTIES:
        w = make_world(WorldConfig(specialty=sp, n_surgeons=3), seed=0)
        assert set(w.true_params.procedure) <= set(SPECIALTIES[sp].procedure)


def test_leave_is_ten_weeks_a_year_in_two_week_blocks():
    leave = make_leave(5, 104, RollingConfig(working_weeks=42), seed=1)
    assert (leave[:, :52].sum(axis=1) == 10).all() and (leave[:, 52:].sum(axis=1) == 10).all()
    starts = np.flatnonzero(np.diff(np.concatenate([[0], leave[0].astype(int)])) == 1)
    assert all(leave[0, s + 1] for s in starts)


def test_demand_scales_with_load(world):
    a = arrival_rates(world, RollingConfig(load=0.8))
    b = arrival_rates(world, RollingConfig(load=0.4))
    assert np.allclose(a, 2 * b)


def test_same_patients_whatever_the_scenario(world):
    p1 = make_patients(world, RollingConfig(policy="in_turn", extra_lists=0, **SHORT), seed=5)
    p2 = make_patients(world, RollingConfig(policy="ratio_fill", extra_lists=2, turnaround=5, **SHORT), seed=5)
    assert np.array_equal(p1.added_day, p2.added_day) and np.array_equal(p1.true_minutes, p2.true_minutes)


@pytest.mark.parametrize("policy", ["in_turn", "ratio_fill", "touch_fill"])
def test_simulation_bookkeeping(world, policy):
    cfg = RollingConfig(policy=policy, extra_lists=0.5, **SHORT)
    out = simulate(world, cfg, seed=5)
    p = out["patients"]
    treated = p[p.treated_day >= 0]
    assert (treated.treated_day >= treated.added_day).all(), "operated on before being added"
    lists = out["lists"]
    assert (lists.operated + lists.cancelled == lists.booked).all()
    assert lists.touch_utilisation.between(0, 1).all()
    leave = make_leave(world.config.n_surgeons, cfg.weeks_warmup + cfg.weeks_measured, cfg, seed=5)
    assert not leave[lists.surgeon.to_numpy(), lists.week.to_numpy()].any(), "a list ran while its surgeon was on leave"
    r = summarise_run(out, cfg)
    for k in ["waiting_within_6wk", "treated_within_6wk", "cohort_within_6wk"]:
        assert 0 <= r[k] <= 1


def test_more_lists_do_not_make_waits_worse(world):
    pts = make_patients(world, RollingConfig(**SHORT), seed=5)
    res = [summarise_run(simulate(world, cfg := RollingConfig(extra_lists=e, **SHORT), 5, pts), cfg)["cohort_within_6wk"]
           for e in (0.0, 1.0, 2.0)]
    assert res[0] <= res[1] + 0.02 and res[1] <= res[2] + 0.02


def test_referrals_fix_demand_whatever_the_number_of_surgeons():
    from theatre_sched.rolling import implied_load
    for n in (3, 6, 9):
        w = make_world(WorldConfig(specialty="ent", n_surgeons=n), seed=4)
        assert arrival_rates(w, RollingConfig(referrals_per_week=30)).sum() == pytest.approx(30)
    w = make_world(WorldConfig(specialty="ent", n_surgeons=6), seed=4)
    assert implied_load(w, RollingConfig(load=0.85)) == pytest.approx(0.85)
    # the same referrals need a smaller share of theatre time when there are more surgeons
    w9 = make_world(WorldConfig(specialty="ent", n_surgeons=9), seed=4)
    assert implied_load(w9, RollingConfig(referrals_per_week=30)) < implied_load(w, RollingConfig(referrals_per_week=30))
