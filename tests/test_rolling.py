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


# ---------------------------------------------------------------- starting from today's waiting list
from theatre_sched.rolling import starting_waits, weeks_to_target  # noqa: E402
from theatre_sched.scenarios import Scenario, recovery_curve, run  # noqa: E402


def test_starting_waits_match_the_profile_asked_for():
    w = starting_waits(20000, 42, 0.3, 182, np.random.default_rng(0))
    assert np.mean(w > 42) == pytest.approx(0.3, abs=0.01)
    assert w.max() <= 183 and w.min() >= 0
    beyond = w[w > 42]
    # triangular beyond the target: mean is a third of the way from target to the longest wait
    assert beyond.mean() == pytest.approx(42 + (182 - 42) / 3, abs=2)


def test_weeks_to_target_needs_the_share_to_hold():
    share = np.array([0.80, 0.96, 0.90, 0.95, 0.96, 0.97, 0.99, 0.98])
    assert weeks_to_target(share, 0.95) == 4.0  # weeks 4-7 hold; week 2 alone doesn't count
    assert np.isnan(weeks_to_target(np.full(10, 0.9), 0.95))


def test_a_run_from_todays_list_starts_where_asked_and_settled_runs_are_unchanged():
    scn = Scenario(referrals_per_week=24, start_waiting=200, start_over_target_pct=40, start_longest_weeks=30,
                   horizon_weeks=26)
    r = run(scn, 2).per_seed
    assert (r["waiting_start"] == 200).all()
    assert r["over_target_start"].between(60, 100).all()
    assert r["longest_wait_days_start"].le(7 * 30 + 1).all()
    settled = run(Scenario(referrals_per_week=24), 1).per_seed
    assert "weeks_to_target" not in settled.columns


def test_more_extra_lists_clear_a_backlog_sooner():
    scn = Scenario(referrals_per_week=24, start_waiting=300, start_over_target_pct=40, start_longest_weeks=40)
    rec = recovery_curve(scn, seeds=2, grid=(0.0, 1.0, 2.0))
    c = rec.curve.set_index("extra_lists")
    assert c.loc[2.0, "over_target_week_26"] < c.loc[0.0, "over_target_week_26"]
    by = rec.lists_to_meet_by(26)
    assert np.isnan(by) or 0 <= by <= 2.0


# ---------------------------------------------------------------- describing today's list in more detail
from theatre_sched.starting_list import CountsError, parse_counts, parse_weeks, summarise, to_scenario_bands  # noqa: E402


def test_parse_weeks_formats():
    assert parse_weeks("12-18") == (12, 18)
    assert parse_weeks("12 to 18 weeks") == (12, 18)
    assert parse_weeks("<6") == (0, 6)
    assert parse_weeks("7") == (7, 8)
    assert parse_weeks("52+") == (52, 78) and parse_weeks(">52") == (52, 78)
    with pytest.raises(CountsError):
        parse_weeks("soon")


def test_parse_counts_reads_pasted_spreadsheet_blocks():
    bands = parse_counts("Consultant\tWeeks\tPatients\nMr A\t0-6\t40\nMr A\t6-18\t20\nMs B\t0-6\t30\nMs B\t52+\t2")
    assert [b.patients for b in bands] == [40, 20, 30, 2] and bands[0].surgeon == "Mr A"
    s = summarise(bands, 6)
    assert s["patients"] == 92 and s["over_target_pct"] == pytest.approx(100 * 22 / 92)
    assert to_scenario_bands(bands, 3)[3] == (1, 52, 78, 2)
    with pytest.raises(CountsError, match="at least 2"):
        to_scenario_bands(bands, 1)
    with pytest.raises(CountsError, match="Line 2"):
        parse_counts("weeks,patients\n0-6,lots")
    # a band straddling the target counts in proportion
    assert summarise(parse_counts("4-8,100"), 6)["over_target_pct"] == pytest.approx(50)


def test_counts_and_per_surgeon_lists_start_where_asked():
    bands = to_scenario_bands(parse_counts("Mr A,0-6,40\nMr A,20-30,20\nMs B,0-6,10"), 4)
    r = run(Scenario(referrals_per_week=24, n_surgeons=4, start_waiting=70, start_bands=bands, horizon_weeks=26), 1)
    p = r.per_seed.iloc[0]
    assert p["waiting_start"] == 70 and p["over_target_start"] == 20
    assert 7 * 20 < p["longest_wait_days_start"] <= 7 * 30 + 1
    by = ((60, 50, 30), (10, 0, 7), (10, 0, 7), (10, 0, 7))
    q = run(Scenario(referrals_per_week=24, n_surgeons=4, start_waiting=90, start_by_surgeon=by, horizon_weeks=26), 1)
    assert q.per_seed.iloc[0]["waiting_start"] == 90
    assert 20 <= q.per_seed.iloc[0]["over_target_start"] <= 40  # about half of S1's 60


def test_a_backlog_on_one_surgeon_clears_more_slowly_than_the_same_backlog_shared():
    kw = dict(referrals_per_week=24, extra_lists=0.5, start_waiting=200, horizon_weeks=52)
    shared = run(Scenario(**kw, start_by_surgeon=((34, 35, 30),) * 2 + ((33, 35, 30),) * 4), 3).per_seed
    one = run(Scenario(**kw, start_by_surgeon=((100, 70, 30),) + ((20, 0, 7),) * 5), 3).per_seed
    assert one["over_target_week_13"].mean() > shared["over_target_week_13"].mean()
    assert one["weeks_to_treat_start_list"].mean() > shared["weeks_to_treat_start_list"].mean()


# ---------------------------------------------------------------- subspecialties and pooled lists
from theatre_sched.synthetic import SUBSPECIALTIES, surgeon_subspecialties  # noqa: E402


def test_subspecialty_case_mixes():
    w = make_world(WorldConfig(specialty="ent", n_surgeons=6, subspecialties=True), seed=3)
    interests = surgeon_subspecialties("ent", 6)
    assert interests[:3] == ["rhinology", "head and neck", "otology"]
    does = w.true_params.groupby("surgeon")["procedure"].apply(set)
    for k, s in enumerate(does.index):
        assert does[s] == set(SUBSPECIALTIES["ent"]["common"]) | set(SUBSPECIALTIES["ent"][interests[k]])


def test_pooled_lists_only_pass_patients_to_surgeons_who_do_the_procedure():
    w = make_world(WorldConfig(specialty="ent", n_surgeons=6, subspecialties=True), seed=3)
    cfg = RollingConfig(referrals_per_week=30, pooling="pooled", weeks_warmup=0, weeks_measured=26,
                        start_waiting=150, start_over_target_share=0.3, start_longest_days=180)
    pts = make_patients(w, cfg, seed=3)
    own_k = pts.surgeon
    assert np.allclose(pts.pool_true[own_k, np.arange(len(own_k))], pts.true_minutes)  # own surgeon: same case
    p = simulate(w, cfg, 3, pts)["patients"]
    done = p[p.treated_day >= 0]
    does = w.true_params.groupby("surgeon")["procedure"].apply(set)
    names = sorted(does.index, key=lambda s: int(s[1:]))
    assert all(proc in does[names[k]] for k, proc in zip(done.treated_by, done.procedure))
    # the otologists (S3, S6) do grommets and tonsils for anyone, but never sinus surgery or a thyroid
    oto = done[done.treated_by.isin([2, 5])]
    assert set(oto.procedure) <= {"grommet_insertion", "tonsillectomy"}
    assert (done.treated_by != done.surgeon).any()  # some patients did move


def test_pooling_within_subspecialty_clears_a_backlog_on_one_surgeon_sooner():
    kw = {"specialty": "general_surgery", "subspecialties": True, "referrals_per_week": 24, "start_waiting": 200,
          "start_by_surgeon": ((90, 60, 40),) + ((22, 10, 12),) * 5, "horizon_weeks": 52}
    own = run(Scenario(**kw), 3).per_seed
    pooled = run(Scenario(**kw, pooling="pooled"), 3).per_seed
    assert pooled["over_target_week_26"].mean() < own["over_target_week_26"].mean()
    assert pooled["weeks_to_treat_start_list"].mean() < own["weeks_to_treat_start_list"].mean()
    assert pooled["treated_per_year"].mean() >= own["treated_per_year"].mean()
