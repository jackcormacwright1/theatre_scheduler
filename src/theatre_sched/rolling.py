"""A year (and more) of elective theatre booking for one specialty, simulated week by week.

The single-window model in ``schedule.py`` can't say anything about waiting times,
because nobody new joins the list and nobody's wait grows. This module adds time.

Each simulated week:

1. **Booking** (start of the week). Everyone added to the list before this week can be
   booked. The booking policy puts them in an order, and each patient goes on the earliest
   of their surgeon's lists this week where they still fit under the booking rule (below).
   Extra lists, if any, go to the surgeons with the most patients about to pass the
   6-week mark.
2. **On the day.** Each list runs its cases longest-expected first, using the TRUE case
   times. Before each case starts, if the theatre team expects it to finish more than
   ``overrun_tolerance`` minutes after the session end, it is cancelled and the patient
   goes back on the waiting list. Their clock keeps running, as under RTT rules.
3. **New patients** are added through the week (Poisson arrivals).
4. **Measurement** at the end of the week.

Booking rules (how full a list may be):

* ``monte_carlo`` a list is allowed only if the planner's Monte Carlo estimate of P(list overruns)
                  is at most ``alpha``.
* ``average``     each case is assumed to take its surgeon's historical average time, and the list is
                  filled up to ``fill_fraction`` of the session. With ``in_turn`` this is plain
                  first-come-first-served booking.

Booking policies (who gets first claim on the space):

* ``in_turn``     urgent (P2) patients first, then everyone else in order of how long they've waited.
* ``ratio_fill``  patients who would pass their target before next week ("due") first, then the
                  most priority per expected theatre minute, which favours short cases.
* ``touch_fill``  "due" patients first, then the longest cases, which maximises patient-in-theatre
                  time per changeover and so the touch-time utilisation measure.

Three ways of measuring the 6-week target (from being added to the list to surgery):

* waiting:  of the patients still waiting at the end of each week, the share who have waited
            6 weeks or less. This is how the national 18-week RTT standard is measured.
* treated:  of the patients operated on, the share who had waited 6 weeks or less.
* cohort:   of the patients added to the list, the share who were operated on within 6 weeks,
            followed until the outcome is known. Treating patients out of turn can't flatter it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .durations import LognormalPairModel, fit_plugin
from .evaluate import list_total_minutes
from .synthetic import World

POLICIES = ("in_turn", "ratio_fill", "touch_fill")
BOOKING_RULES = ("monte_carlo", "average")


@dataclass
class RollingConfig:
    weeks_warmup: int = 26
    weeks_measured: int = 52
    lists_per_surgeon: int = 1  # baseline all-day lists per surgeon per week
    extra_lists: float = 0.0  # additional all-day lists per week for the specialty, e.g. 0.5 = one every other week
    session_minutes: float = 480.0
    turnaround: float = 15.0
    load: float = 0.85  # demand / baseline session time actually available (after leave), at a 15-minute changeover
    referrals_per_week: float | None = None  # if set, demand is this many patients a week for the whole
    # specialty, shared equally between the surgeons, and ``load`` is ignored
    working_weeks: int = 42  # weeks a year each surgeon operates; leave taken in two-week blocks
    alpha: float = 0.10  # booking limit on each list's estimated P(overrun)
    overrun_tolerance: float = 30.0  # minutes past session end the team will accept before cancelling
    target_days: int = 42  # the 6-week target
    urgent_target_days: int = 28  # P2 patients
    urgent_share: float = 0.10
    initial_backlog_weeks: float = 3.0  # starting list: this many weeks of arrivals, waits spread over 6 weeks
    plan_samples: int = 500
    policy: str = "in_turn"
    booking_rule: str = "monte_carlo"  # or "average"
    fill_fraction: float = 1.0  # for the "average" rule: fill lists to this share of the session


@dataclass
class Patients:
    surgeon: np.ndarray  # index into the world's surgeon list
    procedure: np.ndarray  # procedure name
    added_day: np.ndarray
    urgent: np.ndarray
    target: np.ndarray
    true_minutes: np.ndarray  # the case time if it goes ahead
    expected: np.ndarray  # the planner's expected case time (from its lognormal model)
    hist_mean: np.ndarray  # the surgeon's historical average time for the procedure
    plan: np.ndarray  # the planner's Monte Carlo draws, one row per patient


def _minutes_per_case(world: World, reference_turnaround: float) -> np.ndarray:
    """Expected theatre minutes per case for each surgeon (true durations, their case mix, plus changeover)."""
    truth = world.true_params.copy()
    truth["mean_min"] = np.exp(truth["mu"] + 0.5 * truth["sigma"] ** 2)
    share = world.procedures.set_index("procedure")["referral_share"]
    out = []
    for s in sorted(truth["surgeon"].unique(), key=lambda x: int(x[1:])):
        t = truth[truth.surgeon == s]
        w = share.loc[t["procedure"]].to_numpy()
        w = w / w.sum()
        out.append(float(np.sum(w * (t["mean_min"].to_numpy() + reference_turnaround))))
    return np.array(out)


def arrival_rates(world: World, cfg: RollingConfig, reference_turnaround: float = 15.0) -> np.ndarray:
    """Patients added to the list per week, for each surgeon.

    Two ways of setting demand:

    * ``referrals_per_week`` (used by the dashboard): a fixed number of patients a week for the
      whole specialty, shared equally between the surgeons. Adding surgeons adds capacity, not demand.
    * ``load`` (used by the Part 2 scripts): each surgeon's demand is ``load`` x the session time
      they have available after leave, so demand scales with the number of surgeons.

    Demand is in patients, fixed at the reference changeover time, so that a scenario with
    faster changeovers sees the same patients rather than more of them.
    """
    n = world.config.n_surgeons
    if cfg.referrals_per_week is not None:
        return np.full(n, cfg.referrals_per_week / n)
    available = cfg.lists_per_surgeon * cfg.session_minutes * cfg.working_weeks / 52
    return cfg.load * available / _minutes_per_case(world, reference_turnaround)


def implied_load(world: World, cfg: RollingConfig, reference_turnaround: float = 15.0) -> float:
    """Theatre minutes the demand needs, as a share of the baseline session time available after leave."""
    rates = arrival_rates(world, cfg, reference_turnaround)
    needed = float(np.sum(rates * _minutes_per_case(world, reference_turnaround)))
    available = world.config.n_surgeons * cfg.lists_per_surgeon * cfg.session_minutes * cfg.working_weeks / 52
    return needed / available


def make_patients(world: World, cfg: RollingConfig, seed: int) -> Patients:
    """Everyone who will ever be on the list in this run: the starting backlog plus all arrivals.

    Generated up front from a seed that doesn't depend on the policy or the number of lists,
    so every scenario sees exactly the same patients (common random numbers).
    """
    r_arr, r_true, r_plan = (np.random.default_rng(s) for s in np.random.SeedSequence([seed, 99]).spawn(3))
    surgeons = sorted(world.true_params["surgeon"].unique(), key=lambda x: int(x[1:]))
    share = world.procedures.set_index("procedure")["referral_share"]
    rates = arrival_rates(world, cfg)
    total_days = 7 * (cfg.weeks_warmup + cfg.weeks_measured)

    rows = []
    for k, s in enumerate(surgeons):
        procs = world.true_params.loc[world.true_params.surgeon == s, "procedure"].to_numpy()
        w = share.loc[procs].to_numpy()
        w = w / w.sum()
        n_init = r_arr.poisson(rates[k] * cfg.initial_backlog_weeks)
        n_new = r_arr.poisson(rates[k] * total_days / 7)
        days = np.concatenate([r_arr.uniform(-42, 0, n_init), r_arr.uniform(0, total_days, n_new)])
        for d in np.floor(days).astype(int):
            rows.append((k, r_arr.choice(procs, p=w), d, r_arr.random() < cfg.urgent_share))
    df = pd.DataFrame(rows, columns=["surgeon_idx", "procedure", "added_day", "urgent"])
    df["surgeon"] = [surgeons[k] for k in df["surgeon_idx"]]

    truth = LognormalPairModel(world.true_params)
    fit = fit_plugin(world.history)
    true_minutes = truth.sample(df, 1, r_true)[:, 0]
    plan = fit.model.sample(df, cfg.plan_samples, r_plan)
    return Patients(
        surgeon=df["surgeon_idx"].to_numpy(),
        procedure=df["procedure"].to_numpy(),
        added_day=df["added_day"].to_numpy(),
        urgent=df["urgent"].to_numpy(),
        target=np.where(df["urgent"], cfg.urgent_target_days, cfg.target_days),
        true_minutes=true_minutes,
        expected=plan.mean(axis=1),
        hist_mean=fit.historical_mean(df),
        plan=plan,
    )


def make_leave(n_surgeons: int, n_weeks: int, cfg: RollingConfig, seed: int) -> np.ndarray:
    """True where a surgeon is on leave that week. Each 52-week year: (52 - working_weeks) weeks,
    taken as non-overlapping two-week blocks at random points in the year."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
    blocks = (52 - cfg.working_weeks) // 2
    leave = np.zeros((n_surgeons, n_weeks), dtype=bool)
    for k in range(n_surgeons):
        for year_start in range(0, n_weeks, 52):
            for b in rng.choice(26, size=blocks, replace=False):
                w0 = year_start + 2 * b
                leave[k, w0:w0 + 2] = True
    return leave[:, :n_weeks]


def _order(idx: np.ndarray, pts: Patients, waited: np.ndarray, policy: str, expected: np.ndarray) -> np.ndarray:
    """Booking order for one surgeon's eligible patients (positions into ``idx``)."""
    w = waited[idx]
    if policy == "in_turn":
        return np.lexsort((-w, ~pts.urgent[idx]))  # urgent first, then longest wait
    due = w + 13 >= pts.target[idx]  # would pass target before they could be booked next week
    due_order = np.flatnonzero(due)[np.argsort(-w[due])]
    rest = np.flatnonzero(~due)
    if policy == "ratio_fill":
        weight = 1.0 + w[rest] / pts.target[idx][rest]
        rest_order = rest[np.argsort(-(weight / expected[idx][rest]))]
    elif policy == "touch_fill":
        rest_order = rest[np.argsort(-expected[idx][rest])]
    else:
        raise ValueError(f"unknown policy: {policy}")
    return np.concatenate([due_order, rest_order])


def simulate(world: World, cfg: RollingConfig, seed: int, pts: Patients | None = None) -> dict:
    """Run one specialty for warm-up + measured weeks. Returns weekly, list-level and patient-level records."""
    if cfg.booking_rule not in BOOKING_RULES:
        raise ValueError(f"unknown booking_rule: {cfg.booking_rule}")
    pts = pts if pts is not None else make_patients(world, cfg, seed)
    # What the booking team expects each case to take: the planner's model, or the plain historical average.
    expected = pts.hist_mean if cfg.booking_rule == "average" else pts.expected
    n_surg = int(pts.surgeon.max()) + 1
    n = len(pts.added_day)
    treated_day = np.full(n, -1)
    cancellations = np.zeros(n, dtype=int)
    L, T = cfg.session_minutes, cfg.turnaround
    n_weeks = cfg.weeks_warmup + cfg.weeks_measured
    leave = make_leave(n_surg, n_weeks, cfg, seed)
    weekly, lists_log = [], []

    for week in range(n_weeks):
        d0 = 7 * week
        waiting = (treated_day < 0) & (pts.added_day < d0)
        waited = d0 - pts.added_day

        # Sessions this week: baseline lists, plus extra lists for the surgeons with most "due" minutes.
        sessions = [(k, d0 + (k % 5)) for k in range(n_surg) for _ in range(cfg.lists_per_surgeon)
                    if not leave[k, week]]
        n_baseline = len(sessions)
        n_extra = int(np.floor(cfg.extra_lists * (week + 1) + 1e-9) - np.floor(cfg.extra_lists * week + 1e-9))
        if n_extra:
            due_minutes = np.array([
                expected[waiting & (pts.surgeon == k) & (waited + 13 >= pts.target)].sum()
                for k in range(n_surg)
            ])
            due_minutes[leave[:, week]] = -np.inf  # extra lists only go to surgeons who are working
            for _ in range(n_extra):
                k = int(np.argmax(due_minutes))
                sessions.append((k, d0 + 5))  # Saturday
                due_minutes[k] -= 0.8 * L

        # 1. Booking.
        booked: list[list[int]] = [[] for _ in sessions]
        totals = [np.zeros(cfg.plan_samples) for _ in sessions]
        planned = np.zeros(len(sessions))  # for the "average" rule
        left_unbooked = np.zeros(len(sessions), dtype=int)  # surgeon's eligible patients who didn't fit
        for k in range(n_surg):
            mine = [j for j, (sk, _) in enumerate(sessions) if sk == k]
            if not mine:
                continue
            mine.sort(key=lambda j: sessions[j][1])
            idx = np.flatnonzero(waiting & (pts.surgeon == k))
            for pos in _order(idx, pts, waited, cfg.policy, expected):
                i = idx[pos]
                for j in mine:
                    gap = T if booked[j] else 0.0
                    if cfg.booking_rule == "average":
                        if planned[j] + gap + expected[i] <= cfg.fill_fraction * L:
                            booked[j].append(i)
                            planned[j] += gap + expected[i]
                            break
                    else:
                        new = totals[j] + pts.plan[i] + gap
                        if np.mean(new > L) <= cfg.alpha:
                            booked[j].append(i)
                            totals[j] = new
                            break
            left = len(idx) - sum(len(booked[j]) for j in mine)
            for j in mine:
                left_unbooked[j] = left

        # 2. On the day, with true case times.
        for j, (k, day) in enumerate(sessions):
            order = sorted(booked[j], key=lambda i: -expected[i])
            t, touch, done, cancelled, end = 0.0, 0.0, 0, 0, 0.0
            for i in order:
                start = t + (T if done else 0.0)
                if start >= L or start + expected[i] > L + cfg.overrun_tolerance:
                    cancellations[i] += 1
                    cancelled += 1
                    continue
                end = start + pts.true_minutes[i]
                touch += max(0.0, min(end, L) - start)
                treated_day[i] = day
                t = end
                done += 1
            lists_log.append({"week": week, "surgeon": k, "day": day, "extra": j >= n_baseline,
                              "booked": len(order), "operated": done, "cancelled": cancelled,
                              "short_of_patients": bool(left_unbooked[j] == 0),
                              "touch_minutes": touch, "touch_utilisation": touch / L,
                              "overran": end > L, "overtime": max(0.0, end - L)})

        # 3-4. Measurement at the end of the week (arrivals during the week are already in added_day).
        d1 = d0 + 7
        still = (treated_day < 0) & (pts.added_day < d1)
        w_now = d1 - pts.added_day[still]
        just_treated = (treated_day >= d0) & (treated_day < d1)
        w_treated = treated_day[just_treated] - pts.added_day[just_treated]
        weekly.append({
            "week": week,
            "waiting": int(still.sum()),
            "waiting_within_6wk": float(np.mean(w_now <= cfg.target_days)) if still.any() else 1.0,
            "longest_wait_days": int(w_now.max()) if still.any() else 0,
            "treated": int(just_treated.sum()),
            "treated_within_6wk": float(np.mean(w_treated <= cfg.target_days)) if just_treated.any() else np.nan,
        })

    patients = pd.DataFrame({"surgeon": pts.surgeon, "procedure": pts.procedure, "added_day": pts.added_day,
                             "urgent": pts.urgent, "expected": pts.expected, "treated_day": treated_day,
                             "cancellations": cancellations})
    return {"weekly": pd.DataFrame(weekly), "lists": pd.DataFrame(lists_log), "patients": patients}


def summarise_run(out: dict, cfg: RollingConfig) -> dict:
    """Headline numbers over the measured weeks (warm-up excluded)."""
    w = out["weekly"][out["weekly"].week >= cfg.weeks_warmup]
    li = out["lists"][out["lists"].week >= cfg.weeks_warmup]
    p = out["patients"]
    start_day = 7 * cfg.weeks_warmup
    end_day = 7 * (cfg.weeks_warmup + cfg.weeks_measured)
    # Cohort: patients added during the measured period early enough that their 6-week outcome is known.
    cohort = p[(p.added_day >= start_day) & (p.added_day <= end_day - cfg.target_days)]
    within = (cohort.treated_day >= 0) & (cohort.treated_day - cohort.added_day <= cfg.target_days)
    treated = p[(p.treated_day >= start_day) & (p.treated_day < end_day)]
    waits = treated.treated_day - treated.added_day
    return {
        "waiting_within_6wk": float(w["waiting_within_6wk"].mean()),
        "waiting_within_6wk_worst_week": float(w["waiting_within_6wk"].min()),
        "treated_within_6wk": float(np.mean(waits <= cfg.target_days)) if len(waits) else np.nan,
        "cohort_within_6wk": float(within.mean()) if len(cohort) else np.nan,
        "touch_utilisation": float(li["touch_utilisation"].mean()),
        # lists where everyone eligible was booked, so the list could not be filled from its own queue
        "share_lists_short_of_patients": float(li["short_of_patients"].mean()),
        "touch_utilisation_full_lists": float(li.loc[~li["short_of_patients"], "touch_utilisation"].mean())
        if (~li["short_of_patients"]).any() else np.nan,
        "overrun_rate": float(li["overran"].mean()),
        "cancelled_per_100_booked": float(100 * li["cancelled"].sum() / max(li["booked"].sum(), 1)),
        "treated_per_week": float(w["treated"].mean()),
        "lists_run_per_week": float(len(li) / cfg.weeks_measured),
        "session_minutes_per_week": float(len(li) * cfg.session_minutes / cfg.weeks_measured),
        "touch_minutes_per_week": float(li["touch_minutes"].sum() / cfg.weeks_measured),
        "overtime_minutes_per_week": float(li["overtime"].sum() / cfg.weeks_measured),
        "median_wait_days": float(np.median(waits)) if len(waits) else np.nan,
        "p95_wait_days_treated": float(np.percentile(waits, 95)) if len(waits) else np.nan,
        "waiting_list_mean": float(w["waiting"].mean()),
        "waiting_list_growth": float(w["waiting"].iloc[-1] - w["waiting"].iloc[0]),
        "longest_wait_days_end": float(w["longest_wait_days"].iloc[-1]),
    }
