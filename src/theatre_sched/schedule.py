"""Booking rules: who goes on which theatre list.

Every patient gets a weight that says how much booking them is worth:

    urgency = days waited / target days for their clinical priority
    weight  = 1 + lam * urgency

The "1" rewards booking anyone at all (throughput); the urgency term rewards
booking the people who have waited longest relative to their target. ``lam``
sets the balance between the two.

Patients are split into two tiers:

* must-book: every P2 patient, and anyone already at or past their target
  (urgency >= 1). They are placed first, most urgent first.
* the rest: used to fill the remaining space. How they are ordered is where
  the methods differ.

Two methods:

* ``book_to_average``: the usual approach. Each case is assumed to take its
  surgeon's historical average time, and a list is filled until the planned
  total reaches ``fill_fraction`` of the session (1.0 = book right up to the
  end; 0.85 = leave a 15% buffer).

* ``chance_constrained``: each case's time is uncertain. A list is allowed only
  if the Monte Carlo estimate of P(list overruns) is at most ``alpha``. The
  search starts with a greedy fill and then improves it by swaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import list_total_minutes


def add_weights(patients: pd.DataFrame, lam: float = 1.0) -> pd.DataFrame:
    p = patients.reset_index(drop=True).copy()
    p["urgency"] = p["days_waited"] / p["target_days"]
    p["weight"] = 1.0 + lam * p["urgency"]
    p["must_book"] = (p["priority"] == "P2") | (p["urgency"] >= 1.0)
    return p


def _surgeon_sessions(sessions: pd.DataFrame, surgeon: str) -> list[tuple[str, float]]:
    s = sessions[sessions["surgeon"] == surgeon].sort_values("day")
    return list(zip(s["session_id"], s["minutes"]))


def _tiers(pats: pd.DataFrame, expected: np.ndarray, fill_order: str) -> tuple[list[int], list[int]]:
    """Positional indices of must-book patients (most urgent first) and the rest (in fill order)."""
    must = pats[pats["must_book"]].sort_values("urgency", ascending=False)
    rest = pats[~pats["must_book"]]
    if fill_order == "queue":
        # clinical priority first, then longest wait: the conventional booking queue
        rest = rest.sort_values(["priority", "days_waited"], ascending=[True, False])
    elif fill_order == "ratio":
        # most weight per expected minute of theatre time first
        ratio = rest["weight"] / expected[rest.index.to_numpy()]
        rest = rest.assign(_r=ratio).sort_values("_r", ascending=False)
    else:
        raise ValueError(f"unknown fill_order: {fill_order}")
    return must.index.tolist(), rest.index.tolist()


def book_to_average(
    patients: pd.DataFrame,
    sessions: pd.DataFrame,
    expected_minutes: np.ndarray,
    turnaround: float,
    fill_fraction: float = 1.0,
    fill_order: str = "queue",
) -> dict[str, list[int]]:
    """Fill lists using average case times, up to ``fill_fraction`` of each session.

    Each patient goes on the earliest list of their surgeon where they still fit.
    If they don't fit anywhere, the next patient in the queue is tried.
    """
    schedule: dict[str, list[int]] = {sid: [] for sid in sessions["session_id"]}
    planned = {sid: 0.0 for sid in schedule}
    for surgeon in sessions["surgeon"].unique():
        sess = _surgeon_sessions(sessions, surgeon)
        pats = patients[patients["surgeon"] == surgeon]
        must, rest = _tiers(pats, expected_minutes, fill_order)
        for i in must + rest:
            for sid, minutes in sess:
                add = expected_minutes[i] + (turnaround if schedule[sid] else 0.0)
                if planned[sid] + add <= fill_fraction * minutes:
                    schedule[sid].append(i)
                    planned[sid] += add
                    break
    return schedule


def chance_constrained(
    patients: pd.DataFrame,
    sessions: pd.DataFrame,
    plan_samples: np.ndarray,
    turnaround: float,
    alpha: float,
    fill_order: str = "ratio",
    max_passes: int = 50,
) -> dict[str, list[int]]:
    """Book lists so that each list's estimated P(overrun) is at most ``alpha``.

    ``plan_samples`` holds the planner's simulated case times (rows = patients,
    columns = Monte Carlo draws). The constraint is checked on these draws, so
    the planner is only as good as the duration model that produced them.

    Search, per surgeon:
      1. Insert: go through unbooked patients (must-book first, then the rest in
         ``fill_order``) and put each on the earliest list where the constraint
         still holds.
      2. Swap: for each unbooked patient, try replacing a booked patient on one
         of the surgeon's lists, if the swap keeps the constraint and either
         (a) brings in a must-book patient for an optional one, or
         (b) raises the total weight.
      Repeat 1-2 until nothing changes. Each accepted move books more must-book
      patients or raises the total weight, so the loop always ends.
    """
    expected = plan_samples.mean(axis=1)
    weight = patients["weight"].to_numpy()
    must_flag = patients["must_book"].to_numpy()
    schedule: dict[str, list[int]] = {sid: [] for sid in sessions["session_id"]}

    for surgeon in sessions["surgeon"].unique():
        sess = _surgeon_sessions(sessions, surgeon)
        limit = dict(sess)
        totals = {sid: list_total_minutes(plan_samples, [], turnaround) for sid, _ in sess}
        pats = patients[patients["surgeon"] == surgeon]
        must, rest = _tiers(pats, expected, fill_order)
        booked_in: dict[int, str] = {}

        def ok(sid: str, total: np.ndarray) -> bool:
            return float(np.mean(total > limit[sid])) <= alpha

        for _ in range(max_passes):
            changed = False

            # 1. insertion
            for i in must + rest:
                if i in booked_in:
                    continue
                for sid, _m in sess:
                    new = totals[sid] + plan_samples[i] + (turnaround if schedule[sid] else 0.0)
                    if ok(sid, new):
                        schedule[sid].append(i)
                        totals[sid] = new
                        booked_in[i] = sid
                        changed = True
                        break

            # 2. swaps (only optional patients can be swapped out)
            unbooked = [i for i in must + rest if i not in booked_in]
            unbooked.sort(key=lambda i: (not must_flag[i], -weight[i]))
            for u in unbooked:
                done = False
                for sid, _m in sess:
                    candidates = [
                        b for b in schedule[sid]
                        if not must_flag[b] and (must_flag[u] or weight[u] > weight[b])
                    ]
                    for b in sorted(candidates, key=lambda b: weight[b]):
                        new = totals[sid] - plan_samples[b] + plan_samples[u]
                        if ok(sid, new):
                            schedule[sid][schedule[sid].index(b)] = u
                            totals[sid] = new
                            del booked_in[b]
                            booked_in[u] = sid
                            changed = done = True
                            break
                    if done:
                        break

            if not changed:
                break

    return schedule
