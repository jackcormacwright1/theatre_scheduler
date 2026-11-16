"""Monte Carlo evaluation of theatre lists.

Given a matrix of simulated case times (one row per patient, one column per
draw), the total time a list takes in draw s is

    total_s = sum of the booked patients' case times in draw s
              + turnaround * (number of cases - 1)

Averaging over the draws estimates how the list behaves:

    P(overrun)        = share of draws where total_s > session length
    expected overtime = average of max(total_s - session length, 0)
    utilisation       = average of min(total_s, session length) / session length
                        (counts changeover time as used)
    touch utilisation = average patient-in-theatre minutes inside the session / session length
                        (changeovers and time after the session end don't count: the way the
                        national 85% theatre utilisation measure is defined)

Every candidate list is scored on the same pre-drawn matrix ("common random
numbers"), so differences between two candidate lists reflect the lists, not
fresh simulation noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def list_total_minutes(samples: np.ndarray, members: list[int], turnaround: float) -> np.ndarray:
    """Total minutes the list takes in each Monte Carlo draw."""
    if len(members) == 0:
        return np.zeros(samples.shape[1])
    return samples[members].sum(axis=0) + turnaround * (len(members) - 1)


def touch_minutes(durations: np.ndarray, turnaround: float, session_minutes: float) -> np.ndarray:
    """Patient-in-theatre minutes that fall inside the session, per Monte Carlo draw.

    ``durations`` has one row per case, in the order the list runs, and one column per draw.
    Cases run back to back with ``turnaround`` minutes between them, starting at minute 0.
    """
    if durations.shape[0] == 0:
        return np.zeros(durations.shape[1])
    starts = np.vstack([np.zeros((1, durations.shape[1])), np.cumsum(durations[:-1] + turnaround, axis=0)])
    ends = starts + durations
    return np.clip(np.minimum(ends, session_minutes) - starts, 0.0, None).sum(axis=0)


def session_metrics(total: np.ndarray, session_minutes: float) -> dict:
    over = total - session_minutes
    p = float(np.mean(over > 0))
    return {
        "p_overrun": p,
        "p_overrun_se": float(np.sqrt(p * (1 - p) / len(total))),  # Monte Carlo standard error
        "expected_overtime": float(np.mean(np.maximum(over, 0.0))),
        "utilisation": float(np.mean(np.minimum(total, session_minutes)) / session_minutes),
    }


def evaluate_schedule(
    schedule: dict[str, list[int]],
    sessions: pd.DataFrame,
    samples: np.ndarray,
    turnaround: float,
) -> pd.DataFrame:
    """One row of metrics per session, scored on ``samples``."""
    rows = []
    for s in sessions.itertuples(index=False):
        members = schedule.get(s.session_id, [])
        total = list_total_minutes(samples, members, turnaround)
        touch = touch_minutes(samples[members], turnaround, s.minutes)
        rows.append({"session_id": s.session_id, "surgeon": s.surgeon, "n_cases": len(members),
                     **session_metrics(total, s.minutes),
                     "touch_utilisation": float(np.mean(touch) / s.minutes)})
    return pd.DataFrame(rows)


def summarise(
    schedule: dict[str, list[int]],
    per_session: pd.DataFrame,
    patients: pd.DataFrame,
    long_case_procedures: set[str],
) -> dict:
    """Headline numbers for a whole schedule."""
    booked = [i for members in schedule.values() for i in members]
    b = patients.iloc[booked]
    must = patients["must_book"].to_numpy()
    booked_mask = np.zeros(len(patients), dtype=bool)
    booked_mask[booked] = True
    return {
        "patients_booked": len(booked),
        "weight_booked": float(b["weight"].sum()),
        "must_book_unbooked": int(np.sum(must & ~booked_mask)),
        "overrun_rate": float(per_session["p_overrun"].mean()),  # expected share of lists that overrun
        "overtime_per_list": float(per_session["expected_overtime"].mean()),
        "utilisation": float(per_session["utilisation"].mean()),
        "touch_utilisation": float(per_session["touch_utilisation"].mean()),
        "long_case_share": float(b["procedure"].isin(long_case_procedures).mean()) if len(b) else 0.0,
    }
