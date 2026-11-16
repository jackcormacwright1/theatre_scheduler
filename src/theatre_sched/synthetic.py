"""A synthetic elective-surgery world: procedures, surgeons, past case durations,
a waiting list and a set of theatre sessions.

Everything here is invented. Procedure times, surgeon effects, priorities and
waits are illustrative values chosen to be plausible, not figures from any NHS
organisation.

Case durations follow a lognormal distribution: log(duration) is normally
distributed. That gives right-skewed times (a long tail of slow cases), which
is the usual modelling choice for surgical case times.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Illustrative elective general-surgery procedures.
#   median_min     median case time in minutes (patient into theatre to patient out)
#   log_sd         spread of case times on the log scale (0.3 ~ +/-30% typical variation)
#   referral_share relative frequency on the waiting list
PROCEDURES = pd.DataFrame(
    {
        "procedure": [
            "skin_lesion_excision",
            "haemorrhoidectomy",
            "umbilical_hernia_repair",
            "inguinal_hernia_repair",
            "lap_cholecystectomy",
            "incisional_hernia_repair",
        ],
        "median_min": [25.0, 35.0, 45.0, 60.0, 80.0, 110.0],
        "log_sd": [0.35, 0.30, 0.30, 0.30, 0.35, 0.40],
        "referral_share": [0.15, 0.10, 0.15, 0.30, 0.20, 0.10],
    }
)


def _procs(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["procedure", "median_min", "log_sd", "referral_share"])


# Other specialties, same columns, equally illustrative. They differ mainly in case mix:
# orthopaedics is dominated by two-hour joint replacements, ENT has many very short cases,
# gynaecology mixes short day cases with long laparoscopic operations.
SPECIALTIES = {
    "general_surgery": PROCEDURES,
    "orthopaedics": _procs([
        ("carpal_tunnel_release", 25, 0.30, 0.15),
        ("knee_arthroscopy", 45, 0.30, 0.15),
        ("shoulder_arthroscopy", 75, 0.30, 0.10),
        ("acl_reconstruction", 100, 0.30, 0.05),
        ("total_hip_replacement", 120, 0.25, 0.25),
        ("total_knee_replacement", 125, 0.25, 0.30),
    ]),
    "gynaecology": _procs([
        ("operative_hysteroscopy", 25, 0.35, 0.30),
        ("diagnostic_laparoscopy", 40, 0.35, 0.15),
        ("lap_ovarian_cystectomy", 80, 0.35, 0.15),
        ("vaginal_hysterectomy_and_repair", 100, 0.30, 0.15),
        ("lap_endometriosis_excision", 120, 0.45, 0.10),
        ("total_lap_hysterectomy", 150, 0.35, 0.15),
    ]),
    "ent": _procs([
        ("grommet_insertion", 15, 0.35, 0.20),
        ("microlaryngoscopy", 25, 0.35, 0.15),
        ("tonsillectomy", 40, 0.30, 0.25),
        ("septoplasty", 60, 0.30, 0.15),
        ("endoscopic_sinus_surgery", 90, 0.35, 0.15),
        ("thyroidectomy", 130, 0.30, 0.10),
    ]),
}

# Clinical priority categories (P2 = within a month, P3 = within three months,
# P4 = can wait longer). Targets and wait distributions are modelling assumptions.
PRIORITY_TARGET_DAYS = {"P2": 28, "P3": 90, "P4": 180}
PRIORITY_SHARE = {"P2": 0.10, "P3": 0.40, "P4": 0.50}
PRIORITY_MEAN_WAIT_DAYS = {"P2": 14, "P3": 60, "P4": 120}


@dataclass
class WorldConfig:
    specialty: str = "general_surgery"  # a key of SPECIALTIES
    n_surgeons: int = 5
    patients_per_surgeon: int = 60
    weeks: int = 4
    sessions_per_week: int = 1  # all-day lists
    session_minutes: float = 480.0
    turnaround_minutes: float = 15.0  # changeover between consecutive cases
    surgeon_log_sd: float = 0.15  # how much surgeons differ in speed (log scale)
    interaction_log_sd: float = 0.08  # surgeon-specific deviation per procedure
    history_median_cases: float = 15.0  # typical number of past cases per surgeon-procedure pair
    history_log_sd: float = 1.0  # spread of that number: some pairs have 1-3 cases, some 100+
    min_procedures_per_surgeon: int = 4


@dataclass
class World:
    config: WorldConfig
    procedures: pd.DataFrame
    true_params: pd.DataFrame  # surgeon, procedure, mu, sigma: the hidden truth
    history: pd.DataFrame  # surgeon, procedure, minutes: what the planner can see
    patients: pd.DataFrame  # one row per patient on the waiting list
    sessions: pd.DataFrame  # one row per theatre session
    seed: int = field(default=0)


def make_world(config: WorldConfig | None = None, seed: int = 0) -> World:
    """Generate one synthetic world. The same seed always gives the same world."""
    cfg = config or WorldConfig()
    # Separate random streams for each part of the world, so that changing one
    # setting (e.g. how much history there is) leaves everything else identical.
    r_params, r_hist, r_pat = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3))
    procs = SPECIALTIES[cfg.specialty].copy()
    surgeons = [f"S{k + 1}" for k in range(cfg.n_surgeons)]

    # True duration parameters for each surgeon-procedure pair:
    #   log(duration) ~ Normal(mu, sigma^2)
    #   mu = log(procedure median) + surgeon speed effect + small pair-specific deviation
    surgeon_effect = r_params.normal(0.0, cfg.surgeon_log_sd, cfg.n_surgeons)
    rows = []
    for k, surgeon in enumerate(surgeons):
        n_proc = r_params.integers(cfg.min_procedures_per_surgeon, len(procs) + 1)
        chosen = np.sort(r_params.choice(len(procs), size=n_proc, replace=False))
        for p in chosen:
            mu = (
                np.log(procs.at[p, "median_min"])
                + surgeon_effect[k]
                + r_params.normal(0.0, cfg.interaction_log_sd)
            )
            rows.append((surgeon, procs.at[p, "procedure"], mu, procs.at[p, "log_sd"]))
    true_params = pd.DataFrame(rows, columns=["surgeon", "procedure", "mu", "sigma"])

    # Past cases. The number per pair varies a lot, so some estimates rest on very little data.
    n_hist = np.maximum(
        1,
        np.round(
            np.exp(r_hist.normal(np.log(cfg.history_median_cases), cfg.history_log_sd, len(true_params)))
        ),
    ).astype(int)
    hist_frames = []
    for (surgeon, procedure, mu, sigma), n in zip(true_params.itertuples(index=False), n_hist):
        minutes = np.exp(r_hist.normal(mu, sigma, n))
        hist_frames.append(pd.DataFrame({"surgeon": surgeon, "procedure": procedure, "minutes": minutes}))
    history = pd.concat(hist_frames, ignore_index=True)

    # Waiting list. Each patient belongs to one consultant and needs one procedure.
    share = procs.set_index("procedure")["referral_share"]
    priorities = list(PRIORITY_SHARE)
    pri_p = np.array([PRIORITY_SHARE[p] for p in priorities])
    pat_rows = []
    for surgeon in surgeons:
        their_procs = true_params.loc[true_params.surgeon == surgeon, "procedure"].to_numpy()
        w = share.loc[their_procs].to_numpy()
        w = w / w.sum()
        for _ in range(cfg.patients_per_surgeon):
            proc = r_pat.choice(their_procs, p=w)
            pri = r_pat.choice(priorities, p=pri_p)
            waited = int(np.ceil(r_pat.exponential(PRIORITY_MEAN_WAIT_DAYS[pri])))
            pat_rows.append((surgeon, proc, pri, waited, PRIORITY_TARGET_DAYS[pri]))
    patients = pd.DataFrame(
        pat_rows, columns=["surgeon", "procedure", "priority", "days_waited", "target_days"]
    )
    patients.insert(0, "patient_id", [f"P{i:04d}" for i in range(len(patients))])

    # Theatre sessions: each surgeon has their own lists, spread across the weekdays.
    sess_rows = []
    for k, surgeon in enumerate(surgeons):
        for week in range(cfg.weeks):
            for j in range(cfg.sessions_per_week):
                weekday = (k + 2 * j) % 5  # Mon=0 ... Fri=4
                day = 7 * week + weekday
                sess_rows.append((f"{surgeon}-w{week + 1}-{j + 1}", surgeon, day, cfg.session_minutes))
    sessions = pd.DataFrame(sess_rows, columns=["session_id", "surgeon", "day", "minutes"])
    sessions = sessions.sort_values(["day", "surgeon"], ignore_index=True)

    return World(
        config=cfg,
        procedures=procs,
        true_params=true_params,
        history=history,
        patients=patients,
        sessions=sessions,
        seed=seed,
    )
