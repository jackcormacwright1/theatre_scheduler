"""Case-duration models.

A duration model turns a set of patients into a matrix of simulated case times:
one row per patient, one column per Monte Carlo draw. Everything downstream
(evaluating a list, searching for a schedule) works on that matrix.

Two models are provided:

* ``LognormalPairModel`` with the TRUE parameters from the synthetic world.
  This is "reality": schedules are scored against it, and an "oracle" planner
  can use it to show what perfect knowledge of durations would buy.

* ``fit_plugin`` estimates the same lognormal form from the historical cases,
  one surgeon-procedure pair at a time, and then treats those estimates as if
  they were exact. That is the "plug-in" approach. It ignores the fact that an
  estimate based on three past cases is far less certain than one based on a
  hundred, which is the weakness the next version (a hierarchical model) is
  meant to fix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class LognormalPairModel:
    """Case duration ~ LogNormal(mu, sigma) with fixed parameters per surgeon-procedure pair.

    In minutes: duration = exp(mu + sigma * Z) with Z ~ Normal(0, 1).
    The median is exp(mu); the mean is exp(mu + sigma^2 / 2), a little above the median.
    """

    def __init__(self, params: pd.DataFrame):
        missing = {"surgeon", "procedure", "mu", "sigma"} - set(params.columns)
        if missing:
            raise ValueError(f"params is missing columns: {sorted(missing)}")
        self.params = params.set_index(["surgeon", "procedure"])[["mu", "sigma"]]

    def _lookup(self, patients: pd.DataFrame) -> pd.DataFrame:
        idx = pd.MultiIndex.from_frame(patients[["surgeon", "procedure"]])
        p = self.params.reindex(idx)
        if p.isna().any().any():
            bad = p[p.isna().any(axis=1)].index.unique().tolist()
            raise KeyError(f"no duration parameters for: {bad[:5]}")
        return p

    def sample(self, patients: pd.DataFrame, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Return an array of shape (n_patients, n_samples) of case times in minutes."""
        p = self._lookup(patients)
        mu = p["mu"].to_numpy()[:, None]
        sigma = p["sigma"].to_numpy()[:, None]
        z = rng.standard_normal((len(patients), n_samples))
        return np.exp(mu + sigma * z)

    def mean_minutes(self, patients: pd.DataFrame) -> np.ndarray:
        p = self._lookup(patients)
        return np.exp(p["mu"].to_numpy() + 0.5 * p["sigma"].to_numpy() ** 2)


@dataclass
class PluginFit:
    model: LognormalPairModel
    table: pd.DataFrame  # per pair: n, mean_minutes, mu, sigma, sigma_source

    def historical_mean(self, patients: pd.DataFrame) -> np.ndarray:
        """The simple 'average time it takes that surgeon' for each patient's procedure."""
        t = self.table.set_index(["surgeon", "procedure"])["mean_minutes"]
        return t.reindex(pd.MultiIndex.from_frame(patients[["surgeon", "procedure"]])).to_numpy()


def fit_plugin(history: pd.DataFrame, min_cases_for_own_sd: int = 3) -> PluginFit:
    """Estimate lognormal parameters for each surgeon-procedure pair from past cases.

    mu    = mean of log(minutes) for that pair
    sigma = standard deviation of log(minutes) for that pair, if it has at least
            ``min_cases_for_own_sd`` cases; otherwise the procedure's pooled spread
            (a standard deviation cannot be estimated from one case, and is very
            unstable from two).
    """
    h = history.assign(logd=np.log(history["minutes"]))
    g = h.groupby(["surgeon", "procedure"])
    table = g.agg(
        n=("minutes", "size"),
        mean_minutes=("minutes", "mean"),
        mu=("logd", "mean"),
        own_sd=("logd", lambda x: x.std(ddof=1) if len(x) > 1 else np.nan),
    ).reset_index()

    # Pooled within-pair spread for each procedure: sqrt( sum((n-1) * sd^2) / sum(n-1) ),
    # using only pairs with at least two cases.
    usable = table[table["n"] >= 2]
    num = (usable["n"] - 1) * usable["own_sd"] ** 2
    den = usable["n"] - 1
    pooled_by_proc = np.sqrt(num.groupby(usable["procedure"]).sum() / den.groupby(usable["procedure"]).sum())
    overall = np.sqrt(num.sum() / den.sum()) if den.sum() > 0 else 0.3
    pooled = table["procedure"].map(pooled_by_proc).fillna(overall)

    use_own = table["n"] >= min_cases_for_own_sd
    table["sigma"] = np.where(use_own, table["own_sd"], pooled)
    table["sigma_source"] = np.where(use_own, "own", "pooled")
    table = table.drop(columns="own_sd")

    model = LognormalPairModel(table[["surgeon", "procedure", "mu", "sigma"]])
    return PluginFit(model=model, table=table)
