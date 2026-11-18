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


# --------------------------------------------------------------------------------------------
# Hierarchical (partial pooling) model
# --------------------------------------------------------------------------------------------

def _pooled_sigma(table: pd.DataFrame) -> tuple[pd.Series, float]:
    """Within-pair spread of log(minutes) for each procedure, pooled over pairs with 2+ cases."""
    usable = table[table["n"] >= 2]
    num = (usable["n"] - 1) * usable["own_var"]
    den = usable["n"] - 1
    by_proc = np.sqrt(num.groupby(usable["procedure"]).sum() / den.groupby(usable["procedure"]).sum())
    overall = float(np.sqrt(num.sum() / den.sum())) if den.sum() > 0 else 0.3
    return by_proc, overall


class HierarchicalModel:
    """Case durations with partial pooling across surgeon-procedure pairs.

    On the log scale, case k of pair (s, p) takes   y = theta_sp + sigma_p * e,   e ~ Normal(0, 1),
    and the pair's true average theta_sp is itself uncertain:

        prior:      theta_sp ~ Normal(m_sp, tau^2 + se_m^2)     m_sp = alpha_p (+ beta_s)
        data:       ybar_sp  ~ Normal(theta_sp, sigma_p^2 / n_sp)
        posterior:  theta_sp ~ Normal(theta_hat, v)

    with   w = prior_var / (prior_var + sigma_p^2 / n)        (the weight on the pair's own data)
           theta_hat = w * ybar + (1 - w) * m
           v = w * sigma_p^2 / n                              (= 1 / (n / sigma_p^2 + 1 / prior_var))

    ``sample`` draws theta once per pair per Monte Carlo draw and shares it across all of that pair's
    patients, so if a surgeon is slower at a procedure than the data suggest, all their cases of it are
    slower together. Their errors then add up on a list instead of cancelling out.
    """

    def __init__(self, table: pd.DataFrame, alpha: pd.Series, beta: pd.Series, tau: float,
                 sigma_by_proc: pd.Series, prior_extra_var: dict, surgeon_effect: bool):
        self.table = table
        self.alpha, self.beta, self.tau = alpha, beta, tau
        self.sigma_by_proc = sigma_by_proc
        self.prior_extra_var = prior_extra_var  # (surgeon, procedure) -> se_m^2 for pairs with no history
        self.surgeon_effect = surgeon_effect
        self._by_pair = table.set_index(["surgeon", "procedure"])

    def _posterior(self, patients: pd.DataFrame) -> pd.DataFrame:
        """theta_hat, v and sigma for each patient's pair, using the prior for pairs with no history."""
        idx = pd.MultiIndex.from_frame(patients[["surgeon", "procedure"]])
        post = self._by_pair[["mu_hat", "post_var", "sigma"]].reindex(idx)
        unseen = post["mu_hat"].isna().to_numpy()
        if unseen.any():
            for i in np.flatnonzero(unseen):
                s, p = idx[i]
                if p not in self.alpha.index:
                    raise KeyError(f"no history at all for procedure {p!r}")
                m = self.alpha[p] + (self.beta.get(s, 0.0) if self.surgeon_effect else 0.0)
                post.iloc[i] = [m, self.tau ** 2 + self.prior_extra_var.get((s, p), 0.0), self.sigma_by_proc[p]]
        return post

    def sample(self, patients: pd.DataFrame, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Shape (n_patients, n_samples): one shared draw of each pair's average per Monte Carlo draw."""
        post = self._posterior(patients)
        codes, uniques = pd.factorize(pd.MultiIndex.from_frame(patients[["surgeon", "procedure"]]))
        first = np.array([np.flatnonzero(codes == c)[0] for c in range(len(uniques))])
        theta = (post["mu_hat"].to_numpy()[first][:, None]
                 + np.sqrt(post["post_var"].to_numpy()[first])[:, None] * rng.standard_normal((len(uniques), n_samples)))
        z = rng.standard_normal((len(patients), n_samples))
        return np.exp(theta[codes] + post["sigma"].to_numpy()[:, None] * z)

    def mean_minutes(self, patients: pd.DataFrame) -> np.ndarray:
        post = self._posterior(patients)
        return np.exp(post["mu_hat"].to_numpy() + 0.5 * (post["post_var"].to_numpy() + post["sigma"].to_numpy() ** 2))


def fit_hierarchical(history: pd.DataFrame, surgeon_effect: bool = True, n_iter: int = 20,
                     tau_floor: float = 0.01) -> HierarchicalModel:
    """Empirical-Bayes fit of the hierarchical model (method of moments, no MCMC).

    1. sigma_p: within-pair spread of log(minutes), pooled per procedure (as in ``fit_plugin``).
    2. alpha_p, beta_s: a weighted least-squares fit of the pair averages, ybar_sp ~ alpha_p + beta_s,
       with weights 1 / (tau^2 + sigma_p^2 / n). beta_s is the surgeon's speed relative to the average
       surgeon (the betas sum to zero). Without ``surgeon_effect``, ybar_sp ~ alpha_p only.
    3. tau^2: how far pair averages scatter around alpha_p + beta_s beyond what sampling noise explains,
       from the residuals: mean(r^2 * N / (N - k) - sigma_p^2 / n), floored at ``tau_floor^2``.
    Steps 2 and 3 are repeated until tau settles. Then each pair gets the posterior in the class docstring.
    The prior variance adds se_m^2, the uncertainty in alpha_p + beta_s itself.
    """
    h = history.assign(y=np.log(history["minutes"]))
    table = (h.groupby(["surgeon", "procedure"])["y"]
             .agg(n="size", ybar="mean", own_var=lambda x: x.var(ddof=1) if len(x) > 1 else np.nan)
             .reset_index())
    sigma_by_proc, overall = _pooled_sigma(table)
    procs = sorted(table["procedure"].unique())
    sigma_by_proc = sigma_by_proc.reindex(procs).fillna(overall)
    surgeons = sorted(table["surgeon"].unique(), key=lambda s: (len(s), s))

    def design(surgeon: pd.Series, procedure: pd.Series) -> np.ndarray:
        X = (procedure.to_numpy()[:, None] == np.array(procs)[None, :]).astype(float)
        if surgeon_effect and len(surgeons) > 1:
            # sum-to-zero coding: the last surgeon's effect is minus the sum of the others
            S = (surgeon.to_numpy()[:, None] == np.array(surgeons)[None, :]).astype(float)
            X = np.hstack([X, S[:, :-1] - S[:, [-1]]])
        return X

    X = design(table["surgeon"], table["procedure"])
    y = table["ybar"].to_numpy()
    samp_var = (table["procedure"].map(sigma_by_proc).to_numpy() ** 2) / table["n"].to_numpy()
    N, k = X.shape
    tau2 = 0.1 ** 2
    for _ in range(n_iter):
        wts = 1.0 / (tau2 + samp_var)
        XtW = X.T * wts
        cov = np.linalg.pinv(XtW @ X)
        coef = cov @ (XtW @ y)
        r = y - X @ coef
        new = max(tau_floor ** 2, float(np.mean(r ** 2 * N / max(N - k, 1) - samp_var)))
        if abs(new - tau2) < 1e-10:
            tau2 = new
            break
        tau2 = new

    alpha = pd.Series(coef[:len(procs)], index=procs)
    if surgeon_effect and len(surgeons) > 1:
        b = coef[len(procs):]
        beta = pd.Series(np.append(b, -b.sum()), index=surgeons)
    else:
        beta = pd.Series(0.0, index=surgeons)
    m = X @ coef
    se_m2 = np.einsum("ij,jk,ik->i", X, cov, X)
    prior_var = tau2 + se_m2
    w = prior_var / (prior_var + samp_var)
    table["prior_mean"] = m
    table["prior_var"] = prior_var
    table["weight"] = w
    table["mu_hat"] = w * y + (1 - w) * m
    table["post_var"] = w * samp_var
    table["sigma"] = table["procedure"].map(sigma_by_proc)
    table["mean_minutes"] = np.exp(table["mu_hat"] + 0.5 * (table["post_var"] + table["sigma"] ** 2))
    table = table.drop(columns="own_var")

    # Prior uncertainty for every surgeon-procedure combination, for pairs with no history.
    combos = pd.DataFrame([(s, p) for s in surgeons for p in procs], columns=["surgeon", "procedure"])
    Xc = design(combos["surgeon"], combos["procedure"])
    extra = dict(zip(map(tuple, combos.to_numpy()), np.einsum("ij,jk,ik->i", Xc, cov, Xc)))
    return HierarchicalModel(table, alpha, beta, float(np.sqrt(tau2)), sigma_by_proc, extra, surgeon_effect)
