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


# --------------------------------------------------------------------------------------------
# Bayesian model on a grid: the hierarchical model with its spreads uncertain too
# --------------------------------------------------------------------------------------------

def _design(surgeon: np.ndarray, procedure: np.ndarray, procs: list[str], surgeons: list[str],
            surgeon_effect: bool) -> np.ndarray:
    """Columns: one per procedure (alpha_p); then, with ``surgeon_effect``, one per surgeon but the last,
    in sum-to-zero coding (the last surgeon's effect is minus the sum of the others)."""
    X = (np.asarray(procedure)[:, None] == np.array(procs)[None, :]).astype(float)
    if surgeon_effect and len(surgeons) > 1:
        S = (np.asarray(surgeon)[:, None] == np.array(surgeons)[None, :]).astype(float)
        X = np.hstack([X, S[:, :-1] - S[:, [-1]]])
    return X


def tau_log_posterior(tau_grid: np.ndarray, X: np.ndarray, ybar: np.ndarray, samp_var: np.ndarray) -> np.ndarray:
    """log p(tau | pair averages, sigma) on a grid, up to a constant, with a flat prior on tau and on the
    regression coefficients (Gelman et al., Bayesian Data Analysis, section 5.4).

    Each pair average is ybar_i ~ Normal(x_i' gamma, tau^2 + v_i), v_i = sigma_p^2 / n_i. Integrating out
    gamma leaves, for each tau, with w_i = 1 / (tau^2 + v_i) and gamma_hat the weighted least-squares fit:

        log p(tau | ...) = -1/2 log det(X' W X) - 1/2 sum log(tau^2 + v_i) - 1/2 sum w_i (ybar_i - x_i' gamma_hat)^2
    """
    W = 1.0 / (tau_grid[:, None] ** 2 + samp_var[None, :])  # (G, N)
    XtWX = np.einsum("gn,ni,nj->gij", W, X, X)
    XtWy = np.einsum("gn,ni,n->gi", W, X, ybar)
    gamma = np.linalg.solve(XtWX, XtWy[..., None])[..., 0]
    resid = ybar[None, :] - gamma @ X.T
    _, logdet = np.linalg.slogdet(XtWX)
    return -0.5 * logdet + 0.5 * np.log(W).sum(axis=1) - 0.5 * np.sum(W * resid ** 2, axis=1)


class BayesModel:
    """The hierarchical model, but with tau and each sigma_p uncertain instead of fixed at estimates.

    Fitted by drawing ``n_draws`` sets of parameters from their joint posterior, without MCMC:

    1. sigma_p^2 for each procedure, from its within-pair spread: with a prior p(sigma^2) ~ 1/sigma^2,
       sigma_p^2 | data ~ S_p / chi^2(nu_p), where nu_p = sum of (n - 1) over the procedure's pairs and
       S_p the sum of squared deviations of each case from its pair's average. (This uses only the spread
       within pairs, and ignores the little the pair averages say about sigma.)
    2. tau, from its posterior on a grid given those sigmas (``tau_log_posterior``), with a uniform prior
       on 0 to ``tau_max``: pick a grid cell with its probability, then a point uniformly within the cell.
    3. the regression coefficients gamma (procedure averages and surgeon speeds) given tau and sigma:
       Normal(gamma_hat, (X' W X)^-1).
    4. each pair's average theta_i given the rest: Normal(w ybar_i + (1 - w) x_i' gamma, w v_i), with
       w = tau^2 / (tau^2 + v_i). A pair with no history gets Normal(x' gamma, tau^2).

    ``sample`` then uses one parameter set per Monte Carlo draw, shared by all patients of a pair, so
    the planner's draws carry the uncertainty about sigma and tau as well as about each pair.
    """

    def __init__(self, procs, surgeons, surgeon_effect, pairs, X_pairs, sigma2, tau, gamma, theta,
                 tau_grid, tau_post):
        self.procs, self.surgeons, self.surgeon_effect = procs, surgeons, surgeon_effect
        self.pairs = pairs  # DataFrame of observed pairs: surgeon, procedure, n, ybar
        self._pair_index = {k: i for i, k in enumerate(zip(pairs["surgeon"], pairs["procedure"]))}
        self._proc_index = {p: i for i, p in enumerate(procs)}
        self.sigma2 = sigma2  # (K, n_procs)
        self.tau = tau  # (K,)
        self.gamma = gamma  # (K, n_coef)
        self.theta = theta  # (K, n_pairs)
        self.tau_grid, self.tau_post = tau_grid, tau_post  # marginal posterior of tau on the grid

    @property
    def n_draws(self) -> int:
        return len(self.tau)

    def _theta_for(self, patients: pd.DataFrame, cols: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """theta and sigma for each patient (rows) and draw (columns ``cols`` of the parameter bank)."""
        keys = list(zip(patients["surgeon"], patients["procedure"]))
        codes, uniques = pd.factorize(pd.Index(keys))
        theta_u = np.empty((len(uniques), len(cols)))
        for u, (s, p) in enumerate(uniques):
            if p not in self._proc_index:
                raise KeyError(f"no history at all for procedure {p!r}")
            i = self._pair_index.get((s, p))
            if i is not None:
                theta_u[u] = self.theta[cols, i]
            else:  # a pair with no history: the prior for its surgeon and procedure
                x = _design(np.array([s]), np.array([p]), self.procs, self.surgeons, self.surgeon_effect)[0]
                theta_u[u] = self.gamma[cols] @ x + self.tau[cols] * rng.standard_normal(len(cols))
        proc_idx = np.array([self._proc_index[p] for p in patients["procedure"]])
        sigma = np.sqrt(self.sigma2[cols][:, proc_idx]).T
        return theta_u[codes], sigma

    def sample(self, patients: pd.DataFrame, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Shape (n_patients, n_samples): each column uses one parameter set from the posterior."""
        cols = rng.permutation(self.n_draws)[:n_samples] if n_samples <= self.n_draws \
            else rng.integers(0, self.n_draws, n_samples)
        theta, sigma = self._theta_for(patients, cols, rng)
        return np.exp(theta + sigma * rng.standard_normal(theta.shape))

    def mean_minutes(self, patients: pd.DataFrame) -> np.ndarray:
        cols = np.arange(self.n_draws)
        theta, sigma = self._theta_for(patients, cols, np.random.default_rng(0))
        return np.exp(theta + 0.5 * sigma ** 2).mean(axis=1)

    def interval(self, q: float = 0.90) -> pd.DataFrame:
        """Posterior median and central ``q`` interval of each observed pair's average (log scale)."""
        a = (1 - q) / 2
        out = self.pairs[["surgeon", "procedure", "n"]].copy()
        out["median"] = np.median(self.theta, axis=0)
        out["low"] = np.quantile(self.theta, a, axis=0)
        out["high"] = np.quantile(self.theta, 1 - a, axis=0)
        return out


def fit_bayes(history: pd.DataFrame, surgeon_effect: bool = True, n_draws: int = 1000, tau_max: float = 1.0,
              n_grid: int = 200, seed: int = 0) -> BayesModel:
    """Fit ``BayesModel`` (see its docstring): ``n_draws`` joint posterior draws, computed exactly
    apart from the grid spacing and the step-1 simplification."""
    rng = np.random.default_rng(seed)
    h = history.assign(y=np.log(history["minutes"]))
    g = h.groupby(["surgeon", "procedure"])["y"]
    pairs = g.agg(n="size", ybar="mean").reset_index()
    pairs["ss"] = g.apply(lambda x: float(np.sum((x - x.mean()) ** 2))).to_numpy()
    procs = sorted(pairs["procedure"].unique())
    surgeons = sorted(pairs["surgeon"].unique(), key=lambda s: (len(s), s))
    X = _design(pairs["surgeon"].to_numpy(), pairs["procedure"].to_numpy(), procs, surgeons, surgeon_effect)
    ybar, n = pairs["ybar"].to_numpy(), pairs["n"].to_numpy()
    proc_idx = pairs["procedure"].map({p: i for i, p in enumerate(procs)}).to_numpy()

    # 1. sigma_p^2 ~ S_p / chi^2(nu_p); procedures with no pair of 2+ cases borrow the overall spread
    nu = np.bincount(proc_idx, weights=n - 1, minlength=len(procs))
    S = np.bincount(proc_idx, weights=pairs["ss"].to_numpy(), minlength=len(procs))
    nu_all, S_all = nu.sum(), S.sum()
    nu_use = np.where(nu > 0, nu, max(nu_all, 1.0))
    S_use = np.where(nu > 0, S, S_all if nu_all > 0 else 0.09)
    sigma2 = S_use[None, :] / rng.chisquare(nu_use[None, :], (n_draws, len(procs)))

    # 2-4. tau, gamma, theta, one draw at a time (each needs its own grid posterior)
    step = tau_max / n_grid
    grid = (np.arange(n_grid) + 0.5) * step
    tau = np.empty(n_draws)
    gamma = np.empty((n_draws, X.shape[1]))
    theta = np.empty((n_draws, len(pairs)))
    post_sum = np.zeros(n_grid)
    for j in range(n_draws):
        v = sigma2[j, proc_idx] / n
        lp = tau_log_posterior(grid, X, ybar, v)
        p = np.exp(lp - lp.max())
        p /= p.sum()
        post_sum += p
        t = grid[rng.choice(n_grid, p=p)] + (rng.random() - 0.5) * step
        tau[j] = t
        w_inv = 1.0 / (t ** 2 + v)
        prec = (X.T * w_inv) @ X
        cov = np.linalg.inv(prec)
        g_hat = cov @ ((X.T * w_inv) @ ybar)
        gamma[j] = rng.multivariate_normal(g_hat, cov)
        m = X @ gamma[j]
        w = t ** 2 / (t ** 2 + v)
        theta[j] = w * ybar + (1 - w) * m + np.sqrt(w * v) * rng.standard_normal(len(pairs))
    return BayesModel(procs, surgeons, surgeon_effect, pairs[["surgeon", "procedure", "n", "ybar"]],
                      X, sigma2, tau, gamma, theta, grid, post_sum / n_draws)
