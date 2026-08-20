"""Small statistical helpers, kept dependency-light on purpose.

Ordinary least squares with heteroskedasticity-consistent errors and a
percentile bootstrap. Written out rather than pulled from statsmodels so the
repository's numbers depend on four packages instead of five, and so the
standard-error choice is visible in the source rather than a default.
"""

import numpy as np


def ols(y, X, names=None, cluster=None):
    """OLS with HC1 errors, or cluster-robust errors when cluster is given.

    Clustering matters here: a provider contributes many releases and their
    disclosure habits are not independent draws, so provider-clustered errors
    are the honest default for anything pooled across releases.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)

    if cluster is None:
        meat = (X * (resid**2)[:, None]).T @ X * n / max(n - k, 1)
    else:
        cluster = np.asarray(cluster)
        meat = np.zeros((k, k))
        groups = np.unique(cluster)
        for g in groups:
            m = cluster == g
            Xg, ug = X[m], resid[m]
            score = Xg.T @ ug
            meat += np.outer(score, score)
        g_count = len(groups)
        meat *= (g_count / max(g_count - 1, 1)) * ((n - 1) / max(n - k, 1))

    vcov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(vcov))
    with np.errstate(invalid="ignore", divide="ignore"):
        t = beta / se
    names = names or [f"x{i}" for i in range(k)]
    return {
        "names": names, "beta": beta, "se": se, "t": t,
        "n": n, "n_clusters": None if cluster is None else len(np.unique(cluster)),
    }


def print_ols(fit, title):
    print(f"\n{title}  (n={fit['n']}"
          + (f", {fit['n_clusters']} clusters)" if fit["n_clusters"] else ")"))
    for name, b, se, t in zip(fit["names"], fit["beta"], fit["se"], fit["t"]):
        stars = "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.64 else ""
        print(f"  {name:26} {b:+8.2f}  ({se:5.2f})  t={t:+6.2f} {stars}")


def bootstrap_mean(values, draws=10000, seed=0, cluster=None):
    """Percentile bootstrap of a mean. Resamples clusters, not rows, when a
    cluster label is supplied."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, (np.nan, np.nan)

    if cluster is None:
        idx = rng.integers(0, len(values), size=(draws, len(values)))
        means = values[idx].mean(axis=1)
    else:
        cluster = np.asarray(cluster)
        groups = [np.where(cluster == g)[0] for g in np.unique(cluster)]
        means = np.empty(draws)
        for i in range(draws):
            pick = rng.integers(0, len(groups), size=len(groups))
            means[i] = values[np.concatenate([groups[j] for j in pick])].mean()

    return values.mean(), (np.percentile(means, 2.5), np.percentile(means, 97.5))
