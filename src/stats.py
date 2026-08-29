
import numpy as np

MIN_CLUSTERS = 12

GRANULAR_CLUSTERS = 10

WEBB_CLUSTERS = 11

WEBB_SUPPORT = (-np.sqrt(1.5), -1.0, -np.sqrt(0.5),
                np.sqrt(0.5), 1.0, np.sqrt(1.5))

def sign_support(n_groups, weights="auto"):
    if weights == "auto":
        weights = "webb" if n_groups <= WEBB_CLUSTERS else "rademacher"
    if weights == "webb":
        return np.array(WEBB_SUPPORT), 6.0 ** n_groups, weights
    return np.array([-1.0, 1.0]), 2.0 ** n_groups, weights

def p_value_floor(n_groups, draws, weights):
    if weights == "webb":
        distinct = 6.0 ** n_groups
    else:
        distinct = 2.0 ** max(n_groups - 1, 0)
    return float(1.0 / (min(float(draws), distinct) + 1.0))

def _fit(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta

def _cluster_vcov(X, resid, cluster, XtX_inv):
    n, k = X.shape
    groups = np.unique(cluster)
    meat = np.zeros((k, k))
    for g in groups:
        rows = cluster == g
        score = X[rows].T @ resid[rows]
        meat += np.outer(score, score)
    g_count = len(groups)
    meat *= (g_count / max(g_count - 1, 1)) * ((n - 1) / max(n - k, 1))
    return XtX_inv @ meat @ XtX_inv

def ols(y, X, names=None, cluster=None, min_clusters=MIN_CLUSTERS):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    beta, resid = _fit(y, X)
    XtX_inv = np.linalg.pinv(X.T @ X)

    names = names or [f"x{i}" for i in range(k)]
    n_clusters = None if cluster is None else len(np.unique(np.asarray(cluster)))

    if n_clusters is not None and n_clusters < min_clusters:
        nan = np.full(k, np.nan)
        return {
            "names": names, "beta": beta, "se": nan, "t": nan,
            "n": n, "n_clusters": n_clusters,
            "inference": (f"withheld: {n_clusters} clusters is below the "
                          f"{min_clusters} needed for cluster-robust asymptotics; "
                          "use wild_cluster_bootstrap or randomization_test_mean"),
        }

    if n_clusters is None:
        if n - k <= 0:
            nan = np.full(k, np.nan)
            return {"names": names, "beta": beta, "se": nan, "t": nan,
                    "n": n, "n_clusters": None,
                    "inference": f"withheld: {n} observations for {k} parameters"}
        meat = (X * (resid**2)[:, None]).T @ X * n / (n - k)
        vcov = XtX_inv @ meat @ XtX_inv
    else:
        vcov = _cluster_vcov(X, resid, np.asarray(cluster), XtX_inv)

    se = np.sqrt(np.diag(vcov))
    with np.errstate(invalid="ignore", divide="ignore"):
        t = beta / se
    return {"names": names, "beta": beta, "se": se, "t": t,
            "n": n, "n_clusters": n_clusters, "inference": "cluster-robust"
            if n_clusters is not None else "HC1"}

def wild_cluster_bootstrap(y, X, cluster, index, draws=9999, seed=0,
                           weights="auto"):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    cluster = np.asarray(cluster)
    n, k = X.shape

    beta, resid = _fit(y, X)
    XtX_inv = np.linalg.pinv(X.T @ X)
    observed = beta[index] / np.sqrt(np.diag(_cluster_vcov(X, resid, cluster, XtX_inv))[index])

    keep = [j for j in range(k) if j != index]
    X_null = X[:, keep]
    beta_null, resid_null = _fit(y, X_null)
    fitted_null = X_null @ beta_null

    groups = np.unique(cluster)
    codes = np.searchsorted(groups, cluster)
    support, distinct, kind = sign_support(len(groups), weights)
    rng = np.random.default_rng(seed)

    extreme = 0
    for _ in range(draws):
        draw = rng.choice(support, size=len(groups))[codes]
        y_star = fitted_null + draw * resid_null
        beta_star, resid_star = _fit(y_star, X)
        se_star = np.sqrt(
            np.diag(_cluster_vcov(X, resid_star, cluster, XtX_inv))[index])
        if se_star > 0 and abs(beta_star[index] / se_star) >= abs(observed):
            extreme += 1

    return {
        "t_observed": float(observed),
        "p_value": (extreme + 1) / (draws + 1),
        "p_floor": p_value_floor(len(groups), draws, kind),
        "n_clusters": len(groups),
        "granular": len(groups) < GRANULAR_CLUSTERS,
        "weights": kind,
        "distinct_draws_available": float(distinct),
    }

def randomization_test_mean(values, cluster=None, draws=9999, seed=0):
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"mean": np.nan, "p_value": np.nan, "n": 0, "n_clusters": 0}

    if cluster is None:
        codes = np.arange(len(values))
    else:
        cluster = np.asarray(cluster)[: len(values)]
        codes = np.searchsorted(np.unique(cluster), cluster)
    n_groups = int(codes.max()) + 1

    observed = values.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(draws, n_groups))[:, codes]
    null = (signs * values).mean(axis=1)

    return {
        "mean": float(observed),
        "p_value": float(((np.abs(null) >= abs(observed)).sum() + 1) / (draws + 1)),
        "p_floor": p_value_floor(n_groups, draws, "rademacher"),
        "n": int(len(values)),
        "n_clusters": n_groups,
        "granular": n_groups < GRANULAR_CLUSTERS,
        "weights": "rademacher",
        "distinct_draws_available": 2.0 ** n_groups,
    }

def two_way_se(y, X, cluster_a, cluster_b):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    beta, resid = _fit(y, X)
    XtX_inv = np.linalg.pinv(X.T @ X)

    a = np.asarray(cluster_a)
    b = np.asarray(cluster_b)
    ab = np.array([f"{x}||{z}" for x, z in zip(a, b)])
    vcov = (_cluster_vcov(X, resid, a, XtX_inv)
            + _cluster_vcov(X, resid, b, XtX_inv)
            - _cluster_vcov(X, resid, ab, XtX_inv))
    diagonal = np.diag(vcov).copy()
    diagonal[diagonal < 0] = np.nan
    return {
        "beta": beta,
        "se": np.sqrt(diagonal),
        "n": n,
        "n_clusters_a": int(len(np.unique(a))),
        "n_clusters_b": int(len(np.unique(b))),
    }

def print_ols(fit, title):
    header = f"\n{title}  (n={fit['n']}"
    header += f", {fit['n_clusters']} clusters)" if fit["n_clusters"] else ")"
    print(header)

    withheld = fit.get("inference", "").startswith("withheld")
    for name, b, se, t in zip(fit["names"], fit["beta"], fit["se"], fit["t"]):
        if withheld or not np.isfinite(se):
            print(f"  {name:26} {b:+8.2f}  (se not reported)")
        else:
            stars = ("***" if abs(t) > 2.58 else "**" if abs(t) > 1.96
                     else "*" if abs(t) > 1.64 else "")
            print(f"  {name:26} {b:+8.2f}  ({se:5.2f})  t={t:+6.2f} {stars}")
    if withheld:
        print(f"  {fit['inference']}")

def bootstrap_mean(values, draws=10000, seed=0, cluster=None,
                   min_clusters=MIN_CLUSTERS):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, (np.nan, np.nan)

    if cluster is None:
        idx = rng.integers(0, len(values), size=(draws, len(values)))
        means = values[idx].mean(axis=1)
    else:
        cluster = np.asarray(cluster)[: len(values)]
        groups = [np.where(cluster == g)[0] for g in np.unique(cluster)]
        if len(groups) < min_clusters:
            return values.mean(), (np.nan, np.nan)
        means = np.empty(draws)
        for i in range(draws):
            pick = rng.integers(0, len(groups), size=len(groups))
            means[i] = values[np.concatenate([groups[j] for j in pick])].mean()

    return values.mean(), (np.percentile(means, 2.5), np.percentile(means, 97.5))
