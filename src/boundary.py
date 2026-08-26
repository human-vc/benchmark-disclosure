"""What is identified at the availability boundary, when nothing is globally.

Eligibility is $\\mathbf{1}[t_i - \\tau_b > 0]$, a deterministic function of the
two dates. That is what breaks every global repair: equating needs the two
groups comparable, inverse-propensity weighting needs the probability of
observing a cell bounded away from zero, and a deterministic rule gives neither.

It is also, exactly, a sharp regression discontinuity. Identification survives
deterministic assignment locally, at the cutoff, under continuity, so the
question the panel can answer is whether standing jumps where availability
turns on.

Two estimators, following the two frameworks in the RD literature. The
continuity-based one fits local linear regressions either side of zero and
reads off the gap. The local-randomization one picks a window where
pre-determined covariates are balanced and treats assignment inside it as
as-if random, which suits a setting with few provider clusters and few cells
close to the cutoff.
"""

import numpy as np
import pandas as pd

from .config import INTERIM

COVARIATES = ("n_benchmarks_scored", "n_scaffolds", "scaffold_spread")
BANDWIDTHS = (60, 90, 180, 250)
BALANCE_ALPHA = 0.15
MIN_PER_SIDE = 10


def cells(panel):
    """Comparable cells, with maturity in days as the running variable."""
    from .percentiles import within_benchmark_percentile
    scored = within_benchmark_percentile(panel)
    frame = scored[scored["eligible"] | scored["placebo"]].dropna(
        subset=["percentile"]).copy()
    frame["maturity"] = (frame["Release date"]
                         - frame["benchmark_release_date"]).dt.days
    frame["treat"] = (frame["maturity"] > 0).astype(int)
    frame["n_benchmarks_scored"] = frame.groupby("release_id")["slug"].transform("nunique")
    return frame


def local_linear(frame, bandwidth, order=1, outcome="percentile"):
    """Continuity-based estimate: the gap at zero, triangular kernel.

    Standard errors cluster on provider, since a provider's releases are not
    independent draws.
    """
    band = frame[np.abs(frame["maturity"]) <= bandwidth].dropna(subset=[outcome])
    # scale the running variable to [-1, 1]: x squared in days overflows the
    # normal equations, and the jump at the cutoff is invariant to the rescaling
    x = band["maturity"].to_numpy(float) / bandwidth
    y = band[outcome].to_numpy(float)
    t = band["treat"].to_numpy(float)
    weight = np.maximum(0.0, 1.0 - np.abs(x))

    columns = [np.ones(len(x)), t, x, t * x]
    if order >= 2:
        columns += [x ** 2, t * x ** 2]
    design = np.column_stack(columns)
    # weight by broadcasting rather than an n by n diagonal: same fit, and it
    # avoids building a 1266 square matrix to hold 1266 numbers
    root = np.sqrt(weight)[:, None]
    beta = np.linalg.lstsq(root * design, root[:, 0] * y, rcond=None)[0]
    resid = y - design @ beta

    inverse = np.linalg.pinv(design.T @ (weight[:, None] * design))
    meat = np.zeros((design.shape[1], design.shape[1]))
    for org in band["primary_org"].unique():
        rows = (band["primary_org"] == org).to_numpy()
        score = design[rows].T @ (weight[rows] * resid[rows])
        meat += np.outer(score, score)
    groups = band["primary_org"].nunique()
    vcov = inverse @ meat @ inverse * (groups / max(groups - 1, 1))
    se = float(np.sqrt(vcov[1, 1]))
    return {
        "bandwidth": bandwidth, "order": order,
        "jump": float(beta[1]), "se": se,
        "t": float(beta[1] / se) if se > 0 else np.nan,
        "low": float(beta[1] - 1.96 * se), "high": float(beta[1] + 1.96 * se),
        "n": int(len(band)), "n_clusters": int(groups),
    }


def randomization_p(frame, column, draws=4999, seed=0):
    """Two-sided p for a difference in means, permuting treatment within the window."""
    sub = frame.dropna(subset=[column])
    y = sub[column].to_numpy(float)
    t = sub["treat"].to_numpy(bool)
    if t.all() or not t.any():
        return np.nan
    observed = abs(y[t].mean() - y[~t].mean())
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(draws):
        shuffled = rng.permutation(t)
        if abs(y[shuffled].mean() - y[~shuffled].mean()) >= observed:
            count += 1
    return (count + 1) / (draws + 1)


def select_window(frame, covariates=COVARIATES, alpha=BALANCE_ALPHA,
                  widths=range(20, 400, 20), draws=999):
    """The largest window whose covariates, and every window inside it, balance.

    Follows the data-driven procedure of Cattaneo, Frandsen and Titiunik: test
    each pre-determined covariate inside a candidate window, take the smallest
    p-value, and keep widening while that minimum stays above `alpha`. The
    threshold is deliberately looser than a conventional level, because here
    failing to reject is what licenses the window.
    """
    chosen, table = None, []
    for width in widths:
        band = frame[np.abs(frame["maturity"]) <= width]
        left = int((band["treat"] == 0).sum())
        right = int((band["treat"] == 1).sum())
        if min(left, right) < MIN_PER_SIDE:
            continue
        ps = {c: randomization_p(band, c, draws=draws) for c in covariates
              if c in band.columns}
        worst = min([p for p in ps.values() if not np.isnan(p)], default=np.nan)
        table.append({"width": width, "left": left, "right": right,
                      "min_p": worst, **{f"p_{c}": ps.get(c) for c in covariates}})
        if not np.isnan(worst) and worst > alpha:
            chosen = width
        else:
            break
    return chosen, pd.DataFrame(table)


def density_ratio(frame, width=30):
    """Cells just above the cutoff over cells just below, a manipulation check."""
    below = int(((frame["maturity"] >= -width) & (frame["maturity"] < 0)).sum())
    above = int(((frame["maturity"] >= 0) & (frame["maturity"] < width)).sum())
    return above / below if below else np.nan


def main():
    panel = pd.read_csv(
        INTERIM / "panel.csv",
        parse_dates=["Release date", "benchmark_release_date"],
    )
    frame = cells(panel)
    treated, control = frame[frame["treat"] == 1], frame[frame["treat"] == 0]

    print("the availability boundary as a sharp discontinuity")
    print(f"  running variable is benchmark maturity in days, cutoff at zero")
    print(f"  eligible {len(treated)} cells, mean maturity {treated['maturity'].mean():.0f} days")
    print(f"  postdating {len(control)} cells, mean maturity {control['maturity'].mean():.0f} days")
    print(f"  the two group means sit {treated['maturity'].mean() - control['maturity'].mean():.0f} "
          f"days apart, so the global contrast compares cells far from the cutoff")
    print(f"  density just above over just below, 30 day bins: {density_ratio(frame):.2f}")

    print("\ncontinuity framework, local linear, provider-clustered")
    print("   bandwidth  order      jump      se        95% interval        n")
    rows = []
    for h in BANDWIDTHS:
        for order in (1, 2):
            r = local_linear(frame, h, order)
            rows.append(r)
            print(f"   {h:>6}d      {order}    {r['jump']:+7.2f}  {r['se']:6.2f}   "
                  f"[{r['low']:+7.2f}, {r['high']:+7.2f}]  {r['n']:>5}")
    table = pd.DataFrame(rows)
    print(f"\n   negative in {int((table['jump'] < 0).sum())} of {len(table)} specifications, "
          f"every interval covers zero, and the largest upper bound is "
          f"{table['high'].max():+.2f}")

    print("\nlocal randomization framework")
    width, balance = select_window(frame)
    if width is None:
        print("   no window balances at the chosen level")
    else:
        band = frame[np.abs(frame["maturity"]) <= width]
        p = randomization_p(band, "percentile", draws=4999)
        diff = (band.loc[band["treat"] == 1, "percentile"].mean()
                - band.loc[band["treat"] == 0, "percentile"].mean())
        print(f"   window +/-{width} days, chosen on covariate balance at {BALANCE_ALPHA}")
        print(f"   {int((band['treat'] == 0).sum())} postdating and "
              f"{int((band['treat'] == 1).sum())} eligible cells inside it")
        print(f"   difference in mean standing {diff:+.2f}, randomization p = {p:.3f}")
    print("\n   balance across candidate windows:")
    for row in balance.itertuples():
        print(f"     +/-{row.width:>3}d  left {row.left:>4}  right {row.right:>4}  "
              f"min p {row.min_p:.3f}")


if __name__ == "__main__":
    main()
