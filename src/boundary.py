
import numpy as np
import pandas as pd

from .config import INTERIM

COVARIATES = ("n_benchmarks_scored", "n_scaffolds", "scaffold_spread")
BANDWIDTHS = (60, 90, 180, 250)
BALANCE_ALPHA = 0.15
MIN_PER_SIDE = 10

def cells(panel):
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
    band = frame[np.abs(frame["maturity"]) <= bandwidth].dropna(subset=[outcome])
    x = band["maturity"].to_numpy(float) / bandwidth
    y = band[outcome].to_numpy(float)
    t = band["treat"].to_numpy(float)
    weight = np.maximum(0.0, 1.0 - np.abs(x))

    columns = [np.ones(len(x)), t, x, t * x]
    if order >= 2:
        columns += [x ** 2, t * x ** 2]
    design = np.column_stack(columns)
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
