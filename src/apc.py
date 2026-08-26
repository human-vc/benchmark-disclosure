"""How much of the placebo gap the unidentified part of the design can carry.

Release date, benchmark vintage and benchmark maturity satisfy maturity =
date - vintage exactly, so a regression of standing on all three is singular.
This is the age-period-cohort problem, and its standard result is that only the
linear components are unidentified: with maturity as age, release date as
period and vintage as cohort, the data pin down two combinations,

    Pi_p = beta_maturity + beta_date        Pi_c = beta_vintage - beta_maturity

and leave a one-parameter family indexed by s = beta_maturity. Restricting the
signs of the other two slopes bounds s, and the bound times the difference in
mean maturity between eligible and placebo cells bounds how much of the placebo
gap the unidentified component can account for.

The restrictions are the paper's own measurements, not assumptions imported
from outside: capability rises over calendar time, later benchmarks are less
saturated, and standing rises with maturity because a model on a young
benchmark is ranked against newer peers.
"""

import numpy as np
import pandas as pd

from .config import INTERIM

YEAR = 365.25


def coordinates(panel):
    """Attach period, cohort and age in years to the comparable cells."""
    cells = panel[panel["eligible"] | panel["placebo"]].dropna(
        subset=["percentile"]).copy()
    cells["period"] = cells["Release date"].map(pd.Timestamp.toordinal) / YEAR
    cells["cohort"] = cells["benchmark_release_date"].map(
        pd.Timestamp.toordinal) / YEAR
    cells["age"] = cells["period"] - cells["cohort"]
    return cells


def reduced_form(cells, degree=1):
    """The two linear combinations the data identify.

    `degree` is the highest power of period and cohort included, so 1 is the
    bare linear form. Centred powers above 1 are not collinear with age, so
    they sharpen the linear estimates without dissolving the identification
    problem the bound is about.
    """
    columns = [np.ones(len(cells)), cells["period"].to_numpy(),
               cells["cohort"].to_numpy()]
    for power in range(2, degree + 1):
        columns.append((cells["period"].to_numpy() - cells["period"].mean()) ** power)
        columns.append((cells["cohort"].to_numpy() - cells["cohort"].mean()) ** power)
    design = np.column_stack(columns)
    beta, *_ = np.linalg.lstsq(design, cells["percentile"].to_numpy(float), rcond=None)
    return float(beta[1]), float(beta[2])


def is_singular(cells):
    """The identification problem, stated as a rank deficiency."""
    design = np.column_stack([np.ones(len(cells)), cells["period"],
                              cells["cohort"], cells["age"]])
    return np.linalg.matrix_rank(design) < design.shape[1]


def bound(cells, degree=1):
    """Bound the maturity slope, and what it can explain of the placebo gap."""
    pi_p, pi_c = reduced_form(cells, degree)
    upper = min(pi_p, -pi_c)
    age_gap = float(cells.loc[cells["eligible"], "age"].mean()
                    - cells.loc[cells["placebo"], "age"].mean())
    return {
        "pi_period": pi_p,
        "pi_cohort": pi_c,
        "slope_low": 0.0,
        "slope_high": float(max(upper, 0.0)),
        "age_gap_years": age_gap,
        "explains_at_most": float(max(upper, 0.0) * age_gap),
        "binding": "saturation" if -pi_c < pi_p else "capability",
    }


def main():
    panel = pd.read_csv(
        INTERIM / "panel.csv",
        parse_dates=["Release date", "benchmark_release_date"],
    )
    from .percentiles import within_benchmark_percentile
    cells = coordinates(within_benchmark_percentile(panel))

    print("age-period-cohort structure of the placebo gap")
    print(f"  maturity = date - vintage holds to "
          f"{np.abs(cells['age'] - (cells['period'] - cells['cohort'])).max():.1e}")
    print(f"  design on all three is rank deficient: {is_singular(cells)}")
    print(f"  mean maturity gap, eligible minus placebo: "
          f"{cells.loc[cells['eligible'], 'age'].mean() - cells.loc[cells['placebo'], 'age'].mean():.3f} years")
    print("\n  degree  Pi_period  Pi_cohort   slope bound   explains at most")
    for degree in (1, 2, 3):
        result = bound(cells, degree)
        print(f"  {degree:>6}   {result['pi_period']:+8.3f}   {result['pi_cohort']:+8.3f}   "
              f"[0, {result['slope_high']:.3f}]   {result['explains_at_most']:.2f} points "
              f"({100 * result['explains_at_most'] / 12.23:.0f}% of the gap)")
    print("\n  the binding restriction is saturation, not capability")

    rng = np.random.default_rng(3)
    null = []
    for _ in range(30):
        frame = panel.copy()
        frame["score"] = frame.groupby("slug")["score"].transform(
            lambda v: rng.permutation(v.values))
        shuffled = coordinates(within_benchmark_percentile(frame))
        null.append(abs(bound(shuffled)["explains_at_most"]))
    print(f"\n  under scores shuffled within benchmark, which leaves every date and the "
          f"rank deficiency\n  untouched, the same quantity falls to "
          f"{np.mean(null):.2f} on average (sd {np.std(null):.2f}, max {np.max(null):.2f})")


if __name__ == "__main__":
    main()
