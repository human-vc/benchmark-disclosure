"""Within-benchmark standing, computed against contemporaneous models.

docs/design.md: "Percentiles are taken within benchmark across contemporaneous
models, so 'standing' means position in the field rather than raw score."

The contemporaneity is not decoration. Ranking a 2023 model against the 2026
frontier pushes every early release toward the bottom of every benchmark
regardless of how it stood when it shipped, which is not what "standing" means
and which correlates with release date -- the same axis the temporal gate and
the drop design run along. A release is compared against the models a reader
could have compared it against at the time.

### Sides are defined strictly

Splitting the window into older and newer peers is what the side-balanced
measure rests on, and the split admits two conventions. A peer released on the
same day sits on neither side, and a model is trivially its own same-day peer.
The strict convention excludes both from both sides. The loose one folds them
into the older side, which is arithmetically convenient and wrong in a specific
way: a model then counts as its own older peer, ties itself at midrank, and a
release with no older peer at all is assigned an older-side standing of 50
rather than none.

That is not a rounding difference. It is exactly the case the balanced measure
exists to expose -- a model evaluated at the very start of a benchmark's
coverage, ranked almost entirely against models newer than itself -- and the
loose convention gives it a defined number built from nothing but itself. So
the sides are strict, `pct_balanced` is undefined wherever a side is empty, and
the coverage that costs is reported rather than closed by imputation.

The price is that the decomposition identity no longer lands on `percentile`,
which includes the focal model's own midrank contribution. It lands on
`pct_sided`, the same average taken over peers only. Under the loose
convention the identity holds against `percentile` *because* the model is
counted among its own peers, so the exactness is an artifact of the defect
rather than a check on the arithmetic.
"""

import numpy as np
import pandas as pd

from .config import RELEASE_COL, WINDOW_DAYS


def _sides(panel, window_days):
    """Per-row window arithmetic, split strictly by side.

    Returns percentile over the whole window (self included, as the reported
    measure has always been), peer counts, and the older/newer split with
    same-day peers and the focal model itself excluded from both sides.
    """
    panel = panel.copy()
    panel["Release date"] = pd.to_datetime(panel["Release date"])

    percentile = np.full(len(panel), np.nan)
    n_peers = np.zeros(len(panel), dtype=int)
    share_newer = np.full(len(panel), np.nan)
    pct_old = np.full(len(panel), np.nan)
    pct_new = np.full(len(panel), np.nan)

    for _, group in panel.groupby("slug", sort=False):
        rows = panel.index.get_indexer(group.index.to_numpy())
        days = group["Release date"].to_numpy("datetime64[D]").astype(np.int64)
        scores = group["score"].to_numpy(float)

        within = np.abs(days[:, None] - days[None, :]) <= window_days
        below = (scores[None, :] < scores[:, None]) & within
        equal = (scores[None, :] == scores[:, None]) & within
        counts = within.sum(axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            percentile[rows] = (
                100.0 * (below.sum(axis=1) + 0.5 * equal.sum(axis=1)) / counts
            )
        n_peers[rows] = counts

        newer = within & (days[None, :] > days[:, None])
        older = within & (days[None, :] < days[:, None])
        n_new = newer.sum(axis=1).astype(float)
        n_old = older.sum(axis=1).astype(float)
        sided = n_new + n_old

        with np.errstate(invalid="ignore", divide="ignore"):
            share_newer[rows] = np.where(sided > 0, n_new / sided, np.nan)
            pct_new[rows] = 100.0 * (
                (below & newer).sum(axis=1) + 0.5 * (equal & newer).sum(axis=1)
            ) / np.where(n_new > 0, n_new, np.nan)
            pct_old[rows] = 100.0 * (
                (below & older).sum(axis=1) + 0.5 * (equal & older).sum(axis=1)
            ) / np.where(n_old > 0, n_old, np.nan)

    return panel, percentile, n_peers, share_newer, pct_old, pct_new


def within_benchmark_percentile(panel, window_days=WINDOW_DAYS):
    """Percentile of each score among peers on the same benchmark within
    +/- window_days of the focal release date.

    Ties take the midrank, so a benchmark where many models saturate at the
    same score does not hand an arbitrary advantage to whichever row sorts
    first. n_peers is retained: a percentile computed against two peers is not
    the same evidence as one computed against forty, and thin cells have to be
    visible rather than silently equal-weighted.

    `share_newer` is the fraction of a cell's *sided* window released after the
    focal model. The window is symmetric in calendar time but a benchmark's
    coverage is not -- it begins when the benchmark was built -- and that
    asymmetry is the same timing that decides whether the benchmark was
    available to report.
    """
    panel, percentile, n_peers, share_newer, _, _ = _sides(panel, window_days)
    panel["percentile"] = percentile
    panel["n_peers"] = n_peers
    panel["share_newer"] = share_newer
    return panel


def side_balanced_percentile(panel, window_days=WINDOW_DAYS):
    """Standing that does not depend on which side of a benchmark's coverage
    the focal model happens to sit.

    Position among older peers and position among newer peers, averaged with
    equal weight instead of the empirical one. Undefined wherever a side is
    empty; see the module docstring for why that is kept rather than imputed.
    """
    panel, _, _, share_newer, pct_old, pct_new = _sides(panel, window_days)

    panel["pct_old_side"] = pct_old
    panel["pct_new_side"] = pct_new
    panel["pct_balanced"] = np.where(
        np.isnan(pct_old) | np.isnan(pct_new), np.nan, 0.5 * pct_old + 0.5 * pct_new
    )
    # The windowed percentile over peers only, which is what the two sides
    # decompose. It differs from `percentile` only by the focal model's own
    # midrank contribution, and it exists so the decomposition can be checked
    # as an identity rather than asserted.
    panel["pct_sided"] = np.where(
        np.isnan(share_newer),
        np.nan,
        (1 - share_newer) * np.nan_to_num(pct_old)
        + share_newer * np.nan_to_num(pct_new),
    )
    return panel


def add_percentiles(panel, window_days=WINDOW_DAYS):
    """Attach every standing measure at once: the windowed percentile, the
    side-balanced one, and the all-time one.

    The all-time column exists so the windowing choice can be shown not to
    drive the result, rather than asserted not to. The side-balanced columns
    are attached here because the remedy table compares all of them on one
    panel, and a measure missing from that table would read as a repair that
    was never tried rather than one that was tried and failed.
    """
    panel = side_balanced_percentile(
        within_benchmark_percentile(panel, window_days), window_days
    )
    panel["percentile_alltime"] = (
        panel.groupby("slug")["score"].rank(pct=True) * 100
    )
    return panel


def capability_level(panel):
    """A release's overall standing: mean windowed percentile across every
    benchmark it has an independent score on. This is the conditioning
    variable design.md calls the model's overall capability level."""
    return panel.groupby(RELEASE_COL)["percentile"].mean().rename("level")
