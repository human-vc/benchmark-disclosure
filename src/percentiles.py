"""Within-benchmark standing, computed against contemporaneous models.

docs/design.md: "Percentiles are taken within benchmark across contemporaneous
models, so 'standing' means position in the field rather than raw score."

The contemporaneity is not decoration. Ranking a 2023 model against the 2026
frontier pushes every early release toward the bottom of every benchmark
regardless of how it stood when it shipped, which is not what "standing" means
and which correlates with release date -- the same axis the temporal gate and
the drop design run along. A release is compared against the models a reader
could have compared it against at the time.
"""

import numpy as np
import pandas as pd

from .config import RELEASE_COL, WINDOW_DAYS


def within_benchmark_percentile(panel, window_days=WINDOW_DAYS):
    """Percentile of each score among peers on the same benchmark within
    +/- window_days of the focal release date.

    Ties take the midrank, so a benchmark where many models saturate at the
    same score does not hand an arbitrary advantage to whichever row sorts
    first. n_peers is retained: a percentile computed against two peers is not
    the same evidence as one computed against forty, and thin cells have to be
    visible rather than silently equal-weighted.
    """
    panel = panel.copy()
    panel["Release date"] = pd.to_datetime(panel["Release date"])

    percentile = np.full(len(panel), np.nan)
    n_peers = np.zeros(len(panel), dtype=int)
    newer_share = np.full(len(panel), np.nan)
    pct_older = np.full(len(panel), np.nan)
    pct_newer = np.full(len(panel), np.nan)

    for _, group in panel.groupby("slug", sort=False):
        positions = group.index.to_numpy()
        rows = panel.index.get_indexer(positions)
        days = group["Release date"].to_numpy("datetime64[D]").astype(np.int64)
        scores = group["score"].to_numpy(float)

        within = np.abs(days[:, None] - days[None, :]) <= window_days
        below = (scores[None, :] < scores[:, None]) & within
        equal = (scores[None, :] == scores[:, None]) & within
        counts = within.sum(axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            pct = 100.0 * (below.sum(axis=1) + 0.5 * equal.sum(axis=1)) / counts
        percentile[rows] = pct
        n_peers[rows] = counts

        # Split the window by which side of the focal release a peer sits on.
        # The window is symmetric in calendar time but a benchmark's coverage
        # is not: it begins when the benchmark was built, so a model evaluated
        # soon after that is ranked almost entirely against models newer than
        # itself. That asymmetry is a property of timing, and it is the same
        # timing that decides whether the benchmark was available to report.
        # A same-day peer sits on neither side, and the focal model is its own
        # same-day peer, so the sides are defined strictly. That is also what
        # makes the balanced measure *undefined* at the boundary rather than
        # quietly defined: a release at the very start of a benchmark's
        # coverage has no older peer to weight, and inventing one would hide
        # exactly the case the measure exists to expose.
        newer = within & (days[None, :] > days[:, None])
        older = within & (days[None, :] < days[:, None])
        n_new = newer.sum(axis=1).astype(float)
        n_old = older.sum(axis=1).astype(float)
        sided = n_new + n_old

        with np.errstate(invalid="ignore", divide="ignore"):
            newer_share[rows] = np.where(sided > 0, n_new / sided, np.nan)
            pct_newer[rows] = 100.0 * (
                (below & newer).sum(axis=1) + 0.5 * (equal & newer).sum(axis=1)
            ) / np.where(n_new > 0, n_new, np.nan)
            pct_older[rows] = 100.0 * (
                (below & older).sum(axis=1) + 0.5 * (equal & older).sum(axis=1)
            ) / np.where(n_old > 0, n_old, np.nan)

    panel["percentile"] = percentile
    panel["n_peers"] = n_peers
    panel["newer_share"] = newer_share
    panel["percentile_older"] = pct_older
    panel["percentile_newer"] = pct_newer
    panel["percentile_balanced"] = np.where(
        np.isnan(pct_older) | np.isnan(pct_newer),
        np.nan,
        0.5 * pct_older + 0.5 * pct_newer,
    )
    # The windowed percentile over peers only, which is what the two sides
    # decompose. It differs from `percentile` only by the focal model's own
    # midrank contribution, and it exists so the decomposition can be checked
    # as an identity rather than asserted.
    panel["percentile_sided"] = np.where(
        np.isnan(newer_share),
        np.nan,
        (1 - newer_share) * np.nan_to_num(pct_older)
        + newer_share * np.nan_to_num(pct_newer),
    )
    return panel


def add_percentiles(panel, window_days=WINDOW_DAYS):
    """Attach both the windowed percentile and the all-time one.

    The all-time column exists so the windowing choice can be shown not to
    drive the result, rather than asserted not to.
    """
    panel = within_benchmark_percentile(panel, window_days)
    panel["percentile_alltime"] = (
        panel.groupby("slug")["score"].rank(pct=True) * 100
    )
    return panel


def capability_level(panel):
    """A release's overall standing: mean windowed percentile across every
    benchmark it has an independent score on. This is the conditioning
    variable design.md calls the model's overall capability level."""
    return panel.groupby(RELEASE_COL)["percentile"].mean().rename("level")
