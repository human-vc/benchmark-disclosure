
import numpy as np
import pandas as pd

from .config import RELEASE_COL, WINDOW_DAYS

def _sides(panel, window_days):
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
    panel, percentile, n_peers, share_newer, _, _ = _sides(panel, window_days)
    panel["percentile"] = percentile
    panel["n_peers"] = n_peers
    panel["share_newer"] = share_newer
    return panel

def side_balanced_percentile(panel, window_days=WINDOW_DAYS):
    panel, _, _, share_newer, pct_old, pct_new = _sides(panel, window_days)

    panel["pct_old_side"] = pct_old
    panel["pct_new_side"] = pct_new
    panel["pct_balanced"] = np.where(
        np.isnan(pct_old) | np.isnan(pct_new), np.nan, 0.5 * pct_old + 0.5 * pct_new
    )
    panel["pct_sided"] = np.where(
        np.isnan(share_newer),
        np.nan,
        (1 - share_newer) * np.nan_to_num(pct_old)
        + share_newer * np.nan_to_num(pct_new),
    )
    return panel

def add_percentiles(panel, window_days=WINDOW_DAYS):
    panel = side_balanced_percentile(
        within_benchmark_percentile(panel, window_days), window_days
    )
    panel["percentile_alltime"] = (
        panel.groupby("slug")["score"].rank(method="average", pct=True) * 100
        - 50.0 / panel.groupby("slug")["score"].transform("size")
    )
    return panel

def capability_level(panel):
    return panel.groupby(RELEASE_COL)["percentile"].mean().rename("level")
