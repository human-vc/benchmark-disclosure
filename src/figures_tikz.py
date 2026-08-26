"""Native TikZ/pgfplots figures, generated from the panel."""

import numpy as np
import pandas as pd

from .config import INTERIM, RELEASE_COL
from .helm_external import (
    CONTROL,
    CONTROL_HEADLINE,
    FROZEN,
    frozen_models,
    headline_drift,
    load,
)
from .percentiles import side_balanced_percentile, within_benchmark_percentile

OUT = INTERIM.parent.parent / "paper" / "figures"
EDGES = np.arange(0, 1.0001, 0.1)


def _panel():
    frame = pd.read_csv(
        INTERIM / "panel.csv", parse_dates=["Release date", "benchmark_release_date"]
    )
    return side_balanced_percentile(within_benchmark_percentile(frame))


def _binned(frame, xcol, ycol, min_cells=15):
    """Bin means with an interval clustered on organisation."""
    rows = []
    for lo, hi in zip(EDGES, EDGES[1:]):
        cell = frame[(frame[xcol] >= lo) & (frame[xcol] < hi)]
        if len(cell) < min_cells:
            continue
        spread = cell[ycol].std() / np.sqrt(max(cell["primary_org"].nunique(), 1))
        rows.append(((lo + hi) / 2, cell[ycol].mean(), 1.96 * spread))
    return rows


def _coords(rows, errors=False):
    if errors:
        return " ".join(f"({x:.3f},{y:.2f}) +- (0,{e:.2f})" for x, y, e in rows)
    return " ".join(f"({x:.3f},{y:.2f})" for x, y, *_ in rows)


def fig_one_curve(panel):
    """Panel (a) is the problem, panel (b) the correction, on the same axes."""
    cells = panel[panel["eligible"] | panel["placebo"]].dropna(
        subset=["percentile", "share_newer"]
    )
    pooled = _binned(cells, "share_newer", "percentile")
    elig = _binned(cells[cells["eligible"]], "share_newer", "percentile")
    plac = _binned(cells[cells["placebo"]], "share_newer", "percentile")

    only = panel[panel["eligible"]]
    corrected = []
    for column in ("percentile", "pct_balanced"):
        cell = only.dropna(subset=[column, "share_newer"]).copy()
        centred = (cell[column] - cell.groupby(RELEASE_COL)[column].transform("mean")
                   + cell[column].mean())
        corrected.append(_binned(cell.assign(c=centred), "share_newer", "c"))

    axis = ("causalre body, xmin=0, xmax=1, xtick={0,0.5,1}, "
            "xlabel={share of window newer than the focal model}")
    return f"""\\begin{{tikzpicture}}
  \\begin{{axis}}[{axis}, ymin=20, ymax=80,
    ylabel={{within-benchmark standing}}, title={{(a)}},
    legend to name=leg:onecurve, legend columns=3]
    \\addplot[cMUTE, line width=2.2pt, smooth, forget plot]
      coordinates {{{_coords(pooled)}}};
    \\addplot[black, dashed, thin, forget plot] coordinates {{(0.5,20) (0.5,80)}};
    \\addplot[cNYC, only marks, mark=*, mark size=1.4pt,
             error bars/.cd, y dir=both, y explicit]
      coordinates {{{_coords(elig, errors=True)}}};
    \\addlegendentry{{benchmark predates the release}}
    \\addplot[cSF, only marks, mark=square*, mark size=1.4pt,
             error bars/.cd, y dir=both, y explicit]
      coordinates {{{_coords(plac, errors=True)}}};
    \\addlegendentry{{benchmark postdates the release}}
    \\addplot[cMUTE, line width=2.2pt] coordinates {{(0,0)}};
    \\addlegendentry{{all cells pooled}}
  \\end{{axis}}
  \\begin{{axis}}[{axis}, ymin=35, ymax=65, xshift=0.50\\textwidth,
    title={{(b)}}, legend to name=leg:correction, legend columns=2]
    \\addplot[cSF, smooth, mark=*, mark size=1.4pt]
      coordinates {{{_coords(corrected[0])}}};
    \\addlegendentry{{windowed percentile}}
    \\addplot[cNYC, smooth, mark=square*, mark size=1.4pt]
      coordinates {{{_coords(corrected[1])}}};
    \\addlegendentry{{side-balanced}}
  \\end{{axis}}
\\end{{tikzpicture}}
\\\\[2pt]
{{\\footnotesize \\ref{{leg:onecurve}} \\hfill \\ref{{leg:correction}}}}
"""


def fig_helm():
    """Rank at the first release against rank at the last, as a slopegraph.

    The claim is that pairs reverse order while the evidence is frozen, and in
    a two-point rank plot a reversed pair is a crossing, one to one, so the
    statistic counts itself on the page. The fourteen intermediate releases
    added wiggle without adding evidence; a bump chart of them buried the five
    crossings in one-rank noise.
    """
    def endpoints(path, headline=None):
        payload, order = load(path)
        models = frozen_models(payload, order)
        drift = headline_drift(payload, order, models, headline=headline)
        series = drift["series"]
        names = sorted(series)
        def rank_at(t):
            return {m: place for place, m in enumerate(
                sorted(names, key=lambda k: -series[k][t]), 1)}
        first, last = rank_at(0), rank_at(len(order) - 1)
        import itertools
        pairs = [(a, b) for a, b in itertools.combinations(names, 2)
                 if (first[a] - first[b]) * (last[a] - last[b]) < 0]
        sloped = {m for m in names if first[m] != last[m]}
        return first, last, pairs, sloped, order[0], order[-1], len(names)

    def components(pairs):
        """Groups of mutually reversing models, one hue each, so every
        crossing is a same-coloured X and the five are countable."""
        parent = {}
        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, b in pairs:
            parent[find(a)] = find(b)
        groups = {}
        for m in parent:
            groups.setdefault(find(m), []).append(m)
        return list(groups.values())

    def lines(first, last, pairs):
        hues = ["cSF", "cNYC", "cGRN", "cPNK"]
        hue_of = {}
        for k, group in enumerate(sorted(components(pairs), key=min)):
            for m in group:
                hue_of[m] = hues[k % len(hues)]
        out = []
        for m in sorted(first, key=lambda k: k in hue_of):
            if m in hue_of:
                style = f"{hue_of[m]}, line width=0.9pt, mark=*, mark size=1.0pt"
            else:
                style = "cMUTE, line width=0.4pt"
            out.append(f"    \\addplot[{style}, forget plot] "
                       f"coordinates {{(0.3,{first[m]}) (0.7,{last[m]})}};")
        return "\n".join(out)

    lf, ll, lc, ls, l0, l1, lk = endpoints(FROZEN)
    cf, cl, cc, cs, c0, c1, ck = endpoints(CONTROL, CONTROL_HEADLINE)
    print(f"  lite: {lk} models, {len(lc)} reversed pairs in "
          f"{len(components(lc))} groups, {len(ls)} sloped; "
          f"capabilities: {ck} models, {len(cc)} pairs, {len(cs)} sloped")

    axis = ("causalre body, y dir=reverse, ymin=0.4, xmin=0, xmax=1, "
            "xlabel={release}, ytick={1,6,12,18,24}")
    return f"""\\begin{{tikzpicture}}
  \\begin{{axis}}[{axis}, xtick={{0.3,0.7}}, xticklabels={{{l0},{l1}}}, ymax={lk + 0.6},
    ylabel={{rank among the constant models}}, title={{(a)}}]
{lines(lf, ll, lc)}
  \\end{{axis}}
  \\begin{{axis}}[{axis}, xtick={{0.3,0.7}}, xticklabels={{{c0},{c1}}}, ymax={ck + 0.6},
    xshift=0.50\\textwidth, yticklabels={{,,}}, title={{(b)}}]
{lines(cf, cl, cc)}
  \\end{{axis}}
\\end{{tikzpicture}}"""


def fig_geometry(panel, slug="hle", window=182):
    """The mechanism, drawn on real coverage rather than illustrated."""
    cells = panel[panel["slug"] == slug].dropna(subset=["Release date"])
    days = cells["Release date"].map(pd.Timestamp.toordinal).to_numpy()
    order = np.argsort(days)
    days = days[order]
    origin = days.min()
    x = days - origin
    span = x.max()

    boundary = cells["benchmark_release_date"].dropna()
    bx = boundary.iloc[0].toordinal() - origin if len(boundary) else 0

    edge_i = 0
    interior_i = int(np.argmin(np.abs(x - (span * 0.62))))

    def lane(focal_i, y):
        lo, hi = x[focal_i] - window, x[focal_i] + window
        ticks = "\n".join(
            f"    \\draw[cMUTE, line width=0.9pt] ({v:.1f},{y-0.30}) -- ({v:.1f},{y+0.30});"
            for v in x
        )
        return (
            f"    \\fill[cSKY!30] ({lo:.1f},{y-0.42}) rectangle ({hi:.1f},{y+0.42});\n"
            + ticks
            + f"\n    \\draw[cNYC, line width=2.2pt] ({x[focal_i]:.1f},{y-0.42}) -- "
              f"({x[focal_i]:.1f},{y+0.42});"
        )

    return f"""\\begin{{tikzpicture}}[x=0.00062\\textwidth, y=1cm]
{lane(edge_i, 1.15)}
{lane(interior_i, 0.0)}
    \\draw[black, dashed, thin] ({bx:.1f},-0.62) -- ({bx:.1f},1.78);
    \\draw[cMUTE, line width=0.4pt] (0,-0.62) -- ({span:.1f},-0.62);
\\end{{tikzpicture}}
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = _panel()
    written = {
        "fig_one_curve.tex": fig_one_curve(panel),
        "fig_helm.tex": fig_helm(),
        "fig_geometry.tex": fig_geometry(panel),
    }
    for name, body in written.items():
        (OUT / name).write_text(body)
        print(f"wrote {OUT / name}  ({len(body.splitlines())} lines)")


if __name__ == "__main__":
    main()
