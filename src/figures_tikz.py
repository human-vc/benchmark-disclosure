"""Native TikZ/pgfplots figures, generated from the panel.

One figure, one claim. The one-curve panel carries the empirical result alone,
with its correction moved to the appendix. The geometry figure is a schematic,
since its purpose is the mechanism rather than the data. The HELM figure shows
only the models whose published order changed, named, with the control stated
in the caption as a number, because twenty-two flat lines spent half a figure
saying zero.
"""

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

SHORTEN = {
    "text-davinci-002": "davinci-002",
    " 32K seqlen": "",
    "(0613)": "0613",
}


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
    """The single-panel empirical figure: one declining curve, two groups on it."""
    cells = panel[panel["eligible"] | panel["placebo"]].dropna(
        subset=["percentile", "share_newer"]
    )
    elig = _binned(cells[cells["eligible"]], "share_newer", "percentile")
    plac = _binned(cells[cells["placebo"]], "share_newer", "percentile")
    reach = max(max(x for x, *_ in elig), max(x for x, *_ in plac))
    pooled = [row for row in _binned(cells, "share_newer", "percentile")
              if row[0] <= reach + 1e-9]
    mean_e = cells.loc[cells["eligible"], "share_newer"].mean()
    mean_p = cells.loc[cells["placebo"], "share_newer"].mean()

    axis = ("causalre wide, height=4.6cm, xmin=0, xmax=1, xtick={0,0.5,1}, "
            "xlabel={share of window newer than the focal model}")
    return f"""\\begin{{tikzpicture}}
  \\begin{{axis}}[{axis}, ymin=20, ymax=80,
    ylabel={{within-benchmark standing}},
    legend to name=leg:onecurve, legend columns=3]
    \\addplot[cMUTE, line width=2.2pt, smooth, forget plot]
      coordinates {{{_coords(pooled)}}};
    \\addplot[black, dashed, thin, forget plot] coordinates {{(0.5,20) (0.5,80)}};
    \\addplot[cNYC, dashed, thin, forget plot]
      coordinates {{({mean_e:.3f},20) ({mean_e:.3f},80)}};
    \\addplot[cSF, dashed, thin, forget plot]
      coordinates {{({mean_p:.3f},20) ({mean_p:.3f},80)}};
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
\\end{{tikzpicture}}
\\\\[2pt]
{{\\footnotesize \\ref{{leg:onecurve}}}}"""


def fig_correction(panel):
    """The side-balanced repair against the uncorrected measure, for the appendix."""
    only = panel[panel["eligible"]]
    series = []
    for column in ("percentile", "pct_balanced"):
        cell = only.dropna(subset=[column, "share_newer"]).copy()
        centred = (cell[column] - cell.groupby(RELEASE_COL)[column].transform("mean")
                   + cell[column].mean())
        series.append(_binned(cell.assign(c=centred), "share_newer", "c"))

    axis = ("causalre body, xmin=0, xmax=1, xtick={0,0.5,1}, "
            "xlabel={share of window newer than the focal model}")
    return f"""\\begin{{tikzpicture}}
  \\begin{{axis}}[{axis}, ymin=35, ymax=65,
    ylabel={{within-release standing}},
    legend to name=leg:correction, legend columns=2]
    \\addplot[cSF, smooth, mark=*, mark size=1.4pt]
      coordinates {{{_coords(series[0])}}};
    \\addlegendentry{{windowed percentile}}
    \\addplot[cNYC, smooth, mark=square*, mark size=1.4pt]
      coordinates {{{_coords(series[1])}}};
    \\addlegendentry{{side-balanced}}
  \\end{{axis}}
\\end{{tikzpicture}}
\\\\[2pt]
{{\\footnotesize \\ref{{leg:correction}}}}"""


def fig_helm():
    """A labeled slopegraph of the models whose published order changed.

    A reversed pair is a crossing, one to one, so the statistic counts itself.
    The absolute-headline control moved nothing, and a number in the caption
    says so better than twenty-two flat lines.
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
        return first, last, pairs, order[0], order[-1], len(names)

    def components(pairs):
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

    def display(name):
        text = name.replace("(", "").replace(")", "")
        for long, short in SHORTEN.items():
            text = text.replace(long.replace("(", "").replace(")", ""), short)
        return text.strip()

    first, last, pairs, v0, v1, total = endpoints(FROZEN)
    _, _, ctl_pairs, _, _, ctl_total = endpoints(CONTROL, CONTROL_HEADLINE)
    movers = sorted({m for pair in pairs for m in pair}, key=lambda m: first[m])
    labels = [display(m) for m in movers]
    assert len(set(labels)) == len(labels), "shortened names collide"
    print(f"  lite: {len(pairs)} reversed pairs among {len(movers)} of {total} "
          f"models; control: {len(ctl_pairs)} pairs among {ctl_total}")

    hues = ["cSF", "cNYC", "cGRN", "cPNK"]
    hue_of = {}
    for k, group in enumerate(sorted(components(pairs), key=min)):
        for m in group:
            hue_of[m] = hues[k % len(hues)]

    lines, names = [], []
    for m in movers:
        lines.append(f"    \\addplot[{hue_of[m]}, line width=0.9pt, mark=*, "
                     f"mark size=1.2pt, forget plot] "
                     f"coordinates {{(0.45,{first[m]}) (0.75,{last[m]})}};")
        names.append(f"    \\node[anchor=east, font=\\tiny, text=black] "
                     f"at (axis cs:0.43,{first[m]}) {{{display(m)}}};")

    lo = min(first[m] for m in movers) - 1
    hi = max(first[m] for m in movers) + 1
    return f"""\\begin{{tikzpicture}}
  \\begin{{axis}}[causalre wide, height=4.6cm, y dir=reverse,
    xmin=0, xmax=1, ymin={lo - 0.2}, ymax={hi + 0.2},
    xtick={{0.45,0.75}}, xticklabels={{{v0},{v1}}}, xlabel={{release}},
    ytick={{4,8,12,16}}, ylabel={{rank among the {total} constant models}}]
{chr(10).join(lines)}
{chr(10).join(names)}
  \\end{{axis}}
\\end{{tikzpicture}}"""


def fig_geometry():
    """The mechanism as a schematic: one benchmark, two releases, one window."""
    unit = 0.00062
    window = 182
    focal_a, focal_b = 30, 370
    span_lo, span_hi = -200, 580
    top, bottom = 1.15, 0.0
    half = 0.42

    def lane(focal, y, older_label):
        left, right = focal - window, focal + window
        parts = [
            f"    \\fill[cSKY!30] ({max(left, 0)},{y - half}) rectangle ({right},{y + half});",
        ]
        if left < 0:
            parts.append(
                f"    \\fill[pattern=north east lines, pattern color=cMUTE!70] "
                f"({left},{y - half}) rectangle (0,{y + half});")
            parts.append(
                f"    \\node[font=\\tiny, text=black] at ({left / 2},{y + half + 0.16}) "
                f"{{nothing scored}};")
        else:
            parts.append(
                f"    \\node[font=\\tiny, text=black] at ({focal - window / 2},{y + half + 0.16}) "
                f"{{{older_label}}};")
        parts.append(
            f"    \\node[font=\\tiny, text=black] at ({focal + window / 2 + 22},{y + half + 0.16}) "
            f"{{newer peers}};")
        parts.append(
            f"    \\draw[cMUTE, line width=0.6pt] (0,{y}) -- ({span_hi},{y});")
        parts.append(
            f"    \\draw[cNYC, line width=2.2pt] ({focal},{y - half - 0.08}) -- "
            f"({focal},{y + half + 0.08});")
        return "\n".join(parts)

    return f"""\\begin{{tikzpicture}}[x={unit}\\textwidth, y=1cm]
{lane(focal_a, top, "older peers")}
{lane(focal_b, bottom, "older peers")}
    \\draw[cGRN, line width=1.4pt] (0,{bottom - half - 0.22}) -- (0,{top + half + 0.34});
    \\draw[cMUTE, line width=0.4pt, -stealth] ({span_lo},{bottom - half - 0.22}) -- ({span_hi},{bottom - half - 0.22});
\\end{{tikzpicture}}"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = _panel()
    written = {
        "fig_one_curve.tex": fig_one_curve(panel),
        "fig_correction.tex": fig_correction(panel),
        "fig_helm.tex": fig_helm(),
        "fig_geometry.tex": fig_geometry(),
    }
    for name, body in written.items():
        (OUT / name).write_text(body + "\n")
        print(f"wrote {OUT / name}  ({len(body.splitlines())} lines)")


if __name__ == "__main__":
    main()
