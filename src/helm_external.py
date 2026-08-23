"""The same bias on a leaderboard we did not build.

The obvious objection to everything else in this repository is that the measure
being diagnosed and repaired is our own. Nobody else publishes a 182-day
symmetric-window midrank, so a reader can reasonably ask who is harmed by the
problem we found.

HELM Lite answers it. Its headline Accuracy column is Mean win rate, defined as
the fraction of head-to-head comparisons a model wins against the other models
evaluated, which is a pool-relative statistic in exactly the sense that matters
here. Across fourteen public releases the ten scenario columns never change, and
twenty-four models appear in every release. Their two hundred and forty
published scenario scores are bit-identical from the first release to the last.
Every one of their published Mean win rates moves anyway, and a handful of pairs
change order.

Nothing here says HELM is wrong. Mean win rate is correct as defined, and HELM
itself later moved its Capabilities leaderboard to an absolute mean score. The
point is narrower and it is the paper's point: a pool-relative standing statistic
reorders models whose evidence did not change, so the composition of the
comparison set is doing work that reads as capability.

The evidence is frozen in data/external/helm_lite_accuracy.json with a SHA-256
per source file, because a leaderboard is a moving target and the whole argument
depends on the reader seeing the bytes we saw.
"""

import itertools
import json

import numpy as np

from .config import ROOT

FROZEN = ROOT / "data" / "external" / "helm_lite_accuracy.json"
CONTROL = ROOT / "data" / "external" / "helm_capabilities_accuracy.json"

# HELM Capabilities is the control, and it is as close to a natural experiment as
# this question is going to get. Same organisation, same infrastructure, same
# release cadence, pool roughly tripling in both. The single difference is that
# its headline is Mean score, an absolute average, rather than a pool-relative
# win rate. The prediction stated before running it was zero drift.
CONTROL_SCENARIOS = [
    "MMLU-Pro - COT correct",
    "GPQA - COT correct",
    "IFEval - IFEval Strict Acc",
    "WildBench - WB Score",
    "Omni-MATH - Acc",
]
CONTROL_HEADLINE = "Mean score"

SCENARIOS = [
    "NarrativeQA - F1",
    "NaturalQuestions (open-book) - F1",
    "NaturalQuestions (closed-book) - F1",
    "OpenbookQA - EM",
    "MMLU - EM",
    "MATH - Equivalent (CoT)",
    "GSM8K - EM",
    "LegalBench - EM",
    "MedQA - EM",
    "WMT 2014 - BLEU-4",
]
HEADLINE = "Mean win rate"


def load(path=FROZEN):
    payload = json.loads(path.read_text())
    order = sorted(payload["releases"], key=lambda v: int(v.split(".")[1]))
    return payload, order


def scenario_columns_are_stable(payload, order, scenarios=None):
    """The first objection is that they added scenarios. They did not.

    The model column's header is relabelled from "Model/adapter" to "Model" at
    v1.10.0, which is cosmetic. The ten scenario columns are identical
    throughout, and that is what the frozen-cell claim rests on.
    """
    scenarios = scenarios or SCENARIOS
    sets = {
        tuple(c for c in payload["releases"][v]["columns"] if c in scenarios)
        for v in order
    }
    return len(sets) == 1 and len(next(iter(sets))) == len(scenarios)


def frozen_models(payload, order):
    return sorted(set.intersection(*[set(payload["releases"][v]["rows"]) for v in order]))


def frozen_cells(payload, order, models, scenarios=None):
    """Published scenario scores for the models present in every release."""
    scenarios = scenarios or SCENARIOS
    total = moved = 0
    spread = 0.0
    for model in models:
        for scenario in scenarios:
            series = [payload["releases"][v]["rows"][model].get(scenario) for v in order]
            if not all(isinstance(x, (int, float)) for x in series):
                continue
            total += 1
            width = max(series) - min(series)
            spread = max(spread, width)
            if width > 5e-9:
                moved += 1
    return {"cells": total, "changed": moved, "max_spread": spread}


def headline_drift(payload, order, models, headline=None):
    headline = headline or HEADLINE
    series = {
        m: [payload["releases"][v]["rows"][m].get(headline) for v in order]
        for m in models
    }
    series = {m: s for m, s in series.items() if all(isinstance(x, (int, float)) for x in s)}
    change = {m: s[-1] - s[0] for m, s in series.items()}
    values = np.array(list(change.values()))
    initial = np.array([series[m][0] for m in change])

    pairs = list(itertools.combinations(sorted(series), 2))
    endpoint = [
        (a, b) for a, b in pairs
        if (series[a][0] - series[b][0]) * (series[a][-1] - series[b][-1]) < 0
    ]
    ever = [
        (a, b) for a, b in pairs
        if any((series[a][0] - series[b][0]) * (series[a][i] - series[b][i]) < 0
               for i in range(len(order)))
    ]
    return {
        "models": len(series),
        "moved": int((np.abs(values) > 1e-9).sum()),
        "mean_abs_change": float(np.abs(values).mean()),
        "max_abs_change": float(np.abs(values).max()),
        "fell": int((values < 0).sum()),
        "rose": int((values > 0).sum()),
        # the control moves nothing, so the changes are identically zero and the
        # correlation is undefined rather than zero. Say so instead of dividing.
        "corr_change_with_initial": (
            float(np.corrcoef(values, initial)[0, 1])
            if values.std() > 0 and initial.std() > 0 else None
        ),
        "pairs": len(pairs),
        "reversals_endpoint": len(endpoint),
        "reversals_ever": len(ever),
        "reversal_examples": [
            {"a": a, "b": b,
             "first": [round(series[a][0], 4), round(series[b][0], 4)],
             "last": [round(series[a][-1], 4), round(series[b][-1], 4)]}
            for a, b in endpoint
        ],
        "series": series,
    }


def summary(path=FROZEN):
    payload, order = load(path)
    models = frozen_models(payload, order)
    return {
        "releases": len(order),
        "pool_size_first": len(payload["releases"][order[0]]["rows"]),
        "pool_size_last": len(payload["releases"][order[-1]]["rows"]),
        "scenario_columns_stable": scenario_columns_are_stable(payload, order),
        "frozen_models": len(models),
        "frozen_cells": frozen_cells(payload, order, models),
        "headline": {k: v for k, v in headline_drift(payload, order, models).items()
                     if k != "series"},
    }


def control(path=CONTROL):
    """The falsification test, with its prediction stated before it was run.

    If the drift in HELM Lite were about anything other than the statistic being
    pool-relative, it would show up here too. Same evaluator, same machinery,
    same growth. It does not show up. Zero of twenty-two headline values move and
    no pair reorders, against twenty-four of twenty-four and five in Lite.
    """
    payload, order = load(path)
    models = frozen_models(payload, order)
    head = headline_drift(payload, order, models, headline=CONTROL_HEADLINE)
    return {
        "releases": len(order),
        "pool_size_first": len(payload["releases"][order[0]]["rows"]),
        "pool_size_last": len(payload["releases"][order[-1]]["rows"]),
        "headline": CONTROL_HEADLINE,
        "headline_kind": "absolute",
        "scenario_columns_stable": scenario_columns_are_stable(
            payload, order, CONTROL_SCENARIOS),
        "frozen_models": len(models),
        "frozen_cells": frozen_cells(payload, order, models, CONTROL_SCENARIOS),
        "moved": head["moved"],
        "models": head["models"],
        "reversals_endpoint": head["reversals_endpoint"],
        "reversals_ever": head["reversals_ever"],
    }


def main():
    report = summary()
    head = report["headline"]
    print(f"HELM Lite, {report['releases']} public releases, "
          f"pool {report['pool_size_first']} -> {report['pool_size_last']} models")
    print(f"  scenario columns identical throughout: {report['scenario_columns_stable']}")
    print(f"  models present in every release: {report['frozen_models']}")

    cells = report["frozen_cells"]
    print(f"\n  their published scenario scores: {cells['cells']} cells, "
          f"{cells['changed']} changed, max spread {cells['max_spread']:.2e}")
    print(f"  their published Mean win rates : {head['moved']} of {head['models']} moved")
    print(f"    mean |change| {head['mean_abs_change']:.4f}, "
          f"max {head['max_abs_change']:.4f}, "
          f"{head['fell']} fell and {head['rose']} rose")
    print(f"    correlation of change with initial standing: "
          f"{head['corr_change_with_initial']:+.3f}")
    print(f"\n  pairs reordered: {head['reversals_endpoint']} of {head['pairs']} "
          f"at the endpoints, {head['reversals_ever']} at some release")
    for case in head["reversal_examples"]:
        print(f"    {case['a']} vs {case['b']}: "
              f"{case['first'][0]:.3f}/{case['first'][1]:.3f} -> "
              f"{case['last'][0]:.3f}/{case['last'][1]:.3f}")
    print("\n  Mean win rate is correct as defined. The evidence did not change;")
    print("  the pool did, and the published order followed the pool.")

    if CONTROL.exists():
        ctl = control()
        print(f"\n  CONTROL, HELM Capabilities, headline is an absolute {ctl['headline']}:")
        print(f"    {ctl['releases']} releases, pool {ctl['pool_size_first']} -> "
              f"{ctl['pool_size_last']}, {ctl['frozen_models']} models throughout")
        print(f"    frozen cells: {ctl['frozen_cells']['cells']}, "
              f"{ctl['frozen_cells']['changed']} changed")
        print(f"    headline values moved: {ctl['moved']} of {ctl['models']}   "
              f"pairs reordered: {ctl['reversals_endpoint']}")
        print("    Same evaluator, same machinery, same growth. The drift appears")
        print("    where the statistic is pool-relative and nowhere else.")


if __name__ == "__main__":
    main()
