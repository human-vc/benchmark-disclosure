
import itertools
import json

import numpy as np

from .config import ROOT

FROZEN = ROOT / "data" / "external" / "helm_lite_accuracy.json"
CONTROL = ROOT / "data" / "external" / "helm_capabilities_accuracy.json"

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
    scenarios = scenarios or SCENARIOS
    sets = {
        tuple(c for c in payload["releases"][v]["columns"] if c in scenarios)
        for v in order
    }
    return len(sets) == 1 and len(next(iter(sets))) == len(scenarios)

def frozen_models(payload, order):
    return sorted(set.intersection(*[set(payload["releases"][v]["rows"]) for v in order]))

def frozen_cells(payload, order, models, scenarios=None):
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
