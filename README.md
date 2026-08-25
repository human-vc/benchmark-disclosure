# Selective Benchmark Disclosure

Measuring what AI model providers *don't* report.

## The question

Model providers publish benchmark results in model cards, system cards, and release posts. Independent evaluators publish results for the same models. The two sets do not match, and the mismatch is a choice.

Epoch AI has verified that providers report benchmark scores *accurately* — their independently-run GPQA Diamond scores show no significant discrepancy against provider self-reports across eight major labs. So the interesting behavior is not misstatement. It is selection: which of the available results appear at all.

This repository builds the measurement. For each model release we compare the set of benchmarks the provider disclosed against the set of benchmarks independently scored for that model, and ask whether omissions are concentrated where the model performs badly.

## Status

Early. The pipeline runs and a six-model pilot is complete; the full disclosure coding is not.

**Pilot result (n=6):** reported-benchmark percentile exceeded omitted-benchmark percentile in all five codeable cases, with gaps of 20.6 to 40.9 percentile points. Both closed-API and open-weights providers showed the pattern.

**Pilot non-result:** open-weights providers were *not* less selective than closed-API providers. The largest single gap belonged to an open-weights model. Treat any open-vs-closed hypothesis as unsupported until the full sample says otherwise.

## Design

The naive comparison — "provider didn't report benchmark X" — does not identify anything. Non-disclosure has at least four innocent explanations that look identical to strategic omission:

1. The benchmark did not exist when the model shipped.
2. The benchmark is irrelevant to that model class.
3. The provider never ran it (Dye 1985 informed-uncertainty).
4. Providers report a conventional table of fixed size, so omission reflects norm, not strategy.

Four things address this.

**The unit of analysis is a drop, not a gap.** A provider reported benchmark X for model v1, then omitted X for v2, while independent data shows v2 was scored on X. The provider's own earlier release establishes both that the benchmark is relevant to their model class and that they habitually run it. That kills (2) and badly weakens (3). If the reported table stays the same size while X specifically is swapped out, (4) weakens too.

**A temporal gate handles (1).** Epoch publishes `benchmark_release_date` per benchmark. Any benchmark postdating a model's release leaves that model's choice set.

**Omissions are classified, not counted.** We adapt ORBIT (Kirkham et al., *BMJ* 2010;340:c365), the clinical-trials instrument for outcome reporting bias, which was built for exactly this problem: separating "was it measured" from "was it reported given that it was measured." See [`protocol/coding-protocol.md`](protocol/coding-protocol.md).

**A placebo group tests the measure, and testing it is how we found the problem.** Benchmarks postdating a model's release are mechanically impossible to omit strategically — there was nothing to omit. That group cannot carry the disclosed-vs-omitted comparison directly, because everything in it is undisclosed. What it was built to support is **omitted-eligible versus postdating**: both sets are non-disclosures, so any artifact that makes unreported benchmarks look bad for innocent reasons hits both equally and differences out, leaving only the fact that the eligible benchmark could have been reported and was not. Under every innocent explanation that statistic should be zero.

It is not zero. Computed with no disclosure labels at all, eligible benchmarks outscore postdating ones by **12.2 percentile points** within a release, positive in 78% of the 146 releases carrying both sets. That is the estimator's null, and it runs in the direction that manufactures false nulls. Two things carry it. Benchmark composition carries most: the outcome is a within-benchmark rank, but the panel is unbalanced and placebo cells concentrate in benchmarks that entered late, so absorbing release and benchmark effects jointly moves the coefficient from −13.4 to −4.9. The peer window carries the rest: it is symmetric in days, but a benchmark's model coverage begins when the benchmark is built, so a release predating its benchmark is ranked against peers drawn 63% from models newer than itself against 44% for eligible cells. Absorbing both and conditioning on peer-window composition leaves **+0.25 (se 1.62)**, not distinguishable from zero. Restricting to two-sided windows removes only 12%, so it is a conditioning problem rather than a trimming problem. The same asymmetry predicts standing at −29.1 points per unit share among eligible cells alone, so it contaminates the outcome wherever compared groups differ in it.

The estimator is therefore no longer described as identifying. It is reported against its measured null, the placebo group reverts to validating that the coding instrument returns nothing where nothing could have been omitted, and identification moves to the within-release and within-benchmark margin. `python -m src.placebo_calibration` reproduces all of it.

Full design notes, including the identification strategy and residual confounds we cannot rule out, are in [`docs/design.md`](docs/design.md).

## A confound in the data itself

Epoch evaluates API-access models more densely than open-weights models. Reproduced by `src/coverage.py` on the current build: mean 12.35 versus 6.59 benchmarks per release (Mann-Whitney p = 2.9e-07), and the working sample splits 83 API against 59 open. Measured *availability* therefore differs by access type, so raw selectivity rates are not comparable across it. Every comparison in this repository conditions on the number of independent scores per release, and `src/selectivity.py` reports that conditioning explicitly rather than asserting it.

The gap is close to twice as large as earlier builds of this repository recorded, because those counted Epoch's scaffold variants as separate models. API providers ship more reasoning-effort variants, which split their coverage across rows and masked the difference.

## Data

Independent scores come from [Epoch AI's Benchmarking Hub](https://epoch.ai/benchmarks) (CC-BY): 819 model-versions, 74 benchmarks, with columns for organization, release date, model accessibility, and training compute.

Those 819 model-versions are **353 releases**. Epoch's `Model version` splits one shipped model across reasoning-effort and context-window scaffolds — GPT-5.5 appears six times, Claude 3.7 Sonnet ten. A provider publishes one model card per release, not one per scaffold, so the panel is keyed on `(organization, model name, release date)` and scores are maximised across scaffolds. The release date is part of the key because a name alone merges separate disclosure events: "GPT-4o" spans five snapshots across ten months, each with its own release post.

Provider disclosures have no structured source and are read from official release artifacts. An earlier pass assigned the ORBIT categories with a language model; those labels were cleared in `1775c3e` and the sheet now awaits human coding. The coding sheet schema is in [`protocol/`](protocol/).

## Usage

```bash
pip install -r requirements.txt
python -m src.download_data      # fetch Epoch's bundle
python -m src.snapshot           # check it against the pinned build
python -m src.build_matrix       # panel, temporal gate, release collapsing
python -m src.coverage           # the access-type confound
python -m src.families           # seed the family linkage, then hand-review it
python -m src.worklist           # emit the targeted coding worklist
python -m src.date_worklist      # rank the missing benchmark dates by impact

# once data/disclosures.csv is coded:
python -m src.validate_coding    # gate: protocol rules, non-zero exit on failure
python -m src.placebo_calibration  # the placebo null and where it comes from
python -m src.selectivity        # the three estimators
python -m src.falsification      # the four falsification tests
python -m src.reliability        # 20% double-coding draw and Cohen's kappa

pytest                           # 114 tests
```

`build_matrix` produces the release × benchmark eligibility matrix with the temporal gate applied — currently 3,082 pairs across 353 releases, of which 2,355 are eligible and 476 fall in the placebo group. The temporal gate now reaches 91.4% of pairs, up from 45.5% on first reproduction.

`worklist` is what makes the hand-coding finishable. Coding every eligible cell means reading model cards for 1,502 cells, most of which can never produce a drop. A drop is undefined without a predecessor in the same family, so restricting to multi-release families and to releases Epoch scores densely enough cuts the reading to **1,298 cells across 103 releases and 38 families** without discarding a single potential drop. Three providers — OpenAI, Anthropic, Google DeepMind — are most of it. The count rose from 772 as the temporal gate closed: those are pairs previously dropped for unknown vintage, not new work invented. Placebo rows are emitted pre-filled at `reported=0` and cost no reading time.

`selectivity` computes all three estimators; `falsification` runs the four tests; both need the coded sheet and say so until one exists.

The disclosure side has no structured source and does not exist yet. That is the open work, and `data/worklist.csv` is now the exact list of it.

## Which Epoch build

`data/raw/` is gitignored, and Epoch's hub is updated continuously, so the same
code run a few days apart produces different numbers with nothing to signal it.
A rebuild three days after the first pull added twenty model-versions and moved
every headline count in this file.

`data/snapshot.json` is therefore checked in: per-file SHA-256 and row counts for
the build these numbers were computed against. `python -m src.snapshot` compares
the tree on disk against it and names the files that differ. Downloading does not
silently re-pin; `python -m src.download_data --capture` does, and it is a
deliberate act, because it redefines what the reported numbers mean and every
table has to be regenerated behind it.

The build currently pinned is 2026-08-17: 77 files, 819 model-versions. Numbers in
this README that were computed against a later pull are marked where they appear.

## Sample sizes

Filtering on releases with a known organization, a known release date, and at least *n* independent benchmark scores:

- n ≥ 5: 212 releases
- n ≥ 6: 196
- n ≥ 8: 149
- n ≥ 10: 121

The ≥ 8 filter is the working analysis sample. These are lower than earlier builds recorded and describe more data, not less: they count releases rather than scaffold variants.

## Known limitations

Epoch's benchmark metadata carries release dates for 33 of 57 benchmarks. Hand collection has since added **29 of the 41 missing dates**, each with a primary source URL in `data/benchmark_dates.csv`, taking the temporal gate from 45.5% of pairs to **91.4%**. Twelve remain: mostly small creator-run benchmarks (ProofBench, FrontierCode, CL-bench, GBAEval, ExploitBench and others) for which no primary release date could be located. Those rows carry `status=todo-searched` and a note recording what was tried, so the search is not repeated. Two filled rows carry `status=needs-review` because only secondary sources were found, and one carries `status=override`.

Release dates are recorded as the benchmark's first public introduction — arXiv v1, or the launch post where there is no paper — not the date of a later dataset re-release. This matters for at least one case: EnigmaEval's dataset was re-published in 2026 but the benchmark was introduced in February 2025.

**Correcting an earlier claim.** This file previously stated that eight metadata benchmarks — HellaSwag, Winogrande, TriviaQA, ScienceQA, OpenBookQA, CadEval, EBR-Bench, Remote Labor Index — have no score file and are dropped. That was wrong for seven of them. The files ship; the join was failing because Epoch's metadata name and its score filename normalise differently (`HellaSwag` against `hella_swag_external.csv`). Only EBR-bench genuinely has no score file. Recovering the other seven, plus a separate OSWorld file that a bad alias had orphaned, moved date coverage from 45.5% to 53.7% and eligible pairs from 1,856 to 2,109 at version level. An unjoined slug outside an explicit allowlist is now an error rather than a printed line.

Epoch names the score column differently per benchmark; the loader takes the first numeric column after the model identifier, and the maximum across scaffolds both within a file and across a release's scaffold variants. That second collapse touches 16.7% of cells and the scaffolds disagree in 488 of them, so the choice is consequential; `scaffold_spread` is retained in the panel so it can be audited rather than assumed.

Fuller treatment, including the residual confound the design cannot close, is in [`docs/design.md`](docs/design.md).

## Prior work this builds on

The empirical premise is established. Singh et al., [The Leaderboard Illusion](https://arxiv.org/abs/2504.20879) (2025), documents that Meta privately tested 27 Llama-4 variants and disclosed one, and that data access is sharply unequal across providers.

The theory is older than the setting. Grossman (1981) and Milgrom (1981) show verifiable disclosure unravels to full revelation under strong assumptions; Dye (1985) breaks it when receivers are unsure the sender is informed; Hotz & Xiao (*Economic Inquiry* 2013) break it when quality is multidimensional and receivers are heterogeneous. AI benchmark suites are the multidimensional case with the Dye friction layered on.

The method is borrowed from clinical epidemiology. Chan et al. (*JAMA* 2004) compared trial protocols against publications; Kirkham et al. (*BMJ* 2010) built ORBIT from it; Dwan et al. (*PLoS ONE* 2013) established that significant outcomes have higher odds of full reporting. Releases-by-benchmarks is the same matrix as trials-by-outcomes.

## License

Code MIT. Derived data inherits CC-BY from Epoch AI.
