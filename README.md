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

**A placebo group tests the measure.** Benchmarks postdating a model's release are mechanically impossible to omit strategically. The selectivity statistic computed on that group should be indistinguishable from zero. If it isn't, the measure is picking up something other than concealment.

Full design notes, including the identification strategy and residual confounds we cannot rule out, are in [`docs/design.md`](docs/design.md).

## A confound in the data itself

Epoch evaluates API-access models more densely than open-weights models. Reproduced by `src/coverage.py` on the current build: mean 6.36 versus 4.95 benchmarks per model (Mann-Whitney p = 2.6e-03), and the working sample splits 128 API against 52 open. Measured *availability* therefore differs by access type, so raw selectivity rates are not comparable across it. Every comparison in this repository conditions on the number of independent scores per model.

## Data

Independent scores come from [Epoch AI's Benchmarking Hub](https://epoch.ai/benchmarks) (CC-BY): 819 model-versions, 344 base models, 74 benchmarks, with columns for organization, release date, model accessibility, and training compute.

Provider disclosures have no structured source and are hand-coded from official release artifacts. The coding sheet schema is in [`protocol/`](protocol/).

## Usage

```bash
pip install -r requirements.txt
python -m src.download_data
python -m src.build_matrix
python -m src.selectivity
```

`build_matrix` produces the model × benchmark eligibility matrix with the temporal gate applied — currently 4,448 model-benchmark pairs, of which 1,826 are eligible and 234 fall in the placebo group. `coverage` runs the diagnostics above. `selectivity` computes the reported-vs-omitted percentile gap, conditioned on coverage, with a permutation test and the placebo comparison; it needs the hand-coded disclosure sheet and will tell you so until one exists.

The disclosure side has no structured source and does not exist yet. That is the open work.

## Sample sizes

Filtering on models with a known organization, a known release date, and at least *n* independent benchmark scores:

- n ≥ 5: 340 model-versions
- n ≥ 6: 288
- n ≥ 8: 188
- n ≥ 10: 127

The ≥ 8 filter is the working analysis sample.

## Known limitations

Epoch's benchmark metadata carries release dates for 33 of 57 benchmarks, so the temporal gate currently applies to about 46% of pairs. Filling the remaining dates by hand from arXiv and release pages is the highest-value open task here. Eight metadata benchmarks have no score file in the public bundle and are dropped rather than imputed. Epoch names the score column differently per benchmark; the loader takes the first numeric column after the model identifier and the maximum across scaffolds, which is a judgment call that moves the sample by a model or two against a hand-built alternative.

Fuller treatment, including the residual confound the design cannot close, is in [`docs/design.md`](docs/design.md).

## Prior work this builds on

The empirical premise is established. Singh et al., [The Leaderboard Illusion](https://arxiv.org/abs/2504.20879) (2025), documents that Meta privately tested 27 Llama-4 variants and disclosed one, and that data access is sharply unequal across providers.

The theory is older than the setting. Grossman (1981) and Milgrom (1981) show verifiable disclosure unravels to full revelation under strong assumptions; Dye (1985) breaks it when receivers are unsure the sender is informed; Hotz & Xiao (*Economic Inquiry* 2013) break it when quality is multidimensional and receivers are heterogeneous. AI benchmark suites are the multidimensional case with the Dye friction layered on.

The method is borrowed from clinical epidemiology. Chan et al. (*JAMA* 2004) compared trial protocols against publications; Kirkham et al. (*BMJ* 2010) built ORBIT from it; Dwan et al. (*PLoS ONE* 2013) established that significant outcomes have higher odds of full reporting. Releases-by-benchmarks is the same matrix as trials-by-outcomes.

## License

Code MIT. Derived data inherits CC-BY from Epoch AI.
