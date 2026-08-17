# Research design

## Claim under test

Providers report benchmark results truthfully and selectively, and selection
alone is enough to distort model choice.

The first half is not an assumption. Epoch AI independently reran GPQA Diamond
against provider self-reports across eight labs and found no significant
discrepancy. That result does more work than it appears to: it removes the
fraud story, and what remains is the verifiable-disclosure problem in its
classical form. Every number a provider publishes is true. The question is
which true numbers get published.

## Estimand

For a model release *i* with eligible benchmark set K_i and disclosed subset
S_i, the object of interest is whether the model's standing on the omitted
benchmarks K_i \ S_i is systematically worse than on the disclosed ones, after
conditioning on the model's overall capability level.

Percentiles are taken within benchmark across contemporaneous models, so
"standing" means position in the field rather than raw score, which is not
comparable across benchmarks with different scales and difficulties.

## Why the naive comparison identifies nothing

Non-disclosure has at least four innocent explanations observationally
identical to strategic omission.

1. **Vintage.** The benchmark postdates the model.
2. **Relevance.** The benchmark does not apply to that model class.
3. **Informed uncertainty.** The provider never ran it. This is Dye (1985),
   and it is a feature of the theory and a bug for the empirics.
4. **Convention.** Providers report a table of conventional size, so omission
   reflects a norm rather than a choice.

A study that reports raw gap rates has measured relevance and convention.

## The four things that address it

**The unit of analysis is a drop.** A provider reported benchmark X for model
v1, then omitted X for v2 in the same family, with X still eligible and an
independent score for v2 in hand. The provider's own earlier release
establishes relevance and establishes that they run it, which removes (2) and
badly weakens (3). If the reported table holds its size while X specifically is
swapped out, (4) weakens as well. This is the disclosure-cessation design from
empirical accounting, transplanted.

**A temporal gate removes (1).** Epoch publishes `benchmark_release_date`. Any
benchmark postdating a model leaves that model's choice set. This is a filter
on the same file that supplies the outcome variable, not new data collection.

**Omissions are classified, not counted.** See
[`../protocol/coding-protocol.md`](../protocol/coding-protocol.md). The ORBIT
instrument separates "was it measured" from "was it reported given that it was
measured," which is precisely the split (3) demands.

**A placebo group validates the measure.** Benchmarks postdating a model's
release cannot be strategically omitted; there was nothing to omit. The
selectivity statistic computed on that group should be indistinguishable from
zero. If it is not, the statistic is capturing something other than
concealment. The panel builder emits this group explicitly.

## Falsification

- **Placebo group**, as above.
- **Excess significance.** If omission is strategic, omitted-benchmark
  percentiles should sit below what the model's disclosed benchmarks predict.
  If innocent, they should be unrelated to the model's level. Stanley,
  Doucouliagos, Ioannidis & Carter (*Research Synthesis Methods* 2021) show the
  test for excess significance outperforms Egger's test and three-parameter
  selection models in simulation.
- **Permutation.** Relabel dropped benchmarks at random within provider and
  recompute. The observed statistic should sit outside the permutation null.
- **Reverse gap.** Benchmarks a provider reports that the independent source
  does not cover are recorded rather than discarded. Under a pure concealment
  story that set should not be systematically favourable.

## A confound inside the data

Epoch evaluates API-access models more densely than open-weights models. In
this repository's build: mean 6.36 versus 4.95 benchmarks per model,
Mann-Whitney p = 2.6e-03, and the ≥8 analysis sample is 128 API against 52 open.
Measured *availability* therefore varies with access type. Any comparison
across that margin conditions on the number of independent scores. Raw rates
are not interpretable.

A six-model pilot found no support for lower selectivity among open-weights
providers, and the largest single gap belonged to an open-weights model. The
open-versus-closed question is reported as a secondary result and may be a null.

## Identification

The cross-section is descriptive. Two sources of variation could support a
causal reading, and both carry a caveat worth stating before the data is cut.

**Code of Practice signature.** The EU AI Act's GPAI obligations applied from
2 August 2025. Twenty-one firms signed the General-Purpose AI Code of Practice
on 10 July 2025; Meta declined on 18 July; xAI signed the Safety and Security
chapter but not the Transparency chapter. That last case is treatment intensity
on exactly the margin this study measures.

**Compute threshold.** Article 51 presumes systemic risk above 10^25 FLOP of
cumulative training compute, which triggers the Article 55 obligations. Epoch's
index carries a `Training compute (FLOP)` column, so the running variable is in
hand. California SB 53's 10^26 FLOP threshold, effective 1 January 2026, gives a
second cutoff for triangulation.

**The caveat.** The Annex XI duty to document evaluation results runs to the AI
Office and national authorities on request. It is not an automatic public
disclosure mandate. The link from crossing a threshold to publishing more in a
public model card is behavioural, not legislated, and has to be established
rather than assumed. Late 2025 is also crowded with other shocks, so a clean
untreated control barely exists among frontier labs. Within-firm variation, the
same provider's models on either side of a cutoff, is the version worth
attempting.

The closest methodological template is Breuer, Hombach & Müller, "When You
Talk, I Remain Silent," *The Accounting Review* 2022, which studies firms either
side of a regulatory disclosure threshold and finds mandated disclosure crowds
out voluntary disclosure by unregulated peers. That is a live competing
hypothesis here, not just a citation, and it should be committed to in advance.

## Known limitations

- **Vintage coverage is partial.** Epoch's benchmark metadata carries release
  dates for 33 of 57 benchmarks, so the temporal gate currently applies to
  about 46% of pairs. The remaining dates are hand-fillable from arXiv and
  release pages and this is the highest-value open task in the repository.
- **Eight metadata benchmarks have no score file** in the public bundle
  (HellaSwag, Winogrande, TriviaQA, ScienceQA, OpenBookQA, CadEval, EBR-Bench,
  Remote Labor Index). They are dropped, not imputed.
- **Score columns are heterogeneous.** Epoch names the score column per
  benchmark. The loader takes the first numeric column after the model
  identifier, preferring the explicit best-score column where present, and
  takes the maximum across scaffolds where a benchmark reports several. This
  is a judgment call and it moves the sample size by one or two models against
  a hand-built alternative.
- **The residual confound does not go away.** A provider who ran an evaluation
  on v1, genuinely did not re-run it on v2 for reasons uncorrelated with the
  expected score, and happened to skip one where v2 would have looked bad, is
  indistinguishable from a concealer. No design without internal evaluation
  logs closes that gap. The claim available is that the joint pattern across
  many drops, providers, and benchmarks is hard to generate at the required
  rate from independent score-uncorrelated events.
