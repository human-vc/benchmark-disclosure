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

Contemporaneity is operationalised as a +/-182 day window around the focal
release, in `src/percentiles.py`, and is reported as a sensitivity rather than
assumed. It is not cosmetic. Ranking against every model Epoch has ever scored
correlates a release's measured standing with its release date at +0.31, since
later models beat earlier ones on shared benchmarks by construction. Windowing
cuts that to +0.08. Release date is the same axis the temporal gate and the
drop design run along, so leaving that correlation in the outcome variable
would have put a date trend inside every statistic here.

The unit is a **release**, not one of Epoch's `Model version` rows. Those split
a shipped model across reasoning-effort and context-window scaffolds -- GPT-5.5
six ways, Claude 3.7 Sonnet ten -- and a provider publishes one artifact per
release. Left split, a single release enters a benchmark's percentile
distribution up to ten times. The key is (organization, model name, release
date); the date is needed because "GPT-4o" alone merges five snapshots across
ten months, each a separate disclosure event.

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
release cannot be strategically omitted; there was nothing to omit.

The obvious construction does not work, and it is worth recording why, because
the first implementation here got it wrong and a test caught it. The placebo
group contains *no disclosures at all* -- every benchmark postdating a release
is undisclosed -- so the disclosed-versus-omitted statistic is undefined inside
it. Comparing the disclosed set against the placebo set instead does run, but
it inherits whatever selection put benchmarks in the disclosed set.

The contrast the placebo group can actually carry is **omitted-eligible versus
placebo**. Both sets are non-disclosures. Any artifact that makes unreported
benchmarks look bad for innocent reasons -- they are harder, they are newer,
Epoch runs them precisely because they discriminate -- applies to both and
differences out. What is left is the single asymmetry that cannot be anything
else: the eligible benchmark was available to report and was not reported.
Under strategic omission that statistic is negative; under irrelevance, never
having run it, or conventional table size, it is zero.

**That last sentence is false on this panel, and the correction is the most
important thing in this document.** Computed with no disclosure labels of any
kind, eligible benchmarks outscore postdating ones by 12.2 percentile points
within a release, positive in 78 percent of the 146 releases carrying both sets.
Twelve points is the estimator's null. The bias runs in the direction that
manufactures false nulls, because concealment has to overcome a twelve-point
head start before the statistic registers anything at all.

The cause is two things at once, and an earlier version of this paragraph got the
split wrong because the code behind it demeaned by release and then by benchmark in
a single pass, which does not remove both sets of effects on an unbalanced panel.
Absorbing them jointly, benchmark composition carries most of the contrast, moving
the coefficient from -13.4 to -4.9. The outcome is a within-benchmark rank, but the
panel is unbalanced and placebo cells concentrate in the benchmarks that entered
late. Peer-window composition carries the rest. The percentile window is symmetric
in days, but a benchmark's model coverage is not: it begins when the benchmark is
built. A release that predates its benchmark therefore sits at the left edge of
that coverage and is ranked against a peer set drawn 63 percent from models newer
than itself, against 44 percent for eligible cells, and it scores low for a reason
that has nothing to do with standing. Conditioning on peer-window asymmetry within release
absorbs 29 percent of the contrast. Absorbing release and benchmark effects jointly
and conditioning on peer-window composition leaves +0.25 with a standard error of
1.62, which is not distinguishable from zero. Restricting to cells with a genuinely
two-sided window removes only 12 percent, so this is a conditioning problem and not
a trimming problem. Peer count belongs in that specification as a joint control and
not as a further channel: entered alone it is a suppressor, making the asymmetry
slope more negative rather than less.

The consequence reaches past this one estimator. Peer-window asymmetry predicts
standing at -29.1 percentile points per unit share among eligible cells alone, so
it contaminates the outcome variable wherever two groups being compared differ in
it. Every contrast in this repository now reports peer-window composition
alongside the number, and `src/placebo_calibration.py` reproduces the whole
decomposition.

The estimator is therefore **not identifying and is no longer described as such**.
It is reported against its measured null rather than against zero, and the placebo
group returns to the narrower role it can support, which is validating that the
coding instrument returns nothing where nothing could have been omitted.
Identification has to come from the within-release and within-benchmark margin,
where the variation is which providers omitted a given benchmark while both the
release's overall standing and the benchmark's identity are held fixed. The
disclosed-minus-omitted gap remains descriptive and is not promoted by this
correction; it is simply no longer outranked by a statistic that does not work.

## Inference

The cluster count here is small and unevenly distributed. The estimation sample
spans 46 organisations, but 24 of them contribute fewer than ten cells, 20 appear
in a single release, and the five largest hold 68 percent of the sample, so the
effective number of clusters is far below the nominal one. Cluster-robust
asymptotics are unreliable in that range and they fail silently rather than
loudly, which is the dangerous property.

Three rules follow, and they are enforced in `src/stats.py` rather than left to
discipline. Below twelve clusters the analytic standard error is not reported at
all; the function returns a point estimate, a missing error and a sentence saying
why. Regression coefficients in that range are tested with the restricted wild
cluster bootstrap of Cameron, Gelbach and Miller, which imposes the null before
resampling and keeps within-cluster correlation intact through cluster-level
Rademacher weights. Release-level means are tested by flipping the signs of whole
clusters, which is exact under its own null and does not degrade as clusters
become scarce, it only becomes granular. With one provider that test returns a
p-value of one, and that is the correct answer rather than a failure of the test.

No result in this repository is called significant on a cluster-t alone.

An earlier version of `stats.py` did none of this. Handed two releases from a
single provider it reported standard errors of zero, t-statistics of order 1e15
and three significance stars. The arithmetic was correct throughout. The
asymptotics it rested on did not exist, and the code had no way to say so, so it
said something confident instead.

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
this repository's build: mean 12.35 versus 6.59 benchmarks per release,
Mann-Whitney p = 2.9e-07, and the ≥8 analysis sample is 83 API against 59 open.
The gap roughly doubled once scaffold variants were collapsed to releases;
API providers ship more reasoning-effort variants, which spread their coverage
across rows and understated the difference.
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

- **Vintage coverage is now mostly closed.** Epoch published dates for 33 of
  57 benchmarks; hand collection added 29 of the 41 missing ones, each with a
  primary source URL in `data/benchmark_dates.csv`, taking the gate from 54% of
  pairs to 91.4%. Twelve remain undated, mostly small creator-run benchmarks,
  and those pairs leave the analysis rather than being imputed.
- **One metadata benchmark has no score file** in the public bundle
  (EBR-bench). It is dropped, not imputed. An earlier version of this document
  listed eight. Seven of those had score files all along and were being lost to
  a slug-normalisation mismatch between Epoch's metadata names and its
  filenames; a separate OSWorld file was orphaned by an alias that merged
  OSWorld with OSWorld 2.0 and would have attributed the 2024 benchmark's date
  to the later one, admitting models to a choice set containing a benchmark
  that did not yet exist. Fixed, and an unjoined slug outside an explicit
  allowlist now raises.
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

## What the code does that this document did not originally specify

Recorded so the gap between design and implementation stays visible.

- **Release-level collapsing.** See the estimand section. Scaffold variants are
  maximised over, which is the rule this document already stated for scaffolds
  within a benchmark file, applied consistently across a release's rows.
- **Provider-clustered inference.** A provider contributes many releases and
  their disclosure habits are not independent draws, so standard errors and
  bootstrap resampling cluster on provider throughout.
- **The coding worklist is targeted.** `src/worklist.py` restricts hand-coding
  to releases inside a multi-release family, since a drop is undefined without
  a predecessor. This cuts the reading from 1,502 cells to 1,346 without
  discarding a potential drop, and is the difference between the coding being
  finishable and not.
- **The estimators are tested against planted effects.** `tests/` builds
  synthetic panels where omission is strategic by construction and where it is
  random, and requires the estimators to separate them. An estimator never
  shown to recover a known signal is not evidence about anything.


## What the completed coding showed

The coding is done: 100 of the 108 worklist releases were read against their
official artifacts and 8 are blocked, which is a distinct state and is recorded
as one. 1,280 cells derive, with 63 drops.

**The identifying test fails, and it fails without any coding at all.** The
contrast this document proposes -- omitted-eligible against postdating -- is
the labelled version of a comparison that needs no labels: every benchmark a
release *could* have reported against every benchmark that did not yet exist.
Both sets are defined by dates alone. That label-free contrast comes out at
+12.22 percentile points across 146 releases, positive in 78.1% of them, with a
provider-clustered interval clear of zero. Under every innocent explanation
this document lists, it should be zero.

So the argument in "A placebo group validates the measure" is wrong in an
instructive way. It assumed that whatever makes non-disclosed benchmarks look
bad for innocent reasons hits both sets equally and differences out. It does
not, because the two sets are not exchangeable on the thing that drives the
measure. A benchmark's model coverage begins when the benchmark was built, so
the comparison window is symmetric in calendar time and one-sided in coverage:
the mean share of a cell's peer window that is newer than the focal model is
0.469 for available cells and 0.707 for postdating ones, against 0.5 for a
two-sided window. Postdating cells sit systematically nearer the left edge of
their benchmark's coverage, are ranked mostly against models newer than
themselves, and therefore stand lower for reasons that have nothing to do with
disclosure.

That is not a fixable nuisance, and this is the part worth carrying forward.
The date deciding whether a benchmark was *available* to a release is the same
date deciding *where in that benchmark's coverage* the release sits. The
variable identifying the endowment is the variable that moves the outcome. A
standing measure defined by a calendar window around the release cannot escape
it, whatever the width. Reweighting the two sides of the window --
`percentile_balanced` -- is a partial repair and cuts the gap to +9.15 without
removing it.

**The drop estimator is small and its interval spans zero.** Twenty dropped
benchmarks across 17 releases, mean +3.5 percentile points, CI [-2.6, +8.6].
That is the one comparison in this design that isolates withholding, and it is
not powered to detect a gap the size of the contamination.

The honest position is therefore the one this document already gestures at in
"The residual confound does not go away", but stronger: no estimate of
selective disclosure is warranted from this design, and the reason is
measurement rather than sample size. What the pipeline does establish is the
obstacle, which is a result about how this question can be asked at all.
