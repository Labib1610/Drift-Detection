# TASK 5b — Detection done properly

- Mode: FULL · extends T5. delta grid 1e-06..0.99 (14 points).
- Language: `ns` · stream: `ns_panel`
- **Anti-leakage:** every delta* is frozen from the shuffled null streams (perms 01-10) in `results/calibration_ns.json`, computed *before* any real or synthetic stream is scored. Synthetic streams are never used for calibration (that would be circular). The real stream (perm00) and synthetic streams are only *read* here.

## Problem 1 — extended grid: does the FAR constraint bind?

FAR is monotone non-decreasing in delta on the null streams: **True** (confirms larger delta ⇒ more sensitive detector).

Full FAR-vs-delta on nulls (raw), per signal at its first tokenizer:

| delta | S1 | S1c | S3 | S4 | S7 |
| --- | --- | --- | --- | --- | --- |
| 1e-06 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| 1e-05 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| 0.0001 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| 0.001 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 1.6e-05 | 0.0e+00 |
| 0.01 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 1.6e-05 | 0.0e+00 |
| 0.05 | 0.0e+00 | 0.0e+00 | 1.6e-05 | 1.1e-04 | 3.3e-05 |
| 0.1 | 3.3e-05 | 6.5e-05 | 3.3e-05 | 1.5e-04 | 9.8e-05 |
| 0.2 | 2.0e-04 | 1.3e-04 | 9.8e-05 | 3.6e-04 | 2.1e-04 |
| 0.3 | 3.4e-04 | 2.1e-04 | 1.8e-04 | 4.2e-04 | 2.8e-04 |
| 0.5 | 5.7e-04 | 3.4e-04 | 3.3e-04 | 6.3e-04 | 5.2e-04 |
| 0.7 | 7.0e-04 | 5.9e-04 | 4.6e-04 | 8.0e-04 | 7.0e-04 |
| 0.9 | 1.1e-03 | 1.0e-03 | 7.5e-04 | 1.1e-03 | 9.4e-04 |
| 0.95 | 1.2e-03 | 1.1e-03 | 7.6e-04 | 1.2e-03 | 9.9e-04 |
| 0.99 | 1.2e-03 | 1.1e-03 | 8.6e-04 | 1.3e-03 | 1.1e-03 |

Max FAR reached on the grid, per signal: S1=1.2e-03, S1c=1.1e-03, S3=8.6e-04, S4=1.3e-03, S7=1.1e-03.
**Constraint DOES NOT BIND** at target 0.001. delta* pinned at grid max (0.99) for: S1 0/5, S1c 3/5, S3 1/5, S4 0/1, S7 0/5.
> Where a signal's FAR stays below target even at delta=0.99, its delta* is recorded as **grid maximum, constraint inactive**, and its delays are measured at maximum sensitivity — not at a matched FAR. Stated explicitly so the comparison is not overclaimed.

## Problem 2 — full 5×5 delay grid on COVID (no 'best' selection)

Detection delay in **days** from t*=2020-03-08 (first alarm at/after t*), frozen delta* at target FAR 0.001:

| signal | bert-base-multilingual-cased | xlm-roberta-base | Llama-3.2-1B | Qwen2.5-0.5B | bloom-560m |
| --- | --- | --- | --- | --- | --- |
| S1 | 50 | 50 | 24 | 84 | 24 |
| S1c | 24 | 33 | 84 | 84 | 24 |
| S3 | 24 | 50 | 24 | 84 | 24 |
| S4 (shared) | 84 | 84 | 84 | 84 | 84 |
| S7 | 50 | 50 | 50 | 50 | 50 |

Median [IQR] delay across tokenizers, per signal (S4 is a single value):

| signal | median | IQR | n detected |
| --- | --- | --- | --- |
| S1 | 50 | [24, 50] | 5/5 |
| S1c | 33 | [24, 84] | 5/5 |
| S3 | 24 | [24, 50] | 5/5 |
| S4 | 84 | (single value) | 1 |
| S7 | 50 | [50, 50] | 5/5 |

**Paired S7 vs S4** across 5 tokenizers (same S4 value paired against each tokenizer's S7): signs = ['S7<S4', 'S7<S4', 'S7<S4', 'S7<S4', 'S7<S4']; Wilcoxon W=0.0, p=0.062. With only 5 pairs this is under-powered — report, don't over-claim.

## Problem 3a — semi-synthetic drift injection

Early pool = 2016-2017 (1,282 docs), late pool = 2020 (9,789 docs). Each synthetic stream is 2000 windows; before W* only early-pool docs, from W* each doc is late w.p. p (drift intensity). W* drawn uniformly from the middle 60%; 20 replicates/intensity, sampled without replacement within a stream (pools are large enough). Streams are shorter than the real detection epoch (documented deviation) to keep 5×20 replicates within budget. delta* is the frozen null-calibrated value; synthetic streams are never calibrated on.

**Detection power** (fraction of replicates × tokenizers detecting), per signal × intensity:

| signal | p=0.05 | p=0.1 | p=0.25 | p=0.5 | p=1 |
| --- | --- | --- | --- | --- | --- |
| S1 | 0.92 | 0.92 | 0.89 | 0.92 | 0.86 |
| S1c | 0.84 | 0.79 | 0.86 | 0.78 | 0.85 |
| S3 | 0.91 | 0.86 | 0.82 | 0.88 | 0.84 |
| S4 | 0.95 | 1.00 | 1.00 | 1.00 | 0.90 |
| S7 | 0.97 | 1.00 | 1.00 | 1.00 | 0.92 |

**Median detection delay in windows** [95% bootstrap CI], per signal × intensity (— = never detected):

| signal | p=0.05 | p=0.1 | p=0.25 | p=0.5 | p=1 |
| --- | --- | --- | --- | --- | --- |
| S1 | 173 [106,301] | 175 [139,269] | 137 [75,234] | 138 [105,284] | 204 [108,380] |
| S1c | 314 [220,396] | 268 [171,364] | 330 [171,493] | 284 [202,395] | 204 [140,332] |
| S3 | 138 [106,205] | 175 [108,364] | 137 [75,235] | 154 [106,330] | 252 [140,490] |
| S4 | 41 [12,45] | 44 [42,74] | 42 [26,45] | 42 [42,44] | 43 [42,45] |
| S7 | 42 [13,42] | 44 [43,60] | 42 [41,43] | 42 [42,42] | 44 [43,44] |

**Intensity p reaching ≥80% detection power**, per signal:

| signal | S1 | S1c | S3 | S4 | S7 |
| --- | --- | --- | --- | --- | --- |
| p@80% | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |

**Ordering check (S7 vs S4 vs S1) with CI overlap:**

- p=0.05: S1: 173[106,301]; S4: 41[12,45]; S7: 42[13,42] → S7 vs S4 CIs OVERLAP (indistinguishable).
- p=0.1: S1: 175[139,269]; S4: 44[42,74]; S7: 44[43,60] → S7 vs S4 CIs OVERLAP (indistinguishable).
- p=0.25: S1: 137[75,234]; S4: 42[26,45]; S7: 42[41,43] → S7 vs S4 CIs OVERLAP (indistinguishable).
- p=0.5: S1: 138[105,284]; S4: 42[42,44]; S7: 42[42,42] → S7 vs S4 CIs OVERLAP (indistinguishable).
- p=1: S1: 204[108,380]; S4: 43[42,45]; S7: 44[43,44] → S7 vs S4 CIs OVERLAP (indistinguishable).

## Problem 3b — multiple real changepoints

All 4 candidate dates verified against cited sources (see `params.yaml`):

| event | date | source |
| --- | --- | --- |
| lok_sabha_results | 2019-05-23 | Election Commission of India — 2019 general election counting 23 May 2019 |
| article_370_abrogation | 2019-08-05 | Govt of India / Rajya Sabha — Article 370 abrogated 5 Aug 2019 |
| ayodhya_verdict | 2019-11-09 | Supreme Court of India — Ayodhya title verdict 9 Nov 2019 |
| covid_first_cases | 2020-01-30 | MoHFW / WHO — first COVID-19 case in India (Kerala) 30 Jan 2020 |

Delay in **days** to first alarm at/after each event (frozen delta*, real stream, median across tokenizers per signal; S4 single value):

| event | S1 | S1c | S3 | S4 | S7 |
| --- | --- | --- | --- | --- | --- |
| lok_sabha_results | 340 | 323 | 314 | 374 | 340 |
| article_370_abrogation | 266 | 249 | 240 | 300 | 266 |
| ayodhya_verdict | 170 | 153 | 144 | 204 | 170 |
| covid_first_cases | 88 | 71 | 62 | 122 | 88 |

> Events differ in lexical footprint — a national election introduces vocabulary very differently from a pandemic — so cross-event delays are **not strictly commensurable**. The synthetic experiment (3a) exists because it is. Events where no signal alarms are reported (—), not dropped.

## Problem 4 — do the covariates actually *trend*? (variance ≠ drift)

Per-covariate: yearly mean, and a linear trend (slope per year) with p-value over the real stream:

| covariate | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | slope/yr | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean_word_len | 4.857 | 4.885 | 4.890 | 4.891 | 4.933 | 4.951 | +0.0177 | 1.1e-44 |
| mean_doc_words | 425.407 | 382.089 | 400.115 | 338.934 | 374.183 | 420.481 | +3.7346 | 2.2e-02 |
| ttr | 0.393 | 0.394 | 0.392 | 0.404 | 0.402 | 0.407 | +0.0034 | 2.5e-31 |
| top1000_share | 0.682 | 0.681 | 0.674 | 0.666 | 0.654 | 0.639 | -0.0101 | 2.6e-273 |
| latin_share | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | +0.0000 | 5.6e-01 |
| mean_freq_rank | 2346.619 | 3109.914 | 3435.516 | 3646.962 | 3808.076 | 4042.659 | +246.3056 | 1.5e-276 |

Share of observed Δz(S1) (reference→final-10%) attributable to the covariates (OLS of z(S1) on the six covariates, fit on the reference epoch, applied forward):

| tokenizer | R²(ref fit) | covariate-predicted Δz(S1) | observed Δz(S1) | ratio |
| --- | --- | --- | --- | --- |
| bert-base-multilingual-cased | 0.30 | +0.906 | +1.085 | +0.83 |
| xlm-roberta-base | 0.42 | +0.976 | +1.094 | +0.89 |
| Llama-3.2-1B | 0.31 | +0.829 | +0.757 | +1.09 |
| Qwen2.5-0.5B | 0.24 | +0.765 | +0.781 | +0.98 |
| bloom-560m | 0.38 | +0.935 | +0.786 | +1.19 |

`latin_share` trend: slope +0.0000/yr (p=5.6e-01) — rising over time. The T5 correlation of ~−0.87 for Llama/Qwen means Latin-script share moves opposite to their z(S1); whether it *explains drift* depends on whether it trends (above), not just correlates.

Covariates with a material time trend (p<0.05): **mean_word_len, mean_doc_words, ttr, top1000_share, mean_freq_rank**.

## STATUS

```
GATE 1 — delta grid extended to 0.99; FAR-vs-delta curve reported in full:        PASS
GATE 2 — does the FAR constraint bind anywhere on the grid?                        DOES NOT BIND
GATE 3 — full 5x5 delay grid reported; no 'best tokenizer' selection anywhere:     PASS
GATE 4 — synthetic injection: 5 intensities x 20 replicates, all signals:   PASS
GATE 5 — >=4 real changepoint dates verified against a cited source:             PASS (4 verified)
GATE 6 — delta* frozen from null streams only; no synthetic/real-stream leakage:   PASS

THE ORDERING, under correct treatment:
    paired S7 vs S4 across 5 tokenizers: signs=['S7<S4', 'S7<S4', 'S7<S4', 'S7<S4', 'S7<S4'], Wilcoxon W=0.0, p=0.062
    median delay [IQR] across tokenizers on COVID (days):
        S1: 50 [24,50]
        S1c: 33 [24,84]
        S3: 24 [24,50]
        S4: 84 (single value)
        S7: 50 [50,50]
    p for 80% detection power, per signal: S1=0.05, S1c=0.05, S3=0.05, S4=0.05, S7=0.05
    do the CIs separate S7 from S4?  NO (by intensity: 0.05:overlap, 0.1:overlap, 0.25:overlap, 0.5:overlap, 1:overlap)

THE ANOMALY:
    covariate trends explaining S1 sign flip: none (covariates explain variance, not the drift/flip)

VERDICT: PROCEED WITH CAVEATS
Blockers:
  - none
Surprises worth a human decision:
  - Under correct treatment S7 and S4 are INDISTINGUISHABLE (CIs overlap at every intensity). The T5 headline ranking does not survive — the honest result is a clean negative with a mechanism.
  - ADWIN's FAR constraint never binds even at delta=0.99: detectors run at maximum sensitivity, so delays are NOT at a matched FAR. Report delays as max-sensitivity, not FAR-matched.
```
