# TASK 5b — Detection done properly

- Mode: FULL · extends T5. delta grid 1e-06..0.99 (14 points).
- **Anti-leakage:** every delta* is frozen from the shuffled null streams (perms 01-10) in `results/calibration.json`, computed *before* any real or synthetic stream is scored. Synthetic streams are never used for calibration (that would be circular). The real stream (perm00) and synthetic streams are only *read* here.

## Problem 1 — extended grid: does the FAR constraint bind?

FAR is monotone non-decreasing in delta on the null streams: **False** (confirms larger delta ⇒ more sensitive detector).

Full FAR-vs-delta on nulls (raw), per signal at its first tokenizer:

| delta | S1 | S1c | S3 | S4 | S7 |
| --- | --- | --- | --- | --- | --- |
| 1e-06 | 8.7e-05 | 8.7e-05 | 0.0e+00 | 8.7e-06 | 0.0e+00 |
| 1e-05 | 8.7e-05 | 8.7e-05 | 0.0e+00 | 1.7e-05 | 3.5e-05 |
| 0.0001 | 8.7e-05 | 8.7e-05 | 0.0e+00 | 3.5e-05 | 4.3e-05 |
| 0.001 | 8.7e-05 | 8.7e-05 | 0.0e+00 | 5.2e-05 | 7.0e-05 |
| 0.01 | 8.7e-05 | 9.6e-05 | 0.0e+00 | 7.8e-05 | 8.7e-05 |
| 0.05 | 9.6e-05 | 1.2e-04 | 5.2e-05 | 1.2e-04 | 1.2e-04 |
| 0.1 | 1.4e-04 | 2.0e-04 | 1.9e-04 | 2.2e-04 | 2.1e-04 |
| 0.2 | 2.8e-04 | 3.3e-04 | 3.3e-04 | 3.5e-04 | 2.6e-04 |
| 0.3 | 3.3e-04 | 4.5e-04 | 4.2e-04 | 4.3e-04 | 4.0e-04 |
| 0.5 | 6.0e-04 | 6.2e-04 | 7.2e-04 | 6.3e-04 | 5.6e-04 |
| 0.7 | 8.0e-04 | 9.1e-04 | 8.9e-04 | 9.0e-04 | 9.0e-04 |
| 0.9 | 1.1e-03 | 1.1e-03 | 1.2e-03 | 1.2e-03 | 1.1e-03 |
| 0.95 | 1.2e-03 | 1.2e-03 | 1.3e-03 | 1.2e-03 | 1.2e-03 |
| 0.99 | 1.2e-03 | 1.3e-03 | 1.4e-03 | 1.3e-03 | 1.3e-03 |

Max FAR reached on the grid, per signal: S1=1.2e-03, S1c=1.3e-03, S3=1.4e-03, S4=1.3e-03, S7=1.3e-03.
**Constraint BINDS** at target 0.001. delta* pinned at grid max (0.99) for: S1 0/5, S1c 0/5, S3 1/5, S4 0/1, S7 0/5.

## Problem 2 — full 5×5 delay grid on COVID (no 'best' selection)

Detection delay in **days** from t*=2020-03-08 (first alarm at/after t*), frozen delta* at target FAR 0.001:

| signal | bert-base-multilingual-cased | xlm-roberta-base | Llama-3.2-1B | Qwen2.5-0.5B | bloom-560m |
| --- | --- | --- | --- | --- | --- |
| S1 | 59 | 12 | 50 | 50 | 12 |
| S1c | 246 | 4 | 64 | 64 | 7 |
| S3 | 64 | 4 | 68 | 64 | 12 |
| S4 (shared) | 7 | 7 | 7 | 7 | 7 |
| S7 | 7 | 4 | 4 | 4 | 7 |

Median [IQR] delay across tokenizers, per signal (S4 is a single value):

| signal | median | IQR | n detected |
| --- | --- | --- | --- |
| S1 | 50 | [12, 50] | 5/5 |
| S1c | 64 | [7, 64] | 5/5 |
| S3 | 64 | [12, 64] | 5/5 |
| S4 | 7 | (single value) | 1 |
| S7 | 4 | [4, 7] | 5/5 |

**Paired S7 vs S4** across 5 tokenizers (same S4 value paired against each tokenizer's S7): signs = ['tie/na', 'S7<S4', 'S7<S4', 'S7<S4', 'tie/na']; Wilcoxon W=0.0, p=0.250. With only 5 pairs this is under-powered — report, don't over-claim.

## Problem 3a — semi-synthetic drift injection

Early pool = 2016-2017 (39,960 docs), late pool = 2020 (19,980 docs). Each synthetic stream is 2000 windows; before W* only early-pool docs, from W* each doc is late w.p. p (drift intensity). W* drawn uniformly from the middle 60%; 20 replicates/intensity, sampled without replacement within a stream (pools are large enough). Streams are shorter than the real detection epoch (documented deviation) to keep 5×20 replicates within budget. delta* is the frozen null-calibrated value; synthetic streams are never calibrated on.

**Detection power** (fraction of replicates × tokenizers detecting), per signal × intensity:

| signal | p=0.05 | p=0.1 | p=0.25 | p=0.5 | p=1 |
| --- | --- | --- | --- | --- | --- |
| S1 | 0.55 | 0.63 | 0.67 | 0.60 | 0.72 |
| S1c | 0.64 | 0.58 | 0.65 | 0.68 | 0.80 |
| S3 | 0.54 | 0.62 | 0.71 | 0.76 | 0.89 |
| S4 | 0.45 | 0.50 | 0.75 | 0.85 | 1.00 |
| S7 | 0.53 | 0.48 | 0.83 | 0.96 | 1.00 |

**Median detection delay in windows** [95% bootstrap CI], per signal × intensity (— = never detected):

| signal | p=0.05 | p=0.1 | p=0.25 | p=0.5 | p=1 |
| --- | --- | --- | --- | --- | --- |
| S1 | 300 [193,415] | 389 [289,417] | 390 [288,410] | 266 [209,360] | 167 [116,201] |
| S1c | 415 [318,492] | 399 [309,417] | 391 [358,410] | 410 [198,446] | 208 [147,284] |
| S3 | 411 [268,520] | 396 [299,417] | 391 [295,410] | 294 [243,437] | 179 [122,222] |
| S4 | 415 [109,990] | 403 [210,508] | 218 [121,365] | 126 [72,147] | 38 [30,54] |
| S7 | 415 [337,461] | 417 [389,508] | 218 [167,326] | 94 [72,103] | 38 [30,44] |

**Intensity p reaching ≥80% detection power**, per signal:

| signal | S1 | S1c | S3 | S4 | S7 |
| --- | --- | --- | --- | --- | --- |
| p@80% | >1.0 | 1.0 | 1.0 | 0.5 | 0.25 |

**Ordering check (S7 vs S4 vs S1) with CI overlap:**

- p=0.05: S1: 300[193,415]; S4: 415[109,990]; S7: 415[337,461] → S7 vs S4 CIs OVERLAP (indistinguishable).
- p=0.1: S1: 389[289,417]; S4: 403[210,508]; S7: 417[389,508] → S7 vs S4 CIs OVERLAP (indistinguishable).
- p=0.25: S1: 390[288,410]; S4: 218[121,365]; S7: 218[167,326] → S7 vs S4 CIs OVERLAP (indistinguishable).
- p=0.5: S1: 266[209,360]; S4: 126[72,147]; S7: 94[72,103] → S7 vs S4 CIs OVERLAP (indistinguishable).
- p=1: S1: 167[116,201]; S4: 38[30,54]; S7: 38[30,44] → S7 vs S4 CIs OVERLAP (indistinguishable).

## Problem 3b — multiple real changepoints

All 7 candidate dates verified against cited sources (see `params.yaml`):

| event | date | source |
| --- | --- | --- |
| rohingya_influx | 2017-08-25 | Crisis Group / CFR — ARSA attacks 25 Aug 2017 triggered exodus |
| khaleda_zia_jailed | 2018-02-08 | The Daily Star / Business Standard — sentenced 8 Feb 2018 |
| road_safety_protests | 2018-07-29 | Wikipedia / TIME / NPR — bus killed 2 students 29 Jul 2018 |
| parliamentary_election | 2018-12-30 | IPU Parline / APF Canada — 11th election 30 Dec 2018 |
| nusrat_rafi_murder | 2019-04-10 | HRW / CBS — attacked 6 Apr, died 10 Apr 2019 |
| abrar_fahad_killing | 2019-10-07 | Wikipedia / The Daily Star — killed night 6-7 Oct 2019 |
| covid_first_cases | 2020-03-08 | IEDCR / Frontiers Public Health — first 3 cases 8 Mar 2020 |

Delay in **days** to first alarm at/after each event (frozen delta*, real stream, median across tokenizers per signal; S4 single value):

| event | S1 | S1c | S3 | S4 | S7 |
| --- | --- | --- | --- | --- | --- |
| rohingya_influx | 25 | 25 | 46 | 37 | 22 |
| khaleda_zia_jailed | 36 | 36 | 36 | 23 | 41 |
| road_safety_protests | 44 | 44 | 44 | 91 | 62 |
| parliamentary_election | 54 | 67 | 28 | 19 | 19 |
| nusrat_rafi_murder | 198 | 42 | 33 | 92 | 102 |
| abrar_fahad_killing | 37 | 18 | 50 | 1 | 1 |
| covid_first_cases | 50 | 64 | 64 | 7 | 4 |

> Events differ in lexical footprint — a national election introduces vocabulary very differently from a pandemic — so cross-event delays are **not strictly commensurable**. The synthetic experiment (3a) exists because it is. Events where no signal alarms are reported (—), not dropped.

## Problem 4 — do the covariates actually *trend*? (variance ≠ drift)

Per-covariate: yearly mean, and a linear trend (slope per year) with p-value over the real stream:

| covariate | 2016 | 2017 | 2018 | 2019 | 2020 | slope/yr | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mean_word_len | 5.260 | 5.274 | 5.306 | 5.297 | 5.312 | +0.0128 | 3.5e-16 |
| mean_doc_words | 322.958 | 346.212 | 313.269 | 312.743 | 279.794 | -11.8488 | 5.3e-34 |
| ttr | 0.527 | 0.525 | 0.529 | 0.528 | 0.531 | +0.0011 | 2.3e-06 |
| top1000_share | 0.529 | 0.527 | 0.528 | 0.525 | 0.521 | -0.0018 | 4.6e-12 |
| latin_share | 0.014 | 0.014 | 0.012 | 0.014 | 0.017 | +0.0006 | 1.4e-01 |
| mean_freq_rank | 8086.225 | 8859.996 | 8912.423 | 9110.056 | 9744.267 | +353.9699 | 3.0e-127 |

Share of observed Δz(S1) (reference→final-10%) attributable to the covariates (OLS of z(S1) on the six covariates, fit on the reference epoch, applied forward):

| tokenizer | R²(ref fit) | covariate-predicted Δz(S1) | observed Δz(S1) | ratio |
| --- | --- | --- | --- | --- |
| bert-base-multilingual-cased | 0.61 | +0.226 | +0.122 | +1.86 |
| xlm-roberta-base | 0.51 | +0.169 | -0.260 | -0.65 |
| Llama-3.2-1B | 0.96 | +0.223 | +0.201 | +1.11 |
| Qwen2.5-0.5B | 0.95 | +0.216 | +0.185 | +1.17 |
| bloom-560m | 0.39 | +0.083 | -0.491 | -0.17 |

`latin_share` trend: slope +0.0006/yr (p=1.4e-01) — rising over time. The T5 correlation of ~−0.87 for Llama/Qwen means Latin-script share moves opposite to their z(S1); whether it *explains drift* depends on whether it trends (above), not just correlates.

Covariates with a material time trend (p<0.05): **mean_word_len, mean_doc_words, ttr, top1000_share, mean_freq_rank**.
But for the tokenizers whose S1 *falls* (xlm-roberta-base, bloom-560m), the covariate model predicts a **rise** (ratio<0 in the table): the six covariates explain S1 for the byte-level tokenizers (which rise, R²≈0.95) but do **not** explain the sign flip. `latin_share` correlates but does not trend (p>0.1). So the flip remains unexplained by these covariates — the honest correction to T5's claim.

## STATUS

```
GATE 1 — delta grid extended to 0.99; FAR-vs-delta curve reported in full:        PASS
GATE 2 — does the FAR constraint bind anywhere on the grid?                        BINDS
GATE 3 — full 5x5 delay grid reported; no 'best tokenizer' selection anywhere:     PASS
GATE 4 — synthetic injection: 5 intensities x 20 replicates, all signals:   PASS
GATE 5 — >=4 real changepoint dates verified against a cited source:             PASS (7 verified)
GATE 6 — delta* frozen from null streams only; no synthetic/real-stream leakage:   PASS

THE ORDERING, under correct treatment:
    paired S7 vs S4 across 5 tokenizers: signs=['tie/na', 'S7<S4', 'S7<S4', 'S7<S4', 'tie/na'], Wilcoxon W=0.0, p=0.250
    median delay [IQR] across tokenizers on COVID (days):
        S1: 50 [12,50]
        S1c: 64 [7,64]
        S3: 64 [12,64]
        S4: 7 (single value)
        S7: 4 [4,7]
    p for 80% detection power, per signal: S1=>1.0, S1c=1.0, S3=1.0, S4=0.5, S7=0.25
    do the CIs separate S7 from S4?  NO (by intensity: 0.05:overlap, 0.1:overlap, 0.25:overlap, 0.5:overlap, 1:overlap)

THE ANOMALY:
    covariate trends explaining S1 sign flip: covariates trend and explain S1 for byte-level tokenizers, but do NOT explain the XLM-R/BLOOM sign flip (predict rise, observe fall)

VERDICT: PROCEED WITH CAVEATS
Blockers:
  - none
Surprises worth a human decision:
  - Under correct treatment S7 and S4 are INDISTINGUISHABLE (CIs overlap at every intensity). The T5 headline ranking does not survive — the honest result is a clean negative with a mechanism.
```
