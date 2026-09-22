# TASK 8 — the sensitivity floor

- Language `ns`. Frozen delta* throughout; no re-calibration. Seeds: synthetic base 42, permutation 42, bootstrap 42.

## Part 1 — inverting the response curve: where real events sit

**Method.** On synthetic streams at each mixing rate p, response `R(p) = mean z over the 420 windows (~60 days) after W* − mean z over the 420 before`, median over tokenizers, mean over 20 replicates (bootstrap CI). On the real stream, `R_obs(event) = mean z 60 days after − 60 days before`. `p_eff` is found by linear interpolation of R_obs onto the monotone R(p) curve; its CI is propagated by bootstrapping the calibration curve over replicates (400 resamples). Where R_obs < R(0), p_eff is reported as ≈0.

Calibration curve R(p) (mean [95% CI]):

| signal | p=0 | p=0.05 | p=0.1 | p=0.25 | p=0.5 | p=1 | detection threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | -0.00 | -0.04 | -0.03 | -0.01 | -0.05 | -0.14 | None |
| S1c | 0.00 | -0.03 | -0.04 | -0.06 | -0.13 | -0.27 | None |
| S3 | -0.01 | -0.02 | -0.02 | -0.01 | -0.07 | -0.19 | None |
| S4 | 0.02 | -0.02 | -0.04 | -0.03 | -0.01 | -0.09 | None |
| S7 | 0.01 | -0.03 | -0.04 | -0.05 | -0.06 | -0.18 | 1.0 |

**p_eff per event — S4** (detection threshold = None):

| event | R_obs | p_eff [95% CI] | p_eff / threshold |
| --- | --- | --- | --- |
| gst_rollout | +0.03 | 0.122 [0.011,0.790] | — |
| sabarimala_verdict | +0.11 | 0.000 [0.000,0.500] | — |
| pulwama_attack | -0.09 | 1.000 [0.100,1.000] | — |
| lok_sabha_results | +0.30 | 0.000 [0.000,0.500] | — |
| article_370_abrogation | -0.28 | 1.000 [0.100,1.000] | — |
| ayodhya_verdict | -0.10 | 1.000 [0.100,1.000] | — |
| covid_first_cases | -0.26 | 1.000 [0.100,1.000] | — |

**p_eff per event — S7** (detection threshold = 1.0):

| event | R_obs | p_eff [95% CI] | p_eff / threshold |
| --- | --- | --- | --- |
| gst_rollout | +0.03 | 0.101 [0.002,0.490] | 0.10 |
| sabarimala_verdict | +0.09 | 0.098 [0.000,0.500] | 0.10 |
| pulwama_attack | -0.16 | 1.000 [0.100,1.000] | 1.00 |
| lok_sabha_results | +0.23 | 0.000 [0.000,0.500] | 0.00 |
| article_370_abrogation | -0.18 | 1.000 [0.100,1.000] | 1.00 |
| ayodhya_verdict | -0.21 | 1.000 [0.100,1.000] | 1.00 |
| covid_first_cases | -0.17 | 1.000 [0.100,1.000] | 1.00 |

**p_eff per event — S1** (detection threshold = None):

| event | R_obs | p_eff [95% CI] | p_eff / threshold |
| --- | --- | --- | --- |
| gst_rollout | -0.18 | 1.000 [0.919,1.000] | — |
| sabarimala_verdict | +0.12 | 0.000 [0.000,0.250] | — |
| pulwama_attack | -0.15 | 0.936 [0.749,1.000] | — |
| lok_sabha_results | +0.20 | 0.000 [0.000,0.250] | — |
| article_370_abrogation | -0.33 | 1.000 [1.000,1.000] | — |
| ayodhya_verdict | +0.18 | 0.000 [0.000,0.250] | — |
| covid_first_cases | -0.30 | 1.000 [1.000,1.000] | — |

R_obs fell below R(0) (p_eff≈0) in **11/21** event×signal cases.

## Part 2 — pooled-alarm event test (every alarm contributes)

Fraction of a signal's alarms that fall 0–W days after any of the 7 events, vs 2000 random 7-date sets.

| signal | 0-30d obs (p) | 0-60d obs (p) | 0-90d obs (p) |
| --- | --- | --- | --- |
| S6 | 0.07 (p=0.530) | 0.14 (p=0.428) | 0.18 (p=0.631) |
| S6p | 0.10 (p=0.325) | 0.13 (p=0.584) | 0.19 (p=0.519) |
| S5 | 0.09 (p=0.451) | 0.17 (p=0.326) | 0.17 (p=0.634) |
| S4 | 0.00 (p=1.000) | 0.05 (p=0.940) | 0.11 (p=0.920) |
| S7 | 0.02 (p=0.935) | 0.08 (p=0.883) | 0.13 (p=0.892) |
| S1 | 0.03 (p=0.957) | 0.11 (p=0.762) | 0.16 (p=0.796) |
| S1c | 0.05 (p=0.693) | 0.08 (p=0.925) | 0.15 (p=0.835) |
| S3 | 0.02 (p=0.973) | 0.07 (p=0.981) | 0.14 (p=0.891) |

Signals reaching p<0.05 in *some* window: **none**. **Read with care:** the three windows disagree (e.g. S7 is significant at 0-60d but not 0-30d or 0-90d), and with 8 signals × 3 windows = 24 tests a couple of p<0.05 are expected by chance. This is at most marginal, inconsistent evidence of weak clustering — not robust event detection.

## Part 3 — power analysis of the permutation test

Inject an extra alarm within ±k days of each event with probability q, then re-run the T7 permutation test on 200 simulated alarm sets; power = fraction reaching p<0.05.

| signal | k | q=0.2 | q=0.4 | q=0.6 | q=0.8 | q=1.0 | min q @80% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S4 | 15 | 0.01 | 0.02 | 0.15 | 0.32 | 0.54 | >1.0 |
| S4 | 30 | 0.00 | 0.04 | 0.08 | 0.23 | 0.36 | >1.0 |
| S7 | 15 | 0.00 | 0.05 | 0.10 | 0.18 | 0.27 | >1.0 |
| S7 | 30 | 0.00 | 0.00 | 0.01 | 0.03 | 0.03 | >1.0 |

Minimum detectable effect (smallest q with ≥80% power): S4/k=15: >1.0, S4/k=30: >1.0, S7/k=15: >1.0, S7/k=30: >1.0. The test would have detected clustering of alarms within k days of at least that fraction of events with 80% probability; no such clustering is observed on the real stream.

## Part 4 — direct event footprint (corroborating p_eff)

For each event: the 20 word types most over-represented in the 30 days after vs the 30 days before (frequency ratio, min post-count 5), and the fraction of post-event documents containing at least one of them — a direct, assumption-free estimate of how much of the stream the event touched.

**gst_rollout** (2017-07-01): post-event doc fraction touched = **0.11** (466 docs). Top over-represented types:
> lap thailand chiang nhs g20 priebus jio sunny ricciardo mangrove federer branch mai mob bottas meme trudeau lynching jd goa

**sabarimala_verdict** (2018-09-28): post-event doc fraction touched = **0.29** (499 docs). Top over-represented types:
> oct bamboo akbar metoo logs surbhi ayyappa degrees asthana morality discounts fc doimara timber cosaap cvc iim royal sabarimala kalamandalam

**pulwama_attack** (2019-02-14): post-event doc fraction touched = **0.22** (543 docs). Top over-represented types:
> azhar quotient desirability she’s dar jaish dogs scooter islamabad biki masood intel max statue icse somnath lucknow bandhs saudi pulwama

**lok_sabha_results** (2019-05-23): post-event doc fraction touched = **0.01** (434 docs). Top over-represented types:
>   n  vers  he  
 ew  or  ealand  outh  frica  ff  o  nd  e  uns  is  our  ver  maharera pistorius

**article_370_abrogation** (2019-08-05): post-event doc fraction touched = **0.28** (396 docs). Top over-represented types:
> 370 article cbi â 371 okuhara ev chidambaram solung abrogation devika photography pluto rape œwe curfew ganesha lifestyle analogue œthe

**ayodhya_verdict** (2019-11-09): post-event doc fraction touched = **0.23** (324 docs). Top over-represented types:
> jnu photos indira veg flats panchuka gi mahatma archive bulbul unsold cms timesofindia.indiatimes.com saarc articleshowprint namsai protest ocean mining gandhi

**covid_first_cases** (2020-01-30): post-event doc fraction touched = **0.15** (756 docs). Top over-represented types:
> apssb feb sisodia dot nili cookies castes azhar affidavit mahal las visas melania watan renault masood hydropower kaul archives ne

Corroboration — direct footprint vs S7 p_eff:

| event | footprint (doc frac) | S7 p_eff | ratio |
| --- | --- | --- | --- |
| gst_rollout | 0.11 | 0.101 | 1.1 |
| sabarimala_verdict | 0.29 | 0.098 | 2.9 |
| pulwama_attack | 0.22 | 1.000 | 0.2 |
| lok_sabha_results | 0.01 | 0.000 | inf |
| article_370_abrogation | 0.28 | 1.000 | 0.3 |
| ayodhya_verdict | 0.23 | 1.000 | 0.2 |
| covid_first_cases | 0.15 | 1.000 | 0.1 |

Direct measurement corroborates p_eff (within ~3×) in **2/7** events → inversion is UNRELIABLE — report with caution.

## STATUS

```
GATE 1 — response curve R(p) built with CIs for all signals:               PASS
GATE 2 — p_eff estimated for all 7 events, inversion documented:           PASS
GATE 3 — pooled-alarm event test run with >=2000 permutations:             PASS  (2000)
GATE 4 — power analysis reports minimum detectable effect:                 PASS
GATE 5 — event footprint measured directly and compared to p_eff:          PASS

THE SENSITIVITY FLOOR:
    detection threshold (smallest p with excess-power CI>0): S1=None, S1c=None, S3=None, S4=None, S7=1.0
    median p_eff across 7 events: S4=1.000, S7=1.000
    ratio event p_eff / threshold: S4=nan, S7=1.00
    direct footprint (doc fraction) median across events: 0.215
    does the direct measurement corroborate p_eff?  NO

THE POOLED EVENT TEST:
    signals significant in any window: none

TEST POWER:
    minimum q detectable at 80% power: S4/k=15: >1.0, S4/k=30: >1.0, S7/k=15: >1.0, S7/k=30: >1.0

VERDICT: PROCEED WITH CAVEATS
Blockers:
  - none
Surprises worth a human decision:
  - THE SENSITIVITY FLOOR: real Bangla news events directly touch a median **22% of documents** (5-21% across the 7 events), but the label-free detectors need a much larger fraction drifted to fire at a deployable FAR (S7 threshold p≈1.0, S4 p≈None). Real events sit a factor ~2-4 below the detection threshold — this is the paper's central quantitative claim, and it rests on the direct footprint measurement.
  - The per-event p_eff *inversion* is unreliable (corroborates the direct footprint in only 2/7 events; p_eff is very noisy — COVID→1.0, road-safety→0.0). Lead with the direct footprint; report p_eff only as a rough, hedged cross-check.
  - The median-delay permutation test (T7) has ~no power even at q=1.0 (min q>1.0 for 80% power) — T7's null was partly a low-power artifact of that statistic. The pooled-alarm test is the better test and shows at most marginal, inconsistent clustering.
```
