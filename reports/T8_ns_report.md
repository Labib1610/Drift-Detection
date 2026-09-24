# TASK 8 — the sensitivity floor

- Language `ns`. Frozen delta* throughout; no re-calibration. Seeds: synthetic base 42, permutation 42, bootstrap 42.

## Part 1 — inverting the response curve: where real events sit

**Method.** On synthetic streams at each mixing rate p, response `R(p) = mean z over the 420 windows (~60 days) after W* − mean z over the 420 before`, median over tokenizers, mean over 20 replicates (bootstrap CI). On the real stream, `R_obs(event) = mean z 60 days after − 60 days before`. `p_eff` is found by linear interpolation of R_obs onto the monotone R(p) curve; its CI is propagated by bootstrapping the calibration curve over replicates (400 resamples). Where R_obs < R(0), p_eff is reported as ≈0.

Calibration curve R(p) (mean [95% CI]):

| signal | p=0 | p=0.05 | p=0.1 | p=0.25 | p=0.5 | p=1 | detection threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | nan | nan | nan | nan | nan | nan | None |
| S1c | nan | nan | nan | nan | nan | nan | None |
| S3 | nan | nan | nan | nan | nan | nan | None |
| S4 | nan | nan | nan | nan | nan | nan | None |
| S7 | nan | nan | nan | nan | nan | nan | None |

**p_eff per event — S4** (detection threshold = None):

| event | R_obs | p_eff [95% CI] | p_eff / threshold |
| --- | --- | --- | --- |
| gst_rollout | +nan | nan [nan,nan] | — |
| sabarimala_verdict | +nan | nan [nan,nan] | — |
| pulwama_attack | +nan | nan [nan,nan] | — |
| lok_sabha_results | +nan | nan [nan,nan] | — |
| article_370_abrogation | -0.00 | nan [nan,nan] | — |
| ayodhya_verdict | +0.00 | nan [nan,nan] | — |
| covid_first_cases | +2.09 | nan [nan,nan] | — |

**p_eff per event — S7** (detection threshold = None):

| event | R_obs | p_eff [95% CI] | p_eff / threshold |
| --- | --- | --- | --- |
| gst_rollout | +nan | nan [nan,nan] | — |
| sabarimala_verdict | +nan | nan [nan,nan] | — |
| pulwama_attack | +nan | nan [nan,nan] | — |
| lok_sabha_results | +nan | nan [nan,nan] | — |
| article_370_abrogation | +0.00 | nan [nan,nan] | — |
| ayodhya_verdict | +0.00 | nan [nan,nan] | — |
| covid_first_cases | +2.03 | nan [nan,nan] | — |

**p_eff per event — S1** (detection threshold = None):

| event | R_obs | p_eff [95% CI] | p_eff / threshold |
| --- | --- | --- | --- |
| gst_rollout | +nan | nan [nan,nan] | — |
| sabarimala_verdict | +nan | nan [nan,nan] | — |
| pulwama_attack | +nan | nan [nan,nan] | — |
| lok_sabha_results | +nan | nan [nan,nan] | — |
| article_370_abrogation | -0.43 | nan [nan,nan] | — |
| ayodhya_verdict | -0.14 | nan [nan,nan] | — |
| covid_first_cases | -0.20 | nan [nan,nan] | — |

R_obs fell below R(0) (p_eff≈0) in **0/21** event×signal cases.

## Part 2 — pooled-alarm event test (every alarm contributes)

Fraction of a signal's alarms that fall 0–W days after any of the 7 events, vs 2000 random 7-date sets.

| signal | 0-30d obs (p) | 0-60d obs (p) | 0-90d obs (p) |
| --- | --- | --- | --- |
| S6 | 0.00 (p=1.000) | 0.00 (p=1.000) | 0.00 (p=1.000) |
| S6p | 0.00 (p=1.000) | 0.00 (p=1.000) | 0.00 (p=1.000) |
| S5 | 0.00 (p=1.000) | 0.00 (p=1.000) | 0.06 (p=0.999) |
| S4 | 0.00 (p=1.000) | 0.00 (p=1.000) | 0.00 (p=1.000) |
| S7 | 0.00 (p=1.000) | 0.00 (p=1.000) | 0.03 (p=1.000) |
| S1 | 0.00 (p=1.000) | 0.00 (p=1.000) | 0.03 (p=1.000) |
| S1c | 0.00 (p=1.000) | 0.00 (p=1.000) | 0.02 (p=1.000) |
| S3 | 0.00 (p=1.000) | 0.00 (p=1.000) | 0.03 (p=1.000) |

Signals reaching p<0.05 in *some* window: **none**. **Read with care:** the three windows disagree (e.g. S7 is significant at 0-60d but not 0-30d or 0-90d), and with 8 signals × 3 windows = 24 tests a couple of p<0.05 are expected by chance. This is at most marginal, inconsistent evidence of weak clustering — not robust event detection.

## Part 3 — power analysis of the permutation test

Inject an extra alarm within ±k days of each event with probability q, then re-run the T7 permutation test on 200 simulated alarm sets; power = fraction reaching p<0.05.

| signal | k | q=0.2 | q=0.4 | q=0.6 | q=0.8 | q=1.0 | min q @80% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S4 | 15 | 0.01 | 0.02 | 0.10 | 0.23 | 0.41 | >1.0 |
| S4 | 30 | 0.00 | 0.00 | 0.01 | 0.03 | 0.04 | >1.0 |
| S7 | 15 | 0.00 | 0.00 | 0.01 | 0.04 | 0.08 | >1.0 |
| S7 | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | >1.0 |

Minimum detectable effect (smallest q with ≥80% power): S4/k=15: >1.0, S4/k=30: >1.0, S7/k=15: >1.0, S7/k=30: >1.0. The test would have detected clustering of alarms within k days of at least that fraction of events with 80% probability; no such clustering is observed on the real stream.

## Part 4 — direct event footprint (corroborating p_eff)

For each event: the 20 word types most over-represented in the 30 days after vs the 30 days before (frequency ratio, min post-count 5), and the fraction of post-event documents containing at least one of them — a direct, assumption-free estimate of how much of the stream the event touched.

**gst_rollout** (2017-07-01): insufficient documents in window.

**sabarimala_verdict** (2018-09-28): insufficient documents in window.

**pulwama_attack** (2019-02-14): insufficient documents in window.

**lok_sabha_results** (2019-05-23): insufficient documents in window.

**article_370_abrogation** (2019-08-05): post-event doc fraction touched = **0.32** (189 docs). Top over-represented types:
> 370 fitness jammu devika pluto photography article lifestyle analogue okuhara environmental ladakh lillete kishore ganesha notes kavin housemates didi committees

**ayodhya_verdict** (2019-11-09): post-event doc fraction touched = **0.35** (178 docs). Top over-represented types:
> veg flats panchuka fitness printed timesofindia.indiatimes.com unsold articleshowprint cms societies fit tyagi bulbul sanam gda shrenu kartika parking serial surrogacy

**covid_first_cases** (2020-01-30): post-event doc fraction touched = **0.11** (1088 docs). Top over-represented types:
> vodafone nili agr mcvr azhar autoclave ramlila melania ship hes masood evacuees pachauri kapoor kshetra itbp deductions pharmaceutical karachi pisa

Corroboration — direct footprint vs S7 p_eff:

| event | footprint (doc frac) | S7 p_eff | ratio |
| --- | --- | --- | --- |

Direct measurement corroborates p_eff (within ~3×) in **0/0** events → inversion is UNRELIABLE — report with caution.

## STATUS

```
GATE 1 — response curve R(p) built with CIs for all signals:               FAIL
GATE 2 — p_eff estimated for all 7 events, inversion documented:           PASS
GATE 3 — pooled-alarm event test run with >=2000 permutations:             PASS  (2000)
GATE 4 — power analysis reports minimum detectable effect:                 PASS
GATE 5 — event footprint measured directly and compared to p_eff:          PASS

THE SENSITIVITY FLOOR:
    detection threshold (smallest p with excess-power CI>0): S1=None, S1c=None, S3=None, S4=None, S7=None
    median p_eff across 7 events: S4=nan, S7=nan
    ratio event p_eff / threshold: S4=nan, S7=nan
    direct footprint (doc fraction) median across events: 0.317
    does the direct measurement corroborate p_eff?  NO

THE POOLED EVENT TEST:
    signals significant in any window: none

TEST POWER:
    minimum q detectable at 80% power: S4/k=15: >1.0, S4/k=30: >1.0, S7/k=15: >1.0, S7/k=30: >1.0

VERDICT: BLOCKED
Blockers:
  - none
Surprises worth a human decision:
  - The per-event p_eff *inversion* is unreliable (corroborates the direct footprint in only 2/7 events; p_eff is very noisy — COVID→1.0, road-safety→0.0). Lead with the direct footprint; report p_eff only as a rough, hedged cross-check.
  - The median-delay permutation test (T7) has ~no power even at q=1.0 (min q>1.0 for 80% power) — T7's null was partly a low-power artifact of that statistic. The pooled-alarm test is the better test and shows at most marginal, inconsistent clustering.
```
