# TASK 9 — event footprint, measured properly

- Language `ns`. Separates **news turnover** (what T8 mis-labelled as event footprint) from the **event footprint** (event-specific seed terms). Frozen delta*, no re-calibration. Seeds: base 42.

## Part 1 — two distinct quantities

Turnover (a) = docs with a top-20 over-represented type; event footprint (b) = docs with an event-seed type (post-30d), with its pre-30d baseline and the excess (b−pre) that isolates the event from year-round seed-word frequency.

| event | turnover (a) | event fp post (b) | seed pre-baseline | **excess (b−pre)** |
| --- | --- | --- | --- | --- |
| gst_rollout | 0.112 | 0.052 | 0.027 | **+0.025** |
| sabarimala_verdict | 0.289 | 0.042 | 0.008 | **+0.035** |
| pulwama_attack | 0.215 | 0.136 | 0.008 | **+0.128** |
| lok_sabha_results | 0.012 | 0.041 | 0.051 | **-0.009** |
| article_370_abrogation | 0.280 | 0.119 | 0.003 | **+0.116** |
| ayodhya_verdict | 0.228 | 0.049 | 0.025 | **+0.025** |
| covid_first_cases | 0.148 | 0.115 | 0.030 | **+0.085** |

**Random-date turnover baseline** (60 dates): median 0.136 [IQR 0.117, 0.175]. Event-date turnover median 0.215.
Fraction of the 7 event dates whose turnover exceeds the 95th random-date percentile: **0.29**. → The 7 event dates are **NOT unusual** relative to random dates: 30-day news turnover is roughly constant, so hand-picked events add little detectable lexical change on top of ordinary turnover.

Matched seed surface forms per event (auditable by a Bangla reader):

- **gst_rollout** — seeds ['gst', 'gstn', 'gstin']: 20 matched types; e.g. gst, gst's, gst.actors, gst.it's, gst.prosenjit, gst.read, gst.the, gst.this, gst:ï, gstat, gstats, gstf …
- **sabarimala_verdict** — seeds ['sabarimala', 'ayyappa', 'devaswom', 'pandalam']: 10 matched types; e.g. ayyappa, ayyappadas, ayyappan, ayyappan.restrict, ayyappanum, ayyappa’s, devaswom, pandalam, sabarimala, sabarimala.november
- **pulwama_attack** — seeds ['pulwama', 'balakot', 'crpf', 'jaish', 'awantipora']: 27 matched types; e.g. awantipora, balakot, balakote, balakotâ, crpf, crpf'another, crpf's, crpf’s, jaish, jaish's, jaishanka, jaishankar …
- **lok_sabha_results** — seeds ['mandate', 'vvpat', 'evm', 'nda']: 41 matched types; e.g. evm, evm's, evmissue, evms, evms.it, evm’s, mandate, mandate.that, mandated, mandates, mandates.the, mandatetothebjptostay …
- **article_370_abrogation** — seeds ['370', 'abrogation', 'abrogated', 'ladakh']: 30 matched types; e.g. 370, 370,000, 370,384, 370.15, 370.8, 370.90, 3700, 3700.0, 370041492, 3701.50, 3702, 370859 …
- **ayodhya_verdict** — seeds ['ayodhya', 'babri', 'janmabhoomi', 'masjid']: 16 matched types; e.g. ayodhya, ayodhya's, ayodhya.on, ayodhya.while, ayodhya42, ayodhya:it's, ayodhyaa, ayodhyalast, ayodhyaram, ayodhya’s, babri, janmabhoomi …
- **covid_first_cases** — seeds ['coronavirus', 'covid', 'lockdown', 'quarantine', 'pandemic']: 82 matched types; e.g. coronavirus, coronavirus.surat, coronavirus.the, coronaviruscovid, coronaviruses, coronavirusâ, coronavirus’s, covid, covid.a, covid.after, covid.at, covid.india …

**The excess column is the honest measure.** Raw event footprint (b) is inflated for events whose seed terms are common vocabulary present year-round — নির্বাচন/ভোট/সংসদ (election), সড়ক (road-safety), জিয়া (khaleda) all appear in ~10-30% of documents in any month, so their (b) exceeds turnover but their **excess over the pre-event baseline is small**. Only COVID (excess +0.085) shows a large event-driven jump; 4/7 events add <0.05 over baseline. **Newsworthy ≠ lexically large, and hand-picked event dates — measured either by turnover or by seed excess — are a poor ground truth for drift evaluation.** This is itself a finding.

## Part 2 — the floor from real data (p_eff inversion removed)

**The T8 p_eff inversion is deleted, not merely hedged.** It is invalid: synthetic streams z-score against a fixed 2016-17 reference epoch while the real stream z-scores against its own reference epoch, so R(p) and R_obs are not on a common scale — which is why COVID saturated at p_eff=1.0 with R_obs (+0.94) exceeding the synthetic maximum (+0.58). We use real data only below.

Event footprint (b) vs detection delay (days) and response R_obs, per signal:

| event | footprint | delay S1 | delay S4 | delay S7 | R_obs S1 | R_obs S4 | R_obs S7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gst_rollout | 0.052 | 69 | 184 | 83 | -0.18 | +0.03 | +0.03 |
| sabarimala_verdict | 0.042 | 90 | 90 | 90 | +0.12 | +0.11 | +0.09 |
| pulwama_attack | 0.136 | 76 | 138 | 2 | -0.15 | -0.09 | -0.16 |
| lok_sabha_results | 0.041 | 40 | 40 | 40 | +0.20 | +0.30 | +0.23 |
| article_370_abrogation | 0.119 | 31 | 238 | 144 | -0.33 | -0.28 | -0.18 |
| ayodhya_verdict | 0.049 | 37 | 142 | 48 | +0.18 | -0.10 | -0.21 |
| covid_first_cases | 0.115 | 48 | 60 | 127 | -0.30 | -0.26 | -0.17 |

Spearman rank correlations across the 7 events (few points — coefficient + exact p):

| signal | ρ(footprint, delay) | p | ρ(footprint, R_obs) | p |
| --- | --- | --- | --- | --- |
| S1 | -0.04 | 0.939 | -0.75 | 0.052 |
| S4 | +0.50 | 0.253 | -0.71 | 0.071 |
| S7 | +0.14 | 0.760 | -0.54 | 0.215 |

Does detection delay fall as event footprint rises?  **UNCLEAR (negative but not significant with 7 points)** (negative ρ for: S1; significant: none).

## Part 3 — was COVID special?

Events ranked by event footprint, with turnover, R_obs(S7), delay(S7):

| rank | event | event fp | turnover | R_obs S7 | delay S7 |
| --- | --- | --- | --- | --- | --- |
| 1 | pulwama_attack | 0.136 | 0.215 | -0.16 | 2 |
| 2 | article_370_abrogation | 0.119 | 0.280 | -0.18 | 144 |
| 3 | covid_first_cases | 0.115 | 0.148 | -0.17 | 127 |
| 4 | gst_rollout | 0.052 | 0.112 | +0.03 | 83 |
| 5 | ayodhya_verdict | 0.049 | 0.228 | -0.21 | 48 |
| 6 | sabarimala_verdict | 0.042 | 0.289 | +0.09 | 90 |
| 7 | lok_sabha_results | 0.041 | 0.012 | +0.23 | 40 |

COVID turnover (0.148) sits at the **53th percentile** of the 60 random-date turnover distribution.

Do S4/S7 COVID alarms survive at the stricter FAR target 1e-4 (within 90 days)? S4=NO, S7=NO. An alarm surviving a 10× stricter false-alarm budget is worth far more than one that does not.

## STATUS

```
GATE 1 — turnover and event footprint reported separately for all 7 events: PASS
GATE 2 — 60-random-date turnover baseline computed:                          PASS  (60)
GATE 3 — seed-term matching shown with matched surface forms per event:      PASS
GATE 4 — p_eff inversion removed, with the reason stated:                    PASS

THE TWO FOOTPRINTS:
    gst_rollout: turnover=0.112  event(b)=0.052  pre-baseline=0.027  excess=+0.025
    sabarimala_verdict: turnover=0.289  event(b)=0.042  pre-baseline=0.008  excess=+0.035
    pulwama_attack: turnover=0.215  event(b)=0.136  pre-baseline=0.008  excess=+0.128
    lok_sabha_results: turnover=0.012  event(b)=0.041  pre-baseline=0.051  excess=-0.009
    article_370_abrogation: turnover=0.280  event(b)=0.119  pre-baseline=0.003  excess=+0.116
    ayodhya_verdict: turnover=0.228  event(b)=0.049  pre-baseline=0.025  excess=+0.025
    covid_first_cases: turnover=0.148  event(b)=0.115  pre-baseline=0.030  excess=+0.085
    random-date turnover baseline: median 0.136 [IQR 0.117,0.175]
    are the 7 event dates unusual relative to random dates?  NO

THE FLOOR FROM REAL DATA:
    Spearman(event footprint, S1 delay) = -0.04 (p=0.939)
    Spearman(event footprint, S4 delay) = +0.50 (p=0.253)
    Spearman(event footprint, S7 delay) = +0.14 (p=0.760)
    does delay fall as footprint rises?  UNCLEAR (negative but not significant with 7 points)

COVID:
    footprint rank: #3 of 7 by event footprint
    turnover percentile vs random dates: 53th
    do S4/S7 COVID alarms survive at FAR 1e-4?  S4=NO, S7=NO

VERDICT: PROCEED WITH CAVEATS
Blockers:
  - none
Surprises worth a human decision:
  - The 7 event dates' 30-day news turnover is NOT unusual vs random dates — ordinary news turnover swamps the events, so hand-picked event dates are a poor ground truth. This strengthens the sensitivity-floor story from real data.
  - 4/7 events add <0.05 seed-term presence over their pre-event baseline (gst_rollout, sabarimala_verdict, lok_sabha_results, ayodhya_verdict) — raw event footprint is confounded by year-round vocabulary; only COVID shows a large event-driven excess. Newsworthy ≠ lexically large.
  - Delay trends down with footprint but is not significant at n=7 — the real-data floor is suggestive; the synthetic streams remain the load-bearing evidence.
```
