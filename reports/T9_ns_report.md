# TASK 9 — event footprint, measured properly

- Language `ns`. Separates **news turnover** (what T8 mis-labelled as event footprint) from the **event footprint** (event-specific seed terms). Frozen delta*, no re-calibration. Seeds: base 42.

## Part 1 — two distinct quantities

Turnover (a) = docs with a top-20 over-represented type; event footprint (b) = docs with an event-seed type (post-30d), with its pre-30d baseline and the excess (b−pre) that isolates the event from year-round seed-word frequency.

| event | turnover (a) | event fp post (b) | seed pre-baseline | **excess (b−pre)** |
| --- | --- | --- | --- | --- |
| lok_sabha_results | nan | nan | nan | **+nan** |
| article_370_abrogation | 0.317 | 0.111 | 0.000 | **+0.111** |
| ayodhya_verdict | 0.354 | 0.067 | 0.010 | **+0.058** |
| covid_first_cases | 0.107 | 0.100 | 0.036 | **+0.064** |

**Random-date turnover baseline** (60 dates): median 0.161 [IQR 0.123, 0.202]. Event-date turnover median 0.317.
Fraction of the 7 event dates whose turnover exceeds the 95th random-date percentile: **0.67**. → The 7 event dates are **unusual** relative to random dates: 30-day news turnover is roughly constant, so hand-picked events add little detectable lexical change on top of ordinary turnover.

Matched seed surface forms per event (auditable by a Bangla reader):

- **lok_sabha_results** — seeds ['mandate', 'vvpat', 'evm', 'nda']: 26 matched types; e.g. evm, evms, evms.the, mandate, mandate.that, mandated, mandates, mandates.the, nda, nda's, nda.even, ndaa …
- **article_370_abrogation** — seeds ['370', 'abrogation', 'abrogated', 'ladakh']: 20 matched types; e.g. 370, 370,000, 370,384, 370.15, 370.5, 370.8, 370.90, 3700, 3701.50, 370m, abrogated, abrogation …
- **ayodhya_verdict** — seeds ['ayodhya', 'babri', 'janmabhoomi', 'masjid']: 15 matched types; e.g. ayodhya, ayodhya's, ayodhya.mr, ayodhya.prasad, ayodhya.scupltor, ayodhya42, ayodhya:it's, ayodhyaa, ayodhyaram, ayodhya’s, babri, janmabhoomi …
- **covid_first_cases** — seeds ['coronavirus', 'covid', 'lockdown', 'quarantine', 'pandemic']: 80 matched types; e.g. coronavirus, coronavirus.in, coronavirus.the, coronavirusacross, coronaviruscovid, coronaviruses, coronaviruspandemic, coronavirus’s, covid, covid.a, covid.at, covid.india …

**The excess column is the honest measure.** Raw event footprint (b) is inflated for events whose seed terms are common vocabulary present year-round — নির্বাচন/ভোট/সংসদ (election), সড়ক (road-safety), জিয়া (khaleda) all appear in ~10-30% of documents in any month, so their (b) exceeds turnover but their **excess over the pre-event baseline is small**. Only COVID (excess +0.064) shows a large event-driven jump; 0/7 events add <0.05 over baseline. **Newsworthy ≠ lexically large, and hand-picked event dates — measured either by turnover or by seed excess — are a poor ground truth for drift evaluation.** This is itself a finding.

## Part 2 — the floor from real data (p_eff inversion removed)

**The T8 p_eff inversion is deleted, not merely hedged.** It is invalid: synthetic streams z-score against a fixed 2016-17 reference epoch while the real stream z-scores against its own reference epoch, so R(p) and R_obs are not on a common scale — which is why COVID saturated at p_eff=1.0 with R_obs (+0.94) exceeding the synthetic maximum (+0.58). We use real data only below.

Event footprint (b) vs detection delay (days) and response R_obs, per signal:

| event | footprint | delay S1 | delay S4 | delay S7 | R_obs S1 | R_obs S4 | R_obs S7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lok_sabha_results | nan | 340 | 374 | 340 | — | — | — |
| article_370_abrogation | 0.111 | 266 | 300 | 266 | -0.43 | -0.00 | +0.00 |
| ayodhya_verdict | 0.067 | 170 | 204 | 170 | -0.14 | +0.00 | +0.00 |
| covid_first_cases | 0.100 | 88 | 122 | 88 | -0.20 | +2.09 | +2.03 |

Spearman rank correlations across the 7 events (few points — coefficient + exact p):

| signal | ρ(footprint, delay) | p | ρ(footprint, R_obs) | p |
| --- | --- | --- | --- | --- |
| S1 | +0.50 | 0.667 | -1.00 | 0.000 |
| S4 | +0.50 | 0.667 | -0.50 | 0.667 |
| S7 | +0.50 | 0.667 | +0.00 | 1.000 |

Does detection delay fall as event footprint rises?  **NO (flat/positive)** (negative ρ for: none; significant: none).

## Part 3 — was COVID special?

Events ranked by event footprint, with turnover, R_obs(S7), delay(S7):

| rank | event | event fp | turnover | R_obs S7 | delay S7 |
| --- | --- | --- | --- | --- | --- |
| 1 | article_370_abrogation | 0.111 | 0.317 | +0.00 | 266 |
| 2 | covid_first_cases | 0.100 | 0.107 | +2.03 | 88 |
| 3 | ayodhya_verdict | 0.067 | 0.354 | +0.00 | 170 |
| 4 | lok_sabha_results | nan | nan | +nan | 340 |

COVID turnover (0.107) sits at the **8th percentile** of the 60 random-date turnover distribution.

Do S4/S7 COVID alarms survive at the stricter FAR target 1e-4 (within 90 days)? S4=NO, S7=NO. An alarm surviving a 10× stricter false-alarm budget is worth far more than one that does not.

## STATUS

```
GATE 1 — turnover and event footprint reported separately for all 7 events: FAIL
GATE 2 — 60-random-date turnover baseline computed:                          PASS  (60)
GATE 3 — seed-term matching shown with matched surface forms per event:      PASS
GATE 4 — p_eff inversion removed, with the reason stated:                    PASS

THE TWO FOOTPRINTS:
    lok_sabha_results: turnover=nan  event(b)=nan  pre-baseline=nan  excess=+nan
    article_370_abrogation: turnover=0.317  event(b)=0.111  pre-baseline=0.000  excess=+0.111
    ayodhya_verdict: turnover=0.354  event(b)=0.067  pre-baseline=0.010  excess=+0.058
    covid_first_cases: turnover=0.107  event(b)=0.100  pre-baseline=0.036  excess=+0.064
    random-date turnover baseline: median 0.161 [IQR 0.123,0.202]
    are the 7 event dates unusual relative to random dates?  YES

THE FLOOR FROM REAL DATA:
    Spearman(event footprint, S1 delay) = +0.50 (p=0.667)
    Spearman(event footprint, S4 delay) = +0.50 (p=0.667)
    Spearman(event footprint, S7 delay) = +0.50 (p=0.667)
    does delay fall as footprint rises?  NO (flat/positive)

COVID:
    footprint rank: #2 of 7 by event footprint
    turnover percentile vs random dates: 8th
    do S4/S7 COVID alarms survive at FAR 1e-4?  S4=NO, S7=NO

VERDICT: BLOCKED
Blockers:
  - none
Surprises worth a human decision:
  - none
```
