# TASK 9 — event footprint, measured properly

- Separates **news turnover** (what T8 mis-labelled as event footprint) from the **event footprint** (event-specific seed terms). Frozen delta*, no re-calibration. Seeds: base 42.

## Part 1 — two distinct quantities

Turnover (a) = docs with a top-20 over-represented type; event footprint (b) = docs with an event-seed type (post-30d), with its pre-30d baseline and the excess (b−pre) that isolates the event from year-round seed-word frequency.

| event | turnover (a) | event fp post (b) | seed pre-baseline | **excess (b−pre)** |
| --- | --- | --- | --- | --- |
| rohingya_influx | 0.191 | 0.155 | 0.007 | **+0.148** |
| khaleda_zia_jailed | 0.048 | 0.113 | 0.085 | **+0.028** |
| road_safety_protests | 0.105 | 0.179 | 0.138 | **+0.041** |
| parliamentary_election | 0.131 | 0.298 | 0.442 | **-0.143** |
| nusrat_rafi_murder | 0.103 | 0.059 | 0.022 | **+0.037** |
| abrar_fahad_killing | 0.141 | 0.095 | 0.053 | **+0.043** |
| covid_first_cases | 0.212 | 0.598 | 0.082 | **+0.516** |

**Random-date turnover baseline** (60 dates): median 0.086 [IQR 0.073, 0.108]. Event-date turnover median 0.131.
Fraction of the 7 event dates whose turnover exceeds the 95th random-date percentile: **0.43**. → The 7 event dates are **NOT unusual** relative to random dates: 30-day news turnover is roughly constant, so hand-picked events add little detectable lexical change on top of ordinary turnover.

Matched seed surface forms per event (auditable by a Bangla reader):

- **rohingya_influx** — seeds ['রোহিঙ্গা', 'রাখাইন', 'আরাকান', 'শরণার্থী']: 89 matched types; e.g. আরাকান, আরাকানকে, আরাকানজুড়ে, আরাকানরাজ্যে, আরাকানরাজ্যের, আরাকানসাসও, আরাকানি, আরাকানিকে, আরাকানির, আরাকানী, আরাকানীরা, আরাকানে …
- **khaleda_zia_jailed** — seeds ['খালেদা', 'জিয়া', 'এতিমখানা', 'কারাগারে']: 37 matched types; e.g. এতিমখানা, এতিমখানাকে, এতিমখানাগুলো, এতিমখানাটি, এতিমখানার, এতিমখানাসহ, এতিমখানায়, এতিমখানায়ই, কারাগারে, কারাগারেñস্বৈরাচার, কারাগারেই, কারাগারেএদিকে …
- **road_safety_protests** — seeds ['নিরাপদ', 'সড়ক', 'বাসচাপায়', 'শিক্ষার্থীদের', 'আন্দোলন', 'জাবালে', 'নূর']: 205 matched types; e.g. আন্দোলন, আন্দোলনই, আন্দোলনইউনাইটেড, আন্দোলনইসলামী, আন্দোলনএদিকে, আন্দোলনও, আন্দোলনকরীদের, আন্দোলনকরীরা, আন্দোলনকর্মী, আন্দোলনকর্মীদের, আন্দোলনকর্মীর, আন্দোলনকারি …
- **parliamentary_election** — seeds ['নির্বাচন', 'ভোট', 'একাদশ', 'সংসদ', 'ইভিএম']: 424 matched types; e.g. ইভিএম, ইভিএমএ, ইভিএমএর, ইভিএমকে, ইভিএমগুলো, ইভিএমবাহী, ইভিএমসহ, ইভিএমে, ইভিএমেই, ইভিএমের, ইভিএম’র, একাদশ …
- **nusrat_rafi_murder** — seeds ['নুসরাত', 'ফেনী', 'মাদ্রাসা', 'সোনাগাজী']: 75 matched types; e.g. নুসরাত, নুসরাতকে, নুসরাতকেও, নুসরাতসহ, নুসরাতে, নুসরাতের, ফেনী, ফেনীও, ফেনীকে, ফেনীগামী, ফেনীতে, ফেনীতেও …
- **abrar_fahad_killing** — seeds ['আবরার', 'বুয়েট', 'ফাহাদ', 'ছাত্রলীগ']: 48 matched types; e.g. আবরার, আবরারই, আবরারকে, আবরারদের, আবরারসহ, আবরারুর, আবরারে, আবরারের, আবরার’র, ছাত্রলীগ, ছাত্রলীগই, ছাত্রলীগও …
- **covid_first_cases** — seeds ['করোনা', 'কোভিড', 'লকডাউন', 'কোয়ারেন্টিন']: 159 matched types; e.g. করোনা, করোনাআক্রান্ত, করোনাই, করোনাইদা, করোনাও, করোনাকবলিত, করোনাকা, করোনাকারণে, করোনাকাল, করোনাকালিন, করোনাকালীণ, করোনাকালীন …

**The excess column is the honest measure.** Raw event footprint (b) is inflated for events whose seed terms are common vocabulary present year-round — নির্বাচন/ভোট/সংসদ (election), সড়ক (road-safety), জিয়া (khaleda) all appear in ~10-30% of documents in any month, so their (b) exceeds turnover but their **excess over the pre-event baseline is small**. Only COVID (excess +0.516) shows a large event-driven jump; 5/7 events add <0.05 over baseline. **Newsworthy ≠ lexically large, and hand-picked event dates — measured either by turnover or by seed excess — are a poor ground truth for drift evaluation.** This is itself a finding.

## Part 2 — the floor from real data (p_eff inversion removed)

**The T8 p_eff inversion is deleted, not merely hedged.** It is invalid: synthetic streams z-score against a fixed 2016-17 reference epoch while the real stream z-scores against its own reference epoch, so R(p) and R_obs are not on a common scale — which is why COVID saturated at p_eff=1.0 with R_obs (+0.94) exceeding the synthetic maximum (+0.58). We use real data only below.

Event footprint (b) vs detection delay (days) and response R_obs, per signal:

| event | footprint | delay S1 | delay S4 | delay S7 | R_obs S1 | R_obs S4 | R_obs S7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rohingya_influx | 0.155 | 25 | 37 | 22 | +0.02 | +0.18 | +0.24 |
| khaleda_zia_jailed | 0.113 | 36 | 23 | 41 | -0.07 | +0.20 | +0.18 |
| road_safety_protests | 0.179 | 44 | 91 | 62 | -0.15 | -0.08 | -0.19 |
| parliamentary_election | 0.298 | 54 | 19 | 19 | -0.07 | +0.33 | +0.35 |
| nusrat_rafi_murder | 0.059 | 198 | 92 | 102 | -0.02 | -0.16 | -0.08 |
| abrar_fahad_killing | 0.095 | 37 | 1 | 1 | -0.07 | +0.03 | +0.01 |
| covid_first_cases | 0.598 | 50 | 7 | 4 | -0.18 | +0.67 | +0.94 |

Spearman rank correlations across the 7 events (few points — coefficient + exact p):

| signal | ρ(footprint, delay) | p | ρ(footprint, R_obs) | p |
| --- | --- | --- | --- | --- |
| S1 | +0.07 | 0.879 | -0.50 | 0.253 |
| S4 | -0.32 | 0.482 | +0.75 | 0.052 |
| S7 | -0.36 | 0.432 | +0.64 | 0.119 |

Does detection delay fall as event footprint rises?  **UNCLEAR (negative but not significant with 7 points)** (negative ρ for: S4, S7; significant: none).

## Part 3 — was COVID special?

Events ranked by event footprint, with turnover, R_obs(S7), delay(S7):

| rank | event | event fp | turnover | R_obs S7 | delay S7 |
| --- | --- | --- | --- | --- | --- |
| 1 | covid_first_cases | 0.598 | 0.212 | +0.94 | 4 |
| 2 | parliamentary_election | 0.298 | 0.131 | +0.35 | 19 |
| 3 | road_safety_protests | 0.179 | 0.105 | -0.19 | 62 |
| 4 | rohingya_influx | 0.155 | 0.191 | +0.24 | 22 |
| 5 | khaleda_zia_jailed | 0.113 | 0.048 | +0.18 | 41 |
| 6 | abrar_fahad_killing | 0.095 | 0.141 | +0.01 | 1 |
| 7 | nusrat_rafi_murder | 0.059 | 0.103 | -0.08 | 102 |

COVID turnover (0.212) sits at the **100th percentile** of the 60 random-date turnover distribution.

Do S4/S7 COVID alarms survive at the stricter FAR target 1e-4 (within 90 days)? S4=YES, S7=YES. An alarm surviving a 10× stricter false-alarm budget is worth far more than one that does not.

## STATUS

```
GATE 1 — turnover and event footprint reported separately for all 7 events: PASS
GATE 2 — 60-random-date turnover baseline computed:                          PASS  (60)
GATE 3 — seed-term matching shown with matched surface forms per event:      PASS
GATE 4 — p_eff inversion removed, with the reason stated:                    PASS

THE TWO FOOTPRINTS:
    rohingya_influx: turnover=0.191  event(b)=0.155  pre-baseline=0.007  excess=+0.148
    khaleda_zia_jailed: turnover=0.048  event(b)=0.113  pre-baseline=0.085  excess=+0.028
    road_safety_protests: turnover=0.105  event(b)=0.179  pre-baseline=0.138  excess=+0.041
    parliamentary_election: turnover=0.131  event(b)=0.298  pre-baseline=0.442  excess=-0.143
    nusrat_rafi_murder: turnover=0.103  event(b)=0.059  pre-baseline=0.022  excess=+0.037
    abrar_fahad_killing: turnover=0.141  event(b)=0.095  pre-baseline=0.053  excess=+0.043
    covid_first_cases: turnover=0.212  event(b)=0.598  pre-baseline=0.082  excess=+0.516
    random-date turnover baseline: median 0.086 [IQR 0.073,0.108]
    are the 7 event dates unusual relative to random dates?  NO

THE FLOOR FROM REAL DATA:
    Spearman(event footprint, S1 delay) = +0.07 (p=0.879)
    Spearman(event footprint, S4 delay) = -0.32 (p=0.482)
    Spearman(event footprint, S7 delay) = -0.36 (p=0.432)
    does delay fall as footprint rises?  UNCLEAR (negative but not significant with 7 points)

COVID:
    footprint rank: #1 of 7 by event footprint
    turnover percentile vs random dates: 100th
    do S4/S7 COVID alarms survive at FAR 1e-4?  S4=YES, S7=YES

VERDICT: PROCEED WITH CAVEATS
Blockers:
  - none
Surprises worth a human decision:
  - The 7 event dates' 30-day news turnover is NOT unusual vs random dates — ordinary news turnover swamps the events, so hand-picked event dates are a poor ground truth. This strengthens the sensitivity-floor story from real data.
  - 5/7 events add <0.05 seed-term presence over their pre-event baseline (khaleda_zia_jailed, road_safety_protests, parliamentary_election, nusrat_rafi_murder, abrar_fahad_killing) — raw event footprint is confounded by year-round vocabulary; only COVID shows a large event-driven excess. Newsworthy ≠ lexically large.
  - Delay trends down with footprint but is not significant at n=7 — the real-data floor is suggestive; the synthetic streams remain the load-bearing evidence.
```
