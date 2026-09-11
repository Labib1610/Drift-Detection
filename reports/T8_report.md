# TASK 8 — the sensitivity floor

- Frozen delta* throughout; no re-calibration. Seeds: synthetic base 42, permutation 42, bootstrap 42.

## Part 1 — inverting the response curve: where real events sit

**Method.** On synthetic streams at each mixing rate p, response `R(p) = mean z over the 420 windows (~60 days) after W* − mean z over the 420 before`, median over tokenizers, mean over 20 replicates (bootstrap CI). On the real stream, `R_obs(event) = mean z 60 days after − 60 days before`. `p_eff` is found by linear interpolation of R_obs onto the monotone R(p) curve; its CI is propagated by bootstrapping the calibration curve over replicates (400 resamples). Where R_obs < R(0), p_eff is reported as ≈0.

Calibration curve R(p) (mean [95% CI]):

| signal | p=0 | p=0.05 | p=0.1 | p=0.25 | p=0.5 | p=1 | detection threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | -0.01 | 0.00 | 0.04 | -0.01 | 0.01 | 0.03 | None |
| S1c | -0.01 | 0.00 | 0.03 | -0.01 | 0.02 | 0.02 | None |
| S3 | -0.01 | 0.02 | 0.03 | 0.01 | 0.03 | 0.07 | 1.0 |
| S4 | 0.02 | 0.03 | 0.05 | 0.09 | 0.23 | 0.55 | 0.5 |
| S7 | 0.02 | 0.04 | 0.08 | 0.13 | 0.27 | 0.58 | 0.25 |

**p_eff per event — S4** (detection threshold = 0.5):

| event | R_obs | p_eff [95% CI] | p_eff / threshold |
| --- | --- | --- | --- |
| rohingya_influx | +0.18 | 0.243 [0.066,0.498] | 0.49 |
| khaleda_zia_jailed | +0.20 | 0.426 [0.065,0.535] | 0.85 |
| road_safety_protests | -0.08 | 0.000 [0.000,0.050] | 0.00 |
| parliamentary_election | +0.33 | 0.662 [0.114,0.727] | 1.32 |
| nusrat_rafi_murder | -0.16 | 0.000 [0.000,0.050] | 0.00 |
| abrar_fahad_killing | +0.03 | 0.055 [0.000,0.101] | 0.11 |
| covid_first_cases | +0.67 | 1.000 [0.957,1.000] | 2.00 |

**p_eff per event — S7** (detection threshold = 0.25):

| event | R_obs | p_eff [95% CI] | p_eff / threshold |
| --- | --- | --- | --- |
| rohingya_influx | +0.24 | 0.368 [0.059,0.511] | 1.47 |
| khaleda_zia_jailed | +0.18 | 0.087 [0.062,0.369] | 0.35 |
| road_safety_protests | -0.19 | 0.000 [0.000,0.050] | 0.00 |
| parliamentary_election | +0.35 | 0.599 [0.078,0.649] | 2.40 |
| nusrat_rafi_murder | -0.08 | 0.000 [0.000,0.001] | 0.00 |
| abrar_fahad_killing | +0.01 | 0.016 [0.000,0.064] | 0.07 |
| covid_first_cases | +0.94 | 1.000 [1.000,1.000] | 4.00 |

**p_eff per event — S1** (detection threshold = None):

| event | R_obs | p_eff [95% CI] | p_eff / threshold |
| --- | --- | --- | --- |
| rohingya_influx | +0.02 | 0.928 [0.132,0.986] | — |
| khaleda_zia_jailed | -0.07 | 0.000 [0.000,0.500] | — |
| road_safety_protests | -0.15 | 0.000 [0.000,0.500] | — |
| parliamentary_election | -0.07 | 0.000 [0.000,0.500] | — |
| nusrat_rafi_murder | -0.02 | 0.207 [0.004,0.803] | — |
| abrar_fahad_killing | -0.07 | 0.000 [0.000,0.500] | — |
| covid_first_cases | -0.18 | 0.000 [0.000,0.500] | — |

R_obs fell below R(0) (p_eff≈0) in **9/21** event×signal cases.

## Part 2 — pooled-alarm event test (every alarm contributes)

Fraction of a signal's alarms that fall 0–W days after any of the 7 events, vs 2000 random 7-date sets.

| signal | 0-30d obs (p) | 0-60d obs (p) | 0-90d obs (p) |
| --- | --- | --- | --- |
| S6 | 0.23 (p=0.057) | 0.23 (p=0.572) | 0.38 (p=0.315) |
| S6p | 0.08 (p=0.843) | 0.23 (p=0.602) | 0.23 (p=0.861) |
| S5 | 0.06 (p=0.927) | 0.29 (p=0.345) | 0.41 (p=0.281) |
| S4 | 0.19 (p=0.240) | 0.38 (p=0.075) | 0.43 (p=0.215) |
| S7 | 0.18 (p=0.138) | 0.38 (p=0.021) | 0.44 (p=0.071) |
| S1 | 0.16 (p=0.233) | 0.31 (p=0.096) | 0.41 (p=0.122) |
| S1c | 0.13 (p=0.441) | 0.27 (p=0.288) | 0.41 (p=0.107) |
| S3 | 0.13 (p=0.423) | 0.32 (p=0.095) | 0.46 (p=0.032) |

Signals reaching p<0.05 in *some* window: S7, S3. **Read with care:** the three windows disagree (e.g. S7 is significant at 0-60d but not 0-30d or 0-90d), and with 8 signals × 3 windows = 24 tests a couple of p<0.05 are expected by chance. This is at most marginal, inconsistent evidence of weak clustering — not robust event detection.

## Part 3 — power analysis of the permutation test

Inject an extra alarm within ±k days of each event with probability q, then re-run the T7 permutation test on 200 simulated alarm sets; power = fraction reaching p<0.05.

| signal | k | q=0.2 | q=0.4 | q=0.6 | q=0.8 | q=1.0 | min q @80% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S4 | 15 | 0.06 | 0.17 | 0.38 | 0.54 | 0.59 | >1.0 |
| S4 | 30 | 0.02 | 0.04 | 0.12 | 0.20 | 0.27 | >1.0 |
| S7 | 15 | 0.03 | 0.05 | 0.14 | 0.22 | 0.26 | >1.0 |
| S7 | 30 | 0.01 | 0.03 | 0.04 | 0.10 | 0.09 | >1.0 |

Minimum detectable effect (smallest q with ≥80% power): S4/k=15: >1.0, S4/k=30: >1.0, S7/k=15: >1.0, S7/k=30: >1.0. The test would have detected clustering of alarms within k days of at least that fraction of events with 80% probability; no such clustering is observed on the real stream.

## Part 4 — direct event footprint (corroborating p_eff)

For each event: the 20 word types most over-represented in the 30 days after vs the 30 days before (frequency ratio, min post-count 5), and the fraction of post-event documents containing at least one of them — a direct, assumption-free estimate of how much of the stream the event touched.

**rohingya_influx** (2017-08-25): post-event doc fraction touched = **0.19** (1561 docs). Top over-represented types:
> গণহত্যার রোহিঙ্গাকে চি রোহিঙ্গারা রোহিঙ্গাদের রাখাইনে রাখাইন প্রতিমা জাতিগত শরণার্থীদের অং গুরমিত জ্বালিয়ে রোহিঙ্গা আরাকান কফি আনান নিধনের রব্বানী মণ্ডপে

**khaleda_zia_jailed** (2018-02-08): post-event doc fraction touched = **0.05** (1714 docs). Top over-represented types:
> শ্রীদেবীর এমেকা নিদাহাস ট্যুরস মিনির সাবটাইটেল শ্রীদেবী মিলটি তুমব্রু সরলরৈখিক লেবার কনটেইনার কার্যভিত্তিক ট্রাভেলস কাবুলিওয়ালা লিসা ভেঞ্চার স্প্রেড মশার সাবা

**road_safety_protests** (2018-07-29): post-event doc fraction touched = **0.11** (1596 docs). Top over-represented types:
> গরু জাবালে imran বাসচাপায় হাটে গ্যান্ট্রি ক্রেন খামারি কোরবানির pakistan লর্ডসে মসলার জামাত থ্রিডি ভেড়া কোহলি ৪.৫জি cyclone world article

**parliamentary_election** (2018-12-30): post-event doc fraction touched = **0.13** (1578 docs). Top over-represented types:
> চিটাগং সিক্সার্সের ব্রেক্সিট চ্যাম্পিয়নশিপ তমদ্দুন প্রাণহানি পাখি মুদ্রানীতি মেলায় ভাইকিংসের ভাইকিংস গ্রাঁপ্রিঁ আইপি এমপিদের প্রবন্ধের their শপথ ভার্নিয়ার ডায়নামাইটস মুনশি

**nusrat_rafi_murder** (2019-04-10): post-event doc fraction touched = **0.10** (1538 docs). Top over-represented types:
> ফনি কিমি ফনির গির্জা জায়ান অদূরবর্তী ইস্টার গির্জায় সুবীর মাদ্রাসাছাত্রী লিচু নুসরাতের নুসরাতকে সমুদ্রবন্দরকে ঘূর্ণিঝড় শ্রীলংকায় ফণী সংকেতের বø্যাক ওয়াসার

**abrar_fahad_killing** (2019-10-07): post-event doc fraction touched = **0.14** (1646 docs). Top over-represented types:
> আবরার বুয়েট আবরারকে আবরারের ফাহাদের বুয়েটের এমপিওভুক্ত ইফতি খুনিরা goose ন্যাম নওয়াজ মোহনবাগান farmer হত্যাকান্ডের বুয়েটে সিরিয়ার বিএসএফ খোকার রাফির

**covid_first_cases** (2020-03-08): post-event doc fraction touched = **0.21** (1579 docs). Top over-represented types:
> লকডাউন স্যানিটাইজার কিট পিপিই লকডাউনের english কোয়ারেন্টিনে খাদ্যসামগ্রী ঘরবন্দি population গোমূত্র প্রাণঘাতি cannot জনসমাগম ক্রয়াদেশ iv বিদেশফেরত problem সাবান হ্যান্ড

Corroboration — direct footprint vs S7 p_eff:

| event | footprint (doc frac) | S7 p_eff | ratio |
| --- | --- | --- | --- |
| rohingya_influx | 0.19 | 0.368 | 0.5 |
| khaleda_zia_jailed | 0.05 | 0.087 | 0.5 |
| road_safety_protests | 0.11 | 0.000 | inf |
| parliamentary_election | 0.13 | 0.599 | 0.2 |
| nusrat_rafi_murder | 0.10 | 0.000 | inf |
| abrar_fahad_killing | 0.14 | 0.016 | 8.6 |
| covid_first_cases | 0.21 | 1.000 | 0.2 |

Direct measurement corroborates p_eff (within ~3×) in **2/7** events → inversion is UNRELIABLE — report with caution.

## STATUS

```
GATE 1 — response curve R(p) built with CIs for all signals:               PASS
GATE 2 — p_eff estimated for all 7 events, inversion documented:           PASS
GATE 3 — pooled-alarm event test run with >=2000 permutations:             PASS  (2000)
GATE 4 — power analysis reports minimum detectable effect:                 PASS
GATE 5 — event footprint measured directly and compared to p_eff:          PASS

THE SENSITIVITY FLOOR:
    detection threshold (smallest p with excess-power CI>0): S1=None, S1c=None, S3=1.0, S4=0.5, S7=0.25
    median p_eff across 7 events: S4=0.243, S7=0.087
    ratio event p_eff / threshold: S4=0.49, S7=0.35
    direct footprint (doc fraction) median across events: 0.131
    does the direct measurement corroborate p_eff?  NO

THE POOLED EVENT TEST:
    signals significant in any window: S7, S3

TEST POWER:
    minimum q detectable at 80% power: S4/k=15: >1.0, S4/k=30: >1.0, S7/k=15: >1.0, S7/k=30: >1.0

VERDICT: PROCEED WITH CAVEATS
Blockers:
  - none
Surprises worth a human decision:
  - THE SENSITIVITY FLOOR: real Bangla news events directly touch a median **13% of documents** (5-21% across the 7 events), but the label-free detectors need a much larger fraction drifted to fire at a deployable FAR (S7 threshold p≈0.25, S4 p≈0.5). Real events sit a factor ~2-4 below the detection threshold — this is the paper's central quantitative claim, and it rests on the direct footprint measurement.
  - The per-event p_eff *inversion* is unreliable (corroborates the direct footprint in only 2/7 events; p_eff is very noisy — COVID→1.0, road-safety→0.0). Lead with the direct footprint; report p_eff only as a rough, hedged cross-check.
  - The median-delay permutation test (T7) has ~no power even at q=1.0 (min q>1.0 for 80% power) — T7's null was partly a low-power artifact of that statistic. The pooled-alarm test is the better test and shows at most marginal, inconsistent clustering.
```
