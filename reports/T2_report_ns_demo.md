# TASK 2 — ns preparation report

- Generated: 2026-09-22T16:16:52
- Mode: DEMO (5k subsample) · loader `newssumm_csv`
- Params: `params.yaml` · seed 42

Reproduce with:
```
python src/prepare.py --lang ns --demo
```

## PART 1 — ns preparation

### Cleaning ledger

Raw rows read from `data/raw/NewsSumm/Processed/NewsSumm_processed.xlsx` (1 files/splits): **200**

| rule | removed | % of raw | remaining |
| --- | --- | --- | --- |
| (rows in) |  |  | 200 |
| 4. unparseable/null date dropped | 0 | 0.00% | 200 |
| 6. empty/whitespace text dropped | 0 | 0.00% | 200 |
| 7. exact-duplicate text dropped (kept earliest) | 2 | 1.00% | 198 |
| 9. n_words < 50 dropped | 8 | 4.00% | 190 |
| **(rows out / pool)** |  |  | **190** |

### Topic label set

Before canonicalisation (21): `Automotive`, `Business and Finance`, `Business news`, `Crime and Justice`, `Education`, `Entertainment`, `Environment`, `Health and Wellness`, `Human Interest`, `International News`, `Local News`, `National News`, `Natural Disasters`, `Opinion and Editorial`, `Politics`, `Religion and Spirituality`, `Science and Technology`, `Sports`, `Technology and Gadgets`, `Weather`, `legal proceedings`

After canonicalisation (21): `Automotive`, `Business and Finance`, `Business news`, `Crime and Justice`, `Education`, `Entertainment`, `Environment`, `Health and Wellness`, `Human Interest`, `International News`, `Local News`, `National News`, `Natural Disasters`, `Opinion and Editorial`, `Politics`, `Religion and Spirituality`, `Science and Technology`, `Sports`, `Technology and Gadgets`, `Weather`, `legal proceedings`

Mappings applied: none

### Publishers in the cleaned pool

16 distinct publishers. Use these exact strings in `panel_publishers`.

| publisher | docs |
| --- | --- |
| Indian Express | 62 |
| The Hindu | 40 |
| The Mint | 32 |
| The Economic Times | 18 |
| The Pioneer | 7 |
| Hindustan Times | 7 |
| The Asian Age | 7 |
| Business Standard | 4 |
| Financial Express | 4 |
| The Tribune | 3 |
| Deccan Chronicle | 1 |
| The Telegraph | 1 |
| The Times of India | 1 |
| The Statesman | 1 |
| The Shillong Times | 1 |
| Free Press Journal | 1 |

### Publisher coverage and panel suggestion

Range probed: **2000-01..2025-12** (312 months). A publisher-month counts as covered when it has ≥ **100** cleaned docs (`suggest_min_cell`).

Top 10 publishers by months covered:

| publisher | docs in range | months covered (of 312) |
| --- | --- | --- |
| Indian Express | 58 | 0 |
| The Hindu | 40 | 0 |
| The Mint | 32 | 0 |
| The Economic Times | 18 | 0 |
| The Asian Age | 7 | 0 |
| Hindustan Times | 6 | 0 |
| The Pioneer | 6 | 0 |
| Business Standard | 4 | 0 |
| Financial Express | 4 | 0 |
| The Tribune | 2 | 0 |

Docs per year for those publishers:

| publisher | 2000 | 2001 | 2002 | 2003 | 2004 | 2006 | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Indian Express | 0 | 13 | 1 | 3 | 2 | 3 | 0 | 0 | 11 | 0 | 2 | 3 | 0 | 0 | 1 | 0 | 1 | 2 | 1 | 6 | 0 | 9 |
| The Hindu | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 5 | 0 | 0 | 0 | 2 | 2 | 2 | 1 | 2 | 1 | 9 | 3 | 3 | 3 |
| The Mint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 12 | 2 |
| The Economic Times | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 2 | 0 | 3 | 0 | 0 | 1 | 0 | 0 | 2 | 4 | 0 | 2 | 2 |
| The Asian Age | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 2 | 2 |
| Hindustan Times | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 4 |
| The Pioneer | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 2 | 1 | 0 | 0 | 0 | 0 |
| Business Standard | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 |
| Financial Express | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| The Tribune | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

**No 3-publisher combination has a common covered month** among the top 10 at ≥100 docs/month. Lower `suggest_min_cell` or `suggest_k`, or narrow the range.

### Panel

No `panel_publishers` in the `ns` config — panel not built. See the suggestion above.

### Output files

| stream | docs | date span | total words | mean n_words | file MB |
| --- | --- | --- | --- | --- | --- |
| ns_full | 171 | 2000-10-07..2023-10-15 | 39,571 | 231.4 | 0.1 |

`ns_full` sampling: uniform across time = equal per-month quota of **16** over 312 months (2000-01..2025-12).

## STATUS

```
PART 1
GATE 4 — ns_full.parquet exists, 36 publishers, 2000-01..2025-12:   FAIL   (15 publishers)

VERDICT: BLOCKED
Blockers:
  - none
Surprises worth a human decision:
  - ns_full uses equal-per-month sampling (uniform across time); confirm this is the intended reading of the brief.
```
