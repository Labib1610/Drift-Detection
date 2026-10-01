# TASK 2 — ns preparation report

- Generated: 2026-09-25T15:20:02
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

Range probed: **2019-07..2024-09** (63 months). A publisher-month counts as covered when it has ≥ **30** cleaned docs (`suggest_min_cell`).

Top 10 publishers by months covered:

| publisher | docs in range | months covered (of 63) |
| --- | --- | --- |
| The Mint | 29 | 0 |
| The Hindu | 18 | 0 |
| Indian Express | 17 | 0 |
| The Economic Times | 8 | 0 |
| The Asian Age | 6 | 0 |
| Hindustan Times | 5 | 0 |
| Business Standard | 3 | 0 |
| Financial Express | 2 | 0 |
| Deccan Chronicle | 1 | 0 |
| Free Press Journal | 1 | 0 |

Docs per year for those publishers:

| publisher | 2019 | 2020 | 2021 | 2022 | 2023 |
| --- | --- | --- | --- | --- | --- |
| The Mint | 0 | 15 | 0 | 12 | 2 |
| The Hindu | 0 | 9 | 3 | 3 | 3 |
| Indian Express | 1 | 1 | 6 | 0 | 9 |
| The Economic Times | 0 | 4 | 0 | 2 | 2 |
| The Asian Age | 0 | 2 | 0 | 2 | 2 |
| Hindustan Times | 0 | 0 | 1 | 0 | 4 |
| Business Standard | 2 | 0 | 0 | 1 | 0 |
| Financial Express | 0 | 0 | 1 | 0 | 1 |
| Deccan Chronicle | 0 | 1 | 0 | 0 | 0 |
| Free Press Journal | 0 | 0 | 0 | 0 | 1 |

**No 3-publisher combination has a common covered month** among the top 10 at ≥30 docs/month. Lower `suggest_min_cell` or `suggest_k`, or narrow the range.

### Panel fill (month × source quota)

Quota per cell: **26** (= 5000 ÷ (63 months × 3 sources) = 189 cells). Cells underfilled: **189 / 189** (100.0%). Panel total: **24** docs.

> ⚠️ **More than 5% of cells underfilled** — the 26-doc quota is too high for the available data; consider lowering it.

Worst 10 cells by shortfall:

| month | publisher | available | taken | shortfall |
| --- | --- | --- | --- | --- |
| 2019-07 | Hindustan Times | 0 | 0 | 26 |
| 2019-07 | The Hindu | 0 | 0 | 26 |
| 2019-07 | The Times of India | 0 | 0 | 26 |
| 2019-08 | Hindustan Times | 0 | 0 | 26 |
| 2019-08 | The Hindu | 0 | 0 | 26 |
| 2019-08 | The Times of India | 0 | 0 | 26 |
| 2019-09 | Hindustan Times | 0 | 0 | 26 |
| 2019-09 | The Hindu | 0 | 0 | 26 |
| 2019-09 | The Times of India | 0 | 0 | 26 |
| 2019-10 | Hindustan Times | 0 | 0 | 26 |

### Realised publisher shares per year in `ns_panel`

(Confound check — each source should be ~33.3% in every year.)

| year | Hindustan Times | The Hindu | The Times of India |
| --- | --- | --- | --- |
| 2020 | 0.0% | 100.0% | 0.0% |
| 2021 | 25.0% | 75.0% | 0.0% |
| 2022 | 0.0% | 100.0% | 0.0% |
| 2023 | 50.0% | 37.5% | 12.5% |

### Topic composition per year in `ns_panel`

(Measured, not corrected — publisher and topic may be correlated, so we size any topic drift before interpreting alarms.)

| year | Business and Finance | Crime and Justice | Health and Wellness | International News | Politics | Science and Technology | Sports | Weather | legal proceedings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | 11.1% | 11.1% | 11.1% | 33.3% | 22.2% | 11.1% | 0.0% | 0.0% | 0.0% |
| 2021 | 0.0% | 25.0% | 50.0% | 0.0% | 25.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2022 | 0.0% | 0.0% | 33.3% | 33.3% | 0.0% | 0.0% | 0.0% | 33.3% | 0.0% |
| 2023 | 0.0% | 12.5% | 12.5% | 37.5% | 0.0% | 0.0% | 25.0% | 0.0% | 12.5% |

### Output files

| stream | docs | date span | total words | mean n_words | file MB |
| --- | --- | --- | --- | --- | --- |
| ns_panel | 24 | 2020-01-03..2023-10-15 | 4,686 | 195.2 | 0.0 |
| ns_full | 171 | 2000-10-07..2023-10-15 | 39,571 | 231.4 | 0.1 |

`ns_full` sampling: uniform across time = equal per-month quota of **16** over 312 months (2000-01..2025-12).

### Calibration epoch of `ns_panel`

First 10% of the panel = first **2** docs, spanning **2020-01-03 → 2020-01-04**.

## STATUS

```
PART 1
GATE 0 — every panel_publishers name exists in the cleaned pool:   PASS   (all found)
GATE 1 — ns_panel.parquet exists, chronologically sorted, schema exact:   PASS   (schema match=True, 24 docs)
GATE 2 — publisher shares within 33.3% ± 2pp in EVERY panel year:   FAIL   (a year exceeds ±2pp)
GATE 3 — <5% of (month × source) quota cells underfilled:   FAIL   (100.0% underfilled)
GATE 4 — ns_full.parquet exists, >= 30 publishers, 2000-01..2025-12:   FAIL   (15 publishers (target: >=30))

VERDICT: BLOCKED
Blockers:
  - none
Surprises worth a human decision:
  - 100.0% of panel cells underfilled (> 5% threshold) — quota may need lowering.
  - ns_full uses equal-per-month sampling (uniform across time); confirm this is the intended reading of the brief.
```
