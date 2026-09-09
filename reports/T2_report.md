# TASK 2 — Bangla preparation + CC-News probe report

- Generated: 2026-09-09T08:56:42
- Mode: FULL
- Params: `params.yaml` · seed 42

Reproduce with:
```
python src/prepare.py --lang bn --params params.yaml --report reports/T2_report.md
```

## PART 1 — Bangla preparation

### Cleaning ledger

Raw rows read from `data/raw/bn_potrika/RawDataset` (63 files): **664,884**

| rule | removed | % of raw | remaining |
| --- | --- | --- | --- |
| (rows in) |  |  | 664,884 |
| 4. unparseable/null date dropped | 1 | 0.00% | 664,883 |
| 6. empty/whitespace text dropped | 2,341 | 0.35% | 662,542 |
| 7. exact-duplicate text dropped (kept earliest) | 2,604 | 0.39% | 659,938 |
| 9. n_words < 50 dropped | 10,247 | 1.54% | 649,691 |
| **(rows out / pool)** |  |  | **649,691** |

### Topic label set

Before canonicalisation (9): `Economy`, `Education`, `Entertainment`, `International`, `National`, `Politics`, `Science_Technology`, `Sports`, `science-and-tech`

After canonicalisation (8): `Economy`, `Education`, `Entertainment`, `International`, `National`, `Politics`, `Science_Technology`, `Sports`

Mappings applied: `science-and-tech`→`Science_Technology`

### Panel fill (month × source quota)

Quota per cell: **555** (= 100000 ÷ (60 months × 3 sources) = 180 cells). Cells underfilled: **0 / 180** (0.0%). Panel total: **99,900** docs.

No cell underfilled.

### Realised publisher shares per year in `bn_panel`

(Confound check — each source should be ~33.3% in every year.)

| year | Inqilab | Jugantor | Kaler Kontho |
| --- | --- | --- | --- |
| 2016 | 33.3% | 33.3% | 33.3% |
| 2017 | 33.3% | 33.3% | 33.3% |
| 2018 | 33.3% | 33.3% | 33.3% |
| 2019 | 33.3% | 33.3% | 33.3% |
| 2020 | 33.3% | 33.3% | 33.3% |

### Topic composition per year in `bn_panel`

(Measured, not corrected — publisher and topic are correlated in Potrika, so we size any topic drift before interpreting alarms.)

| year | Economy | Education | Entertainment | International | National | Politics | Science_Technology | Sports |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2016 | 11.6% | 7.0% | 4.3% | 7.7% | 39.8% | 1.7% | 6.5% | 21.3% |
| 2017 | 10.7% | 6.7% | 4.8% | 10.3% | 36.8% | 2.2% | 6.0% | 22.5% |
| 2018 | 8.3% | 6.0% | 3.4% | 12.2% | 42.4% | 6.4% | 4.8% | 16.6% |
| 2019 | 9.3% | 6.6% | 3.4% | 15.7% | 37.4% | 4.7% | 4.6% | 18.5% |
| 2020 | 9.5% | 5.3% | 3.7% | 18.1% | 41.6% | 2.8% | 3.5% | 15.5% |

### Output files

| stream | docs | date span | total words | mean n_words | file MB |
| --- | --- | --- | --- | --- | --- |
| bn_panel | 99,900 | 2016-01-01..2020-12-30 | 28,609,185 | 286.4 | 111.9 |
| bn_full | 95,935 | 2014-06-25..2020-12-30 | 24,187,251 | 252.1 | 95.2 |

`bn_full` sampling: uniform across time = equal per-month quota of **1265** over 79 months (2014-06..2020-12). The T2 brief's phrase *'sample proportionally within each month'* conflicts with *'uniformly across time … do not over-represent high-volume years'*; the equal-per-month reading is used and flagged below for your decision.

### Calibration epoch of `bn_panel`

First 10% of the panel = first **9,990** docs, spanning **2016-01-01 → 2016-06-30**.

## PART 2 — CC-News volume probe

_Not yet run. Execute:_
```
python src/probe_ccnews.py --langs hi,ar,uk --years 2016-2024 --max-rows-per-year 200000
```
_It will replace this section and finalise the STATUS block below._

## STATUS

```
PART 1
GATE 1 — bn_panel.parquet exists, chronologically sorted, schema exact:   PASS   (schema match=True, 99,900 docs)
GATE 2 — publisher shares within 33.3% ± 2pp in EVERY panel year:   PASS   (all years within tolerance)
GATE 3 — <5% of (month × source) quota cells underfilled:   PASS   (0.0% underfilled)
GATE 4 — bn_full.parquet exists, 6 publishers, 2014-06..2020-12:   PASS   (6 publishers: ['Inqilab', 'Ittefaq', 'Jaijaidin', 'Jugantor', 'Kaler Kontho', 'Somoyer Alo'])

PART 2
GATE 5 — hi: top-3 domains each >=15K over >=36 clean months:   PENDING (run probe)
GATE 6 — ar: same condition:                                    PENDING (run probe)
GATE 7 — uk: same condition:                                    PENDING (run probe)

VERDICT: PROCEED WITH CAVEATS (Part 2 pending)
Blockers:
  - none (Part 1)
Surprises worth a human decision:
  - bn_full uses equal-per-month sampling (uniform across time). The brief also says 'proportionally within each month' — confirm which you meant; only affects the appendix stream.
```
