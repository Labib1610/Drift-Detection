# TASK 2 — bn preparation report

- Generated: 2026-10-02T20:23:33
- Mode: FULL · loader `potrika_csv`
- Params: `params.yaml` · seed 42

Reproduce with:
```
python src/prepare.py --lang bn --params params.yaml --report reports/T2_report.md
```

## PART 1 — bn preparation

### Cleaning ledger

Raw rows read from `data/raw/bn_potrika/RawDataset` (63 files/splits): **664,884**

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

### Publishers in the cleaned pool

6 distinct publishers. Use these exact strings in `panel_publishers`.

| publisher | docs |
| --- | --- |
| Inqilab | 201,572 |
| Jugantor | 189,382 |
| Ittefaq | 162,915 |
| Kaler Kontho | 69,085 |
| Somoyer Alo | 20,390 |
| Jaijaidin | 6,347 |

### Publisher coverage and panel suggestion

Range probed: **2016-01..2020-12** (60 months). A publisher-month counts as covered when it has ≥ **100** cleaned docs (`suggest_min_cell`).

Top 6 publishers by months covered:

| publisher | docs in range | months covered (of 60) |
| --- | --- | --- |
| Inqilab | 201,572 | 60 |
| Jugantor | 189,382 | 60 |
| Kaler Kontho | 53,590 | 60 |
| Ittefaq | 132,154 | 54 |
| Somoyer Alo | 20,390 | 12 |
| Jaijaidin | 6,347 | 3 |

Docs per year for those publishers:

| publisher | 2016 | 2017 | 2018 | 2019 | 2020 |
| --- | --- | --- | --- | --- | --- |
| Inqilab | 36,230 | 26,684 | 45,489 | 42,853 | 50,316 |
| Jugantor | 37,448 | 38,299 | 34,540 | 40,446 | 38,649 |
| Kaler Kontho | 11,092 | 10,934 | 11,504 | 10,581 | 9,479 |
| Ittefaq | 40,991 | 39,248 | 32,337 | 12,139 | 7,439 |
| Somoyer Alo | 0 | 0 | 0 | 0 | 20,390 |
| Jaijaidin | 0 | 0 | 0 | 270 | 6,077 |

**Suggested panel** — longest common covered run for 3 publishers: ['Inqilab', 'Jugantor', 'Kaler Kontho'] over **2016-01..2020-12** (60 months → quota 555 docs/cell at cap 100,000).

```yaml
panel_publishers: ["Inqilab", "Jugantor", "Kaler Kontho"]
panel_start: "2016-01"
panel_end: "2020-12"
```

_This is a starting point: the search looks only at the top publishers by months covered, and ignores topic mix and article length._

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

(Measured, not corrected — publisher and topic may be correlated, so we size any topic drift before interpreting alarms.)

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

`bn_full` sampling: uniform across time = equal per-month quota of **1265** over 79 months (2014-06..2020-12).

### Calibration epoch of `bn_panel`

First 10% of the panel = first **9,990** docs, spanning **2016-01-01 → 2016-06-30**.

## STATUS

```
PART 1
GATE 0 — every panel_publishers name exists in the cleaned pool:   PASS   (all found)
GATE 1 — bn_panel.parquet exists, chronologically sorted, schema exact:   PASS   (schema match=True, 99,900 docs)
GATE 2 — publisher shares within 33.3% ± 2pp in EVERY panel year:   PASS   (all years within tolerance)
GATE 3 — <5% of (month × source) quota cells underfilled:   PASS   (0.0% underfilled)
GATE 4 — bn_full.parquet exists, >= 6 publishers, 2014-06..2020-12:   PASS   (6 publishers (target: >=6))

VERDICT: PROCEED WITH CAVEATS
Blockers:
  - none
Surprises worth a human decision:
  - bn_full uses equal-per-month sampling (uniform across time); confirm this is the intended reading of the brief.
```
