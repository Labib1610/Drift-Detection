# HOW DRIFT IS DETECTED — the pipeline, ADWIN, and the baselines, with real examples

This file answers, concretely: **what is "drift," how does the pipeline turn raw news into a
drift alarm, how does ADWIN actually fire, which report/result file to open to see it (with a
real example from this repo), and in plain words what the `embed` and `classify` steps do.**

Companions: [`PROJECT_WORKFLOW.md`](PROJECT_WORKFLOW.md) (the science in depth),
[`RESULTS_ANALYSIS.md`](RESULTS_ANALYSIS.md) (every result key), and the T1–T9 reports.

---

## 1. The pipeline, annotated (what flows where)

```
                     +-------------------------+
                     | data/raw/bn_potrika.dvc |   4.3 GB of raw Bangla news (DVC-tracked)
                     +-------------------------+
                                  |
                             +---------+   prepare.py  (T2)
                             | prepare |   clean + build the chronological "panel"
                             +---------+   -> data/interim/bn_panel.parquet  (99,900 docs)
             ____________________|________________________________________
            |                    |                    |                    |
       +---------+          +-----------+         +----------+         +--------+
       | streams |          | fertility |         | classify |         | embed  |
       +---------+          +-----------+         +----------+         +--------+
       streams.py (T3)      fertility.py (T3/4)   classifier.py (T6)   embeddings.py (T6)
       11 orderings:        tokenize + window +    train a frozen       LaBSE vectors +
       1 real + 10          z-score the signals    newspaper classifier MMD vs 2016 pool
       shuffled "nulls"     S1,S4,S7,...           -> error = S6        -> S5
       -> data/streams/     -> features/windows/   -> features/         -> features/
          *_perms.npz          *.parquet              classifier/*         embeddings/*
                                      \                  |                   /
                                       \                 |                  /
                                        +----------------+-----------------+
                                                   +--------+
                                                   | detect |   detect.py  (T5-T9)
                                                   +--------+   CALIBRATE each signal to a
                                                                matched false-alarm rate on the
                                                                nulls, then run ADWIN on the REAL
                                                                stream -> alarms + reports T5..T9
```

**One-line summary of each box:**
- **prepare** — clean the raw CSVs into one tidy, date-sorted table (`bn_panel.parquet`).
- **streams** — make 11 orderings of that table: ordering #0 = the real timeline; #1–#10 =
  random shuffles that have *no* real drift (the "null" used to set the alarm sensitivity).
- **fertility** — for each tokenizer, cut the stream into ~2000-word windows and compute the
  per-window drift signals (fertility S1, novelty S4, novel-mass S7, …), z-scored.
- **classify** — the supervised baseline S6 (explained in §6).
- **embed** — the embedding baseline S5 (explained in §5).
- **detect** — calibrate every signal to the same false-alarm rate, then run **ADWIN** on the
  real stream and record the alarms. This is where "drift is detected."

---

## 2. What "drift" means here (tokenizer / concept drift), plainly

A **tokenizer** has a frozen dictionary of subword pieces. **Fertility** = pieces ÷ words. When
the news drifts toward words the tokenizer never saw (new events, new slang, new names), words
fragment into more pieces, so fertility and *novelty* (share of never-before-seen words) go up.
**Concept drift** = the input distribution has moved away from what a model was trained on. The
project turns these into per-window numbers and asks: **can a detector notice the shift?**

- *Tokenizer/fertility drift* = the signals **S1, S1c, S3, S4, S7** (computed from raw text,
  no labels).
- *Meaning drift* = **S5** (embedding MMD).
- *Model-degradation drift* = **S6** (a frozen classifier's error rising).

All of them are fed to the **same** change-detector (ADWIN) at the **same** false-alarm rate so
the comparison is fair.

---

## 3. Where to SEE that drift was detected (the hard files)

| you want… | open this |
|---|---|
| the alarms themselves (dates, counts, per signal) | **`results/detection_metrics.json`** |
| the frozen detector settings used (δ\*) | **`results/calibration.json`** |
| a readable detection summary + the first "lead-time" story | **`reports/T5_report.md`**, **`reports/T5b_report.md`** |
| the picture of signals + alarm marks + the COVID line | **`reports/figs/T5_detection_timeline.png`** |
| the honest validity verdict (do the alarms mean anything?) | **`reports/T7_report.md`** + **`reports/figs/T7_alarm_census.png`** |
| the per-window signal values ADWIN consumed | **`features/windows/bn_panel__<tokenizer>__perm00.parquet`** (`z_S1, z_S4, z_S7, …`) |
| the supervised drift curve (S6) | **`results/classifier_metrics.json`** |
| the embedding drift series (S5) | **`features/embeddings/s5__perm00.parquet`** |

> `perm00` = the real chronological stream; `perm01..10` = the shuffled nulls used only to set
> sensitivity.

---

## 4. HOW ADWIN detects drift — mechanism + a REAL example from this repo

### 4.1 The mechanism in plain words
**ADWIN (ADaptive WINdowing)** reads the signal one window at a time (each window ≈ 2000 words
≈ a slice of chronological time). It keeps a growing memory of recent values and constantly
asks: *"is the average of the newer part noticeably higher/lower than the average of the older
part?"* If the gap is bigger than a statistical threshold (a Hoeffding bound controlled by the
sensitivity knob **δ** — larger δ = lower threshold = more sensitive), ADWIN declares **drift**,
raises an **alarm**, and forgets the old part. So **what it detects is a change in the mean
level of the signal stream** — the signal has moved to a new regime. The window's `median_date`
tells you *when* in calendar time.

It consumes the **z-scored** signal (centred on the early "reference" period), so "alarm" means
"this signal is now far above/below where it sat when the model was fresh."

### 4.2 A real worked example (S7 = novel-token fertility, XLM-R, real stream)
Taken directly from `features/windows/bn_panel__xlm-roberta-base__perm00.parquet` +
`results/calibration.json` (reproduce with the snippet in §7):

- The frozen sensitivity for this detector is **δ\* = 0.7** (from `calibration.json`, chosen so
  it false-alarms only ~1/1000 windows on the shuffled nulls).
- Feeding `z_S7` window-by-window, the z-level of novelty **rises from a 2016 mean of ≈ 0.20 to
  a 2020 mean of ≈ 0.89** — the stream is drifting.
- At **detection-window 9631**, whose median date is **2020-03-12** (4 days after Bangladesh's
  first COVID cases on 2020-03-08), the z-values ran:
  ```
  window:   9627   9628   9629   9630   9631   9632
  z_S7:     1.12   0.34   0.36   0.35   2.74   0.15
  ```
  ADWIN saw the newer mean jump well above the older mean and **raised an alarm at window
  9631** → recorded as `delay_days: 4` from the COVID date.

That exact alarm is stored in **`results/detection_metrics.json`** under the key
`"S7|xlm-roberta-base|raw" → "0.001"`:
```json
{ "delta": 0.7, "alarmed": true, "n_alarms": 21, "delay_days": 4,
  "near_tstar": true, "alarm_dates": ["2016-08-23", ... , "2020-03-12", ...] }
```

**So: drift IS detected** — ADWIN fires an alarm (window 9631, 2020-03-12) when novelty spikes
near COVID. That is the literal "drift detected" event, visible as a mark in
`reports/figs/T5_detection_timeline.png`.

### 4.3 The honest caveat (read this — it is the project's main finding)
ADWIN *fires*, but firing **near** one event is not the same as **detecting** that event. Look
at the same record: it has **21 alarms scattered across 2016–2020** (18 of them *before*
COVID). The "4-day delay" is partly luck — a detector that alarms 21 times will land near any
date by chance. **T7's permutation test proved this rigorously:** for every signal (fertility,
novelty, the embedding baseline S5, *and even the supervised S6*), the alarms are **no closer
to real events than to random dates** (p > 0.05). **T8–T9** explain why: real Bangla news
events change only a few % of the stream (a "sensitivity floor"), below what these detectors
need — only a pandemic-scale shift is lexically large enough.

→ **ADWIN genuinely detects mean-shifts in the signal** (and does so cleanly on *synthetic*
injected drift — see `reports/T5b_report.md`, where detection power rises with drift
intensity). On *real* Bangla events, the alarms exist but are not reliably event-aligned. The
trustworthy conclusion lives in **T7/T8/T9**, not in the raw delays of T5.

---

## 5. What `embed` (S5) does — simple example

**Goal:** detect drift in *meaning*, not just words — the strongest label-free baseline.

**How, step by step:**
1. `embeddings.py` runs each article through **LaBSE**, a model that turns any sentence into a
   list of 768 numbers (an "embedding") capturing its meaning; articles about similar things
   land near each other in that 768-dimensional space.
2. It fixes a **reference pool** of 2,000 early-2016 articles — a cloud of points that
   represents "what the news looked like when the model was fresh."
3. For each later window, it takes that window's article-embeddings and measures **how far this
   cloud is from the reference cloud**, using **MMD²** (Maximum Mean Discrepancy with an RBF
   kernel). MMD² ≈ 0 if the two clouds come from the same distribution, and grows as they
   diverge. `S5 = MMD²`.

**Toy picture (2-D instead of 768-D):**
```
reference 2016 articles  ·· · ·  (a blob centred near the origin)
a 2018 window            · ·· ·   overlaps the blob         -> MMD² small  -> z_S5 ≈ 0
a 2020 COVID window              ·· ··  shifted far away     -> MMD² large  -> z_S5 high
```
So S5 rises when the *topics/meaning* of the news move away from 2016 — even if the exact words
differ from what fertility would catch. **File:** `features/embeddings/s5__perm00.parquet`
(columns `S5`, `z_S5`); the frozen kernel bandwidth is in
`results/embeddings_calibration.json` (`gamma`, `median_dist`). Its result: even this strong
baseline does **not** reliably beat chance on real events (T7), because of the same sensitivity
floor.

---

## 6. What `classify` (S6) does — simple example

**Goal:** the *incumbent* drift detector we compare against — "just watch a deployed model's
error rate."

**How, step by step:**
1. `classifier.py` trains a small online classifier (River logistic regression over hashed
   bag-of-words) to answer: *given an article's words, which newspaper published it?*
2. It trains **only on the first 10%** of the stream (early 2016), then **freezes** it — this
   stands in for a model you deployed in 2016 and never retrained.
3. It then measures that frozen model's **error rate per window** across 2016→2020.
   `S6 = per-window error rate`. As the world drifts away from 2016, the frozen model gets more
   wrong, so its error climbs — and that climb is the drift signal.

**Toy example:**
- In 2016, words like *"cricket, Eid, একাদশ (eleventh), জিয়া"* strongly hint which outlet wrote
  the piece; the frozen model learns those associations and is ~86% accurate.
- In 2020, the news is full of *"লকডাউন (lockdown), করোনা (corona), কোয়ারেন্টিন (quarantine)"* —
  words the 2016 model barely weighted — so it guesses the publisher **worse**.
- Measured on this repo (`results/classifier_metrics.json`), the frozen model's accuracy slides
  **monotonically**:
  ```
  2016: 0.859   2017: 0.789   2018: 0.718   2019: 0.670   2020: 0.668
  ```
  i.e. error rose from 0.14 → 0.33. That rising-error curve is **S6**, and it is a *textbook*
  concept-drift signature (the class mix is held at 33.3% each year, so the drop is drift, not
  a class-balance artifact).

**File:** `features/classifier/publisher_frozen__perm00.parquet` (`err`, `z`); the summary in
`results/classifier_metrics.json`. There is also **S6p** (the same model but it *keeps
learning*) — its error stays flat, showing that a self-updating model *masks* drift (which is
exactly why you might want a label-free early-warning signal in the first place).

---

## 7. See it yourself — one snippet

```python
import pandas as pd, numpy as np, json
from river import drift

w = pd.read_parquet("features/windows/bn_panel__xlm-roberta-base__perm00.parquet").sort_values("window_idx")
n = len(w); det = w.iloc[int(0.10*n):].reset_index(drop=True)      # detection epoch
z = det["z_S7"].to_numpy(float); dates = pd.to_datetime(det["median_date"])
delta = json.load(open("results/calibration.json"))["calibration"]["S7|xlm-roberta-base|raw"]["targets"]["0.001"]["delta"]

a = drift.ADWIN(delta=delta); alarms = []
for i, x in enumerate(z):
    a.update(float(x))
    if a.drift_detected:
        alarms.append(i)
print("alarms:", len(alarms))                                   # -> 21
print("alarm dates:", [str(dates[i].date()) for i in alarms])   # scattered 2016-2020, incl. 2020-03-12
```
Change `z_S7` to `z_S4` (novelty), or load `features/classifier/publisher_frozen__perm00.parquet`
(`z`) for S6, or `features/embeddings/s5__perm00.parquet` (`z_S5`) for S5, to watch any signal's
alarms the same way.

---

## 8. The whole thing in one paragraph

`prepare` makes a clean, date-ordered table of Bangla news; `streams` orders it into the real
timeline plus 10 shuffles; `fertility` turns each ~2000-word window into drift signals
(fertility S1, novelty S4, novel-mass S7), `classify` adds the supervised error signal S6, and
`embed` adds the meaning-shift signal S5. `detect` sets each signal's alarm sensitivity so they
all false-alarm equally often on the shuffles, then runs **ADWIN** on the real timeline:
whenever a signal's z-level shifts to a new regime, ADWIN fires an alarm (e.g. S7 at window
9631 / 2020-03-12). Those alarms are in `results/detection_metrics.json` and
`reports/T5*_report.md`. The catch — proven in `reports/T7_report.md` — is that on real Bangla
events the alarms don't line up with the events better than random dates (the sensitivity
floor), even though ADWIN detects injected synthetic drift cleanly. So: **the detection
machinery works; the real-world events are simply too small for it — that is the paper's
result.**
