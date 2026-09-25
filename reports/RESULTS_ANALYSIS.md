# RESULTS ANALYSIS — what every result file and every key means (Bangla / main branch)

**Purpose.** Reading the raw outputs by hand is hectic. This file explains **every result
file, every key inside it, how the number was produced, and how to read it — with a real
worked example pulled from this branch.** All numbers below are the actual Bangla-panel
results in this repo. For the *science* behind the signals see
[`PROJECT_WORKFLOW.md`](PROJECT_WORKFLOW.md); for *running/editing* see
[`PROJECT_HANDBOOK.md`](PROJECT_HANDBOOK.md).

---

## 0. First: the two kinds of output, and which to trust

There are two layers of output. **Read the reports; the JSON/parquet are the evidence behind
them.**

1. **`reports/T*_report.md`** — human-readable. Each ends with a `## STATUS` block: PASS/FAIL
   gates + a VERDICT + "Surprises worth a human decision." **Start here.** T1→T9 tell the
   story in order.
2. **`results/*.json`** — the machine numbers the reports summarise (calibration δ*, detection
   delays, classifier accuracy, MMD bandwidth).
3. **`features/**/*.parquet` + `data/streams/*.npz`** — the per-document / per-window matrices
   the JSON was computed from. You rarely open these by hand; §5 gives one-line loaders.

**The single most important reading rule (do not skip):** a big number in
`detection_metrics.json` like `delay_days: 4` looks like "S7 detected COVID in 4 days" — **but
T7 proved these delays are not real event detection** (permutation test p > 0.05 for every
signal, including the supervised one). The trustworthy conclusions live in **T7, T8, T9**, not
in the raw delays. §3.2 explains exactly how to see this in the file itself.

---

## 1. Terminology you need to read any result (quick reference)

| term | meaning in one line |
|---|---|
| **stream** | the documents in an order. `bn_panel` = publisher-balanced primary; `bn_full` = all-6-outlets robustness. |
| **perm (permutation)** | an ordering of the stream. `perm_00` = the **real** chronological order. `perm_01..10` = **shuffles = the "null"** (no real drift → any alarm is a false alarm). |
| **window** | ~2000 consecutive ICU words; one row of signal values. ~12,765 windows for `bn_panel`. |
| **signal** | one drift indicator per window: S1 fertility, S4 novelty, S7 novel-token mass, S5 MMD, S6 classifier error, etc. (full list in WORKFLOW §4/§4B). |
| **z-score** | a signal centred/scaled on the reference epoch: `z = (value − μ_ref)/σ_ref`. Detection runs on z, not raw. |
| **epoch** | window ranges: **vocab [0,5%)** (freezes the S4 word-vocabulary), **reference [5%,10%)** (gives μ_ref/σ_ref), **detection [10%,end)** (everything tested). |
| **delta (δ)** | ADWIN's sensitivity knob. Larger δ = more sensitive = more alarms. |
| **δ\*** | the δ chosen by calibration to hit the target false-alarm rate on the nulls, then **frozen**. |
| **FAR (false-alarm rate)** | alarms per window on the null (shuffled) streams. Target = 0.001 = 1 alarm / 1000 windows. |
| **alarm** | a window where ADWIN declares a change. |
| **delay_days** | days from the reference event (t\* = 2020-03-08, COVID) to the first alarm at/after it. |
| **variant `raw` vs `resid`** | `raw` = the z-signal as-is; `resid` = after regressing out window length `n_words` (a robustness check that window-size doesn't drive alarms). |
| **tokenizer** | which subword model produced the fertility: `bert-base-multilingual-cased`, `xlm-roberta-base`, `meta-llama/Llama-3.2-1B`, `Qwen/Qwen2.5-0.5B`, `bigscience/bloom-560m`. |

---

## 2. `results/calibration.json` — the frozen detector settings

**What it is.** For every detector we swept δ over a grid, measured its false-alarm rate on
the 10 shuffled null streams, and picked the δ\* that hits the target FAR. This file freezes
those δ\* so nothing downstream can re-tune them. **This is the fairness backbone of the whole
project: every signal is set to be "wrong equally often," so comparing them is fair.**

**Top-level keys:**
- `grid` — the 14 δ values tried: `[1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5,
  0.7, 0.9, 0.95, 0.99]`.
- `far_targets` — the three FAR targets: `[0.001, 0.01, 0.0001]` (main + 10× looser + 10×
  stricter, for the robustness sweep).
- `calibration` — a dict of **42 entries**, one per detector. The key format is
  **`signal|tokenizer|variant`**, e.g. `S7|xlm-roberta-base|raw`. (42 = 4 tokenizer-specific
  signals × 5 tokenizers × 2 variants + S4 which is tokenizer-independent × 2, etc.)

**Inside one entry** (real example, `S7|xlm-roberta-base|raw`):
```json
{ "signal": "S7", "tokenizer": "xlm-roberta-base", "variant": "raw",
  "null_windows": 115058,                     // total null windows used to measure FAR (≈ 11.5k × 10 perms)
  "far_curve": { "1e-06": 0.0, ... "0.5": 0.000591, "0.7": 0.000834, "0.99": 0.001243 },
  "targets": {
    "0.001":  {"delta": 0.7,  "far_achieved": 0.000834, "extended": false},
    "0.01":   {"delta": 0.99, "far_achieved": 0.001243, "extended": false},
    "0.0001": {"delta": 0.01, "far_achieved": 0.0000869,"extended": false} } }
```
**Key-by-key:**
- `null_windows` — how many null windows the FAR was averaged over (bigger = more reliable
  FAR estimate).
- `far_curve` — `{δ: FAR at that δ}`. **Read it as: "as I make the detector more sensitive
  (δ↑), how often does it false-alarm on shuffled data?"** Here FAR rises from 0 at δ=1e-6 to
  0.00124 at δ=0.99 — a proper monotone curve, which is why the calibration *binds* (the
  target lies inside the curve's range).
- `targets` — for each FAR target, the chosen `delta`, the `far_achieved` at that δ, and
  `extended` (true only if no grid δ met the target and the grid had to be extended).
  **How δ\* is chosen:** the *largest* δ whose FAR ≤ target. For target 0.001 here, δ=0.7
  gives FAR 0.000834 (≤ 0.001) but δ=0.9 gives 0.00114 (> 0.001), so **δ\* = 0.7**.

**How you use it:** you don't tune anything — this file is the frozen input to detection. To
sanity-check a detector, confirm `far_achieved ≤ 2×target` (calibration is honest) and that
the curve is monotone.

---

## 3. `results/detection_metrics.json` — the real-stream alarms (READ THE CAVEAT)

**What it is.** Apply each frozen δ\* to the **real** stream (perm_00) and record what
happened, relative to the COVID reference date **t\* = 2020-03-08**.

**Structure:** same 42 keys (`signal|tokenizer|variant`) → three FAR targets → an object.
Real example, `S7|xlm-roberta-base|raw` at target `0.001`:
```json
{ "delta": 0.7, "alarmed": true, "n_alarms": 21, "delay_days": 4,
  "pre_tstar_alarms": 18, "near_tstar": true,
  "alarm_dates": ["2016-08-23","2016-12-09","2017-01-14", ... "2020-03-12","2020-06-06","2020-09-10"] }
```
**Key-by-key:**
- `delta` — the frozen δ\* used (matches calibration.json).
- `alarmed` — did it alarm at all in the detection epoch?
- `n_alarms` — **total** alarms across 2016–2020 (here 21).
- `delay_days` — days from t\* (2020-03-08) to the first alarm on/after it. Here the first
  alarm ≥ t\* is 2020-03-12 → **4 days**.
- `pre_tstar_alarms` — alarms **before** t\* (here 18). *We do not know if these are real
  earlier drift or false alarms — do not characterise them.*
- `near_tstar` — was any alarm within ±60 days of t\*?
- `alarm_dates` — every alarm's calendar date.

### 3.2 How to read this correctly (the caveat, shown in the numbers)
Look at `alarm_dates`: the 21 alarms are **scattered across the whole 2016–2020 span**, with
**18 of them before COVID**. The "4-day delay" is not evidence of detection — it's that the
detector alarms often (~21 times / ~1640 days ≈ one every ~78 days), so *some* alarm lands
near any date by chance. **T7's permutation test made this rigorous:** shuffling the event
dates, the real events' delays are no smaller than random dates' (p = 0.124 for this exact
signal; p > 0.05 for all, including the supervised S6). **Therefore:** treat
`detection_metrics.json` as *descriptive alarm bookkeeping*, not as proof of event detection.
The honest verdict is in `reports/T7_report.md`. This is the project's central, deliberate
finding — not a mistake in the file.

---

## 4. `results/classifier_*.json` — the supervised baseline S6

### 4.1 `results/classifier_metrics.json` — how well/badly the frozen model does over time
This is the most intuitive result file. Two keys, `publisher` and `topic` (the two prediction
tasks). Real Bangla values:
```json
"publisher": { "ref_accuracy": 0.8952,
   "year_accuracy": {"2016":0.8588,"2017":0.7886,"2018":0.7177,"2019":0.6702,"2020":0.6685},
   "frozen_error_monotone_rising": true },
"topic":     { "ref_accuracy": 0.9361,
   "year_accuracy": {"2016":0.8969,"2017":0.8383,"2018":0.7892,"2019":0.7875,"2020":0.7938},
   "frozen_error_monotone_rising": false }
```
**Key-by-key:**
- `ref_accuracy` — accuracy of the frozen classifier on the reference epoch (first 10%, ~early
  2016) — i.e. how good it is *before* drift. Publisher 0.895 (3 balanced classes; chance =
  0.333, so the model genuinely learns). Topic 0.936 (8 classes; strong).
- `year_accuracy` — accuracy per calendar year **without retraining**. **This is the drift
  signal in its clearest form:** publisher accuracy slides **0.859 → 0.789 → 0.718 → 0.670 →
  0.668** as the world moves away from 2016. That falling curve *is* concept drift, and its
  mirror (error rate) is signal **S6**.
- `frozen_error_monotone_rising` — `true` if accuracy only ever falls (error only rises) year
  on year. Publisher = **true** (a textbook, clean drift reference — this is why publisher was
  chosen). Topic = **false** (topic accuracy dips then recovers a little in 2020 — so topic is
  a noisier drift reference; publisher is the primary target).

> **Why publisher and not topic is the reference:** the `bn_panel` holds each publisher at
> exactly 33.3% every year, so accuracy changes are drift, not class-balance shifts. (Keep
> this fact in mind for the English anomaly — see `ENGLISH_CLASSIFICATION_ANOMALY.md`.)

### 4.2 `results/classifier_calibration.json` — δ\* for the classifier signals
Same shape as `calibration.json` but with three entries: `S6` (frozen publisher error), `S6p`
(prequential publisher error), `topic_frozen`. Each has `target`, `variant`, `far_curve`,
`targets`, `null_windows`. Example `S6` targets:
```json
"0.001": {"delta":0.95,"far":0.000991}, "0.01":{"delta":0.99,"far":0.001069}, "0.0001":{"delta":0.2,"far":0.0000608}
```
Read identically to §2: δ\* is the largest δ with FAR ≤ target (S6 needs a high δ=0.95 to reach
the 0.001 FAR — its error series is smooth, so ADWIN needs to be sensitive to fire at all).

---

## 5. `results/embeddings_*.json` — the MMD baseline S5

### 5.1 `results/embeddings_calibration.json`
```json
"S5": { "far_curve": {δ: FAR ...},
        "targets": {"0.001":{"delta":0.5,"far":0.000765}, "0.01":{...}, "0.0001":{...}},
        "gamma": 0.36492, "median_dist": 1.171, "ref_pool_size": 2000, "null_windows": 115058 }
```
- `far_curve`, `targets` — as in §2 (S5's δ\* at FAR 0.001 is 0.5).
- `gamma` — the RBF-kernel bandwidth **γ = 1/(2·median_dist²)**, set once on the reference pool
  and frozen (recomputing per window would leak the future). Larger γ = the kernel treats
  points as "different" more readily.
- `median_dist` — the median pairwise distance among the 2000 reference-pool embeddings (the
  "median heuristic" that sets γ). Here 1.171.
- `ref_pool_size` — number of reference documents MMD compares each window against (2000,
  sampled from the reference epoch, fixed).
- `null_windows` — as before.

### 5.2 `results/embeddings_status.json`
`{"status":"ok","n_docs":99900,"gamma":0.36492}` — a health flag: `ok` means the LaBSE encode
+ MMD completed and `bn_panel.npy` holds embeddings for all 99,900 docs. If it ever says
`blocked`, the GPU stack (torch/sentence-transformers) wasn't installed — see the handbook.

---

## 6. `data/streams/streams_meta.json` — the null construction, audited
```json
"bn_panel": {"n":99900, "n_perms":11, "seeds":[-1,42,43,...,51], "valid":true, "file":"..."}
```
- `n` — documents in the stream (99,900).
- `n_perms` — 11 orderings (1 identity + 10 shuffles).
- `seeds` — RNG seed per permutation; `-1` marks the identity (perm_00), 42–51 the shuffles.
- `valid` — `true` means every permutation was verified to be a genuine bijection over
  `[0,n)`, perm_00 is the identity, and no two permutations are equal (so the nulls are real
  shuffles, not accidental repeats).

---

## 7. The feature parquet files (the per-window / per-document evidence)

You rarely open these by hand, but here is what each column is.

### 7.1 `features/windows/{stream}__{tokenizer}__perm{NN}.parquet`
One row per window. **Filename tells you which stream, tokenizer, and permutation** (perm00 =
real; 01–10 = nulls). Columns:
- `window_idx` — 0-based window number (also stream order).
- `median_date` — the median publication date of the window's documents (maps a window to
  calendar time).
- `n_docs`, `n_words`, `n_tokens` — documents / ICU words / subword tokens in the window.
- **Raw signals:** `S1, S1b, S1c, S2, S3, S4, S4_type, S7, S8, S9, S8tok, S9tok` (definitions
  in WORKFLOW §4B).
- **Z-scored signals:** `z_S1, z_S1b, z_S1c, z_S2, z_S3, z_S4, z_S4_type, z_S7, z_S8, z_S9`
  (what ADWIN actually consumes; `null`/NaN where a signal is undefined, e.g. z_S2 for
  byte-level tokenizers).
- **Topic mix:** `topic_Economy … topic_Sports` (8 columns) — each window's share of that
  topic (used by S1b and diagnostics).
- `s1b_fallback_hits` — bookkeeping: how often S1b had to fall back to a topic's reference
  mean because that topic was absent in a window.

Example read:
```python
import pandas as pd
d = pd.read_parquet("features/windows/bn_panel__xlm-roberta-base__perm00.parquet")
print(d[["median_date","S1","S4","S7","z_S1","z_S4","z_S7"]].tail())   # late-2020 windows
```

### 7.2 `features/fertility/{stream}__{tokenizer}.parquet`
Per-**document** token counts (before windowing): `doc_id, n_words, n_tokens,
n_byte_fallback, n_continuation`. This is the raw material windows are built from.

### 7.3 `features/type_fertility/{tokenizer}.parquet`
`type, n_pieces` — for every distinct ICU word type, how many subword pieces that tokenizer
splits it into (tokenised in word-initial form). Used by S1c/S7/S8/S9.

### 7.4 `features/classifier/{target}_{variant}__perm{NN}.parquet`
Per-window classifier error: `window_idx, median_date, err` (window error rate),
`z` (its z-score = S6/S6p). `{target}` ∈ {publisher, topic}, `{variant}` ∈ {frozen, preq}.

### 7.5 `features/embeddings/`
- `s5__perm{NN}.parquet` — `window_idx, median_date, S5` (raw MMD²), `z_S5`.
- `bn_panel.npy` — the cached LaBSE embeddings (float16, 99,900 × 768). Regenerating S5 from
  this cache needs no GPU.

### 7.6 `features/windows/zscore_params.json`
The frozen normaliser audit: for every `(stream → tokenizer → "{signal}@perm{NN}")` it stores
`{mu, sigma, n_reference_windows}` — the exact μ_ref and σ_ref used to z-score, so you can
verify no leakage (μ/σ come only from the reference epoch).

---

## 8. `reports/ccnews_probe.json` — the multilingual volume probe (T3b)
Not part of the Bangla result; it's the counts from streaming CC-News for hi/ar/uk. Per
language it stores the acceptance funnel (`seen, score_ok, date_ok, valid`), publisher and
per-month histograms, date formats, and a word-count sample. Read `reports/T3b_report.md` for
the human summary. (Relevant only when adding those languages.)

---

## 9. How each report's STATUS line traces back to a file

| report line | comes from |
|---|---|
| T2 "publisher shares 33.3% each year" | `data/interim/bn_panel.parquet` (`build_panel`) |
| T3/T4 "raw mean fertility per tokenizer" | `features/fertility/*` (Σn_tokens/Σn_words) |
| T3/T4 "σ_ref(S4) > 0", "mean z final 10%" | `features/windows/*` + `zscore_params.json` |
| T5/T5b "δ\* per signal, FAR achieved" | `results/calibration.json` |
| T5/T6 "detection delay, alarms" | `results/detection_metrics.json` |
| T6 "frozen accuracy per year, monotone?" | `results/classifier_metrics.json` |
| T6 "S5 δ\*, γ" | `results/embeddings_calibration.json` |
| T7 "alarm count A, permutation p" | `detection_metrics.json` (counts) + in-memory permutation test |
| T8 "detection threshold, sensitivity floor" | synthetic sweep (in `detect.py`) + `calibration.json` |
| T9 "turnover vs event footprint, excess" | `data/interim/bn_panel.parquet` text + `changepoints` seeds |

---

## 10. Fast recipes — read any result in one line

```python
import json, pandas as pd

# a detector's frozen delta* and achieved FAR at the main target
c = json.load(open("results/calibration.json"))["calibration"]["S7|xlm-roberta-base|raw"]
print(c["targets"]["0.001"])                      # {'delta':0.7,'far_achieved':0.00083,...}

# a detector's real-stream alarms
d = json.load(open("results/detection_metrics.json"))["S4|shared|raw"]["0.001"]
print(d["n_alarms"], d["delay_days"], d["pre_tstar_alarms"])   # note pre_tstar_alarms!

# the frozen classifier's drift curve
print(json.load(open("results/classifier_metrics.json"))["publisher"]["year_accuracy"])

# late-period z of the key signals for one tokenizer
w = pd.read_parquet("features/windows/bn_panel__xlm-roberta-base__perm00.parquet")
tail = w.sort_values("median_date").tail(int(len(w)*0.1))
print(tail[["z_S1","z_S4","z_S7"]].mean())        # the "final 10% mean z" numbers
```

---

## 11. The one-paragraph takeaway (so you don't get lost)

The result files record, at a **matched false-alarm rate**, how a set of label-free signals
(fertility S1/S1c/S3, novelty S4, novel-mass S7, embedding-MMD S5) and a supervised reference
(classifier error S6) alarm on the real Bangla stream and on shuffled nulls. The **frozen
classifier genuinely degrades** (publisher accuracy 0.86 → 0.67), so drift is real. But the
**alarm delays in `detection_metrics.json` are not real event detection** — T7's permutation
test shows no signal (label-free *or* supervised) aligns its alarms with real events better
than chance, and T8/T9 explain why (real news events change only a few percent of the stream,
below the detectors' sensitivity floor; only COVID is lexically large). So: read the STATUS
blocks of **T7, T8, T9** for the conclusions; use the JSON/parquet described above as the
auditable evidence behind them.
