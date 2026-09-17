# PROJECT WORKFLOW — the whole research, from scratch, explained

**Project:** Label-Free Concept Drift Detection via Tokenizer Fertility
**Companion:** [`PROJECT_HANDBOOK.md`](PROJECT_HANDBOOK.md) (how to run the code). This
document explains **what we are doing and why**, every concept from first principles, the
nine experiments in order, and the final result — with worked numeric examples so anyone,
including a reader new to the topic, can follow it.

---

## 1. The one-sentence claim we set out to test

> **Tokenizer fertility is a label-free, cheap, model-free signal that could detect when a
> text stream drifts away from what a model was trained on — potentially *earlier* than
> waiting for the model's accuracy to drop.**

Why anyone cares: in production you rarely get ground-truth labels quickly (a model
classifies millions of documents; nobody hand-checks them). So "watch the error rate" is
often impossible in real time. If a signal computed **from the raw input text alone** — no
labels, no model inference — could warn you that the world has moved, that would be valuable.
Tokenizer fertility is an intuitive candidate for that signal. This project tests, rigorously
and honestly, whether it actually works.

**The honest bottom line (established across T1–T9):** on real Bangla news it does *not* buy
you warning time over error monitoring — but the reason is precise and defensible (a
"sensitivity floor"), the methodology is the contribution, and it *does* work in controlled
conditions. A clean, mechanism-backed negative is a stronger paper than an over-claimed
positive.

---

## 2. Vocabulary, from zero (with tiny examples)

**Concept drift.** The input distribution changes over time. New words, new topics, new
events. Example: before 2020 the word "লকডাউন" (lockdown) barely appears in Bangla news;
after March 2020 it is everywhere. A model trained on 2016 data slowly becomes wrong.

**Tokenizer.** A fixed dictionary that chops text into subword pieces. Its vocabulary is
**frozen at training time**. Example (XLM-R): the word `শব্দ` → `["▁শব্দ"]` (1 piece); a rare
word it never saw might split into `["▁শ", "ব্দ", ...]` (many pieces).

**Fertility.** = (number of subword tokens) ÷ (number of words). It measures how badly the
tokenizer fragments the text. A word the tokenizer "knows" = ~1 piece; a novel/foreign/rare
word = many pieces. **Intuition:** when text drifts toward words the tokenizer never saw,
fertility rises — *before* any model runs, *without* any label. That is the whole idea.

  - Worked example. A 4-word Bangla sentence. XLM-R produces 12 tokens → fertility 12/4 = 3.0.
    mBERT produces 18 → 4.5. Qwen (byte-level, poor Bengali coverage) produces 39 → 9.75.
    Same text, very different fertility per tokenizer — which is why we must **normalize**
    (Section 5) before comparing.

**Word (the denominator).** Defined **once**, identically everywhere, using Unicode
UAX-#29 word boundaries via ICU (`PyICU`'s `BreakIterator`). This matters: naive
whitespace-splitting under-counts words in Arabic (clitics) and stdlib `re`'s `\w+`
*over*-counts Bangla ~2.5× because it breaks conjunct characters at the hasant. ICU is the
project's frozen word definition. (Verified in T1: ICU vs whitespace r=0.9998; `re \w+`
inflates counts.)

**Window.** We don't score per document (documents have wildly different lengths → noisy).
We concatenate documents in stream order and cut into **windows of ~2000 words**, then
compute one fertility number per window. Uniform-size windows make the noise comparable.

**Stream.** The documents in an order. **Stream 0** = the real chronological order (2016 →
2020). **Streams 1–10** = random shuffles of the same documents — these are the **null**: by
construction they contain no temporal drift, so any alarm on them is a false alarm.

**Z-score.** To compare signals with different scales, transform each window value to
`z = (value − μ_ref) / σ_ref`, where μ_ref, σ_ref are computed on a frozen early
"calibration epoch" (standing in for "when the model was deployed"). After this every signal
is unit-free, mean 0 / variance 1 during calibration.

**ADWIN.** An online change-detector (from the `river` library). You feed it the z-scored
window values one by one; it raises an alarm when the recent mean differs enough from the
older mean. It has a sensitivity knob **δ** (larger δ = more sensitive = more alarms).

**False-alarm rate (FAR).** How often a detector alarms on the null (shuffled) streams,
where there is no real drift. **The central methodological rule:** compare every detector at
the **same FAR**, so no signal wins just by being tuned looser. We pick each signal's δ* to
hit a common target FAR on the nulls, then freeze it, then measure on the real stream.

**Lead time.** How many days *earlier* a label-free signal alarms than the supervised
error-rate detector, at matched FAR. Positive lead = the free signal warned you first.

---

**Tokenizer family (three kinds, and why it matters).** Every tokenizer belongs to one of
three families, and each marks "this piece starts a new word" differently. We must detect the
family to count fertility variants correctly.
- **WordPiece** (mBERT): word-start pieces are bare, *continuation* pieces begin with `##`.
  Example: `শব্দ` → `["শ", "##ব", "##্দ"]` = 1 word-start + 2 continuations.
- **SentencePiece** (XLM-R): word-start pieces begin with `▁` (U+2581, a visible underscore);
  anything without it is a continuation. Example: `["▁নতুন", "▁শব্দ"]` = two word-starts.
- **Byte-level BPE** (Qwen, Llama, BLOOM): word-start pieces begin with `Ġ` (U+0120); the
  vocabulary is literally bytes, so rare scripts fragment into many byte-pieces. This is why
  Qwen/Llama have fertility 7–10 on Bangla — they barely "know" Bengali and fall back to
  encoding raw UTF-8 bytes.
The code auto-detects the family by the *fraction* of the vocabulary carrying each marker
(`▁` checked before `Ġ`, because the byte-level alphabet never contains `▁`).

**Byte-fallback token.** When a SentencePiece tokenizer meets a character it cannot encode, it
emits literal byte tokens like `<0xE0>`. A rising byte-fallback rate (S2) means novel
scripts/emoji are appearing. Byte-level tokenizers have *no* byte-fallback concept (they are
already all bytes), so S2 is emitted as **null** for them, never a fake zero.

**OOV / unseen type.** "Out-of-vocabulary" at the *word* level: a word type (a distinct
lowercase ICU word) that did not appear in the frozen vocabulary epoch. S4 counts these. It
touches no tokenizer at all — it is pure lexical novelty — which is exactly why it is the
baseline fertility must beat.

**Three epochs (by window position).** vocabulary epoch `[0, 5%)` freezes the word-type
vocabulary V; reference epoch `[5%, 10%)` computes the μ_ref/σ_ref used to z-score; detection
epoch `[10%, end)` is everything we actually test on. Splitting the first two is what fixed a
subtle bug where S4 was identically zero in its own calibration window (see T4).

**Null stream & the permutation test.** The "null" is any arrangement of the data with no real
temporal structure. Two uses: (1) *shuffled null streams* (permutations of the documents) to
calibrate the false-alarm rate; (2) the *permutation test* (T7) — hold a signal's alarm dates
fixed and shuffle the *event* dates instead, to ask "are the alarms closer to real events than
to random dates?" If not, the signal isn't detecting events.

**Bootstrap confidence interval (CI).** To put error bars on a number (say, detection power at
p=0.25), resample the 20 replicates *with replacement* thousands of times, recompute the
number each time, and take the 2.5th/97.5th percentiles. If that interval excludes zero, the
effect is real; if two signals' intervals overlap, they are statistically indistinguishable.

**Detection power & excess power.** *Power* = the fraction of synthetic replicates in which a
detector fires after the injected changepoint. *Excess power* = power at drift intensity p
minus power at p=0 (the false-positive floor). Subtracting the floor is essential: without it,
a detector that alarms constantly looks "powerful" even with no drift.

**Sensitivity floor.** The smallest amount of drift (fraction of the stream that must change)
a detector needs before it fires at a deployable false-alarm rate. The project's punchline:
real news events change *less* than that floor, so they go undetected — not because the
detector is broken, but because the events are lexically small.

## 3. The data

**Potrika** — 665K Bangla news articles, six newspapers (Inqilab, Jugantor, Ittefaq, Kaler
Kontho, Jaijaidin, Somoyer Alo), eight topics, 2014–2020. Each article has: text, category,
headline, **publication date**, source. The publication date is what makes a chronological
stream possible.

The downstream supervised task (for the baseline) is **publisher prediction**: given an
article's text, which of the outlets published it? Chosen because it needs no annotation and
is genuinely drift-sensitive (outlets change their beat and style over years).

---

## 4. The signals we monitor (formulas + examples)

Per window, over that window's documents:

| id | name | formula | needs labels? | intuition |
|---|---|---|---|---|
| **S1** | fertility | Σtokens / Σwords | no | the headline signal |
| **S1b** | topic-adjusted fertility | fertility with topic mix held at the 2016 level | no | controls for topic drift |
| **S1c** | type-weighted fertility | mean pieces over the *distinct* word types | no | every word counts once (not by frequency) |
| **S2** | byte-fallback rate | Σ byte-fallback tokens / Σtokens | no | spikes on novel scripts/emoji (null for byte-level tokenizers) |
| **S3** | continuation ratio | Σ continuation pieces / Σtokens | no | fertility variant |
| **S4** | unseen-type rate | word tokens whose type ∉ frozen vocab / all word tokens | no | **the critical cheap baseline** — pure novelty |
| **S7** | novel-token fertility mass | Σ pieces of novel tokens / Σ pieces of all tokens | no | novelty *weighted by* how badly it fragments |
| **S8/S9** | mean fertility of novel / seen types | mean f(t) over unique novel / seen types | no | the *mechanism*, not detectors |
| **S5** | embedding MMD | MMD² of LaBSE embeddings vs a fixed reference pool | no | strong but GPU-heavy label-free baseline |
| **S6** | classifier error rate | per-window error of a frozen publisher classifier | **yes** | the incumbent we're trying to beat |
| **S6p** | prequential error | same but the model keeps learning | yes | a self-updating model masks drift |

**S4 is non-negotiable.** The first thing a reviewer asks is "isn't fertility just a proxy
for out-of-vocabulary rate, which is trivial?" So we run OOV rate (S4) as an explicit
baseline and check whether fertility beats it. (It does not reliably — see T7.)

---

## 4B. Every signal in depth (definition, worked example, what it catches, how it fails)

To explain these to anyone, use this running toy window: **10 documents, 2000 ICU words
total, 1500 distinct word types.** Suppose the frozen vocabulary V (from the 5% vocab epoch)
contains 1470 of those types, so **30 types are "novel."** Those 30 novel types happen to
occur 45 times in total (novel *tokens* = 45 of 2000 word tokens). With XLM-R the window
tokenizes to **4200 subword tokens**; seen words average 2.7 pieces, novel words average 4.5
pieces (novel words fragment worse — they're rare/new). We'll compute every signal on this one
window.

### S1 — fertility (the headline signal)
- **Definition:** `S1 = (total subword tokens) / (total ICU words)`.
- **Toy value:** `4200 / 2000 = 2.10`.
- **Catches:** overall fragmentation. If the *whole* stream shifts toward words the tokenizer
  handles badly, S1 rises.
- **Fails (this is the project's key finding):** it is a *frequency-weighted average*. Novel
  words are rare (here 45/2000 = 2.25% of tokens), so even though each fragments badly, they
  barely move the average. This is **dilution** — S1 stays flat while novelty genuinely rises.
- **Example of dilution:** if novelty doubles from 2.25% to 4.5% of tokens, S1 moves by roughly
  `ΔS4 × (novel_fert − seen_fert) = 0.0225 × (4.5 − 2.7) ≈ +0.04` tokens/word — invisible under
  normal window-to-window noise.

### S1b — topic-adjusted fertility
- **Definition:** S1 recomputed with the topic mix held at its 2016 (calibration-epoch) level:
  `S1b = Σ_t w_t^ref · f_t^window`, where `w_t^ref` is topic t's 2016 share and `f_t^window` is
  the mean fertility of topic-t documents in the window.
- **Why:** the Bangla panel holds the *publisher* mix constant but the *topic* mix drifts
  (International rose 7.7%→18.1% over 2016–2020). A reviewer will ask "is rising fertility just
  a shift toward topics that tokenize worse?" S1b answers it: if S1 rises but S1b does not, the
  drift is compositional (topic), not lexical.
- **Reading:** in this project S1b ≈ S1 ≈ flat, so topic composition is not the story.

### S1c — type-weighted fertility (the direct dilution test)
- **Definition:** the mean of `f(t)` over the **distinct** word types in the window, each type
  counted **once** regardless of frequency.
- **Toy value:** average pieces over the 1500 distinct types (seen types ~2.7, novel ~4.5) ≈
  `(1470·2.7 + 30·4.5)/1500 ≈ 2.74`.
- **Catches:** whether giving every distinct word equal weight (so a rare novel word counts as
  much as "the") escapes dilution. It is the *direct* test of the dilution hypothesis.
- **Fails:** in this project S1c is **as flat as S1** — even equal-weighting distinct types,
  novel types are too few relative to the seen vocabulary to move the mean. Dilution is deep.

### S2 — byte-fallback rate
- **Definition (SentencePiece only):** `Σ <0xHH> tokens / Σ tokens`. **Null** for byte-level
  BPE (no such concept) and reported as UNK-rate for WordPiece.
- **Catches:** sudden appearance of characters the tokenizer cannot encode — new scripts,
  emoji, transliteration.
- **Reading:** on Bangla news it stays ~0 (the corpus is clean Bangla script), so its z-score
  is degenerate for most tokenizers — an honest null, documented, not hidden.

### S3 — continuation ratio
- **Definition:** `Σ continuation pieces / Σ tokens` (continuation = `##` for WordPiece, "no
  ▁/Ġ prefix" for the others). Higher = more within-word fragmentation.
- **Toy value (XLM-R):** if 2000 pieces start a word and 2200 are continuations,
  `S3 = 2200/4200 = 0.52`.
- **Catches:** the same phenomenon as S1 from a different angle (it correlates ~0.74–0.91 with
  S1 — a fertility variant, not an independent signal).

### S4 — unseen-type rate (THE critical baseline)
- **Definition:** `(word tokens whose type ∉ V) / (all word tokens)`. Purely lexical — **no
  tokenizer involved**, so it is identical across all five tokenizers.
- **Toy value:** `45 / 2000 = 0.0225` (i.e. 2.25% of word tokens are novel).
- **Catches:** raw novelty — how many words are new relative to the frozen vocabulary. It rises
  cleanly and monotonically across 2016→2020 (final-10% z ≈ +0.63).
- **Why it is non-negotiable:** the obvious cheap competitor to fertility. If fertility can't
  beat "just count new words," the fertility contribution is thin. In this project **corr(S1,
  S4) is only −0.35…+0.20** — so fertility carries *different* information than novelty (good),
  but under matched-FAR detection they perform similarly.

### S7 — novel-token fertility mass (novelty × fragmentation)
- **Definition:** `Σ_{tokens whose type ∉ V} f(type) / Σ_{all tokens} f(type)` — the share of
  the *total subword mass* contributed by novel words. Algebraically `S7 = S4 × (A / S1)` where
  A is the token-weighted fertility of novel tokens.
- **Toy value:** novel pieces = 45·4.5 = 202.5; total pieces = 4200 → `S7 = 202.5/4200 = 0.048`.
- **Catches:** novelty *weighted by* how badly it fragments — the best-performing fertility
  signal (final-10% z ≈ +0.75–0.82, COVID Q1-Q2 ≈ +0.98). It is the one that recovers a real
  drift signal from the diluted S1.
- **Subtlety (T7):** because A/S1 is nearly constant in time, S7 is close to a rescaling of S4
  (corr 0.82–0.93). It *is* a statistically distinct signal (nonzero partial correlation), but
  in matched-FAR detection S7 and S4 are indistinguishable.

### S8 / S9 — mean fertility of novel / seen *types* (the mechanism, not detectors)
- **Definition:** S8 = mean `f(t)` over the unique novel types; S9 = the same over seen types.
- **Toy value:** S8 ≈ 4.5, S9 ≈ 2.7, so **S8 − S9 ≈ 1.8** ("excess fragmentation" of novel
  words).
- **Purpose:** they are *not* detectors — they explain *why* S1 is flat. The dilution identity
  is `S1(W) − S1_ref ≈ S4 × (S8tok − S9tok)`: novelty rate (~6%) times excess fragmentation
  (~1.8) ≈ 0.1 — tiny. (`S8tok/S9tok` are the token-weighted versions that make the identity
  close exactly; S8/S9 are the type-weighted "do novel *types* fragment worse?" versions.)

### S5 — embedding MMD (the strong label-free baseline)
- **Definition:** encode every document with LaBSE (a multilingual sentence embedder) once;
  per window compute **MMD²** — Maximum Mean Discrepancy — between the window's embeddings and
  a fixed 2000-document reference pool, using an RBF (Gaussian) kernel whose bandwidth is set
  once on the reference pool and **frozen**.
- **MMD in one sentence:** a number that is ~0 if two sets of points come from the same
  distribution and grows as they diverge; here it measures how far a window's *meaning* has
  moved from the reference period. Formula: `MMD² = mean k(x,x′) + mean k(y,y′) − 2·mean k(x,y)`
  for window points x and reference points y.
- **Role:** the accurate-but-expensive (GPU) baseline. The label-free fertility signals aim to
  recover most of MMD's detection power at O(1) CPU cost.
- **Reading:** even S5 does **not** reliably lead the supervised detector on real events
  (permutation p=0.33) — the sensitivity floor limits the strong baseline too.

### S6 / S6p — the supervised reference (the incumbent we're compared against)
- **S6 (frozen):** train a River online logistic-regression classifier to predict the
  publisher from 2¹⁸ hashed bag-of-words features, using only the first 10% of the stream, then
  **freeze** it. `S6 = per-window error rate` of that frozen model. As the world drifts away
  from 2016, the frozen model gets more wrong → error rises. Its accuracy falls **monotonically
  0.859 → 0.789 → 0.718 → 0.670 → 0.668** across 2016→2020 — a textbook drift reference and the
  thing a label-free signal must beat on warning time.
- **S6p (prequential):** the same classifier but it keeps learning (test-then-train). A
  self-updating model *masks* drift (its error stays low), so S6p ≈ flat — reported as a
  contrast, to show why "just retrain continuously" hides the very drift we want to detect.
- **Why publisher, not topic:** the panel holds publisher shares at exactly 33.3% every year,
  so the class prior is flat by construction and any error-rate change is drift, not a
  class-balance artifact. That is the whole reason the T2 panel design matters here.

### The statistical tools, plainly
- **ADWIN (ADaptive WINdowing):** keeps a growing window of recent values; whenever the mean of
  an older sub-window differs from a newer sub-window by more than a threshold set by δ, it
  declares a change and drops the old data. Feed it z-scores; it emits alarm positions. Larger
  δ ⇒ lower threshold ⇒ more alarms.
- **FAR calibration:** for each δ on a grid, run ADWIN on the 10 shuffled null streams, count
  alarms ÷ null windows = false-alarm rate; pick the largest δ whose FAR ≤ 0.001. This is the
  "everyone wrong equally often" fairness rule. δ* is frozen to `results/calibration.json`.
- **Wilcoxon signed-rank test:** a paired, non-parametric test — used to ask "across the 5
  tokenizers, is S7's delay consistently shorter than S4's?" With only 5 pairs it is
  underpowered; we report the exact p and don't over-claim.
- **Spearman rank correlation:** monotone-relationship strength — used in T9 to ask "does
  detection delay fall as event footprint rises?" across the 7 events (n=7 → suggestive, not
  significant).
- **Decile decomposition (T6):** bucket every word type into a reference-epoch frequency decile
  (1 = most frequent … 11 = unseen), then split ΔS1 exactly into *composition* (which words are
  used), *within-bucket* (how those words tokenize), and *interaction* terms — an exact,
  non-correlational explanation of the S1 sign flip between tokenizer families.

## 5. The four methodological pillars (this is what makes it a paper)

### 5.1 Fixed-size windows, not documents
Greedy packing: walk the stream, accumulate whole documents until cumulative ICU words ≥
2000, close the window, repeat. ~12,765 windows for the Bangla panel. This removes
document-length heteroscedasticity so ADWIN reacts to drift, not to length.

### 5.2 The split calibration epoch (the T4 fix)
Three epochs by window position:
- **Vocabulary epoch [0, 5%)** — freeze the set of word types V (used by S4: "seen" = in V).
- **Reference epoch [5%, 10%)** — compute μ_ref, σ_ref for **every** signal.
- **Detection epoch [10%, end)** — everything downstream.

Why split? In T3 we used one 10% epoch for both, but then S4 ≡ 0 over that whole epoch (every
type in the vocab window is "seen"), giving σ_ref = 0 and undefined z-scores. Splitting fixes
it: V comes from [0,5%), so windows in [5%,10%) already contain some novelty → σ_ref(S4) > 0.
**Freeze μ_ref, σ_ref — never recompute on later data** (that would leak the future into the
normalizer; it's the point-in-time-correctness discipline a feature store enforces).

### 5.3 FAR calibration on shuffled nulls (spec §3, the core)
For each (signal, tokenizer): sweep δ over a grid (1e-6 … 0.99), run ADWIN on each of the 10
shuffled null streams, count alarms ÷ null windows = FAR. Pick **δ\* = the largest δ whose
FAR ≤ target** (target = 1 alarm / 1000 windows). Freeze all δ\* to
`results/calibration.json`. **Only then** run on the real stream. This guarantees every
detector is "wrong equally often," so whichever alarms first genuinely carries more
information. **No δ\* is ever chosen using real-stream data** — that would be the leakage that
invalidates the whole comparison.

### 5.4 Matched comparison
Because all detectors sit at the same FAR, comparing their detection delays / lead times is
fair. The supervised S6 is the reference; label-free signals are measured against it.

---

## 6. A fully worked example (trace one window end to end)

1. **Documents.** Take the first ~7 chronological panel documents whose ICU word counts sum
   to ≥ 2000. Say they total 2160 words. That is **window 0**.
2. **Tokenize** (XLM-R). Those 2160 words produce, say, 4400 subword tokens. **S1 = 4400 /
   2160 = 2.04.** (This is roughly XLM-R's panel-average fertility.)
3. **Word types.** The window contains ~1500 distinct ICU word types. If the frozen
   vocabulary V (built from the first 5% of windows) contains 1480 of them, then 20 are
   "novel." Suppose novel *tokens* (with repetition) are 60 of the 2160 word tokens → **S4 =
   60 / 2160 = 0.028.**
4. **Novelty mass.** Those 60 novel tokens fragment worse (mean 3.8 pieces) than seen tokens
   (mean 2.0). Novel pieces = 60·3.8 = 228; total pieces ≈ 4400 → **S7 = 228 / 4400 = 0.052.**
5. **Z-score.** If over the reference epoch S1 had μ_ref = 2.05, σ_ref = 0.08, then this
   window's **z(S1) = (2.04 − 2.05)/0.08 = −0.13** — essentially flat.
6. **Detect.** Feed z(S1) window-by-window to ADWIN at the frozen δ\*. If the recent mean of
   z(S1) drifts far enough from the older mean, ADWIN raises an alarm at that window; its
   median date maps the alarm back to a calendar date.
7. **Compare.** Do the same for S4, S7, S5, S6. The event of interest (COVID, 2020-03-08) has
   a known date; "detection delay" = days from that date to the first alarm at or after it.

This single trace is the atom the whole project is built from; the experiments differ only in
*which* signals, *which* stream order, and *what statistic* we compute over the alarms.

---

## 7. The nine experiments, in order (question → method → result)

Each experiment is one `md_files/T*.md` brief → one `src` script → one `reports/T*_report.md`
with a PASS/FAIL gate block and a VERDICT. The arc is a story of increasingly rigorous
self-scrutiny.

### T1 — Data audit (`audit_potrika.py`)
**Q:** do the dates parse, and is the corpus usable?
**Result:** 664,884 rows; dates parse **100.00%** (`%Y/%m/%d`; the only unparseable string
was `2019/02/30`, an invalid calendar date). All six sources present. Publisher imbalance
**32×** (Inqilab 205k vs Jaijaidin 6k). Discovered two families — a dated `RawDataset` (use
this) and an undated `BalancedDataset` (unusable). Also found a category-label collision
(`science-and-tech` vs `Science_Technology`). **PROCEED.**

### T2 — Build the streams (`prepare.py`)
**Q:** turn raw CSVs into a clean, confound-free chronological stream.
**Key design — the panel.** The six outlets *enter the corpus at different times*, so a naive
all-six stream confounds lexical drift with a changing publisher mix. Fix: **`bn_panel`** uses
only the three outlets with full 2016–2020 coverage (Inqilab, Jugantor, Kaler Kontho) and
samples an **equal quota per (month × publisher) cell** (555 = 100000 / (60 months × 3)). The
result: publisher shares are **exactly 33.3% in every year** → any fertility drift must be
lexical, not compositional. `bn_full` (all six, uniform-across-time) is the appendix stream.
**Result:** cleaned pool 649,691 docs; `bn_panel` = **99,900** docs (33,300 each), 0/180
cells underfilled; `bn_full` = 95,935. Calibration epoch (first 10%) = **2016-01 → 2016-06**.
Output is byte-identical across runs. **PROCEED.**

### T3 — Signals S1–S4 + windowing + z-scoring (`streams.py`, `fertility.py`)
**Q:** compute the label-free signals and take the first honest look at whether fertility
moves.
**Result:** 5 tokenizers loaded (mBERT wordpiece; XLM-R sentencepiece; Llama/Qwen/BLOOM
byte-level BPE). **Raw mean fertility:** mBERT 2.72, XLM-R 2.04, BLOOM 1.68, Qwen 6.99, Llama
7.71 (byte-level fragment Bangla heavily — expected). 12,765 windows. **The sobering first
read:** mean z(S1) over the final 10% (late 2020) is **flat and inconsistent in sign** across
tokenizers (−0.37 to +0.14), while S4 (novelty) rises cleanly. Also found z(S4) was
degenerate (σ_ref = 0) → fixed in T4. **PROCEED WITH CAVEATS.**

### T3b — CC-News re-probe (`probe_ccnews.py`)
**Q:** do Hindi/Arabic/Ukrainian have enough dated data for the other four languages?
**Result:** with the corrected schema (`published_date`, `publisher`, `plain_text`), dates
parse ~100%. Volume: ar healthy, hi thinner, **uk very thin** (~1.5k/yr). Flagged an honest
caveat: the bounded head-of-shard scan understates temporal coverage. Decision surfaced: uk
likely needs substituting (Turkish/Russian from MLSUM). **PROCEED WITH CAVEATS.**

### T4 — Calibration fix + signal variants + the dilution mechanism (`fertility.py`)
**Q:** *why* is S1 flat while S4 rises, and does a differently-weighted fertility escape it?
**Method:** the split epoch (§5.2) — fixes σ_ref(S4) > 0. New signals S1c (type-weighted),
S7 (novel-token mass), S8/S9 (novel/seen fertility). The **dilution decomposition:**
`S1(W) − S1_ref ≈ novelty term`.
**Result — the mechanism, quantified:** by late 2020 only **~6% of tokens are novel**
(S4 = 0.059), and although novel *types* fragment ~50% worse than seen ones (S8 ≫ S9), 6% ×
modest excess ≈ negligible → **S1 stays flat by dilution**. Meanwhile **S4 rises +0.63 and S7
rises +0.75–0.82** (monotone, peaking at COVID). And **corr(S1, S4) is only −0.35 to +0.20**
≪ 0.95 → fertility is *not* a duplicate of OOV rate (good for the contribution). The exact
decile decomposition (T6) later shows composition vs interaction reproduces the byte-level
vs SP sign difference. **PROCEED WITH CAVEATS.**

### T5 — FAR-calibrated ADWIN detection (`detect.py`)
**Q:** turn descriptive z-scores into calibrated detection with matched FAR; how much earlier
does a label-free signal fire than the supervised S6?
**Result (first pass):** S7 detects COVID in **4 days**, S4 in 7, plain fertility in 12 — an
apparent label-free lead. Redundancy test: S7 is a *distinct* signal from S4 (R² 0.67–0.87,
nonzero partial correlation). **But this pass had flaws** (below). **PROCEED WITH CAVEATS.**

### T5b — Detection done properly (`detect.py`)
**Q:** three flaws in T5 — did the calibration actually bind, was "best tokenizer" cherry
-picked, and is one COVID observation enough?
**Method:** extend δ grid to 0.99 (now the FAR constraint **binds**); report the full 5×5
delay grid (no cherry-picking); and the load-bearing addition — **semi-synthetic drift
injection**: build streams that are early-pool documents before a known changepoint W\* and a
mixture (late-pool with probability p) after, sweep drift intensity p ∈ {0.05…1.0}, 20
replicates each, and measure detection **power** and delay with confidence intervals.
**Result:** on synthetic drift, S4 and S7 genuinely detect (power rises with p, CIs exclude
zero at p≥0.25). **But S7 vs S4 are statistically indistinguishable** (delay CIs overlap at
every intensity; Wilcoxon p=0.25). The T5 "S7 leads S4 leads S1" ranking **does not survive**.
Honest. **PROCEED WITH CAVEATS.**

### T6 — Baselines: control arm, decile decomposition, S5 (MMD), S6 (supervised) (`classifier.py`, `embeddings.py`, `detect.py`)
**Q:** put the label-free signals against a real supervised detector and the strongest
label-free baseline, at matched FAR.
**Method:** S6 = frozen publisher classifier's per-window error (River logistic, 2¹⁸ hashed
features); S5 = LaBSE embedding MMD vs a frozen reference pool; plus a **p=0 control arm**
(no drift → false-positive floor) and the exact decile decomposition of the sign anomaly.
**Results:** the frozen classifier degrades **monotonically** (accuracy 0.859 → 0.789 → 0.718
→ 0.670 → 0.668 across 2016→2020) — a textbook drift reference. The sign flip is reproduced
directionally by composition-vs-interaction in the decile decomposition. **The headline:** at
matched FAR, **no label-free signal (S1/S1c/S3/S4/S7) nor even the MMD baseline (S5) reliably
leads S6** across the 7 verified events (median leads negative; sign-test p all > 0.05).
**PROCEED WITH CAVEATS.**

### T7 — Is anything actually detecting anything? (`detect.py`) — **the pivotal check**
**Q:** six signals clustered around 22–44-day delays looks like alarm-rate noise, not event
detection. Is it?
**Method:** the **alarm census** (nobody had reported a total alarm count!) + a **permutation
test**: hold each signal's alarm positions fixed, draw 1000 random sets of 7 pseudo-event
dates, and ask whether the real events' median delay is unusually small.
**Result — the project's central finding:** every signal alarms frequently (A = 13–26 per
tokenizer), and **NO signal passes the permutation test** — including the supervised S6
(p=0.230), S5 (0.327), S4 (0.146), S7 (0.124). **The real-event "delays" in T6 were
indistinguishable from delays to random dates.** So T6's lead-time comparison was between two
chance processes. Also, the S7−S4 excess-power gap includes zero at replicate level (not
significant). This reframes everything: the question is no longer "who leads whom" but "does
*anything* detect discrete events" — and the answer is no. **PROCEED WITH CAVEATS.**

### T8 — The sensitivity floor (`detect.py`)
**Q:** T7's two facts look contradictory — detectors work on synthetic drift but not on real
events. Reconcile them into one number.
**Method:** build a response curve R(p) from the synthetic streams, measure real events'
response, and compare event size to the detection threshold; plus a higher-powered pooled
-alarm test and a power analysis; plus a **direct** event-footprint measurement.
**Result:** the detection thresholds (smallest p with excess-power CI > 0) are S7 ≈ 0.25, S4 ≈
0.5; real events directly touch a **median ~13% of documents** — a factor ~2–4 **below** the
threshold. The median-delay permutation test has **≈ zero power** (min q > 1.0), confirming
T7's null was partly low-power. **The sensitivity floor: real news events are too small for
these detectors to fire at a deployable FAR.** (The synthetic→real inversion `p_eff` was
found invalid and removed in T9.) **PROCEED WITH CAVEATS.**

### T9 — Event footprint, measured properly (`detect.py`) — the final correction
**Q:** T8's "13%" actually measured 30-day **news turnover**, not event impact. Separate them.
**Method:** for each event compute (a) turnover (top-20 over-represented types) and (b) event
footprint from **event-specific seed terms** (e.g. রোহিঙ্গা, করোনা), prefix-matched against
ICU types to catch inflections — and crucially the **excess over the pre-event baseline**,
which isolates the event from year-round vocabulary. Compare event dates to 60 random dates.
**Result — the clean, corrected finding:**

| event | turnover | event-seed post | pre-baseline | **excess** |
|---|---|---|---|---|
| covid_first_cases | 0.212 | 0.598 | 0.082 | **+0.516** |
| rohingya_influx | 0.191 | 0.155 | 0.007 | **+0.148** |
| abrar_fahad_killing | 0.141 | 0.095 | 0.053 | **+0.043** |
| road_safety_protests | 0.105 | 0.179 | 0.138 | **+0.041** |
| nusrat_rafi_murder | 0.103 | 0.059 | 0.022 | **+0.037** |
| khaleda_zia_jailed | 0.048 | 0.113 | 0.085 | **+0.028** |
| parliamentary_election | 0.131 | 0.298 | 0.442 | **−0.143** |

**Only COVID is lexically large** (+0.52); the rest add ≤4% over baseline, and the election is
*negative* (election vocabulary peaked in the campaign month *before* polling and is common
year-round). The 7 event dates' turnover is **not unusual** vs random dates (median 0.13 vs
baseline 0.086). Delay trends down with footprint for S4/S7 (right direction) but is not
significant at n=7. **Newsworthy ≠ lexically large — hand-picked event dates are a poor
ground truth for drift evaluation.** **PROCEED WITH CAVEATS.**

---

## 8. The synthesized conclusion (what the paper says)

1. **Controlled setting (works):** novelty-weighted fertility signals (S4, S7) are valid,
   FAR-calibratable drift detectors — on semi-synthetic streams their detection power rises
   monotonically with drift intensity and their CIs exclude zero above p≈0.25.
2. **Plain fertility is diluted (mechanism):** token-weighted fertility (S1) stays flat
   because novelty is token-rare (~6%), so the excess fragmentation of novel words is diluted
   away — quantified exactly by the decomposition. Weighting by novelty (S7) or counting
   novelty directly (S4) recovers the signal, but S7 and S4 are statistically
   indistinguishable.
3. **Real events (does not work):** on real Bangla news, no label-free signal — nor a
   supervised error monitor (S6), nor an MMD embedding baseline (S5) — aligns its alarms with
   discrete real-world events better than chance (permutation test).
4. **The reason (sensitivity floor):** real news events touch only ~a few percent of the
   stream above baseline (only a pandemic-scale event clears the bar), far below the ~25–50%
   these FAR-calibrated detectors need. Hand-picked event dates are anyway swamped by ordinary
   news turnover.

**Positioning:** the NLP community studies temporal *degradation* (does accuracy drop?) and
the streaming-ML community studies *detection* (does an alarm fire?), and they barely speak.
This work connects them with a **label-free signal evaluated under matched false-alarm rates
across five tokenizer families**, and its contribution is (i) the matched-FAR evaluation
methodology, (ii) the dilution mechanism, and (iii) the **permutation-based validity check
and sensitivity-floor quantification** that most drift papers omit — delivered as a rigorous
negative-with-mechanism.

---

## 9. Honest limitations (the paper's limitations section, pre-written)

- **One language for the full analysis.** Only Bangla/Potrika went through T1–T9. English
  (NewsSumm) and Turkish (MLSUM) are set up but not yet run (see the handbook §11).
- **Hand-picked events are a weak ground truth.** T9 shows most "major" events are lexically
  tiny and swamped by turnover; synthetic streams are the commensurable evidence.
- **n = 7 events** → the real-data footprint↔delay relationship is suggestive, not
  significant.
- **Publisher-panel design is Bangla-specific.** Single-outlet corpora (Turkish MLSUM) need a
  different label (topic) or label-free-only treatment.
- **Determinism is version-pinned** (parquet/tokenizer library versions).

---

## 10. Where to take it next

- **Add English + Turkish** (handbook §11) — English as the high-resource, long-span
  sanity-check; Turkish (agglutinative, high fertility) as a stress test with a topic label.
- **Better ground truth** — instead of hand-picked event dates, use the *lexically largest*
  windows (top decile of turnover excess) as pseudo-changepoints; that is a fairer test than
  news salience.
- **Longer synthetic streams** matched to the real detection epoch length, to tighten the
  power curves.
- **The five-language cross-tokenizer table** the spec envisions, once ≥3 languages are
  through the pipeline.

---

*For exact reproduction commands, file/function references, DVC/Git setup, and the
language-addition recipes, see [`PROJECT_HANDBOOK.md`](PROJECT_HANDBOOK.md).*
