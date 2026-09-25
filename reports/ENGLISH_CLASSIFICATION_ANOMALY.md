# WHY DOES ENGLISH CLASSIFICATION "DO GREAT" vs BANGLA? — a diagnostic

**Context.** On the English (NewsSumm) branch the supervised classification task (signal **S6**
— predict the publisher from the text) performs noticeably *better* than on Bangla, and that
is confusing. This note explains **what "better" can mean, the most likely causes ranked, how
to check each one, and what actually matters for the paper.** It is written from the Bangla
`main` branch, so it reasons from the shared design + general principles and gives you exact
commands to run **on the English branch** to settle it.

**The headline you must internalise first:** *high classifier accuracy is **not** the paper's
result.* The result is (1) whether the frozen model **degrades over time** (that degradation
*is* the drift signal S6) on a **balanced** task, and (2) whether the **label-free** signals
lead S6. A very accurate English classifier can be completely uninformative — or even
misleading — about drift. So "English does great" is only good news if it survives the checks
below.

---

## 1. First pin down what "does great" means

There are three different things people call "English does better." Decide which you're seeing
— the diagnosis differs:

- **(A) Higher `ref_accuracy` / `year_accuracy`** in `results/classifier_metrics.json` (the
  frozen model is just more accurate on English). Bangla publisher `ref_accuracy` = **0.895**
  on 3 balanced classes; if English is ~0.97–0.99, this is (A).
- **(B) A steeper, cleaner drift curve** (`year_accuracy` falls a lot more across the years, so
  S6 is a stronger drift reference on English).
- **(C) Better *detection*** — label-free signals (S4/S7) or S6 actually align with events /
  lead on English where they didn't on Bangla.

Run this on the English branch to see which:
```python
import json
m = json.load(open("results/classifier_metrics.json"))["publisher"]
print("ref_acc:", m["ref_accuracy"])
print("per year:", m["year_accuracy"])          # is the level high? does it fall steeply?
print("monotone drift:", m["frozen_error_monotone_rising"])
```

---

## 2. The most likely causes, ranked, with a check for each

### Cause 1 — the English panel is NOT balanced (class-prior artifact) ⚠️ most likely "problem"
The whole point of the Bangla `bn_panel` is that each publisher is **exactly 33.3% in every
year**, so any accuracy change is drift, not a change in class mix. NewsSumm has **36
newspapers with very skewed volumes**; if the English `prepare` did not enforce the equal
`(month × publisher)` quota (or used a different K, or one paper dominates the chosen panel),
then the classifier can score high just by **predicting the majority class**, and the
"year_accuracy" curve then tracks **class-balance shift, not drift**. This would make English
"do great" for the wrong reason and make its S6 **not comparable** to Bangla's.

**Check (English branch):**
```python
import pandas as pd
d = pd.read_parquet("data/interim/en_panel.parquet")
print("overall publisher share:\n", d.publisher.value_counts(normalize=True))
d["yr"] = pd.to_datetime(d.date).dt.year
print("\nper-year share:\n", d.groupby("yr").publisher.value_counts(normalize=True))
```
- **If shares are ~1/K each, every year** → balanced, this is not the problem; go to Cause 2/3.
- **If one publisher dominates or the mix shifts across years** → *this is the bug.* The high
  accuracy is a prior artifact, and S6's drift is confounded. Fix by rebuilding the English
  panel with the same equal-quota `build_panel` design as Bangla (see the handbook), or report
  a **balanced-accuracy** / macro-F1 instead of raw accuracy.

Compare against Bangla, where this is clean:
```json
"publisher year_accuracy": {"2016":0.859,"2017":0.789,"2018":0.718,"2019":0.670,"2020":0.668}
// each class is 33.3% every year, so this fall is pure drift.
```

### Cause 2 — outlet-identifying boilerplate (leakage of the label into the text)
English news text frequently carries **masthead / byline / dateline / section / copyright
boilerplate** — "The Times of India", "(Reuters)", "NEW DELHI:", "Follow us on…", standardized
footers. The classifier uses **hashed bag-of-words**, so it will trivially memorise these
outlet fingerprints and hit near-perfect publisher accuracy **that has nothing to do with
article content or with drift**. Bangla Potrika bodies are comparatively clean, so this
inflates English much more than Bangla.

**Check:** eyeball a few documents and look for repeated outlet strings; then test whether the
accuracy collapses when you strip the first/last tokens:
```python
d = pd.read_parquet("data/interim/en_panel.parquet")
for pub, g in d.groupby("publisher"):
    print(pub, "::", g.text.iloc[0][:120])          # do the openings scream the outlet?
# quick sensitivity test: retrain S6 on text with the first 15 and last 15 words removed;
# if accuracy drops a lot, boilerplate (not content) was driving it.
```
If boilerplate dominates, S6 "detects" the *presence of a masthead*, not concept drift — strip
it (drop the first/last N words, or a regex for known footers) and re-run.

### Cause 3 — legitimately easier + a much longer span (this part is real, not a bug)
Two genuine reasons English can *and should* do better, which are fine to report as long as
Causes 1–2 are cleared:
- **High-resource language + BoW.** English has far less morphological inflection than Bangla.
  Bangla's rich case-suffixing splinters the bag-of-words vocabulary (রোহিঙ্গা / রোহিঙ্গাদের /
  রোহিঙ্গারা are three different BoW features), so the *same* hashed-BoW logistic model is
  simply weaker on Bangla. English words are more "reusable" features → higher accuracy. This
  is expected and benign.
- **NewsSumm spans 2000–2025 (~25 years) vs Potrika 2016–2020 (~5 years).** A frozen model has
  **5× more time** to fall out of date, so its error *rises much more* → S6 looks like a far
  stronger drift detector on English. Also benign — but it means **English and Bangla S6 are
  not on the same footing** (different horizon), so don't compare their delays directly; the
  fair comparison is within each language, label-free vs S6.

**Check:** print the actual date span and number of classes used on the English branch:
```python
d = pd.read_parquet("data/interim/en_panel.parquet")
print("span:", d.date.min(), "→", d.date.max(), "| classes:", d.publisher.nunique(), "| docs:", len(d))
```

### Cause 4 — near-duplicate / syndication leakage
Wire-service articles (AP/PTI/Reuters) republished by several outlets create either **label
noise** (same text, different "publisher" — hurts accuracy) or, if near-duplicates land in both
the frozen-train first-10% and later windows, **train/test leakage** (inflates accuracy). The
Bangla pipeline dedups on normalised-text hash; confirm the English loader did the same.

**Check:**
```python
d = pd.read_parquet("data/interim/en_panel.parquet")
import hashlib
h = d.text.str.lower().str.replace(r"\s+"," ",regex=True).map(lambda t: hashlib.md5(t.encode()).hexdigest())
print("exact-dup rate:", 1 - h.nunique()/len(h))     # should be ~0 if dedup ran
```

### Cause 5 — different K or a much more separable class set
If the English panel used a different number of publishers (K), or 3 very stylistically
distinct outlets (a business daily vs a tabloid vs a sports site), the task is easier than
Bangla's 3 general-news outlets. Not wrong, but note it — accuracy across languages is only
comparable at the **same K** and similar class distinctiveness.

**Check:** confirm K matches `params.yaml → data.top_k_publishers` and list the chosen outlets.

---

## 3. The decision tree (what to conclude)

1. **Is the English panel balanced (Cause 1)?** If no → **that is the problem**: the high
   accuracy is a class-prior artifact and S6's "drift" is confounded. Rebuild the balanced
   panel or switch to balanced-accuracy/macro-F1. Stop here; fix this first.
2. **If balanced, is it boilerplate (Cause 2)?** If stripping mastheads collapses accuracy →
   S6 was detecting outlet fingerprints, not drift. Strip boilerplate and re-run.
3. **If balanced and content-driven**, then English legitimately does better because it is
   high-resource (Cause 3a) and has a much longer drift horizon (Cause 3b). That is fine and
   even expected — but **report it as "not directly comparable to Bangla"** (different span,
   different language difficulty), and keep the real comparison *within* English: do label-free
   signals lead the (now trustworthy) English S6? Also re-verify no duplicate leakage (Cause 4).

---

## 4. Why this does not change the project's story

The paper's claim is about **label-free vs supervised drift detection at matched false-alarm
rates**, and the honest Bangla finding is a **sensitivity floor** (real events are too small
for any detector — label-free or S6 — to beat chance; T7–T9). A stronger English S6 does not
overturn that; it either (a) reflects a class-prior/boilerplate artifact to be fixed, or (b)
reflects English being an easier, longer-horizon control — in which case the interesting
question becomes **whether label-free signals lead English's S6**, tested the same way (T7
permutation test + T8/T9 footprint). If, on English's 25-year span, fertility/novelty *do*
lead a clean, balanced, boilerplate-free S6 and pass the permutation test, that is a genuine
positive result for the high-resource control — exactly what the spec predicted ("if fertility
doesn't lead on the longest span, it won't lead anywhere"). Run T7's permutation test on the
English branch before believing any lead:
```bash
# on the English branch, after fixing Causes 1–2:
python src/detect.py --lang en --params params.yaml
less reports/T7_en_report.md      # does ANY English signal pass the permutation test?
```

---

## 5. TL;DR

- "English does great" is **not automatically good** — first rule out a **class-imbalanced
  panel** (Cause 1, most likely) and **outlet boilerplate** (Cause 2), both of which inflate
  accuracy without measuring drift.
- If it's balanced and content-driven, English legitimately outperforms because it is
  **high-resource** and has a **~25-year drift horizon vs Bangla's 5** — real, but it makes the
  two languages **not directly comparable**; compare label-free-vs-S6 *within* each language.
- The number that matters is **not** classifier accuracy; it is whether any **label-free signal
  leads S6** and **passes the T7 permutation test**. Check that on the English branch before
  concluding English "worked."
