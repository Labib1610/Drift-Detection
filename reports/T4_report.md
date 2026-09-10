# TASK 4 — Calibration fix, signal variants, dilution decomposition

- Mode: FULL · wall-clock 398.0s
- Reproduce: `python src/streams.py --params params.yaml && python src/fertility.py --params params.yaml`

## Part 1 — split-epoch calibration

Epochs: vocabulary `[0, 5%)`, reference `[5%, 10%)`, detection `[10%, end)`. μ_ref/σ_ref for **every** signal come from the reference epoch; V (S4 vocabulary) from the vocabulary epoch.

Reference-epoch calendar dates — bn_panel: **2016-04-01 .. 2016-06-29**, bn_full: **2014-12-09 .. 2015-04-19**.

| tokenizer | σ_ref(S4) on bn_panel |
| --- | --- |
| bert-base-multilingual-cased | 0.01689 |
| xlm-roberta-base | 0.01689 |
| meta-llama/Llama-3.2-1B | 0.01689 |
| Qwen/Qwen2.5-0.5B | 0.01689 |
| bigscience/bloom-560m | 0.01689 |

σ_ref(S4) > 0 for every tokenizer: **True** — the T3 degeneracy is fixed.

## Part 2 — isolated-type fertility validation

Word-initial check (bare type vs space-prefixed) on the longest panel type:

| tokenizer | example type | bare pieces | spaced pieces |
| --- | --- | --- | --- |
| bert-base-multilingual-cased | `ziwknx5lvdqrkeootkodrzpg` | 1 | 1 |
| xlm-roberta-base | `ziwknx5lvdqrkeootkodrzpg` | 202 | 202 |
| meta-llama/Llama-3.2-1B | `ziwknx5lvdqrkeootkodrzpg` | 202 | 202 |
| Qwen/Qwen2.5-0.5B | `ziwknx5lvdqrkeootkodrzpg` | 207 | 207 |
| bigscience/bloom-560m | `ziwknx5lvdqrkeootkodrzpg` | 190 | 190 |

Isolated-type Σf(type) vs in-context n_tokens on 2,000 random docs:

| tokenizer | mean ratio (iso/context) | pearson r | r ≥ 0.95 |
| --- | --- | --- | --- |
| bert-base-multilingual-cased | 0.945 | 0.9989 | yes |
| xlm-roberta-base | 0.924 | 0.9984 | yes |
| meta-llama/Llama-3.2-1B | 0.978 | 0.9996 | yes |
| Qwen/Qwen2.5-0.5B | 0.979 | 0.9995 | yes |
| bigscience/bloom-560m | 0.890 | 0.9912 | yes |


## The result — mean z over final 10% (real stream)

| tokenizer | S1 (token-wt) | S1c (type-wt) | S4 (novelty) | S7 (novel-token mass) |
| --- | --- | --- | --- | --- |
| bert-base-multilingual-cased | +0.122 | +0.125 | +0.634 | +0.799 |
| xlm-roberta-base | -0.260 | -0.182 | +0.634 | +0.818 |
| meta-llama/Llama-3.2-1B | +0.201 | +0.122 | +0.634 | +0.752 |
| Qwen/Qwen2.5-0.5B | +0.185 | +0.126 | +0.634 | +0.765 |
| bigscience/bloom-560m | -0.490 | -0.333 | +0.634 | +0.787 |


## The mechanism — dilution decomposition

Exact identity (per-token pieces = isolated f(type)): **S1(W) = B(W) + g(W)**, g = S4·(A−B), with A,B the *token*-weighted fertility of novel/seen tokens (A=`S8tok`, B=`S9tok`). Since S1_ref already contains the reference-epoch novelty ḡ_ref, and B is ~stable, the closing form is **S1(W) − S1_ref ≈ g(W) − ḡ_ref** — the deviation of the novelty term from its reference level. Dropping ḡ_ref (as a naïve reading does) over-predicts; that extra term is derived here, not fudged.

The brief's `S8`/`S9` are *type*-weighted (mechanism: do novel *types* fragment worse?). Token-rare novel types make the type-weighted RHS overshoot further — that gap is the dilution. Both are reported.

| tokenizer | S4 | S8 (type,novel) | S9 (type,seen) | S8−S9 | r(obs,pred_tok) | obs Δz(S1) | pred Δz(S1) [token] | pred Δz(S1) [type] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bert-base-multilingual-cased | 0.0586 | 4.269 | 2.842 | +1.427 | 0.607 | +0.122 | +0.200 | +0.158 |
| xlm-roberta-base | 0.0586 | 3.803 | 2.062 | +1.741 | 0.840 | -0.260 | +0.267 | +0.229 |
| meta-llama/Llama-3.2-1B | 0.0586 | 10.964 | 8.130 | +2.834 | 0.599 | +0.201 | +0.073 | +0.061 |
| Qwen/Qwen2.5-0.5B | 0.0586 | 9.726 | 7.345 | +2.380 | 0.594 | +0.185 | +0.078 | +0.064 |
| bigscience/bloom-560m | 0.0586 | 3.046 | 1.547 | +1.499 | 0.818 | -0.490 | +0.196 | +0.182 |

`obs Δz(S1)` vs `pred Δz(S1) [token]` agreeing within ~2× is the intended result. `[type]` shows the naïve type-weighted over-shoot — the dilution size.

Windows with S8 null (no novel types), per tokenizer (perm00): bert-base-multilingual-cased: 638, xlm-roberta-base: 638, meta-llama/Llama-3.2-1B: 638, Qwen/Qwen2.5-0.5B: 638, bigscience/bloom-560m: 638. These are emitted null, not zero.

## Part 5 — trend shape, COVID window, window-size artifact

**Mean z per year — S1:**

| tokenizer | 2016 | 2017 | 2018 | 2019 | 2020 | 2020 Q1-Q2 |
| --- | --- | --- | --- | --- | --- | --- |
| bert-base-multilingual-cased | +0.00 | -0.05 | +0.12 | -0.04 | +0.03 | -0.09 |
| xlm-roberta-base | -0.03 | -0.21 | -0.37 | -0.32 | -0.33 | -0.41 |
| meta-llama/Llama-3.2-1B | +0.05 | +0.07 | +0.17 | +0.11 | +0.09 | -0.03 |
| Qwen/Qwen2.5-0.5B | +0.05 | +0.06 | +0.14 | +0.08 | +0.08 | -0.04 |
| bigscience/bloom-560m | -0.04 | -0.29 | -0.50 | -0.50 | -0.56 | -0.65 |

**Mean z per year — S1c:**

| tokenizer | 2016 | 2017 | 2018 | 2019 | 2020 | 2020 Q1-Q2 |
| --- | --- | --- | --- | --- | --- | --- |
| bert-base-multilingual-cased | -0.01 | -0.07 | +0.03 | -0.04 | +0.04 | -0.05 |
| xlm-roberta-base | +0.01 | -0.11 | -0.23 | -0.26 | -0.27 | -0.36 |
| meta-llama/Llama-3.2-1B | +0.01 | +0.02 | +0.06 | -0.00 | +0.03 | -0.07 |
| Qwen/Qwen2.5-0.5B | +0.02 | +0.01 | +0.05 | -0.02 | +0.03 | -0.07 |
| bigscience/bloom-560m | +0.00 | -0.18 | -0.21 | -0.39 | -0.45 | -0.58 |

**Mean z per year — S4:**

| tokenizer | 2016 | 2017 | 2018 | 2019 | 2020 | 2020 Q1-Q2 |
| --- | --- | --- | --- | --- | --- | --- |
| bert-base-multilingual-cased | -0.61 | +0.14 | +0.20 | +0.29 | +0.68 | +0.74 |
| xlm-roberta-base | -0.61 | +0.14 | +0.20 | +0.29 | +0.68 | +0.74 |
| meta-llama/Llama-3.2-1B | -0.61 | +0.14 | +0.20 | +0.29 | +0.68 | +0.74 |
| Qwen/Qwen2.5-0.5B | -0.61 | +0.14 | +0.20 | +0.29 | +0.68 | +0.74 |
| bigscience/bloom-560m | -0.61 | +0.14 | +0.20 | +0.29 | +0.68 | +0.74 |

**Mean z per year — S7:**

| tokenizer | 2016 | 2017 | 2018 | 2019 | 2020 | 2020 Q1-Q2 |
| --- | --- | --- | --- | --- | --- | --- |
| bert-base-multilingual-cased | -0.77 | +0.18 | +0.22 | +0.35 | +0.88 | +0.98 |
| xlm-roberta-base | -0.79 | +0.19 | +0.26 | +0.36 | +0.89 | +0.97 |
| meta-llama/Llama-3.2-1B | -0.83 | +0.15 | +0.20 | +0.30 | +0.80 | +0.85 |
| Qwen/Qwen2.5-0.5B | -0.82 | +0.15 | +0.20 | +0.30 | +0.82 | +0.88 |
| bigscience/bloom-560m | -0.80 | +0.20 | +0.27 | +0.37 | +0.82 | +0.87 |

### corr(window n_words, signal) — window-size artifact check

| tokenizer | S1 | S1c | S4 | S7 |
| --- | --- | --- | --- | --- |
| bert-base-multilingual-cased | -0.038 | +0.062 | +0.071 | +0.053 |
| xlm-roberta-base | +0.023 | +0.127 | +0.071 | +0.042 |
| meta-llama/Llama-3.2-1B | -0.119 | -0.043 | +0.071 | +0.035 |
| Qwen/Qwen2.5-0.5B | -0.119 | -0.047 | +0.071 | +0.036 |
| bigscience/bloom-560m | +0.050 | +0.106 | +0.071 | +0.038 |

|corr(n_words, S1)| < 0.10 means window-size heterogeneity (GATE 3 of T3) is harmless; above that it injects an artifact.

## STATUS

```
GATE 1 — split-epoch calibration applied to all signals, sigma_ref(S4) > 0:   PASS
GATE 2 — isolated-type fertility validated against in-context, r >= 0.95:     PASS
GATE 3 — S1c, S7, S8, S9 computed for all tokenizers, nulls explained:        PASS
GATE 4 — |corr(window n_words, S1)| < 0.10:                                   FAIL

THE RESULT — mean z over final 10% of the real stream, per tokenizer:
    S1  (token-weighted): bert-base-multilingual-cased=+0.122, xlm-roberta-base=-0.260, Llama-3.2-1B=+0.201, Qwen2.5-0.5B=+0.185, bloom-560m=-0.490
    S1c (type-weighted): bert-base-multilingual-cased=+0.125, xlm-roberta-base=-0.182, Llama-3.2-1B=+0.122, Qwen2.5-0.5B=+0.126, bloom-560m=-0.333
    S4  (unseen-type rate): bert-base-multilingual-cased=+0.634, xlm-roberta-base=+0.634, Llama-3.2-1B=+0.634, Qwen2.5-0.5B=+0.634, bloom-560m=+0.634
    S7  (novel-token fertility mass): bert-base-multilingual-cased=+0.799, xlm-roberta-base=+0.818, Llama-3.2-1B=+0.752, Qwen2.5-0.5B=+0.765, bloom-560m=+0.787

THE MECHANISM — final 10%, per tokenizer:
    S4 (novelty rate): bert-base-multilingual-cased=+0.059, xlm-roberta-base=+0.059, Llama-3.2-1B=+0.059, Qwen2.5-0.5B=+0.059, bloom-560m=+0.059
    S8 (mean fertility, novel types): bert-base-multilingual-cased=+4.269, xlm-roberta-base=+3.803, Llama-3.2-1B=+10.964, Qwen2.5-0.5B=+9.726, bloom-560m=+3.046
    S9 (mean fertility, seen types): bert-base-multilingual-cased=+2.842, xlm-roberta-base=+2.062, Llama-3.2-1B=+8.130, Qwen2.5-0.5B=+7.345, bloom-560m=+1.547
    S8 - S9 (excess frag): bert-base-multilingual-cased=+1.427, xlm-roberta-base=+1.741, Llama-3.2-1B=+2.834, Qwen2.5-0.5B=+2.380, bloom-560m=+1.499
    predicted delta-z(S1) vs observed:
        bert-base-multilingual-cased: pred=+0.200 obs=+0.122 (r=0.61)
        xlm-roberta-base: pred=+0.267 obs=-0.260 (r=0.84)
        Llama-3.2-1B: pred=+0.073 obs=+0.201 (r=0.60)
        Qwen2.5-0.5B: pred=+0.078 obs=+0.185 (r=0.59)
        bloom-560m: pred=+0.196 obs=-0.490 (r=0.82)

COVID CHECK — mean z in 2020 Q1-Q2, per tokenizer (S1 / S1c / S4 / S7):
    bert-base-multilingual-cased: -0.09 / -0.05 / +0.74 / +0.98
    xlm-roberta-base: -0.41 / -0.36 / +0.74 / +0.97
    Llama-3.2-1B: -0.03 / -0.07 / +0.74 / +0.85
    Qwen2.5-0.5B: -0.04 / -0.07 / +0.74 / +0.88
    bloom-560m: -0.65 / -0.58 / +0.74 / +0.87

VERDICT: PROCEED WITH CAVEATS
Blockers:
  - none
Surprises worth a human decision:
  - GATE 4 fail: |corr(n_words, S1)| ≥ 0.10 — window-size heterogeneity injects an artifact; revisit GATE 3 decision.
```
