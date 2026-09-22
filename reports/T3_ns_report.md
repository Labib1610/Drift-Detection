# TASK 3 — Fertility signals (S1-S4) report

- Mode: FULL · wall-clock 516.2s
- Language: `ns` · headline stream: `ns_panel`
- Reproduce:
```
python src/streams.py --params params.yaml
python src/fertility.py --lang ns --params params.yaml --report reports/T4_report.md
```

## Permutation validity

| stream | n | n_perms | checks |
| --- | --- | --- | --- |
| bn_full | 95,935 | 11 | valid=True & bijections & no-dups=True |
| bn_panel | 99,900 | 11 | valid=True & bijections & no-dups=True |
| ns_full | 83,893 | 11 | valid=True & bijections & no-dups=True |
| ns_panel | 67,670 | 11 | valid=True & bijections & no-dups=True |

## Tokenizers

| requested | actually loaded | family | vocab | fast | substitution |
| --- | --- | --- | --- | --- | --- |
| bert-base-multilingual-cased | (as requested) | wordpiece | 119,547 | True | — |
| xlm-roberta-base | (as requested) | sentencepiece | 250,002 | True | — |
| meta-llama/Llama-3.2-1B | (as requested) | byte_bpe | 128,256 | True | — |
| Qwen/Qwen2.5-0.5B | (as requested) | byte_bpe | 151,665 | True | — |
| bigscience/bloom-560m | (as requested) | byte_bpe | 250,680 | True | — |

Detected word-boundary / fallback markers, confirmed on the Bangla two-word probe `নতুন শব্দ`:

| tokenizer | continuation marker | byte-fallback handling | probe pieces |
| --- | --- | --- | --- |
| bert-base-multilingual-cased | ## (continuation prefix) | UNK-rate ([UNK] tokens) | নতুন শ ##ব ##্দ |
| xlm-roberta-base | U+2581 '▁' (word-start prefix) | byte-fallback (<0xHH> tokens) | ▁নতুন ▁শব্দ |
| meta-llama/Llama-3.2-1B | U+0120 'Ġ' (word-start prefix) | byte-fallback UNDEFINED (byte-level → null) | à¦¨ à¦ ¤ à§ ģ à¦¨ Ġà¦ ¶ à¦ ¬ à§įà¦ ¦ |
| Qwen/Qwen2.5-0.5B | U+0120 'Ġ' (word-start prefix) | byte-fallback UNDEFINED (byte-level → null) | à¦¨ à¦¤ à§ ģ à¦¨ Ġà¦ ¶ à¦¬ à§įà¦ ¦ |
| bigscience/bloom-560m | U+0120 'Ġ' (word-start prefix) | byte-fallback UNDEFINED (byte-level → null) | à¦¨à¦¤à§ģà¦¨ Ġà¦¶à¦¬à§įà¦¦ |

## Raw mean fertility (whole headline stream `ns_panel`)

Descriptive tokens-per-word over all of `ns_panel` (spec §5.5). Amendment: Qwen ~8-10 is expected (byte-level BPE, negligible Bengali vocabulary).

| tokenizer | raw mean fertility (headline stream) |
| --- | --- |
| bert-base-multilingual-cased | 1.394 |
| xlm-roberta-base | 1.430 |
| meta-llama/Llama-3.2-1B | 1.303 |
| Qwen/Qwen2.5-0.5B | 1.335 |
| bigscience/bloom-560m | 1.300 |

### Fertility vs orthographic word length (amendment)

Per-document Pearson r between fertility and mean word length in characters. r above ~0.9 means the tokenizer is measuring **script encoding**, not vocabulary novelty (observation, not a gate).

| tokenizer | corr(fertility, mean word len) | flag |
| --- | --- | --- |
| bert-base-multilingual-cased | 0.299 | — |
| xlm-roberta-base | 0.453 | — |
| meta-llama/Llama-3.2-1B | 0.124 | — |
| Qwen/Qwen2.5-0.5B | 0.082 | — |
| bigscience/bloom-560m | 0.228 | — |

## Windows (real stream, perm 00)

Windows: **9,844**. Window `n_words`: min 2000, p50 2196, p99 3522, max 6543.

Deviation from spec §5.2 (verbatim for the paper): *windows are built by greedy packing of whole documents until cumulative ICU words ≥ 2,000, and fertility is the exact ratio Σtokens/Σwords over the window rather than a division by a nominal 2,000.*

## Calibration params and end-of-stream z (real stream)

μ_ref/σ_ref from the first 10% of windows; mean z over the **final 10%** is the crude uncalibrated answer to *'does fertility rise by 2020?'*

| tokenizer | signal | μ_ref | σ_ref | mean z (final 10%) |
| --- | --- | --- | --- | --- |
| bert-base-multilingual-cased | S1 | 1.3890 | 0.0520 | +0.678 |
| bert-base-multilingual-cased | S2 | 0.0113 | 0.0072 | +0.539 |
| bert-base-multilingual-cased | S3 | 0.1566 | 0.0204 | +0.338 |
| bert-base-multilingual-cased | S4 | 0.0372 | 0.0138 | +0.653 |
| bert-base-multilingual-cased | S1b | 1.3950 | 0.0200 | +0.629 |
| xlm-roberta-base | S1 | 1.4222 | 0.0463 | +0.961 |
| xlm-roberta-base | S2 | 0.0000 | 0.0000 | null |
| xlm-roberta-base | S3 | 0.3045 | 0.0240 | +0.903 |
| xlm-roberta-base | S4 | 0.0372 | 0.0138 | +0.653 |
| xlm-roberta-base | S1b | 1.4296 | 0.0179 | +0.907 |
| meta-llama/Llama-3.2-1B | S1 | 1.3003 | 0.0500 | +0.427 |
| meta-llama/Llama-3.2-1B | S2 | null | null | null |
| meta-llama/Llama-3.2-1B | S3 | 0.2535 | 0.0310 | +0.592 |
| meta-llama/Llama-3.2-1B | S4 | 0.0372 | 0.0138 | +0.653 |
| meta-llama/Llama-3.2-1B | S1b | 1.3094 | 0.0199 | +0.412 |
| Qwen/Qwen2.5-0.5B | S1 | 1.3294 | 0.0595 | +0.418 |
| Qwen/Qwen2.5-0.5B | S2 | null | null | null |
| Qwen/Qwen2.5-0.5B | S3 | 0.2694 | 0.0343 | +0.571 |
| Qwen/Qwen2.5-0.5B | S4 | 0.0372 | 0.0138 | +0.653 |
| Qwen/Qwen2.5-0.5B | S1b | 1.3399 | 0.0233 | +0.424 |
| bigscience/bloom-560m | S1 | 1.2976 | 0.0423 | +0.613 |
| bigscience/bloom-560m | S2 | null | null | null |
| bigscience/bloom-560m | S3 | 0.2532 | 0.0273 | +0.754 |
| bigscience/bloom-560m | S4 | 0.0372 | 0.0138 | +0.653 |
| bigscience/bloom-560m | S1b | 1.3061 | 0.0176 | +0.493 |

## Null / undefined (tokenizer, signal) pairs

| stream | tokenizer | signal |
| --- | --- | --- |
| ns_panel | meta-llama/Llama-3.2-1B | S2 byte-fallback (byte-level BPE) |
| ns_panel | Qwen/Qwen2.5-0.5B | S2 byte-fallback (byte-level BPE) |
| ns_panel | bigscience/bloom-560m | S2 byte-fallback (byte-level BPE) |
| ns_panel | xlm-roberta-base | S2 (z all null) |
| ns_panel | meta-llama/Llama-3.2-1B | S2 (z all null) |
| ns_panel | Qwen/Qwen2.5-0.5B | S2 (z all null) |
| ns_panel | bigscience/bloom-560m | S2 (z all null) |
| ns_full | meta-llama/Llama-3.2-1B | S2 byte-fallback (byte-level BPE) |
| ns_full | Qwen/Qwen2.5-0.5B | S2 byte-fallback (byte-level BPE) |
| ns_full | bigscience/bloom-560m | S2 byte-fallback (byte-level BPE) |
| ns_full | xlm-roberta-base | S2 (z all null) |
| ns_full | meta-llama/Llama-3.2-1B | S2 (z all null) |
| ns_full | Qwen/Qwen2.5-0.5B | S2 (z all null) |
| ns_full | bigscience/bloom-560m | S2 (z all null) |

All nulls above are intentional: byte-level BPE has no byte-fallback concept, so S2 is emitted as null rather than a fabricated zero.

## S1 vs S1b (topic-adjusted)

| tokenizer | mean z S1 (final 10%) | mean z S1b (final 10%) | topic-absent fallbacks |
| --- | --- | --- | --- |
| bert-base-multilingual-cased | +0.678 | +0.629 | 2007534 |
| xlm-roberta-base | +0.961 | +0.907 | 2007534 |
| meta-llama/Llama-3.2-1B | +0.427 | +0.412 | 2007534 |
| Qwen/Qwen2.5-0.5B | +0.418 | +0.424 | 2007534 |
| bigscience/bloom-560m | +0.613 | +0.493 | 2007534 |

If S1 rises but S1b does not, late-period drift is compositional (topic mix); if both rise, it is lexical.

## Signal correlation matrix (raw signals, real stream)

> T4 split-epoch calibration: vocabulary V from windows [0, 5%), μ_ref/σ_ref from [5%, 10%). z(S4) is now well-defined (σ_ref>0). Correlations use raw signals (scale-invariant → equal to z-correlations).

**bert-base-multilingual-cased:**

|  | S1 | S2 | S3 | S4 |
| --- | --- | --- | --- | --- |
| S1 | 1.000 | 0.028 | 0.780 | 0.427 |
| S2 | 0.028 | 1.000 | -0.109 | -0.073 |
| S3 | 0.780 | -0.109 | 1.000 | 0.347 |
| S4 | 0.427 | -0.073 | 0.347 | 1.000 |

**xlm-roberta-base:**

|  | S1 | S2 | S3 | S4 |
| --- | --- | --- | --- | --- |
| S1 | 1.000 | null | 0.978 | 0.419 |
| S2 | null | null | null | null |
| S3 | 0.978 | null | 1.000 | 0.404 |
| S4 | 0.419 | null | 0.404 | 1.000 |

**meta-llama/Llama-3.2-1B:**

|  | S1 | S2 | S3 | S4 |
| --- | --- | --- | --- | --- |
| S1 | 1.000 | null | 0.956 | 0.407 |
| S2 | null | null | null | null |
| S3 | 0.956 | null | 1.000 | 0.374 |
| S4 | 0.407 | null | 0.374 | 1.000 |

**Qwen/Qwen2.5-0.5B:**

|  | S1 | S2 | S3 | S4 |
| --- | --- | --- | --- | --- |
| S1 | 1.000 | null | 0.964 | 0.384 |
| S2 | null | null | null | null |
| S3 | 0.964 | null | 1.000 | 0.358 |
| S4 | 0.384 | null | 0.358 | 1.000 |

**bigscience/bloom-560m:**

|  | S1 | S2 | S3 | S4 |
| --- | --- | --- | --- | --- |
| S1 | 1.000 | null | 0.953 | 0.432 |
| S2 | null | null | null | null |
| S3 | 0.953 | null | 1.000 | 0.392 |
| S4 | 0.432 | null | 0.392 | 1.000 |

## Wall-clock timing

| stage | seconds |
| --- | --- |
| load bert-base-multilingual-cased | 7.3 |
| load xlm-roberta-base | 9.8 |
| load meta-llama_Llama-3.2-1B | 2.5 |
| load Qwen_Qwen2.5-0.5B | 8.4 |
| load bigscience_bloom-560m | 9.2 |
| ns_panel: ICU types | 8.3 |
| ns_panel: tokenize bert-base-multilingual-cased | 22.8 |
| ns_panel: tokenize xlm-roberta-base | 25.9 |
| ns_panel: tokenize meta-llama_Llama-3.2-1B | 9.7 |
| ns_panel: tokenize Qwen_Qwen2.5-0.5B | 10.8 |
| ns_panel: tokenize bigscience_bloom-560m | 10.4 |
| ns_panel: type-fertility | 13.0 |
| ns_panel: total | 276.6 |
| ns_full: ICU types | 11.0 |
| ns_full: tokenize bert-base-multilingual-cased | 29.5 |
| ns_full: tokenize xlm-roberta-base | 28.1 |
| ns_full: tokenize meta-llama_Llama-3.2-1B | 27.6 |
| ns_full: tokenize Qwen_Qwen2.5-0.5B | 29.5 |
| ns_full: tokenize bigscience_bloom-560m | 27.8 |
| ns_full: type-fertility | 14.7 |
| ns_full: total | 199.5 |

## STATUS

```
GATE 1 — per-doc counts exist for all 5 tokenizers, no unexpected nulls:  PASS
GATE 2 — 11 valid permutations, perm_00 = identity:                          PASS
GATE 3 — window n_words p99 < 3000 (windows are tight):                      FAIL   (p99=3522)
GATE 4 — raw mean fertility in [1.0, 12.0] for every tokenizer:  PASS   (per-language range)
GATE 5 — z-scored calibration epoch has mean ~0, sd ~1 for every stream:     PASS
GATE 6 — S4 computed with a frozen calibration vocabulary, no leakage:       PASS

OBSERVATION — mean z of S1 in the final 10% of the real stream, per tokenizer:
    bert-base-multilingual-cased: +0.678
    xlm-roberta-base: +0.961
    meta-llama/Llama-3.2-1B: +0.427
    Qwen/Qwen2.5-0.5B: +0.418
    bigscience/bloom-560m: +0.613
OBSERVATION — same for S1b (topic-adjusted):
    bert-base-multilingual-cased: +0.629
    xlm-roberta-base: +0.907
    meta-llama/Llama-3.2-1B: +0.412
    Qwen/Qwen2.5-0.5B: +0.424
    bigscience/bloom-560m: +0.493
OBSERVATION — corr(S1, S4) on the real stream:
    bert-base-multilingual-cased: 0.427
    xlm-roberta-base: 0.419
    meta-llama/Llama-3.2-1B: 0.407
    Qwen/Qwen2.5-0.5B: 0.384
    bigscience/bloom-560m: 0.432

VERDICT: PROCEED WITH CAVEATS
Blockers:
  - none
Surprises worth a human decision:
  - GATE 3 fail: window n_words p99=3522 ≥ 3000. A few very long articles create oversized windows (median is tight at ~2160). Non-fatal — signals are valid. Decide: accept, raise the gate, or drop docs above ~1500 words.
```
