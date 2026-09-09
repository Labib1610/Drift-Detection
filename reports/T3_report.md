# TASK 3 — Fertility signals (S1-S4) report

- Mode: FULL · wall-clock 379.8s
- Reproduce:
```
python src/streams.py --params params.yaml
python src/fertility.py --params params.yaml --report reports/T4_report.md
```

## Permutation validity

| stream | n | n_perms | checks |
| --- | --- | --- | --- |
| bn_panel | 99,900 | 11 | valid=True & bijections & no-dups=True |
| bn_full | 95,935 | 11 | valid=True & bijections & no-dups=True |

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

## Raw mean fertility (whole panel)

Descriptive tokens-per-word over all of `bn_panel` (spec §5.5). Amendment: Qwen ~8-10 is expected (byte-level BPE, negligible Bengali vocabulary).

| tokenizer | raw mean fertility (panel) |
| --- | --- |
| bert-base-multilingual-cased | 2.715 |
| xlm-roberta-base | 2.040 |
| meta-llama/Llama-3.2-1B | 7.710 |
| Qwen/Qwen2.5-0.5B | 6.989 |
| bigscience/bloom-560m | 1.678 |

### Fertility vs orthographic word length (amendment)

Per-document Pearson r between fertility and mean word length in characters. r above ~0.9 means the tokenizer is measuring **script encoding**, not vocabulary novelty (observation, not a gate).

| tokenizer | corr(fertility, mean word len) | flag |
| --- | --- | --- |
| bert-base-multilingual-cased | 0.658 | — |
| xlm-roberta-base | 0.293 | — |
| meta-llama/Llama-3.2-1B | 0.845 | — |
| Qwen/Qwen2.5-0.5B | 0.829 | — |
| bigscience/bloom-560m | 0.110 | — |

## Windows (real stream, perm 00)

Windows: **12,765**. Window `n_words`: min 2000, p50 2160, p99 3377, max 6207.

Deviation from spec §5.2 (verbatim for the paper): *windows are built by greedy packing of whole documents until cumulative ICU words ≥ 2,000, and fertility is the exact ratio Σtokens/Σwords over the window rather than a division by a nominal 2,000.*

## Calibration params and end-of-stream z (real stream)

μ_ref/σ_ref from the first 10% of windows; mean z over the **final 10%** is the crude uncalibrated answer to *'does fertility rise by 2020?'*

| tokenizer | signal | μ_ref | σ_ref | mean z (final 10%) |
| --- | --- | --- | --- | --- |
| bert-base-multilingual-cased | S1 | 2.7151 | 0.1046 | +0.122 |
| bert-base-multilingual-cased | S2 | 0.0540 | 0.0067 | -0.134 |
| bert-base-multilingual-cased | S3 | 0.5695 | 0.0249 | +0.305 |
| bert-base-multilingual-cased | S4 | 0.0479 | 0.0169 | +0.634 |
| bert-base-multilingual-cased | S1b | 2.7486 | 0.0699 | +0.014 |
| xlm-roberta-base | S1 | 2.0597 | 0.0806 | -0.260 |
| xlm-roberta-base | S2 | 0.0000 | 0.0000 | null |
| xlm-roberta-base | S3 | 0.5180 | 0.0197 | -0.253 |
| xlm-roberta-base | S4 | 0.0479 | 0.0169 | +0.634 |
| xlm-roberta-base | S1b | 2.0638 | 0.0595 | -0.108 |
| meta-llama/Llama-3.2-1B | S1 | 7.6799 | 0.3846 | +0.201 |
| meta-llama/Llama-3.2-1B | S2 | null | null | null |
| meta-llama/Llama-3.2-1B | S3 | 0.8727 | 0.0081 | +0.218 |
| meta-llama/Llama-3.2-1B | S4 | 0.0479 | 0.0169 | +0.634 |
| meta-llama/Llama-3.2-1B | S1b | 7.8039 | 0.1520 | +0.134 |
| Qwen/Qwen2.5-0.5B | S1 | 6.9689 | 0.3270 | +0.185 |
| Qwen/Qwen2.5-0.5B | S2 | null | null | null |
| Qwen/Qwen2.5-0.5B | S3 | 0.8597 | 0.0083 | +0.212 |
| Qwen/Qwen2.5-0.5B | S4 | 0.0479 | 0.0169 | +0.634 |
| Qwen/Qwen2.5-0.5B | S1b | 7.0738 | 0.1281 | +0.120 |
| bigscience/bloom-560m | S1 | 1.7074 | 0.0806 | -0.490 |
| bigscience/bloom-560m | S2 | null | null | null |
| bigscience/bloom-560m | S3 | 0.4291 | 0.0278 | -0.402 |
| bigscience/bloom-560m | S4 | 0.0479 | 0.0169 | +0.634 |
| bigscience/bloom-560m | S1b | 1.7101 | 0.0497 | -0.287 |

## Null / undefined (tokenizer, signal) pairs

| stream | tokenizer | signal |
| --- | --- | --- |
| bn_panel | meta-llama/Llama-3.2-1B | S2 byte-fallback (byte-level BPE) |
| bn_panel | Qwen/Qwen2.5-0.5B | S2 byte-fallback (byte-level BPE) |
| bn_panel | bigscience/bloom-560m | S2 byte-fallback (byte-level BPE) |
| bn_panel | xlm-roberta-base | S2 (z all null) |
| bn_panel | meta-llama/Llama-3.2-1B | S2 (z all null) |
| bn_panel | Qwen/Qwen2.5-0.5B | S2 (z all null) |
| bn_panel | bigscience/bloom-560m | S2 (z all null) |
| bn_full | meta-llama/Llama-3.2-1B | S2 byte-fallback (byte-level BPE) |
| bn_full | Qwen/Qwen2.5-0.5B | S2 byte-fallback (byte-level BPE) |
| bn_full | bigscience/bloom-560m | S2 byte-fallback (byte-level BPE) |
| bn_full | xlm-roberta-base | S2 (z all null) |
| bn_full | meta-llama/Llama-3.2-1B | S2 (z all null) |
| bn_full | Qwen/Qwen2.5-0.5B | S2 (z all null) |
| bn_full | bigscience/bloom-560m | S2 (z all null) |

All nulls above are intentional: byte-level BPE has no byte-fallback concept, so S2 is emitted as null rather than a fabricated zero.

## S1 vs S1b (topic-adjusted)

| tokenizer | mean z S1 (final 10%) | mean z S1b (final 10%) | topic-absent fallbacks |
| --- | --- | --- | --- |
| bert-base-multilingual-cased | +0.122 | +0.014 | 49697 |
| xlm-roberta-base | -0.260 | -0.108 | 49697 |
| meta-llama/Llama-3.2-1B | +0.201 | +0.134 | 49697 |
| Qwen/Qwen2.5-0.5B | +0.185 | +0.120 | 49697 |
| bigscience/bloom-560m | -0.490 | -0.287 | 49697 |

If S1 rises but S1b does not, late-period drift is compositional (topic mix); if both rise, it is lexical.

## Signal correlation matrix (raw signals, real stream)

> T4 split-epoch calibration: vocabulary V from windows [0, 5%), μ_ref/σ_ref from [5%, 10%). z(S4) is now well-defined (σ_ref>0). Correlations use raw signals (scale-invariant → equal to z-correlations).

**bert-base-multilingual-cased:**

|  | S1 | S2 | S3 | S4 |
| --- | --- | --- | --- | --- |
| S1 | 1.000 | -0.152 | 0.775 | -0.173 |
| S2 | -0.152 | 1.000 | 0.049 | -0.222 |
| S3 | 0.775 | 0.049 | 1.000 | -0.421 |
| S4 | -0.173 | -0.222 | -0.421 | 1.000 |

**xlm-roberta-base:**

|  | S1 | S2 | S3 | S4 |
| --- | --- | --- | --- | --- |
| S1 | 1.000 | null | 0.744 | 0.147 |
| S2 | null | null | null | null |
| S3 | 0.744 | null | 1.000 | -0.010 |
| S4 | 0.147 | null | -0.010 | 1.000 |

**meta-llama/Llama-3.2-1B:**

|  | S1 | S2 | S3 | S4 |
| --- | --- | --- | --- | --- |
| S1 | 1.000 | null | 0.905 | -0.477 |
| S2 | null | null | null | null |
| S3 | 0.905 | null | 1.000 | -0.514 |
| S4 | -0.477 | null | -0.514 | 1.000 |

**Qwen/Qwen2.5-0.5B:**

|  | S1 | S2 | S3 | S4 |
| --- | --- | --- | --- | --- |
| S1 | 1.000 | null | 0.914 | -0.471 |
| S2 | null | null | null | null |
| S3 | 0.914 | null | 1.000 | -0.517 |
| S4 | -0.471 | null | -0.517 | 1.000 |

**bigscience/bloom-560m:**

|  | S1 | S2 | S3 | S4 |
| --- | --- | --- | --- | --- |
| S1 | 1.000 | null | 0.738 | 0.200 |
| S2 | null | null | null | null |
| S3 | 0.738 | null | 1.000 | 0.153 |
| S4 | 0.200 | null | 0.153 | 1.000 |

## Wall-clock timing

| stage | seconds |
| --- | --- |
| load bert-base-multilingual-cased | 2.7 |
| load xlm-roberta-base | 3.3 |
| load meta-llama_Llama-3.2-1B | 1.9 |
| load Qwen_Qwen2.5-0.5B | 1.6 |
| load bigscience_bloom-560m | 2.5 |
| bn_panel: ICU types | 10.1 |
| bn_panel: tokenize bert-base-multilingual-cased | 19.2 |
| bn_panel: tokenize xlm-roberta-base | 17.7 |
| bn_panel: tokenize meta-llama_Llama-3.2-1B | 36.4 |
| bn_panel: tokenize Qwen_Qwen2.5-0.5B | 37.9 |
| bn_panel: tokenize bigscience_bloom-560m | 18.1 |
| bn_panel: type-fertility | 24.2 |
| bn_panel: total | 215.7 |
| bn_full: ICU types | 8.4 |
| bn_full: tokenize bert-base-multilingual-cased | 18.5 |
| bn_full: tokenize xlm-roberta-base | 18.1 |
| bn_full: tokenize meta-llama_Llama-3.2-1B | 31.3 |
| bn_full: tokenize Qwen_Qwen2.5-0.5B | 31.7 |
| bn_full: tokenize bigscience_bloom-560m | 15.4 |
| bn_full: type-fertility | 22.5 |
| bn_full: total | 150.7 |

## STATUS

```
GATE 1 — per-doc counts exist for all 5 tokenizers, no unexpected nulls:  PASS
GATE 2 — 11 valid permutations, perm_00 = identity:                          PASS
GATE 3 — window n_words p99 < 3000 (windows are tight):                      FAIL   (p99=3377)
GATE 4 — raw mean fertility in [1.5, 12.0] for every tokenizer:              PASS   (amended range)
GATE 5 — z-scored calibration epoch has mean ~0, sd ~1 for every stream:     PASS
GATE 6 — S4 computed with a frozen calibration vocabulary, no leakage:       PASS

OBSERVATION — mean z of S1 in the final 10% of the real stream, per tokenizer:
    bert-base-multilingual-cased: +0.122
    xlm-roberta-base: -0.260
    meta-llama/Llama-3.2-1B: +0.201
    Qwen/Qwen2.5-0.5B: +0.185
    bigscience/bloom-560m: -0.490
OBSERVATION — same for S1b (topic-adjusted):
    bert-base-multilingual-cased: +0.014
    xlm-roberta-base: -0.108
    meta-llama/Llama-3.2-1B: +0.134
    Qwen/Qwen2.5-0.5B: +0.120
    bigscience/bloom-560m: -0.287
OBSERVATION — corr(S1, S4) on the real stream:
    bert-base-multilingual-cased: -0.173
    xlm-roberta-base: 0.147
    meta-llama/Llama-3.2-1B: -0.477
    Qwen/Qwen2.5-0.5B: -0.471
    bigscience/bloom-560m: 0.200

VERDICT: PROCEED WITH CAVEATS
Blockers:
  - none
Surprises worth a human decision:
  - GATE 3 fail: window n_words p99=3377 ≥ 3000. A few very long articles create oversized windows (median is tight at ~2160). Non-fatal — signals are valid. Decide: accept, raise the gate, or drop docs above ~1500 words.
```
