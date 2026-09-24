# TASK 3 — Fertility signals (S1-S4) report

- Mode: FULL · wall-clock 426.0s
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
| ns_panel | 46,513 | 11 | valid=True & bijections & no-dups=True |

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
| bert-base-multilingual-cased | 1.397 |
| xlm-roberta-base | 1.432 |
| meta-llama/Llama-3.2-1B | 1.304 |
| Qwen/Qwen2.5-0.5B | 1.338 |
| bigscience/bloom-560m | 1.300 |

### Fertility vs orthographic word length (amendment)

Per-document Pearson r between fertility and mean word length in characters. r above ~0.9 means the tokenizer is measuring **script encoding**, not vocabulary novelty (observation, not a gate).

| tokenizer | corr(fertility, mean word len) | flag |
| --- | --- | --- |
| bert-base-multilingual-cased | 0.255 | — |
| xlm-roberta-base | 0.404 | — |
| meta-llama/Llama-3.2-1B | 0.071 | — |
| Qwen/Qwen2.5-0.5B | 0.035 | — |
| bigscience/bloom-560m | 0.177 | — |

## Windows (real stream, perm 00)

Windows: **6,824**. Window `n_words`: min 2000, p50 2202, p99 3548, max 6652.

Deviation from spec §5.2 (verbatim for the paper): *windows are built by greedy packing of whole documents until cumulative ICU words ≥ 2,000, and fertility is the exact ratio Σtokens/Σwords over the window rather than a division by a nominal 2,000.*

## Calibration params and end-of-stream z (real stream)

μ_ref/σ_ref from the first 10% of windows; mean z over the **final 10%** is the crude uncalibrated answer to *'does fertility rise by 2020?'*

| tokenizer | signal | μ_ref | σ_ref | mean z (final 10%) |
| --- | --- | --- | --- | --- |
| bert-base-multilingual-cased | S1 | 1.3781 | 0.0392 | +1.082 |
| bert-base-multilingual-cased | S2 | 0.0191 | 0.0059 | -1.034 |
| bert-base-multilingual-cased | S3 | 0.1480 | 0.0184 | +0.857 |
| bert-base-multilingual-cased | S4 | 0.0350 | 0.0127 | +1.476 |
| bert-base-multilingual-cased | S1b | 1.3921 | 0.0194 | +0.577 |
| xlm-roberta-base | S1 | 1.4166 | 0.0365 | +1.091 |
| xlm-roberta-base | S2 | 0.0000 | 0.0000 | null |
| xlm-roberta-base | S3 | 0.2998 | 0.0191 | +1.175 |
| xlm-roberta-base | S4 | 0.0350 | 0.0127 | +1.476 |
| xlm-roberta-base | S1b | 1.4316 | 0.0181 | +0.557 |
| meta-llama/Llama-3.2-1B | S1 | 1.2922 | 0.0399 | +0.756 |
| meta-llama/Llama-3.2-1B | S2 | null | null | null |
| meta-llama/Llama-3.2-1B | S3 | 0.2526 | 0.0260 | +0.735 |
| meta-llama/Llama-3.2-1B | S4 | 0.0350 | 0.0127 | +1.476 |
| meta-llama/Llama-3.2-1B | S1b | 1.3118 | 0.0191 | +0.398 |
| Qwen/Qwen2.5-0.5B | S1 | 1.3219 | 0.0460 | +0.780 |
| Qwen/Qwen2.5-0.5B | S2 | null | null | null |
| Qwen/Qwen2.5-0.5B | S3 | 0.2691 | 0.0281 | +0.742 |
| Qwen/Qwen2.5-0.5B | S4 | 0.0350 | 0.0127 | +1.476 |
| Qwen/Qwen2.5-0.5B | S1b | 1.3435 | 0.0209 | +0.437 |
| bigscience/bloom-560m | S1 | 1.2903 | 0.0376 | +0.784 |
| bigscience/bloom-560m | S2 | null | null | null |
| bigscience/bloom-560m | S3 | 0.2519 | 0.0249 | +0.780 |
| bigscience/bloom-560m | S4 | 0.0350 | 0.0127 | +1.476 |
| bigscience/bloom-560m | S1b | 1.3067 | 0.0187 | +0.379 |

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
| bert-base-multilingual-cased | +1.082 | +0.577 | 1126553 |
| xlm-roberta-base | +1.091 | +0.557 | 1126553 |
| meta-llama/Llama-3.2-1B | +0.756 | +0.398 | 1126553 |
| Qwen/Qwen2.5-0.5B | +0.780 | +0.437 | 1126553 |
| bigscience/bloom-560m | +0.784 | +0.379 | 1126553 |

If S1 rises but S1b does not, late-period drift is compositional (topic mix); if both rise, it is lexical.

## Signal correlation matrix (raw signals, real stream)

> T4 split-epoch calibration: vocabulary V from windows [0, 5%), μ_ref/σ_ref from [5%, 10%). z(S4) is now well-defined (σ_ref>0). Correlations use raw signals (scale-invariant → equal to z-correlations).

**bert-base-multilingual-cased:**

|  | S1 | S2 | S3 | S4 |
| --- | --- | --- | --- | --- |
| S1 | 1.000 | -0.147 | 0.797 | 0.485 |
| S2 | -0.147 | 1.000 | -0.267 | -0.226 |
| S3 | 0.797 | -0.267 | 1.000 | 0.429 |
| S4 | 0.485 | -0.226 | 0.429 | 1.000 |

**xlm-roberta-base:**

|  | S1 | S2 | S3 | S4 |
| --- | --- | --- | --- | --- |
| S1 | 1.000 | null | 0.973 | 0.479 |
| S2 | null | null | null | null |
| S3 | 0.973 | null | 1.000 | 0.470 |
| S4 | 0.479 | null | 0.470 | 1.000 |

**meta-llama/Llama-3.2-1B:**

|  | S1 | S2 | S3 | S4 |
| --- | --- | --- | --- | --- |
| S1 | 1.000 | null | 0.953 | 0.428 |
| S2 | null | null | null | null |
| S3 | 0.953 | null | 1.000 | 0.398 |
| S4 | 0.428 | null | 0.398 | 1.000 |

**Qwen/Qwen2.5-0.5B:**

|  | S1 | S2 | S3 | S4 |
| --- | --- | --- | --- | --- |
| S1 | 1.000 | null | 0.962 | 0.403 |
| S2 | null | null | null | null |
| S3 | 0.962 | null | 1.000 | 0.381 |
| S4 | 0.403 | null | 0.381 | 1.000 |

**bigscience/bloom-560m:**

|  | S1 | S2 | S3 | S4 |
| --- | --- | --- | --- | --- |
| S1 | 1.000 | null | 0.950 | 0.476 |
| S2 | null | null | null | null |
| S3 | 0.950 | null | 1.000 | 0.435 |
| S4 | 0.476 | null | 0.435 | 1.000 |

## Wall-clock timing

| stage | seconds |
| --- | --- |
| load bert-base-multilingual-cased | 4.2 |
| load xlm-roberta-base | 3.6 |
| load meta-llama_Llama-3.2-1B | 2.3 |
| load Qwen_Qwen2.5-0.5B | 1.8 |
| load bigscience_bloom-560m | 2.7 |
| ns_panel: ICU types | 5.7 |
| ns_panel: tokenize bert-base-multilingual-cased | 13.9 |
| ns_panel: tokenize xlm-roberta-base | 17.2 |
| ns_panel: tokenize meta-llama_Llama-3.2-1B | 9.9 |
| ns_panel: tokenize Qwen_Qwen2.5-0.5B | 11.4 |
| ns_panel: tokenize bigscience_bloom-560m | 10.5 |
| ns_panel: type-fertility | 10.7 |
| ns_panel: total | 175.5 |
| ns_full: ICU types | 10.5 |
| ns_full: tokenize bert-base-multilingual-cased | 38.6 |
| ns_full: tokenize xlm-roberta-base | 38.5 |
| ns_full: tokenize meta-llama_Llama-3.2-1B | 30.4 |
| ns_full: tokenize Qwen_Qwen2.5-0.5B | 32.0 |
| ns_full: tokenize bigscience_bloom-560m | 31.5 |
| ns_full: type-fertility | 19.2 |
| ns_full: total | 233.6 |

## STATUS

```
GATE 1 — per-doc counts exist for all 5 tokenizers, no unexpected nulls:  PASS
GATE 2 — 11 valid permutations, perm_00 = identity:                          PASS
GATE 3 — window n_words p99 < 3000 (windows are tight):                      FAIL   (p99=3548)
GATE 4 — raw mean fertility in [1.0, 12.0] for every tokenizer:  PASS   (per-language range)
GATE 5 — z-scored calibration epoch has mean ~0, sd ~1 for every stream:     PASS
GATE 6 — S4 computed with a frozen calibration vocabulary, no leakage:       PASS

OBSERVATION — mean z of S1 in the final 10% of the real stream, per tokenizer:
    bert-base-multilingual-cased: +1.082
    xlm-roberta-base: +1.091
    meta-llama/Llama-3.2-1B: +0.756
    Qwen/Qwen2.5-0.5B: +0.780
    bigscience/bloom-560m: +0.784
OBSERVATION — same for S1b (topic-adjusted):
    bert-base-multilingual-cased: +0.577
    xlm-roberta-base: +0.557
    meta-llama/Llama-3.2-1B: +0.398
    Qwen/Qwen2.5-0.5B: +0.437
    bigscience/bloom-560m: +0.379
OBSERVATION — corr(S1, S4) on the real stream:
    bert-base-multilingual-cased: 0.485
    xlm-roberta-base: 0.479
    meta-llama/Llama-3.2-1B: 0.428
    Qwen/Qwen2.5-0.5B: 0.403
    bigscience/bloom-560m: 0.476

VERDICT: PROCEED WITH CAVEATS
Blockers:
  - none
Surprises worth a human decision:
  - GATE 3 fail: window n_words p99=3548 ≥ 3000. A few very long articles create oversized windows (median is tight at ~2160). Non-fatal — signals are valid. Decide: accept, raise the gate, or drop docs above ~1500 words.
```
