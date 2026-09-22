# TASK 7 — is anything actually detecting anything?

- Language `ns`. Uses only the frozen delta* in `results/calibration_ns.json` (+ classifier/embeddings calibration). No re-tuning, no new signals. Permutation seed = 42.

## Part 1 — alarm census (real stream, detection epoch)

| signal | tokenizer | alarms A | rate/1000 win | mean interval (days) | null FAR | rate÷FAR |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | bert-base-multilingual-cased | 28 | 3.16 | 112.4 | 9.8e-04 | 3.2 |
| S1 | xlm-roberta-base | 37 | 4.18 | 85.1 | 9.6e-04 | 4.4 |
| S1 | Llama-3.2-1B | 36 | 4.06 | 87.4 | 6.4e-04 | 6.3 |
| S1 | Qwen2.5-0.5B | 41 | 4.63 | 76.8 | 9.9e-04 | 4.7 |
| S1 | bloom-560m | 31 | 3.50 | 101.5 | 7.9e-04 | 4.4 |
| S1c | bert-base-multilingual-cased | 35 | 3.95 | 89.9 | 6.9e-04 | 5.8 |
| S1c | xlm-roberta-base | 37 | 4.18 | 85.1 | 9.6e-04 | 4.4 |
| S1c | Llama-3.2-1B | 41 | 4.63 | 76.8 | 9.5e-04 | 4.9 |
| S1c | Qwen2.5-0.5B | 38 | 4.29 | 82.8 | 9.3e-04 | 4.6 |
| S1c | bloom-560m | 41 | 4.63 | 76.8 | 8.9e-04 | 5.2 |
| S3 | bert-base-multilingual-cased | 32 | 3.61 | 98.3 | 9.8e-04 | 3.7 |
| S3 | xlm-roberta-base | 34 | 3.84 | 92.6 | 9.1e-04 | 4.2 |
| S3 | Llama-3.2-1B | 46 | 5.19 | 68.4 | 9.3e-04 | 5.6 |
| S3 | Qwen2.5-0.5B | 46 | 5.19 | 68.4 | 9.9e-04 | 5.2 |
| S3 | bloom-560m | 43 | 4.85 | 73.2 | 9.3e-04 | 5.2 |
| S7 | bert-base-multilingual-cased | 35 | 3.95 | 89.9 | 9.1e-04 | 4.3 |
| S7 | xlm-roberta-base | 37 | 4.18 | 85.1 | 9.1e-04 | 4.6 |
| S7 | Llama-3.2-1B | 36 | 4.06 | 87.4 | 8.7e-04 | 4.7 |
| S7 | Qwen2.5-0.5B | 29 | 3.27 | 108.5 | 8.8e-04 | 3.7 |
| S7 | bloom-560m | 37 | 4.18 | 85.1 | 8.2e-04 | 5.1 |
| S4 | (shared) | 19 | 2.14 | 165.6 | 8.2e-04 | 2.6 |
| S5 | (single) | 23 | 2.60 | 136.8 | 9.8e-04 | 2.7 |
| S6 | (single) | 57 | 6.43 | 55.2 | 9.0e-04 | 7.1 |
| S6p | (single) | 31 | 3.50 | 101.5 | 7.7e-04 | 4.6 |

A ratio near 1 means the signal alarms no more on real data than on shuffled nulls (sees no temporal structure); a large ratio means it does — but says nothing yet about *where* the alarms fall (Part 2 settles that).

## Part 2 — event-detection permutation test

S6 first (the supervised reference). Observed = median days event→next alarm; null = same statistic over 1000 random 7-date sets (60-day end margins).

| signal | obs median delay | null median [2.5,97.5] | perm p | within±30d | detects events? |
| --- | --- | --- | --- | --- | --- |
| S6 | 30 | 31 [8,70] | 0.484 | 5/7 (p=0.748) | no |
| S6p | 40 | 56 [17,129] | 0.284 | 5/7 (p=0.270) | no |
| S5 | 40 | 83 [26,209] | 0.110 | 3/7 (p=0.537) | no |
| S4 | 138 | 94 [32,198] | 0.815 | 0/7 (p=1.000) | no |
| S7 | 83 | 46 [15,99] | 0.911 | 3/7 (p=0.976) | no |
| S1 | 48 | 44 [18,87] | 0.580 | 4/7 (p=0.989) | no |
| S1c | 60 | 43 [16,89] | 0.821 | 5/7 (p=0.940) | no |
| S3 | 60 | 37 [15,74] | 0.905 | 5/7 (p=0.962) | no |

**Does S6 pass its own permutation test? NO.** If the supervised reference does not detect the events either, T6's lead-time comparison was between two chance processes and the paper's claim must be reframed: the question becomes whether *anything* detects discrete events, not who leads whom.

Signals passing the delay permutation test (p<0.05): **none**.

## Part 3 — bootstrap CIs on excess power (replicate-level, primary)

Units = **20 replicates** (per-replicate power = mean over the 5 tokenizers). Pooling tokenizers×replicates would overstate n (they share the underlying stream), so the replicate is the independent unit.

| p | signal | power(p) [95% CI] | excess = power(p)−power(0) [95% CI] |
| --- | --- | --- | --- |
| 0.05 | S1 | 0.67 [0.55,0.79] | +0.06 [-0.10,+0.22] |
| 0.05 | S1c | 0.89 [0.82,0.95] | -0.00 [-0.10,+0.11] |
| 0.05 | S3 | 0.74 [0.61,0.86] | -0.04 [-0.18,+0.10] |
| 0.05 | S4 | 0.55 [0.35,0.75] | +0.00 [-0.30,+0.30] |
| 0.05 | S7 | 0.70 [0.54,0.86] | -0.05 [-0.27,+0.17] |
| 0.1 | S1 | 0.75 [0.65,0.85] | +0.14 [-0.02,+0.29] |
| 0.1 | S1c | 0.84 [0.73,0.94] | -0.05 [-0.19,+0.08] |
| 0.1 | S3 | 0.79 [0.69,0.88] | +0.01 [-0.12,+0.13] |
| 0.1 | S4 | 0.40 [0.20,0.60] | -0.15 [-0.45,+0.15] |
| 0.1 | S7 | 0.63 [0.46,0.80] | -0.12 [-0.34,+0.10] |
| 0.25 | S1 | 0.61 [0.49,0.73] | +0.00 [-0.16,+0.17] |
| 0.25 | S1c | 0.73 [0.59,0.86] | -0.16 [-0.33,+0.00] |
| 0.25 | S3 | 0.62 [0.49,0.74] | -0.16 [-0.31,-0.02] |
| 0.25 | S4 | 0.45 [0.25,0.65] | -0.10 [-0.40,+0.20] |
| 0.25 | S7 | 0.66 [0.49,0.82] | -0.09 [-0.31,+0.13] |
| 0.5 | S1 | 0.48 [0.36,0.61] | -0.13 [-0.31,+0.04] |
| 0.5 | S1c | 0.75 [0.63,0.85] | -0.14 [-0.28,+0.00] |
| 0.5 | S3 | 0.53 [0.38,0.68] | -0.25 [-0.42,-0.07] |
| 0.5 | S4 | 0.65 [0.45,0.85] | +0.10 [-0.20,+0.40] |
| 0.5 | S7 | 0.70 [0.52,0.86] | -0.05 [-0.27,+0.18] |
| 1 | S1 | 0.76 [0.66,0.86] | +0.15 [-0.01,+0.31] |
| 1 | S1c | 0.89 [0.82,0.95] | +0.00 [-0.11,+0.11] |
| 1 | S3 | 0.87 [0.80,0.94] | +0.09 [-0.02,+0.20] |
| 1 | S4 | 0.80 [0.60,0.95] | +0.25 [-0.05,+0.50] |
| 1 | S7 | 0.94 [0.87,0.99] | +0.19 [+0.03,+0.35] |

**S7 − S4 excess-power difference (replicate-level, 95% CI):**

| p | S7−S4 excess [95% CI] | CI excludes 0? |
| --- | --- | --- |
| 0.05 | -0.05 [-0.35,+0.24] | no |
| 0.1 | +0.03 [-0.27,+0.34] | no |
| 0.25 | +0.01 [-0.29,+0.32] | no |
| 0.5 | -0.15 [-0.48,+0.17] | no |
| 1 | -0.06 [-0.37,+0.27] | no |

Intensities where the S7−S4 excess-power CI excludes zero: **none**.

## Part 4 — S6 detrended (is it events or just the monotone slide?)

z(S6) linear slope on the detection epoch = +5.70e-06/window. Alarm count **trended 57 → detrended 57**; permutation p **trended 0.484 → detrended 0.484**.


## STATUS

```
GATE 1 — total alarm counts reported for every signal on the real stream:      PASS
GATE 2 — permutation test run with >=1000 random date sets:                     PASS  (1000)
GATE 3 — bootstrap CIs on excess power, replicate-level:                        PASS

THE VERDICT ON DETECTION:
    signal | alarms A | rate/1000 | null FAR | ratio | median delay | perm p | detects?
    S6   | A=57 | 6.43 | 9.0e-04 | 7.1 | 30d | p=0.484 | no
    S6p  | A=31 | 3.50 | 7.7e-04 | 4.6 | 40d | p=0.284 | no
    S5   | A=23 | 2.60 | 9.8e-04 | 2.7 | 40d | p=0.110 | no
    S4   | A=19 | 2.14 | 8.2e-04 | 2.6 | 138d | p=0.815 | no
    S7   | A=35 | 3.95 | 9.1e-04 | 4.3 | 83d | p=0.911 | no
    S1   | A=28 | 3.16 | 9.8e-04 | 3.2 | 48d | p=0.580 | no
    S1c  | A=35 | 3.95 | 6.9e-04 | 5.8 | 60d | p=0.821 | no
    S3   | A=32 | 3.61 | 9.8e-04 | 3.7 | 60d | p=0.905 | no

    Does S6 pass its own permutation test?  NO
    Do ANY signals pass?  NONE

THE POWER DIFFERENCE:
    p=0.05: S7 excess -0.05[-0.27,+0.17], S4 excess +0.00[-0.30,+0.30]
    p=0.1: S7 excess -0.12[-0.34,+0.10], S4 excess -0.15[-0.45,+0.15]
    p=0.25: S7 excess -0.09[-0.31,+0.13], S4 excess -0.10[-0.40,+0.20]
    p=0.5: S7 excess -0.05[-0.27,+0.18], S4 excess +0.10[-0.20,+0.40]
    p=1: S7 excess +0.19[+0.03,+0.35], S4 excess +0.25[-0.05,+0.50]
    intensities where S7−S4 excess CI excludes zero: none

S6 DETRENDED:
    trended A=57 p=0.484 → detrended A=57 p=0.484

VERDICT: PROCEED WITH CAVEATS
Blockers:
  - none
Surprises worth a human decision:
  - NO signal — including the supervised S6 — passes the event-detection permutation test. The 7-event delays in T6 are indistinguishable from random dates: none of these detectors responds to discrete events. This reframes the whole comparison and must be the paper's opening sentence.
  - The S7−S4 excess-power CI includes zero at every intensity — the T6 point gap (+0.41 vs +0.20) is not significant at replicate level.
```
