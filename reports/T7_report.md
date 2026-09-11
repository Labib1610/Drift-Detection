# TASK 7 — is anything actually detecting anything?

- Uses only the frozen delta* in `results/calibration.json` (+ classifier/embeddings calibration). No re-tuning, no new signals. Permutation seed = 42.

## Part 1 — alarm census (real stream, detection epoch)

| signal | tokenizer | alarms A | rate/1000 win | mean interval (days) | null FAR | rate÷FAR |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | bert-base-multilingual-cased | 19 | 1.65 | 86.6 | 8.0e-04 | 2.1 |
| S1 | xlm-roberta-base | 24 | 2.09 | 68.5 | 9.4e-04 | 2.2 |
| S1 | Llama-3.2-1B | 15 | 1.31 | 109.7 | 7.0e-04 | 1.9 |
| S1 | Qwen2.5-0.5B | 14 | 1.22 | 117.5 | 7.3e-04 | 1.7 |
| S1 | bloom-560m | 31 | 2.70 | 53.1 | 9.4e-04 | 2.9 |
| S1c | bert-base-multilingual-cased | 19 | 1.65 | 86.6 | 9.1e-04 | 1.8 |
| S1c | xlm-roberta-base | 22 | 1.91 | 74.8 | 7.7e-04 | 2.5 |
| S1c | Llama-3.2-1B | 15 | 1.31 | 109.7 | 8.1e-04 | 1.6 |
| S1c | Qwen2.5-0.5B | 18 | 1.57 | 91.4 | 9.9e-04 | 1.6 |
| S1c | bloom-560m | 30 | 2.61 | 54.8 | 9.6e-04 | 2.7 |
| S3 | bert-base-multilingual-cased | 16 | 1.39 | 102.8 | 8.9e-04 | 1.6 |
| S3 | xlm-roberta-base | 30 | 2.61 | 54.8 | 9.9e-04 | 2.6 |
| S3 | Llama-3.2-1B | 12 | 1.04 | 137.1 | 8.3e-04 | 1.3 |
| S3 | Qwen2.5-0.5B | 12 | 1.04 | 137.1 | 7.9e-04 | 1.3 |
| S3 | bloom-560m | 28 | 2.44 | 58.8 | 9.7e-04 | 2.5 |
| S7 | bert-base-multilingual-cased | 25 | 2.18 | 65.8 | 9.0e-04 | 2.4 |
| S7 | xlm-roberta-base | 21 | 1.83 | 78.3 | 8.3e-04 | 2.2 |
| S7 | Llama-3.2-1B | 22 | 1.91 | 74.8 | 8.4e-04 | 2.3 |
| S7 | Qwen2.5-0.5B | 23 | 2.00 | 71.5 | 8.5e-04 | 2.4 |
| S7 | bloom-560m | 20 | 1.74 | 82.2 | 8.3e-04 | 2.1 |
| S4 | (shared) | 21 | 1.83 | 78.3 | 9.0e-04 | 2.0 |
| S5 | (single) | 17 | 1.48 | 96.8 | 7.6e-04 | 1.9 |
| S6 | (single) | 26 | 2.26 | 63.3 | 9.9e-04 | 2.3 |
| S6p | (single) | 13 | 1.13 | 126.5 | 9.5e-04 | 1.2 |

A ratio near 1 means the signal alarms no more on real data than on shuffled nulls (sees no temporal structure); a large ratio means it does — but says nothing yet about *where* the alarms fall (Part 2 settles that).

## Part 2 — event-detection permutation test

S6 first (the supervised reference). Observed = median days event→next alarm; null = same statistic over 1000 random 7-date sets (60-day end margins).

| signal | obs median delay | null median [2.5,97.5] | perm p | within±30d | detects events? |
| --- | --- | --- | --- | --- | --- |
| S6 | 24 | 39 [11,101] | 0.230 | 5/7 (p=0.495) | no |
| S6p | 143 | 76 [23,210] | 0.874 | 2/7 (p=0.871) | no |
| S5 | 41 | 49 [16,89] | 0.327 | 3/7 (p=0.904) | no |
| S4 | 23 | 41 [13,85] | 0.146 | 5/7 (p=0.477) | no |
| S7 | 22 | 43 [14,82] | 0.124 | 5/7 (p=0.909) | no |
| S1 | 44 | 50 [23,114] | 0.383 | 7/7 (p=0.750) | no |
| S1c | 42 | 44 [18,85] | 0.461 | 7/7 (p=0.814) | no |
| S3 | 44 | 54 [24,104] | 0.313 | 7/7 (p=0.701) | no |

**Does S6 pass its own permutation test? NO.** If the supervised reference does not detect the events either, T6's lead-time comparison was between two chance processes and the paper's claim must be reframed: the question becomes whether *anything* detects discrete events, not who leads whom.

Signals passing the delay permutation test (p<0.05): **none**.

## Part 3 — bootstrap CIs on excess power (replicate-level, primary)

Units = **20 replicates** (per-replicate power = mean over the 5 tokenizers). Pooling tokenizers×replicates would overstate n (they share the underlying stream), so the replicate is the independent unit.

| p | signal | power(p) [95% CI] | excess = power(p)−power(0) [95% CI] |
| --- | --- | --- | --- |
| 0.05 | S1 | 0.55 [0.38,0.70] | +0.03 [-0.20,+0.25] |
| 0.05 | S1c | 0.53 [0.36,0.70] | -0.10 [-0.33,+0.14] |
| 0.05 | S3 | 0.51 [0.37,0.67] | +0.08 [-0.11,+0.28] |
| 0.05 | S4 | 0.45 [0.25,0.65] | +0.10 [-0.20,+0.40] |
| 0.05 | S7 | 0.44 [0.26,0.64] | +0.06 [-0.20,+0.32] |
| 0.1 | S1 | 0.42 [0.26,0.58] | -0.10 [-0.32,+0.12] |
| 0.1 | S1c | 0.46 [0.32,0.60] | -0.17 [-0.38,+0.04] |
| 0.1 | S3 | 0.45 [0.31,0.60] | +0.02 [-0.18,+0.24] |
| 0.1 | S4 | 0.55 [0.30,0.80] | +0.20 [-0.10,+0.50] |
| 0.1 | S7 | 0.62 [0.46,0.77] | +0.24 [+0.00,+0.47] |
| 0.25 | S1 | 0.61 [0.46,0.75] | +0.09 [-0.12,+0.30] |
| 0.25 | S1c | 0.65 [0.51,0.78] | +0.02 [-0.18,+0.23] |
| 0.25 | S3 | 0.61 [0.45,0.78] | +0.18 [-0.04,+0.39] |
| 0.25 | S4 | 0.65 [0.45,0.85] | +0.30 [+0.00,+0.60] |
| 0.25 | S7 | 0.71 [0.54,0.87] | +0.33 [+0.09,+0.57] |
| 0.5 | S1 | 0.55 [0.39,0.70] | +0.03 [-0.19,+0.26] |
| 0.5 | S1c | 0.58 [0.41,0.75] | -0.05 [-0.27,+0.19] |
| 0.5 | S3 | 0.55 [0.41,0.68] | +0.12 [-0.07,+0.31] |
| 0.5 | S4 | 0.90 [0.75,1.00] | +0.55 [+0.30,+0.80] |
| 0.5 | S7 | 0.99 [0.97,1.00] | +0.61 [+0.42,+0.79] |
| 1 | S1 | 0.70 [0.59,0.80] | +0.18 [-0.00,+0.38] |
| 1 | S1c | 0.72 [0.61,0.83] | +0.09 [-0.09,+0.28] |
| 1 | S3 | 0.82 [0.71,0.91] | +0.39 [+0.21,+0.56] |
| 1 | S4 | 0.90 [0.75,1.00] | +0.55 [+0.30,+0.80] |
| 1 | S7 | 0.98 [0.94,1.00] | +0.60 [+0.41,+0.79] |

**S7 − S4 excess-power difference (replicate-level, 95% CI):**

| p | S7−S4 excess [95% CI] | CI excludes 0? |
| --- | --- | --- |
| 0.05 | -0.04 [-0.37,+0.29] | no |
| 0.1 | +0.04 [-0.35,+0.40] | no |
| 0.25 | +0.03 [-0.33,+0.38] | no |
| 0.5 | +0.06 [-0.24,+0.36] | no |
| 1 | +0.05 [-0.26,+0.36] | no |

Intensities where the S7−S4 excess-power CI excludes zero: **none**.

## Part 4 — S6 detrended (is it events or just the monotone slide?)

z(S6) linear slope on the detection epoch = +1.49e-04/window. Alarm count **trended 26 → detrended 27**; permutation p **trended 0.230 → detrended 0.223**.


## STATUS

```
GATE 1 — total alarm counts reported for every signal on the real stream:      PASS
GATE 2 — permutation test run with >=1000 random date sets:                     PASS  (1000)
GATE 3 — bootstrap CIs on excess power, replicate-level:                        PASS

THE VERDICT ON DETECTION:
    signal | alarms A | rate/1000 | null FAR | ratio | median delay | perm p | detects?
    S6   | A=26 | 2.26 | 9.9e-04 | 2.3 | 24d | p=0.230 | no
    S6p  | A=13 | 1.13 | 9.5e-04 | 1.2 | 143d | p=0.874 | no
    S5   | A=17 | 1.48 | 7.6e-04 | 1.9 | 41d | p=0.327 | no
    S4   | A=21 | 1.83 | 9.0e-04 | 2.0 | 23d | p=0.146 | no
    S7   | A=25 | 2.18 | 9.0e-04 | 2.4 | 22d | p=0.124 | no
    S1   | A=19 | 1.65 | 8.0e-04 | 2.1 | 44d | p=0.383 | no
    S1c  | A=19 | 1.65 | 9.1e-04 | 1.8 | 42d | p=0.461 | no
    S3   | A=16 | 1.39 | 8.9e-04 | 1.6 | 44d | p=0.313 | no

    Does S6 pass its own permutation test?  NO
    Do ANY signals pass?  NONE

THE POWER DIFFERENCE:
    p=0.05: S7 excess +0.06[-0.20,+0.32], S4 excess +0.10[-0.20,+0.40]
    p=0.1: S7 excess +0.24[+0.00,+0.47], S4 excess +0.20[-0.10,+0.50]
    p=0.25: S7 excess +0.33[+0.09,+0.57], S4 excess +0.30[+0.00,+0.60]
    p=0.5: S7 excess +0.61[+0.42,+0.79], S4 excess +0.55[+0.30,+0.80]
    p=1: S7 excess +0.60[+0.41,+0.79], S4 excess +0.55[+0.30,+0.80]
    intensities where S7−S4 excess CI excludes zero: none

S6 DETRENDED:
    trended A=26 p=0.230 → detrended A=27 p=0.223

VERDICT: PROCEED WITH CAVEATS
Blockers:
  - none
Surprises worth a human decision:
  - NO signal — including the supervised S6 — passes the event-detection permutation test. The 7-event delays in T6 are indistinguishable from random dates: none of these detectors responds to discrete events. This reframes the whole comparison and must be the paper's opening sentence.
  - The S7−S4 excess-power CI includes zero at every intensity — the T6 point gap (+0.41 vs +0.20) is not significant at replicate level.
```
