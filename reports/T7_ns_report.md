# TASK 7 — is anything actually detecting anything?

- Language `ns`. Uses only the frozen delta* in `results/calibration_ns.json` (+ classifier/embeddings calibration). No re-tuning, no new signals. Permutation seed = 42.

## Part 1 — alarm census (real stream, detection epoch)

| signal | tokenizer | alarms A | rate/1000 win | mean interval (days) | null FAR | rate÷FAR |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | bert-base-multilingual-cased | 30 | 4.88 | 55.3 | 7.0e-04 | 7.0 |
| S1 | xlm-roberta-base | 29 | 4.72 | 57.2 | 9.6e-04 | 4.9 |
| S1 | Llama-3.2-1B | 29 | 4.72 | 57.2 | 9.8e-04 | 4.8 |
| S1 | Qwen2.5-0.5B | 31 | 5.05 | 53.5 | 9.8e-04 | 5.2 |
| S1 | bloom-560m | 26 | 4.23 | 63.8 | 7.8e-04 | 5.4 |
| S1c | bert-base-multilingual-cased | 33 | 5.37 | 50.2 | 5.9e-04 | 9.2 |
| S1c | xlm-roberta-base | 36 | 5.86 | 46.1 | 9.1e-04 | 6.4 |
| S1c | Llama-3.2-1B | 37 | 6.02 | 44.8 | 8.5e-04 | 7.1 |
| S1c | Qwen2.5-0.5B | 34 | 5.54 | 48.8 | 8.9e-04 | 6.2 |
| S1c | bloom-560m | 39 | 6.35 | 42.5 | 9.4e-04 | 6.7 |
| S3 | bert-base-multilingual-cased | 36 | 5.86 | 46.1 | 8.6e-04 | 6.8 |
| S3 | xlm-roberta-base | 31 | 5.05 | 53.5 | 9.8e-04 | 5.2 |
| S3 | Llama-3.2-1B | 32 | 5.21 | 51.8 | 7.8e-04 | 6.7 |
| S3 | Qwen2.5-0.5B | 32 | 5.21 | 51.8 | 7.6e-04 | 6.8 |
| S3 | bloom-560m | 27 | 4.40 | 61.4 | 7.6e-04 | 5.7 |
| S7 | bert-base-multilingual-cased | 32 | 5.21 | 51.8 | 9.9e-04 | 5.2 |
| S7 | xlm-roberta-base | 29 | 4.72 | 57.2 | 7.3e-04 | 6.4 |
| S7 | Llama-3.2-1B | 32 | 5.21 | 51.8 | 9.3e-04 | 5.6 |
| S7 | Qwen2.5-0.5B | 28 | 4.56 | 59.2 | 9.9e-04 | 4.6 |
| S7 | bloom-560m | 32 | 5.21 | 51.8 | 9.8e-04 | 5.3 |
| S4 | (shared) | 27 | 4.40 | 61.4 | 8.0e-04 | 5.5 |
| S5 | (single) | 17 | 2.77 | 97.5 | 7.8e-04 | 3.5 |
| S6 | (single) | 51 | 8.30 | 32.5 | 7.6e-04 | 10.9 |
| S6p | (single) | 25 | 4.07 | 66.3 | 9.4e-04 | 4.3 |

A ratio near 1 means the signal alarms no more on real data than on shuffled nulls (sees no temporal structure); a large ratio means it does — but says nothing yet about *where* the alarms fall (Part 2 settles that).

## Part 2 — event-detection permutation test

S6 first (the supervised reference). Observed = median days event→next alarm; null = same statistic over 1000 random 7-date sets (60-day end margins).

| signal | obs median delay | null median [2.5,97.5] | perm p | within±30d | detects events? |
| --- | --- | --- | --- | --- | --- |
| S6 | 227 | 20 [6,61] | 1.000 | 0/7 (p=1.000) | no |
| S6p | 227 | 44 [14,98] | 1.000 | 0/7 (p=1.000) | no |
| S5 | 192 | 62 [19,136] | 1.000 | 0/7 (p=1.000) | no |
| S4 | 252 | 44 [11,119] | 1.000 | 0/7 (p=1.000) | no |
| S7 | 218 | 35 [10,90] | 1.000 | 0/7 (p=1.000) | no |
| S1 | 218 | 39 [14,111] | 1.000 | 0/7 (p=1.000) | no |
| S1c | 201 | 33 [10,75] | 1.000 | 0/7 (p=1.000) | no |
| S3 | 192 | 34 [11,80] | 1.000 | 0/7 (p=1.000) | no |

**Does S6 pass its own permutation test? NO.** If the supervised reference does not detect the events either, T6's lead-time comparison was between two chance processes and the paper's claim must be reframed: the question becomes whether *anything* detects discrete events, not who leads whom.

Signals passing the delay permutation test (p<0.05): **none**.

## Part 3 — bootstrap CIs on excess power (replicate-level, primary)

Units = **20 replicates** (per-replicate power = mean over the 5 tokenizers). Pooling tokenizers×replicates would overstate n (they share the underlying stream), so the replicate is the independent unit.

| p | signal | power(p) [95% CI] | excess = power(p)−power(0) [95% CI] |
| --- | --- | --- | --- |
| 0.05 | S1 | 0.87 [0.76,0.96] | -0.03 [-0.16,+0.09] |
| 0.05 | S1c | 0.90 [0.80,0.97] | +0.12 [-0.03,+0.28] |
| 0.05 | S3 | 0.80 [0.68,0.91] | -0.05 [-0.19,+0.09] |
| 0.05 | S4 | 1.00 [1.00,1.00] | +0.05 [+0.00,+0.15] |
| 0.05 | S7 | 0.97 [0.92,1.00] | -0.02 [-0.08,+0.02] |
| 0.1 | S1 | 0.95 [0.88,1.00] | +0.05 [-0.04,+0.13] |
| 0.1 | S1c | 0.83 [0.73,0.91] | +0.05 [-0.10,+0.21] |
| 0.1 | S3 | 0.89 [0.78,0.97] | +0.04 [-0.09,+0.16] |
| 0.1 | S4 | 1.00 [1.00,1.00] | +0.05 [+0.00,+0.15] |
| 0.1 | S7 | 1.00 [1.00,1.00] | +0.01 [+0.00,+0.03] |
| 0.25 | S1 | 0.90 [0.80,0.98] | +0.00 [-0.12,+0.11] |
| 0.25 | S1c | 0.85 [0.73,0.94] | +0.07 [-0.08,+0.23] |
| 0.25 | S3 | 0.86 [0.74,0.96] | +0.01 [-0.13,+0.14] |
| 0.25 | S4 | 1.00 [1.00,1.00] | +0.05 [+0.00,+0.15] |
| 0.25 | S7 | 0.99 [0.97,1.00] | +0.00 [-0.03,+0.03] |
| 0.5 | S1 | 0.90 [0.80,0.98] | +0.00 [-0.11,+0.10] |
| 0.5 | S1c | 0.78 [0.66,0.89] | +0.00 [-0.16,+0.17] |
| 0.5 | S3 | 0.86 [0.75,0.96] | +0.01 [-0.13,+0.14] |
| 0.5 | S4 | 1.00 [1.00,1.00] | +0.05 [+0.00,+0.15] |
| 0.5 | S7 | 1.00 [1.00,1.00] | +0.01 [+0.00,+0.03] |
| 1 | S1 | 0.96 [0.91,1.00] | +0.06 [-0.01,+0.14] |
| 1 | S1c | 0.83 [0.69,0.95] | +0.05 [-0.13,+0.23] |
| 1 | S3 | 0.95 [0.89,1.00] | +0.10 [+0.00,+0.19] |
| 1 | S4 | 1.00 [1.00,1.00] | +0.05 [+0.00,+0.15] |
| 1 | S7 | 1.00 [1.00,1.00] | +0.01 [+0.00,+0.03] |

**S7 − S4 excess-power difference (replicate-level, 95% CI):**

| p | S7−S4 excess [95% CI] | CI excludes 0? |
| --- | --- | --- |
| 0.05 | -0.07 [-0.19,+0.01] | no |
| 0.1 | -0.04 [-0.15,+0.02] | no |
| 0.25 | -0.05 [-0.16,+0.02] | no |
| 0.5 | -0.04 [-0.15,+0.02] | no |
| 1 | -0.04 [-0.15,+0.03] | no |

Intensities where the S7−S4 excess-power CI excludes zero: **none**.

## Part 4 — S6 detrended (is it events or just the monotone slide?)

z(S6) linear slope on the detection epoch = +9.59e-05/window. Alarm count **trended 51 → detrended 51**; permutation p **trended 1.000 → detrended 1.000**.


## STATUS

```
GATE 1 — total alarm counts reported for every signal on the real stream:      PASS
GATE 2 — permutation test run with >=1000 random date sets:                     PASS  (1000)
GATE 3 — bootstrap CIs on excess power, replicate-level:                        PASS

THE VERDICT ON DETECTION:
    signal | alarms A | rate/1000 | null FAR | ratio | median delay | perm p | detects?
    S6   | A=51 | 8.30 | 7.6e-04 | 10.9 | 227d | p=1.000 | no
    S6p  | A=25 | 4.07 | 9.4e-04 | 4.3 | 227d | p=1.000 | no
    S5   | A=17 | 2.77 | 7.8e-04 | 3.5 | 192d | p=1.000 | no
    S4   | A=27 | 4.40 | 8.0e-04 | 5.5 | 252d | p=1.000 | no
    S7   | A=32 | 5.21 | 9.9e-04 | 5.2 | 218d | p=1.000 | no
    S1   | A=30 | 4.88 | 7.0e-04 | 7.0 | 218d | p=1.000 | no
    S1c  | A=33 | 5.37 | 5.9e-04 | 9.2 | 201d | p=1.000 | no
    S3   | A=36 | 5.86 | 8.6e-04 | 6.8 | 192d | p=1.000 | no

    Does S6 pass its own permutation test?  NO
    Do ANY signals pass?  NONE

THE POWER DIFFERENCE:
    p=0.05: S7 excess -0.02[-0.08,+0.02], S4 excess +0.05[+0.00,+0.15]
    p=0.1: S7 excess +0.01[+0.00,+0.03], S4 excess +0.05[+0.00,+0.15]
    p=0.25: S7 excess +0.00[-0.03,+0.03], S4 excess +0.05[+0.00,+0.15]
    p=0.5: S7 excess +0.01[+0.00,+0.03], S4 excess +0.05[+0.00,+0.15]
    p=1: S7 excess +0.01[+0.00,+0.03], S4 excess +0.05[+0.00,+0.15]
    intensities where S7−S4 excess CI excludes zero: none

S6 DETRENDED:
    trended A=51 p=1.000 → detrended A=51 p=1.000

VERDICT: PROCEED WITH CAVEATS
Blockers:
  - none
Surprises worth a human decision:
  - NO signal — including the supervised S6 — passes the event-detection permutation test. The 7-event delays in T6 are indistinguishable from random dates: none of these detectors responds to discrete events. This reframes the whole comparison and must be the paper's opening sentence.
  - The S7−S4 excess-power CI includes zero at every intensity — the T6 point gap (+0.41 vs +0.20) is not significant at replicate level.
```
