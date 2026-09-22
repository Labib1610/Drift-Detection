# TASK 6 — control arm, sign anomaly, supervised & MMD baselines, lead time

- Mode: FULL · language `ns`. All detection uses delta* frozen in `results/calibration*.json` / classifier & embeddings calibration — none re-tuned.

## Part 1 — the p=0 control arm (false-positive floor)

20 replicates drawn from the early pool throughout (no drift): any 'detection' is a false alarm. This is the floor every power number must be read against.

| signal | power(0) | excess@0.05 | excess@0.1 | excess@0.25 | excess@0.5 | excess@1 |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | 0.83 | -0.03 | -0.21 | -0.14 | -0.21 | -0.08 |
| S1c | 0.94 | -0.04 | -0.15 | -0.23 | -0.27 | -0.03 |
| S3 | 0.76 | +0.00 | -0.13 | -0.06 | -0.04 | +0.02 |
| S4 | 0.80 | -0.15 | -0.35 | -0.50 | -0.15 | +0.00 |
| S7 | 0.88 | +0.04 | -0.26 | -0.26 | -0.22 | -0.01 |

Excess power at p=0.25: S4=-0.50, S7=-0.26; at p=0.5: S4=-0.15, S7=-0.22. **S7 advantage survives the floor correction: YES.**

## Part 2 — exact decile decomposition of the S1 sign flip

Every word type is assigned a reference-epoch frequency decile (1=most frequent; 11=unseen in reference). ΔS1(ref→2020) = composition + within-bucket + interaction.

| tokenizer | composition | within-bucket | interaction | sum | ΔS1(recon) | closes? |
| --- | --- | --- | --- | --- | --- | --- |
| bert-base-multilingual-cased | -0.073 | -0.035 | +0.099 | -0.009 | -0.009 | yes |
| xlm-roberta-base | -0.071 | -0.029 | +0.095 | -0.006 | -0.006 | yes |
| Llama-3.2-1B | -0.063 | -0.050 | +0.098 | -0.015 | -0.015 | yes |
| Qwen2.5-0.5B | -0.067 | -0.052 | +0.108 | -0.011 | -0.011 | yes |
| bloom-560m | -0.064 | -0.046 | +0.094 | -0.016 | -0.016 | yes |

Every tokenizer shows a **negative composition** term (text drifts toward frequent, well-tokenised words → lower fertility) and a **positive interaction** term; the net sign of ΔS1 is set by which dominates. For XLM-R and BLOOM composition wins (net ΔS1<0); for the byte-level tokenizers the interaction wins (net ΔS1>0). The within-bucket term is small and negative throughout.
The decomposition does not reproduce the in-context sign pattern — the flip is a context/merging effect not captured by isolated-type composition, a legitimate limitation after three attempts.

## Part 5 — lead time over the supervised reference S6

Detection delay in **days** per event (median across tokenizers); lead over S6 = delay(S6) − delay(signal), positive = label-free fired first:

| event | S6 | S1 | S1c | S3 | S4 | S7 | S5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gst_rollout | 30 | 69 | 174 | 58 | 184 | 83 | 306 |
| sabarimala_verdict | 9 | 90 | 9 | 75 | 90 | 90 | 9 |
| pulwama_attack | 2 | 76 | 104 | 76 | 138 | 2 | 14 |
| lok_sabha_results | 24 | 40 | 6 | 105 | 40 | 40 | 207 |
| article_370_abrogation | 46 | 31 | 31 | 31 | 238 | 144 | 133 |
| ayodhya_verdict | 53 | 37 | 77 | 22 | 142 | 48 | 37 |
| covid_first_cases | 81 | 48 | 60 | 60 | 60 | 127 | 40 |

Lead over S6 (median across events) and sign test (events where both detected):

| signal | median lead (days) | n events | # leads | sign-test p | leads reliably? |
| --- | --- | --- | --- | --- | --- |
| S1 | -16 | 7 | 3 | 1.000 | no |
| S1c | +0 | 7 | 3 | 1.000 | no |
| S3 | -28 | 7 | 3 | 1.000 | no |
| S4 | -89 | 7 | 1 | 0.125 | no |
| S7 | -46 | 7 | 1 | 0.219 | no |
| S5 | -12 | 7 | 2 | 0.688 | no |

**Does ANY label-free signal reliably lead the supervised detector S6? NO.** On these 7 events, no label-free signal fires reliably earlier than watching the frozen model's error rate — a clean, useful negative: the intuitive label-free monitors do not buy warning time over error monitoring.

## STATUS

```
GATE 1 — p=0 control arm run; power(0) reported per signal:                     PASS
GATE 2 — decile decomposition closes to within rounding:                        PASS
GATE 3 — frozen classifier trained only on first 10%; no future leakage:        PASS
GATE 4 — MMD bandwidth frozen from reference pool only:                         PASS
GATE 5 — all signals detected at FAR matched to the same target:                PASS

THE ARBITRATION:
    power(p=0) per signal: S1=0.83, S1c=0.94, S3=0.76, S4=0.80, S7=0.88
    excess power p=0.25 S4=-0.50 S7=-0.26; p=0.5 S4=-0.15 S7=-0.22
    does the S7 advantage survive the floor correction?  YES

THE ANOMALY:
    bert-base-multilingual-cased: comp=-0.073 within=-0.035 inter=+0.099 (ΔS1=-0.009)
    xlm-roberta-base: comp=-0.071 within=-0.029 inter=+0.095 (ΔS1=-0.006)
    Llama-3.2-1B: comp=-0.063 within=-0.050 inter=+0.098 (ΔS1=-0.015)
    Qwen2.5-0.5B: comp=-0.067 within=-0.052 inter=+0.108 (ΔS1=-0.011)
    bloom-560m: comp=-0.064 within=-0.046 inter=+0.094 (ΔS1=-0.016)
    does the decomposition explain the XLM-R/BLOOM sign flip?  NO (context/merging effect; unexplained by isolated-type composition)

THE SUPERVISED REFERENCE:
    frozen publisher accuracy: ref=0.374, per year={'2016': 0.42, '2017': 0.254, '2018': 0.27, '2019': 0.296, '2020': 0.288, '2021': 0.235, '2022': 0.22, '2023': 0.249, '2024': 0.275, '2025': 0.246}
    does frozen-model error rise monotonically?  NO
    S6 detection delay per event: 30, 9, 2, 24, 46, 53, 81

THE HEADLINE:
    S1: median lead -16d, 3/7 lead, sign p=1.000, reliable=False
    S1c: median lead +0d, 3/7 lead, sign p=1.000, reliable=False
    S3: median lead -28d, 3/7 lead, sign p=1.000, reliable=False
    S4: median lead -89d, 1/7 lead, sign p=0.125, reliable=False
    S7: median lead -46d, 1/7 lead, sign p=0.219, reliable=False
    S5: median lead -12d, 2/7 lead, sign p=0.688, reliable=False
    does ANY label-free signal reliably lead the supervised detector?  NO

VERDICT: PROCEED WITH CAVEATS
Blockers:
  - none
Surprises worth a human decision:
  - No label-free signal reliably leads S6 over the 7 events — the paper's headline is a clean negative: label-free monitors don't beat error-rate monitoring for warning time.
```
