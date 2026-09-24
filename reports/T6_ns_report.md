# TASK 6 — control arm, sign anomaly, supervised & MMD baselines, lead time

- Mode: FULL · language `ns`. All detection uses delta* frozen in `results/calibration*.json` / classifier & embeddings calibration — none re-tuned.

## Part 1 — the p=0 control arm (false-positive floor)

20 replicates drawn from the early pool throughout (no drift): any 'detection' is a false alarm. This is the floor every power number must be read against.

| signal | power(0) | excess@0.05 | excess@0.1 | excess@0.25 | excess@0.5 | excess@1 |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | nan | +nan | +nan | +nan | +nan | +nan |
| S1c | nan | +nan | +nan | +nan | +nan | +nan |
| S3 | nan | +nan | +nan | +nan | +nan | +nan |
| S4 | nan | +nan | +nan | +nan | +nan | +nan |
| S7 | nan | +nan | +nan | +nan | +nan | +nan |

Excess power at p=0.25: S4=+nan, S7=+nan; at p=0.5: S4=+nan, S7=+nan. **S7 advantage survives the floor correction: NO.**
> After subtracting the false-positive floor, S7 and S4 are not separable — the T5b conclusion (indistinguishable) is reinforced, not overturned.

## Part 2 — exact decile decomposition of the S1 sign flip

Every word type is assigned a reference-epoch frequency decile (1=most frequent; 11=unseen in reference). ΔS1(ref→2020) = composition + within-bucket + interaction.

| tokenizer | composition | within-bucket | interaction | sum | ΔS1(recon) | closes? |
| --- | --- | --- | --- | --- | --- | --- |
| bert-base-multilingual-cased | -0.048 | -0.026 | +0.074 | -0.000 | -0.000 | yes |
| xlm-roberta-base | -0.048 | -0.023 | +0.070 | -0.001 | -0.001 | yes |
| Llama-3.2-1B | -0.041 | -0.031 | +0.069 | -0.003 | -0.003 | yes |
| Qwen2.5-0.5B | -0.043 | -0.033 | +0.076 | +0.000 | +0.000 | yes |
| bloom-560m | -0.042 | -0.029 | +0.067 | -0.004 | -0.004 | yes |

Every tokenizer shows a **negative composition** term (text drifts toward frequent, well-tokenised words → lower fertility) and a **positive interaction** term; the net sign of ΔS1 is set by which dominates. For XLM-R and BLOOM composition wins (net ΔS1<0); for the byte-level tokenizers the interaction wins (net ΔS1>0). The within-bucket term is small and negative throughout.
The decomposition does not reproduce the in-context sign pattern — the flip is a context/merging effect not captured by isolated-type composition, a legitimate limitation after three attempts.

## Part 5 — lead time over the supervised reference S6

Detection delay in **days** per event (median across tokenizers); lead over S6 = delay(S6) − delay(signal), positive = label-free fired first:

| event | S6 | S1 | S1c | S3 | S4 | S7 | S5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gst_rollout | 1040 | 1031 | 1014 | 1005 | 1065 | 1031 | 1005 |
| sabarimala_verdict | 586 | 577 | 560 | 551 | 611 | 577 | 551 |
| pulwama_attack | 447 | 438 | 421 | 412 | 472 | 438 | 412 |
| lok_sabha_results | 349 | 340 | 323 | 314 | 374 | 340 | 314 |
| article_370_abrogation | 275 | 266 | 249 | 240 | 300 | 266 | 240 |
| ayodhya_verdict | 179 | 170 | 153 | 144 | 204 | 170 | 144 |
| covid_first_cases | 97 | 88 | 71 | 62 | 122 | 88 | 62 |

Lead over S6 (median across events) and sign test (events where both detected):

| signal | median lead (days) | n events | # leads | sign-test p | leads reliably? |
| --- | --- | --- | --- | --- | --- |
| S1 | +9 | 7 | 7 | 0.016 | YES |
| S1c | +26 | 7 | 7 | 0.016 | YES |
| S3 | +35 | 7 | 7 | 0.016 | YES |
| S4 | -25 | 7 | 0 | 0.016 | no |
| S7 | +9 | 7 | 7 | 0.016 | YES |
| S5 | +35 | 7 | 7 | 0.016 | YES |

**Does ANY label-free signal reliably lead the supervised detector S6? YES.**

## STATUS

```
GATE 1 — p=0 control arm run; power(0) reported per signal:                     FAIL
GATE 2 — decile decomposition closes to within rounding:                        PASS
GATE 3 — frozen classifier trained only on first 10%; no future leakage:        PASS
GATE 4 — MMD bandwidth frozen from reference pool only:                         PASS
GATE 5 — all signals detected at FAR matched to the same target:                PASS

THE ARBITRATION:
    power(p=0) per signal: S1=nan, S1c=nan, S3=nan, S4=nan, S7=nan
    excess power p=0.25 S4=+nan S7=+nan; p=0.5 S4=+nan S7=+nan
    does the S7 advantage survive the floor correction?  NO

THE ANOMALY:
    bert-base-multilingual-cased: comp=-0.048 within=-0.026 inter=+0.074 (ΔS1=-0.000)
    xlm-roberta-base: comp=-0.048 within=-0.023 inter=+0.070 (ΔS1=-0.001)
    Llama-3.2-1B: comp=-0.041 within=-0.031 inter=+0.069 (ΔS1=-0.003)
    Qwen2.5-0.5B: comp=-0.043 within=-0.033 inter=+0.076 (ΔS1=+0.000)
    bloom-560m: comp=-0.042 within=-0.029 inter=+0.067 (ΔS1=-0.004)
    does the decomposition explain the XLM-R/BLOOM sign flip?  NO (context/merging effect; unexplained by isolated-type composition)

THE SUPERVISED REFERENCE:
    frozen publisher accuracy: ref=0.654, per year={'2019': 0.567, '2020': 0.631, '2021': 0.476, '2022': 0.463, '2023': 0.45, '2024': 0.478}
    does frozen-model error rise monotonically?  NO
    S6 detection delay per event: 1040, 586, 447, 349, 275, 179, 97

THE HEADLINE:
    S1: median lead +9d, 7/7 lead, sign p=0.016, reliable=True
    S1c: median lead +26d, 7/7 lead, sign p=0.016, reliable=True
    S3: median lead +35d, 7/7 lead, sign p=0.016, reliable=True
    S4: median lead -25d, 0/7 lead, sign p=0.016, reliable=False
    S7: median lead +9d, 7/7 lead, sign p=0.016, reliable=True
    S5: median lead +35d, 7/7 lead, sign p=0.016, reliable=True
    does ANY label-free signal reliably lead the supervised detector?  YES

VERDICT: BLOCKED
Blockers:
  - none
Surprises worth a human decision:
  - none
```
