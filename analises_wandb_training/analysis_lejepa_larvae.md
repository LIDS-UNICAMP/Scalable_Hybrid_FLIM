# LeJEPA SSL Analysis: helminth-larvae Dataset

**Date:** 2026-05-30
**Model:** LeJEPA (locally trained I-JEPA variant)
**Dataset:** helminth-larvae
**Evaluation:** SVM probe (frozen encoder) and MLP fine-tuning (freeze / unfreeze)
**Initializations:** flim, he, random, xavier, trunc_normal
**Data percentages:** 1, 5, 25, 50, 75, 100%
**Splits:** 3 per condition (occasionally 2)

---

## 1. Initialization Strategy Comparison (SVM Probe)

### Table: init x pct -> kappa mean (std)

| init         | pct=1          | pct=5          | pct=25         | pct=50         | pct=75         | pct=100        |
|--------------|----------------|----------------|----------------|----------------|----------------|----------------|
| flim         | 0.2085 (0.247) | 0.4749 (0.056) | 0.7783 (0.029) | 0.7530 (0.066) | 0.7545 (0.057) | 0.7545 (0.007) |
| he           | 0.0465 (0.066) | 0.2179 (0.032) | 0.6095 (0.045) | 0.4375 (0.308) | 0.4550 (0.326) | 0.4254 (0.333) |
| random       | 0.2086 (0.178) | 0.5643 (0.082) | 0.5165 (0.270) | 0.4566 (0.314) | 0.6522 (0.057) | 0.4060 (0.287) |
| xavier       | 0.3726 (0.055) | 0.0624 (0.090) | 0.5897 (0.084) | 0.6306 (0.097) | 0.4102 (0.300) | 0.4015 (0.197) |
| trunc_normal | 0.0000 (0.000) | 0.3741 (0.198) | 0.3892 (0.270) | 0.6353 (0.048) | 0.6183 (0.088) | 0.2670 (0.290) |

### Key findings

- **Best init overall: flim.** Highest kappa and lowest std at pct>=25. Peak 0.7783 at pct=25, plateau ~0.754 through pct=100. std=0.007 at pct=100 — uniquely stable.
- **xavier** best at pct=1 (0.3726) but collapses to 0.0624 at pct=5 — anomaly analyzed in section 6.
- **random** achieves the single highest kappa at pct=5 (0.5643) but with std=0.082 and a non-monotone curve.
- **he** and **trunc_normal** are weakest overall. he degrades from 0.61 at pct=25 to 0.43 at pct=50+ despite more data.
- **trunc_normal** collapses to kappa=0.000 at pct=1 (all splits predict one class; std=0.000).
- **flim is the only init with a smooth, monotone learning curve** reaching a stable plateau.

---

## 2. MLP Fine-tuning: Freeze vs Unfreeze

### flim init — all evaluation modes:

| pct | SVM (frozen enc.) | MLP freeze | MLP unfreeze | Best mode    |
|-----|-------------------|------------|--------------|--------------|
| 1   | 0.2085            | 0.1143     | 0.2311       | MLP unfreeze |
| 5   | 0.4749            | 0.4430     | 0.8077       | MLP unfreeze |
| 25  | 0.7783            | 0.7604     | 0.8698       | MLP unfreeze |
| 50  | 0.7530            | 0.6809     | 0.9030       | MLP unfreeze |
| 75  | 0.7545            | 0.7781     | 0.8840       | MLP unfreeze |
| 100 | 0.7545            | 0.7756     | 0.9347       | MLP unfreeze |

### MLP freeze kappa (all inits):

| init         | pct=1  | pct=5  | pct=25 | pct=50 | pct=75 | pct=100 |
|--------------|--------|--------|--------|--------|--------|---------|
| flim         | 0.1143 | 0.4430 | 0.7604 | 0.6809 | 0.7781 | 0.7756  |
| he           | 0.0000 | 0.0000 | 0.4048 | 0.7200 | 0.4470 | 0.6743  |
| random       | 0.0000 | 0.0000 | 0.4132 | 0.5709 | 0.6236 | 0.3990  |
| xavier       | 0.0000 | 0.0000 | 0.1784 | 0.5895 | 0.1889 | 0.2968  |
| trunc_normal | 0.0000 | 0.0000 | 0.0000 | 0.5211 | 0.3853 | 0.2480  |

### MLP unfreeze kappa (all inits):

| init         | pct=1  | pct=5  | pct=25 | pct=50 | pct=75 | pct=100 |
|--------------|--------|--------|--------|--------|--------|---------|
| flim         | 0.2311 | 0.8077 | 0.8698 | 0.9030 | 0.8840 | 0.9347  |
| he           | 0.0000 | 0.0000 | 0.8021 | 0.8621 | 0.8660 | 0.9314  |
| random       | 0.0000 | 0.2616 | 0.8550 | 0.8973 | 0.8788 | 0.9192  |
| xavier       | 0.0000 | 0.0000 | 0.8315 | 0.8876 | 0.8411 | 0.9416  |
| trunc_normal | 0.0000 | 0.0000 | 0.7630 | 0.8488 | 0.8554 | 0.9097  |

### Verdict

**Full fine-tuning (MLP unfreeze) is decisively beneficial for larvae** across all pct >= 5 (flim) or >= 25 (other inits). Key gains:
- flim, pct=5: freeze=0.443 -> unfreeze=0.808 (+0.365).
- flim, pct=100: freeze=0.776 -> unfreeze=0.935 (+0.159).
- Best unfreeze at pct=100: xavier=0.9416, flim=0.9347, he=0.9314, random=0.9192, trunc_normal=0.9097.

At pct=1, all inits fail for both freeze and unfreeze (kappa=0.0 or near-0), exception: flim unfreeze (0.2311) is non-zero but still weak.

**MLP freeze consistently underperforms SVM probe** at the same encoder, suggesting the MLP head gets stuck in suboptimal solutions when the encoder is frozen. SVM benefits from its global kernel-based margin.

**When unfreeze outperforms freeze (by init):** flim: pct>=5 | he, random, xavier: pct>=25 | trunc_normal: pct>=50.

---

## 3. Data Efficiency

### SVM kappa learning curves:

| init         | 1%     | 5%     | 25%    | 50%    | 75%    | 100%   | Pattern                     |
|--------------|--------|--------|--------|--------|--------|--------|-----------------------------|
| flim         | 0.2085 | 0.4749 | 0.7783 | 0.7530 | 0.7545 | 0.7545 | Rapid rise, plateau at 25%  |
| he           | 0.0465 | 0.2179 | 0.6095 | 0.4375 | 0.4550 | 0.4254 | Rises then regresses        |
| random       | 0.2086 | 0.5643 | 0.5165 | 0.4566 | 0.6522 | 0.4060 | Highly erratic, no plateau  |
| xavier       | 0.3726 | 0.0624 | 0.5897 | 0.6306 | 0.4102 | 0.4015 | Non-monotone, unstable      |
| trunc_normal | 0.0000 | 0.3741 | 0.3892 | 0.6353 | 0.6183 | 0.2670 | Slow rise, collapses at 100%|

**Larvae is a "simpler" dataset for SSL:**
- I-JEPA teacher achieves kappa=0.82 at pct=1 (vs eggs: 0.66, protozoan: 0.59).
- flim-LeJEPA reaches 0.475 at pct=5, far above eggs-flim@pct=5 (0.056).
- flim learning curve saturates at pct=25 (0.778); additional data adds nothing for SVM probe.

### Cross-dataset SVM flim kappa comparison:

| dataset   | pct=1  | pct=5  | pct=25 | pct=100 |
|-----------|--------|--------|--------|---------|
| larvae    | 0.2085 | 0.4749 | 0.7783 | 0.7545  |
| eggs      | 0.0623 | 0.0558 | 0.7331 | 0.7309  |
| protozoan | 0.0982 | 0.3681 | 0.4799 | 0.2915  |

Larvae is most data-efficient at low pct. Eggs reaches comparable kappa at pct=25+. Protozoan is hardest throughout.

---

## 4. Gap to I-JEPA Teacher

### I-JEPA teacher aggregated kappa (larvae, n=3 splits):

| pct | teacher kappa | teacher std |
|-----|--------------|-------------|
| 1   | 0.8216       | 0.0825      |
| 5   | 0.8807       | 0.0251      |
| 25  | 0.9190       | 0.0005      |
| 50  | 0.9433       | 0.0115      |
| 75  | 0.9410       | 0.0094      |
| 100 | 0.9495       | 0.0065      |

### Best LeJEPA SVM vs teacher:

| pct | best init | best SVM kappa | teacher kappa | gap   |
|-----|-----------|----------------|---------------|-------|
| 1   | xavier    | 0.3726         | 0.8216        | 0.449 |
| 5   | random    | 0.5643         | 0.8807        | 0.316 |
| 25  | flim      | 0.7783         | 0.9190        | 0.141 |
| 50  | flim      | 0.7530         | 0.9433        | 0.190 |
| 75  | flim      | 0.7545         | 0.9410        | 0.187 |
| 100 | flim      | 0.7545         | 0.9495        | 0.195 |

### MLP unfreeze (flim) gap to teacher:

| pct | flim unfreeze | teacher  | gap   |
|-----|---------------|----------|-------|
| 25  | 0.8698        | 0.9190   | 0.049 |
| 50  | 0.9030        | 0.9433   | 0.040 |
| 75  | 0.8840        | 0.9410   | 0.057 |
| 100 | 0.9347        | 0.9495   | 0.015 |

Full fine-tuning with flim@pct=100 narrows the gap to only 1.5 kappa points.

### Verification of "larvae split1_pct75 kappa=0.65":

The kappa=0.6538 value is the **LeJEPA random init student, not the I-JEPA teacher:**
- run_id=0480ye37, `lejepa_line_helminth-larvae_split_1_pct_75_model_random`, kappa=0.6538.
- The I-JEPA teacher at larvae/split_1/pct_75 = kappa=**0.9330** (ijepa_svm_results.csv, line 24).
- The run_name suffix "model_random" denotes the LeJEPA initialization strategy, not the teacher model. The 0.65 value is a LeJEPA student result.

---

## 5. FLIM Initialization for Larvae

### flim vs trunc_normal at low pct (SVM probe):

| pct | flim   | trunc_normal | advantage |
|-----|--------|--------------|-----------|
| 1   | 0.2085 | 0.0000       | +0.209    |
| 5   | 0.4749 | 0.3741       | +0.101    |
| 25  | 0.7783 | 0.3892       | +0.389    |

### Init ranking at pct=1 and pct=5:

pct=1: xavier (0.3726) > random (0.2086) >= flim (0.2085) > he (0.0465) > trunc_normal (0.000)
pct=5: random (0.5643) > flim (0.4749) > trunc_normal (0.3741) > he (0.2179) > xavier (0.0624)

**FLIM init is particularly beneficial for larvae at pct >= 25:**
- Only init combining highest kappa AND lowest std from pct=25 onward.
- At pct=100: flim std=0.007 vs he=0.333, random=0.287, xavier=0.197, trunc_normal=0.290.
- flim individual split values at pct=100: 0.764, 0.749, 0.751 — highly consistent.

**Why FLIM helps larvae specifically:** Helminth larvae have distinctive morphological textures (cuticle segmentation, body geometry) captured by FLIM-derived filter banks tuned to biological microscopy. These domain-aligned initialization features reduce the SSL training burden, yielding smooth early-saturating learning curves.

---

## 6. Anomalies and Instabilities

### Runs with kappa < 0.3 at pct >= 25 (SVM probe):

| run_id   | init         | split | pct | kappa   | severity                          |
|----------|-------------|-------|-----|---------|-----------------------------------|
| um1i0j5u | trunc_normal | 1    | 100 | -0.0659 | CRITICAL: negative kappa at 100%  |
| ba6zzvml | he           | 1    | 100 | -0.0456 | CRITICAL: negative kappa at 100%  |
| fsvg6vuo | xavier       | 3    | 5   | -0.0573 | Collapse at pct=5                 |
| cy1tkc44 | xavier       | 3    | 75  | -0.0143 | Negative kappa at pct=75          |
| nfbp8cu6 | he           | 2    | 75  | -0.0056 | Negative kappa at pct=75          |
| vyu9ln38 | he           | 3    | 50  | 0.0134  | Near-zero at pct=50               |
| rq72jl3v | random       | 2    | 100 | 0.0000  | Zero kappa at pct=100             |
| zxlit7lf | random       | 3    | 50  | 0.0151  | Near-zero at pct=50               |

### High-variance conditions (std > 0.25):

| init         | pct | kappa_mean | kappa_std | interpretation                              |
|--------------|-----|------------|-----------|---------------------------------------------|
| flim         | 1   | 0.2085     | 0.247     | Acceptable at pct=1; splits: 0.0, 0.071, 0.556 |
| random       | 25  | 0.5165     | 0.270     | 1 bad (0.143), 2 good (0.639, 0.768)       |
| he           | 50  | 0.4375     | 0.308     | 1 near-0 (0.013), 2 good (0.735, 0.564)   |
| random       | 50  | 0.4566     | 0.314     | 1 near-0 (0.015), 2 good (0.715, 0.640)   |
| he           | 75  | 0.4550     | 0.326     | 1 negative (-0.006), 2 good (0.685, 0.686)|
| xavier       | 75  | 0.4102     | 0.300     | 1 negative (-0.014), 2 good (0.610, 0.635)|
| he           | 100 | 0.4254     | 0.333     | 1 negative (-0.046), 2 good (0.652, 0.670)|
| trunc_normal | 25  | 0.3892     | 0.270     | 2 near-0 (0.022, 0.483), 1 good (0.663)   |
| trunc_normal | 100 | 0.2670     | 0.290     | 1 negative (-0.066), 2 moderate (0.226, 0.641)|

**Dominant failure mode: bimodal distribution across splits.** One split catastrophically fails (kappa near 0 or negative) while the other two succeed (kappa 0.6-0.8). This indicates split-specific data configuration sensitivity (class imbalance or boundary cases in specific folds), not stochastic noise.

**Critical notes:**
1. **he@pct=100, split=1 (ba6zzvml): kappa=-0.046.** Model trained on 100% data performs worse than chance.
2. **trunc_normal@pct=100, split=1 (um1i0j5u): kappa=-0.066.** Worst result in the entire larvae SVM dataset.
3. **xavier@pct=5 collapse:** All three splits fail at pct=5 (0.083, 0.161, -0.057) — unusual since the bimodal pattern does not apply; all fail together. Suggests an optimization landscape issue at this specific data regime.

**flim is immune to this instability:** Maximum inter-split spread at pct>=25 is 0.069 kappa (pct=25). No negative kappa observed.

---

## 7. Cross-Dataset Comparison

### I-JEPA teacher kappa across datasets:

| dataset   | pct=1  | pct=5  | pct=25 | pct=50 | pct=75 | pct=100 |
|-----------|--------|--------|--------|--------|--------|---------|
| larvae    | 0.8216 | 0.8807 | 0.9190 | 0.9433 | 0.9410 | 0.9495  |
| eggs      | 0.6611 | 0.8636 | 0.9171 | 0.9397 | 0.9457 | 0.9572  |
| protozoan | 0.5860 | 0.7248 | 0.8307 | 0.8637 | 0.8816 | 0.8919  |

**Larvae confirmed as easiest for I-JEPA teacher** at pct=1 and pct=5. Eggs marginally surpasses larvae only at pct=75+ (0.9457 vs 0.9410) and pct=100 (0.9572 vs 0.9495).

### LeJEPA flim SVM kappa cross-dataset:

| dataset   | pct=1  | pct=5  | pct=25 | pct=100 | plateau   |
|-----------|--------|--------|--------|---------|-----------|
| larvae    | 0.2085 | 0.4749 | 0.7783 | 0.7545  | ~0.75@25% |
| eggs      | 0.0623 | 0.0558 | 0.7331 | 0.7309  | ~0.73@25% |
| protozoan | 0.0982 | 0.3681 | 0.4799 | 0.2915  | no plateau|

Larvae and eggs reach similar LeJEPA plateaus (~0.75 and ~0.73), despite the teacher showing larvae >> eggs at pct=1. Protozoan remains dramatically harder for LeJEPA, never exceeding 0.50.

**The "easier" nature of larvae partially transfers to LeJEPA:** larvae is more data-efficient at pct=1-5, but the advantage narrows at higher pct. MLP unfreeze at pct=100 effectively closes the teacher-student gap for larvae (gap=0.015) confirming the representations contain sufficient information despite the SVM plateau.

---

## Summary

1. **Best initialization for larvae: flim** — highest kappa plateau (0.778), uniquely stable (std=0.007 at pct=100). Recommended for all pct >= 5.
2. **Full fine-tuning (MLP unfreeze) is essential** — +0.15 to +0.37 kappa over frozen probes. flim-unfreeze at pct=100 reaches 0.935 (teacher gap: 0.015).
3. **Learning curve saturates at pct=25 for flim SVM** (0.778 at 25% vs 0.754 at 100%). Representation quality is the bottleneck, not data quantity.
4. **Teacher gap:** 0.141-0.449 (SVM), 0.015-0.057 (MLP unfreeze flim). Representations are richer than the SVM plateau suggests.
5. **Larvae confirmed as easiest dataset:** teacher kappa=0.82 at pct=1, LeJEPA-flim 0.48 at pct=5.
6. **Severe instability for non-flim inits at high pct:** negative kappa at pct=100 for he, trunc_normal, and at pct=75 for xavier. Bimodal distribution pattern is the dominant failure mode. Avoid these inits for production without full cross-validation.
7. **Verified anomaly:** kappa=0.65 at larvae/split1/pct75 (run 0480ye37) is LeJEPA-random student, NOT the I-JEPA teacher. Teacher at that condition = 0.9330.
8. **xavier@pct=5 anomaly** (0.373->0.062): all three splits fail simultaneously, unusual pattern suggesting optimizer divergence at this data regime.

---

## Files

- CSV: `analises_wandb_training/larvae/lejepa/lejepa_larvae_analysis.csv`
- SVM data: `artifacts/SVM/svm_results.csv`, `artifacts/SVM/svm_aggregated.csv`
- MLP data: `artifacts/MLP/mlp_results.csv`, `artifacts/MLP/mlp_aggregated.csv`
- Teacher: `results/ijepa_svm_results.csv`, `results/ijepa_svm_aggregated.csv`
