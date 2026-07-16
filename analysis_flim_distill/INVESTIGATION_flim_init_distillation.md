# Investigation — Does FLIM initialization help I‑JEPA → FLIM‑CNN distillation?

*Author: analysis run on 2026‑06‑05. Scope: distillation of a frozen I‑JEPA ViT‑H/14 teacher into a 3‑layer FLIM‑CNN student (48‑dim GAP embedding) + a small conv projection head. Compares two student initializations — `flim` (data‑driven FLIM kernels) vs `trunc_normal` (random) — at two projection‑head capacities: **126k** (one‑layer 1×1) and **400k** (two‑layer 48→256→1280).*

---

> ## ⚠️ ADDENDUM (2026‑06‑05, later same day) — supersedes the "FLIM hurts at 400k" claim below
>
> A follow‑up controlled re‑evaluation (`distill_destroys_flim.py`, `ruler_mismatch.py`) revealed that the original "FLIM loses at 400k (Δκ −0.08)" finding was a **measurement artifact: the two CSVs used different rulers.** The FLIM 400k results (`svm_2l_1x1_init_flim_256_1280_results.csv`) were scored on the **48‑dim encoder** (`method=SVM_Distillation_Conv`, embed_dim=48); the random/trunc 400k results (`svm_proj1280_2l_1x1_BN2d_256_1280_results.csv`) were scored on the **1280‑dim projection head** (`method=SVM_Distill_Proj1280`, embed_dim=1280). Comparing them is apples‑to‑oranges.
>
> **Re‑scored on the SAME ruler (split 1, pct 100, κ):**
>
> | ruler | eggs flim/trunc 126k | eggs flim/trunc 400k | larvae flim/trunc 126k | larvae flim/trunc 400k |
> |---|---|---|---|---|
> | **48‑dim encoder** | 0.481 / 0.000 | **0.715 / 0.000** | 0.682 / 0.000 | **0.899 / 0.592** |
> | **1280‑dim projection** | 0.673 / 0.252 | **0.889 / 0.832** | −0.152 / −0.213 | **0.925 / 0.879** |
>
> **On either consistent ruler, FLIM‑init ≥ random at every scale — FLIM never actually hurts.** The random encoder's 48‑dim output is essentially useless (κ≈0); only its projection head is good. So "random beats FLIM at 400k" was FLIM's honest encoder score vs random's head score.
>
> **Why standalone FLIM still looks higher than "distilled FLIM" in the user's tables** (the real question): two compounding reasons, quantified on the 48‑dim encoder —
> 1. **Normalization bug** (see §7/Cause 4): feeding FLIM the distillation pipeline's input (ImageNet‑RGB `Normalize` over LAB, per‑marker `conv*-mean/stdev.txt` never applied) drops standalone FLIM *before any training*: eggs 0.748→0.556, protozoan 0.685→0.330 (larvae robust, 0.818→0.813).
> 2. **Distillation overwrites the deep discriminative filters**: conv1 survives (cosine≈1.0) but conv3 is largely rewritten (cosine eggs 0.34 / protozoan 0.19 at 400k). The encoder trades class‑discriminability for teacher‑imitation in the 48‑dim bottleneck.
>
> **But "more params makes it worse" is backwards:** FLIM distilled at **400k beats 126k on every dataset** (eggs 0.715>0.481, larvae 0.899>0.682, proto 0.625>0.522) — a bigger head shields the encoder, preserving FLIM structure. And read at the **projection** (1280‑dim), FLIM‑init 400k distillation **matches/exceeds standalone FLIM** (eggs 0.889 vs 0.885; larvae 0.925 vs 0.868). So distillation is *not* destroying a good model when measured consistently; it recovers standalone quality and FLIM init beats random throughout.
>
> Data: `analysis_flim_distill/distill_destroys_flim.json`, `analysis_flim_distill/ruler_mismatch.json`. The sections below are the original (pre‑correction) analysis; read them with this addendum in mind — specifically, the "§3b 400k FLIM hurts" table mixes rulers and should not be read as FLIM underperforming.

## TL;DR — the headline is the opposite of "FLIM doesn't help"

**FLIM initialization _does_ help — strongly — but only when the projection head is small (126k). It stops helping, and slightly hurts, when the head is large (400k).** This is a clean, explainable result, not a failure:

| Scale (proj head) | FLIM vs random (SVM κ) | Verdict |
|---|---|---|
| **126k** (1×1, ~123k params) | FLIM wins **18/18** cells, avg **Δκ = +0.34** | FLIM clearly helps |
| **400k** (48→256→1280, ~402k params) | FLIM wins **3/18** cells, avg **Δκ = −0.08** | FLIM slightly hurts |

The mechanism is **projection‑head dominance that scales with head capacity**. A bigger head can map *any* encoder (even a random one) onto the teacher, so the encoder's starting point becomes irrelevant — FLIM's prior gets overwritten. A small head cannot do this alone, so it leans on the encoder, and there FLIM's structured filters prevent the representational collapse that the random init suffers.

I verified this with three independent lines of evidence (downstream SVM κ, embedding‑space geometry, and W&B training dynamics), plus a full code audit that found the FLIM init path is **real and correctly wired**, but uncovered **one genuine confound (input‑normalization mismatch)** and two stale docs.

---

## 1. Experimental setup, decoded

- **"126k" and "400k" are projection‑head sizes, not dataset sizes.** `126k` = the one‑layer `OneLayer1x1ConvDistillationProjectionHead` (48→1280, ≈123k params); `400k` = the two‑layer `TwoLayer1x1ConvBN2dDistillationProjectionHead` (48→256→1280, ≈402k params). The FLIM encoder itself is ~59.5k params in both. (See `scripts/plot_comparison_flim.py` labels and `src/models/distillation.py`.)
- **Student** = `LeJEPAFLIMModel` — 3× (Conv5×5 + ReLU + MaxPool) → GAP → **48‑dim** embedding. That 48‑dim vector is what the downstream SVM uses.
- **Teacher** = frozen I‑JEPA ViT‑H/14, mean‑pooled to 1280‑dim. Loss = MSE between the projected student (1280) and the teacher (1280) in `direct` mode.
- **Inits compared**: `flim` (loads real FLIM conv kernels) vs `trunc_normal` (the random baseline). Grid: 3 datasets (eggs/9‑class, larvae/2‑class, protozoan/7‑class) × 3 splits × 6 pcts × 2 inits × 2 head scales.
- **Eval**: linear SVM, Cohen's κ. Source CSVs: 126k = `results/svm_proj1280_1x1_BN2d_results.csv` (+ `_flim_init` runs); 400k = `results/svm_proj1280_2l_1x1_BN2d_256_1280_results.csv` (trunc) and `results/svm_2l_1x1_init_flim_256_1280_results.csv` (flim).

---

## 2. Is the code *really* initializing with FLIM? — **Yes, verified end‑to‑end**

This was the user's explicit doubt. The answer is **yes** for the current modules:

- `src/modules/distillation_onelayer_module.py:144‑149` and `distillation_twolayer_module.py:145‑150` call `load_FLIM_encoder(...)` / `load_FLIM_encoder_from_arch_dict(...)` **whenever `encoder_init=='flim'`**.
- `load_FLIM_encoder` (`src/models/models.py:299‑339`) reads `conv{n}-kernels.npy` + `conv{n}-bias.txt` and writes them into `model_block[0].weight/.bias` (the Conv2d). Kernel reshaping (`shift_weights`) was checked against the on‑disk shapes: conv1 `(75,24)`→`(24,3,5,5)`, conv2 `(600,32)`→`(32,24,5,5)`, conv3 `(800,48)`→`(48,32,5,5)`. Consistent.
- Channel auto‑detection (`get_actual_channels_from_weights`) reads the real kernel counts from the bias‑file headers, so **protozoan's 30‑channel conv2** (not 32) loads without shape mismatch, and the head is rebuilt to match (`override_arch_channels`).
- The launcher flips exactly one thing between flim and trunc runs: `encoder_init` + the corresponding `flim_weights_path` (`scripts/distillation_conv_ray.py:394,411`). Seed (42), data, splits, pct, transforms, lr (5e‑4), scheduler, weight decay (0.05), epochs (100), warmup (10), batch size, and proj head are **identical**. The comparison is mechanically controlled.

> ⚠️ **Stale documentation found.** `distillation_model_architecture.md:25‑27` states *"o `load_FLIM_encoder()` existe mas **não é chamado** no pipeline de destilação … o que vem do FLIM é apenas a arquitetura, não os pesos."* This is **wrong for the current code** — it predates the `flim_init` experiments. `memory/distillation_validation_report.md:19` ("NOT initialized with FLIM weights") is similarly stale as a general claim. These should be corrected so future‑you doesn't re‑derive the wrong mental model.

---

## 3. Quantitative confirmation of the three observations

### 3a. 126k random < FLIM (FLIM helps) — **confirmed, large effect**

κ delta (flim − trunc), averaged over the 6 pcts, per dataset (126k one‑layer):

| Dataset | avg Δκ (flim − trunc) | cells FLIM wins |
|---|---|---|
| eggs | **+0.31** | 6/6 |
| larvae | **+0.48** | 6/6 |
| protozoan | **+0.24** | 6/6 |
| **all** | **+0.34** | **18/18** |

The gap is widest at low pct (random init is near‑chance, κ≈0, at pct 1–5; FLIM already reaches κ 0.24–0.74). This is the few‑shot regime where a good prior matters most.

### 3b. 400k FLIM ≤ random (FLIM stops helping) — **confirmed, small effect**

κ delta (flim − trunc), 400k two‑layer:

| Dataset | avg Δκ (flim − trunc) | cells FLIM wins |
|---|---|---|
| eggs | **−0.07** (loses at every pct) | 0/6 |
| larvae | **−0.02** (mixed; wins p5/p25) | 2/6 |
| protozoan | **−0.14** (loses at every pct, worse with pct) | 1/6 |
| **all** | **−0.08** | **3/18** |

The bigger head recovers a random encoder and erases FLIM's edge — most decisively on protozoan and eggs.

### 3c. Larvae @ pct=100 high variance — **confirmed, and it is an instability artifact, not a real effect**

Per‑split κ at larvae/pct=100:

| Scale | init | split1 | split2 | split3 | std |
|---|---|---|---|---|---|
| 126k 1‑layer | flim | −0.152 | 0.625 | 0.818 | **0.513** |
| 126k 1‑layer | trunc | −0.213 | 0.050 | 0.765 | **0.506** |
| 400k 2‑layer | flim | 0.899 | **0.116** | 0.872 | **0.444** |
| 400k 2‑layer | trunc | 0.879 | 0.890 | 0.879 | 0.006 |

Larvae's std jumps from ≤0.17 at every other pct to **0.44–0.51 at pct=100**, driven by a *single collapsing split* (split1 at 126k; split2 for 400k‑flim) that produces **negative κ** (worse than chance). Root cause: larvae is a **2‑class, 1:6.9 class‑imbalanced** problem (test n=1757, 223 vs 1534), and the weak 1×1 head occasionally collapses on it. The 400k‑**trunc** cell is perfectly stable (std 0.006), confirming this is head/training instability amplified by imbalance, **not** a degenerate data split. Splits have *identical* class balance, so it is not a data‑split bug.

---

## 4. Why? — Ranked causes, each with a supporting metric

### Cause 1 (primary): **Projection‑head dominance scales with capacity** → encoder init is washed out at 400k

Embedding‑space CKA between the **flim‑trained** and **trunc‑trained** 48‑dim encoders (same eval images, split1, pct=100):

| Dataset | CKA(flim, trunc) @126k | CKA(flim, trunc) @400k |
|---|---|---|
| eggs | **0.30** (very different spaces) | **0.91** (nearly identical) |
| larvae | **0.51** | **0.94** |

At 126k the two inits converge to *different* representations (CKA 0.30/0.51); at 400k they converge to *the same* representation (CKA 0.91/0.94). The 400k head is expressive enough to drive the encoder toward one common, teacher‑aligned solution regardless of where it started — so FLIM's prior cannot express itself. **This is the core explanation for "helps at 126k, hurts at 400k."**

### Cause 2: **FLIM's prior buys class separability and prevents collapse — but only the small head needs it**

48‑dim encoder geometry (split1, pct=100):

| Cell | effective rank (flim / trunc) | Fisher ratio (flim / trunc) | silhouette (flim / trunc) |
|---|---|---|---|
| eggs 126k | **1.74 / 1.00** | 0.33 / 0.39 | −0.28 / −0.46 |
| larvae 126k | **1.00 / 0.025** | **0.78 / 0.41** | **0.52 / 0.40** |
| eggs 400k | 1.21 / 1.16 | 0.35 / 0.35 | −0.31 / −0.32 |
| larvae 400k | 1.24 / 1.36 | 0.61 / 0.52 | 0.51 / 0.46 |

At **126k**, the random‑init encoder **collapses**: larvae‑trunc has an *effective rank of 0.025* (essentially a single direction; 99% of variance in 1 PCA component), and eggs‑trunc collapses to ≈1.0. FLIM stays full‑rank‑ish (1.0–1.74) and markedly more separable (larvae Fisher 0.78 vs 0.41, silhouette 0.52 vs 0.40). **This supports the user's intuition that FLIM encodes a good separability pattern** — and shows the random encoder, with a weak head, fails to build one. At **400k** the gap closes (ranks and Fisher near‑parity) because the head, not the encoder, now carries separability.

### Cause 3: **FLIM ↔ I‑JEPA representational mismatch** (the "high norm, low cosine" paradox)

From `analises_wandb_training/section_flim_init_comparison.md` (training‑time, projection space):

- FLIM init reaches **170× larger** embedding norm (11.25 vs 0.066 at epoch 99) yet **lower** cosine‑to‑teacher (0.571 vs 0.637) and slightly higher MSE loss (0.241 vs 0.229).
- FLIM kernels are local segmentation filters; the teacher is a global‑attention ViT. They fire strongly (high norm) but in directions ~orthogonal to the teacher's — a local minimum the optimizer doesn't escape in 100 epochs.

Important nuance: **better cosine‑to‑teacher ≠ better downstream κ.** The teacher's 1280‑dim space is far richer than the 48‑dim bottleneck can mirror, so MSE/cosine is a loose proxy. That's why trunc "wins" on cosine yet **loses** badly on 126k SVM κ — the metrics disagree, and κ is the one that matters.

### Cause 4 (real confound / likely bug): **Input‑normalization mismatch starves the FLIM kernels**

The FLIM kernels were computed on **LABNorm2 patches in [0,1]**, with per‑layer marker normalization `(x − conv{n}-mean)/conv{n}-stdev`. In the distillation pipeline:

- `ift_lab_loader` correctly returns LAB in [0,1] (`src/data_modules/datasets/dataset.py:41‑47`), **but** the transform then applies `v2.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])` — **ImageNet RGB stats** — on top of the LAB values (`src/data_modules/datasets/lejepa_dataset.py:41,52`).
- The per‑marker files **`conv{n}-mean.txt` / `conv{n}-stdev.txt` exist** in the FLIM model dir (verified) but are **never loaded** anywhere (`grep` for them in `src/` returns nothing; `load_FLIM_encoder` reads only bias + kernels).

So FLIM kernels receive input that is (a) shifted off their calibrated [0,1] LAB range and (b) missing the per‑marker normalization they were built for. **This is the direct cause of the emb_norm≈270 explosion at epoch 0** and forces FLIM to spend the critical first ~10 epochs re‑normalizing instead of aligning — handicapping it relative to a random encoder that has no scale expectation. This is a confound: part of FLIM's "loss" at 400k (and its cosine deficit) may be an artifact of feeding it mis‑scaled input, not an intrinsic limitation of the FLIM prior.

### Cause 5: **larvae instability** (Section 3c) — weak head + 1:6.9 imbalance → occasional negative‑κ split → inflated variance at pct=100.

---

## 5. On "FLIM doesn't adapt well during distillation"

The data refines this. **Representational drift from init** (CKA between the untrained init encoder and the trained encoder; low = moved a lot):

| Cell | drift CKA flim | drift CKA trunc |
|---|---|---|
| eggs 126k | **0.36** (moves a lot) | 0.81 (barely moves) |
| eggs 400k | 0.55 | 0.81 |
| larvae 126k | 0.57 | 0.56 |
| larvae 400k | 0.75 | 0.81 |

For eggs, **FLIM actually adapts _more_ than the random init**, which stays near its (collapsed) starting point. So "FLIM fails to adapt" is not the right frame in the encoder space at 126k — there FLIM both adapts and helps. The accurate statement is:

- **In projection/cosine space** (the teacher‑matching objective) FLIM gets *stuck* in a local minimum (Cause 3) — that is the sense in which it "doesn't adapt well."
- **In the 400k regime** the head adapts so much that the encoder doesn't need to, so FLIM's prior is *overwritten/irrelevant* (Cause 1) — the other sense of "not helping."
- **In the 126k encoder** FLIM is exactly where it shines.

---

## 6. Embedding‑space evidence (PCA, t‑SNE, CKA, rank, correlation)

Full numbers: [`embedding_metrics.json`](embedding_metrics.json). 2D projections (flim vs trunc side‑by‑side, colored by class): [`plots/`](plots/) — e.g. [`larvae_126k_tsne.png`](plots/larvae_126k_tsne.png), [`eggs_126k_pca.png`](plots/eggs_126k_pca.png). Script: [`embedding_analysis.py`](embedding_analysis.py).

Summary of what the geometry shows:
- **Two spaces, not one, at 126k** (CKA 0.30/0.51) → one space at 400k (CKA 0.91/0.94).
- **Random init collapses at 126k** (larvae effective rank 0.025; eggs ≈1.0) while FLIM stays structured and more separable (Fisher, silhouette).
- **Feature redundancy** (mean |off‑diag corr|): at 126k FLIM is more diverse where it matters (larvae 0.25 vs 0.47 — random's features are more redundant/collapsed).
- *Caveat:* these are single‑split (split1), single‑seed snapshots on the test set; treat magnitudes as indicative, the direction of every effect agrees with the 3‑split SVM κ.

---

## 7. Repository correctness audit

| Severity | Finding | Where |
|---|---|---|
| **Major (confound)** | **Input‑normalization mismatch**: FLIM kernels fed ImageNet‑RGB‑normalized LAB; per‑marker `conv*-mean/stdev.txt` never applied. Causes emb_norm≈270 and disadvantages FLIM. | `lejepa_dataset.py:41,52`; `dataset.py:41‑47`; `models.py:299‑339` |
| **Major (method)** | `svm_distill_with_projection.py` evaluates the **1280‑dim projection output**, not the 48‑dim encoder; `svm_distillation_conv.py` evaluates the **48‑dim encoder**. Both agree FLIM helps@126k/hurts@400k, but init conclusions should cite the encoder‑level eval. | `svm_distill_with_projection.py:179,211`; `svm_distillation_conv.py:57‑101` |
| **Major (docs)** | Stale docs deny FLIM weights are loaded (now false). | `distillation_model_architecture.md:25‑27,51,199,230`; `memory/distillation_validation_report.md:19` |
| **Minor (naming)** | `init_weights_trunc_normal` only touches `nn.Linear`; the encoder is all Conv2d, so the "trunc_normal"/"random" baseline conv weights are actually **Kaiming‑uniform(a=√5)** (PyTorch default). The label is a misnomer; the baseline is still a valid random init. `encoder_init='random'` has no branch and is identical to it. | `models.py:450‑498`; `distillation_onelayer_module.py:138‑148` |
| **Minor** | 10‑epoch warmup may be too short for FLIM to renormalize before the BN head over‑adapts (secondary to Cause 4). | `distillation_onelayer_module.py:256‑264` |
| **Known** | Checkpoints are ~2.5 GB because the frozen teacher is saved in the state_dict (see `memory/project_checkpoint_teacher_bloat.md`). Not a correctness issue, but a disk one. | `distillation_*_module.py` |
| **OK** | FLIM weight loading, channel detection (incl. protozoan 30‑ch), best‑checkpoint (min val/loss) selection, and flim‑vs‑trunc training fairness all check out. | see §2 |

**Verdict:** the experiment is *mechanically* controlled and the FLIM‑helps@126k result is robust across three metrics. But the **normalization mismatch** is a real confound — FLIM may be losing at 400k partly because it is fed input it was never calibrated for, not purely because the prior is useless there. Fix it before treating "FLIM ≈ random at 400k" as a final scientific result.

---

## 8. Recommendations / next experiments

1. **Fix the normalization path (highest priority).** For `ift_lab` inputs, drop the ImageNet `v2.Normalize`, and either (a) apply the per‑marker `conv{n}-mean/stdev.txt` between conv layers as FLIM training did, or (b) at minimum keep inputs in the calibrated [0,1] LAB range. Then re‑run a small grid (eggs+larvae, split1, pct∈{5,100}, both scales). Hypothesis: this narrows or closes FLIM's 400k deficit and further widens its 126k lead.
2. **Report the 48‑dim encoder eval (`svm_distillation_conv.py`) as the headline for init claims**, not the 1280‑dim projection eval — it isolates what the encoder actually learned.
3. **Decouple head capacity from the story.** Sweep head sizes (1×1 / 2‑layer / 3×3) × init to draw the "FLIM advantage vs head capacity" curve directly. Predict: advantage decays monotonically as head params grow.
4. **Stabilize larvae.** Use balanced/stratified SVM or class‑weighted training and report balanced accuracy; the negative‑κ splits are an imbalance artifact, not signal.
5. **Add an auxiliary loss on the 48‑dim bottleneck** (contrastive/clustering) so the encoder is forced to learn discriminative features even when a large head could otherwise do all the work — this is the way to make FLIM's prior matter even at 400k.
6. **Correct the stale docs** (`distillation_model_architecture.md`, `memory/distillation_validation_report.md`).

---

## Appendix — reproduction & sources

- **SVM κ tables**: `results/svm_proj1280_1x1_BN2d_results.csv`, `results/svm_proj1280_2l_1x1_BN2d_256_1280_results.csv`, `results/svm_2l_1x1_init_flim_256_1280_results.csv`; teacher upper bound `results/ijepa_svm_results.csv`; FLIM‑CNN baseline `artifacts/normalized/svm_flim_aggregated.csv`.
- **Embedding metrics**: `analysis_flim_distill/embedding_metrics.json` (+ `plots/`, `embedding_analysis.py`, `run.log`).
- **Training dynamics**: `analises_wandb_training/section_flim_init_comparison.md`, `.../section_lejepa_larvae.md`.
- **Code**: `src/models/models.py` (FLIM load + inits), `src/modules/distillation_{onelayer,twolayer}_module.py`, `scripts/distillation_conv_ray.py`, `src/evaluate/svm_distill_with_projection.py`, `src/evaluate/svm_distillation_conv.py`, `src/data_modules/datasets/{dataset,lejepa_dataset}.py`.
- Context: I‑JEPA upper bound (eggs κ 0.957 / larvae 0.950 / protozoan 0.892) and FLIM‑CNN baseline (0.885/0.868/0.847) both exceed *every* distilled student, i.e. distilling into the 48‑dim bottleneck does not yet recover teacher quality regardless of init.
