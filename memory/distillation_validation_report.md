# Distillation Experiment Validation Report
## I-JEPA → FLIM CNN (next_layers_direct)
**Generated:** 2026-05-17

---

## 1. Experimental Setup

### 1.1 Teacher Network
- **Model:** I-JEPA ViT-H/14 (`facebook/ijepa_vith14_1k`)
- **Parameters:** 630,762,240
- **Status during distillation:** Permanently frozen (`requires_grad=False`, `eval()` enforced)
- **Input:** `[B, 3, 224, 224]` — student images resized via bilinear interpolation
- **Output:** `[B, 1280]` — mean-pooled patch embeddings (32 Transformer blocks, 16 heads)
- **Weights source:** Local HuggingFace cache via `safetensors`

### 1.2 Student Network
- **Architecture:** Same FLIM CNN architecture used per dataset (defined by `architecture.json`)
- **Initialization:** `trunc_normal` (timm ViT-style, std=0.02) — **NOT initialized with FLIM weights**
- **Parameters:** 59,504 (encoder only)

| Layer | Operation | Output shape |
|---|---|---|
| conv1 | Conv2d(3→24, 5×5, pad=2) + ReLU + MaxPool(3×3, s=2) | `[B, 24, 99, 99]` |
| conv2 | Conv2d(24→32, 5×5, pad=2) + ReLU + MaxPool(3×3, s=2) | `[B, 32, 49, 49]` |
| conv3 | Conv2d(32→48, 5×5, pad=2) + ReLU + MaxPool(3×3, s=2) | `[B, 48, 24, 24]` |
| pool  | GlobalAvgPool2d(1) + flatten | `[B, 48]` |

Architecture JSON per dataset:
- `eggs`, `larvae`: `data/to_mateus/model/ch24_32_48_a0.5_f5/<dataset>/train<N>/architecture.json`
- `protozoan`: `data/to_mateus/model/ch24_30_48_a0.5_f5/protozoan/train<N>/architecture.json`

### 1.3 Projection Head (ConvDistillationProjectionHead)
Convolutional layers introduced to align student output to teacher dimensionality:

| Layer | Operation | Output shape |
|---|---|---|
| proj[0] | Conv2d(48→128, 1×1, no bias) + BN2d + ReLU | `[B, 128, 24, 24]` |
| proj[1] | Conv2d(128→256, 1×1, no bias) + BN2d + ReLU | `[B, 256, 24, 24]` |
| proj[2] | Conv2d(256→512, 1×1, no bias) + BN2d + ReLU | `[B, 512, 24, 24]` |
| proj[3] | Conv2d(512→1280, 1×1, no bias) + BN2d | `[B, 1280, 24, 24]` |
| pool   | AdaptiveAvgPool2d(1) + flatten | `[B, 1280]` |

**Parameters:** 829,696  
**Total student + proj:** 889,200 params

### 1.4 Loss Function
```
L = MSE( proj_kd(encoder(x)), teacher_emb(resize(x)) )
  = mean( (student_proj - teacher_emb)² )
```
- **Class:** `MSEDistillationLoss` (`src/models/distillation.py`)
- **Mode:** `direct` — no SSL auxiliary loss, only distillation
- **Gradient flow:** through `proj_kd` → `student.encoder` only; teacher is frozen

### 1.5 Optimizer & Schedule
| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Peak LR | 5e-4 |
| Weight decay | 5e-2 |
| Warmup | 10 epochs (linear, start_factor=0.01) |
| Schedule | CosineAnnealingLR after warmup |
| Max epochs | 100 |
| Batch size | 32 |

---

## 2. Data Splits

### 2.1 Datasets
| Dataset | Classes | Architecture |
|---|---|---|
| Helminth Eggs | 9 | ch24_32_48 |
| Helminth Larvae | 2 | ch24_32_48 |
| Protozoan Cysts | 7 | ch24_30_48 |

### 2.2 Split and Percentage Configurations
- **Splits used:** 1, 2, 3 (pre-defined in repository)
- **Percentages used:** 1%, 5%, 25%, 50%, 75%, 100%
- **Split JSON source:** `data/to_modules/new_split_parasito/<dataset>/splits_incremental/split<N>/data_descriptor_perc<P>.json`
- **Status:** All 54 split JSON files present and validated ✓
- **DataModule:** `ParasiteLejepaDataModuleSplited` — same train/val/test splits used across all experiments
- **Train/val/test:** defined by the split JSON files already in the repository (not modified)

### 2.3 Grid
```
3 datasets × 3 splits × 6 percentages × 1 type (direct) = 54 experiments
```

---

## 3. Experiment Status

### 3.1 Overall Summary
| Status | Count |
|---|---|
| **Completed (local)** | **54 / 54** |
| W&B `finished` | 49 |
| W&B `crashed` + local `ok` | 5 |
| W&B `failed` (OOM, retried) | 25 (all re-ran successfully) |

**All 54 experiments completed training successfully.** All have checkpoints saved locally.

### 3.2 Helminth Eggs — All Completed ✓

| Split | Pct | Best Epoch | Best val/loss | Elapsed |
|---|---|---|---|---|
| 1 | 1% | 69 | 0.3443 | 585.5 min |
| 1 | 5% | 80 | 0.2870 | 591.8 min |
| 1 | 25% | 95 | 0.1621 | 595.0 min |
| 1 | 50% | 88 | 0.0905 | 608.4 min |
| 1 | 75% | 95 | 0.0557 | 606.6 min |
| 1 | 100% | 93 | 0.0382 | 1036.1 min |
| 2 | 1% | 69 | 0.3396 | 590.4 min |
| 2 | 5% | 60 | 0.2806 | 591.9 min |
| 2 | 25% | 93 | 0.1599 | 594.9 min |
| 2 | 50% | 95 | 0.0915 | 604.1 min |
| 2 | 75% | 95 | 0.0568 | 604.8 min |
| 2 | 100% | 86 | 0.0376 | 1036.4 min |
| 3 | 1% | 81 | 0.3390 | 592.7 min |
| 3 | 5% | 80 | 0.2781 | 590.8 min |
| 3 | 25% | 82 | 0.1607 | 591.3 min |
| 3 | 50% | 84 | 0.0917 | 605.6 min |
| 3 | 75% | 95 | 0.0556 | 605.8 min |
| 3 | 100% | 93 | 0.0374 | 1033.7 min |

### 3.3 Helminth Larvae — All Completed ✓

| Split | Pct | Best Epoch | Best val/loss | Elapsed |
|---|---|---|---|---|
| 1 | 1% | 77 | 0.3570 | 390.9 min |
| 1 | 5% | 67 | 0.3112 | 406.3 min |
| 1 | 25% | 93 | 0.2057 | 336.1 min |
| 1 | 50% | 88 | 0.1329 | 252.3 min |
| 1 | 75% | 95 | 0.0907 | 222.4 min |
| 1 | 100% | 95 | 0.0657 | 362.5 min |
| 2 | 1% | 77 | 0.3553 | 310.0 min |
| 2 | 5% | 85 | 0.3131 | 254.2 min |
| 2 | 25% | 88 | 0.2075 | 136.5 min |
| 2 | 50% | 93 | 0.1348 | 325.6 min |
| 2 | 75% | 84 | 0.0912 | 162.0 min |
| 2 | 100% | 97 | 0.0656 | 530.9 min |
| 3 | 1% | 98 | 0.3556 | 255.3 min |
| 3 | 5% | 67 | 0.3078 | 147.4 min |
| 3 | 25% | 81 | 0.2056 | 153.5 min |
| 3 | 50% | 89 | 0.1347 | 130.2 min |
| 3 | 75% | 95 | 0.0921 | 133.7 min |
| 3 | 100% | 99 | 0.0657 | 460.6 min |

### 3.4 Protozoan Cysts — All Completed ✓

| Split | Pct | Best Epoch | Best val/loss | Elapsed |
|---|---|---|---|---|
| 1 | 1% | 59 | 0.3072 | 554.4 min |
| 1 | 5% | 84 | 0.2331 | 354.5 min |
| 1 | 25% | 96 | 0.0926 | 372.8 min |
| 1 | 50% | 99 | 0.0378 | 349.1 min |
| 1 | 75% | 98 | 0.0199 | 376.2 min |
| 1 | 100% | 98 | 0.0122 | 700.6 min |
| 2 | 1% | 71 | 0.3097 | 359.9 min |
| 2 | 5% | 89 | 0.2328 | 351.9 min |
| 2 | 25% | 96 | 0.0927 | 367.5 min |
| 2 | 50% | 99 | 0.0376 | 365.4 min |
| 2 | 75% | 92 | 0.0199 | 359.7 min |
| 2 | 100% | 92 | 0.0123 | 626.8 min |
| 3 | 1% | 71 | 0.3090 | 329.3 min |
| 3 | 5% | 84 | 0.2323 | 331.1 min |
| 3 | 25% | 96 | 0.0905 | 334.1 min |
| 3 | 50% | 86 | 0.0376 | 339.8 min |
| 3 | 75% | 98 | 0.0198 | 336.0 min |
| 3 | 100% | 98 | 0.0120 | 310.0 min |

---

## 4. W&B Logging Incidents

W&B logging crashed for 5 runs after the training session was terminated while the W&B `finish()` call was still in progress. **Training completed successfully in all cases** — local checkpoints and `run_metadata.json` with `status=ok` are present.

| Run | W&B ID | W&B State | Local Status | Reason |
|---|---|---|---|---|
| `eggs_split1_pct100` | `03et3x6r` | crashed | ok | tmux session killed before `wandb.finish()` |
| `eggs_split2_pct100` | `uwi04k6s` | crashed | ok | tmux session killed before `wandb.finish()` |
| `eggs_split3_pct100` | `q3xclzge` | crashed | ok | tmux session killed before `wandb.finish()` |
| `larvae_split1_pct50` | `cq3od8ph` | crashed | ok | tmux session killed before `wandb.finish()` |
| `larvae_split1_pct75` | `a0uci292` | crashed | ok | tmux session killed before `wandb.finish()` |

Additionally, 25 W&B runs show `failed` state with ~0.2 min runtime — these were the original OOM failures from the first run attempt (before fixing `--max-concurrent-per-gpu`). All were successfully re-run and completed.

**Root cause of OOM failures:** Running `--max-concurrent-per-gpu 3` caused 3 concurrent I-JEPA ViT-H/14 instances per GPU (~5 GB each), exhausting the 44 GB VRAM. Fixed by setting `--max-concurrent-per-gpu 1`.

---

## 5. Artifacts and Logs

### 5.1 Checkpoint Storage
- **Location:** `artifacts/distillation/<run_name>/checkpoints/`
- **Format:** `best-epoch=<N>-val/loss=<L>.ckpt` (directory due to `/` in filename)
- **All 54 experiments have `last.ckpt` saved** ✓
- **Best checkpoint selection:** minimum `val/loss` from filename via recursive glob

### 5.2 Metadata Storage
- **Location:** `artifacts/distillation/<run_name>/run_metadata.json`
- **Contains:** dataset, split, percentage, arch_json, encoder_init, teacher_model, student_embed_dim, max_epochs, best_checkpoint path, elapsed_s, git_sha, timestamp
- **All 54 present and valid** ✓

### 5.3 Manifest
- `artifacts/distillation/run_manifest_conv.csv` — full run grid status
- `artifacts/distillation/run_manifest_conv_retry.csv` — retry verification log

### 5.4 Training Logs
- **Location:** `logs/distill_*.log`
- **Count:** 13 log files covering initial run and retry sessions

---

## 6. SVM Evaluation Results

### 6.1 Evaluation Strategy 1 — Backbone Only [B, 48]
Script: `src/evaluate/svm_distillation_conv.py`
CSV: `results/svm_distillation_conv_results.csv`
- Projection head removed; only `student.encode()` → `[B, 48]` used
- SVM: linear kernel, C=100, max_iter=10,000

| Dataset | Mean Kappa (all pcts) | Kappa at pct=100 |
|---|---|---|
| Eggs (9 classes) | 0.0125 ± 0.031 | 0.073 |
| Larvae (2 classes) | 0.5259 ± 0.387 | 0.850 |
| Protozoan (7 classes) | 0.1124 ± 0.117 | 0.245 |

### 6.2 Evaluation Strategy 2 — With Projection Head [B, 1280]
Script: `src/evaluate/svm_distill_with_projection.py`
CSV: `results/svm_distill_proj1280_results.csv`
- Full pipeline: `student.encoder` + `proj_kd` → `[B, 1280]`
- SVM: `StandardScaler` + linear kernel, C=100, max_iter=20,000

| Dataset | Mean Kappa (all pcts) | Kappa at pct=100 |
|---|---|---|
| Eggs (9 classes) | 0.7384 ± 0.241 | 0.907 |
| Larvae (2 classes) | 0.8330 ± 0.117 | 0.923 |
| Protozoan (7 classes) | 0.7060 ± 0.174 | 0.859 |

### 6.3 Comparison at pct=100 (mean across 3 splits)

| Method | Embedding | Eggs κ | Larvae κ | Protozoan κ |
|---|---|---|---|---|
| FLIM (supervised) | — | 0.885 | 0.868 | 0.847 |
| LeJEPA trunc_normal | SSL [B,48] | 0.314 | 0.267 | — |
| I-JEPA (teacher) | [B,1280] | **0.957** | **0.950** | **0.892** |
| Distil-Conv [48] | encoder only | 0.073 | 0.850 | 0.245 |
| **Distil-Conv+Proj [1280]** | **full pipeline** | **0.907** | **0.923** | **0.859** |

---

## 7. Key Finding: Projection Head Dominance

The `ConvDistillationProjectionHead` (48→1280 via 1×1 convs) is sufficiently powerful to absorb the representational learning during distillation. Evidence:

- **Inverse correlation:** lower MSE loss → worse backbone classification
  - Protozoan: val/loss=0.012 (best MSE) → kappa=0.245 (backbone only)
  - Larvae: val/loss=0.066 (worst MSE) → kappa=0.850 (backbone only)
- **Recovery with projection:** adding `proj_kd` back to evaluation recovers performance
  - Eggs at pct=100: 0.073 (backbone) → 0.907 (backbone + proj)

The backbone [B,48] does not learn discriminative features for multi-class datasets (9 classes eggs, 7 classes protozoan) — the projection head does all the mapping work. For binary larvae (2 classes), even generic features suffice.

---

## 8. File Reference

| File | Purpose |
|---|---|
| `src/modules/distillation_conv_module.py` | Lightning training module |
| `src/models/distillation.py` | Model components (proj head, loss, teacher) |
| `src/models/ijepa_encoder.py` | I-JEPA ViT-H/14 implementation |
| `src/evaluate/svm_distillation_conv.py` | SVM evaluation — backbone [B,48] |
| `src/evaluate/svm_distill_with_projection.py` | SVM evaluation — full pipeline [B,1280] |
| `scripts/distillation_conv_ray.py` | Ray launcher (--retry, --skip-existing, --check-wandb) |
| `scripts/check_distill_conv_status.py` | Status checker (4 sources: process, W&B, metadata, ckpt) |
| `distillation_model_architecture.md` | Layer-by-layer architecture documentation |
| `artifacts/distillation/run_manifest_conv.csv` | Experiment manifest |
| `results/svm_distillation_conv_results.csv` | SVM results [B,48] |
| `results/svm_distill_proj1280_results.csv` | SVM results [B,1280] |
