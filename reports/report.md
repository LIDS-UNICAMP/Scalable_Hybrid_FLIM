# SDD Verification Report: LeJEPA SSL → Fine-Tune → Evaluation Pipeline

---

## 13.1 Experiment Matrix Audit

**Verdict: Partially Compliant**

| Dimension | Intended | Found | Match? |
|-----------|----------|-------|--------|
| Datasets | protozoan, eggs, larvae | `protozoan-cysts`, `helminth-eggs`, `helminth-larvae` (+ legacy `parasito`) | ✅ |
| Splits | 3 per dataset | 3 per dataset (split_1, split_2, split_3) | ✅ |
| Percentages | 1, 5, 25, 50, 75, 100 | 1, 5, 25, 50, 75, 100 | ✅ |
| Initializers | 4 | `xavier`, `random`, `he`, `flim` | ✅ |
| Total canonical | 216 | 216 (3 × 3 × 6 × 4) + legacy parasito | ✅ |
| Config YAML coverage | 432 (216 × 2 modes) | 644 YAML files present (includes legacy parasito) | ✅ |

**Note**: `helminth-larvae` is present in configs but was not mentioned in the spec's dataset list ("protozoan, eggs, larvae"). The repository **correctly includes it**.

---

## 13.2 Naming Audit

**Verdict: Compliant**

The canonical naming pattern `lejepa_line_{dataset}_split_{N}_pct_{pct}_model_{init}` is implemented and enforced:

- **Parser**: `src/utils/evaluate.py` — `parse_experiment_name()` uses a strict regex:
  ```
  ^lejepa_line_([a-z\-]+)_split_(\d+)_pct_(\d+)_model_(xavier|random|he|flim)$
  ```
- **All four fields** (dataset, split, percentage, initializer) are parsed deterministically.
- **YAML configs** preserve all fields as explicit keys: `dataset_name`, `split`, `percentage`, `initialization_type`, `run_id`, `experiment_name`.
- **W&B runs** use the same naming scheme; `get_runs_dict()` maps `run_id → run_name` from the live API.
- **Legacy runs** (`line_p{pct}_{init}`) are handled by a separate branch in `parse_experiment_name()` — mapped to `parasito` dataset, split=1.
- **Finetune W&B names** (`src/utils/get_names_wandb.py`) strip `pct_*` to avoid collision-per-percentage, with run_id suffix appended when two runs share a base name.

**Conclusion**: Naming is stable, deterministic, and consistent across training configs, W&B metadata, YAML generation, and evaluation outputs.

---

## 13.3 SSL Lineage Audit

**Verdict: Compliant**

- Checkpoints are stored at `logs/flim-ssl/{run_id}/checkpoints/best-*/loss=*.ckpt`.
- **297 run directories** confirmed in `logs/flim-ssl/`.
- `find_best_checkpoint(run_id)` (`src/utils/evaluate.py`) deterministically selects the lowest-loss checkpoint from the `best-*/` subdirs, falling back to `last.ckpt`.
- W&B run IDs are the canonical reference: YAML filenames are `{run_id}.yaml`; checkpoint directories are `logs/flim-ssl/{run_id}/`.
- Deduplication by `created_at` is implemented in `get_runs_dict(deduplicate=True)` to handle re-ran experiments.

**Gap**: There is no script that explicitly compares the **expected experiment list** against **actual W&B runs** to report which (dataset, split, pct, init) combinations are missing SSL pretraining. `resolve_available_experiments()` does the intersection of W&B+local but does not enumerate the theoretical full 216-entry grid to flag gaps.

---

## 13.4 Fine-Tune Lineage Audit

**Verdict: Compliant**

- **One-to-one mapping is enforced**: Each YAML config contains exactly one `run_id`, which corresponds to exactly one SSL checkpoint. There is no all-vs-all combination.
- `load_classification_model(run_id, ...)` (`src/evaluate/mlp.py`) loads exclusively the encoder of the specified `run_id` checkpoint.
- The `experiment_name` in the YAML encodes the full identity (dataset, split, pct, init), ensuring each downstream run is an exact descendant of its SSL parent.
- **Auto-detection** of conv2 channels from checkpoint state_dict handles the protozoan-cysts FLIM edge case (30 vs 32 channels) without violating lineage.
- Ray run name format: `ray_finetune_lejepa_line_{dataset}_split_{split}_{init}_pct{pct}_{run_id}` is fully traceable.

---

## 13.5 Evaluation Audit

**Verdict: Compliant**

All three downstream evaluation modes are present:

| Mode | Entry Point | Status |
|------|------------|--------|
| **MLP Freeze** | `src/evaluate/mlp.py`, `src/evaluate/ray_mlp.py` | ✅ Implemented, ray-parallel |
| **MLP Unfreeze** | Same, `freeze_encoder=False` | ✅ Implemented |
| **SVM on embeddings** | `src/evaluate/svm.py` | ✅ Implemented |

- All three modes are also orchestrated in the unified pipeline: `src/evaluate/unified_eval.py` (718 lines).
- Both MLP modes share the same `train_and_evaluate()` function with the `frozen` flag controlling which parameters are optimized.
- SVM uses frozen encoder embeddings (no gradient), consistent with the intended design.

---

## 13.6 Aggregation Audit

**Verdict: Partially Compliant**

**What exists**:
- `aggregate_metrics(scores, return_std=True)` in `src/metrics/classification.py` computes mean ± std over any list of score dicts.
- Aggregated output files exist:
  - `artifacts/SVM/svm_aggregated.csv` — columns: `dataset_short, pretrained_pct, init, n_splits, kappa, kappa_std, acc, acc_std, f1, f1_std`
  - `artifacts/MLP/mlp_aggregated.csv` — same structure + `model_type` (MLP_unfreeze, MLP_freeze)
- Aggregation groups by (dataset, pct, init) and collapses splits into mean ± std. ✅
- Comparison across all 3 modes (SVM, MLP-freeze, MLP-unfreeze) at each percentage level. ✅
- Plotting: `src/evaluate/eval_plotter.py` — `plot_metric_vs_pct()` and `plot_composite()` with all modes overlaid.

**Gap**: **No standalone aggregation script** was found in `scripts/`. The aggregated CSVs in `artifacts/` appear to be generated by `unified_eval.py` or produced externally. The generation pipeline for `artifacts/svm_aggregated.csv` and `artifacts/mlp_aggregated.csv` is not independently auditable from a dedicated script.

---

## 13.7 Artifact Audit

**Verdict: Partially Compliant**

| Artifact | Expected | Found | Persisted? |
|----------|----------|-------|------------|
| SSL checkpoint | `logs/flim-ssl/{run_id}/checkpoints/best-*/loss=*.ckpt` | 297 run dirs with checkpoints | ✅ |
| MLP freeze weights | `results/mlp_weights/freeze/{run_id}/model_best.pth` | Confirmed files exist | ✅ |
| MLP unfreeze weights | `results/mlp_weights/unfreeze/{run_id}/model_best.pth` | Confirmed files exist | ✅ |
| Ray MLP weights | `results/ray_finetune/{ray_run_name}/weights/model_final.pth` | Directory structure present | ✅ |
| MLP metrics | `results/ray_finetune/{ray_run_name}/metrics/test_metrics.json` | Written per experiment | ✅ |
| MLP config snapshot | `results/ray_finetune/{ray_run_name}/config/config.yaml` | Written per experiment | ✅ |
| SVM metrics | `results/svm_results.csv` | Present (193 KB) | ✅ |
| **SVM classifier** | (expected for reuse) | **NOT saved** — only in-memory | ⚠️ |
| **SVM embeddings** | (expected for reuse) | **NOT persisted** | ⚠️ |
| Aggregated results | `artifacts/SVM/svm_aggregated.csv`, `artifacts/MLP/mlp_aggregated.csv` | Present | ✅ |

**Key gap**: The fitted SVM model is never serialized. If inference on new data is needed or results need to be reproduced without re-training, the SVM must be re-fitted from scratch each time.

---

## 13.8 Reuse Audit

**Verdict: Compliant**

The repository follows a strong reuse-first pattern. No significant reimplementation was found.

| Component | Module | Reused By |
|-----------|--------|-----------|
| Checkpoint discovery | `src/utils/evaluate.py::find_best_checkpoint()` | SVM, MLP, Ray MLP, unified_eval |
| Experiment name parsing | `src/utils/evaluate.py::parse_experiment_name()` | All evaluation modules + config gen |
| W&B run fetching | `src/utils/get_names_wandb.py::get_runs_dict()` | Config gen, resolver, MLP |
| Experiment resolution | `src/utils/evaluate.py::resolve_available_experiments()` | SVM, MLP, unified_eval |
| Model loading | `src/evaluate/mlp.py::load_classification_model()` | MLP, Ray MLP, unified_eval |
| Training + evaluation | `src/evaluate/mlp.py::train_and_evaluate()` | MLP, Ray MLP, unified_eval |
| SVM training | `src/utils/evaluate.py::train_svm()` | SVM, unified_eval |
| Feature extraction | `src/utils/evaluate.py::extract_features()` | SVM, unified_eval |
| Metric computation | `src/metrics/classification.py::compute_metrics()` | All evaluation modules |
| Metric aggregation | `src/metrics/classification.py::aggregate_metrics()` | unified_eval |
| Dataset loading | `src/data_modules/datasets/dataset.py::DatasetParasite` | SVM, MLP, Ray MLP, unified_eval |

**Minor duplications** (non-critical):
- Label conversion (`np.argmax(...) + 1` and `y_pred - 1`) appears in two places instead of a shared utility.
- Column ordering for result DataFrames is defined separately in SVM and MLP modules.

---

## 13.9 Gap List

| # | Gap | Severity | Location |
|---|-----|----------|----------|
| G1 | No script compares the theoretical 216-experiment grid against actual W&B runs to report missing SSL pretraining | Medium | `scripts/` — missing |
| G2 | Aggregation generation script not found; `artifacts/*.csv` origin is opaque | Medium | `scripts/` — missing |
| G3 | SVM classifier not serialized to disk; must re-train for inference | Low | `src/evaluate/svm.py` |
| G4 | SVM embeddings not persisted; must re-extract for new analyses | Low | `src/evaluate/svm.py` |
| G5 | Ray experiments have no automatic retry logic; failures require manual identification + re-run | Low | `src/evaluate/ray_mlp.py` |
| G6 | `constant.py` is referenced in project memory as containing `MY_EXPERIMENTS` but the file only contains `DATASET_NUM_CLASSES` — the registry is now W&B-live, not static | Informational | `src/utils/constant.py` |

---

## Final Compliance Verdict

### **Partially Compliant** — Score: 4/5

---

### Confirmed (matches intended design)

- Experiment matrix: 3 datasets × 3 splits × 6 percentages × 4 initializers = 216 canonical entries ✅
- Naming convention `lejepa_line_{dataset}_split_{N}_pct_{pct}_model_{init}` is deterministic, parsed, and consistent across all layers ✅
- SSL checkpoints exist (297 run dirs) and are tracked by W&B run ID ✅
- Fine-tuning is strictly one-to-one — no all-vs-all combination ✅
- All three downstream modes are present: MLP freeze, MLP unfreeze, SVM ✅
- Split-based aggregation (mean ± std) is implemented in `aggregate_metrics()` and output files exist ✅
- MLP weights saved for both freeze and unfreeze modes ✅
- Reuse-first principle is well-followed — no significant reimplementation found ✅
- W&B deduplication by `created_at` ensures stale run_ids are not evaluated ✅

---

### Missing or Inconsistent

- **G1**: No script enumerates the expected 216-entry grid and cross-checks against W&B to surface missing SSL experiments
- **G2**: Aggregated result CSVs (`artifacts/`) have no traceable generation script in `scripts/`
- **G3–G4**: SVM fitted model and extracted embeddings are ephemeral (not saved to disk)

---

### Reusable Existing Components

- `src/utils/evaluate.py` — checkpoint discovery, experiment parsing, feature extraction, SVM training
- `src/utils/get_names_wandb.py` — W&B run fetching, collision-safe naming
- `src/evaluate/mlp.py` — model loading, fine-tuning, evaluation
- `src/evaluate/ray_mlp.py` — parallel experiment execution via Ray
- `src/evaluate/svm.py` — SVM evaluation pipeline
- `src/evaluate/unified_eval.py` — full SVM + MLP orchestration with aggregation
- `src/metrics/classification.py` — `compute_metrics()`, `aggregate_metrics()`
- `src/data_modules/datasets/dataset.py` — `DatasetParasite`
- `scripts/generate_mlp_configs.py` — YAML generation with deduplication

---

### Required Corrections

Only three changes are needed to reach full compliance:

1. **Add `scripts/check_missing_ssl.py`**: Enumerate all 216 expected (dataset, split, pct, init) combinations, cross-reference against W&B runs and local checkpoints, and print a gap report.

2. **Add `scripts/aggregate_results.py`**: Read `results/svm_results.csv` and `results/ray_mlp_results.csv`, apply `aggregate_metrics()` grouped by (dataset, pct, init), and write `artifacts/SVM/svm_aggregated.csv` and `artifacts/MLP/mlp_aggregated.csv` — making the aggregation reproducible and auditable.

3. **Optionally add SVM persistence** in `src/evaluate/svm.py`: After `train_svm()`, serialize the fitted `SVC` with `joblib.dump()` to `results/svm_models/{run_id}.pkl` for reuse without re-training.
