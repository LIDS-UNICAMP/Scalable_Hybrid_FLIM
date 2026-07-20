# Architecture Verification Report
**Generated:** 2026-05-30  
**Source:** `/artifacts/distillation/distillation_*/run_metadata.json`  
**Total run directories:** 270 (3 datasets × 3 splits × 6 pcts × 5 groups)  
**Total metadata files found:** 245  
**Missing metadata files:** 25  

---

## 1. arch_json Alignment — Summary

All 245 runs with metadata have `arch_json` paths that correctly match their dataset name and expected channel configuration:

| Dataset   | Expected channels | arch_json pattern observed | Mismatch? |
|-----------|------------------|---------------------------|-----------|
| protozoan | ch24_30_48       | `ch24_30_48_a0.5_f5/protozoan/train{1,2,3}/` | NONE |
| larvae    | ch24_32_48       | `ch24_32_48_a0.5_f5/larvae/train{1,2,3}/`    | NONE |
| eggs      | ch24_32_48       | `ch24_32_48_a0.5_f5/eggs/train{1,2,3}/`      | NONE |

The split number in the run name (split1, split2, split3) always matches the `train{N}` subdirectory in `arch_json`.

**No cross-dataset arch_json contamination found in any of the 245 verified runs.**

---

## 2. Architectural Inconsistencies

**None found.** All 245 verified runs satisfy every constraint below:

### proj_head consistency by group

| Model group | Expected proj_head | Observed | OK? |
|-------------|-------------------|----------|-----|
| 1x1_BN2d_1280_one_layer | Conv1x1(48→1280)+BN2d+GELU | All 54 runs (eggs+larvae) + 18 protozoan runs with metadata | YES |
| 1x1_BN2d_1280_one_layer_flim_init | Conv1x1(48→1280)+BN2d+GELU | All 36 runs (eggs+larvae) with metadata | YES |
| 3x3_BN2d_1280_one_layer | Conv3x3(48→1280)+BN2d+GELU | All 54 runs (3 datasets) | YES |
| next_layers_direct | conv_next_layers | All 54 runs (3 datasets) | YES |
| modeldirect | (no proj_head field) | All runs with metadata | YES |

### encoder_init consistency

| Group pattern | Expected encoder_init | Observed | OK? |
|---------------|----------------------|----------|-----|
| *_flim_init | flim | All 36 flim_init runs with metadata | YES |
| all other groups | trunc_normal | All non-flim_init runs | YES |

### teacher_embed_dim and student_embed_dim

- All 245 runs: `teacher_embed_dim = 1280` — PASS  
- All 245 runs: `student_embed_dim = 48` — PASS  

### status

- All 245 runs with metadata: `status = ok` — PASS  
- 0 failed runs found.

---

## 3. Missing Runs per Group

### Overview

| Dataset   | Group                            | Expected | Has metadata | Missing |
|-----------|----------------------------------|----------|-------------|---------|
| eggs      | 1x1_BN2d_1280_one_layer          | 18       | 18          | 0       |
| eggs      | 1x1_BN2d_1280_one_layer_flim_init| 18       | 18          | 0       |
| eggs      | 3x3_BN2d_1280_one_layer          | 18       | 18          | 0       |
| eggs      | modeldirect                      | 18       | 18          | 0       |
| eggs      | next_layers_direct               | 18       | 18          | 0       |
| larvae    | 1x1_BN2d_1280_one_layer          | 18       | 18          | 0       |
| larvae    | 1x1_BN2d_1280_one_layer_flim_init| 18       | 18          | 0       |
| larvae    | 3x3_BN2d_1280_one_layer          | 18       | 18          | 0       |
| larvae    | modeldirect                      | 18       | 18          | 0       |
| larvae    | next_layers_direct               | 18       | 18          | 0       |
| protozoan | 1x1_BN2d_1280_one_layer          | 18       | 18          | 0       |
| protozoan | 1x1_BN2d_1280_one_layer_flim_init| 18       | 0           | **18**  |
| protozoan | 3x3_BN2d_1280_one_layer          | 18       | 18          | 0       |
| protozoan | modeldirect                      | 18       | 11          | **7**   |
| protozoan | next_layers_direct               | 18       | 18          | 0       |
| **TOTAL** |                                  | **270**  | **245**     | **25**  |

### Specific missing runs (no run_metadata.json found)

**Group: protozoan / 1x1_BN2d_1280_one_layer_flim_init — ALL 18 MISSING**

Directories exist (contain `checkpoints/` and `wandb/`) but have no `run_metadata.json`:

```
distillation_protozoan_split1_pct1_1x1_BN2d_1280_one_layer_flim_init
distillation_protozoan_split1_pct5_1x1_BN2d_1280_one_layer_flim_init
distillation_protozoan_split1_pct25_1x1_BN2d_1280_one_layer_flim_init
distillation_protozoan_split1_pct50_1x1_BN2d_1280_one_layer_flim_init
distillation_protozoan_split1_pct75_1x1_BN2d_1280_one_layer_flim_init
distillation_protozoan_split1_pct100_1x1_BN2d_1280_one_layer_flim_init
distillation_protozoan_split2_pct1_1x1_BN2d_1280_one_layer_flim_init
distillation_protozoan_split2_pct5_1x1_BN2d_1280_one_layer_flim_init
distillation_protozoan_split2_pct25_1x1_BN2d_1280_one_layer_flim_init
distillation_protozoan_split2_pct50_1x1_BN2d_1280_one_layer_flim_init
distillation_protozoan_split2_pct75_1x1_BN2d_1280_one_layer_flim_init
distillation_protozoan_split2_pct100_1x1_BN2d_1280_one_layer_flim_init
distillation_protozoan_split3_pct1_1x1_BN2d_1280_one_layer_flim_init
distillation_protozoan_split3_pct5_1x1_BN2d_1280_one_layer_flim_init
distillation_protozoan_split3_pct25_1x1_BN2d_1280_one_layer_flim_init
distillation_protozoan_split3_pct50_1x1_BN2d_1280_one_layer_flim_init
distillation_protozoan_split3_pct75_1x1_BN2d_1280_one_layer_flim_init
distillation_protozoan_split3_pct100_1x1_BN2d_1280_one_layer_flim_init
```

**Group: protozoan / modeldirect — 7 of 18 MISSING**

Has metadata: split1 (pct1,5,25,50,75), split2 (pct1,5,25,50,75), split3 (pct1) = 11 runs  
Missing (directory exists but no `run_metadata.json`):

```
distillation_protozoan_split1_pct100_modeldirect
distillation_protozoan_split2_pct100_modeldirect
distillation_protozoan_split3_pct5_modeldirect
distillation_protozoan_split3_pct25_modeldirect
distillation_protozoan_split3_pct50_modeldirect
distillation_protozoan_split3_pct75_modeldirect
distillation_protozoan_split3_pct100_modeldirect
```

---

## 4. Failed Runs

**None.** All 245 runs with metadata have `status = ok`.  
There are 25 runs with directories but no `run_metadata.json` — their actual run status is unknown.

---

## 5. Summary

### Are experiments configured correctly?

**YES — for all 245 verified runs, the configuration is fully correct.**

Specific findings:

1. **arch_json dataset alignment**: 100% correct. Every protozoan run uses `ch24_30_48` architecture, every eggs and larvae run uses `ch24_32_48` architecture. No cross-dataset contamination.

2. **arch_json split alignment**: 100% correct. The `train{N}` folder in `arch_json` always matches the `splitN` in the run name.

3. **proj_head per group**: All groups have the expected projection head type consistently across all datasets and splits.

4. **encoder_init**: All `*_flim_init` runs correctly have `encoder_init = flim`. All others correctly have `encoder_init = trunc_normal`.

5. **teacher_embed_dim**: All 245 runs = 1280 (I-JEPA ViT-H/14). PASS.

6. **student_embed_dim**: All 245 runs = 48 (FLIM CNN conv3 output). PASS.

7. **Status**: All 245 runs = "ok". Zero failures among completed runs.

### Outstanding issues (not architectural errors, just missing data)

- **25 protozoan runs** lack `run_metadata.json` (18 flim_init + 7 modeldirect). The directories and checkpoint/wandb subdirectories exist, suggesting the training jobs were launched but either the metadata-writing step failed or the runs were not completed on the server and the metadata file was never synced locally. These runs cannot be verified architecturally without their metadata.
- Protozoan flim_init is entirely unverified (0/18 have metadata).
- Protozoan modeldirect split3 is mostly unverified (only split3_pct1 has metadata; 5 of 6 pcts missing).
