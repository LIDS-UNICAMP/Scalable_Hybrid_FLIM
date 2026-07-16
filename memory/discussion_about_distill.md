# Discussão: Destilação I-JEPA → FLIM CNN (next_layers_direct)

---

## Arquitetura de Destilação

**Dois módulos distintos:**

| Módulo | Proj Head | Loss | Nome do run |
|---|---|---|---|
| `distillation_module.py` | Linear (1728→1280): pool6×6 → flatten → Linear | MSE/KL | `..._modeldirect` |
| `distillation_conv_module.py` | 1×1 Convs (48→128→256→512→1280) + GAP | MSE/KL | `..._next_layers_direct` |

**O módulo ativo nos experimentos é `distillation_conv_module.py`** (next_layers_direct).

### FLIM Encoder (student backbone)
- Input: `[B, 3, 200, 200]`
- conv1: Conv2d(3→24, 5×5) + ReLU + MaxPool(3,s=2) → `[B, 24, 99, 99]`
- conv2: Conv2d(24→32, 5×5) + ReLU + MaxPool(3,s=2) → `[B, 32, 49, 49]`
- conv3: Conv2d(32→48, 5×5) + ReLU + MaxPool(3,s=2) → `[B, 48, 24, 24]`
- GlobalAvgPool + flatten → `[B, 48]`
- 59,504 params — pesos trunc_normal init (NÃO pesos FLIM)

### ConvDistillationProjectionHead (Red Projection)
- 1×1 convs: 48→128→256→512→1280 + BN2d + ReLU
- AdaptiveAvgPool2d(1) + flatten → `[B, 1280]`
- 829,696 params

### I-JEPA Teacher (frozen)
- ViT-H/14, `facebook/ijepa_vith14_1k`, 630M params
- Input: `[B, 3, 224, 224]` (resize bilinear do student)
- 32 Transformer blocks, 16 heads, hidden=1280, mean pool → `[B, 1280]`

---

## Duas Avaliações SVM

### 1. Sem proj head — `svm_distillation_conv.py` → `[B, 48]`
```python
student = _load_student_from_ckpt(ckpt_path, device)  # só pesos "student.*"
feats = student.encode(x)  # [B, 48]
```
**I-JEPA NÃO é carregado** — extrai só `student.*` do state_dict manualmente.

### 2. Com proj head (Red Projection) — `svm_distill_with_projection.py` → `[B, 1280]`
```python
student, proj_kd = _load_student_and_proj(ckpt_path, device)  # "student.*" + "proj_kd.*"
feat_map = student.encoder(x)   # [B, 48, 24, 24]
feats    = proj_kd(feat_map)     # [B, 1280]
```
Usa `StandardScaler + SVM` (1280 dims requer normalização para convergir).

---

## Resultados

### kappa SVM sem proj head [B, 48] (pct=100, média dos splits)

| Dataset | Classes | val/loss best | Distil [48] | I-JEPA | FLIM |
|---|---|---|---|---|---|
| larvae | 2 | 0.0657 | **0.850** | 0.950 | 0.868 |
| eggs | 9 | 0.0382 | 0.073 | 0.957 | 0.885 |
| protozoan | 7 | 0.0122 | 0.245 | 0.892 | 0.847 |

**Padrão crítico: menor MSE → pior classificação (relação invertida).**

### kappa SVM com proj head [B, 1280] (parcial, split3 pct=100)

| Dataset | Distil+Proj [1280] | Distil [48] |
|---|---|---|
| eggs | **0.8899** | 0.073 |

A proj head recupera o desempenho → confirma projection head dominance.

---

## Diagnóstico: Projection Head Dominance

A `ConvProjectionHead` (48→1280) é poderosa demais. Durante o treino, **a head aprende a fazer todo o trabalho**. O encoder FLIM aprende só o mínimo para a head funcionar — os 48 features não ficam discriminativos.

- Para larvae (2 classes): features genéricas bastam → funciona
- Para eggs (9 classes) e protozoan (7 classes): encoder não aprende features discriminativas → falha
- Com proj head ativa no SVM: desempenho volta → a head tinha a informação, não o encoder

---

## Possíveis Correções

1. **Head linear fraca** (`distillation_module.py`) — comparar resultados
2. **Supervision direta nos 48 dims** (contrastive, clustering)
3. **Aumentar capacidade do encoder** (mais canais no FLIM)
4. **Loss auxiliar nos 48 dims** além do MSE nos 1280

---

## Scripts

| Script | Embedding | CSV de saída |
|---|---|---|
| `src/modules/distillation_conv_module.py` | — | — |
| `src/evaluate/svm_distillation_conv.py` | `[B, 48]` sem proj | `svm_distillation_conv_results.csv` |
| `src/evaluate/svm_distill_with_projection.py` | `[B, 1280]` com proj | `svm_distill_proj1280_results.csv` |
| `scripts/check_distill_conv_status.py` | status 4 fontes | — |
| `scripts/distillation_conv_ray.py` | launcher Ray | `run_manifest_conv.csv` |
| `distillation_model_architecture.md` | doc layer-by-layer | — |

---

## Lógica Retry (`--skip-existing --check-wandb`)

1. W&B `running` → SKIP
2. W&B `finished` → SKIP
3. W&B `crashed/failed` + local `ok` → SKIP (treino terminou, W&B crashou ao matar tmux)
4. W&B `crashed/failed` + local != ok → QUEUE
5. Sem W&B → checa metadata local

---

## Problemas Operacionais

- **OOM**: sempre `--max-concurrent-per-gpu 1` — I-JEPA usa ~5 GB por processo
- **Checkpoint path**: Lightning cria subdir `val/loss=X.ckpt` porque `/` no filename vira path separator. Glob `**/*.ckpt` resolve.
- **sorted(os.scandir())**: TypeError — fix: `key=lambda e: e.name`
- **SVM 1280 dims**: sem `StandardScaler` o solver não converge — usar `Pipeline([StandardScaler, SVC])`

---

## Comparação Geral (kappa, pct=100, média splits)

| Método | Eggs | Larvae | Protozoan |
|---|---|---|---|
| FLIM | 0.885 | 0.868 | 0.847 |
| LeJEPA trunc_normal | ~0.31 | ~0.27 | — |
| I-JEPA | **0.957** | **0.950** | **0.892** |
| Distil-Conv [48] | 0.073 | 0.850 | 0.245 |
| Distil-Conv+Proj [1280] | ~0.88+ | — | — |
