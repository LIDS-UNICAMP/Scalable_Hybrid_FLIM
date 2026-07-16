# Distillation Direct — Architecture Summary `ch24_32_48`

**Input:** `(B, 3, 200, 200)` → **Output:** `student_emb (B, 48)` + `student_proj (B, 1280)`

---

## 🧊 TEACHER — I-JEPA (frozen, sem grad)

| Etapa | Detalhe |
|---|---|
| Resize | `F.interpolate` → `(B, 3, 224, 224)` |
| Modelo | `IJEPAEncoder` ViT-H/14 · `facebook/ijepa_vith14_1k` |
| Params | ≈ **632,000,000** (frozen) |
| Output | `teacher_emb (B, 1280)` |

---

## 🔵 FLIM CNN BACKBONE — `student.encoder` (gradientes fluem)

| Layer | Operação | Input | Output | Params |
|---|---|---|---|---|
| `conv1[0]` | Conv2d 3→24ch k=5 pad=2 s=1 | (B,3,200,200) | (B,24,200,200) | 1,824 |
| `conv1[1]` | ReLU | (B,24,200,200) | (B,24,200,200) | 0 |
| `conv1[2]` | MaxPool2d k=3 s=2 | (B,24,200,200) | (B,24,99,99) | 0 |
| `conv2[0]` | Conv2d 24→32ch k=5 pad=2 s=1 | (B,24,99,99) | (B,32,99,99) | 19,232 |
| `conv2[1]` | ReLU | (B,32,99,99) | (B,32,99,99) | 0 |
| `conv2[2]` | MaxPool2d k=3 s=2 | (B,32,99,99) | (B,32,49,49) | 0 |
| `conv3[0]` | Conv2d 32→48ch k=5 pad=2 s=1 | (B,32,49,49) | (B,48,49,49) | 38,448 |
| `conv3[1]` | ReLU | (B,48,49,49) | (B,48,49,49) | 0 |
| `conv3[2]` | MaxPool2d k=3 s=2 | (B,48,49,49) | (B,48,24,24) | 0 |
| `pool` | AdaptiveAvgPool2d(1) | (B,48,24,24) | (B,48,1,1) | 0 |
| **Subtotal FLIM** | | | | **59,504** |

> `student_emb = (B, 48)` ← embedding bruto, usado no SVM / avaliação


---

## 🟡 CONV KD PROJECTION HEAD — Variante A: `ConvDistillationProjectionHead` (next_layers 1×1)

> 4 convoluções pointwise (k=1×1) a partir de `feat_map (B, 48, 24, 24)` — **sem pool no FLIM**

| Layer | Operação | Input | Output | Params |
|---|---|---|---|---|
| `proj[0]` | Conv2d 48→128 k=1×1 bias=False | (B,48,24,24) | (B,128,24,24) | 6,144 |
| `proj[1]` | BN2d(128) + ReLU | (B,128,24,24) | (B,128,24,24) | 256 |
| `proj[3]` | Conv2d 128→256 k=1×1 bias=False | (B,128,24,24) | (B,256,24,24) | 32,768 |
| `proj[4]` | BN2d(256) + ReLU | (B,256,24,24) | (B,256,24,24) | 512 |
| `proj[6]` | Conv2d 256→512 k=1×1 bias=False | (B,256,24,24) | (B,512,24,24) | 131,072 |
| `proj[7]` | BN2d(512) + ReLU | (B,512,24,24) | (B,512,24,24) | 1,024 |
| `proj[9]` | Conv2d 512→1280 k=1×1 bias=False | (B,512,24,24) | (B,1280,24,24) | 655,360 |
| `proj[10]` | BN2d(1280) | (B,1280,24,24) | (B,1280,24,24) | 2,560 |
| `pool` | AdaptiveAvgPool2d(1) | (B,1280,24,24) | (B,1280,1,1) | 0 |
| `flatten` | — | (B,1280,1,1) | (B,1280) | 0 |
| **Subtotal** | | | | **829,696** |

> `student_proj = (B, 1280)` ← entra na `KLLoss` contra `teacher_emb (B, 1280)`

---

## 🟠 CONV KD PROJECTION HEAD — Variante B: `OneLayerConvDistillationProjectionHead` (3×3 BN2d)

> 1 convolução 3×3 direta 48→1280 a partir de `feat_map (B, 48, 24, 24)` — **sem pool no FLIM**

| Layer | Operação | Input | Output | Params |
|---|---|---|---|---|
| `proj[0]` | Conv2d 48→1280 k=3×3 pad=1 bias=False | (B,48,24,24) | (B,1280,24,24) | 552,960 |
| `proj[1]` | BN2d(1280) | (B,1280,24,24) | (B,1280,24,24) | 2,560 |
| `proj[2]` | GELU | (B,1280,24,24) | (B,1280,24,24) | 0 |
| `pool` | AdaptiveAvgPool2d(1) | (B,1280,24,24) | (B,1280,1,1) | 0 |
| `flatten` | — | (B,1280,1,1) | (B,1280) | 0 |
| **Subtotal** | | | | **555,520** |


---

## Comparação dos Projection Heads

| Arquitetura | Camadas | Kernel | Ativação | Params proj head | Params totais (FLIM + head) |
|---|---|---|---|---|---|
| `ConvDistillationProjectionHead` (next_layers 1×1) | 4 | 1×1 | ReLU | 829,696 | 889,200 |
| `OneLayerConvDistillationProjectionHead` (3×3 BN2d) | 1 | 3×3 | GELU | 555,520 | 615,024 |

---

### Fluxo de dados resumido

```
INPUT (B,3,200,200)
    │
    ├──► [FLIM CNN] ──► feat_map (B,48,24,24)
    │         │
    │         ├── pool+flatten ──► student_emb (B,48) ──► SVM / avaliação
    │         │
    │         ├── [Variante A: next_layers 1×1]  ──► student_proj (B,1280)
    │         └── [Variante B: 3×3 BN2d]         ──► student_proj (B,1280)
    │
    └──► [TEACHER I-JEPA ViT-H/14 frozen]
              │ resize (224,224)
              └──► teacher_emb (B,1280)
```

---

## Resultados SVM — Destilação 1×1 BN2d (Variante A) com norma ImageNet

> Métricas: média ± desvio-padrão · 3 splits · `percentage=100` · `status=ok`

### FLIM init + 2L 1×1 BN2d — `trunc_normal` init, com norma ImageNet, proj 1280

> CSV: `svm_proj1280_2l_1x1_BN2d_256_1280_results.csv` (filtro: `encoder_init==trunc_normal`)

| Dataset | F1 | κ | Acc |
|---|---|---|---|
| eggs | 0.84 ± 0.01 | 0.82 ± 0.01 | 0.86 ± 0.01 |
| larvae | 0.94 ± 0.00 | 0.88 ± 0.01 | 0.95 ± 0.00 |
| protozoan | 0.80 ± 0.03 | 0.79 ± 0.02 | 0.82 ± 0.03 |

### FLIM init + 2L 1×1 BN2d — FLIM init, com norma ImageNet, proj 1280

> CSV: `svm_2l_1x1_init_flim_256_1280_results.csv`

| Dataset | F1 | κ | Acc |
|---|---|---|---|
| eggs | 0.80 ± 0.07 | 0.77 ± 0.07 | 0.84 ± 0.05 |
| larvae | 0.79 ± 0.27 | 0.63 ± 0.44 | 0.84 ± 0.17 |
| protozoan | 0.67 ± 0.04 | 0.56 ± 0.07 | 0.70 ± 0.05 |

> ⚠️ Alta variância em larvae (κ=0.63±0.44) indica instabilidade do checkpoint FLIM init com norma ImageNet.

---

## Resultados SVM — Destilação 1×1 BN2d (Variante A) SEM norma

> Métricas: média ± desvio-padrão · 3 splits · `percentage=100` · `status=ok`

### FLIM init + 1×1 BN2d, SEM norma — encoder 48

> CSV: `svm_nonorm_1x1_encoder48_results.csv`

| Dataset | F1 | κ | Acc |
|---|---|---|---|
| eggs | 0.27 ± 0.21 | 0.24 ± 0.28 | 0.28 ± 0.19 |
| larvae | 0.92 ± 0.01 | 0.84 ± 0.01 | 0.92 ± 0.02 |
| protozoan | 0.17 ± 0.06 | 0.07 ± 0.07 | 0.19 ± 0.05 |

> ⚠️ Alta variância e valores baixos em eggs e protozoan sugerem colapso de representação sem norma no encoder 48.

### FLIM init + 1×1 BN2d, SEM norma — proj 1280

> CSV: `svm_nonorm_1x1_proj1280_results.csv`

| Dataset | F1 | κ | Acc |
|---|---|---|---|
| eggs | 0.53 ± 0.11 | 0.42 ± 0.13 | 0.75 ± 0.03 |
| larvae | 0.93 ± 0.02 | 0.86 ± 0.03 | 0.93 ± 0.01 |
| protozoan | 0.29 ± 0.06 | 0.16 ± 0.03 | 0.47 ± 0.05 |

> A proj 1280 recupera parcialmente larvae mas eggs e protozoan permanecem fracos sem norma.

---

## Resultados SVM — Destilação 2L 1×1 BN2d (Variante A) SEM norma

> Métricas: média ± desvio-padrão · 3 splits · `percentage=100` · `status=ok`

### FLIM init + 2L 1×1 BN2d, SEM norma — encoder 48

> CSV: `svm_2l_1x1_init_flim_256_1280_nonorm_encoder48.csv`

| Dataset | F1 | κ | Acc |
|---|---|---|---|
| eggs | 0.57 ± 0.25 | 0.57 ± 0.25 | 0.57 ± 0.25 |
| larvae | 0.95 ± 0.00 | 0.90 ± 0.00 | 0.95 ± 0.01 |
| protozoan | 0.63 ± 0.10 | 0.59 ± 0.09 | 0.60 ± 0.09 |

> A 2ª camada 1×1 melhora protozoan vs. 1L sem norma (0.63 vs. 0.17 F1), mas eggs ainda instável.

### FLIM init + 2L 1×1 BN2d, SEM norma — proj 1280

> CSV: `svm_2l_1x1_init_flim_256_1280_nonorm_proj1280.csv`

| Dataset | F1 | κ | Acc |
|---|---|---|---|
| eggs | 0.86 ± 0.05 | 0.84 ± 0.06 | 0.89 ± 0.02 |
| larvae | 0.96 ± 0.01 | 0.92 ± 0.02 | 0.96 ± 0.01 |
| protozoan | 0.77 ± 0.06 | 0.74 ± 0.09 | 0.80 ± 0.04 |

> Melhor resultado sem norma: 2L 1×1 + proj 1280 supera inclusive a variante com norma ImageNet (eggs 0.86 vs. 0.84, protozoan 0.77 vs. 0.80).

---

## Resultados SVM — FLIM frozen + proj conv 1×1 BN2d (destilação com encoder congelado)

> Métricas: média ± desvio-padrão · **6 splits** (média dos dois checkpoints: knn + loss) · `percentage=100` · `status=ok`

### FLIM frozen + proj 1×1 BN2d conv

> CSV: `svm_distillation_conv_flim_frozen_results.csv`

| Dataset | F1 | κ | Acc |
|---|---|---|---|
| eggs | 0.77 ± 0.02 | 0.79 ± 0.03 | 0.75 ± 0.03 |
| larvae | 0.95 ± 0.01 | 0.91 ± 0.02 | 0.95 ± 0.01 |
| protozoan | 0.74 ± 0.03 | 0.75 ± 0.02 | 0.69 ± 0.03 |

> Encoder FLIM congelado: proj conv aprende boa representação para larvae e protozoan, mas eggs fica abaixo dos experimentos com encoder fine-tunable (0.77 vs. 0.84–0.86 F1).

---

## Visão Geral Comparativa (SVM, `percentage=100`)

| Experimento | Init encoder | Norma | Avaliação | Splits | eggs F1 | larvae F1 | protozoan F1 |
|---|---|---|---|---|---|---|---|
| 1×1 BN2d sem norma — enc48 | FLIM | ✗ | encoder 48 | 3 | 0.27 ± 0.21 | 0.92 ± 0.01 | 0.17 ± 0.06 |
| 1×1 BN2d sem norma — proj1280 | FLIM | ✗ | proj 1280 | 3 | 0.53 ± 0.11 | 0.93 ± 0.02 | 0.29 ± 0.06 |
| 2L 1×1 BN2d sem norma — enc48 | FLIM | ✗ | encoder 48 | 3 | 0.57 ± 0.25 | 0.95 ± 0.00 | 0.63 ± 0.10 |
| **2L 1×1 BN2d sem norma — proj1280** | FLIM | ✗ | proj 1280 | 3 | **0.86 ± 0.05** | **0.96 ± 0.01** | **0.77 ± 0.06** |
| 2L 1×1 BN2d com norma — trunc_normal | trunc_normal | ✓ | proj 1280 | 3 | 0.84 ± 0.01 | 0.94 ± 0.00 | 0.80 ± 0.03 |
| 2L 1×1 BN2d com norma — FLIM init | FLIM | ✓ | proj 1280 | 3 | 0.80 ± 0.07 | 0.79 ± 0.27 | 0.67 ± 0.04 |
| FLIM frozen + proj conv | FLIM | — | proj 1280 | 6 | 0.77 ± 0.02 | 0.95 ± 0.01 | 0.74 ± 0.03 |

---

## Catálogo de Modelos — chave do plot × parâmetros

> Cada método registrado em `METHOD_COMPARE_LABEL` (`src/evaluate/eval_plotter.py`), com a contagem de parâmetros treináveis (FLIM backbone ≈ 59,504 + proj head, quando aplicável). A avaliação SVM usa o embedding indicado (`student_emb (B,48)` bruto ou `student_proj (B,1280)`).

| Chave do método | Rótulo no plot | Params | Embedding eval |
|---|---|---|---|
| `SVM_FLIM` | FLIM SVM | ~60K (59,504) | `(B,48)` |
| `SVM_LeJEPA_flim` | LeJEPA SVM — FLIM init | ~60K | `(B,48)` |
| `SVM_LeJEPA_he` | LeJEPA SVM — HE init | ~60K | `(B,48)` |
| `SVM_LeJEPA_xavier` | LeJEPA SVM — Xavier init | ~60K | `(B,48)` |
| `SVM_LeJEPA_random` | LeJEPA SVM — Random init | ~60K | `(B,48)` |
| `SVM_LeJEPA_trunc_normal` | LeJEPA SVM — TruncNorm | ~60K | `(B,48)` |
| `SVM_IJEPA` | I-JEPA SVM (teacher, frozen) | ~632M | `(B,1280)` |
| `SVM_Distillation_Conv` | Distil-Conv [48] | ~60K | `(B,48)` |
| `SVM_Distill_Proj1280` | Distil 3 (proj 4× 1×1, next_layers) | 889,200 | `(B,1280)` |
| `SVM_Distill_3x3BN` | Distil 2 (3×3 BN2d, trunc_normal) | 615,024 | `(B,1280)` |
| `SVM_Distill_3x3BN_flim` | Distil 2 — FLIM init | 615,024 (~615K) | `(B,1280)` |
| `SVM_Distill_1x1BN` | Distil 1 (1×1 BN2d one-layer) | 123,504 | `(B,1280)` |
| `SVM_Distill_1x1BN_flim` | Distil 1 — FLIM init | 123,504 (~123K) | `(B,1280)` |
| `SVM_Distill_1x1BN_nonorm` | Distil 1 — FLIM sem ImageNet-norm | 123,504 | `(B,1280)` |
| `SVM_Distill_2l400K` | Distil 2L 1×1 | 402,608 | `(B,1280)` |
| `SVM_Distill_2l400K_flim` | Distil 2L — FLIM init | 402,608 (~402K) | `(B,1280)` |
| `SVM_Distill_2l400K_flim_nonorm` | Distil 2L — FLIM init sem norm | 402,608 (~402K) | `(B,1280)` |
| `SVM_Distill_1x1BN_flim_frozen_eval_knn` | Distil 1×1 FLIM congelado — ckpt knn+κ | 123,504 (~123K) | `(B,1280)` |
| `SVM_Distill_1x1BN_flim_frozen_eval_loss` | Distil 1×1 FLIM congelado — ckpt best-loss | 123,504 (~123K) | `(B,1280)` |

> **Notas de contagem:**
> - Os `SVM_LeJEPA_*` compartilham o mesmo backbone (~60K), variando apenas a inicialização dos pesos (`flim`, `he`, `xavier`, `random`, `trunc_normal`).
> - `Distil 1/2/3` referem-se ao tamanho da proj head: **Distil 1** (1×1 BN2d, 123K) < **Distil 2** (3×3 BN2d, 615K) < **Distil 3** (4× 1×1 next_layers, 889K). A `Distil 2L 1×1` (402K) fica entre Distil 1 e Distil 2.
> - As variantes `_flim` / `_nonorm` / `_frozen_eval_*` **não alteram a contagem de parâmetros** — diferem na inicialização do encoder (`flim` vs `trunc_normal`), no uso de normalização ImageNet, ou no checkpoint avaliado (knn+κ vs best-loss).

---

## Como cada experimento foi gerado (procedência)

> Documentação completa (treino → checkpoint → eval SVM → CSV → unificação → plot) em
> [`data_provenance.md`](data_provenance.md). Resumo dos **8 modelos oficiais** dos plots:

| Curva (rótulo) | método (chave) | CSV de origem | gerador / comando |
|---|---|---|---|
| FLIM | `SVM_FLIM` | `data/reports_felipe/svm/report_svm_*.csv` | **externo** (não há gerador no repo; dir gitignored) |
| LeJEPA | `SVM_LeJEPA` (`trunc_normal`) | `artifacts/SVM/*/lejepa_pct_*/metrics_SVM_*.csv` | `run_ssl_ray.py` (pré-treino) → `python -m src.evaluate.unified_eval --model svm --dataset all` |
| I-JEPA | `SVM_IJEPA` | `results/ijepa_svm_aggregated.csv` | `python -m src.evaluate.svm_ijepa` (teacher ViT-H/14 frozen) |
| Distill 4 | `SVM_Distill_Proj1280` | `results/svm_distill_proj1280_results.csv` | `python -m src.evaluate.svm_distill_with_projection --output-csv svm_distill_proj1280_results` |
| Distill 3 | `SVM_Distill_3x3BN` | `results/svm_proj1280_3x3_BN2d_results.csv` | `... svm_distill_with_projection --run-filter 3x3_BN2d_1280_one_layer --output-csv svm_proj1280_3x3_BN2d_results` |
| Distill 1 | `SVM_Distill_1x1BN` | `results/svm_proj1280_1x1_BN2d_results.csv` | `... svm_distill_with_projection --run-filter 1x1_BN2d_1280_one_layer --output-csv svm_proj1280_1x1_BN2d_results` |
| Distill 2 | `SVM_Distill_2l400K` | `results/svm_proj1280_2l_1x1_BN2d_256_1280_results.csv` | `... svm_distill_with_projection --run-filter 2l_1x1_BN2d_256_1280` |
| Distill 1 (FLIM init) | `SVM_Distill_1x1BN_flim_frozen_eval_loss` | `results/svm_distillation_conv_flim_frozen_results.csv` | `... svm_distillation_conv --run-filter 1x1_BN2d_1280_flim_frozen --output-csv svm_distillation_conv_flim_frozen_results` (ckpt best-loss) |

**Notas de geração:**
- Os 4 "Distill 1/2/3/4" (encoder **treinável**, init `trunc_normal`) saem do mesmo avaliador
  `svm_distill_with_projection.py`; o que muda é `--run-filter`/`--output-csv` e a proj head auto-detectada pelo shape dos pesos.
  Checkpoints treinados via `scripts/distillation_conv_ray.py` (`--proj-type`) → trainers `src/modules/distillation_{onelayer,twolayer,conv}_module.py`, selecionados por `val/knn_kappa`.
- "Distill 1 (FLIM init)" usa o **encoder FLIM congelado** (sem backprop) + proj 1×1 BN2d treinada por cima, sem norma ImageNet;
  o CSV traz 2 checkpoints (`_eval_knn` por `val/knn_kappa`, `_eval_loss` por `val/loss`) — o plot oficial usa só `_eval_loss`.
- Toda agregação média ± std sobre splits é feita em `scripts/normalize_reports.py` (cada linha do CSV de destilação é 1 run = 1 dataset × 1 split × 1 percentage).
