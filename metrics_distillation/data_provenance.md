# Data Provenance — de onde vem cada curva dos plots

Documenta **como cada CSV é gerado** (treino → checkpoint → avaliação SVM → CSV) para
os **8 modelos oficiais** do experimento, exatamente os que aparecem nas curvas/legenda
de `scripts/plot_comparison_flim.py`.

## Pipeline geral

```
                          [treino / checkpoint]            [avaliação SVM]                 [normalização + plot]
FLIM (externo) ─────────────────────────────────────────► data/reports_felipe/svm/*.csv ┐
LeJEPA  ── run_ssl_ray.py ──► ckpt LeJEPA ── unified_eval.py ──► artifacts/SVM/*/metrics_SVM_*.csv ┤
I-JEPA  ── teacher frozen ───────────────── svm_ijepa.py ──────► results/ijepa_svm_aggregated.csv  ┤
Distill ── distillation_conv_ray.py ──► ckpt ── svm_distill_with_projection.py ──► results/svm_*.csv ┼─► normalize_reports.py
                                                svm_distillation_conv.py (frozen) ──► results/svm_*.csv ┘        │
                                                                                                                ▼
                                                                              artifacts/normalized/unified_svm_comparison.csv
                                                                                                                │
                                                                                                                ▼
                                                                               plot_comparison_flim.py → merge_plots/{kappa,acc,f1}.png
```

- **Hub central:** `scripts/normalize_reports.py` lê cada CSV de origem, filtra
  (`status==ok`, `encoder_init`, `encoder_mode`…) e **agrega média ± std sobre os splits/folds**
  por `(dataset, percentage)` com `ddof=1`. O resultado vai para
  `artifacts/normalized/unified_svm_comparison.csv`.
- **Importante:** cada linha dos CSVs de destilação é **UM run** (1 dataset × 1 split × 1 percentage).
  Não há cross-validation interno no avaliador — a média ± std só é computada depois, no `normalize_reports.py`.
- **SVM** (todos os avaliadores): `sklearn.svm.SVC(kernel="linear", C=1e2, gamma="auto",
  decision_function_shape="ovo")`. `max_iter`: 20000 em `svm_distill_with_projection.py`,
  10000 em `svm_distillation_conv.py` / `unified_eval.py` / `svm_ijepa.py`.
- **Datasets / classes:** `eggs`=9, `larvae`=2, `protozoan`=7. Percentages: 1/5/25/50/75/100. Splits: 1/2/3.
- **Métricas:** `compute_metrics()` (`src/metrics/classification.py`) → `kappa` (Cohen), `acc`, `f1`.

---

## Resumo (chave → CSV → gerador)

| # | Curva | método (chave) | CSV de origem | gerador |
|---|---|---|---|---|
| 1 | FLIM | `SVM_FLIM` | `data/reports_felipe/svm/report_svm_*.csv` | **externo** (não há gerador no repo) |
| 2 | LeJEPA | `SVM_LeJEPA` (`trunc_normal`) | `artifacts/SVM/*/lejepa_pct_*/metrics_SVM_*.csv` | `src/evaluate/unified_eval.py` |
| 3 | I-JEPA | `SVM_IJEPA` | `results/ijepa_svm_aggregated.csv` | `src/evaluate/svm_ijepa.py` |
| 4 | Distill 4 | `SVM_Distill_Proj1280` | `results/svm_distill_proj1280_results.csv` | `src/evaluate/svm_distill_with_projection.py` |
| 5 | Distill 3 | `SVM_Distill_3x3BN` | `results/svm_proj1280_3x3_BN2d_results.csv` | `src/evaluate/svm_distill_with_projection.py` |
| 6 | Distill 1 | `SVM_Distill_1x1BN` | `results/svm_proj1280_1x1_BN2d_results.csv` | `src/evaluate/svm_distill_with_projection.py` |
| 7 | Distill 2 | `SVM_Distill_2l400K` | `results/svm_proj1280_2l_1x1_BN2d_256_1280_results.csv` | `src/evaluate/svm_distill_with_projection.py` |
| 8 | Distill 1 (FLIM init) | `SVM_Distill_1x1BN_flim_frozen_eval_loss` | `results/svm_distillation_conv_flim_frozen_results.csv` | `src/evaluate/svm_distillation_conv.py` |

---

## 1. FLIM — `SVM_FLIM`

- **Origem: EXTERNA.** `data/reports_felipe/svm/` está no `.gitignore` (`.gitignore:46`), com 0 commits e 0 arquivos rastreados.
  Nenhum script do repo escreve `report_svm_*.csv` nem produz a coluna `encoder_mode`/`test_cohen_kappa` — só há **leitores**
  (`normalize_reports.py`, `plot_comparison_flim.py`, `eval_plotter.py`). Os 54 CSVs vieram de fora (pipeline do "Felipe").
- **O que é:** encoder FLIM (CNN ~60K params construída por superpixels/marcadores **sem backprop** e **sem pré-treino LeJEPA**),
  frozen, avaliado por SVM linear. Todos os CSVs têm `init_method=flim`, `encoder_mode=frozen`.
- **Normalização:** `normalize_felipe_svm()` (`normalize_reports.py:80-148`) — filtra `encoder_mode=="frozen"`, lê
  `test_cohen_kappa→kappa`, `test_accuracy→acc`, `test_f1_weighted→f1`, agrega por `(dataset, percentage)`.
- **Regenerar:** ❌ não existe no repo (fonte externa). Para atualizar, basta recolocar os CSVs em `data/reports_felipe/svm/`.

## 2. LeJEPA — `SVM_LeJEPA` (init `trunc_normal`)

Gerado em **2 estágios** dentro do repo:

1. **Pré-treino SSL LeJEPA** (`LejepaLineModule`, loss `src/losses/lejepa_loss.py`), um experimento por
   `(dataset × split × pct × init)`. A `percentage` seleciona a fração de dados usada no pré-treino auto-supervisionado.
   - `python scripts/run_ssl_ray.py --inits trunc_normal --num-gpus 2 --max-concurrent-per-gpu 3`
2. **Avaliação SVM** (`src/evaluate/unified_eval.py`, `run_svm` em `:140-254`): carrega o ckpt LeJEPA, **congela** o encoder
   (`:182-188`), treina SVM linear sobre embeddings do `train`, prediz no `test`, e `aggregate_and_save` (`:407-486`) grava
   **um CSV por `init`** já agregado: `artifacts/SVM/{ds}/lejepa_pct_{pct}/metrics_SVM_{init}.csv`.
   - `python -m src.evaluate.unified_eval --model svm --dataset all`

- **`init`** = inicialização dos pesos do encoder LeJEPA **antes** do SSL (`trunc_normal`, `flim`, `he`, `xavier`, `random`).
  O baseline oficial usa **`trunc_normal`** (filtrado em `plot_comparison_flim.py::_filter`).
- **Normalização:** `load_lejepa_svm_metrics()` (`normalize_reports.py:153-190`), glob `*/*/metrics_SVM_*.csv`, `method="SVM_LeJEPA"`.
- **Colunas (já agregadas):** `model_type, dataset_short, pretrained_pct, init, n_splits, kappa, kappa_std, acc, acc_std, f1, f1_std`.

## 3. I-JEPA — `SVM_IJEPA`

- **Gerador:** `src/evaluate/svm_ijepa.py` (`main` em `:235-388`). Teacher I-JEPA ViT-H/14 (`facebook/ijepa_vith14_1k`, ~632M,
  **frozen**) via `IJEPAEncoder` (`src/models/ijepa_encoder.py`); inputs 224×224 ImageNet-norm; saída = patch embeddings
  mean-pooled `(B, 1280)`. Loop `DATASETS × SPLITS[1,2,3] × PCTS[1,5,25,50,75,100]`; SVM linear; `_save_aggregated`
  (`:173-229`) agrega média ± std sobre os 3 splits.
- **`pretrained_pct`** aqui **não** é pré-treino (o teacher é frozen) — é o **% de dados rotulados usados para treinar o SVM**
  (eixo x, consistente com as outras curvas).
- **Regenerar:** `python -m src.evaluate.svm_ijepa` (ou `--aggregate-only` para só re-agregar o raw).
- **Normalização:** `normalize_ijepa_svm()` (`normalize_reports.py:195-222`), `method="SVM_IJEPA"`, `init="ijepa"`.

---

## 4–7. Distill 1/2/3/4 — projection heads (encoder treinável, init `trunc_normal`)

Os 4 CSVs vêm do **mesmo avaliador**: `src/evaluate/svm_distill_with_projection.py`
(`python -m src.evaluate.svm_distill_with_projection`). O que distingue cada um é o par `--run-filter` / `--output-csv`.
Comandos canônicos em `scripts/regen_svm_queue.sh:42-45`.

**Infra comum:**
- `find_distillation_runs()` (`src/evaluate/svm_distillation.py:219`) varre `artifacts/distillation/`, casa runs por **substring**
  no nome da pasta (`--run-filter`), e seleciona o checkpoint por `val/knn_kappa` (probe kNN, `KnnKappaProbeMixin`,
  `src/models/distillation.py:503`), preferindo o mais treinado por `(epoch, global_step, -val/loss, mtime)`.
- `_load_student_and_proj()` (`svm_distill_with_projection.py:82`) carrega só `student.*` + `proj_kd.*` (sem o teacher) e
  **auto-detecta a arquitetura da proj head pelo shape dos pesos** (não por flag).
- **Embedding avaliado = 1280d** (proj head **ATIVA**): `student.encoder(x)` → `[B,48,H,W]` → `proj_kd(...)` → `[B,1280]`.
- **Treino dos checkpoints:** launcher Ray `scripts/distillation_conv_ray.py` (`--proj-type` escolhe a variante), que roda cada
  experimento como subprocess invocando um dos trainers Lightning em `src/modules/`.

| # | Curva | `--run-filter` | `--output-csv` | proj head (classe em `src/models/distillation.py`) | params | trainer |
|---|---|---|---|---|---|---|
| 4 | Distill 4 | *(base `next_layers_direct`)* | `svm_distill_proj1280_results` | `ConvDistillationProjectionHead` (4× conv 1×1: 48→128→256→512→1280, BN+ReLU) `:213` | ~889K | `distillation_conv_module.py` |
| 5 | Distill 3 | `3x3_BN2d_1280_one_layer` | `svm_proj1280_3x3_BN2d_results` | `OneLayerConvDistillationProjectionHead` (1× conv 3×3 BN2d + GELU) `:260` | ~615K | `distillation_onelayer_module.py` (`--proj-kernel 3`) |
| 6 | Distill 1 | `1x1_BN2d_1280_one_layer` | `svm_proj1280_1x1_BN2d_results` | `OneLayer1x1ConvDistillationProjectionHead` (1× conv 1×1 BN2d + GELU) `:302` | ~123K | `distillation_onelayer_module.py` (`--proj-kernel 1`) |
| 7 | Distill 2 | `2l_1x1_BN2d_256_1280` | *(derivado do filtro)* | `TwoLayer1x1ConvBN2dDistillationProjectionHead` (2× conv 1×1: 48→256→1280, BN+GELU) `:335` | ~402K | `distillation_twolayer_module.py` |

**Comandos de regeneração:**
```bash
python -m src.evaluate.svm_distill_with_projection --output-csv svm_distill_proj1280_results                                  # Distill 4
python -m src.evaluate.svm_distill_with_projection --run-filter 3x3_BN2d_1280_one_layer --output-csv svm_proj1280_3x3_BN2d_results   # Distill 3
python -m src.evaluate.svm_distill_with_projection --run-filter 1x1_BN2d_1280_one_layer --output-csv svm_proj1280_1x1_BN2d_results   # Distill 1
python -m src.evaluate.svm_distill_with_projection --run-filter 2l_1x1_BN2d_256_1280                                          # Distill 2
```

**Normalização** (todos `init=trunc_normal`):
- `_normalize_distill_proj1280()` (`:326`) — `status==ok`
- `_normalize_distill_3x3bn()` (`:370`) — `status==ok & encoder_init==trunc_normal` (as linhas `flim` do mesmo CSV viram `SVM_Distill_3x3BN_flim`)
- `_normalize_distill_1x1bn()` (`:414`) — `status==ok & encoder_init==trunc_normal` (linhas `flim` → `SVM_Distill_1x1BN_flim`)
- `_normalize_distill_2l_400k()` (`:686`) — `status==ok`

> ⚠️ Nota: a coluna `method` é `SVM_Distill_Proj1280` em **todos** os 4 CSVs, e `proj_head` é a string fixa
> `"conv_next_layers_active"` — nenhuma das duas reflete a arquitetura real. Quem distingue Distill 1/2/3/4 é o
> `normalize_reports.py` pelo **nome do arquivo de origem**.

## 8. Distill 1 (FLIM init) — `SVM_Distill_1x1BN_flim_frozen_eval_loss`

Variante com **encoder FLIM congelado** (init `flim`, sem backprop) + proj head 1×1 BN2d treinada por cima, **sem norma ImageNet**.

- **Gerador:** `src/evaluate/svm_distillation_conv.py`
  ```bash
  python -m src.evaluate.svm_distillation_conv --run-filter 1x1_BN2d_1280_flim_frozen --output-csv svm_distillation_conv_flim_frozen_results
  ```
- **Dois checkpoints → dois métodos** (`_discover_ckpts`, `:255-307`): para runs frozen cada run gera 2 linhas:

  | método (coluna `method`) | checkpoint | monitor | critério |
  |---|---|---|---|
  | `SVM_Distill_1x1BN_flim_frozen_eval_knn` | `best_knn_kappa.ckpt` | `val/knn_kappa` | best por kNN-kappa de validação (max) |
  | `SVM_Distill_1x1BN_flim_frozen_eval_loss` | `best_loss.ckpt` | `val/loss` | best por loss de validação (min, MSE/KD) |

  Os dois `ModelCheckpoint` são criados no trainer (`distillation_onelayer_module.py:429-440`). **O plot oficial usa só o
  `_eval_loss`** (rótulo "Distill 1 (FLIM init)").
- **Arquitetura:** encoder FLIM (`LeJEPAFLIMModel`, `src/models/lejepa_flim.py`, frozen `requires_grad=False`) +
  `OneLayer1x1ConvDistillationProjectionHead` (`distillation.py:302`). Embedding avaliado = proj 1280d (o encoder 48d é
  constante quando congelado).
- **Treino frozen + FLIM init** (`distillation_onelayer_module.py`): `--freeze-encoder` (`:194-217`), `--encoder-init flim
  --flim-weights-path <dir>`, `--proj-kernel 1`, `--no-imagenet-norm`, `--knn-probe projection`.
- **Normalização:** `_normalize_distill_flim_frozen()` (`normalize_reports.py:272`) — `status==ok`, `kappa` não-NaN, agrupa por
  `(method, dataset, percentage)` → cada método com `n_splits=3` (3 folds). `init=flim`.

---

### Como reconstruir tudo do zero

```bash
# 1) (re)gerar os CSVs de destilação a partir dos checkpoints em artifacts/distillation/
bash scripts/regen_svm_queue.sh           # roda os svm_distill_with_projection.py / svm_distillation_conv.py
# 2) (re)gerar os baselines LeJEPA e I-JEPA
python -m src.evaluate.unified_eval --model svm --dataset all
python -m src.evaluate.svm_ijepa
# 3) unificar tudo e plotar
python scripts/normalize_reports.py        # → artifacts/normalized/unified_svm_comparison.csv
python scripts/plot_comparison_flim.py --mode merge
```

> O FLIM (`SVM_FLIM`) **não** é regenerável aqui — depende dos CSVs externos em `data/reports_felipe/svm/`.
