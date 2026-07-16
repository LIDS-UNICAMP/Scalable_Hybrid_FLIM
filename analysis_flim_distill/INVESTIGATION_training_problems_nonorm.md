# Investigação — problemas no treino da destilação (flim-init, no-imagenet-norm)

*2026-06-09, 4 agentes em paralelo + curvas de treino do W&B. Read-only. Gatilho: o κ de eggs do `1x1` parecia maior que o do `2L`, o que não fazia sentido.*

## TL;DR — dois problemas independentes

**Problema 1 (pipeline / corrupção de métrica): checkpoint stale.** O eval SVM leu o checkpoint ERRADO em ≥17 células do grupo `1x1`. O número "eggs 1x1 κ=0.864" vem de um `best.ckpt` de **epoch 0** (encoder quase = FLIM-init puro, não destilado). O modelo `1x1` **treinado de verdade** (`best-v1.ckpt`, ep99) tem κ=**0.376**. Logo a premissa "1x1 ganha do 2L em eggs" é falsa: corrigido, **2L (0.772) >> 1x1 (0.376)**.

**Problema 2 (científico / treino degrada features): colapso induzido pela destilação no-norm.** Treinar PIORA o encoder: eggs 48d κ cai de **0.768 (FLIM-init) → 0.194 (1x1 treinado)**; effective rank 7.6 → **1.55** (quase rank-1). Causa: o teacher I-JEPA recebe **LAB-como-RGB sem ImageNet-norm** (alvo degradado) e a MSE arrasta o student para uma solução de baixo posto. A projection head infla a norma de saída para casar o teacher (loss↓, cosine↑) enquanto o encoder colapsa → **as curvas de treino parecem saudáveis, mas medem a head, não o encoder**.

---

## 1. Evidência das curvas de treino (W&B)

Acesso: W&B API (`ophira-ai/flim-ssl`, `run.scan_history()`, 100 épocas completas) + leitura offline do `wandb/run-*/run-*.wandb` via `wandb.sdk.internal.datastore` (fallback funcional).

Trajetória típica (eggs_2L `e07vygbi`, protozoan_2L `wbx5bs97`):
- **val/loss e train/loss**: descem suave, sem overfit (gap val−train ~0), sem spikes — curvas "exemplares".
- **val/cosine_sim**: sobe monotonicamente até ~0.81 (eggs) / 0.836 (protozoan).
- **val/student_emb_norm (encoder 48d)**: DESPENCA de ~28–35 (ep0) para o mínimo já na **ep3–6** (~2–5) e fica presa lá. Teacher ≈ 21.8.
- **val/student_proj_norm (saída 1280d da head)**: CRESCE de ~5 → 14–16 (aproxima do teacher).

**Anomalia central:** a loss continua caindo ~90 épocas DEPOIS de o embedding do encoder já ter colapsado (ep6). Toda a "melhora" pós-ep6 vem da head inflando a saída, não do encoder. `larvae` é a exceção (emb_norm fica ~0.77–1.16× teacher, não colapsa) — coerente com larvae ser o caso benigno.

## 2. Problema 1 — bug de seleção de checkpoint (corrompe resultados)

- **Bug:** `src/evaluate/svm_distillation.py:252-259` (`_pick`): ordena por `(exact, loss, -mtime)` com `exact = 0 if name in ("best.ckpt","last.ckpt") else 1`. Como nem `best.ckpt` nem `best-v1.ckpt` têm `loss=` no nome (ambos `inf`), o desempate é só pelo `exact` → **`best.ckpt` sempre vence `best-v1.ckpt`**, mesmo sendo epoch 0. O `run_metadata.json["best_checkpoint"]` aponta corretamente para `best-v1.ckpt`, mas o eval **ignora** esse campo. `svm_distill_with_projection.py` importa a mesma função → herda o bug.
- **Por que existe um best.ckpt epoch-0:** o `distillation_onelayer_module.py` (grupo 1x1) **não tem lógica de resume** (`:399-401`, `trainer.fit` sem `ckpt_path`), enquanto o `twolayer` tem (`:388-399`). Sequência: run #1 treina até ep99 e grava `best.ckpt`; estoura o disco (teacher bloat de 2.5 GB); run #2 é relançado, NÃO resume, começa do ep0, e o `ModelCheckpoint` (vendo `best.ckpt` já existir) renomeia o antigo para `best-v1.ckpt` e grava o novo em `best.ckpt`; run #2 também morre cedo (disco) → `best.ckpt` congela em ep0. O `strip_teacher_from_ckpt.py` (glob `best*.ckpt`) processou os dois, preservando o stale.
- **Escopo:** grupo `1x1` = **17 células contaminadas** (eggs todos os splits + larvae split1, eval no encoder ep0/ep1 não-treinado) + **4 células ausentes** (eggs s1p50, s2p1, s2p50, s3p50 — disk-full, nunca completaram). Grupo `2L` = **0 contaminadas** (resume correto, 54/54 células no ckpt treinado ep90-99).
- **Prova de que importa:** em eggs_split1_pct100, sum|Δpesos| ep0→ep99 = ~1745 no encoder e ~73k na head → o encoder treina (não é congelado). O κ 0.864 é do encoder ep0 ≈ FLIM-init.

## 3. Problema 2 — a destilação no-norm erode as features (encoder + head)

κ_linear (SVM linear) e effective rank, test split, no-imagenet-norm transform:

| dataset/head | ponto | κ 48d (effrank) | κ 1280d (effrank) |
|---|---|---|---|
| eggs / 1x1 | epoch0 (≈FLIM-init) | 0.741 (7.6) | 0.864 (34) |
| eggs / 1x1 | **treinado ep99** | **0.194 (1.55)** | **0.376 (3.1)** |
| eggs / 2L | treinado ep94 | 0.466 (3.2) | 0.772 (11) |
| proto / 1x1 | treinado | 0.260 (5.2) | 0.536 (28) |
| proto / 2L | treinado | 0.397 (5.7) | 0.636 (48) |

Encoder 48d puro (FLIM-init, sem treino): eggs **0.768**, protozoan **0.497**. → treinar derruba o encoder em todos os casos (eggs 1x1 0.768→0.194; eggs 2L 0.768→0.466; proto 0.497→0.26/0.40).

- **Colapso é do ENCODER e da HEAD** (não só "projection-head dominance"): o effective rank do encoder 48d desaba (eggs 1x1 7.6→1.55). Isto **refina** a tese antiga ("o teacher não erode o FLIM") — sob `no_imagenet_norm` + MSE direct, ele erode.
- **Head maior amortece o dano:** 2L (48→256→1280) preserva mais estrutura que 1x1 (48→1280): eggs 48d 0.466 vs 0.194; 1280d 0.772 vs 0.376. A head 1x1 tem pouca capacidade para absorver o alvo ruim → o gradiente joga a erosão direto no encoder.
- **Causa-raiz (confirmada no código):** `prepare_teacher_input` (`src/models/distillation.py:396`) só faz resize bilinear; `IJEPAEncoder` (`src/models/ijepa_encoder.py`) exige ImageNet-norm. Com `--no-imagenet-norm`, o teacher recebe LAB-como-RGB em [0,1] → alvo degradado → a MSE força o student a imitar lixo → colapso de baixo posto.

## 4. Outras anomalias operacionais
- **Não-determinismo:** `deterministic=False` no Trainer (twolayer:385 / onelayer:396), apesar de `seed_everything(42)`. Runs de mesma config dão val/loss diferentes (ex.: larvae_s1_pct5 0.276 vs 0.285).
- **Teacher bloat** (2.5 GB/ckpt) ainda presente em ckpts antigos (ex.: eggs_s2_pct1 `last.ckpt` = 2.5 GB) — causa dos disk-full.
- **Churn de re-runs:** 2L com 4–5 `wandb/run-*` (maioria crashes de startup); só o último treinou.

## 5. Recomendações (ordem)
1. **Corrigir o `_pick`** (`svm_distillation.py:252-259`): preferir `run_metadata["best_checkpoint"]`; fallback por MAIOR epoch/step (carregar via torch), não pelo nome `best.ckpt`. Depois **re-rodar o SVM do grupo 1x1** (as 17 células contaminadas). Limpar os `best.ckpt` stale ep0.
2. **Corrigir o caminho de normalização do teacher** (causa do Problema 2): tirar ImageNet-norm do student E aplicá-lo DENTRO de `prepare_teacher_input`, só no ramo do teacher. Predição: para de degradar o alvo → encoder deixa de colapsar. É o fix já apontado em `REPORT_why_protozoan_collapses.md §5`.
3. Adicionar resume ao `distillation_onelayer_module.py` (igual ao twolayer) e `deterministic=True` (ou documentar a tolerância). Garantir teacher fora do save (já feito via `FrozenTeacherCheckpointMixin`).
4. **Não usar val/loss como proxy de qualidade** — monitorar κ/effective-rank no encoder 48d, ou `student_emb_norm` (sinal de colapso).

## Fontes
- Curvas: W&B `ophira-ai/flim-ssl` (run ids e07vygbi/wbx5bs97/349pgqgz/fiv7rlvm/jy81o106/wzq5s10m); `wandb/run-*/run-*.wandb`.
- Bug: `src/evaluate/svm_distillation.py:252-259`; módulos `src/modules/distillation_{onelayer,twolayer}_module.py` (resume 388-399 vs 399-401).
- Mecanismo: `src/models/distillation.py:396`, `src/models/ijepa_encoder.py`, `_teacher_emb` nos módulos.
- κ/rank: ckpts em `artifacts/distillation/distillation_{eggs,protozoan}_split1_pct100_*`; CSVs `results/svm_proj1280_1x1_BN2d_1280_one_layer_flim_init_no_imagenet_norm_results.csv`, `results/svm_2l_1x1_init_flim_256_1280_nonorm_proj1280.csv`.
