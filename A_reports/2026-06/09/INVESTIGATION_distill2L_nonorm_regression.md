# Investigação — "Distil 2L 1×1 400K sem norm": por que os resultados parecem inconsistentes

*Investigação 2026-06-09, 4 agentes em paralelo (código/git, config/hparams, resultados/métricas, documentação). Read-only. Pergunta do usuário: o "Distil 2L 1×1 400K sem norm" aparenta desempenho superior/inconsistente vs o último experimento coerente, o que não faz sentido dadas as mudanças do pipeline.*

## TL;DR

1. **Não há bug de código que degrade o treino.** As mudanças de arquitetura desta sessão (`ProjectionHeadHawk`, `FrozenTeacherCheckpointMixin`, linear-48) são **uncommitted e inertes** (rename comentado; mixin equivalente ao código antigo; linear-48 revertido). O modo `direct` treina só `encoder + proj_kd` e **nunca** usa o `multi_layer_perceptron` do student → o rename não toca o encoder. (Agente 1)
2. **A "superioridade" é majoritariamente artefato**, por dois motivos:
   - **Régua trocada no plot de comparação**: a curva nova (encoder **48d**) é plotada contra o baseline trunc 400k lido em **1280d** — exatamente o "ruler mismatch" que a própria doc classifica como measurement artifact. (Agente 4)
   - **Outlier no baseline**: o único ganho real aparente (larvae pct100, +0.27) vem de um **split colapsado** no baseline antigo (κ=0.116), não de o novo ser melhor. (Agente 3)
3. **A degradação real existe e é metodológica, não bug**: NEW (2L, sem-norm) é **pior** que OLD (2L, com-norm) em eggs (−0.19) e protozoan (−0.13). É o **conflito de normalização de duas pontas já documentado**: sem ImageNet-norm o teacher recebe LAB-como-RGB → alvo degradado; o student casa melhor em MSE (val/loss igual/menor) mas as features 48d ficam piores no SVM (inversão MSE↓→κ↓). (Agentes 3 e 4)

## 1. Lista cronológica das modificações (com evidência)

| Data | Commit/Status | Mudança | Arquivo | Impacto |
|---|---|---|---|---|
| 30/mai | `f9ff52c` | `load_FLIM_encoder_from_arch_dict` + `PROTOZOAN_FLIM_ARCH` (aditivo) | `models.py:339-405` | Afeta init FLIM do protozoan; eggs/larvae inalterados |
| 02–04/jun | `5f67d6c` | **Baseline coerente** treinada (regime com ImageNet-norm) | — | É o "antes". `git_sha=5f67d6c` |
| 05/jun | `b9c72d8` | `filename="best-{epoch}-{val/loss}"` → `"best"`; regen svm queue | `distillation_*layer_module.py` | Neutro (só nome do ckpt) |
| 05/jun | `7f637c6` | Split por `encoder_init` em memória no plotting | `eval_plotter.py`, `normalize_reports.py`, `plot_comparison_flim.py` | Afeta só como os números aparecem nos relatórios |
| 07/jun | `9e5bce1` | **Caminho `imagenet_norm=False`** (treino+eval); `on_save_checkpoint` exclui teacher; SVM `--no-imagenet-norm`/`--only-ok`/`map_location=cpu` | `lejepa_dataset.py:30-58`, `parasite_data_module_lejepa_splited.py:118-148`, `distillation_*layer_module.py`, `svm_distillation_conv.py` | **Única mudança com impacto nos pesos/embeddings.** Define este grupo experimental. NEW treinado 07–09/jun aqui |
| 08–09/jun | execução | **Re-lançamento por cima de runs já concluídos** (até 5 `wandb/run-*` por run; resumes de ckpt em ep99/100 rodando 11min–1.5h) | `artifacts/distillation/*_no_imagenet_norm/` | Churn operacional; best.ckpt=last.ckpt (md5) em 51/54; best é early-best (ep 67–99) |
| sessão atual | uncommitted | `ProjectionHeadHawk` (não instanciada), bloco `_hawk` **comentado**, `FrozenTeacherCheckpointMixin` | `lejepa.py`, `lejepa_flim.py`, `distillation.py` | **Inerte** — não ativo nos retreinos (`9e5bce1`); não afeta `student.*` |

## 2. Hipóteses ordenadas por probabilidade

### H1 — Régua trocada no plot de comparação (ALTA) — explica o "parece inconsistente/superior"
No `plot_comparison_flim.py`, as três curvas "Distil 2L 400K" vêm de réguas diferentes (mapeamento em `normalize_reports.py`, verificado pela coluna `method` de cada CSV):

| Curva | CSV fonte | Régua |
|---|---|---|
| `SVM_Distill_2l400K` (trunc) | `svm_proj1280_2l_1x1_BN2d_256_1280_results.csv` | **1280d** (proj) |
| `SVM_Distill_2l400K_flim` | `svm_2l_1x1_init_flim_256_1280_results.csv` | **48d** (encoder) |
| `SVM_Distill_2l400K_flim_nonorm` (NOVO) | `svm_2l_1x1_init_flim_256_1280_nonorm_encoder48.csv` | **48d** (encoder) |

A curva nova (48d) é comparada visualmente com um baseline trunc em 1280d → maçã-com-laranja. A doc (`INVESTIGATION_flim_init_distillation.md`, ADDENDUM) já classificou isso como artefato de medição. **Evidência:** coluna `method`/`embed_dim` dos CSVs; `normalize_reports.py` sources.

### H2 — "Ganho" no larvae é outlier do baseline (ALTA)
Único lugar onde NEW parece claramente superior = larvae pct100 (Δ +0.268). Per-split:
- OLD (2L, com-norm): **0.899 / 0.116 / 0.872** ← split-2 colapsado puxa a média para 0.629.
- NEW (2L, sem-norm): 0.900 / 0.892 / 0.898 (estável).

A "vitória" é o baseline degenerado, não NEW melhor. **Evidência:** `svm_2l_1x1_init_flim_256_1280_results.csv` (larvae pct100, por split). (Agente 3)

### H3 — Degradação real por conflito de normalização (ALTA, é o efeito verdadeiro)
NEW (2L, sem-norm) é **pior** que OLD (2L, com-norm) em eggs (Δ −0.19) e protozoan (Δ −0.13); larvae ~flat (o +0.05 é outlier). Mean κ NEW=0.499 vs OLD_2L_norm=0.587 (Δ **−0.088**).
Mecanismo documentado (`REPORT_why_protozoan_collapses.md` §3): student e teacher compartilham o mesmo tensor; sem ImageNet-norm o **teacher** recebe LAB-como-RGB → alvo degradado.
- `val/loss` NEW=0.2061 ≈ OLD=0.2060 (igual/menor para NEW por célula), mas κ pior → **inversão MSE↓→κ↓**.
- `val/student_emb_norm` colapsa **40–59 (com-norm) → 12–15 (sem-norm)**; teacher_emb_norm inalterado (~21). O student imita melhor em MSE/cosine, mas o encoder 48d resultante é mais fraco. **Evidência:** `wandb-summary.json` dos runs novos vs antigos. (Agente 3 §3)

### H4 — Churn de re-run/resume (BAIXA-MÉDIA)
O grupo foi re-disparado por cima de runs concluídos; o `best.ckpt` avaliado é o early-best (ep tão baixos quanto 67/70/73), variando run-a-run. Provavelmente ruído menor (gstep idêntico nos resumes → pesos ~inalterados), mas adiciona inconsistência run-a-run e dificulta reprodução. **Evidência:** múltiplos `wandb/run-*`, runtimes 30–100× distintos por célula, best=last md5 em 51/54. (Agente 2 §3)

### Descartado — mudanças de código de arquitetura/loader
`hawk`/mixin/linear-48: inertes/uncommitted/revertidos; SVM loader inalterado (`strict=True` em `student.*`); `load_FLIM_encoder` byte-idêntico para eggs/larvae. **Não** são causa. (Agente 1)

## 3. Comparação direta: melhor anterior vs último (κ, média de 3 splits, régua encoder-48d)

| dataset | pct | NEW (2L, sem-norm) | OLD (2L, com-norm) | Δ |
|---|---|---|---|---|
| eggs | 5 | 0.268 | 0.431 | **−0.163** |
| eggs | 25 | 0.309 | 0.597 | **−0.288** |
| eggs | 50 | 0.401 | 0.652 | **−0.252** |
| eggs | 100 | 0.570 | 0.773 | **−0.203** |
| larvae | 75 | 0.894 | 0.796 | +0.098 |
| larvae | 100 | 0.897 | 0.629* | +0.268* (*outlier OLD) |
| protozoan | 25 | 0.180 | 0.441 | **−0.261** |
| protozoan | 50 | 0.332 | 0.544 | **−0.212** |
| protozoan | 100 | 0.594 | 0.556 | +0.038 |

Mean κ: **NEW 0.499 vs OLD_2L_norm 0.587 (Δ −0.088)**. Por dataset: eggs −0.192, protozoan −0.125, larvae +0.053 (inflado por outlier). Fonte: `svm_2l_1x1_init_flim_256_1280_nonorm_encoder48.csv` vs `svm_2l_1x1_init_flim_256_1280_results.csv`.

## 4. Aderência metodológica (Agente 4)

- Implementação **aderente**: FLIM init real (`distillation_twolayer_module.py:145-150`), flag no-norm wirada corretamente, CSV novo na régua certa (encoder48), conflito de norm permanece como documentado (`prepare_teacher_input` só faz resize — fix pendente).
- **Divergência principal**: a régua trocada (H1) foi reintroduzida no plot — contra a recomendação explícita de `INVESTIGATION_flim_init_distillation.md` §8.2.
- **Docs desatualizados**: `distillation_model_architecture.md:26-27` ("FLIM weights não carregados" — falso); `distillation.py:399-400` docstring ("views already ImageNet-normalised" — falso p/ no-norm); `analises_wandb_training/section_flim_init_comparison.md` (concl. superada pelo ADDENDUM).

## 5. Recomendação (ordem de prioridade)

1. **Não há código para reverter** — a degradação não é regressão de software. Foco: comparação e metodologia.
2. **Corrigir a régua antes de qualquer conclusão** (limpeza pedida): comparar apenas experimentos equivalentes — **FLIM init + sem-norm + MESMA régua (encoder 48d)**. Hoje, mesmo dentro do grupo "flim no-norm", o `SVM_Distill_1x1BN_nonorm` vem de um CSV **proj1280**, enquanto o 2L vem de encoder48 → ainda é régua trocada. É preciso gerar/usar todos no encoder48.
3. **Se o objetivo é que "sem-norm" não piore eggs/protozoan**: aplicar o fix documentado — tirar ImageNet-norm do student E aplicá-lo dentro de `prepare_teacher_input` (só no ramo do teacher). Predição da doc: protozoan/eggs deixam de perder e o ganho do student se mantém.
4. **Higienizar execução**: garantir treino do zero (apagar `last.ckpt` antes) em vez de resume sobre runs concluídos, para reprodutibilidade.
5. Atualizar docs desatualizados (§4).

## Fontes
- Código: `src/modules/distillation_twolayer_module.py`, `distillation_onelayer_module.py`, `src/models/{distillation,lejepa_flim,lejepa,models}.py`, `src/data_modules/datasets/lejepa_dataset.py`, `src/evaluate/svm_distillation_conv.py`.
- Resultados: `results/svm_2l_1x1_init_flim_256_1280_nonorm_encoder48.csv`, `svm_2l_1x1_init_flim_256_1280_results.csv`, `svm_nonorm_1x1_encoder48_results.csv`, `artifacts/normalized/unified_svm_comparison.csv`, `artifacts/distillation/*_no_imagenet_norm/{run_metadata.json,wandb/*/files/wandb-summary.json}`.
- Docs: `analysis_flim_distill/INVESTIGATION_flim_init_distillation.md`, `REPORT_why_protozoan_collapses.md`, memórias do projeto.
