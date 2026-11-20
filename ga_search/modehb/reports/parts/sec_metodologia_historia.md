# Metodologia e história das técnicas (Hybrid SPiFiL / MO-DEHB / LEMONADE / PNAS-CS)

_Lido do código, das configs e dos MDs em 16/11/2026. Todas as citações são `arquivo:linha` relativas à raiz do repositório. Não há números novos aqui: os valores vêm dos MDs citados._

---

## 1. Glossário de siglas

### 1.1 Buscas (técnicas nossas)

| Sigla | Nome | O que é | Config / código | Saída |
|---|---|---|---|---|
| **H-MODEHB** | Hybrid SPiFiL MO-DEHB estratificado | MO-DEHB (Awad et al. 2023) multiobjetivo, κ × FLOPs, com população estratificada por parâmetros. Cada candidato tem init SPiFiL, depois AE local e depois SSL global (LeJEPA+SIGReg). | `ga_search/config/modehb/search_v3.yaml`; `ga_search/modehb/modehb.py:1`, `strata.py:1` | `artifacts/modehb_hybrid/<base>` (antes `artifacts/modehb/`) |
| **H-LEM** | Hybrid SPiFiL LEMONADE | Igual ao H-MODEHB, mais: warm start com 3 sementes de receita, morfismos de rede como mutação e herança lamarckiana (Elsken et al. 2019). | `search_v3_lemonade.yaml:` bloco `lemonade:`; `ga_search/modehb/lemonade.py:1-24` | `artifacts/modehb_lemonade/<base>` |
| **H-SEED** | Busca semeada | H-MODEHB e H-LEM semeados com o encoder externo "SPiFiL pequeno + LeJEPA" e uma grade de crescimento que preserva a função. Estratos vão até xl ≤ 5,5M. Duas variantes: H-SEED-MODEHB e H-SEED-LEM. | `search_v3_seeded.yaml`, `search_v3_lemonade_seeded.yaml`; `external_seed.py:1-22` | `artifacts/modehb_seeded/<base>`, `artifacts/modehb_lemonade_seeded/<base>` |
| **PNAS-CS** | Progressive NAS + Compound Scaling | MO-DEHB sobre famílias (base, α, β), com níveis P0–P4 de 0,5/1/2/3,5/5,5M, Network Morphism com herança e Fase 0 de ablação do AE. | `pnas_cs.yaml`; `pnas_cs.py:1-24` | `artifacts/pnas_cs/<base>` (athena GPU 2) |
| *_C01_* | Buscas arquivadas | H-MODEHB/H-LEM rodados com o fitness antigo (C=0,1, kw=0,75). Também são avaliados no teste. | `monitor_report.py:22-23`; `FILA_E_INSTRUCOES.md:229` | `artifacts/modehb_{hybrid,lemonade}_C01_*` |

### 1.2 Encoders externos (equipe CosFace / discovery)

| Sigla | O que é | Fonte |
|---|---|---|
| **SPF-P-L0** | "SPiFiL pequeno" puro (FPS, sem SSL). RegNet 40-80-152-152-152-312, conv k3, 0,98M parâmetros. | `FILA_E_INSTRUCOES.md:28`, `:39-44` |
| **SPF-P-LeJEPA** | SPF-P-L0 + SSL LeJEPA (estágio L6), checkpoint `best_kappa.ckpt`. É a semente do H-SEED e do PNAS-CS. | `search_v3_seeded.yaml` bloco `external_seed:`; `FILA_E_INSTRUCOES.md:28` |
| **SPF-P-CosFace**, **SPF-P-SimCLR** | O mesmo encoder pequeno com outro SSL/perda da sessão CosFace. Comparados na régua igual do remote_sensing. | `FILA_E_INSTRUCOES.md:93-95`; `ga_search/modehb/reports/remote_sensing_regua_igual.md` |
| SPF-P-*-trainonly | Semente do ham10000 retreinada só no nosso treino. A versão antiga viu o nosso val. | `FILA_E_INSTRUCOES.md:29`, `:44`, `:113` |

### 1.3 Leituras (réguas) — nunca comparar entre elas

| Sigla | Definição | Fonte |
|---|---|---|
| **ORIG** | "SVM original". `core.svm.fit_svm` (SVC linear, **C=100**, ovo, `max_iter=-1`, **sem scaler**) no embedding `head(encoder(x))`, com 1280 dimensões, ou 2048 no LEMONADE c00098. Mesmo protocolo dos FMs congelados. **É o número reportado.** | `monitor_report.py:4-8`, `:168`; `FILA_E_INSTRUCOES.md:10-11` |
| **COV** | `cov_head`: leitura do fitness. Para cada bloco: PCA top-k=64 da covariância de canais → por imagem, sqrtm (eigh fp64) da covariância k×k → triângulo superior. Depois anexa o embedding do head, aplica signed-sqrt+L2 por grupo e StandardScaler, e roda o SVM com C=100. No teste é só uma coluna extra (com `MONITOR_COV=top`, só para o melhor por validação de cada estrato). | `pooling_head.py:321-338`; `monitor_report.py:173-175`; `FILA_E_INSTRUCOES.md:216` |
| **GTP** | `gem_tta_pw`: régua R-SPF da discovery. GeM p=3 do último bloco, média nas 8 vistas rot90×flip, `pw_whiten` (PCA-whitening com shrinkage 0,1) + L2, sem scaler, SVM C=100. É a régua do "SPiFiL híbrido" de outubro. | `ga_search/modehb/reports/remote_sensing_regua_igual.md:10`; `A_reports/october/report_halloween_october.md:26-28` |
| 48d / 1280d | Réguas antigas da destilação: encoder GAP com 48 dimensões (`SVM_Distillation_Conv`) e projeção com 1280 dimensões (`SVM_Distill_Proj1280`). | `CLAUDE.md` ("Rulers"); `results_from_mds.md:5` |

### 1.4 Foundation models (FMs, baselines congelados)

AIMv2, DINOv2, I-JEPA (ViT-H/14, 1280 dimensões), ResNet-50 e ConvNeXtV2-pico. Todos são avaliados congelados, com SVM C=100 sem scaler (o mesmo protocolo da ORIG). Nas bases *_raw usam o split `split1_modehb_raw` (`FILA_E_INSTRUCOES.md:35`).

### 1.5 Baselines antigos do repo (famílias de `results_from_mds.md`)

| Família (nome em `results_from_mds.md`) | Exemplo / régua | Fonte |
|---|---|---|
| **FLIM** (SVM_FLIM, Felipe; "SPiFiL" nas tabelas `report_results_only_spifil*` = FLIM sem treino) | eggs pct100 κ 0,885 | `ga_search/modehb/reports/results_from_mds.md:20-21`, `:481` |
| **LeJEPA-SSL** (SVM_lejepa_view; LeJEPA flim init MLP_unfreeze) | 0,3142 (flatten 27648d); 0,929 (MLP-U) | `results_from_mds.md:23-24` |
| **Distill-IJEPA-FLIM 48d / 1280d** (Distill1 1x1BN, Distill2 2l400K, Distill3 3x3BN, Distill4 next_layers) | 2L 400k: 0,715 (48d) × 0,889 (1280d) | `results_from_mds.md:27-28`; `analysis_flim_distill/INVESTIGATION_flim_init_distillation.md:15-16` |
| **SPiFiL-hibrido** (bp01w64c1280, e_bp10, régua GTP) | eggs 0,967 | `results_from_mds.md:22`; `A_reports/october/report_halloween_october.md:325` |
| **FM-IJEPA** | eggs 0,96 | `results_from_mds.md:36-37` |
| **Supervisionado / MLP-U** (MLP ReLU unfrozen, Felipe) | eggs 0,947 | `results_from_mds.md:38` |

---

## 2. Metodologia (lida do código e das configs)

### 2.1 Espaço de busca (genoma)

- O genoma é um vetor contínuo em [0,1]^DIM: genes globais + 8 camadas × 8 genes. Os genes categóricos são decodificados como `choices[min(int(v·n), n−1)]`. `depth` mascara as camadas inativas. A semente da busca não é gene (`modehb.py:1-13`).
- **Genes globais** (`modehb.py:109-112`): init {spifil, random}, depth, ssl {simclr, lejepa, ae}, reg {none, sigreg, weak_sigreg}, embed_dim {1280, 1536, 2048, 2560}, pooling, ae_target {pixel, feature, masked}, spifil_images_per_class, projector, protect {full, lowlr, freeze}, ae_schedule {joint, decoder_first}.
- **Genes por camada** (`modehb.py:113-114`): channels, kernel, conv, residual, activation, norm, pool, softhebb.
- **Espaço restrito pelo YAML** (`search_v3.yaml:45-46`, `:65-89`; `operators_from_start` e `search_space_from_yaml`):
  - operator {conv, dwsep, inception}, softhebb {T, F}, kernel {3, 5, 7};
  - activation {relu, gelu, silu, selu};
  - norm {none, batch, zca, dbn, batch_zca, batch_dbn}, pool {none, max, avg}, residual;
  - pooling final {gap, gem, moments}, protect, projector, ae_schedule;
  - canais inteiros de 1 a 1024 em escala log, monotônicos (`search_v3.yaml:26-31`);
  - spifil_images_choices [1, 2, 4, 8, 16, 32] (`search_v3.yaml:20`).
- **`repair`** (`modehb.py:15-19`): embed_dim ≥ 1280; pointwise ⇒ k1; dwsep fora da camada 0; residual só com cin=cout e sem pool; os pools são limitados para o mapa final ficar ≥ `min_spatial`; com init spifil, toda camada passa por `init.repair_for_spifil`.
- **Cabeça** (`pooling_head.py:1-24`): conv 1×1 treinável → pooling (gap / gem p=3 / gem_learn / meanstd / moments / combo) → embedding de dimensão exatamente `embed_dim`. O pooling roda em fp32 e o GeM faz clamp(min=eps).
- **DE**: rand/1/bin, F=0,5, CR=0,5, ε=0,1 de amostra uniforme (`search_v3.yaml:59-63`).

### 2.2 Estratos e teto de parâmetros

- A contagem é o encoder + a projeção 1×1 final: estimativa estrutural (device `meta`) antes da avaliação, reclassificação pelo valor medido depois (`strata.py:3-12`; `STRATIFIED.md:9`).
- **H-MODEHB / H-LEM**: Tiny ≤ 300k < Small ≤ 800k < Medium ≤ 1,5M < Huge ≤ 3,5M; teto 3,5M (medido acima → `infeasible=param_cap`) (`search_v3.yaml:94-101`; `STRATIFIED.md:11-18`).
- **H-SEED**: small ≤ 0,8M / medium ≤ 1,5M / large ≤ 3M / xl ≤ 5,5M; teto 5,5M (`search_v3_seeded.yaml` bloco `stratified`; `FILA_E_INSTRUCOES.md:82-84`).
- **PNAS-CS**: os níveis P0..P4 (0,5 / 1 / 2 / 3,5 / 5,5M, tolerância 10%) são os estratos; teto 5,5M × 1,05 (`pnas_cs.py:10-16`; `pnas_cs.yaml` bloco `pnas_cs:`).
- **Evolução estratificada** (`STRATIFIED.md:23-32`):
  - categoria-alvo em rodízio;
  - pais da mesma categoria com 70% de chance (`cross_category_prob 0.3`);
  - projeção `fit_to_stratum` com p=0,7 (`strata.py:91`);
  - cota justa por categoria;
  - 16 sementes em pirâmide (filtros 16/32/64/128 × prof. 2/3, `pyramid_growth 2.0`, `search_v3.yaml:48-56`).
- **Promoção protegida** (`strata.py:159`; `STRATIFIED.md:37-42`):
  - nos 5 primeiros brackets, ≥25% das vagas por categoria (`min_share`), com ranqueamento não dominado dentro da categoria;
  - +1 promoção exploratória para Medium/Huge (large/xl, P2–P4) com inclinação > 0 (`learning_slope`, `strata.py:151`);
  - depois disso, promoção global;
  - HV final global, só na fidelidade máxima.
- **Reparos**:
  - `mem_repair` (orçamento de 60 GB reais; 40 no PNAS-CS) (`strata.py:310`);
  - `spifil_images_repair` (N = 100 superpixels × classes × imagens > D) (`strata.py:435`; `STRATIFIED.md:52-64`);
  - cabeça 1×1 processada em pedaços (`STRATIFIED.md:54-56`).

### 2.3 Fidelidades e orçamento

- Fidelidades: **5 / 15 / 45 / 100 / 200 épocas**, η=3 (`search_v3.yaml:34-36`).
- `max_brackets`: 10 (H-MODEHB, H-SEED-MODEHB) e 4 (H-LEM, H-SEED-LEM, PNAS-CS) (diffs dos YAMLs, linha 20 das configs filtradas).
- Divisão AE:SSL = 1:2 dentro de cada fidelidade, com o AE arredondado para baixo (`search_v3.yaml:37-40`; `run_search.py:30`, `:666`).
- 1 "época" = ≤ 2048 imagens sorteadas (32 passos de bs64) (`search_v3.yaml` bloco `training`).
- Promoções retomam o checkpoint da fidelidade anterior.
- Calibração: 1 representante por categoria passa por todas as fidelidades antes do bracket 0 (`STRATIFIED.md:47`).

### 2.4 Inicialização SPiFiL

- SPiFiL é o inicializador primário. `random` existe só como gêmea de baseline e está desligado (`random_baseline: false`, `search_v3.yaml:49`; `init.py:1-4`).
- `flim.progressive.fit_new_block` (não modificado) só ajusta conv densa k3/k5 sequencial. `repair_for_spifil` mapeia o resto (`init.py:6-25`).
- Com `spifil_ops: true`, `spifil_ops.py` estende o ajuste para 1×1, 7×7, dwsep e inception (`spifil_ops.py:1-23`; `search_v3.yaml:16`).
- Mecânica (`spifil_ops.py:9-21`):
  - patches nos superpixels;
  - FisherScorer + Mahalanobis (Ledoit-Wolf);
  - alocador por classe e seleção por diversidade;
  - z-score e normalização unitária dos filtros.
- Os doadores são as primeiras `spifil_images_per_class` imagens de treino por classe (`run_search.py:16-17`).

### 2.5 AE local (Fase B)

- O prefixo fica congelado. Treinam só o bloco novo e um decoder temporário, que é descartado no fim (`local_ae.py:1-15`).
- Alvo: pixel, feature ou masked.
- `protect` multiplica o lr das convs SPiFiL: full 1 / lowlr 0,01 / freeze 0 (`modehb.py:102-103`; `local_ae.py:144-159`).
- `decoder_first` congela o bloco na primeira metade das épocas, treinando só o decoder; depois o bloco entra (`modehb.py:106-107`; `local_ae.py:153-154`).

### 2.6 SSL global (Fase C)

- `train_global_ssl` descongela tudo e usa AdamW com dois grupos: lr_old = 1e-4 para as camadas herdadas, lr_new = 1e-3 para a camada nova + head (`ssl_trainers.py:7-10`; `run_search.py:65`, `:704-714`).
- `lejepa` = invariância + **RealSIGReg (Epps–Pulley)**, com perda total `λ·SIGReg + (1−λ)·inv`, λ = 0,05 (`ssl_trainers.py:15`, `:395`; `run_search.py:65`).
- V = 4 vistas para o LeJEPA (`ssl_trainers.py:293`).
- No PNAS-CS, o espaço fixa ssl = lejepa e reg = sigreg, com autocast bf16 e operações sensíveis em fp32 (`pnas_cs.yaml`: `ssl: [lejepa]`, `amp_dtype: bf16`).
- O projetor/decoder é descartado depois do SSL.

### 2.7 protect e WiSE (garantia "o SSL só soma ao SPiFiL")

- `ElementProtect` aplica a escala do `protect` só aos elementos herdados (exato para Adam/AdamW). Os filtros novos treinam livres (`external_seed.py:20-22`; `run_search.py:709-711`).
- **WiSE-FT** (`run_search.py:718-725`, `:806-850`):
  - depois do SSL, avalia `(1−α)·âncora_SPiFiL(+AE) + α·SSL` para α ∈ {0, 0,25, 0,5, 0,75, 1} (`search_v3.yaml:142-143`);
  - mantém o α de maior quality na **validação**;
  - α=0 é o próprio SPiFiL, então o backprop não piora o SPiFiL na validação.

### 2.8 Fitness (seleção)

- `fitness_mode: svm_kappa` (`search_v3.yaml:136`).
- `svm_fitness: {C: 100, scaler: standard, max_train: 4000, kappa_weight: 1.0, solver: libsvm, readout: cov_head, cov_k: 64, tta: 1, wise_alphas: [...]}` (`search_v3.yaml:142-143`; idem em `search_v3_lemonade.yaml:145-146` e `pnas_cs.yaml:145-146`).
- `svm_fitness` usa `core.svm.fit_svm` e calcula `quality = clip(kw·max(κ,0) + (1−kw)·F1)`. Com kw = 1, quality = κ de validação (`supervised_fitness.py:1-6`, `:101-109`, `:162`, `:175`).
- O treino é subamostrado de forma estratificada (≤ 4000). A validação é inteira (`val_max_samples: null`, `search_v3.yaml:10`). O teste nunca é tocado (`supervised_fitness.py:5-6`).
- **Inconsistência documental**: o comentário em `search_v3.yaml:136` e o texto em `STRATIFIED.md:19` ainda dizem "0,75·κ + 0,25·F1". O valor efetivo é `kappa_weight: 1.0`.
- A assinatura do fitness (`C<C>_kw<kw>`) faz parte da identidade do controller W&B, para não misturar curvas C=0,1 e C=100 (`run_search.py:1161-1164`).
- **Colapso** (std < 1e-4, rank efetivo < 5%, cosseno médio > 0,95, NaN) → quality = 0 (`search_v3.yaml` bloco `collapse`).

### 2.9 Objetivos

- (max quality, min FLOPs/MACs).
- HV com ponto de referência (quality 0; FLOPs normalizados 1,1), FLOPs log-normalizados por `max_flops_in_space` = 1e11 (`search_v3.yaml:118-128`; `modehb.py:693`).
- Parâmetros e latência são registrados, mas não são objetivos.

### 2.10 Técnicas específicas

- **H-LEM**:
  - 3 sementes de receita: seed_B 40-80-152-152-152-312 k3 zca gem; seed_B_wide; seed_A 32-48-64 k5;
  - 5 descendentes (8 no H-SEED-LEM);
  - morph_prob 0,5, com operadores deepen/widen/residual/kernel;
  - herança lamarckiana do pai com fidelidade ≤ a do filho (`search_v3_lemonade.yaml` bloco `lemonade:`; `lemonade.py:74`, `:124`, `:190`, `:226`).
- **H-SEED**:
  - `load_external_seed` converte o FLIM (conv k3 + relu + MaxPool(3, s2), `pool_padding 0`) em `SpifilOpsBlock`s (`external_seed.py:123`; `STRATIFIED.md:139`);
  - grade: largura ×{0,75, 1, 1,25, 1,5, 2, 2,5} × profundidade +{0, 1, 2} + variante residual zero/silu (`search_v3_seeded.yaml` bloco `grid`);
  - morfismos Net2Wider/Net2Deeper (`widen_block`, `identity_block`, `residual_zero_block`, `grow_blocks`: `external_seed.py:357-444`);
  - filtros novos = SPiFiL + AE local;
  - protect lowlr/freeze nos herdados e âncora WiSE.
- **PNAS-CS** (`pnas_cs.yaml` bloco `pnas_cs:`; `pnas_cs.py:252`, `:287`, `:429`, `:584`):
  - α ∈ {1,0 … 1,4} (profundidade), β ∈ {1,05 … 1,3} (largura), resolução fixa em 224;
  - filho de crescimento P_k → P_{k+1} com p=0,4, mutação (add_layer/add_residual/widen/pooling) com p=0,3, DE no restante;
  - sementes = SPF-P-LeJEPA + checkpoints do GA (c00004, c00010, c00098 do `_C01_final`) + o melhor por **validação** de cada `search_state` (`best_from`, nunca teste);
  - Fase 0: ablação do AE (sem AE × decoder_first × joint) em P0 com fid 45, 2 sementes;
  - lei de escala: saturação quando o ganho de κ por duplicação < 0,01.

### 2.11 Protocolo de teste (reporte)

- `monitor_report.py` (`:1-9`, `:142-175`) avalia o **teste de todo job terminado, em todas as fidelidades**, inclusive as buscas `_C01_`. A ordem é: fidelidade mais alta primeiro, com prioridade para o melhor por validação de cada estrato/P-level (`FILA_E_INSTRUCOES.md:207`).
- Leituras:
  - **ORIG**: `evaluate_checkpoint(..., scaler=False, readout="head", C=100.0)` (`monitor_report.py:168`);
  - **COV**: `scaler=True, readout="cov_head", C=100.0` (`:175`).
- Saída: `reports/test_svm_C100.jsonl` → `HYBRID_PROGRESS.md` e `ranking_ga_C100.csv` (`FILA_E_INSTRUCOES.md:90-91`).
- **O teste nunca entra na seleção** (R2, `FILA_E_INSTRUCOES.md:11`; `search_split_key: validation`, `search_v3.yaml:11`).

---

## 3. Como chegamos em cada técnica (cronologia e motivações)

### 3.1 Antecedentes (antes do GA, maio–outubro)

- **Destilação I-JEPA → FLIM**: a cabeça de 1280 dimensões pontuava muito mais que o encoder de 48 dimensões (eggs 2L 400k: 0,889 × 0,715). Além disso, a `val/loss` era desacoplada do κ. Daí vêm a disciplina de "réguas" e a seleção de checkpoint por κ, não por loss (`analysis_flim_distill/INVESTIGATION_flim_init_distillation.md:15-16`; `CLAUDE.md`, "Critical domain gotchas").
- **LeJEPA puro no FLIM** ficou fraco no flatten (eggs 0,3142) (`results_from_mds.md:23`).
- **SPiFiL híbrido de outubro** (crescimento camada a camada SPiFiL + AE + backprop, régua GTP) empatou ou superou o I-JEPA: eggs 0,941 × 0,940, larvae 0,968 × 0,943, protozoan 0,928 × 0,864 (`A_reports/october/report_halloween_october.md:26-28`). Daí o princípio "Hybrid SPiFiL: o backprop soma ao SPiFiL, nunca desfaz" (R6, `FILA_E_INSTRUCOES.md:15`). Ele motivou a busca automática da arquitetura (MO-DEHB) e o WiSE com α=0 = SPiFiL.

### 3.2 14/11 — H-MODEHB estratificado

- **Motivo**: sem estratos, a promoção ia toda para as redes Tiny. No teste sintético, as 8 vagas iam para Tiny; com estratos, ficam ≥ 2 por categoria (`STRATIFIED.md:96`).
- **Fidelidades**: voltaram a ser 5/15/45/100/200 a pedido do usuário (a 1ª versão usava 5/20/60/120) (`STRATIFIED.md:46`).
- **Auditoria de falhas**: a busca anterior (`modehb_spifil/remote_sensing`) perdeu 127/315 redes (40%) (`STRATIFIED.md:124-132`). Causas e correções:
  - OOM da cabeça 1×1 em resolução cheia → cabeça em pedaços + `mem_repair`;
  - SPiFiL com N ≤ D → `spifil_images_repair`;
  - LeJEPA sem regularizador → `repair` troca para sigreg;
  - NaN em eigh → `safe_eigh`;
  - **validação cortada nas primeiras 512 imagens, ordenadas por classe** → `val_max_samples: null`. No coconut, só a classe 1 era avaliada.
- Só init SPiFiL (`random_baseline: false`, decisão do usuário) (`STRATIFIED.md:33`).

### 3.3 Fitness: do label-free ao κ do SVM; cov_head

- O fitness original era sem rótulo (consistência k-NN entre vistas + estabilidade de clustering, `search_v3.yaml` bloco `fitness`). Isso ficou como diagnóstico, e a seleção passou a `svm_kappa` (`run_search.py:736-746`).
- **cov_head** foi adotado por ser a leitura de segunda ordem que mais subiu o κ de teste médio em 7 checkpoints NAS do remote_sensing, de 0,51 para 0,61 contra a leitura head (`pooling_head.py:322-323`). O próprio relatório alerta que "parte do ganho pode ser a leitura" (`ga_search/modehb/reports/remote_sensing_comparacao.md:27`).

### 3.4 15/11 — Incidente C=0,1 → C=100 e kappa_weight 0,75 → 1,0

- Um agente tinha escolhido C=0,1 para o SVM do fitness na validação. O usuário ordenou C=100 ("NUNCA MAIS MUDE ... C=100!"), e isso virou a regra R1 (`FILA_E_INSTRUCOES.md:10`, `:229`).
- `kappa_weight` passou a 1,0 (R3: "só considere o SVM e o FLOPS ... svm kappa", `:12`, `:229`).
- As buscas C=0,1 foram arquivadas em `*_C01_*` e continuam avaliadas no teste. O controller passou a incluir a assinatura do fitness (`run_search.py:1161-1164`).
- **Correção do monitor**: a assinatura passou a sair do nome do diretório. Antes, tudo virava C0,1 e os jobs novos eram pulados. A jsonl foi limpa: 97 linhas mantidas, 41 refeitas (`FILA_E_INSTRUCOES.md:105`).

### 3.5 15/11 — SVM original (ORIG) no teste

- O usuário disse: "no final das contas eu preciso ver o SVM original". O número reportado passou a ser `fit_svm` C=100 sem scaler no `head(encoder(x))`, igual aos FMs. O COV ficou como coluna extra (R2, `FILA_E_INSTRUCOES.md:11`; `monitor_report.py:4-8`).
- Na régua igual do remote_sensing, a ordem muda conforme a régua (`FILA_E_INSTRUCOES.md:93-95`):
  - ORIG: CosFace 0,605 > LEMONADE c00098 0,598 > SimCLR 0,585 > L0/LeJEPA 0,570 > MO-DEHB c00010 0,524;
  - COV: SPF-P-LeJEPA 0,693 > c00004 0,667.

### 3.6 15/11 — H-LEM

- LEMONADE como variante B: 4 brackets "bastam para a 1ª comparação com o run A". Usa warm start com as receitas dos relatórios: seed_B, o melhor SPiFiL pequeno do remote_sensing, e seed_A, "provado contra o I-JEPA" (`search_v3_lemonade.yaml` bloco `lemonade:`; `lemonade.py:7-11`).

### 3.7 15–16/11 — H-SEED (semear com SPF-P-LeJEPA)

- Decisão do usuário: "Sim, em todas", mas **na fila**, depois das buscas atuais (`FILA_E_INSTRUCOES.md:230`).
- Grade de crescimento + teto de 5M (soft) / 5,5M (duro): "se precisar passar um pouquinho ... passe" (`:84`, `:231`).
- **Vazamento**: a semente ham10000 viu 878 imagens do nosso val. Foi retreinada só no nosso treino; a antiga foi renomeada `*_VIU_NOSSO_VAL_nao_usar` (`:29`, `:113`; `reports/external_seed_leakage.md`).
- Validação de ponta a ponta antes de entrar na fila: mini-buscas de 23 e 22 jobs, 0 falhas, kill -9 com retomada (`:99-103`). Isso virou a regra R9 (`:19`).

### 3.8 15–16/11 — PNAS-CS

- Pedido para crescer de forma controlada por porte (P0–P4) com Network Morphism, AE camada a camada "obrigatório e testado PRIMEIRO, por ablação", e LeJEPA+SIGReg como SSL principal (`FILA_E_INSTRUCOES.md:46-65`).
- Fase 0 no e2e: decoder_first 0,496 > joint 0,481 > sem AE 0,467 (`:145`).
- Pendências: `random_twins: false`, e `widen_block` só preserva a função em blocos planos (`:148-151`).
- Começou a rodar no athena em 16/11 09:30 (`:154`).

### 3.9 16/11 — Sementes LeJEPA escolhidas por κ, não por loss

- Com `best_by_probe=False`, o `best_kappa.ckpt` era o da **menor val_ssl_loss**, frequentemente na época 0 (flowers 6, coconut 0, protozoan 1, aid 0).
- Usuário: "Retreinar pelo κ". A correção foi `best_by_probe=True` (monitor = κ da sonda GeM na validação) (`FILA_E_INSTRUCOES.md:158-166`).
- Ecoa o achado da destilação: loss ≠ κ.
- Padrão aberto: o κ da sonda é máximo nas primeiras épocas, e o LeJEPA longo piora (`:223-225`).

### 3.10 16/11 — Incidentes de CPU / BLAS (sem mudar o SVM)

- **athena**: load 220 em 96 núcleos e GPUs a 0%. Causa: eigh do DBN na CPU fp64, mais todas as buscas usando todos os núcleos.
  - Correções: `safe_eigh(on_cpu=...)` e `_cap_cpu_threads` (cpu_count//12) (`FILA_E_INSTRUCOES.md:127-138`; `run_search.py:1174-1187`).
- **BLAS de 64 threads no fit do libsvm**: com cov_head de D > 10k, um fit chegou a 9560 s; com 1 thread, 2,6 s e κ idêntico.
  - Correções: `OPENBLAS_NUM_THREADS=2`; depois o patch `svm_fitness_fast` (BLAS realmente limitado, eigh em 8 streams CUDA, WiSE em paralelo; features bitwise iguais; `core/svm.py` intacto) (`:177-183`, `:199-203`; `run_search.py:509-521`).
- Custo do cov_head no teste (~1 h por fit no athena) → `MONITOR_COV=top` (`:216`).

### 3.11 16/11 — Memória e "zero failed"

- **OOMs/cuDNN com a GPU cheia**:
  - `_is_oom` passou a cobrir "unable to find an engine" (`modehb.py:681`);
  - OOM_RETRIES 4 → 10, sleep 180 → 300 s;
  - gates com `flock` e 1 início de job a cada 30 min;
  - states cortados no job que falhou e relançados (`FILA_E_INSTRUCOES.md:185-197`).
- **Fit SPiFiL com N×N de ~13 GB**: patch `spifil_oom.py` (score em blocos só sob OOM, bitwise igual). Por R9, só roda numa cópia de teste em produção (`:211-214`).

---

## 4. Regras permanentes que definem a metodologia (resumo)

R1 SVM = `core.svm.fit_svm` C=100 (`FILA_E_INSTRUCOES.md:10`) · R2 ORIG no teste para todo job (`:11`) · R3 fitness = κ SVM + FLOPs (`:12`) · R6 o backprop soma ao SPiFiL (`:15`) · R9 código novo só depois de testado de ponta a ponta (`:19`) · teste nunca na seleção (`:57`; `search_v3.yaml:11`).
