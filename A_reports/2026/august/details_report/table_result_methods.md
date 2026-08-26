# Resultado de TESTE — comparação entre métodos

Retrato de 2026-08-25 18:10, com a linha `hybrid_FLIM (head)` acrescentada em 2026-08-26 a partir
de CSVs de 2026-08-26 12:51. Uma tabela por porcentagem.

## Procedência — leia antes de olhar qualquer número

Estes números não foram recalculados aqui. A tabela é a **junção de três arquivos** que já
estavam no disco:

| O quê | De onde |
|---|---|
| F1, κ, Acc | `artifacts/normalized/unified_svm_comparison.csv` |
| Params, TFLOPs, Pesos (MB) | `statistics/tools/compute_cost.csv` |
| F1, κ, Acc da linha hybrid_FLIM | `results/eval_growth_stages_pct{5,50}.csv` |
| F1, κ, Acc da linha hybrid_FLIM (feat) | `results/eval_growth_stages_g5_in_feature_pct{5,50}.csv` |
| F1, κ, Acc da linha hybrid_FLIM (img) | `results/eval_growth_stages_g5_in_image_pct{5,50}.csv` |
| F1, κ, Acc da linha hybrid_FLIM (head) | `results/eval_growth_stages_g5_head_pct{5,50}.csv` |

O CSV unificado é montado por `scripts/normalize_reports.py`, que lê os CSVs primários de cada
método (última coluna da legenda). O custo é medido por
`statistics/tools/measure_compute_cost.py` — `FlopCounterMode`, lote 1, float32, e
`weights_MB` = params × 4 bytes.

- Conjunto: **teste**. O SVM treina nos embeddings de `train` e prediz em `test`.
- Cada célula é **média ± desvio sobre 3 splits**, 4 casas decimais. Única exceção: larvae
  em 50% na linha hybrid_FLIM, com **n=2**. As linhas `(feat)` e `(img)` têm n=3 nas seis.
- **Negrito = melhor valor daquela métrica naquele dataset**, entre as linhas preenchidas da
  tabela.
- Em **Params**, o negrito marca o **menor** valor (menos parâmetros é melhor), não o maior.
- Cobertura: 8 métodos × 6 porcentagens × 3 datasets = 144 células **do CSV unificado**,
  nenhuma faltando. As quatro linhas `hybrid_FLIM*` só existem em 5% e 50% — 6 das 18
  células de cada uma, e a `(head)` só 4 das 18, porque o `larvae` dela ainda não rodou.

**Não confundir com `table_result_stages.md`.** Aquele é o currículo de crescimento SPiFiL
avaliado com o SVM oficial; este é a comparação entre famílias de método. Protocolos diferentes —
e a única ponte entre os dois é a linha `hybrid_FLIM`, que traz de lá o estágio de maior kappa e
carrega junto o protocolo de lá (ver a ressalva na Legenda).

## Legenda — o que é cada linha

| Linha | `method` no unified | `init` | Params | TFLOPs | Pesos (MB) | CSV primário |
|---|---|---|---|---|---|---|
| **LeJEPA** | `SVM_lejepa_view` | `trunc_normal` | 59.504 | 0.0007 | 0.23 | `artifacts/SVM/<ds>/lejepa_pct_<P>/metrics_SVM_trunc_normal.csv` |
| **FLIM** | `SVM_FLIM` | `flim` | 59.504 | 0.0007 | 0.23 | `data/reports_felipe/svm/` |
| **hybrid_FLIM** | `SVM_SPiFiL_growth_flatten_labcru` | `flim` | - | - | - | `results/eval_growth_stages_pct{5,50}.csv` |
| **hybrid_FLIM (feat)** | `SVM_SPiFiL_growth_flatten_labcru` | `flim` | - | - | - | `results/eval_growth_stages_g5_in_feature_pct{5,50}.csv` |
| **hybrid_FLIM (img)** | `SVM_SPiFiL_growth_flatten_labcru` | `flim` | - | - | - | `results/eval_growth_stages_g5_in_image_pct{5,50}.csv` |
| **hybrid_FLIM (head)** | `SVM_SPiFiL_growth_flatten_labcru` | `flim` | - | - | - | `results/eval_growth_stages_g5_head_pct{5,50}.csv` |
| **Distill1** | `SVM_Distill_1x1BN` | `trunc_normal` | 123.504 | 0.0008 | 0.47 | `results/svm_proj1280_1x1_BN2d_results.csv` |
| **Distill1 (FLIM)** | `SVM_Distill_1x1BN_flim_frozen_eval_loss` | `flim` | 123.504 | 0.0008 | 0.47 | `results/svm_distillation_conv_flim_frozen_results.csv` |
| **Distill2** | `SVM_Distill_2l400K` | `trunc_normal` | 402.544 | 0.0011 | 1.54 | `results/svm_proj1280_2l_1x1_BN2d_256_1280_results.csv` |
| **Distill3** | `SVM_Distill_3x3BN` | `trunc_normal` | 615.024 | 0.0013 | 2.35 | `results/svm_proj1280_3x3_BN2d_results.csv` |
| **Distill4** | `SVM_Distill_Proj1280` | `trunc_normal` | 889.200 | 0.0017 | 3.39 | `results/svm_distill_proj1280_results.csv` |
| **I-JEPA** | `SVM_IJEPA` | `ijepa` | 630.762.240 | 0.3332 | 2406.17 | `results/ijepa_svm_aggregated.csv` |

O `init` faz parte da identidade da linha, não é detalhe: o LeJEPA existe no CSV unificado com
**cinco** inicializações (`flim`, `he`, `random`, `trunc_normal`, `xavier`), e só a
`trunc_normal` é a do artigo. Pegar outra dá número diferente.

LeJEPA e FLIM têm custo idêntico (59.504 params) porque usam o **mesmo encoder** — o que muda é
como ele foi treinado. Uma pegadinha de nome: o `compute_cost.csv` ainda registra o LeJEPA como
`SVM_LeJEPA`, enquanto o CSV unificado já usa `SVM_lejepa_view`.

A linha **hybrid_FLIM** não vem do CSV unificado: ela é o currículo de crescimento SPiFiL,
consolidado em [`table_result_stages.md`](table_result_stages.md). Cada célula é o **estágio
de maior kappa médio daquele dataset** — não uma média sobre estágios, e não uma mistura: as três
métricas (F1, κ, Acc) saem sempre do mesmo estágio. Os estágios escolhidos são `stage2` para eggs
e protozoan nas duas porcentagens, `round2_stage3` para larvae em 5% e `round2_stage4` para larvae
em 50%.

**Params, TFLOPs e Pesos ficam em `-` nas seis tabelas** porque não existe um número único: o
encoder muda de tamanho conforme o dataset e o estágio escolhido — **59.504** parâmetros em eggs
(`stage2`, 3 camadas), **101.072** em larvae (`round2_*`, 5 camadas) e **55.902** em protozoan
(`stage2`, 3 camadas). Uma coluna só não comporta os três. Note que em eggs e protozoan o estágio
vencedor é o `stage2`, ou seja o encoder FLIM **antes** de qualquer crescimento: a diferença entre
59.504 e 55.902 vem de o `conv2` do próprio FLIM ter 30 canais em vez de 32, e isso se propagar
para a entrada do `conv3` — dois terços dos 3.602 parâmetros de diferença estão lá, não no SPiFiL. A contagem
por dataset e por profundidade está em [`table_result_stages.md`](table_result_stages.md).

**Ressalva de protocolo — a porcentagem desta linha significa outra coisa.** Como já avisado
acima, em `table_result_stages.md` a porcentagem é o tamanho do **conjunto de treino do
SVM**, enquanto no LeJEPA e no I-JEPA ela é a fração do **pré-treino auto-supervisionado**. A
linha **hybrid_FLIM** ela é as **duas coisas**: `scripts/spifil_growth_loop.py:213` repassa
`--percentage` ao treinador do autoencoder, que o entrega ao DataModule
(`src/modules/autoencoder_flim_module.py:928`); é o DataModule que dimensiona o conjunto de treino
da reconstrução (`src/data_modules/parasite_data_module_lejepa_splited.py:193`). O mesmo valor
define o conjunto de treino do SVM. Comparar 5% do hybrid_FLIM com 5% do I-JEPA continua não sendo comparar a mesma
coisa.

**Cobertura menor.** O experimento de crescimento só rodou em 5% e 50%; nas tabelas de 1%, 25%,
75% e 100% a linha é toda `-`. E em 50% o `larvae` tem **n=2 splits** (o split 3 não chegou à
rodada 2 do crescimento), contra n=3 em todas as outras células do documento.

### As duas linhas `(feat)` e `(img)` — mesma receita, um fator trocado

São a ablação do laço de crescimento: `g5_in_feature` calcula o superpixel do SPiFiL na **grade
de features 24×24**, `g5_in_image` na **imagem LAB 200×200** (o default). Todo o resto é igual
entre as duas e igual à linha `hybrid_FLIM`: `--one-per-class --impurities --pool-stride 2
--max-rounds 2`, seed 42, 3 splits. O protocolo de leitura também é o mesmo — a célula é o
estágio de maior kappa médio daquele dataset, com as três métricas saindo desse mesmo estágio.

**O estágio vencedor é `stage2` nas seis células de cada linha.** Nas duas famílias, nos dois
percentuais e nos três datasets, o topo é o encoder FLIM destravado **antes de qualquer
crescimento**. Nenhuma camada SPiFiL enxertada venceu em célula nenhuma — por isso essas duas
linhas não precisam do rótulo de estágio por dataset que a linha `hybrid_FLIM` carrega.

Isso as separa da linha `hybrid_FLIM` (grid4), onde o larvae vence com `round2_*`. A diferença
não é de qualidade do crescimento: no grid4 o larvae **chegou** à rodada 2, e nestas duas ele
nunca cresce. `--one-per-class` dá 2 imagens-semente (larvae tem 2 classes) e o orçamento de
covariância não fecha — N=286 ≤ D=432 no `(feat)`, N=200 ≤ D=432 no `(img)`. O `spifil_grow`
recusa com `GROW_EXHAUSTED` e o braço para no `stage2`.

**Cobertura melhor que a da linha `hybrid_FLIM`:** n=3 em todas as seis células das duas linhas,
inclusive larvae em 50%, onde a `hybrid_FLIM` tem n=2.

**Params:** como o vencedor é sempre `stage2`, o encoder é o FLIM de 3 camadas — **59.504** em
eggs e larvae (canais `[3,24,32,48]`, idênticos) e **55.902** em protozoan (`conv2` com 30
canais). Continuam em `-` porque uma coluna só não comporta dois valores.

**Uma escolha de leitura que precisa ficar explícita.** Em `(img)`, 5%, eggs, o `round2_stage3`
tem kappa médio 0.7141 — acima do `stage2` (0.6759). Ele **não** foi escolhido porque tem
**n=1**: é o único braço de toda a ablação que chegou à rodada 2 (`eggs_split2_pct5`). Publicar
`0.7141 ± 0.0000` ao lado de células de 3 splits leria como estabilidade quando é ausência de
amostra. A regra aplicada aqui é: entre os estágios de **cobertura máxima**, o de maior kappa.

### A linha `(head)` — crescimento com fine-tune supervisionado por Head

É o braço `g5_head`. A cadeia tem quatro etapas por dataset e split:

| Estágio | O que roda | Supervisão |
|---|---|---|
| `stage1` | encoder FLIM congelado + decoder | reconstrução |
| `stage2` | tudo destravado | reconstrução |
| `round{r}_grow` | cresce uma camada SPiFiL | nenhuma — não treina |
| `round{r}_head` | tudo destravado + Head de perceptrons, sem decoder | rótulo |

A diferença de receita para a linha `hybrid_FLIM` (grid4) está em três flags: `--pool-stride 2`,
`--one-per-class` (uma imagem por classe monta a camada SPiFiL, em vez das 87–200 do grid4) e
`--impurities` (sem máscara). São as mesmas três flags que as linhas `(feat)` e `(img)` já usam;
o que a `(head)` acrescenta é o fine-tune supervisionado nos estágios `round{r}_head`.

**Qual estágio representa a linha.** A mesma regra das outras três linhas `hybrid_FLIM*`: entre os
estágios de **cobertura máxima** daquele dataset, o de **maior kappa médio** — e as três métricas
saem sempre desse mesmo estágio. Isso dá `round1_head` (4 camadas) para eggs nas duas
porcentagens e para protozoan em 5%, e `round2_head` (5 camadas) para protozoan em 50%. Em nenhuma
das quatro células o vencedor é `stage1` ou `stage2`, então, ao contrário de `(feat)` e `(img)`, a
linha carrega rótulo de estágio por dataset.

**Ressalva 1 — o treino não terminou.** Os dois CSVs somam **58 dos 72 estágios** previstos
(3 datasets × 3 splits × 2 porcentagens × 4 estágios). O que falta:

- **`larvae` não tem nenhum estágio de Head**, em nenhuma das duas porcentagens. Só `stage1` e
  `stage2` rodaram. Por isso as três células de larvae desta linha estão em `-` nas seis tabelas —
  é ausência de execução, não resultado ruim. São 12 dos 14 estágios faltantes.
- Em 5%, faltam ainda `eggs_split1_round2_head` e `protozoan_split3_round2_head`; por isso o
  `round2_head` de 5% tem **n=2** e não entrou como vencedor em célula nenhuma.

Todas as quatro células publicadas têm **n=3 splits**.

**Ressalva 2 — o embedding é muito menor que o das outras linhas `hybrid_FLIM*`.** O
`--pool-stride 2` encolhe o mapa de features a cada rodada, e o SVM lê o `flatten` desse mapa:

| Estágio | eggs | protozoan |
|---|---|---|
| `stage1` / `stage2` | 27.648 (48×24×24) | 27.648 |
| `round1_head` | 5.445 (45×11×11) | 5.082 (42×11×11) |
| `round2_head` | 1.125 (45×5×5) | 1.050 (42×5×5) |

As outras três linhas `hybrid_FLIM*` publicam sempre um estágio de **27.648** dimensões, porque
nelas o vencedor é sempre `stage2`. Aqui as células publicadas têm 5.445, 5.082 e 1.050 — e no fim
da cadeia o embedding é **~25× menor** (1.125 contra 27.648). **Não é comparação linha a linha**:
o SVM desta linha recebe um vetor de outra ordem de grandeza que o das linhas vizinhas.

**Ressalva 3 — a porcentagem acumula mais um papel nesta linha.** Na `hybrid_FLIM` ela já era duas
coisas (treino auto-supervisionado do autoencoder **e** treino do SVM). Na `(head)` é uma terceira
também: o mesmo `--percentage` dimensiona o conjunto rotulado do **fine-tune da Head**. Ler "5%" nesta
linha e "5%" no I-JEPA é comparar três usos do dado contra um.

**Params, TFLOPs e Pesos ficam em `-`** pelo mesmo motivo das outras: o encoder muda de tamanho
conforme o dataset e o estágio escolhido, e uma coluna só não comporta os valores.

## O que a porcentagem significa — e o cuidado que ela exige

**A semântica não é a mesma em todas as linhas.** No LeJEPA e no I-JEPA, `pretrained_pct` é a
fração usada no **pré-treino auto-supervisionado**. Nas linhas de destilação e no FLIM ela se
refere ao conjunto rotulado. Na linha **hybrid_FLIM** ela é as duas coisas ao mesmo tempo:
dimensiona o treino auto-supervisionado do autoencoder **e** o conjunto de treino do SVM. Na linha
**hybrid_FLIM (head)** ela é três: as duas acima **mais** o conjunto rotulado do fine-tune da Head. Comparar
linhas dentro de uma mesma tabela é comparar modelos que viram quantidades diferentes de dado em
etapas diferentes.

Vale também para fora: em `table_result_stages.md` a porcentagem é o tamanho do conjunto de
treino do SVM. Os dois documentos usam a mesma palavra para coisas diferentes.

---

## 1% dos dados

| Método | Params | eggs F1 | eggs κ | eggs Acc | larvae F1 | larvae κ | larvae Acc | protozoan F1 | protozoan κ | protozoan Acc |
|---|---|---|---|---|---|---|---|---|---|---|
| **LeJEPA** | **59.504** | 0.3473 ± 0.0198 | 0.2366 ± 0.0215 | 0.4767 ± 0.0558 | 0.4661 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 | 0.4355 ± 0.0116 | 0.2701 ± 0.0538 | 0.4704 ± 0.0180 |
| **FLIM** | **59.504** | 0.0952 ± 0.0170 | 0.1278 ± 0.0284 | 0.1738 ± 0.0297 | 0.3365 ± 0.4386 | 0.0284 ± 0.1004 | 0.3899 ± 0.4281 | 0.1422 ± 0.0183 | 0.1447 ± 0.0455 | 0.2083 ± 0.0455 |
| **hybrid_FLIM** | - | - | - | - | - | - | - | - | - | - |
| **hybrid_FLIM (feat)** | - | - | - | - | - | - | - | - | - | - |
| **hybrid_FLIM (img)** | - | - | - | - | - | - | - | - | - | - |
| **hybrid_FLIM (head)** | - | - | - | - | - | - | - | - | - | - |
| **Distill1** | 123.504 | 0.0879 ± 0.0000 | 0.0000 ± 0.0000 | 0.1111 ± 0.0000 | 0.4661 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 | 0.1449 ± 0.0324 | -0.0154 ± 0.0223 | 0.2091 ± 0.0751 |
| **Distill1 (FLIM)** | 123.504 | 0.2233 ± 0.0298 | 0.0631 ± 0.0645 | 0.2250 ± 0.0247 | 0.7112 ± 0.1147 | 0.4286 ± 0.2238 | 0.6843 ± 0.1120 | 0.3258 ± 0.0256 | 0.2710 ± 0.0069 | 0.3257 ± 0.0374 |
| **Distill2** | 402.544 | 0.4067 ± 0.0276 | 0.3312 ± 0.0501 | 0.4722 ± 0.0208 | 0.8080 ± 0.0425 | 0.6168 ± 0.0851 | 0.8281 ± 0.0754 | 0.4264 ± 0.0363 | 0.3302 ± 0.0437 | 0.4662 ± 0.0442 |
| **Distill3** | 615.024 | 0.3528 ± 0.0192 | 0.2683 ± 0.0332 | 0.4270 ± 0.0266 | 0.8174 ± 0.0679 | 0.6350 ± 0.1359 | 0.8135 ± 0.0765 | 0.4455 ± 0.0529 | 0.3447 ± 0.0276 | 0.4781 ± 0.0417 |
| **Distill4** | 889.200 | 0.3666 ± 0.0286 | 0.2637 ± 0.0653 | 0.4433 ± 0.0115 | 0.8058 ± 0.0308 | 0.6129 ± 0.0620 | 0.8301 ± 0.0646 | 0.4646 ± 0.0203 | 0.3915 ± 0.0254 | 0.5124 ± 0.0474 |
| **I-JEPA** | 630.762.240 | **0.6518** ± 0.0198 | **0.6608** ± 0.0273 | **0.6805** ± 0.0449 | **0.9105** ± 0.0416 | **0.8216** ± 0.0825 | **0.9241** ± 0.0413 | **0.5593** ± 0.0151 | **0.5855** ± 0.0273 | **0.5756** ± 0.0245 |

Negrito = maior valor daquela métrica naquele dataset, entre as linhas preenchidas da tabela. Na coluna Params, o negrito marca o **menor** número de parâmetros — aqui LeJEPA e FLIM empatam, pois compartilham o mesmo encoder.

---

## 5% dos dados

| Método | Params | eggs F1 | eggs κ | eggs Acc | larvae F1 | larvae κ | larvae Acc | protozoan F1 | protozoan κ | protozoan Acc |
|---|---|---|---|---|---|---|---|---|---|---|
| **LeJEPA** | **59.504** | 0.4589 ± 0.1554 | 0.3361 ± 0.1757 | 0.4686 ± 0.1751 | 0.6838 ± 0.0988 | 0.3741 ± 0.1977 | 0.7089 ± 0.1600 | 0.3935 ± 0.0674 | 0.2585 ± 0.0354 | 0.4371 ± 0.1024 |
| **FLIM** | **59.504** | 0.7577 ± 0.0355 | 0.5852 ± 0.0513 | 0.7512 ± 0.0374 | 0.9125 ± 0.0232 | 0.5712 ± 0.1208 | 0.9213 ± 0.0189 | **0.7674** ± 0.0164 | 0.6230 ± 0.0254 | **0.7646** ± 0.0184 |
| **hybrid_FLIM**<br>`stage2` 3 cam. / `round2_stage3` 5 cam. / `stage2` 3 cam. | - | 0.7152 ± 0.0020 | 0.6749 ± 0.0120 | 0.7603 ± 0.0476 | 0.8334 ± 0.0565 | 0.6679 ± 0.1117 | 0.8057 ± 0.0685 | 0.6780 ± 0.0194 | 0.6746 ± 0.0199 | 0.6990 ± 0.0205 |
| **hybrid_FLIM (feat)**<br>`stage2` 3 cam. nos três | - | 0.7153 ± 0.0021 | 0.6725 ± 0.0133 | 0.7609 ± 0.0460 | 0.8011 ± 0.0610 | 0.6045 ± 0.1198 | 0.7607 ± 0.0625 | 0.6771 ± 0.0180 | 0.6705 ± 0.0240 | 0.7005 ± 0.0205 |
| **hybrid_FLIM (img)**<br>`stage2` 3 cam. nos três | - | 0.7177 ± 0.0038 | 0.6759 ± 0.0103 | 0.7624 ± 0.0465 | 0.7967 ± 0.0573 | 0.5959 ± 0.1125 | 0.7554 ± 0.0587 | 0.6785 ± 0.0258 | 0.6754 ± 0.0141 | 0.7015 ± 0.0272 |
| **hybrid_FLIM (head)**<br>`round1_head` 4 cam. / larvae não rodou / `round1_head` 4 cam. | - | 0.7200 ± 0.0707 | 0.6949 ± 0.0801 | 0.7271 ± 0.0615 | - | - | - | 0.6532 ± 0.0286 | 0.6937 ± 0.0121 | 0.6580 ± 0.0178 |
| **Distill1** | 123.504 | 0.0954 ± 0.0118 | -0.0022 ± 0.0052 | 0.1160 ± 0.0086 | 0.5042 ± 0.0188 | 0.0615 ± 0.0315 | 0.5185 ± 0.0098 | 0.1648 ± 0.1005 | 0.0399 ± 0.0691 | 0.2163 ± 0.1272 |
| **Distill1 (FLIM)** | 123.504 | 0.3840 ± 0.0310 | 0.3793 ± 0.0374 | 0.3863 ± 0.0183 | 0.8768 ± 0.0301 | 0.7537 ± 0.0601 | 0.8736 ± 0.0199 | 0.3959 ± 0.1210 | 0.4400 ± 0.0930 | 0.3898 ± 0.0935 |
| **Distill2** | 402.544 | 0.5475 ± 0.0694 | 0.4891 ± 0.0983 | 0.6186 ± 0.0390 | 0.8357 ± 0.0335 | 0.6716 ± 0.0669 | 0.8339 ± 0.0381 | 0.5261 ± 0.0099 | 0.5099 ± 0.0382 | 0.5458 ± 0.0136 |
| **Distill3** | 615.024 | 0.6208 ± 0.0162 | 0.5963 ± 0.0266 | 0.6722 ± 0.0272 | 0.8510 ± 0.0252 | 0.7024 ± 0.0504 | 0.8424 ± 0.0373 | 0.5198 ± 0.0191 | 0.5039 ± 0.0282 | 0.5407 ± 0.0122 |
| **Distill4** | 889.200 | 0.7002 ± 0.0503 | 0.6679 ± 0.0585 | 0.7423 ± 0.0470 | 0.8829 ± 0.0120 | 0.7660 ± 0.0238 | 0.8880 ± 0.0163 | 0.5786 ± 0.0238 | 0.5578 ± 0.0127 | 0.6009 ± 0.0316 |
| **I-JEPA** | 630.762.240 | **0.8683** ± 0.0144 | **0.8639** ± 0.0135 | **0.8696** ± 0.0278 | **0.9403** ± 0.0126 | **0.8807** ± 0.0251 | **0.9469** ± 0.0026 | 0.6644 ± 0.0267 | **0.7251** ± 0.0098 | 0.6628 ± 0.0200 |

Negrito = maior valor daquela métrica naquele dataset, entre as linhas preenchidas da tabela. Na coluna Params, o negrito marca o **menor** número de parâmetros — aqui LeJEPA e FLIM empatam, pois compartilham o mesmo encoder. Nas linhas **hybrid_FLIM** e **hybrid_FLIM (head)**, o rótulo traz o estágio vencedor e a profundidade do encoder naquele estágio, na ordem **eggs / larvae / protozoan** — os três não são o mesmo estágio. Na **(head)**, o `larvae` está em `-` porque nenhum estágio de Head rodou nele, e o embedding das células publicadas tem 5.445 (eggs) e 5.082 (protozoan) dimensões, não as 27.648 das linhas vizinhas.

---

## 25% dos dados

| Método | Params | eggs F1 | eggs κ | eggs Acc | larvae F1 | larvae κ | larvae Acc | protozoan F1 | protozoan κ | protozoan Acc |
|---|---|---|---|---|---|---|---|---|---|---|
| **LeJEPA** | **59.504** | 0.4581 ± 0.0923 | 0.3502 ± 0.1287 | 0.4795 ± 0.0986 | 0.6869 ± 0.1438 | 0.3892 ± 0.2701 | 0.7242 ± 0.1726 | 0.4137 ± 0.0401 | 0.2313 ± 0.0524 | 0.4846 ± 0.0902 |
| **FLIM** | **59.504** | 0.8885 ± 0.0161 | 0.8050 ± 0.0263 | 0.8870 ± 0.0171 | 0.9566 ± 0.0027 | 0.8049 ± 0.0142 | 0.9566 ± 0.0023 | **0.8628** ± 0.0152 | 0.7750 ± 0.0248 | **0.8619** ± 0.0158 |
| **hybrid_FLIM** | - | - | - | - | - | - | - | - | - | - |
| **hybrid_FLIM (feat)** | - | - | - | - | - | - | - | - | - | - |
| **hybrid_FLIM (img)** | - | - | - | - | - | - | - | - | - | - |
| **hybrid_FLIM (head)** | - | - | - | - | - | - | - | - | - | - |
| **Distill1** | 123.504 | 0.0878 ± 0.0000 | -0.0004 ± 0.0004 | 0.1110 ± 0.0001 | 0.5036 ± 0.0185 | 0.0589 ± 0.0309 | 0.5179 ± 0.0096 | 0.2745 ± 0.0488 | 0.1499 ± 0.0652 | 0.3671 ± 0.1234 |
| **Distill1 (FLIM)** | 123.504 | 0.6555 ± 0.0443 | 0.6754 ± 0.0624 | 0.6478 ± 0.0513 | 0.9425 ± 0.0037 | 0.8850 ± 0.0074 | 0.9407 ± 0.0132 | 0.6308 ± 0.0810 | 0.6606 ± 0.0253 | 0.5871 ± 0.0632 |
| **Distill2** | 402.544 | 0.6854 ± 0.0069 | 0.6544 ± 0.0137 | 0.7481 ± 0.0075 | 0.8593 ± 0.0119 | 0.7186 ± 0.0237 | 0.8590 ± 0.0198 | 0.5968 ± 0.0230 | 0.5771 ± 0.0317 | 0.6294 ± 0.0181 |
| **Distill3** | 615.024 | 0.7813 ± 0.0194 | 0.7576 ± 0.0245 | 0.8187 ± 0.0173 | 0.9267 ± 0.0082 | 0.8535 ± 0.0164 | 0.9332 ± 0.0053 | 0.7122 ± 0.0210 | 0.7069 ± 0.0254 | 0.7304 ± 0.0181 |
| **Distill4** | 889.200 | 0.8449 ± 0.0275 | 0.8271 ± 0.0241 | 0.8686 ± 0.0177 | 0.9349 ± 0.0101 | 0.8698 ± 0.0201 | 0.9380 ± 0.0189 | 0.7510 ± 0.0216 | 0.7459 ± 0.0033 | 0.7699 ± 0.0231 |
| **I-JEPA** | 630.762.240 | **0.9278** ± 0.0089 | **0.9171** ± 0.0138 | **0.9362** ± 0.0111 | **0.9595** ± 0.0002 | **0.9190** ± 0.0005 | **0.9659** ± 0.0029 | 0.8189 ± 0.0024 | **0.8307** ± 0.0023 | 0.8266 ± 0.0091 |

Negrito = maior valor daquela métrica naquele dataset, entre as linhas preenchidas da tabela. Na coluna Params, o negrito marca o **menor** número de parâmetros — aqui LeJEPA e FLIM empatam, pois compartilham o mesmo encoder.

---

## 50% dos dados

| Método | Params | eggs F1 | eggs κ | eggs Acc | larvae F1 | larvae κ | larvae Acc | protozoan F1 | protozoan κ | protozoan Acc |
|---|---|---|---|---|---|---|---|---|---|---|
| **LeJEPA** | **59.504** | 0.4416 ± 0.0903 | 0.3137 ± 0.0747 | 0.4876 ± 0.0921 | 0.8176 ± 0.0238 | 0.6353 ± 0.0476 | 0.8206 ± 0.0313 | 0.3062 ± 0.0185 | 0.1230 ± 0.0525 | 0.3617 ± 0.0845 |
| **FLIM** | **59.504** | 0.9109 ± 0.0128 | 0.8423 ± 0.0222 | 0.9097 ± 0.0129 | 0.9606 ± 0.0065 | 0.8213 ± 0.0308 | 0.9607 ± 0.0064 | 0.8880 ± 0.0076 | 0.8158 ± 0.0128 | **0.8875** ± 0.0079 |
| **hybrid_FLIM**<br>`stage2` 3 cam. / `round2_stage4` 5 cam. / `stage2` 3 cam. | - | 0.9137 ± 0.0038 | 0.9062 ± 0.0032 | 0.9245 ± 0.0095 | 0.9344 ± 0.0060 | 0.8687 ± 0.0120 | 0.9382 ± 0.0161 | 0.8511 ± 0.0055 | 0.8587 ± 0.0062 | 0.8661 ± 0.0025 |
| **hybrid_FLIM (feat)**<br>`stage2` 3 cam. nos três | - | 0.9245 ± 0.0053 | 0.9153 ± 0.0044 | 0.9307 ± 0.0077 | 0.9285 ± 0.0206 | 0.8569 ± 0.0412 | 0.9306 ± 0.0207 | 0.8532 ± 0.0029 | 0.8583 ± 0.0064 | 0.8654 ± 0.0060 |
| **hybrid_FLIM (img)**<br>`stage2` 3 cam. nos três | - | 0.9152 ± 0.0102 | 0.9064 ± 0.0107 | 0.9249 ± 0.0117 | 0.9291 ± 0.0189 | 0.8582 ± 0.0378 | 0.9320 ± 0.0178 | 0.8529 ± 0.0079 | 0.8573 ± 0.0100 | 0.8652 ± 0.0026 |
| **hybrid_FLIM (head)**<br>`round1_head` 4 cam. / larvae não rodou / `round2_head` 5 cam. | - | 0.9408 ± 0.0114 | 0.9309 ± 0.0093 | 0.9355 ± 0.0202 | - | - | - | **0.9002** ± 0.0040 | **0.9198** ± 0.0089 | 0.8850 ± 0.0092 |
| **Distill1** | 123.504 | 0.1533 ± 0.0608 | 0.0219 ± 0.0407 | 0.2043 ± 0.0911 | 0.5225 ± 0.0078 | 0.0883 ± 0.0490 | 0.5613 ± 0.0668 | 0.3438 ± 0.1389 | 0.2407 ± 0.1491 | 0.4757 ± 0.1073 |
| **Distill1 (FLIM)** | 123.504 | 0.7379 ± 0.0068 | 0.7579 ± 0.0336 | 0.7178 ± 0.0242 | 0.9431 ± 0.0098 | 0.8863 ± 0.0195 | 0.9390 ± 0.0012 | 0.7011 ± 0.0611 | 0.7106 ± 0.0150 | 0.6555 ± 0.0505 |
| **Distill2** | 402.544 | 0.7341 ± 0.0075 | 0.7034 ± 0.0051 | 0.7802 ± 0.0153 | 0.9030 ± 0.0239 | 0.8061 ± 0.0479 | 0.9079 ± 0.0188 | 0.6835 ± 0.0155 | 0.6818 ± 0.0093 | 0.7223 ± 0.0165 |
| **Distill3** | 615.024 | 0.8447 ± 0.0101 | 0.8334 ± 0.0098 | 0.8691 ± 0.0211 | 0.9398 ± 0.0075 | 0.8796 ± 0.0150 | 0.9412 ± 0.0104 | 0.7769 ± 0.0187 | 0.7871 ± 0.0054 | 0.7843 ± 0.0253 |
| **Distill4** | 889.200 | 0.8876 ± 0.0096 | 0.8755 ± 0.0103 | 0.9011 ± 0.0177 | 0.9572 ± 0.0078 | 0.9144 ± 0.0157 | 0.9622 ± 0.0089 | 0.8141 ± 0.0127 | 0.8060 ± 0.0116 | 0.8228 ± 0.0062 |
| **I-JEPA** | 630.762.240 | **0.9484** ± 0.0072 | **0.9397** ± 0.0081 | **0.9489** ± 0.0094 | **0.9717** ± 0.0057 | **0.9433** ± 0.0115 | **0.9698** ± 0.0031 | 0.8559 ± 0.0118 | 0.8636 ± 0.0032 | 0.8570 ± 0.0050 |

Negrito = maior valor daquela métrica naquele dataset, entre as linhas preenchidas da tabela. Na coluna Params, o negrito marca o **menor** número de parâmetros — aqui LeJEPA e FLIM empatam, pois compartilham o mesmo encoder. Nas linhas **hybrid_FLIM** e **hybrid_FLIM (head)**, o rótulo traz o estágio vencedor e a profundidade do encoder naquele estágio, na ordem **eggs / larvae / protozoan** — os três não são o mesmo estágio. Na **(head)**, o `larvae` está em `-` porque nenhum estágio de Head rodou nele, e o embedding das células publicadas tem 5.445 (eggs) e 1.050 (protozoan) dimensões, não as 27.648 das linhas vizinhas — o negrito em protozoan F1 e κ compara vetores de tamanhos muito diferentes.

---

## 75% dos dados

| Método | Params | eggs F1 | eggs κ | eggs Acc | larvae F1 | larvae κ | larvae Acc | protozoan F1 | protozoan κ | protozoan Acc |
|---|---|---|---|---|---|---|---|---|---|---|
| **LeJEPA** | **59.504** | 0.4852 ± 0.0860 | 0.3654 ± 0.1175 | 0.6147 ± 0.0138 | 0.8091 ± 0.0438 | 0.6183 ± 0.0876 | 0.8063 ± 0.0450 | 0.2405 ± 0.0909 | 0.1174 ± 0.0879 | 0.3253 ± 0.1510 |
| **FLIM** | **59.504** | 0.9267 ± 0.0105 | 0.8696 ± 0.0182 | 0.9261 ± 0.0106 | 0.9634 ± 0.0028 | 0.8353 ± 0.0132 | 0.9632 ± 0.0026 | **0.9014** ± 0.0112 | 0.8371 ± 0.0185 | **0.9005** ± 0.0118 |
| **hybrid_FLIM** | - | - | - | - | - | - | - | - | - | - |
| **hybrid_FLIM (feat)** | - | - | - | - | - | - | - | - | - | - |
| **hybrid_FLIM (img)** | - | - | - | - | - | - | - | - | - | - |
| **hybrid_FLIM (head)** | - | - | - | - | - | - | - | - | - | - |
| **Distill1** | 123.504 | 0.2516 ± 0.1524 | 0.1713 ± 0.1501 | 0.4744 ± 0.3255 | 0.5102 ± 0.0073 | 0.0665 ± 0.0143 | 0.5205 ± 0.0044 | 0.4239 ± 0.0535 | 0.2973 ± 0.0898 | 0.5632 ± 0.0451 |
| **Distill1 (FLIM)** | 123.504 | 0.7610 ± 0.0070 | 0.7823 ± 0.0301 | 0.7395 ± 0.0237 | 0.9482 ± 0.0071 | 0.8964 ± 0.0142 | 0.9442 ± 0.0062 | 0.7138 ± 0.0434 | 0.7290 ± 0.0199 | 0.6665 ± 0.0394 |
| **Distill2** | 402.544 | 0.7971 ± 0.0270 | 0.7769 ± 0.0273 | 0.8368 ± 0.0091 | 0.9280 ± 0.0089 | 0.8560 ± 0.0177 | 0.9311 ± 0.0061 | 0.7522 ± 0.0132 | 0.7514 ± 0.0110 | 0.7726 ± 0.0156 |
| **Distill3** | 615.024 | 0.8836 ± 0.0155 | 0.8696 ± 0.0151 | 0.9010 ± 0.0149 | 0.9563 ± 0.0027 | 0.9127 ± 0.0054 | 0.9563 ± 0.0061 | 0.7961 ± 0.0203 | 0.8109 ± 0.0114 | 0.8131 ± 0.0185 |
| **Distill4** | 889.200 | 0.9146 ± 0.0160 | 0.9021 ± 0.0150 | 0.9249 ± 0.0163 | 0.9620 ± 0.0009 | 0.9240 ± 0.0018 | 0.9579 ± 0.0051 | 0.8286 ± 0.0072 | 0.8240 ± 0.0047 | 0.8423 ± 0.0091 |
| **I-JEPA** | 630.762.240 | **0.9527** ± 0.0073 | **0.9455** ± 0.0087 | **0.9528** ± 0.0128 | **0.9705** ± 0.0047 | **0.9410** ± 0.0094 | **0.9708** ± 0.0059 | 0.8692 ± 0.0143 | **0.8817** ± 0.0049 | 0.8730 ± 0.0029 |

Negrito = maior valor daquela métrica naquele dataset, entre as linhas preenchidas da tabela. Na coluna Params, o negrito marca o **menor** número de parâmetros — aqui LeJEPA e FLIM empatam, pois compartilham o mesmo encoder.

---

## 100% dos dados

| Método | Params | eggs F1 | eggs κ | eggs Acc | larvae F1 | larvae κ | larvae Acc | protozoan F1 | protozoan κ | protozoan Acc |
|---|---|---|---|---|---|---|---|---|---|---|
| **LeJEPA** | **59.504** | 0.4272 ± 0.0609 | 0.3142 ± 0.0584 | 0.4935 ± 0.1188 | 0.5312 ± 0.2688 | 0.2670 ± 0.2900 | 0.5953 ± 0.1804 | 0.2624 ± 0.0617 | 0.1030 ± 0.0619 | 0.3471 ± 0.1022 |
| **FLIM** | **59.504** | 0.9355 ± 0.0134 | 0.8848 ± 0.0236 | 0.9348 ± 0.0138 | 0.9706 ± 0.0023 | 0.8677 ± 0.0106 | 0.9706 ± 0.0023 | **0.9075** ± 0.0151 | 0.8470 ± 0.0250 | **0.9069** ± 0.0156 |
| **hybrid_FLIM** | - | - | - | - | - | - | - | - | - | - |
| **hybrid_FLIM (feat)** | - | - | - | - | - | - | - | - | - | - |
| **hybrid_FLIM (img)** | - | - | - | - | - | - | - | - | - | - |
| **hybrid_FLIM (head)** | - | - | - | - | - | - | - | - | - | - |
| **Distill1** | 123.504 | 0.4219 ± 0.0612 | 0.2959 ± 0.0493 | 0.7394 ± 0.0050 | 0.5273 ± 0.3419 | 0.2005 ± 0.5062 | 0.5259 ± 0.3366 | 0.3145 ± 0.0580 | 0.2016 ± 0.0427 | 0.4376 ± 0.0567 |
| **Distill1 (FLIM)** | 123.504 | 0.7877 ± 0.0197 | 0.8003 ± 0.0292 | 0.7624 ± 0.0257 | 0.9561 ± 0.0084 | 0.9121 ± 0.0169 | 0.9531 ± 0.0083 | 0.7329 ± 0.0390 | 0.7410 ± 0.0199 | 0.6826 ± 0.0375 |
| **Distill2** | 402.544 | 0.8391 ± 0.0109 | 0.8189 ± 0.0117 | 0.8609 ± 0.0111 | 0.9412 ± 0.0031 | 0.8824 ± 0.0062 | 0.9497 ± 0.0047 | 0.7989 ± 0.0300 | 0.7935 ± 0.0249 | 0.8217 ± 0.0274 |
| **Distill3** | 615.024 | 0.9118 ± 0.0098 | 0.8997 ± 0.0083 | 0.9131 ± 0.0150 | 0.9695 ± 0.0067 | 0.9391 ± 0.0135 | 0.9749 ± 0.0074 | 0.8175 ± 0.0175 | 0.8326 ± 0.0137 | 0.8273 ± 0.0165 |
| **Distill4** | 889.200 | 0.9168 ± 0.0109 | 0.9120 ± 0.0082 | 0.9267 ± 0.0124 | 0.9634 ± 0.0037 | 0.9268 ± 0.0074 | 0.9657 ± 0.0010 | 0.8399 ± 0.0025 | 0.8377 ± 0.0062 | 0.8543 ± 0.0074 |
| **I-JEPA** | 630.762.240 | **0.9626** ± 0.0063 | **0.9574** ± 0.0067 | **0.9626** ± 0.0114 | **0.9748** ± 0.0033 | **0.9495** ± 0.0065 | **0.9751** ± 0.0054 | 0.8841 ± 0.0046 | **0.8917** ± 0.0024 | 0.8902 ± 0.0071 |

Negrito = maior valor daquela métrica naquele dataset, entre as linhas preenchidas da tabela. Na coluna Params, o negrito marca o **menor** número de parâmetros — aqui LeJEPA e FLIM empatam, pois compartilham o mesmo encoder.

---

## Nota de método — quatro ressalvas que mudam a leitura

**1. O desvio do LeJEPA usa fórmula diferente do resto.** Os `metrics_SVM_*.csv` do LeJEPA foram
gerados com `ddof=0` (populacional) e o `normalize_reports.py:176-190` apenas copia. Todas as
outras linhas usam `ddof=1` (amostral). Com a mesma régua, o LeJEPA a 5% em eggs seria
`0.46 ± 0.19 / 0.34 ± 0.22 / 0.47 ± 0.21` em vez de `± 0.16 / 0.18 / 0.18`. As **médias não
mudam**, só os desvios.

**2. As métricas não são plenamente comensuráveis entre braços.** `compute_metrics` usa
`average="macro"`, então a maioria das linhas reporta acurácia **balanceada**; o braço FLIM
carrega acurácia micro e F1 ponderado, de outra origem. A auditoria em
`A_reports/2026/august/report_auditoria_figura_kappa_2026-08-03.md` mede discrepâncias de até
0,373 e conclui que **κ é a única das três métricas comensurável entre braços**.

**3. A linha do LeJEPA não é reproduzível com o código de hoje.** Ela vem do caminho **flatten
27.648-d**; o código atual roda **pooling 48-d**, e o rerun com pooling
(`results/svm_lejepa_pooled_results.csv`) dá quase-acaso, com κ ≈ 0 em larvae e protozoan.
Documentado em `A_reports/2026/july/details_report/lejepa_view_pooling_handoff_2026-07-31.md`.

**4. O I-JEPA roda em resolução diferente:** 224×224 contra 200×200 dos demais
(`statistics/tools/compute_cost.csv`, coluna `input_shape`). Os TFLOPs dele não são comparáveis
aos outros na mesma base de entrada.

## Como reproduzir

```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM

python scripts/normalize_reports.py                    # regenera o CSV unificado
python statistics/tools/measure_compute_cost.py        # regenera as colunas de custo

# as tres linhas hybrid_FLIM* vem daqui, nao do CSV unificado
python -m src.evaluate.eval_growth_stages --pct 5  --device cuda:0   # grid4 -> hybrid_FLIM
python -m src.evaluate.eval_growth_stages --pct 50 --device cuda:0

python -m src.evaluate.eval_growth_stages --pct 5  --family g5_in_feature --device cuda:0
python -m src.evaluate.eval_growth_stages --pct 50 --family g5_in_feature --device cuda:0
python -m src.evaluate.eval_growth_stages --pct 5  --family g5_in_image   --device cuda:0
python -m src.evaluate.eval_growth_stages --pct 50 --family g5_in_image   --device cuda:0

python -m src.evaluate.eval_growth_stages --pct 5  --family g5_head       --device cuda:0
python -m src.evaluate.eval_growth_stages --pct 50 --family g5_head       --device cuda:0
```

Extrair as 8 linhas do CSV unificado de uma porcentagem:

```bash
python -c "
import pandas as pd
d = pd.read_csv('artifacts/normalized/unified_svm_comparison.csv')
alvo = {'SVM_lejepa_view':'trunc_normal', 'SVM_FLIM':'flim',
        'SVM_Distill_1x1BN':'trunc_normal', 'SVM_Distill_1x1BN_flim_frozen_eval_loss':'flim',
        'SVM_Distill_2l400K':'trunc_normal', 'SVM_Distill_3x3BN':'trunc_normal',
        'SVM_Distill_Proj1280':'trunc_normal', 'SVM_IJEPA':'ijepa'}
print(d[(d.pretrained_pct==5) & d.apply(lambda r: alvo.get(r.method)==r.init, axis=1)].to_string())"
```

Não existe script que emita este documento pronto — ele é a junção das três fontes acima.
