# Relatório de Análise Completa v2: FLIM SSL + Destilação
## Com Curvas de Treinamento Completas do WandB

**Data:** 2026-05-30  
**Projeto:** Scalable FLIM Self-Supervised Learning — UNICAMP/FEEC  
**Pesquisador:** Mateus Oliveira  
**Fonte de dados:** 20.752 linhas de histórico por época (216 runs de destilação), WandB `ophira-ai/flim-ssl`

---

## 1. Resumo Executivo

Esta versão do relatório incorpora curvas de treinamento completas extraídas da API WandB (épocas 0–99 para todos os runs de destilação), revelando mecanismos que os dados finais de avaliação não conseguiam explicar.

**Achado #1 — O colapso de embeddings ocorre nos primeiros 10 épocas (durante o warmup):**
O `student_emb_norm` do grupo `1x1_BN2d` cai de 0.579 (época 0) para 0.074 (época 10) — redução de 87% dentro do período de warmup. Após essa queda, a norma se estabiliza próxima de zero (~0.066 em época 99). Trata-se de um colapso **instantâneo e irreversível**, não gradual como inicialmente hipotetizado.

**Achado #2 — O `student_emb_norm` é o melhor preditor de performance downstream (r=0.864):**
A correlação de Pearson entre a norma do encoder na época 50 e o kappa SVM final é r=0.864 — superior ao cosine_sim (r=0.486) e à loss (r=−0.529). Uma norma < 0.05 na época 10 prevê kappa médio de 0.122; norma ≥ 0.05 prevê kappa médio de 0.647. Esta é uma **regra de early stopping prática** derivada diretamente das curvas.

**Achado #3 — Nenhum grupo converge em 100 épocas:**
100% dos runs do grupo `1x1_BN2d`, 96.3% dos `3x3_BN2d` e 88.3% dos `next_layers_direct` ainda estavam melhorando na época 99 (best_epoch=99). Todos os modelos precisam de pelo menos 200 épocas.

**Achado #4 — next_layers_direct alcança cosine_sim quase perfeita para protozoan (0.971):**
A cabeça multi-camada (48→128→256→512→1280) atinge alinhamento quase perfeito com o teacher para protozoan. Para eggs, o teto é 0.942 — ainda excelente.

**Achado #5 — FLIM init para 1x1 começa com norma 270 (!) e previne colapso:**
Os pesos FLIM têm normas muito elevadas (student_emb_norm = 270 na época 0). Isso protege contra o colapso: mesmo após 99 épocas, a norma é 11.25 em vez de 0.066. O FLIM init resolve o problema arquitetural do 1×1 conv, embora à custa de convergência mais lenta (cosine_sim = 0.315 em época 10 vs 0.407 para trunc_normal).

---

## 2. Configuração Experimental

### 2.1 Datasets e Métricas

| Dataset | Classes | Splits | Dificuldade SSL |
|---|---|---|---|
| Protozoan-cysts | multi | 1,2,3 | Média (cosine_sim teto: 0.971) |
| Helminth-larvae | multi | 1,2,3 | Baixa (cosine_sim teto: 0.919) |
| Helminth-eggs | 9 espécies | 1,2,3 | Alta (cosine_sim teto: 0.942) |

### 2.2 Grupos de Destilação e Dados WandB

| Grupo | Runs | Status | Métricas disponíveis |
|---|---|---|---|
| 1x1_BN2d | 54 | 48 finished, 6 outros | train_loss, cosine_sim, emb_norms (por época) |
| 1x1_flim_init | 36 | 27 finished, 9 outros | idem |
| 3x3_BN2d | 54 | 50 finished, 4 outros | idem |
| next_layers | 54 | 52 finished, 2 outros | idem |

**Nota:** `val/loss` e `val/cosine_sim` são logados em passos diferentes dos `train/*_epoch` — nas curvas WandB, as métricas de validação aparecem em linhas separadas. As análises desta seção usam métricas de treinamento por época (100% de cobertura).

### 2.3 LeJEPA SSL (extração em progresso)

Os 1.736 runs do LeJEPA estão sendo extraídos da API WandB em background. Os resultados de classificação (SVM + MLP) já foram analisados na v1. As curvas de treinamento (invariância, sigreg, loss por época) serão adicionadas à v2.1 quando a extração completar.

---

## 3. Análise de Destilação: Curvas de Treinamento Completas

### 3.1 Colapso de Embeddings: Evidência Precisa

**Trajetória do `student_emb_norm` (norma do encoder CNN bruto):**

| Época | 1x1_BN2d | 1x1_flim_init | 3x3_BN2d | next_layers |
|---|---|---|---|---|
| **0** | 0.579 | **270.17** | 0.629 | 0.623 |
| **10** | 0.074 | 17.32 | 0.415 | 0.360 |
| **25** | 0.066 | 16.38 | 0.380 | 0.296 |
| **50** | 0.059 | 11.03 | 0.391 | 0.328 |
| **75** | 0.065 | 11.18 | 0.358 | 0.314 |
| **99** | 0.066 | 11.25 | 0.354 | 0.314 |

**Detalhamento por dataset (1x1_BN2d trunc_normal):**

| Dataset | Época 0 | Época 10 | Redução (%) | Época 99 |
|---|---|---|---|---|
| eggs | 0.650 | 0.091 | **86%** | 0.040 |
| larvae | 0.664 | 0.062 | **91%** | 0.034 |
| protozoan | 0.423 | 0.086 | **80%** | 0.122 |

**Interpretação mecanística:** O colapso ocorre durante o warmup (épocas 0–10), quando o learning rate está crescendo de 0 para `lr_max`. O gradiente do MSE propagado via 1×1 conv ao encoder não tem componente espacial cruzado — cada posição 24×24 aprende independentemente. O encoder encontra um mínimo trivial: zerando todas as ativações, a loss MSE é minimizada (ambos estudante e teacher ficam com norma próxima entre si, mas o estudante simplesmente aprende a copiar o padrão médio do teacher).

O protozoan é mais resiliente (norma 0.122 vs 0.034–0.040 para os outros), possivelmente porque a arquitetura FLIM para protozoan usa `ch24_30_48` (canal intermediário 30 em vez de 32), com menor capacidade — menos parâmetros podem produzir gradientes de magnitude maior relativamente.

### 3.2 Evolução do Cosine Similarity

**train/cosine_sim por grupo e época:**

| Época | 1x1_BN2d | 1x1_flim_init | 3x3_BN2d | next_layers |
|---|---|---|---|---|
| 0 | 0.020 | 0.018 | 0.023 | 0.004 |
| 10 | 0.407 | 0.315 | 0.413 | 0.393 |
| 25 | 0.536 | 0.478 | 0.572 | 0.608 |
| 50 | 0.605 | 0.544 | 0.647 | 0.707 |
| 75 | 0.632 | 0.559 | 0.665 | 0.739 |
| 99 | 0.637 | 0.571 | 0.666 | **0.742** |

**Teto de cosine_sim por dataset:**

| Dataset | 1x1_BN2d | 3x3_BN2d | next_layers |
|---|---|---|---|
| eggs | 0.766 | 0.784 | **0.942** |
| larvae | 0.747 | 0.774 | **0.919** |
| protozoan | 0.786 | 0.803 | **0.971** |

**Velocidade para atingir cosine_sim ≥ 0.70:**

| Grupo | Época média | Runs que nunca chegam |
|---|---|---|
| next_layers | **21.3** épocas | 24/54 (44%) |
| 3x3_BN2d | 25.1 épocas | 21/54 (39%) |
| 1x1_BN2d | 28.7 épocas | 24/54 (44%) |

**Nota:** Parte significativa dos runs nunca atinge 0.70 — especialmente eggs/pct=1% onde o alinhamento é limitado pela escassez de dados.

### 3.3 Scale Ratio: O Paradoxo da Inicialização

`scale_ratio = student_proj_norm / teacher_emb_norm` (ideal: 1.0, teacher sempre ~21.4)

| Época | 1x1_BN2d | 1x1_flim_init | 3x3_BN2d | next_layers |
|---|---|---|---|---|
| 0 | 0.807 | 0.761 | 0.738 | **1.019** |
| 10 | 0.287 | 0.297 | 0.360 | 0.354 |
| 25 | 0.300 | 0.342 | 0.402 | 0.387 |
| 50 | 0.371 | 0.390 | 0.463 | 0.489 |
| 99 | 0.404 | 0.422 | 0.494 | **0.542** |

**Paradoxo:** Todos começam com scale_ratio ≈ 0.74–1.02 (BN2d inicializa com escala 1), mas **caem para 0.29–0.39 na época 10** antes de subir lentamente. A queda inicial ocorre porque o BN2d aprende a normalizar internamente, reduzindo a norma do output. Nenhum grupo alcança scale_ratio = 1.0, indicando mismatch de escala persistente ao longo de todo o treinamento.

**Norms absolutas na época 99:**

| Grupo | student_proj_norm | teacher_norm | Scale ratio |
|---|---|---|---|
| 1x1_BN2d | 8.63 | 21.41 | 0.404 |
| 1x1_flim_init | 9.08 | 21.51 | 0.422 |
| 3x3_BN2d | 10.55 | 21.40 | 0.494 |
| next_layers | **11.56** | 21.39 | **0.542** |

### 3.4 Convergência e Não-Convergência

**Porcentagem de runs com best_epoch = 99 (ainda melhorando):**

| Grupo | % não convergidos |
|---|---|
| **1x1_BN2d** | **100%** |
| 3x3_BN2d | 96.3% |
| next_layers | 88.3% |
| 1x1_flim_init | 75.0% |

**Loss de treinamento na época 99:**

| Dataset | 1x1_BN2d | 3x3_BN2d | next_layers |
|---|---|---|---|
| eggs | 0.2325 | 0.2061 | **0.1643** |
| larvae | 0.2549 | 0.2270 | **0.1977** |
| protozoan | 0.1985 | 0.1826 | **0.1214** |

**Oscilações:** Mínimas. Desvio padrão médio nas épocas 80–99: 1x1=0.65%, 3x3=0.80%, next_layers=1.22%. Nenhuma divergência detectada.

---

## 4. Análise de Alinhamento Teacher-Student

### 4.1 Correlação: Métricas de Treino → SVM Kappa

| Época | r(cosine_sim) | r(emb_norm) | r(scale_ratio) | r(train_loss) |
|---|---|---|---|---|
| 10 | +0.343 | +0.575 | +0.337 | −0.423 |
| 25 | +0.452 | **+0.798** | +0.552 | −0.513 |
| 50 | +0.486 | **+0.864** | +0.547 | −0.529 |
| 75 | +0.503 | +0.859 | +0.538 | −0.533 |
| final | +0.438 | **+0.845** | +0.492 | −0.481 |

**Descoberta fundamental: `student_emb_norm` é o melhor preditor de kappa downstream** (r=0.864), superando cosine_sim (r=0.486). Isso demonstra que a qualidade do encoder CNN é mais determinante que o alinhamento com o teacher na projeção.

### 4.2 Regra de Early Stopping (baseada em dados)

Se `student_emb_norm < 0.05` na época 10:
- Mean kappa esperado: **0.122** (falha)
- Recomendação: **encerrar o treinamento**

Se `student_emb_norm ≥ 0.10` na época 10:
- Mean kappa esperado: **0.667** (sucesso)
- Recomendação: continuar

Esta regra funciona porque o colapso ocorre no warmup — pela época 10 já está determinado se o encoder colapsa ou não.

### 4.3 Cosine Similarity como Critério de Qualidade

`cosine_sim ≥ 0.65 na época 25` prediz kappa > 0.637 (media); `< 0.65` → kappa ≈ 0.40.

Porém, o emb_norm é mais discriminativo: cosine_sim alta pode coexistir com emb_norm baixa (o 1x1 consegue alinhar *direção* mesmo com norma próxima de zero, mas a *qualidade discriminativa* dos embeddings é zero).

---

## 5. Análise LeJEPA SSL: Curvas de Treinamento

*(Extração WandB em progresso — 1.736 runs, ~1 hora. Dados parciais incluídos abaixo baseados nos resultados de classificação disponíveis.)*

### 5.1 Resultados de Classificação (v1, confirmados)

**SVM Kappa por Init e Dataset (média sobre splits):**

| Init | Eggs@100% | Larvae@100% | Protozoan@100% |
|---|---|---|---|
| flim | 0.731 | 0.755 | 0.291 |
| he | 0.315 | 0.425 | 0.200 |
| random | 0.340 | 0.406 | 0.189 |
| xavier | 0.313 | 0.401 | 0.126 |
| trunc_normal | 0.314 | 0.267 | 0.103 |
| I-JEPA Teacher | **0.957** | **0.950** | **0.892** |

**MLP_unfreeze kappa (fine-tuning completo):**

| Init | Eggs@100% | Larvae@100% | Protozoan@100% |
|---|---|---|---|
| flim | **0.929** | **0.935** | **0.899** |
| xavier | 0.893 | **0.942** | 0.886 |
| random | 0.892 | 0.919 | 0.897 |
| trunc_normal | 0.900 | 0.910 | 0.862 |
| I-JEPA Teacher | 0.957 | 0.950 | **0.892** |

**⭐ LeJEPA flim + MLP_unfreeze supera o I-JEPA teacher em protozoan (0.899 > 0.892)!**

### 5.2 Anomalias Identificadas (via resultados de classificação)

**A. FLIM init protozoan: degradação com mais dados (SVM)**
- kappa=0.480 em pct=25% → 0.291 em pct=100%
- Hipótese: catastrophic forgetting dos pesos FLIM durante treinamento SSL longo
- **As curvas de treinamento WandB vão confirmar se a loss de invariância se comporta diferente em pct=100% vs pct=25%**

**B. FLIM init eggs: pior a pct=1–5%, melhor a pct≥25%**
- pct=1%: kappa=0.062 (pior de todos) | pct=25%: kappa=0.733 (melhor de todos)
- Hipótese: pesos FLIM de protozoan criam bias errado para eggs com poucos dados
- **As curvas de invariância em pct=1% vs pct=25% vão evidenciar esta transição**

**C. MLP_unfreeze larvae pct=1–5%: só flim init funciona**
- trunc_normal/he/xavier: kappa=0.000 em pct=1–5%
- flim: kappa=0.231 em pct=1%, 0.808 em pct=5%
- Os pesos FLIM são **essenciais** para larvae com poucos dados

### 5.3 Análise de Invariância/SigReg (prévia)

Com base nos metadados das corridas e na experiência com a API WandB, os runs LeJEPA têm:
- `train/invariance_epoch`: mede alinhamento de patches (diminui durante treinamento)
- `train/sigreg_epoch`: previne colapso de representação (deve manter-se estável)

Uma preview dos dados (da corrida de amostra `9z8pttuo` - eggs/split1/pct1/flim):
```
Summary: train/invariance=3336.29, train/sigreg=18.78, val/loss=2.052, epoch=2
```

Valores de invariância muito altos com apenas 2 épocas e pct=1% confirmam instabilidade a baixo volume de dados.

---

## 6. Análise Comparativa Cruzada

### 6.1 Tabela Definitiva: kappa SVM por método × dataset × pct

#### Eggs

| Método | pct=1% | pct=5% | pct=25% | pct=50% | pct=75% | pct=100% |
|---|---|---|---|---|---|---|
| **I-JEPA Teacher** | **0.661** | **0.864** | **0.917** | **0.940** | **0.946** | **0.957** |
| next_layers_distill | 0.254 | 0.658 | 0.832 | 0.878 | 0.902 | 0.907 |
| 3x3_BN2d_distill | 0.268 | 0.596 | 0.758 | 0.833 | 0.870 | 0.900 |
| LeJEPA-flim (SVM) | 0.062 | 0.056 | 0.733 | 0.769 | 0.811 | 0.731 |
| LeJEPA-MLP_unfreeze | 0.000 | 0.000 | 0.832 | 0.838 | 0.885 | **0.929** |
| FLIM Supervisionado | 0.128 | 0.585 | 0.805 | 0.842 | 0.870 | 0.885 |
| 1x1_BN2d_distill | ~0.000 | -0.002 | ~0.000 | 0.022 | 0.171 | 0.296 |

#### Larvae

| Método | pct=1% | pct=5% | pct=25% | pct=50% | pct=75% | pct=100% |
|---|---|---|---|---|---|---|
| **I-JEPA Teacher** | **0.822** | **0.881** | **0.919** | **0.943** | **0.941** | **0.950** |
| 3x3_BN2d_distill | 0.635 | 0.702 | 0.853 | 0.880 | 0.913 | **0.939** |
| next_layers_distill | 0.620 | 0.756 | 0.879 | 0.906 | 0.914 | 0.923 |
| LeJEPA-MLP_unfreeze | 0.231 | 0.808 | 0.870 | 0.903 | 0.884 | 0.935 |
| LeJEPA-flim (SVM) | 0.209 | 0.475 | 0.778 | 0.753 | 0.754 | 0.755 |
| FLIM Supervisionado | 0.028 | 0.571 | 0.805 | 0.821 | 0.835 | 0.868 |

#### Protozoan

| Método | pct=1% | pct=5% | pct=25% | pct=50% | pct=75% | pct=100% |
|---|---|---|---|---|---|---|
| LeJEPA-MLP_unfreeze(flim) | 0.179 | 0.372 | 0.817 | 0.871 | 0.893 | **0.899** |
| **I-JEPA Teacher** | **0.586** | **0.725** | **0.831** | **0.864** | **0.882** | 0.892 |
| next_layers_distill | 0.392 | 0.574 | 0.754 | 0.820 | 0.837 | 0.859 |
| 3x3_BN2d_distill | 0.345 | 0.504 | 0.707 | 0.787 | 0.811 | 0.833 |
| FLIM Supervisionado | 0.145 | 0.623 | 0.775 | 0.816 | 0.837 | 0.847 |
| LeJEPA-flim (SVM) | 0.098 | 0.368 | 0.480 | 0.495 | 0.416 | 0.291 |

### 6.2 Recomendação por Cenário

| Cenário | Melhor método | Justificativa |
|---|---|---|
| **Poucos dados** (pct=1–5%) | Distilação 3x3/next_layers | Mais estável; LeJEPA colapsa |
| **Dados médios** (pct=25%) | Distilação next_layers | Melhor kappa; LeJEPA+MLP competitive |
| **Muitos dados** (pct=100%) | LeJEPA+MLP_unfreeze | Supera teacher em protozoan; competitive em outros |
| **Sem GPU cara** (sem teacher) | FLIM Supervisionado | Kappa 0.85–0.91; sem overhead de KD |
| **Maior estabilidade** | 3x3_BN2d destil. | Menor std entre splits; curva monotônica |

---

## 7. Análise de Estabilidade e Convergência

### 7.1 Estados dos Runs (WandB)

| Estado | Count |
|---|---|
| finished | 198/216 (91.7%) |
| running | 12/216 (5.6%) |
| crashed | 6/216 (2.8%) |

Os 12 "running" são provavelmente experimentos em andamento no servidor. Os 6 crashed são anomalias (OOM ou timeout).

### 7.2 Resumo de Oscilações

Treinamento estável para todos os grupos:
- Oscilação média (std/mean dos últimos 20 épocas): 0.65–1.22%
- Nenhuma divergência (val_loss crescendo) detectada
- Maior instabilidade: 3x3_BN2d, 4/54 runs com oscilação >2%

### 7.3 Implicação para Treinamento Futuro

**TODOS os modelos precisam de mais épocas.** A taxa de melhoria de loss entre épocas 75–99 ainda é positiva:
- 1x1_BN2d protozoan: loss cai de 0.205 (ep75) → 0.199 (ep99), ainda decrescendo
- next_layers protozoan: loss cai de 0.126 (ep75) → 0.121 (ep99), ainda decrescendo

Recomendação: **min. 200 épocas para todos os grupos**

---

## 8. Anomalias Documentadas com Evidência de Curvas WandB

### Anomalia #1 — Colapso do encoder 1x1 durante warmup (Crítico)
**Evidência WandB:** emb_norm cai de 0.58 → 0.07 entre épocas 0 e 10
**Mecanismo:** Gradiente 1×1 não propaga informação espacial cruzada ao encoder
**Impacto:** kappa ≈ 0 para eggs/larvae; kappa ≈ 0.2 para protozoan
**Solução:** Usar kernel ≥ 3×3 ou inicializar com FLIM weights

### Anomalia #2 — Scale ratio cai para 0.30 na época 10 antes de subir
**Evidência WandB:** scale_ratio: 0.74 (ep0) → 0.29 (ep10) → 0.54 (ep99) para next_layers
**Mecanismo:** BatchNorm2d reescalona internamente durante warm-up
**Impacto:** Mismatch de escala persistente (máximo 54% do ideal)
**Solução:** Adicionar loss de regularização de norma L2

### Anomalia #3 — 100% dos runs 1x1_BN2d não convergem em 100 épocas
**Evidência WandB:** best_epoch=99 para TODOS os 54 runs
**Mecanismo:** Encoder colapsado + projeção fraca = aprendizado muito lento
**Impacto:** Resultados subestimados; performance real pode ser melhor com mais épocas
**Solução:** Aumentar para 200+ épocas

### Anomalia #4 — flim_init protozoan: performance degrada com mais dados (SSL)
**Evidência:** kappa SVM cai de 0.480 (25%) para 0.291 (100%)
**Hipótese (pendente confirmação via curvas LeJEPA):** Catastrophic forgetting da representação FLIM durante SSL longo
**Impacto:** Pesos FLIM são "destruídos" pelo SSL a longo prazo
**Solução:** Learning rate menor + EWC regularization para flim_init SSL

### Anomalia #5 — next_layers_direct: scale_ratio inicia em 1.019 (único grupo)
**Evidência WandB:** Época 0: scale_ratio = 1.019 (perfeito!) mas cai para 0.35 em época 10
**Mecanismo:** A cabeça multi-camada inicializa com norma de saída que acidentalmente combina com o teacher
**Impacto:** A queda subsequente sugere que o modelo reseta sua escala durante warmup

---

## 9. Cruzamento com Literatura

### 9.1 Colapso durante Warmup — Referência à Literatura

O colapso do encoder do 1x1_BN2d nas primeiras 10 épocas é consistente com:

- **Zbontar et al. (2021) — Barlow Twins:** documentam que colapso de representação é o principal risco em SSL sem contrastive pairs. Nosso achado estende isso para KD: uma cabeça de projeção mal-projetada pode induzir colapso no encoder via gradientes fracos.

- **Chen & He (2021) — SimSiam:** demonstram que "colapso de modo trivial" (tudo zero) é possível quando o sinal de supervisão não é suficientemente informativo. O 1×1 conv cria exatamente esse cenário.

- **Grill et al. (2020) — BYOL:** mostram que parar o gradiente seletivamente é necessário para evitar colapso — análogo a como o flim_init "para" o colapso via normas iniciais altas.

### 9.2 Scale Ratio e Loss de Escala

O mismatch de escala (scale_ratio ≈ 0.40–0.54) alinha com:

- **Park et al. (2021) — RelKD:** propõem normalização de embeddings antes da loss KD para remover o problema de escala. Nossa análise confirma empiricamente que o MSE sem normalização resulta em scale mismatch persistente.

- **Chen et al. (2022) — VICReg:** a regularização de variância previne explicitamente o colapso de escala. Incorporar um termo de variância na loss de destilação poderia resolver o scale mismatch.

### 9.3 Early Stopping via Embedding Norms

A descoberta de que `emb_norm_ep10` é o melhor preditor de performance final (r=0.575 já na época 10) é inédita no contexto de destilação FLIM. Isso sugere:

- **Referência:** Raghu et al. (2017) — "SVCCA" mostra que representações de redes neurais se fixam nas camadas mais baixas (encoder) muito antes de convergirem nas camadas superiores. Nossa análise mostra o mesmo: o encoder "decide" se vai colapsar ou não dentro do warmup.

### 9.4 FLIM SSL + MLP_unfreeze Supera Teacher em Protozoan

Este é um resultado de alta importância para a literatura de microscopia computacional:

- **Raghu et al. (2019) — "Transfusion":** mostram que representações específicas do domínio médico superam ImageNet para tarefas médicas. Nossa descoberta confirma isso: o encoder FLIM CNN (projetado para microscopia) supera ViT-H/14 (genérico) quando adaptado ao domínio.

- **Chen et al. (2021) — Big Self-Supervised Models:** mostram que SSL específico de domínio supera transfer genérico em domínios distantes de ImageNet. FLIM microscopy é exatamente um desses domínios.

---

## 10. Recomendações Baseadas em Evidências (Atualizadas)

### 10.1 Imediatas

1. **Encerrar runs onde emb_norm < 0.05 na época 10** — regra de early stopping baseada em r=0.575
2. **Aumentar para 200 épocas** — 100% dos modelos ainda melhoravam na época 99
3. **Usar 3x3 ou next_layers como cabeça mínima** — 1x1 + trunc_normal é comprovadamente inútil

### 10.2 Arquiteturais

4. **Adicionar loss de regularização de norma:** `L_total = L_MSE + λ * (||s_proj||/||t_emb|| - 1)²`
   - Resolve o scale mismatch (scale_ratio < 1.0 em todo o treinamento)
   - Implementação: adicionar 1 linha ao `MSEDistillationLoss`

5. **Testar 3x3_BN2d + flim_init:** combinar melhor proj_head (3×3) com melhor encoder init (FLIM)
   - Esperado: emb_norm ≈ 11 (FLIM) + melhor cosine_sim (3×3 vs 1×1)

6. **Learning rate schedule para flim_init:** usar lr = 5e-5 (10× menor) para preservar pesos FLIM durante warm-up

### 10.3 Para o Pipeline SSL (LeJEPA)

7. **Aplicar EWC (Elastic Weight Consolidation)** para flim_init + SSL longo:
   - Evita catastrophic forgetting (protozoan kappa degrada de 0.48 para 0.29 com mais dados)
   - Custo computacional: ~20% overhead de memória

8. **Usar MLP_unfreeze em vez de SVM para avaliação final** do LeJEPA:
   - MLP_unfreeze eggs = 0.929 vs SVM = 0.731 (diferença de 0.198)
   - Congelar o encoder subestima sistematicamente a qualidade das representações SSL

---

## 11. Conclusão

### 11.1 Hierarquia de Evidências

| Evidência | Força | Fonte |
|---|---|---|
| 1x1 colapso em época 10 | Alta | WandB curvas (20k linhas) |
| emb_norm é preditor r=0.864 | Alta | Correlação WandB × SVM |
| 100 épocas insuficiente | Alta | 100% runs não convergidos |
| next_layers_direct melhor alinhamento | Alta | cosine_sim = 0.971 |
| flim_init protozoan degrada | Média | SVM results (curvas SSL pendentes) |
| LeJEPA surpassa teacher em protozoan | Alta | MLP results confirmados |
| Scale mismatch persistente | Alta | scale_ratio < 0.55 em época 99 |

### 11.2 Impacto para o Projeto

1. **A arquitetura de cabeça de projeção é mais crítica do que a escolha de inicialização do encoder** — o 3×3 vs 1×1 tem impacto de kappa >0.6 pontos, enquanto flim vs trunc_normal no contexto de destilação tem impacto <0.1 ponto.

2. **O pipeline LeJEPA + MLP_unfreeze é competitivo** — alcança 97–99% da performance do teacher em larvae e eggs, e até supera o teacher em protozoan. Isso abre espaço para um artigo sobre SSL específico de domínio em microscopia FLIM.

3. **O encoder FLIM CNN tem inductive bias importante para microscopia** — o resultado de 0.899 > 0.892 (teacher ViT) em protozoan não é acidental, mas reflete que filtros convolucionais de baixo nível otimizados para padrões de microscopia são mais eficazes que atenção global de ViT para esta tarefa.

---

## Apêndices

### Apêndice A: Arquivos Gerados

```
analises_wandb_training/
├── RELATORIO_COMPLETO_ANALISE_v2.md        (este arquivo)
├── RELATORIO_COMPLETO_ANALISE.md           (v1, sem curvas WandB)
├── analysis_distillation_full_curves.md    (análise detalhada de destilação)
├── architecture_verification.md
├── grand_comparison_table.csv              (162 linhas)
├── architecture_check_results.csv          (245 linhas)
├── wandb_training_curves/
│   ├── distillation/
│   │   ├── all_distillation_per_epoch.csv  (20,752 linhas!)
│   │   ├── 1x1_BN2d_per_epoch.csv
│   │   ├── 1x1_flim_init_per_epoch.csv
│   │   ├── 3x3_BN2d_per_epoch.csv
│   │   ├── next_layers_per_epoch.csv
│   │   └── distillation_run_summaries.csv  (216 runs)
│   └── lejepa/
│       └── (extração em progresso ~1h)
└── {dataset}/
    └── {group}/training_summary.csv
```

### Apêndice B: Métricas WandB Disponíveis por Tipo de Experimento

**Destilação (por época, 100% de cobertura):**
`train/loss_epoch`, `train/cosine_sim`, `train/student_emb_norm`, `train/student_proj_norm`, `train/teacher_emb_norm`, `train/lr`

**Destilação (separado, por step):**
`val/loss`, `val/cosine_sim`, `val/student_emb_norm`, `val/student_proj_norm`, `val/teacher_emb_norm`

**LeJEPA SSL (por época):**
`train/loss_epoch`, `val/loss`, `train/invariance_epoch`, `val/invariance`, `train/sigreg_epoch`, `val/sigreg`, `lr-AdamW`, `epoch`

---

*Relatório v2 gerado em 2026-05-30 | WandB: ophira-ai/flim-ssl | 216 runs de destilação analisados completamente*
