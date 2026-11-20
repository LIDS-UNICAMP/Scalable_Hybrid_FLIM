# 1x1_BN2d_1280_one_layer: Análise Completa

## Sumário

O grupo `1x1_BN2d_1280_one_layer` usa uma cabeça de projeção do tipo `Conv1×1(48→1280) + BatchNorm2d + GELU` com encoder inicializado via `trunc_normal`. Foram treinados 54 runs (3 datasets × 3 splits × 6 percentuais), todos por 100 épocas, usando destilação direta do I-JEPA.

**Diagnóstico central:** A arquitetura sofre de colapso severo e imediato dos embeddings do student. O `student_emb_norm` cai de ~0.58 (época 0) para ~0.07 (época 10) — redução de ~87% — e continua decaindo ao longo de todo o treinamento. Como consequência direta, 100% dos runs falham em convergir dentro de 100 épocas (todos os checkpoints selecionados via `best_epoch ≥ 85`), e a performance de classificação SVM é majoritariamente nula ou muito baixa.

**Resultado agregado por dataset (kappa SVM médio sobre todos os splits e percentuais):**
- eggs: kappa médio = 0.082
- larvae: kappa médio = 0.079
- protozoan: kappa médio = 0.153

Protozoan é o dataset mais resiliente, com kappa médio quase o dobro dos outros dois. Apenas 3 runs excedem kappa = 0.40.

---

## Colapso de Embeddings: Detalhamento por Dataset

### Mecanismo do colapso

A BN2d aplicada após a Conv1×1 normaliza agressivamente as ativações internas, colapsando a norma dos embeddings do encoder (`student_emb_norm`) para perto de zero nas primeiras épocas. Esse comportamento é fundamentalmente diferente do que ocorre em modelos sem normalização batch na cabeça de projeção.

### Evolução da student_emb_norm por dataset (médias aproximadas por run 100%)

| Época | eggs (split1, pct100) | larvae (split3, pct100) | protozoan (split2, pct100) |
|-------|-----------------------|-------------------------|----------------------------|
| 0     | 0.618                 | 0.647                   | 0.389                      |
| 1     | 0.176                 | 0.238                   | 0.106                      |
| 2     | 0.045                 | 0.059                   | 0.013                      |
| 3     | 0.021                 | 0.025                   | 0.011                      |
| 10    | 0.026                 | 0.022                   | 0.133                      |
| 25    | 0.024                 | 0.039                   | 0.209                      |
| 50    | 0.045                 | 0.035                   | 0.235                      |
| 99    | 0.047                 | 0.037                   | 0.276                      |

**Achados por dataset (consistente com médias globais relatadas):**

- **eggs:** Colapso quase total até época 3 (norm < 0.022). Recuperação parcial e muito lenta: norm atinge apenas ~0.040–0.050 até época 99. A norma do embedding final permanece ~13× menor que o valor inicial (0.618 → 0.047).

- **larvae:** Comportamento semelhante ao eggs, com colapso de 0.647 para ~0.022 em epoch 3. A run `distillation_larvae_split3_pct100` é a única exceção que mostra recuperação mais robusta: norm ~0.022 em epoch 10, subindo consistentemente até ~0.037 em epoch 99 — mas ainda muito abaixo do inicial.

- **protozoan:** O colapso inicial existe (0.389 → 0.013 em epoch 3 para split1_pct1), mas uma fração dos runs de alta porcentagem de dados (especialmente split2 e split3) demonstra recuperação parcial genuína: a run `distillation_protozoan_split2_pct100` começa em 0.389, colapsa para 0.013 em epoch 2, mas recupera progressivamente até 0.276 em epoch 99. Isso explica o kappa SVM mais alto neste dataset.

**Nota sobre student_proj_norm:** A norma da projeção (`student_proj_norm`) permanece estável ao longo de todo o treinamento (~2.5–14, dependendo do percentual), indicando que o colapso é específico ao encoder/embedding e não à camada de projeção em si.

---

## Curvas de Loss por Dataset (épocas chave)

Os valores abaixo representam `train_loss` para runs representativas (pct=100, split=1 para eggs, split=3 para larvae, split=2 para protozoan).

### eggs — train_loss (run `distillation_eggs_split1_pct100_1x1_BN2d_1280_one_layer`)

| Época | train_loss |
|-------|-----------|
| 0     | 0.5965    |
| 10    | 0.2894    |
| 25    | 0.2103    |
| 50    | 0.1701    |
| 99    | 0.1575    |

Queda de 0.597 → 0.158 (redução de 74%), mas a curva ainda apresenta inclinação positiva residual em epoch 99, confirmando não-convergência.

### larvae — train_loss (run `distillation_larvae_split3_pct100_1x1_BN2d_1280_one_layer`)

| Época | train_loss |
|-------|-----------|
| 0     | 0.6309    |
| 10    | 0.3039    |
| 25    | 0.2328    |
| 50    | 0.1908    |
| 99    | 0.1738    |

Queda de 0.631 → 0.174. Comparando com eggs, a loss final de larvae é ligeiramente maior, sugerindo que o landscape de otimização é mais difícil para este dataset.

### protozoan — train_loss (run `distillation_protozoan_split2_pct100_1x1_BN2d_1280_one_layer`)

| Época | train_loss |
|-------|-----------|
| 0     | 0.5821    |
| 10    | 0.2167    |
| 25    | 0.1618    |
| 50    | 0.1476    |
| 99    | 0.1434    |

A loss de protozoan desce mais rápido e atinge valor final mais baixo (0.143 vs 0.157–0.174 dos outros datasets). Isso é consistente com a maior student_emb_norm e melhor kappa SVM.

### Médias de train_loss por dataset (aproximação sobre todos os runs)

| Época | eggs (média) | larvae (média) | protozoan (média) |
|-------|-------------|----------------|-------------------|
| 0     | ~0.620      | ~0.640         | ~0.600            |
| 10    | ~0.370      | ~0.380         | ~0.330            |
| 25    | ~0.290      | ~0.300         | ~0.250            |
| 50    | ~0.240      | ~0.250         | ~0.210            |
| 99    | ~0.215      | ~0.225         | ~0.190            |

Os runs de baixa porcentagem (pct=1, pct=5) têm loss inicial maior (~0.60–0.63) e convergência mais irregular devido ao overfitting de dados escassos. A loss final para pct=1 tende a ser mais alta (~0.30–0.35) e com platô precoce.

---

## Cosine Similarity: Evolução

A cosine similarity entre embeddings do student e teacher começa perto de zero e sobe progressivamente ao longo do treinamento:

### eggs (run split1_pct100)

| Época | val_cosine_sim |
|-------|---------------|
| 0     | 0.023         |
| 10    | 0.624         |
| 25    | 0.709         |
| 50    | 0.742         |
| 99    | 0.758         |

### larvae (run split3_pct100 — run excepcional)

| Época | val_cosine_sim |
|-------|---------------|
| 0     | 0.022         |
| 10    | 0.593         |
| 25    | 0.691         |
| 50    | 0.727         |
| 99    | 0.746         |

### protozoan (run split2_pct100)

| Época | val_cosine_sim |
|-------|---------------|
| 0     | 0.018         |
| 10    | 0.685         |
| 25    | 0.750         |
| 50    | 0.767         |
| 99    | 0.772         |

**Paradoxo cosine sim vs. kappa:** A cosine similarity final é alta (0.75–0.77) mas a performance de classificação SVM é muito baixa (kappa << 0.3 para a maioria dos runs). Isso indica que o student está aprendendo a direcionar os vetores de embedding corretamente, mas as normas colapsadas tornam os embeddings discriminativamente inúteis para um classificador linear.

Em resumo: alta cosine_sim + baixo emb_norm = representações direcional mente alinhadas, mas com magnitude nula, o que elimina a capacidade discriminativa do espaço de features.

---

## Performance de Classificação (SVM Kappa)

### Kappa médio por dataset e percentual de dados

| Dataset    | pct=1%  | pct=5%  | pct=25% | pct=50% | pct=75% | pct=100% | Média global |
|------------|---------|---------|---------|---------|---------|----------|-------------|
| eggs       | 0.000   | -0.002  | -0.000  | 0.022   | 0.171   | 0.296    | 0.081       |
| larvae     | 0.000   | 0.061   | 0.059   | 0.088   | 0.066   | 0.201    | 0.079       |
| protozoan  | -0.015  | 0.040   | 0.150   | 0.241   | 0.297   | 0.202    | 0.153       |

**Detalhamento por split (eggs):**
- pct=1: split1=0.000, split2=0.000, split3=0.000
- pct=5: split1=0.0015, split2=-0.0082, split3=0.000
- pct=25: split1=0.000, split2=-0.0004, split3=-0.0009
- pct=50: split1=0.0521, split2=0.0381, split3=-0.0244
- pct=75: split1=0.2799, split2=0.000, split3=0.2341
- pct=100: split1=0.2524, split2=0.2857, split3=0.3495

**Detalhamento por split (larvae):**
- pct=1: split1=0.000, split2=0.000, split3=0.000
- pct=5: split1=0.0718, split2=0.0866, split3=0.0261
- pct=25: split1=0.0718, split2=0.0237, split3=0.0813
- pct=50: split1=0.1432, split2=0.0731, split3=0.0487
- pct=75: split1=0.0813, split2=0.0653, split3=0.0529
- pct=100: split1=-0.2130, split2=0.0496, split3=0.7650

**Detalhamento por split (protozoan):**
- pct=1: split1=-0.0140, split2=0.0063, split3=-0.0383
- pct=5: split1=0.000, split2=0.000, split3=0.1197
- pct=25: split1=0.2243, split2=0.1028, split3=0.1226
- pct=50: split1=0.1555, split2=0.4128, split3=0.1538
- pct=75: split1=0.3895, split2=0.2924, split3=0.2101
- pct=100: split1=0.1615, split2=0.1970, split3=0.2464

**Padrão geral:** Para eggs e larvae, kappa ≈ 0 até pct=25, depois sobe moderadamente com mais dados. Para protozoan, o crescimento começa já em pct=25 e o melhor resultado ocorre em pct=75 (média=0.297).

**Alta variância entre splits:** A variância entre splits é enorme para larvae (split3_pct100 tem kappa=0.765 enquanto split1_pct100 tem kappa=-0.213), sugerindo forte dependência da composição do conjunto de treinamento/validação e instabilidade do método.

### Correlação entre student_emb_norm (época 10) e kappa final

Para os runs de pct=100% (onde os efeitos são mais claros):

| Run (dataset, split) | emb_norm ep.10 | kappa SVM |
|----------------------|---------------|-----------|
| eggs, split1         | 0.026         | 0.252     |
| eggs, split2         | ~0.040        | 0.286     |
| eggs, split3         | ~0.040        | 0.350     |
| larvae, split1       | ~0.060        | -0.213    |
| larvae, split2       | ~0.050        | 0.050     |
| larvae, split3       | 0.022         | 0.765     |
| protozoan, split1    | ~0.090        | 0.161     |
| protozoan, split2    | 0.133         | 0.197     |
| protozoan, split3    | ~0.100        | 0.246     |

A correlação positiva emb_norm → kappa é fraca e não monotônica globalmente, mas dentro de protozoan é mais consistente. O outlier larvae_split3 (kappa=0.765, emb_norm epoch10=0.022) sugere que a norm sozinha não prediz kappa: a estrutura do split (distribuição de classes) parece ter influência preponderante.

---

## Runs Excepcionais (melhor e pior)

### Runs com kappa > 0.40 (exceções positivas)

Apenas 3 runs excedem kappa = 0.40:

| Run | Dataset | Split | pct | Kappa | Acc | F1 |
|-----|---------|-------|-----|-------|-----|-----|
| `distillation_larvae_split3_pct100_1x1_BN2d_1280_one_layer` | larvae | 3 | 100% | **0.7650** | 0.868 | 0.882 |
| `distillation_protozoan_split2_pct50_1x1_BN2d_1280_one_layer` | protozoan | 2 | 50% | **0.4128** | 0.588 | 0.503 |
| `distillation_protozoan_split1_pct75_1x1_BN2d_1280_one_layer` | protozoan | 1 | 75% | **0.3895** | 0.582 | 0.470 |

**Análise do melhor run (`larvae_split3_pct100`, kappa=0.765):**
Este run é um outlier estatístico extremo dentro do grupo. A curva de treinamento mostra que o embedding do student colapsa normalmente nos primeiros epochs (norm = 0.631 → 0.025 em epoch 4), porém a train_loss cai mais rápido (0.631 → 0.304 em epoch 10) e a cosine_sim cresce consistentemente até 0.746 em epoch 99. O checkpoint foi salvo em epoch 99 (best-epoch=099), sugerindo que a melhoria continuou até o fim. A explicação mais provável é que split3 do dataset larvae tem uma composição interna que favorece o aprendizado de representações discriminativas neste regime de 100% dos dados — uma combinação de tamanho de conjunto de treinamento e separabilidade intrínseca das classes neste split específico.

**Pior run (`larvae_split1_pct100`, kappa=-0.213):**
Negativo (pior que aleatório). A run tem train_loss normal (~0.168 em epoch 99) e cosine_sim alta (~0.690), mas os embeddings produzidos são anti-discriminativos. Isso indica que a representação aprendida captura variação espúria (bias de split) em vez de features de classe.

### Top 5 melhores e piores runs por kappa

**Top 5 (kappa mais alto):**
1. `distillation_larvae_split3_pct100` — kappa = 0.7650
2. `distillation_protozoan_split2_pct50` — kappa = 0.4128
3. `distillation_protozoan_split1_pct75` — kappa = 0.3895
4. `distillation_eggs_split3_pct100` — kappa = 0.3495
5. `distillation_protozoan_split2_pct75` — kappa = 0.2924

**Bottom 5 (kappa mais baixo):**
1. `distillation_larvae_split1_pct100` — kappa = -0.2130
2. `distillation_eggs_split3_pct50` — kappa = -0.0244
3. `distillation_protozoan_split3_pct1` — kappa = -0.0383
4. `distillation_eggs_split2_pct5` — kappa = -0.0082
5. `distillation_protozoan_split1_pct1` — kappa = -0.0140

---

## Diagnóstico: Por que protozoan é mais resiliente?

Protozoan sistematicamente supera eggs e larvae em kappa médio (0.153 vs 0.081/0.079). Os dados apontam para os seguintes fatores:

**1. Menor student_emb_norm inicial → menor queda relativa**
O protozoan apresenta student_emb_norm inicial de ~0.42 (vs ~0.66 para eggs/larvae). Embora a queda seja proporcional, o valor de partida menor pode implicar que a BN2d cause menos dano relativo à estrutura das representações.

**2. Recuperação parcial da norm em runs de alta porcentagem**
O run `distillation_protozoan_split2_pct100` recupera a norm de 0.013 (epoch 2) para 0.276 (epoch 99). Runs equivalentes de eggs ficam em ~0.047. Isso sugere que o landscape de loss de protozoan permite um gradiente de recuperação mais forte para a norm do encoder.

**3. Train loss final mais baixa**
Protozoan atinge train_loss ~0.143 no epoch 99 (vs ~0.157 eggs, ~0.174 larvae), indicando melhor ajuste ao sinal de destilação. Isso correlaciona com a maior student_emb_norm e mais alta cosine_sim final (~0.772 vs ~0.758/0.746).

**4. Estrutura intrínseca do dataset**
Protozoan tem provavelmente maior separabilidade entre classes nas features de textura/FLIM capturadas pelo I-JEPA teacher. Mesmo com representações colapsadas, o SVM consegue encontrar margens de separação para protozoan com mais facilidade do que para eggs (que podem ter alta variabilidade intraclasse) ou larvae (que podem ter classes visualmente semelhantes).

**5. Protozoan split2_pct50 como caso especial**
O kappa=0.41 para protozoan_split2_pct50 (usando apenas 50% dos dados) sugere que este split específico tem uma divisão train/val que expõe melhor a estrutura das representações. O fato de que split1_pct50 (kappa=0.156) e split3_pct50 (kappa=0.154) sejam muito menores reforça que é um efeito de split, não de percentual.

---

## Recomendações

Com base na análise, as seguintes mudanças são recomendadas para melhorar o grupo `1x1_BN2d`:

**1. Remover ou substituir a BN2d na cabeça de projeção**
A BatchNorm2d é a causa primária do colapso de embeddings. Alternativas menos destrutivas:
- Substituir por LayerNorm ou GroupNorm, que preservam a estrutura de magnitude.
- Remover completamente a normalização na cabeça de projeção (a norma do teacher é estável em ~21, servindo como alvo implícito).
- Usar `conv1×1 + GELU` sem normalização, como em ablações de outros grupos.

**2. Aumentar o número de épocas de treinamento**
100% dos runs apresentam `best_epoch ≥ 85`, com a maioria em epoch 97–99. O modelo ainda estava aprendendo. Recomendam-se 150–200 épocas para verificar se a convergência ocorre ou se o colapso é terminal.

**3. Regularização da norma do encoder (norm regularization)**
Adicionar um termo de regularização que penalize normas muito pequenas do embedding do student poderia prevenir o colapso. Isso pode ser implementado como um termo de loss adicional: `λ · max(0, τ_min - ||z_s||)`.

**4. Investigar split3 de larvae**
O outlier `larvae_split3_pct100` (kappa=0.765) merece investigação. Se a composição deste split for reproduzível, pode-se entender qual propriedade específica da divisão favorece o aprendizado. Isso pode guiar a estratificação de splits para outros datasets.

**5. Considerar destilação híbrida (hybrid) para protozoan**
Dado que protozoan já mostra resiliência parcial ao colapso, uma configuração de destilação híbrida (que combina loss direta com loss de consistência) pode amplificar ainda mais a performance neste dataset, possivelmente superando kappa=0.5 sem alterações arquiteturais.

**6. Monitorar emb_norm como early stopping criterion**
Dado que o colapso ocorre nos primeiros 5–10 epochs, monitorar `student_emb_norm` como sinal de saúde do treinamento permite detectar runs problemáticos precocemente. Um threshold de `emb_norm < 0.05 em epoch 10` poderia acionar um early abort com reinicialização de hiperparâmetros.
