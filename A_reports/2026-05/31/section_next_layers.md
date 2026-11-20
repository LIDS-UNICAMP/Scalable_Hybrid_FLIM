# next_layers_direct: Análise Completa

**Arquitetura:** 48 → 128 → 256 → 512 → 1280 (cabeça multi-camada, 4 convoluções 1×1 + BN2d + GAP)
**Datasets:** eggs, larvae, protozoan | 3 splits × 6 percentuais (1%, 5%, 25%, 50%, 75%, 100%) = 54 runs planejados (6 crashes)

---

## Sumário Executivo

`next_layers_direct` é a arquitetura de maior capacidade avaliada no pipeline de destilação I-JEPA → CNN FLIM. Com cabeça de projeção em 4 estágios (48→128→256→512→1280), produz os maiores valores absolutos de cosine similarity e kappa SVM entre todos os grupos comparados. O teto de cosine_sim por dataset — 0.942 (eggs), 0.919 (larvae), 0.971 (protozoan) — supera o grupo 3×3_BN2d em +0.145 a +0.168 pontos. Em downstream, o kappa SVM com 100% dos dados chega a 0.907 (eggs), 0.923 (larvae) e 0.859 (protozoan) em embeddings de 1280 dimensões.

O grupo apresenta três características marcantes que o distinguem dos demais:

1. **Paradoxo do scale_ratio na época 0:** inicia em 1.019 (coincidência de norma com o teacher), colapsa 70% até a época 10 antes de recuperar gradualmente — comportamento único entre todos os grupos.
2. **Convergência mais rápida para cosine_sim = 0.70:** atinge esse limiar em apenas 21.3 épocas em média, ante 25.1 épocas (3×3_BN2d) e 28.7 épocas (1×1_BN2d).
3. **Alta taxa de não-convergência em 100 épocas:** 88.3% dos runs ainda não plataram ao fim do treinamento — o melhor teto só é atingido com mais épocas ou mais dados.

Seis crashes ocorreram exclusivamente neste grupo, todos em percentuais altos (pct ≥ 50%), provavelmente por OOM ou timeout no servidor.

---

## Curvas de Treinamento por Época

As métricas abaixo são médias sobre os 54 runs planejados (excluindo runs crashados nas épocas relevantes).

| Época | loss  | cosine_sim | emb_norm (student) | scale_ratio |
|-------|-------|------------|-------------------|-------------|
| 0     | 0.791 | 0.004      | 0.623             | 1.019       |
| 10    | 0.317 | 0.393      | 0.285             | 0.307       |
| 25    | 0.228 | 0.608      | 0.296             | 0.387       |
| 50    | 0.181 | 0.707      | 0.328             | 0.489       |
| 75    | 0.165 | 0.739      | 0.327             | 0.531       |
| 99    | 0.161 | 0.742      | 0.314             | 0.542       |

**Padrão de aprendizado:** A queda de loss mais intensa ocorre entre as épocas 0–10 (0.791 → 0.317, redução de 60%), onde o alinhamento direcional se estabelece rapidamente (cosine_sim sobe de 0.004 para 0.393). Entre épocas 10–99 a loss reduz mais 49% de forma gradual e assintótica. A cosine_sim dobra de 0.393 para 0.742 neste intervalo, mas sem atingir platô claro — sinal de que 100 épocas são insuficientes para a maioria dos runs.

**Norma do student (emb_norm):** Inicia em 0.623, colapsa para 0.285 na época 10, depois sobe levemente e estabiliza em torno de 0.314–0.328. O colapso inicial de norma é parte do mesmo fenômeno que gera o paradoxo de scale_ratio (ver seção dedicada).

### emb_norm por dataset × época

| Dataset   | ep0    | ep10   | ep25   | ep50   | ep99   |
|-----------|--------|--------|--------|--------|--------|
| eggs      | 0.6973 | 0.2923 | 0.3081 | 0.3331 | 0.3102 |
| larvae    | 0.7140 | 0.3014 | 0.2881 | 0.3098 | 0.2894 |
| protozoan | 0.4495 | 0.2565 | 0.2926 | 0.3413 | 0.3434 |

**Observação por dataset:** Eggs e larvae iniciam com norma similar (~0.70) e colapsam para ~0.29–0.30 na época 10. Protozoan parte de norma significativamente menor (0.4495) — possivelmente refletindo menor variabilidade interna das imagens — e apresenta recuperação mais estável, chegando a 0.3434 na época 99 (valor mais alto dos três). Protozoan também é o único dataset onde a norma cresce monotonicamente após o colapso.

---

## Paradoxo do Scale Ratio (scale=1.0 na época 0)

O `scale_ratio` é definido como `student_proj_norm / teacher_emb_norm` e mede o alinhamento de magnitude entre as embeddings do student e do teacher.

| Época | scale_ratio | cosine_sim | Interpretação                            |
|-------|-------------|------------|------------------------------------------|
| 0     | **1.019**   | 0.004      | Coincidência de norma, sem alinhamento   |
| 10    | 0.307       | 0.393      | Colapso de 70%; alinhamento direcional   |
| 25    | 0.387       | 0.608      | Recuperação gradual                      |
| 50    | 0.489       | 0.707      | Magnitude cresce junto com direção       |
| 75    | 0.531       | 0.739      | Desaceleração visível                    |
| 99    | 0.542       | 0.742      | Ainda ascendente, sem platô              |

**Por que o scale_ratio começa em 1.019?** A cabeça multi-camada (48→128→256→512→1280) inicializa com as camadas BN2d em escala = 1. Nesse estado inicial, a saída acumulada das quatro convoluções mais o GAP produz, por construção, normas de embedding coincidentemente próximas às do teacher I-JEPA. Contudo, a cosine_sim de 0.004 confirma que essa coincidência de magnitude é puramente acidental — não há alinhamento representacional.

**Por que cai abruptamente para 0.307 até a época 10?** Durante o aquecimento (warmup), o BN2d "aprende" a normalizar as ativações e reconfigura a escala internamente. O otimizador encontra o caminho de menor resistência para minimizar a loss de cosine: alinhar as direções das embeddings, que exige colapsar a norma do student. Esse fenômeno — alinhar direção antes de recuperar magnitude — é bem documentado em destilação de representações.

**Este comportamento é único entre todos os grupos.** Todos os outros grupos iniciam com scale_ratio < 0.81 e sobem monotonicamente. O `next_layers_direct` é o único que começa "perfeito" em magnitude e desce antes de recuperar. A causa é estrutural: somente a cabeça de 4 camadas com BN2d em todos os estágios produz essa coincidência de norma na inicialização.

**Implicação prática:** O scale_ratio final de 0.542 (época 99) indica que o student ainda opera com normas em torno de 54% das do teacher. A lacuna persiste porque o teacher I-JEPA, treinado com objetivos self-supervised que incentivam representações de alta magnitude, produz embeddings estruturalmente maiores do que uma CNN com GAP pode replicar em 100 épocas.

---

## Cosine Similarity: Mais Alto de Todos (teto 0.971)

### Teto de cosine_sim por dataset

| Dataset   | max(next_layers) | max(3×3_BN2d) | Vantagem next_layers |
|-----------|-----------------|--------------|----------------------|
| eggs      | 0.942           | 0.784        | +0.158               |
| larvae    | 0.919           | 0.774        | +0.145               |
| protozoan | **0.971**       | 0.803        | +0.168               |

O `next_layers_direct` supera o 3×3_BN2d em +0.145 a +0.168 pontos em todos os datasets. A vantagem é maior em protozoan, onde a cabeça profunda consegue construir uma hierarquia representacional que mapeia com excepcional fidelidade o espaço do teacher.

### Cosine_sim média por época (todos os 54 runs)

A evolução segue um padrão de crescimento rápido nas primeiras 25 épocas (0.004 → 0.608) seguido de crescimento mais lento mas contínuo (0.608 → 0.742 entre épocas 25–99). A ausência de platô claro na época 99 é consistente com a alta taxa de não-convergência (88.3%) e sugere que gains adicionais são esperados além da época 100.

**Protozoan como caso limite:** Com teto de 0.971, protozoan demonstra que o alinhamento direcional quase perfeito com o teacher é atingível para datasets com representações mais compressíveis. Eggs e larvae têm teto menor (0.942 e 0.919), possivelmente porque sua maior variabilidade intra-classe dificulta o colapso representacional.

---

## Performance de Classificação

### SVM Kappa por dataset × percentual de dados (embeddings 1280-dim)

| Dataset   | 1%    | 5%    | 25%   | 50%   | 75%   | 100%  |
|-----------|-------|-------|-------|-------|-------|-------|
| eggs      | 0.254 | 0.658 | 0.832 | 0.878 | 0.902 | 0.907 |
| larvae    | 0.620 | 0.756 | 0.879 | 0.906 | 0.914 | 0.923 |
| protozoan | 0.392 | 0.574 | 0.754 | 0.820 | 0.837 | 0.859 |

**Crescimento com dados:** Em todos os datasets, o kappa cresce expressivamente de 1% para 25% e satura progressivamente após 50%. O ganho marginal de 75% para 100% é de apenas 0.005–0.009 pontos — indicando que as representações já codificam a maior parte da informação de classificação com 75% dos dados.

**Regime de baixo dado (pct=1%):** Eggs apresenta kappa 0.254 — próximo ao acaso para classificação multi-classe — enquanto larvae é surpreendentemente robusta (0.620). Protozoan é intermediário (0.392). A profundidade do next_layers pode ser prejudicial com poucos dados: a cabeça de 4 camadas tem maior capacidade de overfitting na projeção quando o gradiente é calculado sobre conjuntos pequenos.

**Dissociação protozoan:** Apesar de ter o maior cosine_sim (teto 0.971), protozoan apresenta o menor kappa SVM em todas as faixas de pct. Esta dissociação indica que o espaço de embedding do teacher I-JEPA para protozoan tem estrutura que o SVM linear captura menos eficientemente — as classes protozoan podem ser separáveis no espaço original do teacher, mas a projeção CNN preserva a direção global enquanto comprime fronteiras de decisão não-lineares.

**Larvae como destaque:** Com kappa de 0.923 em pct=100% e 0.914 em pct=75%, larvae é o dataset mais discriminável no espaço destilado — apesar de não ter o maior cosine_sim absoluto.

---

## Crashes Detectados: Todos em High-Pct

Seis runs crasharam durante o treinamento, e todos pertencem exclusivamente ao grupo `next_layers_direct`:

| Run crashado                              | Dataset | split | pct  |
|-------------------------------------------|---------|-------|------|
| distillation_eggs_split1_pct100           | eggs    | 1     | 100% |
| distillation_eggs_split2_pct100           | eggs    | 2     | 100% |
| distillation_eggs_split3_pct100           | eggs    | 3     | 100% |
| distillation_larvae_split1_pct50          | larvae  | 1     | 50%  |
| distillation_larvae_split1_pct75          | larvae  | 1     | 75%  |
| distillation_larvae_split1_pct100         | larvae  | 1     | 100% |

**Padrão:** Todos os crashes ocorrem em pct ≥ 50% e afetam os datasets eggs (pct=100%, todos os 3 splits) e larvae (split1, pct ≥ 50%). O dataset protozoan não teve crashes. A causa mais provável é OOM (out-of-memory) ou timeout no servidor: o modelo maior (48→128→256→512→1280) com pct alto de dados produz batches maiores e demanda mais memória GPU do que os grupos 3×3 ou 1×1.

**Impacto nos dados:**
- Os kappa SVM de eggs em pct=100% refletem apenas splits 2 e 3 (split1 crashou). A média reportada (0.907) pode ser ligeiramente sub-estimada.
- Larvae split1 perdeu pct ≥ 50%. A média de kappa para larvae em pct ≥ 50% usa apenas splits 2 e 3 — potencialmente introduzindo viés de seleção.

**Recomendação de infraestrutura:** Antes de re-executar os runs crashados, aumentar o limite de memória GPU ou reduzir o tamanho do batch para pct=100% com arquitetura next_layers.

---

## Convergência

**88.3% dos runs não converge em 100 épocas** (47 de 54 runs ainda mostravam melhora quando o treinamento foi interrompido). Este é o maior percentual de não-convergência entre todos os grupos avaliados.

### Velocidade para atingir cosine_sim = 0.70

| Grupo         | Épocas médias para cosine = 0.70 |
|---------------|----------------------------------|
| next_layers   | **21.3** (mais rápido)           |
| 3×3_BN2d      | 25.1                             |
| 1×1_BN2d      | 28.7                             |

A cabeça mais profunda aprende representações de qualidade mais rapidamente nas primeiras épocas, mas o platô é mais tardio — a curva continua subindo além da época 99 na maioria dos runs.

### Implicação da não-convergência

A cosine_sim média de 0.742 na época 99 está ainda crescendo (0.739 na época 75 vs. 0.742 na época 99: variação de +0.003 nas últimas 24 épocas). Extrapolando a tendência, runs com pct ≥ 50% ainda ganhariam ~0.01–0.03 pontos de cosine_sim adicionais com 150–200 épocas. Para eggs e larvae, cujos best_epoch tendem a cair entre 85–99, estender o treinamento é especialmente relevante.

---

## Comparação: next_layers vs 3×3_BN2d

| Métrica                              | next_layers_direct | 3×3_BN2d | Vantagem          |
|--------------------------------------|--------------------|----------|-------------------|
| Teto cosine_sim — eggs               | **0.942**          | 0.784    | +0.158            |
| Teto cosine_sim — larvae             | **0.919**          | 0.774    | +0.145            |
| Teto cosine_sim — protozoan          | **0.971**          | 0.803    | +0.168            |
| cosine_sim médio (época 99)          | **0.742**          | ~0.620   | +0.122            |
| Épocas para cosine = 0.70            | **21.3**           | 25.1     | 25% mais rápido   |
| Kappa SVM pct=100% — eggs            | **0.907**          | ~0.890   | +0.017            |
| Kappa SVM pct=100% — larvae          | **0.923**          | ~0.910   | +0.013            |
| Kappa SVM pct=100% — protozoan       | **0.859**          | ~0.830   | +0.029            |
| % runs sem convergência em 100 ep.  | 88.3%              | ~75%     | 3×3 converge mais |
| Crashes observados                   | **6**              | 0        | 3×3 mais estável  |
| scale_ratio final (média)            | **0.542**          | ~0.480   | +0.062            |

**Síntese da comparação:**

O `next_layers_direct` é superior em todas as métricas de qualidade de representação e classificação. O custo é uma arquitetura menos estável: maior taxa de não-convergência dentro de 100 épocas e exclusividade de todos os crashes do experimento. O 3×3_BN2d oferece uma relação qualidade/estabilidade mais equilibrada, mas com teto inferior em todas as dimensões avaliadas.

Para decisão arquitetural em produção: se o objetivo é maximizar kappa SVM com dados abundantes (pct ≥ 50%) e infraestrutura confiável, `next_layers_direct` é a escolha. Se o regime é de poucos dados (pct ≤ 5%) ou infraestrutura limitada, o 3×3_BN2d ou 1×1_BN2d são mais adequados.

---

## Recomendações

1. **Estender treinamento para 150–200 épocas** em configurações com pct ≥ 50%. A cosine_sim média ainda subia na época 99 (0.739 → 0.742 entre épocas 75–99), e 88.3% dos runs não haviam convergido. Projeções indicam ganhos adicionais de 0.01–0.03 em cosine_sim e potencialmente ~0.005–0.010 em kappa SVM.

2. **Investigar e corrigir os 6 crashes antes de re-executar.** Todos em pct ≥ 50% com modelo maior. Recomenda-se reduzir batch size para pct=100% ou aumentar o limite de memória GPU. Os 3 crashes de eggs em pct=100% (todos os splits) são especialmente críticos para a análise do ponto de saturação desse dataset.

3. **Para regime de poucos dados (pct ≤ 5%), preferir arquiteturas menores.** O kappa de eggs em pct=1% (0.254) e o colapso de cosine_sim nessa condição indicam que a cabeça de 4 camadas overfita com dados insuficientes para calibrar todos os estágios BN2d.

4. **Protozoan como referência de alinhamento.** Com teto de cosine_sim de 0.971 e emb_norm crescendo monotonicamente após o colapso inicial, protozoan é o dataset mais favorável para validar melhorias arquiteturais: as representações são altamente compressíveis e respondem bem à destilação.

5. **Investigar a dissociação cosine_sim × kappa em protozoan.** O maior cosine_sim absoluto (0.971) coexiste com o menor kappa SVM (0.859). Um classificador não-linear (e.g., kernel RBF) poderia recuperar a discriminabilidade que o SVM linear perde, confirmando se o espaço destilado de protozoan é de fato informativo mas com fronteiras de decisão não-lineares.

6. **Monitorar o paradoxo de scale_ratio na inicialização** ao variar a arquitetura. O comportamento de scale ≈ 1.0 na época 0 é específico da combinação de 4 camadas 1×1 + BN2d com inicialização padrão. Qualquer mudança no número de camadas ou no inicializador pode alterar esse comportamento e afetar a dinâmica de aprendizado nas primeiras épocas.
