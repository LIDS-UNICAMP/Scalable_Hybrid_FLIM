# 3x3_BN2d: Análise Completa

## Sumário Executivo

O grupo `3x3_BN2d_1280_one_layer` representa a configuração de referência do pipeline de destilação I-JEPA → FLIM CNN. Com 54 runs distribuídos por três datasets (eggs, larvae, protozoan) e múltiplos hiperparâmetros, este grupo é o único que combina um encoder com receptivo campo espacial real (kernel 3×3) e uma cabeça de projeção simples (uma camada com Batch Normalization 2D). O resultado é um perfil de treinamento saudável, livre de crashes, com cosine_sim final de 0.666 e kappa SVM de até 0.939 (larvae, 100% dos dados). Apesar de não atingir convergência total em 100 épocas (96,3% dos runs ainda em queda de loss), a configuração 3×3 domina o grupo 1×1 em todas as métricas mensuráveis e serve como âncora de comparação para os demais grupos.

---

## Por que o 3×3 funciona onde o 1×1 falha

A diferença fundamental entre os grupos `1x1_BN2d` e `3x3_BN2d` está no receptivo campo do encoder convolutivo. Um kernel 1×1 opera pixel a pixel, sem agregar contexto espacial vizinho. Em imagens de fluorescência de tempo de vida (FLIM), onde a informação relevante é distribuída localmente em manchas e estruturas de tamanho variável, a ausência de contexto espacial limita drasticamente a qualidade dos embeddings.

O kernel 3×3, por outro lado, agrega informação de uma vizinhança de 9 pixels a cada ativação, permitindo ao encoder capturar bordas, texturas locais e gradientes de intensidade — os sinais primários que distinguem ovos de larvas ou protozoários em imagens FLIM.

O efeito prático é dramático: no dataset `eggs`, o salto de kappa entre os dois grupos chega a 0.6+ pontos — enquanto o grupo 3×3 atinge kappa 0.900 com 100% dos dados de treino, o grupo 1×1 está em colapso de representação a ponto de não produzir embeddings discriminativos. As métricas confirmam: o `emb_norm` no grupo 1×1 (0.066 na época 99) está 5,4 vezes abaixo do grupo 3×3 (0.354), indicando que o encoder 1×1 produziu embeddings degenerados e quasi-nulos — um modo de colapso típico em arquiteturas sem capacidade representacional suficiente.

Em resumo: o kernel 3×3 não é um detalhe de arquitetura — é condição necessária para que o encoder FLIM CNN aprenda representações úteis a partir da destilação do I-JEPA.

---

## Curvas de Treinamento

### Evolução média (54 runs, todas as épocas monitoradas)

| Época | loss  | cosine_sim | emb_norm | scale_ratio |
|------:|------:|-----------:|---------:|------------:|
|     0 | 0.585 |      0.023 |    0.629 |       0.738 |
|    10 | 0.309 |      0.413 |    0.351 |       0.328 |
|    25 | 0.245 |      0.572 |    0.380 |       0.402 |
|    50 | 0.215 |      0.647 |    0.391 |       0.463 |
|    75 | 0.207 |      0.665 |    0.372 |       0.485 |
|    99 | 0.205 |      0.666 |    0.354 |       0.494 |

**Leitura das curvas:**

- **loss**: queda de 0.585 → 0.205 ao longo de 99 épocas (−65%). A maior parte da queda ocorre nas primeiras 25 épocas (0.585 → 0.245), com desaceleração progressiva a partir daí. A taxa de queda entre épocas 75 e 99 ainda é positiva (0.207 → 0.205), sinalizando que o treinamento não convergiu.
- **cosine_sim**: sobe de 0.023 (alinhamento quase nulo na inicialização) para 0.666 na época 99. O salto mais expressivo ocorre entre épocas 0 e 10 (+0.390 pontos), seguido de ganhos menores mas consistentes.
- **emb_norm**: apresenta comportamento não-monotônico. Parte em 0.629 (norma elevada por inicialização), cai abruptamente para 0.351 na época 10, sobe levemente até 0.391 (época 50), e depois decresce novamente para 0.354 (época 99). Este padrão reflete a regulação das normas pelo BN2d em conjunto com a adaptação da cabeça de projeção.
- **scale_ratio**: sobe monotonicamente de 0.328 (época 10) até 0.494 (época 99), indicando que o encoder está alinhando progressivamente a magnitude dos seus embeddings à do teacher. Um scale_ratio próximo a 1.0 seria ideal; 0.494 indica que o encoder ainda subestima a norma do teacher por um fator ~2×.

---

## Performance de Classificação (SVM Kappa)

### Kappa por dataset e fração de dados de treino (embeddings de 1280 dimensões)

| Dataset   |   1%  |   5%  |  25%  |  50%  |  75%  | 100%  |
|-----------|------:|------:|------:|------:|------:|------:|
| eggs      | 0.268 | 0.596 | 0.758 | 0.833 | 0.870 | 0.900 |
| larvae    | 0.635 | 0.702 | 0.853 | 0.880 | 0.913 | 0.939 |
| protozoan | 0.345 | 0.504 | 0.707 | 0.787 | 0.811 | 0.833 |

**Observações:**

- **larvae** apresenta a melhor performance absoluta (kappa 0.939 com 100% dos dados) e também o melhor desempenho com poucos dados (kappa 0.635 com apenas 1%). Isso sugere que os embeddings 3×3 capturam bem as características morfológicas das larvas.
- **eggs** tem a maior variância entre frações: kappa de 0.268 com 1% dos dados salta para 0.900 com 100%, uma amplitude de 0.632 pontos. O dataset de ovos é, portanto, mais sensível à quantidade de dados de classificação.
- **protozoan** fica no meio-termo, com kappa 0.833 no limite de dados completos, mas desempenho intermediário em regimes de poucos dados (kappa 0.345 com 1%).
- Em todos os datasets, a curva de kappa é côncava — os ganhos por fração adicional diminuem com o aumento da fração disponível, o que é comportamento esperado para embeddings com capacidade representacional razoável.

---

## Gap para o Teacher I-JEPA

### Comparação direta: SVM Kappa — Student 3x3_BN2d vs. Teacher I-JEPA

| Dataset   | Fração | Student | Teacher | Gap   |
|-----------|-------:|--------:|--------:|------:|
| eggs      |    1%  |  0.268  |  0.661  | −0.393 |
| eggs      |    5%  |  0.596  |  0.864  | −0.268 |
| eggs      |   25%  |  0.758  |  0.917  | −0.159 |
| eggs      |   50%  |  0.833  |  0.940  | −0.107 |
| eggs      |   75%  |  0.870  |  0.946  | −0.076 |
| eggs      |  100%  |  0.900  |  0.957  | −0.057 |
| larvae    |    1%  |  0.635  |  0.822  | −0.187 |
| larvae    |    5%  |  0.702  |  0.881  | −0.179 |
| larvae    |   25%  |  0.853  |  0.919  | −0.066 |
| larvae    |   50%  |  0.880  |  0.943  | −0.063 |
| larvae    |   75%  |  0.913  |  0.941  | −0.028 |
| larvae    |  100%  |  0.939  |  0.950  | −0.011 |
| protozoan |    1%  |  0.345  |  0.586  | −0.241 |
| protozoan |    5%  |  0.504  |  0.725  | −0.221 |
| protozoan |   25%  |  0.707  |  0.831  | −0.124 |
| protozoan |   50%  |  0.787  |  0.864  | −0.077 |
| protozoan |   75%  |  0.811  |  0.882  | −0.071 |
| protozoan |  100%  |  0.833  |  0.892  | −0.059 |

**Padrão geral:** o gap é consistentemente maior nos regimes de poucos dados (1%–5%) e diminui com mais dados de classificação. Em `larvae` com 100% dos dados, o gap é apenas −0.011 — praticamente paridade com o teacher. Em `eggs` com 1%, o gap é de −0.393, o maior do grupo.

**Interpretação:** os embeddings do student 3×3 são discriminativos o suficiente para suportar classificadores SVM eficientes quando há dados suficientes. A desvantagem relativa em low-shot se deve a representações menos refinadas — o teacher I-JEPA foi treinado com mais capacidade e em mais épocas sobre os dados originais, resultando em embeddings naturalmente mais separáveis.

O cosine_sim máximo atingido por dataset (0.784 para eggs, 0.774 para larvae, 0.803 para protozoan) indica que o espaço de embeddings do student ainda não espelha completamente o do teacher, deixando margem para melhora com mais épocas ou ajuste de hiperparâmetros.

---

## Embedding Norms: Estabilidade ao longo das Épocas

### student_emb_norm por dataset

| Dataset   |  ep0   |  ep10  |  ep25  |  ep50  |  ep99  |
|-----------|-------:|-------:|-------:|-------:|-------:|
| eggs      | 0.7269 | 0.3586 | 0.3755 | 0.3792 | 0.3450 |
| larvae    | 0.7381 | 0.3847 | 0.3633 | 0.3743 | 0.3237 |
| protozoan | 0.4589 | 0.2983 | 0.4002 | 0.4188 | 0.3920 |

**Observações:**

- **eggs e larvae** partem de normas similares (~0.73) e convergem para valores próximos (~0.34–0.35 na época 99). O padrão de queda e leve recuperação entre épocas 10 e 50 é comum nos dois datasets.
- **protozoan** parte com norma menor (0.4589 na época 0) e exibe uma dinâmica distinta: cai para 0.2983 na época 10 (o menor valor absoluto entre todos os datasets) antes de se recuperar para 0.4188 (época 50) e retornar a 0.3920 (época 99). Isso pode refletir diferenças na distribuição de intensidade das imagens FLIM de protozoários, que possuem características espectrais distintas.
- Nenhum dos três datasets mostra colapso de norma (valores tendendo a 0), o que é uma validação importante da estabilidade do encoder 3×3 com BN2d. Isso contrasta fortemente com o grupo 1×1, onde a norma cai para 0.066.
- A variação de norma ao longo das épocas (amplitude típica de ~0.15) é moderada e não indica instabilidade.

---

## Comparação Direta com 1x1_BN2d

### Métricas na época 99 (médias sobre todos os runs)

| Métrica     | 1x1_BN2d | 3x3_BN2d | Diferença          |
|-------------|:--------:|:--------:|:------------------:|
| emb_norm    |   0.066  |   0.354  | 3×3 é **5,4× maior** |
| cosine_sim  |   0.637  |   0.666  | 3×3 **+4,5%**      |
| train_loss  |   0.229  |   0.205  | 3×3 **−10,5%** menor |
| scale_ratio |   0.404  |   0.494  | 3×3 **+22%** melhor  |

### Cosine_sim máximo atingido por dataset

| Dataset   | 1x1_BN2d | 3x3_BN2d | next_layers |
|-----------|:--------:|:--------:|:-----------:|
| eggs      |  0.766   |  0.784   |    0.942    |
| larvae    |  0.747   |  0.774   |    0.919    |
| protozoan |  0.786   |  0.803   |    0.971    |

**Conclusão da comparação:**

O grupo 3×3 supera o 1×1 em todas as quatro métricas de treinamento. A diferença mais crítica é o `emb_norm`: o valor de 0.066 do grupo 1×1 é patológico — embeddings com norma tão baixa perdem poder discriminativo porque as distâncias no espaço de embedding colapsam. O SVM pode dificilmente separar classes quando os vetores são quasi-nulos.

O `cosine_sim` máximo do grupo 3×3 (0.784 em eggs) está consistentemente acima do 1×1 (0.766) em todos os datasets, confirmando que o receptor de campo 3×3 produz representações mais alinhadas com o teacher.

Vale notar que o grupo `next_layers`, com cosine_sim máximo de 0.942–0.971, supera o 3×3 em alinhamento bruto, mas ao custo de 6 crashes — enquanto o grupo 3×3 tem zero crashes.

---

## Convergência e Necessidade de Mais Épocas

**96,3% dos runs do grupo 3×3_BN2d não convergiram em 100 épocas.**

Os dados de treinamento confirmam esta afirmação: a loss ainda estava decrescendo (0.207 → 0.205) entre as épocas 75 e 99, e a cosine_sim não atingiu platô (0.665 → 0.666). O scale_ratio continua subindo (0.485 → 0.494), e o cosine_sim máximo observado (até 0.803 em protozoan) ainda está longe do máximo do grupo next_layers (~0.97).

**Estimativa de épocas necessárias:**

Com base na taxa de queda da loss entre épocas 75–99 (≈0.001 por época), e assumindo que convergência ocorre próxima a loss ~0.19–0.20, seriam necessárias aproximadamente 150–200 épocas totais. Para o cosine_sim atingir ~0.75, projetando a curva atual, também são necessárias ao menos 150 épocas.

**Implicações práticas:**

- Os resultados de kappa SVM reportados são sub-ótimos — treinamento mais longo melhorará os embeddings.
- O gap para o teacher I-JEPA em regimes low-shot (1%–5%) deve diminuir com mais épocas, à medida que os embeddings se tornem mais discriminativos.
- Configurações com os melhores hiperparâmetros devem ser re-treinadas com 200+ épocas antes de qualquer conclusão definitiva sobre o potencial do grupo 3×3.

---

## Anomalias

1. **Queda e recuperação de emb_norm em protozoan**: a norma cai para 0.2983 na época 10 (menor que eggs e larvae) antes de se recuperar. Embora não seja um sinal de instabilidade, é uma dinâmica distinta. Hipótese: as imagens FLIM de protozoários têm menor contraste local, o que faz o encoder precisar de mais épocas para calibrar as normas.

2. **Comportamento não-monotônico do emb_norm agregado**: o emb_norm médio sobe entre épocas 10 e 50 (0.351 → 0.391) e depois cai até 0.354 na época 99. Este padrão em sino sugere que o BN2d passa por um período de expansão de norma durante a fase de aprendizado mais intenso, seguido de uma compressão quando a loss se estabiliza. Não é anomalia — é comportamento esperado do BN2d em destilação, mas deve ser monitorado em runs mais longos.

3. **scale_ratio inicial alto (0.738 na época 0)**: o valor inicial elevado indica que, antes de qualquer gradiente ser aplicado, as normas do student e teacher têm escala similar. A queda para 0.328 na época 10 sugere que as primeiras atualizações de gradiente colapsam temporariamente a norma do student. A recuperação progressiva para 0.494 (época 99) é um sinal positivo de estabilização.

4. **Ausência total de crashes**: nenhum dos 54 runs do grupo 3×3_BN2d terminou em crash. Todos os 6 crashes observados no experimento são do grupo `next_layers`. Isso valida a robustez da combinação kernel 3×3 + BN2d + one_layer como backbone estável para destilação FLIM.

---

## Recomendações

1. **Estender o treinamento para 200–300 épocas**: com 96,3% dos runs sem convergência, os números atuais representam um limite inferior do potencial do grupo 3×3. Runs selecionados (melhores hiperparâmetros em cada dataset) devem ser re-treinados com mais épocas antes de comparações finais.

2. **Usar 3×3_BN2d como configuração base para ablações futuras**: este é o único grupo com encoder saudável (emb_norm ~0.35), zero crashes, e performance SVM competitiva. Qualquer nova variante arquitetural deve ser comparada contra este baseline.

3. **Investigar scale_ratio como métrica de parada**: o scale_ratio de 0.494 indica que o encoder ainda subestima a norma do teacher em ~2×. Monitorar este valor em runs mais longos; um platô próximo a 0.8–0.9 pode ser um critério de convergência mais informativo que a perda.

4. **Priorizar dataset eggs em análises de baixa amostragem**: eggs tem o maior gap para o teacher em regimes de 1%–5% (−0.393 e −0.268 respectivamente) e a maior variância de kappa entre frações. É o dataset mais sensível à qualidade dos embeddings e o que mais se beneficiará de mais épocas de treinamento.

5. **Considerar aumentar a capacidade da cabeça de projeção para larvae**: larvae já está em kappa 0.939 com 100% dos dados, muito próximo do teacher (0.950). Para este dataset, ganhos adicionais podem requerer aumento de capacidade da projeção — duas camadas ou dimensão de saída maior que 1280.

6. **Não migrar para next_layers sem análise de estabilidade**: embora o grupo `next_layers` tenha cosine_sim máximo ~0.94–0.97 (muito acima do 3×3), os 6 crashes e a menor estabilidade observada tornam esta configuração inadequada para produção sem antes identificar e corrigir as fontes de instabilidade.
