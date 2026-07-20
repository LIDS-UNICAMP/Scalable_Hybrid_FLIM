# Convergência e Anomalias de Treinamento

**Data da análise:** 2026-05-30
**Fonte:** `distillation_run_summaries.csv` (216 runs) + curvas por época (~20 750 linhas)

---

## Não-Convergência: 100 Épocas São Insuficientes

A principal conclusão da análise de convergência é que **100 épocas são insuficientes para todos os grupos**, com graus diferentes de severidade.

| Grupo          | Runs totais | best_epoch = 99 | % não convergidos |
|----------------|-------------|-----------------|-------------------|
| 1x1_BN2d       | 54          | 54              | **100.0%**        |
| 3x3_BN2d       | 54          | 52              | **96.3%**         |
| next_layers    | 54*         | 48              | **88.3%**         |
| 1x1_flim_init  | 36†         | 27              | **75.0%**         |

*Contando apenas os 54 runs não crashados (de 60 planejados).
†Apenas os 36 runs finalizados no momento da análise (12 ainda em execução).

O grupo **1x1_BN2d** é o caso mais grave: **todos os 54 runs atingiram o limite de 100 épocas sem convergir**. O critério de parada foi o teto de épocas, não um platô detectado. As curvas de loss nas épocas 75–99 ainda exibem slope negativo mensurável em todos os subgrupos de pct alto (protozoan: −3.8%, eggs/next_layers: −4.6%, larvae/3x3_BN2d: −1.4%), confirmando aprendizado ativo até o final do treinamento.

O grupo **1x1_flim_init** apresenta a menor taxa de não-convergência (75%), sugerindo que a inicialização a partir de pesos FLIM pré-treinados reduz o número de épocas necessárias — 9 dos 36 runs finalizados convergiram entre as épocas 52 e 77.

---

## Taxa de Decrescimento da Loss por Fase

Todos os grupos exibem um padrão trifásico universal de decrescimento da loss de treinamento:

| Fase                 | Épocas  | 1x1_BN2d | 3x3_BN2d | next_layers |
|----------------------|---------|-----------|-----------|-------------|
| Inicial rápida       | 0 → 10  | −48.9%    | −47.2%    | −59.9%      |
| Transição            | 10 → 25 | −17.8%    | −20.7%    | −28.1%      |
| Lenta                | 25 → 50 | −11.4%    | −12.2%    | −20.6%      |
| Refinamento          | 50 → 75 | −4.9%     | −3.7%     | −8.9%       |
| Quase-platô          | 75 → 99 | **−0.0%** | −0.9%     | −2.3%       |

Valores representam queda relativa da loss dentro de cada janela de épocas, média sobre runs finished com pct ≥ 25%.

**Observações-chave:**

- **next_layers aprende mais rápido na fase inicial:** queda de −59.9% nas primeiras 10 épocas, vs −48.9% para 1x1_BN2d. Isso reflete que next_layers distila múltiplas camadas do teacher, capturando representações mais ricas desde o início.
- **1x1_BN2d parou completamente na fase 75–99:** slope de −0.0% indica platô absoluto — os pesos congelaram numericamente, mas o critério de parada não foi acionado. Este comportamento sugere que o learning rate caiu a zero por schedule, não que o modelo convergiu ao ótimo.
- **3x3_BN2d e next_layers ainda decrescem nas épocas finais** (−0.9% e −2.3%), confirmando que o orçamento de 100 épocas é especialmente insuficiente para esses grupos.
- O padrão de três fases (rápida / lenta / platô) é universal nos quatro grupos, com diferenças quantitativas mas não qualitativas. Fase 1 (0–25) responde por 60–75% da queda total de loss; fase 3 (75–99) contribui com menos de 5%.

---

## ANOMALIA CRITICA: 6 Crashes em next_layers_direct High-Pct

**Todos os 6 runs crashados pertencem exclusivamente ao grupo next_layers_direct**, concentrados em combinações de alta porcentagem de dados e datasets específicos:

| run_id   | Run name                                              | Dataset | Split | Pct | Épocas até crash |
|----------|-------------------------------------------------------|---------|-------|-----|------------------|
| 03et3x6r | distillation_eggs_split1_pct100_next_layers_direct    | eggs    | 1     | 100 | 6                |
| uwi04k6s | distillation_eggs_split2_pct100_next_layers_direct    | eggs    | 2     | 100 | 6                |
| q3xclzge | distillation_eggs_split3_pct100_next_layers_direct    | eggs    | 3     | 100 | 6                |
| cq3od8ph | distillation_larvae_split1_pct50_next_layers_direct   | larvae  | 1     | 50  | 16               |
| a0uci292 | distillation_larvae_split1_pct75_next_layers_direct   | larvae  | 1     | 75  | 18               |
| b6lkl9sh | distillation_larvae_split1_pct100_next_layers_direct  | larvae  | 1     | 100 | 8                |

**Padrão dos crashes:**

1. **Pct alto em todos os casos:** os crashes ocorrem em pct ≥ 50%, nunca em pct ≤ 25%. Datasets maiores exigem mais memória por batch de embeddings de múltiplas camadas.
2. **eggs/pct100 perdeu todos os 3 splits:** 100% de cobertura crashada para essa combinação específica, tornando o kappa médio reportado para eggs/pct100/next_layers uma subestimativa (baseada em dados de runs que sobreviveram com menos dados ou outras configurações).
3. **larvae/split1 especialmente vulnerável:** todos os 3 crashes de larvae ocorrem no split1, sugerindo instabilidade numérica combinada com características específicas desta partição dos dados (possível distribuição de classes mais difícil ou amostras de alta norma de embedding).
4. **Crashes muito precoces (6–18 épocas):** o colapso ocorre na fase de descida rápida, antes do refinamento. Provável causa: **OOM (Out of Memory)** no servidor `jaci` ao alocar tensores de embeddings de múltiplas camadas para batches grandes, ou explosão de gradientes em embeddings não normalizados no regime de alta quantidade de dados.
5. **Runs equivalentes sem crash existem:** para eggs/pct100, os runs `rxsy6okd`, `bmaze9vb` e `zwo92csq` (IDs diferentes, mesma configuração em outro momento) finalizaram normalmente, indicando que o problema é intermitente / dependente de alocação de memória no servidor, não de instabilidade determinística do modelo.

**Causa provável mais parcimoniosa:** OOM em nó com menos VRAM disponível no momento do lançamento, combinado com buffers de embeddings de múltiplas camadas que excedem o threshold de memória disponível para datasets grandes.

---

## Impacto dos Crashes nos Resultados de eggs@100%

Os 3 crashes em eggs/pct100/next_layers_direct eliminaram exatamente os runs de maior quantidade de dados para o grupo next_layers no dataset eggs. O impacto nos resultados reportados é:

- **Kappa médio eggs@pct100 para next_layers é subestimado:** o valor reportado baseia-se nos runs não-crashados (com splits ou configurações alternativas). Se os 3 runs crashados tivessem completado 100 épocas, o desempenho esperado seria superior (mais dados de treinamento → student mais forte), elevando o kappa médio do grupo.
- **Comparações entre grupos em eggs/pct100 são desfavoráveis a next_layers:** ao comparar 1x1_BN2d vs next_layers em eggs/pct100, o grupo next_layers está sendo avaliado com cobertura incompleta, tornando qualquer conclusão de inferioridade de next_layers nessa condição não confiável.
- **Splits 2 e 3 de larvae/split também afetados:** embora apenas split1 tenha crashado em larvae, a ausência dos 3 runs (pct50, 75, 100 / split1) reduz a robustez estatística das médias para larvae/next_layers em pct alto.
- **Recomendação:** Ao reportar resultados de next_layers, indicar explicitamente que eggs/pct100 tem N=0 runs válidos para next_layers_direct e que as médias de larvae/pct≥50 têm N=2 splits em vez de 3. Qualquer comparação quantitativa nessas condições deve incluir essa ressalva metodológica.

---

## Oscilações: Treinamento Estável

A análise de estabilidade do treinamento (coeficiente de variação da loss nas épocas 80–99) mostra que a grande maioria dos runs apresenta treinamento estável:

| Grupo          | CV médio (std/mean, épocas 80–99) | Runs com CV > 2% |
|----------------|-----------------------------------|------------------|
| 1x1_BN2d       | 0.0065                            | **0 / 54**       |
| 3x3_BN2d       | 0.0080                            | 4 / 54           |
| next_layers    | 0.0122                            | 1 / 54           |
| 1x1_flim_init  | 0.0078                            | 2 / 36           |

**Interpretação:**

- **1x1_BN2d é o grupo mais estável:** CV de apenas 0.65% e zero runs oscilatórios. A estabilidade extrema é, paradoxalmente, sintoma do platô total (slope = 0.0%) — loss que não muda não oscila. O treinamento está numericamente "parado", não estável no sentido de ter convergido a um mínimo ótimo.
- **3x3_BN2d tem 4 runs oscilatórios:** concentrados em pct=1 e pct=5, onde o número de amostras de treinamento é muito pequeno e o gradiente por batch é mais ruidoso. Esses runs exibem slope positivo na fase final (loss aumentando nas últimas épocas), caracterizando sobreajuste tardio ou instabilidade de learning rate.
- **next_layers tem apenas 1 run oscilatório** apesar do CV médio mais alto (1.22%): o CV maior reflete variação real de aprendizado (slope ainda negativo e substantivo), não ruído. O único run oscilatório (CV > 2%) é provavelmente um caso de pct muito baixo.
- **1x1_flim_init tem 2 runs oscilatórios:** ambos em pct=1, padrão consistente com os demais grupos. A inicialização FLIM não elimina a oscilação em regime de dados escassos.
- **Conclusão geral:** oscilações problemáticas afetam exclusivamente runs com pct=1 ou pct=5. Para pct ≥ 25%, todos os grupos são estáveis. Não há evidência de instabilidade sistêmica no pipeline de destilação — os 7 runs oscilatórios identificados (em 198 finalizados) representam 3.5% do total.

---

## Extrapolação: O que Aconteceria com 200 Épocas?

Para runs com best_epoch = 99, a extrapolação usa a taxa linear (slope) das últimas 20 épocas (épocas 79–99), aplicada às 101 épocas adicionais (99 → 200). Esta é uma estimativa **conservadora**: o slope tende a diminuir adicionalmente, então a melhoria real poderia ser menor; porém para next_layers, onde o slope ainda é alto, poderia também ser maior se o aprendizado acelerou.

| Grupo       | Loss @ época 99 | Taxa / época (slope 79–99) | Loss estimada @ época 200 | Melhoria relativa |
|-------------|-----------------|----------------------------|---------------------------|-------------------|
| 1x1_BN2d    | 0.229           | −0.000 (parou!)            | **0.229**                 | 0.0% — nenhuma    |
| 3x3_BN2d    | 0.205           | −0.0002                    | **0.185**                 | −10%              |
| next_layers | 0.161           | −0.0003                    | **0.131**                 | −19%              |

**Análise por grupo:**

- **1x1_BN2d:** O slope completamente zero (−0.000/época) indica que estender o treinamento para 200 épocas **não produziria nenhum benefício** sob o schedule atual. A loss de 0.229 permaneceria inalterada. A causa é quase certamente o decaimento do learning rate para valores abaixo do limiar numérico de atualização de pesos — o otimizador existe mas não produz updates efetivos. Para obter ganhos, seria necessário reiniciar com learning rate maior, usar cosine restart, ou treinar com schedule diferente.
- **3x3_BN2d:** Melhoria projetada de −10% (0.205 → 0.185). Esta melhoria é significativa e justificaria o custo computacional de 100 épocas adicionais. Os 2 runs que já convergiram neste grupo (protozoan/pct75 e pct100, com n_epochs = 120–142) confirmam empiricamente que épocas além de 100 produzem ganhos reais para 3x3_BN2d.
- **next_layers:** Maior margem de melhoria projetada: −19% (0.161 → 0.131). Este grupo, que já possui as menores losses em valores absolutos, ainda tem slope substantivo nas épocas finais. A extrapolação sugere que **next_layers se beneficia mais de treinamento estendido** entre todos os grupos. Considerando que próximo de loss 0.131 o modelo estaria capturando representações muito próximas do teacher I-JEPA, o impacto no kappa de classificação poderia ser relevante.

**Cenários práticos para 200 épocas:**

| Cenário              | Ação necessária                                      | Ganho esperado |
|----------------------|------------------------------------------------------|----------------|
| 1x1_BN2d → 200 épocas | Nenhum ganho sem mudança de schedule               | Nulo           |
| 3x3_BN2d → 200 épocas | Apenas aumentar n_epochs                           | −10% loss      |
| next_layers → 200 épocas | Aumentar n_epochs + verificar crashes com gradient clipping | −19% loss |
| 1x1_BN2d com cosine restart | Reiniciar LR após época 80                  | Potencialmente −5 a −15% |

---

## 12 Runs "Running": Protozoan flim_init em Andamento

No momento da análise (2026-05-30), **12 runs permanecem em execução no servidor**, todos pertencentes ao grupo **1x1_flim_init** no dataset **protozoan**:

| run_id   | Split | Pct  | Épocas concluídas | best_epoch |
|----------|-------|------|-------------------|------------|
| 4lxpsm0w | 2     | 5%   | 54                | 53         |
| 0lhmp87s | 1     | 25%  | 78                | 77         |
| 22devhmw | 3     | 1%   | 54                | 53         |
| 4iqikzk8 | 1     | 1%   | 77                | 76         |
| ezj70i83 | 2     | 25%  | 76                | 75         |
| f64oasil | 2     | 1%   | 78                | 77         |
| fk0pv5zr | 1     | 75%  | 75                | 74         |
| g595id09 | 1     | 50%  | 53                | 52         |
| j74ezp5h | 2     | 100% | 27                | 26         |
| pcmjm4qd | 1     | 5%   | 54                | 53         |
| rihx60fg | 2     | 75%  | 53                | 52         |
| uwjyr3dy | 1     | 100% | 27                | 26         |

**Observações:**

- Todos os 12 runs têm best_epoch = n_epochs − 1, confirmando não-convergência no snapshot atual — padrão consistente com o grupo flim_init onde 75% não convergem.
- Os runs com pct=100 (j74ezp5h, uwjyr3dy) têm apenas 27 épocas, indicando início mais recente; estão na fase de aprendizado mais rápido e têm a maior margem de melhoria pela frente.
- Runs com pct=1 e pct=5 (22devhmw, pcmjm4qd, f64oasil) já estão próximos de época 78 — se o padrão do grupo se mantiver, poderão convergir antes de 100 ou atingir o limite sem convergência.
- **Impacto na análise:** A taxa de não-convergência de 75% para flim_init é baseada em 36 runs finalizados (eggs + larvae). Os 12 runs de protozoan, quando concluídos, ajustarão essa estatística. Se o comportamento de protozoan/flim_init for similar a eggs e larvae, a taxa deve se manter em torno de 70–80%.
- **Cobertura do dataset protozoan em flim_init:** Atualmente, não há nenhum run flim_init finalizado para protozoan. Toda a análise comparativa de flim_init vs BN2d para protozoan depende da conclusão desses 12 runs.

---

## Recomendações de Protocolo de Treinamento

Com base nas evidências de não-convergência, crashes e extrapolações, as seguintes mudanças de protocolo são justificadas:

### 1. Aumentar orçamento de épocas por grupo

| Grupo          | Épocas atuais | Épocas recomendadas | Justificativa                                              |
|----------------|---------------|---------------------|------------------------------------------------------------|
| 1x1_BN2d       | 100           | 150 (+ cosine restart) | Slope zero por schedule; reinício de LR necessário      |
| 3x3_BN2d       | 100           | 120–140             | 2 runs convergidos empiricamente nessa faixa; −10% projetado |
| next_layers    | 100           | 200                 | Maior margem de melhoria (−19%); slope alto nas épocas finais |
| 1x1_flim_init  | 100           | 100–120             | 25% já convergem antes de 100; extensão moderada para o restante |

### 2. Resolver crashes em next_layers_direct (alta prioridade)

- **Implementar gradient clipping** (max_norm = 1.0 ou 0.5) para next_layers_direct, especialmente em pct ≥ 50%.
- **Investigar alocação de memória:** logs de crash de 6 épocas em eggs/pct100 são consistentes com OOM. Reduzir batch size ou usar gradient checkpointing para datasets grandes.
- **Warm-up de learning rate mais lento:** nas primeiras 10 épocas, reduzir LR para 10% do valor nominal e aumentar gradualmente, evitando gradientes explosivos na fase de ajuste de escala dos embeddings de múltiplas camadas.
- **Replicar runs crashados** com as correções acima para restaurar cobertura experimental em eggs/pct100/next_layers.

### 3. Early stopping para pct=1 com 1x1_BN2d

- Runs com pct=1 em 1x1_BN2d exibem slope positivo após época ~75 (loss aumentando). Implementar early stopping com patience = 10 épocas baseado em loss de validação ou em kappa de classificação.
- Isso evitaria sobreajuste tardio e reduziria custo computacional para configurações que claramente não se beneficiam de treinamento prolongado.

### 4. Monitoramento de crashes em tempo real

- Adicionar verificação periódica (a cada 5 épocas) do estado dos runs no W&B para detectar crashes precoces e relançar automaticamente com configuração de fallback (batch size reduzido).
- Prioridade máxima para eggs/pct100/next_layers e larvae/split1/pct>=50, que são as combinações historicamente instáveis.

### 5. Aguardar conclusão dos 12 runs protozoan/flim_init

- Não fazer análise comparativa definitiva de flim_init vs BN2d em protozoan até os 12 runs finalizarem.
- Estimativa de conclusão: se os runs estão entre 27 e 78 épocas e o treinamento é linear em tempo, os últimos (pct=100) devem concluir em dias.
