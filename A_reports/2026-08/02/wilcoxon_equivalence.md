# Testes par-a-par e de equivalencia (TOST) entre os 8 modelos oficiais

Este relatorio existe por duas razoes. Primeira: os testes de `wilcoxon_f1.md`, `wilcoxon_acc.md` e `wilcoxon_kappa.md` sao um-contra-todos com o FLIM como baseline, logo cobrem 7 dos 28 pares possiveis — **Distill 3 vs Distill 4**, entre outros, ficava sem teste. Segunda, e mais importante: **ausencia de significancia nao e empate**.

Um Wilcoxon que nao rejeita diz apenas "nao detectei diferenca", o que e compativel tanto com "os modelos sao iguais" quanto com "o teste nao tem poder". Para escrever *empate* no artigo e preciso um **teste de equivalencia**, que inverte o onus da prova: a hipotese nula passa a ser "a diferenca e grande", e so se rejeita essa nula diante de evidencia positiva de que a diferenca e pequena. E o que o TOST faz aqui.

## Desenho

Unidade pareada: celula `(dataset, % de rotulos)`, 3 datasets x 6 fracoes = **n = 18 pares**. Cada celula ja e a media sobre 3 splits. Pareamento pela chave, verificado antes de cada teste.

**Convencao de sinal:** `diff = metrica(modelo B) - metrica(modelo A)`. Positivo favorece o **modelo B**.

Cada par recebe **dois** testes, que respondem a perguntas diferentes:

| Teste | Pergunta | H0 | Rejeitar significa |
|---|---|---|---|
| **Superioridade** (Wilcoxon bilateral) | Existe diferenca? | mediana = 0 | ha diferenca |
| **Equivalencia** (TOST) | A diferenca e pequena? | \|mediana\| >= 0.05 | a diferenca cabe em +/-0.05 |

Combinando os dois, cada par cai em uma de quatro caixas:

| | TOST rejeita (dif. pequena) | TOST nao rejeita |
|---|---|---|
| **Superioridade rejeita** | diferenca real porem menor que a margem | diferenca real e nao-desprezivel |
| **Superioridade nao rejeita** | **EQUIVALENTES** (empate demonstrado) | inconclusivo (sem poder) |

Apenas a caixa **EQUIVALENTES** autoriza a palavra "empate" no texto. A caixa "inconclusivo" e onde caem hoje varias afirmacoes do artigo.

## Margem de equivalencia

`delta = 0.05` (5 pontos da metrica), **pre-especificada** — escolhida antes de ver os resultados, nao ajustada depois. Justificativa dupla:

1. E o arredondamento convencional para "diferenca praticamente irrelevante" nesta escala de metrica.
2. Fica em torno de **2x o ruido de medicao do proprio experimento**: o desvio-padrao tipico entre os 3 splits da mesma celula e (f1: 0.0179, kappa: 0.0248, acc: 0.0182); no percentil 90 chega a (f1: 0.0845, kappa: 0.0921, acc: 0.1011). Uma diferenca menor que 0.05 esta na ordem de grandeza do ruido entre splits.

Para nao obrigar o leitor a aceitar essa escolha, toda linha traz tambem **`delta_min`**: a menor margem para a qual a equivalencia se sustenta a alfa = 0.05. Leitura direta: *"estes dois modelos sao equivalentes dentro de +/- delta_min"*. Quanto menor, mais forte o empate. `n/a` significa que a equivalencia nao se sustenta em margem nenhuma.

## 1. Familia focal (tabela do artigo)

Os pares que sustentam afirmacoes explicitas de empate ou de ganho desprezivel no texto. Familia pre-especificada de 5 pares, correcao de Holm por metrica, alfa = 0.05.

### F1

| A | B | Mediana B-A | IC95% | B vence | p superior. (Holm) | p TOST (Holm) | delta_min | Veredito |
|---|---|---:|---|---:|---:|---:|---:|---|
| FLIM (59.504) | I-JEPA (632M) | +0.0185 | [-0.010, +0.038] | 13/18 | 0.3251 | 0.4557 | 0.1925 | **inconclusivo** |
| Distill 3 (615K) | Distill 4 (889K) | +0.0267 | [+0.012, +0.038] | 16/18 | 0.0010 | 0.0039 | 0.0366 | **diferenca real porem menor que a margem** |
| Distill 1 - FLIM init (123K) | Distill 2 (402K) | +0.0330 | [-0.019, +0.081] | 10/18 | 0.3251 | 0.4557 | 0.0676 | **inconclusivo** |
| Distill 2 (402K) | Distill 3 (615K) | +0.0404 | [+0.019, +0.080] | 16/18 | 0.0017 | 0.5339 | 0.0670 | **diferenca real e nao-desprezivel** |
| FLIM (59.504) | Distill 4 (889K) | -0.0225 | [-0.063, -0.005] | 3/18 | 0.3251 | 0.2595 | 0.0987 | **inconclusivo** |

### kappa

| A | B | Mediana B-A | IC95% | B vence | p superior. (Holm) | p TOST (Holm) | delta_min | Veredito |
|---|---|---:|---|---:|---:|---:|---:|---|
| FLIM (59.504) | I-JEPA (632M) | +0.1039 | [+0.076, +0.200] | 18/18 | 3.81e-05 | 1.0000 | 0.2613 | **diferenca real e nao-desprezivel** |
| Distill 3 (615K) | Distill 4 (889K) | +0.0257 | [+0.012, +0.044] | 15/18 | 0.0031 | 0.0140 | 0.0400 | **diferenca real porem menor que a margem** |
| Distill 1 - FLIM init (123K) | Distill 2 (402K) | -0.0132 | [-0.047, +0.053] | 8/18 | 1.0000 | 0.1975 | 0.0462 | **inconclusivo** |
| Distill 2 (402K) | Distill 3 (615K) | +0.0665 | [+0.037, +0.103] | 16/18 | 0.0013 | 1.0000 | 0.0894 | **diferenca real e nao-desprezivel** |
| FLIM (59.504) | Distill 4 (889K) | +0.0461 | [+0.006, +0.091] | 13/18 | 0.0095 | 1.0000 | 0.1088 | **diferenca real e nao-desprezivel** |

### Acc

| A | B | Mediana B-A | IC95% | B vence | p superior. (Holm) | p TOST (Holm) | delta_min | Veredito |
|---|---|---:|---|---:|---:|---:|---:|---|
| FLIM (59.504) | I-JEPA (632M) | +0.0175 | [-0.006, +0.044] | 13/18 | 0.2168 | 0.3247 | 0.1699 | **inconclusivo** |
| Distill 3 (615K) | Distill 4 (889K) | +0.0281 | [+0.016, +0.039] | 17/18 | 0.0002 | 0.0010 | 0.0369 | **diferenca real porem menor que a margem** |
| Distill 1 - FLIM init (123K) | Distill 2 (402K) | +0.0979 | [+0.019, +0.139] | 13/18 | 0.0058 | 0.9093 | 0.1197 | **diferenca real e nao-desprezivel** |
| Distill 2 (402K) | Distill 3 (615K) | +0.0369 | [+0.012, +0.062] | 15/18 | 0.0052 | 0.2830 | 0.0533 | **diferenca real e nao-desprezivel** |
| FLIM (59.504) | Distill 4 (889K) | -0.0087 | [-0.036, -0.003] | 4/18 | 0.2168 | 0.2830 | 0.1024 | **inconclusivo** |

### Afirmacao que cada par sustenta

| A | B | Afirmacao no texto | F1 | kappa | Acc |
|---|---|---|---|---|---|
| FLIM (59.504) | I-JEPA (632M) | "FLIM praticamente empatado com o teacher" (F1 medio 0,938 vs 0,940 @100%) | inconclusivo | diferenca real e nao-desprezivel | inconclusivo |
| Distill 3 (615K) | Distill 4 (889K) | "+45% de params rendem +0,007 de F1 e +0,002 de kappa" (retornos decrescentes) | diferenca real porem menor que a margem | diferenca real porem menor que a margem | diferenca real porem menor que a margem |
| Distill 1 - FLIM init (123K) | Distill 2 (402K) | "Distill 1 FLIM init tem kappa 0,82 contra 0,83 do Distill 2, com 1/3 dos params" | inconclusivo | inconclusivo | diferenca real e nao-desprezivel |
| Distill 2 (402K) | Distill 3 (615K) | curva de retornos decrescentes das proj heads (Distill 2 -> 3) | diferenca real e nao-desprezivel | diferenca real e nao-desprezivel | diferenca real e nao-desprezivel |
| FLIM (59.504) | Distill 4 (889K) | fronteira de Pareto: FLIM puro (60K) contra o melhor destilado (889K) | inconclusivo | diferenca real e nao-desprezivel | inconclusivo |

## 1b. Decomposicao por fracao de rotulos (descritiva)

A mediana agregada sobre as 18 celulas esconde a estrutura mais importante do experimento: em varios pares, quase toda a diferenca vem do regime de rotulo escasso. As tabelas abaixo mostram a diferenca media (B - A) em cada fracao, e `B vence` conta em quantos dos 3 datasets o modelo B ficou a frente naquela fracao.

### F1

| Par (A vs B) | 1% | 5% | 25% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| FLIM (59.504) vs I-JEPA (632M) | +0.516 (3/3) | +0.012 (2/3) | -0.001 (2/3) | +0.006 (2/3) | +0.000 (2/3) | +0.003 (2/3) |
| Distill 3 (615K) vs Distill 4 (889K) | +0.007 (2/3) | +0.057 (3/3) | +0.037 (3/3) | +0.033 (3/3) | +0.023 (3/3) | +0.007 (2/3) |
| Distill 1 - FLIM init (123K) vs Distill 2 (402K) | +0.127 (3/3) | +0.084 (2/3) | -0.029 (1/3) | -0.021 (0/3) | +0.018 (2/3) | +0.034 (2/3) |
| Distill 2 (402K) vs Distill 3 (615K) | -0.008 (2/3) | +0.027 (2/3) | +0.093 (3/3) | +0.080 (3/3) | +0.053 (3/3) | +0.040 (3/3) |
| FLIM (59.504) vs Distill 4 (889K) | +0.354 (3/3) | -0.092 (0/3) | -0.059 (0/3) | -0.034 (0/3) | -0.029 (0/3) | -0.031 (0/3) |

### kappa

| Par (A vs B) | 1% | 5% | 25% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| FLIM (59.504) vs I-JEPA (632M) | +0.589 (3/3) | +0.230 (3/3) | +0.094 (3/3) | +0.089 (3/3) | +0.075 (3/3) | +0.066 (3/3) |
| Distill 3 (615K) vs Distill 4 (889K) | +0.007 (1/3) | +0.063 (3/3) | +0.042 (3/3) | +0.032 (3/3) | +0.019 (3/3) | +0.002 (2/3) |
| Distill 1 - FLIM init (123K) vs Distill 2 (402K) | +0.172 (3/3) | +0.033 (2/3) | -0.090 (0/3) | -0.055 (0/3) | -0.008 (1/3) | +0.014 (2/3) |
| Distill 2 (402K) vs Distill 3 (615K) | -0.010 (2/3) | +0.044 (2/3) | +0.123 (3/3) | +0.103 (3/3) | +0.070 (3/3) | +0.059 (3/3) |
| FLIM (59.504) vs Distill 4 (889K) | +0.322 (3/3) | +0.071 (2/3) | +0.019 (2/3) | +0.039 (2/3) | +0.036 (2/3) | +0.026 (2/3) |

### Acc

| Par (A vs B) | 1% | 5% | 25% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| FLIM (59.504) vs I-JEPA (632M) | +0.469 (3/3) | +0.014 (2/3) | +0.008 (2/3) | +0.006 (2/3) | +0.002 (2/3) | +0.005 (2/3) |
| Distill 3 (615K) vs Distill 4 (889K) | +0.022 (3/3) | +0.059 (3/3) | +0.031 (3/3) | +0.030 (3/3) | +0.018 (3/3) | +0.010 (2/3) |
| Distill 1 - FLIM init (123K) vs Distill 2 (402K) | +0.177 (3/3) | +0.116 (2/3) | +0.020 (2/3) | +0.033 (2/3) | +0.063 (2/3) | +0.078 (2/3) |
| Distill 2 (402K) vs Distill 3 (615K) | -0.016 (1/3) | +0.019 (2/3) | +0.082 (3/3) | +0.061 (3/3) | +0.043 (3/3) | +0.028 (3/3) |
| FLIM (59.504) vs Distill 4 (889K) | +0.338 (3/3) | -0.069 (0/3) | -0.043 (0/3) | -0.024 (1/3) | -0.022 (0/3) | -0.022 (0/3) |

## 1c. Familia focal descartando a fracao de 1% (n = 15)

Repeticao integral da familia focal sobre as 5 fracoes de 5% a 100%. Serve para responder a pergunta "a diferenca sobrevive fora do regime de colapso?". Onde um veredito muda entre esta tabela e a de §1, a diferenca agregada era sustentada principalmente pela fracao de 1%.

### F1

| A | B | Mediana B-A | B vence | p superior. (Holm) | delta_min | Veredito (n=15) | Veredito (n=18) |
|---|---|---:|---:|---:|---:|---|---|
| FLIM (59.504) | I-JEPA (632M) | +0.0071 | 10/15 | 1.0000 | 0.0243 | **EQUIVALENTES** ⚠ | inconclusivo |
| Distill 3 (615K) | Distill 4 (889K) | +0.0319 | 14/15 | 0.0009 | 0.0422 | **diferenca real porem menor que a margem** | diferenca real porem menor que a margem |
| Distill 1 - FLIM init (123K) | Distill 2 (402K) | -0.0038 | 7/15 | 1.0000 | 0.0479 | **inconclusivo** | inconclusivo |
| Distill 2 (402K) | Distill 3 (615K) | +0.0674 | 14/15 | 0.0005 | 0.0773 | **diferenca real e nao-desprezivel** | diferenca real e nao-desprezivel |
| FLIM (59.504) | Distill 4 (889K) | -0.0295 | 0/15 | 0.0003 | 0.0658 | **diferenca real e nao-desprezivel** ⚠ | inconclusivo |

### kappa

| A | B | Mediana B-A | B vence | p superior. (Holm) | delta_min | Veredito (n=15) | Veredito (n=18) |
|---|---|---:|---:|---:|---:|---|---|
| FLIM (59.504) | I-JEPA (632M) | +0.0974 | 15/15 | 0.0003 | 0.1180 | **diferenca real e nao-desprezivel** | diferenca real e nao-desprezivel |
| Distill 3 (615K) | Distill 4 (889K) | +0.0326 | 14/15 | 0.0013 | 0.0424 | **diferenca real e nao-desprezivel** ⚠ | diferenca real porem menor que a margem |
| Distill 1 - FLIM init (123K) | Distill 2 (402K) | -0.0288 | 5/15 | 0.2293 | 0.0545 | **inconclusivo** | inconclusivo |
| Distill 2 (402K) | Distill 3 (615K) | +0.0808 | 14/15 | 0.0005 | 0.0999 | **diferenca real e nao-desprezivel** | diferenca real e nao-desprezivel |
| FLIM (59.504) | Distill 4 (889K) | +0.0325 | 10/15 | 0.0707 | 0.0620 | **inconclusivo** ⚠ | diferenca real e nao-desprezivel |

### Acc

| A | B | Mediana B-A | B vence | p superior. (Holm) | delta_min | Veredito (n=15) | Veredito (n=18) |
|---|---|---:|---:|---:|---:|---|---|
| FLIM (59.504) | I-JEPA (632M) | +0.0091 | 10/15 | 0.5245 | 0.0267 | **EQUIVALENTES** ⚠ | inconclusivo |
| Distill 3 (615K) | Distill 4 (889K) | +0.0292 | 14/15 | 0.0009 | 0.0395 | **diferenca real porem menor que a margem** | diferenca real porem menor que a margem |
| Distill 1 - FLIM init (123K) | Distill 2 (402K) | +0.0669 | 10/15 | 0.0302 | 0.1007 | **diferenca real e nao-desprezivel** | diferenca real e nao-desprezivel |
| Distill 2 (402K) | Distill 3 (615K) | +0.0523 | 14/15 | 0.0006 | 0.0631 | **diferenca real e nao-desprezivel** | diferenca real e nao-desprezivel |
| FLIM (59.504) | Distill 4 (889K) | -0.0184 | 1/15 | 0.0007 | 0.0501 | **diferenca real e nao-desprezivel** ⚠ | inconclusivo |

## 1d. Consistencia por dataset (pares focais)

Mediana da diferenca dentro de cada dataset (6 fracoes cada). Um par cujo **sinal muda** entre datasets tem vantagem dependente de dominio, e a mediana agregada esconde isso.

### F1

| Par (A vs B) | eggs | larvae | protozoan | sinal consistente? |
|---|---:|---:|---:|---|
| FLIM (59.504) vs I-JEPA (632M) | +0.038 (6/6) | +0.009 (6/6) | -0.032 (1/6) | **NAO — troca de sinal** |
| Distill 3 (615K) vs Distill 4 (889K) | +0.037 (6/6) | +0.007 (4/6) | +0.035 (6/6) | sim |
| Distill 1 - FLIM init (123K) vs Distill 2 (402K) | +0.044 (5/6) | -0.030 (1/6) | +0.052 (4/6) | **NAO — troca de sinal** |
| Distill 2 (402K) vs Distill 3 (615K) | +0.080 (5/6) | +0.028 (6/6) | +0.032 (5/6) | sim |
| FLIM (59.504) vs Distill 4 (889K) | -0.021 (1/6) | -0.005 (1/6) | -0.073 (1/6) | sim |

### kappa

| Par (A vs B) | eggs | larvae | protozoan | sinal consistente? |
|---|---:|---:|---:|---|
| FLIM (59.504) vs I-JEPA (632M) | +0.105 (6/6) | +0.118 (6/6) | +0.052 (6/6) | sim |
| Distill 3 (615K) vs Distill 4 (889K) | +0.037 (5/6) | +0.014 (4/6) | +0.029 (6/6) | sim |
| Distill 1 - FLIM init (123K) vs Distill 2 (402K) | +0.007 (3/6) | -0.060 (1/6) | +0.037 (4/6) | **NAO — troca de sinal** |
| Distill 2 (402K) vs Distill 3 (615K) | +0.098 (5/6) | +0.057 (6/6) | +0.049 (5/6) | sim |
| FLIM (59.504) vs Distill 4 (889K) | +0.033 (6/6) | +0.091 (6/6) | -0.011 (1/6) | **NAO — troca de sinal** |

### Acc

| Par (A vs B) | eggs | larvae | protozoan | sinal consistente? |
|---|---:|---:|---:|---|
| FLIM (59.504) vs I-JEPA (632M) | +0.044 (6/6) | +0.009 (6/6) | -0.029 (1/6) | **NAO — troca de sinal** |
| Distill 3 (615K) vs Distill 4 (889K) | +0.028 (6/6) | +0.011 (5/6) | +0.036 (6/6) | sim |
| Distill 1 - FLIM init (123K) vs Distill 2 (402K) | +0.099 (6/6) | -0.022 (1/6) | +0.123 (6/6) | **NAO — troca de sinal** |
| Distill 2 (402K) vs Distill 3 (615K) | +0.059 (5/6) | +0.025 (5/6) | +0.026 (5/6) | sim |
| FLIM (59.504) vs Distill 4 (889K) | -0.008 (1/6) | -0.005 (2/6) | -0.061 (1/6) | sim |

## 2. Matriz completa dos 28 pares

Nenhum par fica sem teste. Holm sobre os 28 pares, por metrica — correcao bem mais severa que a da familia focal, logo os vereditos aqui sao mais conservadores. Ordenado por magnitude da mediana da diferenca.

### F1 (28 pares)

| A | B | Mediana B-A | B vence | p superior. (Holm) | p TOST (Holm) | delta_min | Veredito |
|---|---|---:|---:|---:|---:|---:|---|
| FLIM (59.504) | I-JEPA (632M) | +0.0185 | 13/18 | 0.5661 | 1.0000 | 0.1925 | inconclusivo |
| FLIM (59.504) | Distill 4 (889K) | -0.0225 | 3/18 | 0.5661 | 1.0000 | 0.0987 | inconclusivo |
| Distill 3 (615K) | Distill 4 (889K) | +0.0267 | 16/18 | 0.0023 | 0.0221 | 0.0366 | diferenca real porem menor que a margem |
| Distill 1 - FLIM init (123K) | Distill 2 (402K) | +0.0330 | 10/18 | 0.5419 | 1.0000 | 0.0676 | inconclusivo |
| Distill 2 (402K) | Distill 3 (615K) | +0.0404 | 16/18 | 0.0038 | 1.0000 | 0.0670 | diferenca real e nao-desprezivel |
| I-JEPA (632M) | Distill 4 (889K) | -0.0516 | 0/18 | 0.0002 | 1.0000 | 0.0813 | diferenca real e nao-desprezivel |
| FLIM (59.504) | Distill 3 (615K) | -0.0523 | 3/18 | 0.5661 | 1.0000 | 0.0857 | inconclusivo |
| Distill 2 (402K) | Distill 4 (889K) | +0.0649 | 16/18 | 0.0014 | 1.0000 | 0.1003 | diferenca real e nao-desprezivel |
| I-JEPA (632M) | Distill 3 (615K) | -0.0841 | 0/18 | 0.0002 | 1.0000 | 0.1169 | diferenca real e nao-desprezivel |
| Distill 1 - FLIM init (123K) | Distill 3 (615K) | +0.0954 | 15/18 | 0.0033 | 1.0000 | 0.1132 | diferenca real e nao-desprezivel |
| FLIM (59.504) | Distill 2 (402K) | -0.1029 | 3/18 | 0.5661 | 1.0000 | 0.1510 | inconclusivo |
| Distill 1 - FLIM init (123K) | Distill 4 (889K) | +0.1175 | 17/18 | 0.0006 | 1.0000 | 0.1421 | diferenca real e nao-desprezivel |
| I-JEPA (632M) | Distill 2 (402K) | -0.1282 | 0/18 | 0.0002 | 1.0000 | 0.1749 | diferenca real e nao-desprezivel |
| FLIM (59.504) | Distill 1 - FLIM init (123K) | -0.1568 | 3/18 | 0.1611 | 1.0000 | 0.1810 | inconclusivo |
| I-JEPA (632M) | Distill 1 - FLIM init (123K) | -0.1815 | 0/18 | 0.0002 | 1.0000 | 0.2236 | diferenca real e nao-desprezivel |
| LeJEPA (59.504) | Distill 2 (402K) | +0.2052 | 17/18 | 0.0002 | 1.0000 | 0.3125 | diferenca real e nao-desprezivel |
| LeJEPA (59.504) | Distill 1 (123K) | -0.2060 | 3/18 | 0.0146 | 1.0000 | 0.2441 | diferenca real e nao-desprezivel |
| LeJEPA (59.504) | Distill 1 - FLIM init (123K) | +0.2311 | 15/18 | 0.0028 | 1.0000 | 0.2994 | diferenca real e nao-desprezivel |
| LeJEPA (59.504) | Distill 3 (615K) | +0.3109 | 18/18 | 0.0002 | 1.0000 | 0.3631 | diferenca real e nao-desprezivel |
| LeJEPA (59.504) | Distill 4 (889K) | +0.3385 | 18/18 | 0.0002 | 1.0000 | 0.3936 | diferenca real e nao-desprezivel |
| Distill 1 (123K) | Distill 1 - FLIM init (123K) | +0.3692 | 18/18 | 0.0002 | 1.0000 | 0.4247 | diferenca real e nao-desprezivel |
| Distill 1 (123K) | Distill 2 (402K) | +0.3709 | 18/18 | 0.0002 | 1.0000 | 0.4491 | diferenca real e nao-desprezivel |
| FLIM (59.504) | LeJEPA (59.504) | -0.4021 | 3/18 | 0.0054 | 1.0000 | 0.4411 | diferenca real e nao-desprezivel |
| LeJEPA (59.504) | I-JEPA (632M) | +0.4265 | 18/18 | 0.0002 | 1.0000 | 0.4631 | diferenca real e nao-desprezivel |
| Distill 1 (123K) | Distill 3 (615K) | +0.4354 | 18/18 | 0.0002 | 1.0000 | 0.4964 | diferenca real e nao-desprezivel |
| Distill 1 (123K) | Distill 4 (889K) | +0.4439 | 18/18 | 0.0002 | 1.0000 | 0.5370 | diferenca real e nao-desprezivel |
| FLIM (59.504) | Distill 1 (123K) | -0.4955 | 2/18 | 0.0007 | 1.0000 | 0.5829 | diferenca real e nao-desprezivel |
| I-JEPA (632M) | Distill 1 (123K) | -0.5058 | 0/18 | 0.0002 | 1.0000 | 0.6156 | diferenca real e nao-desprezivel |

### kappa (28 pares)

| A | B | Mediana B-A | B vence | p superior. (Holm) | p TOST (Holm) | delta_min | Veredito |
|---|---|---:|---:|---:|---:|---:|---|
| FLIM (59.504) | Distill 3 (615K) | +0.0130 | 10/18 | 0.6694 | 1.0000 | 0.0743 | inconclusivo |
| Distill 1 - FLIM init (123K) | Distill 2 (402K) | -0.0132 | 8/18 | 1.0000 | 1.0000 | 0.0462 | inconclusivo |
| Distill 3 (615K) | Distill 4 (889K) | +0.0257 | 15/18 | 0.0073 | 0.0784 | 0.0400 | diferenca real e nao-desprezivel |
| FLIM (59.504) | Distill 4 (889K) | +0.0461 | 13/18 | 0.0237 | 1.0000 | 0.1088 | diferenca real e nao-desprezivel |
| I-JEPA (632M) | Distill 4 (889K) | -0.0610 | 0/18 | 0.0002 | 1.0000 | 0.1268 | diferenca real e nao-desprezivel |
| Distill 2 (402K) | Distill 3 (615K) | +0.0665 | 16/18 | 0.0030 | 1.0000 | 0.0894 | diferenca real e nao-desprezivel |
| FLIM (59.504) | Distill 2 (402K) | -0.0758 | 6/18 | 0.7386 | 1.0000 | 0.0916 | inconclusivo |
| Distill 1 - FLIM init (123K) | Distill 3 (615K) | +0.0760 | 15/18 | 0.0034 | 1.0000 | 0.0933 | diferenca real e nao-desprezivel |
| FLIM (59.504) | Distill 1 - FLIM init (123K) | -0.0844 | 7/18 | 0.7386 | 1.0000 | 0.0962 | inconclusivo |
| I-JEPA (632M) | Distill 3 (615K) | -0.0914 | 0/18 | 0.0002 | 1.0000 | 0.1657 | diferenca real e nao-desprezivel |
| Distill 2 (402K) | Distill 4 (889K) | +0.0937 | 16/18 | 0.0014 | 1.0000 | 0.1207 | diferenca real e nao-desprezivel |
| FLIM (59.504) | I-JEPA (632M) | +0.1039 | 18/18 | 0.0002 | 1.0000 | 0.2613 | diferenca real e nao-desprezivel |
| Distill 1 - FLIM init (123K) | Distill 4 (889K) | +0.1042 | 17/18 | 0.0005 | 1.0000 | 0.1242 | diferenca real e nao-desprezivel |
| I-JEPA (632M) | Distill 1 - FLIM init (123K) | -0.1602 | 0/18 | 0.0002 | 1.0000 | 0.2646 | diferenca real e nao-desprezivel |
| I-JEPA (632M) | Distill 2 (402K) | -0.2026 | 0/18 | 0.0002 | 1.0000 | 0.2301 | diferenca real e nao-desprezivel |
| LeJEPA (59.504) | Distill 1 (123K) | -0.2276 | 3/18 | 0.0125 | 1.0000 | 0.2854 | diferenca real e nao-desprezivel |
| LeJEPA (59.504) | Distill 2 (402K) | +0.3376 | 18/18 | 0.0002 | 1.0000 | 0.4581 | diferenca real e nao-desprezivel |
| LeJEPA (59.504) | Distill 1 - FLIM init (123K) | +0.4227 | 17/18 | 0.0005 | 1.0000 | 0.4622 | diferenca real e nao-desprezivel |
| FLIM (59.504) | LeJEPA (59.504) | -0.4352 | 2/18 | 0.0008 | 1.0000 | 0.5042 | diferenca real e nao-desprezivel |
| LeJEPA (59.504) | Distill 3 (615K) | +0.4700 | 18/18 | 0.0002 | 1.0000 | 0.5306 | diferenca real e nao-desprezivel |
| LeJEPA (59.504) | Distill 4 (889K) | +0.4976 | 18/18 | 0.0002 | 1.0000 | 0.5618 | diferenca real e nao-desprezivel |
| Distill 1 (123K) | Distill 1 - FLIM init (123K) | +0.5251 | 18/18 | 0.0002 | 1.0000 | 0.6340 | diferenca real e nao-desprezivel |
| LeJEPA (59.504) | I-JEPA (632M) | +0.5735 | 18/18 | 0.0002 | 1.0000 | 0.6410 | diferenca real e nao-desprezivel |
| Distill 1 (123K) | Distill 2 (402K) | +0.5987 | 18/18 | 0.0002 | 1.0000 | 0.6234 | diferenca real e nao-desprezivel |
| FLIM (59.504) | Distill 1 (123K) | -0.6070 | 0/18 | 0.0002 | 1.0000 | 0.6781 | diferenca real e nao-desprezivel |
| Distill 1 (123K) | Distill 3 (615K) | +0.6330 | 18/18 | 0.0002 | 1.0000 | 0.6976 | diferenca real e nao-desprezivel |
| Distill 1 (123K) | Distill 4 (889K) | +0.6531 | 18/18 | 0.0002 | 1.0000 | 0.7248 | diferenca real e nao-desprezivel |
| I-JEPA (632M) | Distill 1 (123K) | -0.7616 | 0/18 | 0.0002 | 1.0000 | 0.8040 | diferenca real e nao-desprezivel |

### Acc (28 pares)

| A | B | Mediana B-A | B vence | p superior. (Holm) | p TOST (Holm) | delta_min | Veredito |
|---|---|---:|---:|---:|---:|---:|---|
| FLIM (59.504) | Distill 4 (889K) | -0.0087 | 4/18 | 0.4335 | 1.0000 | 0.1024 | inconclusivo |
| FLIM (59.504) | I-JEPA (632M) | +0.0175 | 13/18 | 0.4335 | 1.0000 | 0.1699 | inconclusivo |
| Distill 3 (615K) | Distill 4 (889K) | +0.0281 | 17/18 | 0.0006 | 0.0059 | 0.0369 | diferenca real porem menor que a margem |
| FLIM (59.504) | Distill 3 (615K) | -0.0328 | 4/18 | 0.4335 | 1.0000 | 0.0719 | inconclusivo |
| Distill 2 (402K) | Distill 3 (615K) | +0.0369 | 15/18 | 0.0116 | 1.0000 | 0.0533 | diferenca real e nao-desprezivel |
| I-JEPA (632M) | Distill 4 (889K) | -0.0419 | 0/18 | 0.0002 | 1.0000 | 0.0641 | diferenca real e nao-desprezivel |
| Distill 2 (402K) | Distill 4 (889K) | +0.0604 | 17/18 | 0.0007 | 1.0000 | 0.0866 | diferenca real e nao-desprezivel |
| I-JEPA (632M) | Distill 3 (615K) | -0.0762 | 0/18 | 0.0002 | 1.0000 | 0.1044 | diferenca real e nao-desprezivel |
| FLIM (59.504) | Distill 2 (402K) | -0.0883 | 3/18 | 0.4335 | 1.0000 | 0.1199 | inconclusivo |
| Distill 1 - FLIM init (123K) | Distill 2 (402K) | +0.0979 | 13/18 | 0.0154 | 1.0000 | 0.1197 | diferenca real e nao-desprezivel |
| I-JEPA (632M) | Distill 2 (402K) | -0.1112 | 0/18 | 0.0002 | 1.0000 | 0.1488 | diferenca real e nao-desprezivel |
| Distill 1 - FLIM init (123K) | Distill 3 (615K) | +0.1456 | 16/18 | 0.0016 | 1.0000 | 0.1514 | diferenca real e nao-desprezivel |
| LeJEPA (59.504) | Distill 1 - FLIM init (123K) | +0.1665 | 14/18 | 0.0332 | 1.0000 | 0.2269 | diferenca real e nao-desprezivel |
| Distill 1 - FLIM init (123K) | Distill 4 (889K) | +0.1737 | 17/18 | 0.0003 | 1.0000 | 0.1852 | diferenca real e nao-desprezivel |
| FLIM (59.504) | Distill 1 - FLIM init (123K) | -0.1795 | 3/18 | 0.0659 | 1.0000 | 0.2055 | inconclusivo |
| LeJEPA (59.504) | Distill 2 (402K) | +0.1861 | 16/18 | 0.0006 | 1.0000 | 0.2883 | diferenca real e nao-desprezivel |
| LeJEPA (59.504) | Distill 1 (123K) | -0.1983 | 4/18 | 0.0659 | 1.0000 | 0.2401 | inconclusivo |
| I-JEPA (632M) | Distill 1 - FLIM init (123K) | -0.2105 | 0/18 | 0.0002 | 1.0000 | 0.2448 | diferenca real e nao-desprezivel |
| Distill 1 (123K) | Distill 1 - FLIM init (123K) | +0.2550 | 18/18 | 0.0002 | 1.0000 | 0.3435 | diferenca real e nao-desprezivel |
| LeJEPA (59.504) | Distill 3 (615K) | +0.2661 | 17/18 | 0.0004 | 1.0000 | 0.3327 | diferenca real e nao-desprezivel |
| LeJEPA (59.504) | Distill 4 (889K) | +0.2978 | 17/18 | 0.0003 | 1.0000 | 0.3592 | diferenca real e nao-desprezivel |
| FLIM (59.504) | LeJEPA (59.504) | -0.3194 | 3/18 | 0.0105 | 1.0000 | 0.3790 | diferenca real e nao-desprezivel |
| Distill 1 (123K) | Distill 2 (402K) | +0.3439 | 18/18 | 0.0002 | 1.0000 | 0.3973 | diferenca real e nao-desprezivel |
| LeJEPA (59.504) | I-JEPA (632M) | +0.3609 | 18/18 | 0.0002 | 1.0000 | 0.4056 | diferenca real e nao-desprezivel |
| Distill 1 (123K) | Distill 3 (615K) | +0.3716 | 18/18 | 0.0002 | 1.0000 | 0.4360 | diferenca real e nao-desprezivel |
| Distill 1 (123K) | Distill 4 (889K) | +0.4018 | 18/18 | 0.0002 | 1.0000 | 0.4648 | diferenca real e nao-desprezivel |
| FLIM (59.504) | Distill 1 (123K) | -0.4407 | 2/18 | 0.0007 | 1.0000 | 0.4856 | diferenca real e nao-desprezivel |
| I-JEPA (632M) | Distill 1 (123K) | -0.4486 | 0/18 | 0.0002 | 1.0000 | 0.5317 | diferenca real e nao-desprezivel |

## 3. O que isso muda no texto do artigo

### 3.1 "FLIM praticamente empatado com o I-JEPA" — verdadeiro, mas precisa de escopo

Em **F1** o par e **inconclusivo**: nao se detecta diferenca (p Holm = 0.3251) *e* nao se demonstra equivalencia (p TOST Holm = 0.4557). O `delta_min` = **0.1925** diz o tamanho do problema: so seria possivel declarar empate admitindo uma margem de +/-0.19 de F1, o que e largo demais para ter sentido pratico. Com n = 18 o experimento simplesmente nao distingue os dois.

Em **kappa** a situacao e pior para a afirmacao: o I-JEPA vence em 18/18 celulas com mediana +0.1039 e p Holm = 3.81e-05 — **diferenca real e nao-desprezivel**. Em **Acc**, inconclusivo (0.2168).

O tamanho dessa vantagem em kappa depende quase inteiramente da quantidade de rotulos (§1b): +0.589 a 1%, +0.230 a 5%, +0.094 a 25% e +0.066 a 100% (kappa 0.867 do FLIM contra 0.933 do teacher). Quase toda a superioridade se concentra no regime de rotulo escasso, onde o FLIM colapsa, e estabiliza em torno de 0,07 a partir de 25%.

Mas ela **nao desaparece** fora desse regime: descartando a fracao de 1% (§1c), o I-JEPA ainda vence em 15/15 celulas, com mediana +0.0974 e p Holm = 0.0003. E e consistente nos tres dominios (§1d): eggs +0.105, larvae +0.118, protozoan +0.052 — sem a troca de sinal entre datasets que o F1 apresenta.

**O empate existe, mas so fora do regime de colapso e so em F1/Acc.** Esta e a descoberta mais util deste relatorio para a redacao do artigo. Descartando a fracao de 1%, o par FLIM vs I-JEPA passa a **EQUIVALENTES** em F1 (mediana +0.0071, superioridade p = 1.0000, TOST p = 0.0066, `delta_min` = **0.0243**) e tambem em Acc (`delta_min` = 0.0267). O `delta_min` de 0.024 e uma margem **apertada** — cerca de um terco dos 0.05 pre-especificados e da ordem do ruido entre splits. Ou seja: a partir de 5% dos rotulos, FLIM e I-JEPA sao equivalentes em F1 e acuracia dentro de uma margem muito estreita, e isso agora esta **demonstrado**, nao apenas nao-refutado.

O contraste com o agregado de 18 celulas nao e contradicao: as 3 celulas de 1% sozinhas (+0.589 de kappa, e +0,516 de F1) carregavam toda a diferenca e ao mesmo tempo inflavam a dispersao, o que impedia o TOST de concluir qualquer coisa. Separar os dois regimes resolve as duas pontas.

> **Redacao sugerida.** Trocar "estatisticamente empatados" por: *"a partir de 5% dos rotulos, o FLIM e estatisticamente equivalente ao teacher I-JEPA em F1 e acuracia, dentro de uma margem de +/-0.024 (TOST pareado, n = 15, p = 0.0066), com ~10.600x menos parametros. O teacher mantem vantagem em kappa (+0.097, p = 0.0003), e domina no regime de 1% de rotulos, onde o FLIM colapsa."* Essa formulacao e mais forte que a original, porque troca uma afirmacao de empate sem teste por uma equivalencia demonstrada com escopo declarado.

### 3.2 "Distill 3 vs Distill 4: ganho praticamente zero" — quase certo, mas o numero esta errado

Este e o par mais interessante do relatorio, e cai na caixa **diferenca real porem menor que a margem** nas tres metricas. Em F1: o Distill 4 vence em 16/18 celulas, com p Holm = 0.0010 — a diferenca **existe e e consistente**. Mas o TOST tambem rejeita (p Holm = 0.0039, `delta_min` = 0.0366): ela esta **comprovadamente abaixo de 0.037 de F1**. Mesmo padrao em kappa (0.0400) e Acc (0.0369).

Sob a correcao mais severa da matriz completa (Holm sobre 28 pares), a metade de equivalencia sobrevive em 2 das 3 metricas — em kappa o TOST deixa de rejeitar. A conclusao a citar e a da familia focal, que e pre-especificada; a queda na matriz completa e o custo de testar tudo, nao um resultado contraditorio.

O numero "+0,007 de F1" do texto vem so do recorte a **100% dos rotulos**, com n = 3 datasets — sem poder para teste algum. Sobre as 18 celulas a mediana e +0.0267, cerca de 4x maior. A conclusao qualitativa ("nao vale a pena pagar +45% de params") se mantem, e agora com respaldo estatistico bem mais forte do que a versao original — mas o numero citado precisa mudar.

> **Redacao sugerida.** *"Subir de Distill 3 (615K) para Distill 4 (889K) — +45% de parametros — produz um ganho consistente porem minusculo: mediana de +0.027 de F1 sobre as 18 celulas (p = 0.0010), com equivalencia demonstrada dentro de +/-0.037 (TOST, p = 0.0039). Ou seja: a diferenca e real, e e comprovadamente desprezivel."* Essa formulacao e mais forte que "praticamente zero", porque poe um limite superior demonstrado no tamanho do ganho.

### 3.3 Distill 1 FLIM init (123K) vs Distill 2 (402K)

O resultado e **dependente da metrica**, e o texto precisa dizer isso. Em **kappa** o veredito e **inconclusivo**: mediana -0.0132 (a favor de Distill 1 - FLIM init (123K)), superioridade p = 1.0000, TOST p = 0.1975, delta_min = 0.0462.

E o par que **mais perto chega** de um empate demonstrado em todo o estudo, e ainda assim nao chega: o TOST bruto da p = 0.0494 (passaria sozinho), mas nao sobrevive a correcao de Holm sobre os 5 pares da familia (0.1975). O `delta_min` = 0.0462 explica por que e tao apertado: a equivalencia se sustentaria com uma margem de +/-0.046, logo abaixo dos 0.05 pre-especificados. **Nao escreva "mesmo kappa" no artigo** — escreva que a diferenca em kappa e indistinguivel de zero nesta bateria (mediana -0.013, p = 1.0000) e que o experimento nao tem poder para converter isso em equivalencia formal.

Em **F1** o par e inconclusivo (p = 0.3251) e em **Acc** o Distill 2 leva vantagem real (+0.0979, p = 0.0058, delta_min = 0.1197). A afirmacao "1/3 dos parametros pelo mesmo desempenho" so e defensavel em kappa, e mesmo la apenas como ausencia de diferenca detectavel — nao como equivalencia. Declarar essa dependencia de metrica evita a critica de cherry-picking.

### 3.4 Panorama dos 28 pares

Sobre as 84 linhas da matriz completa (28 pares x 3 metricas):

- **66** sao diferencas reais e nao-despreziveis;
- **2** sao diferencas reais porem menores que a margem de 0.05;
- **0** sao equivalencias demonstradas;
- **16** ficam inconclusivas.

**Sobre as 18 celulas completas, em nenhum dos 28 pares e em nenhuma das 3 metricas ha empate estatisticamente demonstrado.** Nesse recorte a palavra "empate" nao deveria aparecer no texto. Restam duas construcoes defensaveis:

1. **"diferenca real porem limitada a X"** — quando o TOST rejeita junto com a superioridade. Ocorre em 2 celula(s): Distill 3 (615K) vs Distill 4 (889K) (F1, Acc). E a formulacao mais forte disponivel, porque poe um teto demonstrado no tamanho do ganho.
2. **"nao se detectou diferenca, e o experimento nao tem poder para estabelecer equivalencia"** — nas 16 celulas inconclusivas, entre elas FLIM vs I-JEPA em F1 e Acc.

**Mas o quadro muda ao separar o regime de rotulo escasso.** Descartando a fracao de 1% (§1c), 2 celula(s) da familia focal atingem equivalencia demonstrada: FLIM (59.504) vs I-JEPA (632M) em F1 (delta_min 0.0243); FLIM (59.504) vs I-JEPA (632M) em Acc (delta_min 0.0267). A licao metodologica e que a fracao de 1% estava envenenando o agregado nas duas direcoes — inflava as diferencas medias e, por aumentar a dispersao, impedia o TOST de concluir. **Qualquer afirmacao de equivalencia no artigo deve declarar o regime de supervisao a que se aplica.**

As 66 celulas de diferenca real e nao-desprezivel mostram que o problema nao e falta de poder generalizada — a bateria separa bem os modelos distantes entre si. O que ela dificilmente consegue, no agregado de 18 celulas, e **provar igualdade** entre os modelos proximos, que e exatamente o que as afirmacoes de empate do texto exigem.

## 4. Ressalvas

- **A coluna `f1` e F1 ponderado, nao macro.** `scripts/normalize_reports.py:114` mapeia `test_f1_weighted -> f1`; `src/evaluate/svm_classification_flim.py:144` confirma `average="weighted"`. O cabecalho de `metrics_distillation/params_vs_quality_flim.md` diz "F1 (macro)" e **esta errado**. Isso importa para a leitura deste relatorio: F1 ponderado e inflado pela classe majoritaria em dados desbalanceados, enquanto o kappa desconta concordancia por acaso. Quando as duas metricas divergem — como em FLIM vs I-JEPA, inconclusivo em F1 e diferenca real em kappa — o kappa e o numero mais honesto, e a divergencia provavelmente reflete desempenho pior nas classes minoritarias.
- **As 18 celulas nao sao independentes.** Dentro de um dataset, os subconjuntos de treino sao estritamente encaixados (1% ⊂ 5% ⊂ ... ⊂ 100%, contencao verificada nos JSONs de `data/to_mateus/splits_incremental/`) e o conjunto de teste e identico nas 6 fracoes. O Wilcoxon supoe pares independentes, logo n = 18 e otimista para **ambos** os testes. Para a superioridade isso e anticonservador (p otimista); para o TOST tambem, o que significa que as equivalencias declaradas aqui devem ser lidas como o limite superior do que os dados sustentam.
- **A margem delta e uma escolha, nao um fato.** Por isso `delta_min` esta em toda linha: quem discordar de 0.05 le a coluna e aplica o proprio criterio.
- **Equivalencia nao e identidade.** Declarar que dois modelos sao equivalentes dentro de +/-0.05 nao os torna intercambiaveis para todo fim — apenas afirma que a diferenca media de qualidade nesta bateria esta abaixo desse limiar.
- Os `.ckpt` das runs Distill 1/2/3/4 nao estao mais no disco; contagens de parametros vem da topologia reinstanciada (secao 5 do `README.md` deste diretorio).
- A matriz completa usa Holm sobre 28 pares. Um par pode aparecer como significativo na familia focal e nao-significativo na matriz completa: nao e contradicao, e o preco da cobertura exaustiva. A familia focal e pre-especificada e e a que sustenta as conclusoes.

## 5. Reproducao

```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM
LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib \
/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python \
    statistics/tools/wilcoxon_equivalence.py
```

Versoes: scipy 1.17.1, statsmodels 0.14.6, numpy 1.26.4, pandas 3.0.3.
