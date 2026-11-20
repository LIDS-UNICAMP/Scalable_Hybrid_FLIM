# Wilcoxon pareado (signed-rank) na métrica kappa de Cohen

Baseline: **FLIM (59.504)** (`SVM_FLIM`). Desenho um-contra-todos, 7 comparações. Unidades pareadas: 18 células `(dataset, fração de pré-treino)` = 3 datasets x 6 frações (1, 5, 25, 50, 75, 100 %). alfa = 0.05.

Convenção de sinal: `diferença = modelo - baseline`. Diferença positiva e `r` positivo indicam que o modelo supera o FLIM; valores negativos indicam que o FLIM supera o modelo. `LeJEPA` restrito a `init == trunc_normal`. Teste bilateral, `zero_method="wilcox"`, p exato quando n efetivo <= 25. Correção de múltiplas comparações por Holm (principal) e Bonferroni (referência conservadora) sobre a família das 7 comparações. IC 95% da mediana da diferença por bootstrap percentil (10.000 reamostragens das 18 células, semente 42).

## Tabela principal (n = 18 pares)

| Modelo | Mediana da dif. vs FLIM | IC95% bootstrap | W | p bruto | p Holm | p Bonferroni | r rank-biserial | Melhor | Signif. (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| I-JEPA (632M) | +0.1039 | [+0.0759, +0.2004] | 0.0 | 7.63e-06 | 5.34e-05 | 5.34e-05 | +1.000 | I-JEPA (632M) | sim |
| Distill 4 (889K) | +0.0461 | [+0.0064, +0.0909] | 23.0 | 0.0047 | 0.0190 | 0.0332 | +0.731 | Distill 4 (889K) | sim |
| Distill 3 (615K) | +0.0130 | [-0.0176, +0.0743] | 53.0 | 0.1674 | 0.5021 | 1.0000 | +0.380 | Distill 3 (615K) | nao |
| Distill 2 (402K) | -0.0758 | [-0.1046, +0.0177] | 64.0 | 0.3692 | 0.5021 | 1.0000 | -0.251 | FLIM (59.504) | nao |
| Distill 1 - FLIM init (123K) | -0.0844 | [-0.1070, +0.0623] | 58.0 | 0.2462 | 0.5021 | 1.0000 | -0.322 | FLIM (59.504) | nao |
| LeJEPA (59.504) | -0.4352 | [-0.5572, -0.2071] | 5.0 | 7.63e-05 | 0.0004 | 0.0005 | -0.942 | FLIM (59.504) | sim |
| Distill 1 (123K) | -0.6070 | [-0.7066, -0.5575] | 0.0 | 7.63e-06 | 5.34e-05 | 5.34e-05 | -1.000 | FLIM (59.504) | sim |

Colunas auxiliares (n efetivo após descarte de zeros, W+/W-, médias, contagem de células em que cada lado vence, IC da diferença média) estão em `wilcoxon_kappa.csv`.

### Suporte por contagem de células

| Modelo | Células modelo > FLIM | Células FLIM > modelo | Empates | n efetivo | Método p | Média kappa modelo | Média kappa FLIM |
|---|---|---|---|---|---|---|---|
| I-JEPA (632M) | 18/18 | 0/18 | 0 | 18 | exact | 0.8621 | 0.6715 |
| Distill 4 (889K) | 13/18 | 5/18 | 0 | 18 | exact | 0.7569 | 0.6715 |
| Distill 3 (615K) | 10/18 | 8/18 | 0 | 18 | exact | 0.7296 | 0.6715 |
| Distill 2 (402K) | 6/18 | 12/18 | 0 | 18 | exact | 0.6650 | 0.6715 |
| Distill 1 - FLIM init (123K) | 7/18 | 11/18 | 0 | 18 | exact | 0.6541 | 0.6715 |
| LeJEPA (59.504) | 2/18 | 16/18 | 0 | 18 | exact | 0.2946 | 0.6715 |
| Distill 1 (123K) | 0/18 | 18/18 | 0 | 18 | exact | 0.1042 | 0.6715 |

## Análise secundária por dataset (exploratória, n = 6)

Wilcoxon dentro de cada dataset, pareando pelas 6 frações de pré-treino. Com n = 6 o menor p bilateral atingível é 0,0312, e nenhum resultado sobreviveria à correção para 21 testes. Os p abaixo são **brutos** e servem só para inspecionar se o sinal do efeito é consistente entre datasets ou vem de um deles. Não use esta tabela para inferência confirmatória.

| Modelo | Dataset | Mediana da dif. | Células modelo > FLIM | W | p bruto | r rank-biserial |
|---|---|---|---|---|---|---|
| I-JEPA (632M) | eggs | +0.1048 | 6/6 | 0.0 | 0.0312 | +1.000 |
| I-JEPA (632M) | larvae | +0.1180 | 6/6 | 0.0 | 0.0312 | +1.000 |
| I-JEPA (632M) | protozoan | +0.0518 | 6/6 | 0.0 | 0.0312 | +1.000 |
| Distill 4 (889K) | eggs | +0.0329 | 6/6 | 0.0 | 0.0312 | +1.000 |
| Distill 4 (889K) | larvae | +0.0909 | 6/6 | 0.0 | 0.0312 | +1.000 |
| Distill 4 (889K) | protozoan | -0.0115 | 1/6 | 6.0 | 0.4375 | -0.429 |
| Distill 3 (615K) | eggs | +0.0055 | 3/6 | 8.0 | 0.6875 | +0.238 |
| Distill 3 (615K) | larvae | +0.0743 | 6/6 | 0.0 | 0.0312 | +1.000 |
| Distill 3 (615K) | protozoan | -0.0275 | 1/6 | 6.0 | 0.4375 | -0.429 |
| Distill 2 (402K) | eggs | -0.0944 | 1/6 | 6.0 | 0.4375 | -0.429 |
| Distill 2 (402K) | larvae | +0.0177 | 4/6 | 6.0 | 0.4375 | +0.429 |
| Distill 2 (402K) | protozoan | -0.0994 | 1/6 | 5.0 | 0.3125 | -0.524 |
| Distill 1 - FLIM init (123K) | eggs | -0.0859 | 0/6 | 0.0 | 0.0312 | -1.000 |
| Distill 1 - FLIM init (123K) | larvae | +0.0726 | 6/6 | 0.0 | 0.0312 | +1.000 |
| Distill 1 - FLIM init (123K) | protozoan | -0.1070 | 1/6 | 5.0 | 0.3125 | -0.524 |
| LeJEPA (59.504) | eggs | -0.4794 | 1/6 | 1.0 | 0.0625 | -0.905 |
| LeJEPA (59.504) | larvae | -0.2071 | 0/6 | 0.0 | 0.0312 | -1.000 |
| LeJEPA (59.504) | protozoan | -0.6183 | 1/6 | 1.0 | 0.0625 | -0.905 |
| Distill 1 (123K) | eggs | -0.6436 | 0/6 | 0.0 | 0.0312 | -1.000 |
| Distill 1 (123K) | larvae | -0.7001 | 0/6 | 0.0 | 0.0312 | -1.000 |
| Distill 1 (123K) | protozoan | -0.5791 | 0/6 | 0.0 | 0.0312 | -1.000 |

## Leitura dos resultados

Na métrica kappa, 3 dos 7 modelos apresentam mediana de diferença positiva contra o FLIM e 4 apresentam mediana negativa. Antes da correção, I-JEPA (632M), Distill 4 (889K), LeJEPA (59.504), Distill 1 (123K) ficam abaixo de 0.05. Depois de Holm, 4 comparações permanecem significativas (acima do FLIM: I-JEPA (632M), Distill 4 (889K); abaixo do FLIM: LeJEPA (59.504), Distill 1 (123K)). As demais (Distill 3 (615K), Distill 2 (402K), Distill 1 - FLIM init (123K)) não se separam do FLIM. O IC95% bootstrap da mediana da diferença exclui zero em 4 de 7 comparações (I-JEPA (632M), Distill 4 (889K), LeJEPA (59.504), Distill 1 (123K)); nas 3 restantes o intervalo contém zero, de acordo com os p ajustados.

Na análise por dataset o sinal da mediana da diferença troca entre datasets para Distill 4 (889K), Distill 3 (615K), Distill 2 (402K), Distill 1 - FLIM init (123K). Mantém o mesmo sinal nos três datasets: I-JEPA (632M), LeJEPA (59.504), Distill 1 (123K), ou seja, nesses casos o resultado agregado não vem de um único dataset. Ressalva: Distill 4 (889K) aparece como significativo no teste agregado apesar de inverter o sinal em pelo menos um dataset, logo o efeito não é uniforme entre os três problemas.

Limitação de poder: n = 18 pares, teste bilateral e família de 7 hipóteses corrigida por Holm. O menor p bilateral exato alcançável com n = 18 é 7.63e-06, portanto o teste só detecta efeito quando o sinal da diferença é quase uniforme entre as 18 células. Diferenças moderadas mas inconsistentes ao longo das frações de pré-treino ficam indetectáveis, e ausência de significância não é evidência de equivalência. As células pareadas agregam 3 splits cada (média já calculada no CSV de entrada), logo a variabilidade entre splits não entra no teste: o Wilcoxon trata a heterogeneidade de dataset e de fração de pré-treino como a única fonte de variação pareada.
