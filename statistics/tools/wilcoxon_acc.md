# Wilcoxon pareado de sinais - Acuracia (`acc`)

Baseline: **FLIM (59.504)** (`SVM_FLIM`). Desenho um-contra-todos, 7 comparacoes, alpha = 0.05.

Unidade pareada: celula `(dataset, fracao de pre-treino)` = 3 datasets x 6 fracoes, **n = 18 pares** por comparacao. Cada celula e a media sobre 3 splits.

Convencao de sinal: **diferenca = modelo - FLIM**. Positivo significa que o modelo tem acuracia maior que o FLIM; negativo significa que o FLIM e melhor. A rank-biserial `r` segue a mesma convencao.

Teste: `scipy.stats.wilcoxon(..., alternative="two-sided", zero_method="wilcox")`. Correcao de multiplicidade: `statsmodels.stats.multitest.multipletests` (`holm` principal, `bonferroni` como referencia conservadora). IC95%: bootstrap percentil com 10.000 reamostragens das 18 celulas pareadas, semente 42.

## Tabela principal

| Modelo | Mediana da dif. vs FLIM | IC95% bootstrap (mediana) | Media da dif. | IC95% (media) | vitorias mod./FLIM | W | n_ef | p bruto | p Holm | p Bonferroni | r rank-biserial | Signif. Holm | Melhor |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|---|
| Distill 1 (123K) | -0.4407 | [-0.4821, -0.3684] | -0.3947 | [-0.4959, -0.2846] | 2/16 | 4.0 | 18 | 5.34e-05 | 3.74e-04 | 3.74e-04 | -0.953 | sim | **FLIM (59.504)** |
| LeJEPA (59.504) | -0.3194 | [-0.4075, -0.1846] | -0.2596 | [-0.3699, -0.1366] | 3/15 | 15.0 | 18 | 0.0010 | 0.0063 | 0.0073 | -0.825 | sim | **FLIM (59.504)** |
| Distill 1 - FLIM init (123K) | -0.1795 | [-0.2320, -0.0182] | -0.1196 | [-0.1955, -0.0368] | 3/15 | 29.0 | 18 | 0.0120 | 0.0602 | 0.0842 | -0.661 | nao | sem diferenca detectada (tendencia: FLIM (59.504)) |
| I-JEPA (632M) | 0.0175 | [-0.0061, 0.0442] | 0.0841 | [0.0083, 0.1770] | 13/5 | 48.0 | 18 | 0.1084 | 0.4335 | 0.7587 | 0.439 | nao | sem diferenca detectada (tendencia: I-JEPA (632M)) |
| Distill 2 (402K) | -0.0883 | [-0.1311, -0.0425] | -0.0383 | [-0.1123, 0.0525] | 3/15 | 51.0 | 18 | 0.1415 | 0.4335 | 0.9906 | -0.404 | nao | sem diferenca detectada (tendencia: FLIM (59.504)) |
| Distill 3 (615K) | -0.0328 | [-0.0793, -0.0132] | -0.0021 | [-0.0677, 0.0766] | 4/14 | 52.0 | 18 | 0.1540 | 0.4335 | 1.0000 | -0.392 | nao | sem diferenca detectada (tendencia: FLIM (59.504)) |
| Distill 4 (889K) | -0.0087 | [-0.0429, -0.0030] | 0.0265 | [-0.0359, 0.1025] | 4/14 | 53.0 | 18 | 0.1674 | 0.4335 | 1.0000 | -0.380 | nao | sem diferenca detectada (tendencia: FLIM (59.504)) |

`W` e a estatistica devolvida pelo scipy (min(W+, W-)). `n_ef` e o numero de pares apos descartar diferencas nulas. Todos os p-valores desta tabela usaram exact.

## Analise secundaria por dataset (exploratoria)

n = 6 fracoes por dataset. **Sem correcao de multiplicidade e com poder muito baixo**: o menor p bilateral exato alcancavel com n = 6 e 0.03125. Serve so para checar se o sinal do efeito e consistente entre os tres datasets, nao para sustentar conclusao isolada.

| Modelo | Dataset | Mediana da dif. | vitorias mod./FLIM | W | n_ef | p bruto | r | Melhor |
|---|---|---:|:---:|---:|---:|---:|---:|---|
| Distill 1 (123K) | eggs | -0.5434 | 0/6 | 0.0 | 6 | 0.0312 | -1.000 | FLIM |
| Distill 1 (123K) | larvae | -0.4207 | 1/5 | 1.0 | 6 | 0.0625 | -0.905 | FLIM |
| Distill 1 (123K) | protozoan | -0.4405 | 1/5 | 1.0 | 6 | 0.0625 | -0.905 | FLIM |
| LeJEPA (59.504) | eggs | -0.3594 | 1/5 | 2.0 | 6 | 0.0938 | -0.810 | FLIM |
| LeJEPA (59.504) | larvae | -0.1846 | 1/5 | 1.0 | 6 | 0.0625 | -0.905 | FLIM |
| LeJEPA (59.504) | protozoan | -0.4515 | 1/5 | 1.0 | 6 | 0.0625 | -0.905 | FLIM |
| Distill 1 - FLIM init (123K) | eggs | -0.1892 | 1/5 | 1.0 | 6 | 0.0625 | -0.905 | FLIM |
| Distill 1 - FLIM init (123K) | larvae | -0.0182 | 1/5 | 6.0 | 6 | 0.4375 | -0.429 | FLIM |
| Distill 1 - FLIM init (123K) | protozoan | -0.2330 | 1/5 | 1.0 | 6 | 0.0625 | -0.905 | FLIM |
| I-JEPA (632M) | eggs | 0.0442 | 6/0 | 0.0 | 6 | 0.0312 | 1.000 | modelo |
| I-JEPA (632M) | larvae | 0.0092 | 6/0 | 0.0 | 6 | 0.0312 | 1.000 | modelo |
| I-JEPA (632M) | protozoan | -0.0290 | 1/5 | 6.0 | 6 | 0.4375 | -0.429 | FLIM |
| Distill 2 (402K) | eggs | -0.1094 | 1/5 | 6.0 | 6 | 0.4375 | -0.429 | FLIM |
| Distill 2 (402K) | larvae | -0.0425 | 1/5 | 6.0 | 6 | 0.4375 | -0.429 | FLIM |
| Distill 2 (402K) | protozoan | -0.1465 | 1/5 | 6.0 | 6 | 0.4375 | -0.429 | FLIM |
| Distill 3 (615K) | eggs | -0.0328 | 1/5 | 6.0 | 6 | 0.4375 | -0.429 | FLIM |
| Distill 3 (615K) | larvae | -0.0132 | 2/4 | 7.0 | 6 | 0.5625 | -0.333 | FLIM |
| Distill 3 (615K) | protozoan | -0.0953 | 1/5 | 6.0 | 6 | 0.4375 | -0.429 | FLIM |
| Distill 4 (889K) | eggs | -0.0083 | 1/5 | 6.0 | 6 | 0.4375 | -0.429 | FLIM |
| Distill 4 (889K) | larvae | -0.0051 | 2/4 | 7.0 | 6 | 0.5625 | -0.333 | FLIM |
| Distill 4 (889K) | protozoan | -0.0615 | 1/5 | 6.0 | 6 | 0.4375 | -0.429 | FLIM |

## Leitura

Apos Holm, 2 de 7 comparacoes ficam abaixo de alpha = 0.05: Distill 1 (123K), LeJEPA (59.504).

Em todas elas o vencedor e o FLIM (59.504), que tem acuracia mediana maior que Distill 1 (123K) (0.441 de acuracia, r = -0.95, p Holm = 3.74e-04), LeJEPA (59.504) (0.319 de acuracia, r = -0.82, p Holm = 0.0063). A diferenca e negativa na convencao modelo - FLIM, ou seja, esses modelos perdem para o baseline.

Sem diferenca detectavel apos Holm: Distill 1 - FLIM init (123K), I-JEPA (632M), Distill 2 (402K), Distill 3 (615K), Distill 4 (889K). Isso e ausencia de evidencia de diferenca, nao evidencia de equivalencia; os IC95% bootstrap da mediana mostram a faixa de efeitos ainda compativel com os dados.

Limitacao de poder: com n = 18 pares o menor p bruto bilateral exato alcancavel e 7.6e-06, e numa familia de 7 testes o p mais baixo e confrontado com alpha/7 = 0.0071 sob Holm. Comparacoes com efeito mediano pequeno e sinal inconsistente entre as 18 celulas nao tem chance de sobreviver a correcao neste desenho; nas cinco comparacoes nao significativas o |mediana da diferenca| fica em 0.180 no maximo e o p bruto ja nao passaria nem sem correcao em 4 de 5 delas.

Sinal oposto entre mediana e media da diferenca em Distill 4 (889K): o efeito nao e homogeneo ao longo das fracoes de pre-treino, com as fracoes baixas puxando a media para o lado contrario da mediana. A leitura correta ai e que a diferenca depende do regime de pre-treino, nao que exista um vencedor unico.

Consistencia entre datasets na analise secundaria: Distill 1 (123K) (- nos 3); LeJEPA (59.504) (- nos 3); Distill 1 - FLIM init (123K) (- nos 3); I-JEPA (632M) (sinal misto); Distill 2 (402K) (- nos 3); Distill 3 (615K) (- nos 3); Distill 4 (889K) (- nos 3). As duas comparacoes que sobrevivem a Holm mantem o mesmo sinal nos tres datasets, logo o efeito agregado nao vem de um dataset isolado.

Numeros brutos em `wilcoxon_acc.csv`. Reproduzir com `python statistics/tools/wilcoxon_acc.py`.
