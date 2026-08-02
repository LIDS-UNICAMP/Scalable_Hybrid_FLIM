# Wilcoxon pareado: init FLIM congelada vs init aleatoria, mesma cabeca de 123K

Comparacao controlada de **um fator**. Os dois bracos tem arquitetura e contagem de parametros **identicas** — 123.504 params = encoder FLIM `ch24_32_48` (59.504) + cabeca de projecao 1x1 BN2d 48->1280 (64.000, ou 1.08x o backbone). A unica diferenca e a origem dos pesos do encoder:

- **baseline** `SVM_Distill_1x1BN` (init `trunc_normal`) — Distill 1 - init aleatoria (123K), encoder treinavel;
- **modelo** `SVM_Distill_1x1BN_flim_frozen_eval_loss` — Distill 1 - FLIM init congelada (123K), pesos FLIM reais, **encoder congelado** (zero gradiente no encoder), checkpoint por best-loss.

Metricas: colunas `f1`, `kappa` e `acc` de `artifacts/normalized/unified_svm_comparison.csv` (embedding `proj (B,1280)` avaliado por SVM linear). Familia principal = essas 3 metricas, alfa = 0.05.

Unidade pareada: celula `(dataset, % de rotulos)`, 3 datasets x 6 fracoes = **n = 18 pares** por metrica. Cada celula ja e a media sobre 3 splits. O pareamento usa a chave `(dataset, fracao)`, verificada antes de cada teste.

> **Nota de nomenclatura:** a coluna `pretrained_pct` do CSV e o **% de dados rotulados usados para treinar o SVM linear**, nao uma fracao de pre-treino (o teacher e frozen) — ver secao 3 de `metrics_distillation/data_provenance.md`. Os relatorios `wilcoxon_f1.md`/`_acc.md`/`_kappa.md` chamam esse mesmo eixo de "pre-treino"; a nomenclatura correta e a usada aqui.

Teste: `scipy.stats.wilcoxon(baseline, modelo, alternative="two-sided", zero_method="wilcox")`, metodo `exact`. Correcao de multiplicidade por Holm (principal) e Bonferroni (referencia) sobre as 3 metricas, via `statsmodels.stats.multitest.multipletests`. IC95% por bootstrap percentil das 18 celulas pareadas (10.000 reamostragens, seed 42).

**Convencao de sinal**: diferenca = metrica(FLIM init congelada) - metrica(init aleatoria). Positivo significa que a init FLIM supera a aleatoria. O mesmo vale para `r` rank-biserial.

## Tabela principal (n = 18 pares por linha)

| Metrica | Celulas pro FLIM init | Mediana da dif. | IC95% bootstrap (mediana) | Media da dif. | W | p bruto | p Holm | p Bonferroni | r rank-biserial | Sig. Holm? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **F1** | 18/18 | +0.3692 | [+0.2900, +0.4334] | +0.3683 | 0.0 | 7.63e-06 | 2.29e-05 | 2.29e-05 | +1.000 | sim |
| **kappa** | 18/18 | +0.5251 | [+0.4302, +0.7019] | +0.5498 | 0.0 | 7.63e-06 | 2.29e-05 | 2.29e-05 | +1.000 | sim |
| **Acc** | 18/18 | +0.2550 | [+0.1766, +0.4003] | +0.2751 | 0.0 | 7.63e-06 | 2.29e-05 | 2.29e-05 | +1.000 | sim |

Medianas por braco nas 18 celulas:

| Metrica | Init aleatoria | FLIM init congelada |
|---|---:|---:|
| F1 | 0.3291 | 0.7234 |
| kappa | 0.0640 | 0.7350 |
| Acc | 0.4751 | 0.6835 |

## F1 medio por fracao de rotulos (descritiva, 3 datasets por linha)

| % de rotulos | Init aleatoria | FLIM init congelada | Delta | Celulas pro FLIM init |
|---:|---:|---:|---:|---:|
| 1% | 0.2330 | 0.4201 | +0.1871 | 3/3 |
| 5% | 0.2548 | 0.5522 | +0.2974 | 3/3 |
| 25% | 0.2886 | 0.7429 | +0.4543 | 3/3 |
| 50% | 0.3398 | 0.7940 | +0.4542 | 3/3 |
| 75% | 0.3952 | 0.8077 | +0.4125 | 3/3 |
| 100% | 0.4212 | 0.8255 | +0.4043 | 3/3 |

O teto da init aleatoria e **0.4212**, alcancado com **100% dos rotulos** — ela nunca ultrapassa esse valor em nenhum regime de supervisao. A init FLIM congelada ja supera esse teto a partir de **25%** dos rotulos (0.7429).

## Tabela secundaria por dataset (exploratoria, F1)

Wilcoxon dentro de cada dataset, n = 6 fracoes. Serve para checar consistencia do efeito entre dominios. Com n = 6 o menor p bilateral alcancavel pelo teste exato e 0.03125, o poder e minimo e nenhuma correcao de multiplicidade foi aplicada aqui: estes p nao sustentam conclusao isolada.

| Dataset | Mediana da dif. | Celulas pro FLIM init | W | p bruto (nao corrigido) | r rank-biserial |
|---|---:|---:|---:|---:|---:|
| eggs | +0.4376 | 6/6 | 0.0 | 0.0312 | +1.000 |
| larvae | +0.4247 | 6/6 | 0.0 | 0.0312 | +1.000 |
| protozoan | +0.3231 | 6/6 | 0.0 | 0.0312 | +1.000 |

## Robustez do braco FLIM (F1, 18 celulas, fora da familia de correcao)

Duas variantes do lado FLIM, sempre contra o mesmo baseline aleatorio. A primeira troca o criterio de selecao de checkpoint; a segunda libera o encoder, separando "init FLIM" de "encoder congelado".

| Braco FLIM | Mediana da dif. vs aleatoria | IC95% (mediana) | Celulas pro FLIM | W | p bruto | F1 medio |
|---|---:|---:|---:|---:|---:|---:|
| FLIM init congelada, ckpt knn+kappa (123K) | +0.3641 | [+0.2938, +0.4324] | 18/18 | 0.0 | 7.63e-06 | 0.6956 |
| FLIM init destreinada, encoder liberado (123K) | +0.2765 | [+0.2254, +0.3488] | 18/18 | 0.0 | 7.63e-06 | 0.5939 |
| Distill 1 - FLIM init congelada (123K) *(braco principal)* | +0.3692 | [+0.2900, +0.4334] | 18/18 | 0.0 | 7.63e-06 | 0.6904 |

## Leitura dos resultados

### O que este teste faz, em linguagem simples

Os dois modelos enfrentaram um ao outro em **18 situacoes identicas**: 3 conjuntos de imagens (eggs, larvae, protozoan) x 6 quantidades de rotulos (1%, 5%, 25%, 50%, 75%, 100%). Como rodam exatamente na mesma condicao e com exatamente o mesmo numero de parametros, da para comparar um a um quem obteve a metrica maior. Chamamos cada uma dessas situacoes de *celula*.

O teste de Wilcoxon olha o placar dessas 18 comparacoes e responde a uma unica pergunta: **um resultado tao desequilibrado assim apareceria por puro acaso?** Ele nao conta apenas quantas celulas cada lado venceu: ordena as 18 diferencas e da mais peso as vitorias mais folgadas. E isso que a palavra *posto* (rank) significa.

A mesma pergunta foi feita 3 vezes (uma por metrica). Fazer varias perguntas aumenta a chance de uma delas dar "positivo" por acaso, e as correcoes de **Holm** e **Bonferroni** compensam isso exigindo evidencia mais forte de cada comparacao. Por isso a coluna que decide e `p Holm`, nao `p bruto`.

### O que o teste demonstrou

Nas **3 de 3** metricas a init FLIM congelada supera a aleatoria ao nivel alfa = 0.05, e o resultado e o **maximo que o desenho permite**: **18/18 celulas** a favor do FLIM em todas as tres metricas, **W = 0** (nenhuma celula discordante) e `r` rank-biserial = +1.000. O `p` bruto de 7.63e-06 e o piso do teste exato bilateral com n = 18; nao existe evidencia mais forte alcancavel com esse n.

- **F1** — mediana da diferenca **+0.369** (IC95% [+0.290, +0.433]); mediana 0.329 da init aleatoria contra 0.723 da init FLIM. p Holm = 2.29e-05.
- **kappa** — mediana da diferenca **+0.525** (IC95% [+0.430, +0.702]); mediana 0.064 da init aleatoria contra 0.735 da init FLIM. p Holm = 2.29e-05.
- **Acc** — mediana da diferenca **+0.255** (IC95% [+0.177, +0.400]); mediana 0.475 da init aleatoria contra 0.683 da init FLIM. p Holm = 2.29e-05.

Diferente das linhas inconclusivas de `wilcoxon_f1.md`, aqui a conclusao nao depende de detalhe estatistico: o efeito e grande, aparece em **todas** as celulas e em **todos** os datasets (6/6 em eggs, larvae e protozoan), e sobrevive a Bonferroni. Se esta comparacao fosse embutida na familia de 7 hipoteses daquele relatorio, o p corrigido seria 7 x 7.63e-06 = 5.34e-05 — ainda significativo.

### Congelar nao e concessao: e melhor

A variante com init FLIM mas **encoder liberado** tambem bate a init aleatoria (18/18 celulas, p bruto 7.63e-06), mas fica em F1 medio **0.5939** contra **0.6904** da versao congelada. Ou seja: destreinar o encoder inicializado por FLIM **piora** o embedding. A contribuicao nao e "FLIM como ponto de partida para o gradiente", e "FLIM como backbone pronto" — o gradiente fica so na cabeca.

A conclusao tambem nao depende do criterio de selecao de checkpoint: com o ckpt escolhido por `val/knn_kappa` em vez de best-loss, o resultado e praticamente o mesmo (18/18 celulas, mediana +0.3641, F1 medio 0.6956).

### Onde o ganho aparece

O ganho existe em **todos** os regimes de supervisao, mas cresce com a quantidade de rotulos: +0.19 de F1 a 1%, +0.30 a 5%, +0.45 a 25% e +0.40 a 100%. Isso importa para a redacao: os valores frequentemente citados de **0,42 -> 0,83** sao os de **100% dos rotulos**, nao de 5% — a 5% os numeros sao 0.25 -> 0.55. A leitura correta de 0,42 e "o teto da init aleatoria, atingido so com todos os rotulos".

### O que este teste NAO diz

- Nao compara a init FLIM com o **FLIM puro** nem com o teacher: para isso ver `wilcoxon_f1.md` (baseline FLIM), onde `Distill 1 - FLIM init (123K)` fica *abaixo* do FLIM de 59.504 params (mediana -0.157, inconclusivo apos Holm).
- Nao estabelece um limiar de capacidade. Pela descritiva de `unified_svm_comparison.csv`, o menor braco de init aleatoria que escapa do colapso e o **Distill 2** (402.608 params, cabeca 5,77x o backbone, F1 medio 0,86 a 100%), nao o Distill 3/4. O que a init FLIM entrega com 123K, a init aleatoria so alcanca com 402K-615K.
- Nao mede custo de inferencia: params, FLOPs, latencia e memoria estao em `compute_cost.md`.
- **Nao diz nada sobre os outros pares de modelos.** Para FLIM vs I-JEPA, Distill 3 vs Distill 4 e os demais 28 pares — e, principalmente, para separar "nao detectei diferenca" de "demonstrei que a diferenca e pequena" (teste de equivalencia TOST) — ver `wilcoxon_equivalence.md`.

### De onde vem o p, e o que ele NAO mede

O `p` de 7.63e-06 nao e um numero opaco: com n = 18 o teste exato enumera as 2^18 = 262.144 atribuicoes de sinal possiveis sob a nula, e apenas **2** delas dao W = 0 ("todas positivas" e "todas negativas"). Logo p = 2 / 2^18 = 7.629395e-06. Verificado por enumeracao exaustiva, nao so pelo scipy. Como consequencia, esse e o **piso** do teste: nenhum resultado com n = 18 pode dar p menor. O teste do sinal (binomial) sobre as mesmas 18 celulas da exatamente o mesmo valor, o que era esperado com 18/18.

O que isso implica: **o `p` mede apenas a consistencia da direcao**, nao a magnitude. Dezoito diferencas de 1e-9 no mesmo sentido dariam o mesmo p. A magnitude tem de ser lida na mediana (+0.369 de F1, IC95% [+0.290, +0.433]) e no fato de que a **menor** das 18 celulas ja e +0.135 de F1 (eggs a 1%) — nao e um caso de 18 vitorias minusculas.

## Ressalvas

- **As 18 celulas nao sao independentes.** Dentro de um dataset, os 6 subconjuntos de treino sao estritamente encaixados (1% ⊂ 5% ⊂ 25% ⊂ 50% ⊂ 75% ⊂ 100%, contencao de 100% verificada nos JSONs de `data/to_mateus/splits_incremental/`) e o conjunto de teste e **identico** nas 6 fracoes. O Wilcoxon supoe pares independentes, portanto n = 18 e otimista e o `p` deve ser lido como evidencia de direcao consistente, nao como probabilidade calibrada. No extremo conservador — colapsando cada dataset na mediana das 6 fracoes, n = 3 — a direcao se mantem (3/3 datasets a favor da init FLIM, mediana +0.4356), mas o p = 0.2500 e o piso alcancavel com n = 3, logo nada pode ser declarado significativo nesse recorte. Os tres splits, por outro lado, **sao** genuinamente diferentes (~51% de sobreposicao entre os conjuntos de treino), o que valida a media sobre splits dentro de cada celula.
- Os `.ckpt` das runs Distill 1/2/3/4 nao estao mais no disco; a contagem de parametros vem da topologia reinstanciada (ver secao 5 de `README.md` deste diretorio). Para **protozoan** o encoder real usa `conv2` com 30 canais, nao 32, logo o total e 3.602 params menor que os 123.504 de eggs/larvae. Isso nao afeta o teste (os dois bracos tem a mesma topologia), mas afeta qualquer numero de params citado por dataset.
- O IC95% vem de reamostrar as 18 celulas (*bootstrap*) e nao e o mesmo procedimento do teste; aqui os dois concordam (IC exclui zero e o Wilcoxon rejeita), mas em caso de conflito a conclusao conservadora e a do Wilcoxon corrigido.
- Ausencia de significancia com n = 18 seria inconclusividade, nao equivalencia. Nao e o caso aqui.

## Reproducao

```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM
LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib \
/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python \
    statistics/tools/wilcoxon_flim_init.py
```

Versoes: scipy 1.17.1, statsmodels 0.14.6, numpy 1.26.4, pandas 3.0.3.
