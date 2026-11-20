# Wilcoxon pareado de sinais em F1: FLIM (59.504) como baseline

Métrica: coluna `f1` de `artifacts/normalized/unified_svm_comparison.csv`. Baseline: `SVM_FLIM` (FLIM (59.504)). Família de 7 comparações um-contra-todos, alfa = 0.05.

Unidade pareada: célula `(dataset, fração de pré-treino)`, 3 datasets x 6 frações = **n = 18 pares** por comparação. Cada célula já é a média sobre 3 splits. O pareamento usa a chave `(dataset, fração)`, verificada antes de cada teste.

Teste: `scipy.stats.wilcoxon(baseline, modelo, alternative="two-sided", zero_method="wilcox")`, método `exact`. Correção de multiplicidade por Holm (principal) e Bonferroni (referência), via `statsmodels.stats.multitest.multipletests`. IC95% por bootstrap percentil das 18 células pareadas (10000 reamostragens, seed 42).

**Convenção de sinal**: diferença = F1(modelo) - F1(FLIM). Valor positivo significa que o modelo supera o FLIM; negativo, que o FLIM supera o modelo. O mesmo vale para `r` rank-biserial. A coluna *Melhor* nomeia o vencedor pelo sinal da mediana, independentemente de haver significância.

## Tabela principal (n = 18 pares por linha)

| Modelo | Mediana da dif. vs FLIM | IC95% bootstrap (mediana) | Melhor | W | p bruto | p Holm | p Bonferroni | r rank-biserial | Sig. Holm? |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| Distill 1 (123K) | -0.4955 | [-0.5978, -0.4407] | FLIM (59.504) | 4.0 | 5.34e-05 | 0.0004 | 0.0004 | -0.953 | sim |
| LeJEPA (59.504) | -0.4021 | [-0.4592, -0.1915] | FLIM (59.504) | 13.0 | 0.0007 | 0.0040 | 0.0047 | -0.848 | sim |
| Distill 1 - FLIM init (123K) | -0.1568 | [-0.1872, -0.0149] | FLIM (59.504) | 35.0 | 0.0268 | 0.1342 | 0.1879 | -0.591 | nao |
| I-JEPA (632M) | +0.0185 | [-0.0102, +0.0384] | I-JEPA (632M) | 51.0 | 0.1415 | 0.5661 | 0.9906 | +0.404 | nao |
| Distill 3 (615K) | -0.0523 | [-0.1053, -0.0139] | FLIM (59.504) | 51.0 | 0.1415 | 0.5661 | 0.9906 | -0.404 | nao |
| Distill 4 (889K) | -0.0225 | [-0.0582, -0.0068] | FLIM (59.504) | 51.0 | 0.1415 | 0.5661 | 0.9906 | -0.404 | nao |
| Distill 2 (402K) | -0.1029 | [-0.1899, -0.0465] | FLIM (59.504) | 51.0 | 0.1415 | 0.5661 | 0.9906 | -0.404 | nao |

Mediana de F1 do baseline FLIM (59.504) nas 18 células: 0.9044.

Detalhe dos sinais e da média das diferenças:

| Modelo | Vitórias do modelo | Vitórias do FLIM | Empates | W+ | W- | Média da dif. | IC95% bootstrap (média) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Distill 1 (123K) | 2 | 16 | 0 | 4.0 | 167.0 | -0.4603 | [-0.5678, -0.3400] |
| LeJEPA (59.504) | 3 | 15 | 0 | 13.0 | 158.0 | -0.3011 | [-0.4226, -0.1668] |
| Distill 1 - FLIM init (123K) | 3 | 15 | 0 | 35.0 | 136.0 | -0.0920 | [-0.1732, -0.0016] |
| I-JEPA (632M) | 13 | 5 | 0 | 120.0 | 51.0 | +0.0893 | [+0.0062, +0.1872] |
| Distill 3 (615K) | 3 | 15 | 0 | 51.0 | 120.0 | -0.0089 | [-0.0808, +0.0785] |
| Distill 4 (889K) | 3 | 15 | 0 | 51.0 | 120.0 | +0.0183 | [-0.0472, +0.0998] |
| Distill 2 (402K) | 3 | 15 | 0 | 51.0 | 120.0 | -0.0564 | [-0.1388, +0.0432] |

O teste exato depende apenas de W e de n, logo comparações com o mesmo W recebem o mesmo p bruto. As quatro linhas com W = 51 não são erro de cálculo: I-JEPA tem W- = 51 (13 vitórias contra 5) e Distill 2, 3 e 4 têm W+ = 51 (3 vitórias contra 15, justamente as três de maior magnitude). A magnitude do efeito difere entre elas, o p não.

## Tabela secundária por dataset (exploratória)

Wilcoxon dentro de cada dataset, n = 6 frações. Serve para checar consistência do efeito entre datasets. Com n = 6 o menor p bilateral alcançável pelo teste exato é 0.03125, o poder é mínimo e nenhuma correção de multiplicidade foi aplicada aqui: estes p não sustentam conclusão isolada.

| Modelo | Dataset | Mediana da dif. | Vitórias mod./FLIM | W | p bruto (não corrigido) | r rank-biserial |
|---|---|---:|---:|---:|---:|---:|
| Distill 1 (123K) | eggs | -0.6687 | 0/6 | 0.0 | 0.0312 | -1.000 |
| Distill 1 (123K) | larvae | -0.4407 | 1/5 | 1.0 | 0.0625 | -0.905 |
| Distill 1 (123K) | protozoan | -0.5663 | 1/5 | 1.0 | 0.0625 | -0.905 |
| LeJEPA (59.504) | eggs | -0.4359 | 1/5 | 1.0 | 0.0625 | -0.905 |
| LeJEPA (59.504) | larvae | -0.1915 | 1/5 | 1.0 | 0.0625 | -0.905 |
| LeJEPA (59.504) | protozoan | -0.5155 | 1/5 | 1.0 | 0.0625 | -0.905 |
| Distill 1 - FLIM init (123K) | eggs | -0.1694 | 1/5 | 1.0 | 0.0625 | -0.905 |
| Distill 1 - FLIM init (123K) | larvae | -0.0149 | 1/5 | 6.0 | 0.4375 | -0.429 |
| Distill 1 - FLIM init (123K) | protozoan | -0.1872 | 1/5 | 2.0 | 0.0938 | -0.810 |
| I-JEPA (632M) | eggs | +0.0384 | 6/0 | 0.0 | 0.0312 | +1.000 |
| I-JEPA (632M) | larvae | +0.0091 | 6/0 | 0.0 | 0.0312 | +1.000 |
| I-JEPA (632M) | protozoan | -0.0321 | 1/5 | 6.0 | 0.4375 | -0.429 |
| Distill 3 (615K) | eggs | -0.0547 | 1/5 | 6.0 | 0.4375 | -0.429 |
| Distill 3 (615K) | larvae | -0.0139 | 1/5 | 6.0 | 0.4375 | -0.429 |
| Distill 3 (615K) | protozoan | -0.1082 | 1/5 | 6.0 | 0.4375 | -0.429 |
| Distill 4 (889K) | eggs | -0.0209 | 1/5 | 6.0 | 0.4375 | -0.429 |
| Distill 4 (889K) | larvae | -0.0053 | 1/5 | 6.0 | 0.4375 | -0.429 |
| Distill 4 (889K) | protozoan | -0.0734 | 1/5 | 6.0 | 0.4375 | -0.429 |
| Distill 2 (402K) | eggs | -0.1532 | 1/5 | 6.0 | 0.4375 | -0.429 |
| Distill 2 (402K) | larvae | -0.0465 | 1/5 | 6.0 | 0.4375 | -0.429 |
| Distill 2 (402K) | protozoan | -0.1769 | 1/5 | 6.0 | 0.4375 | -0.429 |

## Leitura dos resultados

### O que este teste faz, em linguagem simples

Cada modelo enfrentou o FLIM (59.504) em **18 situações idênticas**: 3 conjuntos de imagens (eggs, larvae, protozoan) x 6 quantidades de pré-treino (1%, 5%, 25%, 50%, 75%, 100%). Como os dois rodam exatamente na mesma condição, dá para comparar um a um quem obteve o F1 maior — é o mesmo raciocínio de testar dois remédios no *mesmo* paciente em vez de comparar dois grupos de pacientes diferentes. Chamamos cada uma dessas situações de *célula*.

O teste de Wilcoxon olha o placar dessas 18 comparações e responde a uma única pergunta: **um resultado tão desequilibrado assim apareceria por puro acaso?** O valor `p` é essa resposta em número — quanto menor, mais difícil explicar o resultado por sorte. Ele não conta apenas quantas células cada lado venceu: ordena as 18 diferenças da menor para a maior e dá mais peso às vitórias mais folgadas. É isso que a palavra *posto* (rank) significa. O que ele **não** faz é somar as diferenças — uma vitória gigantesca não vale por dez, vale por ficar em primeiro lugar na fila.

A mesma pergunta foi feita 7 vezes (o FLIM (59.504) contra cada um dos 7 modelos). Fazer muitas perguntas aumenta a chance de uma delas dar "positivo" por acaso, como comprar sete bilhetes de loteria em vez de um. As correções de **Holm** e **Bonferroni** compensam isso exigindo evidência mais forte de cada comparação; Bonferroni é a mais rígida e entra só como checagem. Por isso a coluna que decide é `p Holm`, não `p bruto`.

**Regra do sinal**: número negativo = FLIM (59.504) melhor; positivo = o outro modelo melhor. A diferença está na escala do F1, que vai de 0 a 1, então -0.10 quer dizer 10 pontos de F1 abaixo do FLIM (59.504).

### O que o teste conseguiu demonstrar

Após a correção de Holm, 2 das 7 comparações diferem do FLIM (59.504) em F1 ao nível alfa = 0.05: Distill 1 (123K) (a favor de FLIM (59.504)), LeJEPA (59.504) (a favor de FLIM (59.504)). Sob Bonferroni, a régua mais rígida, sobrevivem 2. Em detalhe:

- **Distill 1 (123K)** — o FLIM (59.504) vence em 16 das 18 células, com diferença típica de -0.496 de F1 (mediana). F1 mediano: 0.904 do FLIM (59.504) contra 0.329. p Holm = 0.0004.
- **LeJEPA (59.504)** — o FLIM (59.504) vence em 15 das 18 células, com diferença típica de -0.402 de F1 (mediana). F1 mediano: 0.904 do FLIM (59.504) contra 0.450. p Holm = 0.0040.

Nesses dois casos a diferença não é marginal nem depende de detalhe estatístico: ela é grande, aparece na maioria das células e sobrevive à régua mais conservadora. Nenhum modelo supera o FLIM (59.504) com significância.

### O que ficou sem veredito (e o que isso *não* quer dizer)

As outras 5 comparações (Distill 1 - FLIM init (123K), I-JEPA (632M), Distill 3 (615K), Distill 4 (889K), Distill 2 (402K)) ficam **inconclusivas**. Inconclusivo não é empate. É como pesar duas malas numa balança de banheiro: se as duas marcam 20 kg, isso não prova que têm o mesmo peso, prova apenas que a balança não enxerga a diferença. Com 18 células e uma família de 7 perguntas, o teste só detecta diferenças grandes ou muito consistentes; as pequenas passam despercebidas.

O caso mais ilustrativo é **Distill 1 - FLIM init (123K)**: p bruto = 0.0268, abaixo de 0.05. Sozinha, essa comparação passaria. Dentro da família de 7 perguntas feitas ao mesmo tempo, não passa (p Holm = 0.1342). É exatamente o preço de comprar sete bilhetes.

Nessas linhas, a leitura útil é a mediana da diferença e o IC95%, não o `p`. Uma ressalva honesta: esse IC vem de reamostrar as 18 células (*bootstrap*) e não é o mesmo procedimento do teste. Ele pode excluir o zero enquanto o Wilcoxon não rejeita, e nesse caso a conclusão conservadora — a que vai para o artigo — é a do Wilcoxon corrigido.

### Onde o FLIM (59.504) perde: só quando quase não há pré-treino

Para 6 dos 7 modelos (Distill 1 (123K), LeJEPA (59.504), Distill 1 - FLIM init (123K), Distill 3 (615K), Distill 4 (889K), Distill 2 (402K)), *todas* as células em que o modelo bate o FLIM (59.504) estão na fração de 1% de pré-treino. De 5% em diante o FLIM (59.504) vence em todas as células, sem exceção. Ou seja: a vantagem dos concorrentes existe apenas no regime de dado escasso e desaparece assim que há pré-treino disponível. Essa é a leitura prática do experimento, e ela não depende de nenhum valor de `p`.

### Por que a mediana e a média discordam em alguns modelos

A **mediana** é o caso típico: enfileire as 18 diferenças e pegue a do meio. A **média** soma tudo e divide por 18, então poucos valores extremos a puxam — é o efeito de "a renda média do bar sobe quando entra um bilionário". Em Distill 3 (615K), Distill 4 (889K), Distill 2 (402K) acontece exatamente isso: 3 vitórias grandes (todas em 1%) contra 15 derrotas pequenas. A mediana é negativa (no caso típico o FLIM (59.504) é melhor), enquanto a média fica perto de zero ou positiva.

O Wilcoxon se comporta como a mediana, não como a média: ele vê 3 vitórias contra 15 derrotas e a folga dessas 3 compra, no máximo, os primeiros lugares da fila. Daí o veredito "inconclusivo" para esses modelos, apesar da média favorável.

### O resultado se repete nos três conjuntos de imagens?

A tabela secundária repete a mesma disputa dentro de cada dataset (6 células cada) para responder a uma pergunta simples: o efeito vale nos três domínios ou é peculiaridade de um só? Se um modelo ganha num dataset e perde noutro, a comparação agregada sobre as 18 células mistura as duas coisas e esconde essa dependência.

- **I-JEPA (632M)** troca de sinal entre datasets (eggs +0.038, larvae +0.009, protozoan -0.032). A mediana agregada (+0.019) esconde esse comportamento: a vantagem depende do domínio, não é geral.

Nos outros 6 modelos o sinal é o mesmo nos três conjuntos, o que indica efeito consistente e não artefato de um dataset isolado.

### Resumo

Pelo sinal da mediana, 1 modelo fica acima do FLIM (59.504) em F1 (I-JEPA (632M)) e 6 modelos ficam abaixo (Distill 1 (123K), LeJEPA (59.504), Distill 1 - FLIM init (123K), Distill 3 (615K), Distill 4 (889K), Distill 2 (402K)). Com correção de multiplicidade, 2 dessas diferenças se sustentam estatisticamente (todas a favor do FLIM (59.504)); as demais 5 ficam sem veredito. Onde os concorrentes ganham, ganham apenas na fração de 1% de pré-treino.

## Reprodução

```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM
LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib \
/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python \
    statistics/tools/wilcoxon_f1.py
```

Versões: scipy 1.17.1, statsmodels 0.14.6, numpy 1.26.4, pandas 3.0.3.

`scikit-posthocs` não foi usado: seu `posthoc_wilcoxon` é all-vs-all (28 comparações com 8 modelos), enquanto o desenho pedido é um-contra-todos (7 comparações contra o FLIM). A família de correção seria diferente e o ajuste de p ficaria mais conservador sem necessidade.
