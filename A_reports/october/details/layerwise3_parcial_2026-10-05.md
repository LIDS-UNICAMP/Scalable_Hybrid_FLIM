# layerwise3: resultado final (08/10/2026)

**Resumo em uma frase:** crescer o encoder camada por camada ganha do SPiFiL e dos
Distill quase sempre. A camada que vence muda com a quantidade de dados: com poucos
dados ganha a L1, com muitos ganha a L3. O I-JEPA continua na frente.

> Este arquivo substitui o relatório parcial de 05/10, que olhava o κ de **validação**
> no meio do treino, ainda com a sonda SVM ligada. Aqui os números são de **teste**,
> com o treino completo e o SVM rodando só no final.

---

## Como o modelo foi treinado

A ideia é **empilhar blocos**: o SPiFiL corta cada bloco novo, os de baixo ficam colados, e só
o bloco novo é ajustado.

**Para cada camada k = 1, 2, 3:**

1. **Cortar a camada com o SPiFiL, sem gradiente.** O SPiFiL escolhe uma imagem de treino por
   classe, divide cada uma em superpixels (pedacinhos parecidos de cor) e recorta os filtros
   em volta deles. A L1 sai da imagem LAB; a L2 e a L3 saem das features que as camadas
   anteriores, já treinadas, produzem.
   - Config do artigo: **16 / 32 / 48 filtros**, kernel 3×3, max-pool 3 com stride 2 (a imagem
     encolhe 200 → 99 → 49 → 24).
   - O SPiFiL arredonda os filtros pelo número de classes, então sobram 9 / 27 / 45 no eggs,
     16 / 32 no larvae e 14 / 28 / 42 no protozoan.
2. **Congelar as camadas antigas.** L1 … L(k-1) não recebem gradiente. O loop confere isso
   comparando os pesos antes e depois; se mudarem, a célula para com erro.
3. **Encaixar um decoder novo e treinar por reconstrução.** O decoder espelha o encoder e tenta
   recriar a imagem de entrada (perda BCE sobre o LAB, sem rótulo nenhum). Só a camada nova e o
   decoder aprendem.
   - Teto de 1000 épocas, com parada após 50 épocas sem melhora na perda de reconstrução de
     validação. O checkpoint guardado é o de menor perda.
   - O SVM fica **desligado durante o treino** (`evaluate.SVM: false`), porque era ele que
     travava a grade.
4. **Jogar o decoder fora.** Ele é só um apoio para a camada nova poder ser treinada sem rótulo.

**No fim**, cada encoder (com 1, 2 ou 3 camadas) é congelado e avaliado pelo **SVM oficial** do
repositório, treinado no conjunto de treino e medido no de teste, com o mapa da última camada
achatado (`flatten`).

Tudo é configurado por YAML, sem flag na linha de comando:
`experiments/autoencoder/growth_layerwise3.yaml` (treino) e `eval_svm_layerwise3.yaml` (SVM),
os dois rodados com `python -m experiments.ray.launch <yaml>`.

## 1. O que foi rodado

- **Grade:** 3 datasets × 3 splits × 6 porcentagens = **54 células**, todas completas.
- **Camadas:** cada célula cresce L1 → L2 → L3. O SPiFiL corta cada camada nova (config do
  artigo: 16/32/48 filtros, kernel 3, max-pool 3/2). As camadas antigas ficam congeladas,
  só a nova aprende, e um decoder novo serve de apoio (reconstrução, sem rótulo).
- **Larvae só tem L1 e L2.** O SPiFiL recusa a L3 por ter poucas amostras (N=196 contra
  D=288). Isso é esperado, não falha.
- **Avaliação:** o SVM oficial sobre o encoder congelado, no conjunto de teste. São
  **144 avaliações**, todas ok. Cada número abaixo é a média dos 3 splits.
- **Treino sem SVM** (`evaluate.SVM: false`). O melhor checkpoint e a parada passam a
  olhar a perda de reconstrução.
- **Uma falha:** a célula `eggs s3 25%` caiu por timeout do W&B e foi repetida com sucesso.

Arquivos:

| o quê | onde |
|---|---|
| CSV por célula | `results/svm_layerwise3.csv` |
| média ± desvio | `artifacts/normalized/unified_svm_comparison.csv` (`SVM_Layerwise3_L*`) |
| gráficos | `artifacts/plots/layerwise3/merge_plots/{kappa,acc,f1}.png` e `.../comparison/` |

## 2. κ de teste por camada, ao lado das referências

O negrito marca a melhor camada do layerwise3 em cada linha.

| dataset | % | L1 | L2 | L3 | SPiFiL | Distill 3 | Distill 4 | I-JEPA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eggs | 1   | **0,436** | 0,397 | 0,336 | 0,128 | 0,268 | 0,264 | 0,661 |
| eggs | 5   | **0,734** | 0,688 | 0,670 | 0,585 | 0,596 | 0,668 | 0,864 |
| eggs | 25  | 0,856 | **0,858** | 0,856 | 0,805 | 0,758 | 0,827 | 0,917 |
| eggs | 50  | 0,893 | 0,902 | **0,905** | 0,842 | 0,833 | 0,875 | 0,940 |
| eggs | 75  | 0,905 | 0,915 | **0,922** | 0,870 | 0,870 | 0,902 | 0,945 |
| eggs | 100 | 0,913 | 0,924 | **0,934** | 0,885 | 0,900 | 0,912 | 0,957 |
| larvae | 1   | 0,204 | **0,337** | — | 0,028 | 0,635 | 0,613 | 0,822 |
| larvae | 5   | 0,543 | **0,605** | — | 0,571 | 0,702 | 0,766 | 0,881 |
| larvae | 25  | 0,752 | **0,778** | — | 0,805 | 0,853 | 0,870 | 0,919 |
| larvae | 50  | 0,788 | **0,812** | — | 0,821 | 0,880 | 0,914 | 0,943 |
| larvae | 75  | 0,802 | **0,839** | — | 0,835 | 0,913 | 0,924 | 0,941 |
| larvae | 100 | 0,830 | **0,870** | — | 0,868 | 0,939 | 0,927 | 0,950 |
| protozoan | 1   | 0,471 | **0,475** | 0,457 | 0,145 | 0,345 | 0,391 | 0,586 |
| protozoan | 5   | 0,618 | **0,640** | 0,634 | 0,623 | 0,504 | 0,558 | 0,725 |
| protozoan | 25  | 0,713 | 0,775 | **0,810** | 0,775 | 0,707 | 0,746 | 0,831 |
| protozoan | 50  | 0,756 | 0,816 | **0,848** | 0,816 | 0,787 | 0,806 | 0,864 |
| protozoan | 75  | 0,776 | 0,832 | **0,863** | 0,837 | 0,811 | 0,824 | 0,882 |
| protozoan | 100 | 0,790 | 0,853 | **0,875** | 0,847 | 0,833 | 0,838 | 0,892 |

"SPiFiL" é a curva `SVM_FLIM` dos gráficos (o encoder SPiFiL/FLIM de 3 camadas, sem
treino). Ela vem dos relatórios do Felipe, um pipeline diferente; por isso a comparação
com ela é boa, mas não idêntica.

## 3. O que isso diz

**Empilhar camadas ajuda quando há dados, e atrapalha quando não há.** É como um prédio:
com pouco material, o andar de cima fica bambo.

- **Eggs:** com 1% e 5%, a L1 é a melhor e cada camada a mais perde κ (de 0,436 para
  0,336 em 1%). De 50% em diante a ordem inverte, e a L3 vira a melhor (0,934 em 100%).
- **Protozoan:** a L3 é a melhor a partir de 25%. Em 1% e 5% as três camadas empatam.
- **Larvae:** a L2 ganha da L1 em todas as porcentagens.

**Contra o SPiFiL original, que é a pergunta do experimento:**

- **Ganha muito com pouco dado:** em 1%, 0,436 contra 0,128 em eggs e 0,475 contra 0,145
  em protozoan. É o triplo.
- **Ganha um pouco no resto, em eggs e protozoan:** +0,03 a +0,06 em eggs; +0,02 a +0,03 em
  protozoan de 25% para cima.
- **Larvae empata ou perde um pouco** de 25% para cima (0,778 contra 0,805 em 25%), porque
  fica preso na L2.

**Contra os outros:**

- **Distill 3 e 4:** o layerwise3 ganha em eggs e protozoan em todas as porcentagens. No
  larvae perde para os dois.
- **I-JEPA** (632 milhões de parâmetros) continua na frente em tudo. Mas a distância cai:
  em eggs 100% fica em 0,023, e em protozoan 100% em 0,017.

## 4. Comparação com o crescimento antigo (`grid4`, mesmo SVM, mesmo `flatten`)

O `grid4` é o FLIM de 3 camadas com uma camada SPiFiL por cima, e só existe para 5% e 50%.

| % | dataset | layerwise3 (melhor camada) | grid4 FLIM congelado | grid4 melhor estágio |
|---|---|---:|---:|---:|
| 5  | eggs      | **0,734** (L1) | 0,617 | 0,675 |
| 5  | larvae    | 0,605 (L2) | 0,565 | **0,668** |
| 5  | protozoan | 0,640 (L2) | 0,617 | **0,675** |
| 50 | eggs      | 0,905 (L3) | 0,852 | **0,906** |
| 50 | larvae    | 0,812 (L2) | 0,812 | **0,869** |
| 50 | protozoan | 0,848 (L3) | 0,808 | **0,859** |

O layerwise3 ganha do FLIM congelado em quase tudo. Mas o melhor estágio do `grid4`, que
destrava e treina o encoder inteiro, ainda é um pouco melhor, de +0,01 a +0,06. A exceção
é eggs a 5%, onde o layerwise3 ganha por 0,06.

## 5. Ressalvas

- **Tamanho do embedding.** L1 e L2 entregam mapas grandes para o SVM (9×99×99 e 27×49×49
  no eggs), contra 45×24×24 na L3 e no `grid4`. Parte da vantagem da L1 com pouco dado
  pode vir só de "mais números". **A L3 é a comparação justa** com o `grid4`, porque tem
  a mesma resolução.
- **O treino vai até o teto de 1000 épocas.** Sem o SVM, a parada olha a perda de
  reconstrução, e ela não para de cair (a L3 de eggs 1% parou na época 998). Ou seja, o
  treino escolhe pela reconstrução, não pela classificação.
- **Larvae varia muito** em 1% e 5%: o desvio entre splits chega a 0,19.

## 6. Próximo passo

O **layerwise5** (L4 e L5 por cima) responde se continuar empilhando ainda ajuda com muitos
dados. A avaliação dele está rodando; faltam 2 células para treinar (`eggs s2 75%` e
`eggs s3 5%`).
