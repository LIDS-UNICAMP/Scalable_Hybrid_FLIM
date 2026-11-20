# Restrição ≤ 5 M parâmetros (11/11)

**Resposta curta:** com o que já foi rodado, **nenhum modelo de até 5 M parâmetros empata com a receita de 8,2 M**. O melhor caminho que já temos é cortar a receita na camada 4 (4,94 M), e isso custa em média **−0,023 de κ** (cat e cat_tta). Para chegar perto de 1 M, a grade nova aplica 9 leis de construção da literatura (§3) com orçamentos fixos de ~1 M, ~2 M e ~4,5 M. Uma pista dos dados antigos: o kernel 3 empata com o kernel 5 e usa só 9/25 dos parâmetros. A grade está pronta para o jaci (§4).

Contagem de parâmetros = pesos das convoluções + bias. O whitening é absorvido nos pesos, então não soma nada. Para k5 e 256 canais: a L1 tem 19.456 parâmetros e cada camada de 256→256 tem 1.638.656.

| arquitetura | params |
|---|---|
| receita 6×256 k5 | 8.212.736 |
| 5×256 k5 | 6.574.080 |
| 4×256 k5 (receita cortada na L4) | 4.935.424 |
| 6×[256,256,256,192,192,128] k5 (exemplo do pedido) | 6.062.080, **acima de 5 M** |
| 5×256 k5-5-5-3-3 (encdeep g0) | 4.477.088 |
| 6×256 k3 | 2.957.568 |
| 4×256 k3 | 1.780.928 |
| 6×[256,256,128,128,128,128] k3 | 1.334.784 |
| 3×256 k3 | 1.190.848 |
| 6×[256,256,128,128,64,64] k3 | 1.150.592 |
| 4×192 k3 | 1.001.280 |
| 6×128 k3 | 741.504 |

## 1. Receita cortada por profundidade (dados existentes, TTA, 6 bases, 50 %, 3 splits)

Os filtros crescem uma camada por vez. Por isso a linha `layer=k` do CSV da receita **é** o modelo de k camadas: a camada 4 não depende de existirem as camadas 5 e 6. Braço lido: fpsw nas bases de objeto, fps em aid e remote_sensing. Os CSVs ficam em `/exps/moliveira/results/svm_disc_md_receita_tta_*` (aid, remote_sensing, flowers) e `/exps/moliveira/recebido_jaci/exps/results/svm_disc_md_receita_tta_*` (Eggs, Protozoan, Larvae). A coluna random_z L4 é o controle aleatório puro do mesmo tamanho (4,94 M).

### κ cat (média de 3 splits; Δ contra a receita 8,21 M = camada 6)

| base | L6 8,21 M | L5 6,57 M | L4 4,94 M | L3 3,30 M | L2 1,66 M | random_z L4 | CSV (split 1, L4) |
|---|---|---|---|---|---|---|---|
| Eggs | 0.866 | 0.862 (-0.004) | 0.835 (-0.031) | 0.807 (-0.059) | 0.732 (-0.133) | 0.613 | Eggs_sh1.csv:68 |
| Protozoan | 0.834 | 0.825 (-0.009) | 0.796 (-0.038) | 0.773 (-0.061) | 0.750 (-0.084) | 0.531 | Protozoan_sh1.csv:68 |
| Larvae | 0.855 | 0.854 (-0.001) | 0.841 (-0.015) | 0.771 (-0.084) | 0.684 (-0.171) | 0.408 | Larvae_sh1.csv:68 |
| aid | 0.771 | 0.767 (-0.004) | 0.749 (-0.022) | 0.735 (-0.036) | 0.698 (-0.073) | 0.647 | aid_sh0.csv:68 |
| remote_sensing | 0.492 | 0.484 (-0.008) | 0.502 (+0.010) | 0.502 (+0.011) | 0.480 (-0.011) | 0.508 | remote_sensing_sh0.csv:68 |
| flowers | 0.655 | 0.638 (-0.016) | 0.614 (-0.041) | 0.621 (-0.034) | 0.605 (-0.050) | 0.540 | flowers_sh1.csv:68 |
| **Δ médio** | — | -0.007 | -0.023 | -0.044 | -0.087 | | |

### κ cat_tta (média de 3 splits; Δ contra a receita 8,21 M = camada 6)

| base | L6 8,21 M | L5 6,57 M | L4 4,94 M | L3 3,30 M | L2 1,66 M | random_z L4 | CSV (split 1, L4) |
|---|---|---|---|---|---|---|---|
| Eggs | 0.919 | 0.915 (-0.005) | 0.901 (-0.018) | 0.867 (-0.053) | 0.809 (-0.110) | 0.735 | Eggs_sh1.csv:72 |
| Protozoan | 0.887 | 0.879 (-0.008) | 0.859 (-0.029) | 0.833 (-0.055) | 0.789 (-0.098) | 0.657 | Protozoan_sh1.csv:72 |
| Larvae | 0.878 | 0.841 (-0.038) | 0.840 (-0.038) | 0.838 (-0.041) | 0.792 (-0.087) | 0.630 | Larvae_sh1.csv:72 |
| aid | 0.844 | 0.836 (-0.009) | 0.818 (-0.027) | 0.786 (-0.058) | 0.754 (-0.091) | 0.756 | aid_sh0.csv:72 |
| remote_sensing | 0.539 | 0.534 (-0.005) | 0.530 (-0.009) | 0.518 (-0.021) | 0.517 (-0.022) | 0.498 | remote_sensing_sh0.csv:72 |
| flowers | 0.660 | 0.651 (-0.009) | 0.640 (-0.020) | 0.618 (-0.042) | 0.600 (-0.060) | 0.598 | flowers_sh1.csv:72 |
| **Δ médio** | — | -0.012 | -0.023 | -0.045 | -0.078 | | |

O que a tabela mostra: a perda começa já na L4 (−0,02 a −0,04 em 5 das 6 bases) e dobra na L3. A L5 quase empata (−0,007), mas tem 6,57 M. Em remote_sensing a profundidade não muda nada. A L4 continua bem acima do random_z do mesmo tamanho em todas as bases, menos remote_sensing.

## 2. Outras fontes existentes (só para orientar, sem TTA e com protocolo diferente)

**Leis (`svm_disc_md_leis_p1*`, split 1, sorteios 42 e 43, sem TTA)** comparadas com a âncora sol_wsal 6×256 k5 (split 1, sorteio 42). A âncora vem de `recebido_jaci/exps/results/svm_disc_md_sol_wsal_<base>_s0200_split1_pct50_sh1.csv:198/242`: aid 0,786; flowers 0,656; Protozoan 0,829; remote_sensing 0,432. Melhores arquiteturas ≤ 5 M (κ cat, média dos 2 sorteios):

| arquitetura (k5) | params | aid | flowers | Protozoan | remote_sensing | CSV (sorteio 42) |
|---|---|---|---|---|---|---|
| 256-256-128-128-64-64 | 3,19 M | 0,754 (−0,032) | 0,604 (−0,052) | 0,796 (−0,034) | 0,46–0,48 | `svm_disc_md_leis_p1_aid_..._leidepth6_leiA_r0.71_w0256.csv:44` |
| 512-256-128-128-64-32 | 4,80 M | 0,752 (−0,034) | 0,624 (−0,032) | 0,787 (−0,043) | 0,486 | `..._leidepth6_flin_leiB_w0512_wL32.csv:44` |
| 256-256-256-256 (L4 de lei C) | 4,94 M | 0,755 | 0,609 | — | 0,50 | `..._leiC.csv:36` |

Conclusão das leis: afinar o fim da rede não salva mais que cortar a profundidade. As duas saídas perdem cerca de 0,03, e a troca de sorteio sozinha mexe 0,01–0,04.

**encdeep g0~z0 (256×5, kernels 5-5-5-3-3, 4,48 M, split 1, sem TTA)** fica em `/exps/moliveira/results_sol/encdeep/`: aid 0,770 (−0,016 contra a âncora), flowers 0,641 (−0,015), remote_sensing 0,451. É o melhor ≤ 5 M que já existe, mas vem de 2 bases e 1 split.

**Kernel 3 vs 5 (`recebido_jaci/exps/results/svm_disc_md_spf_r4_*`, protocolo antigo r4: 4 camadas, pool 2, sem whitening entre camadas, 3 splits, largura 256, κ cat L4, fps@sift):**

| base | k3 | k5 | k7 |
|---|---|---|---|
| aid | 0,716 | 0,711 | 0,686 |
| flowers | 0,587 | 0,581 | 0,587 |
| Protozoan_subgroup | 0,818 | 0,808 | 0,806 |
| helminth_eggs | 0,935 | 0,943 | 0,934 |
| helminth_larvae | 0,960 | 0,955 | 0,925 |

O k3 empata com o k5 (a diferença fica em ±0,01). Esse é o único caminho que leva o modelo para perto de 1 M sem cortar profundidade nem largura.

## 3. Leis da literatura

Ideia em uma frase: em vez de chutar arquiteturas, cada artigo abaixo dá uma regra de "como encolher ou crescer uma rede". Aplicamos cada regra à receita até ela caber em ~1 M, ~2 M e ~4,5 M de parâmetros.

| # | lei | autor, ano, veículo | o que diz | URL |
|---|---|---|---|---|
| L0 | multiplicador de largura α | Howard et al., 2017, arXiv (MobileNets) | encolha todas as camadas pelo mesmo fator α; o custo cai com α² | https://arxiv.org/abs/1704.04861 |
| L1 | pilha de 3×3 | Simonyan & Zisserman, 2015, ICLR (VGG) | duas 3×3 enxergam o mesmo que uma 5×5 com 18 em vez de 25 pesos | https://arxiv.org/abs/1409.1556 |
| L2 | largo e raso | Zagoruyko & Komodakis, 2016, BMVC (Wide ResNet) | poucas camadas largas igualam muitas finas | https://arxiv.org/abs/1605.07146 |
| L3 | dobrar canais quando a imagem encolhe | He et al., 2016, CVPR (ResNet) | cada pool 2× vem com 2× canais, mantendo o custo por camada | https://arxiv.org/abs/1512.03385 |
| L4 | larguras lineares quantizadas | Radosavovic et al., 2020, CVPR (RegNet) | u_j = w0 + wa·j, quantizado como w0·wm^round(s_j); a forma u_j = wa(j+1) foi ainda melhor | https://arxiv.org/abs/2003.13678 |
| L5 | razão de estágios 1:1:3:1 | Liu et al., 2022, CVPR (ConvNeXt) | o 3º estágio leva 3× mais blocos; larguras C, 2C, 4C, 8C | https://arxiv.org/abs/2201.03545 |
| L6 | campo receptivo cobre a imagem | Araujo, Norris & Sim, 2019, Distill; Luo et al., 2016, NeurIPS | o campo receptivo teórico deve cobrir o objeto ou cena; o efetivo é bem menor (gaussiano) | https://distill.pub/2019/computing-receptive-fields/ · https://arxiv.org/abs/1701.04128 |
| L7 | escala composta | Tan & Le, 2019, ICML (EfficientNet) | profundidade α^φ, largura β^φ, resolução γ^φ, com α·β²·γ² ≈ 2 (α=1,2; β=1,1; γ=1,15) | https://arxiv.org/abs/1905.11946 |
| L8 | escala rápida dWr | Dollár, Singh & Girshick, 2021, CVPR | escale sobretudo a largura: e_d = e_r = (1−α)/2 e e_w = α, com α = 0,8 | https://arxiv.org/abs/2103.06877 |

Contexto que não vira arquitetura própria: Bello et al. (2021, NeurIPS, *Revisiting ResNets*, https://arxiv.org/abs/2103.07579) recomendam escalar a largura, e não a profundidade, quando há risco de sobreajuste, como no nosso caso. É a mesma direção da L8.

### Tradução para o SPiFiL

O quadro fixo da receita é este: 6 camadas, pools 2-2-2-2-1-1 (mapas de 99, 49, 24, 11, 11 e 11 px a 200 px), whitening entre camadas, GeM + TTA, e leitura cat abaixo de 2000 dimensões. As larguras são arredondadas a múltiplos de 8, como no RegNet. As contas seguem `eval/spf_grade.py:conv_cost`:

- **Parâmetros:** Σ c_in·c_out·k² (o bias acrescenta menos de 0,1 %).
- **Campo receptivo (RF):** cada camada soma (k−1)·salto, e cada pool soma 2·salto e dobra o salto.

Fórmulas:

- **L0 e L1:** uniforme w, com P ≈ 5·k²·w².
- **L2:** 4 camadas, com P ≈ 27·w² (k3).
- **L3:** larguras w·(1,2,4,8,8,8), P ≈ 1530·w².
- **L4:** wa·(1,2,4,4,4,8) (u_j = wa(j+1) quantizado com wm = 2), P ≈ 666·wa².
- **L5:** as mesmas larguras, mas com pools 2-2-1-1-2-1, para pôr 3 camadas no 3º estágio.
- **L6:** k3 em L1–L3 e k5 em L4–L6; é o jeito mais barato de levar o RF a pelo menos 200 px, porque o kernel grande fica onde o salto é 8–16.
- **L7:** a receita (6×256, k5, 200 px) escalada por φ < 0.
- **L8:** a receita escalada por um fator s de custo: d = 6·s^0,1, w = 256·s^0,4, r = 200·s^0,05.

| lei | ~1 M | ~2 M | ~4,5 M |
|---|---|---|---|
| L0 ref. uniforme k5 | 6×88 · 0,97 M · RF 219 | 6×128 · 2,06 M · RF 219 | 6×192 · 4,62 M · RF 219 |
| L1 VGG k3 | 6×152 · 1,04 M · RF 125 | 6×208 · 1,95 M · RF 125 | 6×312 · 4,39 M · RF 125 |
| L2 WRN 4 camadas k3 | 4×192 · 1,00 M · RF 61 | 4×272 · 2,00 M · RF 61 | 4×408 · 4,51 M · RF 61 |
| L3 ResNet k3 | 24-48-104-208-208-208 · 1,03 M · RF 125 | 40-72-144-288-288-288 · 1,99 M | 56-112-216-432-432-432 · 4,47 M |
| L4 RegNet k3 | 40-80-152-152-152-312 · 0,98 M · RF 125 | 56-112-216-216-216-440 · 1,97 M | 80-168-328-328-328-656 · 4,49 M |
| L5 ConvNeXt k3 | as larguras de L4 · pools 2-2-1-1-2-1 · RF 61 | idem · 1,97 M | idem · 4,49 M |
| L6 RF k3-3-3-5-5-5 | 6×104 · 1,01 M · RF 205 | 6×144 · 1,93 M · RF 205 | 6×216 · 4,34 M · RF 205 |
| L7 EfficientNet k5 | φ = −6: 3×160 a 104 px · 1,29 M · RF 43 | φ = −3,7: 3×192 a 128 px · 1,86 M | φ = −1,6: 4×224 a 160 px · 3,78 M · RF 91 |
| L8 dWr k5 | s = 0,10: 5×96 a 176 px · 0,93 M · RF 155 | s = 0,25: 5×144 a 184 px · 2,09 M | s = 0,55: 6×192 a 192 px ≈ L0, não roda |

O cálculo é reproduzível pelo script de derivação, com a mesma conta de `conv_cost`. São 26 arquiteturas, 2 a 3 por lei em cada orçamento. Cada uma roda com o seu random_z puro, da mesma arquitetura.

Leituras a priori, antes de rodar:

- **Campo receptivo:** só L0, L6 e L8 têm RF ≥ 150 px. L2 e L5 enxergam apenas 61 px de 200. Se o RF importa, L2 e L5 devem perder nas bases de cena.
- **Achados anteriores:** em k5, 256 canais vencia larguras menores em 12 de 12 bases, e 3–4 camadas bastavam com pouco dado. A L2 (WRN) testa exatamente isso, com o orçamento fixo.

## 4. Grade nova (YAMLs para o jaci)

- **Bases (só 50 %, 3 splits):** fiquem de fora as recortadas de fundo preto (helminth_eggs, helminth_larvae, protozoan_cysts, com 96–98 % de pixels pretos).
- **Os *_subgroup ficam.** Eles têm 5–41 % de pixels pretos, mas é margem de bounding box: o fundo real em volta do objeto continua lá (conferido em amostras). Não são objetos recortados.
- **Dois grupos:**
  - **obj** (fpsw): Eggs, Protozoan, Larvae, flowers, ham10000, Eggs_subgroup, Protozoan_subgroup, Larvae_subgroup;
  - **cena** (fps): aid, remote_sensing, coconut_trees. Em coconut_trees o fps venceu o fpsw (0,647 vs 0,625, `svm_disc_md_sol_wsal_coconut_trees_s0200_split1_pct50_sh1.csv`).
- **food101 fica de fora:** são 101 classes, o que pesa demais para uma fila cheia. Posso incluir se você quiser.
- **Código:** duas chaves opt-in em `eval/spf_grade.py`. Sem elas, o comportamento é idêntico ao de antes.
  - `lei: W, w: "a-b-c"`: larguras explícitas por camada;
  - `lei.k: "3-3-3-5-5-5"`: kernel por camada.

  Elas existem porque `law_channels` só gerava potências de 2 e o kernel era um só para a rede toda. O pacote spifil não foi tocado.

Ordem de lançamento: dentro de cada YAML, os orçamentos vão de 1 M para 4,5 M e depois 2 M.

1. `disc_md_restr5m_k3_obj_jaci.yaml`, `disc_md_restr5m_k3_cena_jaci.yaml`: L1, L2, L3, L4 e L6.
2. `disc_md_restr5m_k5_obj_jaci.yaml`, `disc_md_restr5m_k5_cena_jaci.yaml`: L0, a referência.
3. `disc_md_restr5m_convnext_obj_jaci.yaml`, `disc_md_restr5m_convnext_cena_jaci.yaml`: L5.
4. `disc_md_restr5m_effnet_s104_jaci.yaml`, `disc_md_restr5m_dwr_s176_jaci.yaml`, `disc_md_restr5m_effnet_s160_jaci.yaml`, `disc_md_restr5m_effnet_s128_jaci.yaml`, `disc_md_restr5m_dwr_s184_jaci.yaml`: L7 e L8, cada uma na própria resolução, com fps e fpsw nas 11 bases.

## 5. Recomendação (provisória)

- **Hoje, sem rodar nada:** a receita cortada na **L4 (4×256 k5, 4,94 M)**. Custa −0,023 de κ médio e fica acima do random_z do mesmo tamanho. Não é "sem perda".
- **Depois da grade:** a lei vencedora é a de menor orçamento que fica a até 0,01 da receita de 8,2 M na maioria das 11 bases e se mantém entre elas. Esse é o critério de "lei que generaliza". A curva params × κ de cada lei mostra onde a perda começa.

**Smoke no athena (11/11 02:30–02:56, GPU 2, GPU_CAP_MIB = 4200):** rodei a lei L6 de 1 M (6×104, k3-3-3-5-5-5) em Larvae, split 1. O run terminou sem erro e o CSV saiu com `params` = 1.008.696 e `rf` = 205. O pico de GPU foi 1,2 GB. Levou 26 minutos para uma lei num job de base pequena, com o cache de imagens já pronto.

Os números são de um split só, então valem como indício, não como resultado:

| camada | fpsw cat / cat_tta | random_z cat / cat_tta |
|---|---|---|
| L4 | 0,804 / 0,889 | — |
| L6 | 0,752 / 0,858 | 0,388 / 0,614 |

A receita de 8,2 M fica em 0,855 / 0,878 (média dos 3 splits).

Custo estimado das 26 arquiteturas: algumas dezenas de horas de fila no jaci, com 8 jobs simultâneos.
