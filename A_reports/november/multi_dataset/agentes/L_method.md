# L_method: como as regras de escala nasceram, e o menor experimento para achar a nossa

**Resposta curta:** quase nenhuma "regra famosa" de CNN é lei de verdade.
ResNet, EfficientNet, RegNet e ConvNeXt são receitas bem testadas. Lei, no sentido de prever o erro antes de treinar, só existe em Rosenfeld, Kaplan, Hoffmann e parentes. LeJEPA é outro bicho: a regra dele vem de um teorema.

Para nós, o menor desenho é: **6 datasets (um por família), 16 células por dataset, sempre SPiFiL e aleatório lado a lado.**
Isso dá 218 cadeias se o ruído for baixo, 398 se for alto. Mais 126 de confirmação em datasets que a regra nunca viu.

A regra só vale se, num dataset deixado de fora, ela empatar com a melhor opção da grade e gastar menos que a melhor config comum.

Marcas usadas:
- **[li]:** li o texto no PDF.
- **[ag]:** um agente leu e trouxe citação com página; conferi os números principais.
- **[conta]:** conta minha.
- **[hip]:** hipótese minha, sem medida.
- pN é a página do PDF do arXiv. Nada aqui foi citado de memória.

---

## 1. Como cada um chegou à regra

| | O que mediu | Quantas redes | Regra | Forma | Por que pegou |
|---|---|---|---|---|---|
| **ResNet** [li] | Erro de treino e teste, rede simples contra residual com os mesmos parâmetros. ImageNet 18/34 (e 50/101/152), CIFAR-10 de 20 a 1202 camadas | Total não dado. Cerca de 16 redes nomeadas; a de 110 camadas rodou 5 vezes (p7) | Somar atalho. Mesmo mapa, mesmos filtros; mapa pela metade, filtros em dobro (p3) | y = F(x) + x | Experimento de causa, não ajuste |
| **EfficientNet** [li] | Top-1 de ImageNet contra FLOPs, um eixo por vez (Fig. 3), largura sobre 4 bases (Fig. 4), grade pequena em B0 | Não dado; só "small grid search" (p5) | α=1,2, β=1,1, γ=1,15 | d=α^φ, w=β^φ, r=γ^φ, com α·β²·γ²≈2 (p5) | Ganha a FLOPs iguais: até +2,5% (p8) |
| **RegNet** [li] | Distribuição do erro de uma população de redes, 400 MF, 10 épocas (p3) | n=500 por espaço (A a E e X), depois 100 redes de 25 épocas por regime (p7). Por regime, 25 sorteios e a melhor 5 vezes (p8). Total não dado | Largura cresce em linha. Profundidade ~20 blocos, estável. wm≈2,5, b=1, resolução 224 fixa (p7) | u_j = w0 + wa·j; w_j = w0·wm^round(s_j) (p5-6) | Vale para a população, não para uma rede. Ordem dos espaços se mantém em 6 cenários (p6) |
| **ConvNeXt** [li] | Top-1 de ImageNet passo a passo, FLOPs quase fixos, 2 regimes | 15 passos na Fig. 2; 3 sementes por passo (p3) | Receita 76,1→78,8 sem mudar a rede. Estágios 3:3:9:3, kernel 7 satura (p5). Variantes só mudam C e B | Nenhuma | Não é lei, é trilha: uma mudança por vez, mesmo orçamento |
| **LeJEPA** [li] | Sonda linear, 60+ arquiteturas, 10+ datasets (p1), 50 modelos abaixo de 20M (p14) | 60+ | Embedding deve ser gaussiana isotrópica. λ=0,05, 2+8 vistas, lote ≥128 (p13) | L = λ·SIGReg + (1−λ)·predição (p12) | Teorema (Lema 1, 2, Teorema 1, p5-6) mais um λ estável |

**ResNet.** A sacada foi separar "rede funda não aprende" de "rede funda decora".
A rede simples de 34 camadas tem erro maior que a de 18 (28,54 contra 27,94), e o erro de **treino** sobe junto (p1, p5). Logo não é overfitting. Com atalho, a ordem inverte (25,03 contra 27,88).
Eles dizem onde quebra: 1202 camadas dão 7,93%, pior que os 6,43% de 110, com treino parecido (p8).
Lição: teste primeiro **se o problema é treino ou dado**. Temos κ de treino e de validação por época, então esse teste sai de graça para a queda depois da L4.

**EfficientNet.** A conta α·β²·γ²≈2 **não é medida**. É só o custo de uma convolução, que cresce com d·w²·r² (p5).
O medido são três números, numa grade, num modelo, num dataset. Conferindo: 1,2·1,1²·1,15² = 1,92 [conta].
O paper nunca prevê erro. Depois, Bello diz que regra tirada de regime pequeno quebra (p7) [ag], e Dollár troca os pesos por α=0,8, puxado para largura (p5) [ag].

**RegNet.** A ideia boa é comparar **populações**, não campeões. Sorteia 500 redes, olha a curva de erro, simplifica uma regra por vez e confere se a curva não piora.
De 16 dimensões caiu para 6, e a linha reta "apareceu" olhando as 20 melhores redes (p5).
Resultados surpreendentes: profundidade não cresce com orçamento e resolução fixa é melhor, mesmo com mais FLOPs (p7). O oposto do EfficientNet.

**ConvNeXt.** Não há lei. A parte útil para nós é o aviso: **a receita sozinha valeu +2,7%** (p3). Sem congelar a receita, qualquer lei mede a receita.

**LeJEPA.** Tem "scaling law" no texto, mas não é de tamanho: é dividir a loss por λ^0,4, o que sobe a correlação de Spearman com a sonda de ~0,85 para ~0,99 (p15). Em ResNet-18 no flowers102, vai de 0,60 para 0,95.
O que serve para nós é outra coisa: **escolher modelo sem rótulo pela loss**.
Hoje não temos isso. A nossa loss "não vê a vantagem do SPiFiL" (`report_halloween_october.md` §4.8).

## 2. Leis por dados e parâmetros

| | Medido | Pontos | Forma | Prova de que é lei |
|---|---|---|---|---|
| **Rosenfeld 2019** [li] | Erro de teste numa grade (modelo, dados), 6 datasets de imagem, 3 de texto | 42 a 49 configs por dataset, até 6 constantes (p6) | ε = a·n^−α + b·m^−β + c∞, mais um envelope que parte do chute ao acaso (p6) | Ajuste com 1/16 do modelo e 1/8 dos dados prevê o resto no ImageNet: erro médio 4,5%, desvio 4,7% (p8) |
| **Kaplan 2020** [li] | Loss de texto, N de 768 a 1,5B, D de 22M a 23B tokens (p7) | Não dado | L(N)=(Nc/N)^0,076; L(D)=(Dc/D)^0,095; penalidade por N^0,74/D (p3-4) | 6 ordens de grandeza. Forma da rede quase não importa (p3) |
| **Hoffmann 2022** [li] | Loss de texto, 70M a 16B | Mais de 400 modelos (p1) | L = E + A/N^α + B/D^β; E=1,69, A=406,4, B=410,7, α=0,34, β=0,28 | Três métodos diferentes dão quase o mesmo: a≈b≈0,5 (p5-8) |
| **Zhai 2021** (visão) [ag] | Erro few-shot e finetune contra compute, 5M a 2B params | 11 arquiteturas, ~260 linhas nas tabelas | E = a(C+d)^−b + c (p3). Expoente b vai de 0,20 a 0,77 conforme o dataset (p12) | Só qualitativo. O menor modelo sai da lei (p3) |
| **Hestness 2017** (visão) [ag] | ResNets de 89K a 121M, ImageNet de 1 a 800 imagens por classe | Não dado | Erro ~ m^−0,31; **melhor tamanho s(m)=2,76·m^0,57** (milhões de params, m em imagens por classe) (p9) | Repetição em 4 domínios; sem predição fora da amostra |
| **Alabdulmohsin 2022** [ag] | 72 tarefas de visão, cada uma uma curva | Não dado por curva | M4: (ε−ε∞)/(ε0−ε)^α = β·x^c (p4) | Mede extrapolação: ajusta até metade, prevê o dobro; erro 1 a 3% (p6, p14) |
| **Alabdulmohsin 2023** (ViT em forma) [ag] | Erro 10-shot por eixo de forma | 115 experimentos (p2) | f = α·x^−a + (β·x^b + ξ)·t^−c + ε, com ótimo x* ∝ t^s, s=c/(a+b) (p4-5) | O centro da estrela nunca entra no ajuste (p6). Modelo novo (SoViT) com forma nunca vista |
| **Sharma & Kaplan 2020** [ag] | Loss contra N e dimensão intrínseca | CNN até ~70 mil params (p20) | α ≈ 4/d (p1) | CIFAR10 +5%, FMNIST +8%, SVHN −6%, **MNIST −37%** |

**O que separa lei de bom modelo.** Quatro coisas:
1. Poucas constantes para muitos pontos.
2. Previsão de um ponto que **não entrou no ajuste**, com o erro dito em número.
3. Mesma forma em vários datasets.
4. Um jeito claro de dar errado.

Por esse critério, EfficientNet e RegNet ficam abaixo de Rosenfeld e Hoffmann, embora sejam mais usados.

**Como um ajuste engana** [ag]:
- Melhor ajuste não é melhor previsão. Curvas com expoente de −0,24 a −0,40 se sobrepõem (Alabdulmohsin 2022, p5).
- Intervalo apertado demais é falha de otimizador. A replicação achou intervalo **50 vezes mais largo** que o de Hoffmann (Besiroglu, p5). Causa: média em vez de soma na loss, e o otimizador parou cedo.
- Com pouco ponto, quebra não aparece. Caballero: com N no eixo, às vezes 7 pontos bastam, muitas vezes não (p18).
- Bello: "do not recommend generating scaling rules exclusively in a small scale regime" (p7). A escolha entre profundidade e largura virou entre 10 e 350 épocas (p6).

**Nenhum dos papers lidos entrega "dataset entra, modelo sai"** [ag]. O mais perto é α≈4/d, mas esse d exige rede treinada e muda 35% com o estimador (Sharma, p22).
Também nenhum mede lei em rede treinada camada a camada, nem CNN ≤5M em dataset pequeno. Essa parte é nova.
O mais perto é Belilovsky: camada a camada empata com ponta a ponta em VGG-11, mas com modelos de 46M para cima, sem lei, e com sinal de que auxiliar raso satura antes (p6-7, p13) [ag].

---

## 3. O que muda no nosso caso

| Clássico | Nosso |
|---|---|
| Orçamento é FLOPs | Teto é **5M parâmetros**. Parâmetro não depende da resolução (Dollár, p3), então resolução é "de graça" no teto e cara em memória |
| Inicialização é detalhe | Cada camada nasce do SPiFiL. Largura e profundidade pedem **candidatos**, e essa conta é nova |
| Treino de ponta a ponta | Camada a camada: uma cadeia até L5 já contém L2, L3 e L4. **Profundidade sai quase de graça** |
| Um dataset grande | 12 datasets, mas só **6 famílias**: as irmãs têm 78%, 83% e 21% da antiga mapeados (R2 §5) |
| Erro alto | κ perto de 0,9: ruído de ±0,02 por run (HAL §4.15) é do tamanho dos efeitos |

Ponto importante: **6 famílias sustentam no máximo 1 medida de dataset por eixo.** Mais que isso é decorar seis pontos.

### 3.1 As contas exatas

Conferi contra o oficial: 32/48/64/1280 com k 5/5/5/1 dá 200.944 params, batendo com o CSV [conta].

| m | Convs | Params (c=1280) | Maior D | N pedido |
|---|---|---|---|---|
| 1/4 | 8/12/16 | 29.596 | 300 | 3.840 |
| 1/2 | 16/24/32 | 72.312 | 600 | 3.840 |
| 1 | 32/48/64 | 200.944 | 1.200 | 3.840 |
| 2 | 64/96/128 | 631.008 | 2.400 | 3.840 |
| 4 | 128/192/256 | 2.182.336 | 4.800 | 4.801 |
| 6 | 192/288/384 | 4.655.264 | 7.200 | 7.201 |

O oficial usa 4% do teto. São 2,2 décadas de tamanho disponíveis.

Profundidade: 4 convs cabem até m=4 (4,80M). 5 convs com m=4 passam do teto (9,9M). Com m=1: 395.600 (4 convs) e 743.888 (5 convs).

Campo de visão em pixels, com k5 e pool 3/2: **RF(d) = 6·2^d − 5**, ou seja 7, 19, 43, 91, 187 [conta]. Bate com os "~40 px" do HAL §4.5.
Cada camada com pool dobra a visão. Por isso **profundidade cresce com log da resolução**, não como potência.

Memória: pico ≈ 0,065 MiB por pixel² no oficial (B1: 2.828 / 6.792 / 26.420 MiB a 200 / 320 / 640 px).
Se crescer com m [conta, não medido]:
- m=4 a 320 px dá ~27 GB.
- m=6 a 320 px dá ~40 GB, no limite. Ali o topo fica m=4.
- 640 px só aguenta m≈1.

### 3.2 O que o SPiFiL pede em cada largura e profundidade

Esta é a parte que nenhum paper tem. A regra precisa dizer, para cada camada ℓ:

**Demanda de candidatos:** N_ℓ ≥ max(3·w_ℓ, k²·w_{ℓ−1} + 1).
- O 3 vem do `pool_factor`: sem 3 candidatos por filtro, a seleção não escolhe nada (`SP/selection.py:126-129`, R4).
- O segundo termo é a covariância da Mahalanobis: com N ≤ D ela perde posto, e o shrinkage a mantém "inversível, não informativa" (`flim/spifil.py:327-329`).
- Hoje estamos abaixo disso: eggs tem N=400 para D=800 e 1200, e por isso `budget_check: false` (R4).

**Oferta por imagem:** sementes distintas numa grade g ≈ g²·(1 − e^(−n/g²)).
É bola em caixa: as sementes caem nas casas da grade, e várias caem na mesma casa [conta].
Confere com a medida sem máscara de R4 em 24×24:

| n | 25 | 50 | 100 | 200 | 450 |
|---|---|---|---|---|---|
| medido | 23 | 45 | 77 | 159 | 280 |
| fórmula | 24,5 | 47,9 | 91,8 | 169 | 312 |

Erro de até 19%, só 3 imagens. Serve como ponto de partida, não como lei.

**Imagens por classe I**, com 800 px por superpixel (a maior exigência entre as camadas) [conta]:

| Tamanho (n) | C=2 | C=7 | C=9 | C=30 |
|---|---|---|---|---|
| 200 px (n=50), m=1/4/6 | 41 / 49 / 73 | 12 / 14 / 21 | 9 / 11 / 17 | 3 / 4 / 5 |
| 320 px (n=128), m=1/4/6 | 16 / 19 / 29 | 5 / 6 / 9 | 4 / 5 / 7 | 2 / 2 / 2 |
| 640 px (n=512), m=1/4/6 | 4 / 5 / 8 | 2 / 2 / 3 | 1 / 2 / 2 | 1 / 1 / 1 |

**Fração que nasce do SPiFiL:**
- Convs: f=1 sempre que N ≥ w. Com a oferta pedida acima, é o caso.
- Cabeça 1×1: f = min(1, C·I·distintos / c). Hoje, com I=1 e c=1280 [conta]:

| Tamanho | C=2 | C=7 | C=9 | C=30 |
|---|---|---|---|---|
| 200 px | 7% | 26% | 34% | 100% |
| 320 px | 19% | 67% | 86% | 100% |
| 640 px | 77% | 100% | 100% | 100% |

Bate com R4 (8%, 23%, 31%). R4 mostra pior: contando só filtros distintos, são 3% a 13%.

**Três consequências:**
1. **Resolução alimenta o SPiFiL.** Dobrar r dá até 4 vezes mais casas nas camadas fundas. Por isso C=2 a 640 px pede 10 vezes menos imagens que a 200 px.
2. **Profundidade esvazia o SPiFiL.** Cada pool divide as casas por 4. A 200 px, a L5 só tem 11×11, e a cabeça depois dela, 5×5: no máximo 25 sementes por imagem.
3. **Poucas classes é o pior caso.** A verba é por classe (`out_channels // C`), e a largura deveria ser múltiplo de C, para não jogar filtro fora (X1 já mostra 64→63 e 32→27).

**Candidato não é filtro útil.** Nas X1 sem máscara, 38 a 51 dos 63-64 filtros da L1 têm um vizinho com |cos|>0,9 (7 braços medidos agora).
Logo a largura nominal engana. A regra precisa contar filtros **distintos**, e X3 vai dizer quanto a poda custa.

### 3.3 Forma candidata da lei

Parte de ViT-em-forma, trocando compute por **número de imagens n**. É troca minha, não do paper; lá eles supõem dado sem limite (p4).

e(x, n) = E_t + A_t·x^−a + B_t·x^b·n^−c, para cada eixo x

- Primeiro termo: o piso do dataset.
- Segundo: modelo pequeno demais.
- Terceiro: modelo grande para pouco dado.

Daí sai a regra fechada: **x\* = K_t · n^s, com s = c/(a+b).**

Bahri ajuda a escolher onde cada medida entra: trocar o número de classes não moveu o expoente, trocar o dataset e somar ruído moveram (p3, p10) [ag]. Então C deve entrar em K, não em s [hip].

Como a queda depois da L4 e a largura plana (HAL §4.3-4.4) sugerem que pode nem haver ótimo interno, o alvo da regra é o **menor modelo a até 0,01 do platô**. Esse alvo existe mesmo em curva plana, e combina com "CNN leve".

**A regra final, em 5 passos:**
1. **Medir o dataset sem treinar:** C, n, tamanho nativo, fração de sementes não constantes, filtros que sobram após redundância.
2. **Resolução:** r = regra ajustada, limitada por memória e pelo tamanho nativo. Vem de X1 e X1s.
3. **Profundidade:** d = menor d com RF(d) cobrindo a escala do objeto, sem usar máscara.
4. **Largura:** m = K·n^s, arredondado para múltiplo de C, até 5M.
5. **SPiFiL:** n = r²/(px por superpixel); I pela demanda acima; N ≤ 30.000 (a matriz N×N, ~9 GB, D1); f resultante.

Nosso análogo do α·β²·γ²≈2 são três contas exatas: params ≤ 5M, memória ≤ 40 GB, N ≤ 30.000. O medido são só os expoentes.

**Três formas de o SPiFiL valer** (comparando com o aleatório na mesma célula):
- Só A muda: é um fator fixo.
- Muda o expoente a: ajuda mais quanto maior o modelo.
- Só E muda: sobe o teto.

Para falar em número, proponho o **multiplicador**: quantas vezes o aleatório precisa ser maior para empatar com o SPiFiL. É a ideia de "dados efetivos transferidos" de Hernandez 2021 (p1, p5) [li, por trechos].
Se o multiplicador cair para 1 nas larguras altas, o SPiFiL só ajuda modelo pequeno.

---

## 4. O menor desenho

**Regras gerais:**
- Receita congelada antes: sai de E0 a E4 (ConvNeXt, Rosenfeld p4).
- Leitura final `gem_tta_pw`, sem passar de 2000-d, ajuste **só com validação**.
- c=1280 fixa no eixo largura, para a dimensão da leitura não mudar. Isso difere do prompt salvo, que escalava a 1×1 junto.
- Controle pareado: `random_layer: true` (`flim/spifil.py:949`). Mesma forma, mesma alocação, mesma norma por filtro; só muda a direção. `random_layer_classic` entra só no centro, para ligar com a família antiga.
- 60 épocas para a forma, 200 para conferir (Bello p7).

**Datasets do núcleo, um por família:** Eggs_subgroup, Protozoan_subgroup, Larvae_subgroup, flowers, remote_sensing, aid.
r0 = 200 para os três primeiros, 320 para os outros.

**Por dataset, 16 células, cada uma com SPiFiL e aleatório:**

| Bloco | Células | Para quê |
|---|---|---|
| Largura × dados | m {1/4, 1, 4} × pct {10, 50, 100} = 9 | Dá o expoente s dentro do próprio dataset |
| Largura extra | m {1/2, 2, 6}, 50% = 3 | Curva e o ponto de quase 5M |
| Resolução | m=1, r {r0/2, 1,6·r0} = 2 | Eixo r |
| Cantos | (m=4, r0/2) e (m=1/4, 1,6·r0) = 2 | Mede se os eixos conversam (Observação 2 do EfficientNet) |

Profundidade sai das cadeias: d {2, 3, 4} em 5 células (m {1/4, 1, 4} a 50% e as 2 de resolução).

**Por dataset, mais 8 cadeias:**
- **Oferta do SPiFiL (6, só SPiFiL):** m {1, 4} com N/D≈1 e N/D≈3; e no centro, "cabeça aleatória" e "só L1 SPiFiL". É aqui que se mede o que candidatos e fração valem.
- **Sementes do aleatório (2):** no centro, porque ele varia 0,012 a 0,022 entre sementes (HAL §4.8).

aid vai fino, porque cada estágio custa perto de 10 vezes mais: 7 células e 4 extras.

**Por que esse tamanho (simulação em `tools/sim.py` e `sim2.py`, s verdadeiro = 0,5, parâmetros supostos):**
- **Sem o eixo pct, o expoente sai errado:** dá 0,22 a 0,34. É o que não pode cair.
- **Ruído manda no resto.** Intervalo de 95% de s com 6 datasets, 12 células:

| Ruído por run | 1 split | 3 splits no bloco largura×dados | 3 splits em tudo |
|---|---|---|---|
| 0,02 | 0,15 a 1,51 | 0,23 a 1,09 | 0,30 a 0,73 |
| 0,012 | 0,28 a 0,90 | 0,35 a 0,75 | não rodei |

Por isso o desenho começa medindo o ruído.

**Quantidade e custo:**

| Bloco | O que é | Cadeias | Horas de slot |
|---|---|---|---|
| B0 | Contas e medidas sem treino, 12+3 datasets, CPU | 0 | horas de CPU (não estimei) |
| B1 | Aleatório nas X1: p0800 × 3 larguras × 12 datasets × 3 splits, só L1 | 108 braços | ~26 |
| B2 | Ruído: 2 repetições do split 1 no centro, S e R, 6 datasets | 24 | ~45 |
| B3 | Núcleo. Decisão: se ruído ≤0,012, 1 split; se ~0,02, 3 splits no bloco largura×dados | **218 ou 398** | **327 ou 511** |
| B4 | Confirmação (seção 5) | 126 | ~272 |

- Em 4 slots, B3 leva 3,4 ou 5,3 dias.
- A conta usa a velocidade de m=1, mas m≥4 é limitado por GPU. Use margem de **×1,5**, ou seja 5 a 8 dias [conta, não medido].
- Base: `epoch_s` medido nas X1, e fórmula de D1 para o resto. Reproduz os ~11 h de triagem de D1 (deu 10).
- SVM: ~590 avaliações somando B1 a B4, em paralelo ao treino, uma por vez. Protozoan a 1280-d nunca foi cronometrado.

**O que precisa mudar no código** (depois das filas; schema não se mexe com fila rodando):
1. Bifurcar a cabeça a partir de um estágio pronto. Sem isso, as 5 células com profundidade custam 1,7 vez mais, e o pipeline não é bit-reprodutível, então prefixos repetidos não seriam idênticos.
2. Chave por camada dizendo quem nasce do SPiFiL. Hoje `random_layer` vale para o run inteiro.
3. `fill_random` só existe na última camada (`growth.py:414-415`). Com a oferta da seção 3.2 isso não atrapalha.

Tudo em YAML, `experiments/autoencoder/disc_md_law_*.yaml`, com chaves novas em `RUNNER_ARGS`.

---

## 5. O que refuta a lei

Escrever isto e congelar **antes** de ver os dados de B4.

**Confirmação (B4, 126 cadeias):**
- **Irmãos:** Eggs, Protozoan, Larvae, helminth_eggs, helminth_larvae, protozoan_cysts. 4 braços × 3 splits (regra com SPiFiL, regra com aleatório, melhor config comum, modelo trocado) = 72 cadeias.
- **Nunca vistos:** `coconut_trees` (90 px, 2 classes), `medmnist_retinamnist` (128 px, 5), `medmnist_dermamnist` (28 px, 7). Já estão em `datasets/organized/`, fora dos 12. Mesmos 4 braços = 36 cadeias. Precisam de registro (A0).
- **Épocas:** 3 datasets × 3 larguras × S e R a 200 épocas = 18 cadeias.

O braço "regra" a 200 épocas já é a corrida final prevista em D1, então não é custo novo.

| Teste | Como | Refuta se |
|---|---|---|
| **T1 Tamanho** | Ajusta até m=2 (631K), prevê m=4 e m=6 | Erro médio > 0,015 em κ, ou mesmo sinal em 5 de 6 datasets |
| **T2 Família de fora** | Tira uma família, reajusta (6 vezes, sem treinar) | Expoente da família fora da faixa de 95% em 2 ou mais das 6 |
| **T3 Regra contra config comum** | Melhor config única, escolhida sem a família testada | Regra fica > 0,01 abaixo do melhor da grade; **ou** não gasta 2× menos params nem ganha ≥ 0,01 |
| **T4 Troca** | Dá a cada dataset o modelo do dataset mais diferente | κ e custo não pioram: a dependência do dataset é enfeite |
| **T5 Medida embaralhada** | Permuta a medida entre famílias, reajusta | Regra embaralhada prevê igual: a medida não serve |
| **T6 Controle** | SPiFiL contra aleatório, pareado, no modelo da regra | Intervalo do ganho inclui 0 na maioria dos datasets; ou multiplicador < 1,25 em m=4. Cai a parte SPiFiL da lei |
| **T7 Oferta** | N/D de 0,1 a 3 e f de 0 a 1 | κ muda < 0,01 em 4 de 6: a conta de candidatos não importa, use a mais barata |
| **T8 Nunca vistos** | Previsão **escrita antes** de treinar: modelo, κ e intervalo | Fora do intervalo em 2 dos 3 |
| **T9 Épocas** | Ordem dos tamanhos a 60 e a 200 épocas | Ordem muda: a lei vale só para aquele regime |

T2 é por **família**, não por dataset: tirar só Eggs_subgroup deixa Eggs e helminth_eggs ensinando a resposta.

**Aviso sobre T3.** Pela simulação, com erros de escala 0,03, errar a largura por 2× custa 0,004 em κ, e por 4×, 0,015.
Uma regra perfeita ganharia da melhor config comum só 0,001 a 0,0065, com ótimos espalhados por 4× a 32× [conta, supondo a=b=0,5].
Ou seja, **no eixo largura a regra provavelmente não vai ganhar em κ**. Ela ganha em tamanho: mesmo κ com menos params. Por isso T3 aceita economia. Resolução e oferta devem ter efeito maior [hip].

**Resultado que rebaixa, sem matar:** se valer só dentro de cada família, é "lei por domínio". Se a curva for plana, é "regra de mínimo suficiente".

**Objetivo (a parte LeJEPA):** a lei vale para um objetivo fixo. Depois de congelar, vale testar nas 3 larguras se o λ escolhido segue bom, sem reajuste. Esse é o análogo do "λ estável". Se a loss nova prever a sonda, ganhamos escolha de modelo sem rótulo.

---

## 6. Avisos encontrados

1. **Sonda com κ de validação 0 e treino alto.** Ex.: `helminth_larvae_split3` tem κ de treino 0,90 e validação 0,0 nas 20 épocas. Treino menos validação passou de 0,3 em 7 dos 24 braços de helminth_eggs, helminth_larvae, protozoan_cysts e Larvae_subgroup. Fora dessas quatro, 0 de 20.
   Como `best_by_probe: true` usa essa sonda (`autoencoder_flim_module.py:179`), o melhor checkpoint pode estar errado nesses braços. Não achei a causa. Palpite: canal quase morto, com desvio pouco acima do limite 1e-5, explode na validação [hip]. **Conferir antes de ajustar qualquer lei.**
2. **Ruído desconhecido na leitura final.** ±0,02 vem do protocolo antigo com gem. Na sonda de L1 de X1, o desvio entre splits vai de 0,024 a 0,47, mas isso mistura o problema acima.
3. **Há CSV de teste das X1 sendo gerado** (`results/svm_disc_md_eval_x1_s200_test_gemttapw.csv`, 04:13). Não usei; um `head` mostrou uma linha. D1 previa SVM de validação. Vale conferir.
4. **aid custa perto de 10 vezes mais por estágio** (leitura de 120 ms por imagem): 347 h se fosse completo. Um cache redimensionado em `/exps` poderia baixar isso; não medi.
5. **O prompt salvo** (`A_reports/october/discovery_scaling_law_prompt.md`) está velho: teto de 20 GB, sem resolução, sem controle, 1×1 junto com a largura. Este desenho o substitui.
6. **Inversão a poucos dados:** o aleatório ganha a 1% em eggs e larvae (R1 §1.3). Como I=1, o número de candidatos é o mesmo a 1% e a 50%, então a oferta sozinha não explica isso. A lei precisa de ganho SPiFiL que dependa de n, por isso pct entra com controle.

## 7. Fontes

PDFs do arXiv, texto em `<scratchpad>/law/txt/`: 1512.03385, 1905.11946, 2003.13678, 1905.13214 (só baixado), 2201.03545, 2511.08544, 1909.12673, 2001.08361, 2203.15556, 2102.01293, 2103.06877, 2103.07579, 2104.00298, 2106.04560, 2209.06640, 2305.13035, 1712.00409, 2004.10802, 2102.06701, 2206.14486, 2210.14891, 2404.10102, 1812.11446.

Leitura: Kaplan, Hoffmann e Hernandez foram lidos por trechos; ResNet, ConvNeXt e LeJEPA nas seções centrais. Os demais [ag] vieram com citação e página; confirmei no texto 115 experimentos, c≈0,65, os s_k, 50-fold e 600.000.

Do repositório: `flim/spifil.py`, `experiments/ray/runners/growth.py`, `methods/autoencoder/autoencoder_flim_module.py`, `A_reports/october/report_halloween_october.md` (HAL), relatórios R1, R2, R4, B1, D1, K.
O grafo achou funções e arquivos. Grep só serviu para linhas dentro delas e valores de YAML, que ele não indexa.

Contas: `<scratchpad>/law/tools/acct.py`, `cost.py`, `sim.py`, `sim2.py`. Nada foi escrito no repositório, nenhuma GPU foi usada.
