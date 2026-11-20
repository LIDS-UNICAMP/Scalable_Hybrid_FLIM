# L_classic: ideias clássicas para montar kernels da camada 3 em diante

## 0. A resposta

**Encher a 1×1 com patches crus do SPiFiL piora.** Na sonda sem treino, a 1.600 sementes, perde 0,049 em eggs e 0,076 em protozoan contra a 1×1 oficial, nos 6 casos (2 datasets × 3 splits).

O que os artigos ensinam, e a sonda confirma em parte:

1. **Na 1×1, o candidato tem que passar por "branqueamento" e ter os dois sinais.** Branquear é tirar a correlação entre canais, para o brilho não mandar sozinho. Os dois sinais são usar o filtro `f` e o `-f`. Assim o banco empata com o oficial a 1.600 tokens e ganha 0,024 / 0,016 a 6.000.
2. **Na L3, PCA dos próprios patches, sem treino nem rótulo, chega onde 200 épocas de backprop chegaram.** Ela funciona até com 400 patches.
3. **κ caindo com a profundidade aparece sem treino nenhum.** Logo o culpado não é o treino. A informação não some: concatenando as camadas no vetor final, κ sobe 0,056 / 0,078 sobre a L3.
4. **A dimensão 2,6 é quase só brilho.** A primeira direção do vetor tem correlação 0,94 com o brilho da imagem em eggs e em protozoan. Subir esse número, sozinho, não subiu κ.
5. **Canal "morto" não custa κ.** Centrar o corte elimina todos eles e κ não muda. Essa hipótese era minha, e caiu.

Tudo isso é sonda em CPU, sem backprop em cima do banco novo. Ela mostra o ponto de partida, não o resultado final do híbrido.

---

## 1. O que medi

**Como.** Peguei os checkpoints oficiais `disc_bp01w64c1280_bg1`, a 50%, troquei só o banco de filtros de uma camada e li com GeM + regressão logística padronizada. Usei 1.000 imagens de treino e 600 de validação. **Teste não entrou. GPU não entrou. Nada foi treinado.** A escolha dos filtros não usa rótulo.

**A régua é esta:**
- Reproduz o R8: dimensão 2,59, 108 canais mortos, 46,5% de zeros.
- Os mesmos filtros da L3 medidos por dois caminhos deram 0,867 e 0,854 em eggs, 0,775 e 0,766 em protozoan. **Diferença abaixo de ~0,02 na média de 3 splits não decide.**
- Larvae satura (κ perto de 1), não separa nada. Ficou fora da replicação.
- A leitura com whitening que tentei imitar variou 0,03 entre duas sementes do mesmo braço. Descartei essa coluna.

### 1.1 A 1×1 (1280 filtros), média de 3 splits

| banco | eggs | protozoan | Δ pareado contra o oficial inicial |
|---|---|---|---|
| oficial depois do treino | 0,916 | 0,863 | +0,013 / +0,023 (6 de 6 positivos) |
| **oficial inicial** (SPiFiL + aleatório) | **0,903** | **0,839** | referência |
| 100% aleatório | 0,908 | 0,835 | empate |
| patches crus, 1.600 tokens | 0,854 | 0,763 | **−0,049 / −0,076** (6 de 6 negativos) |
| patches crus, 120.000 tokens | 0,893 | 0,799 | −0,010 / −0,040 |
| branco, 1.600 tokens | 0,893 | 0,803 | −0,010 / −0,036 |
| branco, 120.000 tokens | 0,913 | 0,854 | +0,010 / +0,015 |
| branco + dois sinais, 1.600 | 0,917 | 0,839 | +0,013 / 0,000 |
| **branco + dois sinais, 6.000** | **0,927** | **0,855** | **+0,024 / +0,016** |
| branco + dois sinais, 120.000 | 0,923 | 0,860 | +0,019 / +0,020 |
| SPiFiL + resto branco | 0,908 | 0,853 | +0,005 / +0,013 |

"Cru" é o patch com z-score por canal, como `builder.py:8-11`, escolhido longe dos já escolhidos. "Branco" é o mesmo com ZCA. Parente do SPiFiL, não o SPiFiL: sem Fisher e sem rótulo.

Na largura do SPiFiL (400 / 300 filtros): SPiFiL 0,868 / 0,744, aleatório 0,908 / 0,804, branco a 1.600 tokens 0,892 / 0,824. **Parte dessa perda do SPiFiL são cópias:** dos 400 de eggs só 160 são distintos, e dos 300 de protozoan só 132 (R4). O checkpoint oficial foi feito com máscara, que agora está proibida.

### 1.2 A L3 (5×5 sobre o L2, 64 filtros), média de 3 splits

| banco | largura | eggs | protozoan |
|---|---|---|---|
| SPiFiL inicial | 64 | 0,814 | 0,737 |
| SPiFiL treinado 200 épocas | 64 | 0,867 | 0,775 |
| PCA, 400 patches | 64 | 0,879 | 0,783 |
| PCA, 1.600 patches | 64 | 0,852 | 0,779 |
| PCA, 60.000 patches | 64 | 0,876 | 0,776 |
| PCA 32 + dois sinais, 60.000 | 64 | 0,895 | 0,807 |
| SPiFiL inicial + dois sinais | 128 | 0,872 | 0,772 |
| SPiFiL treinado + dois sinais | 128 | 0,881 | 0,809 |
| PCA 64 + dois sinais, 1.600 | 128 | 0,890 | 0,827 |

Leitura:
- **PCA contra SPiFiL treinado: empate** (−0,015 / +0,003 a 1.600). Só com 60.000 patches e dois sinais passa: +0,028 / +0,031.
- **Dois sinais dobram a largura sem parâmetro novo:** +0,059 / +0,035 no SPiFiL inicial. No treinado, +0,014 / +0,033.
- Prova limpa de que o sinal negativo traz informação (split 1): PCA 32 + dois sinais contra os mesmos 32 repetidos, +0,068 em eggs e +0,092 em protozoan.
- **Comparação injusta a favor da PCA:** ela vê o L2 final, e o SPiFiL inicial nasceu de um L2 anterior. Os +0,04 sobre o "inicial" estão inflados. A comparação honesta é contra o treinado.
- **Branquear na 5×5 é perigoso** (split 1): 0,637 / 0,533 / 0,676 com exemplares brancos. Os próprios filtros SPiFiL branqueados caíram em protozoan e larvae (0,641 e 0,667). Em D = 1200 o ZCA amplifica ruído. Na 1×1 (D = 64) funciona.

### 1.3 Pilha gulosa sobre a L3 (+3 camadas 5×5, largura 64, sem treino), média de 3 splits

L3 sozinha: 0,854 / 0,766.

| banco | só a última: L4 → L5 → L6 (eggs) | idem (protozoan) | concat L3..L6 (252–256-d) |
|---|---|---|---|
| patches crus | 0,783 → 0,536 → **0,436** | 0,653 → 0,513 → **0,447** | 0,891 / 0,785 |
| aleatório | 0,826 → 0,798 → 0,766 | 0,731 → 0,684 → 0,678 | 0,900 / 0,817 |
| PCA + dois sinais | 0,823 → 0,806 → 0,800 | 0,751 → 0,712 → 0,709 | **0,910 / 0,844** |

Leitura:
- **Todo banco cai lendo só a última camada.** É a mesma curva do HAL:239, sem treino nenhum.
- **Patch cru desaba:** dimensão efetiva 1,04, duas direções guardam 99% (split 1). Cada camada reforça a direção do brilho.
- **A informação continua lá:** concatenar recupera tudo.
- **Concat não bate a largura:** 256-d de PCA ficam −0,007 / −0,019 do oficial treinado de 1280-d. É um empate com 5× menos dimensões, não uma vitória da profundidade.

### 1.4 Hipóteses que caíram ou mudaram

- **Canal morto (minha).** Com bias 0, 15–23% dos canais ficam quase sempre desligados e 20–26% sempre ligados. Centrar o corte zera os dois grupos e κ fica igual: 0,919 → 0,919 e 0,853 → 0,849 (split 1).
- **Dimensão efetiva como alvo.** Banco com dimensão 1,3 deu 0,919; com 6,1 deu 0,932 (split 1, eggs). Patches crus têm dimensão baixa *e* κ baixo, mas a relação não é geral.
- **Grupos de canais (Coates & Ng) não ajudaram aqui**, em split 1: L3 0,831 / 0,643, 1×1 0,903 / 0,815. Faz sentido: a regra deles existe para entrada de milhares de canais, e nossa L3 tem 64. Não testei sobre os 1280.
- **"O backprop quase não mexe".** Vale na 1×1 (cosseno 0,93 a 0,98 entre início e fim, ganho +0,013 / +0,023). Não vale na L3 (cosseno 0,36 a 0,83, split 1), embora o cosseno ali seja inflado pelo mesmo descompasso de L2 citado acima.

---

## 2. Artigos sobre filtros em camadas profundas sem rótulo

Formato: citação, o que mede, a regra, como traduzir sem tocar no pacote SPiFiL. "Lido" = conferi no PDF.

### 2.1 Coates, Lee, Ng. *An Analysis of Single-Layer Networks in Unsupervised Feature Learning*. AISTATS 2011 (lido)
- **Mede:** k-means, GMM, autoencoder e RBM esparsos em CIFAR, variando largura, whitening, passo e tamanho do campo.
- **Resultado:** k-means com 1.600 filtros dá 77,9%, com 4.000 dá 79,6%. "Whitening, muitos filtros e passo pequeno" ajudam qualquer algoritmo. Campo de 6 px venceu; 12 px ficou igual ou pior.
- **Regra:** gaste o orçamento em largura e passo, não no algoritmo. Branqueie antes.
- **Tradução:** a largura vale na 1×1 (ver 1.1). Em split 1, indo de 64 a 1280 filtros brancos, κ sobe de 0,894 para 0,926 em eggs e de 0,769 para 0,880 em protozoan. Eggs já para em 320 (0,931).

### 2.2 Coates, Ng. *The Importance of Encoding versus Training with Sparse Coding and Vector Quantization*. ICML 2011 (lido)
- **Mede:** troca o dicionário e o codificador de forma independente.
- **Resultado:** em NORB, patches sorteados ao acaso com codificador bom dão 95,0%, acima do dicionário treinado (94,1%). A nota 2 diz que separar polaridade "sempre melhorou".
- **Regra:** o dicionário importa menos que o jeito de codificar. Exemplares ao acaso bastam; use `f` e `-f`.
- **Tradução:** explica por que 69–92% de aleatório não quebra o modelo. E aponta o sinal negativo como o ganho mais barato, confirmado aqui.

### 2.3 Coates, Ng. *Selecting Receptive Fields in Deep Networks*. NIPS 2011 (lido)
- **Mede:** 1, 2 e 3 camadas de k-means em CIFAR, com três formas de ligar canais.
- **Resultado:** 1 camada 78,3%; 2 camadas ligando tudo **77,4%**; 2 com grupos ao acaso 77,6%; 2 com grupos aprendidos 81,2%; 3 camadas 82,0%; 1 camada larga 80,6%.
- **Regra:** ZCA nos canais, similaridade = correlação dos quadrados, sorteia uma semente e pega seus T vizinhos (T = 200 entre milhares de mapas), roda k-means só nesse grupo. Sem isso, a 2ª camada fica pior que a 1ª.
- **Detalhe que quase passa batido:** o vetor final deles **concatena as camadas** (Seção 3.4). O ganho de profundidade vem junto com isso.
- **Tradução:** não use agora em L3 (64 canais). Passa a valer se alguém empilhar sobre a 1×1 de 1280, onde D = 1280 × k².

### 2.4 Coates, Ng. *Learning Feature Representations with K-means*. Tricks of the Trade 2012 (lido)
- **Mede:** receita prática de k-means em imagem, seção 5 sobre camadas.
- **Regra:** sem whitening os centroides seguem a correlação dos dados; em dimensão alta o k-means precisa de muito dado; empilhar sem controlar a conexão é "essencialmente infrutífero".
- **Tradução:** é a explicação direta da linha "patch cru" das tabelas 1.1 e 1.3.

### 2.5 Chan et al. *PCANet*. IEEE TIP 2015 (lido)
- **Mede:** cascata de filtros PCA, com RandNet (aleatório) e LDANet como controles.
- **Resultado:** em MultiPIE com iluminação + expressão + pose, RandNet-2 42,47%, PCANet-2 66,49%. "Dois estágios bastam em geral; mais fundo não melhora necessariamente." LDANet, que usa rótulo, não ganhou de PCANet nesse teste (46,72%).
- **Regra:** filtros = maiores autovetores das patches com a média removida, estágio por estágio. Exige k² ≥ número de filtros.
- **Tradução:** na L3, PCA das `state.patches` do próprio grow. Cada filtro é combinação de patches nascidos dos superpixels. Na 1×1 tem teto: D = 64 dá 64 direções, e 128 com dois sinais (0,901 / 0,814, split 1). Não enche 1280 sozinha.

### 2.6 Krähenbühl et al. *Data-dependent Initializations of CNNs*. ICLR 2016 (lido)
- **Mede:** Gaussiana, PCA e k-means, com e sem acerto de escala, em VOC 2007.
- **Resultado:** k-means sem escala **41,2%**, Gaussiana 50,8%. Com escala por camada e entre camadas: k-means **56,6%**, Gaussiana 53,3%, PCA 52,8%.
- **Regra:** k-means esférico em ativações branqueadas, camada por camada, **e depois igualar a velocidade de mudança de cada camada**. Sem esse acerto o filtro bom perde do aleatório.
- **Tradução:** minha sonda padroniza o vetor e por isso **não vê esse efeito**. É o maior risco do híbrido: um banco novo pode ganhar na sonda e perder no treino. Vale a regra F3 do R9.
- **Discordância:** eles acham k-means melhor que PCA. Aqui PCA foi melhor na L3 (0,880 / 0,792 contra 0,827 / 0,700, split 1). Só com treino saberemos.

### 2.7 Dundar, Jin, Culurciello. *Convolutional Clustering*. ICLR 2016 workshop (lido)
- **Mede:** k-means convolucional em STL-10, 2 e 3 camadas.
- **Resultado:** 74,1% com 3 camadas, contra 60,1% de Coates & Ng. Ligações aleatórias entre camadas 65,4%, aprendidas com rótulo 67,1%.
- **Regra:** o k-means comum gera filtros que são a mesma coisa deslocada. Use janela 2× maior e tire o trecho que mais ativa, para cair alinhado no centroide.
- **Tradução:** vale para 5×5, não para 1×1, onde não há deslocamento. Parente do `redundancy_cos` / `redundancy_merge_seeds`. O passo das ligações usa rótulo: fora da nossa regra para SSL.

### 2.8 Mairal. *End-to-End Kernel Learning with Supervised CKNs*. NIPS 2016 (lido)
- **Regra:** em cada camada, patches com norma 1 → k-means esférico → filtros. Depois o backprop refina. No treino, pré-condiciona pela covariância inversa das patches da camada.
- **Tradução:** é o precedente mais próximo do "híbrido": cada camada nasce de patches da anterior e o gradiente vem depois.

### 2.9 Thiry, Arbel, Belilovsky, Oyallon. *The Unreasonable Effectiveness of Patches*. ICLR 2021 (lido)
- **Resultado:** com patches branqueados ao acaso, 1 camada chega a resultado competitivo em CIFAR. Trocar patches por ruído branco custa 8,1 pontos. Treinar o dicionário de ponta a ponta não ajudou (−0,1).
- **Regra:** patch + whitening já é quase tudo. Mesmo depois de branquear, a dimensão intrínseca fica em ~10% da total.
- **Tradução:** apoia o branco na 1×1. Nota: lá o patch vence o ruído; aqui só vence depois de branqueado.

### 2.10 Comparação, como pedido: LSUV e Delta-orthogonal

- **Mishkin, Matas. *All You Need Is a Good Init*. ICLR 2016 (lido).** Regra: base ortonormal, depois igualar a variância de saída a 1, camada a camada.
- **Xiao et al. *Dynamical Isometry and a Mean Field Theory of CNNs*. ICML 2018 (lido).** Regra: ortogonal no centro do kernel, zero no resto. Treina 10.000 camadas sem atalho.
- **Medi (split 1):** 1×1 LSUV 0,918 / 0,848, Delta 0,912 / 0,851, aleatório 0,930 / 0,838: empate. L3 0,822 / 0,745 e 0,815 / 0,745: abaixo da PCA. Pilha Delta: 0,836 → 0,831 → 0,760 em eggs.
- **Conclusão:** servem para o gradiente passar, não para escolher filtro. Os dois nascem sem olhar o dado. Não substituem o SPiFiL; a escala do LSUV é útil como passo final.

---

## 3. As outras ideias pedidas

| ideia | insight | prevê para nós | teste barato |
|---|---|---|---|
| Saxe et al., ICML 2011 (lido) | Arquitetura sozinha faz muito: 53% com aleatório contra 54,2% treinado (Caltech101, citado no artigo) | Busca de arquitetura pode usar banco sem treino | Esta sonda é isso |
| Bengio et al., NIPS 2006 (lido; o ano citado era 2007) | Guloso supervisado é "guloso demais" e descarta informação; guloso sem rótulo preserva | Reconstrução preserva, por isso a concat tem o que recuperar | Já visto em 1.3 |
| Belilovsky et al., ICML 2019 (lido) | Camada a camada escala: top-5 79,7 (k=1), 86,3 (k=2), 88,7 (k=3). Cabeça auxiliar mais funda ajuda muito | Decoder/cabeça rasa por camada limita o que a camada aprende | Sem chave para k>1 |
| Vincent et al., JMLR 2010 (lido) | Representação igual ou maior que a entrada reconstrói com identidade; reconstruir bem não basta | Mapa 1280×24×24 tem 6× mais números que a imagem: o piso é copiar | `train_aug` com perturbação |
| Schoenholz et al., ICLR 2017 (lido) | Em rede sem treino, duas entradas podem convergir para correlação 1 com a profundidade | Explica a pilha de patch cru | Medir correlação por camada |
| Balduzzi et al., ICML 2017 (lido) | Sem atalho, a correlação dos gradientes decai exponencial | Depois da L4, atalho antes de mais filtros | `residual_gain` |
| He et al., CVPR 2016 (lido) | Mais camadas sobem o erro de treino, e não é sobreajuste | Idem | idem |
| Luo et al., NIPS 2016 (lido) | Campo efetivo cresce com √n e encolhe 1/√n relativo; cresce linear com k | k grande rende mais que camada a mais: casa com k3 0,875 → k5 ~0,900 | `layer_kernels` |
| Bruna, Mallat, TPAMI 2013 (lido) | Mais de 99% da energia está em caminhos de tamanho ≤ 3 | 3 cascatas bastam: casa com o pico em L3–L4 | Já visto |
| Oyallon et al., ICCV 2017 (resumo) | Camadas iniciais fixas + rede aprendida vencem com pouco dado | Apoia o híbrido a 1–5% | Grade de pct |
| Net2Net, ICLR 2016 (lido) | Crescer preservando a função; copiar exige ruído para quebrar a simetria; treinar tudo é seguro | Cópias exatas da 1×1 (R4) não ajudam sem ruído | `layerwise_backprop` |
| Karras et al., ICLR 2018 (lido) | Camada nova entra aos poucos, com α de 0 a 1, tudo treinável | Mesma ideia do `residual_gain` | idem |
| VGG, ICLR 2015 (lido) | Rede funda iniciada pela rasa, sem baixar lr das camadas herdadas | Casa com bp 0→0,1 (HAL:216-217) | Já visto |
| Inception, CVPR 2015 (lido); Arora et al., ICML 2014 (resumo) | Construir camada agrupando neurônios correlacionados; 1×1 para reduzir dimensão | Mesma família do 2.3 | `knn_smooth` |
| DenseNet, CVPR 2017 (lido) | Toda camada vê as anteriores; camadas estreitas somam ao "conhecimento coletivo" | Concat de 1.3 | `gem_dense` |
| Sánchez et al., IJCV 2013 (lido) | Potência `sign(z)√|z|` e L2 antes do classificador | Já na leitura (`_pw`) | Já feito |
| Bag of words, SPM (de memória; SPP-net lido) | Pooling em pirâmide guarda posição | Briga com TTA (R9) | Fechado |
| BagNet, ICLR 2019 (resumo) | 33×33 px de campo já dão 87,6% top-5 | Campo de 40 px cobre o objeto: profundidade não é por campo | Já visto |
| Masci et al. 2011, ReZero, SkipInit (de memória) | Autoencoder convolucional empilhado; ganho do atalho partindo de zero | Parentes de `residual_gain`; **não conferi** | — |

---

## 4. Ordem por insight sobre custo

Custo em hora de slot é conta minha: 4 estágios × 60 épocas × ~15 s ≈ 1 h por braço (J1: 8–22 s por época). Cada linha de treino abaixo é 2 datasets × 3 splits.

| # | teste | chaves | custo | previsão |
|---|---|---|---|---|
| 1 | **Só SVM nos 5 braços já treinados** | YAMLs prontos `eval_svm_disc_g20_e_aug_{wh,pn,div,knn,ipc3}_val_gemtta.yaml` | só CPU | `wh` piora ou empata na L3; `div` e `ipc3` empatam. Atenção: s3 incompleto (HAL:162-168) |
| 2 | **Curva de largura sem treinar** | YAMLs prontos `eval_svm_disc_sl_c{160,320,640,1920}_e60_val_gemttapwr.yaml` | só CPU | eggs satura perto de 320; protozoan sobe até 640+ |
| 3 | **Encher a 1×1 com SPiFiL cru** | `images_per_class: 4`, `fill_random_last: false` | 6 h + controle 6 h | Perde perto de 0,05 / 0,08 no início; o treino recupera parte |
| 4 | Idem, branco e diverso | + `whiten: true`, `select_diverse: 0.9` | 6 h | 1×1 empata, mas `whiten` é global e deve ferir a L3 |
| 5 | Profundidade com atalho | `layerwise: 7`, listas do exemplo de J1, `residual_gain: 0.1` | 12 h | "Só a última" para de cair |
| 6 | Mesma rede, leitura dupla | `embed_mode: gem_dense` (já existe, `eval/svm.py:306`) | só CPU | Sobe mais em rede funda; na L3+L4 empatou (R9) |
| 7 | Sementes no mapa de ativação | `spifil_in_feature: true`, `n_superpixels: 144` (`growth_g5_in_feature.yaml`) | 6 h | Mais tokens, mesmo problema se ficar cru |
| 8 | Sair do piso da reconstrução | `train_aug` com ruído/blur (valores em p1/Y1.md) | 6 h | L3 muda mais; 1×1 pouco |

Para a linha 3: `images_per_class` é 4 em eggs (8 classes), 5 em cysts e 13 em larvae, para dar 1280 ÷ classes sementes por classe a 50 superpixels. Os 12 datasets precisam dessa conta um a um.

**Ressalva da linha 3 e 4:** `images_per_class` e `whiten` valem para todas as camadas (`GROW_KNOBS`, `experiments/ray/runners/growth.py:151`). Mudam L1–L3 junto. Por YAML não dá para isolar a 1×1. É isso que pede a chave nova abaixo.

---

## 5. Chaves novas necessárias (sem código aqui)

Duas bastam para o teste que importa:

1. **`layer_bank`** em `runner_args`, lista com um valor por camada: `spifil | pca | pca_pol | white_pol | random`.
   Exemplo: `layer_bank: [spifil, spifil, pca_pol, white_pol]`.
   Mora em `flim/spifil.py`, no mesmo ponto do `fill_random` (`:971`), usando as `patches` que o `Learner` já tem. O pacote SPiFiL não muda: o superpixel continua sorteando, só o último passo troca. Entra em `RUNNER_ARGS` e `GROW_KNOBS`.
2. **`embed_mode: gem_cat`** em `eval/svm.py`: GeM de todas as camadas concatenado. `gem_dense` já faz isso com duas. Fica abaixo de 2000-d: 32+48+64+1280 = 1424.

Três cuidados para quem implementar:
- **Dois sinais dobram a entrada da camada seguinte:** a 1×1 passa a ler 128 canais.
- **Tokens:** na 1×1 precisa de ~6.000 (1.600 só empata). Uma imagem dá 576 posições a 24×24, então 8 imagens já dão 4.600 sem pedir mais imagens, se usar todas as posições em vez de só as sementes.
- **Escala:** acertar a escala do banco novo antes de treinar (2.6, F3). A sonda não mede isso.

**Decisão para vocês:** a regra "SPiFiL inicializa toda camada" aceita PCA das patches do SPiFiL? Cada filtro é combinação dos patches dos superpixels, mas não é mais um patch. Se não aceitar, sobra `white_pol` também na L3, e ali o branco em 5×5 é o caso perigoso.

---

## 6. Limites

- **Sem backprop em cima de nenhum banco novo.** Nenhum número aqui é κ do modelo final.
- Regressão logística sem TTA e sem SVM: o κ absoluto é mais baixo que o oficial. A própria J1 diz que sonda e SVM divergem em família aleatória.
- O solver avisou que não convergiu em parte dos braços (200–300 iterações). É parte do ruído de ±0,02.
- Só 2 datasets com replicação (`helminth_eggs`, `protozoan_cysts`), objeto em fundo preto. **Nada de aid, flowers, remote_sensing.**
- Checkpoints oficiais usam máscara e `pre`-imagens, e a nova regra não usa. Os tokens que sorteei são uniformes no mapa, não sementes de superpixel: a linha "1.600" aproxima o SPiFiL, não o reproduz.
- Fundo preto domina os tokens uniformes. Isso pesa contra os patches crus; superpixels equilibram melhor as regiões, e a perda real pode ser menor.
- Só 50% dos dados. Nada sobre 1–5%.
- Grupos de canais, LSUV, Delta, k-means esférico e largura vêm só do split 1.
- Bug meu no meio do caminho (`repeat` no bias de um braço), corrigido e relançado. Nenhum número acima vem da versão com erro.

**Busca:** o grafo achou `grow`, `spifil.py`, `schema.py`, `growth.py`. Não resolveu o nó `LAYERWISE_SPECS` nem números citados em relatório; para esses usei grep.

**Arquivos:** sondas e logs em `/tmp/claude-1033/-dados-home-moliveira-Scalable-Hybrid-FLIM/990a33ea-4327-4228-ac99-2136fb213621/scratchpad/law/classic_tools/` (`bank.py`, `deep.py`, `pol.py`, `rep.py`; replicação em `rep_*.log`). Textos dos artigos em `.../scratchpad/law/src/<nome>/p.txt`.

## Fontes

Coates 2011: https://proceedings.mlr.press/v15/coates11a.html · Encoding vs training: https://icml.cc/2011/papers/485_icmlpaper.pdf · Selecting RF: https://cs.stanford.edu/~acoates/papers/coatesng_nips_2011.pdf · K-means: https://www-cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf · PCANet: https://arxiv.org/abs/1404.3606 · Krähenbühl: https://arxiv.org/abs/1511.06856 · Conv clustering: https://arxiv.org/abs/1511.06241 · CKN: https://arxiv.org/abs/1605.06265 · Thiry: https://arxiv.org/abs/2101.07528 · LSUV: https://arxiv.org/abs/1511.06422 · Delta: https://arxiv.org/abs/1806.05393 · Saxe: https://icml.cc/2011/papers/551_icmlpaper.pdf · Bengio: https://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks · Belilovsky: https://arxiv.org/abs/1812.11446 · Vincent: https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf · Schoenholz: https://arxiv.org/abs/1611.01232 · Balduzzi: https://arxiv.org/abs/1702.08591 · ResNet: https://arxiv.org/abs/1512.03385 · Luo: https://arxiv.org/abs/1701.04128 · Scattering: https://arxiv.org/abs/1203.1513 · Hybrid scattering: https://arxiv.org/abs/1703.08961 · Net2Net: https://arxiv.org/abs/1511.05641 · Progressive GAN: https://arxiv.org/abs/1710.10196 · VGG: https://arxiv.org/abs/1409.1556 · Inception: https://arxiv.org/abs/1409.4842 · Arora: https://arxiv.org/abs/1310.6343 · DenseNet: https://arxiv.org/abs/1608.06993 · SPP: https://arxiv.org/abs/1406.4729 · Fisher vector: https://pure.uva.nl/ws/files/19494509/SanchezIJCV2013.pdf · BagNet: https://arxiv.org/abs/1904.00760
