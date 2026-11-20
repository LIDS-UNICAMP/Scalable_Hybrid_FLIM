# Agente 2 — Preparação: literatura e desenho do SPiFiL abstrato (antes do gate do Agente 1)

> Data: 16/11/2026. Autor: Agente 2 (fase de preparação).
> **Escopo desta fase:** só literatura e desenho. Nada foi implementado no código de produção, nada foi treinado, nenhuma GPU foi usada.
> **Status científico:** SPiFiL-1, 2, 3 e 4 são **propostas não testadas**. Nenhuma frase deste documento afirma que eles funcionam. Todo "espera-se" é hipótese, com critério de falsificação na §9.
> **Régua:** toda métrica de seleção citada aqui é κ de **validação** na leitura ORIG (`core.svm.fit_svm`, C=100, sem scaler, embedding 1280 `head(encoder(x))`), conforme R1/R2 e a missão. O teste não entra em nada deste plano, exceto o relato final de um candidato já aprovado.
> **Dependência:** as arquiteturas-semente, o protocolo (`evaluation_contract.md`) e a priorização vêm do gate do Agente 1. Onde este plano precisa deles, há um marcador `[A1]`.
> **Decisão do usuário (MISSION.md, linhas 7–9), válida em todo este desenho:** todo backbone tem cabeça com embedding de dimensão **≥ 1280**. Ela nunca é removida nem reduzida. O custo dela **conta** no orçamento Bk; para caber no orçamento, otimiza-se a **forma** da cabeça (pooling moments/meanstd, projeção de baixo rank, bilinear compacto), nunca a dimensão. Os métodos SPiFiL-0..4 deste documento inicializam só os **blocos do encoder**; a cabeça (`EmbeddingHead`, ≥ 1280) é a mesma em todos os braços de uma comparação (§6, §9.1).

## Sumário

0. [Fontes lidas e lacunas](#0-fontes-lidas-e-lacunas)
1. [Literatura (9 papers)](#1-literatura)
2. [O que o código faz hoje (SPiFiL-0 real) e achados relevantes](#2-o-que-o-código-faz-hoje)
3. [Interface `KernelInitializer`](#3-interface-kernelinitializer)
4. [SPiFiL-0 dentro da interface, sem modificar o original](#4-spifil-0-dentro-da-interface)
5. [SPiFiL-1/2/3/4: desenho, pseudocódigo, custo e rótulos](#5-spifil-1234)
6. [Aprendizagem local e global](#6-aprendizagem-local-e-global)
7. [Diagnósticos por camada](#7-diagnósticos-por-camada)
8. [Testes unitários e de integração](#8-testes)
9. [Matriz mínima de ablação, orçamento e critério de decisão](#9-matriz-mínima-de-ablação)
10. [Riscos e perguntas para o gate do A1](#10-riscos-e-perguntas)
11. [Referências](#11-referências)

---

## 0. Fontes lidas e lacunas

| Paper | Como foi lido | Observação |
|---|---|---|
| De Souza & Falcão 2020 (FLIM) | texto completo (PDF arXiv 2008.03549, extraído localmente) | O DOI da missão (IEEE Xplore 9265976) não abriu (HTTP 202 vazio). O arXiv corresponde ao artigo do SIBGRAPI 2020, pp. 464–468, confirmado na SBC SOL. A página do arXiv também lista o DOI 10.1109/LGRS.2020.3020098 (versão em periódico, não verificada). |
| PCANet (Chan et al.) | PDF completo (arXiv 1404.3606v2, extraído) + leitura completa do A1 (`agent1/parts/A1d_literatura.md` §2) | — |
| Coates, Lee & Ng 2011 | PDF completo (PMLR v15, extraído) + A1d §1 | — |
| SoftHebb (Moraitis et al. 2022) | HTML do arXiv + PDF extraído | Só MNIST/F-MNIST e um CIFAR-10 preliminar; **o follow-up convolucional (Journé et al., ICLR 2023, arXiv 2209.11883), que é o que `softhebb.py` implementa, não foi lido nesta fase.** Não cito números dele. |
| SwAV (Caron et al. 2020) | HTML + PDF | — |
| VICRegL (Bardes et al. 2022) | HTML do arXiv 2210.01571 | Números de custo vieram do extrator de página; conferir no PDF antes de citar em relatório final. |
| I-JEPA (Assran et al. 2023) | HTML + PDF (escalas de máscara e EMA conferidas no texto) | — |
| MAE (He et al. 2022) | HTML + PDF | A página do arXiv lida não mostra o venue; o CVPR 2022 vem da missão. |
| Belilovsky et al. 2019 | PDF completo (extraído) + A1d §3 | — |

Os números marcados com † vieram de um resumo automático da página. Confira no PDF antes de usá-los em `abstract_spifil_study.md`.

---

## 1. Literatura

Formato de cada item: **princípio · adaptação CNN/SPiFiL · custo · limitações · hipótese verificável**. As hipóteses são numeradas H-L1…H-L9 e ligadas a braços da matriz da §9.

### 1.1 De Souza & Falcão 2020 — FLIM (Feature Learning from Image Markers)

- **Princípio.** O usuário escolhe poucas imagens (4, duas por classe, a partir de uma projeção t-SNE) e desenha marcadores rotulados nas regiões que discriminam as classes. Para cada classe *i*, os patches k×k×m em volta dos pixels marcados formam um conjunto P_i. Um **k-means por classe** agrupa P_i. Antes disso, todos os patches são centrados pela média global e padronizados pelo desvio (a "marker-based batch normalization"). **Cada centróide vira um filtro, com ‖vec(F)‖ = 1.** A centralização faz o filtro responder positivamente ao padrão da sua classe e negativamente aos outros (o filtro é a normal de um hiperplano). Depois vêm ReLU e max-pool, e cada camada é treinada sozinha, sem backprop.
- **Números.** Coconut-tree (aéreo, 2 classes, 90×90, LAB [0,1]); uma camada Conv(60×7×7), MaxPool 3×3 s4. F-score: FLIM+SVM 0,838 ± 0,017 contra VGG-FE+SVM 0,826 ± 0,006 (VGG-16 ajustada no ImageNet). FLIM+MLP 0,854; VGG-FT 0,851; a mesma arquitetura do zero por backprop (CL+MLP) 0,811; FLIM com fine-tuning (FLIM-FT+MLP) 0,832, **abaixo** do FLIM sem fine-tuning. Marcadores em 4, 8 ou 12 imagens: 0,838, 0,844 e 0,844 (inconclusivo, segundo os autores). SVM com C = 0,01.
- **Adaptação ao SPiFiL.** O SPiFiL troca o marcador desenhado por sementes de superpixel em máscaras e o k-means por **seleção** de patches individuais (Fisher/Mahalanobis + diversidade). O FLIM original é, portanto, o parente direto do SPiFiL-1 supervisionado: **centróide por classe em vez de patch escolhido**. O `knn_smooth` de `flim.spifil.GrowOneLayer`, que já existe, é um "k-means-lite" nessa direção (§2.2).
- **Custo.** k-means sobre alguns milhares de patches: segundos em CPU.
- **Limitações.** Uma única camada, um dataset binário, 3 splits. Depende do usuário (seleção de imagens e marcadores). Não diz nada sobre camadas profundas, e o efeito do número de imagens marcadas não foi conclusivo.
- **H-L1.** *Na camada 1 (stem), centróides por classe (S1-sup, estilo FLIM) dão κ_val ≥ patches selecionados por Fisher (S0) com o mesmo número de filtros, em RS (aéreo, como coconut).* Falsifica se Δκ_val(S1-sup − S0) < 0 em 3/3 seeds.

### 1.2 PCANet (Chan et al., TIP 2015)

- **Princípio.** O filtro do estágio 1 é um dos L1 autovetores principais de XXᵀ, em que X junta todos os patches k1×k2 com a média de cada patch removida. O estágio 2 repete a PCA sobre os L1 mapas. A saída passa por hashing binário, histogramas por bloco e SVM linear. Não há não linearidade entre estágios. O padrão é k = 5 e L1 = L2 = 8.
- **Números (A1d §2).** CIFAR-10 com embedding PCA-1280 + SVM linear: 77,14% (multiescala 78,67%). MNIST: erro de 0,66%. "Dois estágios bastam; mais profundo não melhora necessariamente." Sem remover a média do patch, o desempenho "degrada significativamente".
- **Adaptação.** PCA é o "AE linear" mínimo e o piso natural para "filtro derivado dos dados, sem gradiente". Vira o sub-braço `S1-pca` (seleção = top-C_out autovetores no espaço branqueado, ou seja, sem branqueamento, já que ZCA + PCA dá identidade; ver §5.1). Serve também de controle do AE local: o AE não linear deveria bater a PCA do mesmo bloco.
- **Custo.** Covariância O(N·D²) e decomposição O(D³). Para D = C_in·k² ≤ 1368, leva segundos.
- **Limitações.** Testado só em imagens alinhadas e pequenas. A dimensão da saída explode com os estágios. Os filtros PCA ficam ordenados por variância, o que não significa discriminação.
- **H-L2.** *Nas camadas ≥ 2, o rank efetivo do banco S0 é menor que o de um banco PCA do mesmo tamanho nos mesmos dados, e a diferença de rank se correlaciona com Δκ.* Mede-se só na leitura init-only (barata).

### 1.3 Coates, Lee & Ng 2011 (AISTATS)

- **Princípio.** Uma camada de features aprendida sem rótulos: patches w×w aleatórios, normalização por patch (média e desvio dos próprios elementos: brilho e contraste), **ZCA opcional**, um aprendiz (sparse AE, sparse RBM, k-means ou GMM) com K bases, mapeamento por patch, sum-pooling em quadrantes e SVM linear. O k-means "triangle" usa f_k(x) = max{0, μ(z) − z_k}, com z_k = ‖x − c_k‖₂.
- **Números (texto, Fig. 3; A1d §1).** O melhor por validação cruzada foi branqueamento, stride 1, campo receptivo de 6 px e 1600 features: 77,9% no CIFAR-10 e 79,6% com 4000 features. Com k-means, o branqueamento é **crucial** ("os algoritmos de clustering não lidam com as correlações dos dados"). O triangle supera o hard por cerca de 9 pp. Mais K, sempre melhor; stride 2 já custa ≥ 3 pp.
- **Adaptação.** É a receita direta do SPiFiL-1 label-free: patches densos de muitas imagens de treino → ZCA → k-means esférico → centróides como filtros, **com o ZCA dobrado nos pesos** (§5.1). O triangle pode virar uma ativação candidata do bloco, como variante registrada à parte (não neste plano mínimo).
- **Custo.** k-means O(N·K·D·iter). Com N = 10⁵, K = 3·C_out ≈ 450, D ≈ 576 e 30 iterações, dá cerca de 8·10¹¹ FLOPs: segundos em GPU.
- **Limitações.** Uma camada, 32×32 px. A normalização por patch é não linear e não cabe exatamente numa conv (só a parte linear, a remoção da média, é dobrável; `GrowOneLayer` já faz essa aproximação).
- **H-L3.** *Dentro do S1, o ZCA melhora κ init-only e o rank efetivo do banco em relação ao S1 sem ZCA, em todas as profundidades.* É um sub-ablation barato (init-only).

### 1.4 SoftHebb (Moraitis et al., Neuromorph. Comput. Eng. 2022)

- **Princípio.** WTA suave. Com u_k = w_k·x (pesos e entrada normalizados), y_k = softmax_k((u_k + w_0k)/T) e a regra Δw_ik = η·y_k·(x_i − u_k·w_ik), a regra é local, sem rótulos, e minimiza a cross-entropy de uma mistura. Em equilíbrio, ‖w_k‖ → 1 e w_0k → T·ln P(C_k).
- **Números.** MNIST: 96,31 ± 0,06% (1 camada) e 97,80 ± 0,02% (2 camadas, com perceptron supervisionado); MLP com backprop: 98,65%. CIFAR-10 preliminar: 50,27% (hard WTA 49,78%). Robustez a ruído e perturbações adversariais maior que a do MLP†.
- **Adaptação.** `ga_search/modehb/softhebb.py` já implementa a versão convolucional do follow-up (soft anti-Hebbian, η adaptativo por neurônio) como **refino depois do SPiFiL**, em fases sequenciais (SPiFiL → SoftHebb → AE local). No SPiFiL-1, o SoftHebb entra como refino opcional dos protótipos: é um k-means online "suave", coerente com protótipos.
- **Custo.** Um passe por época: conv + softmax por posição, O(B·K·H·W·D). Já existe chunking (`SOFTHEBB_CHUNK_ELEMS`).
- **Limitações.** O paper lido não testa convolução nem profundidade, e nos datasets mais difíceis os retornos são decrescentes. A regra puxa os filtros para a norma 1 sem bias, o que **muda a escala** em relação ao SPiFiL-0 (cujos filtros têm o z-score dobrado e norma ≠ 1). A interação com o bias dobrado precisa ser verificada.
- **H-L4.** *O refino SoftHebb (1 época) aumenta a entropia de vencedores e o rank efetivo dos protótipos S1 sem reduzir κ_val (Δκ ≥ −0,005).* Se reduzir, o SoftHebb sai do SPiFiL-1.

### 1.5 SwAV (Caron et al., NeurIPS 2020)

- **Princípio.** Clustering online com troca de predição: o código q de uma vista (atribuição aos K protótipos, calculada por Sinkhorn-Knopp com **equipartição**, ε = 0,05 e 3 iterações) é predito pelas features da outra vista: L = ℓ(z_t, q_s) + ℓ(z_s, q_t), com ℓ = cross-entropy contra softmax(zᵀc_k/τ), τ = 0,1. K = 3000. Multi-crop 2×224 + 6×96. A equipartição evita o colapso sem pares negativos.
- **Números.** ResNet-50 linear no ImageNet: 75,3%. Com 1% de rótulos: 53,9 top-1. Custo: 6h15 por 100 épocas em 64 V100 (multi-crop)†.
- **Adaptação.** No SPiFiL-2 os protótipos são **os próprios filtros candidatos**. A consistência de atribuição entre vistas, em **posições correspondentes**, vira critério de seleção (sem gradiente) e, opcionalmente, loss de um refino curto. A equipartição de Sinkhorn impede que poucos filtros "ganhem" todas as posições, o que ataca os canais mortos que o A1 vai medir.
- **Custo.** Sinkhorn sobre (posições × K) é desprezível. O custo dominante é a conv dos K candidatos nas V vistas.
- **Limitações.** Precisa de batch grande ou fila (o próprio paper usa fila de cerca de 3000 com batch pequeno). Um ε alto leva à solução trivial. É um objetivo global (de imagem), não local; a adaptação para posições é nossa.
- **H-L5.** *Filtros escolhidos por consistência de atribuição entre vistas com equipartição (S2) têm fração de canais mortos menor e estabilidade maior que S0, e o ganho em κ_val é maior em RS (invariante a rotação) que em Flowers.*

### 1.6 VICRegL (Bardes, Ponce & LeCun, NeurIPS 2022)

- **Princípio.** VICReg (invariância λ = 25, variância μ = 25, covariância ν = 1) aplicado ao vetor global e a **pares de vetores locais** antes do pooling. Os pares são casados **por posição** (vizinho mais próximo usando a geometria conhecida da transformação) e **por feature** (vizinho mais próximo no embedding), mantendo os top-γ pares (γ = 20). L = α·ℓ_global + (1 − α)·(L_pos + L_feat).
- **Números.** ResNet-50, α = 0,75: ImageNet linear 70,4 e VOC linear 55,9 (VICReg: 47,8). Ablação: só posição 73,5/58,9; só feature 73,6/57,7; ambos 73,6/60,3 (ImageNet/VOC). Custo: +18% de tempo e +39% de memória sobre o VICReg†.
- **Adaptação.** Dá ao SPiFiL-2 (i) o **casamento por posição usando a geometria da transformação**: `make_views` gera `theta` afim e nós a reconstruímos (§5.2); (ii) os termos de variância e covariância como anti-colapso do refino opcional `S2-ref`. O trade-off α (local × global) do paper avisa que excesso de invariância local pode custar acurácia linear de imagem, que é o nosso ORIG.
- **Custo.** Matching O(P²) por par de imagens no casamento por feature. Só usamos o casamento por posição, que é O(P).
- **Limitações.** O campo receptivo das features "locais" de uma ResNet já cobre a imagem toda. No nosso bloco recém-adicionado, o RF é pequeno, o que é mais favorável à localidade, mas não foi testado.
- **H-L6.** *O score de estabilidade casado por posição (Pearson das respostas em posições correspondentes) de um filtro se correlaciona positivamente (Spearman > 0,3) com a sua contribuição marginal ao κ_val init-only, em RS e Flowers.* É um teste de validade do critério, antes de usá-lo para selecionar.

### 1.7 I-JEPA (Assran et al., ICCV 2023)

- **Princípio.** A partir de um bloco de contexto (escala 0,85–1,0, aspecto 1, sem as regiões-alvo), um preditor estreito prevê as **representações** (não os pixels) de M = 4 blocos-alvo (escala 0,15–0,2, aspecto 0,75–1,5) dadas pelo encoder-alvo EMA (momentum 0,996 → 1). A loss é L2 médio nos alvos. Não usa augmentations manuais.
- **Números.** ViT-H/14, 300 épocas: 79,3 linear no IN-1K (MAE ViT-H 1600 épocas: 77,2). Treino em menos de 1200 GPU-h (16 A100 em menos de 72 h). Em tarefas locais (Clevr/Count) fica **abaixo** do MAE (86,7 × 90,5).
- **Adaptação.** O SPiFiL-3 aplica o princípio na **grade de features do prefixo congelado**: um encoder-latente pequeno e local (contexto mascarado → alvo EMA) aprende um latente preditivo. Os protótipos latentes viram kernels por uma **transformação latente→kernel explícita** (regressão ridge do patch k×k×C_in para a atribuição ao protótipo; §5.3). A escala dos blocos é adaptada a grades de 14–56 posições.
- **Custo.** Encoder-latente de 2 convs e preditor de 2 convs na grade do prefixo: na ordem de 10¹⁴ FLOPs para 3000 passos, ou seja, minutos (§5.3).
- **Limitações.** Sem EMA ou anti-colapso há risco de solução trivial. Com grade pequena (14×14), os blocos-alvo ficam com poucas posições. A versão "pixel" degrada a semântica, segundo os próprios autores.
- **H-L7.** *Nas camadas ≥ 2, o S3 (latente preditivo treinado) supera o S3-rand (o mesmo pipeline com encoder-latente aleatório, não treinado) em κ_val pós-treino.* Se não superar, a parte "preditiva" do S3 não está contribuindo, e qualquer ganho viria só do mapeamento ridge e dos protótipos.

### 1.8 MAE (He et al., CVPR 2022)

- **Princípio.** Mascarar 75% dos patches ao acaso; o encoder vê só os visíveis; um decoder leve reconstrói os pixels (alvo normalizado por patch); a loss só conta os mascarados.
- **Números.** Fine-tuning ViT-H 86,9 (87,8 a 448 px). **Linear probe fraco**: ViT-L 73,5–75,8 (contrastivos e JEPA são mais separáveis linearmente). Ablação: alvo normalizado ajuda; máscara em grade bate máscara em bloco no linear (66,0 × 63,9).
- **Adaptação.** O `local_ae.py` já tem o alvo `masked` (máscara em blocos 2×2, ratio 0,4, alvo normalizado por patch via `masked_patch_mse`). É um braço de **aprendizagem local**, não de inicialização. O paper sugere que reconstrução de pixels favorece fine-tuning e não sonda linear, que é o nosso ORIG.
- **Custo.** Igual ao AE local atual.
- **Limitações.** O ganho do MAE vem da escala (ViT, 1600 épocas). Num bloco local de CNN, a máscara alta pode só ensinar interpolação.
- **H-L8.** *No AE local, subir mask_ratio de 0,4 para 0,75 não melhora o κ_val ORIG (|Δ| < 0,01).* É um braço secundário, rodado só se o A1 apontar o AE como gargalo.

### 1.9 Belilovsky, Eickenberg & Oyallon 2019 (ICML) — Greedy Layerwise Learning

- **Princípio.** Cada novo bloco (conv 3×3 + ReLU, downsampling **invertível** space-to-depth) é treinado sozinho, com os anteriores congelados, para resolver um problema auxiliar **supervisionado** raso: k = 1 (média espacial + linear) ou k = 2, 3 (camadas extras no classificador auxiliar).
- **Números.** ImageNet k = 1: 58,1/79,7 (> AlexNet 56,5/79,1); k = 3: 69,7/88,7. VGG-11 treinada camada a camada (k = 3) dá 67,6 top-1 contra 67,9 ponta a ponta. A separabilidade linear cresce com a profundidade, mas **com k = 1 satura num patamar mais baixo**.
- **Adaptação.** Sustenta o pipeline atual (SPiFiL por bloco + AE local + prefixo protegido) e a sonda linear por bloco como **diagnóstico**. Avisa que um objetivo local "raso" satura cedo com a profundidade, e é por isso que o AE local (reconstrução, ainda mais distante do κ) pode ser o fator que limita a profundidade.
- **Custo.** Por bloco, menor em memória; o tempo total não é menor que o ponta a ponta.
- **Limitações.** O objetivo local é supervisionado, e o nosso é label-free. Arquiteturas VGG grandes, sem skip, com datasets inteiros.
- **H-L9.** *Na varredura de profundidade (3, 4, 6), a curva κ_val(profundidade) do S0 + AE satura (ganho < 0,01 por bloco) antes da curva do melhor SPiFiL-1/2/3 + AE.* Se as duas saturarem juntas, o limite está no objetivo local, não na inicialização. Esse é um resultado negativo útil para o Agente 3.

---

## 2. O que o código faz hoje

### 2.1 O pipeline SPiFiL-0, lido no código

- **Entrada.** `init.init_block(genome, layer_idx, prefix, samples, config)` (`ga_search/modehb/init.py:180`). `samples` são **doadores** (`run_search.spifil_subset`): as primeiras `spifil_images_per_class` (1, 2, 4, 8 ou 16) imagens de **treino** de cada classe, tiradas de um pool fixo (`build_spifil_pool`, que verifica que os doadores estão só no treino; `run_search.py:79-99`).
- **Rotas.**
  - `spifil_ops: false` usa `flim.progressive.fit_new_block` (conv densa k3/k5, sequencial).
  - Largura < n_classes ou `spifil_ops: true` usa `spifil_ops.fit_ops_block` (conv k1/3/5/7, pointwise, depthwise, dwsep, inception, residual).
- **Mecânica** (biblioteca `/dados/home/moliveira/SPiFiL/src/spifil`, nunca modificada):
  1. **Sementes.** SLIC no LAB (ou DISF), medóides por superpixel, até `n_superpixels` = 100 por imagem, **dentro da máscara** (`Cropped(use_mask=not impurities)`), reprojetadas para a grade do prefixo (`flim/spifil.py:635 build_layer0`).
  2. **Patches.** Vetor channel-major de C_in·span² na grade do prefixo (`patches.extract_patches`).
  3. **Score.** `FisherScorer` = média das distâncias às outras classes / média das distâncias à própria classe, com distância de Mahalanobis (Ledoit-Wolf). Depois, ranks por classe.
  4. **Alocação.** `UniformAllocator`: out_channels // n_classes por classe, então **a largura pode encolher**. `AutoAllocator`/`ExactBudgetAllocator` (aproximações registradas) dão a largura exata.
  5. **Seleção.** `DiversitySelector` (α = 0,5, pool_factor = 3): busca gulosa rank × distância mínima, por classe.
  6. **Construção.** `build_filters`: z-score sobre os patches **selecionados** (desvio populacional + 0,01), normalização unitária no espaço z, e a normalização **dobrada** em `weight` e `bias`. O filtro final tem norma ≠ 1 no espaço de entrada.
  7. **Orçamento.** `N ≤ D` gera `BudgetExhausted`, convertido em `InfeasibleCandidate`.
- **Depois do fit.** Norm pós-hoc (BN 1/0; ZCA ajustado por `_fit_zca` com `zca_fit_batches` do treino), SoftHebb opcional (`softhebb: true` no gene), diagnóstico `_log_init_diag`, AE local (`train_local_ae`, com `protect` = escala do lr das convs e `decoder_first` = metade das épocas só com o decoder), e depois o SSL global (`train_global_ssl`, LeJEPA + RealSIGReg, λ = 0,05, V = 4) com WiSE α ∈ {0, …, 1} escolhido na validação.

### 2.2 Achados que afetam o desenho (verificar no gate)

1. **Já existem "pedaços de SPiFiL-1" no repositório, mas fora do caminho do MO-DEHB.** `flim.spifil.GrowOneLayer` (`flim/spifil.py:288-372`) aceita:
   - `whiten`: ZCA nos patches antes de Fisher e seleção;
   - `patch_norm`: Coates, só a parte linear dobrável;
   - `knn_smooth`: o candidato vira a média dos k vizinhos da mesma classe ("k-means-lite");
   - `select_diverse`: NMS por |cos|.

   O `grow()` dobra o ZCA e a escala nos pesos (`flim/spifil.py:910-922`). Já o `fit_new_block` instancia `GrowOneLayer(budget_check)` **sem** essas opções (`flim/progressive.py:94`). Consequência para a honestidade do estudo: precisa haver um braço **S0+** (SPiFiL-0 com essas opções já existentes), para não creditar ao SPiFiL-1 o que o repositório já tinha. Se ligar essas opções exigir mexer em produção, o braço é implementado no módulo novo, compondo a biblioteca (como `spifil_ops` faz).
2. **O gargalo N > D nas camadas profundas é estrutural.** D = C_in·k². Com C_in = 64 e k = 3, D = 576, o que exige mais de 576 sementes. Com 100 superpixels por imagem dentro da máscara, Flowers com ipc = 1 (5 imagens, no máximo cerca de 500 sementes) **não fecha o orçamento** a partir de cerca de 56 canais de entrada. O próprio `seeds_in_features` documenta que, com máscara, `spifil_in_feature` quase nunca fecha. Parte da "insuficiência de patches simples em camadas profundas" (hipótese do A1) pode ser **escassez de candidatos**, não a natureza do patch. Por isso o braço de controle **S0-data** (S0 com o mesmo número de imagens que o S1–S3 usam) é obrigatório.
3. **Fisher + Mahalanobis é O(N²) em memória** (o inventário registra um N×N de cerca de 13 GB e o patch `spifil_oom.py`). O S1–S3 usam N ~ 10⁵ patches com custo O(N·K·D) e nunca formam N×N.
4. **Os rótulos entram no SPiFiL-0 por três canais:** o rótulo da imagem (Fisher, alocação por classe), **a máscara do objeto** (anotação de localização) e a escolha dos doadores por classe. Uma versão label-free precisa desligar os três. Ela pode ser montada com o `DistanceSumScorer` da biblioteca, uma pseudo-classe única e `impurities: true`. Isso é um braço diagnóstico (**S0-lf**), não uma promessa.
5. **`_log_init_diag` usa `self.val_x`** (16 imagens de validação) para medir ativação (`run_search.py:1051`). Não entra no fitness, mas no protocolo novo os diagnósticos de inicialização devem usar um **lote fixo de treino** (separado do usado no fit), para a validação ficar intocada até a seleção.
6. **`make_views` (`ssl_trainers.py:88`) não devolve `theta`.** O SPiFiL-2 precisa da geometria para casar posições. A solução é um gerador irmão no módulo novo que reproduza exatamente a mesma amostragem e devolva `theta` (teste de igualdade bit a bit das vistas; §8). Não se altera `ssl_trainers.py`.
7. **A política de augmentation é única** (`vflip_p` = 0,5, pensada para microscopia). Para Flowers, o flip vertical não é uma invariância natural. Uma política por domínio é uma **variante registrada** (§5.2), nunca uma mudança silenciosa.

---

## 3. Interface `KernelInitializer`

### 3.1 Localização proposta (módulo opcional, nada em produção)

```
ga_search/modehb/abstract_init/
  __init__.py          # registro: {"spifil0": Spifil0Adapter, "spifil1": ..., ...}
  interface.py         # dataclasses, Protocol, validate_init(), assemble_block()
  spifil0.py           # adaptador do SPiFiL atual (só chama código existente)
  dense_patches.py     # amostragem densa, unfold, ZCA (fit/dobra), k-means esférico mini-batch, seleção
  spifil1.py  spifil2.py  spifil3.py  spifil4.py
  views.py             # vistas com theta (réplica de make_views) + correspondências
  diagnostics.py       # §7
tests/abstract_init/   # §8
```

O roteamento em produção fica **desligado por padrão**: uma chave YAML `abstract_init: {method: ...}`. Ausente, o comportamento é bit a bit o atual (há um teste de regressão para isso). O ponto de integração (um hook opt-in em `run_search._grow` ou um runner de ablação separado) é decidido com o Agente 5 sob a R9. Até lá, as ablações podem rodar num runner próprio que reutiliza `train_local_ae`, `train_global_ssl` e `svm_fitness` sem tocá-los.

### 3.2 Tipos

```python
# interface.py — PROPOSTA (não implementado)
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping, Optional, Protocol, Sequence
import torch
from torch import nn, Tensor

LabelPolicy = Literal["supervised", "label_free"]
WeightDict = dict[str, Tensor]   # chaves = nomes do state_dict de SpifilOpsBlock(effective_spec)

@dataclass(frozen=True)
class TrainData:
    """Somente TREINO. Nunca contém validação/teste."""
    donors: Sequence["spifil.data.Sample"]       # doadores SPiFiL (rótulo + máscara), = run_search.spifil_subset
    loader_factory: Callable[[int], Iterable[tuple[Tensor, Tensor]]]
    #   seed -> iterador de (x [B,3,S,S] float32 LAB em [0,1], y [B] int64), split de treino, ordem semeada
    split_key: str                               # deve ser "train"; fit() recusa outro valor
    split_fingerprint: str                       # sha256 da lista ordenada de ids de treino
    n_classes: int
    image_size: int                              # S
    label_policy: LabelPolicy = "supervised"
    mask_policy: Literal["mask", "no_mask"] = "mask"   # máscara = anotação (conta como uso de rótulo)

@dataclass(frozen=True)
class InitBudget:
    max_images: int = 256          # imagens de treino lidas pelo método
    max_patches: int = 100_000     # N máximo
    max_seconds: float = 600.0
    max_gpu_gb: float = 8.0        # acima disso o método divide em blocos ou lança InfeasibleCandidate

@dataclass(frozen=True)
class ParentInfo:                  # só SPiFiL-4 / crescimento
    parent_block: nn.Module        # SpifilOpsBlock do pai (camada mapeada)
    keep: Sequence[int]            # filtros herdados (external_seed.keep_filters)
    in_map: Sequence[int]          # canal de entrada do filho -> do pai (-1 = novo)
    n_new: int                     # nº de filtros novos a inicializar
    inherited_masks: Mapping[str, Tensor]

@dataclass
class InitContext:
    seed: int                      # seed da camada (= genome.seed + layer_idx, como hoje)
    device: torch.device
    layer_idx: int
    prefix_id: str
    budget: InitBudget = field(default_factory=InitBudget)
    config: Mapping[str, Any] = field(default_factory=dict)   # hiperparâmetros do método (hash vai à proveniência)
    parent: Optional[ParentInfo] = None
    genome: Optional[dict] = None  # só o adaptador SPiFiL-0 usa (equivalência com init_block)
    diag_batch: Optional[Tensor] = None   # lote FIXO de treino p/ diagnóstico (nunca validação)
    logger: Any = None
    amp_dtype: Optional[torch.dtype] = None  # só no forward do prefixo; estatísticas sempre em fp32

class KernelInitializer(Protocol):
    name: str          # "spifil0" | "spifil1" | ...
    version: str       # semver do método; muda a cada mudança científica
    def fit(self, prefix: Optional[nn.Module], train_data: TrainData, block_spec: dict,
            context: InitContext) -> tuple[WeightDict, WeightDict, dict]:
        """Retorna (weights, bias, provenance). Não treina o prefixo nem altera a topologia pedida
        sem registrar. Lança flim.progressive.InfeasibleCandidate quando inviável."""
```

`block_spec` é o **ops spec** de `spifil_ops.to_ops_spec(layer, in_channels)`, com as chaves `builder, conv, kernel, in_channels, channels, requested_channels, dw_multiplier, activation, norm, pool, pool_stride, residual, skip, dropout` (e `pool_padding`/`pool_size` nos blocos externos). Para a rota legada (`fit_new_block`), o adaptador converte de/para `init.to_block_spec`.

### 3.3 Shapes exatos por tipo de bloco

Notação: C_in = canais do prefixo, C_out = canais **efetivos**, k = kernel. As chaves são as do `state_dict` de `SpifilOpsBlock` (`spifil_ops.py:173`). A rota legada `fit_new_block` produz `nn.Sequential(conv, [bn], activation, [pool])`, com as **mesmas** chaves `conv.weight`/`conv.bias`.

| `conv` | `weights` | `bias` | Restrições |
|---|---|---|---|
| `conv` (k ∈ {1,3,5,7}; legado k ∈ {3,5}) | `conv.weight` (C_out, C_in, k, k) | `conv.bias` (C_out,) | padding k//2 |
| `pointwise` | `conv.weight` (C_out, C_in, 1, 1) | `conv.bias` (C_out,) | k = 1 |
| `depthwise` (mult. m) | `conv.weight` (C_in·m, 1, k, k) | `conv.bias` (C_in·m,) | groups = C_in; C_out = C_in·m; ordem de grupo: canal c ocupa as linhas [c·m, (c+1)·m) |
| `dwsep` | `conv.depthwise.weight` (C_in, 1, k, k); `conv.pointwise.weight` (C_out, C_in, 1, 1) | `conv.depthwise.bias` (C_in,); `conv.pointwise.bias` (C_out,) | depthwise m = 1; sem ativação entre os dois |
| `inception` (ramos de `inception_split(C_out)`, ordem `b1, b3, b5, pool`) | `conv.branches.b1.weight` (c1, C_in, 1, 1); `conv.branches.b3.weight` (c3, C_in, 3, 3); `conv.branches.b5.weight` (c5, C_in, 5, 5); `conv.branches.pool.proj.weight` (cp, C_in, 1, 1) | um `.bias` (c_i,) por ramo | c1 + c3 + c5 + cp = C_out; os c_i efetivos vão em `effective_spec["inception_branches"]`; o ramo pool vê `MaxPool2d(3,1,1)(x)` |
| `residual: true`, skip `identity` | conv como acima | conv como acima | exige C_in = C_out; y = pool(drop(act(norm(conv x))) + x) |
| `residual: true`, skip `project` | conv + `skip.weight` (C_out, C_in, 1, 1) | conv + `skip.bias` (C_out,) | C_in ≠ C_out |

Geometria de saída. A conv preserva H×W. O pool da biblioteca (3×3, s2, pad 1) leva H a ⌊(H − 1)/2⌋ + 1; o pool externo (`pool_padding` = 0) leva H a ⌊(H − 3)/2⌋ + 1. A norm (BN 1/0, ZCA, DBN) é pós-hoc e não faz parte do retorno.

### 3.4 Contrato (invariantes testadas na §8)

- **I1.** O prefixo não muda: `flim.progressive._fingerprint(prefix)` é igual antes e depois, o prefixo fica em `eval()` com `requires_grad = False` durante o fit, e os parâmetros não recebem gradiente.
- **I2.** `set(weights) ∪ set(bias)` é exatamente o conjunto das chaves conv/skip de `SpifilOpsBlock(effective_spec).state_dict()`, com shapes exatos (§3.3). Tudo float32 e finito.
- **I3.** 1 ≤ C_out efetivo ≤ C_out pedido. Qualquer diferença vai a `provenance["channel_repair"]`, e a topologia **nunca** muda em silêncio.
- **I4.** Determinismo em CPU: os mesmos (seed, dados, config, versão) dão tensores idênticos bit a bit. Em CUDA, tolerância de 1e-5 com `nondeterminism` registrado.
- **I5.** Não há acesso a validação nem teste. `split_key == "train"` é verificado, e os ids lidos são registrados por hash e comparados com o fingerprint do treino.
- **I6.** Se `label_policy == "label_free"`, a saída é **invariante a qualquer permutação dos rótulos** e não lê máscara (`mask_policy == "no_mask"` é forçado), e `provenance["labels_used"] = False`.
- **I7.** Falha determinística gera `InfeasibleCandidate(reason, details)`, sem erro genérico.
- **I8.** Orçamento: se o método passar de `InitBudget`, lança `InfeasibleCandidate("budget: ...")` ou divide em blocos (bit a bit igual, testado).

### 3.5 Proveniência (schema mínimo, serializável em JSON)

```json
{
  "method": "spifil1", "version": "0.1.0", "config_sha256": "...",
  "origin": "abstract_init", "faithful_to": null, "approximations": ["zca_folded", "patch_norm:mean_only"],
  "seed": 1043, "layer_idx": 2, "prefix_id": "...", "prefix_state_sha256": "...",
  "requested_spec": {...}, "effective_spec": {...}, "channel_repair": null,
  "labels_used": false, "label_source": "none | image_labels | donor_masks+image_labels",
  "mask_used": false,
  "data": {"split_key": "train", "split_fingerprint": "...", "n_images": 256,
           "image_ids_sha256": "...", "n_patches": 100000, "patch_dim": 576},
  "timing": {"fit_seconds": 12.3, "peak_gpu_gb": 1.9},
  "diagnostics": {"init": {...}},
  "env": {"torch": "2.7.1+cu128", "numpy": "...", "spifil": "...", "git_sha": "..."}
}
```

### 3.6 Montagem e validação (funções puras)

```python
def validate_init(spec: dict, weights: WeightDict, bias: WeightDict) -> None:
    ref = SpifilOpsBlock(spec, spec["in_channels"]).state_dict()
    conv_keys = {k for k in ref if k.startswith(("conv.", "skip."))}
    assert set(weights) | set(bias) == conv_keys, (set(weights) | set(bias)) ^ conv_keys
    for k, v in {**weights, **bias}.items():
        assert tuple(v.shape) == tuple(ref[k].shape), (k, v.shape, ref[k].shape)
        assert v.dtype == torch.float32 and torch.isfinite(v).all(), k

def assemble_block(spec: dict, weights: WeightDict, bias: WeightDict, device=None) -> SpifilOpsBlock:
    validate_init(spec, weights, bias)
    blk = SpifilOpsBlock(spec, spec["in_channels"])
    with torch.no_grad():
        sd = blk.state_dict()
        for k, v in {**weights, **bias}.items():
            sd[k].copy_(v)
    for p in blk.parameters():
        p.requires_grad_(True)
    return blk if device is None else blk.to(device)
```

---

## 4. SPiFiL-0 dentro da interface

O objetivo é que o adaptador **só chame código existente** e reproduza bit a bit o que `init.init_block` faz hoje. Nada em `init.py`, `spifil_ops.py`, `flim/` ou na biblioteca SPiFiL é modificado.

```python
# spifil0.py — PROPOSTA
class Spifil0Adapter:
    name, version = "spifil0", "frozen-ref"

    def fit(self, prefix, train_data, block_spec, context):
        _check_train_only(train_data)                     # I5
        cfg = {**context.config.get("spifil", {}), "device": str(context.device),
               "image_size": train_data.image_size}
        if context.genome is not None:
            # Caminho de EQUIVALÊNCIA: exatamente a chamada de produção
            # (repair, rota de largura pequena, seed = genome.seed + layer_idx, fit_new_block|fit_ops_block).
            block, prov = init.init_block(context.genome, context.layer_idx, prefix,
                                          train_data.donors, cfg, context.prefix_id)
        else:
            # Caminho direto (ops spec já reparado), como fit_ops_block faz hoje.
            block, prov = spifil_ops.fit_ops_block(prefix, train_data.donors, block_spec,
                                                   {**cfg, "seed": context.seed},
                                                   prefix_id=context.prefix_id,
                                                   exact_width=context.config.get("exact_width", False))
        sd = block.state_dict()
        weights = {k: v.detach().float().cpu().clone() for k, v in sd.items()
                   if k.startswith(("conv.", "skip.")) and k.endswith("weight")}
        bias = {k: v.detach().float().cpu().clone() for k, v in sd.items()
                if k.startswith(("conv.", "skip.")) and k.endswith("bias")}
        # Rota legada: Sequential(conv,[bn],act,[pool]); as chaves bn.* ficam de fora (pós-hoc).
        prov = {**prov, "method": "spifil0", "version": self.version,
                "labels_used": True, "label_source": "donor_masks+image_labels",
                "mask_used": not cfg.get("impurities", False),
                "diagnostics": {"init": diagnostics.layer_init(prefix, weights, bias,
                                                               prov["effective_spec"], context.diag_batch),
                                "candidates": diagnostics.spifil0_candidates(prefix, train_data, prov, cfg)}}
        return weights, bias, prov
```

- **Diagnósticos de candidatos do SPiFiL-0, sem tocar no original.** `diagnostics.spifil0_candidates` refaz, de forma determinística, só a parte barata com as funções públicas: `build_layer0` → `extract_patches` (via `spifil_ops._learner` + `_budget`). Isso dá N, D, N/D, rank efetivo dos candidatos (após centragem), cobertura (fração de candidatos com cos máximo com algum filtro selecionado > 0,8, medida no espaço z dos selecionados) e distribuição dos filtros por classe e por imagem. Para verificar que os filtros saíram dos candidatos, desdobra-se o z-score (`W·stdev`) e checa-se cos ≈ 1 com algum candidato. **Não** se refaz o Fisher (O(N²)).
- **Variantes S0 da matriz** (todas compõem código existente):
  - **S0**: produção.
  - **S0-data**: ipc maior (8 ou 16), para casar com o número de imagens do S1–S3.
  - **S0+**: `GrowOneLayer(whiten, patch_norm, knn_smooth, select_diverse)` mais a dobra de ZCA/escala de `grow()`, compostos no módulo novo (Learner + strategy injetada, o mesmo padrão de `spifil_ops._learner`).
  - **S0-lf**: `DistanceSumScorer`, pseudo-classe única e sem máscara. É só diagnóstico.

---

## 5. SPiFiL-1/2/3/4

Convenções comuns aos quatro:

- **Dados.** Imagens de treino amostradas por `loader_factory(seed)`, até `max_images` (padrão 256, estratificado por classe só quando a política é supervisionada; uniforme quando label-free).
- **Features.** F = prefix(x), sem gradiente, em blocos; amp opcional só no forward. Grade H×W, C_in canais.
- **Patches densos.** P posições por imagem (padrão 400, uniformes; nunca dentro de máscara no label-free), `unfold` k×k gera X ∈ ℝ^{N×D}, com D = C_in·k² e N ≤ `max_patches`. **N > D é garantido por construção** (o gargalo de §2.2-2 desaparece; registra-se N/D).
- **Dobra linear** (o que torna o resultado um kernel de conv). Se o filtro é definido no espaço transformado x̃ = Z·P·(x − μ) (Z = ZCA simétrica, P = I − 11ᵀ/D, a remoção opcional da média do patch, que é linear), então c·x̃ = (PZc)·x − (PZc)·μ, e portanto **w = PZc** e **b = −w·μ**. É exato porque Z e P são simétricas.
- **Escala.** Para comparar com o S0 (z-score dobrado, resposta com escala próxima de 1), cada filtro é dividido pelo desvio da sua resposta em patches de treino separados do ajuste, e o bias é recalculado. Fica registrado como `approximations: ["unit_response_std"]` e há um sub-braço sem isso.
- **Por tipo de bloco.**
  - `conv`/`pointwise`: o banco inteiro.
  - `depthwise`: por canal (D = k²), m protótipos por canal.
  - `dwsep`: depthwise (m = 1, então o protótipo é o 1º componente principal ou centróide único do canal) seguido de um pointwise ajustado na saída linear do depthwise (D = C_in), como o S0 já faz.
  - `inception`: por ramo, nos k de cada ramo (o ramo pool sobre `max_pool2d(3,1,1)`), com diversidade entre ramos opcional (rejeita um candidato com |cos| > 0,9 em relação a um filtro já escolhido de um ramo de mesmo k).
  - `residual project`: o skip 1×1 recebe as top-C_out componentes principais do 1×1 de entrada (PCA, label-free), para preservar a variância máxima no atalho. É uma escolha nova registrada; a alternativa é o S0-1×1 de hoje.

### 5.1 SPiFiL-1 — protótipos hierárquicos

**Ideia.** Trocar "escolher K patches individuais por Fisher" por "resumir a distribuição de patches da camada em protótipos" (Coates, FLIM, PCANet), em espaço branqueado, escolhendo C_out protótipos por **representatividade (massa) + diversidade (FPS)**. Opcionalmente, refinar com SoftHebb.

```python
class Spifil1Prototypes:
    name, version = "spifil1", "0.1.0"
    # config: whiten ∈ {zca, none}; eps_zca = 0.01·mean(eig) (igual a GrowOneLayer);
    #         patch_norm ∈ {none, mean_only}; k_mult = 3 (K = k_mult·C_out);
    #         kmeans: esférico, mini-batch, k-means++ semeado, iters = 30;
    #         select ∈ {mass_fps, fps, pca, all}; a_rep = 0.5; softhebb: {epochs: 0|1, lr: 0.01, T: 1.0}
    def fit(self, prefix, data, spec, ctx):
        X = dense_patches(prefix, data, spec["kernel"], ctx)          # (N, D)
        mu = X.mean(0); Xc = X - mu
        P = I - 11ᵀ/D if cfg.patch_norm == "mean_only" else I
        Z = zca(Xc @ P, eps) if cfg.whiten == "zca" else I            # simétrica (D, D)
        Xw = Xc @ P @ Z
        if policy == "label_free":
            C, assign = spherical_kmeans(normalize_rows(Xw), K, seed)  # C: (K, D) unitários
        else:                                                          # estilo FLIM: k-means POR CLASSE
            C, assign, cls = per_class_kmeans(Xw, y_patch, K_per_class, seed)   # y_patch = rótulo da imagem
        mass = bincount(assign) / N
        if cfg.select == "pca":
            sel = top_eigvecs(cov(Xc @ P), C_out)       # PCANet: sem ZCA (ZCA+PCA = identidade)
        elif cfg.select == "mass_fps":
            sel = greedy(C, score=lambda j, S: a_rep·norm_log(mass[j]) + (1-a_rep)·min_{s∈S}(1 - cos(C_j, C_s)))
        #   supervisionado: as quotas por classe seguem o alocador do S0 (Uniform/Exact) e o desempate
        #   usa a razão de Fisher da resposta GAP do protótipo nas imagens de treino.
        c = C[sel]                                       # (C_out, D) unitários no espaço branqueado
        w = c @ Z @ P                                    # dobra: (C_out, D)   [Z, P simétricas]
        b = -(w @ mu)
        w, b = unit_response_std(w, b, X_holdout)        # opcional
        if cfg.softhebb.epochs:                          # refino label-free, código existente
            conv = conv_from(w, b, spec); fit_softhebb(conv, loader, ..., input_transform=prefix)
            w, b = conv.weight, conv.bias                # bias intocado pela regra (como hoje)
        return to_weight_dict(spec, w, b), ..., prov
```

- **Custo** (N = 10⁵, D = 576, C_out = 152, K = 456):
  - covariância e ZCA: N·D² ≈ 3,3·10¹⁰ mais eigh D³ ≈ 2·10⁸;
  - k-means: 30·N·K·D ≈ 7,9·10¹¹ FLOPs;
  - memória: X tem 230 MB em fp32.

  Na ordem de segundos em GPU e de 1 a 3 min em CPU. **Não há matriz N×N.** Para comparação, o Fisher do S0 com N = 10⁴ já forma 10⁸ distâncias.
- **Rótulos.** Label-free por padrão. A versão supervisionada usa só o rótulo da imagem de treino (não a máscara, a menos que `mask_policy = "mask"`, registrado).
- **O que pode falhar** (hipóteses para medir, não promessas):
  - protótipos com muita massa tendem a ser "fundo/textura comum", que não discrimina. Daí o termo de diversidade e o sub-braço `select = fps` puro;
  - com ZCA, o ruído é amplificado nas direções de baixa variância (eps controla isso).
- **Sub-ablations init-only** (baratas, rodam antes da matriz principal): whiten {zca, none} × select {mass_fps, fps, pca} × patch_norm {none, mean_only}. Testam a H-L2 e a H-L3.

### 5.2 SPiFiL-2 — invariância entre vistas

**Ideia.** Entre muitos candidatos (protótipos S1 e, opcionalmente, patches S0), preferir os filtros cuja resposta é **estável e repetível sob as transformações que não mudam a classe no domínio**, com equipartição (SwAV) para evitar canais mortos. Em posições correspondentes conhecidas pela geometria (VICRegL), não por similaridade de feature.

**Vistas e correspondência.** `views.make_views_with_theta(x, V, aug_cfg, seed)` reproduz `ssl_trainers.make_views` e devolve também θ_v (2×3, coordenadas normalizadas de saída → entrada; o flip entra no sinal). Para uma posição p da grade da vista a, com coordenada normalizada u_a, a coordenada na imagem é u = θ_a·[u_a; 1], e na vista b é u_b = A_b⁻¹(u − t_b). O par é mantido se ‖u_b‖_∞ ≤ 1 − 1/H; q é a célula mais próxima na grade de b. A tolerância é de 1 célula. Só correspondências entre vistas da **mesma** imagem.

**Política de invariâncias por domínio.** É uma variante registrada; o padrão de hoje fica como braço "aug_default".

| Domínio | Geometria | Fotometria | Justificativa (hipótese) |
|---|---|---|---|
| RS, AID (aéreo) | rot90 + hflip + vflip; escala 0,5–1 | brilho/contraste 0,2; canal 0,05 | sem orientação canônica |
| Flowers | **só hflip**; escala 0,3–1 | brilho 0,1; **canal 0,0** (a cor é pista de classe) | a flor tem "para cima"; a cor discrimina espécies |
| HAM10000 | rot + flips; escala 0,6–1 | brilho 0,1; canal 0,02 | a cor é diagnóstica (melanoma) |
| Parasitas raw/pre | rot + flips; escala 0,7–1 | coloração 0,1 | microscopia, sem orientação; o pre tem fundo preto (o crop pode cair no fundo, então descarta-se vista com > 90% de fundo) |

```python
class Spifil2Invariant:
    name, version = "spifil2", "0.1.0"
    # config: V = 2; cand = spifil1 (K = 3·C_out) [+ spifil0]; tau = 0.1; eps_sk = 0.05; sk_iters = 3
    #         lam_stab = 1, lam_rep = 0.5, lam_mass = 0.5, red_max = 0.9; stab_floor = 0.2
    #         refine: {steps: 0 | 300, lr: 1e-3, loss: vicregl_local | swav_local}
    def fit(self, prefix, data, spec, ctx):
        Wc, bc, Z, mu = spifil1_candidates(prefix, data, spec, ctx)    # (K, D) no espaço de entrada + dobras
        stats = zeros(K)
        for x in loader(ctx.seed, max_images):
            views, thetas = make_views_with_theta(x, V, aug_cfg(dataset), seed)
            R = [conv(prefix(v), Wc, bc) for v in views]               # (B, K, H, W) cada
            pairs = correspondences(thetas, grid=R[0].shape[-2:])      # [(b, pa, pb)]
            ra, rb = gather(R[0], pairs.a), gather(R[1], pairs.b)      # (n_pares, K)
            stats.stab += pearson_per_column(ra, rb)                   # estabilidade por filtro
            stats.rep  += topm_peak_overlap(R[0], R[1], pairs, m=8)    # repetibilidade
            qa = sinkhorn(softmax(ra / tau), eps_sk, sk_iters)         # equipartição (SwAV)
            pb = softmax(rb / tau)
            stats.agree += column_agreement(qa, pb)                    # consistência de atribuição
            stats.mass  += qa.mean(0)                                  # uso sob equipartição
        score = lam_stab·z(stab) + lam_rep·z(rep) + lam_mass·z(log mass) + z(agree)
        if policy == "supervised":
            score += lam_fisher·z(fisher_gap_response(Wc, bc, donors_or_train_labels))
        sel = greedy_nms(score, Wc_whitened, red_max, floor=stab_floor, n=C_out)   # completa por score se faltar
        w, b = Wc[sel], bc[sel]
        if cfg.refine.steps:                         # S2-ref: gradiente LOCAL, prefixo congelado
            w, b = refine_local(w, b, prefix, loader, steps, loss=cfg.refine.loss)
            #  vicregl_local: L = 25·inv(z_a[p], z_b[q]) + 25·var(z) + 1·cov(z), com z = resposta pré-ativação
            #  swav_local:    L = CE(q_a(p), p_b(q)) + CE(q_b(q), p_a(p)), protótipos = filtros normalizados
            #  ao fim: renormaliza no espaço branqueado e recalcula a escala (unit_response_std)
        return to_weight_dict(spec, w, b), ..., prov   # prov inclui stab/rep/agree de cada filtro escolhido
```

- **Custo.** Conv dos K candidatos em M·V imagens: M·V·H·W·K·D ≈ 256·2·56²·456·576 ≈ 4,2·10¹¹ MACs, ou seja, segundos. O refino (300 passos, bs 32, 2 vistas, só o bloco novo) leva minutos.
- **Rótulos.** Label-free por padrão. O termo de Fisher só existe em `S2-sup`.
- **Risco conhecido.** Estabilidade alta é trivialmente obtida por filtros de baixa frequência (DC/cor). Por isso exige-se diversidade, mede-se a fração de energia DC dos escolhidos (§7) contra o S0, e a H-L6 valida o critério antes de ele selecionar em produção.

### 5.3 SPiFiL-3 — contextual/preditivo, com transformação latente→kernel explícita

**Ideia.** Aprender, **sobre a grade do prefixo congelado**, um latente local que seja previsível a partir do contexto (I-JEPA no espaço de features, não de pixels). Agrupar esse latente em protótipos e **projetar cada protótipo para o espaço de kernels k×k×C_in** com a melhor aproximação linear. O kernel resultante tem exatamente o shape de uma conv comum. O encoder-latente e o preditor são **descartados**.

**Estágio 1 — latente preditivo (label-free, só treino).**

- **Encoder de contexto f_θ.** Conv k×k (C_in → d) + GELU + conv 3×3 (d → d), com d = 64 (≤ 128). O RF na grade é k + 2. Entrada: F mascarada (posições-alvo zeradas, mais um token de máscara aprendível por canal).
- **Encoder-alvo f_ξ.** A mesma arquitetura, por EMA de θ (m de 0,996 a 1, linear). Entrada: F inteira, sem gradiente.
- **Preditor h_φ.** 2 convs 3×3 (d → 64 → d) sobre a saída de f_θ, com embedding posicional 2D somado nos alvos.
- **Máscaras** (I-JEPA adaptado a grades de 14 a 56): M = 4 blocos-alvo com escala 0,15–0,2 da grade e aspecto 0,75–1,5, **com lado mínimo de 2 células**. Contexto: um bloco de escala 0,85–1,0 menos os alvos. Com grade < 14×14, M = 2.
- **Loss.** Média de ‖h_φ(f_θ(F_ctx))_p − sg(f_ξ(F))_p‖² sobre os alvos. Anti-trivial, somado com peso pequeno: piso de variância VICReg (μ = 1, γ = 1) e covariância (ν = 0,04) sobre f_θ nas posições de contexto, e o monitor erank(f_ξ)/d ≥ 0,2 (abaixo disso, `InfeasibleCandidate("ssl_collapse")`).
- **Otimização.** 2000 passos, bs 32, AdamW lr 1e-3, wd 0,04. Nenhum parâmetro do prefixo entra no otimizador (I1).

**Estágio 2 — protótipos latentes.** Z = f_ξ(F) em N posições amostradas (N ≤ 10⁵), dimensão d. Aplicam-se ZCA em Z e k-means esférico com K = 3·C_out, e a seleção de C_out por massa + FPS (como no S1). Logits de atribuição: L_pj = cos(z̃_p, μ_j)/τ, com τ = 0,1.

**Estágio 3 — transformação latente→kernel (explícita).**

- **Entrada da regressão.** O patch X_p = unfold(F, k) na mesma posição p: dimensão **D = C_in·k²**, **suporte espacial k×k** centrado em p (o mesmo do kernel final), **todos os C_in canais**. Padroniza-se X com (μ_X, σ_X).
- **Alvo.** T = L − mean(L), de dimensão (N, C_out), um alvo por kernel.
- **Ridge.** W = (X̃ᵀX̃ + λN·I)⁻¹ X̃ᵀT, com W de dimensão (D, C_out). λ vem de uma grade {1e-4, …, 1} escolhida por MSE em patches de treino separados (nunca validação).
- **Kernel.** w_j = W[:, j], normalizado no espaço z (como o S0), e depois a dobra: weight_j = (w_j/σ_X).reshape(C_in, k, k) e b_j = −Σ μ_X·w_j/σ_X. O tensor final é (C_out, C_in, k, k), exatamente o shape de `conv.weight`.
- **Diagnóstico obrigatório.** R²_j, a fração da variância da atribuição ao protótipo j que um detector linear k×k consegue reproduzir. Se R² mediano < 0,3, o latente "vê" além do suporte (RF k + 2 > k) ou é muito não linear. Opções registradas: k' = k + 2 (5 em vez de 3, que **muda params/FLOPs** e por isso é outro braço), ou ativar um f_θ só com k×k e 1×1 (RF = k).
- **Mapeamento alternativo exato (sub-braço `S3-lin`).** Se f é linear no patch (z = A·x̃, ablação "JEPA linear"), então w_j = Aᵀμ_j exatamente, e R² = 1 por construção. Isso isola o efeito da não linearidade.

```python
class Spifil3Predictive:
    name, version = "spifil3", "0.1.0"
    def fit(self, prefix, data, spec, ctx):
        f_ctx, f_tgt, pred = LatentEnc(C_in, d, k), ema_copy(...), Predictor(d)
        train_jepa_local(prefix, data.loader_factory(ctx.seed), f_ctx, f_tgt, pred,
                         steps, masks=block_masks(grid, M, scales), anti_collapse=vic_floor)  # prefixo congelado
        Z, X = sample_latents_and_patches(prefix, f_tgt, k, N)            # (N, d), (N, C_in·k²)
        mu_j = select_mass_fps(spherical_kmeans(zca(Z), 3*C_out), C_out)  # (C_out, d)
        T = cos(zca(Z), mu_j) / tau; T -= T.mean(0)
        W, lam, R2 = ridge_cv(standardize(X), T, holdout=train_holdout)  # (C_in·k², C_out)
        w, b = fold_zscore(unit_norm_cols(W), mu_X, sd_X)                 # -> (C_out, C_in, k, k), (C_out,)
        del f_ctx, f_tgt, pred                                            # descartados
        return to_weight_dict(spec, w, b), ..., prov(R2=R2, lam=lam, erank_Z=..., jepa_loss_curve=...)
```

- **Custo.** Na grade 28×28 com C_in = 64, d = 128 e bs 32, um forward custa cerca de 32·784·(64·9·128 + 128·9·128) ≈ 5,5·10⁹ MACs, ou ~3·10¹⁰ FLOPs por passo com backward e EMA. Com 2000 passos, ~6·10¹³ FLOPs, ou seja, minutos. A ridge custa D³ + N·D·C_out, segundos. A memória fica abaixo de 3 GB.
- **Rótulos.** Label-free por padrão. `S3-sup` faz k-means por classe no Estágio 2 (rótulo da imagem de treino).
- **Controle obrigatório `S3-rand`.** O mesmo pipeline com f_ξ **aleatório e não treinado** (sem Estágio 1). Testa a H-L7: se S3 ≈ S3-rand, a parte preditiva não contribui.

### 5.4 SPiFiL-4 — composição e herança no crescimento

**Contexto honesto.** O H-SEED já faz parte disto: Net2Wider/Net2Deeper (`external_seed.widen_block`, `identity_block`, `residual_zero_block`, `grow_blocks`), filtros novos via SPiFiL-0 com largura exata (`fit_new_filters`), AE local só nos elementos novos com `ElementProtect`, e âncora WiSE. O que o SPiFiL-4 propõe de **novo**, para testar contra esse baseline:

1. **Filtros novos no complemento dos herdados.** No espaço branqueado da entrada da camada, U = orth(Z⁻¹·W_herdadoᵀ) (a base do subespaço dos filtros herdados vistos como direções) e X_res = X̃ − X̃UUᵀ. Os n_new filtros novos saem de SPiFiL-1/2/3 **sobre X_res**, ou seja, padrões que os herdados não cobrem. Diagnóstico: o cos máximo de cada novo com os herdados deve ficar < 0,3 (medido).
2. **As colunas consumidoras dos canais novos ficam = 0** (o `widen_block` existente), o que preserva exatamente a função no passo 0. Os gradientes das colunas consumidoras não são nulos (a ativação dos novos é ≠ 0), então eles "entram" no treino.
3. **Profundidade nova como "residual-ε".** Variante de `residual_zero_block`: conv = ε·(kernels SPiFiL-1/2/3 com C_in = C_out) e ativação silu, com ε ∈ {0, 10⁻³, 10⁻²}. Com ε = 0, é exatamente o `residual_zero_block` de hoje. O erro de preservação é medido (`preservation_error`), e o limite teórico registrado é ‖Δy‖ ≤ ε·‖W‖_op·‖x‖·sup|silu'|.
4. **Aprendizagem local só nos novos** (AE local com `ElementProtect`: protect ∈ {freeze, lowlr} nos herdados), decoder descartado, depois o SSL global LeJEPA + SIGReg e WiSE com âncora no filho pré-SSL.

```python
class Spifil4Compose:
    name, version = "spifil4", "0.1.0"
    def fit(self, prefix, data, spec, ctx):          # ctx.parent obrigatório
        par = ctx.parent
        W_old = inherited_weights(par.parent_block, par.keep, par.in_map)   # já em ordem do filho
        X, mu, Z, P = dense_patch_space(prefix, data, spec, ctx)            # mesmo espaço do S1
        U = orth((W_old @ inv(P @ Z)).T)            # direções herdadas no espaço transformado (pinv se singular)
        X_res = X_tilde - X_tilde @ U @ U.T
        gen = REGISTRY[cfg.new_filter_method]       # spifil1 | spifil2 | spifil3 (o vencedor do braço A1)
        w_new, b_new, _ = gen.fit_on_space(X_res, ..., n=par.n_new)
        weights, bias = widen_layout(W_old, w_new, par)                     # = external_seed.widen_block
        return weights, bias, prov(novelty_cos=max_cos(w_new, W_old), preservation_error=...)
```

- **Custo.** O do gerador escolhido mais uma ortogonalização O(D·n_old²).
- **Rótulos.** Herdados do gerador e do pai (registrar `labels_used` do pai e do método).
- **Demonstração contra o crescimento padrão** (braços A4 da §9):
  - C1: Net2Net puro (novos = 0, identity/residual_zero);
  - C2: H-SEED atual (novos = S0 com largura exata);
  - C3: o mesmo filho reinicializado do zero com o melhor init;
  - T: SPiFiL-4.

  Métrica: Δκ_val = κ_val(filho) − κ_val(pai) com o mesmo protocolo e o mesmo orçamento adicional de passos, mais o erro de preservação no passo 0 e o CKA dos canais herdados antes e depois.

---

## 6. Aprendizagem local e global

- **AE local (principal)**: `train_local_ae` sem alteração. O alvo `pixel` com `decoder_first` é o padrão de produção; `joint` fica como sub-braço. O decoder é sempre descartado (já é).
- **noAE (controle)**: a etapa local é pulada; o bloco segue direto para a próxima camada (o mesmo mecanismo do braço `phase0_ae_off` do PNAS-CS).
- **Variantes locais secundárias** (só se o A1 apontar o AE como gargalo):
  - `feature`, `masked` 0,4 e `masked` 0,75 (H-L8);
  - uma variante **preditiva** local, que é o Estágio 1 do SPiFiL-3 usado como objetivo de treino do próprio bloco (f_θ = o bloco) — um braço distinto, nunca misturado com "AE".
- **Proteção do SPiFiL**: `protect` ∈ {freeze (0), lowlr (0,01), full (1)} via `conv_lr_scale`. No plano mínimo fixa-se o valor de produção (lowlr). freeze e full são sub-braços só para o vencedor.
- **Global**: `train_global_ssl(ssl="lejepa", reg="sigreg")` real do repositório (RealSIGReg Epps–Pulley, λ = 0,05, V = 4), lr_old 1e-4 e lr_new 1e-3, WiSE α ∈ {0, 0,25, 0,5, 0,75, 1} escolhido **na validação**. Outras losses só como braços identificados (fora do plano mínimo).
- **Cabeça ≥ 1280 (obrigatória).** A interface `KernelInitializer` não toca a cabeça. Em toda a matriz, a cabeça é a `EmbeddingHead` de produção, com `embed_dim` ≥ 1280 e o mesmo pooling, inicialização (seed fixa) e lr em todos os braços. Assim, os params e FLOPs da cabeça entram no total Bk de forma idêntica e não confundem a comparação de init. Se um braço estourar o orçamento Bk, ajusta-se a **forma** da cabeça (moments/meanstd, baixo rank, bilinear compacto) de modo **igual para todos os braços**, nunca a dimensão < 1280. A leitura de seleção é sempre o embedding ≥ 1280 (ORIG). A sonda GAP por bloco (§7) é só diagnóstico e **não** substitui a cabeça.
- **Medida de "o backprop soma ao SPiFiL"**: a fração de filtros com cos(init, pós-AE) > 0,9 e cos(init, pós-SSL) > 0,9, e o α escolhido pelo WiSE. Se α = 0 for frequente, o treino global está destruindo ou não somando ao init.

---

## 7. Diagnósticos por camada

**Lote de diagnóstico.** 64 imagens **de treino**, fixas por hash, disjuntas das usadas no fit. A validação só é usada para o κ ORIG final e para as sondas, marcadas como `probe_diag`, que nunca entram em seleção.

| Diagnóstico | Definição | Como medir | Quando | Rótulos? |
|---|---|---|---|---|
| Shapes, params, FLOPs | shapes do §3.3; params; MACs na resolução de treino | `validate_init`; contagem analítica conv/pool (a mesma do MO-DEHB) | init | não |
| N, D, N/D | nº de candidatos e dimensão do patch | proveniência do método (S0 via `spifil0_candidates`) | init | não |
| Rank efetivo dos candidatos | exp(entropia dos valores singulares normalizados) de X centrado | `init._erank` (em blocos de 8192 linhas) | init | não |
| Cobertura / representatividade | fração de candidatos com max_j cos(x̃, c_j) > 0,8; média do cos máximo | no espaço transformado do método (z-score no S0, branqueado no S1–S4) | init | não |
| Alocação por classe | nº de filtros por classe e por imagem de origem | proveniência (S0 e supervisionados) | init | sim (só leitura) |
| Normas | média/desvio de ‖w_j‖ (espaço de entrada) e de \|b_j\| | direto | init, pós-AE, pós-SSL | não |
| Redundância e diversidade | fração de pares com \|cos\| > 0,9; 1 − cos médio | `init.kernel_diagnostics` (já existe) | init, pós-AE, pós-SSL | não |
| Rank efetivo do banco | erank dos filtros achatados | `kernel_diagnostics.weight_effective_rank` | idem | não |
| Ativação | média, variância por canal, fração de positivos | ReLU(conv) no lote de diagnóstico | idem | não |
| % canais mortos | std < 1e-8 **ou** fração de positivos < 1% | idem (o 2º critério é novo, porque std > 0 com ReLU quase sempre zerada também é "morto") | idem | não |
| Similaridade entre mapas | média de \|cos\| entre mapas de canais | `kernel_diagnostics.featmap_similarity` | idem | não |
| Rank efetivo das ativações | erank (posições × canais) | `kernel_diagnostics.activation_effective_rank` | idem | não |
| Estabilidade entre vistas | Pearson das respostas em posições correspondentes (§5.2), por canal; média e p10 | `views.correspondences` com a política de aug do domínio, 2 vistas, seed fixa | idem | não |
| Repetibilidade | sobreposição dos top-8 picos mapeados (tolerância de 1 célula) | idem | idem | não |
| Energia espectral do kernel | fração DC: k²·‖média espacial de w‖² / ‖w‖²; fração em alta frequência (FFT 2D com padding 8×8, raio > 0,5·Nyquist) | direto nos pesos | idem | não |
| Espectro das ativações | decaimento dos autovalores da covariância de canais (inclinação α em log-log) | eigh da covariância C×C do lote | idem | não |
| Similaridade antes/depois de AE e SSL | cos por filtro (init × pós-AE × pós-SSL); CKA linear das ativações; fração com cos > 0,9 | snapshots dos pesos guardados na proveniência (hash + tensores no ckpt do experimento) | pós-AE, pós-SSL | não |
| Preservação (crescimento) | `preservation_error` no passo 0; CKA dos canais herdados pai × filho após o treino | `external_seed.preservation_error` | SPiFiL-4 | não |
| Fidelidade latente→kernel | R² por kernel, λ escolhido | ridge (§5.3) | SPiFiL-3 | não |
| Saúde do latente | curva da loss JEPA, erank(Z)/d, std mínima por dimensão | Estágio 1 | SPiFiL-3 | não |
| Qualidade do embedding final | **κ_val ORIG** (seleção); erank do embedding 1280; flags de colapso do fitness | `svm_fitness` com régua ORIG + o detector de colapso existente | fim de cada fidelidade | sim (SVM) |
| Sondas intermediárias (**só diagnóstico**) | SVM C=100 (`core.svm.fit_svm`) no GAP da saída de cada bloco, treino → validação | marcada `probe_diag`; **nunca** entra em fitness, ranking ou seleção | init e fim | sim |
| Uso de rótulos | `labels_used`, `label_source`, `mask_used` | proveniência | init | — |

Tudo vai para W&B com o prefixo `abstract_init/layer{i}/...` e para a proveniência JSON do candidato.

---

## 8. Testes

### 8.1 Unitários (CPU, dados sintéticos, < 2 min no total)

Dados sintéticos: 3 classes e 12 imagens 32×32 LAB, com padrões de classe (listras horizontais, verticais e bolhas), máscaras circulares e `Sample` falsos em `tmp_path`.

1. **Contrato de shapes (I2/I3)**, parametrizado em `conv` (k = 1, 3, 5, 7), `pointwise`, `depthwise` (m = 1, 2), `dwsep`, `inception`, `residual` identity/project, para os métodos `spifil0`, `spifil1`, `spifil2` e `spifil3`: `validate_init` passa, `assemble_block` carrega estrito e o forward tem o shape esperado (incluindo a geometria do pool).
2. **Equivalência do SPiFiL-0.** `Spifil0Adapter.fit` com `context.genome` reproduz, **bit a bit** em CPU, `init.init_block` (rota legada `fit_new_block`, rota de largura pequena e rota `spifil_ops`), comparando o `state_dict` inteiro e `effective_spec`.
3. **Prefixo intocado (I1).** O fingerprint é igual antes e depois; nenhum `p.grad` no prefixo; o prefixo está em `eval()`. Inclui o SPiFiL-3, que treina módulos auxiliares.
4. **Só treino (I5).** O `TrainData` com `split_key = "val"` é recusado; um id sentinela de validação injetado no loader dispara erro pelo `split_fingerprint`.
5. **Label-free de verdade (I6).** Com rótulos permutados, os métodos `*-lf` dão saída **idêntica**; com máscara presente, o método não a lê (um mock de `load_mask` que lança erro).
6. **Determinismo (I4).** A mesma seed dá o mesmo resultado; uma seed diferente dá resultado diferente (exceto no S0 com 1 doador por classe, que não tem aleatoriedade, e isso é documentado).
7. **Dobra correta.** Para ZCA e mean_only, conv(x)[p] = c·Z·P·(x_p − μ) + 0 com tolerância 1e-5 (forma explícita contra a conv).
8. **Latente→kernel.** (a) Com latente linear sintético z = Ax, a ridge com λ → 0 recupera Aᵀμ_j (cos > 0,999) e R² > 0,99. (b) Com latente não linear (|Ax|), R² < 1 e é reportado. (c) O shape do kernel é (C_out, C_in, k, k).
9. **Correspondências.** θ identidade dá o mapa identidade; um hflip puro dá o espelho (exato); um crop com escala 0,5 e um ponto marcado é mapeado com tolerância de 1 célula; `make_views_with_theta` produz vistas **iguais bit a bit** a `ssl_trainers.make_views` com a mesma seed.
10. **Anti-colapso.** No S3, entrada constante gera `InfeasibleCandidate("ssl_collapse")`; com ruído branco, o erank(Z)/d fica ≥ 0,2.
11. **Orçamento (I7/I8).** `max_patches` pequeno com divisão em blocos dá o mesmo resultado que sem divisão; estourar o limite de memória simulado gera `InfeasibleCandidate("budget: ...")`.
12. **SPiFiL-4.** Com ε = 0, o filho reproduz o pai nos canais antigos (`preservation_error` < 1e-6); com ε > 0, o erro fica ≤ o limite teórico; o cos máximo dos novos com os herdados no espaço transformado é < 1e-4 por construção.
13. **Diagnósticos com resposta conhecida.** Filtros idênticos dão redundância 1; um banco ortonormal de n filtros dá erank = n; um canal sempre 0 conta como morto; um kernel constante dá fração DC 1.

### 8.2 Integração

- **E2E CPU (R9, antes de qualquer fila).** Sintético 3 classes, encoder de 3 blocos, um ciclo por método: init → AE local 1 época → SSL LeJEPA + SIGReg 1 época → WiSE → κ_val ORIG. Verifica:
  - sem NaN;
  - checkpoint salvo e recarregado com forward idêntico;
  - proveniência serializável;
  - `labels_used` correto.
- **Regressão de produção.** Com `abstract_init` ausente, ou com `method: spifil0`, um candidato de RS em f5 dá `effective_specs`, pesos pós-init e κ_val **iguais** aos do pipeline atual (mesma seed, CPU ou CUDA com tolerância registrada).
- **E2E real pequeno (Fase A da fila GPU 3, via Agente 5).** RS e Flowers, 1 candidato por método, f5. Mede tempo, pico de memória e N/D por camada. Inclui um teste de **kill -9 com retomada**, que não deve duplicar o job nem refazer o init já registrado (o init é determinístico; o checkpoint guarda o hash dos pesos iniciais).
- **Reprodutibilidade.** Dois jobs idênticos em CUDA: os pesos iniciais batem com tolerância de 1e-5 e o κ_val com variação registrada. Configs versionadas (`configs` com hash na proveniência) e ambiente registrado.

---

## 9. Matriz mínima de ablação

### 9.1 Princípios

- **RS e Flowers primeiro.** HAM10000 e parasitas (raw/pre separados) só depois do gate desta matriz. Coconut e AID entram como controle.
- **Uma variável por vez.** Primeiro, arquitetura fixa e init variando (A1). Depois, init fixa e profundidade variando (A2). Sem cartesiano completo.
- **Arquitetura fixa por dataset** `[A1]`: a semente do `seed_registry.csv` no porte B1 (≤ 800k) com profundidade 4. Fallback, se o A1 não definir: o melhor por **validação** do estrato small de cada dataset. A **camada 0 (stem) é S0 em todos os braços**; os braços variam as camadas 1..L−1, que é onde a hipótese de "patch simples insuficiente" vive. O braço "todas as camadas" fica só para o vencedor.
- **Mesmos params e FLOPs.** Todos os braços usam o `effective_spec` do S0 daquela seed (a largura que o alocador do S0 produziu), e o teste `params(arm) == params(S0)` é feito antes de rodar. Os params são contados **encoder + cabeça ≥ 1280** (orçamento Bk), com decoders, projetores e o encoder-latente do S3 registrados à parte, porque são descartados.
- **Cabeça ≥ 1280 fixa.** A mesma `EmbeddingHead` (dimensão ≥ 1280, mesma forma e mesma seed de init) em todos os braços. Nenhum braço reduz a dimensão para caber no orçamento (§6).
- **Mesmos dados, passos, resolução, seeds auxiliares** (k-means, vistas, máscaras JEPA derivam da seed da camada). Só treino no init.
- **Controle de quantidade de dados.** O S1–S3 leem até 256 imagens; o **S0-data** usa ipc = 16 (Flowers: 80 imagens; RS: até 223), o máximo do pool atual. Se o S0-data der `BudgetExhausted` numa camada, isso é registrado como resultado (é evidência para o §2.2-2), não como falha do braço.

### 9.2 Braços

| ID | Init (camadas 1..L−1) | Rótulos | Papel |
|---|---|---|---|
| R | He aleatório (gêmeo exato, `random_twin_ops`) | não | piso |
| S0 | SPiFiL-0 de produção (ipc de produção) | sim + máscara | baseline |
| S0-data | SPiFiL-0, ipc = 16 | sim + máscara | controle de dados |
| S1-lf | protótipos, ZCA, mass_fps | não | H-L1/H-L3 |
| S1-sup | k-means por classe (FLIM) | sim (imagem) | H-L1 |
| S2-lf | seleção por invariância, sem refino | não | H-L5/H-L6 |
| S3-lf | latente preditivo + ridge | não | H-L7 |
| S3-rand | S3 com latente aleatório | não | controle da H-L7 |
| *(secundários, só se o pai passar)* S0+, S0-lf, S1-lf+SH, S1-pca, S2-ref, S2-sup, S3-lin | — | — | H-L2/H-L4 etc. |

### 9.3 Fases, contagem e orçamento

As estimativas de GPU-h vêm das medianas de `agent1/parts/A1a_data.csv` (RS/Flowers): f45 de 0,06 a 0,35 GPU-h, com mediana de cerca de 0,15 para 0,4–0,9M params; f100 em torno de 0,4. O custo do init novo (de segundos a minutos por camada) é desprezível diante disso, **mas é medido** no E2E (§8.2).

| Fase | Desenho | Runs | GPU-h (estimativa) |
|---|---|---|---|
| A0 init-only | sub-ablations do S1 (whiten × select × patch_norm = 12) + os 8 braços, sem AE/SSL; leitura: κ_val ORIG com head 1280 aleatória de seed fixa (a mesma em todos os braços) + sonda GAP diagnóstica | 2 ds × 20 × 3 seeds = 120 | ~2 |
| A1 init × local | 8 braços × {AE, noAE} × 3 seeds × 2 ds, f45 (AE:SSL = 1:2 como em produção; LeJEPA + SIGReg + WiSE) | 96 | ~15 |
| A2 profundidade | {S0, melhor de A1} × depth {3, 4, 6} × AE × 3 seeds × 2 ds, f45 (depth 8 só depois da validação) | 36 | ~7 |
| A3 promoção | top-2 de A1 + S0 em f100, 3 seeds; o vencedor ganha +2 seeds (5 no total) | ~22 | ~9 |
| A4 SPiFiL-4 | crescimento B0→B1: {C1 Net2Net, C2 H-SEED atual, C3 reinit, T SPiFiL-4} × 3 seeds × 2 ds, f45 | 24 | ~6 |
| **Total** | | **~300** | **~39 GPU-h** |

Na GPU 3 do athena (49 GB, com 3 a 4 slots concorrentes, a confirmar com o Agente 5), isso dá **~10–14 h de relógio**. A0 roda em minutos e pode até usar CPU (o init é barato), o que antecipa a resposta sem ocupar a GPU de forma longa.

### 9.4 Critério de decisão (só validação)

- **Métrica primária:** κ_val ORIG (C = 100, sem scaler, embedding 1280). Δ pareado por seed contra o S0 (mesmas seeds).
- **Benefício reproduzível** (para passar ao A3): média de Δκ_val ≥ +0,02 em pelo menos um dataset, com **3/3 seeds positivas**, e sem regressão maior que 0,01 no outro dataset.
- **Confirmação** (A3, f100, 5 seeds): ≥ 4/5 seeds positivas e IC95 bootstrap de Δ médio > 0. Só então o mecanismo é declarado "com benefício reproduzível" para o gate do Agente 3.
- **Controles que derrubam a conclusão:**
  - se S0-data ≈ S1–S3, o ganho é de **quantidade de dados**, não do mecanismo;
  - se S3 ≈ S3-rand, a parte preditiva é inócua;
  - se o ganho some com AE e aparece só no noAE (ou o contrário), o resultado é **interação** e é relatado como tal.
- **Resultado negativo** (também é um entregável do gate): nenhum braço passa. Registra-se a tabela completa, com os diagnósticos que explicam o porquê (por exemplo, R² baixo no S3 ou fração DC alta no S2).
- **Teste:** só o vencedor confirmado e o S0 são medidos no teste, uma vez, depois da decisão, e relatados à parte (R2).

---

## 10. Riscos e perguntas para o gate do A1

1. **Arquitetura fixa da A1** `[A1]`: qual semente B1 por dataset? Profundidade 4 basta para ver o efeito em "camadas profundas"? Se a melhor semente tiver profundidade ≤ 3, a A1 testa pouca profundidade.
2. **A leitura init-only** com head 1280 aleatória é ruidosa. O A1 aceita a sonda GAP como diagnóstico auxiliar (nunca de seleção)?
3. **Máscaras em RS/Flowers.** É preciso confirmar se esses datasets têm máscaras. Sem máscara, o S0 já roda "sem máscara" (`load_mask` dá None), e o S0-lf difere do S0 só no Fisher/rótulo.
4. **Diagnóstico em validação** (`_log_init_diag` usa `val_x`): o A1 confirma que o protocolo novo deve usar um lote de treino?
5. **Política de augmentation por domínio** (§5.2): precisa de aprovação, porque também afeta o LeJEPA global, se adotada lá. Neste plano, ela vale **só** dentro do SPiFiL-2. O LeJEPA global continua com `AugConfig` padrão em todos os braços.
6. **Cabeça ≥ 1280 dentro do B1.** Com a cabeça contando no orçamento, num encoder estreito a cabeça (conv 1×1 C_in → 1280 + pooling) pode ser uma fração grande dos ≤ 800k params. O A1 deve informar quanto do Bk da semente fixa é cabeça. Se for preciso otimizar a forma da cabeça (moments/baixo rank), isso é feito **antes** da matriz e fica igual em todos os braços.
7. **O fitness atual usa `cov_head` + scaler** (`search_v3.yaml:142`), e a missão pede ORIG. Este plano mede ORIG; a mudança no MO-DEHB é assunto do Agente 4.
8. **Ponto de integração** (hook opt-in em `run_search._grow` × runner de ablação separado): decidir com o Agente 5 sob a R9.
9. **Risco de custo escondido.** O S2/S3 leem 256 imagens e fazem forward do prefixo nelas a cada camada. Com prefixo profundo e grade grande, isso pode passar de minutos por camada. O E2E mede; se passar de `max_seconds`, o caminho é o cache de features por camada (só treino, em disco temporário, com hash).

---

## 11. Referências

- De Souza, I. E.; Falcão, A. X. *Learning CNN Filters From User-Drawn Image Markers for Coconut-Tree Image Classification.* SIBGRAPI 2020, pp. 464–468. DOI 10.1109/SIBGRAPI51738.2020.00024. Preprint: https://arxiv.org/abs/2008.03549 — SBC SOL: https://sol.sbc.org.br/index.php/sibgrapi/article/view/14156
- Chan, T.-H. et al. *PCANet: A Simple Deep Learning Baseline for Image Classification?* IEEE TIP 2015. https://arxiv.org/abs/1404.3606
- Coates, A.; Lee, H.; Ng, A. Y. *An Analysis of Single-Layer Networks in Unsupervised Feature Learning.* AISTATS 2011, PMLR 15:215–223. https://proceedings.mlr.press/v15/coates11a.html
- Moraitis, T. et al. *SoftHebb: Bayesian Inference in Unsupervised Hebbian Soft Winner-Take-All Networks.* Neuromorph. Comput. Eng. 2(4) 044017, 2022. https://arxiv.org/abs/2107.05747 — follow-up convolucional (não lido nesta fase): Journé et al., ICLR 2023, https://arxiv.org/abs/2209.11883
- Caron, M. et al. *Unsupervised Learning of Visual Features by Contrasting Cluster Assignments (SwAV).* NeurIPS 2020. https://arxiv.org/abs/2006.09882
- Bardes, A.; Ponce, J.; LeCun, Y. *VICRegL: Self-Supervised Learning of Local Visual Features.* NeurIPS 2022. https://arxiv.org/abs/2210.01571
- Assran, M. et al. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA).* ICCV 2023. https://arxiv.org/abs/2301.08243
- He, K. et al. *Masked Autoencoders Are Scalable Vision Learners.* CVPR 2022. https://arxiv.org/abs/2111.06377
- Belilovsky, E.; Eickenberg, M.; Oyallon, E. *Greedy Layerwise Learning Can Scale to ImageNet.* ICML 2019, PMLR 97:583–593. https://proceedings.mlr.press/v97/belilovsky19a.html
- Código lido: `ga_search/modehb/{init.py, spifil_ops.py, softhebb.py, local_ae.py, ssl_trainers.py, external_seed.py, run_search.py}`, `flim/{progressive.py, spifil.py}`, `/dados/home/moliveira/SPiFiL/src/spifil/{patches.py, scoring.py, allocation.py, selection.py, learner.py, nn/builder.py}`; literatura do A1: `research/spifil_b0b4/agent1/parts/A1d_literatura.md`.
