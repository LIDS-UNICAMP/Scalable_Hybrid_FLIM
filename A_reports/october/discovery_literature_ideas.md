# [A5] Ideias da literatura para levar o κ do SVM linear até o I-JEPA (2026-11-01)

## A resposta, em uma frase

**A aposta mais barata e mais forte é trocar a "leitura" do mapa no SVM, sem treinar nada:** uma pirâmide de pooling (1×1 + 2×2 + 4×4), normalização L2 em cada pixel do mapa e GeM. A literatura dá +2 a +6 pontos para esses truques, e o nosso próprio dado já mostra que a leitura mexe ~0,09 no κ, contra ≤0,02 de qualquer mudança de treino.

Treinar diferente vem depois. Mascarar pedaços da entrada no autoencoder e o regularizador SIGReg (que já está no repo) são as duas apostas de treino com mais evidência.

## Por que a leitura é a alavanca (em palavras simples)

- O **flatten** guarda tudo, inclusive *onde* cada coisa está: são 36.864 números para cada imagem. O SVM se perde no meio de tanto número (é o "mais ruído" que vemos).
- O **avgpool** joga fora o *onde*: fica uma média por canal, só 64 números. Perde informação.
- **Uma pirâmide fica no meio do caminho.** É como olhar a imagem inteira, depois em 4 quadrantes, depois em 16 quadradinhos, e anotar a média de cada pedaço: 64 × (1+4+16) = 1.344 números. Guarda um pouco do *onde* sem explodir o tamanho.

## Top 8, ranqueado por (ganho esperado × facilidade)

| # | Ideia | Tipo | Ganho reportado na fonte | Custo no repo | Fonte |
|---|---|---|---|---|---|
| 1 | **Pirâmide espacial** (avg 1×1+2×2+4×4 do L3, também sobre o 1x1×1280) | só eval | +2 a +6 pts (DSP vs pooling sem ordem) | ~4 linhas + YAML eval | [Deep Spatial Pyramid](https://arxiv.org/abs/1504.05277); [SPP-net](https://arxiv.org/abs/1406.4729) |
| 2 | **L2 por pixel + raiz com sinal** antes de flatten/pool | só eval | +0,4 a +3,6 pts | ~2 linhas | [Deep Spatial Pyramid](https://arxiv.org/abs/1504.05277) (Tab. 2) |
| 3 | **GeM p=3** no lugar do avg (e dentro da pirâmide) | só eval | +5 a +6 mAP sobre avg/max | 1 linha | [Radenović et al., GeM](https://arxiv.org/abs/1711.02512) |
| 4 | **Concatenar camadas** (pirâmide do L2 + do L3) | só eval | sem delta isolado; VGG/AlexNet piora no fim da rede | ~5 linhas | [Kolesnikov et al.](https://arxiv.org/abs/1901.09005); [Asano et al.](https://arxiv.org/abs/1904.13132) |
| 5 | **Pooling de 2ª ordem** (covariância 64×64 + raiz de matriz) | só eval | +2,7 a +3,8 pts com a raiz de matriz | ~6 linhas | [Improved Bilinear Pooling](https://arxiv.org/abs/1707.06772) |
| 6 | **Pré-processamento do SVM:** StandardScaler/whitening e `class_weight="balanced"` (larvae) | só eval | misto: whitening ajuda em [Coates](https://proceedings.mlr.press/v15/coates11a.html), PCA atrapalha em DSP | parâmetro do `fit_svm` | [Coates, Lee, Ng](https://proceedings.mlr.press/v15/coates11a.html); [DSP](https://arxiv.org/abs/1504.05277) |
| 7 | **Autoencoder mascarado** (tapa 50–60% em blocos, loss só no tapado, alvo normalizado por bloco) | treino | SparK: loss só no mascarado +0,8; a reconstrução pura aprende o que não serve | training_step + chave YAML | [Balestriero & LeCun 2024](https://arxiv.org/abs/2402.11337); [SparK](https://arxiv.org/abs/2301.03580); [ConvNeXt V2](https://arxiv.org/abs/2301.00808) |
| 8 | **SIGReg** (LeJEPA) como regularizador no embedding pooled | treino | CNN pequena in-domain: ResNet-34 com probe congelado 78,2 vs 67,6 do DINOv2; loss↔probe Spearman 85–99% | `core/losses/real_sigreg.py` já existe | [LeJEPA](https://arxiv.org/abs/2511.08544) |

**Regra de justiça (vale para 1–6):** toda leitura nova tem que rodar **também nas features do I-JEPA**, senão a comparação fica torta. O I-JEPA também pode ganhar com a pirâmide. Além disso, o nosso κ é de validação e o do I-JEPA é de teste. E a escolha da leitura/C precisa sair só de treino+val, nunca do teste.

---

## Detalhe de cada ideia

### 1. Pirâmide espacial (SPP / DSP)
- **Fonte:** Gao et al., *Deep Spatial Pyramid: The Devil is Once Again in the Details*, https://arxiv.org/abs/1504.05277. Também He et al., *Spatial Pyramid Pooling in Deep Convolutional Networks*, https://arxiv.org/abs/1406.4729.
- **Por que aqui:** a nossa diferença flatten↔avg é de 0,09. Logo, o *onde* importa e muitos números atrapalham. A pirâmide é exatamente o meio-termo. Com 1.344 dimensões (L3) em vez de 36.864, ela deve ficar menos ruidosa que o flatten e menos cega que o avg.
- **Efeito reportado (SVM linear, C=1):** pirâmide de 6 blocos vs FV sem ordem: Caltech-101 +4,6; Caltech-256 +4,7; SUN397 +5,9; Indoor67 +4,8; Stanford40 +5,8; VOC +0,5. Multiescala soma +0,5 a +2,5.
- **Como (mínimo):** em `eval/svm.py`, `_encode_pooled`, acrescentar `"spp"` a `EMBED_MODES` e um ramo:
  `torch.cat([F.adaptive_avg_pool2d(out, s).flatten(1) for s in (1, 2, 4)], 1)`.
  O `embed_mode` já viaja do YAML de eval (`experiments/ray/runners/eval.py:230`). Basta copiar `experiments/autoencoder/eval_svm_layerwise5_avg.yaml` com `embed_mode: spp`. Testar no L3 (eggs, protozoan) e no 1x1×1280 (larvae: 1280×21 = 26.880 dim; talvez só (1, 2)).

### 2. L2 por pixel + raiz com sinal ("ℓ2 matrix" + power norm)
- **Fonte:** a mesma DSP (Tab. 2), https://arxiv.org/abs/1504.05277. A raiz com sinal vem dos Fisher vectors (Perronnin et al.) e do bilinear.
- **Por que aqui:** cada pixel do mapa vira um vetor de 64 canais. Normalizar cada um para comprimento 1 faz o SVM olhar o *padrão* dos canais, e não o *brilho* da região. Fundo claro e ovo escuro passam a pesar igual. A raiz com sinal (`sign(x)·sqrt|x|`) achata os picos que dominam o produto interno do SVM linear.
- **Efeito reportado:** ℓ2 por pixel vs nada: Caltech-101 +1,9; Stanford40 +3,6; Indoor67 +3,4; Scene15 +0,2. **Atenção:** ℓ2 no vetor inteiro chegou a piorar (Stanford40 −1,4) e PCA piorou em todos (−0,6 a −2,8).
- **Como:** antes do `flatten`/pool, `out = F.normalize(out, dim=1)`; opcional `out = out.sign() * out.abs().sqrt()`. É um modo novo (`"l2flatten"`) ou um prefixo combinável com o `spp`.

### 3. GeM (média generalizada)
- **Fonte:** Radenović, Tolias, Chum, *Fine-tuning CNN Image Retrieval with No Human Annotation*, https://arxiv.org/abs/1711.02512.
- **Por que aqui:** o avg dilui um ovo pequeno no meio de muito fundo. O max olha só um pixel. O GeM com p=3 é o meio: dá mais peso aos pixels fortes sem ignorar o resto. Como as nossas features saem de ReLU (≥0), ele funciona sem ajuste.
- **Efeito reportado (retrieval, mAP):** avg 61,2 / max 62,2 / GeM p=3 67,9 (Oxford5k); avg 70,8 / max 68,9 / GeM 74,8 (Paris6k). O p aprendido não ganhou do p=3 fixo. É retrieval, não classificação: espere menos.
- **Como:** `out.clamp(min=1e-6).pow(3).mean((2, 3)).pow(1/3)`. Pode virar o pooling dentro da pirâmide (ideia 1). Interessa sobretudo em larvae, que hoje usa avg.

### 4. Concatenar camadas (L2 + L3)
- **Fontes:** Kolesnikov, Zhai, Beyer, *Revisiting Self-Supervised Visual Representation Learning*, https://arxiv.org/abs/1901.09005. Asano, Rupprecht, Vedaldi, *A critical analysis of self-supervision, or what we can learn from a single image*, https://arxiv.org/abs/1904.13132. No protocolo DINO/DINOv2, o probe padrão concatena as 4 últimas camadas.
- **Por que aqui:** redes sem atalho (VGG/AlexNet, como a nossa) **perdem qualidade no fim da rede** no probe linear (Kolesnikov). Nos auto-supervisionados de Asano, o probe cai depois da conv3. Isso bate com "profundidade além do L3 com pooling atrapalha". Juntar o L2 (detalhe fino) com o L3 dá ao SVM as duas escalas.
- **Efeito reportado:** sem delta isolado limpo. A evidência é a de que "a melhor camada nem sempre é a última" em redes sem atalho.
- **Como:** no laço de `_encode_pooled`, guardar `out` das camadas pedidas e concatenar as pirâmides de cada uma (`"spp_multi"`).

### 5. Pooling de 2ª ordem (covariância + raiz de matriz)
- **Fonte:** Lin & Maji, *Improved Bilinear Pooling with CNNs*, https://arxiv.org/abs/1707.06772. Também *Bilinear CNN*, https://arxiv.org/abs/1504.07889, e iSQRT-COV, https://arxiv.org/abs/1712.01034.
- **Por que aqui:** é a média de "quais canais acendem juntos" em vez de "quanto cada canal acende". Isso captura textura e casca (bom para ovo × cisto) e não depende da posição. Com 64 canais são 64·65/2 = 2.080 números.
- **Efeito reportado (fine-grained, sem fine-tuning):** raiz de matriz sobre o bilinear normalizado: CUB +2,7; Cars +3,8; Aircraft +3,2.
- **Como:** `X = out.flatten(2)` (B×64×576); `G = X @ X.transpose(1,2) / 576`; raiz via `torch.linalg.eigh` (autovalores^0,5); pegar o triângulo superior; aplicar raiz com sinal + L2. Cuidado: no 1x1×1280 vira 820k dimensões. Usar só no L3.

### 6. Pré-processamento do SVM (whitening/scaler, peso de classe)
- **Fontes:** Coates, Lee, Ng, *An Analysis of Single-Layer Networks in Unsupervised Feature Learning*, AISTATS 2011, https://proceedings.mlr.press/v15/coates11a.html (whitening é peça central). Contra: DSP (PCA piora) e Radenović (PCA-whitening pode piorar o max; whitening *aprendido* ganha sempre).
- **Por que aqui:** hoje o `fit_svm` (`core/svm.py:143`) usa C=100 fixo, sem scaler e sem peso de classe. Larvae tem 2 classes desbalanceadas e o κ castiga errar a classe pequena; `class_weight="balanced"` é o remédio clássico. O scaler iguala canais de escalas diferentes, o mesmo motivo pelo qual o MAE põe um BatchNorm sem afim antes do probe linear.
- **Efeito esperado:** misto. Teste rápido, não aposta.
- **Como:** respeitando "um SVM, uma métrica", é **um parâmetro novo no `fit_svm`** (como o `scaler` já é), nunca outro SVM. Precisa ser aplicado igual ao I-JEPA. C escolhido só por validação cruzada dentro do treino.

### 7. Autoencoder mascarado (denoising do jeito certo)
- **Fontes:**
  - Balestriero & LeCun, *Learning by Reconstruction Produces Uninformative Features For Perception*, https://arxiv.org/abs/2402.11337.
  - Tian et al., *SparK: Designing BERT for Convolutional Networks*, https://arxiv.org/abs/2301.03580.
  - Woo et al., *ConvNeXt V2 (FCMAE)*, https://arxiv.org/abs/2301.00808.
  - Chen, Liu, Xie, He, *Deconstructing Denoising Diffusion Models for SSL (l-DAE)*, https://arxiv.org/abs/2401.14404.
- **Por que aqui:** um autoencoder puro gasta a capacidade copiando o que mais varia em pixel (cor, fundo, iluminação), e isso não é o que separa as classes. No TinyImageNet, o subespaço de 90% da variância dá 45% de acerto e o de 20% da variância dá 55%. **Mascarar ajuda; ruído gaussiano no pixel não** (provado no artigo). Isso também explica o "a loss não prevê o κ".
- **Efeito reportado:**
  - SparK: loss só no mascarado +0,8; conv esparsa vs "zerar o buraco" +0,9; decoder hierárquico +0,5 (ImageNet FT). Probe linear: SparK 54,7 vs BEiT 15,7.
  - l-DAE: ruído no espaço de pixel completo 23,6% vs no espaço PCA de baixa dimensão 64,5%.
- **Como (mínimo, sem conv esparsa):** no `training_step` de `methods/autoencoder/autoencoder_flim_module.py` (L602), trocar 50–60% dos blocos 8×8 da entrada pela média LAB (não zero, para não deslocar a distribuição que o SPiFiL calibrou). Calcular a BCE só nos blocos tapados, mantendo o peso 0,1 do fundo. A chave nova (`mask_ratio`) entra em `RUNNER_ARGS` (`experiments/ray/schema.py:50`) e é lida em `experiments/ray/runners/growth.py`. YAML irmão: `experiments/autoencoder/growth_layerwise5_aug.yaml`.
- **Risco:** o "mask pattern vanishing" do SparK (conv densa come a borda do buraco a cada camada). Com 3 conv5 isso é pequeno, mas existe.

### 8. SIGReg (LeJEPA) como regularizador
- **Fonte:** Balestriero & LeCun, *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*, https://arxiv.org/abs/2511.08544.
- **Por que aqui:** o SIGReg empurra o embedding a ficar "redondo" (gaussiano isotrópico), sem direções mortas nem dominantes. É exatamente o formato em que um classificador linear trabalha melhor. Funciona com CNN.
- **Efeito reportado:**
  - Galaxy10, pré-treino no próprio domínio, probe congelado com todos os dados: ResNet-34 78,2 e ConvNeXt-V2 Nano 76,5, contra DINOv2 ViT-S 67,6 e DINOv3 71,4.
  - A loss correlaciona ~85% (Spearman) com o probe, e ~99% com a escala λ^0,4. Isso daria um critério sem rótulo, coisa que hoje nos falta.
  - Defaults: λ=0,05, 1.024 fatias, batch ≥128.
- **Como:** `core/losses/real_sigreg.py` já existe. Somar `λ·SIGReg(avgpool(L3))` à BCE no `training_step`; λ entra como chave YAML (mesmo caminho da ideia 7). A versão completa precisa de 2 vistas (hoje V=1). Começar só com o SIGReg numa vista, como regularizador.

---

## (c) Leis de escala: como ajustar 1/5/25/50/75/100% × largura × profundidade

- **Forma a ajustar:** Rosenfeld et al., *A Constructive Prediction of the Generalization Error Across Scales*, https://arxiv.org/abs/1909.12673.
  - Erro conjunto: `ε(n, m) = a·n^(−α) + b·m^(−β) + c∞`, mais um "envelope" que leva ao nível de chute: `ε̂ = ε0·|ε̃/(ε̃ − iη)|`.
  - Para nós: `ε = 1 − κ`, e o chute é κ = 0, logo `ε0 = 1`.
  - Regra deles: pelo menos 2 pontos por eixo e nunca menos pontos que parâmetros (≤6). 6 frações × 3 larguras × 3 splits = 54 pontos dá folga.
  - Erro de extrapolação reportado: 4,5% ± 4,7% no ImageNet, a partir de 1/16 do modelo e 1/8 dos dados.
- **Profundidade não é monotônica para nós** (L4+ com pooling piora). Uma potência simples não descreve isso. Use *Broken Neural Scaling Laws*, Caballero et al., https://arxiv.org/abs/2210.14891, que modela curvas que quebram e voltam.
- **O que esperar:**
  - Kolesnikov (https://arxiv.org/abs/1901.09005): largura e tamanho da representação sobem o probe linear de forma consistente, também com 5–10% dos dados (RevNet 4×→16×: 47,3→53,7; representação maior numa rede fina: 31→43).
  - Cole et al. (https://arxiv.org/abs/2105.05837): o ganho de mais dados de pré-treino satura (~500k imagens no SimCLR).
  - Asano et al. (https://arxiv.org/abs/1904.13132): camadas iniciais aprendidas de **1 imagem** com augmentation empatam com milhões. Isso sustenta o SPiFiL com 1 imagem por classe no L1–L2; a diferença aparece nas camadas mais fundas.
- **Cuidado prático:** ajustar a lei no κ **flatten** (ou na pirâmide), não no avg. O avg é 3× mais ruidoso e o ruído vira um "expoente" falso. Pesar cada ponto pelo inverso da variância entre splits.

## (d) FLIM / grupo do Falcão: o que pode estar faltando

| Ideia | Fonte | Uso aqui |
|---|---|---|
| **Viés = −⟨média do patch, kernel⟩** no lugar da normalização por bloco; mesmos pontos (BoFP) em todos os blocos | Martinelli et al., *FLIM Networks with Bag of Feature Points*, https://arxiv.org/abs/2602.20845 | Variante do SPiFiL nas camadas 2–3 sem z-score por bloco. Mais leve e menos sensível à normalização (lembra o −0,33 κ do `imagenet_norm`). F-measure 0,860 vs 0,844, e 0,792 vs 0,604 em zero-shot (Entamoeba). |
| **Decoder adaptativo:** pesos + para canais de objeto e − para canais de fundo, por imagem | *FLIM-based Salient Object Detection Networks with Adaptive Decoders*, https://arxiv.org/abs/2504.11112 | Versão para classificação: **pooling só no objeto** (média dentro da máscara de fundo que já existe para o peso 0,1). É outra leitura só de eval, irmã da 1. |
| **DISF automático** para marcadores (superpixels → pontos característicos) nos mesmos 3 datasets (ovos, larvas, cistos) | Miranda (orient. Falcão), dissertação *A Study on Lightweight Networks for the Classification of Enteroparasites*, defesa 17/04/2026, https://www.ic.unicamp.br/en/evento/defesa-de-mestrado-de-maria-anglica-krger-miranda/ | É o vizinho mais próximo do SPiFiL. Vale pedir o texto: tem κ nos nossos datasets e pode ter escolhas (nº de pontos, kernel) prontas. |
| **Escolha iterativa de imagens:** adiciona a imagem que o modelo atual trata pior e fica com ela só se a validação sobe | https://arxiv.org/abs/2602.20845 (trabalho futuro); SIBGRAPI 2024, https://sol.sbc.org.br/index.php/sibgrapi/article/view/31697 | Trocar "1 imagem por classe" por 2–3 escolhidas assim, só com treino+val. |
| FLIM original para classificação (filtros de marcadores, sem backprop) | de Souza & Falcão, https://arxiv.org/abs/2008.03549 | Linha de base histórica; nada novo além do que já temos. |

## Fora do top 8, mas anotado

- **CroW** (pesos espaciais pela soma das ativações + pesos de canal por raridade), Kalantidis et al., https://arxiv.org/abs/1512.04065: outra leitura só de eval, parecida com o "pooling só no objeto". Não conferi os números.
- **l-DAE com ruído no espaço PCA de patches**: forte no ImageNet (ver ideia 7), mas exige mudar a entrada do decoder. Mais caro.
- **VICRegL** (VICReg global + local para CNN), https://arxiv.org/abs/2210.01571: o local ajuda segmentação e **mantém** a classificação. Não promete ganho no nosso κ.
- **Joint-embedding vence reconstrução quando o irrelevante tem grande amplitude** (Van Assel et al., https://arxiv.org/abs/2505.12477). É o argumento teórico para as ideias 7–8: o fundo e a cor são "irrelevante grande" aqui.

## Ordem sugerida de execução

1. Um único job de eval com os modos `spp`, `l2flatten`, `gem`, `spp_multi`, `cov` sobre os checkpoints *best* que já existem (bp01w64 L3; c1280 L4 larvae), **e os mesmos modos nas features do I-JEPA**. Não precisa treinar nada, e os modos rodam um por vez (SVM saturando CPU).
2. Se algum modo bater o flatten em ≥0,02 nos 3 splits, ele vira a leitura padrão do growth.
3. Só depois: growth com `mask_ratio` (ideia 7) e com SIGReg (ideia 8), lidos na leitura vencedora do passo 1.
