# [A5] Técnicas de convolução para a nossa CNN que cresce (2026-11-01)

## A resposta, em uma frase

**A melhor aposta é o GRN do ConvNeXt V2 logo depois da nossa L4 1×1×1280.** Ele foi feito para o mesmo problema que temos (autoencoder mascarado + camada de expansão larga), começa como identidade, custa 2.560 parâmetros e ganhou do SE e do CBAM no próprio artigo. Depois vêm o ECA (o "SE de 80 parâmetros"), a leitura densa L3+L4 e a troca do max-pool por BlurPool/SoftPool.

## GeM em 3 linhas (nível criança)

1. Cada canal é um mapa de "onde eu vi este padrão"; a gente precisa resumir o mapa num número só.
2. A **média** dilui um ovo pequeno num fundo grande. O **máximo** escuta só o ponto mais forte, e um grão de sujeira pode ganhar.
3. O **GeM** eleva cada valor ao cubo, tira a média e tira a raiz cúbica: os pontos fortes falam mais alto, mas todo mundo vota. É o meio-termo entre média e máximo.

## As 3 perguntas que valem para tudo

- **Onde entra no código:** o encoder é um `nn.ModuleDict` de `nn.Sequential` montado por `build_encoder_from_arch` (`flim/arch.py:79`): `conv → [ReLU] → [MaxPool]` por camada. Há dois jeitos de entrar:
  - **(A) módulo novo depois da conv:** inserir em `self.model.encoder.blocks[f"conv{n}"]`, como o BatchNorm entrou no commit `e26349c` (`methods/autoencoder/autoencoder_flim_module.py`, flag `batch_norm`, `.insert(1, ...)`). Uma flag booleana no módulo + chave em `RUNNER_ARGS` + YAML irmão de `growth_layerwise5_bn.yaml`.
  - **(B) mudança no bloco:** um tipo novo em `pool_desc["type"]` ou na conv dentro de `build_encoder_from_arch`, também no espelho `build_decoder_from_arch`.
- **SPiFiL:** a camada n+1 estima os filtros com patches da *saída* do prefixo. Se um módulo entra depois da conv n, o SPiFiL da n+1 precisa ver a saída **já com o módulo**. Então é preciso conferir que o prefixo passa pelo `Sequential` inteiro. O caminho do NAS (`flim/progressive.py:61`) recusa `groups≠1` e só aceita kernel 3/5: depthwise não entra sem SPiFiL novo.
- **Por que precisa de init identidade:** os kernels FLIM carregam a normalização dos marcadores no par kernel/viés. Qualquer coisa que mude a escala da ativação no início desloca o corte da ReLU que o SPiFiL calibrou. O repo já mediu isso: `imagenet_norm` custou −0,33 κ. Uma porta com `sigmoid` inicializada em zero dá 0,5 e **corta tudo pela metade** no primeiro passo. A regra é a do Net2Net/ReZero/LayerScale: **toda peça nova começa como identidade** (γ=0 com atalho `+x`, ou porta `2·σ(z)` com z começando em 0).

## Ranking (ganho esperado × facilidade × compatibilidade)

| # | Técnica | Ganho na fonte | Compatível com o growth? | Onde | Init identidade? | Parâmetros a mais |
|---|---|---|---|---|---|---|
| 1 | **GRN** (ConvNeXt V2) após ReLU da L4 1×1×1280 | FCMAE ConvNeXt-B: 83,7 → **84,6**; SE 84,4 e CBAM 84,5 com +20M params; sem o atalho, 84,0 | **Sim, ótima.** Nasceu para MAE em conv; vai exatamente na camada de expansão, onde eles viram o colapso | (A) `blocks["conv4"]` | **Já é**: γ=β=0 e `+x` | 2×1280 = 2.560 |
| 2 | **ECA-Net** (canal, conv1d k=3) após cada ReLU | ResNet-50 **+2,28** top-1, com ~80 params no total | Sim. É o SE sem MLP; o A7 já tem o encaixe | (A), mesmo lugar do SE | Precisa: porta `2σ(z)`, conv1d com peso 0 | ~3–5 por camada |
| 3 | **Leitura densa L3 + L4** (GeM 64 + GeM 1280 = 1.344 dim) | DenseNet; Kolesnikov: rede sem atalho perde qualidade no fim, o probe linear lê melhor com atalho | Sim, nada muda no treino. O atalho de verdade (`x + f(x)`) não encaixa porque largura e resolução mudam | só eval (`_encode_pooled`) | não se aplica | 0 |
| 4 | **BlurPool** (max stride 1 + blur 3×3 stride 2) | ImageNet ~+0,5 a +1, e mais consistência a deslocamento | Sim, sem parâmetro. A próxima camada SPiFiL vê o mapa borrado. Exige recrescer do L1 | (B) tipo `max_blur_pool` no builder | não tem parâmetro | 0 |
| 5 | **SoftPool** no lugar do max-pool | +1,17 em média (máx. +1,51 ResNet-18) treinando do zero | Sim, sem parâmetro. Fica entre max e avg dentro de cada janela 2×2 | (B) tipo novo de pooling | não tem parâmetro | 0 |
| 6 | **Coordinate Attention** | MobileNetV2 72,3 → **74,3** (SE 73,5; CBAM 73,6) | Sim. Guarda *onde* (linha e coluna), útil para objeto pequeno descentrado | (A) após ReLU | Precisa: porta `2σ`, última conv com peso 0 | ~2C²/r |
| 7 | **GCNet** (contexto global somado) | ResNet-50 **+0,8** top-1; COCO +2 AP | Sim. É soma com atalho | (A) após ReLU | Fácil: última conv com peso 0 vira identidade | ~2C²/r |
| 8 | **SimAM** (atenção 3D sem parâmetro) | SimAM-R50 77,45 top-1 (~+1 sobre a base) | **Média.** Multiplica por `σ(·)` ∈ (0,5, 1), muda a escala e não dá para começar como identidade sem pôr um α | (A) após ReLU | Só se envolver em `x + α·(SimAM(x) − x)` com α=0 | 0 (1 com α) |

**Meu corte prático:** fazer 1 (GRN) e 3 (leitura densa) já, porque não brigam com o SPiFiL. O 2 (ECA) sai como variante barata do SE do A7. Os 4–5 só valem se houver orçamento para recrescer desde o L1.

## O resto, com o motivo de não estar no topo

| Técnica | Ganho na fonte | Compatibilidade | Veredito |
|---|---|---|---|
| **SE** | ResNet-50 erro 24,8 → 23,3 | Sim, com porta `2σ` | O A7 já está fazendo |
| **CBAM** (canal + espacial) | ResNet-50 erro 24,56 → 22,66 | Sim, com porta `2σ` | A parte espacial briga com o GeM, que já pesa espaço; ECA/CA fazem o mesmo por menos |
| **SKNet** (escolhe kernel 3 ou 5 por imagem) | SKNet-50 erro 20,8 vs ResNeXt-50 22,2 | **Ruim**: precisa de 2 bancos de filtro por camada e o SPiFiL estima 1 | Só com SPiFiL por ramo |
| **Depthwise separable / inverted residual** (MobileNetV2) | eficiência, não acurácia | **Não**: `groups≠1` é recusado (`flim/progressive.py:63`); SPiFiL estima filtro cheio | Fora |
| **Bloco ConvNeXt** (dw 7×7 + LN + MLP 4×) | ResNet-50 76,1 → ConvNeXt-T 82,1 (receita toda) | **Não**: depthwise + LayerNorm | Fora; só o GRN se aproveita |
| **Residual de verdade** (`x + f(x)`) | ResNet | **Difícil**: a largura muda (32→48→64→1280) e há pool. Precisa de projeção 1×1. Com `f` em zero o SPiFiL vira enfeite; sem zero, a soma muda a escala | Use a leitura densa (#3) |
| **Conv com stride no lugar do max-pool** | All-Conv: empata no CIFAR | Possível, mas tira o max que o FLIM sempre usou | Ganho esperado ~0 |
| **LIP** (pooling com importância aprendida) | ganho relatado no ImageNet/COCO (não conferi o número) | Média: tem parâmetros e precisa começar igual ao avg | Depois do SoftPool |
| **Dilatada** | campo receptivo maior sem parâmetro | O builder já lê `dilation_rate` do JSON (`flim/arch.py:97`), mas o patch do SPiFiL precisa ser amostrado com a mesma dilatação. Na L4 1×1 não muda nada | Testar só se o SPiFiL suportar; barato |
| **ASPP / Inception / MixConv** (vários kernels juntos) | segmentação (ASPP); MixNet em mobile | **Média-ruim**: vários ramos, um SPiFiL por ramo | Fora por ora |
| **Deformável** (DCNv2) | sobretudo detecção | Sim (offsets em zero = conv normal), precisa de `torchvision.ops.deform_conv2d` | Ganho em classificação incerto |
| **GroupNorm / LayerNorm2d** | GN vence BN em batch 2 (erro 24,1 vs 34,7); empata em batch 32 | **Não começa como identidade** (normaliza na hora) → desloca o viés FLIM | Esperar o resultado do BN já lançado: se o BN perder, estes também perdem |
| **Weight Standardization** | GN+WS ≈ BN | **Não**: reescreve os kernels SPiFiL (média 0, variância 1 por filtro), que já vêm normalizados pelos marcadores | Redundante ou destrutivo |

## O que a literatura diz sobre o probe linear de CNN auto-supervisionada

- **Atalho importa para ler a última camada** (Kolesnikov et al., 1901.09005). Com atalho (RevNet/ResNet), o probe melhora até a última camada. Sem atalho (VGG, AlexNet, como a nossa), ele piora no fim. Isso explica "L4+ com pooling piora", e é o motivo da leitura densa (#3).
- **Camada de expansão larga colapsa sob MAE** (ConvNeXt V2). Canais mortos ou saturados e muito redundantes aparecem **na camada de expansão**, que é a nossa 1×1×1280. O GRN conserta, e no mesmo teste o BN *derrubou* para 80,5.
- **O GRN agrega com L2, não com média:** GAP 83,7, L1 84,3, L2 84,6. Ou seja, o resumo do mapa que dá certo ali é "o tamanho do sinal", o mesmo espírito do GeM.

## Fontes

- **GRN, ConvNeXt V2:** Woo et al., *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders*, https://arxiv.org/abs/2301.00808
- **SE:** Hu et al., *Squeeze-and-Excitation Networks*, https://arxiv.org/abs/1709.01507
- **ECA:** Wang et al., *ECA-Net: Efficient Channel Attention*, https://arxiv.org/abs/1910.03151
- **CBAM:** Woo et al., *CBAM: Convolutional Block Attention Module*, https://arxiv.org/abs/1807.06521
- **Coordinate Attention:** Hou et al., *Coordinate Attention for Efficient Mobile Network Design*, https://arxiv.org/abs/2103.02907
- **SimAM:** Yang et al., *SimAM: A Simple, Parameter-Free Attention Module*, ICML 2021, https://proceedings.mlr.press/v139/yang21o.html
- **SKNet:** Li et al., *Selective Kernel Networks*, https://arxiv.org/abs/1903.06586
- **GCNet:** Cao et al., *GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond*, https://arxiv.org/abs/1904.11492
- **BlurPool:** Zhang, *Making Convolutional Networks Shift-Invariant Again*, https://arxiv.org/abs/1904.11486
- **SoftPool:** Stergiou et al., *Refining activation downsampling with SoftPool*, https://arxiv.org/abs/2101.00440
- **LIP:** Gao et al., *LIP: Local Importance-based Pooling*, https://arxiv.org/abs/1908.04156
- **All-Conv:** Springenberg et al., *Striving for Simplicity: The All Convolutional Net*, https://arxiv.org/abs/1412.6806
- **MobileNetV2:** Sandler et al., https://arxiv.org/abs/1801.04381
- **ConvNeXt:** Liu et al., https://arxiv.org/abs/2201.03545
- **ResNet:** He et al., https://arxiv.org/abs/1512.03385
- **DenseNet:** Huang et al., https://arxiv.org/abs/1608.06993
- **Dilatada:** Yu & Koltun, https://arxiv.org/abs/1511.07122
- **ASPP (DeepLab):** Chen et al., https://arxiv.org/abs/1606.00915
- **Inception:** Szegedy et al., https://arxiv.org/abs/1409.4842
- **MixConv:** Tan & Le, https://arxiv.org/abs/1907.09595
- **Deformável:** Dai et al., https://arxiv.org/abs/1703.06211; DCNv2, https://arxiv.org/abs/1811.11168
- **GroupNorm:** Wu & He, https://arxiv.org/abs/1803.08494
- **Weight Standardization:** Qiao et al., https://arxiv.org/abs/1903.10520
- **Probe linear em SSL:** Kolesnikov et al., *Revisiting Self-Supervised Visual Representation Learning*, https://arxiv.org/abs/1901.09005
- **Init identidade:**
  - Net2Net: Chen et al., https://arxiv.org/abs/1511.05641
  - ReZero: Bachlechner et al., https://arxiv.org/abs/2003.04887
  - LayerScale: Touvron et al., https://arxiv.org/abs/2103.17239
- **GeM:** Radenović et al., https://arxiv.org/abs/1711.02512

Números de ImageNet e COCO não são números de κ em microscopia. Servem para ordenar, não para prometer.
