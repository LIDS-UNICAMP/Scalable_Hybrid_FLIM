# L_new_papers: como montar as camadas 3+ sem backprop, e o que fazer com backprop depois

**Resposta curta:** a literatura nova aponta para quatro trocas. Encha os filtros com dado real em vez de ruído, dê nota pela resposta do filtro por imagem em vez de olhar o patch, faça a largura pagar o que o pooling tira, e depois deixe o backprop solto. O ganho esperado do backprop é pequeno e fica nos datasets pequenos.

Essa frase é aposta minha, tirada dos fatos abaixo. Ninguém testou o nosso caso exato.

## Como ler este relatório

- **[FATO]**: está no artigo, com número ou trecho copiado.
- **[APOSTA]**: ideia minha. Nenhum artigo testou.
- **[NOSSO]**: medida do projeto, com arquivo e linha.
- **[CLÁSSICOS]**: sonda do agente que cobre Coates & Ng, PCANet e k-means. Ainda em andamento, eggs ou protozoan, split 1, 1000 imagens. Serve de pista, não de resultado.
- **[SONDA]**: sonda do agente de inicialização, em 240 imagens (seção F). Também é pista.

Sete agentes leram os artigos no texto do PDF, e o SWIM li eu. Conferi por amostragem 88 trechos citados contra o texto baixado, e todos bateram. Os outros números vêm dos relatórios dos agentes, sem conferência minha. Os textos estão em `<scratchpad>/law/newpapers/txt/`, com `<scratchpad>` = `/tmp/claude-1033/-dados-home-moliveira-Scalable-Hybrid-FLIM/990a33ea-4327-4228-ac99-2136fb213621/scratchpad`.

## 1. Por que a camada 3+ fica fraca: quatro contas

Pense na última camada (1×1) como um time de 1280 jogadores. De 100 a 400 foram escolhidos vendo o jogo. Os outros 880 a 1180 foram sorteados de olhos fechados. E o técnico, que é o backprop, quase não mexe no time.

1. **L3: a régua quebra.** [NOSSO] O candidato tem D = 1200 números (48 canais × 5 × 5) e há só N = 100 a 400 candidatos na receita oficial, 50 sementes × classes (`p0/R1.md:78`; `L3_grow/conv3-kernels.npy`: 1200 × 64). A Mahalanobis precisa de N maior que D, senão mede pouco.
2. **L4: o problema é outro.** [NOSSO] Ali D = 64 e N já é maior que D, mas se pedem 1280 filtros com só 100 a 400 candidatos (`conv4-kernels.npy`: 64 × 1280). Faltam candidatos, não dimensão. E a dimensão real é ainda menor: posto efetivo perto de 30 (`p1/Y4.md`).
3. **A rede encolhe.** [NOSSO, conta minha] Com 32/48/64 canais em grades 99/49/24, os valores por imagem caem de 313.632 para 115.248 e para 36.864, ou 8,5 vezes menos. [FATO] O SoftHebb multiplica a largura por 4 a cada camada, enquanto o pooling divide a área por 4. Isso mantém esse número quase constante (conta minha).
4. **A nota pode jogar fora o que serviria depois.** [APOSTA] Fisher por classe é um critério guloso. [FATO] Em treino local, camada gulosa perde informação que a seguinte não recupera (InfoPro, Sakamoto, ContSup).

## 2. O que o backprop faz depois: o que já medimos

- [NOSSO] **O meio é reescrito.** Kernels de L2 e L3 saem com cosseno 0,59 e 0,54 do início (`A_reports/october/discovery_act_drift.md:28-29`).
- [NOSSO] **A 1×1 fica como está.** Cosseno 0,97, nos filtros SPiFiL e nos aleatórios (`discovery_act_drift.md:30,47`). Ou seja, os pesos que você põe na 1×1 são os que ficam. A ativação muda mais (0,45), porque a L3 debaixo dela mudou.
- [NOSSO] **Segurar perto do SPiFiL piora.** Âncora L2-SP: −0,069 em eggs e −0,048 em larvae (`discovery_insights_board.md:1774,1784`). Soltar tudo ajuda um pouco: +0,010 em eggs e +0,004 em larvae (`discovery_insights_board.md:1711`).
- [NOSSO] **Mais camadas pioram hoje.** Eggs, 60 épocas, gem: L4 0,875, d5 0,857, d6 0,851 (`discovery_insights_board.md:2009`).
- [NOSSO] **O backprop apaga o contraste objeto × fundo que o SPiFiL trazia.** Na L3, de 2,86 para 0,99 (`discovery_act_drift.md:29`).
- [NOSSO] **Como inicializador, o SPiFiL é bom onde o aleatório vai mal.** Ganha 0,050 a 0,072 em protozoan e empata em eggs, no split 1 (`discovery_insights_board.md:919,935`).

## 3. O que a literatura concorda

**Por quanto tempo uma boa inicialização dura.** [FATO] Há três efeitos, com durações diferentes. A separação em três é leitura do agente, os números são dos artigos.

| Efeito | Quanto dura | Evidência |
|---|---|---|
| Escala boa dos pesos | Só acelera. Some com treino longo | Ciga 2021: ganho de 0,7 a 5,5% em 100 épocas, igual em 1000. Raghu: só velocidade |
| Conteúdo bom dos filtros | Não acaba por relógio, acaba por quantidade de dados | Lagani: +4,6 com 1% dos rótulos, −0,3 com 100%. Matsoukas: ganho claro com 3.662 imagens, quase nada com 224 mil. Deep Incubation: +7,2 com 25%, +0,6 com 100% |
| Dano de treinar demais a fase 1 | Pode não sumir nunca | DASH, Ash & Adams. Com augmentation e schedule o dano cai para 0,4 a 1,9 pontos |

**Quanto o backprop rende depois de pesos sem backprop global.** [FATO]

| Artigo | Dado | Sem backprop | Com backprop | Do zero |
|---|---|---|---|---|
| SCFF | STL-10 | 77,30 | 80,13 | 77,02 |
| SoftHebb | STL-10 | 76,23 | 78,5 | 74,51 |
| Lagani, L5 | CIFAR-10, 1% | 38,46 | 39,75 | 35,18 |
| Lagani, L5 | CIFAR-10, 25% | 56,46 | 64,77 | 68,04 |
| Lagani, L5 | CIFAR-10, 100% | 61,16 | 84,38 | 84,71 |
| ReduNet | MNIST | 0,898 | 0,978 | 0,932 |
| SPHeRe, só L1 | CIFAR-10 | 78,78 | 79,93 | 79,46 |
| FFCL | Raio-X, só local | n/d | 62,50, AUC 50 (colapso) | 73,08 |
| FFCL | Raio-X, local e global | n/d | 80,45 | 73,08 |
| MS-FLIM | BraTS, Dice ET | 0,746 | 0,747 ("unnecessary") | 0,665 |

Todos usam rótulo no backprop.

**A lacuna.** [FATO] Nenhum artigo mede pesos sem gradiente, seguidos de autoencoder sem rótulo, seguidos de SVM congelado. Nenhum dá a curva por época de "início sem gradiente contra início aleatório". Medir isso é contribuição nova.

**Discordâncias entre artigos, que só o nosso teste resolve.**
- **Patch cru ou branqueado.** Thiry: patches sorteados e branqueados valem tanto quanto otimizados. Pre-Warm: patch cru como início nunca ajudou em dados coloridos. A diferença provável é o branqueamento [APOSTA].
- **Direção do sorteio.** Thiry, ACnnL e Forward Projection branqueiam. Rainbow reforça as direções de muita variância. [CLÁSSICOS] Na sonda de eggs, filtro aleatório branqueado deu 0,9324 na leitura padronizada e 0,7265 na leitura `pw`. A leitura muda a resposta.
- **Soltar cedo ou tarde.** Ash diz que poucas épocas já danificam. DASH diz que até um limiar não custa nada.
- **Taxa ao soltar.** Gomez: 0,1 ganhou de 0,01. Layer Grafted usa baixa. [NOSSO] Em nós, mais taxa ajudou pouco.

## 4. Fichas dos artigos

Regra de leitura: "Regra" é sempre [APOSTA], e o resto é [FATO]. Tudo importa do SPiFiL e fica no repositório principal: `Learner` aceita `scorer=`, `metric=`, `selector=`, `allocator=`, `strategy=`. O SPiFiL não é editado.

### A. Forward-Forward e derivados

Atenção: todo FF convolucional usa gradiente dentro da camada. O que ele não tem é gradiente entre camadas. Não é o "sem gradiente" do SPiFiL.

**SCFF.** Chen, Liu, Laydevant, Grollier. Nature Communications 2025. https://arxiv.org/abs/2409.11593
- Ideia: sem rótulo, o par positivo é a imagem somada com ela mesma, e o negativo é somada com outra.
- Resultado: CIFAR-10 80,75 contra BP 84,9. STL-10 por camada 66,2 / 72,4 / 76 / 77,3 contra BP 67 / 73,3 / 76,2 / 77. Tiny ImageNet 35,67.
- Código: github.com/neurophysics-cnrsthales/contrastive-forward-forward
- Escala: 3 a 5 conv, até 6144 filtros, imagens até 96 px.
- Depois com backprop: STL-10 sobe para 80,13 (+2,83). Sem curva e sem desvio. O início usou imagens sem rótulo, então parte do ganho pode vir disso.
- Regra: nota do candidato = energia no par "mesma imagem" menos no par "classes diferentes". Um número por candidato, sem matriz D × D.

**FFCL.** Ahamed, Chen, Imran. MIDL 2023. https://arxiv.org/abs/2305.02927
- Ideia: contraste local, depois contraste global, depois backprop na tarefa.
- Resultado (Tabela 1, ResNet-18, pneumonia): aleatório 73,08; ImageNet 76,76; local e global 80,45; só global 76,92; só local 62,50 com AUC 50; local e global sobre ImageNet 72,28.
- Código: não indicado.
- Escala: ResNet-18 e 34, 224 px, 5232 imagens, 2 classes, 100 épocas.
- Depois com backprop: é o desenho do artigo. Início só local colapsou. Sobre pesos já bons, atrapalhou (−4,48). Uma rodada, sem desvio.
- Regra: medir κ logo depois da montagem, para pegar colapso cedo.

**DeeperForward.** Sun, Zhang, He, Wen, Shen, Xie. ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/file/7dd309df03d37643b96f5048b44da798-Paper-Conference.pdf
- Ideia: canais por classe, e a nota da camada fica separada da feature passada adiante.
- Resultado: CIFAR-10 com 4 / 14 / 17 camadas: 79,49 / 81,76 / 86,22. Com 33 e 100: 85,43 e 83,06. BP ResNet18 94,03. Sem residual 81,16. Fim da rede: FC 76,92, conv 5×5 82,85, conv 3×3 83,25.
- Trecho: "significant drops after each 1×1 convolution layer, indicating that our method is unsuitable for the bottleneck structure" (p.17).
- Código: github.com/tobysunsun/deeperforward
- Escala: até 100 camadas, 800 canais, 32 px, 45 mil imagens.
- Depois com backprop: não testado.
- Regra: trocar a 1×1 final por 3×3, porque 1×1 enxerga um pixel só. Ver caminho 2.

**BiCovG + HGB.** Jiang, Al-Hashimi, Xu. Preprint 2026. https://arxiv.org/abs/2605.04346
- Ideia: a nota por canal perde a relação entre canais, então soma projeções entre canais e várias escalas espaciais.
- Resultado: ImageNet-100 73,01. Com blocos de 4 camadas 83,98. BP 87,61. Distância para o BP vai de 5% em CIFAR-100 a 28% em ImageNet-100.
- Código: não indicado.
- Escala: VGG-16, 224 px, 100 classes, 34,9M parâmetros.
- Depois com backprop: não testado. Blocos são gradiente interno, não início.
- Regra: no Scorer, medir separação também em grade 2×2 das respostas. Escala rendeu mais que correlação entre canais.

**Scalable FF.** Krutsylo. Preprint 2025. https://arxiv.org/abs/2501.03176
- Ideia: nota por classe sem separar canais, com backprop só dentro do bloco.
- Resultado: CIFAR-10 com 150 mil parâmetros: 70,34 por camada, 81,38 em blocos, BP 79,10. Imagenette com 1000 imagens: 58,95 ± 1,62 contra BP 49,93 ± 2,04.
- Código: github.com/DentonJC/sff (conferir, ligadura no PDF).
- Escala: ResNet18, MobileNetV3, até 224 px.
- Depois com backprop: não testado.
- Regra: testar primeiro no dataset de 741 imagens, onde o artigo sugere a maior vantagem.

**CaFo.** Zhao, Wang, Li, Jin, Lang. Preprint 2023. https://arxiv.org/abs/2303.09728
- Ideia: blocos conv que podem nem ser treinados, com classificador em cada bloco.
- Resultado: erro CIFAR-10: blocos aleatórios 32,57; DFA 30,52; BP 26,30.
- Código: github.com/Graph-ZKY/CaFo
- Escala: 3 a 10 blocos, 32 px.
- Depois com backprop: não testado.
- Regra: sempre rodar o piso aleatório na mesma largura e leitura.

**CwComp.** Papachristodoulou et al. AAAI 2024. https://arxiv.org/abs/2312.12668
- Ideia: cada classe é dona de um grupo de canais, a mesma ideia do `UniformAllocator`.
- Resultado: erro 21,89% em CIFAR-10, 48,77% em CIFAR-100. No CIFAR-100, treinou 2 blocos nas 20 superclasses e só o 3º nas 100.
- Código: github.com/andreaspapac/CwComp
- Escala: 2 a 3 blocos, 32 px.
- Depois com backprop: não testado.
- Regra: em aid (30 classes) e remote_sensing (14), `Allocator` por grupo de classes nas camadas do meio. Com 2 classes não muda nada.

**Scodellaro et al.** Scientific Reports 2025. https://arxiv.org/abs/2312.14924
- Ideia: espalhar o rótulo pela imagem para a conv enxergar.
- Resultado: CIFAR-10: BP 85,4, morfológico 68,6, Fourier 60,9. É a linha de base honesta do FF sem truques: perde 17 pontos.
- Código: Zenodo 10.5281/zenodo.11571949, "upon request".
- Escala: 6 conv, 32 px.
- Depois com backprop: não testado.
- Regra: medir κ por camada. Separação boa só em camada baixa é sinal de atalho.

**Família PEPITA: descartar.** Dellaferrera & Kreiman, ICML 2022, https://arxiv.org/abs/2201.11665. Srinivasan et al., ICLR 2024, https://arxiv.org/abs/2302.05440. Caillon et al. (FOTON), ECAI 2025, https://arxiv.org/abs/2512.20668.
- Ideia: dois passes para frente, o segundo com a entrada perturbada pelo erro.
- Resultado: 1 conv em CIFAR-10: 56,33 contra BP 64,99. FOTON em CIFAR-100: 1 conv 28,48, 2 conv 21,98, BP 34,45.
- Escala: 1 a 2 conv, 32 px. Depois com backprop: não testado.
- Regra: não usar. Nenhum chega à nossa terceira camada.

**Vistos com menos fundo.**
- Ghader et al. 2606.09928: 17 camadas, 89,37% em CIFAR-10.
- ASGE 2509.12394: VGG8 90,50%.
- Trifecta 2311.18130: 12 camadas 83,51%; com layer norm, a feature perde utilidade rápido com a profundidade.
- HCL-FF 2605.24797: CIFAR-100 de 63,77 para 70,76 com hierarquia de classes.
- Distance-Forward 2408.14925: 84,75 por camada, 88,15 em blocos, BP 90,1.
- Mono-Forward 2501.09238: 4 conv, 84,97 contra BP 86,22.
- SaFF-Net 2406.14038: encoder congelado, não é início de backprop.
- Hinton 2212.13345, SymBa 2303.08418, Lorberbom 2305.12393, Tosato 2305.18353: sem conv de verdade.

### B. Hebbian profundo

**SoftHebb.** Journé, Garcia Rodriguez, Guo, Moraitis. ICLR 2023. https://arxiv.org/abs/2209.11883
- Ideia: cada filtro aprende na proporção de um softmax entre todos, sem gradiente, em uma passada.
- Resultado: CIFAR-10 80,31; STL-10 76,23; ImageNet 27,3. Na última camada, aleatório / WTA duro / SoftHebb: CIFAR-10 64,6 / 60,3 / 80,3. Ativação: ReLU 70,68, RePU e Triangle 80,31.
- Largura (leitura de figura, CIFAR-10, camadas 1/2/3): 1x vai de 71 a 71,5 a 66. 4x vai de 71,1 a 77,7 a 80,3. Trecho: "to increase performance with depth, the network needs to also grow in width".
- Competição: 1/τ ótimo de 1, 0,65 e 0,25 nas camadas 1, 2, 3. Cerca de 5000 atualizações bastam.
- Código: github.com/NeuromorphicComputing/SoftHebb
- Escala: 96 / 384 / 1536 em CIFAR; 5 camadas até 12288 a 160 px em ImageNet.
- Depois com backprop: STL-10 de 76,23 para 78,5 (+3,99 sobre do zero). Único teste, sem curva. Triangle "would be detrimental to BP".
- Regra: encher a 1×1 com SoftHebb em uma passada, sobre todas as posições, com os filtros Fisher parados na competição. Na 1×1, N = imagens × 576. Ver caminho 5.
- Limite: leitura achatada em 24.576 valores, a nossa é GeM. WTA duro perdeu para aleatório, então pesos aleatórios são o controle obrigatório.

**Lagani et al. 2021 (base, fora da janela).** Hebbian Semi-Supervised Learning in a Sample Efficiency Setting. Neural Networks. https://arxiv.org/abs/2103.09002
- Ideia: camadas internas por PCA Hebbiana sem rótulo, depois SGD.
- Resultado: tabela da seção 3. Com largura quase constante, a sonda para: L1 a L5 64,69 / 65,92 / 64,43 / 61,24 / 61,16.
- Código: github.com/GabrieleLagani/HebbianPCA/tree/hebbpca
- Escala: 96 a 256 canais, 32 px, 40 a 90 mil imagens, 20 épocas.
- Depois com backprop: não acelera o fine-tuning. Com 1% quase não muda os pesos, com 100% muda tudo. BatchNorm comum foi "disruptive" nas camadas altas.
- Regra: testar nas camadas 3+ um desvio único para o patch todo, no lugar do z-score por dimensão.

**FastHebb.** Lagani, Gennaro, Fassold, Amato. SISAP 2022. https://arxiv.org/abs/2207.03172
- Ideia: a regra Hebbiana reescrita como multiplicação de matrizes em lote, até 50 vezes mais rápida.
- Resultado: ImageNet de 467 h para 10 h 57. Top-5 com fine-tuning, 1%: HPCA 8,65, VAE 2,72. 100%: 77,28 e 76,84.
- Código: github.com/GabrieleLagani/HebbianLearning/tree/fasthebb
- Escala: 10 camadas, 96 a 512 canais, 210 px.
- Depois com backprop: sempre. Hebbian ganha do VAE abaixo de 4 a 5% de rótulos.
- Regra: fazer qualquer passada Hebbiana como matmul sobre patches, reusando o extrator do SPiFiL.

**Miconi 2021 (base).** https://arxiv.org/abs/2107.01729
- Ideia: explica por que a camada profunda Hebbian falha.
- Resultado: CIFAR-10 34,5 na rede padrão, 64,55 com Triangle e máscara de 1% das conexões. Sem treinar nada, a mesma arquitetura dá 56,3.
- Código: github.com/ThomasMiconi/HebbianCNNPyTorch
- Escala: 3 camadas, 100 / 196 / 400, 32 px.
- Depois com backprop: não testado.
- Trecho: filtros profundos viram "larger Gabor-like features and poorly localized blobs".
- Regra: olhar se os filtros SPiFiL da L3 são versões borradas da L1. Candidato que olha só parte dos canais também reduz D.

**Nimmo & Mondragón.** Neural Networks 2025. https://arxiv.org/abs/2501.17266
- Ideia: WTA duro funciona na rede larga, com inibição lateral e regra BCM.
- Resultado: CIFAR-10 75,2, igual ao BP da mesma rede. Mesmo WTA: estreita 59,7, larga 74,8. Sem competição 19,9. Depthwise separável: 5,9M vira 0,9M, 71,9 contra 74,8.
- Código: github.com/Julian-JN/Advancing-the-Biological-Plausibility-and-Efficacy-of-Hebbian-Convolutional-Neural-Networks
- Escala: 3 a 4 camadas, até 5,9M, 32 a 96 px.
- Depois com backprop: não testado.
- Regra: para caber em 5M, largura nas 1×1 e espacial em depthwise.

**SPHeRe.** Deng et al. NeurIPS 2025. https://arxiv.org/abs/2510.14810
- Ideia: cada camada preserva a matriz de similaridade entre imagens do lote, sem decoder.
- Resultado: Tiny ImageNet 40,33 contra SoftHebb 34,12. CIFAR-10 81,11.
- Código: github.com/brain-intelligence-lab/SPHeRe
- Escala: 3 camadas, 384 / 768 / 1536. Usa AdamW local, não é livre de gradiente.
- Depois com backprop: +0,47 só na primeira camada, sem barra de erro.
- Regra: testar essa perda no lugar da reconstrução, uma camada 3+ por vez.

**Ciampi et al.** Hebbian em microscopia. 2024. https://arxiv.org/abs/2412.03192
- Ideia: pré-treino Hebbian em U-Net, depois fine-tuning com poucos rótulos.
- Resultado: GlaS Dice com 1%: 69,95 contra VAE 68,77. Ganho de 1 a 2 pontos sobre o estado da arte. Temperatura ótima muda entre datasets, de 20 a 100.
- Código: github.com/ciampluca/hebbian-bootstraping-semi-supervised-medical-imaging
- Escala: U-Net, 128 px, de 165 a 11.897 imagens. Segmentação com rótulo de pixel.
- Depois com backprop: é o método. Melhora "in most cases".
- Regra: é a evidência mais perto do nosso tamanho de dataset. Buscar temperatura por dataset.

**Hua et al. Hebbian with Global Direction.** 2026. https://arxiv.org/abs/2601.21367
- Ideia: passo vem da regra Hebbian, sinal vem do gradiente global.
- Resultado: ResNet-50 ImageNet 73,14 contra 75,99. Trecho sobre SoftHebb: "Any modifications to its architecture can lead to a drastic performance drop".
- Código: github.com/huawjcn/GHL
- Depois com backprop: não testado como início.
- Regra: não adotar. Fica o aviso de que SoftHebb é frágil.

**Liu et al. Self-Defined Target.** 2024. https://arxiv.org/abs/2403.12116
- Ideia: o alvo é o neurônio vencedor, e uma homeostase obriga todos a vencer igual.
- Resultado: SVHN 81,53, CIFAR-10 71,5.
- Código: github.com/neurophysics-cnrsthales/unsupervised-target
- Escala: 2 conv, 32 px.
- Depois com backprop: o treino já é backprop.
- Regra: usar a homeostase como medida de canal morto na 1×1.

**Fora, em uma linha.** Lagani survey 2307.16236: Hebbian "not very effective for training intermediate network layers" (p.22). Bahroun 2308.02427: só LeNet em MNIST. Cekic 2202.13074: custo somado ao SGD, não é início.

### C. Treino local por camada ou bloco

Por que camada local perde informação. [FATO] Ela só pensa no próprio objetivo, e o que sai não volta. O sintoma é sempre o mesmo: a sonda sobe cedo e fica plana.

**Sakamoto & Sato.** TMLR 2024. https://arxiv.org/abs/2402.09050
- Ideia: treino ponta a ponta dá papéis diferentes às camadas, e o local dá o mesmo papel a todas.
- Resultado: ResNet-50 CIFAR-10: E2E 94,8, local 89,9, retreino das camadas de cima 90,8.
- Aviso: forçar preservação da entrada subiu o sinal de informação e piorou a acurácia. Só preservar não basta, e um autoencoder puro corre esse risco.
- Código: github.com/keitaroskmt/E2E-info
- Escala: LeNet5 a ResNet-50, 32 px.
- Depois com backprop: se só as de cima retreinam, quase não ajuda (+0,9).
- Regra: achar a camada em que a sonda fica plana. Se for a 2, o conserto é na 2.

**Siddiqui et al. Blockwise SSL.** TMLR 2024. https://arxiv.org/abs/2302.01647
- Ideia: 4 blocos com Barlow Twins local quase igualam o ponta a ponta.
- Resultado: ImageNet 70,48 contra 71,57. Simultâneo 68,84, sequencial 67,12, blocos anteriores aleatórios 51,79.
- Código: github.com/shoaibahmed/blockwise_ssl
- Escala: ResNet-50, 4 blocos, 300 épocas.
- Depois com backprop: não testado.
- Regra: manter lr nas camadas antigas, que é o mais perto de "simultâneo" que temos. Já fazemos.

**InfoPro (base, 2021).** Wang et al. ICLR 2021. https://arxiv.org/abs/2101.10832
- Ideia: perda local = reconstrução + classificação.
- Resultado: erro ResNet-32 por número de módulos: 7,37 / 10,30 / 16,07 / 21,19 / 24,59. Com K=8: sem nada 21,19, com decodificador 14,13, ambos 11,13.
- Código: github.com/blackfeather-wang/InfoPro-Pytorch
- Escala: até ResNet-152, K até 16.
- Depois com backprop: não testado.
- Regra: camadas baixas pesam reconstrução, altas pesam discriminação, que no nosso caso já vem do Fisher.

**AugLocal.** Ma et al. ICLR 2024. https://arxiv.org/abs/2402.17318
- Ideia: a cabeça auxiliar copia a forma das camadas seguintes.
- Resultado: ResNet-110 CIFAR-10: BP 94,61, InfoPro 86,95, AugLocal 93,96. Cabeça uniforme 90,98 contra 1×1 80,61.
- Código: github.com/ChenxiangMA/AugLocal
- Escala: até 55 camadas locais, ImageNet.
- Depois com backprop: não testado.
- Regra: o decoder da camada 3+ com a forma das camadas que virão.

**CLAPP++.** Wu, Delrocq, Gerstner, Bellec. ICML 2026. https://arxiv.org/abs/2601.21683
- Ideia: com pesos ortonormais, gradiente local = global. Em CNN, a perda precisa de dependência espacial.
- Resultado: CIFAR-10: BP 80,49, sem espaço 73,21, com espaço 80,51. ImageNet: 48,52 contra 44,16.
- Código: github.com/zihan-wu/local-SSL
- Escala: VGG 6 a 8 camadas, 96 a 224 px.
- Depois com backprop: não testado, mas empata com BP global em até 6 camadas.
- Regra: filtros mais ortogonais em 3+ aproximam o teorema. Testar leitura que concatena camadas.

**ContSup.** Yu et al. 2023. https://arxiv.org/abs/2312.07636
- Ideia: dar à camada profunda um caminho extra com a entrada original.
- Resultado: erro K=16: 16,21 para 10,32 só com a imagem. Trecho: "the deep modules are unable to recover the information lost in the shallow layers".
- Código: github.com/Tab-ct/ContSup
- Depois com backprop: não testado.
- Regra: juntar ao candidato da L3 um canal LAB reduzido. Muda o encoder, então não é barato.

**Interlocking Backprop.** Gomez et al. JMLR 2022. https://arxiv.org/abs/2010.04116
- Ideia: blocos em pares sobrepostos, e o teste direto de local seguido de E2E.
- Resultado: CIFAR-10 E2E 92,86, local e depois E2E 94,21. CIFAR-100 73,98 e 76,40. Melhor metade do tempo local, metade E2E. Taxa 0,1 ganhou de 0,01.
- Código: não indicado.
- Escala: ResNet-32, 32 px.
- Depois com backprop: é o tema. Com orçamento fixo, não até convergir.
- Regra: ao soltar, não começar com taxa muito baixa.

**Deep Incubation.** Ni et al. 2022. https://arxiv.org/abs/2212.04129
- Ideia: cada módulo treina dentro de um meta-modelo pequeno e fixo, depois tudo é montado e ajustado.
- Resultado: ViT-B com 100 / 50 / 25% dos dados: E2E 81,8 / 74,7 / 65,7, modular 82,4 / 78,6 / 72,9. Uma época de ajuste já supera o E2E.
- Código: github.com/LeapLabTHU/Deep-Incubation
- Escala: quase tudo ViT, até 632M.
- Depois com backprop: sempre. O ganho vem da fase modular, não do reinício.
- Regra: treinar a camada 3+ dentro de uma rede rasa já montada, para ela servir quem vem depois.

**Fora.** LocalProp 2609.32517: ResNet-50 CIFAR-10 local 89,79, BP 97,40. Ren et al. 2210.03310: cortar o gradiente custa mais que a perda local. MAN 2407.05623, MAN++ 2507.16279, Layer Grafted 2302.14138: ideia útil, não aplicável à seleção sem gradiente.

**O que não precisa de experimento novo.** [APOSTA] Os artigos de blocos (BiCovG, Gomez, LocalProp) mostram que deixar o gradiente andar entre camadas recupera quase tudo. Nós já fazemos isso com `layerwise_backprop`, e soltar tudo rendeu +0,010.

### D. Redes analíticas, forma fechada e kernels

Quase toda conta fechada é uma de três: ridge sobre todas as posições, matriz usada para sortear filtros, ou diferença entre dois pontos.

| Artigo | Conta da camada L (A = patches em todas as posições, y = rótulo da imagem) |
|---|---|
| Forward Projection | W = (AᵀA + λI)⁻¹ Aᵀ Z, com Z = sign(AQ) + sign(yU) |
| ACnnL | W = (AᵀA + γI)⁻¹ Aᵀ Z̄, com Z̄ = YQ |
| Deep ConvRFM | filtro = g M^½, com M = média de ∇f ∇fᵀ sobre patches (AGOP) |
| Rainbow | filtro ~ N(0, C); sem rótulo, C ≈ covariância da entrada |
| SWIM | w = (x₂ − x₁)/‖x₂ − x₁‖², b = ⟨w, x₁⟩ |
| Saab | filtro DC + PCA dos patches sem a média local |
| ReduNet | Z ← P(Z + η(EZ − Σ γⱼ Cʲ Z πⱼ)), E e Cʲ em forma fechada |

**SWIM (li eu).** Bolager, Burak, Datar, Sun, Dietrich. NeurIPS 2023. https://arxiv.org/abs/2306.16830
- Ideia: cada neurônio é a diferença entre dois pontos de dado, com pares sorteados onde a saída muda muito em pouca distância.
- Resultado: 72 tarefas OpenML, "on average, 30 times faster" e acurácia "comparable". Cabeça sobre ResNet50, VGG19 e Xception em CIFAR-10: na ResNet50, sampling ganha do Adam acima de 1024 de largura.
- Código: gitlab.com/felix.dietrich/swimnetworks
- Escala: 1 a 5 camadas de 500 neurônios. "Many architectures like convolutional or transformer networks cannot be sampled with our method yet" (p.10).
- Depois com backprop: testado só na cabeça. "close to the iterative approach", ou seja, a vantagem some depois do ajuste.
- Regra: N candidatos dão N² pares, então 1280 filtros deixam de depender de 1280 sementes. Na 1×1, w = (p₂ − p₁) normalizado na norma média SPiFiL, b = −w·p₁. Sorteie pares de classes diferentes só entre candidatos bem pontuados.
- Limite: sem máscara, ~97% dos candidatos são fundo (`p0/R1.md`), e "perto e de classe diferente" descreve dois pedaços de fundo. Sem a trava de rank, o peso do SWIM escolhe justamente os piores pares.

**Rainbow.** Guth, Ménard, Rochette, Mallat. JMLR 2024. https://arxiv.org/abs/2305.18512
- Ideia: filtros de rede treinada são sorteios, o que importa é a covariância.
- Resultado: CIFAR-10: Gaussian rainbow 85%, 89% com classificador retreinado, rede treinada 92%. Parte útil de cerca de 30 dimensões em largura 128.
- Código: github.com/FlorentinGuth/Rainbow
- Escala: 7 camadas em CIFAR-10, 10 em ImageNet. Espacial fixo por ondaletas, só 1×1 aprendida.
- Depois com backprop: mede o que o SGD faz: "training amounts to learning the weight covariance". O sorteio inicial segue reconhecível até o fim.
- Regra: canais de enchimento da 1×1 = g C^½, com C = covariância real da entrada dessa camada.
- Limite: 85% usa covariância de rede treinada. A versão sem rótulo não tem acurácia reportada. Inadequado em ImageNet (p.33).

**Universalidade em CNNs.** Guth & Ménard. ICLR 2024 Workshop. https://arxiv.org/abs/2409.19460
- Ideia: da 2ª camada em diante, formas espaciais são quase iguais em qualquer dataset, o que muda é a mistura de canais.
- Resultado: "such encodings do not need to be learned and could be originally preset". Sem tabela de acurácia no texto lido.
- Código: não indicado.
- Escala: VGG-11, 8 conv, 32 e 224 px.
- Depois com backprop: não se aplica.
- Regra: na L3, medir a nota no vetor de canais da semente, D = 48, não 1200. Ver caminho 3.

**Forward Projection.** O'Shea & Rajendran. Nature Communications 2026 (local vem da busca; li arXiv v4). https://arxiv.org/abs/2501.16476
- Ideia: cada camada é uma ridge para um alvo que mistura entrada e rótulo, em uma passada, sem gradiente nenhum.
- Resultado: AUC 84,5 em OCT com N = 10, contra 61,8 do BP com N = 5. CIFAR-10: FP 48,8, aleatório 40,6, BP 54,8.
- Código: github.com/robertoshea/forward_projection (espaço no PDF, conferir).
- Escala: CNN 3×3, 16·2^(l−1) filtros, 128 px, de 5 a 50 por classe.
- Depois com backprop: não testado.
- Regra: somar AᵀA em todas as posições. Com 741 imagens são 426 mil linhas, bem acima de D. Vale para os percentuais pequenos.
- Limite: alvo feito só de rótulo dá no máximo K direções (conferido pelo agente em numpy). Não enche 1280 canais.

**ACnnL.** Zhuang, Lin, Yang, Toh. 2022. https://arxiv.org/abs/2202.06504
- Ideia: a mesma ridge, com alvo aleatório tirado do rótulo.
- Resultado: CIFAR-10 0,7049 com C = 512. Sobe com a profundidade sem gradiente: 0,4953 / 0,5353 / 0,6046 / 0,6277.
- Código: não indicado.
- Escala: 4 conv, até 2048 canais, 32 px.
- Depois com backprop: não testado. Avisa que "this incorporation is not trivial due to numerical differences", porque a inversão dá pesos grandes.
- Regra: normalizar a norma do filtro antes do autoencoder.

**Deep ConvRFM.** Beaglehole, Radhakrishnan, Pandit, Belkin. 2023. https://arxiv.org/abs/2309.00570
- Ideia: a covariância dos filtros treinados segue a matriz de gradientes sobre patches, que dá para calcular sem backprop.
- Resultado: Pearson em geral acima de 0,9. CNTK / ConvRFM / Deep ConvRFM / CNN: SVHN 81,38 / 85,02 / 87,28 / 84,48. CIFAR-10 67,74 / 67,97 / 67,97 / 64,90.
- Código: github.com/aradha/convrfm
- Escala: 3 camadas, 64 filtros, imagens pequenas.
- Depois com backprop: não testado.
- Regra: usar M como métrica no lugar da Mahalanobis.
- Por que fica de fora dos 5: profundidade não ganhou nada em CIFAR-10, precisa de kernel caro, e a versão barata seria invenção minha.

**Deep RFM.** Beaglehole, Súkeník, Mondelli, Belkin. 2024. https://arxiv.org/abs/2402.13728
- Ideia: projetar com a raiz da AGOP junta as classes, e a camada aleatória só dá largura.
- Resultado: "the improvement in NC1 is entirely due to M^1/2", e a parte aleatória "actually worsens" esse número. Sem acurácia.
- Código: github.com/dmbeaglehole/neural_collapse_rfm
- Escala: até 19 camadas, sem convolução.
- Depois com backprop: não testado.
- Regra: medir separação antes e depois de cada etapa.

**Green Learning, DFT e LNT.** Kuo & Madni 2022, https://arxiv.org/abs/2210.00965. Yang et al. 2022, https://arxiv.org/abs/2203.11924. Wang et al. 2023, https://arxiv.org/abs/2312.14968.
- Ideia: PCA sem rótulo faz as camadas, e um teste de pureza em 1 dimensão escolhe o que presta.
- Resultado: DFT usa 25 a 35% das features e perde 0,62 a 0,94%. LNT CIFAR-10 76,28.
- Código: não indicado.
- Escala: 3 hops, até 1441 canais, 32 px, XGBoost no fim.
- Depois com backprop: não testado.
- Regra: aplicar o filtro, fazer pooling por imagem, medir pureza no melhor corte. N vira número de imagens e D vira 1. O cotovelo diz quantos filtros a camada merece. Ver caminho 3.

**ReduNet.** Chan, Yu, You, Qi, Wright, Ma. JMLR 2022. https://arxiv.org/abs/2105.10446
- Ideia: cada camada é um passo de subida no ganho de codificação, com matrizes calculadas dos dados.
- Resultado: MNIST: só construção 0,898, backprop do zero 0,932, construção mais backprop 0,978.
- Código: github.com/Ma-Lab-Berkeley/ReduNet
- Escala: 25 a 30 camadas, 100 a 500 imagens, 28 px.
- Depois com backprop: +4,6 pontos sobre início aleatório. Sem número de épocas.
- Regra: Selector guloso que maximiza o log det das respostas pooled dos filtros já escolhidos.

**Sparse Manifold Transform.** Chen, Yun, Ma, Olshausen, LeCun. ICLR 2023. https://arxiv.org/abs/2209.15261
- Ideia: dicionário de patches aleatórios mais uma projeção fechada que aproxima patches vizinhos.
- Resultado: KNN CIFAR-10 81,1. SimCLR 90,1 com augmentation completa.
- Código: não indicado.
- Escala: 2 camadas, 32 px, dicionário até 65.536. Hierarquia profunda segue em aberto.
- Depois com backprop: não testado.
- Regra: vizinhos = patches do mesmo superpixel.

**Lei de separação.** He & Su. PNAS 2023. https://arxiv.org/abs/2210.17020
- Ideia: em rede bem treinada, cada camada reduz a mistura das classes pelo mesmo fator.
- Resultado: D_l ≈ ρ^l D_0, ρ = 0,818 em rede de 20 camadas, Pearson de −0,997 a −1,000. "Somewhat fuzzier" em CNN.
- Código: github.com/HornHehhf/Equi-Separation
- Escala: 4 a 20 camadas, 32 px.
- Depois com backprop: é medição em rede treinada.
- Regra: medir log D por camada em GeM, nos ckpts que já existem. Mostra em número se a L3 é fraca. É o passo zero, sem treino.

**Fora, em uma linha.**
- Wu et al. 2202.01210: W = soma por classe, posto K. A v2 dá 404 e a busca diz que foi retirada.
- RFM 2212.13881: redes densas e tabular.
- Milsom 2309.09814 e 2410.06171: 92,7 e 94,52 em CIFAR-10, mas com Adam e cerca de 2000 h de GPU.
- Bietti 2102.10032: teoria.

### E. Filtros, dicionários e nota sem treino

**Thiry et al. (base, 2021, o par mais próximo do SPiFiL).** ICLR 2021. https://arxiv.org/abs/2101.07528
- Ideia: dicionário de patches sorteados e branqueados, mais classificador linear, chega perto de redes treinadas.
- Resultado: CIFAR-10 86,7 com 60k patches. Trocar patches por ruído branco: "this step leads to a drop of 8.1%". ImageNet 128 px: 35,9 top-1.
- Código: github.com/louity/patches
- Escala: 1 a 2 camadas, 2k a 60k átomos, 32 a 128 px.
- Depois com backprop: com 2048 átomos, treinar o dicionário dá "the same accuracy (- 0.1 %)". Sortear é tão eficiente quanto otimizar.
- Outros fatos: dicionário dobrado com d e −d. Distância em patch branqueado é Mahalanobis.
- Regra: encher a 1×1 com vetores do mapa sorteados, e não com ruído. Ver caminho 1.
- Limite: tudo em camada 1 sobre pixels. Ninguém testou em mapa de camada 3+.

**Gavrikov & Keuper, Random Convolutions.** 2023. https://arxiv.org/abs/2301.11360
- Ideia: filtros espaciais aleatórios congelados bastam se houver 1×1 aprendida depois.
- Resultado: CIFAR-10: congelado 91,89, base 91,60. ImageNet: 37,85 sem 1×1, 72,31 com, base 72,60.
- Código: prometido, sem link.
- Escala: ResNet-20, E até 1024, kernels 3 a 9.
- Depois com backprop: o artigo é todo backprop.
- Regra: o nosso 1×1 aleatório está do lado ruim. Lá o aleatório é o espacial e o 1×1 é aprendido. Aqui é o inverso.

**Cazenavette, Julin, Lucey. Rethinking the Role of Spatial Mixing.** 2025. https://arxiv.org/abs/2503.16760
- Ideia: separa conv em espacial e canal, e congela uma de cada vez.
- Resultado: "networks only learning spatial mixing (Space) consistently perform far worse than those learning only channel mixing". Valores só em figura. "Random filters work best when they are uncorrelated".
- Código: não indicado.
- Escala: ResNet separável e ConvMixer, CIFAR e ImageNet.
- Depois com backprop: não testa aleatório como início.
- Regra: tratar a 1×1 aleatória como o elo mais fraco. É a primeira coisa a trocar.

**Trockman, Willmott, Kolter.** 2022. https://arxiv.org/abs/2210.03651
- Ideia: amostrar filtros de uma Gaussiana com covariância estruturada, sem treino.
- Resultado: até +1,6% em ConvMixer. ImageNet 65,00 para 68,78. "our method works poorly for 3 × 3 filters".
- Código: trecho nas Figs. 15 e 16.
- Escala: só depthwise, kernels 3 a 15, até 32 camadas.
- Depois com backprop: testado. Em 200 épocas o ganho continua, chega a 90% com cerca de 50% menos épocas. Na transferência de covariância, a vantagem "quickly fades".
- Regra: se sobrar canal aleatório, sortear de N(0, Σ) dos filtros SPiFiL, não isotrópico.

**CNN Filter DB.** Gavrikov & Keuper. CVPR 2022. https://arxiv.org/abs/2203.15331
- Ideia: 1,4 bilhão de filtros 3×3 treinados, medindo diversidade e esparsidade.
- Resultado: entropia de 0,75 para 0,5 com a profundidade. Banco com entropia alta parece ruído.
- Código: github.com/paulgavrikov/cnn-filter-db
- Escala: centenas de modelos, só 3×3.
- Depois com backprop: não testado como início.
- Regra: termômetro barato nas camadas 3×3. Não se aplica à 1×1.

**FLIM com Bag of Feature Points.** Martinelli et al. 2026. https://arxiv.org/abs/2602.20845
- Ideia: um único clustering na entrada escolhe os pontos, e cada bloco usa os mesmos pontos mapeados.
- Resultado: S. mansoni 0,860 ± 0,091 contra 0,844 ± 0,017, com faixas sobrepostas. Cluster melhor com até 2 blocos, BoFP com 2 ou 4.
- Código: "released after review".
- Escala: até 4 blocos, 3×3, 2 a 3 imagens de treino.
- Depois com backprop: não testado.
- Regra: é o que o SPiFiL já faz com sementes reprojetadas. Não resolve colisão na grade 24×24.

**MS-FLIM.** Cerqueira et al. 2024. https://arxiv.org/abs/2403.12748
- Ideia: usuário escolhe filtros da camada 1 entre várias rodadas de k-means.
- Resultado: Dice ET/NC/WT: backprop 0,665 / 0,734 / 0,721; MS-FLIM congelado 0,746 / 0,813 / 0,785; com fine-tuning 0,747 / 0,805 / 0,780. "the fine-tuning of the encoder by backpropagation was unnecessary".
- Código: não indicado.
- Escala: 3 camadas, 6 imagens para o encoder.
- Depois com backprop: encoder bom não ganhou nada.
- Regra: acertar as primeiras camadas vale mais que refinar as de cima.
- Sobre FLIM acima de 2 camadas: não existe nota automática. A seleção é humana (2306.14840) ou nenhuma (2504.20872). Não achei FLIM sem marcador.

**CoSeP.** Levin, Singer. arXiv v3 2026. https://arxiv.org/abs/2505.13225
- Ideia: cada canal vira um perfil de distâncias entre pares de classes, calculadas na saída com média espacial.
- Resultado: ResNet-56 CIFAR-10 com 2,28× menos FLOPs: aleatório 91,75 ± 0,31, CoSeP 93,82 ± 0,09.
- Código: não indicado.
- Escala: ResNet, VGG, MobileNet, CIFAR e ImageNet.
- Depois com backprop: obrigatório, 2 épocas após cada camada.
- Regra: é o melhor modelo para a nota de filtro profundo. Ver caminho 3.
- Limite: filtros de rede já treinada. Com 2 classes o perfil tem 1 número só.

**MeCo.** Jiang, Wang, Bie. NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/bfa815ac6f08f4ada34fe22be054f2b9-Abstract.html
- Ideia: nota da rede é o menor autovalor da correlação entre canais, sem rótulo e sem gradiente.
- Resultado: Spearman 0,894 em NATS-TSS, mas −0,79 em NATS-SSS, onde muda a largura: "highly sensitive to the number of channels".
- Código: github.com/HamsterMimi/MeCo
- Escala: benchmarks NAS.
- Depois com backprop: não usa.
- Regra: só comparar bancos de mesma largura.

**NEAR.** Husistein, Reiher, Eckhoff. ICLR 2025. https://arxiv.org/abs/2408.08776
- Ideia: soma do posto efetivo antes e depois da ativação, e dá para estimar a largura boa.
- Resultado: largura prevista 52, 58, 60 contra 102, 61, 44 usadas.
- Código: github.com/ReiherGroup/NEAR
- Escala: largura só em MLP.
- Depois com backprop: não usa.
- Regra: varrer a última camada de 128 a 1280 e ver onde o posto satura. Já está na família E1.

**Fora, em uma linha.** SDNet 2210.12945 e LCANets (PMLR v162): dicionário com backprop ou só primeira camada. BagSSL 2206.08954: precisa de SSL. ZiCo 2301.11300: pede gradiente e rótulo. Rare Gems 2202.12002: acha a máscara com gradiente, mas deu o protocolo de sanidade: rodar top, pior e sorteio. Luther & Seung 2206.02716: só MNIST.

### F. Inicialização dependente de dados e crescimento

A literatura é fraca no ponto exato. O agente não achou artigo de 2023 a 2026 que monte camadas profundas de CNN a partir de dados sem gradiente. O mais próximo, Pre-Warm, só mexe na primeira camada.

**Três regras que saem daqui.** [APOSTA]
- **Escala:** calibre cada filtro pela resposta dele nos dados, sem copiar a norma média.
- **Ortogonalidade:** na 1×1 é impossível, porque são 1280 filtros em 64 dimensões. A meta é espalhar por igual. Na L3 cabe (D = 1200, 64 filtros).
- **Filtro novo:** nota por quanto do resíduo ele explica, em todos os pixels. Escolher o pixel de maior resíduo piorou.

**Sonda do agente deste tema.** [SONDA] CPU, treino apenas, sem backprop, `disc_bp01w64c1280_bg1`, split 1, 50%, 240 imagens por dataset, GeM + LinearSVC próprio, não o `fit_svm`. Logs em `<scratchpad>/law/newpapers/probe/`.

| Enchimento (1280 filtros) | κ eggs | κ protozoan | κ larvae | Filtros novos quase parados |
|---|---|---|---|---|
| Só SPiFiL | 0,391 | 0,613 | 0,973 | n/a |
| Atual (aleatório, bias 0) | 0,612 | 0,622 | 0,973 | 65 / 51 / 69% |
| Mesmo W, centrado e desvio 1 | 0,648 | 0,603 | 0,973 | 0% |
| Aleatório branco, centrado | 0,631-0,650 | 0,581-0,592 | 0,959 | 0% |
| Mais patches, sem seleção | 0,533-0,538 | 0,619-0,634 | 0,986 | 0% |
| Pixel de maior resíduo | 0,585-0,595 | 0,566-0,571 | 0,959-0,973 | 73 / 68 / 70% |

- **Firme nos 3 datasets:** centrar acaba com os filtros parados.
- **Não confirmado:** κ. Eggs sobe perto de 0,03, protozoan cai 0,02 a 0,04, larvae está no teto. O sorteio sozinho já move 0,02, e a leitura usa 120 imagens.
- **Surpresa:** em eggs, os 880 aleatórios sozinhos deram 0,599 contra 0,391 do SPiFiL sozinho. Em protozoan empatam (0,600 e 0,613).
- **Não testado:** o efeito depois do autoencoder.
- **Inferência dele:** na L4 parece não haver seleção. Com 50 sementes por classe e verba de 160, todo candidato vira filtro.
- **Cuidado:** esta sonda e a [CLÁSSICOS] dão κ em níveis muito diferentes (0,6 contra 0,9 em eggs), com leitura e amostra diferentes. Nenhuma decide sozinha.

**Krähenbühl et al. (base, 2016, a busca não achou sucessor).** ICLR 2016. https://arxiv.org/abs/1511.06856
- Ideia: normalizar cada canal com dados, depois equilibrar as camadas.
- Resultado: VOC2007 com SGD: k-means sem escala 41,2%, com escala 56,6%. Gaussiano 50,8 para 53,3. "160 images are sufficient to robustly estimate activation statistics".
- Código: github.com/philkr/magic_init
- Escala: CaffeNet, também VGG e GoogLeNet, 5011 imagens.
- Depois com backprop: é o objetivo, 80 mil iterações.
- Regra: filtro vindo de dado sem escala perdeu para gaussiano. Medir média `m` e desvio `s` por filtro em todos os pixels. Aleatório: `W/=s`, `b=-m/s`. SPiFiL: só dividir por `s`, sem recentrar.

**TINY.** Verbockhaven, Rudkiewicz, Chevallier, Charpiat. TMLR 2024. https://arxiv.org/abs/2405.19816
- Ideia: neurônio novo em forma fechada, por SVD, cruzando a entrada embranquecida com o que falta.
- Resultado: CIFAR-100: TINY 68,1 ± 0,5, GradMax 56,8 ± 0,2, referência 72,8. Só acrescentando neurônios passa de 17%, aleatório fica perto de 1%.
- Código: gitlab.inria.fr/mverbock/tinypub
- Escala: ResNet18 partindo de 1/64 da largura, 32 px.
- Depois com backprop: sim. Após 250 épocas extras, 68,7. O efeito fica, mas treino longo dilui a direção inicial.
- Regra: na L3, trocar o gradiente pelo resíduo de reconstrução e pegar os maiores valores singulares. A troca é minha, sem teste. Limite: no máximo 64 filtros na L4.

**NORTH.** Maile, Rachelson, Luga, Wilson. AutoML Conf 2022. https://arxiv.org/abs/2202.08539
- Ideia: acrescentar neurônio quando a dimensão efetiva pede, escolhendo candidatos que sobem a ortogonalidade pós-ativação.
- Resultado: "NORTH-Select with γa = 0.9 outperforms even Big Static with less than half as many parameters". Só em figura.
- Código: github.com/neurogenesisauthors/Neurogenesis
- Escala: VGG-11 e WRN-28, 32 px.
- Depois com backprop: é crescimento durante o treino.
- Regra: sorteie candidatos e guarde os que sobem a dimensão efetiva pós-ReLU. Pare quando estagnar.

**GradMax.** Evci et al. ICLR 2022. https://arxiv.org/abs/2201.05125
- Ideia: pesos do neurônio novo por SVD, para maximizar o gradiente sem mudar a saída.
- Resultado: MobileNet-V1 ImageNet: Random 66,9, GradMax 68,6. Saída nula: Random de 90,6 a 92,1.
- Código: github.com/google-research/growneuron
- Escala: WRN-28-1, VGG-11, MobileNet-V1.
- Depois com backprop: "improvements in the training dynamics which last for over 500 training steps".
- Regra: se crescer durante o autoencoder, filtro de entrada por dados e saída nula.

**Variance Transfer.** Yuan, Savarese, Maire. 2023. https://arxiv.org/abs/2306.12700
- Ideia: reescalar pesos ao crescer, e lr por bloco pela razão de normas.
- Resultado: ResNet-20 CIFAR-10: Net2Net 91,60, completo 92,53. A lr rendeu mais que a escala (+0,64 contra +0,40).
- Código: não indicado.
- Escala: ResNet, VGG, MobileNet, CIFAR e ImageNet.
- Depois com backprop: sim, mesmo total de épocas.
- Regra: `lr_bloco = lr · ||W_bloco|| / ||W_base||` em vez de 0,1 fixo.

**ZerO.** Zhao, Schäfer, Anandkumar. TMLR 2022. https://arxiv.org/abs/2110.12661
- Ideia: início sem sorteio, com Hadamard na camada que aumenta dimensão.
- Resultado: CIFAR-10 ResNet-18: erro 5,13 ± 0,08 contra 5,15 ± 0,13. O ganho real é menos variância entre sementes.
- Código: github.com/jiaweizzhao/ZerO-initialization
- Escala: ResNet até cerca de 500 camadas.
- Depois com backprop: feito para isso.
- Regra: na 64 para 1280, linhas de Hadamard na base branca. Não testado.

**IDInit.** Pan et al. Preprint 2025. https://arxiv.org/abs/2503.04626
- Ideia: identidade repetida em bloco quando a camada não é quadrada.
- Resultado: ResNet-20 CIFAR-10: 87,01 sem as melhorias, 93,22 com elas.
- Código: não indicado.
- Escala: ResNet-20 a 110, ImageNet, BERT.
- Depois com backprop: sim. A força vem do atalho residual, que não temos.
- Regra: não priorizar.

**SENN.** Mitchell, Menzenbach, Kersting, Mundt. 2023. https://arxiv.org/abs/2307.04526
- Ideia: acrescentar neurônio quando o ganho no escore natural passa de um limiar.
- Resultado: CIFAR-10 acima de 93%.
- Código: github.com/ml-research/self-expanding-neural-networks
- Escala: MLP e DenseNet de 3 blocos, 32 px.
- Depois com backprop: cresce durante o treino.
- Regra: nota = correlação quadrada da ativação do candidato com o resíduo de reconstrução. A troca de gradiente por resíduo é minha.

**Menor profundidade.**
- LSUV 1511.06422: uma escala por camada, "virtually identical to the orthonormal initialization". Um número por camada não conserta filtro parado.
- AutoInit 2109.08958: a média entra na escala, ResNet-50 de 74,33 para 75,35.
- TAT 2203.08120: Orthogonal Delta "does not have any clear advantage" (74,6 contra 74,8). Com 4 camadas, ortogonalidade só vale por diversidade.
- Sylvester 2105.10335: camada a camada sem gradiente, 50-shot 43,86 contra 40,84, 10-shot 26,84 contra 27,91.

**Fora.** GradInit 2102.08098 e NIO 2210.05956: precisam de gradiente. Weight Selection 2311.18823: precisa de modelo grande pré-treinado. GHN-3 2303.04143: precisa de hiper-rede treinada no ImageNet.

### G. Quanto dura, medido

**Pre-Warm.** Martnishn. 2026, um autor. https://arxiv.org/abs/2606.25256
- Ideia: escrever PCA ou k-means de patches do primeiro lote na metade dos filtros de conv1.
- Resultado: PCA +0,98 ponto sobre He em CIFAR-10, p = 0,0003, 8 de 8 sementes. "Random patches never improve the color datasets".
- Código: não indicado.
- Escala: 2 conv, 32 px, 5000 passos.
- Depois com backprop: em ResNet-20, ficou abaixo de He. O autor culpa o BatchNorm, que recentra e reescala o filtro.
- Regra: criar o controle "patch ao acaso, sem Fisher, mesma escala" na L3.

**DASH.** Shin, Oh, Cho, Yun. NeurIPS 2024. https://arxiv.org/abs/2410.23495
- Ideia: warm start piora porque a rede decora ruído, então encolhe cada peso conforme o alinhamento com o gradiente.
- Resultado: Tiny-ImageNet sem augmentation: aleatório 25,69, warm 9,57, DASH 46,11. Com augmentation, CIFAR-100: 75,98, 74,10, 74,84.
- Código: github.com/baekrok/DASH
- Escala: ResNet-18, VGG-16.
- Depois com backprop: é o tema. Dano só depois de um limiar de pré-treino, e ele fica.
- Regra: limitar épocas por estágio em vez de treinar até convergir.

**Matsoukas et al.** CVPR 2022. https://arxiv.org/abs/2203.01825
- Ideia: separar conteúdo dos pesos (WT) de estatística por camada (ST).
- Resultado: RI / ST / WT: APTOS (3.662) 0,845 / 0,848 / 0,888. CheXpert (224.316) 0,796 / 0,794 / 0,800.
- Código: github.com/ChrisMats/feature-reuse
- Escala: 22M a 29M parâmetros, 224 px.
- Depois com backprop: o ganho sobrevive em N pequeno, e em N grande vira quase zero.
- Regra: no fim do treino, voltar uma camada ao peso inicial e ver quanto o κ cai.

**Kumar et al. LP-FT.** ICLR 2022. https://arxiv.org/abs/2202.10054
- Ideia: com cabeça aleatória, as primeiras atualizações deformam as features. Treinar antes só a cabeça evita isso.
- Resultado: ID: FT 85,1, LP-FT 85,7. OOD: 59,3 e 68,9. LP-FT muda as features 10 a 100 vezes menos. Exceção: FMoW (satélite) FT 56,5 contra LP-FT 51,8.
- Código: não indicado no PDF.
- Escala: ResNet-50 e ViT.
- Depois com backprop: a proteção dura o treino todo, mas é ganho fora da distribuição.
- Regra: nosso decoder novo é a "cabeça aleatória". Ver caminho 4.

**Davila et al.** 2024. https://arxiv.org/abs/2406.10050
- Ideia: 8 estratégias, 8 datasets médicos, 3 CNNs, mesmo protocolo.
- Resultado: casos com melhora, em 24: LP-FT 58,3%, Last→First 45,8%, L2-SP 29,2%, First→Last 20,8%.
- Código: não indicado.
- Escala: 224 px, 587 a 64.540 imagens.
- Regra: "só o fim, depois tudo". Sem descongelar aos poucos de baixo para cima.

**Zhou et al. Fortuitous Forgetting.** ICLR 2022. https://arxiv.org/abs/2202.00155
- Ideia: reinicializar as camadas finais várias vezes força as iniciais a guardar o que é útil.
- Resultado: Flower: 66,89 para 70,87. Congelar as iniciais mata o ganho.
- Código: github.com/hlml/fortuitous_forgetting
- Escala: ResNet18, datasets pequenos.
- Regra: sortear de novo a parte aleatória da 1×1 entre rodadas. Custa N vezes mais.

**Kunin et al. Get rich quick.** NeurIPS 2024. https://arxiv.org/abs/2406.06158
- Ideia: escala total e desequilíbrio entre camadas decidem quanto a rede muda.
- Resultado: "rapid rich learning happens when both τ is small and δ is large". Em CNN real, "all networks achieve comparable training and test accuracy".
- Código: github.com/allanraventos/getrichquick
- Escala: 2 camadas na teoria, LeNet-5 e ResNet pequena.
- Regra: botão barato, mas no CNN real a escala muda a aparência do filtro, não a acurácia. Fica de fora.

**Base, leitura rápida.** Ash & Adams 1910.08475: CIFAR-10 56,2 contra 51,7, logística convexa não sofre. Frankle 2002.10365: a parte decisiva está nas primeiras 500 iterações. Raghu 1902.07208: guardar só média e variância por camada acelera, e modelo pequeno quase não ganha (93,6 para 93,9), o que importa para o nosso teto de 5M. Ciga 2109.01721: a única medida em épocas.

**Fora.** Lee Surgical 2210.11466: confirma que descongelar aos poucos não paga. Kleinman 2308.12221: rede linear. Domine 2409.14623: rede linear de 2 camadas. Lyle 2303.01486 e Dohare 2306.13812: alvo que muda sem parar, o nosso é fixo.

## 5. Não consegui ler

| Item | Motivo |
|---|---|
| Draelos 2016/2017, King & Mortazavi 2021 | Crescimento por erro de reconstrução, fora da janela, só resumo de busca. Valem leitura se a nota por resíduo for adiante |
| VFF-Net (Neural Networks 2025) | Sem texto aberto, vi só resumo |
| Deep-CBN (Sci. Reports 2025) | Só trecho de busca, fora do tema |
| FastHebb, versão Neurocomputing 2024 | Repositório devolveu 403 |
| Lagani et al., Zenodo 6367175 e PDF do ISTI | Não abri, ou veio binário |
| TVSPrune (ICLR 2023) | OpenReview devolveu 403. Parece par direto do CoSeP |
| Tese da TUM sobre SWIM em convolução (mediatum.ub.tum.de/node?id=1785711) | Página bloqueada para robô. É o único SWIM em convolução que achei |
| Nakayama BMVC 2013 e FPCANet 2019 | Fora da janela. Parentes do FisherScorer |
| Delrocq et al. 2024 | Não achei o ID |
| NASWOT original | Fora da janela, fiquei nos sucessores |
| LocalProp 2609.32517, Layer Grafted, FBA, SLL, HPFF, NM-Hebb | Leitura parcial ou só resumo |
| Cazenavette | Números só em figura |
| Fukase 2506.15954, 2502.12379, 2306.09320, 2308.06534 | Só resumo ou busca |

Convolutional kernel networks: a busca de 2024 a 2026 não trouxe nada que monte camadas por conta. Os kernels novos (Milsom) treinam com Adam.

## 6. Os 5 caminhos

Ordem por valor esperado sobre custo, e essa ordem é [APOSTA].

**Passo zero, sem treino.** Medir log D (He & Su) por camada nos ckpts de `disc_bp01w64c1280_bg1`. Mostra em número onde a camada para de separar.

**Regra comum.** Base = `disc_md_example.yaml`, recortada para 200 px, igual à triagem do Estágio 2 de D1: 4 camadas, 60 épocas, 3 splits, uma mudança por família. Datasets `Larvae_subgroup, Eggs_subgroup, Protozoan_subgroup, helminth_eggs`, para ter 2, 9, 7 e 9 classes.
- Decide: κ de validação com `gem_tta_pw`, média de 3 splits, Δ abaixo de 0,02 é empate. Olhar dataset por dataset, porque o efeito depende do tamanho.
- Custo: D1 conta ~3 h de slot por família com 4 datasets. O meu corte é só 200 px, então deve sair por menos. Não medi.
- A base (`disc_md_n0_base.yaml`) roda 1 vez, e todos comparam com ela.

YAMLs prontos em `<scratchpad>/law/newpapers/yaml/`, fora do repositório. Na hora de rodar, copie para `experiments/autoencoder/` do worktree. SVM depois do treino: `disc_md_eval_n_s200_val_gemttapw.yaml` (`family: disc_md_n*`, `stages: [L3, L4]`).

Conferi todos com `Experiment.from_dict` (`experiments/ray/schema.py`). Passam hoje: `n0_base`, `n2_ipc4`, `n2_x4`, `n2_k3last`, `n4_rand` e o eval. Os outros são recusados só pela chave nova, nomeada na mensagem. `n4_dec5` passa no schema, mas o CLI recusa até o parâmetro existir.

Lançar, por exemplo:
```
cd /dados/home/moliveira/Scalable_Hybrid_FLIM_discovery; export LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib:$LD_LIBRARY_PATH RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0; /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m experiments.ray.launch experiments/autoencoder/disc_md_n0_base.yaml
```

Base, igual em todos (`disc_md_n0_base.yaml`):
```yaml
name:   spifil_growth_disc_md_n0_base
method: autoencoder
runner: growth
grid:
  init:        [flim]
  datasets:    [Larvae_subgroup, Eggs_subgroup, Protozoan_subgroup, helminth_eggs]
  splits:      [1, 2, 3]
  percentages: [50]
resources: {gpu_ids: [1], max_per_gpu: 1, cpus_per_experiment: 10}
runner_args:
  layerwise: 4
  impurities: true
  no_crop: true
  image_size: 200
  layer_channels: [32, 48, 64, 1280]
  layer_kernels:  [5, 5, 5, 1]
  layer_pools:    [2, 2, 2, 1]
  fill_random_last: true
  superpixels: disf
  n_superpixels: 50
  budget_check: false
  embed_mode: avgpool2d
  layerwise_backprop: 0.1
output: {work_dir: /exps/moliveira/spifil_growth/disc_md_n0_base}
wandb: {enabled: true, project: phd_thesis_layerwise}
skip: state
evaluate: {SVM: false}
overrides:
  model.init_args.max_epochs: 60
  trainer.max_epochs: 60
  model.init_args.warmup_epochs: 2
  model.init_args.recon_bg_weight: 1.0
  data.init_args.num_workers: 8
  model.init_args.epoch_probe: gem
  model.init_args.best_by_probe: true
```

### Caminho 1: encher a 1×1 com dado real, não com ruído

**A ideia.** Hoje 880 a 1180 filtros da 1×1 são ruído com bias zero. Troque por filtros que nascem do próprio mapa da L3, que o SPiFiL já extraiu.

**Por que este vem primeiro.**
- [NOSSO] É a camada onde o início fica: cosseno 0,97 depois do treino.
- [FATO] Trocar patch por ruído custa 8,1 pontos (Thiry). Aprender só a mistura de canais vence aprender só o espacial (Cazenavette).
- [CLÁSSICOS] Em eggs, 1/3 dos filtros aleatórios fica sempre desligado e 1/3 sempre ligado (0,347 e 0,328). Só centrar o bias zera os dois e leva `k_pw` de 0,8787 a 0,9316, com `k_std` igual (0,9186 e 0,9191). Outra semente do mesmo ruído deu `k_pw` 0,9075, então a diferença real é menor que parece. Fonte: `<scratchpad>/law/classic_tools/bank_eggs.log`.
- [APOSTA] Isso também fecha a pergunta aberta de D1 ("SPiFiL inicializa toda camada"), porque `patch` e `pair` nascem de patches do SPiFiL.

**Chave nova:** `fill_mode: noise | center | calib | patch | pair | cov`, padrão `noise`, igual a hoje.
- Onde entra: `RUNNER_ARGS` (`experiments/ray/schema.py:96`), `GROW_KNOBS` (`experiments/ray/runners/growth.py:151`) e o bloco `fill_random` (`flim/spifil.py:971`).
- De onde vêm os vetores: `learn.state[0].features`. Nada no SPiFiL muda.

| Braço | Direção do filtro | Bias | Base no artigo |
|---|---|---|---|
| `disc_md_n1_fill_center` | ruído | −w·μ | [CLÁSSICOS], [SONDA] |
| `disc_md_n1_fill_calib` | ruído, `W/=s` por filtro | −m/s por filtro | Krähenbühl |
| `disc_md_n1_fill_patch` | vetor do mapa sorteado, z-score do SPiFiL | −w·μ | Thiry |
| `disc_md_n1_fill_pair` | p₂ − p₁, classes diferentes, só candidatos de bom rank | −w·p₁ | SWIM |
| `disc_md_n1_fill_cov` | g C^½, C sobre todas as posições | −w·μ | Rainbow, Trockman |

Todos na norma média dos filtros SPiFiL, como o `fill_random` já faz. YAML: a base mais `fill_mode: <braço>`.

**Lê-se assim.** Se `center` empatar com `patch`, fique com `center`, que é uma linha. Se nenhum passar da base, a parte aleatória não é o gargalo e o problema está na L3.

**Risco.**
- [SONDA] O κ não confirmou: centrar subiu eggs e baixou protozoan, dentro do ruído. Mais patches pioraram eggs (0,53 contra 0,61). Em eggs os 880 aleatórios sozinhos bateram o SPiFiL sozinho. Só o treino real decide.
- [FATO] Filtro de dado sem escala perdeu para gaussiano (Krähenbühl: 41,2 contra 50,8). Por isso todos os braços calibram.
- [APOSTA] Sem máscara quase todo candidato é fundo, então `pair` pode gerar filtros de fundo. E a leitura com `pw` já branqueia, o que pode engolir o ganho.

### Caminho 2: largura que paga o pooling, e última camada com contexto

**A ideia.** Só YAML, nenhuma chave nova.

**Por quê.**
- [FATO] Com a mesma regra, largura constante faz a camada 3 piorar e largura 4× faz subir (SoftHebb). O mesmo aparece em Nimmo (59,7 contra 74,8) e em Lagani.
- [FATO] DeeperForward vê queda depois de cada 1×1, e no fim da rede 3×3 ganha de FC por 6 pontos.
- [NOSSO] A escada de largura deu plano, mas parou em 64/96/128, fator ×1,5 (`discovery_insights_board.md:1762-1767`). w192 e w256 foram planejadas, e não achei κ delas. O fator ×4 nunca rodou.
- [APOSTA, conta minha] 24/96/384 mantém os valores por imagem em 235 mil, 230 mil e 221 mil, contra 313 mil, 115 mil e 37 mil hoje.

| YAML | Mudança sobre a base | Parâmetros | Pergunta |
|---|---|---|---|
| `disc_md_n2_ipc4` | `images_per_class: 4` | 200.944 | Mais candidatos sobem a fração SPiFiL da 1×1? |
| `disc_md_n2_x4` | `images_per_class: 4` + `layer_channels: [24, 96, 384, 1280]` | 1.474.304 | O fator de largura importa? |
| `disc_md_n2_k3last` | `images_per_class: 4` + `layer_kernels: [5, 5, 5, 3]` | 856.304 | Contexto na última camada ajuda? |

Compare `n2_x4` e `n2_k3last` com `n2_ipc4`, não com a base. Largura maior precisa de mais sementes, e `images_per_class: 4` dá 400 candidatos até em 2 classes.

**Risco.** [NOSSO] Largura já deu plano em eggs. O ganho pode não vir, e isso também fecha a pergunta. O ganho do artigo veio com leitura achatada, e a nossa é GeM.

### Caminho 3: nota que não depende de D

**A ideia.** Hoje a L3 mede com uma régua de 1200 números e só 100 a 400 exemplos. Olhe de outro jeito: resumindo o patch, ou olhando só a resposta do filtro por imagem. A resposta é um número, então o problema de N contra D some.

**Por quê.**
- [FATO] Nota na saída com média espacial escolhe melhor que sorteio em poda (CoSeP: 93,82 contra 91,75). DFT faz o mesmo em 1 dimensão.
- [FATO] Da 2ª camada em diante o que muda entre datasets é a mistura de canais, não a forma espacial (Guth & Ménard).
- [FATO] Nota no início às vezes não mede nada, por isso vem o teste de sanidade (Rare Gems, Pre-Warm).

**Chaves novas:**
- `layer_grow: {<camada>: {<knob>: <valor>}}`. Knob por camada, uma linha em `experiments/ray/runners/growth.py:411`, porque `_grow_once` já faz `{...} | layer` (`:300`). Serve para qualquer knob.
- `score_mode: fisher | random | patch_mean | response`, padrão `fisher`, em `GrowOneLayer.run` (`flim/spifil.py:356-359`). Importa `per_class_ranks` do SPiFiL. Muda só `state.scores` e `state.ranks`, e o filtro segue sendo o patch inteiro.

| YAML | `layer_grow` | O que faz | Custo de código |
|---|---|---|---|
| `disc_md_n3_score_rand` | `{3: {score_mode: random}}` | Sanidade: rank sorteado | 2 linhas |
| `disc_md_n3_score_mean` | `{3: {score_mode: patch_mean}}` | Média espacial do patch: D de 1200 para 48, N/D de 0,33 para 8,3 em eggs | 3 linhas |
| `disc_md_n3_score_resp` | `{3: {score_mode: response}}` | ReLU e GeM por imagem, Fisher em 1 dimensão, 30 imagens de treino por classe via `train_samples` e `encoder_features` | ~25 linhas |

Só a L3: [SONDA] na L4 da base não há seleção, porque todo candidato já vira filtro (50 sementes por classe, verba de 160). Lá a nota só passa a valer com `images_per_class` maior.

Variante para depois: `score_mode: residual`, nota pelo resíduo de reconstrução em todos os pixels (SENN, TINY), sem teste e com a troca sendo minha. [SONDA] Escolher o pixel de maior resíduo piorou.

**Lê-se assim.** Rode `n3_score_rand` antes. Se empatar com a base, o Fisher não paga na L3+. Aí a escolha vira só diversidade, e `response` nem precisa ser escrita.

**Risco.** [FATO] CoSeP foi medido em filtros de rede treinada. [APOSTA] Com 2 classes há pouco perfil para separar.

### Caminho 4: backprop depois, decoder primeiro, e medir quanto dura

**A ideia.** Cada camada nova começa com decoder aleatório. Nas primeiras épocas esse decoder manda gradiente ruim para filtros que estavam bons. Deixe o decoder aprender 5 épocas, com o encoder parado, e depois solte tudo.

**Por quê.**
- [FATO] LP-FT muda as features 10 a 100 vezes menos que soltar tudo de saída (Kumar). Foi o melhor em 58,3% dos casos médicos (Davila).
- [NOSSO] O backprop apaga o contraste da L3: de 2,86 para 0,99.
- [NOSSO] Segurar por muito tempo piora (âncora −0,069). Por isso são só 5 de 60 épocas, sem âncora.

**Chave nova:** `model.init_args.decoder_first_epochs: 5`, padrão 0.
- Onde entra: `AutoEncoderFlimModule.__init__` (`methods/autoencoder/autoencoder_flim_module.py:383`) e um `on_train_epoch_start` que liga `requires_grad` dos convs depois da época N.
- Atenção: o schema não olha `overrides`. O YAML passa na validação, mas o CLI recusa até o parâmetro existir no módulo (hoje não existe, conferi).

| YAML | Mudança | Pergunta |
|---|---|---|
| `disc_md_n4_dec5` | `decoder_first_epochs: 5` | Proteger no começo melhora o κ final? |
| `disc_md_n4_rand` | `random_layer_classic: true` (já existe) | Por quantas épocas o SPiFiL fica à frente do aleatório? |

**A medida que falta na literatura.** `n0_base` contra `n4_rand`, `val_kappa` por época em `epoch_metrics.csv`, por camada e por dataset. Curva da sonda contra curva da sonda, porque J1 avisa que a sonda não segue o SVM na família aleatória. Nenhum artigo tem essa curva.

**Risco.** [FATO] O ganho de LP-FT é fora da distribuição (+0,6 dentro), e em satélite perdeu (FMoW). Rode também em remote_sensing antes de concluir.

### Caminho 5: competição suave no lugar do ruído

**A ideia.** Em vez de sortear, deixe os filtros livres da 1×1 disputarem os vetores do mapa em uma passada sem gradiente, com os filtros Fisher parados na disputa. Cada filtro livre é empurrado para o que o Fisher não cobre.

**Por quê.**
- [FATO] Na última camada em CIFAR-10, sem gradiente: aleatório 64,6, SoftHebb 80,3.
- [FATO] Aprende em todas as posições. Aqui daria N = imagens × 576, não sementes (conta minha). Com D = 64 sobra amostra.
- [FATO] Ciampi mostra Hebbian ajudando em microscopia com datasets do nosso tamanho.

**Chave nova:** `fill_mode: softhebb`, a mesma chave do caminho 1. A regra em matmul, 1/τ = 0,25, 5000 passos.

**YAML:** `disc_md_n5_hebb`, a base mais `fill_mode: softhebb`. Compare com `n1_fill_center` e `n1_fill_patch`. Se ganhar, o passo seguinte é a L3, com `layer_grow`.

**Por que vem por último.**
- [FATO] SoftHebb é frágil: "Any modifications to its architecture can lead to a drastic performance drop". Nimmo só conseguiu aprender com uma inicialização específica.
- [APOSTA] Ninguém testou filtros congelados dentro da competição.
- [FATO] Parte do ganho vem da ativação Triangle, que "would be detrimental to BP", e nós ficamos com ReLU.

## 7. O que deixei de fora, e por quê

| Ideia | Motivo |
|---|---|
| Forward-Forward puro | Usa gradiente dentro da camada e perde 17 pontos sem truques. Fica só a ideia de nota por par |
| PEPITA | Para em 1 a 2 conv |
| AGOP e Deep ConvRFM | Profundidade não ganhou em CIFAR-10 (67,97 e 67,97), e o kernel é caro |
| Forward Projection e LNT para encher canais | Alvo de rótulo dá no máximo K direções |
| Deep kernel machines | Treinam com Adam, cerca de 2000 h de GPU |
| Sparse Manifold Transform | Nunca empilhado |
| Blocos de camadas | Já temos com `layerwise_backprop` |
| Escala α (Kunin) | Em CNN real muda o filtro, não a acurácia |
| Banco PCA e polaridade na L3, leitura concatenada | Com o agente dos clássicos. [CLÁSSICOS] `L3_pca32_pol64` deu 0,9015 em eggs e 0,8310 em protozoan, contra 0,8361 e 0,7234 do SPiFiL cru (`<scratchpad>/law/classic_tools/pol_*.log`). Se isso se confirmar, é a resposta para a L3 e combina com o caminho 1, que cuida da L4 |

**Limite geral.** [FATO] Quase tudo aqui é 32 px, com rótulo, e dezenas de milhares de imagens. Nada testa 640 px, e só FFCL, Scalable FF, FastHebb e Ciampi chegam perto do nosso tamanho de dataset. Nenhum experimento acima foi rodado: são propostas.
