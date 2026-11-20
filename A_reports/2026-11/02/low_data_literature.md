# Poucos dados (1–5%) com SPiFiL + autoencoder: literatura e plano

**Resposta curta:** com 1–5% dos dados, o que mais deve ajudar é:
- deixar o SVM ver mais "fotos" de cada exemplo (vistas aumentadas no treino do SVM);
- regularizar o leitor (o SVM no fim);
- treinar menos o autoencoder, ou nem treinar: FLIM puro já foi feito para ~5 imagens.

Aumento de dados forte, sim, mas sem mexer na escala.


## 0. Prioridade do usuário: domain shift + teoria dos filtros vindos do dado

**Em uma frase:** um filtro tirado de pedacinhos reais da imagem só "sabe" ver o que existe naquele domínio. Com pouco dado e domínio deslocado, isso protege mais do que pesos aprendidos por gradiente, que se ajustam ao ruído de 25 exemplos.

### (1) FLIM/SPiFiL: o que a literatura reporta
- **de Souza & Falcão, "Learning CNN filters from user-drawn image markers for coconut-tree image classification"** (IEEE GRSL 2020) — https://arxiv.org/abs/2008.03549
  - Como funciona: filtros estimados camada a camada a partir de marcadores em poucas imagens, sem backprop.
  - O que reporta: fica competitivo com CNNs treinadas.
  - Backprop depois do FLIM: a ideia central é que o encoder é *estimado*, não treinado, e só a cabeça é ajustada.
- **Joao et al., "Building Flyweight FLIM-based CNNs with Adaptive Decoding for Object Detection"** (2023) — https://arxiv.org/abs/2306.14840
  - Dado: ovos de S. mansoni.
  - O que reporta: encoder FLIM sem backprop, a partir de poucas imagens marcadas, mais um decoder adaptativo sem treino. Concorre com detectores grandes.
- **FLIM-BoFP, "FLIM-Bag of Feature Points"** (2026) — https://arxiv.org/abs/2602.20845
  - Dado: cistos de E. histolytica e ovos de Ancylostoma.
  - O que muda: estima filtros a partir de pontos de marcadores, mais rápido que FLIM-Cluster.
  - O que reporta: robustez com poucas imagens.
- **Multi-level Cellular Automata for FLIM networks** (2025) — https://arxiv.org/abs/2504.11406
  - Usa FLIM para inicializar, de novo em ovos de parasita.
- **Dissertação UNICAMP (Krüger Miranda): FLIM para classificar enteroparasitas com poucos dados** — https://www.ic.unicamp.br/en/evento/defesa-de-mestrado-de-maria-anglica-krger-miranda/
- Projeto FAPESP: descreve o FLIM com "~5 imagens, sem backprop" — https://bv.fapesp.br/en/bolsas/225141/

**Limite desta busca:** não verifiquei, texto a texto, as contagens exatas de imagens por classe e de marcadores/superpixels em cada paper. O padrão comum da linha é de poucas imagens (≈3–5 no total, ou 1–2 por classe) e nenhum backprop no encoder.

**Implicação para o caso:** nosso AE camada a camada é um *passo de gradiente em cima do FLIM*. Em 1%, ele pode desfazer a vantagem.

### (2) Por que filtros de estatística de patches resistem a domain shift
- **Coates, Lee & Ng, "An Analysis of Single-Layer Networks in Unsupervised Feature Learning"** (AISTATS 2011) — https://proceedings.mlr.press/v15/coates11a.html
  - Achado: k-means sobre patches com whitening rivaliza com métodos treinados. O que importa é o número de filtros e o pré-processamento, não o algoritmo.
- **Chan et al., "PCANet: A Simple Deep Learning Baseline for Image Classification?"** (TIP 2015) — https://arxiv.org/abs/1404.3606
  - Achado: filtros = autovetores dos patches do próprio domínio, competitivos sem backprop.
- **Ben-David et al., "A theory of learning from different domains"** (MLJ 2010) — https://link.springer.com/article/10.1007/s10994-009-5152-4
  - Ideia: o erro no alvo ≤ erro na fonte + distância entre domínios *no espaço de features* + erro ideal conjunto.
  - Consequência: features estimadas sobre o domínio, com marcadores representativos, encolhem essa distância.
- **Kumar et al., "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution"** (ICLR 2022) — https://arxiv.org/abs/2202.10054
  - Achado: ajustar o encoder com pouco dado distorce boas features e piora fora da distribuição. Já o linear probe, que é o nosso caso SVM, preserva.
  - Para nós: é a base teórica para "AE com 0 épocas" ou "AE curto" em 1%.
- **Li et al., "Revisiting Batch Normalization For Practical Domain Adaptation" (AdaBN)** (2016) — https://arxiv.org/abs/1603.04779
  - Ideia: reestimar estatísticas por domínio corrige o shift.
  - Para nós: o PCA-whitening estimado com n = 25 fica *pior* que estimado com todo o dado sem rótulo.

### Justificativa teórica de cada experimento do ranking (§3)
1. **SVM com vistas aumentadas** → minimização de risco vicinal (Chapelle et al., "Vicinal Risk Minimization", NIPS 2000, https://papers.nips.cc/paper/1876-vicinal-risk-minimization). Com n = 25, o risco empírico é instável. As vistas aumentam a vizinhança sem tocar no encoder (Kumar 2022).
2. **Whitening (estimar sem rótulo em todo o treino, ou com shrinkage)** — sem rótulo ou com shrinkage** → Ledoit–Wolf 2004 (covariância com n ≪ d) + AdaBN 2016. O whitening não precisa de rótulo: estime-o com todas as imagens de treino sem rótulo.
3. **AE com 0 épocas (FLIM puro)** → de Souza & Falcão 2020; Joao et al. 2023; Kumar et al. 2022.
4. **strong_aug no AE** → Tellez et al. 2019. A cor e o foco são justamente os eixos do shift entre lâminas.
5. **SPiFiL com mais imagens por classe** → Coates et al. 2011: a cobertura do dicionário de patches pesa mais que o algoritmo. Mais imagens dão filtros mais representativos do domínio.
6. **Orçamento fixo de passos no AE** → Kumar et al. 2022: menos gradiente = menos distorção das features FLIM.

**Ranking revisado (domain shift):** 1, 2 e 3 ficam empatados no topo, e o 3 sobe para testar a hipótese central do usuário.

## 1. Aumentos coerentes com microscopia de parasitas

| Op | O que imita no mundo real | Pode quebrar uma pista de classe? |
|---|---|---|
| Rotação arbitrária (0–360°) | O ovo cai na lâmina em qualquer ângulo | Não. Use reflexão nas bordas, porque o fundo é homogêneo |
| Flips H/V | Mesma coisa: orientação | Não. Mas a TTA de 8 já cobre isso, então sozinho rende pouco |
| Jitter LAB/HSV (stain aug) | Corante (lugol/iodo), lote, câmera, balanço de branco | Pouco risco se for leve. A cor ajuda a separar cisto e impureza, então limite o matiz |
| Blur gaussiano / desfoque | Foco errado, profundidade do ovo | Moderado. Casca e estrias são textura fina; σ ≤ 1,5 px |
| Ruído gaussiano/Poisson | Sensor, pouca luz | Baixo, com σ ≤ 0,03 |
| Brilho/gama | Iluminação do microscópio | Baixo |
| Escala | Zoom | **SIM, quebra.** O tamanho separa as espécies (±5–10% dentro de uma classe). Use no máximo ±5%, ou nada |
| Deformação elástica | Ovo amassado, larva curvada | Ovo: quebra a forma oval, que é uma pista. Larva: ajuda |
| Cutout | Oclusão por detrito | **Arriscado.** O objeto é só 2–7% do quadro: o corte erra o objeto (inútil) ou apaga o objeto todo |
| Mixup/CutMix no AE | Regularização | Pouca evidência para reconstrução. Sem rótulo, mixup no AE só ensina a reconstruir misturas. Não recomendo |
| RandAugment para microscopia | Política automática | Funciona, mas é só um empacotamento das ops acima. Com 25 imagens, não dá para ajustar a magnitude com segurança |

**`strong_aug` recomendado (5 ops):**
1. Rotação contínua em [0, 360) com reflexão nas bordas.
2. Flip H/V com p = 0,5.
3. Jitter em LAB (Tellez et al. "HED-light", adaptado):
   - L × [0,9; 1,1] + [−5; 5];
   - a, b ± 4;
   - p = 0,8.
4. Blur gaussiano com σ ∈ [0,1; 1,2] e p = 0,3.
5. Ruído gaussiano com σ ∈ [0; 0,02] e p = 0,3.
6. (Opcional) Escala em [0,97; 1,03]. **Nunca** acima de ±5%.

Referências:
- Tellez et al., "Quantifying the effects of data augmentation and stain color normalization in CNNs for computational pathology", MedIA 2019 — https://arxiv.org/abs/1902.06543
- Cubuk et al., "RandAugment", 2020 — https://arxiv.org/abs/1909.13719
- DeVries & Taylor, "Improved Regularization of CNNs with Cutout", 2017 — https://arxiv.org/abs/1708.04552
- Balestriero et al., "The Effects of Regularization and Data Augmentation are Class Dependent", 2022. Mostra que um aumento pode prejudicar classes específicas — https://arxiv.org/abs/2204.03632

## 2. Técnicas para MUITO poucos dados (~25 imagens)

- **Vistas aumentadas no treino do SVM:**
  - O que é: aumento no nível das features. Cada imagem de treino gera K vistas, e todas entram no SVM.
  - Por que deve ajudar: hoje a TTA só faz média no teste. Com 25 pontos, o SVM fica com fronteira instável.
  - Referência: DeCAF/"CNN features off-the-shelf" (Razavian 2014, https://arxiv.org/abs/1403.6382). O ganho com jitter no treino aparece lá.
- **Shrinkage LDA / SVM mais regularizado:**
  - Com n ≪ d, a covariância estimada é ruim. Ledoit–Wolf (2004, https://doi.org/10.1016/S0047-259X(03)00096-4) resolve isso.
  - Cuidado: o PCA-whitening com n = 25 amplifica ruído. Corte para k ≤ n−1 componentes, ou use shrinkage.
  - Outro caminho: escolher C pequeno por validação.
- **Não treinar o backbone (FLIM puro):**
  - FLIM foi desenhado para ~5 imagens, sem backprop. O AE aprende com o próprio dado pequeno e pode sobreajustar e apagar o viés dos marcadores.
  - Refs:
    - de Souza & Falcão, "Learning CNN filters from user-drawn image markers for coconut-tree image classification", IEEE GRSL 2020 — https://arxiv.org/abs/2008.03549
    - Joao et al., "Building Flyweight FLIM-based CNNs with Adaptive Decoding for Object Detection", 2023 (ovos de S. mansoni) — https://arxiv.org/abs/2306.14840
    - "FLIM-Bag of Feature Points", 2026 (cistos de E. histolytica, ovos de Ancylostoma) — https://arxiv.org/abs/2602.20845
    - Dissertação UNICAMP (Krüger Miranda) sobre FLIM para enteroparasitas com poucos dados — https://www.ic.unicamp.br/en/evento/defesa-de-mestrado-de-maria-anglica-krger-miranda/
- **Filtros k-means / PCANet:**
  - O que é: filtros tirados do próprio dado, sem treino.
  - Refs: Coates & Ng 2011, "The importance of encoding versus training with sparse coding and vector quantization" — https://icml.cc/2011/papers/485_icmlpaper.pdf ; Chan et al., "PCANet", 2015 — https://arxiv.org/abs/1404.3606
  - Comparação: são parentes do SPiFiL. Servem como baseline, não como knob.
- **SPiFiL com mais imagens/superpixels:**
  - Hoje é 1 imagem por classe. Passar para 3–5 por classe, ou dobrar os superpixels, ainda cabe em 1% para ovos.
  - Rende mais filtros diversos sem custo de treino.
- **Self-training/pseudo-rótulos no SVM:**
  - Usa os ~95% sem rótulo. Funciona quando a acurácia inicial é razoável, e 0,27 é baixo, então o risco de reforçar erro é alto.
  - Mitigação: confiança alta e balanceamento por classe.
  - Ref: Lee 2013, "Pseudo-Label" — https://www.researchgate.net/publication/280581078
- **Early stopping por passos no AE:**
  - Com menos dados, o AE dá menos passos, mas cada imagem é vista muito mais vezes, o que leva a sobreajuste.
  - Fixe o número de passos, ou pare pela reconstrução na validação.

## 3. Ranking: ganho esperado em 1–5% ÷ custo

Cada experimento muda um knob só e mantém o protocolo do AE.

1. **Treinar o SVM com as vistas aumentadas** (as 8 da TTA, ou o strong_aug): custo ~0, só no eval, e ganho alto.
2. **Ajustar o whitening ao tamanho do treino** (n_comp = min(atual, n_train−1), ou trocar por shrinkage): custo ~0 e ganho alto. Suspeita forte de que esta é a causa do colapso em 1%.
3. **AE com 0 épocas** (só filtros SPiFiL): custo baixo, porque é diagnóstico. Se ganhar em 1%, o AE está sobreajustando.
4. **strong_aug no treino do AE** (as 5 ops acima): custo de um treino, ganho médio.
5. **SPiFiL com 3 imagens por classe** (em vez de 1): custo baixo e ganho médio, maior nas classes difíceis.
6. **Orçamento fixo de passos no AE** (igual ao de 100%, em vez de épocas): custo de um treino, ganho médio.

Self-training ficou de fora. Só vale tentar depois que 1–2 subirem a base acima de ~0,5.
