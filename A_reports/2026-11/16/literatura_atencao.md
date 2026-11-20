# Agente 6: literatura de atenção e o que sobra dela SEM pesos treináveis

> 17/11/2026. As fontes foram lidas no arXiv (abs) e no ar5iv (HTML do método). As marcações **[memória]** indicam itens que não foram conferidos nas páginas.
> A "versão sem pesos" de cada item é **hipótese própria** (ATTENTION_ADDENDUM.md), e não reprodução do método. O código está em `ga_search/modehb/fixed_attention.py` (cópia `scratchpad/attn6`; patch em `agent6/fixed_attention.patch`).

Contexto de uso: CNN SPiFiL rasa com 2 a 7 blocos e 5 a 1024 canais. Os mapas finais têm de 56² a 7² posições (ótimo de 28² a 14², A1a §A2). A cabeça é uma `EmbeddingHead` com 1×1 treinado, pooling e embedding ≥ 1280 (R11). A leitura é um SVM linear C=100 sem scaler (ORIG).

---

## 1. Vaswani et al. 2017, *Attention Is All You Need* (arXiv:1706.03762; NeurIPS 2017 [memória])

**Princípio** (§3.2.1): a atenção é calculada como Attention(Q,K,V) = softmax(QKᵀ/√d_k)V.
- O fator √d_k evita a saturação da softmax: se q e k têm componentes i.i.d. de variância 1, q·k tem variância d_k (nota de rodapé).
- Na multi-head (§3.2.2), cada cabeça é headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V), com saída Concat·W^O, h=8 e d_k=64.
- O custo por camada é O(n²·d) (Tab. 1).

**Versão sem pesos** (operador A, `FixedSelfAttention`):
- Q=K=V=X, com Y = softmax(sim(X̂ᵢ, X̂ⱼ)/τ)·X. A similaridade é `cosine` ou `normalized_dot` (= q·k/(√d·τ), a escala do paper).
- A fusão é Y' = (1−α)X + αY, com α fixo.
- As keys e values são `tokens` ∈ {16, 49, 196}, tirados de um pooling adaptativo do próprio mapa. As queries ficam em resolução cheia, então o custo é O(N·M·C) em vez de O(N²·C). Com tokens ≥ H·W o operador é exatamente Q=K=V=X (há teste).

**Limitações:**
- O ganho do Transformer vem de W^Q, W^K e W^V aprendidas, que escolhem *o que comparar*. Sem elas, a afinidade é só a similaridade das features SPiFiL.
- As features SPiFiL são muito redundantes: o rank efetivo das ativações fica em 0,11–0,16·C nos blocos profundos, e 40–59% dos filtros têm um par com cos > 0,9 (A1a §B3). Com isso a similaridade cosseno tende a ser alta para quase todos os pares, e a softmax fica perto de uniforme.
- Por isso τ precisa ser pequeno (0,03–0,1) e a normalização `zscore` por canal ajuda a decorrelacionar.

## 2. Wang, Girshick, Gupta, He 2018, *Non-local Neural Networks* (arXiv:1711.07971, CVPR 2018)

**Princípio** (§3): yᵢ = (1/C(x)) Σⱼ f(xᵢ,xⱼ) g(xⱼ). As formas de f são:

| Forma | f(xᵢ,xⱼ) | C(x) | Pesos |
|---|---|---|---|
| Gaussiana | exp(xᵢᵀxⱼ) | Σⱼ f | **nenhum** |
| Embedded Gaussian | exp(θᵀφ), equivalente a softmax | Σⱼ f | θ, φ |
| Dot product | θᵀφ | N | θ, φ |
| Concatenação | ReLU(w_fᵀ[θ,φ]) | N | w_f, θ, φ |

- O bloco é zᵢ = W_z yᵢ + xᵢ. O W_z (com BN de escala zero no §4) começa em zero, então o bloco começa como identidade.
- Na Tab. 2a (Kinetics), todas as formas dão +0,7 a +1,1 ponto. Os autores concluem que o ganho vem do caráter não-local, e não da softmax.

**Versão sem pesos:**
- É a f Gaussiana com g = identidade, que é o operador A.
- Trocamos W_z por α fixo. **α=0 reproduz exatamente o truque de inicialização do paper**: o bloco nasce como identidade, o que é útil para crescimento com preservação de função.
- O max-pool em φ/g do paper corresponde às nossas keys/values por pooling (`tokens`).

**Limitações:**
- O paper nunca testou a versão totalmente sem pesos: g e W_z são sempre aprendidos, e os ganhos foram medidos com fine-tuning de ResNets profundas.
- exp(xᵢᵀxⱼ) satura sem escala. Usamos cosseno/τ, ou normalized_dot/(√d·τ).

## 3. Touvron et al. 2021, *Going deeper with Image Transformers* (CaiT, arXiv:2103.17239; ICCV 2021 [memória])

**Princípio:**
- **LayerScale** (§2): x' = x + diag(λ)·SA(η(x)), com λ iniciado em ε (0,1 até profundidade 18; 1e-5/1e-6 nas mais profundas).
- **Class-attention** (§3): só o token CLS é query, com Q = W_q·x_cls e K, V sobre [cls, patches]; os patches ficam congelados.
- O custo é linear em p, contra p² da self-attention.
- Na Tab. 2, com DeiT-S, o average pooling dá 80,3 e o CaiT 10+2 também dá 80,3, com menos FLOPs.

**Versão sem pesos** (operador B, `FixedAttentionPooling`):
- A query é a GAP (ou GeM, ou uma GAP refinada por uma passada de atenção). Os pesos são a = softmax(sim(q, xᵢ)/τ).
- O pooling da cabeça vira a versão ponderada: weighted mean, weighted GeM ou momentos ponderados.
- O LayerScale vira o α fixo: w = (1−α)/N + α·a. Com α=0, o resultado é a cabeça original exata.
- O custo é O(N·C), e o gene `tokens` não se aplica.

**Limitações:**
- O próprio CaiT mostra que o average pooling já é muito competitivo, e sem W_q/W_k a vantagem do class-attention desaparece.
- A query GAP pondera mais as posições parecidas com a média, isto é, o fundo dominante. Isso pode **prejudicar objetos pequenos** (ovos, cistos) e ajudar cenas homogêneas.
- Há um caso degenerado identificado: com `zscore` e query GAP, a query vira o vetor nulo e os pesos ficam uniformes. O reparo troca para `normalization=none`.

## 4. Chen, Fan, Panda 2021, *CrossViT* (arXiv:2103.14899, ICCV 2021)

**Princípio** (§3.3, Eq. 5–7):
- O CLS do ramo L, depois de uma projeção f, é a query; os tokens do ramo S são keys e values. A saída é y = f(x_cls) + MCA(·), e g projeta de volta.
- f e g são projeções **aprendidas** que alinham as dimensões dos ramos.
- Na Tab. 6 (top-1): sem fusão 80,2; class token 80,3; pairwise 80,3; cross-attention 81,0, com +3M parâmetros.

**Versão sem pesos** (operadores C e D, `FixedCrossAttention`):
- As queries vêm da camada profunda (L3) e as keys/values da rasa (L2), ou, no D, de um banco multiescala de L1..L3 mais a própria camada.
- **Os papéis de f e g são feitos por estatística de treino, sem rótulos:**
  - `pca`: PCA-whitening por camada, mais uma rotação de Procrustes ajustada em posições co-localizadas. Os eixos de PCA de camadas diferentes não são tratados como equivalentes.
  - `svd`: CCA, ou seja, SVD da covariância cruzada das camadas branqueadas. Os eixos são os pares canônicos de correlação máxima.
  - `shared_channels`: z-score por canal. Só vale quando há correspondência de canais, por exemplo um skip de identidade.
  - O caminho de volta usa a pseudo-inversa (`back`).
- As estatísticas são ajustadas **só no treino** e reutilizadas em validação e teste (há teste).

**Limitações:**
- Mesmo aprendidas, as fusões simples ficam a 0,1 ponto do baseline; só a cross-attention completa ganha 0,8.
- Sem f/g aprendidas, o ganho esperado é marginal. A informação nova só aparece se a camada rasa tiver detalhe que a profunda perdeu (resolução maior).
- O alinhamento linear entre camadas não-lineares é aproximado. Vale a regra do adendo: "não tratar projeções arbitrárias como espaços semânticos equivalentes". Por isso fazemos Procrustes/CCA, e sempre comparamos com a versão sem cross (E0).

## 5. Hu et al. 2018, *Squeeze-and-Excitation Networks* (arXiv:1709.01507; CVPR 2018, TPAMI)

**Princípio** (§III): z_c = GAP(u_c); s = σ(W₂δ(W₁z)), com r=16; x̃_c = s_c·u_c.
- Ablações em ResNet-50, top-1 err: baseline 23,30; SE 22,28.
- Com ReLU no lugar da sigmoid o erro é 23,47, **pior que o baseline**.
- Sem squeeze (NoSqueeze) o erro é 22,93.
- Não há ablação sem as FCs.

**Versão sem pesos:**
- Gating por estatística do próprio canal, s_c = σ(z-score de z_c). Mas, como notou a revisão, **antes de um SVM linear sobre GAP isso é quase redundante**: z_c·f(z_c) é só uma não-linearidade por dimensão.
- Por isso **não** implementamos SE puro. A parte útil da ideia, a interação entre canais, aparece no operador E (`FixedCovarianceAttention`): A = softmax(corr(canais)/τ) de tamanho C×C, com Y = A·X. É a "channel attention" sem pesos, no estilo do módulo de canais do DANet, que também não tem parâmetros. O E age **antes** do 1×1 e do pooling da cabeça, onde não é redundante.

**Limitações:**
- O custo é O(C²·N). Na c00009 do RS (280 canais, 56² posições) isso dá 0,99 GFLOP (2·MAC), ou 57% do encoder com a cabeça (1,75 GFLOP = 2 × 0,875 GMAC; smoke).
- O reparo limita o operador por memória. Com C=1280 (location=head), o Gram por lote fica grande.

## 6. Lin, RoyChowdhury, Maji 2015, *Bilinear CNN Models for Fine-grained Visual Recognition* (arXiv:1504.07889, ICCV 2015; versão estendida TPAMI 2018 [memória])

**Princípio** (§III-A):
- Φ(I) = Σ_l f_A(l)ᵀ f_B(l), o produto externo somado sobre as posições (orderless).
- Depois vêm signed sqrt e ℓ2. O classificador é um SVM linear.
- Projetar um dos lados com PCA (16× menor) perde menos de 1%.
- Sem fine-tuning já é forte: 80,1 → 84,0 com FT.

**Versão sem pesos:** o bilinear já não tem pesos. No repo ele existe como `CovReadout`, uma leitura de 2ª ordem fora da cabeça: PCA k=64, sqrtm, triângulo superior, signed-sqrt+ℓ2, scaler. O COV ganha do ORIG em 90% dos jobs, com mediana +0,123 (A1b §D1).
- **Não duplicamos** essa funcionalidade.
- O operador E usa a 2ª ordem para **pesar** a mistura de canais dentro do mapa, mantendo a cabeça ≥1280 e a régua ORIG. Por isso E5 é comparável a E0, e o COV fica como coluna de referência.

**Limitações:**
- A dimensão é C² (com 300 canais, 90k).
- A covariância fica mal condicionada com poucas posições (k ≤ nº de posições, A1d §6.2).
- O eigh fp64 leva cerca de 1 h por fit no athena.

## 7. Shanmugam et al. 2021, *Better Aggregation in Test-Time Augmentation* (arXiv:2011.11156, ICCV 2021)

**Princípio e achados:**
- O TTA quase sempre ajuda no saldo, mas **"can change many correct predictions into incorrect predictions"**: cerca de 1/3 das previsões alteradas viram erradas (ResNet-18/ImageNet, §3.2).
- **Causas de dano** (§3.3):
  - crops e zoom deslocam o foco para componentes pequenos ou descentralizados;
  - em classes parecidas, o zoom enfatiza traços não discriminativos.
- **As invariâncias dependem da classe** (§3.4, Flowers): crops prejudicam classes com traço pequeno.
- **Agregação:**
  - a média é quase sempre comparável à agregação aprendida;
  - na agregação aprendida (AugTTA/ClassTTA, ajustada na validação), as escalas de 10% e contrast/cutout recebem peso 0, enquanto crop, translação e blur recebem peso alto;
  - o ganho correlaciona com o número de exemplos por classe (r = 0,95).

**Versão sem pesos** (`tta_views` + `aggregate_views`):
- **Agregação:**
  - média uniforme `mean`, igual ao `encode_tta` do repo;
  - `l2_mean` como variante identificada;
  - nenhuma agregação aprendida, conforme o adendo: a política de TTA só pode ser otimizada **depois** e só na validação.
- **Vistas por geometria:**
  - `rot_flip` (cena aérea, microscopia, dermatoscopia): tta4 = id/hflip/vflip/rot180; tta8 = 4 rot90 × {id, hflip}, igual ao `gem_tta`;
  - `flip_scale` (flowers, com orientação canônica): {id, hflip} × escalas.
- `multiscale` usa 0,75 / 1 / 1,25 e custa ×3,125.

**Limitações:**
- Ovos e cistos pequenos são o caso "traço pequeno", onde escala e zoom podem prejudicar. O multiscale 0,75 nesses domínios merece cautela.
- Com poucos exemplos por classe (RS: 682 de treino em 14 classes), o ganho esperado é menor.

---

## Síntese para o desenho

| Operador | Base | O que sobra sem pesos | Expectativa (hipótese) |
|---|---|---|---|
| A self | Vaswani, Non-local | non-local gaussiano, α fixo (= W_z=0 em α=0) | pequeno; depende de τ e normalização |
| B pool | CaiT | GAP refinada por similaridade | ≈ GAP; risco em objetos pequenos |
| C cross | CrossViT | alinhamento por PCA/CCA de treino no lugar de f/g | marginal; útil se L2 tem detalhe que L3 perdeu |
| D multiescala | CrossViT, FPN, Inception (+0,42 z, A1a §A3) | banco de tokens multiescala, pesos de escala pela softmax | o mais plausível nas redes rasas |
| E covariância | Bilinear, SE, DANet-CAM | mistura de canais pela correlação | medir contra o COV; custo C²N |
| TTA | Shanmugam | média uniforme de vistas seguras | +, com custo ×4/×8/×3,1 |

**Achado transversal: todo ganho publicado com atenção vem com projeções aprendidas e treino end-to-end.** A versão sem pesos é uma hipótese a medir, e o próprio CaiT mostra que o average pooling já é forte. Por isso E0 é sempre o controle, e o gene `aggregation=mean` é um controle sem similaridade, que só mistura o contexto global.
