# Network Morphism (NetMorph) — regras de implementação × nossa cadeia

## Referência
Wei, T.; Wang, C.; Rui, Y.; Chen, C. W. *Network Morphism*. ICML 2016, arXiv:1603.01670.
Lido em https://ar5iv.labs.arxiv.org/html/1603.01670 (17/11/2026). Seções usadas: §1 (definição), §3.1 (Eqs. 3–6), §3.2 (Algoritmos 1–2, Claim 1/Eq. 7), §3.3 (P-activation, Eqs. 8–11), §3.4.1 (largura, Eq. 18), §3.4.2 (tamanho do kernel), §3.5 (subnet, Eqs. 19–20), §4.2 (CIFAR10), §4.4 (VGG16).

## Operação
**Definição** (§1): morfismo é "a parameter-transferring map from a parent network to a child network that preserves its function and outputs". Equação do morfismo (Eq. 6): `G̃ = F_{l+1} ⊛ F_l`, em que `G̃` é o filtro do pai com zero-padding e kernel efetivo `K̃ = K₁ + K₂ − 1` (Eq. 5). No caso linear, `G = F_{l+1}·F_l` (Eq. 3).

### Profundidade (§3.2)
- **Algoritmo 1**: `F_l` e `F_{l+1}` começam como **ruído aleatório** e `G` recebe zero-padding até `G̃`. Alternam-se deconvoluções `F_{l+1} = deconv(G̃, F_l)` e `F_l = deconv(G̃, F_{l+1})`, minimizando `‖G̃ − conv(F_l, F_{l+1})‖²` até perda 0 ou `maxIter`. Ao final, os dois são normalizados para o mesmo desvio-padrão.
- **Claim 1 (Eq. 7)**: se `min(C_l·C_{l−1}·K₁², C_{l+1}·C_l·K₂²) ≥ C_{l+1}·C_{l−1}·(K₁+K₂−1)²`, a perda zera em um passo (sistema subdeterminado). O próprio artigo avisa que, no caso geral, "it is not guaranteed that the loss … will always converge to 0".
- **Algoritmo 2 (prático)**: roda o Alg. 1 com `maxIter = 1`, começando de `K₂ʳ = K₂` e reduzindo `K₂ʳ` até a perda zerar. Depois faz zero-padding até `K₂` e normaliza.
- **Não linearidade (§3.3)**: para φ idempotente, como a ReLU, vale o IdMorph (Eq. 8). Para φ geral, usa-se a família P-activation `φ^a = (1 − a)·φ + a·φ_id` (Eq. 9), com **a = 1 no início** (camada linear, morfismo exato, Eqs. 10–11). "The value of a shall be learned when we continue to train the model."

### Largura (§3.4.1, Eq. 18)
- Exige `F̃_l(c̄_l, ·) * F̃_{l+1}(·, c̄_l) = 0` nos canais novos `c̄_l`. "We set the one with less parameters to 0, and the other one to random noises": zera-se o lado (fan-in ou fan-out) com **menos parâmetros** e o outro recebe ruído.
- Depois, **permutação aleatória** dos canais, para que zeros e ruído não fiquem agrupados.

### Tamanho do kernel (§3.4.2)
- Filtros e blobs recebem zero-padding de `(K̃ − K)/2` por lado. É exato e não usa ruído.

### Subnet (§3.5)
- Eq. 19 estende o morfismo a P+1 camadas em sequência. Eq. 20 trata subnets empilhadas com `Σ G_i = G` (por exemplo, `G_i = G/n`).

### Onde e quando crescer
- O artigo não dá critério formal e se restringe ao "expanding mode". Na prática: no VGG16, uma conv é adicionada "at the end of the first three subsets" (§4.4). No CIFAR10, algumas camadas recebem 1×1 com 4× canais (§4.2).
- Sobre o pai: nada sobre convergência ou platô. O pai VGG16 foi "de-multiscaled" com treino adicional, o que custou cerca de 1% (§4.4).

## Receita
- Depois do morfismo, "we continue to train the child network after morphing" (§4.4), a **rede inteira**. Não informa lr, otimizador nem épocas. No VGG16, morfismo mais treino terminaram em 5 dias, cerca de 15× mais rápido que do zero (§4.4).
- O artigo não cita ruído além da inicialização aleatória.
- Aceitação: **nenhuma**. A comparação é com a rede treinada do zero ("Raw"), e o NetMorph vence, com diferença que cresce com a profundidade (§4.2).

## Regras (checklist "como no NetMorph")
- [ ] Largura: zerar o lado (fan-in ou fan-out) com **menos** parâmetros e encher o outro com ruído. Permutar os canais aleatoriamente.
- [ ] Kernel: zero-pad central `(K̃−K)/2`.
- [ ] Profundidade: substituir `G` por `F_{l+1} ⊛ F_l` resolvido por deconvolução (Alg. 2) e normalizar os dois ao mesmo desvio. A não linearidade inserida é P-activation com `a = 1`, aprendível.
- [ ] Continuar treinando a rede inteira (hiperparâmetros não especificados no artigo).
- [ ] Sem gate de aceitação.

## Divergências

| Item | Artigo | Nosso | Ação |
|---|---|---|---|
| Largura: qual lado zerar | O lado com **menos** parâmetros; o outro com ruído | Sempre o fan-out = 0 e o fan-in = SPiFiL (morphisms.py:309, growth_chain.py:642) | Manter o fan-out zero: a regra exige fan-in SPiFiL (≠ 0). Quando `C_{l−1}K_l² < C_{l+1}K_{l+1}²`, o artigo zeraria o fan-in, o que é incompatível com SPiFiL. Divergência registrada. |
| Ruído no lado não zerado | Ruído aleatório | SPiFiL (dados) | Manter. É a regra do projeto. |
| Permutação de canais | Aleatória | Novos vão ao fim do mapa (`out_local = herdados + [-1]*n`) | Irrelevante para a função e para o gradiente. Não mudar. |
| Kernel | Zero-pad central | `kernel_grow` com zero-pad central (`_remap_conv`, morphisms.py:116-126) | **Igual**. |
| Profundidade | `G → F_{l+1} ⊛ F_l` por deconvolução, ruído inicial, P-activation a=1 | `deepen_identity` (IdMorph, Eq. 8, só com entrada ≥ 0) e `deepen_residual_zero` (y = act(conv x)+x com conv = 0) | Ver Adaptação (NetMorph-SPiFiL). Faltam a P-activation e a deconvolução. |
| Restrição de entrada ≥ 0 | Resolvida pela P-activation (a=1 é linear) | `NotExactError` quando a entrada pode ser < 0 (morphisms.py:436) | Implementar `PAct(a=1)`: `deepen_identity` passa a ser exato em qualquer posição. |
| O que treina | Rede inteira | Herdados a 0,01×lr; novos a lr cheio; AE local → LeJEPA curto → WiSE | "Como no NetMorph": `protect: full`, sem AE, sem WiSE. Os hiperparâmetros do artigo não existem; usar os do pai. |
| Aceitação | Nenhuma | κ_filho ≥ κ_pai | Nossa. Para fidelidade, registrar sem filtrar. |
| Estagnação | Não discute | Não verificamos | — |

## Adaptação para SPiFiL ("NetMorph-SPiFiL")
1. **Largura**: fan-in SPiFiL, fan-out 0, sem permutação. Declarar "NetMorph §3.4.1 com o lado zerado fixado no fan-out por causa da regra SPiFiL".
2. **Profundidade por deconvolução com SPiFiL** (opcional, precisa de aval do usuário): para trocar a camada `G` do pai por `F_{l+1} ⊛ F_l`, fixar `F_l` = filtros **SPiFiL** ajustados na entrada da camada `l` e resolver **só** `F_{l+1} = deconv(G̃, F_l)` por mínimos quadrados, um passo do Alg. 1. O resultado é exato quando vale a condição da Eq. 7, com P-activation `a = 1` entre eles e `a` aprendível. **Conflito com "o backprop soma, não desfaz"**: `G` (SPiFiL herdado) deixa de existir como filtro explícito e passa a ser refatorado em `F_{l+1}`. Por isso, só como braço declarado, nunca como padrão.
3. **Profundidade aditiva (padrão compatível)**: em vez de refatorar `G`, inserir uma camada nova com filtros SPiFiL e **saída gateada a zero**. É o `deepen_rezero` descrito em `rezero.md`, que preserva `G` intacto.
4. **P-activation**: adicionar `PAct` com `a` iniciado em 1 e tratado como parâmetro **novo** (lr cheio) aos blocos inseridos. Remove a restrição `_nonneg_output`.
