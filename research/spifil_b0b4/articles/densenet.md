# DenseNet (concatenação) como operação de crescimento — regras × nossa cadeia

## Referência
Huang, G.; Liu, Z.; van der Maaten, L.; Weinberger, K. Q. *Densely Connected Convolutional Networks*. CVPR 2017, arXiv:1608.06993.
Lido em https://ar5iv.labs.arxiv.org/html/1608.06993 (17/11/2026). Seções usadas: §1, §3 (Eqs. 1–2, "Dense connectivity", "Composite function", "Growth rate", "Bottleneck", "Compression"), §4.2 (treino), §5 ("Feature Reuse", Fig. 5).

> DenseNet **não é um método de crescimento** (treina a arquitetura fixa do zero). Mas a sua operação, "cada camada **acrescenta** k mapas ao estado e deixa os anteriores intactos", é, lida como morfismo, a forma mais literal de "o backprop soma, não desfaz".

## Operação (no artigo)
- ResNet (Eq. 1): `x_ℓ = H_ℓ(x_{ℓ−1}) + x_{ℓ−1}`. A soma "may impede the information flow" (§3).
- DenseNet (Eq. 2): `x_ℓ = H_ℓ([x_0, x_1, …, x_{ℓ−1}])`, concatenação. "We never combine features through summation" (§1).
- "Explicitly differentiates between information that is added to the network and information that is preserved". As camadas "keep the remaining feature-maps unchanged" (§1). Cada camada vê o "collective knowledge" (§3).
- `H_ℓ` = BN → ReLU → Conv 3×3. **Taxa de crescimento k**: "each layer adds k feature-maps of its own to this state" (§3).
- DenseNet-B: 1×1 com 4k mapas antes da 3×3. Transição: 1×1 + avg-pool 2×2. DenseNet-C: compressão θ = 0,5 (⌊θm⌋ mapas).
- Inicialização: He et al. 2015 (aleatória). Sem preservação de função, porque não há pai.

### Reúso (§5, Fig. 5)
- Peso absoluto médio como proxy de dependência: "All layers spread their weights over many inputs within the same block". As transições espalham peso por todas as camadas do bloco. O classificador concentra-se nos mapas finais.

## Receita (§4.2)
- SGD, Nesterov 0,9, wd 1e-4. CIFAR/SVHN: batch 64, 300/40 épocas, lr 0,1 ÷10 em 50% e 75%. ImageNet: 90 épocas, batch 256, ÷10 nas épocas 30 e 60. Dropout de 0,2 após cada conv (menos a 1ª) em C10, C100 e SVHN.
- Onde e quando crescer, aceitação, estagnação: **não se aplicam**.

## DenseNet lido como morfismo (operação `add_dense_layer`)
O LEMONADE (§3.1, skip por concatenação "with zero-padding in subsequent convs") dá a regra de preservação:
1. Bloco novo `D` de saída k mapas: `d = H([x_0, …, x_{ℓ−1}])`, com entrada = concatenação dos mapas **da mesma resolução**.
2. O estado passa a ser `[x_0, …, x_{ℓ−1}, d]`. Todo **consumidor** do estado (a camada seguinte ou a cabeça) ganha k **colunas de entrada = 0**.
3. A função é exata para qualquer entrada: os mapas antigos ficam literalmente intactos e os novos têm peso zero no consumidor.
4. O gradiente entra direto nas colunas zero (`∂L/∂W_cons,new = ∂L/∂z · dᵀ`), como no nosso `widen`.

## Regras (checklist "como no DenseNet")
- [ ] Concatenação, nunca soma, dos mapas novos.
- [ ] Bloco novo vê **todos** os mapas anteriores da mesma resolução.
- [ ] Taxa de crescimento k fixa por camada. Opcional: bottleneck 1×1 com 4k e compressão θ = 0,5 nas transições.
- [ ] H = BN-ReLU-Conv3×3.
- [ ] Os mapas antigos não são alterados pela operação.
- [ ] (Como morfismo) colunas novas = 0 em todo consumidor.

## Divergências

| Item | Artigo | Nosso | Ação |
|---|---|---|---|
| Concatenação | Sim (Eq. 2) | **Não existe** no `Backbone` (sequencial; residual identity/gated/project; `add_skip_zero` é **soma**) | Implementar `add_dense_layer` / `add_concat_skip` em `morphisms.py` e `build.py` (estado concatenado por estágio). |
| Entrada do bloco novo | Todos os mapas anteriores `[x_0…x_{ℓ−1}]` | `widen` acrescenta canais que veem só `x_{ℓ−1}` (fan-in SPiFiL sobre o prefixo; as colunas de entrada novas vindas de canais novos são 0, growth_chain.py:696-700) | Na versão densa, o SPiFiL é ajustado sobre a **concatenação** (mapas de várias profundidades). |
| Mapas antigos | Intactos ("preserved") | Os herdados mudam a 0,01×lr (lowlr) no AE e no SSL | "Como no DenseNet" literal, para o morfismo: `protect: freeze` nos mapas antigos. Isso dá "o backprop soma" em sentido estrito. O lowlr é o meio-termo atual. |
| Taxa de crescimento | k fixo por camada (por exemplo, 12, 24, 32) | Larguras por fator (≤ 2,5 por op) até a faixa | Para fidelidade: crescer de k em k mapas por camada densa adicionada. |
| Init dos mapas novos | He (aleatória) | SPiFiL | **Manter SPiFiL** (regra). Divergência declarada. |
| Consumidor | Treinado do zero | Colunas novas = 0 (`_remap_conv` / `_fill_head`) | Igual à regra LEMONADE de concatenação. **Alinhado.** |
| Resolução | Concatena só dentro do bloco denso; a transição reduz | Blocos com `pool` mudam a resolução | Restringir `add_dense_layer` a estágios de mesma resolução. Na fronteira, usar só o widen. |
| Treino | SGD, 300 épocas, do zero | AE local + LeJEPA curto + WiSE + gate | Não comparável (o artigo não cresce). Declarar que só a **operação** vem do DenseNet. |

## Adaptação para SPiFiL ("Dense-SPiFiL", candidata a operação padrão de "somar sem desfazer")
1. `add_dense_layer(stage, k, kernel=3)`: entrada = concatenação dos mapas do estágio; filtros = k **SPiFiL** ajustados sobre essa concatenação (o `NewFilterInit` com o prefixo devolvendo a concatenação); saída anexada ao estado; consumidores com k colunas = 0; cabeça ≥ 1280 com colunas novas = 0 (`head_widen`).
2. Mapas antigos com `protect: freeze` (DenseNet literal) ou `lowlr` (padrão atual), declarado no `OpRecord`.
3. Preservação exata: verificar com `finalize` / `measure_preservation`. Deve dar 0 em eval() e em treino, porque o BN do bloco novo não toca os mapas antigos.
4. Diagnóstico do §5: depois do SSL, medir o peso absoluto médio do consumidor por grupo de mapas (os antigos contra os k novos). Se as colunas novas continuarem perto de 0, o crescimento não foi usado. É o mesmo diagnóstico do α no `rezero.md`.
