# ReZero — regras de implementação × nossa cadeia

## Referência
Bachlechner, T.; Majumder, B. P.; Mao, H. H.; Cottrell, G. W.; McAuley, J. *ReZero is All You Need: Fast Convergence at Large Depth*. arXiv:2003.04887 (2020; UAI 2021).
Lido em https://ar5iv.labs.arxiv.org/html/2003.04887 (17/11/2026). Seções usadas: §1 (Eq. 1), §2.1 (Eq. 3), §3 (Eq. 4), §3.1 (Eqs. 5–8, Fig. 2), §4 (Fig. 3), §5 (nota 3), §6.1–6.3 (Eq. 10, Tab. 3, Fig. 6), §7, Ap. B, C, E.1, E.2.

> ReZero **não é um método de crescimento**. É uma parametrização residual com identidade no início. O §7 só especula que ela "may … allow for progressive stacking of layers". Entra aqui porque é a forma exata de **inserir uma camada com pesos não nulos sem mudar a função**.

## Operação
- `x_{i+1} = x_i + α_i F(x_i)` (Eqs. 1 e 4), com **um escalar α_i por camada e α_i = 0 no início** ("α = 0 at the beginning of training"). No Transformer, um α é compartilhado pelos subblocos de atenção e de feed-forward (Eq. 10).
- **Os pesos de F NÃO são zero.** Seguem a inicialização normal `σ²_W = 2/w` (§4) ou Xavier (§6.1).
- No início, a rede é a identidade e "trivially satisfies dynamical isometry". Todos os valores singulares do Jacobiano entrada–saída valem 1 (§3, §6.1, Eq. 3).
- **Dinâmica** (§3.1): no modelo de brinquedo `x_L = (1 + αw)^L x_0` (Eq. 5), a atualização de w é ∝ α (Eq. 6). Logo, **F não recebe gradiente enquanto α = 0**, e o 1º passo move α (Eq. 8). "Initially the gradients for all parameters defining F vanish" (§1).
- |α_i| sobe cedo e depois decai até cerca de 1/L: 0,0898 com 12 camadas, 0,0185 com 64 (§6.3, Fig. 6).
- Ruído e simetria: não se aplica. F tem inicialização aleatória comum.

## Receita
- ResNet/CIFAR-10 (Ap. E.1): SGD, lr 0,1, ÷10 nas épocas 100 e 150, 200 épocas, batch 128, wd 5e-4, momento 0,9.
- Superconvergência (Ap. E.2): one-cycle de 45 épocas com pico de lr 1,2. "The residual weights cannot tolerate the extremely large learning rates", por isso o **lr de α fica fixo em 0,1**. Nota 3 (§5): é preciso "a small, constant learning rate for the residual weights", senão diverge.
- Transformers (Ap. B): LAMB, lr fixo de 0,016, **sem warm-up**. Ap. C (profundos): lr = 0,0005·√batch, até 100 épocas.
- Comparações (Tab. 2 e 3): ReZero com α = 0 converge mais rápido que Post-Norm, Pre-Norm e FixUp. **ReZero com α = 1 não melhora** sobre o Transformer comum (§6.2), ou seja, o ganho vem do início em zero.
- Onde e quando, aceitação, estagnação: não se aplicam (não é crescimento).

## Regras (checklist "como no ReZero")
- [ ] Bloco novo = `x + α·F(x)`, com **um** α escalar por bloco iniciado em 0.
- [ ] F com inicialização **não nula**: para nós, SPiFiL.
- [ ] α treinável com **lr pequeno e constante**, separado do resto quando o lr do resto for alto ou tiver cronograma.
- [ ] Sem warm-up obrigatório.
- [ ] Não há norm no ramo como pré-requisito ("ReZero alone is sufficient … even in the absence of various norms", §2.2).

## Divergências

| Item | Artigo | Nosso | Ação |
|---|---|---|---|
| Inserir camada residual | `x + α F(x)`, F aleatório ≠ 0, α = 0 | `deepen_residual_zero`: `x + act(conv(x))` com **conv = 0**, sem α (morphisms.py:463-488) | **Implementar `deepen_rezero`**: `x + α·act(norm(conv_SPiFiL(x)))`, α = 0. Resolve a violação da regra: hoje o bloco novo nasce **zerado**, sem nenhum filtro SPiFiL. |
| Gradiente inicial | α recebe gradiente; F só depois que α ≠ 0 | conv = 0: a conv recebe gradiente via `act'(0)·x` (silu: 0,5; relu: 0, morto), e o filtro aprendido sai de zero, não de SPiFiL | Com ReZero-SPiFiL, os filtros já começam informativos e α decide quanto entra. É coerente com "o backprop soma": α soma a contribuição SPiFiL ao caminho herdado. |
| `add_skip_zero` | — (o ReZero gateia o **ramo residual**) | Gateia o **skip**: `y + λx`, λ = 0 (morphisms.py:491) | Não é ReZero; é um skip novo de contribuição zero. Manter, com o nome certo ("skip gateado", estilo LEMONADE). |
| lr de α | Pequeno e constante (por exemplo, 0,1 enquanto o resto vai a 1,2) | Não existe α. lr 1e-3 fixo para os novos, 1e-5 para os herdados | Criar um grupo para α: lr constante ≤ lr dos novos, sem wd, sem lowlr (α é novo). |
| Um α por camada | Escalar por bloco | — | Fiel: um escalar por bloco inserido. A variante "α por canal" é extensão nossa e deve ser declarada. |
| Widen | — | Fan-out = 0 nas colunas novas | Opcional "ReZero no widen": fan-out ≠ 0 (por exemplo, mínimos quadrados) × α_grupo = 0. Não é necessário: o fan-out zero já recebe gradiente direto. Manter. |

## Adaptação para SPiFiL ("ReZero-SPiFiL", recomendado como o deepen padrão)
1. `deepen_rezero(at, kernel=3, act=..., norm=none|batch)`: os filtros da conv vêm do `NewFilterInit` (SPiFiL ajustado na saída do prefixo do pai até `at`, C → C). Escalar `α = 0` (`nn.Parameter(torch.zeros(()))`). Saída `x + α·act(norm(conv(x)))`.
2. Exato para **qualquer** entrada e **qualquer** ativação. Elimina a restrição `_nonneg_output` do `deepen_identity` e o bloco zerado do `deepen_residual_zero`.
3. Máscaras: conv, norm e α são **novos** (lr cheio; α no seu grupo de lr constante). Nada herdado muda pelo morfismo.
4. Validação: preservação medida por `finalize` (deve dar 0 bit a bit em eval() e em treino, porque α = 0 anula o ramo inteiro, inclusive o BN em treino). Registrar α ao fim do SSL: se α ficar perto de 0, o bloco não foi usado, o que é um diagnóstico útil.
5. O WiSE continua nosso: com âncora no pai, α_WiSE = 0 devolve α_ReZero = 0. Declarar que WiSE e gate não são do ReZero.
