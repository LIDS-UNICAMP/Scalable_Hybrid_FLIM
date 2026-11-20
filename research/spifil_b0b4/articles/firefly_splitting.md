# Firefly e Splitting Steepest Descent — regras de implementação × nossa cadeia

## Referências
- Liu, Q.; Wu, L.; Wang, D. *Splitting Steepest Descent for Growing Neural Architectures*. NeurIPS 2019, arXiv:1910.02366. Lido em https://ar5iv.labs.arxiv.org/html/1910.02366. Seções: §1, §2.1 (Eq. 2), Def. 2.1 (Eq. 3), Teorema 2.3, §2.2 (Algoritmo 1), Eq. 7, Ap. B.1–B.5.
- Wu, L.; Liu, B.; Stone, P.; Liu, Q. *Firefly Neural Architecture Descent: a General Approach for Growing Neural Networks*. NeurIPS 2020, arXiv:2102.08574. Lido em https://ar5iv.labs.arxiv.org/html/2102.08574. Seções: §2.1, §2.2 (vizinhança, Eqs. 4–5), §2.3 (camadas), §2.4 (aprendizado contínuo, Alg. 2), Algoritmo 1, Ap. B.1–B.4.
- (Lidos em 17/11/2026.)

## Operação

### Splitting Steepest Descent (Liu et al. 2019)
- **Split** (§2.1, Eq. 2): o neurônio `σ(θ,x)` vira `Σ_i w_i σ(θ_i, x)`, com `w_i > 0`, `Σ w_i = 1` e `‖θ_i − θ‖ ≤ ε`. Com todos os `θ_i = θ`, a função é idêntica.
- **Matriz de splitting** (Def. 2.1, Eq. 3): `S(θ) = E[Φ'(σ(θ,x)) ∇²_θθ σ(θ,x)]`. O índice de splitting é `λ_min(S)` e a direção é o autovetor `v_min`.
- **Split ótimo** (Teorema 2.3b): duas cópias com `w₁ = w₂ = 1/2` (o fan-out de cada cópia é a metade), `θ₁ = θ + ε v_min`, `θ₂ = θ − ε v_min`. A perda cai em `−ε² λ_min / 2`.
- **Critério**: se `λ_min > 0`, o neurônio é "splitting stable" e nenhum split infinitesimal ajuda (Teorema 2.3a). Só se divide com `λ_min < 0`, escolhendo os mais negativos (§2.2).
- **Quando**: "The splitting phase is triggered when no further improvement can be made by only updating parameters" (§1). Também: "introduce splitting only at stable local minima" (§2.1). **É um gatilho por estagnação.**
- **Algoritmo 1**: (1) treinar os parâmetros (SGD) até não melhorar mais; (2) calcular `S`, `λ_min` e `v_min` por neurônio; (3) escolher até `m*` neurônios com `λ_min ≤ λ* ≤ 0`; (4) dividir com passo ε; (5) voltar a (1).
- **Conv (Ap. B.3)**: SGD Nesterov 0,9, wd 1e-4, batch 64, lr 0,1, 160 épocas, ÷10 em 50% e 75%. **+30% de canais por fase**. A ReLU é trocada por Softplus para calcular `S`. ε e λ* não aparecem no texto lido.

### Firefly (Wu et al. 2020)
- **Vizinhança** (§2.2): `f_{ε,δ}(x) = Σ_{i≤m} ½[σ(x, θ_i + ε_i δ_i) + σ(x, θ_i − ε_i δ_i)] + Σ_{i=m+1}^{m+m'} ε_i σ(x, δ_i)`, com `|ε_i| ≤ ϵ`.
  - A 1ª soma é o **split** dos neurônios existentes.
  - A 2ª soma são os **neurônios novos**, com fan-in `δ_i` "far away from the existing neurons" (N(0, 0,1) nos Ap. B.1 e B.2) e **fan-out escalado por ε_i**, pequeno e não nulo. Fica perto do pai, mas **não é exato**.
- **Camadas novas** (§2.3): residual `(I + h_ℓ)` com `h_ℓ = Σ_i ε_{ℓi} σ(·, δ_{ℓi})`. Com todos os ε = 0, a camada é a identidade.
- **Passo 1** (Eq. 4): poucas iterações de gradiente em (ε, δ) sobre o conjunto **grande** de candidatos, com a esparsidade relaxada em penalidade. O número de passos não aparece no texto principal; no Ap. B.1 são 100 iterações e no B.3, 5 épocas.
- **Passo 2** (Eq. 5): score de Taylor `s_i ≈ ∇_{ε_i} L` (média de n ≈ 3 minibatches). Ficam os **η_t de maior |s_i|**, com `ε̂_i = −ϵ · 𝕀(|s_i| ≥ |s_(η_t)|) · sign(s_i)`.
- **Quando**: a descida de arquitetura "is called less frequently (e.g., only when a parametric local optimum is reached)" (§2.1). Também é **gatilho por estagnação**. No aprendizado contínuo (Alg. 2), cresce "while f_t can not solve task t sufficiently well".
- **Quanto**: VGG-19 cresce **+30% de neurônios por vez** (MobileNet, 35%), com m' = 50 neurônios novos candidatos por camada (Ap. B.2). Em profundidade, +25% de largura e camadas identidade em cada bloco.
- **Treino depois** (Ap. B.2): **160 épocas de finetune entre crescimentos** no VGG-19 (mais 1 época antes do score). Profundidade: 11 crescimentos × 40 épocas, depois 200 épocas. NAS: crescer a cada 10 épocas, 100 épocas no total. Otimizador e lr não aparecem no texto lido.
- **Comparações**: só com split, o Firefly equivale ao Splitting, mas sem o autoproblema e mais barato. O Net2Net "grows networks uniformly by randomly selecting the existing neurons". RandSearch serve de linha de base.
- **Aceitação**: nenhuma além da seleção por score. A rede é treinada até o próximo platô.

## Regras (checklist "como no Splitting / Firefly")
- [ ] **Gatilho por estagnação**: crescer só quando o treino do pai não melhora mais (ótimo local paramétrico).
- [ ] Split: duas cópias `θ ± ε v`, cada uma com fan-out × 1/2. A direção v vem de `v_min(S)` (Splitting) ou do gradiente de δ (Firefly).
- [ ] Neurônios novos (Firefly): pool grande, fan-out `ε_i` pequeno, poucos passos de gradiente em (ε, δ), score `|∂L/∂ε_i|`, top-η_t.
- [ ] Camada nova (Firefly): `I + Σ ε σ(·, δ)` com ε = 0.
- [ ] Cerca de +30% de neurônios por crescimento; finetune longo (por exemplo, 160 épocas) antes do próximo.
- [ ] Sem gate "≥ pai".

## Divergências

| Item | Artigo | Nosso | Ação |
|---|---|---|---|
| Quando crescer | **Só na estagnação** (ótimo local; `λ_min < 0`) | Por faixa, sem medir platô do pai | **Adicionar gate de estagnação**: crescer o pai só se Δκ_ORIG-val ≤ τ nas últimas N épocas de SSL, ou se a perda SSL estiver em platô. Registrar a curva do pai. |
| Onde crescer | Neurônios com `λ_min` mais negativo ou maior `|s_i|` | Família round-robin + DE θ (`plan_families`, growth_chain.py:889) | Usar o score de Taylor/gradiente por camada para escolher onde (ver Adaptação). |
| Split de herdado | `θ ± εv` e fan-out ÷ 2 | Não existe | **Conflita com a regra**: altera o filtro SPiFiL herdado (θ → θ + εv), e o backprop "desfaz", e a cópia não é SPiFiL. **Não implementar como padrão.** Variante declarada só com aval do usuário. |
| Neurônio novo: fan-in | Aleatório N(0, 0,1), depois otimizado por poucos passos | SPiFiL (`NewFilterInit`) | Manter SPiFiL. Os poucos passos em δ são compatíveis com "o backprop soma" se forem aplicados **só** aos novos. |
| Neurônio novo: fan-out | `ε_i` pequeno ≠ 0 (aproximado) | 0 exato | Manter 0. O score `∂L/∂ε` no ponto ε = 0 é calculável com o fan-out 0 (ver Adaptação). |
| Pool e seleção | m' candidatos ≫ η_t mantidos | Exatamente o nº de canais do plano, sem seleção | Gerar um pool SPiFiL ≥ 3× e manter o top-η_t por score. |
| Camada nova | `I + Σ ε σ(·, δ)`, ε = 0, δ aleatório | `deepen_residual_zero`: `x + act(conv x)` com **conv = 0** (morphisms.py:463) | Trocar por `x + Σ ε_i σ(SPiFiL_i(x))`, com fan-in SPiFiL e ε = 0 (o mesmo que `deepen_rezero` por canal; ver `rezero.md`). |
| Tamanho do passo | +30% por crescimento | Até a faixa seguinte (fator ≤ 2,5 por op; B0→B1 é até cerca de 2,7× params) | "Como no Firefly": incrementos de 30% com treino entre eles. A faixa é atingida em vários passos. |
| Treino entre crescimentos | Longo (160 épocas de finetune) | f15 (AE 5 + SSL 10) / f45 (AE 15 + SSL 30) | Nosso treino é cerca de 4 a 10× mais curto. Para fidelidade, treinar até o platô (que é também o gate do próximo crescimento). |
| Aceitação | Nenhuma | WiSE + κ ≥ pai | Nossa. Para fidelidade, registrar sem filtrar. |

## Adaptação para SPiFiL ("Firefly-SPiFiL")
1. **Gate de estagnação (Splitting §1, Firefly §2.1)**: só cresce o pai cuja curva κ_ORIG-val (ou perda LeJEPA) está em platô, por exemplo com Δκ ≤ 0,002 nas últimas 3 avaliações. Caso contrário, o pai é treinado mais e reentra. Isso torna o crescimento uma resposta à estagnação, como nos dois artigos.
2. **Pool SPiFiL e score de Taylor**: para a camada ℓ, gerar `m'` candidatos SPiFiL (≥ 3× o que será mantido). Acoplar cada candidato com fan-out 0 e um gate escalar `ε_i = 0`. Calcular `s_i = ∂L_SSL/∂ε_i` em ε = 0, que é `⟨∂L/∂z_{ℓ+1}, W_out,i h_i⟩`. Como o fan-out é zero, usar `‖∂L/∂W_out,i‖`, que coincide com o critério do GradMax-SPiFiL (`gradmax.md`). Manter os top-η_t.
3. **Poucos passos no pool (Passo 1)**: opcional, com gradiente **só** nos candidatos (fan-in SPiFiL ajustado e fan-out), herdados congelados, cerca de 1 época. Compatível com "o backprop soma, não desfaz".
4. **Onde**: a camada com maior `Σ_top |s_i|` recebe os canais, com cerca de 30% por crescimento.
5. **Split**: fora do padrão, porque modifica um filtro herdado. Se o usuário quiser "como no Splitting", a forma menos invasiva é manter o original intacto e acrescentar **um** filho `θ + εv_min` com fan-out 0 (sem o ÷2). Isso deixa de ser split, deixa de ser SPiFiL puro e precisa ser declarado.
