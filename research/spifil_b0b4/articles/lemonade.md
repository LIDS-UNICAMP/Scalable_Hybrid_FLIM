# LEMONADE — regras de implementação × nossa cadeia

## Referência
Elsken, T.; Metzen, J. H.; Hutter, F. *Efficient Multi-objective Neural Architecture Search via Lamarckian Evolution*. ICLR 2019, arXiv:1804.09081.
Lido em https://ar5iv.labs.arxiv.org/html/1804.09081 (17/11/2026). Seções usadas: §3.1 (morfismos), §3.2 (ANM, Eq. 9), §4 (Algoritmo 1, Eqs. 1–2), §5.1, Ap. A.1.1 (Tipos I–IV, Eqs. 3–8), A.1.2 (ANM), A.2, A.3.1, A.3.4 (treino), Fig. 2 e Fig. 6 (ablações).

## Operação

### Morfismos exatos (NM, Eq. 3: `N^w(x) = (TN)^w̃(x)` ∀x; §3.1 e Ap. A.1.1)
- **Inserir Conv-BN-ReLU** (Tipo I, Eq. 4, `A = 1, b = 0`, "Net2DeeperNet"): a conv é identidade. "Offset and scale of BatchNormalization are initialized to be the (moving) batch mean and (moving) batch variance", ou seja, o BN desfaz a normalização com as estatísticas reais. A ReLU é idempotente (Tipo III, Eq. 7).
- **Aumentar filtros** (Tipo II, Eq. 6): a conv seguinte tem os parâmetros "padded with 0's", isto é, **fan-out = 0**. O fan-in dos filtros novos é livre (o artigo não especifica ruído). Net2WiderNet aparece como alternativa.
- **Adicionar skip**: (i) por **concatenação**, com zero-padding nas convs seguintes; ou (ii) por **adição** como combinação convexa com λ aprendível. Em §3.1, λ é "initialized as 0", de modo que a saída começa no caminho original. O Ap. A.1.1 (Tipo IV, Eq. 8) diz λ = 1 para a mesma ideia. A inconsistência é do artigo, mas nos dois casos o ramo novo começa com peso 0.
- **Ruído**: o artigo não menciona.

### Morfismos aproximados (ANM, §3.2, Eq. 9 e Ap. A.1.2)
- Remover uma camada ou skip, podar 1/2 ou 1/4 dos filtros de uma conv, trocar conv por separável. Os pesos são ajustados por **destilação** (entropia cruzada): as camadas não afetadas são herdadas e **congeladas**, e as afetadas treinam **5 épocas**.

### Onde e quando (Algoritmo 1, §4)
- KDE sobre o objetivo **barato** (params, mult-adds) da população. O pai é amostrado com `p(N) ∝ 1/p_KDE(f_cheap(N))` (Eq. 1), favorecendo regiões esparsas da frente.
- Geram-se `n_pc` filhos com operadores **sorteados uniformemente** (§5.1), de 1 a 3 operadores por filho (Ap. A.3.1). Desses, `n_ac` são **aceitos antes de treinar**, com `p_child ∝ 1/p_KDE(f_cheap(N^c))` (Eq. 2). Os valores de `n_pc` e `n_ac` não são dados.
- Não há critério de estagnação. O pai é um membro treinado da frente (herança lamarckiana). 100 gerações.

## Receita (Ap. A.3.4)
- Filhos durante a busca: **20 épocas, SGD com cosine annealing, lr 0,01 → 0**, batch 64, weight decay 5·10⁻⁴, aumentos padrão + mixup + Cutout, 45k treino e 5k validação. Treina-se **toda** a rede, sem lr diferente para herdados.
- ANM: 5 épocas de destilação nas camadas afetadas (as demais congeladas).
- Avaliação final: do zero, 600 épocas, lr 0,025.
- **Aceitação e seleção**: nova população = `ParetoFront(P ∪ filhos, 𝔣)` sobre os objetivos caros (erro de validação) e baratos (params, mult-adds, latência). **Não há exigência "filho ≥ pai"**: um filho entra se não for dominado. Fig. 2: modelos grandes "are discarded in a later generation as smaller, better ones are discovered".
- Estagnação: "no significant changes in the Pareto front after approximately 24 GPU days" de 56 (Ap. A.3.4). O Ap. A.2.1 controla o ganho que vem só de "treinar mais" usando os modelos da geração 1. Ablação (Fig. 6): tirar ANM, lamarckismo ou KDE piora o resultado.

## Regras (checklist "como no LEMONADE")
- [ ] Operadores NM: inserir Conv-BN-ReLU identidade com BN calibrado nas estatísticas móveis; aumentar filtros com fan-out 0; skip por concatenação com zero-pad ou por adição convexa com λ = 0.
- [ ] Operadores ANM (encolher) com destilação de 5 épocas e não afetados congelados.
- [ ] Pais amostrados por `1/p_KDE(f_cheap)`; filhos aceitos **antes do treino** por `1/p_KDE(f_cheap)`.
- [ ] 1 a 3 operadores sorteados uniformemente por filho.
- [ ] Filho: 20 épocas, SGD + cosseno 0,01→0, wd 5e-4, rede inteira.
- [ ] População = frente de Pareto (P ∪ filhos), sem gate "≥ pai".

## Divergências

| Item | Artigo | Nosso | Ação |
|---|---|---|---|
| Aumentar filtros | Fan-out 0; fan-in livre | Fan-out 0 (`_remap_conv`); fan-in SPiFiL (growth_chain.py:642) | **Igual em estrutura**. O SPiFiL é a nossa instância do "fan-in livre". |
| Inserir Conv-BN-ReLU | Identidade; BN = média/variância móveis reais | `deepen_identity`: BN rm=0, rv=1, γ=√(1+eps), exato só em eval() (morphisms.py:450-455) | Calibrar o BN com estatísticas de dados reais. A identidade não é SPiFiL: ver `net2net.md`, Adaptação (b). |
| Skip por adição | Convexa `(1−λ)x + λy`, λ aprendível, ramo novo com peso 0 | `add_skip_zero`: `y + λx` com λ = 0 (gated, cin = cout) ou projeção 1×1 = 0 (morphisms.py:491) | Equivalentes em função. Para fidelidade literal, a parametrização convexa. Não muda a ciência. |
| Skip por concatenação | Sim (zero-pad no consumidor) | **Não existe** | Implementar `add_concat_skip` (ver `densenet.md`). |
| ANM (encolher) | Sim, destilação de 5 épocas | Não (só exatos que crescem; teto 5,5M) | Fora do escopo B0→B4. Registrar a ausência. A ablação do artigo (Fig. 6) mostra que ANM ajuda. |
| Escolha do pai | `∝ 1/p_KDE(f_cheap)` | Frente de Pareto (κ ORIG-val × FLOPs), `n_parents=4`, round-robin (growth_chain.py:973) | "Como no LEMONADE": amostrar o pai por KDE inversa em params ou FLOPs. |
| Operadores do filho | 1–3 sorteados uniformemente | Família fixa (round-robin) + widens encadeados até a faixa (`plan_with_family`, growth_chain.py:911), fator ≤ 2,5 por op, até 12 ops | O nosso salto de faixa (B_k→B_{k+1}) é muito maior que o passo LEMONADE (1–3 ops pequenas). Para ficar fiel: filhos de 1–3 ops sem obrigação de mudar de faixa. |
| Aceitação pré-treino | `n_ac` de `n_pc` por `1/p_KDE` | Triagem por fidelidade (f15 → 1/η → f45), pós-treino | Divergência de desenho (MO-DEHB). Registrar. |
| Treino do filho | 20 ép. SGD cosseno 0,01→0, wd 5e-4, rede inteira | AE local (Adam 1e-3) + LeJEPA/SIGReg (AdamW 1e-3 fixo, wd 0); herdados 0,01× | "Como no LEMONADE": um único SSL, cronograma cosseno até 0, `protect: full`, wd do pai. |
| WiSE | Não existe | α ∈ {0,…,1} pela ORIG-val, âncora no pai | Nosso. Para fidelidade, `wise_alphas: [1.0]`. |
| Aceitação | Não dominado (Pareto P ∪ filhos) | κ_filho ≥ κ_pai re-pontuado (`accept`, growth_chain.py:1344) | "Como no LEMONADE": aceitar por não dominância (κ, FLOPs) sobre a população inteira, sem gate por pai. Um filho pior que o pai e mais barato pode entrar. |
| Estagnação | Monitora a frente (sem mudança após cerca de 24 GPU-dias); controle "geração 1 treinada mais" | Nada | Adicionar o controle do Ap. A.2.1: o pai treinado pelas mesmas épocas extras, sem crescer. Distingue o ganho de crescer do ganho de treinar mais. |

## Adaptação para SPiFiL ("LEMONADE-SPiFiL")
1. Aumentar filtros: fan-in SPiFiL e fan-out 0. Já é o Tipo II do artigo.
2. Inserir camada: Conv identidade com BN calibrado (só como operador estrutural) **mais** k filtros SPiFiL de fan-out 0; ou `deepen_rezero` com SPiFiL (`rezero.md`).
3. Skip: manter `add_skip_zero` e adicionar `add_concat_skip`, com os filtros novos do bloco concatenado em SPiFiL.
4. Seleção fiel: KDE inversa no custo para escolher o pai, Pareto P ∪ filhos para a população e nenhum gate por pai. Se quisermos manter "cada faixa melhor que a anterior", isso é **regra nossa adicional** e deve ser declarado como "LEMONADE + gate κ ≥ pai".
5. Controle obrigatório do Ap. A.2.1: o pai com as mesmas épocas extras de SSL e sem crescer. Só assim o Δκ é atribuível ao crescimento.
6. Treino: os herdados a lowlr seguem a nossa regra (o backprop soma). O artigo treina tudo. Declarar `protect: lowlr` como desvio.
