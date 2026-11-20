# Reinicialização de filtros (ReDo / RePr / Shrink-and-Perturb) — regras fiéis aos artigos

> Fontes lidas (17/11/2026): ReDo arXiv:2302.12902 (ar5iv) + código oficial `google/dopamine/dopamine/labs/redo/weight_recyclers.py`; RePr arXiv:1811.07275 (ar5iv); Shrink-and-Perturb arXiv:1910.08475 (ar5iv).
> Código comparado: `ga_search/modehb/init.py` L291–340 (`kernel_diagnostics`: `filter_redundancy`, `dead_fraction`), chamado por `run_search.py::_log_init_diag` (L1043–1060). Hoje o repo **só mede**; não existe reinicialização de filtros em `ga_search/` nem `flim/`.

---

## 1. Referência

| Sigla | Artigo | Onde |
|---|---|---|
| ReDo | Sokar, Agarwal, Castro, Evci. *The Dormant Neuron Phenomenon in Deep Reinforcement Learning*. ICML 2023 | arXiv:2302.12902 |
| RePr | Prakash, Storer, Florencio, Zhang. *RePr: Improved Training of Convolutional Filters*. CVPR 2019 | arXiv:1811.07275 |
| S&P | Ash & Adams. *On Warm-Starting Neural Network Training*. NeurIPS 2020 | arXiv:1910.08475 |

## 2. Método

**ReDo (Def. 3.1, Eq. 1; Alg. 1).** Score do neurônio i na camada ℓ:
s_i^ℓ = E_x|h_i^ℓ(x)| / ( (1/H^ℓ) Σ_k E_x|h_k^ℓ(x)| ). Neurônio é τ-dormente se s_i^ℓ ≤ τ.
Conv: o código oficial (`estimate_neuron_score`) reduz **batch e posições juntos** (`reduce_axes = range(ndim−1)`), `score = mean(|h|)`, `score /= mean(score) + 1e-9`.
A cada F passos: recicla dormentes — pesos de **entrada** re-sorteados da distribuição de init (código: Xavier uniform; bias de entrada zerado), pesos de **saída zerados** (a saída da rede não muda quando τ = 0), momentos do Adam (μ, ν) zerados nas posições recicladas (`reset_momentum`).

**RePr (Sec. 5, Eqs. 1–2; Alg. 1).** Por camada, filtros achatados (k·k·c) normalizados: Ŵ_ℓ = W_ℓ/‖W_ℓ‖ (por linha);
P_ℓ = |Ŵ_ℓŴ_ℓᵀ − I| (Eq. 1); O_ℓ^f = Σ_j P_ℓ[f, j] / J_ℓ (Eq. 2). Maior O ⇒ mais redundante ⇒ cai primeiro.
Ciclo: treina tudo S1 → ranqueia **globalmente** (todas as camadas juntas) → derruba os p% piores → treina a sub-rede S2 → reinicializa os derrubados **ortogonais** aos filtros restantes da mesma camada (QR / espaço nulo) → repete N ciclos.

**Shrink-and-Perturb (Sec. 4).** A cada rodada t (quando chegam dados novos): θ ← λ·θ + p, p = parâmetros de uma rede recém-inicializada **escalada** ("adding parameters from a scaled, randomly-initialized network"), 0 < λ < 1, em **todos** os parâmetros.

## 3. Receita oficial

| Item | ReDo | RePr | S&P |
|---|---|---|---|
| Critério | τ-dormente, Eq. 1; τ = 0.1 padrão (Sec. 5; 0.025 no DQN default, Table 1; 0 no SAC, Table 4); grade [0, 0.01, 0.1] | ortogonalidade inter-filtros, Eqs. 1–2; melhor que norma, Taylor, ativações, APoZ, gradiente, Hessiana (Fig. 3: Ortho 76.4 vs Weights 74.6 vs Activations 74.1 vs Standard 72.1, CIFAR-10 C³(32)) | — (todos os parâmetros) |
| Quanto | todos com s ≤ τ | **p = 30%** (ganho não significativo acima; instável > 50%) | todos |
| Quando | a cada **F = 1000** passos (DQN/DrQ); 200 000 (SAC); grade [1e3, 1e4, 1e5] | **S1 = 20, S2 = 10, N = 3** ciclos (Alg. 1, Sec. 8) | a cada rodada de dados novos |
| Amostra p/ score | minibatch 64 (DQN), 256 (SAC); {32, 64, 256, 1024} dão ~mesma fração (Ap. C.3) | só pesos (sem dados) | — |
| Reinit entrada | distribuição de init original (Xavier uniform no código); escalar pela norma média dos vivos ≈ igual (Fig. 23) | **ortogonal** aos filtros restantes (QR); com p grande, começar com valores pequenos (Apêndice) | λθ + ruído de init escalado |
| Reinit saída | **zero** (aleatório piora, Fig. 22) | kernels da camada seguinte que consomem o filtro derrubado → valores aleatórios pequenos (Apêndice) | — |
| Otimizador | zera μ, ν do Adam nas entradas/saídas recicladas (código) | reinicializar estado do Adam dos pesos derrubados + γ, β do BN (Sec. 8) | Adam lr 1e-3 com estado resetado a cada rodada (Sec. 2.2) |
| Valores | — | — | online: **λ = 0.6, ruído 0.01** (Fig. 7); pré-treino: **λ = 0.3, ruído 1e-4** (Fig. 9) |
| Controles do paper | reciclar aleatórios ou os de maior score colapsa (Fig. 15) | Random 73.4 < Ortho 76.4 | só shrink da última camada não ajuda (Fig. 14); só ruído ajuda menos (Table 4) |

## 4. Regras (checklist)

Critério "filtro ruim"
- [ ] **Dormente (ReDo):** s_i = mean_{b,h,w}|h_i| / mean_k(mean_{b,h,w}|h_k|) na **saída pós-ativação** do bloco; dormente se s_i ≤ τ, τ = 0.1 (registrar também τ = 0 e 0.025).
- [ ] Score em minibatch de **treino** (64 basta, Ap. C.3) — nunca validação/teste se o score decide reinicialização.
- [ ] **Redundante (RePr):** O_f = (1/J)·Σ_j |cos(ŵ_f, ŵ_j) − δ_fj| sobre filtros achatados; ranking **global** entre camadas; derrubar 30%.
- [ ] Constante (std ≈ 0) é outro critério (útil para SVM linear) — não chamá-lo de "dormente" sem dizer.

Quando
- [ ] ReDo: periódico, F passos (1000 no paper), durante todo o treino.
- [ ] RePr: S1 = 20, S2 = 10, N = 3 (épocas/iterações; o paper usa os dois termos — declarar a unidade).
- [ ] S&P: no início de cada nova rodada de dados/fase.

Como
- [ ] ReDo: entrada = init original; bias de entrada = 0; saída = 0; reset do Adam nas posições.
- [ ] RePr: novo filtro no espaço nulo dos restantes (QR); kernels consumidores na camada seguinte pequenos; reset do Adam e de γ, β do BN.
- [ ] S&P: θ ← λθ + p_init·escala, em todas as camadas.
- [ ] Nunca reciclar aleatoriamente nem os de maior score (ReDo Fig. 15).

## 5. Divergências

| Item | Artigo | Nosso | Ação |
|---|---|---|---|
| Critério de morto | ReDo: mean|h| normalizado ≤ τ (0.1) | `dead_fraction`: std por canal < 1e-8 (init.py L327–329) — mede canal **constante**, não dormente. Canal com |h| alto e constante conta como morto; canal com |h| pequeno mas variável não conta | Adicionar `dormant_fraction_tau{0,0.025,0.1}` com o score de ReDo (média sobre b,h,w de |h|, normalizada pela média da camada). Manter `dead_fraction` como `constant_fraction` |
| Critério de redundância | RePr: O_f por filtro (Eq. 2), ranking global, p = 30% | `filter_redundancy`: fração de **pares** com |cos| > 0.9 (init.py L313–314) — escalar agregado, limiar arbitrário, sem score por filtro | Adicionar `repr_ortho_score` por filtro (Eqs. 1–2) e `repr_mean`; manter a fração de pares só como diagnóstico |
| Redundância por covariância | RePr: ativações são critério pior (74.1 vs 76.4) | não existe | Se usar correlação/covariância de ativações (pedido do usuário), rotular como **variante nossa** (não RePr), e comparar com a Ortho de pesos |
| Dados do score | treino (minibatch) | `_log_init_diag` usa `self.val_x[:16]` (run_search L1050) | Ok enquanto for diagnóstico. Se virar critério de reinit, usar minibatch de **treino** (≥ 64 imagens) |
| Cobertura de convs | ReDo: todas as camadas | só a **primeira** `nn.Conv2d` do bloco (L1047) — inception/dwsep ficam parciais | Iterar sobre todas as convs do bloco |
| Momento da medição | ReDo: ao longo do treino; RePr: após S1 | só logo após o init (antes do AE/SSL) | Medir também após AE local e a cada F passos do SSL |
| Reinicialização | ReDo/RePr/S&P implementam | **inexistente** | Implementar `recycle_filters(block, next_block, criterion, mode)` — ver Adaptação |
| Saída zerada | ReDo: zera pesos de saída | — | Para o último bloco, "saída" = colunas da 1×1 do `EmbeddingHead`; para `CovReadout` (sem pesos) a função muda ⇒ re-ajustar a PCA após reciclar |
| Estado do otimizador | reset Adam (ReDo, RePr) | — | Zerar `exp_avg`/`exp_avg_sq` nas linhas recicladas (entrada) e colunas (saída) |

## 6. Adaptação (sempre com init SPiFiL)

Princípio do projeto: "backprop soma ao SPiFiL, nunca desfaz". As três técnicas são compatíveis desde que a **distribuição de init** seja o próprio SPiFiL e a saída seja preservada no instante da troca.

1. **ReDo-SPiFiL (dormentes):**
   - Score ReDo (τ = 0.1) em minibatch de treino, pós-ativação, a cada F passos de AE/SSL (F = 1000 do paper; com poucos passos por fidelidade, F = 1 vez por época).
   - **Entrada:** re-ajustar os filtros dormentes com SPiFiL sobre o prefixo congelado atual (`spifil_ops.fit_dense` com `ExactBudgetAllocator` pedindo só n_dormentes filtros), sorteando patches de superpixels **diferentes** dos já usados (outra semente / regiões onde o banco atual responde menos). Bias = −(K·μ_patch) (regra SPiFiL).
   - **Escala:** copiar a norma média dos filtros vivos (ReDo Fig. 23 diz que escalar pela norma média ≈ aleatório; é a mesma regra do `fill_random`, `flim/spifil.py` L944–950).
   - **Saída:** zerar os pesos da camada seguinte que leem esses canais ⇒ função idêntica no passo 0 (morfismo), o que respeita o princípio.
   - Resetar Adam nas posições.
2. **RePr-SPiFiL (redundantes):**
   - O_f (Eqs. 1–2) sobre os filtros SPiFiL; ranking global; p = 30% (começar com 10–20% em bancos estreitos, pois p é relativo ao total).
   - Candidatos SPiFiL novos (patches) projetados no **espaço nulo** dos filtros restantes (QR, como RePr); escolher os de maior resíduo (equivale a um FPS no complemento ortogonal) — direção vinda de dados, ortogonalidade vinda do paper.
   - Kernels consumidores na camada seguinte: pequenos (RePr Apêndice) ou zero (para preservar a função, preferível no nosso princípio — declarar a escolha).
   - Cronograma: S1 = 20 → derrubar → S2 = 10 → reinit → N = 3, na unidade de épocas do AE/SSL.
   - Proteção: com `protect=freeze/lowlr`, só filtros não protegidos (ou os novos) podem ser derrubados.
3. **Shrink-and-Perturb-SPiFiL (crescimento B0→B4 / herança):**
   - Ao herdar pesos para um filho (nova rodada de dados ou novo orçamento): θ ← λθ_herdado + ε·θ_SPiFiL_novo, com θ_SPiFiL_novo = SPiFiL re-ajustado no prefixo do filho (mesma forma), em vez de ruído gaussiano.
   - Valores do paper: λ = 0.6, ε = 0.01 (online) ou λ = 0.3, ε = 1e-4 (pré-treino). Para camadas SPiFiL protegidas usar λ próximo de 1 (não desfazer).
   - Sem BN, encolher todas as camadas por λ escala a saída por λ^L (Proposição 1): re-ajustar ZCA/PCA da leitura depois.
4. **Controles:** mesmo critério com reinit aleatório (He/Xavier, como no paper) × reinit SPiFiL; mesmo nº de filtros trocados; reciclar aleatórios como controle negativo (ReDo Fig. 15). Seleção por κ de validação; teste só relatório.
