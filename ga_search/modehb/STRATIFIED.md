# MO-DEHB estratificado por parâmetros (14/11/2026)

Liga-se no YAML: bloco `stratified:` de `ga_search/config/modehb/search_v3.yaml`. Sem o bloco, ou com `enabled: false`, o MO-DEHB antigo roda sem nenhuma mudança (há teste para isso).

O código novo está em `ga_search/modehb/strata.py`. As mudanças pontuais ficaram em `modehb.py`, `run_search.py` (passagem do YAML), `wandb_logger.py` (`log_strata`) e `pooling_head.py` (cabeça em pedaços).

## 1. Categorias

A contagem é o total de parâmetros do encoder **somado à projeção 1×1 final** (`EmbeddingHead`). Antes da avaliação vale a estimativa estrutural (`param_estimate`, montada no device `meta`). Depois da avaliação, a categoria é refeita com o valor **medido** (`result.params`).

| Categoria | Parâmetros |
|---|---|
| Tiny | ≤ 300k |
| Small | 300k – 800k |
| Medium | 800k – 1,5M |
| Huge | 1,5M – 3,5M |

- **Teto de 3,5M:** `enforce_cap` impede gerar redes acima dele. Se a medição passar de 3,5M, o candidato vira `failed/infeasible=param_cap`.
- **Outras métricas:** MACs/FLOPs e latência continuam sendo registrados. O objetivo da busca continua (1 − quality, FLOPs), com quality = 0,75·κ + 0,25·F1 do SVM na validação.

## 2. Evolução estratificada

- **Categoria-alvo em rodízio:** cada novo candidato tem uma categoria-alvo, sorteada em rodízio Tiny → Small → Medium → Huge. Isso garante exploração contínua das quatro.
- **DE (rand/1 + binomial):** o alvo é um membro da categoria-alvo. Os pais vêm da mesma categoria (70%) ou de qualquer uma (`cross_category_prob: 0.3`).
- **Projeção do filho:**
  - Com probabilidade `projection_prob: 0.7`, o filho é projetado para a categoria-alvo (`fit_to_stratum`). A projeção reescala todas as larguras por um fator único, mantendo a monotonia c1 ≤ c2 ≤ …, e adiciona camadas quando só a largura não alcança a faixa.
  - Nos outros 30%, o filho mantém a forma e é reclassificado.
- **Subpopulações por fidelidade, com diversidade:**
  - Um filho só compete dentro da própria categoria.
  - Uma categoria abaixo da cota justa (|subpop|/4) expulsa o pior membro da categoria mais populosa.
  - Assim, Tiny não toma a população.
- **Sementes:** 16 pirâmides projetadas em rodízio nas 4 categorias.
- **Inicialização:** só SPiFiL (`random_baseline: false`, decisão anterior do usuário). O gráfico SPiFiL × aleatório existe (`hv_strata/init_*`), mas o aleatório fica em 0 enquanto o baseline estiver desligado.

## 3. Promoção protegida

Vale nos brackets com successive halving (`protected_brackets: 5`):

- **Cota por categoria:** cada categoria com candidatos válidos recebe pelo menos ⌊25%·n⌋ das vagas (mínimo 1), escolhidas por ordenação não dominada e crowding **dentro da categoria**. As vagas restantes saem da seleção global.
- **Poucas vagas:** quando há menos vagas que categorias, as categorias menos promovidas até então têm prioridade (`promo_counts`).
- **Promoção exploratória:** `explore_extra: 1` vaga extra por rung para o Medium/Huge com maior ganho de quality por log2(época) entre as duas últimas fidelidades (`learning_slope > 0`).
- **Depois dos brackets protegidos:** a promoção volta a ser global. A frente final e o hypervolume principal são globais, só na fidelidade máxima, sem favorecer categoria nenhuma.

## 4. Orçamento adaptativo

- **Fidelidades:** 5 / 15 / 45 / 100 / 200 épocas (usuário pediu de volta em 14/11; a primeira versão usava 5/20/60/120), com AE:SSL = 1:2 em cada uma. Promoções retomam o checkpoint anterior.
- **Calibração (`calibration: true`):** antes do bracket 0, uma semente de cada categoria passa por todas as fidelidades (5 → 200). O resultado fica em `state.calibration`: curva quality × épocas, parâmetros medidos e época de saturação (primeiro ponto em que o ganho até o seguinte é < 0,01).
- **Extensão:** desligada (`extension_fidelities: []`), porque 200 já é a fidelidade máxima; ligar = pôr fidelidades acima de 200 na lista.
  - Extensões não entram nas subpopulações nem no hypervolume principal; aparecem só nas curvas.
  - O protocolo de comparação global é a fidelidade de 120 épocas, igual para todos.

## 5. Reparos de viabilidade (achados ao lançar)

1. **Cabeça 1×1 em pedaços** (`EmbeddingHead.forward`):
   - **Problema:** a projeção para 1280 canais em resolução cheia, com 256 imagens (64 × 4 vistas do LeJEPA), montava mapas de 16 a 43 GB. Até a rede Tiny pedia 66 GB.
   - **Correção:** a cabeça roda por pedaços do batch, com checkpoint no backward. A conta é idêntica (testado: diferença de 0 na saída e menor que 1e-13 no gradiente). A rede Tiny caiu para 0,2 GB.
2. **Reparo de memória** (`mem_repair`, `memory_budget_gb: 100` na escala da estimativa):
   - **Medição:** custo por canal em 224² de ~0,22 GB (conv), 0,36 (dwsep) e 0,75 (inception).
   - **Correção:** quando a estimativa passa do limite, o reparo põe max-pooling na primeira camada que ainda não reduz a resolução. Se isso não basta, estreita as camadas de alta resolução.
   - **Efeito:** as redes grandes crescem depois de reduzir a resolução, como um stem de CNN.
   - **Calibração da estimativa:** é cerca de 2× conservadora; 97 estimados correspondem a 53 GB medidos.
3. **Amostras do SPiFiL** (`spifil_images_repair`):
   - **Problema:** o SPiFiL exige N = 100 superpixels × classes × imagens/classe maior que D = cin·k².
   - **Correção:** o gene `spifil_images_per_class` sobe até a menor escolha que satisfaz essa condição. Antes, essas redes falhavam com `BudgetExhausted`.

## 6. W&B (run controller, `strata_*`)

Cada gráfico pedido e onde ele está:

1. **Qualidade × épocas por categoria:** `strata_plots/quality_vs_epochs` e `strata_plots/calibration`.
2. **HV × GPU-horas:** `hv_strata/global` com eixo x `gpu_hours`.
3. **HV × avaliações:** `hv_strata/global` com eixo x `n_evaluations`.
4. **HV global e por categoria:** `hv_strata/{global,tiny,small,medium,huge}`.
5. **Candidatos por categoria:** `strata/<cat>/n_candidates` e `strata/<cat>/n_evals_f<f>`.
6. **Taxa de promoção:** `strata/<cat>/promotion_rate_f<f>`.
7. **Sobrevivência:** `strata/<cat>/survival_rate` (entrou em 5 épocas e chegou a 120).
8. **Parâmetros × qualidade:** `strata_plots/params_vs_quality`.
9. **Evolução da frente de Pareto:** `strata_plots/pareto_evolution` (acumulada por bracket).
10. **Ganho marginal:** `strata_plots/marginal_gain` (Δquality por 100k parâmetros ao longo da frente).
11. **Mutações e cruzamentos entre categorias:** `strata_plots/category_operations` (`ops/<pai>-><filho>`, `ops/pool_same|cross`, promoções, promoções exploratórias, extensões).
12. **SPiFiL × aleatório:** `hv_strata/init_spifil` e `hv_strata/init_random`.

Os objetivos são normalizados e o ponto de referência é fixo (1,1; 1,1). O HV principal usa só a fidelidade máxima.

## 7. Validação (antes de lançar)

`tests/modehb/test_strata.py`: 14 testes, usando um avaliador sintético em que redes grandes começam piores e continuam melhorando. A suíte inteira tem **2205 testes, todos passando**.

| Item pedido | Teste |
|---|---|
| classificação por parâmetros | `test_classification_bounds`, `test_strata_must_be_contiguous` |
| restrições das 4 categorias | `test_fit_to_stratum_hits_each_category`, `test_cap_never_exceeded`, `test_measured_above_cap_is_infeasible` |
| mutação e crossover dentro e entre categorias | `test_mutation_within_and_across_categories` |
| promoção protegida de Medium/Huge | `test_protected_promotion_quota_and_exploration`, `test_promotion_counts_rotate_when_few_slots` |
| retomada de checkpoint | `test_resume_replays_identically`, `test_calibration_and_extension` (extensões retomam o checkpoint) |
| sem favorecimento de modelos pequenos | `test_all_categories_explored_and_reach_max_fidelity`, `test_insert_keeps_categories`, `test_protected_promotion_quota_and_exploration` (sem estratificação, as 8 vagas iriam para Tiny; com ela, ≥ 2 por categoria) |
| integridade de Pareto e HV | `test_extension_excluded_from_hv_and_front` |
| modo desligado = MO-DEHB original | `test_disabled_is_original_behaviour` |

**Simulação** (avaliador sintético, 10 brackets, 219 avaliações): todas as categorias chegam a 120 épocas (Tiny 7, Small 5, Medium 7, Huge 5). O maior número de parâmetros gerado foi 3,21M, abaixo do teto. A retomada reproduz os 219 jobs com o mesmo hash.

**GPU real** (remote_sensing, 5 épocas, sementes da busca nova):

| Categoria | Parâmetros | Pico GPU | κ validação |
|---|---|---|---|
| Huge (inc34 → conv335 → dwsep1024 ×2) | 2,61M | 27,8 GB | 0,383 |
| Huge (inc34 → inc171 → inc655) | 1,70M | 52,6 GB | 0,448 |
| Medium (conv139 → inc175 → inc263) | 0,90M | 53,1 GB | 0,435 |
| Medium (conv75 → conv150 → inc299) | 0,85M | 28,8 GB | 0,464 |
| Small (inc21 → dwsep162 → inc254) | 0,39M | 17,3 GB | 0,522 |
| Tiny (conv8 → dwsep16) | 23k | 0,2 GB | — |

## 8. Execução

- **Sessão tmux:** `modehb_strat`.
- **Comando:** `python -m ga_search.modehb.run_search --config ga_search/config/modehb/search_v3.yaml --dataset <d> --seed 42 --set output_dir=artifacts/modehb_strat/<d>/state/seed42`.
- **Fila de bases:** remote_sensing → flowers → larvae → ham10000 → aid → eggs → coconut_trees → protozoan.
- **Estados preservados:**
  - `artifacts/modehb_spifil/` (busca anterior; remote_sensing completo, flowers parcial);
  - `artifacts/modehb_strat_oom_1347/` (primeira tentativa, que falhou por falta de memória antes dos reparos).

## 9. Auditoria de falhas (14/11, 4 agentes): zero falhas como meta

Antes das correções, a busca anterior (`modehb_spifil/remote_sensing`) perdeu **127 de 315 redes novas (40%)**. As causas, e o que foi feito para cada uma:

| Causa | Casos | Correção |
|---|---|---|
| Falta de memória: cabeça 1×1 em resolução cheia, camadas largas em 224² e ZCA que guardava todos os mapas | 59 + 3 | cabeça em pedaços; `mem_repair` com modelo de memória por fase, em GB reais (SSL por tipo lejepa/simclr/ae, AE local, SoftHebb, SPiFiL N²) e orçamento de 60 GB; ZCA e SoftHebb processados em blocos, com resultado idêntico |
| SPiFiL com N ≤ D | 43 + 1 | D real por operação (a inception sempre ajusta um ramo 5×5); contagem real de doadores por classe; se nem 32 imagens/classe bastam, a repair reduz o kernel ou estreita a entrada (`spifil_images_repair`) |
| LeJEPA sem regularizador | 14 | `repair` troca para sigreg |
| NaN/SVD/eigh em camadas de 1–7 canais | 13 | filtro único do SPiFiL estimado com estatísticas de todas as sementes (antes saía zerado); ZCA/DBN estáveis (`safe_eigh`, `_InvSqrtPSD`); passo com NaN é pulado, não derruba a rede |
| Validação cortada nas primeiras 512 imagens, ordenadas por classe | todas as bases com val > 512 | `val_max_samples: null`: validação inteira (no coconut só a classe 1 era avaliada) |

**Validação estática:** em cada uma das 8 bases, as 16 sementes mais 300 redes sorteadas têm memória estimada ≤ 60 GB e nenhuma fica inviável para o SPiFiL. As 4 categorias saem balanceadas.

## 10. Semente externa + grade de crescimento (buscas SEMEADAS, 15/11)

- **YAML:** `search_v3_seeded.yaml` e `search_v3_lemonade_seeded.yaml` (= os atuais + bloco `external_seed:` + estratos small ≤0,8M / medium ≤1,5M / large ≤3M / xl ≤5,5M, teto 5,5M). Saída em `artifacts/modehb_seeded/<base>` e `artifacts/modehb_lemonade_seeded/<base>`. Sem o bloco (ou arquivo ausente para a base: aviso) a busca é idêntica à de antes (`test_seed_off_genome_sequence_identical_to_before_patch`).
- **Conversão** (`external_seed.py`, `load_external_seed(base)`): o encoder "SPiFiL pequeno + LeJEPA" (FLIM `conv k3 + relu + MaxPool2d(3, s2)`) vira `SpifilOpsBlock`s com os pesos do checkpoint LeJEPA (projetor/decoder descartados). Única op nova: `pool_padding: 0` (o pool FLIM não tem padding; o nosso tem). Igualdade testada contra o código original do repositório discovery (diferença máxima 0).
- **Sementes do MO-DEHB:** semente pura + grade de crescimento que preserva a função (largura ×{0,75 … 2,5}, +{0,1,2} camadas identidade no fim, variante com bloco residual init zero/silu); acima do teto, descartadas. Vêm primeiro na população inicial (representantes da calibração dos seus estratos). Filhos do DE são genomas normais.
- **Pesos:** filtros mantidos = do encoder; filtros NOVOS = SPiFiL (alocador de largura exata) + AE local; colunas do consumidor para canais novos = 0 (Net2Wider); corte = remove os filtros menos usados pelo consumidor, sem re-treinar. `protect` (lowlr/freeze, alternado nas sementes) vale só para os elementos herdados (`ElementProtect`, exato para Adam/AdamW); os novos treinam livres. Âncora WiSE = rede crescida (+AE), então α=0 recupera a semente.
- **LEMONADE:** as sementes externas também são pais (descendentes iniciais com operador em rodízio); `lemonade.function_preserving: true` faz widen/deepen/kernel herdarem preservando a função (`grow_blocks`); camadas que não dá para mapear (ex.: residual numa camada herdada) voltam ao SPiFiL normal.
