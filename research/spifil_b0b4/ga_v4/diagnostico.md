# GA v3 preso em modelos pequenos: diagnóstico e correção (v4)

Data: 18/11/2026, Jaci. Fontes: `artifacts/{modehb_hybrid,modehb_lemonade,modehb_lemonade_seeded}/<base>/state/seed42/search_state.json`. Bases: flowers, remote_sensing, ham10000, aid e coconut_trees. `modehb_seeded` ainda não tem estado. O script e o JSON bruto estão no scratchpad da sessão (`diag/diag.py`, `diag/diag.json`).

## Resumo

O GA v3 não deixa de tentar os grandes porque eles falham: ele quase não os gera e quase não os promove.
1. **Teto de 3,5M.** A faixa "huge" vai de 1,5M a 3,5M. A faixa B4 (3–5,5M) praticamente não existe: aparecem de 0 a 2 candidatos por run.
2. **Cotas iguais.** O round-robin dá a tiny o mesmo número de filhos que dá a huge.
3. **FLOPs global.** Na promoção, as vagas que sobram, e todas as vagas depois do bracket 5, vêm da frente global κ×FLOPs, e essa frente é dominada por tiny/small.
4. **Memória superestimada (~1,6×).** Com o budget de 15–24 GB, o `mem_repair` comprime a faixa grande para baixo.
5. **Falhas são raras.** Foram 3 em cerca de 900 jobs. O OOM espera 10×300 s; quando esgota, o candidato vira FAIL_OBJ e ocupa uma vaga da subpopulação.

## 1. Distribuição dos params (jobs de busca)

| run | bracket | GPU-h | jobs | cand. únicos | cand. > 1,5M | params p10/p50/p90/máx | memory_budget |
|---|---|---|---|---|---|---|---|
| hybrid/flowers | 0 | 12,6 | 47 | 27 | 6 | 21k / 684k / 1,89M / 3,40M | 20 |
| hybrid/remote_sensing | 0 | 10,8 | 47 | 27 | 6 | 20k / 795k / 1,93M / 2,95M | 60 |
| hybrid/ham10000 | 1 | 36,7 | 147 | 83 | 24 | 20k / 752k / 1,97M / 3,22M | 15 |
| hybrid/aid | 0 | 35,4 | 135 | 81 | 17 | 23k / 612k / 1,83M / 3,35M | 15 |
| hybrid/coconut_trees | 0 | 23,2 | 39 | 19 | 3 | 22k / 552k / 2,30M / 2,30M | 34 |
| lemonade/flowers | 0 | 13,5 | 37 | 17 | 4 | 134k / 791k / 2,18M / 2,33M | 12 |
| lemonade/remote_sensing | 3 (fim) | 28,8 | 208 | 121 | 32 | 30k / 772k / 2,01M / 3,15M | 20 |
| lemonade/ham10000 | 0 | 31,3 | 143 | 79 | 24 | 18k / 847k / 2,14M / 3,21M | 15 |
| lemonade/aid | 0 | 0,3 | 2 | 1 | 0 | 148k | 15 |
| lemonade/coconut_trees | 0 | 13,8 | 24 | 5 | 1 | 201k / 968k / 2,22M | 12 |
| lemonade_seeded/remote_sensing | 0 | 0,2 | 4 | 1 | 0 | 163k | 24 |

**Faixas B0–B4 da missão, na fidelidade 5** (rung 0):

| run | B0 | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|
| hybrid/ham10000 | 22 | 24 | 15 | 22 | 2 |
| hybrid/aid | 25 | 21 | 21 | 16 | 2 |
| hybrid/flowers | 8 | 8 | 8 | 6 | 1 |
| lemonade/remote_sensing | 24 | 19 | 20 | 21 | 1 |
| lemonade/ham10000 | 24 | 17 | 18 | 24 | 2 |

Dentro da faixa "huge" (1,5–3,5M), a mediana medida fica entre 1,94M e 2,19M. Acima de 2,5M aparecem só 0–8 candidatos por run.

**Por bracket:** quase todos os runs ainda estão no **bracket 0**, mesmo depois de 12–37 GPU-h. Com fid 5 e eta 3, o rung 0 tem 81 configurações, preenchidas com amostras aleatórias. O DE praticamente não rodou: há 1 job DE no ham hybrid e 16 no RS lemonade. Sobre os outros operadores:
- os filhos "random" e de "morph" se dividem igualmente entre as 4 faixas;
- a projeção acerta: cerca de 90% dos filhos medidos caem na faixa-alvo. Exemplo: no hybrid/aid, o alvo huge resultou em 22 huge e 3 medium.

## 2. Falhas (failed/infeasible)

| run | cand. | fid | params_est | motivo |
|---|---|---|---|---|
| hybrid/aid | c00051 | 5 | 1,48M | OOM no fit do SPiFiL (GPU compartilhada, 0,5 GB livres) |
| hybrid/aid | c00054 | 5 | 622k | OOM (GPU compartilhada) |
| hybrid/coconut_trees | c00012 | 5 | 3,31M (est.) | param_cap: 3,54M medidos > 3,5M |
| lemonade/remote_sensing | c00104 | 15 | 222k | OOM (GPU compartilhada) |

Há também descartes silenciosos, que não viram job. O `enforce_cap` encolhe qualquer filho acima de 3,5M. O `mem_repair` acrescenta pooling cedo ou estreita as camadas quando a estimativa passa do budget. O `external_seed` descarta a variante quando a memória estimada passa do budget; por exemplo, a `seed_B_wide` de 5,3M foi encolhida para huge.

## 3. O que empurra a busca para o pequeno

- **Promoção protegida** (`strata.stratified_promotions`). A cota por faixa é `max(1, floor(0,25·n))`. Com n=3 (45→100) ou n=1 (100→200), uma faixa fica sem vaga. As vagas extras vêm da frente GLOBAL (1−κ, FLOPs), em que o modelo grande perde no FLOPs. A partir de `protected_brackets=5`, toda a promoção passa a ser global. Frentes globais medidas:
  - lemonade/RS: @200 = 2 tiny + 1 small, @45 = 5 tiny + 2 huge + 1 medium;
  - hybrid/RS: tiny domina em todas as fidelidades;
  - ham hybrid: @200 = 1 tiny + 1 small.
- **DE.** O pool de pais cross-category (prob. 0,3) é escolhido pela frente global (FLOPs), e `projection_prob=0,7` deixa 30% dos filhos com o tamanho do pai. Os dois efeitos herdam o viés para o pequeno.
- **Subpopulação.** Um candidato que falha entra com FAIL_OBJ enquanto a subpopulação não está cheia, e pode ser escolhido como alvo do DE (`_next_target` não filtra o status).
- **A qualidade não justifica o viés.** Na fidelidade 5, o melhor κ é de huge ou medium em 8 de 9 runs:

  | base | huge | tiny |
  |---|---|---|
  | aid | 0,845 | 0,819 |
  | flowers | 0,763 | 0,687 |
  | coconut | 0,790 | 0,714 |
  | RS lemonade | 0,611 | 0,549 |

  O grande perde só porque o FLOPs entra na comparação global.

## 4. Teto, budget de memória e estimador

- `param_cap` é 3,5M em search_v3/lemonade (5,5M só nos `*_seeded`). Os runs usam `memory_budget_gb` de 12 a 34 (via `--set`), e o YAML traz 60.
- **O estimador superestima.** Comparando `strata.mem_estimate_gb` (com margem 1,15) com o `peak_gpu_mem_gb` medido em 1699 jobs, a razão pico/estimativa fica em 0,62 na mediana, 0,77 no p90, 1,11 no p99 e 1,82 no máximo. Restringindo aos modelos > 1,5M (421 jobs): 0,61 / 0,75 / 1,08 / 1,67. Com budget 15, o budget real efetivo fica em ~9,7 GB; o maior pico medido no ham foi 13,4 GB.
- **Simulação do `_decode`** (ham10000, 12 filhos aleatórios projetados por faixa). A projeção acerta 12/12 em todos os casos; o budget só desloca a mediana para baixo:

  | teto | faixa | budget 15 GB | budget 60 GB |
  |---|---|---|---|
  | 3,5M | huge | 2,01M | 2,78M |
  | 5,5M | B4 | 4,19M | 4,65M |

  O budget comprime a faixa e força pooling cedo, mas não impede a busca de chegar às faixas grandes.

## 5. Fitness

A v3 calcula `readout: cov_head` + StandardScaler, enquanto o número reportado vem da SVM original sem scaler sobre so2/GAP. A correlação entre os dois é ρ 0,09 (diagnóstico anterior). Na v4, a fitness passa a ser a **mesma leitura reportada**: `so2_4096_pca` de `readout_x`, seguida da SVM original (linear, C=100, ovo, sem scaler).

## 6. Correção v4 (`ga_v4.diff` + `search_v4.yaml`)

| problema | v3 | v4 |
|---|---|---|
| faixas/teto | tiny/small/medium/huge até 3,5M | **B0–B4** (300k/800k/1,5M/3M/5,5M), teto duro 5,5M |
| quota de geração | round-robin igual | `band_quota`: a faixa-alvo é a mais sub-representada NO BRACKET (contagem/peso, com pesos B0 1, B1 1, B2 1,25, B3 1,5, B4 1,75); filho que falhou ou caiu em outra faixa não conta, então a faixa volta a ser alvo |
| projeção | 0,7 | 0,95; `cross_category_prob` 0,15 |
| pais do DE | frente κ×FLOPs | `parent_select: quality` |
| promoção | cota 25% + resto pela frente global; global depois do bracket 5 | `promotion: per_band` em todos os brackets: vagas proporcionais aos pesos e ≥ 1 por faixa (`promote_min_per_band`); Pareto (κ, FLOPs) DENTRO da faixa; sobras pela qualidade |
| OOM | 10×300 s, depois FAIL_OBJ na subpop | 1 re-tentativa (60 s), depois a escada **bf16 → bf16+batch/2 → bf16+batch/4**, registrada em `result.oom_attempts` / `result.mem_fallback`; se esgotar, vira `infeasible=oom, penalize=False`, não entra na subpop e não conta na quota (a faixa é re-tentada) |
| memória | estimativa ×1,0 | `mem_scale: 0.8` (p90 medido 0,77); budget por GPU em `fila_v4.json` |
| fitness | cov_head + StandardScaler | `readout: so2_4096` + SVM C=100 **sem scaler** (`ga_search/modehb/so2_readout.py`, igual à referência; testado) |
| calibração | 1 representante por faixa × 5 fidelidades | desligada (com B4@200 consumia cerca de 1/3 do bracket 0) |

Tudo é opt-in em `stratified:`, com defaults que mantêm o comportamento da v3. As buscas v3 em andamento não mudam. Uma limitação: o **gradient checkpointing não foi implementado**. A escada usa bf16 e batch dividido (`_SplitLoader`; o efeito de batch menor fica registrado no resultado).

## 7. Testes

- **Simulação sintética** (`scratchpad/diag/sim_ga.py`): avaliador com κ = 0,40 + 0,06·log10(params/1e4) + ruído N(0; 0,02); OOM quando o pico real (0,62 × estimativa) passa de 9 GB; o bf16 reduz o pico a 0,6× e o batch/d a d^-0,7. Rodada com remote_sensing, 3 brackets, budget 24 GB e a escala completa do rung 0:

| | v3 (search_v3.yaml) | v4 (search_v4.yaml) |
|---|---|---|
| gerados no bracket 0 (B0/B1/B2/B3/B4) | 10/7/4/3/0 | 12/12/16/19/22 |
| gerados nos brackets 1+2 | 9/6/5/6/2 | 5/5/7/8/11 |
| promovidos para fid 15 | 10/8/3/3/0 | 4/4/5/5/9 |
| promovidos para fid 200 | 2/2/0/2/0 | 3/3/3/2/3 |
| falhas (OOM) | 65 (12/14/19/17/3) | 0 (154 ok via escada) |
| melhor κ @200 | 0,554 (B3) | 0,607 (B4) |

  Uma variante menor (2 brackets, 1/3 da escala) dá o mesmo resultado: na v3, B3/B4 nunca chegam a 200 e o melhor modelo é B1 (0,538); na v4 há 2 por faixa @200 e o melhor é B4 (0,597).
- **Testes** de `tests/modehb` na cópia com o diff, em CPU: **2308 passed, 4 skipped, 0 falhas** (incluídos os 10 novos de `tests/modehb/test_ga_v4.py`: faixas e teto, `next_band`, promoção por faixa, `mem_scale`, escada de OOM, falha fora da subpop, `_SplitLoader` e so2 igual à referência de readout_x).
- **e2e em CPU** (`run_search` real; remote_sensing; fid [1,3]; imagem 64; `max_train` 300; W&B desligado): a sessão caiu no meio da primeira rodada, e a retomada (`--resume`, replay) funcionou. 4 jobs ok: c00001 B0 20k q 0,342; c00002 B0 253k 0,436; c00003 B2 1,21M 0,388; c00004 B3 2,19M 0,394. A fitness rodou como `[SVM_FITNESS] scaler=none;kernel=linear;C=100;ovo` sobre so2_4096. O processo parou pelo timeout de 1500 s em CPU (EXIT 124), sem nenhum erro.

### 7b. Simulação sem pressão de OOM (GPU real de 40 GB; `sim_ga_gpu40.json`)

Aqui o viés aparece sem nenhuma falha:

| | v3 | v4 |
|---|---|---|
| gerados no bracket 0 (B0/B1/B2/B3/B4) | 22/20/19/18/**2** | 12/12/16/19/**22** |
| promovidos para fid 15 | 8/6/6/6/1 | 4/4/5/6/8 |
| promovidos para fid 45 | 5/4/4/5/**0** | 2/2/5/3/**7** |
| na fidelidade 200 | 1/1/1/1/**0** | 3/3/3/3/**3** |
| melhor κ @200 | 0,562 (B3) | 0,588 (B4) |

Arquivos em `research/spifil_b0b4/ga_v4/`: `ga_v4.diff`, `ga_v4.diff.md5`, `search_v4.yaml` e `fila_v4.json`, mais `sim_ga.py`, `sim_ga_gpu9.0.json`, `sim_ga_gpu40.json`, `diag.py` e `diag.json`. O diff aplica na árvore viva (`git apply --check` ok), mas **não foi aplicado**.
