# Fila e instruções do usuário — Hybrid SPiFiL / MO-DEHB / LEMONADE

> MD auxiliar de memória (pedido do usuário em 15/11). Antes de qualquer decisão, leia isto. Atualize a cada mudança.
> Regra: seguir as instruções ao pé da letra; usar múltiplos agentes; GPU nunca parada.

## 1. Regras permanentes (não mudar sem perguntar)

| # | Regra | Origem |
|---|---|---|
| R1 | **SVM = padrão do repo**: `core.svm.fit_svm`, SVC linear, **C=100**, ovo, max_iter=-1. NUNCA mudar C/scaler/solver. | "NUNCA MAIS MUDE ... C=100!" |
| R2 | **SVM original no TESTE para todo job do GA** (todas as fidelidades), no embedding `head(encoder(x))`, sem scaler, mesmo protocolo dos FMs. É o número reportado. A leitura cov_head (C=100) é só uma coluna extra. | "no final das contas eu preciso ver o SVM original" |
| R3 | **Fitness da reprodução = κ do SVM** (kappa_weight 1.0) e **FLOPs**. Mais nada. | "só considere o SVM e o FLOPS ... svm kappa" |
| R4 | **Toda pergunta**: caixa de opções (AskUserQuestion). **Sem PushNotification e sem commit** (17/11: "não mande push e nem commit, espere que eu faça isso"). | 15/11 e 17/11 |
| R5 | GPU nunca parada; validar em paralelo; zero falhas (OOM é refeito: OOM_RETRIES=10 × 300 s desde 16/11 10:55). | |
| R6 | Hybrid SPiFiL: o backprop soma ao SPiFiL, nunca desfaz. | |
| R7 | Resultados vão direto para a tabela de resultados (ranking da apresentação) e para o W&B, sem segurar. | "Não fica segurando esses resultados" |
| R8 | Português, respostas curtas; sem arquivos .sh; não apagar nada (arquivar). | |
| R10 | Jaci: pode usar a GPU à vontade (picos acima de 91 GB são aceitos; os retries de OOM seguram). Pausar só para evitar que um job vire failed. | "não tem problema, use a vontade" (16/11) |
| R11 | **Cabeça com embedding ≥ 1280 é obrigatória** em todo backbone (vetor de alta dimensão separa bem para uso futuro). Para caber no orçamento, otimizar a forma da cabeça, nunca a dimensão. | usuário 17/11 |
| R9 | **Código novo do GA só entra na fila depois de testado de ponta a ponta**: mini-busca real com todos os caminhos (sementes, DE, promoção, WiSE, xl ~5M, LEMONADE com herança), queda com kill -9 e retomada, arquivo ausente, monitor; zero failed. Não pode quebrar no meio. | "teste ele antes, porque não é legal ele quebrar durante o processo" |


- **R18 (18/11, usuário):** genes novos na v3: spifil_images_per_class (mais amostras/representatividade), spifil_n_superpixels (16/50/100/200) com eliminação de redundância dentro da rede, e a covariância como GATE treinável num residual (y = base + g·cov_branch, g=0 no início, aprendido no SSL; também na cabeça).
- **R17 (18/11, usuário):** cada SSL/técnica segue À RISCA o artigo (batch, vistas, augmentations, otimizador, lr…); regras em research/spifil_b0b4/articles/<metodo>.md; adaptações inevitáveis marcadas como "adaptação". O AE local + SPiFiL é a técnica própria (sem artigo).
- **R16 (18/11):** auditoria de mecanismos e literatura em research/spifil_b0b4/status_mecanismos_literatura.md; todos os itens parciais/ausentes entram na cadeia v3 (pedido do usuário).
- **R15 (17/11 22h, usuário):** parasitas com fundo preto (eggs/larvae/protozoan pre) NÃO se trabalham mais (só validação); *_raw no fim. Init sempre SPiFiL (sem comparar com aleatório); variar o SSL; cabeça ≠ GAP; TTA no teste; DA no SSL; mecanismos: whitening, SoftHebb, inception, concatenação de filtros da layer anterior, covariância para repor filtros ruins.
- **R14 (17/11, usuário):** prioridade SEMPRE flowers, remote_sensing, ham10000, aid, coconut_trees; parasitas (eggs/larvae/protozoan e *_raw) no FIM de toda fila.
- **R13 (17/11, usuário):** a GPU 2 do athena pode rodar a cadeia de crescimento ao lado do PNAS-CS, sem tocar no PNAS.
- **R12 (17/11):** a cada resultado novo (treino de base A2, crescimento B0→B4, atenção E1–E8, PNAS, FM, mapas de calor novos), analisar e atualizar research/spifil_b0b4/heatmaps_athena/INSIGHTS_heatmaps.md, cruzando com o diagnóstico; separar OBSERVADO de HIPÓTESE.

## 2. Fila (ordem de execução)

| # | Item | Status | Quando entra |
|---|---|---|---|
| F1 | Buscas atuais C=100/kw=1 — Jaci: hybrid remote_sensing, protozoan, ham10000, aid; LEMONADE remote_sensing, ham10000. Athena: hybrid flowers, larvae, coconut, eggs, larvae_raw, eggs_raw, protozoan_raw; LEMONADE flowers, larvae, coconut, eggs. | rodando desde 15/11 ~21:00–00:35 | — (não reiniciar) |
| F2 | LEMONADE aid (Jaci) | adiantado às 02:29 (GPU em 73 GB), mas PAUSADO às 03:00 (2 jobs, estado mantido no lugar): a GPU subiu para 92,4 GB (<4 GB livres para o root) e o LEMONADE rs teve OOMs. Voltou para a fila tmux queue_lem_aid, que espera o modehb_ham terminar e retoma | quando modehb_ham terminar |
| F2b | LEMONADE protozoan (Jaci) | fila (tmux queue_lem_proto) | quando modehb_aid terminar |
| F3 | **Buscas SEMEADAS** com o encoder "SPiFiL pequeno + LeJEPA" (sessão CosFace; RegNet 40-80-152-152-152-312 k3, 0,98M; κ teste 0,693 no remote_sensing com cov_head C=100), em `artifacts/modehb_seeded/<base>` e `artifacts/modehb_lemonade_seeded/<base>`, configs `search_v3_seeded.yaml` / `search_v3_lemonade_seeded.yaml`. | implementando (agente), encoders sendo trazidos do athena | **depois que a busca atual da mesma base terminar** ("logo após finalizar alguns experimentos ... entraria na fila") |
| F3a | Treino das sementes que faltam (receita deles, mesmo yaml): **aid** L6 LeJEPA a partir do L0 FPS; **ham10000** L0+L6 só com o NOSSO treino (o deles tinha val=treino → viu 878 imagens do nosso val). Saída em `exps/spifil_growth/ssl_pequeno_lejepa/{aid,ham10000_trainonly}_split1_pct100`. | NA FILA na Jaci:
- tmux seedtrain_aid (dry-run bs64 → L6, ~1,5–2,5 h);
- tmux seedtrain_ham (encoder FPS no split1_trainonly → dry-run → L0 → L6, ~4–5 h);
- cada estágio só sobe com memória livre ≥ pico + 6 GB, por 3 checagens;
- split `datasets/organized/ham10000/splits_incremental/split1_trainonly/` = nosso train/val, test vazio;
- receita idêntica à deles (`ssl_pequeno_lejepa_jaci.yaml`: lr 5e-5, bf16, bs64, LeJEPA λ0,05, 2×200 + 6×96 vistas, patience 3000 passos) | antes da busca semeada de aid / ham10000 (usuário: "Treinar o LeJEPA", "Retreinar só no nosso treino") |
| F5 | **Baseline dos modelos grandes** (AIMv2, DINOv2, I-JEPA, ResNet-50, ConvNeXtV2-pico) em larvae_raw, eggs_raw e protozoan_raw: mesmo protocolo do benchmark (congelado, SVM C=100 sem scaler) e mesmo split split1_modehb_raw | pedido ao claude_explore_models (athena) | assim que sobrar espaço na GPU 2/3 |
| F4 | Hybrid seed43 flowers e larvae (athena) | reserva | se a GPU do athena ficar < 38 GB por 20 min |

### Encoders-semente disponíveis (15/11, split1 pct100)
| base | lejepa (L6) | L0 | situação |
|---|---|---|---|
| remote_sensing, flowers, coconut_trees | sim (athena → jaci `artifacts/external_ssl_pequeno/`) | sim | split1 = o nosso splits/split1.json |
| aid | NÃO (o run parou antes da L6) | sim | precisa treinar a L6 (~1–4 h de GPU) |
| eggs, larvae, protozoan (+ _raw) | sim na jaci (`exps/spifil_growth/ssl_pequeno_lejepa/`, symlinks em `artifacts/external_ssl_pequeno/`) | sim | sem vazamento (0 imagens do nosso test/val no treino deles). Treinados em raw: nas bases pre (fundo preto) o domínio é outro. Relatório: `reports/external_seed_leakage.md` |
| ham10000 | não usar o atual (viu o nosso val) | — | retreinar só no nosso treino (F3a) |

## 2b. Experimento novo (15/11 noite): Progressive NAS + Compound Scaling + MO-DEHB + Network Morphism (PNAS-CS)

- **Hardware:** athena **GPU 2 EXCLUSIVA**. Fila sequencial **remote_sensing → ham10000 → flowers**; orçamento de memória ~40 GB, teto 44 GB; retomada automática; W&B completo.
- **Buscas que estão hoje na GPU 2** (flowers hybrid, larvae, larvae_raw, protozoan_raw, LEMONADE larvae, LEMONADE eggs) vão para a **GPU 3 e a Jaci** ("os outros experimentos ... vão ficar com a GPU 3 e a Jaci todinha"). A troca só acontece quando o código PNAS-CS estiver validado (R9), para a GPU não ficar parada. O estado é preservado (search_state + candidates; rsync para a Jaci no mesmo caminho absoluto).
- **Método:**
  - sementes = melhores backbones SPiFiL (SPiFiL pequeno + LeJEPA externo + os melhores do GA), mais exploração de arquiteturas novas;
  - arquitetura-base em estágios: Stem → blocos conv/residuais → embedding;
  - **Compound Scaling**: famílias P0–P4 com 500k / 1M / 2M / 3,5M / 5,5M parâmetros, profundidade α e largura β, resolução fixa;
  - Network Morphism com herança de pesos (P_k → P_{k+1} preservando a função);
  - mutações: +camada, +bloco residual, +canais, pooling;
  - subpopulações e promoções protegidas por porte (P-level).
- **Seleção:** κ do SVM na validação (C=100) para promoção, e objetivos **κ × FLOPs**. Pareto sempre atualizado. Teste nunca entra na seleção.
- **Treino:**
  - SPiFiL prioritário; aleatório só como baseline;
  - LeJEPA + SIGReg como SSL principal;
  - **AE camada a camada obrigatório e testado PRIMEIRO, por ablação** (Fase 0);
  - BF16, com as operações sensíveis em FP32;
  - orçamento calibrado por porte, para os grandes não serem descartados por convergirem devagar.
- **W&B:** HV × GPU-horas, κ × params, evolução do Pareto, crescimento P0–P4, profundidade × largura, ganho marginal por expansão, taxa de sobrevivência das mutações, lei de crescimento/saturação.
- **Status:** implementando (agente) → validação de ponta a ponta (R9) → libera a GPU 2 → lança a fila.
- **Migração das 6 buscas da GPU 2 do athena para a Jaci:**
  - waiters na Jaci: tmux migr_lem_eggs, migr_lem_larvae, migr_larvae_raw, migr_protozoan_raw, migr_larvae, migr_flowers;
  - cada um espera o marcador `artifacts/migration_markers/<kind>_<base>.ok` (criado pelo claude_explore_models depois do rsync conferido) e GPU livre ≥12 GB (flowers ≥18 GB) por 3 checagens, e então retoma;
  - controller W&B novo (o artifacts do athena é symlink, então o resolve() é diferente);
  - flowers recomeça do zero com 20/24 (só tinha 2 jobs).
- **Usuário OFF ~10 h (desde 15/11 ~22:40, horário da Jaci):** operações seguras sem perguntar; tudo registrado aqui. Vigilância: tmux modehb_watchdog (Jaci e athena) → `reports/watchdog_alerts.log`.

## 2c. MISSÃO SPiFiL Progressivo B0–B4 (17/11, madrugada) — spec em research/spifil_b0b4/MISSION.md
- **Máquinas (regra absoluta):**
  - athena GPU 2 NÃO MEXER;
  - athena GPU 3 dedicada à nova pesquisa: os nossos jobs dela migram para a fila da Jaci, retomando;
  - Jaci: não mexer nos existentes, só acrescentar à fila.
- **Equipe:**
  - Agente 5: athena = claude_explore_models; Jaci = agente local (validação de resume e fila);
  - Agente 1 em 4 partes paralelas (1a arquitetura/init, 1b aprendizagem/embedding, 1c generalização/vazamento, 1d literatura) → síntese → **CHECK**;
  - Agentes 2/3/4 em preparação (literatura + desenho, sem código/GPU) até os gates.
- **Gates:** A1 → A2 → A3 → A4; o CHECK audita cada gate.

- 17/11 ~02:00 Jaci: as buscas migradas da athena GPU 3 foram PUBLICADAS (validação de resume ok: replay a seco, md5, dados, config).
  - Waiters tmux g3_*: hybrid eggs/eggs_raw/coconut, LEMONADE flowers/coconut e as 5 semeadas, com gate de memória + flock e sem preempção. O hybrid coconut precisa de 34/44 (gate de 33 GB livres).
  - seed43 flowers/larvae continuam como reserva F4 (conflito de diretório; aguardar OK).
- Adendo de atenção sem pesos (ATTENTION_ADDENDUM.md) → Agente 6.

- 17/11 ~02:20:
  - **CHECK do gate B-attn: APROVADO COM CONDIÇÕES** (check/gate_B_attn_audit.md). A fase B (E1–E8) só roda depois de C1–C4. Ficou só o E0 (val) liberado, pelo gate parcial gates/gate_B_attn_E0only.approved.
  - Treino em curso na GPU 3: Fase 0 do PNAS-CS (ablação do AE) em flowers e ham10000 (artifacts/pnas_cs_phase0_gpu3/).
  - Implementação em paralelo: 2a driver + interface, 2b SPiFiL-1, 2c SPiFiL-2, 2d SPiFiL-3, 3 specs/morfismos, 4 infraestrutura hmodehb.

## 3. Especificação das buscas semeadas (F3)

- **Semente** em todas as bases que tiverem o encoder (L6 lejepa `best_kappa.ckpt` + L0 FPS). Sem o arquivo → a busca roda como hoje (aviso no log).
- **Grid de crescimento** a partir da semente, sem estragar o que o SPiFiL/LeJEPA aprendeu:
  - largura por estágio ×{0,75, 1, 1,25, 1,5, 2, 2,5} (mais filtros, menos filtros);
  - profundidade +{0, 1, 2} camadas;
  - morfismos que preservam a função (Net2Wider/Deeper, saída zerada nos filtros novos);
  - pesos importados protegidos (protect freeze/lowlr + âncora WiSE);
  - filtros novos com SPiFiL init + AE local + SSL.
- **Crescer mais:**
  - estratos small ≤0,8M, medium ≤1,5M, large ≤3M, xl ≤5M;
  - teto duro 5,5M ("se precisar passar um pouquinho ... passe").
- MO-DEHB: variações do grid entram como sementes da população inicial. LEMONADE: a semente e as variações são pais com herança Lamarckiana.
- Fitness/SVM iguais às regras R1–R3.

## 4. Resultados — onde estão

- Teste com o SVM original, todo job: `ga_search/modehb/reports/test_svm_C100.jsonl` → `HYBRID_PROGRESS.md` (automático, monitor `modehb_monitor`).
- Export para a tabela de resultados da apresentação: `ga_search/modehb/reports/ranking_ga_C100.csv` (gerado pelo monitor). Cópia no athena a cada 30 min (`ranking_ga_C100_jaci.csv`, tmux modehb_rsync_ranking do claude_explore_models).
- A tabela da apresentação é mantida pela sessão **cluster_presentation_athena**: athena:/dados/home/moliveira/presentation_results_cluster_november/results_grids.md, §2.2 (SVM original) e §5 (cov_head, fora do ranking). Os resultados novos vão para ESSA sessão.
- Régua igual (nossos × SPiFiL pequeno): `ga_search/modehb/reports/remote_sensing_regua_igual.md` (pronto 16/11, enviado ao cluster_presentation_athena).
  - SVM original: CosFace 0,605 > LEMONADE c00098 0,598 > SimCLR 0,585 > L0/LeJEPA 0,570 > MO-DEHB c00010 0,524 … MO-DEHB c00004 0,480.
  - cov_head: SPiFiL pequeno + LeJEPA 0,693 > c00004 0,667.
- Resultados das MDs do repo: `ga_search/modehb/reports/results_from_mds.md`.

## 4b. Código aplicado (16/11 ~01:20, Jaci; athena sincronizando)
- Patch da semente externa (`external_seed.py`, `search_v3_seeded.yaml`, `search_v3_lemonade_seeded.yaml`):
  - validado de ponta a ponta: mini-buscas MO-DEHB (23 jobs) e LEMONADE (22 jobs) com 0 falhas, queda com kill -9 e retomada, semente ausente, monitor;
  - com a semente desligada, o código é neutro;
  - 2251 testes passando;
  - backup dos arquivos antes do patch em `scratchpad/pre_seed_backup`.
- Smoke da semente pura no remote_sensing (f5): κ val 0,564 / teste 0,693 (cov_head) / 0,578 (SVM original); 7,9 min; 4,1 GB.
- Correção do monitor: a assinatura do fitness sai do nome do diretório (antes tudo virava C0.1 e jobs novos eram pulados); jobs C=100 têm prioridade. A jsonl foi limpa: 97 linhas mantidas, 41 refeitas (backup `*_prefix_*.jsonl.bak`).
- Budget das semeadas: as sementes nunca são alteradas pelo mem_repair; a que não cabe é descartada.
  - 24/28 preserva todas as xl; 20/24 perde só a ×2; 15/18 é fraco; 12/14 perde até a semente pura (não usar).
- **Fila F3 montada na Jaci (16/11 01:28):** 12 waiters `seedq_*`.
  - Cada um espera o "done" da busca atual da mesma base + o ckpt da semente + memória livre ≥ cap.
  - rs hybrid 24/28 · rs LEMONADE 24/28 · protozoan 24/28 · ham10000 hybrid/LEMONADE 20/24 (semente retreinada só no nosso treino) · aid 20/24 (semente L6 nova) · larvae, larvae_raw, protozoan_raw, flowers (migradas) 20/24 · LEMONADE larvae, eggs 20/24.
  - Log de início: `artifacts/seeded_queue.log`.
- Symlinks das sementes:
  - `external_ssl_pequeno/ham10000_split1_pct100` → retreino trainonly; o antigo foi renomeado para `*_VIU_NOSSO_VAL_nao_usar`;
  - `aid_split1_pct100/lejepa` → o L6 novo; o vazio do athena virou `lejepa_vazio_athena`.
- Athena: a F3 foi pedida ao claude_explore_models (todas 24/28, inclusive coconut).

## 4c. Riscos em aberto (16/11 02:00)
- **seedtrain_ham / seedtrain_aid podem ficar sem vez na Jaci:**
  - o dry-run com bs64 precisa de ≥22,9 GB livres, e a Jaci oscila entre 9 e 17 GB livres;
  - quando as migradas do athena chegarem (o gate delas é ≥12 GB), elas pegam a memória primeiro;
  - a semente do ham é pré-requisito da fase ham do PNAS-CS (2ª da fila da GPU 2, depois do remote_sensing, que deve levar ≥1 dia) e das semeadas aid/ham;
  - se até lá não tiver rodado: pausar uma busca pequena da Jaci (estado preservado) durante o treino (~4–5 h) ou rodar o treino na GPU 2 entre a fase rs e a ham do PNAS. Decidir com o usuário quando ele voltar.
  - **Mitigação (02:10):** os treinos aid L6 (R1) e ham trainonly L6 (R2) foram para a RESERVA da GPU 3 do athena (≥20 GB livres por 20 min, um por vez). O claude_explore_models avisa antes de lançar, para eu desligar o estágio equivalente na Jaci; depois faz rsync do L6 para os mesmos caminhos da Jaci.
  - 02:15: seedtrain_aid/seedtrain_ham da Jaci DESLIGADOS (nada tinha começado); R1/R2 só no athena (tmux seedtrain_athena, cópia seedtrain_athena.py). As réplicas seed43 podem ser pausadas para abrir espaço; sementes têm prioridade.
- **Athena (05:15 lá):** 13 buscas (+ hybrid flowers seed43 e larvae seed43 na GPU 3, da reserva).

## 4d. Gargalo de CPU no athena (16/11 ~03:30 Jaci / 06:21 athena)
- Sintoma: GPUs 2/3 do athena a 0% por ~1 h, load 220 em 96 núcleos, logs parados há horas.
- Causas:
  - (1) o eigh do DBN por passo de treino tinha sido posto na CPU fp64 (correção do c00040);
  - (2) cada busca usava todos os núcleos (torch/BLAS).
- Correções, sem mudar o SVM (R1):
  - `safe_eigh(on_cpu=...)`: o DBN fica na GPU; só o ZCA (ajustado uma vez) vai para a CPU;
  - `run_search._cap_cpu_threads` (padrão cpu_count//12; MODEHB_THREADS);
  - 2253 testes passando.
- Aplicação sem perder job: `restart_between_jobs.py` (espera o job ser salvo → SIGTERM → o laço retoma). Na Jaci: tmux rbj_<pid> para as 6 buscas. No athena: o claude_explore_models aplica; as buscas paradas há mais de 1 h são mortas direto.

- 07:02 athena: o patch resolveu (load 220 → 58, GPUs 2/3 a 100%). Para o R1 abrir, o LEMONADE flowers do athena foi PAUSADO entre jobs (estado no lugar). Um waiter o relança depois do R2.

- 07:51 athena: **R1 (aid L6) COMEÇOU** na GPU 3 (LEMONADE coconut e flowers pausados entre jobs, voltam depois do R2). Depois vem o R2 (ham trainonly L6). Ao terminar: rsync para a Jaci, nos mesmos caminhos.

## 4e. PNAS-CS validado e aplicado (16/11 ~06:20 Jaci)
- Patch `scratchpad/pnasrepo3/pnas_cs.patch`, aplicado na Jaci com md5 conferidos; 2286 testes passando; backup em `scratchpad/pre_pnas_backup`.
- e2e: 49 jobs, 0 falhas.
  - Fase 0: decoder_first 0,496 > joint 0,481 > sem AE 0,467.
  - Família da semente P1→P4 até 5,77M; 4 mutações; kill -9 com retomada; monitor.
- Estimativa: ~220 jobs por base com max_brackets 4. rs ~45 GPU-h, ham ~105, flowers ~100 → ~11 dias.
- Pontos em aberto:
  - (a) `random_twins: false`: o caminho de gêmeas não foi validado com o PNAS (ligado, quebra o contador de HV nos testes). Baseline aleatório: fazer DEPOIS, gêmeas aleatórias da fronteira de Pareto de cada base, validadas antes (R9);
  - (b) `widen_block` só preserva a função em blocos planos (sem norm/skip); as famílias das sementes do GA (não planas) quase não herdam no crescimento. Estender depois;
  - (c) GA seeds best_from no athena: os states de ham/flowers estão na Jaci ou migrando; o rs usa checkpoints explícitos (rsync pedido).
- Athena: o claude_explore_models vai trazer os arquivos, religar o monitor na GPU 2 (para os testes C=100 do athena), migrar as 6 da GPU 2 e lançar `tmux pnas_cs`, com gate no ckpt da semente ham.

- **PNAS-CS RODANDO** na athena GPU 2 desde 16/11 09:30 (horário do athena), tmux pnas_cs, controller ctrl-a48cd6f06c856533.
  - Correção de compatibilidade do torch 2.2 (`_bf16_autocast`) aplicada; 2286 testes passando no athena. Reiniciado às 09:37 sem ReplayMismatch.
  - As 6 da GPU 2 migraram (marcadores criados) e esperam memória na Jaci; o flowers foi arquivado em `modehb_hybrid_gpu2_parado_0929`.

## 4f. Sementes LeJEPA: checkpoint pela loss (16/11 ~09:45)
- O módulo do discovery, com `lejepa` e `best_by_probe=False`, escolhe o best_kappa.ckpt pela MENOR val_ssl_loss, o que muitas vezes cai na época 0.
- Épocas dos best: rs 741, larvae 198, eggs 28; **flowers 6, coconut 0, protozoan 1, aid 0**.
- Usuário: "Retreinar pelo κ".
  - Correção só de config: `best_by_probe=True`, ou seja, monitor = probe/lin_select_kappa (κ da sonda GeM na validação, sem warmup).
  - Fila no athena GPU 3: R1b aid L6_kappa → R2 ham trainonly → R3 coconut → R4 flowers.
  - Protozoan na Jaci quando houver memória.
  - Depois de cada um: reapontar os symlinks `artifacts/external_ssl_pequeno/<base>_split1_pct100/lejepa`.
  - Isso é automático na Jaci: tmux seed_repoint espera o L6_kappa de aid → coconut → flowers chegar e ficar estável, move o lejepa antigo para `lejepa_ep_antigo_*` e cria o symlink; o log fica em `artifacts/seed_repoint.log`. O ham já aponta para `ham10000_trainonly_split1_pct100/L6`, a saída do R2.
  - No athena, o claude_explore_models faz o mesmo (tmux seedtrain_r1b..r4).

- 06:53 Jaci: 5 migradas começaram JUNTAS (cada gate viu os 12 GB livres ao mesmo tempo), a GPU chegou a 96,9 GB e houve retries de OOM. Medidas:
  - mantidas só LEMONADE eggs e LEMONADE larvae; larvae, larvae_raw, protozoan_raw e flowers foram mortas (tinham acabado de começar; estado preservado);
  - recriadas com gate escalonado: só 1 começo a cada 30 min (lock `artifacts/migration_markers/.last_start`), e só com ≥20 GB livres (flowers ≥28) por 5 leituras, porque os picos da GPU passavam de 95 GB.

- 09:05 Jaci: o hybrid protozoan teve OOM 2/4 no job 15 (GPU a 95,5 GB). O processo foi pausado durante a espera do retry (nada perdido; 15 jobs, 0 failed) e relançado no mesmo tmux modehb_hybrid_b com gate de ≥24 GB livres por 3 leituras. A retomada refaz o job 15.

- 16/11 ~09:20 Jaci (R10): o protozoan foi relançado sem gate; as 5 migradas têm gate mais leve (≥8 GB livres, flowers ≥12, 2 leituras, 10 min entre um começo e outro).

## 4g. BLAS de 64 threads no fit do SVM (16/11 ~10:30 Jaci)
- O `_cap_cpu_threads` não limitava o OpenBLAS do scipy (ddot do libsvm) nem o libgomp: os dois carregam depois e sobem com todos os núcleos.
- Com o cov_head de D>10k, um fit levava até 9560 s (LEMONADE larvae). Com BLAS 1 thread: 2,6 s, κ idêntico.
- Correção imediata, só de ambiente:
  - todas as sessões de busca da Jaci foram recriadas com `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`; os jobs travados nos fits foram refeitos;
  - o mesmo foi pedido no athena.
- O patch definitivo (agente svmfast) está em validação: exporta as variáveis no run_search, eigh em streams CUDA bitwise igual e fits dos alphas em paralelo.

- 10:41 Jaci: **1 failed** no larvae (migrado), job 17 c00004 f45: cuDNN "unable to find an engine" com a GPU cheia, porque 4 migradas entraram juntas (corrida na trava por idade).
  - Correções: `_is_oom` passa a cobrir os erros de engine e alloc do cuDNN/cuBLAS (2287 testes passando); o state foi cortado no job 17 (backup `.antes_corte_cudnn_*`; o f45 vai para `modehb_hybrid_cudnnfail_1041`); os gates das migradas passam a usar `flock` com 15 min entre um começo e outro.
  - Aplicação entre jobs: tmux rbj2_<pid> para as 8 buscas da Jaci.

- 10:55 Jaci: OOM_RETRIES 4→10 e sleep 180→300 s (~50 min de paciência), porque a Jaci fica cheia com o rs hybrid (budget 60, ~35 GB reais). Aplicado entre jobs (rbj2_/rbj3_). O protozoan_raw foi pausado no retry 2/4 e voltou para o gate com flock (≥16 GB).

- 11:43 Jaci: **1 failed** no hybrid remote_sensing, job 39 c00020 f5 (OOM com a GPU em 97 GB; o processo ainda rodava o código com 4 retries). O state foi cortado no job 39 (backup `.antes_corte_oom_*`; f5 em `modehb_hybrid_oomfail_1143`) e relançado com o código de 10 retries, sem ReplayMismatch.

- 12:29 Jaci: **1 failed** no hybrid ham10000, job 34 c00015 f5: OOM 11 vezes em ~55 min, com a GPU sempre cheia (9 buscas + root).
  - O state foi cortado no job 34 (backup + f5 em `modehb_hybrid_oomfail_1229`) e relançado.
  - Causa: excesso de buscas. **Concorrência reduzida**: as migradas (LEMONADE eggs, LEMONADE larvae, larvae_raw) foram pausadas, com estado preservado (0 failed).
  - Todas as migradas voltam por gate com flock (≥22 GB livres, flowers ≥28, **1 começo a cada 30 min**).
  - Ficam rodando: hybrid rs, protozoan, aid, ham; LEMONADE rs, ham.

- ~13:00 Jaci: patch **svm_fitness_fast** aplicado (2311 testes passando; backup em `scratchpad/pre_svmfast_backup`).
  - Fitness 2,4–4,8× mais rápido, com features bitwise iguais, mesmos κ e mesmo alpha nos 6 checkpoints testados.
  - Mudanças: BLAS limitado de verdade, eigh em 8 streams CUDA, fits do WiSE em paralelo. O SVM não mudou (core/svm.py intacto).
  - Aplicado entre jobs (rbj5_*); monitor e watchdog religados com OPENBLAS=2.
  - Athena: testar `test_fitness_speed.py` no torch 2.2 antes.

- 16/11 (~13:20 Jaci): o usuário aprovou rodar as sementes R2 ham, R3 coconut e R4 flowers na **GPU 2 do athena, ao lado do PNAS-CS** (gpucap 16 GB, uma por vez; o PNAS tem prioridade). A semente do aid (R1b) foi entregue: κ 0,681, época 61, parada manual depois de ~80 épocas sem melhora.

- ~13:35 Jaci: monitor_report com prioridade para o melhor por validação de cada estrato e P-level (huge e P4 ganham teste cedo), shards (`MONITOR_SHARD`, 2 instâncias: tmux modehb_monitor e modehb_monitor2), escrita com flock e OOM sem contar como erro definitivo. O watchdog religa o shard 0 (WATCHDOG_MONITOR_SHARD=0/2). Athena: as mesmas 2 instâncias (GPU 3 e GPU 2).

- 14:33 Jaci: **1 failed** no hybrid rs, job 43 c00024 f5: o fit do SPiFiL pede 12,96 GiB contíguos e esgotou os 10 retries com 10,6 GB livres. O state foi cortado no job 43 e relançado. O LEMONADE ham10000 (19 jobs, 0 failed) foi PAUSADO para abrir memória e volta sozinho quando o rs passar do job 43 (gate no tmux modehb_lem_ham).

- ~17:20 Jaci: o hybrid remote_sensing estava preso no job 43 (c00024): o fit do SPiFiL pede uma matriz N×N de ~13 GB (N=41700) e a codificação inteira, ~23 GB.
  - Patch `scratchpad/spifil_oom_fallback.patch`: `spifil_oom.py` faz o score em blocos SÓ quando dá OOM (bitwise igual); `_SkipEncode` pula a codificação que não é usada (pesos idênticos em 10 candidatos); 2324 testes passando.
  - **A validação de ponta a ponta (R9) ainda não foi feita**, então o patch NÃO foi para a árvore viva. Só o rs hybrid roda dessa cópia com patch (`scratchpad/rs_patched`, mesmo output_dir e controller), como teste em produção; gate ≥24 GB livres.
  - Se o job 43 passar e os seguintes rodarem sem falha, o patch vai para a árvore viva.

- ~18:00 Jaci: monitor com `MONITOR_COV=top`. A coluna cov_head (ablação, mesmo protocolo) só é calculada para o melhor por validação de cada (busca, base, estrato); o SVM original continua em todos os jobs. Motivo: um fit de cov levava ~1 h no athena (coconut 3741 s), e o SVM não pode mudar (R1).

- 16/11 noite: o usuário pediu HEATMAPS para os slides, comparando o melhor modelo grande de cada base contra o NOSSO melhor (pelo SVM original).
  - 10 imagens de teste por base (as mesmas para os dois), em athena:/dados/home/moliveira/presentation_results_cluster_november/heatmaps/<base>/.
  - Método: CAM do SVM linear (pesos C=100 aplicados ao mapa antes do pooling / tokens de patch).
  - Delegado ao claude_explore_models, porque os FMs e os slides estão no athena.

- 17/11 00:10 athena: R5 protozoan L6_kappa entregue (best ep35, κ sonda 0,551; parada pelas 40 épocas). Fica a ep35: a semente antiga (ep1) tem 0,509; a ep0 (0,614) é ≈ L0 + 1 época, no warmup, sem ckpt.
- **Padrão observado** em aid/coconut/flowers/protozoan/ham: o κ da sonda é máximo nas primeiras épocas e o LeJEPA longo piora. Igual ao WiSE α=0 no coconut c00001.
  - **Investigar** (tarefa futura): lr, vistas, projetor e λ do LeJEPA; o SSL não está somando sobre o SPiFiL nessas bases.

- 17/11 00:30: **ATENÇÃO, comparação**: os κ de FM usados para larvae/eggs/protozoan são medidos em imagens RAW (idênticos ao modehb_raw). As nossas larvae/eggs/protozoan são pre/orig (fundo preto). A comparação justa é *_raw × FM: protozoan_raw 0,874 COV > ResNet-50 0,822 e ConvNeXtV2-pico 0,853; eggs_raw 0,922 e larvae_raw 0,829 ficam abaixo de todos os FMs.

## 5. Histórico de decisões (resumo)

- 15/11: C=0,1 → C=100 (ordem do usuário; C=0,1 tinha sido escolha de um agente na validação). kappa_weight 0,75 → 1,0. As buscas com C=0,1 foram arquivadas em `*_C01_*` e são avaliadas no teste.
- 15/11: semear o GA com o SPiFiL pequeno + LeJEPA: "Sim, em todas", mas na FILA, depois das buscas atuais.
- 15/11: grid de crescimento + teto de 5M (soft) / 5,5M (duro) nas semeadas.
