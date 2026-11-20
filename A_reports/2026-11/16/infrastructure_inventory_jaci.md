# Inventário de infraestrutura — Jaci (só leitura)

_Snapshot: 16/11/2026 22:01–22:10 (horário da Jaci). Agente 5, parte Jaci. Nada foi encerrado, pausado, reiniciado ou reconfigurado._

## 1. Máquina

| item | valor |
|---|---|
| GPU | 1× NVIDIA RTX PRO 6000 Blackwell, 97 887 MiB, driver 615.71.09 (CUDA UMD 13.4) |
| Memória GPU | 89–96 GB usados (7,9–1,6 GB livres), util 43–100 % |
| CPU / carga | 128 núcleos; load 14,7 / 16,3 / 19,5 |
| RAM | 251 GB (142 usados, 108 disponíveis) |
| Disco `/dados` | 2,9 TB, 587 GB livres (79 %); `artifacts/` = 41 GB |
| Disco `/` (onde fica o `/tmp` e o scratchpad) | 437 GB, **11 GB livres (98 %)** — ⚠ o hybrid remote_sensing roda de `scratchpad/rs_patched` (em `/tmp`) |
| Python | 3.11.15 (`/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python`) |
| torch | **2.7.1+cu128** (CUDA 12.8); sklearn 1.8.0; numpy 1.26.4 (o athena usa torch 2.2: checkpoints carregam aqui sem problema, testado) |
| `artifacts/` | diretório real (não é symlink), ext4 no `/dados` → `mv` de `_staging_gpu3` para `artifacts/<kind>` é rename atômico |

## 2. Processos na GPU (dono)

| dono | nº processos | memória |
|---|---|---|
| moliveira (nossos: 10 buscas + 2 monitores) | 12 | ~76 GB |
| **root — terceiros** (`/david/wonderful/npnets/kitkat/bin/python main.py --dataset brisc2025 ... --split 5..9`) | 5 | ~13 GB (2,6 GB cada) — **não mexer** |

Por processo (nossos), MiB: protozoan 14 458 · protozoan_raw 11 456 · LEM ham10000 9 806–9 906 · larvae_raw 8 056 · aid 7 612 · LEM rs 6 824 · flowers 4 044 · larvae 3 592–4 066 · ham10000 3 164 · LEM eggs 2 330–10 868 · monitor0 1 582 · monitor1 1 566 (o pico varia a cada job).

**Situação de memória:** GPU saturada. Nos logs há OOM-retries simultâneos (flowers job 15, ham10000 job 60, LEM eggs job 36, LEM rs job 129, rs hybrid job 43 retry 8/10); watchdog contabiliza 259 retries de OOM no total. Qualquer migrada nova deve ficar presa no gate até abrir memória.

## 3. Sessões tmux (31)

### 3a. Buscas RODANDO (run_search ativo)

| tmux | busca | output_dir | jobs (ok/failed) | último job | obs |
|---|---|---|---|---|---|
| modehb_aid | hybrid aid | artifacts/modehb_hybrid/aid/state/seed42 | 59 (59/0) | c00039 f5 | budget 15/cap 18 |
| modehb_ham | hybrid ham10000 | artifacts/modehb_hybrid/ham10000/state/seed42 | 59 (59/0) | c00039 f5; job 60 em retry OOM | 15/18 |
| modehb_hybrid_b | hybrid protozoan | artifacts/modehb_hybrid/protozoan/state/seed42 | 47 (47/0) | c00027 f5 | 20/24 |
| modehb_lemonade | LEMONADE remote_sensing | artifacts/modehb_lemonade/remote_sensing/state/seed42 | 129 (129/0) | c00004 f15; job 129 retry OOM 3/10 | 20/20 |
| modehb_lem_ham | LEMONADE ham10000 | artifacts/modehb_lemonade/ham10000/state/seed42 | 36 (36/0) | c00016 f5 | 15/18 |
| migr_lem_eggs | LEMONADE eggs (migrada GPU 2) | artifacts/modehb_lemonade/eggs/state/seed42 | 35 (35/0) | c00015 f5; job 36 retry OOM | 12/14 |
| migr_flowers | hybrid flowers (migrada) | artifacts/modehb_hybrid/flowers/state/seed42 | 15 (15/0) | c00003 f200; job 15 retry OOM 2/10 | 20/24, gate 28 GB |
| migr_larvae | hybrid larvae (migrada) | artifacts/modehb_hybrid/larvae/state/seed42 | 48 (47/1) | c00028 f5 = **failed** "params 3 744 470 > cap 3 500 000" (infeasível pelo teto; comportamento esperado da busca, não é falha de infra) | 12/14 |
| migr_larvae_raw | hybrid larvae_raw (migrada) | artifacts/modehb_hybrid/larvae_raw/state/seed42 | 42 (42/0) | c00022 f5 | 12/14 |
| migr_protozoan_raw | hybrid protozoan_raw (migrada) | artifacts/modehb_hybrid/protozoan_raw/state/seed42 | 27 (27/0) | c00007 f5 | 12/14 |

Todas com `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, laço de 5 tentativas. Nenhuma busca com `done=True`; brackets concluídos = 0 em todas (ainda no bracket 0).

### 3b. Buscas ESPERANDO (gate)

| tmux | o que espera | depois roda |
|---|---|---|
| modehb_hybrid | ≥24 GB livres por 2 leituras de 60 s | hybrid remote_sensing **do código com patch** `scratchpad/rs_patched` (spifil_oom, sem R9 ainda), budget 60/cap 64; 43 jobs (0 failed), parado no job 43 (c00024 f5) desde 16:15 |
| migr_lem_larvae | ≥22 GB livres + flock `.last_start` (30 min) | LEMONADE larvae, 30 jobs (0 failed), pausada desde 11:56 |
| queue_lem_aid | tmux `modehb_ham` terminar | LEMONADE aid (2 jobs, pausada); ⚠ sem `OPENBLAS/OMP/MKL` no ambiente (o run_search agora exporta os limites, mas registrar) |
| queue_lem_proto | tmux `modehb_aid` terminar | LEMONADE protozoan (do zero); ⚠ idem, sem as variáveis BLAS |
| seedq_rs_hyb / seedq_rs_lem | `done` da busca rs (hybrid / LEMONADE) + ckpt da semente; ≥28 GB × 3 | semeada rs → `artifacts/modehb_seeded/remote_sensing`, `artifacts/modehb_lemonade_seeded/remote_sensing` (24/28) |
| seedq_proto_hyb | done hybrid protozoan; ≥28 GB × 3 | `artifacts/modehb_seeded/protozoan` (24/28) |
| seedq_aid_hyb, seedq_ham_hyb, seedq_larvae_hyb, seedq_larvae_raw_hyb, seedq_protozoan_raw_hyb, seedq_flowers_hyb | done da hybrid da mesma base; ≥24 GB × 3 | `artifacts/modehb_seeded/<base>` (20/24) |
| seedq_ham_lem, seedq_larvae_lem, seedq_eggs_lem | done da LEMONADE da mesma base; ≥24 GB × 3 | `artifacts/modehb_lemonade_seeded/<base>` (20/24) |

⚠ Os 12 `seedq_*` exportam só `PYTORCH_CUDA_ALLOC_CONF` (sem as variáveis BLAS) e não têm flock nem checagem de controlador duplicado.
⚠ **Risco de 2 controladores**: `seedq_flowers_hyb` vai criar/retomar `artifacts/modehb_seeded/flowers/state/seed42`. Se uma busca semeada de flowers vier do athena para o mesmo caminho, os dois usariam o mesmo `search_state.json`. O mesmo vale para qualquer semeada do athena cuja base tenha um `seedq_*` na Jaci (rs, protozoan, aid, ham10000, larvae, larvae_raw, protozoan_raw, flowers; LEMONADE rs, ham10000, larvae, eggs). O `staging_watch.py` detecta e marca CONFLITO.

### 3c. Serviços

| tmux | processo | obs |
|---|---|---|
| modehb_monitor / modehb_monitor2 | `ga_search.modehb.monitor_report` (shards 0/2 e 1/2, `MONITOR_COV=top`, nice 5) | ~1,6 GB GPU cada |
| modehb_watchdog | `ga_search.modehb.watchdog` (nice 10) → `ga_search/modehb/reports/watchdog_alerts.log` | |
| claude_GA, claude_explore | sessões interativas do Claude | |

Sessões `rbj*`: **nenhuma ativa** (as de restart-between-jobs já terminaram). Sessões `queue_*`: as 2 acima.

### 3d. Marcadores e travas
- `artifacts/migration_markers/`: `.last_start` (último começo 16/11 16:02), `.last_start.flock`, e os `.ok` das 6 migradas da GPU 2 (06:29–06:30).
- `artifacts/migration_from_athena.log`: log de START dos waiters.
- `artifacts/_staging_gpu3/`: **ainda não existe** (22:10).

## 4. Destinos das buscas que virão da GPU 3

| busca | destino na Jaci | existe? |
|---|---|---|
| hybrid coconut_trees | artifacts/modehb_hybrid/coconut_trees | não |
| hybrid eggs | artifacts/modehb_hybrid/eggs | não |
| hybrid eggs_raw | artifacts/modehb_hybrid/eggs_raw | não |
| LEMONADE coconut_trees | artifacts/modehb_lemonade/coconut_trees | não |
| LEMONADE flowers | artifacts/modehb_lemonade/flowers | não |
| hybrid flowers seed43 / larvae seed43 | artifacts/modehb_hybrid/{flowers,larvae}/state/seed43 | **a base existe** (seed42 rodando aqui) → conflito no nível da base |
| semeadas | artifacts/modehb_seeded/<base>, artifacts/modehb_lemonade_seeded/<base> | vazios hoje, mas com `seedq_*` apontando para várias bases |

Datasets na Jaci: `ga_search/config/modehb/datasets/{coconut_trees,eggs,eggs_raw,flowers,larvae,...}.yaml` presentes. Sementes externas convertidas em CPU (sha do encoder): flowers `18e8452036421b1a` (→ L6_kappa), coconut_trees `b0f5ae27fac1637b` (→ L6_kappa), eggs/eggs_raw `7e72fbae9bb744c9`, larvae `5d7a9b1dd914cffb`. Se o athena usou outro L6 (o antigo, pela loss), o sha não bate e o validador bloqueia.
