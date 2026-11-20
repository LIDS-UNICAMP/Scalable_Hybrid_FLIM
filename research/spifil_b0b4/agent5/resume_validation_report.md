# Validação de resume — buscas migradas da GPU 3 do athena → Jaci

_Atualizado em 2026-11-16 22:26:16 (Jaci). Gerado por `research/spifil_b0b4/agent5/tools/staging_watch.py` (validador: `tools/validate_resume.py`, CPU, sem treino). Disco /dados livre: 576G; GPU usada/livre (MiB): 85551, 11793._

**Nada foi publicado nem lançado.** Os comandos abaixo são para o coordenador executar, na ordem de prioridade, um por vez (o waiter já respeita o gate de memória e o espaçamento de 15 min).

## Notas do Agente 5 (Jaci) — ler antes de executar

- **Arquivos do athena nesta pasta:** o `jaci_resume_queue.json` original do athena (que tinha sobrescrito o meu às 22:22) foi preservado como `jaci_resume_queue_athena_proposta.json`. O `jaci_resume_queue.json` atual é o da Jaci (validado). Os demais arquivos do athena (`gpu3_migration_manifest.csv`, `checkpoint_integrity_report.md`, `infrastructure_inventory.md`, `execution_status.md`, `athena_gpu3_research_queue.json`) não foram tocados. O inventário da Jaci está em `infrastructure_inventory_jaci.md`.
- **Manifesto conferido:** em todas as 6 buscas com estado, o nº de jobs e o md5 do `search_state.json` na Jaci são iguais aos do `gpu3_migration_manifest.csv` (athena = jaci). Além disso: todos os checkpoints referenciados existem, as amostras carregam em CPU, os dados batem bit a bit (data_sha256) e o replay reproduz todos os jobs.
- **`cov_eigh_workers`:** os comandos da Jaci NÃO passam `--set eval.svm_fitness.cov_eigh_workers=1`; usam o default (torch 2.7; bitwise igual segundo o athena). Isso não afeta o replay (a seção search é idêntica).
- **Budgets = os originais do athena** (que estão no state e no inventário do athena): hybrid coconut 34/44; hybrid eggs e eggs_raw 12/14; LEMONADE coconut e flowers 12/14. **O hybrid coconut não pode baixar o budget**: com 20 e 24 o replay dá ReplayMismatch no job 5. O pico real foi 25 GB, então o gate é de 33 GB livres.
- **LEMONADE coconut_trees:** o replay gera 1 job extra *cached* (job 24, c00002 f5, custo zero). Isso é normal (resultado em cache), não é divergência.
- **Controller W&B:** o id é NOVO na Jaci em todas (o `artifacts` do athena é symlink, então o `resolve()` muda). O state continua de onde parou; só o painel do controller recomeça.
- **Semeadas (5, nunca iniciaram):** entram como buscas NOVAS, sem publicação. O waiter espera o `done` da busca-mãe migrada na Jaci (regra F3) e o ckpt da semente, depois o gate. Budget 24/28 (padrão F3 do athena: "todas 24/28"); confira com `athena:/exps/moliveira/queues/gpu3_migracao/waiters/seedq_*.txt`, que não está visível da Jaci. Nenhum `seedq_*` da Jaci aponta para esses destinos, então não há conflito.
- **Seed43:** flowers (2 jobs, validada) tem destino dentro de `artifacts/modehb_hybrid/flowers`, que já existe (o seed42 roda lá) → **CONFLITO pela regra**. Deixei um comando ALTERNATIVO, que publica só `state/seed43`. larvae seed43 não tem estado (começa do zero); o waiter dele é opcional. Os dois são a reserva F4 e só rodam com OK.
- **Ordem sugerida** (todos os waiters dividem o flock `.last_start`, com 15 min entre começos): hybrid eggs → hybrid eggs_raw → LEMONADE flowers → LEMONADE coconut → hybrid coconut (gate 33 GB). As semeadas só começam depois do `done` das mães.
- A GPU da Jaci está saturada agora (89–96 GB usados, vários retries de OOM). Os gates (22–33 GB livres) não devem abrir até alguma busca terminar.

## O que cada validação checa
1. `search_state.json` carrega; nº de jobs, status, brackets, último job.
2. Manifesto (marcador `.ok`, `*manifest*` do staging, `gpu3_migration_manifest.csv`): nº de jobs e sha256 (se houver).
3. Checkpoints de TODOS os jobs ok existem; `torch.load` em CPU (`weights_only=False`) de 5 amostras + todos os de fidelidade máxima.
4. Config viva da Jaci (`search_v3*.yaml` + mesmos `--set`) = `state["config"]` (seção search); `config_resolved.yaml` comparado.
5. Dataset na Jaci: yaml, imagens, split (sha256), **identidade de dados** (hash de todas as imagens de treino/val, a mesma do Evaluator) = `data_sha256` dos jobs; `spifil_pool_sizes` iguais.
6. Semente externa (semeadas): arquivos presentes; sha do encoder convertido na Jaci = o do athena.
7. Assinatura do fitness (C100_kw1) e id do controller W&B que a Jaci vai usar.
8. **Replay a seco**: `MODEHB.run(resume=True)` com avaliador-stub sobre uma cópia do state — tem de reproduzir todos os jobs (sem `ReplayMismatch`) e parar no próximo job novo.

## Resumo

| prio | busca | migração | resume | jobs | fid máx | último job | erros/conflitos |
|---|---|---|---|---|---|---|---|
| 1 | modehb_hybrid/eggs/seed42 | validada_pronta_para_publicar | replay_ok | 55 | 200.0 | 54 c00035 f5.0 | — |
| 2 | modehb_hybrid/eggs_raw/seed42 | validada_pronta_para_publicar | replay_ok | 51 | 200.0 | 50 c00031 f5.0 | — |
| 3 | modehb_lemonade/flowers/seed42 | validada_pronta_para_publicar | replay_ok | 36 | 200.0 | 35 c00016 f5.0 | — |
| 4 | modehb_lemonade/coconut_trees/seed42 | validada_pronta_para_publicar | replay_ok | 24 | 200.0 | 23 c00005 f5.0 | — |
| 5 | modehb_hybrid/coconut_trees/seed42 | validada_pronta_para_publicar | replay_ok | 39 | 200.0 | 38 c00019 f5.0 | — |
| 10 | modehb_seeded/eggs/seed42 | nova (nada a migrar) | nao_aplicavel | 0 | — | — | — |
| 11 | modehb_seeded/eggs_raw/seed42 | nova (nada a migrar) | nao_aplicavel | 0 | — | — | — |
| 12 | modehb_lemonade_seeded/flowers/seed42 | nova (nada a migrar) | nao_aplicavel | 0 | — | — | — |
| 13 | modehb_lemonade_seeded/coconut_trees/seed42 | nova (nada a migrar) | nao_aplicavel | 0 | — | — | — |
| 14 | modehb_seeded/coconut_trees/seed42 | nova (nada a migrar) | nao_aplicavel | 0 | — | — | — |
| 20 | modehb_hybrid/flowers/seed43 | validada_CONFLITO | replay_ok | 2 | 15.0 | 1 c00001 f15.0 | destino artifacts/modehb_hybrid/flowers JÁ EXISTE na Jaci (sem state/seed43) |
| 21 | modehb_hybrid/larvae/seed43 | sem_state (0 jobs: resume = começar do zero) | nao_aplicavel | 0 | — | — | destino artifacts/modehb_hybrid/larvae JÁ EXISTE na Jaci |

## modehb_hybrid/eggs/seed42 — MO-DEHB hybrid (SPiFiL) — **validada_pronta_para_publicar**

- jobs 55 {'ok': 55}, brackets concluídos 0, done=False, GPU-h 22.85; fid máx 200.0 (4 ckpts); ckpts carregados em CPU 9, ausentes 0; staging 3.06 GB.
- `search_state.json` sha256 `17cdf6501968df367d7ab1812b433fedceb043dc3de7a38b7aaf42e98309bb3d` (gravado 2026-11-16 22:07:18); jobs failed: nenhum.
- resume: replay_ok; próximo job {'next_job': 55, 'next_fidelity': 5.0, 'next_role': 'search', 'next_cand': 'c00036'}.
- controller Jaci `ctrl-e5f1c41edb782ce7` (C100_kw1); recursos {'memory_budget_gb': 12, 'gpu_process_cap_gb': 14, 'peak_gpu_reserved_gb_max': 11.4, 'gate_free_mib': 22528}.
- config: `{'config': 'ga_search/config/modehb/search_v3.yaml', 'dataset': 'eggs', 'seed': 42, 'sets': ['output_dir=artifacts/modehb_hybrid/eggs/state/seed42', 'search.stratified.memory_budget_gb=12', 'gpu_process_cap_gb=14']}`

Publicação (atômica, só se o destino não existir):
```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM && test -f artifacts/_staging_gpu3/modehb_hybrid_eggs.ok && test ! -e artifacts/modehb_hybrid/eggs && mkdir -p artifacts/modehb_hybrid && mv -T --no-clobber artifacts/_staging_gpu3/modehb_hybrid/eggs artifacts/modehb_hybrid/eggs && ls artifacts/modehb_hybrid/eggs/state/seed*/search_state.json >/dev/null && touch artifacts/migration_markers/modehb_hybrid_eggs.gpu3.ok && echo "PUBLISH modehb_hybrid/eggs $(date)" >> artifacts/migration_from_athena.log || echo 'NAO PUBLICADO: destino existe ou staging incompleto'
```

Waiter (tmux `g3_hyb_eggs_s42`; rodar DEPOIS da publicação):
```bash
tmux new-session -d -s g3_hyb_eggs_s42 bash -c "$(cat <<'G3EOF'
cd /dados/home/moliveira/Scalable_Hybrid_FLIM
NAME=g3_hyb_eggs_s42; OUT=artifacts/modehb_hybrid/eggs/state/seed42; LOG=artifacts/modehb_hybrid/eggs/logs/search_s42.log; GATE=22528
test -f "$OUT/search_state.json" || { echo "ABORT $NAME: $OUT/search_state.json ausente (publicar antes) $(date)" >> artifacts/migration_from_athena.log; exit 1; }
ok=0
while true; do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$f" -ge $GATE ]; then ok=$((ok+1)); else ok=0; fi
  if [ $ok -ge 3 ] && flock -n artifacts/migration_markers/.last_start.flock -c 'age=$(( $(date +%s) - $(stat -c %Y artifacts/migration_markers/.last_start) )); [ $age -ge 900 ] && touch artifacts/migration_markers/.last_start'; then break; fi
  sleep $((90 + RANDOM % 60))
done
echo "START $NAME $(date)" >> artifacts/migration_from_athena.log
ulimit -n 65536
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$(dirname $LOG)"
for i in 1 2 3 4 5; do
  if pgrep -f "^[^ ]*python[^ ]* -m ga_search.modehb.run_search .*output_dir=$OUT( |\$)" >/dev/null; then
    echo "DUP $NAME: outro run_search já usa $OUT (tentativa $i adiada) $(date)" >> artifacts/migration_from_athena.log; sleep 600; continue
  fi
  flock -n artifacts/migration_markers/modehb_hybrid_eggs_s42.ctrl.lock /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.modehb.run_search --config ga_search/config/modehb/search_v3.yaml --dataset eggs --seed 42 --set output_dir=artifacts/modehb_hybrid/eggs/state/seed42 --set search.stratified.memory_budget_gb=12 --set gpu_process_cap_gb=14 >> $LOG 2>&1 && break
  echo "RETRY $NAME tentativa $i rc=$? $(date)" >> artifacts/migration_from_athena.log
  sleep 300
done
echo "END $NAME $(date)" >> artifacts/migration_from_athena.log
G3EOF
)"
```

## modehb_hybrid/eggs_raw/seed42 — MO-DEHB hybrid (SPiFiL) — **validada_pronta_para_publicar**

- jobs 51 {'ok': 51}, brackets concluídos 0, done=False, GPU-h 22.58; fid máx 200.0 (4 ckpts); ckpts carregados em CPU 9, ausentes 0; staging 2.76 GB.
- `search_state.json` sha256 `8b3284b1170010cfe7a695945421948bb26d5b6dcccd2851bd0f4f0c88de3eb7` (gravado 2026-11-16 22:12:04); jobs failed: nenhum.
- resume: replay_ok; próximo job {'next_job': 51, 'next_fidelity': 5.0, 'next_role': 'search', 'next_cand': 'c00032'}.
- controller Jaci `ctrl-26e9134cfb5e9fb7` (C100_kw1); recursos {'memory_budget_gb': 12, 'gpu_process_cap_gb': 14, 'peak_gpu_reserved_gb_max': 9.47, 'gate_free_mib': 22528}.
- config: `{'config': 'ga_search/config/modehb/search_v3.yaml', 'dataset': 'eggs_raw', 'seed': 42, 'sets': ['output_dir=artifacts/modehb_hybrid/eggs_raw/state/seed42', 'search.stratified.memory_budget_gb=12', 'gpu_process_cap_gb=14']}`

Publicação (atômica, só se o destino não existir):
```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM && test -f artifacts/_staging_gpu3/modehb_hybrid_eggs_raw.ok && test ! -e artifacts/modehb_hybrid/eggs_raw && mkdir -p artifacts/modehb_hybrid && mv -T --no-clobber artifacts/_staging_gpu3/modehb_hybrid/eggs_raw artifacts/modehb_hybrid/eggs_raw && ls artifacts/modehb_hybrid/eggs_raw/state/seed*/search_state.json >/dev/null && touch artifacts/migration_markers/modehb_hybrid_eggs_raw.gpu3.ok && echo "PUBLISH modehb_hybrid/eggs_raw $(date)" >> artifacts/migration_from_athena.log || echo 'NAO PUBLICADO: destino existe ou staging incompleto'
```

Waiter (tmux `g3_hyb_eggs_raw_s42`; rodar DEPOIS da publicação):
```bash
tmux new-session -d -s g3_hyb_eggs_raw_s42 bash -c "$(cat <<'G3EOF'
cd /dados/home/moliveira/Scalable_Hybrid_FLIM
NAME=g3_hyb_eggs_raw_s42; OUT=artifacts/modehb_hybrid/eggs_raw/state/seed42; LOG=artifacts/modehb_hybrid/eggs_raw/logs/search_s42.log; GATE=22528
test -f "$OUT/search_state.json" || { echo "ABORT $NAME: $OUT/search_state.json ausente (publicar antes) $(date)" >> artifacts/migration_from_athena.log; exit 1; }
ok=0
while true; do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$f" -ge $GATE ]; then ok=$((ok+1)); else ok=0; fi
  if [ $ok -ge 3 ] && flock -n artifacts/migration_markers/.last_start.flock -c 'age=$(( $(date +%s) - $(stat -c %Y artifacts/migration_markers/.last_start) )); [ $age -ge 900 ] && touch artifacts/migration_markers/.last_start'; then break; fi
  sleep $((90 + RANDOM % 60))
done
echo "START $NAME $(date)" >> artifacts/migration_from_athena.log
ulimit -n 65536
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$(dirname $LOG)"
for i in 1 2 3 4 5; do
  if pgrep -f "^[^ ]*python[^ ]* -m ga_search.modehb.run_search .*output_dir=$OUT( |\$)" >/dev/null; then
    echo "DUP $NAME: outro run_search já usa $OUT (tentativa $i adiada) $(date)" >> artifacts/migration_from_athena.log; sleep 600; continue
  fi
  flock -n artifacts/migration_markers/modehb_hybrid_eggs_raw_s42.ctrl.lock /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.modehb.run_search --config ga_search/config/modehb/search_v3.yaml --dataset eggs_raw --seed 42 --set output_dir=artifacts/modehb_hybrid/eggs_raw/state/seed42 --set search.stratified.memory_budget_gb=12 --set gpu_process_cap_gb=14 >> $LOG 2>&1 && break
  echo "RETRY $NAME tentativa $i rc=$? $(date)" >> artifacts/migration_from_athena.log
  sleep 300
done
echo "END $NAME $(date)" >> artifacts/migration_from_athena.log
G3EOF
)"
```

## modehb_lemonade/flowers/seed42 — LEMONADE — **validada_pronta_para_publicar**

- jobs 36 {'ok': 36}, brackets concluídos 0, done=False, GPU-h 13.44; fid máx 200.0 (4 ckpts); ckpts carregados em CPU 9, ausentes 0; staging 0.71 GB.
- `search_state.json` sha256 `901ba50018c53b4a3df696b34f7b529f71dfcdbff0229d465dfc802e29c732a9` (gravado 2026-11-16 22:05:34); jobs failed: nenhum.
- resume: replay_ok; próximo job {'next_job': 36, 'next_fidelity': 5.0, 'next_role': 'search', 'next_cand': 'c00017'}.
- controller Jaci `ctrl-a50a46a30b8058ed` (C100_kw1); recursos {'memory_budget_gb': 12, 'gpu_process_cap_gb': 14, 'peak_gpu_reserved_gb_max': 9.64, 'gate_free_mib': 22528}.
- config: `{'config': 'ga_search/config/modehb/search_v3_lemonade.yaml', 'dataset': 'flowers', 'seed': 42, 'sets': ['output_dir=artifacts/modehb_lemonade/flowers/state/seed42', 'search.stratified.memory_budget_gb=12', 'gpu_process_cap_gb=14']}`

Publicação (atômica, só se o destino não existir):
```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM && test -f artifacts/_staging_gpu3/modehb_lemonade_flowers.ok && test ! -e artifacts/modehb_lemonade/flowers && mkdir -p artifacts/modehb_lemonade && mv -T --no-clobber artifacts/_staging_gpu3/modehb_lemonade/flowers artifacts/modehb_lemonade/flowers && ls artifacts/modehb_lemonade/flowers/state/seed*/search_state.json >/dev/null && touch artifacts/migration_markers/modehb_lemonade_flowers.gpu3.ok && echo "PUBLISH modehb_lemonade/flowers $(date)" >> artifacts/migration_from_athena.log || echo 'NAO PUBLICADO: destino existe ou staging incompleto'
```

Waiter (tmux `g3_lem_flowers_s42`; rodar DEPOIS da publicação):
```bash
tmux new-session -d -s g3_lem_flowers_s42 bash -c "$(cat <<'G3EOF'
cd /dados/home/moliveira/Scalable_Hybrid_FLIM
NAME=g3_lem_flowers_s42; OUT=artifacts/modehb_lemonade/flowers/state/seed42; LOG=artifacts/modehb_lemonade/flowers/logs/search_s42.log; GATE=22528
test -f "$OUT/search_state.json" || { echo "ABORT $NAME: $OUT/search_state.json ausente (publicar antes) $(date)" >> artifacts/migration_from_athena.log; exit 1; }
ok=0
while true; do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$f" -ge $GATE ]; then ok=$((ok+1)); else ok=0; fi
  if [ $ok -ge 3 ] && flock -n artifacts/migration_markers/.last_start.flock -c 'age=$(( $(date +%s) - $(stat -c %Y artifacts/migration_markers/.last_start) )); [ $age -ge 900 ] && touch artifacts/migration_markers/.last_start'; then break; fi
  sleep $((90 + RANDOM % 60))
done
echo "START $NAME $(date)" >> artifacts/migration_from_athena.log
ulimit -n 65536
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$(dirname $LOG)"
for i in 1 2 3 4 5; do
  if pgrep -f "^[^ ]*python[^ ]* -m ga_search.modehb.run_search .*output_dir=$OUT( |\$)" >/dev/null; then
    echo "DUP $NAME: outro run_search já usa $OUT (tentativa $i adiada) $(date)" >> artifacts/migration_from_athena.log; sleep 600; continue
  fi
  flock -n artifacts/migration_markers/modehb_lemonade_flowers_s42.ctrl.lock /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.modehb.run_search --config ga_search/config/modehb/search_v3_lemonade.yaml --dataset flowers --seed 42 --set output_dir=artifacts/modehb_lemonade/flowers/state/seed42 --set search.stratified.memory_budget_gb=12 --set gpu_process_cap_gb=14 >> $LOG 2>&1 && break
  echo "RETRY $NAME tentativa $i rc=$? $(date)" >> artifacts/migration_from_athena.log
  sleep 300
done
echo "END $NAME $(date)" >> artifacts/migration_from_athena.log
G3EOF
)"
```

## modehb_lemonade/coconut_trees/seed42 — LEMONADE — **validada_pronta_para_publicar**

- jobs 24 {'ok': 24}, brackets concluídos 0, done=False, GPU-h 13.77; fid máx 200.0 (4 ckpts); ckpts carregados em CPU 9, ausentes 0; staging 1.17 GB.
- `search_state.json` sha256 `ae2fb8b46f614b51a93ece0dc7883921241dd82502186337b07bf06a28bbc67b` (gravado 2026-11-16 22:06:28); jobs failed: nenhum.
- resume: replay_ok; próximo job {'next_job': 25, 'next_fidelity': 5.0, 'next_role': 'search', 'next_cand': 'c00006'}.
- controller Jaci `ctrl-c57ebded4a8789c3` (C100_kw1); recursos {'memory_budget_gb': 12, 'gpu_process_cap_gb': 14, 'peak_gpu_reserved_gb_max': 7.98, 'gate_free_mib': 22528}.
- config: `{'config': 'ga_search/config/modehb/search_v3_lemonade.yaml', 'dataset': 'coconut_trees', 'seed': 42, 'sets': ['output_dir=artifacts/modehb_lemonade/coconut_trees/state/seed42', 'search.stratified.memory_budget_gb=12', 'gpu_process_cap_gb=14']}`

Publicação (atômica, só se o destino não existir):
```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM && test -f artifacts/_staging_gpu3/modehb_lemonade_coconut_trees.ok && test ! -e artifacts/modehb_lemonade/coconut_trees && mkdir -p artifacts/modehb_lemonade && mv -T --no-clobber artifacts/_staging_gpu3/modehb_lemonade/coconut_trees artifacts/modehb_lemonade/coconut_trees && ls artifacts/modehb_lemonade/coconut_trees/state/seed*/search_state.json >/dev/null && touch artifacts/migration_markers/modehb_lemonade_coconut_trees.gpu3.ok && echo "PUBLISH modehb_lemonade/coconut_trees $(date)" >> artifacts/migration_from_athena.log || echo 'NAO PUBLICADO: destino existe ou staging incompleto'
```

Waiter (tmux `g3_lem_coconut_trees_s42`; rodar DEPOIS da publicação):
```bash
tmux new-session -d -s g3_lem_coconut_trees_s42 bash -c "$(cat <<'G3EOF'
cd /dados/home/moliveira/Scalable_Hybrid_FLIM
NAME=g3_lem_coconut_trees_s42; OUT=artifacts/modehb_lemonade/coconut_trees/state/seed42; LOG=artifacts/modehb_lemonade/coconut_trees/logs/search_s42.log; GATE=22528
test -f "$OUT/search_state.json" || { echo "ABORT $NAME: $OUT/search_state.json ausente (publicar antes) $(date)" >> artifacts/migration_from_athena.log; exit 1; }
ok=0
while true; do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$f" -ge $GATE ]; then ok=$((ok+1)); else ok=0; fi
  if [ $ok -ge 3 ] && flock -n artifacts/migration_markers/.last_start.flock -c 'age=$(( $(date +%s) - $(stat -c %Y artifacts/migration_markers/.last_start) )); [ $age -ge 900 ] && touch artifacts/migration_markers/.last_start'; then break; fi
  sleep $((90 + RANDOM % 60))
done
echo "START $NAME $(date)" >> artifacts/migration_from_athena.log
ulimit -n 65536
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$(dirname $LOG)"
for i in 1 2 3 4 5; do
  if pgrep -f "^[^ ]*python[^ ]* -m ga_search.modehb.run_search .*output_dir=$OUT( |\$)" >/dev/null; then
    echo "DUP $NAME: outro run_search já usa $OUT (tentativa $i adiada) $(date)" >> artifacts/migration_from_athena.log; sleep 600; continue
  fi
  flock -n artifacts/migration_markers/modehb_lemonade_coconut_trees_s42.ctrl.lock /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.modehb.run_search --config ga_search/config/modehb/search_v3_lemonade.yaml --dataset coconut_trees --seed 42 --set output_dir=artifacts/modehb_lemonade/coconut_trees/state/seed42 --set search.stratified.memory_budget_gb=12 --set gpu_process_cap_gb=14 >> $LOG 2>&1 && break
  echo "RETRY $NAME tentativa $i rc=$? $(date)" >> artifacts/migration_from_athena.log
  sleep 300
done
echo "END $NAME $(date)" >> artifacts/migration_from_athena.log
G3EOF
)"
```

## modehb_hybrid/coconut_trees/seed42 — MO-DEHB hybrid (SPiFiL) — **validada_pronta_para_publicar**

- jobs 39 {'ok': 38, 'failed': 1}, brackets concluídos 0, done=False, GPU-h 23.24; fid máx 200.0 (4 ckpts); ckpts carregados em CPU 9, ausentes 0; staging 1.94 GB.
- `search_state.json` sha256 `db0813424f1007b4bd2c3823f733abf0d2fb2b39d57514102498b36a4423523a` (gravado 2026-11-16 22:04:30); jobs failed: [[31, 'c00012', 5.0, 'params 3541364 > cap 3500000']].
- resume: replay_ok; próximo job {'next_job': 39, 'next_fidelity': 5.0, 'next_role': 'search', 'next_cand': 'c00020'}.
- controller Jaci `ctrl-3f0ecd85783def06` (C100_kw1); recursos {'memory_budget_gb': 34, 'gpu_process_cap_gb': 44, 'peak_gpu_reserved_gb_max': 25.07, 'gate_free_mib': 33863}.
- config: `{'config': 'ga_search/config/modehb/search_v3.yaml', 'dataset': 'coconut_trees', 'seed': 42, 'sets': ['output_dir=artifacts/modehb_hybrid/coconut_trees/state/seed42', 'search.stratified.memory_budget_gb=34', 'gpu_process_cap_gb=44']}`

Publicação (atômica, só se o destino não existir):
```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM && test -f artifacts/_staging_gpu3/modehb_hybrid_coconut_trees.ok && test ! -e artifacts/modehb_hybrid/coconut_trees && mkdir -p artifacts/modehb_hybrid && mv -T --no-clobber artifacts/_staging_gpu3/modehb_hybrid/coconut_trees artifacts/modehb_hybrid/coconut_trees && ls artifacts/modehb_hybrid/coconut_trees/state/seed*/search_state.json >/dev/null && touch artifacts/migration_markers/modehb_hybrid_coconut_trees.gpu3.ok && echo "PUBLISH modehb_hybrid/coconut_trees $(date)" >> artifacts/migration_from_athena.log || echo 'NAO PUBLICADO: destino existe ou staging incompleto'
```

Waiter (tmux `g3_hyb_coconut_trees_s42`; rodar DEPOIS da publicação):
```bash
tmux new-session -d -s g3_hyb_coconut_trees_s42 bash -c "$(cat <<'G3EOF'
cd /dados/home/moliveira/Scalable_Hybrid_FLIM
NAME=g3_hyb_coconut_trees_s42; OUT=artifacts/modehb_hybrid/coconut_trees/state/seed42; LOG=artifacts/modehb_hybrid/coconut_trees/logs/search_s42.log; GATE=33863
test -f "$OUT/search_state.json" || { echo "ABORT $NAME: $OUT/search_state.json ausente (publicar antes) $(date)" >> artifacts/migration_from_athena.log; exit 1; }
ok=0
while true; do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$f" -ge $GATE ]; then ok=$((ok+1)); else ok=0; fi
  if [ $ok -ge 3 ] && flock -n artifacts/migration_markers/.last_start.flock -c 'age=$(( $(date +%s) - $(stat -c %Y artifacts/migration_markers/.last_start) )); [ $age -ge 900 ] && touch artifacts/migration_markers/.last_start'; then break; fi
  sleep $((90 + RANDOM % 60))
done
echo "START $NAME $(date)" >> artifacts/migration_from_athena.log
ulimit -n 65536
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$(dirname $LOG)"
for i in 1 2 3 4 5; do
  if pgrep -f "^[^ ]*python[^ ]* -m ga_search.modehb.run_search .*output_dir=$OUT( |\$)" >/dev/null; then
    echo "DUP $NAME: outro run_search já usa $OUT (tentativa $i adiada) $(date)" >> artifacts/migration_from_athena.log; sleep 600; continue
  fi
  flock -n artifacts/migration_markers/modehb_hybrid_coconut_trees_s42.ctrl.lock /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.modehb.run_search --config ga_search/config/modehb/search_v3.yaml --dataset coconut_trees --seed 42 --set output_dir=artifacts/modehb_hybrid/coconut_trees/state/seed42 --set search.stratified.memory_budget_gb=34 --set gpu_process_cap_gb=44 >> $LOG 2>&1 && break
  echo "RETRY $NAME tentativa $i rc=$? $(date)" >> artifacts/migration_from_athena.log
  sleep 300
done
echo "END $NAME $(date)" >> artifacts/migration_from_athena.log
G3EOF
)"
```

## modehb_seeded/eggs/seed42 — MO-DEHB hybrid semeada — **nova (nada a migrar)**

- origem: athena GPU 3 (waiter seedq_* que nunca iniciou; sem estado)
- depende_de: done de artifacts/modehb_hybrid/eggs/state/seed42/search_state.json (busca-mãe migrada, regra F3: semeada só depois da atual)
- semente: `{'checkpoint': 'artifacts/external_ssl_pequeno/eggs_split1_pct100/lejepa/checkpoints/best_kappa.ckpt', 'realpath': '/dados/home/moliveira/exps/spifil_growth/ssl_pequeno_lejepa/Eggs_split1_pct100/L6/flim-ssl/1mj48qxh/checkpoints/best_kappa.ckpt', 'arch': 'artifacts/external_ssl_pequeno/eggs_split1_pct100/L0/architecture.json'}`
- config: `{'config': 'ga_search/config/modehb/search_v3_seeded.yaml', 'dataset': 'eggs', 'seed': 42, 'sets': ['output_dir=artifacts/modehb_seeded/eggs/state/seed42', 'search.stratified.memory_budget_gb=24', 'gpu_process_cap_gb=28']}`
- recursos: `{'memory_budget_gb': 24, 'gpu_process_cap_gb': 28, 'gate_free_mib': 28672}`
- obs: budget 24/28 = padrão F3 (athena: 'todas 24/28'); o comando original está em athena:/exps/moliveira/queues/gpu3_migracao/waiters/seedq_modehb_seeded_eggs.txt — conferir antes

Waiter (tmux `g3_seedhyb_eggs_s42`; não precisa publicar nada):
```bash
tmux new-session -d -s g3_seedhyb_eggs_s42 bash -c "$(cat <<'G3EOF'
cd /dados/home/moliveira/Scalable_Hybrid_FLIM
NAME=g3_seedhyb_eggs_s42; OUT=artifacts/modehb_seeded/eggs/state/seed42; LOG=artifacts/modehb_seeded/eggs/logs/search_s42.log; GATE=28672
# busca NOVA (sem estado no athena): espera o done da busca-mãe na Jaci + o ckpt da semente
until /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -c "import json,sys; sys.exit(0 if json.load(open('artifacts/modehb_hybrid/eggs/state/seed42/search_state.json')).get('done') else 1)" 2>/dev/null && [ -f artifacts/external_ssl_pequeno/eggs_split1_pct100/lejepa/checkpoints/best_kappa.ckpt ]; do sleep 300; done
ok=0
while true; do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$f" -ge $GATE ]; then ok=$((ok+1)); else ok=0; fi
  if [ $ok -ge 3 ] && flock -n artifacts/migration_markers/.last_start.flock -c 'age=$(( $(date +%s) - $(stat -c %Y artifacts/migration_markers/.last_start) )); [ $age -ge 900 ] && touch artifacts/migration_markers/.last_start'; then break; fi
  sleep $((90 + RANDOM % 60))
done
echo "START $NAME $(date)" >> artifacts/migration_from_athena.log
ulimit -n 65536
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$(dirname $LOG)"
for i in 1 2 3 4 5; do
  if pgrep -f "^[^ ]*python[^ ]* -m ga_search.modehb.run_search .*output_dir=$OUT( |\$)" >/dev/null; then
    echo "DUP $NAME: outro run_search já usa $OUT (tentativa $i adiada) $(date)" >> artifacts/migration_from_athena.log; sleep 600; continue
  fi
  flock -n artifacts/migration_markers/modehb_seeded_eggs_s42.ctrl.lock /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.modehb.run_search --config ga_search/config/modehb/search_v3_seeded.yaml --dataset eggs --seed 42 --set output_dir=artifacts/modehb_seeded/eggs/state/seed42 --set search.stratified.memory_budget_gb=24 --set gpu_process_cap_gb=28 >> $LOG 2>&1 && break
  echo "RETRY $NAME tentativa $i rc=$? $(date)" >> artifacts/migration_from_athena.log
  sleep 300
done
echo "END $NAME $(date)" >> artifacts/migration_from_athena.log
G3EOF
)"
```

## modehb_seeded/eggs_raw/seed42 — MO-DEHB hybrid semeada — **nova (nada a migrar)**

- origem: athena GPU 3 (waiter seedq_* que nunca iniciou; sem estado)
- depende_de: done de artifacts/modehb_hybrid/eggs_raw/state/seed42/search_state.json (busca-mãe migrada, regra F3: semeada só depois da atual)
- semente: `{'checkpoint': 'artifacts/external_ssl_pequeno/eggs_raw_split1_pct100/lejepa/checkpoints/best_kappa.ckpt', 'realpath': '/dados/home/moliveira/exps/spifil_growth/ssl_pequeno_lejepa/Eggs_split1_pct100/L6/flim-ssl/1mj48qxh/checkpoints/best_kappa.ckpt', 'arch': 'artifacts/external_ssl_pequeno/eggs_raw_split1_pct100/L0/architecture.json'}`
- config: `{'config': 'ga_search/config/modehb/search_v3_seeded.yaml', 'dataset': 'eggs_raw', 'seed': 42, 'sets': ['output_dir=artifacts/modehb_seeded/eggs_raw/state/seed42', 'search.stratified.memory_budget_gb=24', 'gpu_process_cap_gb=28']}`
- recursos: `{'memory_budget_gb': 24, 'gpu_process_cap_gb': 28, 'gate_free_mib': 28672}`
- obs: budget 24/28 = padrão F3 (athena: 'todas 24/28'); o comando original está em athena:/exps/moliveira/queues/gpu3_migracao/waiters/seedq_modehb_seeded_eggs_raw.txt — conferir antes

Waiter (tmux `g3_seedhyb_eggs_raw_s42`; não precisa publicar nada):
```bash
tmux new-session -d -s g3_seedhyb_eggs_raw_s42 bash -c "$(cat <<'G3EOF'
cd /dados/home/moliveira/Scalable_Hybrid_FLIM
NAME=g3_seedhyb_eggs_raw_s42; OUT=artifacts/modehb_seeded/eggs_raw/state/seed42; LOG=artifacts/modehb_seeded/eggs_raw/logs/search_s42.log; GATE=28672
# busca NOVA (sem estado no athena): espera o done da busca-mãe na Jaci + o ckpt da semente
until /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -c "import json,sys; sys.exit(0 if json.load(open('artifacts/modehb_hybrid/eggs_raw/state/seed42/search_state.json')).get('done') else 1)" 2>/dev/null && [ -f artifacts/external_ssl_pequeno/eggs_raw_split1_pct100/lejepa/checkpoints/best_kappa.ckpt ]; do sleep 300; done
ok=0
while true; do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$f" -ge $GATE ]; then ok=$((ok+1)); else ok=0; fi
  if [ $ok -ge 3 ] && flock -n artifacts/migration_markers/.last_start.flock -c 'age=$(( $(date +%s) - $(stat -c %Y artifacts/migration_markers/.last_start) )); [ $age -ge 900 ] && touch artifacts/migration_markers/.last_start'; then break; fi
  sleep $((90 + RANDOM % 60))
done
echo "START $NAME $(date)" >> artifacts/migration_from_athena.log
ulimit -n 65536
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$(dirname $LOG)"
for i in 1 2 3 4 5; do
  if pgrep -f "^[^ ]*python[^ ]* -m ga_search.modehb.run_search .*output_dir=$OUT( |\$)" >/dev/null; then
    echo "DUP $NAME: outro run_search já usa $OUT (tentativa $i adiada) $(date)" >> artifacts/migration_from_athena.log; sleep 600; continue
  fi
  flock -n artifacts/migration_markers/modehb_seeded_eggs_raw_s42.ctrl.lock /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.modehb.run_search --config ga_search/config/modehb/search_v3_seeded.yaml --dataset eggs_raw --seed 42 --set output_dir=artifacts/modehb_seeded/eggs_raw/state/seed42 --set search.stratified.memory_budget_gb=24 --set gpu_process_cap_gb=28 >> $LOG 2>&1 && break
  echo "RETRY $NAME tentativa $i rc=$? $(date)" >> artifacts/migration_from_athena.log
  sleep 300
done
echo "END $NAME $(date)" >> artifacts/migration_from_athena.log
G3EOF
)"
```

## modehb_lemonade_seeded/flowers/seed42 — LEMONADE semeada — **nova (nada a migrar)**

- origem: athena GPU 3 (waiter seedq_* que nunca iniciou; sem estado)
- depende_de: done de artifacts/modehb_lemonade/flowers/state/seed42/search_state.json (busca-mãe migrada, regra F3: semeada só depois da atual)
- semente: `{'checkpoint': 'artifacts/external_ssl_pequeno/flowers_split1_pct100/lejepa/checkpoints/best_kappa.ckpt', 'realpath': '/dados/home/moliveira/exps/spifil_growth/ssl_pequeno_lejepa/flowers_split1_pct100/L6_kappa/checkpoints/best_kappa.ckpt', 'arch': 'artifacts/external_ssl_pequeno/flowers_split1_pct100/L0/architecture.json'}`
- config: `{'config': 'ga_search/config/modehb/search_v3_lemonade_seeded.yaml', 'dataset': 'flowers', 'seed': 42, 'sets': ['output_dir=artifacts/modehb_lemonade_seeded/flowers/state/seed42', 'search.stratified.memory_budget_gb=24', 'gpu_process_cap_gb=28']}`
- recursos: `{'memory_budget_gb': 24, 'gpu_process_cap_gb': 28, 'gate_free_mib': 28672}`
- obs: budget 24/28 = padrão F3 (athena: 'todas 24/28'); o comando original está em athena:/exps/moliveira/queues/gpu3_migracao/waiters/seedq_modehb_lemonade_seeded_flowers.txt — conferir antes

Waiter (tmux `g3_seedlem_flowers_s42`; não precisa publicar nada):
```bash
tmux new-session -d -s g3_seedlem_flowers_s42 bash -c "$(cat <<'G3EOF'
cd /dados/home/moliveira/Scalable_Hybrid_FLIM
NAME=g3_seedlem_flowers_s42; OUT=artifacts/modehb_lemonade_seeded/flowers/state/seed42; LOG=artifacts/modehb_lemonade_seeded/flowers/logs/search_s42.log; GATE=28672
# busca NOVA (sem estado no athena): espera o done da busca-mãe na Jaci + o ckpt da semente
until /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -c "import json,sys; sys.exit(0 if json.load(open('artifacts/modehb_lemonade/flowers/state/seed42/search_state.json')).get('done') else 1)" 2>/dev/null && [ -f artifacts/external_ssl_pequeno/flowers_split1_pct100/lejepa/checkpoints/best_kappa.ckpt ]; do sleep 300; done
ok=0
while true; do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$f" -ge $GATE ]; then ok=$((ok+1)); else ok=0; fi
  if [ $ok -ge 3 ] && flock -n artifacts/migration_markers/.last_start.flock -c 'age=$(( $(date +%s) - $(stat -c %Y artifacts/migration_markers/.last_start) )); [ $age -ge 900 ] && touch artifacts/migration_markers/.last_start'; then break; fi
  sleep $((90 + RANDOM % 60))
done
echo "START $NAME $(date)" >> artifacts/migration_from_athena.log
ulimit -n 65536
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$(dirname $LOG)"
for i in 1 2 3 4 5; do
  if pgrep -f "^[^ ]*python[^ ]* -m ga_search.modehb.run_search .*output_dir=$OUT( |\$)" >/dev/null; then
    echo "DUP $NAME: outro run_search já usa $OUT (tentativa $i adiada) $(date)" >> artifacts/migration_from_athena.log; sleep 600; continue
  fi
  flock -n artifacts/migration_markers/modehb_lemonade_seeded_flowers_s42.ctrl.lock /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.modehb.run_search --config ga_search/config/modehb/search_v3_lemonade_seeded.yaml --dataset flowers --seed 42 --set output_dir=artifacts/modehb_lemonade_seeded/flowers/state/seed42 --set search.stratified.memory_budget_gb=24 --set gpu_process_cap_gb=28 >> $LOG 2>&1 && break
  echo "RETRY $NAME tentativa $i rc=$? $(date)" >> artifacts/migration_from_athena.log
  sleep 300
done
echo "END $NAME $(date)" >> artifacts/migration_from_athena.log
G3EOF
)"
```

## modehb_lemonade_seeded/coconut_trees/seed42 — LEMONADE semeada — **nova (nada a migrar)**

- origem: athena GPU 3 (waiter seedq_* que nunca iniciou; sem estado)
- depende_de: done de artifacts/modehb_lemonade/coconut_trees/state/seed42/search_state.json (busca-mãe migrada, regra F3: semeada só depois da atual)
- semente: `{'checkpoint': 'artifacts/external_ssl_pequeno/coconut_trees_split1_pct100/lejepa/checkpoints/best_kappa.ckpt', 'realpath': '/dados/home/moliveira/exps/spifil_growth/ssl_pequeno_lejepa/coconut_trees_split1_pct100/L6_kappa/checkpoints/best_kappa.ckpt', 'arch': 'artifacts/external_ssl_pequeno/coconut_trees_split1_pct100/L0/architecture.json'}`
- config: `{'config': 'ga_search/config/modehb/search_v3_lemonade_seeded.yaml', 'dataset': 'coconut_trees', 'seed': 42, 'sets': ['output_dir=artifacts/modehb_lemonade_seeded/coconut_trees/state/seed42', 'search.stratified.memory_budget_gb=24', 'gpu_process_cap_gb=28']}`
- recursos: `{'memory_budget_gb': 24, 'gpu_process_cap_gb': 28, 'gate_free_mib': 28672}`
- obs: budget 24/28 = padrão F3 (athena: 'todas 24/28'); o comando original está em athena:/exps/moliveira/queues/gpu3_migracao/waiters/seedq_modehb_lemonade_seeded_coconut_trees.txt — conferir antes

Waiter (tmux `g3_seedlem_coconut_trees_s42`; não precisa publicar nada):
```bash
tmux new-session -d -s g3_seedlem_coconut_trees_s42 bash -c "$(cat <<'G3EOF'
cd /dados/home/moliveira/Scalable_Hybrid_FLIM
NAME=g3_seedlem_coconut_trees_s42; OUT=artifacts/modehb_lemonade_seeded/coconut_trees/state/seed42; LOG=artifacts/modehb_lemonade_seeded/coconut_trees/logs/search_s42.log; GATE=28672
# busca NOVA (sem estado no athena): espera o done da busca-mãe na Jaci + o ckpt da semente
until /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -c "import json,sys; sys.exit(0 if json.load(open('artifacts/modehb_lemonade/coconut_trees/state/seed42/search_state.json')).get('done') else 1)" 2>/dev/null && [ -f artifacts/external_ssl_pequeno/coconut_trees_split1_pct100/lejepa/checkpoints/best_kappa.ckpt ]; do sleep 300; done
ok=0
while true; do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$f" -ge $GATE ]; then ok=$((ok+1)); else ok=0; fi
  if [ $ok -ge 3 ] && flock -n artifacts/migration_markers/.last_start.flock -c 'age=$(( $(date +%s) - $(stat -c %Y artifacts/migration_markers/.last_start) )); [ $age -ge 900 ] && touch artifacts/migration_markers/.last_start'; then break; fi
  sleep $((90 + RANDOM % 60))
done
echo "START $NAME $(date)" >> artifacts/migration_from_athena.log
ulimit -n 65536
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$(dirname $LOG)"
for i in 1 2 3 4 5; do
  if pgrep -f "^[^ ]*python[^ ]* -m ga_search.modehb.run_search .*output_dir=$OUT( |\$)" >/dev/null; then
    echo "DUP $NAME: outro run_search já usa $OUT (tentativa $i adiada) $(date)" >> artifacts/migration_from_athena.log; sleep 600; continue
  fi
  flock -n artifacts/migration_markers/modehb_lemonade_seeded_coconut_trees_s42.ctrl.lock /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.modehb.run_search --config ga_search/config/modehb/search_v3_lemonade_seeded.yaml --dataset coconut_trees --seed 42 --set output_dir=artifacts/modehb_lemonade_seeded/coconut_trees/state/seed42 --set search.stratified.memory_budget_gb=24 --set gpu_process_cap_gb=28 >> $LOG 2>&1 && break
  echo "RETRY $NAME tentativa $i rc=$? $(date)" >> artifacts/migration_from_athena.log
  sleep 300
done
echo "END $NAME $(date)" >> artifacts/migration_from_athena.log
G3EOF
)"
```

## modehb_seeded/coconut_trees/seed42 — MO-DEHB hybrid semeada — **nova (nada a migrar)**

- origem: athena GPU 3 (waiter seedq_* que nunca iniciou; sem estado)
- depende_de: done de artifacts/modehb_hybrid/coconut_trees/state/seed42/search_state.json (busca-mãe migrada, regra F3: semeada só depois da atual)
- semente: `{'checkpoint': 'artifacts/external_ssl_pequeno/coconut_trees_split1_pct100/lejepa/checkpoints/best_kappa.ckpt', 'realpath': '/dados/home/moliveira/exps/spifil_growth/ssl_pequeno_lejepa/coconut_trees_split1_pct100/L6_kappa/checkpoints/best_kappa.ckpt', 'arch': 'artifacts/external_ssl_pequeno/coconut_trees_split1_pct100/L0/architecture.json'}`
- config: `{'config': 'ga_search/config/modehb/search_v3_seeded.yaml', 'dataset': 'coconut_trees', 'seed': 42, 'sets': ['output_dir=artifacts/modehb_seeded/coconut_trees/state/seed42', 'search.stratified.memory_budget_gb=24', 'gpu_process_cap_gb=28']}`
- recursos: `{'memory_budget_gb': 24, 'gpu_process_cap_gb': 28, 'gate_free_mib': 28672}`
- obs: budget 24/28 = padrão F3 (athena: 'todas 24/28'); o comando original está em athena:/exps/moliveira/queues/gpu3_migracao/waiters/seedq_modehb_seeded_coconut_trees.txt — conferir antes

Waiter (tmux `g3_seedhyb_coconut_trees_s42`; não precisa publicar nada):
```bash
tmux new-session -d -s g3_seedhyb_coconut_trees_s42 bash -c "$(cat <<'G3EOF'
cd /dados/home/moliveira/Scalable_Hybrid_FLIM
NAME=g3_seedhyb_coconut_trees_s42; OUT=artifacts/modehb_seeded/coconut_trees/state/seed42; LOG=artifacts/modehb_seeded/coconut_trees/logs/search_s42.log; GATE=28672
# busca NOVA (sem estado no athena): espera o done da busca-mãe na Jaci + o ckpt da semente
until /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -c "import json,sys; sys.exit(0 if json.load(open('artifacts/modehb_hybrid/coconut_trees/state/seed42/search_state.json')).get('done') else 1)" 2>/dev/null && [ -f artifacts/external_ssl_pequeno/coconut_trees_split1_pct100/lejepa/checkpoints/best_kappa.ckpt ]; do sleep 300; done
ok=0
while true; do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$f" -ge $GATE ]; then ok=$((ok+1)); else ok=0; fi
  if [ $ok -ge 3 ] && flock -n artifacts/migration_markers/.last_start.flock -c 'age=$(( $(date +%s) - $(stat -c %Y artifacts/migration_markers/.last_start) )); [ $age -ge 900 ] && touch artifacts/migration_markers/.last_start'; then break; fi
  sleep $((90 + RANDOM % 60))
done
echo "START $NAME $(date)" >> artifacts/migration_from_athena.log
ulimit -n 65536
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$(dirname $LOG)"
for i in 1 2 3 4 5; do
  if pgrep -f "^[^ ]*python[^ ]* -m ga_search.modehb.run_search .*output_dir=$OUT( |\$)" >/dev/null; then
    echo "DUP $NAME: outro run_search já usa $OUT (tentativa $i adiada) $(date)" >> artifacts/migration_from_athena.log; sleep 600; continue
  fi
  flock -n artifacts/migration_markers/modehb_seeded_coconut_trees_s42.ctrl.lock /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.modehb.run_search --config ga_search/config/modehb/search_v3_seeded.yaml --dataset coconut_trees --seed 42 --set output_dir=artifacts/modehb_seeded/coconut_trees/state/seed42 --set search.stratified.memory_budget_gb=24 --set gpu_process_cap_gb=28 >> $LOG 2>&1 && break
  echo "RETRY $NAME tentativa $i rc=$? $(date)" >> artifacts/migration_from_athena.log
  sleep 300
done
echo "END $NAME $(date)" >> artifacts/migration_from_athena.log
G3EOF
)"
```

## modehb_hybrid/flowers/seed43 — MO-DEHB hybrid (SPiFiL) — **validada_CONFLITO**

- jobs 2 {'ok': 2}, brackets concluídos 0, done=False, GPU-h 0.54; fid máx 15.0 (1 ckpts); ckpts carregados em CPU 2, ausentes 0; staging 0.02 GB.
- `search_state.json` sha256 `8c1947acadc9742f3dcde9ef8b78c0b3b9bff20f9cb8a96c127254e125790d2c` (gravado 2026-11-16 02:18:04); jobs failed: nenhum.
- resume: replay_ok; próximo job {'next_job': 2, 'next_fidelity': 45.0, 'next_role': 'search', 'next_cand': 'c00001'}.
- controller Jaci `ctrl-524ac822d16d6d11` (C100_kw1); recursos {'memory_budget_gb': 12, 'gpu_process_cap_gb': 14, 'peak_gpu_reserved_gb_max': 4.35, 'gate_free_mib': 22528}.
- config: `{'config': 'ga_search/config/modehb/search_v3.yaml', 'dataset': 'flowers', 'seed': 43, 'sets': ['output_dir=artifacts/modehb_hybrid/flowers/state/seed43', 'search.stratified.memory_budget_gb=12', 'gpu_process_cap_gb=14']}`
- **CONFLITO — PARAR e decidir**: destino artifacts/modehb_hybrid/flowers JÁ EXISTE na Jaci (sem state/seed43)

**NÃO executar sem OK do coordenador** — alternativa: publicar só o subdiretório state/seed43 dentro do destino existente:
```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM && test ! -e artifacts/modehb_hybrid/flowers/state/seed43 && mv -T --no-clobber artifacts/_staging_gpu3/modehb_hybrid_seed43/flowers/state/seed43 artifacts/modehb_hybrid/flowers/state/seed43 && mkdir -p artifacts/modehb_hybrid/flowers/logs && { test ! -e artifacts/_staging_gpu3/modehb_hybrid_seed43/flowers/logs/search_s43.log || mv --no-clobber artifacts/_staging_gpu3/modehb_hybrid_seed43/flowers/logs/search_s43.log artifacts/modehb_hybrid/flowers/logs/search_s43.log; } && echo "PUBLISH modehb_hybrid/flowers/seed43 $(date)" >> artifacts/migration_from_athena.log || echo 'NAO PUBLICADO: destino seed43 existe'
```

Waiter (tmux `g3_hyb_flowers_s43`; rodar DEPOIS da publicação):
```bash
tmux new-session -d -s g3_hyb_flowers_s43 bash -c "$(cat <<'G3EOF'
cd /dados/home/moliveira/Scalable_Hybrid_FLIM
NAME=g3_hyb_flowers_s43; OUT=artifacts/modehb_hybrid/flowers/state/seed43; LOG=artifacts/modehb_hybrid/flowers/logs/search_s43.log; GATE=22528
test -f "$OUT/search_state.json" || { echo "ABORT $NAME: $OUT/search_state.json ausente (publicar antes) $(date)" >> artifacts/migration_from_athena.log; exit 1; }
ok=0
while true; do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$f" -ge $GATE ]; then ok=$((ok+1)); else ok=0; fi
  if [ $ok -ge 3 ] && flock -n artifacts/migration_markers/.last_start.flock -c 'age=$(( $(date +%s) - $(stat -c %Y artifacts/migration_markers/.last_start) )); [ $age -ge 900 ] && touch artifacts/migration_markers/.last_start'; then break; fi
  sleep $((90 + RANDOM % 60))
done
echo "START $NAME $(date)" >> artifacts/migration_from_athena.log
ulimit -n 65536
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$(dirname $LOG)"
for i in 1 2 3 4 5; do
  if pgrep -f "^[^ ]*python[^ ]* -m ga_search.modehb.run_search .*output_dir=$OUT( |\$)" >/dev/null; then
    echo "DUP $NAME: outro run_search já usa $OUT (tentativa $i adiada) $(date)" >> artifacts/migration_from_athena.log; sleep 600; continue
  fi
  flock -n artifacts/migration_markers/modehb_hybrid_flowers_s43.ctrl.lock /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.modehb.run_search --config ga_search/config/modehb/search_v3.yaml --dataset flowers --seed 43 --set output_dir=artifacts/modehb_hybrid/flowers/state/seed43 --set search.stratified.memory_budget_gb=12 --set gpu_process_cap_gb=14 >> $LOG 2>&1 && break
  echo "RETRY $NAME tentativa $i rc=$? $(date)" >> artifacts/migration_from_athena.log
  sleep 300
done
echo "END $NAME $(date)" >> artifacts/migration_from_athena.log
G3EOF
)"
```

## modehb_hybrid/larvae/seed43 — MO-DEHB hybrid (SPiFiL) — **sem_state (0 jobs: resume = começar do zero)**

- origem: athena GPU 3 -> artifacts/_staging_gpu3/modehb_hybrid_seed43/larvae
- config: `{'config': 'ga_search/config/modehb/search_v3.yaml', 'dataset': 'larvae', 'seed': 43, 'sets': ['output_dir=artifacts/modehb_hybrid/larvae/state/seed43', 'search.stratified.memory_budget_gb=12', 'gpu_process_cap_gb=14']}`
- recursos: `{'memory_budget_gb': 12, 'gpu_process_cap_gb': 14, 'gate_free_mib': 22528}`
- obs: busca nova de fato (réplica seed43, F4 = reserva): decidir com o coordenador se entra; não há o que retomar
- marcador: "seed43 larvae: SEM search_state.json (pausado 3 min após iniciar, antes do 1º job salvo); 0 jobs; o resume equivale a começar do zero; arquivos copiados: 2"
- **CONFLITO — PARAR e decidir**: destino artifacts/modehb_hybrid/larvae JÁ EXISTE na Jaci

**Só com OK do coordenador** (réplica seed43, reserva F4; começa do zero em state/seed43, dentro da base larvae que já existe):
```bash
tmux new-session -d -s g3_hyb_larvae_s43 bash -c "$(cat <<'G3EOF'
cd /dados/home/moliveira/Scalable_Hybrid_FLIM
NAME=g3_hyb_larvae_s43; OUT=artifacts/modehb_hybrid/larvae/state/seed43; LOG=artifacts/modehb_hybrid/larvae/logs/search_s43.log; GATE=22528
# busca NOVA (sem estado no athena): espera só o gate
:
ok=0
while true; do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$f" -ge $GATE ]; then ok=$((ok+1)); else ok=0; fi
  if [ $ok -ge 3 ] && flock -n artifacts/migration_markers/.last_start.flock -c 'age=$(( $(date +%s) - $(stat -c %Y artifacts/migration_markers/.last_start) )); [ $age -ge 900 ] && touch artifacts/migration_markers/.last_start'; then break; fi
  sleep $((90 + RANDOM % 60))
done
echo "START $NAME $(date)" >> artifacts/migration_from_athena.log
ulimit -n 65536
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$(dirname $LOG)"
for i in 1 2 3 4 5; do
  if pgrep -f "^[^ ]*python[^ ]* -m ga_search.modehb.run_search .*output_dir=$OUT( |\$)" >/dev/null; then
    echo "DUP $NAME: outro run_search já usa $OUT (tentativa $i adiada) $(date)" >> artifacts/migration_from_athena.log; sleep 600; continue
  fi
  flock -n artifacts/migration_markers/modehb_hybrid_larvae_s43.ctrl.lock /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.modehb.run_search --config ga_search/config/modehb/search_v3.yaml --dataset larvae --seed 43 --set output_dir=artifacts/modehb_hybrid/larvae/state/seed43 --set search.stratified.memory_budget_gb=12 --set gpu_process_cap_gb=14 >> $LOG 2>&1 && break
  echo "RETRY $NAME tentativa $i rc=$? $(date)" >> artifacts/migration_from_athena.log
  sleep 300
done
echo "END $NAME $(date)" >> artifacts/migration_from_athena.log
G3EOF
)"
```

