# athena_ops — runners e filas de teste do athena (18–20/11/2026)

Cópia do código operacional que rodou fora da árvore do repo (filas em /exps/moliveira/queues, clones em
/dados/home/moliveira/exps/*). Os originais seguem nos caminhos de origem; aqui fica o registro versionado.

- `queues/` — escalonador das GPUs 2/3 (`gpu_launcher.py`: backfill pela memória real, reserva = pico medido) e
  utilitários de pausa/relançamento entre nós (`growth_pause_between_nodes.py`, `stop_between.py`,
  `relaunch_after_pause.py`, `v3_swap_fixstate.py`, `ham_b1g1c00_retry_watch.py`).
- `curvas_pct/` — curvas κ × % de treino (melhor + gêmeo só-SPiFiL) de ham10000, protozoan e protozoan_raw.
- `benchmark_all11/run_all11.py` — SVM original dos FMs nas 11 bases (embeddings já extraídos).
- `jaci_migracao/` — passagem dos testes da Jaci para o athena (19/11).
- `retry_oom_node.py` — avaliação avulsa de nó de crescimento morto por OOM de contenção.
- `patches/` — `ga_v4_param_cap_fix.diff` (teto de 5,5M checado antes do treino) e
  `growth_v3_retry_oom_lineage.diff` (oom.retry_failed + lineage com retry_of).

Os testes do readout (`research/spifil_b0b4/readout_x/`) usam a SVM original (core.svm.fit_svm, linear, C=100,
ovo, max_iter=-1, sem scaler); o teste nunca entra na seleção.
