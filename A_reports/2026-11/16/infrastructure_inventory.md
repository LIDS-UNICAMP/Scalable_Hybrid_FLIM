# Inventário de infraestrutura — athena (Agente 5) — 2026-11-17 01:05:32

Somente leitura até a seção "Ações". Host: athena. Usuário: moliveira.

## GPUs (nvidia-smi; índice físico = índice CUDA, sem remapeamento global)
```
0, GPU-731b8e24-92f8-b429-d0b0-080d0fe4961e, 00000000:18:00.0, 9866 MiB, 49140 MiB, 100 %
1, GPU-d6d7765b-7441-9a5d-478c-0592726dc28d, 00000000:3B:00.0, 23140 MiB, 49140 MiB, 100 %
2, GPU-66e3bb43-d4ad-4c81-64a3-47151de4e9c4, 00000000:86:00.0, 20377 MiB, 49140 MiB, 100 %
3, GPU-3c72f1c7-f2c9-d905-fdee-9b0b0ce802be, 00000000:AF:00.0, 26755 MiB, 49140 MiB, 100 %
```
- Nossos processos usam CUDA_VISIBLE_DEVICES=<índice físico> (2 ou 3), ou seja, dentro do processo a GPU vira cuda:0.
- GPUs 0 e 1: cedidas a terceiros desde 12/11; não são nossas.

### GPU 0/1 (terceiros; só listados)
| gpu_uuid | pid | dono | memória |
|---|---|---|---|
| GPU-731b8e24-92f8-b429-d0b0-080d0fe4961e | 2020545 | crispim |  9850 MiB |
| GPU-d6d7765b-7441-9a5d-478c-0592726dc28d | 186265 | crispim |  23124 MiB |

### GPU 2 — NÃO MEXER (PNAS-CS, monitores de teste, benchmark FM pre/orig, heatmap coconut×ConvNeXt)
| pid | dono | memória | comando |
|---|---|---|---|
| 996382 | moliveira |  10484 MiB | /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.modehb.run_search --config ga_search/config/modehb/pnas_cs.yaml  |
| 1648855 | moliveira |  290 MiB | /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python /dados/home/moliveira/presentation_results_cluster_november/figs/heatmaps/mak |
| 1898421 | moliveira |  592 MiB | /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.modehb.monitor_report |
| 1898415 | moliveira |  966 MiB | /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.modehb.monitor_report |
| 1898410 | moliveira |  634 MiB | /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.modehb.monitor_report |
| 2007258 | moliveira |  7378 MiB | /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python /dados/home/moliveira/Scalable_Hybrid_FLIM_discovery/results/benchmark/modehb |

### GPU 3 — antes da migração (01:00 de 17/11): só processos nossos, nenhum de terceiros
| pid | tmux | memória | comando (resumo) | output_dir | jobs (ok/failed) | último ok |
|---|---|---|---|---|---|---|
| 1018842 | modehb_gpu3_eraw | 5,7 GB | run_search search_v3.yaml --dataset eggs_raw, budget 12/cap 14 | artifacts/modehb_hybrid/eggs_raw/state/seed42 | 50 (50/0) | c00030 f5 small |
| 1024539 | modehb_gpu3c | 10,6 GB | run_search search_v3.yaml --dataset coconut_trees, budget 34/cap 44 | artifacts/modehb_hybrid/coconut_trees/state/seed42 | 38 (37/1: job 31 c00012 params > cap) | c00018 f5 small |
| 1027675 | modehb_gpu3e | 5,7 GB | run_search search_v3.yaml --dataset eggs, budget 12/cap 14 | artifacts/modehb_hybrid/eggs/state/seed42 | 54 (54/0) | c00034 f5 small |
| 1265798 | modehb_gpu3_lem_coco3 | 8,0 GB | run_search search_v3_lemonade.yaml --dataset coconut_trees | artifacts/modehb_lemonade/coconut_trees/state/seed42 | 23 (23/0) | c00001 f5 tiny |
| 1267978 | modehb_gpu3_lem_flowers2 | 8,9 GB | run_search search_v3_lemonade.yaml --dataset flowers | artifacts/modehb_lemonade/flowers/state/seed42 | 34 (34/0) | c00014 f5 medium |

Ambiente comum: env conda scalable_FLIM (torch 2.2.2), LD_LIBRARY_PATH do env, OPENBLAS_NUM_THREADS=2, OMP/MKL=8, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, --set eval.svm_fitness.cov_eigh_workers=1, fitness C100_kw1. Cada um roda num laço tmux "for i in 1..5" com retomada pelo search_state.json (ReplayMismatch protege genoma/fidelidade/papel).

## Gerenciador de fila (tmux + waiters) ligado à GPU 3
| sessão | tipo | o que esperava | estado |
|---|---|---|---|
| seedq_modehb_seeded_{coconut_trees,eggs,eggs_raw} | waiter | vaga na GPU 3 → search_v3_seeded.yaml | nunca iniciou (sem search_state) |
| seedq_modehb_lemonade_seeded_{coconut_trees,flowers} | waiter | vaga na GPU 3 → search_v3_lemonade_seeded.yaml | nunca iniciou |
| s43_reserva | waiter | fim de uma busca da GPU 3 → hybrid seed43 flowers, depois larvae | estado arquivado em artifacts/modehb_hybrid_s43_pausado_0518/ (flowers 2 jobs; larvae 0 jobs, sem search_state) |
Os comandos de todos foram salvos em /exps/moliveira/queues/gpu3_migracao/{waiters,loops}/.

Demais controladores (não tocados): watchdog (modehb_watchdog, CUDA 2, WATCH_GPUS=2,3, religa o monitor shard 0/3), monitores de teste modehb_monitor{,2,3} (GPU 2), pnas_cs (GPU 2), fm_pre_bench (GPU 2), heatmap_coconut_fm (GPU 2/CPU), modehb_rsync_ranking (rsync do ranking a cada 30 min).

## Checkpoints e estados (athena, preservados)
- Cada busca: state/seed42/{search_state.json, candidates/cNNNNN/fF/{encoder.pt,result.json,val_embedding.pt}, external_seed/, data/, config_resolved.yaml} + logs/search_s42.log.
- Tamanhos: hybrid eggs_raw 2,5 GB · hybrid coconut 1,8 GB · hybrid eggs 2,8 GB · LEMONADE coconut 1,1 GB · LEMONADE flowers 0,6 GB.
- artifacts/ é symlink para /dados/home/moliveira/artifacts no athena; na Jaci é diretório real.

## Disco e permissões
```
/dev/mapper/ubuntu--vg-lv--dados   37T   36T  1.2T  97% /dados
/dev/nvme0n1                      3.7T  3.5T  187G  95% /exps
jaci: /dev/mapper/dados--vg-lv--dados  2.9T  2.2T  585G  80% /dados
```
- Tudo é do usuário moliveira (grupo moliveira, umask 002). Na Jaci, o staging fica em artifacts/_staging_gpu3/ (o mesmo usuário via ssh BatchMode).

## Dependências
- Código: Scalable_Hybrid_FLIM (main) com o patch de velocidade (run_search 1f61f193, pooling_head 877f8684, supervised_fitness f1cc7bb7) e o monitor b776e2f6. No athena (torch 2.2) os laços usam cov_eigh_workers=1; na Jaci (torch 2.7) o eigh em streams fica ligado.
- Dados: /dados/home/moliveira/datasets/organized/{coconut_trees,Eggs,flowers,...}; para eggs_raw, splits/split1_modehb_raw.json + raw_modehb/.
- Sementes externas: artifacts/external_ssl_pequeno/<base>_split1_pct100/{L0,lejepa}.

## Ações (Agente 5)
1. 01:04 waiters desarmados (seedq_* ×5, s43_reserva). Nenhum tinha processo Python filho.
2. 01:04 cinco sessões mig_<laço>: restart_between_jobs (espera o job atual salvar e manda SIGTERM) → encerra a sessão do laço (sem relançar) → espera o pid sair → rsync -a (sem --delete) para jaci:artifacts/_staging_gpu3/<kind>/<base>/ → confere nº de jobs e md5 do search_state → marcador <kind>_<base>.ok.
3. 01:04 seed43 arquivado: flowers (2 jobs, md5 conferido) e larvae (0 jobs, sem search_state; 2 arquivos) → staging modehb_hybrid_seed43/.
