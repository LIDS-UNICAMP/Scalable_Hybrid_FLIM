# Tempo de execução do GA (MO-DEHB e crescimento B0–B4)

Gerado em 20/11/2026 a partir dos arquivos da Jaci (`artifacts/modehb_*/<base>/state/seed42/search_state.json`). Runs só do athena não entram.

- **GPU-h** = soma de `gpu_hours` de todos os jobs da busca.
- **Duração** = relógio do 1º ao último job (buscas rodaram em paralelo; GPU-h > duração = vários jobs ao mesmo tempo).
- **Terminou** = campo `done` do estado da busca.

## Buscas MO-DEHB

| busca | base | jobs | ok | candidatos | terminou | GPU-h | duração (h) | início | fim |
|---|---|---|---|---|---|---|---|---|---|
| modehb_hybrid | aid | 177 | 175 | 108 | não | 52.5 | 54.5 | 16/11 | 18/11 |
| modehb_hybrid | coconut_trees | 39 | 38 | 19 | não | 24.9 | 8.9 | 16/11 | 16/11 |
| modehb_hybrid | eggs | 55 | 55 | 35 | não | 24.6 | 9.0 | 16/11 | 16/11 |
| modehb_hybrid | eggs_raw | 51 | 51 | 31 | não | 24.5 | 9.1 | 16/11 | 16/11 |
| modehb_hybrid | flowers | 119 | 118 | 81 | não | 22.1 | 29.0 | 17/11 | 18/11 |
| modehb_hybrid | ham10000 | 190 | 190 | 111 | não | 55.7 | 52.5 | 16/11 | 18/11 |
| modehb_hybrid | larvae | 161 | 158 | 99 | não | 25.3 | 30.0 | 16/11 | 17/11 |
| modehb_hybrid | larvae_raw | 162 | 159 | 98 | não | 27.6 | 31.7 | 16/11 | 17/11 |
| modehb_hybrid | protozoan | 120 | 118 | 81 | não | 37.1 | 32.0 | 16/11 | 17/11 |
| modehb_hybrid | protozoan_raw | 112 | 112 | 81 | não | 34.2 | 34.9 | 16/11 | 17/11 |
| modehb_hybrid | remote_sensing | 114 | 111 | 81 | não | 17.8 | 17.7 | 17/11 | 18/11 |
| modehb_hybrid_C01_2049 | aid | 4 | 4 | 1 | não | 2.5 | 2.4 | 15/11 | 15/11 |
| modehb_hybrid_C01_2049 | ham10000 | 4 | 4 | 1 | não | 2.7 | 2.5 | 15/11 | 15/11 |
| modehb_hybrid_C01_2049 | protozoan | 4 | 4 | 1 | não | 2.9 | 2.7 | 15/11 | 15/11 |
| modehb_hybrid_C01_final | remote_sensing | 146 | 138 | 82 | não | 21.3 | 10.0 | 15/11 | 15/11 |
| modehb_hybrid_fullepoch_1555 | aid | 3 | 3 | 1 | não | 3.8 | 3.4 | 15/11 | 15/11 |
| modehb_hybrid_fullepoch_1555 | ham10000 | 3 | 3 | 1 | não | 2.2 | 2.0 | 15/11 | 15/11 |
| modehb_hybrid_fullepoch_1555 | protozoan | 2 | 2 | 1 | não | 2.5 | 0.0 | 15/11 | 15/11 |
| modehb_hybrid_oomfail_2120 | aid | 2 | 0 | 2 | não | 0.0 | 0.1 | 15/11 | 15/11 |
| modehb_hybrid_oomfail_2120 | ham10000 | 1 | 0 | 1 | não | 0.0 | 0.0 | 15/11 | 15/11 |
| modehb_hybrid_oomfail_2120 | protozoan | 1 | 0 | 1 | não | 0.0 | 0.0 | 15/11 | 15/11 |
| modehb_hybrid_pre_aesched_0033 | coconut_trees | 1 | 1 | 1 | não | 0.1 | 0.0 | 15/11 | 15/11 |
| modehb_hybrid_pre_aesched_0033 | remote_sensing | 14 | 14 | 3 | não | 0.7 | 0.0 | 15/11 | 15/11 |
| modehb_lemonade | aid | 2 | 2 | 1 | não | 0.3 | 0.2 | 16/11 | 16/11 |
| modehb_lemonade | coconut_trees | 24 | 24 | 5 | não | 15.0 | 6.9 | 16/11 | 16/11 |
| modehb_lemonade | eggs | 147 | 147 | 83 | não | 31.7 | 28.7 | 16/11 | 17/11 |
| modehb_lemonade | flowers | 154 | 154 | 87 | não | 29.5 | 15.0 | 17/11 | 18/11 |
| modehb_lemonade | ham10000 | 199 | 199 | 114 | não | 49.5 | 48.9 | 16/11 | 18/11 |
| modehb_lemonade | larvae | 167 | 165 | 103 | não | 26.4 | 22.4 | 16/11 | 17/11 |
| modehb_lemonade | remote_sensing | 208 | 207 | 121 | sim | 29.9 | 30.8 | 16/11 | 17/11 |
| modehb_lemonade_C01_final | remote_sensing | 185 | 182 | 106 | não | 20.5 | 14.9 | 15/11 | 15/11 |
| modehb_lemonade_seeded | remote_sensing | 139 | 139 | 81 | não | 10.4 | 15.0 | 17/11 | 18/11 |
| modehb_spifil | flowers | 102 | 45 | 81 | não | 1.9 | 3.0 | 14/11 | 14/11 |
| modehb_spifil | remote_sensing | 374 | 247 | 315 | sim | 7.3 | 9.3 | 14/11 | 14/11 |
| modehb_strat_fid120_1415 | remote_sensing | 3 | 3 | 1 | não | 0.1 | 0.1 | 14/11 | 14/11 |
| modehb_strat_oom_1347 | remote_sensing | 8 | 0 | 4 | não | 0.0 | 0.0 | 14/11 | 14/11 |
| modehb_strat_preHybrid_2325 | remote_sensing | 10 | 10 | 2 | não | 1.1 | 1.1 | 14/11 | 14/11 |
| modehb_strat_prefix_2130 | remote_sensing | 69 | 62 | 49 | não | 6.3 | 1.0 | 14/11 | 14/11 |
| modehb_v3 | remote_sensing | 29 | 20 | 29 | não | 0.1 | 0.2 | 14/11 | 14/11 |
| modehb_v3_old | remote_sensing | 60 | 36 | 60 | não | 0.3 | 0.6 | 14/11 | 14/11 |
| modehb_v4 | remote_sensing | 20 | 19 | 20 | não | 4.9 | 4.9 | 18/11 | 18/11 |
| **TOTAL** | | **3385** | **3119** | | | **620** | **535** | 14/11 | 18/11 |

## GPU-h por base (todas as buscas)

| base | GPU-h |
|---|---|
| remote_sensing | 120.8 |
| ham10000 | 110.1 |
| aid | 59.1 |
| eggs | 56.4 |
| flowers | 53.5 |
| larvae | 51.7 |
| protozoan | 42.5 |
| coconut_trees | 40.0 |
| protozoan_raw | 34.2 |
| larvae_raw | 27.6 |
| eggs_raw | 24.5 |

## Cadeias de crescimento B0–B4

Sem `gpu_hours` gravado; duração estimada pela data dos arquivos (aproximada).

| cadeia | nós | duração aprox. (h) | período |
|---|---|---|---|
| growth/athena_gpu3/ham10000_fix5_nossl/seed42 | 1 | 14.8 | 18/11→18/11 |
| growth_smoke/jaci_gpu0/remote_sensing | 0 | 0.0 | 17/11→17/11 |
| growth_v3_jaci/ae_so2/aid/seed42 | 16 | 2.2 | 18/11→18/11 |
| growth_v3_jaci/ae_so2/coconut_trees/seed42 | 3 | 1.2 | 18/11→18/11 |
| growth_v3_jaci/ae_so2/flowers/seed42 | 1 | 0.0 | 18/11→18/11 |
| growth_v3_jaci/ae_so2/ham10000/seed42 | 8 | 2.1 | 18/11→18/11 |
