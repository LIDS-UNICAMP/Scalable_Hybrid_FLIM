# Inventário das CNNs (GA / MO-DEHB / LEMONADE / crescimento)

3027 registros (busca × base × candidato × fidelidade). κ_val = validação da própria busca (seleção); κ_teste só para relatório.

| base | n | ok | with_ckpt | with_test | top30 | focus |
|---|---|---|---|---|---|---|
| aid | 195 | 195 | 195 | 146 | 59 | 2 |
| coconut_trees | 60 | 60 | 60 | 39 | 18 | 1 |
| eggs | 193 | 193 | 193 | 120 | 58 | 1 |
| eggs_raw | 47 | 47 | 47 | 32 | 15 | 1 |
| flowers | 301 | 301 | 301 | 215 | 91 | 1 |
| ham10000 | 391 | 391 | 391 | 286 | 118 | 1 |
| larvae | 308 | 308 | 308 | 205 | 93 | 1 |
| larvae_raw | 152 | 152 | 152 | 98 | 46 | 1 |
| protozoan | 120 | 120 | 120 | 10 | 36 | 0 |
| protozoan_raw | 107 | 107 | 107 | 4 | 33 | 0 |
| remote_sensing | 1153 | 1153 | 1153 | 874 | 346 | 2 |

## Campos ausentes (contagem)

| campo | n |
|---|---|
| generation | 30 |
| kappa_test | 998 |
| latency | 29 |

## Modelos em foco (draw_io)

| id | base | κ_val | κ_teste | leitura(val) | κ_teste(leitura val) | params | ckpt |
|---|---|---|---|---|---|---|---|
| modehb_hybrid/aid/c00091/f45 | aid | 0.839 | 0.678 | so2_4096_pca_tta8 | 0.846 | 740454 | sim |
| hmodehb_b0b4/growth_v3_jaci/ae_so2/aid/seed42/nodes/B1g0c01/f15 | aid | 0.801 | 0.726 | so2_4096_pca_tta8 | 0.846 | 498110 | sim |
| modehb_hybrid/coconut_trees/c00015/f5 | coconut_trees | 0.761 | 0.804 | so2_4096_pca_tta8 | 0.821 | 1043416 | sim |
| modehb_lemonade/eggs/c00032/f15 | eggs | 0.941 | 0.942 | so2_4096_pca | 0.966 | 2055524 | sim |
| modehb_hybrid/eggs_raw/c00027/f5 | eggs_raw | 0.893 | 0.780 | so2_4096_pca | 0.914 | 999162 | sim |
| modehb_hybrid/flowers/c00064/f5 | flowers | 0.763 | 0.735 | gap1280 | 0.735 | 1630360 | sim |
| hmodehb_b0b4/growth/athena_gpu3/ham10000_fix5_nossl/seed42/nodes/B1g0c06/f15 | ham10000 | 0.475 | AUSENTE | AUSENTE | AUSENTE | 504623 | sim |
| modehb_lemonade/larvae/c00078/f15 | larvae | 0.984 | 0.713 | so2_4096_pca_tta8 | 0.917 | 204080 | sim |
| modehb_hybrid/larvae_raw/c00023/f15 | larvae_raw | 0.884 | 0.565 | so2_4096_pca | 0.866 | 884856 | sim |
| modehb_hybrid_C01_final/remote_sensing/c00015/f15 | remote_sensing | 0.592 | 0.497 | so2_4096_pca_tta8 | 0.654 | 1481220 | sim |
| modehb_lemonade_seeded/remote_sensing/c00002/f200 | remote_sensing | 0.591 | 0.571 | so2_4096_pca_tta8 | 0.665 | 1383536 | sim |
