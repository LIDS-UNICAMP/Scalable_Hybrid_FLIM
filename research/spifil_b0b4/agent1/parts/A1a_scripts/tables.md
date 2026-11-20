## T0 contagem de jobs ok (role=search) por dataset × busca × assinatura de fitness × fidelidade
| dataset        | kind      | fitness_sig                          | host   |   n |   n_f5 |   n_fge45 |   n_test |   kval_max |       params_max |   depth_max |
|:---------------|:----------|:-------------------------------------|:-------|----:|-------:|----------:|---------:|-----------:|-----------------:|------------:|
| aid            | hybrid    | C0.1_kw0.75_cov_head_standard_mt4000 | jaci   |   7 |      2 |         3 |        2 |      0.733 |  23657           |           2 |
| aid            | hybrid    | C100_kw1_cov_head_standard_mt4000    | jaci   |  59 |     43 |        12 |       13 |      0.845 |      3.35163e+06 |           8 |
| aid            | lemonade  | C100_kw1_cov_head_standard_mt4000    | jaci   |   2 |      1 |         0 |        1 |      0.706 | 147950           |           3 |
| coconut_trees  | hybrid    | C0.1_kw0.75_cov_head_standard_mt4000 | jaci   |   1 |      1 |         0 |        0 |      0.758 |  23488           |           2 |
| coconut_trees  | hybrid    | C100_kw1_cov_head_standard_mt4000    | athena |   7 |      0 |         7 |        7 |      0.79  |      2.29787e+06 |           4 |
| coconut_trees  | lemonade  | C100_kw1_cov_head_standard_mt4000    | athena |   2 |      0 |         2 |        2 |      0.732 | 593840           |           5 |
| eggs           | hybrid    | C100_kw1_cov_head_standard_mt4000    | athena |   6 |      0 |         6 |        6 |      0.932 |      1.6973e+06  |           5 |
| eggs           | lemonade  | C100_kw1_cov_head_standard_mt4000    | jaci   |  35 |     19 |        12 |        3 |      0.937 |      2.16986e+06 |           6 |
| eggs_raw       | hybrid    | C100_kw1_cov_head_standard_mt4000    | athena |   6 |      0 |         6 |        6 |      0.896 |      1.6973e+06  |           5 |
| flowers        | hybrid    | C0.1_kw0.75_cov_head_standard_mt4000 | athena |   7 |      0 |         7 |        7 |      0.781 |      2.21758e+06 |           4 |
| flowers        | hybrid    | C100_kw1_cov_head_standard_mt4000    | jaci   |  15 |      3 |         9 |        2 |      0.757 | 894960           |           4 |
| flowers        | lemonade  | C100_kw1_cov_head_standard_mt4000    | athena |   4 |      0 |         4 |        4 |      0.744 |      1.31192e+06 |           6 |
| flowers        | spifil_v1 | C100_kw0.75_head_none_mt4000         | jaci   |  45 |     24 |         0 |        0 |      0.675 |      2.80717e+06 |           7 |
| ham10000       | hybrid    | C0.1_kw0.75_cov_head_standard_mt4000 | jaci   |   7 |      2 |         3 |        2 |      0.526 |  19790           |           2 |
| ham10000       | hybrid    | C100_kw1_cov_head_standard_mt4000    | jaci   |  59 |     43 |        12 |       10 |      0.601 |      2.86396e+06 |           8 |
| ham10000       | lemonade  | C100_kw1_cov_head_standard_mt4000    | jaci   |  37 |     21 |        12 |       13 |      0.587 |      2.86396e+06 |           6 |
| larvae         | hybrid    | C0.1_kw0.75_cov_head_standard_mt4000 | athena |   7 |      0 |         7 |        7 |      0.967 |      1.97372e+06 |           5 |
| larvae         | hybrid    | C100_kw1_cov_head_standard_mt4000    | jaci   |  47 |     31 |        12 |        6 |      0.967 |      3.49382e+06 |           7 |
| larvae         | lemonade  | C100_kw1_cov_head_standard_mt4000    | jaci   |  30 |     14 |        12 |       10 |      0.984 |      2.22307e+06 |           6 |
| larvae_raw     | hybrid    | C100_kw1_cov_head_standard_mt4000    | jaci   |  42 |     26 |        12 |        8 |      0.915 |      2.42488e+06 |           7 |
| protozoan      | hybrid    | C0.1_kw0.75_cov_head_standard_mt4000 | jaci   |   6 |      2 |         2 |        2 |      0.697 |  19790           |           2 |
| protozoan      | hybrid    | C100_kw1_cov_head_standard_mt4000    | jaci   |  47 |     31 |        12 |        7 |      0.897 |      3.04267e+06 |           8 |
| protozoan_raw  | hybrid    | C100_kw1_cov_head_standard_mt4000    | jaci   |  27 |     11 |        12 |        1 |      0.797 |      1.81223e+06 |           5 |
| remote_sensing | hybrid    | C0.1_kw0.75_cov_head_standard_mt4000 | jaci   | 152 |     81 |        36 |       20 |      0.605 |      3.05914e+06 |           8 |
| remote_sensing | hybrid    | C100_kw1_cov_head_standard_mt4000    | jaci   |  43 |     27 |        12 |       18 |      0.599 |      2.3765e+06  |           7 |
| remote_sensing | lemonade  | C0.1_kw0.75_cov_head_standard_mt4000 | jaci   | 182 |     82 |        42 |       65 |      0.591 |      3.38931e+06 |           8 |
| remote_sensing | lemonade  | C100_kw1_cov_head_standard_mt4000    | jaci   | 129 |     85 |        12 |       42 |      0.611 |      3.1543e+06  |           8 |
| remote_sensing | pnas_cs   | C100_kw1_cov_head_standard_mt4000    | athena |  79 |     59 |        15 |        9 |      0.605 |      5.63359e+06 |           8 |
| remote_sensing | spifil_v1 | C100_kw0.75_head_none_mt4000         | jaci   | 247 |     63 |        94 |        0 |      0.572 |      2.00215e+07 |           8 |
| remote_sensing | strat     | C100_kw0.75_head_none_mt4000         | jaci   |  75 |     49 |        19 |        0 |      0.517 |      2.31119e+06 |           8 |
| remote_sensing | v3        | C100_kw0.75_head_none_mt10000        | jaci   |  36 |     36 |         0 |        0 |      0.487 |      1.98184e+07 |           8 |
| remote_sensing | v3        | C100_kw0.75_head_none_mt4000         | jaci   |  20 |     20 |         0 |        0 |      0.514 |      8.97296e+06 |           6 |

## T0b jobs falhos por run (motivo truncado)
| run                                 | dataset        | motivo                                                       |   n |
|:------------------------------------|:---------------|:-------------------------------------------------------------|----:|
| artifacts/modehb_spifil             | remote_sensing | InfeasibleCandidate: SPiFiL fit failed: BudgetExhausted: den |  43 |
| artifacts/modehb_spifil             | flowers        | InfeasibleCandidate: SPiFiL fit failed: BudgetExhausted: den |  28 |
| artifacts/modehb_spifil             | remote_sensing | ValueError: lejepa requires a regularizer ('sigreg' = origin |  14 |
| artifacts/modehb_v3_old             | remote_sensing | InfeasibleCandidate: SPiFiL fit failed: ValueError: build_fi |  13 |
| artifacts/modehb_spifil             | remote_sensing | OutOfMemoryError: CUDA out of memory. Tried to allocate 21.4 |   8 |
| artifacts/modehb_hybrid_C01_final   | remote_sensing | RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x |   7 |
| artifacts/modehb_v3                 | remote_sensing | InfeasibleCandidate: SPiFiL fit failed: ValueError: build_fi |   6 |
| artifacts/modehb_spifil             | remote_sensing | FloatingPointError: grad norm non-finite in new_block.conv.b |   5 |
| artifacts/modehb_spifil             | remote_sensing | OutOfMemoryError: CUDA out of memory. Tried to allocate 24.1 |   3 |
| artifacts/modehb_spifil             | remote_sensing | OutOfMemoryError: CUDA out of memory. Tried to allocate 26.8 |   3 |
| artifacts/modehb_spifil             | remote_sensing | OutOfMemoryError: CUDA out of memory. Tried to allocate 15.3 |   3 |
| artifacts/modehb_spifil             | remote_sensing | _LinAlgError: linalg.svd: The algorithm failed to converge b |   3 |
| artifacts/modehb_lemonade_C01_final | remote_sensing | RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x |   3 |
| artifacts/modehb_spifil             | flowers        | ValueError: lejepa requires a regularizer ('sigreg' = origin |   3 |
| artifacts/modehb_strat_prefix_2130  | remote_sensing | OutOfMemoryError: CUDA out of memory. Tried to allocate 18.7 |   2 |
| artifacts/modehb_strat_prefix_2130  | remote_sensing | RuntimeError: linalg.eigh: The input tensor A must have at l |   2 |
| artifacts/modehb_v3                 | remote_sensing | InfeasibleCandidate: SPiFiL fit failed: BudgetExhausted: den |   2 |
| artifacts/modehb_spifil             | flowers        | OutOfMemoryError: CUDA out of memory. Tried to allocate 15.3 |   2 |
| artifacts/modehb_strat_oom_1347     | remote_sensing | OutOfMemoryError: CUDA out of memory. Tried to allocate 15.3 |   2 |
| artifacts/modehb_spifil             | flowers        | FloatingPointError: grad norm non-finite in new_block.conv.b |   2 |
| artifacts/modehb_spifil             | remote_sensing | OutOfMemoryError: CUDA out of memory. Tried to allocate 38.8 |   2 |
| artifacts/modehb_spifil             | remote_sensing | OutOfMemoryError: CUDA out of memory. Tried to allocate 9.38 |   2 |
| artifacts/modehb_spifil             | remote_sensing | OutOfMemoryError: CUDA out of memory. Tried to allocate 73.5 |   2 |
| artifacts/modehb_spifil             | flowers        | OutOfMemoryError: CUDA out of memory. Tried to allocate 23.9 |   2 |
| artifacts/modehb_strat_oom_1347     | remote_sensing | OutOfMemoryError: CUDA out of memory. Tried to allocate 42.8 |   2 |

## T0c tipos de op por run (jobs ok)
| row_0                                         |   calibration |   de |   external_seed |   grow |   morph |   mutation |   none |   promotion |   random |   seed |
|:----------------------------------------------|--------------:|-----:|----------------:|-------:|--------:|-----------:|-------:|------------:|---------:|-------:|
| modehb_hybrid/aid                             |            20 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |       22 |     16 |
| modehb_hybrid/coconut_trees                   |             5 |    0 |               0 |      0 |       0 |          0 |      2 |           0 |        0 |      0 |
| modehb_hybrid/eggs                            |             6 |    0 |               0 |      0 |       0 |          0 |      0 |           0 |        0 |      0 |
| modehb_hybrid/eggs_raw                        |             4 |    0 |               0 |      0 |       0 |          0 |      2 |           0 |        0 |      0 |
| modehb_hybrid/flowers                         |            14 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |        0 |      0 |
| modehb_hybrid/ham10000                        |            20 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |       22 |     16 |
| modehb_hybrid/larvae                          |            20 |    0 |               0 |      0 |       0 |          0 |      0 |           0 |       11 |     16 |
| modehb_hybrid/larvae_raw                      |            20 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |        5 |     16 |
| modehb_hybrid/protozoan                       |            20 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |       10 |     16 |
| modehb_hybrid/protozoan_raw                   |            20 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |        0 |      6 |
| modehb_hybrid/remote_sensing                  |            20 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |        6 |     16 |
| modehb_hybrid_C01_2049/aid                    |             3 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |        0 |      0 |
| modehb_hybrid_C01_2049/ham10000               |             3 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |        0 |      0 |
| modehb_hybrid_C01_2049/protozoan              |             3 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |        0 |      0 |
| modehb_hybrid_C01_final/flowers               |             7 |    0 |               0 |      0 |       0 |          0 |      0 |           0 |        0 |      0 |
| modehb_hybrid_C01_final/larvae                |             7 |    0 |               0 |      0 |       0 |          0 |      0 |           0 |        0 |      0 |
| modehb_hybrid_C01_final/remote_sensing        |            20 |    0 |               0 |      0 |       0 |          0 |      1 |          43 |       58 |     16 |
| modehb_hybrid_fullepoch_1555/aid              |             2 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |        0 |      0 |
| modehb_hybrid_fullepoch_1555/ham10000         |             2 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |        0 |      0 |
| modehb_hybrid_fullepoch_1555/protozoan        |             1 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |        0 |      0 |
| modehb_hybrid_pre_aesched_0033/coconut_trees  |             0 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |        0 |      0 |
| modehb_hybrid_pre_aesched_0033/remote_sensing |            13 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |        0 |      0 |
| modehb_lemonade/aid                           |             1 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |        0 |      0 |
| modehb_lemonade/coconut_trees                 |             1 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |        0 |      0 |
| modehb_lemonade/eggs                          |            20 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |        0 |     14 |
| modehb_lemonade/flowers                       |             3 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |        0 |      0 |
| modehb_lemonade/ham10000                      |            20 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |        0 |     16 |
| modehb_lemonade/larvae                        |            20 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |        0 |      9 |
| modehb_lemonade/remote_sensing                |            20 |    0 |               0 |      0 |      30 |          0 |      1 |          27 |       27 |     24 |
| modehb_lemonade_C01_final/remote_sensing      |            20 |   18 |               0 |      0 |      40 |          0 |      1 |          55 |       24 |     24 |
| modehb_spifil/flowers                         |             0 |    0 |               0 |      0 |       0 |          0 |     45 |           0 |        0 |      0 |
| modehb_spifil/remote_sensing                  |             0 |    0 |               0 |      0 |       0 |          0 |    247 |           0 |        0 |      0 |
| modehb_strat_fid120_1415/remote_sensing       |             2 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |        0 |      0 |
| modehb_strat_preHybrid_2325/remote_sensing    |             9 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |        0 |      0 |
| modehb_strat_prefix_2130/remote_sensing       |            20 |    0 |               0 |      0 |       0 |          0 |      1 |           0 |       26 |     15 |
| modehb_v3/remote_sensing                      |             0 |    0 |               0 |      0 |       0 |          0 |     20 |           0 |        0 |      0 |
| modehb_v3_old/remote_sensing                  |             0 |    0 |               0 |      0 |       0 |          0 |     36 |           0 |        0 |      0 |
| pnas_cs/remote_sensing                        |            25 |    0 |               1 |      9 |       0 |         14 |      1 |           0 |       10 |     19 |

## T1 Spearman ρ(κ_val, X) na fidelidade 5 (n≥12), por dataset × busca × assinatura de fitness
| dataset        | busca     | fitness                              |   n |   ρ_params |   ρ_flops |   ρ_depth |   ρ_max_width |   ρ_n_pools |   ρ_spifil_ipc |   ρ_depth_parcial_params |   ρ_ipc_parcial_params |
|:---------------|:----------|:-------------------------------------|----:|-----------:|----------:|----------:|--------------:|------------:|---------------:|-------------------------:|-----------------------:|
| remote_sensing | hybrid    | C0.1_kw0.75_cov_head_standard_mt4000 |  81 |       0.48 |      0.59 |     -0.27 |          0.58 |       -0.22 |           0.13 |                    -0.38 |                   0.05 |
| remote_sensing | hybrid    | C100_kw1_cov_head_standard_mt4000    |  27 |       0.65 |      0.53 |      0.08 |          0.64 |        0.22 |           0.49 |                    -0.14 |                   0.25 |
| remote_sensing | lemonade  | C0.1_kw0.75_cov_head_standard_mt4000 |  82 |       0.42 |      0.3  |      0.02 |          0.45 |        0.12 |           0.28 |                    -0.21 |                   0.15 |
| remote_sensing | lemonade  | C100_kw1_cov_head_standard_mt4000    |  85 |       0.48 |      0.54 |     -0.37 |          0.64 |       -0.23 |           0.02 |                    -0.53 |                  -0.08 |
| remote_sensing | pnas_cs   | C100_kw1_cov_head_standard_mt4000    |  59 |      -0.13 |     -0.61 |      0.18 |          0.24 |        0.3  |          -0.02 |                     0.32 |                  -0.05 |
| remote_sensing | spifil_v1 | C100_kw0.75_head_none_mt4000         |  63 |      -0.3  |     -0.01 |     -0.43 |         -0.19 |       -0.37 |           0.01 |                    -0.32 |                   0.16 |
| remote_sensing | strat     | C100_kw0.75_head_none_mt4000         |  49 |       0.1  |      0.37 |     -0.52 |          0.19 |       -0.46 |          -0.36 |                    -0.57 |                  -0.38 |
| remote_sensing | v3        | C100_kw0.75_head_none_mt10000        |  36 |      -0.05 |     -0.01 |     -0.26 |          0.03 |       -0.2  |          -0.04 |                    -0.34 |                  -0.03 |
| remote_sensing | v3        | C100_kw0.75_head_none_mt4000         |  20 |       0.28 |      0.32 |     -0.09 |          0.27 |       -0.09 |          -0.19 |                    -0.43 |                  -0.24 |
| ham10000       | hybrid    | C100_kw1_cov_head_standard_mt4000    |  43 |       0.48 |      0.55 |     -0.05 |          0.58 |        0.09 |           0.29 |                    -0.28 |                   0.17 |
| ham10000       | lemonade  | C100_kw1_cov_head_standard_mt4000    |  21 |       0.62 |      0.05 |      0.58 |          0.59 |        0.56 |           0.46 |                     0.15 |                   0.33 |
| flowers        | spifil_v1 | C100_kw0.75_head_none_mt4000         |  24 |       0.32 |      0.53 |     -0.16 |          0.21 |       -0.26 |           0.09 |                    -0.3  |                  -0.19 |
| aid            | hybrid    | C100_kw1_cov_head_standard_mt4000    |  43 |       0.51 |      0.66 |     -0.17 |          0.51 |       -0.03 |           0.06 |                    -0.43 |                  -0.14 |
| eggs           | lemonade  | C100_kw1_cov_head_standard_mt4000    |  19 |       0.68 |      0.53 |      0.35 |          0.74 |        0.41 |           0.36 |                    -0.33 |                   0.27 |
| larvae         | hybrid    | C100_kw1_cov_head_standard_mt4000    |  31 |       0.7  |      0.63 |      0.33 |          0.71 |        0.29 |           0.51 |                     0.11 |                   0.18 |
| larvae         | lemonade  | C100_kw1_cov_head_standard_mt4000    |  14 |      -0.08 |      0.53 |     -0.12 |         -0.14 |       -0.08 |          -0.13 |                    -0.1  |                  -0.1  |
| larvae_raw     | hybrid    | C100_kw1_cov_head_standard_mt4000    |  26 |       0.72 |      0.68 |      0.16 |          0.71 |        0.24 |           0.43 |                    -0.21 |                  -0.06 |
| protozoan      | hybrid    | C100_kw1_cov_head_standard_mt4000    |  31 |       0.61 |      0.54 |      0    |          0.63 |        0.23 |           0.61 |                    -0.18 |                   0.36 |

- agregado f5 cov_head (atual): n=584, ρ(percentil κ, params)=0.41, ρ(percentil κ, depth)=-0.05, ρ(percentil κ, n_pools)=0.05, ρ(percentil, max_width)=0.49
- agregado f5 head sem scaler (legado): n=192, ρ(percentil κ, params)=-0.05, ρ(percentil κ, depth)=-0.36, ρ(percentil κ, n_pools)=-0.33, ρ(percentil, max_width)=0.03

## T1b Spearman com a fidelidade MAIS ALTA de cada candidato (fitness atual C100_kw1_cov_head; mistura fidelidades)
| dataset        |   n |   ρ_params |   ρ_flops |   ρ_depth |   ρ_max_width |   ρ_n_pools |
|:---------------|----:|-----------:|----------:|----------:|--------------:|------------:|
| aid            |  40 |       0.52 |      0.65 |     -0.17 |          0.53 |       -0.05 |
| eggs           |  19 |       0.7  |      0.65 |      0.43 |          0.68 |        0.49 |
| ham10000       |  56 |       0.48 |      0.49 |      0.04 |          0.57 |        0.16 |
| larvae         |  37 |       0.53 |      0.59 |      0.19 |          0.55 |        0.19 |
| larvae_raw     |  22 |       0.69 |      0.64 |      0.15 |          0.68 |        0.24 |
| protozoan      |  27 |       0.65 |      0.52 |     -0.04 |          0.66 |        0.22 |
| remote_sensing | 156 |       0.36 |      0.33 |     -0.17 |          0.52 |       -0.1  |

## T2 redução espacial: n_pools / resolução final × κ (f5, fitness atual; percentil de κ dentro de cada busca×dataset)
|   n_pools |   n |   pct_mediano |   kz_medio |   res_final_med |   depth_med |       params_med |
|----------:|----:|--------------:|-----------:|----------------:|------------:|-----------------:|
|         0 |   1 |          0.19 |      -0.92 |             224 |           1 | 371880           |
|         1 |  34 |          0.25 |      -0.75 |             112 |           2 |  22654           |
|         2 |  80 |          0.38 |      -0.29 |              56 |           3 | 320874           |
|         3 |  90 |          0.6  |       0.22 |              28 |           4 | 632260           |
|         4 |  95 |          0.74 |       0.54 |              14 |           4 |      1.38512e+06 |
|         5 | 114 |          0.43 |      -0.18 |               7 |           6 |      1.25059e+06 |

## T2b profundidade × κ (f5, fitness atual)
|   depth |   n |   pct_mediano |   kz_medio |   pools_med |       params_med |
|--------:|----:|--------------:|-----------:|------------:|-----------------:|
|       1 |   1 |          0.19 |      -0.92 |           0 | 371880           |
|       2 |  48 |          0.28 |      -0.52 |           1 |  23408           |
|       3 |  88 |          0.44 |      -0.1  |           2 | 338611           |
|       4 | 116 |          0.67 |       0.34 |           4 | 942886           |
|       5 |  60 |          0.55 |       0.23 |           5 |      1.32419e+06 |
|       6 |  44 |          0.67 |       0.44 |           5 |      1.36246e+06 |
|       7 |  34 |          0.2  |      -0.74 |           5 | 932468           |
|       8 |  23 |          0.31 |      -0.59 |           5 | 880662           |

## T2c z-score médio de κ por n_pools e dataset (f5, fitness atual)
| dataset        |    0.0 |   1.0 |    2.0 |    3.0 |    4.0 |    5.0 |
|:---------------|-------:|------:|-------:|-------:|-------:|-------:|
| aid            | nan    | -0.14 |   0.02 |   0.24 |  -0.08 |  -0.24 |
| eggs           | nan    | -3.27 |  -1.24 |   0.17 |   0.65 |   0.29 |
| flowers        | nan    | -1.15 | nan    |   0.58 | nan    | nan    |
| ham10000       | nan    | -0.33 |  -0.5  |   0.07 |   0.18 |   0.15 |
| larvae         | nan    | -2.57 |  -0.97 |   0.51 |   0.52 |   0.15 |
| larvae_raw     | nan    | -0.71 |  -0.44 |   0.29 |   0.5  |  -0.25 |
| protozoan      | nan    | -1.12 |  -0.7  |   0.75 |   0.37 |  -1.11 |
| protozoan_raw  | nan    | -1.87 |   0.19 | nan    |   0.89 |   0.18 |
| remote_sensing |  -0.92 | -0.2  |  -0.13 |   0.02 |   0.78 |  -0.5  |

## T3 operador dominante (>50% dos blocos) × κ (f5, fitness atual)
| op_dom    |   n |   kz_medio |   pct_mediano |       params_med |   flops_med |
|:----------|----:|-----------:|--------------:|-----------------:|------------:|
| conv      | 171 |       0.08 |          0.52 | 923204           | 1.04421e+09 |
| dwsep     |  13 |       0.56 |          0.68 |      1.98446e+06 | 1.60443e+09 |
| inception |  52 |       0.31 |          0.6  | 886743           | 7.8151e+08  |
| misto     | 178 |      -0.21 |          0.47 | 632260           | 5.125e+08   |

## T3b OLS z(κ val) ~ log10(params) + depth + n_pools + frac_dwsep + frac_inception + n_residual (f5, fitness atual; IC 95% bootstrap)
| termo          |   coef |   ic_lo |   ic_hi |
|:---------------|-------:|--------:|--------:|
| intercepto     | -4.7   |  -5.572 |  -3.762 |
| depth          | -0.202 |  -0.285 |  -0.111 |
| n_pools        |  0.134 |   0.01  |   0.241 |
| frac_dwsep     |  0.151 |  -0.291 |   0.553 |
| frac_inception |  0.424 |   0.105 |   0.697 |
| n_residual     | -0.013 |  -0.079 |   0.054 |
| log10_params   |  0.877 |   0.701 |   1.034 |
n=414

## T3c composição média: top-10% × resto (f5, fitness atual)
| top   |   frac_conv |   frac_dwsep |   frac_inception |   n_residual |   depth |   n_pools |
|:------|------------:|-------------:|-----------------:|-------------:|--------:|----------:|
| False |        0.55 |         0.19 |             0.26 |         1.25 |    4.39 |      3.37 |
| True  |        0.44 |         0.17 |             0.38 |         1.59 |    4.27 |      3.71 |

## T5 eficiência das mutações: Δκ_val = κ(filho) − κ(pai), pai na MESMA fidelidade quando existe, senão na fidelidade herdada (≤ filho)
| busca    | fitness     | op              |   n |   dk_mediana |   dk_media |   frac_melhora |   dparams_med |   dk_por_100k |   frac_mesma_fid |
|:---------|:------------|:----------------|----:|-------------:|-----------:|---------------:|--------------:|--------------:|-----------------:|
| lemonade | C0.1_kw0.75 | deepen          |  15 |       -0.014 |     -0.02  |          0.267 |         28392 |        -0.009 |                1 |
| lemonade | C0.1_kw0.75 | kernel          |   9 |       -0.008 |     -0.006 |          0.333 |           576 |        -0.102 |                1 |
| lemonade | C0.1_kw0.75 | residual        |   6 |       -0.014 |     -0.012 |          0.5   |             0 |         0.007 |                1 |
| lemonade | C0.1_kw0.75 | widen           |  14 |       -0.007 |     -0.002 |          0.429 |         78970 |        -0.001 |                1 |
| lemonade | C100_kw1    | deepen          |  11 |       -0.012 |     -0.011 |          0.364 |         28280 |        -0.018 |                1 |
| lemonade | C100_kw1    | kernel          |   6 |       -0.027 |     -0.026 |          0     |           840 |        -0.223 |                1 |
| lemonade | C100_kw1    | residual        |   4 |       -0.022 |     -0.014 |          0.25  |             0 |        -0.092 |                1 |
| lemonade | C100_kw1    | widen           |  13 |       -0.014 |     -0.014 |          0.308 |          8578 |        -0.012 |                1 |
| pnas_cs  | C100_kw1    | add_layer       |   4 |       -0.01  |     -0.007 |          0.25  |         29729 |        -0.168 |                1 |
| pnas_cs  | C100_kw1    | add_residual    |   4 |       -0.005 |     -0.005 |          0.25  |        293114 |        -0.004 |                1 |
| pnas_cs  | C100_kw1    | grow(P_k→P_k+1) |   9 |       -0.011 |     -0.01  |          0.222 |        832160 |        -0.002 |                1 |
| pnas_cs  | C100_kw1    | pooling         |   3 |       -0.007 |     -0.014 |          0     |             0 |               |                1 |
| pnas_cs  | C100_kw1    | widen           |   3 |       -0.022 |     -0.017 |          0.333 |         68197 |        -0.006 |                1 |

## T5b agregado por operador (todas as buscas/assinaturas; só remote_sensing tem pares)
| op              |   n |   dk_mediana |   dk_media |   frac_melhora |   dparams_med |   dk_por_100k |   frac_mesma_fid |
|:----------------|----:|-------------:|-----------:|---------------:|--------------:|--------------:|-----------------:|
| add_layer       |   4 |       -0.01  |     -0.007 |          0.25  |         29729 |        -0.168 |                1 |
| add_residual    |   4 |       -0.005 |     -0.005 |          0.25  |        293114 |        -0.004 |                1 |
| deepen          |  26 |       -0.013 |     -0.016 |          0.308 |         28336 |        -0.013 |                1 |
| grow(P_k→P_k+1) |   9 |       -0.011 |     -0.01  |          0.222 |        832160 |        -0.002 |                1 |
| kernel          |  15 |       -0.014 |     -0.014 |          0.2   |           576 |        -0.102 |                1 |
| pooling         |   3 |       -0.007 |     -0.014 |          0     |             0 |               |                1 |
| residual        |  10 |       -0.017 |     -0.013 |          0.4   |             0 |         0.007 |                1 |
| widen           |  30 |       -0.008 |     -0.008 |          0.367 |         26954 |        -0.002 |                1 |

## T5c datasets com pares pai→filho
| dataset        | busca    |   n |
|:---------------|:---------|----:|
| remote_sensing | lemonade |  78 |
| remote_sensing | pnas_cs  |  24 |

## T5d promoções (mesmo candidato, fidelidade f_lo → f_hi), todas as buscas
|   f_lo |   f_hi |   n |   dk_mediana |   frac_melhora |
|-------:|-------:|----:|-------------:|---------------:|
|      5 |     15 | 202 |        0     |          0.406 |
|      5 |     20 |   1 |       -0.076 |          0     |
|     15 |     45 | 119 |        0.001 |          0.529 |
|     20 |     60 |   1 |       -0.011 |          0     |
|     45 |    100 |  91 |        0     |          0.363 |
|    100 |    200 |  91 |        0     |          0.462 |

| dataset        |   n |   dk_mediana |   frac_melhora |
|:---------------|----:|-------------:|---------------:|
| aid            |  22 |        0     |          0.364 |
| coconut_trees  |   3 |       -0.007 |          0     |
| eggs           |  18 |        0.001 |          0.556 |
| eggs_raw       |   2 |        0.007 |          1     |
| flowers        |  37 |        0.006 |          0.541 |
| ham10000       |  37 |        0.005 |          0.541 |
| larvae         |  35 |        0     |          0.2   |
| larvae_raw     |  16 |        0.002 |          0.5   |
| protozoan      |  20 |       -0.001 |          0.45  |
| protozoan_raw  |  16 |        0     |          0.5   |
| remote_sensing | 299 |        0     |          0.428 |

## T4 benefício marginal por bloco: pares pai→filho com profundidade maior (deepen / add_layer / grow)
| busca    | fitness     | op              | child   | parent   |   fid |   parent_fid_ref |   k_parent |   k_child |     dk |   d_depth |   dk_por_bloco |         p_parent |          p_child | layer_map                              |
|:---------|:------------|:----------------|:--------|:---------|------:|-----------------:|-----------:|----------:|-------:|----------:|---------------:|-----------------:|-----------------:|:---------------------------------------|
| pnas_cs  | C100_kw1    | add_layer       | c00055  | c00008   |     5 |                5 |      0.577 |     0.534 | -0.043 |         1 |         -0.043 |      1.38354e+06 |      1.44122e+06 | [0, 1, "identity", 2, 3, 4, 5]         |
| pnas_cs  | C100_kw1    | add_layer       | c00034  | c00027   |     5 |                5 |      0.544 |     0.528 | -0.016 |         1 |         -0.016 | 532746           | 534524           | [0, 1, "identity", 2, 3]               |
| pnas_cs  | C100_kw1    | add_layer       | c00027  | c00012   |     5 |                5 |      0.549 |     0.544 | -0.005 |         1 |         -0.005 | 530968           | 532746           | [0, "identity", 1, 2]                  |
| pnas_cs  | C100_kw1    | add_layer       | c00039  | c00019   |     5 |                5 |      0.499 |     0.534 |  0.035 |         1 |          0.035 |      3.37438e+06 |      3.94616e+06 | [0, 1, 2, 3, "identity", 4, 5]         |
| pnas_cs  | C100_kw1    | add_residual    | c00035  | c00027   |     5 |                5 |      0.544 |     0.528 | -0.016 |         1 |         -0.016 | 532746           | 534524           | [0, 1, "residual_zero", 2, 3]          |
| pnas_cs  | C100_kw1    | add_residual    | c00040  | c00039   |     5 |                5 |      0.534 |     0.525 | -0.009 |         1 |         -0.009 |      3.94616e+06 |      4.51795e+06 | [0, 1, 2, 3, 4, "residual_zero", 5, 6] |
| pnas_cs  | C100_kw1    | add_residual    | c00057  | c00008   |     5 |                5 |      0.577 |     0.576 | -0.001 |         1 |         -0.001 |      1.38354e+06 |      1.39798e+06 | [0, "residual_zero", 1, 2, 3, 4, 5]    |
| pnas_cs  | C100_kw1    | add_residual    | c00030  | c00019   |     5 |                5 |      0.499 |     0.505 |  0.007 |         1 |          0.007 |      3.37438e+06 |      3.94616e+06 | [0, 1, 2, "residual_zero", 3, 4, 5]    |
| lemonade | C0.1_kw0.75 | deepen          | c00076  | c00014   |     5 |                5 |      0.584 |     0.514 | -0.069 |         1 |         -0.069 |      1.31834e+06 |      1.32196e+06 | nan                                    |
| lemonade | C0.1_kw0.75 | deepen          | c00063  | c00019   |     5 |                5 |      0.582 |     0.512 | -0.069 |         1 |         -0.069 |      1.73351e+06 |      1.73371e+06 | nan                                    |
| lemonade | C100_kw1    | deepen          | c00045  | c00026   |     5 |                5 |      0.434 |     0.367 | -0.067 |         1 |         -0.067 | 471476           | 471632           | nan                                    |
| lemonade | C0.1_kw0.75 | deepen          | c00041  | c00017   |     5 |                5 |      0.541 |     0.5   | -0.041 |         1 |         -0.041 | 334992           | 335922           | nan                                    |
| lemonade | C0.1_kw0.75 | deepen          | c00042  | c00031   |     5 |                5 |      0.57  |     0.534 | -0.036 |         1 |         -0.036 |      1.34662e+06 |      3.1543e+06  | nan                                    |
| lemonade | C0.1_kw0.75 | deepen          | c00098  | c00067   |    15 |               15 |      0.578 |     0.543 | -0.035 |         1 |         -0.035 | 390856           | 530968           | nan                                    |
| lemonade | C100_kw1    | deepen          | c00041  | c00026   |     5 |                5 |      0.434 |     0.398 | -0.035 |         1 |         -0.035 | 471476           | 471490           | nan                                    |
| lemonade | C100_kw1    | deepen          | c00056  | c00016   |     5 |                5 |      0.459 |     0.428 | -0.031 |         1 |         -0.031 |  18878           |  18986           | nan                                    |
| lemonade | C0.1_kw0.75 | deepen          | c00024  | c00001   |     5 |                5 |      0.466 |     0.445 | -0.021 |         1 |         -0.021 | 163386           | 179304           | nan                                    |
| lemonade | C0.1_kw0.75 | deepen          | c00020  | c00003   |     5 |                5 |      0.548 |     0.534 | -0.014 |         1 |         -0.014 |      1.24382e+06 |      1.42036e+06 | nan                                    |
| lemonade | C100_kw1    | deepen          | c00024  | c00001   |     5 |                5 |      0.466 |     0.452 | -0.014 |         1 |         -0.014 | 163386           | 179304           | nan                                    |
| lemonade | C0.1_kw0.75 | deepen          | c00028  | c00019   |     5 |                5 |      0.582 |     0.568 | -0.014 |         1 |         -0.014 |      1.73351e+06 |      1.81555e+06 | nan                                    |
| lemonade | C0.1_kw0.75 | deepen          | c00031  | c00014   |     5 |                5 |      0.584 |     0.57  | -0.013 |         1 |         -0.013 |      1.31834e+06 |      1.34662e+06 | nan                                    |
| lemonade | C100_kw1    | deepen          | c00058  | c00014   |     5 |                5 |      0.582 |     0.571 | -0.012 |         1 |         -0.012 |      1.31834e+06 |      1.38512e+06 | nan                                    |
| lemonade | C100_kw1    | deepen          | c00031  | c00014   |     5 |                5 |      0.582 |     0.571 | -0.012 |         1 |         -0.012 |      1.31834e+06 |      1.34662e+06 | nan                                    |
| lemonade | C100_kw1    | deepen          | c00042  | c00031   |     5 |                5 |      0.571 |     0.562 | -0.009 |         1 |         -0.009 |      1.34662e+06 |      3.1543e+06  | nan                                    |
| lemonade | C0.1_kw0.75 | deepen          | c00059  | c00043   |     5 |                5 |      0.583 |     0.575 | -0.007 |         1 |         -0.007 |      1.74947e+06 |      1.83151e+06 | nan                                    |
| lemonade | C0.1_kw0.75 | deepen          | c00059  | c00043   |     5 |                5 |      0.583 |     0.575 | -0.007 |         1 |         -0.007 |      1.74947e+06 |      1.83151e+06 | nan                                    |
| lemonade | C100_kw1    | deepen          | c00072  | c00016   |     5 |                5 |      0.459 |     0.46  |  0.002 |         1 |          0.002 |  18878           |  18992           | nan                                    |
| lemonade | C100_kw1    | deepen          | c00061  | c00049   |     5 |                5 |      0.456 |     0.462 |  0.006 |         1 |          0.006 | 490804           | 577338           | nan                                    |
| lemonade | C0.1_kw0.75 | deepen          | c00057  | c00049   |     5 |                5 |      0.541 |     0.548 |  0.007 |         1 |          0.007 | 665804           | 671754           | nan                                    |
| lemonade | C100_kw1    | deepen          | c00020  | c00003   |     5 |                5 |      0.548 |     0.555 |  0.007 |         1 |          0.007 |      1.24382e+06 |      1.42036e+06 | nan                                    |
| lemonade | C0.1_kw0.75 | deepen          | c00092  | c00050   |    15 |               15 |      0.555 |     0.563 |  0.008 |         1 |          0.008 |      1.29504e+06 |      1.74729e+06 | nan                                    |
| lemonade | C0.1_kw0.75 | deepen          | c00081  | c00043   |    15 |               15 |      0.583 |     0.591 |  0.008 |         1 |          0.008 |      1.74947e+06 |      1.77787e+06 | nan                                    |
| lemonade | C0.1_kw0.75 | deepen          | c00060  | c00048   |     5 |                5 |      0.515 |     0.523 |  0.008 |         1 |          0.008 |  19988           |  21320           | nan                                    |
| lemonade | C100_kw1    | deepen          | c00028  | c00019   |     5 |                5 |      0.567 |     0.611 |  0.044 |         1 |          0.044 |      1.73351e+06 |      1.81555e+06 | nan                                    |
| pnas_cs  | C100_kw1    | grow(P_k→P_k+1) | c00054  | c00044   |     5 |                5 |      0.583 |     0.548 | -0.034 |         1 |         -0.034 |      2.25994e+06 |      3.73002e+06 | [0, 1, 2, "identity", 3, 4, 5, 6]      |
| pnas_cs  | C100_kw1    | grow(P_k→P_k+1) | c00043  | c00027   |     5 |                5 |      0.544 |     0.533 | -0.011 |         1 |         -0.011 | 532746           | 581942           | [0, 1, "identity", 2, 3]               |
| pnas_cs  | C100_kw1    | grow(P_k→P_k+1) | c00029  | c00008   |     5 |                5 |      0.577 |     0.576 | -0.001 |         1 |         -0.001 |      1.38354e+06 |      2.25994e+06 | [0, 1, 2, 3, 4, 5, "identity"]         |
| pnas_cs  | C100_kw1    | grow(P_k→P_k+1) | c00044  | c00033   |     5 |                5 |      0.577 |     0.583 |  0.006 |         1 |          0.006 |      1.38354e+06 |      2.25994e+06 | [0, 1, 2, 3, 4, 5, "identity"]         |

- resumo: n=38, Δκ/bloco mediana=-0.0114, média=-0.0133, fração>0=0.29; mesma fidelidade: n=38, mediana=-0.0114
- controle (mesma profundidade: widen/kernel/residual/pooling): n=64, Δκ mediana=-0.0103, fração>0=0.30

## T4b PNAS-CS: famílias (mesma base, níveis P0..P4)
| op_family   |   n | levels   |   kmax |             pmin |             pmax |
|:------------|----:|:---------|-------:|-----------------:|-----------------:|
| F480cf5f505 |   3 | P1,P2,P3 |  0.583 |      1.38354e+06 |      3.73002e+06 |
| F675b105060 |   2 | P2,P3    |  0.578 |      1.92644e+06 |      2.7586e+06  |
| Fb3c5a34236 |   2 | P0       |  0.577 | 530968           | 676078           |
| F9a1155a011 |   2 | P1,P2    |  0.577 |      1.38354e+06 |      2.25994e+06 |
| Fea412afbe3 |   1 | P2       |  0.576 |      2.25994e+06 |      2.25994e+06 |
| F99277a6364 |   1 | P1       |  0.576 |      1.39798e+06 |      1.39798e+06 |
| F067d7d31c2 |   1 | P1       |  0.562 |      1.03199e+06 |      1.03199e+06 |
| Fb695726199 |   2 | P0,P1    |  0.556 | 461474           | 826943           |
| F6ce5ae4193 |   1 | P2       |  0.555 |      1.75164e+06 |      1.75164e+06 |
| F5221a25cb0 |   1 | P1       |  0.554 |      1.19538e+06 |      1.19538e+06 |
| F60d5a71e5c |   1 | P0       |  0.55  | 245586           | 245586           |
| F9634011927 |   1 | P2       |  0.549 |      1.78744e+06 |      1.78744e+06 |
| F053b7b1a8f |   1 | P1       |  0.548 |      1.05146e+06 |      1.05146e+06 |
| F8130f640c6 |   1 | P4       |  0.548 |      5.37609e+06 |      5.37609e+06 |
| F0e81eaa605 |   2 | P0       |  0.544 | 532746           | 581942           |

| op_family   |      P0 |      P1 |      P2 |      P3 |      P4 |
|:------------|--------:|--------:|--------:|--------:|--------:|
| F053b7b1a8f | nan     |   0.548 | nan     | nan     | nan     |
| F062f96ea9b | nan     | nan     |   0.472 | nan     | nan     |
| F067d7d31c2 | nan     |   0.562 | nan     | nan     | nan     |
| F0e81eaa605 |   0.544 | nan     | nan     | nan     | nan     |
| F0fb594b987 |   0.523 | nan     | nan     | nan     | nan     |
| F14a85f0ec1 |   0.512 | nan     | nan     | nan     | nan     |
| F171dd209e0 |   0.515 | nan     | nan     | nan     | nan     |
| F1f8278e9a6 | nan     | nan     | nan     |   0.507 |   0.5   |
| F283de84f09 | nan     |   0.472 | nan     | nan     | nan     |
| F3227bc1a8a |   0.528 | nan     | nan     | nan     | nan     |
| F3675ce7dd2 | nan     | nan     |   0.534 | nan     | nan     |
| F3b640b242d | nan     |   0.488 | nan     | nan     | nan     |
| F3ed4ec5735 | nan     |   0.501 | nan     | nan     | nan     |
| F480cf5f505 | nan     |   0.577 |   0.583 |   0.548 | nan     |
| F5221a25cb0 | nan     |   0.554 | nan     | nan     | nan     |
| F55be5f2b3a |   0.494 | nan     | nan     | nan     | nan     |
| F59fc5511b5 | nan     | nan     |   0.534 | nan     | nan     |
| F60d5a71e5c |   0.55  | nan     | nan     | nan     | nan     |
| F630423429d |   0.528 | nan     | nan     | nan     | nan     |
| F675b105060 | nan     | nan     |   0.578 |   0.563 | nan     |
| F67d9747918 | nan     | nan     |   0.528 | nan     | nan     |
| F6a64ea54cd | nan     | nan     | nan     |   0.534 | nan     |
| F6b7554c584 | nan     | nan     | nan     | nan     |   0.486 |
| F6ce5ae4193 | nan     | nan     |   0.555 | nan     | nan     |
| F74d832a089 | nan     | nan     |   0.52  | nan     | nan     |
| F77f69aac37 | nan     | nan     | nan     | nan     |   0.525 |
| F7d137ea4cf |   0.522 | nan     | nan     | nan     | nan     |
| F8130f640c6 | nan     | nan     | nan     | nan     |   0.548 |
| F85d8ce9ab6 |   0.535 | nan     | nan     | nan     | nan     |
| F9311edde34 | nan     | nan     | nan     |   0.534 | nan     |
| F9634011927 | nan     | nan     |   0.549 | nan     | nan     |
| F99277a6364 | nan     |   0.576 | nan     | nan     | nan     |
| F9a1155a011 | nan     |   0.577 |   0.576 | nan     | nan     |
| F9e9220a544 |   0.529 | nan     | nan     | nan     | nan     |
| Fa6183bee3a | nan     | nan     | nan     |   0.505 | nan     |
| Fb3c5a34236 |   0.577 | nan     | nan     | nan     | nan     |
| Fb695726199 |   0.556 |   0.535 | nan     | nan     | nan     |
| Fbd2d53d118 |   0.501 | nan     | nan     | nan     | nan     |
| Fc45c563ee9 |   0.537 |   0.501 | nan     | nan     | nan     |
| Fcf401ac515 | nan     | nan     | nan     | nan     |   0.423 |
| Fdcf839c6dc | nan     | nan     | nan     | nan     |   0.528 |
| Fea412afbe3 | nan     | nan     |   0.576 | nan     | nan     |
| Fed2c70a56e | nan     | nan     | nan     |   0.486 | nan     |
| Ff9fce1c2d9 | nan     | nan     | nan     |   0.499 | nan     |

## T6 melhor checkpoint por orçamento (≤ cap, cumulativo) selecionado por κ VALIDAÇÃO (fitness atual C100_kw1_cov_head); teste só informativo
| dataset        | B   |   n_cand |   n_na_faixa | busca    | run             | host   | cand   |   fid |   params |   flops_G |   depth | widths              |   pools |   k_val |   k_test_ORIG |   k_test_COV | veio_de_faixa_menor   |
|:---------------|:----|---------:|-------------:|:---------|:----------------|:-------|:-------|------:|---------:|----------:|--------:|:--------------------|--------:|--------:|--------------:|-------------:|:----------------------|
| remote_sensing | B0  |       31 |           31 | hybrid   | modehb_hybrid   | jaci   | c00009 |     5 |   132620 |     0.875 |       2 | 70-280              |       2 |   0.591 |               |              | False                 |
| remote_sensing | B1  |       54 |           34 | hybrid   | modehb_hybrid   | jaci   | c00009 |     5 |   132620 |     0.875 |       2 | 70-280              |       2 |   0.591 |               |              | True                  |
| remote_sensing | B2  |       71 |           33 | hybrid   | modehb_hybrid   | jaci   | c00009 |     5 |   132620 |     0.875 |       2 | 70-280              |       2 |   0.591 |               |              | True                  |
| remote_sensing | B3  |       80 |           30 | lemonade | modehb_lemonade | jaci   | c00028 |     5 |  1815554 |     0.417 |       6 | 5-14-56-280-280-448 |       4 |   0.611 |         0.494 |        0.568 | False                 |
| remote_sensing | B4  |       80 |           14 | lemonade | modehb_lemonade | jaci   | c00028 |     5 |  1815554 |     0.417 |       6 | 5-14-56-280-280-448 |       4 |   0.611 |         0.494 |        0.568 | True                  |
| ham10000       | B0  |       13 |           13 | hybrid   | modehb_hybrid   | jaci   | c00025 |     5 |   169748 |     0.625 |       3 | 28-28-56            |       2 |   0.586 |         0.499 |        0.624 | False                 |
| ham10000       | B1  |       25 |           15 | hybrid   | modehb_hybrid   | jaci   | c00034 |     5 |   466666 |     2.072 |       2 | 35-196              |       2 |   0.601 |               |              | False                 |
| ham10000       | B2  |       33 |           11 | hybrid   | modehb_hybrid   | jaci   | c00034 |     5 |   466666 |     2.072 |       2 | 35-196              |       2 |   0.601 |               |              | True                  |
| ham10000       | B3  |       39 |           12 | hybrid   | modehb_hybrid   | jaci   | c00034 |     5 |   466666 |     2.072 |       2 | 35-196              |       2 |   0.601 |               |              | True                  |
| ham10000       | B4  |       39 |            0 | hybrid   | modehb_hybrid   | jaci   | c00034 |     5 |   466666 |     2.072 |       2 | 35-196              |       2 |   0.601 |               |              | True                  |
| flowers        | B0  |        1 |            1 | lemonade | modehb_lemonade | athena | c00001 |   200 |   181715 |     0.808 |       3 | nan                 |         |   0.699 |         0.476 |        0.717 | False                 |
| flowers        | B1  |        2 |            1 | hybrid   | modehb_hybrid   | jaci   | c00002 |   200 |   683680 |     0.827 |       4 | 15-50-135-1020      |       3 |   0.748 |               |              | False                 |
| flowers        | B2  |        3 |            1 | hybrid   | modehb_hybrid   | jaci   | c00003 |    45 |   894960 |     2.398 |       3 | 25-60-205           |       3 |   0.757 |               |              | False                 |
| flowers        | B3  |        3 |            0 | hybrid   | modehb_hybrid   | jaci   | c00003 |    45 |   894960 |     2.398 |       3 | 25-60-205           |       3 |   0.757 |               |              | True                  |
| flowers        | B4  |        3 |            0 | hybrid   | modehb_hybrid   | jaci   | c00003 |    45 |   894960 |     2.398 |       3 | 25-60-205           |       3 |   0.757 |               |              | True                  |
| aid            | B0  |       11 |           11 | hybrid   | modehb_hybrid   | jaci   | c00025 |     5 |   212710 |     0.763 |       3 | 29-29-71            |       2 |   0.819 |         0.719 |        0.849 | False                 |
| aid            | B1  |       23 |           12 | hybrid   | modehb_hybrid   | jaci   | c00038 |     5 |   502304 |     2.104 |       5 | 29-90-112-112-112   |       5 |   0.83  |               |              | False                 |
| aid            | B2  |       31 |            8 | hybrid   | modehb_hybrid   | jaci   | c00038 |     5 |   502304 |     2.104 |       5 | 29-90-112-112-112   |       5 |   0.83  |               |              | True                  |
| aid            | B3  |       38 |            7 | hybrid   | modehb_hybrid   | jaci   | c00020 |     5 |  2137000 |     3.038 |       4 | 30-240-420-420      |       3 |   0.845 |               |              | False                 |
| aid            | B4  |       39 |            1 | hybrid   | modehb_hybrid   | jaci   | c00020 |     5 |  2137000 |     3.038 |       4 | 30-240-420-420      |       3 |   0.845 |               |              | True                  |
| coconut_trees  | B0  |        1 |            1 | lemonade | modehb_lemonade | athena | c00001 |   200 |   200944 |     0.907 |       3 | nan                 |         |   0.732 |         0.699 |        0.692 | False                 |
| coconut_trees  | B1  |        2 |            1 | hybrid   | modehb_hybrid   | athena | c00002 |   200 |   473912 |     3.633 |       3 | nan                 |         |   0.763 |         0.743 |        0.736 | False                 |
| coconut_trees  | B2  |        3 |            1 | hybrid   | modehb_hybrid   | athena | c00003 |   100 |   991302 |     3.282 |       3 | nan                 |         |   0.766 |         0.801 |        0.76  | False                 |
| coconut_trees  | B3  |        4 |            1 | hybrid   | modehb_hybrid   | athena | c00004 |   100 |  2297872 |     3.131 |       4 | nan                 |         |   0.79  |         0.772 |        0.79  | False                 |
| coconut_trees  | B4  |        4 |            0 | hybrid   | modehb_hybrid   | athena | c00004 |   100 |  2297872 |     3.131 |       4 | nan                 |         |   0.79  |         0.772 |        0.79  | True                  |
| eggs           | B0  |        4 |            4 | lemonade | modehb_lemonade | jaci   | c00001 |    45 |   185330 |     0.768 |       3 | 27-45-63            |       3 |   0.899 |               |              | False                 |
| eggs           | B1  |        7 |            3 | lemonade | modehb_lemonade | jaci   | c00009 |     5 |   752924 |     0.491 |       4 | 9-27-99-198         |       4 |   0.937 |               |              | False                 |
| eggs           | B2  |       11 |            4 | lemonade | modehb_lemonade | jaci   | c00009 |     5 |   752924 |     0.491 |       4 | 9-27-99-198         |       4 |   0.937 |               |              | True                  |
| eggs           | B3  |       15 |            4 | lemonade | modehb_lemonade | jaci   | c00009 |     5 |   752924 |     0.491 |       4 | 9-27-99-198         |       4 |   0.937 |               |              | True                  |
| eggs           | B4  |       15 |            0 | lemonade | modehb_lemonade | jaci   | c00009 |     5 |   752924 |     0.491 |       4 | 9-27-99-198         |       4 |   0.937 |               |              | True                  |
| eggs_raw       | B0  |        1 |            1 | hybrid   | modehb_hybrid   | athena | c00001 |   200 |    12919 |     0.147 |       2 | nan                 |         |   0.527 |         0.521 |        0.564 | False                 |
| eggs_raw       | B1  |        2 |            1 | hybrid   | modehb_hybrid   | athena | c00002 |   200 |   562463 |     0.133 |       5 | nan                 |         |   0.753 |         0.608 |        0.797 | False                 |
| eggs_raw       | B2  |        3 |            1 | hybrid   | modehb_hybrid   | athena | c00003 |   200 |  1342070 |     1.319 |       4 | nan                 |         |   0.878 |         0.22  |        0.922 | False                 |
| eggs_raw       | B3  |        4 |            1 | hybrid   | modehb_hybrid   | athena | c00004 |   200 |  1697303 |     0.731 |       5 | nan                 |         |   0.896 |         0.684 |        0.922 | False                 |
| eggs_raw       | B4  |        4 |            0 | hybrid   | modehb_hybrid   | athena | c00004 |   200 |  1697303 |     0.731 |       5 | nan                 |         |   0.896 |         0.684 |        0.922 | True                  |
| larvae         | B0  |        8 |            8 | lemonade | modehb_lemonade | jaci   | c00001 |   200 |   200944 |     0.907 |       3 | 32-48-64            |       3 |   0.984 |         0.693 |        0.915 | False                 |
| larvae         | B1  |       15 |            8 | lemonade | modehb_lemonade | jaci   | c00001 |   200 |   200944 |     0.907 |       3 | 32-48-64            |       3 |   0.984 |         0.693 |        0.915 | True                  |
| larvae         | B2  |       22 |            9 | lemonade | modehb_lemonade | jaci   | c00001 |   200 |   200944 |     0.907 |       3 | 32-48-64            |       3 |   0.984 |         0.693 |        0.915 | True                  |
| larvae         | B3  |       26 |            6 | lemonade | modehb_lemonade | jaci   | c00001 |   200 |   200944 |     0.907 |       3 | 32-48-64            |       3 |   0.984 |         0.693 |        0.915 | True                  |
| larvae         | B4  |       27 |            1 | lemonade | modehb_lemonade | jaci   | c00001 |   200 |   200944 |     0.907 |       3 | 32-48-64            |       3 |   0.984 |         0.693 |        0.915 | True                  |
| larvae_raw     | B0  |        6 |            6 | hybrid   | modehb_hybrid   | jaci   | c00021 |     5 |   119806 |     0.373 |       3 | 2-40-62             |       2 |   0.785 |               |              | False                 |
| larvae_raw     | B1  |       12 |            6 | hybrid   | modehb_hybrid   | jaci   | c00022 |     5 |   643620 |     0.717 |       4 | 1-130-162-162       |       4 |   0.863 |               |              | False                 |
| larvae_raw     | B2  |       17 |            5 | hybrid   | modehb_hybrid   | jaci   | c00003 |    15 |  1354728 |     1.32  |       4 | 10-32-136-428       |       4 |   0.863 |               |              | False                 |
| larvae_raw     | B3  |       22 |            5 | hybrid   | modehb_hybrid   | jaci   | c00004 |    45 |  1973718 |     0.906 |       5 | 6-16-104-818-1024   |       5 |   0.915 |         0.188 |        0.74  | False                 |
| larvae_raw     | B4  |       22 |            0 | hybrid   | modehb_hybrid   | jaci   | c00004 |    45 |  1973718 |     0.906 |       5 | 6-16-104-818-1024   |       5 |   0.915 |         0.188 |        0.74  | True                  |
| protozoan      | B0  |        7 |            7 | hybrid   | modehb_hybrid   | jaci   | c00021 |     5 |   137396 |     0.426 |       3 | 1-49-49             |       2 |   0.773 |               |              | False                 |
| protozoan      | B1  |       14 |            7 | hybrid   | modehb_hybrid   | jaci   | c00006 |     5 |   332576 |     0.453 |       3 | 21-84-224           |       3 |   0.897 |               |              | False                 |
| protozoan      | B2  |       20 |            6 | hybrid   | modehb_hybrid   | jaci   | c00006 |     5 |   332576 |     0.453 |       3 | 21-84-224           |       3 |   0.897 |               |              | True                  |
| protozoan      | B3  |       26 |            6 | hybrid   | modehb_hybrid   | jaci   | c00006 |     5 |   332576 |     0.453 |       3 | 21-84-224           |       3 |   0.897 |               |              | True                  |
| protozoan      | B4  |       27 |            1 | hybrid   | modehb_hybrid   | jaci   | c00006 |     5 |   332576 |     0.453 |       3 | 21-84-224           |       3 |   0.897 |               |              | True                  |
| protozoan_raw  | B0  |        2 |            2 | hybrid   | modehb_hybrid   | jaci   | c00005 |     5 |    54591 |     0.207 |       3 | 7-24-35             |       2 |   0.681 |               |              | False                 |
| protozoan_raw  | B1  |        4 |            2 | hybrid   | modehb_hybrid   | jaci   | c00006 |     5 |   775172 |     0.503 |       4 | 7-28-105-196        |       4 |   0.776 |               |              | False                 |
| protozoan_raw  | B2  |        6 |            2 | hybrid   | modehb_hybrid   | jaci   | c00003 |   100 |  1306514 |     1.185 |       4 | 7-28-133-427        |       4 |   0.797 |               |              | False                 |
| protozoan_raw  | B3  |        7 |            1 | hybrid   | modehb_hybrid   | jaci   | c00003 |   100 |  1306514 |     1.185 |       4 | 7-28-133-427        |       4 |   0.797 |               |              | True                  |
| protozoan_raw  | B4  |        7 |            0 | hybrid   | modehb_hybrid   | jaci   | c00003 |   100 |  1306514 |     1.185 |       4 | 7-28-133-427        |       4 |   0.797 |               |              | True                  |

## T6b idem para assinaturas antigas (NÃO comparáveis com T6: outra régua/C)
| dataset        | fitness                              | B   |   n | cand   | run                            |   fid |   params |   k_val |   k_test_ORIG |
|:---------------|:-------------------------------------|:----|----:|:-------|:-------------------------------|------:|---------:|--------:|--------------:|
| aid            | C0.1_kw0.75_cov_head_standard_mt4000 | B0  |   1 | c00001 | modehb_hybrid_fullepoch_1555   |     5 |    23657 |   0.733 |               |
| aid            | C0.1_kw0.75_cov_head_standard_mt4000 | B1  |   1 | c00001 | modehb_hybrid_fullepoch_1555   |     5 |    23657 |   0.733 |               |
| aid            | C0.1_kw0.75_cov_head_standard_mt4000 | B2  |   1 | c00001 | modehb_hybrid_fullepoch_1555   |     5 |    23657 |   0.733 |               |
| aid            | C0.1_kw0.75_cov_head_standard_mt4000 | B3  |   1 | c00001 | modehb_hybrid_fullepoch_1555   |     5 |    23657 |   0.733 |               |
| aid            | C0.1_kw0.75_cov_head_standard_mt4000 | B4  |   1 | c00001 | modehb_hybrid_fullepoch_1555   |     5 |    23657 |   0.733 |               |
| coconut_trees  | C0.1_kw0.75_cov_head_standard_mt4000 | B0  |   1 | c00001 | modehb_hybrid_pre_aesched_0033 |     5 |    23488 |   0.758 |               |
| coconut_trees  | C0.1_kw0.75_cov_head_standard_mt4000 | B1  |   1 | c00001 | modehb_hybrid_pre_aesched_0033 |     5 |    23488 |   0.758 |               |
| coconut_trees  | C0.1_kw0.75_cov_head_standard_mt4000 | B2  |   1 | c00001 | modehb_hybrid_pre_aesched_0033 |     5 |    23488 |   0.758 |               |
| coconut_trees  | C0.1_kw0.75_cov_head_standard_mt4000 | B3  |   1 | c00001 | modehb_hybrid_pre_aesched_0033 |     5 |    23488 |   0.758 |               |
| coconut_trees  | C0.1_kw0.75_cov_head_standard_mt4000 | B4  |   1 | c00001 | modehb_hybrid_pre_aesched_0033 |     5 |    23488 |   0.758 |               |
| flowers        | C0.1_kw0.75_cov_head_standard_mt4000 | B0  |   1 | c00001 | modehb_hybrid_C01_final        |   200 |    21079 |   0.626 |         0.649 |
| flowers        | C0.1_kw0.75_cov_head_standard_mt4000 | B1  |   2 | c00002 | modehb_hybrid_C01_final        |   100 |   449360 |   0.781 |         0.635 |
| flowers        | C0.1_kw0.75_cov_head_standard_mt4000 | B2  |   3 | c00002 | modehb_hybrid_C01_final        |   100 |   449360 |   0.781 |         0.635 |
| flowers        | C0.1_kw0.75_cov_head_standard_mt4000 | B3  |   4 | c00002 | modehb_hybrid_C01_final        |   100 |   449360 |   0.781 |         0.635 |
| flowers        | C0.1_kw0.75_cov_head_standard_mt4000 | B4  |   4 | c00002 | modehb_hybrid_C01_final        |   100 |   449360 |   0.781 |         0.635 |
| flowers        | C100_kw0.75_head_none_mt4000         | B0  |  15 | c00003 | modehb_spifil                  |    15 |   236600 |   0.675 |               |
| flowers        | C100_kw0.75_head_none_mt4000         | B1  |  21 | c00003 | modehb_spifil                  |    15 |   236600 |   0.675 |               |
| flowers        | C100_kw0.75_head_none_mt4000         | B2  |  22 | c00003 | modehb_spifil                  |    15 |   236600 |   0.675 |               |
| flowers        | C100_kw0.75_head_none_mt4000         | B3  |  24 | c00003 | modehb_spifil                  |    15 |   236600 |   0.675 |               |
| flowers        | C100_kw0.75_head_none_mt4000         | B4  |  24 | c00003 | modehb_spifil                  |    15 |   236600 |   0.675 |               |
| ham10000       | C0.1_kw0.75_cov_head_standard_mt4000 | B0  |   1 | c00001 | modehb_hybrid_C01_2049         |    45 |    19550 |   0.526 |         0.511 |
| ham10000       | C0.1_kw0.75_cov_head_standard_mt4000 | B1  |   1 | c00001 | modehb_hybrid_C01_2049         |    45 |    19550 |   0.526 |         0.511 |
| ham10000       | C0.1_kw0.75_cov_head_standard_mt4000 | B2  |   1 | c00001 | modehb_hybrid_C01_2049         |    45 |    19550 |   0.526 |         0.511 |
| ham10000       | C0.1_kw0.75_cov_head_standard_mt4000 | B3  |   1 | c00001 | modehb_hybrid_C01_2049         |    45 |    19550 |   0.526 |         0.511 |
| ham10000       | C0.1_kw0.75_cov_head_standard_mt4000 | B4  |   1 | c00001 | modehb_hybrid_C01_2049         |    45 |    19550 |   0.526 |         0.511 |
| larvae         | C0.1_kw0.75_cov_head_standard_mt4000 | B0  |   1 | c00001 | modehb_hybrid_C01_final        |   200 |    21900 |   0.74  |         0     |
| larvae         | C0.1_kw0.75_cov_head_standard_mt4000 | B1  |   2 | c00002 | modehb_hybrid_C01_final        |   100 |   593840 |   0.921 |         0.808 |
| larvae         | C0.1_kw0.75_cov_head_standard_mt4000 | B2  |   3 | c00003 | modehb_hybrid_C01_final        |   200 |  1354728 |   0.95  |         0.78  |
| larvae         | C0.1_kw0.75_cov_head_standard_mt4000 | B3  |   4 | c00004 | modehb_hybrid_C01_final        |   200 |  1973718 |   0.967 |         0.699 |
| larvae         | C0.1_kw0.75_cov_head_standard_mt4000 | B4  |   4 | c00004 | modehb_hybrid_C01_final        |   200 |  1973718 |   0.967 |         0.699 |
| protozoan      | C0.1_kw0.75_cov_head_standard_mt4000 | B0  |   1 | c00001 | modehb_hybrid_C01_2049         |    15 |    19790 |   0.697 |               |
| protozoan      | C0.1_kw0.75_cov_head_standard_mt4000 | B1  |   1 | c00001 | modehb_hybrid_C01_2049         |    15 |    19790 |   0.697 |               |
| protozoan      | C0.1_kw0.75_cov_head_standard_mt4000 | B2  |   1 | c00001 | modehb_hybrid_C01_2049         |    15 |    19790 |   0.697 |               |
| protozoan      | C0.1_kw0.75_cov_head_standard_mt4000 | B3  |   1 | c00001 | modehb_hybrid_C01_2049         |    15 |    19790 |   0.697 |               |
| protozoan      | C0.1_kw0.75_cov_head_standard_mt4000 | B4  |   1 | c00001 | modehb_hybrid_C01_2049         |    15 |    19790 |   0.697 |               |
| remote_sensing | C0.1_kw0.75_cov_head_standard_mt4000 | B0  |  47 | c00105 | modehb_lemonade_C01_final      |    45 |    92792 |   0.583 |         0.527 |
| remote_sensing | C0.1_kw0.75_cov_head_standard_mt4000 | B1  |  69 | c00061 | modehb_lemonade_C01_final      |    15 |   390856 |   0.585 |               |
| remote_sensing | C0.1_kw0.75_cov_head_standard_mt4000 | B2  |  91 | c00039 | modehb_hybrid_C01_final        |     5 |   896038 |   0.596 |               |
| remote_sensing | C0.1_kw0.75_cov_head_standard_mt4000 | B3  | 104 | c00020 | modehb_hybrid_C01_final        |    45 |  2755294 |   0.605 |               |
| remote_sensing | C0.1_kw0.75_cov_head_standard_mt4000 | B4  | 106 | c00020 | modehb_hybrid_C01_final        |    45 |  2755294 |   0.605 |               |
| remote_sensing | C100_kw0.75_head_none_mt10000        | B0  |  17 | c00004 | modehb_v3_old                  |     5 |    52512 |   0.478 |               |
| remote_sensing | C100_kw0.75_head_none_mt10000        | B1  |  26 | c00008 | modehb_v3_old                  |     5 |   316800 |   0.487 |               |
| remote_sensing | C100_kw0.75_head_none_mt10000        | B2  |  28 | c00008 | modehb_v3_old                  |     5 |   316800 |   0.487 |               |
| remote_sensing | C100_kw0.75_head_none_mt10000        | B3  |  30 | c00008 | modehb_v3_old                  |     5 |   316800 |   0.487 |               |
| remote_sensing | C100_kw0.75_head_none_mt10000        | B4  |  33 | c00008 | modehb_v3_old                  |     5 |   316800 |   0.487 |               |
| remote_sensing | C100_kw0.75_head_none_mt4000         | B0  | 141 | c00306 | modehb_spifil                  |   100 |   196990 |   0.572 |               |
| remote_sensing | C100_kw0.75_head_none_mt4000         | B1  | 177 | c00306 | modehb_spifil                  |   100 |   196990 |   0.572 |               |
| remote_sensing | C100_kw0.75_head_none_mt4000         | B2  | 194 | c00306 | modehb_spifil                  |   100 |   196990 |   0.572 |               |
| remote_sensing | C100_kw0.75_head_none_mt4000         | B3  | 204 | c00306 | modehb_spifil                  |   100 |   196990 |   0.572 |               |
| remote_sensing | C100_kw0.75_head_none_mt4000         | B4  | 210 | c00306 | modehb_spifil                  |   100 |   196990 |   0.572 |               |

## T6c máximo e mediana(f5) de κ val por FAIXA (não cumulativo), fitness atual
| dataset        |   max_B0 |   max_B1 |   max_B2 |   max_B3 |   max_B4 |   max_>B4 |   medf5_B0 |   medf5_B1 |   medf5_B2 |   medf5_B3 |   medf5_B4 |   medf5_>B4 |
|:---------------|---------:|---------:|---------:|---------:|---------:|----------:|-----------:|-----------:|-----------:|-----------:|-----------:|------------:|
| aid            |    0.819 |    0.83  |    0.815 |    0.845 |    0.768 |           |      0.559 |      0.755 |      0.791 |      0.77  |      0.768 |             |
| coconut_trees  |    0.732 |    0.763 |    0.766 |    0.79  |          |           |            |            |            |            |            |             |
| eggs           |    0.899 |    0.937 |    0.915 |    0.932 |          |           |      0.856 |      0.885 |      0.907 |      0.904 |            |             |
| eggs_raw       |    0.527 |    0.753 |    0.878 |    0.896 |          |           |            |            |            |            |            |             |
| flowers        |    0.699 |    0.748 |    0.757 |          |          |           |      0.593 |      0.735 |      0.738 |            |            |             |
| ham10000       |    0.586 |    0.601 |    0.577 |    0.586 |          |           |      0.449 |      0.545 |      0.521 |      0.556 |            |             |
| larvae         |    0.984 |    0.952 |    0.967 |    0.95  |    0.967 |           |      0.816 |      0.917 |      0.95  |      0.933 |      0.967 |             |
| larvae_raw     |    0.785 |    0.863 |    0.863 |    0.915 |          |           |      0.655 |      0.698 |      0.825 |      0.866 |            |             |
| protozoan      |    0.773 |    0.897 |    0.89  |    0.836 |    0.835 |           |      0.667 |      0.782 |      0.851 |      0.812 |      0.835 |             |
| protozoan_raw  |    0.681 |    0.776 |    0.797 |    0.785 |          |           |      0.378 |      0.63  |      0.789 |      0.716 |            |             |
| remote_sensing |    0.591 |    0.577 |    0.591 |    0.611 |    0.562 |     0.423 |      0.482 |      0.523 |      0.541 |      0.544 |      0.516 |       0.423 |

## T7 (contexto) ρ entre κ val (fitness) e κ teste ORIG / COV nas linhas testadas
| dataset        | fitness     |   n |   ρ_val_testORIG |   ρ_val_testCOV |   ρ_params_testORIG |
|:---------------|:------------|----:|-----------------:|----------------:|--------------------:|
| aid            | C100_kw1    |  14 |            -0.14 |            0.93 |               -0.53 |
| coconut_trees  | C100_kw1    |   9 |            -0.03 |            0.9  |               -0.34 |
| eggs           | C100_kw1    |   9 |             0.27 |            0.97 |                0.02 |
| eggs_raw       | C100_kw1    |   6 |             0.54 |            0.83 |                0.5  |
| flowers        | C0.1_kw0.75 |   7 |            -0.39 |            0.75 |               -0.97 |
| flowers        | C100_kw1    |   6 |             0.89 |            1    |                0.46 |
| ham10000       | C100_kw1    |  23 |             0.14 |                 |               -0.43 |
| larvae         | C0.1_kw0.75 |   7 |            -0.19 |            0.09 |               -0.21 |
| larvae         | C100_kw1    |  16 |             0.61 |            0.69 |               -0.4  |
| larvae_raw     | C100_kw1    |   8 |            -0.42 |            0.69 |               -0.55 |
| protozoan      | C100_kw1    |   7 |            -0.35 |            0.96 |               -0.55 |
| remote_sensing | C0.1_kw0.75 |  85 |             0.47 |            0.68 |                0.24 |
| remote_sensing | C100_kw1    |  69 |             0.07 |            0.65 |                0.1  |

