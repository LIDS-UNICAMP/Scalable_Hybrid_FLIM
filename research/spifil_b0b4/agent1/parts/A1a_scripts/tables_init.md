## TB1 sementes DISTINTAS por imagem após reescala para a grade da camada (100 superpixels SLIC/imagem; medido)
| dataset        |   distinct_224 |   distinct_112 |   distinct_56 |   distinct_28 |   distinct_14 |   distinct_7 |   distinct_4 |   mask_coverage |
|:---------------|---------------:|---------------:|--------------:|--------------:|--------------:|-------------:|-------------:|----------------:|
| aid            |          100   |          100   |          99.8 |          99   |          90.6 |         47.9 |         16   |             1   |
| eggs           |          100   |           99.7 |          99.1 |          94.7 |          70.7 |         27   |         11.1 |             0.4 |
| flowers        |          100   |           99.9 |          99.9 |          98.6 |          90.6 |         47.9 |         16   |             1   |
| ham10000       |          100   |          100   |         100   |          99.5 |          93.2 |         48.2 |         16   |             1   |
| larvae         |           99.7 |           98.3 |          88.3 |          54.7 |          25.8 |         10.3 |          7   |             0.1 |
| larvae_raw     |          100   |           99.3 |          95.5 |          67   |          30   |         14   |          7.5 |             0.1 |
| protozoan      |          100   |           99.5 |          94.5 |          61.6 |          24.5 |          9.8 |          4.4 |             0.1 |
| protozoan_raw  |           99.9 |           99.7 |          98.2 |          85   |          37.8 |         12.3 |          5.3 |             0.2 |
| remote_sensing |          100   |           99.9 |          99.7 |          98.7 |          90.9 |         48   |         16   |             1   |

## TB2 orçamento de covariância por camada: N nominal (com duplicatas) × N efetivo (posições distintas) e D = C_in·k² (fitness atual, camadas SPiFiL)
| grupo                 |   in_res |   n |   N_nominal |    N_eff |      D |   ND_nominal |   ND_eff |   frac_ND_eff_lt5 |   frac_ND_eff_lt1 |
|:----------------------|---------:|----:|------------:|---------:|-------:|-------------:|---------:|------------------:|------------------:|
| natural (sem máscara) |      224 | 243 |       12000 | 12000    |   75   |       297.33 |   297.33 |              0    |              0    |
| natural (sem máscara) |      112 | 261 |       16000 | 15978.7  |   49   |       176.98 |   176.73 |              0.11 |              0    |
| natural (sem máscara) |       56 | 211 |       22300 | 22225.7  |  100   |       114.29 |   113.9  |              0.14 |              0    |
| natural (sem máscara) |       28 | 153 |       22300 | 22002.7  |  168   |       100    |    99.52 |              0.12 |              0    |
| natural (sem máscara) |       14 |  91 |       22300 | 20261.1  |  252   |        80    |    72.48 |              0.24 |              0    |
| natural (sem máscara) |        7 |  72 |       22300 | 10709.3  |  123   |        94.25 |    45.18 |              0.35 |              0.04 |
| parasitas (máscara)   |      224 | 132 |        6362 |  6378.67 |   75   |       118.52 |   118.52 |              0    |              0    |
| parasitas (máscara)   |      112 | 137 |        6362 |  6293.33 |   25   |       254.48 |   251.73 |              0.04 |              0    |
| parasitas (máscara)   |       56 | 118 |        6384 |  6112    |   49.5 |       129.84 |   115.37 |              0.19 |              0    |
| parasitas (máscara)   |       28 |  93 |        6384 |  4288    |  200   |        61.17 |    33.64 |              0.35 |              0.03 |
| parasitas (máscara)   |       14 |  58 |        6373 |  1920    |  248.5 |        34.78 |     9.04 |              0.45 |              0.21 |
| parasitas (máscara)   |        7 |  14 |        6362 |   661.33 | 1296   |         3.18 |     0.5  |              0.64 |              0.57 |

## TB3 largura efetiva/requisitada pelo alocador por classe e rota exact_small (fitness atual)
| dataset        |   n_camadas |   ipc_mediano |   n_imgs_mediano |   shrink_mediano |   shrink_p10 |   frac_shrink_lt_09 |   frac_exact_small |
|:---------------|------------:|--------------:|-----------------:|-----------------:|-------------:|--------------------:|-------------------:|
| aid            |         186 |             4 |              120 |            0.996 |        0.727 |               0.312 |              0.468 |
| eggs           |          82 |            16 |              144 |            0.973 |        0.797 |               0.183 |              0.207 |
| flowers        |           9 |            32 |              160 |            0.99  |        0.868 |               0.333 |              0.111 |
| ham10000       |         268 |            16 |              112 |            0.971 |        0.745 |               0.328 |              0.157 |
| larvae         |         197 |            32 |               64 |            1     |        0.901 |               0.102 |              0.066 |
| larvae_raw     |         108 |            32 |               64 |            1     |        0.879 |               0.139 |              0.093 |
| protozoan      |         121 |            16 |              112 |            0.98  |        0.778 |               0.256 |              0.231 |
| protozoan_raw  |          44 |            32 |              224 |            0.979 |        0.713 |               0.364 |              0.182 |
| remote_sensing |         568 |            16 |              223 |            0.97  |        0.808 |               0.246 |              0.215 |

## TB4 estatísticas do banco de filtros por bloco (checkpoint final; 38 checkpoints = melhor κ val por dataset×estrato + melhor freeze; convs densas)
|   block |   C |   D |   erank_ratio |   rank99_ratio |   mean_abs_cos |   frac_cos_gt09 |   frac_with_dup |   frac_with_dup09 |   frac_near_null |
|--------:|----:|----:|--------------:|---------------:|---------------:|----------------:|----------------:|------------------:|-----------------:|
|       0 |   7 |  27 |         0.639 |          0.6   |          0.516 |           0.104 |           0     |             0.286 |                0 |
|       1 |  24 |  42 |         0.529 |          0.5   |          0.486 |           0.077 |           0.4   |             0.525 |                0 |
|       2 |  37 |  90 |         0.43  |          0.402 |          0.436 |           0.043 |           0.39  |             0.571 |                0 |
|       3 | 204 | 261 |         0.344 |          0.289 |          0.375 |           0.02  |           0.321 |             0.488 |                0 |
|       4 | 112 | 642 |         0.335 |          0.3   |          0.368 |           0.022 |           0.4   |             0.5   |                0 |
|       5 | 112 | 642 |         0.465 |          0.438 |          0.301 |           0.016 |           0.321 |             0.455 |                0 |
|       6 | 642 | 642 |         0.297 |          0.252 |          0.343 |           0.004 |           0.003 |             0.447 |                0 |
n convs = 338

## TB4b idem, só âncoras de candidatos protect=freeze em camadas de origem spifil (= filtros SPiFiL puros, sem AE/SSL nos pesos conv)
|   block |   C |    D |   erank_ratio |   rank99_ratio |   mean_abs_cos |   frac_cos_gt09 |   frac_with_dup |   frac_with_dup09 |   frac_near_null |
|--------:|----:|-----:|--------------:|---------------:|---------------:|----------------:|----------------:|------------------:|-----------------:|
|       0 |   8 |   27 |         0.609 |          0.5   |          0.488 |           0.143 |           0.25  |             0.5   |                0 |
|       1 | 210 |   34 |         0.472 |          0.5   |          0.501 |           0.069 |           0.679 |             0.786 |                0 |
|       2 |  16 |   81 |         0.525 |          0.556 |          0.497 |           0.055 |           0.25  |             0.562 |                0 |
|       3 | 210 |  576 |         0.306 |          0.25  |          0.424 |           0.036 |           0.392 |             0.587 |                0 |
|       4 | 140 | 1377 |         0.335 |          0.307 |          0.265 |           0.022 |           0.393 |             0.488 |                0 |
n convs = 33

## TB4c por kernel (final)
|   k |   C |    D |   erank_ratio |   rank99_ratio |   mean_abs_cos |   frac_cos_gt09 |   frac_with_dup |   frac_with_dup09 |   frac_near_null |
|----:|----:|-----:|--------------:|---------------:|---------------:|----------------:|----------------:|------------------:|-----------------:|
|   1 |  49 |   33 |         0.411 |          0.386 |          0.492 |           0.072 |           0.5   |             0.714 |                0 |
|   3 |  30 |  261 |         0.477 |          0.457 |          0.417 |           0.048 |           0.249 |             0.411 |                0 |
|   5 |  16 |  350 |         0.605 |          0.607 |          0.36  |           0.034 |           0.143 |             0.286 |                0 |
|   7 |  42 | 1372 |         0.635 |          0.622 |          0.352 |           0.012 |           0.025 |             0.142 |                0 |

## TB5 deriva dos pesos conv: âncora (SPiFiL+AE) → SSL e → final (WiSE), ‖ΔW‖/‖W‖ mediana
| ('protect', '')   |   ('drift_rel_norm_anchor_ssl', 'median') |   ('drift_rel_norm_anchor_ssl', 'max') |   ('drift_rel_norm', 'median') |   ('drift_rel_norm', 'max') |   ('drift_cos_anchor_final', 'median') |   ('drift_cos_anchor_final', 'max') |
|:------------------|------------------------------------------:|---------------------------------------:|-------------------------------:|----------------------------:|---------------------------------------:|------------------------------------:|
| freeze            |                                    0      |                                 0      |                         0      |                      0      |                                 1      |                                   1 |
| full              |                                    0.0362 |                                 1.6721 |                         0.0158 |                      1.6721 |                                 0.9999 |                                   1 |
| lowlr             |                                    0.0003 |                                 0.1412 |                         0.0001 |                      0.1095 |                                 1      |                                   1 |

## TB6 ativações (16 imagens doadoras de treino, encoder final): canais mortos, rank efetivo e |corr| entre canais por bloco
|   block |     C |   H |   frac_dead |   act_erank_ratio |   mean_abs_corr |
|--------:|------:|----:|------------:|------------------:|----------------:|
|       0 |  17.5 | 112 |           0 |             0.491 |           0.415 |
|       1 |  48.5 |  56 |           0 |             0.227 |           0.502 |
|       2 | 106   |  28 |           0 |             0.178 |           0.469 |
|       3 | 420   |  14 |           0 |             0.118 |           0.473 |
|       4 | 477   |   7 |           0 |             0.157 |           0.259 |
|       5 | 448   |   7 |           0 |             0.185 |           0.291 |
|       6 | 642   |   7 |           0 |             0.111 |           0.272 |

| dataset        | stratum   | cand   |   block |    C |   H |   frac_dead |
|:---------------|:----------|:-------|--------:|-----:|----:|------------:|
| larvae_raw     | huge      | c00004 |       3 |  818 |  14 |       0.076 |
| larvae_raw     | huge      | c00004 |       4 | 1024 |   7 |       0.322 |
| protozoan_raw  | huge      | c00004 |       3 |  812 |  14 |       0.001 |
| protozoan_raw  | huge      | c00004 |       4 | 1022 |   7 |       0.22  |
| flowers        | small     | c00002 |       0 |   15 | 112 |       0.133 |
| remote_sensing | small     | c00054 |       5 |  168 |   7 |       0.226 |

