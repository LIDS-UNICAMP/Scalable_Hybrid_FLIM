# Backbone x cabeca (sem treino) -- SVM original C=100, sem scaler

Fonte: `research/spifil_b0b4/readout_x/results_jaci_cpu.jsonl` (7 linhas). Cada celula: kappa_val / **kappa_test** (F1_test). Teste = so relatorio (never_feeds_fitness).

| backbone | dataset | GAP 1280 | 2a ordem 1280 | 2a ordem 4096 | cross-block 1280 | cross-block 4096 | controle rnd 1280 / 4096 (k_test) |
|---|---|---|---|---|---|---|---|
| RS_B0_seed01_c00306 | remote_sensing | 0.571 / **0.534** (0.539) [2560d] | 0.501 / **0.599** (0.577) | 0.543 / **0.605** (0.591) | 0.453 / **0.500** (0.475) | 0.443 / **0.507** (0.492) | 0.570/0.563 |

Dimensoes: 2a ordem 1280 = k48 (48 mu + 48 sigma + 1176 triu = 1272 + 8 zeros); 4096 = k88 (88+88+3916 = 4092 + 4 zeros). cross-block: kp=floor(sqrt(D/n_pares)) por par, zeros ate D. GAP do growth/modehb = embedding da propria cabeca (dim marcada se != 1280).

† validation == train no descritor do split (perc100 da discovery): o kappa_val dessa linha e medido no TREINO; so o kappa_test e informativo.

Pendentes (rodar no athena com `--preset table`, ver README.md): RS_B1_fix5_B1g0c00_f45, RS_B2_fix4_B2g1c02_f15, RS_B2_fix4_B2g1c02_f45, RS_B1_fix5_nossl, FL_B0_seed01, HYB_oct_eggs_s1_pct100, HYB_oct_eggs_s2_pct100, HYB_oct_eggs_s3_pct100, HYB_oct_larvae_s1_pct100, HYB_oct_larvae_s2_pct100, HYB_oct_larvae_s3_pct100, HYB_oct_protozoan_s1_pct100, HYB_oct_protozoan_s2_pct100, HYB_oct_protozoan_s3_pct100.
