# Backbone x cabeca (sem treino) -- SVM original C=100, sem scaler

Fonte: `/dados/home/moliveira/Scalable_Hybrid_FLIM/research/spifil_b0b4/readout_x/results/growth_v3_jaci_test.jsonl` (111 linhas). Cada celula: kappa_val / **kappa_test** (F1_test). Teste = so relatorio (never_feeds_fitness).

| backbone | dataset | GAP 1280 | 2a ordem 1280 | 2a ordem 4096 | cross-block 1280 | cross-block 4096 | controle rnd 1280 / 4096 (k_test) |
|---|---|---|---|---|---|---|---|
| jaci_v3ae_aid_B1g0c00_f15 | aid | 0.717 / **0.719** (0.728) [2560d] | — | 0.802 / **0.811** (0.817) | — | — | —/— |
| jaci_v3ae_aid_B1g0c01_f15 | aid | 0.728 / **0.726** (0.734) [2560d] | — | 0.841 / **0.841** (0.846) | — | — | —/— |
| jaci_v3ae_aid_B1g0c02_f15 | aid | 0.734 / **0.741** (0.749) [2560d] | — | 0.817 / **0.815** (0.821) | — | — | —/— |
| jaci_v3ae_aid_B1g0c03_f15 | aid | 0.708 / **0.717** (0.726) [2560d] | — | 0.830 / **0.843** (0.849) | — | — | —/— |
| jaci_v3ae_aid_B1g0c04_f15 | aid | 0.710 / **0.719** (0.727) [2560d] | — | 0.843 / **0.834** (0.839) | — | — | —/— |
| jaci_v3ae_aid_B1g0c04_f45 | aid | 0.697 / **0.724** (0.733) [2560d] | — | 0.843 / **0.829** (0.835) | — | — | —/— |
| jaci_v3ae_aid_B1g0c05_f15 | aid | 0.708 / **0.726** (0.735) [2560d] | — | 0.855 / **0.841** (0.847) | — | — | —/— |
| jaci_v3ae_aid_B1g0c06_f15 | aid | 0.728 / **0.739** (0.748) [2560d] | — | 0.826 / **0.829** (0.833) | — | — | —/— |
| jaci_v3ae_aid_B1g0c06_f45 | aid | 0.737 / **0.736** (0.746) [2560d] | — | 0.830 / **0.821** (0.826) | — | — | —/— |
| jaci_v3ae_aid_B1g0c07_f15 | aid | 0.708 / **0.719** (0.729) [2560d] | — | 0.838 / **0.834** (0.839) | — | — | —/— |
| jaci_v3ae_aid_B1g0c07_f45 | aid | 0.703 / **0.728** (0.736) [2560d] | — | 0.844 / **0.845** (0.850) | — | — | —/— |
| jaci_v3ae_aid_B1g1c00_f15 | aid | 0.717 / **0.719** (0.728) [2560d] | — | 0.802 / **0.811** (0.817) | — | — | —/— |
| jaci_v3ae_aid_B1g1c01_f15 | aid | 0.729 / **0.726** (0.734) [2560d] | — | 0.840 / **0.836** (0.840) | — | — | —/— |
| jaci_v3ae_aid_B1g1c02_f15 | aid | 0.717 / **0.720** (0.729) [2560d] | — | 0.828 / **0.834** (0.840) | — | — | —/— |
| jaci_v3ae_aid_B1g1c03_f15 | aid | 0.717 / **0.726** (0.734) [2560d] | — | 0.839 / **0.837** (0.841) | — | — | —/— |
| jaci_v3ae_aid_B1g1c04_f15 | aid | 0.701 / **0.722** (0.730) [2560d] | — | 0.848 / **0.835** (0.841) | — | — | —/— |
| jaci_v3ae_coconut_trees_B1g0c00_f15 | coconut_trees | 0.749 / **0.738** (0.869) | — | 0.775 / **0.771** (0.885) | — | — | —/— |
| jaci_v3ae_coconut_trees_B1g0c01_f15 | coconut_trees | 0.732 / **0.752** (0.876) | — | 0.776 / **0.788** (0.894) | — | — | —/— |
| jaci_v3ae_coconut_trees_B1g0c02_f15 | coconut_trees | 0.759 / **0.730** (0.865) | — | 0.769 / **0.769** (0.885) | — | — | —/— |
| jaci_v3ae_flowers_B1g0c00_f15 | flowers | 0.632 / **0.647** (0.716) | — | 0.745 / **0.784** (0.828) | — | — | —/— |
| jaci_v3ae_ham10000_B1g0c00_f15 | ham10000 | 0.494 / **0.511** (0.513) [2560d] | — | 0.580 / **0.599** (0.598) | — | — | —/— |
| jaci_v3ae_ham10000_B1g0c01_f15 | ham10000 | 0.472 / **0.512** (0.514) [2560d] | — | 0.570 / **0.618** (0.623) | — | — | —/— |
| jaci_v3ae_ham10000_B1g0c02_f15 | ham10000 | 0.470 / **0.503** (0.533) [2560d] | — | 0.565 / **0.587** (0.591) | — | — | —/— |
| jaci_v3ae_ham10000_B1g0c03_f15 | ham10000 | 0.499 / **0.514** (0.515) [2560d] | — | 0.574 / **0.633** (0.651) | — | — | —/— |
| jaci_v3ae_ham10000_B1g0c04_f15 | ham10000 | 0.463 / **0.512** (0.529) [2560d] | — | 0.562 / **0.613** (0.630) | — | — | —/— |
| jaci_v3ae_ham10000_B1g0c05_f15 | ham10000 | 0.499 / **0.515** (0.515) [2560d] | — | 0.563 / **0.622** (0.659) | — | — | —/— |
| jaci_v3ae_ham10000_B1g0c06_f15 | ham10000 | 0.498 / **0.524** (0.559) [2560d] | — | 0.556 / **0.611** (0.613) | — | — | —/— |
| jaci_v3ae_ham10000_B1g0c07_f15 | ham10000 | 0.490 / **0.501** (0.526) [2560d] | — | 0.567 / **0.629** (0.652) | — | — | —/— |

Dimensoes: 2a ordem 1280 = k48 (48 mu + 48 sigma + 1176 triu = 1272 + 8 zeros); 4096 = k88 (88+88+3916 = 4092 + 4 zeros). cross-block: kp=floor(sqrt(D/n_pares)) por par, zeros ate D. GAP do growth/modehb = embedding da propria cabeca (dim marcada se != 1280).

† validation == train no descritor do split (perc100 da discovery): o kappa_val dessa linha e medido no TREINO; so o kappa_test e informativo.

Pendentes (rodar no athena com `--preset table`, ver README.md): RS_B0_seed01_c00306, RS_B1_fix5_B1g0c00_f45, RS_B2_fix4_B2g1c02_f15, RS_B2_fix4_B2g1c02_f45, RS_B1_fix5_nossl, FL_B0_seed01, HYB_oct_eggs_s1_pct100, HYB_oct_eggs_s2_pct100, HYB_oct_eggs_s3_pct100, HYB_oct_larvae_s1_pct100, HYB_oct_larvae_s2_pct100, HYB_oct_larvae_s3_pct100, HYB_oct_protozoan_s1_pct100, HYB_oct_protozoan_s2_pct100, HYB_oct_protozoan_s3_pct100.
