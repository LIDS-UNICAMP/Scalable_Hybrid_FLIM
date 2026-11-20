# fontes_filtros: resultados no teste e limpeza (14/11)

**Resposta curta:** quase nada dessas pastas tem resultado no **teste**. Elas guardam as "fontes" dos filtros (os pedacinhos de imagem de onde o SPiFiL escolhe os filtros) dos experimentos `disc_md_*`, e esses experimentos foram medidos só na **validação**. Pela regra (só apaga o que já tem teste), quase tudo fica.

- **athena** `/exps/moliveira/fontes_filtros`: 192 GB, 53 pastas. Só 3 experimentos rodaram no teste: `teste_aid_receita_tta`, `teste_aid_restr5m_45M` e `receita_tta_ham100`. Só o último terminou.
- Nenhuma pasta está em uso: nenhum processo com arquivo aberto, nenhum symlink, nenhum YAML da fila (`g*q_*`, `rand_g*`, `g1x1024_ae` e seus `eval_svm_*`) aponta para lá, e nenhum tmux usa o caminho.
- Leitura usada: **L6 `cat_tta`**, a leitura oficial dos reports `disc_md` (`ssl_exploracao/ponto_de_partida.md`, `multi_dataset/ham10000_curva_receita.md`). Com `amostra.avaliar: test`, a coluna `val_kappa` do CSV guarda o κ de **teste**.
- Não rodei SVM de teste para as pastas sem teste. Cada ponto `disc_md` leva cerca de 1 h (seleção + extração de 6 camadas × 2 braços), muito acima dos 15 min.

## athena — 1. `teste_aid_receita_tta` (receita 8,2 M, 6×256, k5, fps, 200 px)

YAML: `experiments/autoencoder/disc_md_teste_aid_receita_tta_athena.yaml`. CSVs: `/exps/moliveira/results/svm_disc_md_teste_aid_receita_tta_aid_s0200_split*_pct*.csv`. Dim 1536 = cat das 6 camadas de 256; L6 = 256×12×12 (200 px ÷ 16).

| base | split | pct | κ test SPiFiL (fps) | F1 test | κ test random_z (mesma forma) | dim |
|---|---|---|---|---|---|---|
| aid | 1 | 50 | 0,826 | 0,834 | 0,770 | 1536 |
| aid | 2 | 50 | 0,818 | 0,825 | 0,781 | 1536 |
| aid | 3 | 50 | **sem teste** (fps morreu na L2, OOM) | — | 0,752 | 1536 |
| aid | 1 | 100 | 0,859 | 0,864 | 0,808 | 1536 |
| aid | 2 | 100 | 0,854 | 0,859 | 0,809 | 1536 |
| aid | 3 | 100 | **sem teste** (CSV vazio) | — | — | — |

## athena — 2. `teste_aid_restr5m_45M` (L3 ResNet 4,5 M, 56-112-216-432-432-432, k3, fps, 200 px)

YAML: `experiments/autoencoder/disc_md_teste_aid_restr5m_45M_athena.yaml`. CSVs: `/exps/moliveira/results/svm_disc_md_teste_aid_restr5m_45M_aid_s0200_split*_pct*_leidepth6_leiW_w56-112-216-432-432-432.csv`. Dim 1680 = 56+112+216+432+432+432; L6 = 432×12×12.

| base | split | pct | κ test SPiFiL (fps) | F1 test | κ test random_z (mesma forma) | dim |
|---|---|---|---|---|---|---|
| aid | 1 | 50 | 0,808 | 0,817 | 0,733 | 1680 |
| aid | 2 | 50 | **sem teste** (OOM, parou na L4) | — | — | — |
| aid | 3 | 50 | 0,817 | 0,825 | 0,746 | 1680 |
| aid | 1 | 100 | 0,863 | 0,868 | 0,785 | 1680 |
| aid | 2 | 100 | 0,861 | 0,867 | 0,813 | 1680 |
| aid | 3 | 100 | **sem teste** (CSV vazio) | — | — | — |

**Modelos grandes no aid** (`results/benchmark/*.csv`, κ test):

| pct | split | DINOv2-g | AIMv2-3B | I-JEPA-g | ResNet-50 | ConvNeXtV2-pico |
|---|---|---|---|---|---|---|
| 50 | 1 | 0,955 | 0,975 | 0,892 | 0,884 | 0,851 |
| 50 | 2 | 0,963 | 0,976 | 0,897 | 0,888 | 0,855 |
| 50 | 3 | 0,970 | 0,977 | 0,899 | 0,901 | 0,863 |
| 100 | 1 | 0,970 | 0,979 | 0,916 | 0,898 | 0,888 |
| 100 | 2 | 0,974 | 0,981 | 0,923 | 0,913 | 0,890 |
| 100 | 3 | 0,975 | 0,982 | 0,925 | 0,910 | 0,888 |

**Aleatório puro no aid** (`results/svm_rand_g2_test_gemtta.csv`, dim 2048): 50% → 0,699 / 0,722 / 0,698 (splits 1/2/3). A 100%: não rodou. `rand_g1`, `rand_g3` e `rand_g4` não têm aid.

## athena — 3. `receita_tta_ham100` (receita 6×256, k5, ham10000 a 100%, 200 px)

YAML: `experiments/autoencoder/disc_md_receita_tta_ham100.yaml`. CSVs: `/exps/moliveira/results/svm_disc_md_receita_tta_ham100_test_ham10000_s0200_split*_pct100_sh*.csv`. Dim 1536; L6 = 256×12×12. Completo: 3 splits × 4 braços.

| base | split | pct | κ test fps | F1 fps | κ test fpsw (saliência) | κ test orth | κ test random_z | dim |
|---|---|---|---|---|---|---|---|---|
| ham10000 | 1 | 100 | 0,549 | 0,594 | 0,541 | 0,458 | 0,448 | 1536 |
| ham10000 | 2 | 100 | 0,554 | 0,595 | 0,527 | 0,466 | 0,461 | 1536 |
| ham10000 | 3 | 100 | 0,532 | 0,584 | 0,516 | 0,468 | 0,462 | 1536 |

**Modelos grandes no ham10000 a 100%** (κ test): ResNet-50 0,493 / 0,500 / 0,491 (splits 1/2/3, dim 2048). DINOv2, AIMv2, I-JEPA e ConvNeXt-pico: não rodaram a 100%.
**Aleatório puro no ham10000:** não rodou (`svm_rand_g*_test_gemtta.csv` não tem ham10000). O random_z da mesma forma está na tabela.

## athena — as outras 50 pastas: sem resultado de teste

Todas são `disc_md_*` (descoberta) medidos só na validação. Ficaram todas.

## athena — inventário completo

| pasta | tamanho | última mudança | YAML que usa (`experiments/autoencoder/`) | em uso? | tem teste? | destino |
|---|---|---|---|---|---|---|
| `atencao_combo` | 2.5G | 2026-11-10 | disc_md_atencao_combo | não |não | sem resultado de teste: fica |
| `atencao_combo320` | 3.2G | 2026-11-10 | disc_md_relglob_320tta disc_md_atencao_combo320 | não |não | sem resultado de teste: fica |
| `atencao_conf` | 2.5G | 2026-11-10 | disc_md_atencao_conf | não |não | sem resultado de teste: fica |
| `atencao_sal320` | 5.1G | 2026-11-11 | disc_md_atencao_sal320 | não |não | sem resultado de teste: fica |
| `atencao_triagem` | 818M | 2026-11-10 | disc_md_atencao_triagem | não |não | sem resultado de teste: fica |
| `cl` | 38G | 2026-11-08 | disc_md_cl_ctl_ext disc_md_cl_a_ext disc_md_cl_a disc_md_cl_ctl | não |não | sem resultado de teste: fica |
| `dino_pontos` | 11M | 2026-11-10 | (script analise_textura/dino_pontos.py) | não |não | sem teste + lida por figura de report: fica |
| `disc_md_e07_off` | 552M | 2026-11-09 | disc_md_e07_off | não |não | sem resultado de teste: fica |
| `disc_md_vlad_off` | 931M | 2026-11-08 | disc_md_vlad_off | não |não | sem resultado de teste: fica |
| `disc_md_vlad_on` | 1.1G | 2026-11-08 | disc_md_vlad_on | não |não | sem resultado de teste: fica |
| `encdeep_nw_off` | 2.8G | 2026-11-09 | disc_md_encdeep_nw_off | não |não | sem resultado de teste: fica |
| `encdeep_off` | 466M | 2026-11-08 | disc_md_encdeep_geo_off disc_md_encdeep_mech_off | não |não | sem resultado de teste: fica |
| `encdeep_on` | 559M | 2026-11-08 | disc_md_encdeep_mech_on disc_md_encdeep_geo_on | não |não | sem resultado de teste: fica |
| `gelu` | 2.5G | 2026-11-09 | disc_md_gelu | não |não | sem resultado de teste: fica |
| `leis_p1` | 764M | 2026-11-08 | disc_md_leis_p1 | não |não | sem resultado de teste: fica |
| `leis_p1b` | 799M | 2026-11-08 | disc_md_leis_p1b | não |não | sem resultado de teste: fica |
| `leis_p1b_s43_athena` | 910M | 2026-11-11 | disc_md_leis_p1b_s43_athena | não |não | sem resultado de teste: fica |
| `leis_p1c_s43` | 797M | 2026-11-08 | disc_md_leis_p1c_s43 | não |não | sem resultado de teste: fica |
| `leis_p1_s43` | 764M | 2026-11-08 | disc_md_leis_p1_s43 | não |não | sem resultado de teste: fica |
| `leis_p1_s43_athena` | 280M | 2026-11-11 | disc_md_leis_p1_s43_athena | não |não | sem resultado de teste: fica |
| `leitura_a2` | 8.3G | 2026-11-08 | disc_md_leitura_a2 | não |não | sem resultado de teste: fica |
| `ms` | 46G | 2026-11-07 | disc_md_ms_b disc_md_ms_ctl disc_md_ms_ctl_ext disc_md_ms_a_ext disc_md_sol_hebb2_piloto disc_md_ms_ctl2 disc_md_ms_b_ext disc_md_sol_hebb3 disc_md_sol_hebb3_conf disc_md_sol_hebb1 disc_md_sol_hebb2 disc_md_ms_a disc_md_sol_hebb1_piloto | não |não | sem resultado de teste: fica |
| `pequeno320_aid` | 2.0G | 2026-11-12 | disc_md_pequeno320_aid_athena | não |não | sem resultado de teste: fica |
| `pequeno320_coconut_trees` | 2.0G | 2026-11-11 | disc_md_pequeno320_coconut_trees_athena | não |não | sem resultado de teste: fica |
| `pequeno320_flowers` | 2.0G | 2026-11-11 | disc_md_pequeno320_flowers_athena | não |não | sem resultado de teste: fica |
| `pequeno320_remote_sensing` | 1.4G | 2026-11-11 | disc_md_pequeno320_remote_sensing_athena | não |não | sem resultado de teste: fica |
| `pf_hebb_d3_off` | 931M | 2026-11-08 | disc_md_pf_hebb_d3_off | não |não | sem resultado de teste: fica |
| `pf_hebb_d3_on` | 1.1G | 2026-11-08 | disc_md_pf_hebb_d3_on | não |não | sem resultado de teste: fica |
| `pf_hebb_d4_off` | 931M | 2026-11-08 | disc_md_pf_hebb_d4_off | não |não | sem resultado de teste: fica |
| `pf_hebb_d4_on` | 1.1G | 2026-11-08 | disc_md_pf_hebb_d4_on | não |não | sem resultado de teste: fica |
| `pf_leis_off_athena` | 931M | 2026-11-12 | disc_md_pf_leis_off_athena | não |não | sem resultado de teste: fica |
| `pf_leis_on_athena` | 1.1G | 2026-11-12 | disc_md_pf_leis_on_athena | não |não | sem resultado de teste: fica |
| `pf_res128_on` | 292M | 2026-11-08 | disc_md_pf_res128_on | não |não | sem resultado de teste: fica |
| `pf_res224_off_b` | 1.2G | 2026-11-08 | disc_md_pf_res224_off_b | não |não | sem resultado de teste: fica |
| `pf_res256_on_b` | 1.7G | 2026-11-09 | disc_md_pf_res256_on_b | não |não | sem resultado de teste: fica |
| `pf_res320_off` | 2.1G | 2026-11-08 | disc_md_pf_res320_off | não |não | sem resultado de teste: fica |
| `pf_res448_off` | 1.9G | 2026-11-09 | disc_md_pf_res448_off | não |não | sem resultado de teste: fica |
| `pf_res448_off_athena` | 8.0G | 2026-11-11 | disc_md_pf_res448_off_athena | não |não | sem resultado de teste: fica |
| `plano_helminth` | 2.6G | 2026-11-09 | disc_md_plano_helminth | não |não | sem resultado de teste: fica |
| `receita_tta` | 3.6G | 2026-11-13 | disc_md_dino_pontos disc_md_receita_tta disc_md_relglob_somar disc_md_relglob disc_md_receita_tta_fundopreto disc_md_receita_tta_ham | não |não | sem resultado de teste: fica |
| `receita_tta_ham100` | 828M | 2026-11-11 | disc_md_receita_tta_ham100 | não |sim (completo) | **apagada** |
| `receita_tta_ham_fix` | 1.9G | 2026-11-11 | disc_md_receita_tta_ham_fix | não |não | sem resultado de teste: fica |
| `restr5m_dwr_s184_athena` | 7.8G | 2026-11-11 | disc_md_restr5m_dwr_s184_athena | não |não | sem resultado de teste: fica |
| `restr5m_effnet_s128_athena` | 4.8G | 2026-11-11 | disc_md_restr5m_effnet_s128_athena | não |não | sem resultado de teste: fica |
| `restr5m_k3_cena` | 5.2G | 2026-11-11 | hyb_ssl_encoder_remote_sensing_athena hyb_ssl_encoder3m_aid_athena hyb_ssl_encoder_aid_athena hyb_ssl_encoder_coconut_trees_athena | não |não | sem resultado de teste: fica |
| `restr5m_k3_obj` | 1.8G | 2026-11-11 | hyb_ssl_encoder_flowers_athena | não |não | sem resultado de teste: fica |
| `restr5m_k5_obj_athena` | 3.5G | 2026-11-11 | disc_md_restr5m_k5_obj_athena | não |não | sem resultado de teste: fica |
| `salw` | 1.8G | 2026-11-08 | disc_md_salw_a_ext | não |não | sem resultado de teste: fica |
| `sol` | 9.4G | 2026-11-08 | disc_md_sol_p3_ext disc_md_sol_p1_n6_ext disc_md_sol_p5_ext disc_md_sol_p1_n6 disc_md_sol_p5 disc_md_sol_p1 disc_md_sol_p2 disc_md_sol_p1_ext disc_md_sol_p3 disc_md_ext_piloto disc_md_sol_p2_ext | não |não | sem resultado de teste: fica |
| `teste_aid_receita_tta` | 1.5G | 2026-11-12 | disc_md_teste_aid_receita_tta_athena | não |parcial | teste incompleto: fica |
| `teste_aid_restr5m_45M` | 1.5G | 2026-11-13 | disc_md_teste_aid_restr5m_45M_athena | não |parcial | teste incompleto: fica |
| `wsal` | 308M | 2026-11-08 | disc_md_sol_wsal_ext disc_md_sol_wsal | não |não | sem resultado de teste: fica |
| `wsal_piloto` | 47M | 2026-11-08 | disc_md_sol_wsal_piloto | não |não | sem resultado de teste: fica |

## jaci — `/dados/home/moliveira/exps/fontes_filtros` (91 GB, 33 pastas)

Levantamento só de leitura em 14/11. O jaci está parado: o K3 de 13/11 às 21:40 derrubou todos os runs, o `monitor_gpu` e o `ssl_jaci`. Não há nenhum processo do moliveira com arquivo aberto nem com `fontes_filtros` na linha de comando. Nenhum symlink aponta para lá. A única sessão tmux (`claude_explore`) não menciona o caminho. Das filas, `g2fila_jaci.txt`, `g2svm_jaci.txt`, `fila4.txt` e `prioridade.txt` estão vazias. A única que ainda cita YAML de fontes é `fila4_espera.txt` (parada desde 10/11 22:37).

Os YAMLs existem nos dois hosts (`Scalable_Hybrid_FLIM_discovery/experiments/autoencoder/`), com o mesmo nome. A coluna YAML mostra só o `_jaci` e quem compartilha a pasta.

### (i) Tabela

| pasta | tamanho | última modificação | bases (.pt) | YAML | em uso (motivo) | tem teste? |
|---|---|---|---|---|---|---|
| disc_md_e07_on | 1.1G | 2026-11-10 17:21 | coconut, flowers | disc_md_e07_on(_jaci) | NÃO | NÃO (4 CSV val) |
| encdeep_mech320_off | 1.1G | 2026-11-08 20:39 | aid, remote | disc_md_encdeep_mech320_off_jaci | NÃO | NÃO (8 val) |
| encdeep_nw_on | 3.3G | 2026-11-08 22:47 | coconut, flowers | disc_md_encdeep_nw_on(_jaci) | NÃO | NÃO (24 val) |
| encdeep_off | 466M | 2026-11-08 19:39 | aid, remote | disc_md_encdeep_{geo,mech}_off(_jaci) | NÃO | NÃO (8 val; geo sem CSV) |
| encdeep_on | 560M | 2026-11-08 19:50 | coconut, flowers | disc_md_encdeep_{geo,mech}_on(_jaci) | NÃO | NÃO (8 val; geo sem CSV) |
| gelu | 4.9G | 2026-11-10 20:59 | aid, Eggs, flowers, Larvae, Protozoan, remote | disc_md_gelu(_jaci) | NÃO | NÃO (36 val) |
| leis_p1 | 553M | 2026-11-08 12:55 | Eggs, Larvae | disc_md_leis_p1(_jaci) | NÃO | NÃO (130 val) |
| leis_p1b | 910M | 2026-11-08 12:57 | Eggs, helminth, Larvae | disc_md_leis_p1b(_jaci) | NÃO | NÃO (148 val) |
| leis_p1b_s43 | 633M | 2026-11-08 17:42 | helminth, Larvae | disc_md_leis_p1b_s43_jaci | **SIM** (citado em `queues/fila4_espera.txt`) | NÃO (12 val) |
| leis_p1_s43 | 553M | 2026-11-08 14:21 | Eggs, Larvae | disc_md_leis_p1_s43(_jaci) | **SIM** (citado em `queues/fila4_espera.txt`) | NÃO (62 val) |
| pequeno320_Eggs | 2.0G | 2026-11-11 22:34 | Eggs | disc_md_pequeno320_Eggs_jaci | NÃO | NÃO (6 val) |
| pequeno320_ham10000 | 1.9G | 2026-11-12 00:00 | ham10000 | disc_md_pequeno320_ham10000_jaci | NÃO | NÃO (6 val) |
| pequeno320_Larvae | 2.0G | 2026-11-11 19:19 | Larvae | disc_md_pequeno320_Larvae_jaci | NÃO | NÃO (6 val) |
| pequeno320_Protozoan | 1.9G | 2026-11-12 01:32 | Protozoan | disc_md_pequeno320_Protozoan_jaci | NÃO | NÃO (6 val) |
| pf_hebb_d3_off | 931M | 2026-11-08 17:59 | aid, remote | disc_md_pf_hebb_d3_off(_jaci) | NÃO | NÃO (2 val) |
| pf_hebb_d3_on | 560M | 2026-11-08 15:57 | coconut, flowers | disc_md_pf_hebb_d3_on(_jaci) | NÃO | NÃO (2 val) |
| pf_res128_on_b | 652M | 2026-11-08 19:31 | helminth, protozoan | disc_md_pf_res128_on_b(_jaci) | NÃO | NÃO (6 val) |
| pf_res160_on_b | 378M | 2026-11-08 20:26 | Eggs | disc_md_pf_res160_on_b(_jaci) | NÃO | NÃO (2 val) |
| pf_res224_on_b | 1.4G | 2026-11-08 22:03 | Eggs, flowers | disc_md_pf_res224_on_b(_jaci) | NÃO | NÃO (4 val) |
| pf_res320_conf | 4.3G | 2026-11-08 19:10 | aid, remote | disc_md_pf_res320_conf_jaci | NÃO | NÃO (8 val) |
| pf_res448_off | 4.0G | 2026-11-10 22:56 | aid, remote | disc_md_pf_res448_off(_jaci) | NÃO | NÃO (4 val) |
| pf_res64_on_b | 99M | 2026-11-08 19:24 | Protozoan | disc_md_pf_res64_on_b(_jaci) | NÃO | NÃO (2 val) |
| pf_res90_on_b | 196M | 2026-11-08 21:41 | coconut | disc_md_pf_res90_on_b(_jaci) | NÃO | NÃO (2 val) |
| receita_tta | 5.2G | 2026-11-11 01:56 | aid, coconut, Eggs, ham10000, Larvae, Protozoan | disc_md_receita_tta_jaci, disc_md_relglob_somar_bases_jaci | NÃO (a tmux `receita_tta` foi fechada pelo K3) | NÃO (18+18 val)* |
| restr5m_convnext_cena | 2.5G | 2026-11-11 13:53 | aid, coconut, remote | disc_md_restr5m_convnext_cena_jaci | NÃO | NÃO (54 val) |
| restr5m_convnext_obj | 6.5G | 2026-11-12 03:11 | Eggs, flowers, ham10000, Larvae, Protozoan | disc_md_restr5m_convnext_obj_jaci | NÃO | NÃO (144 val) |
| restr5m_dwr_s176 | 7.3G | 2026-11-11 19:12 | 8 bases | disc_md_restr5m_dwr_s176_jaci | NÃO | NÃO (66 val) |
| restr5m_effnet_s104 | 3.7G | 2026-11-11 14:33 | 8 bases | disc_md_restr5m_effnet_s104_jaci | NÃO | NÃO (66 val) |
| restr5m_effnet_s160 | 6.4G | 2026-11-12 02:43 | 8 bases | disc_md_restr5m_effnet_s160_jaci | NÃO | NÃO (66 val) |
| restr5m_k3_cena | 2.5G | 2026-11-11 11:06 | aid, coconut, remote | disc_md_restr5m_k3_cena_jaci + hyb_ssl_encoder_{aid,coconut_trees,remote_sensing}_jaci | **SIM (dúvida)**: fontes dos encoders do estudo SSL (`ssl_cadeia.py`), cadeia derrubada pelo K3, pode voltar | NÃO (270 val)** |
| restr5m_k3_obj | 13G | 2026-11-13 01:31 | Eggs, flowers, ham10000, helminth, Larvae, Protozoan | disc_md_restr5m_k3_obj_jaci + hyb_ssl_encoder_{Eggs,flowers,ham10000,helminth_eggs,helminth_larvae,Larvae}_jaci | **SIM (dúvida)**: mesmo motivo | NÃO (720 val)** |
| restr5m_k5_cena | 2.5G | 2026-11-11 13:06 | aid, coconut, remote | disc_md_restr5m_k5_cena_jaci | NÃO | NÃO (18 val) |
| restr5m_k5_obj | 8.9G | 2026-11-12 03:34 | Eggs, flowers, ham10000, Larvae, protozoan, Protozoan | disc_md_restr5m_k5_obj_jaci + hyb_ssl_encoder_{Protozoan,protozoan_cysts}_jaci | **SIM (dúvida)**: mesmo motivo | NÃO (48 val)** |

"8 bases" = aid, coconut, Eggs, flowers, ham10000, Larvae, Protozoan, remote. Nos .pt, `remote` é o prefixo de remote_sensing e `helminth`/`protozoan` são de helminth_eggs/larvae e protozoan_cysts.

\* O teste da receita existe só no **athena**, e com outras pastas de fontes (`/exps/moliveira/fontes_filtros/teste_aid_receita_tta` em aid e `receita_tta_ham100` em ham10000 pct100). Pela regra, isso não vale para a `receita_tta` do jaci.
\** Os encoders montados a partir dessas fontes (`exps/encoders/ssl_pequeno/<base>/{fps,random_z}`) foram medidos em teste depois do SSL (`svm_ssl_pequeno_{lejepa,barlow}_test_catttapw_jaci.csv`). Esse é um teste do modelo treinado, não da leitura direta das fontes. Pela regra, não conta.

### (ii) κ de teste por base/split/pct

Nenhuma pasta do jaci tem resultado de teste. Nenhum YAML que aponta para elas tem `avaliar: test` ou `eval_set: test`, e nenhum CSV com test/teste no nome sai delas. No jaci, os únicos CSVs `disc_md_*test*` são os de `ijepa_pontos_test` (em `results_sol`), e esse YAML não usa `fontes`. Por isso não há tabela de κ, F1 nem dim.

### (iii) Candidatas a apagar (sem uso E com teste)

**Nenhuma: 0 GB.** Todas as 33 pastas falham no critério "tem teste".

- Se o critério fosse só "sem uso": 28 pastas, cerca de 65 GB (91 GB menos os ~25,7 GB das 5 em uso).
- As 5 em uso são `leis_p1_s43`, `leis_p1b_s43`, `restr5m_k3_cena`, `restr5m_k3_obj` e `restr5m_k5_obj`.

## O que foi apagado e o que ficou

**Apagado (athena):**

| pasta | GB | motivo | lista salva antes |
|---|---|---|---|
| `/exps/moliveira/fontes_filtros/receita_tta_ham100` | 0,83 | sem uso; teste completo (3 splits × 4 braços) já está na tabela 3 | `/exps/moliveira/queues/logs/apagados_fontes_filtros_receita_tta_ham100_20261114.txt` |

**Apagado (jaci):** nada.

**Ficou:**
- **athena, 52 pastas (~191 GB):**
  - 50 não têm resultado de teste, só validação.
  - `teste_aid_receita_tta` e `teste_aid_restr5m_45M` têm teste incompleto: aid split3 a 50% (fps) e split3 a 100% na receita; split2 a 50% e split3 a 100% no 4,5 M, por OOM ou porque o run morreu. As fontes servem para terminar esses pontos.
  - `dino_pontos` também é lida por `A_reports/november/analise_textura/dino_pontos.py`.
- **jaci, 33 pastas (91 GB):**
  - nenhuma tem resultado de teste;
  - 5 estão em uso ou na dúvida: `leis_p1_s43` e `leis_p1b_s43` estão em `fila4_espera.txt`; `restr5m_k3_cena`, `restr5m_k3_obj` e `restr5m_k5_obj` são as fontes dos encoders do estudo SSL.

**Para liberar o resto,** um destes dois caminhos:
1. Rodar o teste de cada `disc_md`, cerca de 1 h por ponto.
2. Aceitar apagar sem teste. As fontes são só cache: o SPiFiL é determinístico, e a pasta se refaz ao rodar o YAML de novo.

## Atualização 14/11 ~01:10: limpeza do resto (pedido do Mateus: "apague somente o que não está sendo usado")
- **athena:** apagadas as outras 50 pastas sem uso (≈170 GB; os resultados que existiam, só de validação, estão no inventário acima). Ficaram `teste_aid_receita_tta` e `teste_aid_restr5m_45M` (teste incompleto; as fontes servem para fechar os pontos que faltam). Lista: `/exps/moliveira/queues/logs/apagados_fontes_filtros_resto_athena_20261114.txt`.
- **jaci:** apagadas 28 pastas sem uso (≈67 GB). Ficaram `leis_p1_s43`, `leis_p1b_s43` (fila4_espera) e `restr5m_k3_cena`, `restr5m_k3_obj`, `restr5m_k5_obj` (fontes dos encoders do estudo SSL). Lista: `/dados/home/moliveira/exps/queues/logs/apagados_fontes_filtros_jaci_20261114.txt` (no jaci).
- O SPiFiL é determinístico: rodar o YAML de novo refaz qualquer pasta apagada.
