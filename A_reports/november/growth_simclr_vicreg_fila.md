# Fila do crescimento AE / SimCLR / VICReg

**Última atualização:** 15/11/2026 12:28

**⏸ Fila g2 PAUSADA às 12:20 de 15/11 por ordem do Mateus (`g2.hold`).** Está rodando um SVM parcial das 2 últimas camadas dos 15 runs que estavam em andamento (tmux `svm_parcial_gpu2`/`svm_parcial_gpu3`; CSVs em `Scalable_Hybrid_FLIM_discovery/results/parcial_15nov/`; log em `/exps/moliveira/queues/logs/parcial_15nov/fim.log`). Quando ele acabar, os treinos g2 serão derrubados, e as GPUs 2/3 passam a rodar o MO-DEHB Hybrid SPiFiL mandado pela Jaci (`artifacts/modehb_hybrid/` no repo Scalable_Hybrid_FLIM). Este arquivo para de ser reescrito a cada 30 min até nova ordem.

**Tags:** `doing` rodando agora · `done` terminou · `wait` esperando (não rodou ou foi parado) · `failed` falhou. 🎲 = aleatório (rand/random_z): fica listado, mas não volta sem ordem do Mateus.

**Prioridade (Mateus, 13/11 ~03:45):** 50% e 100% nas 4 redes (2, 1, 3, 4) primeiro; 75% depois; Inception/residual e augmentation por último.

| seção | itens | doing | done | wait | failed | SVM done |
|---|---|---|---|---|---|---|
| Esperando (derrubados no K0) | 438 | 0 | 97 | 340 | 1 | — |
| 50% — Rede 2 (D=2048) | 81 | 10 | 22 | 49 | 0 | 21 |
| 50% — Rede 1 (compacta 4×4, final 1024) | 81 | 0 | 12 | 69 | 0 | 12 |
| 50% — Rede 3 (GeM+cat, final 1024) | 81 | 5 | 7 | 69 | 0 | 7 |
| 50% — Rede 4 (Inception 2×2, final 1024) | 81 | 2 | 6 | 73 | 0 | 6 |
| 100% — Rede 2 (D=2048) | 81 | 0 | 0 | 81 | 0 | 0 |
| 100% — Rede 1 (compacta 4×4, final 1024) | 81 | 0 | 0 | 81 | 0 | 0 |
| 100% — Rede 3 (GeM+cat, final 1024) | 81 | 0 | 0 | 81 | 0 | 0 |
| 100% — Rede 4 (Inception 2×2, final 1024) | 81 | 0 | 0 | 81 | 0 | 0 |
| 75% — Rede 2 (D=2048) | 81 | 0 | 0 | 81 | 0 | 0 |
| 75% — Rede 1 (compacta 4×4, final 1024) | 81 | 0 | 0 | 81 | 0 | 0 |
| 75% — Rede 3 (GeM+cat, final 1024) | 81 | 0 | 0 | 81 | 0 | 0 |
| 75% — Rede 4 (Inception 2×2, final 1024) | 81 | 0 | 0 | 81 | 0 | 0 |
| Inception e residual na camada nova (rede 2, só growth/AE) | 162 | 0 | 0 | 162 | 0 | 0 |
| Augmentation 1ª rodada (sem backprop, spf_grade da rede 2, 50%) | 108 | 0 | 0 | 108 | 0 | 0 |
| **fila nova (total)** | **1242** | 17 | 47 | 1178 | 0 | 46 |

| máquina | rodando agora |
|---|---|
| jaci GPU 0 | liberado para outros experimentos (14/11 ~00:40); nada do g2 roda lá (hoje roda o MO-DEHB estratificado, tmux `modehb_strat`, fora desta fila) |
| athena GPU 2 | g2_ae_fps_coco_s3_p50 (L5 congelada); g2_ae_fps_ham_s1_p50 (L4 congelada); g2_ae_fps_ham_s3_p50 (L4 congelada); g2_ae_fps_food_s1_p50 (L3 congelada); g2_ae_fps_food_s2_p50 (L3 congelada); g2_ae_fps_food_s3_p50 (L2 congelada); g2_vicreg_fps_aid_s1_p50 (L3 congelada); g3_ae_fps_aid_s1_p50 (L3 congelada); g3_vicreg_fps_aid_s1_p50 (L2 solta (_ft)) |
| athena GPU 3 | g2_ae_fps_coco_s1_p50 (L6 solta (_ft)); g2_ae_fps_ham_s2_p50 (L6 congelada); g2_simclr_fps_aid_s1_p50 (L6 solta (_ft)); g3_simclr_fps_aid_s1_p50 (L4 congelada); g3_simclr_fps_rs_s3_p50 (L6 congelada); g3_vicreg_fps_rs_s3_p50 (L6 congelada); g4_ae_fps_aid_s1_p50 (L3 congelada); g4_simclr_fps_aid_s1_p50 (L1 solta (_ft)) |

Fora da fila: o benchmark dos modelos grandes em coconut_trees e ham10000 a 100% também roda nas GPUs 2 e 3 do athena (14/11 ~03:00).

## Resultados até agora

Fonte: `g2_resultados.tsv` (camada final: L4_ft na rede 1, L6_ft nas redes 2-4, largura cheia; média dos splits prontos). Embaixo de cada rede/%/base vem o **Aleatório puro**: a mesma rede com pesos aleatórios init He (kaiming_normal, bias 0, seed 0) e sem treino, no estágio L0, pela mesma leitura gemtta (`results/svm_rand_g*_{val,test}_gemtta.csv`; "—" enquanto o CSV não tem o número). O aleatório da rede 2 em food101 ficou para depois (14/11), porque o SVM dele era lento demais.

| arquitetura | % | método | base | splits | κ val médio | κ test médio |
|---|---|---|---|---|---|---|
| Rede 2 (D=2048) | 50% | growth (AE) | Eggs | 3 | 0.8534 | 0.8542 |
| Rede 2 (D=2048) | 50% | **Aleatório puro** | Eggs | 3 | 0.8015 | 0.8064 |
| Rede 2 (D=2048) | 50% | growth (AE) | Larvae | 2 | 0.6442 | 0.7267 |
| Rede 2 (D=2048) | 50% | **Aleatório puro** | Larvae | 3 | 0.4257 | 0.4635 |
| Rede 2 (D=2048) | 50% | growth (AE) | Protozoan | 1 | 0.8497 | 0.8445 |
| Rede 2 (D=2048) | 50% | **Aleatório puro** | Protozoan | 3 | 0.7046 | 0.7003 |
| Rede 2 (D=2048) | 50% | growth (AE) | aid | 3 | 0.7444 | 0.7460 |
| Rede 2 (D=2048) | 50% | **Aleatório puro** | aid | 3 | 0.6937 | 0.7065 |
| Rede 2 (D=2048) | 50% | **Aleatório puro** | coconut_trees | 3 | 0.8021 | 0.8130 |
| Rede 2 (D=2048) | 50% | growth (AE) | flowers | 3 | 0.6246 | 0.6245 |
| Rede 2 (D=2048) | 50% | **Aleatório puro** | flowers | 3 | 0.5833 | 0.5635 |
| Rede 2 (D=2048) | 50% | **Aleatório puro** | ham10000 | 3 | 0.4932 | 0.4916 |
| Rede 2 (D=2048) | 50% | growth (AE) | remote_sensing | 3 | 0.5006 | 0.4862 |
| Rede 2 (D=2048) | 50% | SimCLR | remote_sensing | 3 | 0.5022 | 0.4665 |
| Rede 2 (D=2048) | 50% | VICReg | remote_sensing | 3 | 0.4425 | 0.4882 |
| Rede 2 (D=2048) | 50% | **Aleatório puro** | remote_sensing | 3 | 0.5155 | 0.5011 |
| Rede 1 (compacta 4×4, final 1024) | 50% | growth (AE) | aid | 1 | 0.7491 | 0.7539 |
| Rede 1 (compacta 4×4, final 1024) | 50% | SimCLR | aid | 1 | 0.6739 | 0.6705 |
| Rede 1 (compacta 4×4, final 1024) | 50% | VICReg | aid | 1 | 0.6766 | 0.6814 |
| Rede 1 (compacta 4×4, final 1024) | 50% | **Aleatório puro** | aid | 0 | — | — |
| Rede 1 (compacta 4×4, final 1024) | 50% | growth (AE) | remote_sensing | 3 | 0.5076 | 0.4883 |
| Rede 1 (compacta 4×4, final 1024) | 50% | SimCLR | remote_sensing | 3 | 0.5147 | 0.4775 |
| Rede 1 (compacta 4×4, final 1024) | 50% | VICReg | remote_sensing | 3 | 0.5023 | 0.4761 |
| Rede 1 (compacta 4×4, final 1024) | 50% | **Aleatório puro** | remote_sensing | 3 | 0.4881 | 0.4782 |
| Rede 3 (GeM+cat, final 1024) | 50% | growth (AE) | remote_sensing | 3 | 0.5133 | 0.5007 |
| Rede 3 (GeM+cat, final 1024) | 50% | SimCLR | remote_sensing | 2 | 0.5425 | 0.4437 |
| Rede 3 (GeM+cat, final 1024) | 50% | VICReg | remote_sensing | 2 | 0.5336 | 0.4575 |
| Rede 3 (GeM+cat, final 1024) | 50% | **Aleatório puro** | remote_sensing | 3 | 0.5103 | 0.4868 |
| Rede 4 (Inception 2×2, final 1024) | 50% | growth (AE) | remote_sensing | 2 | 0.5098 | 0.4755 |
| Rede 4 (Inception 2×2, final 1024) | 50% | SimCLR | remote_sensing | 2 | 0.4777 | 0.4839 |
| Rede 4 (Inception 2×2, final 1024) | 50% | VICReg | remote_sensing | 2 | 0.5158 | 0.4867 |
| Rede 4 (Inception 2×2, final 1024) | 50% | **Aleatório puro** | remote_sensing | 3 | 0.4866 | 0.4959 |

**⚠ Destaque — onde o aleatório não perde.** Na média dos splits, conta empate quando a diferença de κ test é de até 0,02; por split, só entra quando o aleatório ganha:
- Rede 2 (D=2048), remote_sensing 50%: o aleatório puro (κ test 0.501) empata com growth (AE) treinado (0.486) ou ganha dele
- Rede 2 (D=2048), remote_sensing 50%: o aleatório puro (κ test 0.501) empata com SimCLR treinado (0.466) ou ganha dele
- Rede 2 (D=2048), remote_sensing 50%: o aleatório puro (κ test 0.501) empata com VICReg treinado (0.488) ou ganha dele
- Rede 1 (compacta 4×4, final 1024), remote_sensing 50%: o aleatório puro (κ test 0.478) empata com growth (AE) treinado (0.488) ou ganha dele
- Rede 1 (compacta 4×4, final 1024), remote_sensing 50%: o aleatório puro (κ test 0.478) empata com SimCLR treinado (0.477) ou ganha dele
- Rede 1 (compacta 4×4, final 1024), remote_sensing 50%: o aleatório puro (κ test 0.478) empata com VICReg treinado (0.476) ou ganha dele
- Rede 3 (GeM+cat, final 1024), remote_sensing 50%: o aleatório puro (κ test 0.487) empata com growth (AE) treinado (0.501) ou ganha dele
- Rede 3 (GeM+cat, final 1024), remote_sensing 50%: o aleatório puro (κ test 0.487) empata com SimCLR treinado (0.444) ou ganha dele
- Rede 3 (GeM+cat, final 1024), remote_sensing 50%: o aleatório puro (κ test 0.487) empata com VICReg treinado (0.457) ou ganha dele
- Rede 4 (Inception 2×2, final 1024), remote_sensing 50%: o aleatório puro (κ test 0.496) empata com growth (AE) treinado (0.475) ou ganha dele
- Rede 4 (Inception 2×2, final 1024), remote_sensing 50%: o aleatório puro (κ test 0.496) empata com SimCLR treinado (0.484) ou ganha dele
- Rede 4 (Inception 2×2, final 1024), remote_sensing 50%: o aleatório puro (κ test 0.496) empata com VICReg treinado (0.487) ou ganha dele
- Rede 1 (compacta 4×4, final 1024), remote_sensing s2 50%: o aleatório (0.503) ganha de VICReg (0.436), growth (AE) (0.449), SimCLR (0.478)
- Rede 1 (compacta 4×4, final 1024), remote_sensing s3 50%: o aleatório (0.459) ganha de SimCLR (0.444)
- Rede 2 (D=2048), remote_sensing s2 50%: o aleatório (0.517) ganha de growth (AE) (0.448), SimCLR (0.449), VICReg (0.424)
- Rede 2 (D=2048), remote_sensing s3 50%: o aleatório (0.479) ganha de SimCLR (0.414)
- Rede 3 (GeM+cat, final 1024), remote_sensing s1 50%: o aleatório (0.494) ganha de SimCLR (0.438), VICReg (0.460)
- Rede 3 (GeM+cat, final 1024), remote_sensing s2 50%: o aleatório (0.481) ganha de SimCLR (0.450), VICReg (0.455)
- Rede 4 (Inception 2×2, final 1024), remote_sensing s1 50%: o aleatório (0.526) ganha de VICReg (0.511), SimCLR (0.513), growth (AE) (0.499)
- Rede 4 (Inception 2×2, final 1024), remote_sensing s2 50%: o aleatório (0.483) ganha de VICReg (0.462), growth (AE) (0.452), SimCLR (0.455)

Modelos grandes nas mesmas base/% (média dos splits; `Scalable_Hybrid_FLIM_discovery/results/benchmark/*.csv`).

| modelo | % | base | splits | κ val médio | κ test médio |
|---|---|---|---|---|---|
| aimv2-3B-patch14-448 | 50% | Eggs | 3 | 0.9830 | 0.9825 |
| aimv2-3B-patch14-448 | 50% | Larvae | 3 | 0.9406 | 0.9731 |
| aimv2-3B-patch14-448 | 50% | Protozoan | 3 | 0.9530 | 0.9521 |
| aimv2-3B-patch14-448 | 50% | aid | 3 | 0.9807 | 0.9759 |
| aimv2-3B-patch14-448 | 50% | flowers | 3 | 0.9777 | 0.9807 |
| aimv2-3B-patch14-448 | 50% | remote_sensing | 3 | 0.6881 | 0.7432 |
| convnextv2-pico-1k-224 | 50% | Eggs | 3 | 0.9316 | 0.9379 |
| convnextv2-pico-1k-224 | 50% | Larvae | 3 | 0.8762 | 0.8728 |
| convnextv2-pico-1k-224 | 50% | Protozoan | 3 | 0.8229 | 0.8303 |
| convnextv2-pico-1k-224 | 50% | aid | 3 | 0.8644 | 0.8561 |
| convnextv2-pico-1k-224 | 50% | coconut_trees | 2 | 0.6962 | 0.7217 |
| convnextv2-pico-1k-224 | 50% | flowers | 3 | 0.8884 | 0.8894 |
| convnextv2-pico-1k-224 | 50% | remote_sensing | 3 | 0.6007 | 0.6290 |
| dinov2-giant | 50% | Eggs | 3 | 0.9804 | 0.9807 |
| dinov2-giant | 50% | Larvae | 3 | 0.9679 | 0.9739 |
| dinov2-giant | 50% | Protozoan | 3 | 0.9393 | 0.9426 |
| dinov2-giant | 50% | aid | 3 | 0.9699 | 0.9625 |
| dinov2-giant | 50% | flowers | 3 | 0.9726 | 0.9767 |
| dinov2-giant | 50% | remote_sensing | 3 | 0.7080 | 0.7098 |
| ijepa_vitg16_22k | 50% | Eggs | 3 | 0.9701 | 0.9652 |
| ijepa_vitg16_22k | 50% | Larvae | 3 | 0.8499 | 0.8767 |
| ijepa_vitg16_22k | 50% | Protozoan | 3 | 0.9205 | 0.9237 |
| ijepa_vitg16_22k | 50% | aid | 3 | 0.9026 | 0.8961 |
| ijepa_vitg16_22k | 50% | flowers | 3 | 0.9371 | 0.9401 |
| ijepa_vitg16_22k | 50% | remote_sensing | 3 | 0.6033 | 0.6482 |
| resnet50.a1_in1k | 50% | Eggs | 3 | 0.9445 | 0.9488 |
| resnet50.a1_in1k | 50% | Larvae | 3 | 0.8448 | 0.8793 |
| resnet50.a1_in1k | 50% | Protozoan | 3 | 0.7999 | 0.8135 |
| resnet50.a1_in1k | 50% | aid | 3 | 0.9009 | 0.8911 |
| resnet50.a1_in1k | 50% | coconut_trees | 3 | 0.5970 | 0.6307 |
| resnet50.a1_in1k | 50% | flowers | 3 | 0.8822 | 0.8823 |
| resnet50.a1_in1k | 50% | ham10000 | 3 | 0.4624 | 0.4686 |
| resnet50.a1_in1k | 50% | remote_sensing | 3 | 0.6564 | 0.7015 |

**Teste à parte:** g1x1024_ae rs s1 p50 (rede 1 com L4 de 1024 aleatória, AE; tmux `g1x1024_ae_rs`, log `/exps/moliveira/queues/logs/g1x1024_ae_rs.log`). Último evento: OK às 18:22:12; tmux encerrado. Resultado L4_ft (1024-d, gemtta): κ val 0.4443, κ test 0.5353, F1 test 0.5121 (`results/svm_g1x1024_ae_{val,test}_gemtta.csv`); tag `done`.

**⚠ Destaque — aleatório puro passa modelo grande:**
- coconut_trees 50%: o aleatório puro da rede 2 (0.813) passa convnextv2-pico-1k-224 (0.722)
- coconut_trees 50%: o aleatório puro da rede 2 (0.813) passa resnet50.a1_in1k (0.631)
- ham10000 50%: o aleatório puro da rede 2 (0.492) passa resnet50.a1_in1k (0.469)

**⚠ Nota fixa: o split de validação do ham10000 está viciado.** Nos `splits/split{1,2,3}.json` do ham10000, o conjunto de validação é igual ao de treino (100% de sobreposição). Por isso o κ de validação dos runs de ham não vale, e o checkpoint escolhido por `ckpt_by: val_kappa` também sai viciado; só o κ de test vale. Existe `splits/split1_modehb.json`, separado por lesion_id, como alternativa; o Mateus ainda não decidiu.

## Ordem da fila (próximos 24 de cada host)

As filas `g2fila_athena.txt` e `g2fila_jaci.txt` estão intercaladas por rede e método desde 13:37–13:57 de 13/11. A sequência alterna as 12 listas (rede 2, 1, 3, 4 × AE, SimCLR, VICReg). O teto subiu para 24 runs no jaci (90 GB) e 6 runs por GPU no athena (46,5 GB; 9 runs por GPU de 13/11 19:56 até o OOM de RAM de 14/11 01:06, depois de volta a 6; desde 14/11 21:28, 9 por GPU, até 18 em doing). A posição de cada run aparece na coluna máquina/GPU da tabela da fila nova.

| # | athena (de 908) | jaci (de 0) |
|---|---|---|
| 1 | g4_ae_fps_rs_s3_p50 | — |
| 2 | g4_vicreg_fps_aid_s1_p50 | — |
| 3 | g4_simclr_fps_rs_s3_p50 | — |
| 4 | g2_ae_fps_rprot_s2_p50 | — |
| 5 | g4_vicreg_fps_rs_s3_p50 | — |
| 6 | g2_simclr_fps_aid_s2_p50 | — |
| 7 | g2_ae_fps_rlarv_s3_p50 | — |
| 8 | g2_vicreg_fps_aid_s2_p50 | — |
| 9 | g2_simclr_fps_flw_s1_p50 | — |
| 10 | g1_ae_fps_aid_s2_p50 | — |
| 11 | g2_vicreg_fps_flw_s1_p50 | — |
| 12 | g1_simclr_fps_aid_s2_p50 | — |
| 13 | g1_ae_fps_flw_s1_p50 | — |
| 14 | g1_vicreg_fps_aid_s2_p50 | — |
| 15 | g1_simclr_fps_flw_s1_p50 | — |
| 16 | g3_ae_fps_aid_s2_p50 | — |
| 17 | g1_vicreg_fps_flw_s1_p50 | — |
| 18 | g3_simclr_fps_aid_s2_p50 | — |
| 19 | g3_ae_fps_flw_s1_p50 | — |
| 20 | g3_vicreg_fps_aid_s2_p50 | — |
| 21 | g3_simclr_fps_flw_s1_p50 | — |
| 22 | g4_ae_fps_aid_s2_p50 | — |
| 23 | g3_vicreg_fps_flw_s1_p50 | — |
| 24 | g4_simclr_fps_aid_s2_p50 | — |

**Estado:** trava `g2.hold` removida às ~04:20 de 13/11. A fila nova voltou com whitening entre camadas, lançada pelos monitores por GPU. Os runs antigos sem whitening entre camadas foram movidos para `artifacts/g2_ae_sem_whiten_between` e não contam. Histórico: às ~03:50 os treinos foram PAUSADOS por ordem do Mateus (`rc=PAUSADO_loadtune` no g2.status); os runs que estavam rodando voltaram para `wait` (coluna camada diz até onde chegaram). Voltam pelos 3 monitores (tmux `g2mon_gpu2`/`g2mon_gpu3` no athena, `g2mon_gpu0` no jaci) quando existirem `/exps/moliveira/queues/loadtune/athena.json` e `jaci:/dados/home/moliveira/exps/queues/loadtune/jaci.json`. Filas: athena 0 itens, jaci 0 itens (um run pode estar nas duas; o monitor que pegar primeiro tira a linha).

**Quem lança:** desde 14/11 ~00:40 só o athena, pelos monitores `g2mon_gpu2`/`g2mon_gpu3`, que puxam de `g2fila_athena.txt`. O jaci foi liberado para outros experimentos e a fila dele foi intercalada na do athena; `g2fila_jaci.txt` está vazia. Os 16 runs que rodavam no jaci voltaram ao topo da fila do athena, como `wait`. Os resultados que o jaci já tinha fechado continuam valendo e vêm das cópias em `/exps/moliveira/queues/logs_jaci/logs/` (`g2.status`, `g2_resultados.tsv`); os artifacts estão em `/exps/moliveira/artifacts_jaci`.

Fontes da fila nova: `g2.status` (START/END/SVMSTART/SVMEND/RETRY/PARADO), `logs/g2runs/<run>.log`, `g2fila_*.txt` (aparece como "na fila <host>"), `logs/g2_resultados.tsv` (SVM). Início e fim são a hora no log, pelo relógio de cada host. A camada atual é a última pasta `L{k}`/`L{k}_ft` com `.ckpt` em `artifacts/g<rede>_<m>/<base>_split<s>_pct<p>/`, contando só os ckpts posteriores ao START do run. Seção Esperando: os itens retomados são lidos de `pgrep` dos launchers vivos, da cauda do `sol.<yaml>.log` e de `sol.status` (athena), e de `pgrep` + `execution_state.json` dos evals (jaci).

## Esperando (derrubados no K0, 13/11) — 438 itens

Fontes: athena `/exps/moliveira/queues/logs/k0_derrubados_2026-11-13_0121.md`, `/exps/moliveira/queues/prioridade_athena.txt.antes_k0` e os logs dos launchers em `/exps/moliveira/queues/logs/` (último evento por ponto); jaci `/dados/home/moliveira/exps/queues/logs/k0_derrubados_jaci_20261112.md`, `k0_backup_fila_20261112_2223/` (ssl_cadeia.py + ordens, prioridade.txt vazia, fila4.txt vazia, fila4_espera.txt, execution_state) e os logs dos launchers em `Scalable_Hybrid_FLIM_discovery/logs/`. Os itens da ssl_cadeia que nunca começaram são um sub-launcher por base (6 pontos = 3 splits × 50/100%, ou 3 pontos quando diz só 50%).
Às 10:55 (hora do jaci) de 13/11 o disc_md_e07_on foi derrubado no jaci (`k1_antigos_jaci_2026-11-13_1055.md`); os 2 pontos de coconut_trees voltam a `wait`. Às 12:28 de 13/11 os SSL antigos que tinham voltado no athena (simclr aid, simclr flowers e cosface aid) foram derrubados de novo (`/exps/moliveira/queues/logs/k1_antigos_athena_2026-11-13_1228.md`). Os pontos que estavam treinando voltam a `wait`.
Contagem: `doing` 0 · `done` 97 · `wait` 340 · `failed` 1 · 🎲 183.

| # | experimento/run | máquina | tag | onde parou / estado | último ckpt | posição na fila antiga |
|---|---|---|---|---|---|---|
| 1 | `benchmark_coco_ham convnextv2 (coconut_trees, ham10000)` | athena GPU 2 | `wait` | começou: extração coconut_trees+ham10000 ok; SVM coconut_trees s1 p50 (κ val 0,679) e s2 p50 (κ val 0,714) feitos; parado no SVM seguinte (SVM fit 1:47:46) | — | laço do benchmark_coco_ham (1/4) |
| 2 | `benchmark_coco_ham dinov2 (coconut_trees, ham10000)` | athena GPU 2 | `wait` | não começou | — | laço do benchmark_coco_ham (2/4) |
| 3 | `benchmark_coco_ham ijepa (coconut_trees, ham10000)` | athena GPU 2 | `wait` | não começou | — | laço do benchmark_coco_ham (3/4) |
| 4 | `benchmark_coco_ham aimv2 (coconut_trees, ham10000)` | athena GPU 2 | `wait` | não começou | — | laço do benchmark_coco_ham (4/4) |
| 5 | `ssl_pequeno_barlow2048_athena aid split1 pct50` | athena GPU 3 | `done` | terminou (OK no log) | — | sol_ssl_pequeno_barlow2048_aid [1/6] |
| 6 | `ssl_pequeno_barlow2048_athena aid split1 pct100` | athena GPU 3 | `done` | terminou (OK no log) | — | sol_ssl_pequeno_barlow2048_aid [2/6] |
| 7 | `ssl_pequeno_barlow2048_athena aid split2 pct50` | athena GPU 3 | `wait` | rodando no K0, parado (registro K0) | — | sol_ssl_pequeno_barlow2048_aid [3/6] |
| 8 | `ssl_pequeno_barlow2048_athena aid split2 pct100` | athena GPU 3 | `wait` | não começou | — | sol_ssl_pequeno_barlow2048_aid [4/6] |
| 9 | `ssl_pequeno_barlow2048_athena aid split3 pct50` | athena GPU 3 | `wait` | não começou | — | sol_ssl_pequeno_barlow2048_aid [5/6] |
| 10 | `ssl_pequeno_barlow2048_athena aid split3 pct100` | athena GPU 3 | `wait` | não começou | — | sol_ssl_pequeno_barlow2048_aid [6/6] |
| 11 | `ssl_pequeno_barlow2048_athena flowers split1 pct50` | athena GPU 3 | `done` | terminou (OK no log) | — | sol_ssl_pequeno_barlow2048_flowers [1/6] |
| 12 | `ssl_pequeno_barlow2048_athena flowers split1 pct100` | athena GPU 3 | `done` | terminou (OK no log) | — | sol_ssl_pequeno_barlow2048_flowers [2/6] |
| 13 | `ssl_pequeno_barlow2048_athena flowers split2 pct50` | athena GPU 3 | `wait` | rodando no K0, parado (registro K0) | — | sol_ssl_pequeno_barlow2048_flowers [3/6] |
| 14 | `ssl_pequeno_barlow2048_athena flowers split2 pct100` | athena GPU 3 | `wait` | não começou | — | sol_ssl_pequeno_barlow2048_flowers [4/6] |
| 15 | `ssl_pequeno_barlow2048_athena flowers split3 pct50` | athena GPU 3 | `wait` | não começou | — | sol_ssl_pequeno_barlow2048_flowers [5/6] |
| 16 | `ssl_pequeno_barlow2048_athena flowers split3 pct100` | athena GPU 3 | `wait` | não começou | — | sol_ssl_pequeno_barlow2048_flowers [6/6] |
| 17 | `ssl_pequeno_cosface_athena aid split1 pct50` | athena GPU 3 | `done` | terminou (OK no log) | — | sol_ssl_pequeno_cosface_aid [1/6] |
| 18 | `ssl_pequeno_cosface_athena aid split1 pct100` | athena GPU 3 | `done` | terminou (OK no log) | — | sol_ssl_pequeno_cosface_aid [2/6] |
| 19 | `ssl_pequeno_cosface_athena aid split2 pct50` | athena GPU 3 | `wait` | começou (SUBMIT sem OK no log); parado | — | sol_ssl_pequeno_cosface_aid [3/6] |
| 20 | `ssl_pequeno_cosface_athena aid split2 pct100` | athena GPU 3 | `wait` | rodando no K0, parado (registro K0) | — | sol_ssl_pequeno_cosface_aid [4/6] |
| 21 | `ssl_pequeno_cosface_athena aid split3 pct50` | athena GPU 3 | `wait` | não começou | — | sol_ssl_pequeno_cosface_aid [5/6] |
| 22 | `ssl_pequeno_cosface_athena aid split3 pct100` | athena GPU 3 | `wait` | não começou | — | sol_ssl_pequeno_cosface_aid [6/6] |
| 23 | `ssl_pequeno_simclr_athena aid split1 pct50` | athena GPU 2 | `done` | terminou (OK no log) | — | sol_ssl_pequeno_simclr_aid [1/6] |
| 24 | `ssl_pequeno_simclr_athena aid split1 pct100` | athena GPU 2 | `done` | terminou (OK no log) | — | sol_ssl_pequeno_simclr_aid [2/6] |
| 25 | `ssl_pequeno_simclr_athena aid split2 pct50` | athena GPU 2 | `wait` | começou (SUBMIT sem OK no log); parado | — | sol_ssl_pequeno_simclr_aid [3/6] |
| 26 | `ssl_pequeno_simclr_athena aid split2 pct100` | athena GPU 2 | `wait` | começou (SUBMIT sem OK no log); parado | — | sol_ssl_pequeno_simclr_aid [4/6] |
| 27 | `ssl_pequeno_simclr_athena aid split3 pct50` | athena GPU 2 | `wait` | não começou | — | sol_ssl_pequeno_simclr_aid [5/6] |
| 28 | `ssl_pequeno_simclr_athena aid split3 pct100` | athena GPU 2 | `wait` | não começou | — | sol_ssl_pequeno_simclr_aid [6/6] |
| 29 | `ssl_pequeno_simclr_athena flowers split1 pct50` | athena GPU 2 | `done` | terminou (OK no log) | — | sol_ssl_pequeno_simclr_flowers [1/6] |
| 30 | `ssl_pequeno_simclr_athena flowers split1 pct100` | athena GPU 2 | `done` | terminou (OK no log) | — | sol_ssl_pequeno_simclr_flowers [2/6] |
| 31 | `ssl_pequeno_simclr_athena flowers split2 pct50` | athena GPU 2 | `done` | terminou (OK no log) | — | sol_ssl_pequeno_simclr_flowers [3/6] |
| 32 | `ssl_pequeno_simclr_athena flowers split2 pct100` | athena GPU 2 | `done` | terminou (OK no log) | — | sol_ssl_pequeno_simclr_flowers [4/6] |
| 33 | `ssl_pequeno_simclr_athena flowers split3 pct50` | athena GPU 2 | `wait` | começou (SUBMIT sem OK no log); parado | — | sol_ssl_pequeno_simclr_flowers [5/6] |
| 34 | `ssl_pequeno_simclr_athena flowers split3 pct100` | athena GPU 2 | `wait` | começou (SUBMIT sem OK no log); parado | — | sol_ssl_pequeno_simclr_flowers [6/6] |
| 35 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_eggs split1 pct1` | athena GPU 2/3 | `done` | terminou (OK no log) | — | rand_impuro eggs [1/18] |
| 36 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_eggs split1 pct5` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro eggs [2/18] |
| 37 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_eggs split1 pct25` | athena GPU 2/3 | `done` | terminou (OK no log) | — | rand_impuro eggs [3/18] |
| 38 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_eggs split1 pct50` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro eggs [4/18] |
| 39 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_eggs split1 pct75` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro eggs [5/18] |
| 40 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_eggs split1 pct100` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro eggs [6/18] |
| 41 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_eggs split2 pct1` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro eggs [7/18] |
| 42 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_eggs split2 pct5` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro eggs [8/18] |
| 43 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_eggs split2 pct25` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro eggs [9/18] |
| 44 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_eggs split2 pct50` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro eggs [10/18] |
| 45 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_eggs split2 pct75` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro eggs [11/18] |
| 46 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_eggs split2 pct100` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro eggs [12/18] |
| 47 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_eggs split3 pct1` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro eggs [13/18] |
| 48 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_eggs split3 pct5` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro eggs [14/18] |
| 49 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_eggs split3 pct25` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro eggs [15/18] |
| 50 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_eggs split3 pct50` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro eggs [16/18] |
| 51 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_eggs split3 pct75` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro eggs [17/18] |
| 52 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_eggs split3 pct100` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro eggs [18/18] |
| 53 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_larvae split1 pct1` | athena GPU 2/3 | `done` | terminou (OK no log) | — | rand_impuro larvae [1/18] |
| 54 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_larvae split1 pct5` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro larvae [2/18] |
| 55 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_larvae split1 pct25` | athena GPU 2/3 | `done` | terminou (OK no log) | — | rand_impuro larvae [3/18] |
| 56 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_larvae split1 pct50` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro larvae [4/18] |
| 57 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_larvae split1 pct75` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro larvae [5/18] |
| 58 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_larvae split1 pct100` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro larvae [6/18] |
| 59 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_larvae split2 pct1` | athena GPU 2/3 | `done` | terminou (OK no log) | — | rand_impuro larvae [7/18] |
| 60 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_larvae split2 pct5` | athena GPU 2/3 | `done` | terminou (OK no log) | — | rand_impuro larvae [8/18] |
| 61 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_larvae split2 pct25` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro larvae [9/18] |
| 62 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_larvae split2 pct50` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro larvae [10/18] |
| 63 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_larvae split2 pct75` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro larvae [11/18] |
| 64 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_larvae split2 pct100` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro larvae [12/18] |
| 65 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_larvae split3 pct1` | athena GPU 2/3 | `done` | terminou (OK no log) | — | rand_impuro larvae [13/18] |
| 66 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_larvae split3 pct5` | athena GPU 2/3 | `done` | terminou (OK no log) | — | rand_impuro larvae [14/18] |
| 67 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_larvae split3 pct25` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro larvae [15/18] |
| 68 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_larvae split3 pct50` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro larvae [16/18] |
| 69 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_larvae split3 pct75` | athena GPU 2/3 | `done` | terminou (OK no log) | — | rand_impuro larvae [17/18] |
| 70 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_larvae split3 pct100` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro larvae [18/18] |
| 71 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_protozoan split1 pct1` | athena GPU 2/3 | `done` | terminou (OK no log) | — | rand_impuro protozoan [1/18] |
| 72 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_protozoan split1 pct5` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro protozoan [2/18] |
| 73 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_protozoan split1 pct25` | athena GPU 2/3 | `wait` | rodando no K0 (processo de treino no registro), parado | — | rand_impuro protozoan [3/18] |
| 74 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_protozoan split1 pct50` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro protozoan [4/18] |
| 75 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_protozoan split1 pct75` | athena GPU 2/3 | `wait` | começou (SUBMIT sem OK no log); parado | — | rand_impuro protozoan [5/18] |
| 76 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_protozoan split1 pct100` | athena GPU 2/3 | `wait` | rodando no K0 (processo de treino no registro), parado | — | rand_impuro protozoan [6/18] |
| 77 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_protozoan split2 pct1` | athena GPU 2/3 | `wait` | rodando no K0 (processo de treino no registro), parado | — | rand_impuro protozoan [7/18] |
| 78 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_protozoan split2 pct5` | athena GPU 2/3 | `wait` | rodando no K0 (processo de treino no registro), parado | — | rand_impuro protozoan [8/18] |
| 79 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_protozoan split2 pct25` | athena GPU 2/3 | `wait` | não começou | — | rand_impuro protozoan [9/18] |
| 80 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_protozoan split2 pct50` | athena GPU 2/3 | `wait` | não começou | — | rand_impuro protozoan [10/18] |
| 81 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_protozoan split2 pct75` | athena GPU 2/3 | `wait` | não começou | — | rand_impuro protozoan [11/18] |
| 82 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_protozoan split2 pct100` | athena GPU 2/3 | `wait` | não começou | — | rand_impuro protozoan [12/18] |
| 83 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_protozoan split3 pct1` | athena GPU 2/3 | `wait` | não começou | — | rand_impuro protozoan [13/18] |
| 84 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_protozoan split3 pct5` | athena GPU 2/3 | `wait` | não começou | — | rand_impuro protozoan [14/18] |
| 85 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_protozoan split3 pct25` | athena GPU 2/3 | `wait` | não começou | — | rand_impuro protozoan [15/18] |
| 86 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_protozoan split3 pct50` | athena GPU 2/3 | `wait` | não começou | — | rand_impuro protozoan [16/18] |
| 87 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_protozoan split3 pct75` | athena GPU 2/3 | `wait` | não começou | — | rand_impuro protozoan [17/18] |
| 88 | 🎲 `growth_disc_bp01w64c1280_bg1_rand_impuro_protozoan split3 pct100` | athena GPU 2/3 | `wait` | não começou | — | rand_impuro protozoan [18/18] |
| 89 | `disc_md_receita_tta_fundopreto helminth_eggs_s0200_split1_pct50_sh0` | athena GPU 2 | `done` | terminou (OK no log) | — | sol_disc_md_receita_tta_fundopreto [1/18] |
| 90 | `disc_md_receita_tta_fundopreto helminth_eggs_s0200_split1_pct50_sh1` | athena GPU 2 | `done` | terminou (OK no log) | — | sol_disc_md_receita_tta_fundopreto [2/18] |
| 91 | `disc_md_receita_tta_fundopreto helminth_eggs_s0200_split2_pct50_sh0` | athena GPU 2 | `wait` | não começou | — | sol_disc_md_receita_tta_fundopreto [3/18] |
| 92 | `disc_md_receita_tta_fundopreto helminth_eggs_s0200_split2_pct50_sh1` | athena GPU 2 | `wait` | não começou | — | sol_disc_md_receita_tta_fundopreto [4/18] |
| 93 | `disc_md_receita_tta_fundopreto helminth_eggs_s0200_split3_pct50_sh0` | athena GPU 2 | `wait` | não começou | — | sol_disc_md_receita_tta_fundopreto [5/18] |
| 94 | `disc_md_receita_tta_fundopreto helminth_eggs_s0200_split3_pct50_sh1` | athena GPU 2 | `wait` | não começou | — | sol_disc_md_receita_tta_fundopreto [6/18] |
| 95 | `disc_md_receita_tta_fundopreto helminth_larvae_s0200_split1_pct50_sh0` | athena GPU 2 | `done` | terminou (OK no log) | — | sol_disc_md_receita_tta_fundopreto [7/18] |
| 96 | `disc_md_receita_tta_fundopreto helminth_larvae_s0200_split1_pct50_sh1` | athena GPU 2 | `done` | terminou (OK no log) | — | sol_disc_md_receita_tta_fundopreto [8/18] |
| 97 | `disc_md_receita_tta_fundopreto helminth_larvae_s0200_split2_pct50_sh0` | athena GPU 2 | `wait` | não começou | — | sol_disc_md_receita_tta_fundopreto [9/18] |
| 98 | `disc_md_receita_tta_fundopreto helminth_larvae_s0200_split2_pct50_sh1` | athena GPU 2 | `wait` | não começou | — | sol_disc_md_receita_tta_fundopreto [10/18] |
| 99 | `disc_md_receita_tta_fundopreto helminth_larvae_s0200_split3_pct50_sh0` | athena GPU 2 | `wait` | não começou | — | sol_disc_md_receita_tta_fundopreto [11/18] |
| 100 | `disc_md_receita_tta_fundopreto helminth_larvae_s0200_split3_pct50_sh1` | athena GPU 2 | `wait` | não começou | — | sol_disc_md_receita_tta_fundopreto [12/18] |
| 101 | `disc_md_receita_tta_fundopreto protozoan_cysts_s0200_split1_pct50_sh0` | athena GPU 2 | `done` | terminou (OK no log) | — | sol_disc_md_receita_tta_fundopreto [13/18] |
| 102 | `disc_md_receita_tta_fundopreto protozoan_cysts_s0200_split1_pct50_sh1` | athena GPU 2 | `wait` | rodando no K0 (spf_grade no registro), parado | — | sol_disc_md_receita_tta_fundopreto [14/18] |
| 103 | `disc_md_receita_tta_fundopreto protozoan_cysts_s0200_split2_pct50_sh0` | athena GPU 2 | `wait` | não começou | — | sol_disc_md_receita_tta_fundopreto [15/18] |
| 104 | `disc_md_receita_tta_fundopreto protozoan_cysts_s0200_split2_pct50_sh1` | athena GPU 2 | `wait` | não começou | — | sol_disc_md_receita_tta_fundopreto [16/18] |
| 105 | `disc_md_receita_tta_fundopreto protozoan_cysts_s0200_split3_pct50_sh0` | athena GPU 2 | `wait` | não começou | — | sol_disc_md_receita_tta_fundopreto [17/18] |
| 106 | `disc_md_receita_tta_fundopreto protozoan_cysts_s0200_split3_pct50_sh1` | athena GPU 2 | `wait` | não começou | — | sol_disc_md_receita_tta_fundopreto [18/18] |
| 107 | `disc_md_teste_aid_restr5m_45M aid_s0200_split1_pct50` | athena GPU 3 | `done` | terminou (OK no log) | — | sol_disc_md_teste_aid_restr5m_45M [1/6] |
| 108 | `disc_md_teste_aid_restr5m_45M aid_s0200_split1_pct100` | athena GPU 3 | `done` | terminou (OK no log) | — | sol_disc_md_teste_aid_restr5m_45M [2/6] |
| 109 | `disc_md_teste_aid_restr5m_45M aid_s0200_split2_pct50` | athena GPU 3 | `done` | terminou (OK no log) | — | sol_disc_md_teste_aid_restr5m_45M [3/6] |
| 110 | `disc_md_teste_aid_restr5m_45M aid_s0200_split2_pct100` | athena GPU 3 | `done` | terminou (OK no log) | — | sol_disc_md_teste_aid_restr5m_45M [4/6] |
| 111 | `disc_md_teste_aid_restr5m_45M aid_s0200_split3_pct50` | athena GPU 3 | `done` | terminou (OK no log) | — | sol_disc_md_teste_aid_restr5m_45M [5/6] |
| 112 | `disc_md_teste_aid_restr5m_45M aid_s0200_split3_pct100` | athena GPU 3 | `wait` | rodando no K0 (spf_grade no registro), parado | — | sol_disc_md_teste_aid_restr5m_45M [6/6] |
| 113 | `disc_md_teste_aid_receita_tta aid_s0200_split1_pct50` | athena GPU 3 | `done` | terminou (OK no log) | — | sol_disc_md_teste_aid_receita_tta [1/6] |
| 114 | `disc_md_teste_aid_receita_tta aid_s0200_split1_pct100` | athena GPU 3 | `done` | terminou (OK no log) | — | sol_disc_md_teste_aid_receita_tta [2/6] |
| 115 | `disc_md_teste_aid_receita_tta aid_s0200_split2_pct50` | athena GPU 3 | `done` | terminou (OK no log) | — | sol_disc_md_teste_aid_receita_tta [3/6] |
| 116 | `disc_md_teste_aid_receita_tta aid_s0200_split2_pct100` | athena GPU 3 | `done` | terminou (OK no log) | — | sol_disc_md_teste_aid_receita_tta [4/6] |
| 117 | `disc_md_teste_aid_receita_tta aid_s0200_split3_pct50` | athena GPU 3 | `done` | terminou (OK no log) | — | sol_disc_md_teste_aid_receita_tta [5/6] |
| 118 | `disc_md_teste_aid_receita_tta aid_s0200_split3_pct100` | athena GPU 3 | `failed` | ERROR no log | — | sol_disc_md_teste_aid_receita_tta [6/6] |
| 119 | `seldiv.py helminth_eggs (CPU)` | athena CPU | `wait` | começou (4 dias de CPU); derrubado 01:37:38 | — | fora da fila (scratchpad) |
| 120 | `eval_svm_ssl_pequeno_barlow2048_test_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 1 |
| 121 | `ssl_pequeno_lejepa_vicreg_athena [flowers]` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 2 |
| 122 | `eval_svm_ssl_pequeno_simclr_val_pct50_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 3 |
| 123 | `eval_svm_ssl_pequeno_simclr_val_pct50_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 4 |
| 124 | `eval_svm_ssl_pequeno_simclr_val_pct100_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 5 |
| 125 | `eval_svm_ssl_pequeno_simclr_val_pct100_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 6 |
| 126 | `eval_svm_ssl_pequeno_simclr_test_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 7 |
| 127 | `eval_svm_ssl_pequeno_simclr_test_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 8 |
| 128 | `eval_svm_ssl_pequeno_cosface_val_pct50_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 9 |
| 129 | `eval_svm_ssl_pequeno_cosface_val_pct100_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 10 |
| 130 | `eval_svm_ssl_pequeno_cosface_val_pct100_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 11 |
| 131 | `eval_svm_ssl_pequeno_cosface_test_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 12 |
| 132 | `eval_svm_ssl_pequeno_cosface_test_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 13 |
| 133 | `eval_svm_ssl_pequeno_dino_val_pct50_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 14 |
| 134 | `eval_svm_ssl_pequeno_dino_val_pct100_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 15 |
| 135 | `eval_svm_ssl_pequeno_dino_val_pct100_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 16 |
| 136 | `eval_svm_ssl_pequeno_dino_test_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 17 |
| 137 | `eval_svm_ssl_pequeno_dino_test_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 18 |
| 138 | `eval_svm_ssl_pequeno_lejepa_vicreg_val_pct50_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 19 |
| 139 | `eval_svm_ssl_pequeno_lejepa_vicreg_val_pct100_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 20 |
| 140 | `eval_svm_ssl_pequeno_lejepa_vicreg_val_pct100_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 21 |
| 141 | `eval_svm_ssl_pequeno_lejepa_vicreg_test_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 22 |
| 142 | `eval_svm_ssl_pequeno_lejepa_vicreg_test_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 23 |
| 143 | `ssl_pequeno_lejepa_barlow_athena [flowers]` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 24 |
| 144 | `ssl_pequeno_lejepa_barlow_athena [coconut_trees]` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 25 |
| 145 | `ssl_pequeno_lejepa_barlow_athena [aid]` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 26 |
| 146 | `eval_svm_ssl_pequeno_lejepa_barlow_val_pct50_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 27 |
| 147 | `eval_svm_ssl_pequeno_lejepa_barlow_val_pct100_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 28 |
| 148 | `eval_svm_ssl_pequeno_lejepa_barlow_val_pct100_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 29 |
| 149 | `eval_svm_ssl_pequeno_lejepa_barlow_test_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 30 |
| 150 | `eval_svm_ssl_pequeno_lejepa_barlow_test_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 31 |
| 151 | 🎲 `ssl_pequeno_rand_simclr_athena [flowers]` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 32 |
| 152 | 🎲 `ssl_pequeno_rand_simclr_athena [coconut_trees]` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 33 |
| 153 | 🎲 `ssl_pequeno_rand_simclr_athena [aid]` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 34 |
| 154 | 🎲 `eval_svm_ssl_pequeno_rand_simclr_val_pct50_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 35 |
| 155 | 🎲 `eval_svm_ssl_pequeno_rand_simclr_val_pct50_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 36 |
| 156 | 🎲 `eval_svm_ssl_pequeno_rand_simclr_val_pct100_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 37 |
| 157 | 🎲 `eval_svm_ssl_pequeno_rand_simclr_val_pct100_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 38 |
| 158 | 🎲 `eval_svm_ssl_pequeno_rand_simclr_test_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 39 |
| 159 | 🎲 `eval_svm_ssl_pequeno_rand_simclr_test_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 40 |
| 160 | 🎲 `ssl_pequeno_rand_cosface_athena [remote_sensing]` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 41 |
| 161 | 🎲 `ssl_pequeno_rand_cosface_athena [flowers]` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 42 |
| 162 | 🎲 `ssl_pequeno_rand_cosface_athena [coconut_trees]` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 43 |
| 163 | 🎲 `ssl_pequeno_rand_cosface_athena [aid]` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 44 |
| 164 | `eval_svm_ssl_pequeno_cosface_val_pct50_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 45 |
| 165 | 🎲 `eval_svm_ssl_pequeno_rand_cosface_val_pct50_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 46 |
| 166 | 🎲 `eval_svm_ssl_pequeno_rand_cosface_val_pct50_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 47 |
| 167 | 🎲 `eval_svm_ssl_pequeno_rand_cosface_val_pct100_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 48 |
| 168 | 🎲 `eval_svm_ssl_pequeno_rand_cosface_val_pct100_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 49 |
| 169 | 🎲 `eval_svm_ssl_pequeno_rand_cosface_test_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 50 |
| 170 | 🎲 `eval_svm_ssl_pequeno_rand_cosface_test_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 51 |
| 171 | 🎲 `ssl_pequeno_rand_dino_athena [remote_sensing]` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 52 |
| 172 | 🎲 `ssl_pequeno_rand_dino_athena [flowers]` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 53 |
| 173 | 🎲 `ssl_pequeno_rand_dino_athena [coconut_trees]` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 54 |
| 174 | 🎲 `ssl_pequeno_rand_dino_athena [aid]` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 55 |
| 175 | `eval_svm_ssl_pequeno_dino_val_pct50_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 56 |
| 176 | 🎲 `eval_svm_ssl_pequeno_rand_dino_val_pct50_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 57 |
| 177 | 🎲 `eval_svm_ssl_pequeno_rand_dino_val_pct50_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 58 |
| 178 | 🎲 `eval_svm_ssl_pequeno_rand_dino_val_pct100_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 59 |
| 179 | 🎲 `eval_svm_ssl_pequeno_rand_dino_val_pct100_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 60 |
| 180 | 🎲 `eval_svm_ssl_pequeno_rand_dino_test_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 61 |
| 181 | 🎲 `eval_svm_ssl_pequeno_rand_dino_test_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 62 |
| 182 | 🎲 `ssl_pequeno_rand_lejepa_vicreg_athena [remote_sensing]` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 63 |
| 183 | 🎲 `ssl_pequeno_rand_lejepa_vicreg_athena [flowers]` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 64 |
| 184 | 🎲 `ssl_pequeno_rand_lejepa_vicreg_athena [coconut_trees]` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 65 |
| 185 | 🎲 `ssl_pequeno_rand_lejepa_vicreg_athena [aid]` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 66 |
| 186 | `eval_svm_ssl_pequeno_lejepa_vicreg_val_pct50_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 67 |
| 187 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_vicreg_val_pct50_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 68 |
| 188 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_vicreg_val_pct50_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 69 |
| 189 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_vicreg_val_pct100_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 70 |
| 190 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_vicreg_val_pct100_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 71 |
| 191 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_vicreg_test_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 72 |
| 192 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_vicreg_test_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 73 |
| 193 | 🎲 `ssl_pequeno_rand_lejepa_barlow_athena [remote_sensing]` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 74 |
| 194 | 🎲 `ssl_pequeno_rand_lejepa_barlow_athena [flowers]` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 75 |
| 195 | 🎲 `ssl_pequeno_rand_lejepa_barlow_athena [coconut_trees]` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 76 |
| 196 | 🎲 `ssl_pequeno_rand_lejepa_barlow_athena [aid]` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 77 |
| 197 | `eval_svm_ssl_pequeno_lejepa_barlow_val_pct50_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) | — | prioridade_athena.txt linha 78 |
| 198 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_barlow_val_pct50_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 79 |
| 199 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_barlow_val_pct50_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 80 |
| 200 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_barlow_val_pct100_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 81 |
| 201 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_barlow_val_pct100_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 82 |
| 202 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_barlow_test_gemttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 83 |
| 203 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_barlow_test_catttapw_athena` | athena | `wait` | não começou (na fila do monitor) — comentado #rand_espera# | — | prioridade_athena.txt linha 84 |
| 204 | `ssl_8M_lejepa_jaci_Protozoan_s1` | jaci | `done` | terminou (Fim — 1 ok, 0 falha(s).) | — | ssl_cadeia treino 1/152 |
| 205 | `ssl_8M_lejepa_jaci_Protozoan_s2` | jaci | `done` | terminou (Fim — 1 ok, 0 falha(s).) | — | ssl_cadeia treino 2/152 |
| 206 | `ssl_8M_lejepa_jaci_Protozoan_s3` | jaci | `done` | terminou (Fim — 1 ok, 0 falha(s).) | — | ssl_cadeia treino 3/152 |
| 207 | `ssl_8M_lejepa_jaci_Eggs_s1` | jaci | `done` | terminou (Fim — 1 ok, 0 falha(s).) | — | ssl_cadeia treino 4/152 |
| 208 | `ssl_8M_lejepa_jaci_Eggs_s2` | jaci | `done` | terminou (Fim — 1 ok, 0 falha(s).) | — | ssl_cadeia treino 5/152 |
| 209 | `ssl_8M_lejepa_jaci_Eggs_s3` | jaci | `wait` | não começou | — | ssl_cadeia treino 6/152 |
| 210 | `ssl_8M_lejepa_jaci_Larvae_s1` | jaci | `wait` | não começou | — | ssl_cadeia treino 7/152 |
| 211 | `ssl_8M_lejepa_jaci_Larvae_s2` | jaci | `wait` | não começou | — | ssl_cadeia treino 8/152 |
| 212 | `ssl_8M_lejepa_jaci_Larvae_s3` | jaci | `wait` | não começou | — | ssl_cadeia treino 9/152 |
| 213 | `ssl_8M_lejepa_jaci_ham10000_s1` | jaci | `wait` | não começou | — | ssl_cadeia treino 10/152 |
| 214 | `ssl_8M_lejepa_jaci_ham10000_s2` | jaci | `wait` | não começou | — | ssl_cadeia treino 11/152 |
| 215 | `ssl_8M_lejepa_jaci_ham10000_s3` | jaci | `wait` | não começou | — | ssl_cadeia treino 12/152 |
| 216 | `ssl_8M_lejepa_jaci_aid_s1` | jaci | `wait` | não começou | — | ssl_cadeia treino 13/152 |
| 217 | `ssl_8M_lejepa_jaci_aid_s2` | jaci | `wait` | não começou | — | ssl_cadeia treino 14/152 |
| 218 | `ssl_8M_lejepa_jaci_aid_s3` | jaci | `wait` | não começou | — | ssl_cadeia treino 15/152 |
| 219 | `ssl_8M_lejepa_jaci_remote_sensing_s1` | jaci | `wait` | não começou | — | ssl_cadeia treino 16/152 |
| 220 | `ssl_8M_lejepa_jaci_remote_sensing_s2` | jaci | `wait` | não começou | — | ssl_cadeia treino 17/152 |
| 221 | `ssl_8M_lejepa_jaci_remote_sensing_s3` | jaci | `wait` | não começou | — | ssl_cadeia treino 18/152 |
| 222 | `ssl_8M_lejepa_jaci_flowers_s1` | jaci | `wait` | não começou | — | ssl_cadeia treino 19/152 |
| 223 | `ssl_8M_lejepa_jaci_flowers_s2` | jaci | `wait` | não começou | — | ssl_cadeia treino 20/152 |
| 224 | `ssl_8M_lejepa_jaci_flowers_s3` | jaci | `wait` | não começou | — | ssl_cadeia treino 21/152 |
| 225 | `ssl_pequeno_lejepa_jaci_Larvae` | jaci | `done` | terminou (Fim — 6 ok, 0 falha(s).) | — | ssl_cadeia treino 22/152 |
| 226 | `ssl_pequeno_lejepa_jaci_ham10000` | jaci | `done` | terminou (Fim — 6 ok, 0 falha(s).) | — | ssl_cadeia treino 23/152 |
| 227 | `ssl_pequeno_lejepa_jaci_Eggs` | jaci | `done` | terminou (Fim — 6 ok, 0 falha(s).) | — | ssl_cadeia treino 24/152 |
| 228 | `ssl_pequeno_lejepa_jaci_Protozoan` | jaci | `done` | terminou (Fim — 6 ok, 0 falha(s).) | — | ssl_cadeia treino 25/152 |
| 229 | `ssl_pequeno_lejepa_jaci_aid` | jaci | `done` | terminou (Fim — 3 ok, 0 falha(s).) | — | ssl_cadeia treino 26/152 |
| 230 | `ssl_pequeno_lejepa_jaci_flowers` | jaci | `wait` | não começou (só 50%) | — | ssl_cadeia treino 27/152 |
| 231 | `ssl_pequeno_lejepa_jaci_helminth_larvae` | jaci | `wait` | não começou (só 50%) | — | ssl_cadeia treino 28/152 |
| 232 | `ssl_pequeno_lejepa_jaci_helminth_eggs` | jaci | `wait` | não começou (só 50%) | — | ssl_cadeia treino 29/152 |
| 233 | `ssl_pequeno_lejepa_jaci_protozoan_cysts` | jaci | `wait` | não começou (só 50%) | — | ssl_cadeia treino 30/152 |
| 234 | `ssl_pequeno_barlow2048_jaci_Larvae split1 pct50` | jaci GPU 0 | `done` | terminou (OK no log) | — | ssl_cadeia treino 31/152, ponto 1/6 |
| 235 | `ssl_pequeno_barlow2048_jaci_Larvae split1 pct100` | jaci GPU 0 | `done` | terminou (OK no log) | — | ssl_cadeia treino 31/152, ponto 2/6 |
| 236 | `ssl_pequeno_barlow2048_jaci_Larvae split2 pct50` | jaci GPU 0 | `done` | terminou (OK no log) | — | ssl_cadeia treino 31/152, ponto 3/6 |
| 237 | `ssl_pequeno_barlow2048_jaci_Larvae split2 pct100` | jaci GPU 0 | `done` | terminou (OK no log) | — | ssl_cadeia treino 31/152, ponto 4/6 |
| 238 | `ssl_pequeno_barlow2048_jaci_Larvae split3 pct50` | jaci GPU 0 | `wait` | rodando no K0, parado em L6 | `.../ssl_pequeno_barlow2048/Larvae_split3_pct50/L6/flim-ssl/5h6sv2z7/checkpoints/last.ckpt (22:15)` | ssl_cadeia treino 31/152, ponto 5/6 |
| 239 | `ssl_pequeno_barlow2048_jaci_Larvae split3 pct100` | jaci GPU 0 | `wait` | não começou | — | ssl_cadeia treino 31/152, ponto 6/6 |
| 240 | `ssl_pequeno_barlow2048_jaci_Protozoan split1 pct50` | jaci GPU 0 | `done` | terminou (OK no log) | — | ssl_cadeia treino 32/152, ponto 1/6 |
| 241 | `ssl_pequeno_barlow2048_jaci_Protozoan split1 pct100` | jaci GPU 0 | `wait` | rodando no K0, parado em L6 | `.../ssl_pequeno_barlow2048/Protozoan_split1_pct100/L6/flim-ssl/g402b9r3/checkpoints/last.ckpt (22:22)` | ssl_cadeia treino 32/152, ponto 2/6 |
| 242 | `ssl_pequeno_barlow2048_jaci_Protozoan split2 pct50` | jaci GPU 0 | `wait` | não começou | — | ssl_cadeia treino 32/152, ponto 3/6 |
| 243 | `ssl_pequeno_barlow2048_jaci_Protozoan split2 pct100` | jaci GPU 0 | `wait` | não começou | — | ssl_cadeia treino 32/152, ponto 4/6 |
| 244 | `ssl_pequeno_barlow2048_jaci_Protozoan split3 pct50` | jaci GPU 0 | `wait` | não começou | — | ssl_cadeia treino 32/152, ponto 5/6 |
| 245 | `ssl_pequeno_barlow2048_jaci_Protozoan split3 pct100` | jaci GPU 0 | `wait` | não começou | — | ssl_cadeia treino 32/152, ponto 6/6 |
| 246 | `ssl_pequeno_barlow2048_jaci_Eggs split1 pct50` | jaci GPU 0 | `done` | terminou (OK no log) | — | ssl_cadeia treino 33/152, ponto 1/6 |
| 247 | `ssl_pequeno_barlow2048_jaci_Eggs split1 pct100` | jaci GPU 0 | `wait` | rodando no K0, parado em L6 | `.../ssl_pequeno_barlow2048/Eggs_split1_pct100/L6/flim-ssl/i32vqx89/checkpoints/last.ckpt (22:22)` | ssl_cadeia treino 33/152, ponto 2/6 |
| 248 | `ssl_pequeno_barlow2048_jaci_Eggs split2 pct50` | jaci GPU 0 | `wait` | não começou | — | ssl_cadeia treino 33/152, ponto 3/6 |
| 249 | `ssl_pequeno_barlow2048_jaci_Eggs split2 pct100` | jaci GPU 0 | `wait` | não começou | — | ssl_cadeia treino 33/152, ponto 4/6 |
| 250 | `ssl_pequeno_barlow2048_jaci_Eggs split3 pct50` | jaci GPU 0 | `wait` | não começou | — | ssl_cadeia treino 33/152, ponto 5/6 |
| 251 | `ssl_pequeno_barlow2048_jaci_Eggs split3 pct100` | jaci GPU 0 | `wait` | não começou | — | ssl_cadeia treino 33/152, ponto 6/6 |
| 252 | `ssl_pequeno_barlow2048_jaci_helminth_larvae` | jaci | `wait` | não começou | — | ssl_cadeia treino 34/152 |
| 253 | `ssl_pequeno_barlow2048_jaci_helminth_eggs` | jaci | `wait` | não começou | — | ssl_cadeia treino 35/152 |
| 254 | `ssl_pequeno_barlow2048_jaci_protozoan_cysts` | jaci | `wait` | não começou | — | ssl_cadeia treino 36/152 |
| 255 | `ssl_pequeno_simclr_jaci_Larvae split1 pct50` | jaci GPU 0 | `done` | terminou (OK no log) | — | ssl_cadeia treino 37/152, ponto 1/6 |
| 256 | `ssl_pequeno_simclr_jaci_Larvae split1 pct100` | jaci GPU 0 | `wait` | rodando no K0, parado em L6 | `.../ssl_pequeno_simclr/Larvae_split1_pct100/L6/flim-ssl/ku1wcgmk/checkpoints/last.ckpt (22:16)` | ssl_cadeia treino 37/152, ponto 2/6 |
| 257 | `ssl_pequeno_simclr_jaci_Larvae split2 pct50` | jaci GPU 0 | `wait` | não começou | — | ssl_cadeia treino 37/152, ponto 3/6 |
| 258 | `ssl_pequeno_simclr_jaci_Larvae split2 pct100` | jaci GPU 0 | `wait` | não começou | — | ssl_cadeia treino 37/152, ponto 4/6 |
| 259 | `ssl_pequeno_simclr_jaci_Larvae split3 pct50` | jaci GPU 0 | `wait` | não começou | — | ssl_cadeia treino 37/152, ponto 5/6 |
| 260 | `ssl_pequeno_simclr_jaci_Larvae split3 pct100` | jaci GPU 0 | `wait` | não começou | — | ssl_cadeia treino 37/152, ponto 6/6 |
| 261 | `ssl_pequeno_simclr_jaci_Protozoan split1 pct50` | jaci GPU 0 | `done` | terminou (OK no log) | — | ssl_cadeia treino 38/152, ponto 1/6 |
| 262 | `ssl_pequeno_simclr_jaci_Protozoan split1 pct100` | jaci GPU 0 | `wait` | rodando no K0, parado em L6 | `.../ssl_pequeno_simclr/Protozoan_split1_pct100/L6/flim-ssl/cuztz2ys/checkpoints/last.ckpt (22:21)` | ssl_cadeia treino 38/152, ponto 2/6 |
| 263 | `ssl_pequeno_simclr_jaci_Protozoan split2 pct50` | jaci GPU 0 | `wait` | não começou | — | ssl_cadeia treino 38/152, ponto 3/6 |
| 264 | `ssl_pequeno_simclr_jaci_Protozoan split2 pct100` | jaci GPU 0 | `wait` | não começou | — | ssl_cadeia treino 38/152, ponto 4/6 |
| 265 | `ssl_pequeno_simclr_jaci_Protozoan split3 pct50` | jaci GPU 0 | `wait` | não começou | — | ssl_cadeia treino 38/152, ponto 5/6 |
| 266 | `ssl_pequeno_simclr_jaci_Protozoan split3 pct100` | jaci GPU 0 | `wait` | não começou | — | ssl_cadeia treino 38/152, ponto 6/6 |
| 267 | `ssl_pequeno_simclr_jaci_Eggs` | jaci | `wait` | não começou | — | ssl_cadeia treino 39/152 |
| 268 | `ssl_pequeno_simclr_jaci_ham10000` | jaci | `wait` | não começou (só 50%) | — | ssl_cadeia treino 40/152 |
| 269 | `ssl_pequeno_simclr_jaci_helminth_larvae` | jaci | `wait` | não começou (só 50%) | — | ssl_cadeia treino 41/152 |
| 270 | `ssl_pequeno_simclr_jaci_helminth_eggs` | jaci | `wait` | não começou (só 50%) | — | ssl_cadeia treino 42/152 |
| 271 | `ssl_pequeno_simclr_jaci_protozoan_cysts` | jaci | `wait` | não começou (só 50%) | — | ssl_cadeia treino 43/152 |
| 272 | `ssl_pequeno_dino_jaci_Larvae` | jaci | `wait` | não começou | — | ssl_cadeia treino 44/152 |
| 273 | `ssl_pequeno_dino_jaci_Protozoan` | jaci | `wait` | não começou | — | ssl_cadeia treino 45/152 |
| 274 | `ssl_pequeno_dino_jaci_Eggs` | jaci | `wait` | não começou | — | ssl_cadeia treino 46/152 |
| 275 | `ssl_pequeno_dino_jaci_aid` | jaci | `wait` | não começou (só 50%) | — | ssl_cadeia treino 47/152 |
| 276 | `ssl_pequeno_dino_jaci_flowers` | jaci | `wait` | não começou (só 50%) | — | ssl_cadeia treino 48/152 |
| 277 | `ssl_pequeno_dino_jaci_ham10000` | jaci | `wait` | não começou (só 50%) | — | ssl_cadeia treino 49/152 |
| 278 | `ssl_pequeno_dino_jaci_helminth_larvae` | jaci | `wait` | não começou (só 50%) | — | ssl_cadeia treino 50/152 |
| 279 | `ssl_pequeno_dino_jaci_helminth_eggs` | jaci | `wait` | não começou (só 50%) | — | ssl_cadeia treino 51/152 |
| 280 | `ssl_pequeno_dino_jaci_protozoan_cysts` | jaci | `wait` | não começou (só 50%) | — | ssl_cadeia treino 52/152 |
| 281 | `ssl_pequeno_barlow2048_jaci_ham10000` | jaci | `wait` | não começou (só 50%) | — | ssl_cadeia treino 53/152 |
| 282 | `ssl_pequeno_cosface_jaci_Larvae` | jaci | `wait` | não começou | — | ssl_cadeia treino 54/152 |
| 283 | `ssl_pequeno_cosface_jaci_Protozoan` | jaci | `wait` | não começou | — | ssl_cadeia treino 55/152 |
| 284 | `ssl_pequeno_cosface_jaci_Eggs` | jaci | `wait` | não começou | — | ssl_cadeia treino 56/152 |
| 285 | `ssl_pequeno_lejepa_barlow_jaci_Larvae` | jaci | `wait` | não começou | — | ssl_cadeia treino 57/152 |
| 286 | `ssl_pequeno_lejepa_barlow_jaci_Protozoan` | jaci | `wait` | não começou | — | ssl_cadeia treino 58/152 |
| 287 | `ssl_pequeno_lejepa_barlow_jaci_Eggs` | jaci | `wait` | não começou | — | ssl_cadeia treino 59/152 |
| 288 | `ssl_pequeno_lejepa_vicreg_jaci_Larvae` | jaci | `wait` | não começou | — | ssl_cadeia treino 60/152 |
| 289 | `ssl_pequeno_lejepa_vicreg_jaci_Protozoan` | jaci | `wait` | não começou | — | ssl_cadeia treino 61/152 |
| 290 | `ssl_pequeno_lejepa_vicreg_jaci_Eggs` | jaci | `wait` | não começou | — | ssl_cadeia treino 62/152 |
| 291 | 🎲 `ssl_8M_rand_lejepa_jaci_Protozoan_s1` | jaci | `wait` | não começou | — | ssl_cadeia treino 63/152 |
| 292 | 🎲 `ssl_8M_rand_lejepa_jaci_Protozoan_s2` | jaci | `wait` | não começou | — | ssl_cadeia treino 64/152 |
| 293 | 🎲 `ssl_8M_rand_lejepa_jaci_Protozoan_s3` | jaci | `wait` | não começou | — | ssl_cadeia treino 65/152 |
| 294 | 🎲 `ssl_8M_rand_lejepa_jaci_Eggs_s1` | jaci | `wait` | não começou | — | ssl_cadeia treino 66/152 |
| 295 | 🎲 `ssl_8M_rand_lejepa_jaci_Eggs_s2` | jaci | `wait` | não começou | — | ssl_cadeia treino 67/152 |
| 296 | 🎲 `ssl_8M_rand_lejepa_jaci_Eggs_s3` | jaci | `wait` | não começou | — | ssl_cadeia treino 68/152 |
| 297 | 🎲 `ssl_8M_rand_lejepa_jaci_Larvae_s1` | jaci | `wait` | não começou | — | ssl_cadeia treino 69/152 |
| 298 | 🎲 `ssl_8M_rand_lejepa_jaci_Larvae_s2` | jaci | `wait` | não começou | — | ssl_cadeia treino 70/152 |
| 299 | 🎲 `ssl_8M_rand_lejepa_jaci_Larvae_s3` | jaci | `wait` | não começou | — | ssl_cadeia treino 71/152 |
| 300 | 🎲 `ssl_8M_rand_lejepa_jaci_ham10000_s1` | jaci | `wait` | não começou | — | ssl_cadeia treino 72/152 |
| 301 | 🎲 `ssl_8M_rand_lejepa_jaci_ham10000_s2` | jaci | `wait` | não começou | — | ssl_cadeia treino 73/152 |
| 302 | 🎲 `ssl_8M_rand_lejepa_jaci_ham10000_s3` | jaci | `wait` | não começou | — | ssl_cadeia treino 74/152 |
| 303 | 🎲 `ssl_8M_rand_lejepa_jaci_aid_s1` | jaci | `wait` | não começou | — | ssl_cadeia treino 75/152 |
| 304 | 🎲 `ssl_8M_rand_lejepa_jaci_aid_s2` | jaci | `wait` | não começou | — | ssl_cadeia treino 76/152 |
| 305 | 🎲 `ssl_8M_rand_lejepa_jaci_aid_s3` | jaci | `wait` | não começou | — | ssl_cadeia treino 77/152 |
| 306 | 🎲 `ssl_8M_rand_lejepa_jaci_remote_sensing_s1` | jaci | `wait` | não começou | — | ssl_cadeia treino 78/152 |
| 307 | 🎲 `ssl_8M_rand_lejepa_jaci_remote_sensing_s2` | jaci | `wait` | não começou | — | ssl_cadeia treino 79/152 |
| 308 | 🎲 `ssl_8M_rand_lejepa_jaci_remote_sensing_s3` | jaci | `wait` | não começou | — | ssl_cadeia treino 80/152 |
| 309 | 🎲 `ssl_8M_rand_lejepa_jaci_flowers_s1` | jaci | `wait` | não começou | — | ssl_cadeia treino 81/152 |
| 310 | 🎲 `ssl_8M_rand_lejepa_jaci_flowers_s2` | jaci | `wait` | não começou | — | ssl_cadeia treino 82/152 |
| 311 | 🎲 `ssl_8M_rand_lejepa_jaci_flowers_s3` | jaci | `wait` | não começou | — | ssl_cadeia treino 83/152 |
| 312 | 🎲 `ssl_pequeno_rand_lejepa_jaci_Larvae` | jaci | `wait` | não começou | — | ssl_cadeia treino 84/152 |
| 313 | 🎲 `ssl_pequeno_rand_lejepa_jaci_ham10000` | jaci | `wait` | não começou | — | ssl_cadeia treino 85/152 |
| 314 | 🎲 `ssl_pequeno_rand_lejepa_jaci_Eggs` | jaci | `wait` | não começou | — | ssl_cadeia treino 86/152 |
| 315 | 🎲 `ssl_pequeno_rand_lejepa_jaci_Protozoan` | jaci | `wait` | não começou | — | ssl_cadeia treino 87/152 |
| 316 | 🎲 `ssl_pequeno_rand_simclr_jaci_Larvae` | jaci | `wait` | não começou | — | ssl_cadeia treino 88/152 |
| 317 | 🎲 `ssl_pequeno_rand_simclr_jaci_Protozoan` | jaci | `wait` | não começou | — | ssl_cadeia treino 89/152 |
| 318 | 🎲 `ssl_pequeno_rand_simclr_jaci_Eggs` | jaci | `wait` | não começou | — | ssl_cadeia treino 90/152 |
| 319 | 🎲 `ssl_pequeno_rand_dino_jaci_Larvae` | jaci | `wait` | não começou | — | ssl_cadeia treino 91/152 |
| 320 | 🎲 `ssl_pequeno_rand_dino_jaci_Protozoan` | jaci | `wait` | não começou | — | ssl_cadeia treino 92/152 |
| 321 | 🎲 `ssl_pequeno_rand_dino_jaci_Eggs` | jaci | `wait` | não começou | — | ssl_cadeia treino 93/152 |
| 322 | 🎲 `ssl_pequeno_rand_cosface_jaci_Larvae` | jaci | `wait` | não começou | — | ssl_cadeia treino 94/152 |
| 323 | 🎲 `ssl_pequeno_rand_cosface_jaci_Protozoan` | jaci | `wait` | não começou | — | ssl_cadeia treino 95/152 |
| 324 | 🎲 `ssl_pequeno_rand_cosface_jaci_Eggs` | jaci | `wait` | não começou | — | ssl_cadeia treino 96/152 |
| 325 | 🎲 `ssl_pequeno_rand_lejepa_barlow_jaci_Larvae` | jaci | `wait` | não começou | — | ssl_cadeia treino 97/152 |
| 326 | 🎲 `ssl_pequeno_rand_lejepa_barlow_jaci_Protozoan` | jaci | `wait` | não começou | — | ssl_cadeia treino 98/152 |
| 327 | 🎲 `ssl_pequeno_rand_lejepa_barlow_jaci_Eggs` | jaci | `wait` | não começou | — | ssl_cadeia treino 99/152 |
| 328 | 🎲 `ssl_pequeno_rand_lejepa_vicreg_jaci_Larvae` | jaci | `wait` | não começou | — | ssl_cadeia treino 100/152 |
| 329 | 🎲 `ssl_pequeno_rand_lejepa_vicreg_jaci_Protozoan` | jaci | `wait` | não começou | — | ssl_cadeia treino 101/152 |
| 330 | 🎲 `ssl_pequeno_rand_lejepa_vicreg_jaci_Eggs` | jaci | `wait` | não começou | — | ssl_cadeia treino 102/152 |
| 331 | `eval_svm_ssl_8M_lejepa_val_pct50_gemttapw_jaci` | jaci | `wait` | não começou (bases: 8M (7 bases × 3 splits)) | — | ssl_cadeia eval 103/152 |
| 332 | `eval_svm_ssl_8M_lejepa_val_pct50_catttapw_jaci` | jaci | `wait` | não começou (bases: 8M (7 bases × 3 splits)) | — | ssl_cadeia eval 104/152 |
| 333 | `eval_svm_ssl_8M_lejepa_test_gemttapw_jaci` | jaci | `wait` | não começou (bases: 8M (7 bases × 3 splits)) | — | ssl_cadeia eval 105/152 |
| 334 | `eval_svm_ssl_8M_lejepa_test_catttapw_jaci` | jaci | `wait` | não começou (bases: 8M (7 bases × 3 splits)) | — | ssl_cadeia eval 106/152 |
| 335 | `eval_svm_ssl_pequeno_lejepa_val_pct50_gemttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,ham10000,Eggs,Protozoan) | — | ssl_cadeia eval 107/152 |
| 336 | `eval_svm_ssl_pequeno_lejepa_val_pct50_catttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,ham10000,Eggs,Protozoan) | — | ssl_cadeia eval 108/152 |
| 337 | `eval_svm_ssl_pequeno_lejepa_val_pct100_gemttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,ham10000,Eggs,Protozoan) | — | ssl_cadeia eval 109/152 |
| 338 | `eval_svm_ssl_pequeno_lejepa_val_pct100_catttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,ham10000,Eggs,Protozoan) | — | ssl_cadeia eval 110/152 |
| 339 | `eval_svm_ssl_pequeno_lejepa_test_gemttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,ham10000,Eggs,Protozoan) | — | ssl_cadeia eval 111/152 |
| 340 | `eval_svm_ssl_pequeno_lejepa_test_catttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,ham10000,Eggs,Protozoan) | — | ssl_cadeia eval 112/152 |
| 341 | 🎲 `eval_svm_ssl_8M_rand_lejepa_val_pct50_gemttapw_jaci` | jaci | `wait` | não começou (bases: 8M (7 bases × 3 splits)) | — | ssl_cadeia eval 113/152 |
| 342 | 🎲 `eval_svm_ssl_8M_rand_lejepa_val_pct50_catttapw_jaci` | jaci | `wait` | não começou (bases: 8M (7 bases × 3 splits)) | — | ssl_cadeia eval 114/152 |
| 343 | 🎲 `eval_svm_ssl_8M_rand_lejepa_test_gemttapw_jaci` | jaci | `wait` | não começou (bases: 8M (7 bases × 3 splits)) | — | ssl_cadeia eval 115/152 |
| 344 | 🎲 `eval_svm_ssl_8M_rand_lejepa_test_catttapw_jaci` | jaci | `wait` | não começou (bases: 8M (7 bases × 3 splits)) | — | ssl_cadeia eval 116/152 |
| 345 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_val_pct50_gemttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,ham10000,Eggs,Protozoan) | — | ssl_cadeia eval 117/152 |
| 346 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_val_pct50_catttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,ham10000,Eggs,Protozoan) | — | ssl_cadeia eval 118/152 |
| 347 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_val_pct100_gemttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,ham10000,Eggs,Protozoan) | — | ssl_cadeia eval 119/152 |
| 348 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_val_pct100_catttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,ham10000,Eggs,Protozoan) | — | ssl_cadeia eval 120/152 |
| 349 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_test_gemttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,ham10000,Eggs,Protozoan) | — | ssl_cadeia eval 121/152 |
| 350 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_test_catttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,ham10000,Eggs,Protozoan) | — | ssl_cadeia eval 122/152 |
| 351 | 🎲 `eval_svm_ssl_pequeno_rand_simclr_val_pct50_gemttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 123/152 |
| 352 | 🎲 `eval_svm_ssl_pequeno_rand_simclr_val_pct50_catttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 124/152 |
| 353 | 🎲 `eval_svm_ssl_pequeno_rand_simclr_val_pct100_gemttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 125/152 |
| 354 | 🎲 `eval_svm_ssl_pequeno_rand_simclr_val_pct100_catttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 126/152 |
| 355 | 🎲 `eval_svm_ssl_pequeno_rand_simclr_test_gemttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 127/152 |
| 356 | 🎲 `eval_svm_ssl_pequeno_rand_simclr_test_catttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 128/152 |
| 357 | 🎲 `eval_svm_ssl_pequeno_rand_cosface_val_pct50_gemttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 129/152 |
| 358 | 🎲 `eval_svm_ssl_pequeno_rand_cosface_val_pct50_catttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 130/152 |
| 359 | 🎲 `eval_svm_ssl_pequeno_rand_cosface_val_pct100_gemttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 131/152 |
| 360 | 🎲 `eval_svm_ssl_pequeno_rand_cosface_val_pct100_catttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 132/152 |
| 361 | 🎲 `eval_svm_ssl_pequeno_rand_cosface_test_gemttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 133/152 |
| 362 | 🎲 `eval_svm_ssl_pequeno_rand_cosface_test_catttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 134/152 |
| 363 | 🎲 `eval_svm_ssl_pequeno_rand_dino_val_pct50_gemttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 135/152 |
| 364 | 🎲 `eval_svm_ssl_pequeno_rand_dino_val_pct50_catttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 136/152 |
| 365 | 🎲 `eval_svm_ssl_pequeno_rand_dino_val_pct100_gemttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 137/152 |
| 366 | 🎲 `eval_svm_ssl_pequeno_rand_dino_val_pct100_catttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 138/152 |
| 367 | 🎲 `eval_svm_ssl_pequeno_rand_dino_test_gemttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 139/152 |
| 368 | 🎲 `eval_svm_ssl_pequeno_rand_dino_test_catttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 140/152 |
| 369 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_vicreg_val_pct50_gemttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 141/152 |
| 370 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_vicreg_val_pct50_catttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 142/152 |
| 371 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_vicreg_val_pct100_gemttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 143/152 |
| 372 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_vicreg_val_pct100_catttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 144/152 |
| 373 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_vicreg_test_gemttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 145/152 |
| 374 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_vicreg_test_catttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 146/152 |
| 375 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_barlow_val_pct50_gemttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 147/152 |
| 376 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_barlow_val_pct50_catttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 148/152 |
| 377 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_barlow_val_pct100_gemttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 149/152 |
| 378 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_barlow_val_pct100_catttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 150/152 |
| 379 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_barlow_test_gemttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 151/152 |
| 380 | 🎲 `eval_svm_ssl_pequeno_rand_lejepa_barlow_test_catttapw_jaci` | jaci | `wait` | não começou (bases: Larvae,Protozoan,Eggs) | — | ssl_cadeia eval 152/152 |
| 381 | `eval_svm_ssl_pequeno_lejepa_test_catttapw_jaci_Protozoan` | jaci | `wait` | não começou ("eval de teste pendente: na fila logo") | — | ssl_cadeia LOGO 1 |
| 382 | `disc_md_e07_on flowers_s0200_split1_pct50_sh0` | jaci | `done` | terminou (execution_state ok) | — | disc_md_e07_on [1/4] |
| 383 | `disc_md_e07_on flowers_s0200_split1_pct50_sh1` | jaci | `done` | terminou (execution_state ok) | — | disc_md_e07_on [2/4] |
| 384 | `disc_md_e07_on coconut_trees_s0200_split1_pct50_sh0` | jaci GPU 0 | `wait` | rodando no K0 (spf_grade), parado | — | disc_md_e07_on [3/4] |
| 385 | `disc_md_e07_on coconut_trees_s0200_split1_pct50_sh1` | jaci GPU 0 | `wait` | rodando no K0 (spf_grade), parado | — | disc_md_e07_on [4/4] |
| 386 | `disc_md_restr5m_k3_obj Eggs_s0200_split1_pct50_sh0` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [1/48] |
| 387 | `disc_md_restr5m_k3_obj Eggs_s0200_split1_pct50_sh1` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [2/48] |
| 388 | `disc_md_restr5m_k3_obj Eggs_s0200_split2_pct50_sh0` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [3/48] |
| 389 | `disc_md_restr5m_k3_obj Eggs_s0200_split2_pct50_sh1` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [4/48] |
| 390 | `disc_md_restr5m_k3_obj Eggs_s0200_split3_pct50_sh0` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [5/48] |
| 391 | `disc_md_restr5m_k3_obj Eggs_s0200_split3_pct50_sh1` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [6/48] |
| 392 | `disc_md_restr5m_k3_obj Protozoan_s0200_split1_pct50_sh0` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [7/48] |
| 393 | `disc_md_restr5m_k3_obj Protozoan_s0200_split1_pct50_sh1` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [8/48] |
| 394 | `disc_md_restr5m_k3_obj Protozoan_s0200_split2_pct50_sh0` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [9/48] |
| 395 | `disc_md_restr5m_k3_obj Protozoan_s0200_split2_pct50_sh1` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [10/48] |
| 396 | `disc_md_restr5m_k3_obj Protozoan_s0200_split3_pct50_sh0` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [11/48] |
| 397 | `disc_md_restr5m_k3_obj Protozoan_s0200_split3_pct50_sh1` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [12/48] |
| 398 | `disc_md_restr5m_k3_obj Larvae_s0200_split1_pct50_sh0` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [13/48] |
| 399 | `disc_md_restr5m_k3_obj Larvae_s0200_split1_pct50_sh1` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [14/48] |
| 400 | `disc_md_restr5m_k3_obj Larvae_s0200_split2_pct50_sh0` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [15/48] |
| 401 | `disc_md_restr5m_k3_obj Larvae_s0200_split2_pct50_sh1` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [16/48] |
| 402 | `disc_md_restr5m_k3_obj Larvae_s0200_split3_pct50_sh0` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [17/48] |
| 403 | `disc_md_restr5m_k3_obj Larvae_s0200_split3_pct50_sh1` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [18/48] |
| 404 | `disc_md_restr5m_k3_obj flowers_s0200_split1_pct50_sh0` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [19/48] |
| 405 | `disc_md_restr5m_k3_obj flowers_s0200_split1_pct50_sh1` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [20/48] |
| 406 | `disc_md_restr5m_k3_obj flowers_s0200_split2_pct50_sh0` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [21/48] |
| 407 | `disc_md_restr5m_k3_obj flowers_s0200_split2_pct50_sh1` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [22/48] |
| 408 | `disc_md_restr5m_k3_obj flowers_s0200_split3_pct50_sh0` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [23/48] |
| 409 | `disc_md_restr5m_k3_obj flowers_s0200_split3_pct50_sh1` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [24/48] |
| 410 | `disc_md_restr5m_k3_obj ham10000_s0200_split1_pct50_sh0` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [25/48] |
| 411 | `disc_md_restr5m_k3_obj ham10000_s0200_split1_pct50_sh1` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [26/48] |
| 412 | `disc_md_restr5m_k3_obj ham10000_s0200_split2_pct50_sh0` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [27/48] |
| 413 | `disc_md_restr5m_k3_obj ham10000_s0200_split2_pct50_sh1` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [28/48] |
| 414 | `disc_md_restr5m_k3_obj ham10000_s0200_split3_pct50_sh0` | jaci | `wait` | rodando no K0 (spf_grade), parado (308329 já tinha saído sozinho) | — | disc_md_restr5m_k3_obj [29/48] |
| 415 | `disc_md_restr5m_k3_obj ham10000_s0200_split3_pct50_sh1` | jaci | `wait` | rodando no K0 (spf_grade), parado | — | disc_md_restr5m_k3_obj [30/48] |
| 416 | `disc_md_restr5m_k3_obj Eggs_subgroup_s0200_split1_pct50_sh0` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [31/48] |
| 417 | `disc_md_restr5m_k3_obj Eggs_subgroup_s0200_split1_pct50_sh1` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [32/48] |
| 418 | `disc_md_restr5m_k3_obj Eggs_subgroup_s0200_split2_pct50_sh0` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [33/48] |
| 419 | `disc_md_restr5m_k3_obj Eggs_subgroup_s0200_split2_pct50_sh1` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [34/48] |
| 420 | `disc_md_restr5m_k3_obj Eggs_subgroup_s0200_split3_pct50_sh0` | jaci | `wait` | não começou | — | disc_md_restr5m_k3_obj [35/48] |
| 421 | `disc_md_restr5m_k3_obj Eggs_subgroup_s0200_split3_pct50_sh1` | jaci | `wait` | não começou | — | disc_md_restr5m_k3_obj [36/48] |
| 422 | `disc_md_restr5m_k3_obj Protozoan_subgroup_s0200_split1_pct50_sh0` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [37/48] |
| 423 | `disc_md_restr5m_k3_obj Protozoan_subgroup_s0200_split1_pct50_sh1` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [38/48] |
| 424 | `disc_md_restr5m_k3_obj Protozoan_subgroup_s0200_split2_pct50_sh0` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [39/48] |
| 425 | `disc_md_restr5m_k3_obj Protozoan_subgroup_s0200_split2_pct50_sh1` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [40/48] |
| 426 | `disc_md_restr5m_k3_obj Protozoan_subgroup_s0200_split3_pct50_sh0` | jaci | `wait` | não começou | — | disc_md_restr5m_k3_obj [41/48] |
| 427 | `disc_md_restr5m_k3_obj Protozoan_subgroup_s0200_split3_pct50_sh1` | jaci | `wait` | não começou | — | disc_md_restr5m_k3_obj [42/48] |
| 428 | `disc_md_restr5m_k3_obj Larvae_subgroup_s0200_split1_pct50_sh0` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [43/48] |
| 429 | `disc_md_restr5m_k3_obj Larvae_subgroup_s0200_split1_pct50_sh1` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [44/48] |
| 430 | `disc_md_restr5m_k3_obj Larvae_subgroup_s0200_split2_pct50_sh0` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [45/48] |
| 431 | `disc_md_restr5m_k3_obj Larvae_subgroup_s0200_split2_pct50_sh1` | jaci | `done` | terminou (execution_state ok) | — | disc_md_restr5m_k3_obj [46/48] |
| 432 | `disc_md_restr5m_k3_obj Larvae_subgroup_s0200_split3_pct50_sh0` | jaci | `wait` | não começou | — | disc_md_restr5m_k3_obj [47/48] |
| 433 | `disc_md_restr5m_k3_obj Larvae_subgroup_s0200_split3_pct50_sh1` | jaci | `wait` | não começou | — | disc_md_restr5m_k3_obj [48/48] |
| 434 | `disc_md_largura_jaci` | jaci | `wait` | não começou (fila4_espera) | — | fila4_espera.txt linha 1 |
| 435 | `disc_md_leis_p1_s43_jaci` | jaci | `wait` | não começou (fila4_espera) | — | fila4_espera.txt linha 2 |
| 436 | `disc_md_leis_p1b_s43_jaci` | jaci | `wait` | não começou (fila4_espera) | — | fila4_espera.txt linha 3 |
| 437 | `disc_md_pf_leis_off_jaci` | jaci | `wait` | não começou (fila4_espera) | — | fila4_espera.txt linha 4 |
| 438 | `disc_md_pf_leis_on_jaci` | jaci | `wait` | não começou (fila4_espera) | — | fila4_espera.txt linha 5 |

## Fila nova

Plano: `growth_simclr_vicreg_plano.md`. São 9 bases, sem as 3 de fundo preto: o grupo A (aid, remote_sensing, flowers, coconut_trees, ham10000, food101) vem sempre antes do grupo B (Eggs, Larvae, Protozoan de fundo real). Toda camada nova começa pelo SPiFiL (FPS). Cada run tem 6 camadas (L1..L6), cada uma com fase congelada e fase solta (`_ft`). O checkpoint é o de melhor κ de validação, e o número final é o do teste.

**Splits conferidos (13/11):** as 9 bases têm 3 splits (`/dados/home/moliveira/datasets/organized/<base>/splits/split{1,2,3}.json`). food101 não aparece nos CSVs de benchmark do discovery.

**Abreviações:** aid = aid · rs = remote_sensing · flw = flowers · coco = coconut_trees · ham = ham10000 · food = food101 · reggs = Eggs · rlarv = Larvae · rprot = Protozoan (r = fundo real). Prefixos: g1/g2/g3/g4 = rede; ae_inc/ae_res = AE com inception/residual; aug_base/aug_fps/aug_svm/aug_2 = casos de augmentation.

### 50% — Rede 2 (D=2048) — 81 runs

_grupo A → B > método (AE → SimCLR → VICReg) > base > split_

| # | run | método | init | base | split | tag | máquina/GPU | camada atual | início | fim | SVM | κ val (camada final) | κ test (camada final) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | g2_ae_fps_aid_s1_p50 | growth (AE) | SPiFiL (FPS) | aid | 1 | `done` | jaci GPU 0 | L6 solta (_ft) | 05:35:03 | 13:34:33 | `done` | 0.7573 (L6_ft) | 0.7628 |
| 2 | g2_ae_fps_aid_s2_p50 | growth (AE) | SPiFiL (FPS) | aid | 2 | `done` | jaci GPU 0 | L6 solta (_ft) | 05:38:08 | 12:50:31 | `done` | 0.7380 (L6_ft) | 0.7463 |
| 3 | g2_ae_fps_aid_s3_p50 | growth (AE) | SPiFiL (FPS) | aid | 3 | `done` | athena GPU 2 | L6 solta (_ft) | 01:20:15 | 17:46:42 | `done` | 0.7380 (L6_ft) | 0.7290 |
| 4 | g2_ae_fps_rs_s1_p50 | growth (AE) | SPiFiL (FPS) | remote_sensing | 1 | `done` | athena GPU 3 | L6 solta (_ft) | 04:24:53 | 08:52:09 | `done` | 0.4926 (L6_ft) | 0.5269 |
| 5 | g2_ae_fps_rs_s2_p50 | growth (AE) | SPiFiL (FPS) | remote_sensing | 2 | `done` | athena GPU 2 | L6 solta (_ft) | 04:24:58 | 08:53:40 | `done` | 0.5194 (L6_ft) | 0.4479 |
| 6 | g2_ae_fps_rs_s3_p50 | growth (AE) | SPiFiL (FPS) | remote_sensing | 3 | `done` | athena GPU 2 | L6 solta (_ft) | 12:38:35 | 15:06:28 | `done` | 0.4897 (L6_ft) | 0.4839 |
| 7 | g2_ae_fps_flw_s1_p50 | growth (AE) | SPiFiL (FPS) | flowers | 1 | `done` | athena GPU 3 | L6 solta (_ft) | 12:39:20 | 16:10:15 | `done` | 0.6013 (L6_ft) | 0.6290 |
| 8 | g2_ae_fps_flw_s2_p50 | growth (AE) | SPiFiL (FPS) | flowers | 2 | `done` | athena GPU 2 | L6 solta (_ft) | 12:41:50 | 15:41:17 | `done` | 0.6260 (L6_ft) | 0.6166 |
| 9 | g2_ae_fps_flw_s3_p50 | growth (AE) | SPiFiL (FPS) | flowers | 3 | `done` | athena GPU 3 | L6 solta (_ft) | 12:42:35 | 17:00:58 | `done` | 0.6466 (L6_ft) | 0.6280 |
| 10 | g2_ae_fps_coco_s1_p50 | growth (AE) | SPiFiL (FPS) | coconut_trees | 1 | `doing` | athena GPU 3 | L6 solta (_ft) | 15:56:51 | — | `wait` | — | — |
| 11 | g2_ae_fps_coco_s2_p50 | growth (AE) | SPiFiL (FPS) | coconut_trees | 2 | `done` | athena GPU 3 | L6 solta (_ft) | 16:07:37 | 12:26:56 | `wait` | — | — |
| 12 | g2_ae_fps_coco_s3_p50 | growth (AE) | SPiFiL (FPS) | coconut_trees | 3 | `doing` | athena GPU 2 | L5 congelada | 17:47:49 | — | `wait` | — | — |
| 13 | g2_ae_fps_ham_s1_p50 | growth (AE) | SPiFiL (FPS) | ham10000 | 1 | `doing` | athena GPU 2 | L4 congelada | 20:53:09 | — | `wait` | — | — |
| 14 | g2_ae_fps_ham_s2_p50 | growth (AE) | SPiFiL (FPS) | ham10000 | 2 | `doing` | athena GPU 3 | L6 congelada | 21:26:21 | — | `wait` | — | — |
| 15 | g2_ae_fps_ham_s3_p50 | growth (AE) | SPiFiL (FPS) | ham10000 | 3 | `doing` | athena GPU 2 | L4 congelada | 21:31:50 | — | `wait` | — | — |
| 16 | g2_ae_fps_food_s1_p50 | growth (AE) | SPiFiL (FPS) | food101 | 1 | `doing` | athena GPU 2 | L3 congelada | 01:45:35 | — | `wait` | — | — |
| 17 | g2_ae_fps_food_s2_p50 | growth (AE) | SPiFiL (FPS) | food101 | 2 | `doing` | athena GPU 2 | L3 congelada | 01:34:37 | — | `wait` | — | — |
| 18 | g2_ae_fps_food_s3_p50 | growth (AE) | SPiFiL (FPS) | food101 | 3 | `doing` | athena GPU 2 | L2 congelada | 14:28:48 | — | `wait` | — | — |
| 19 | g2_simclr_fps_aid_s1_p50 | SimCLR | SPiFiL (FPS) | aid | 1 | `doing` | athena GPU 3 | L6 solta (_ft) | 21:40:25 | — | `wait` | — | — |
| 20 | g2_simclr_fps_aid_s2_p50 | SimCLR | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #6 | — | — | — | `wait` | — | — |
| 21 | g2_simclr_fps_aid_s3_p50 | SimCLR | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #30 | — | — | — | `wait` | — | — |
| 22 | g2_simclr_fps_rs_s1_p50 | SimCLR | SPiFiL (FPS) | remote_sensing | 1 | `done` | athena GPU 2 | L6 solta (_ft) | 14:00:32 | 17:15:32 | `done` | 0.4877 (L6_ft) | 0.5365 |
| 23 | g2_simclr_fps_rs_s2_p50 | SimCLR | SPiFiL (FPS) | remote_sensing | 2 | `done` | athena GPU 3 | L6 solta (_ft) | 17:27:21 | 22:32:29 | `done` | 0.5455 (L6_ft) | 0.4487 |
| 24 | g2_simclr_fps_rs_s3_p50 | SimCLR | SPiFiL (FPS) | remote_sensing | 3 | `done` | athena GPU 2 | L6 solta (_ft) | 01:42:13 | 09:07:40 | `done` | 0.4733 (L6_ft) | 0.4143 |
| 25 | g2_simclr_fps_flw_s1_p50 | SimCLR | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #9 | — | — | — | `wait` | — | — |
| 26 | g2_simclr_fps_flw_s2_p50 | SimCLR | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #33 | — | — | — | `wait` | — | — |
| 27 | g2_simclr_fps_flw_s3_p50 | SimCLR | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #57 | — | — | — | `wait` | — | — |
| 28 | g2_simclr_fps_coco_s1_p50 | SimCLR | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #54 | — | — | — | `wait` | — | — |
| 29 | g2_simclr_fps_coco_s2_p50 | SimCLR | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #78 | — | — | — | `wait` | — | — |
| 30 | g2_simclr_fps_coco_s3_p50 | SimCLR | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #102 | — | — | — | `wait` | — | — |
| 31 | g2_simclr_fps_ham_s1_p50 | SimCLR | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #126 | — | — | — | `wait` | — | — |
| 32 | g2_simclr_fps_ham_s2_p50 | SimCLR | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #150 | — | — | — | `wait` | — | — |
| 33 | g2_simclr_fps_ham_s3_p50 | SimCLR | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #174 | — | — | — | `wait` | — | — |
| 34 | g2_simclr_fps_food_s1_p50 | SimCLR | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #81 | — | — | — | `wait` | — | — |
| 35 | g2_simclr_fps_food_s2_p50 | SimCLR | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #105 | — | — | — | `wait` | — | — |
| 36 | g2_simclr_fps_food_s3_p50 | SimCLR | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #129 | — | — | — | `wait` | — | — |
| 37 | g2_vicreg_fps_aid_s1_p50 | VICReg | SPiFiL (FPS) | aid | 1 | `doing` | athena GPU 2 | L3 congelada | 21:42:42 | — | `wait` | — | — |
| 38 | g2_vicreg_fps_aid_s2_p50 | VICReg | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #8 | — | — | — | `wait` | — | — |
| 39 | g2_vicreg_fps_aid_s3_p50 | VICReg | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #32 | — | — | — | `wait` | — | — |
| 40 | g2_vicreg_fps_rs_s1_p50 | VICReg | SPiFiL (FPS) | remote_sensing | 1 | `done` | athena GPU 3 | L6 solta (_ft) | 14:03:45 | 18:44:31 | `done` | 0.5060 (L6_ft) | 0.5574 |
| 41 | g2_vicreg_fps_rs_s2_p50 | VICReg | SPiFiL (FPS) | remote_sensing | 2 | `done` | athena GPU 3 | L6 solta (_ft) | 01:45:22 | 07:20:10 | `done` | 0.3999 (L6_ft) | 0.4238 |
| 42 | g2_vicreg_fps_rs_s3_p50 | VICReg | SPiFiL (FPS) | remote_sensing | 3 | `done` | athena GPU 3 | L6 solta (_ft) | 01:34:35 | 06:52:31 | `done` | 0.4217 (L6_ft) | 0.4834 |
| 43 | g2_vicreg_fps_flw_s1_p50 | VICReg | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #11 | — | — | — | `wait` | — | — |
| 44 | g2_vicreg_fps_flw_s2_p50 | VICReg | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #35 | — | — | — | `wait` | — | — |
| 45 | g2_vicreg_fps_flw_s3_p50 | VICReg | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #59 | — | — | — | `wait` | — | — |
| 46 | g2_vicreg_fps_coco_s1_p50 | VICReg | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #56 | — | — | — | `wait` | — | — |
| 47 | g2_vicreg_fps_coco_s2_p50 | VICReg | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #80 | — | — | — | `wait` | — | — |
| 48 | g2_vicreg_fps_coco_s3_p50 | VICReg | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #104 | — | — | — | `wait` | — | — |
| 49 | g2_vicreg_fps_ham_s1_p50 | VICReg | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #128 | — | — | — | `wait` | — | — |
| 50 | g2_vicreg_fps_ham_s2_p50 | VICReg | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #152 | — | — | — | `wait` | — | — |
| 51 | g2_vicreg_fps_ham_s3_p50 | VICReg | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #176 | — | — | — | `wait` | — | — |
| 52 | g2_vicreg_fps_food_s1_p50 | VICReg | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #83 | — | — | — | `wait` | — | — |
| 53 | g2_vicreg_fps_food_s2_p50 | VICReg | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #107 | — | — | — | `wait` | — | — |
| 54 | g2_vicreg_fps_food_s3_p50 | VICReg | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #131 | — | — | — | `wait` | — | — |
| 55 | g2_ae_fps_reggs_s1_p50 | growth (AE) | SPiFiL (FPS) | Eggs | 1 | `done` | athena GPU 3 | L6 solta (_ft) | 01:09:31 | 10:57:40 | `done` | 0.8562 (L6_ft) | 0.8503 |
| 56 | g2_ae_fps_reggs_s2_p50 | growth (AE) | SPiFiL (FPS) | Eggs | 2 | `done` | athena GPU 3 | L6 solta (_ft) | 01:16:54 | 13:21:28 | `done` | 0.8436 (L6_ft) | 0.8463 |
| 57 | g2_ae_fps_reggs_s3_p50 | growth (AE) | SPiFiL (FPS) | Eggs | 3 | `done` | athena GPU 2 | L6 solta (_ft) | 01:16:59 | 14:27:34 | `done` | 0.8603 (L6_ft) | 0.8660 |
| 58 | g2_ae_fps_rlarv_s1_p50 | growth (AE) | SPiFiL (FPS) | Larvae | 1 | `done` | athena GPU 2 | L6 solta (_ft) | 17:16:52 | 20:51:29 | `done` | 0.7262 (L6_ft) | 0.7802 |
| 59 | g2_ae_fps_rlarv_s2_p50 | growth (AE) | SPiFiL (FPS) | Larvae | 2 | `done` | athena GPU 3 | L6 solta (_ft) | 01:31:18 | 06:57:34 | `done` | 0.5623 (L6_ft) | 0.6732 |
| 60 | g2_ae_fps_rlarv_s3_p50 | growth (AE) | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #7 | — | — | — | `wait` | — | — |
| 61 | g2_ae_fps_rprot_s1_p50 | growth (AE) | SPiFiL (FPS) | Protozoan | 1 | `done` | athena GPU 3 | L6 solta (_ft) | 01:20:09 | 03:24:59 | `done` | 0.8497 (L6_ft) | 0.8445 |
| 62 | g2_ae_fps_rprot_s2_p50 | growth (AE) | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #4 | — | — | — | `wait` | — | — |
| 63 | g2_ae_fps_rprot_s3_p50 | growth (AE) | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #28 | — | — | — | `wait` | — | — |
| 64 | g2_simclr_fps_reggs_s1_p50 | SimCLR | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #198 | — | — | — | `wait` | — | — |
| 65 | g2_simclr_fps_reggs_s2_p50 | SimCLR | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #222 | — | — | — | `wait` | — | — |
| 66 | g2_simclr_fps_reggs_s3_p50 | SimCLR | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #246 | — | — | — | `wait` | — | — |
| 67 | g2_simclr_fps_rlarv_s1_p50 | SimCLR | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #153 | — | — | — | `wait` | — | — |
| 68 | g2_simclr_fps_rlarv_s2_p50 | SimCLR | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #177 | — | — | — | `wait` | — | — |
| 69 | g2_simclr_fps_rlarv_s3_p50 | SimCLR | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #201 | — | — | — | `wait` | — | — |
| 70 | g2_simclr_fps_rprot_s1_p50 | SimCLR | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #270 | — | — | — | `wait` | — | — |
| 71 | g2_simclr_fps_rprot_s2_p50 | SimCLR | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #294 | — | — | — | `wait` | — | — |
| 72 | g2_simclr_fps_rprot_s3_p50 | SimCLR | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #318 | — | — | — | `wait` | — | — |
| 73 | g2_vicreg_fps_reggs_s1_p50 | VICReg | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #200 | — | — | — | `wait` | — | — |
| 74 | g2_vicreg_fps_reggs_s2_p50 | VICReg | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #224 | — | — | — | `wait` | — | — |
| 75 | g2_vicreg_fps_reggs_s3_p50 | VICReg | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #248 | — | — | — | `wait` | — | — |
| 76 | g2_vicreg_fps_rlarv_s1_p50 | VICReg | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #155 | — | — | — | `wait` | — | — |
| 77 | g2_vicreg_fps_rlarv_s2_p50 | VICReg | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #179 | — | — | — | `wait` | — | — |
| 78 | g2_vicreg_fps_rlarv_s3_p50 | VICReg | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #203 | — | — | — | `wait` | — | — |
| 79 | g2_vicreg_fps_rprot_s1_p50 | VICReg | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #272 | — | — | — | `wait` | — | — |
| 80 | g2_vicreg_fps_rprot_s2_p50 | VICReg | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #296 | — | — | — | `wait` | — | — |
| 81 | g2_vicreg_fps_rprot_s3_p50 | VICReg | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #320 | — | — | — | `wait` | — | — |

Subtotal: 81

### 50% — Rede 1 (compacta 4×4, final 1024) — 81 runs

_grupo A → B > método (AE → SimCLR → VICReg) > base > split_

| # | run | método | init | base | split | tag | máquina/GPU | camada atual | início | fim | SVM | κ val (camada final) | κ test (camada final) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 82 | g1_ae_fps_aid_s1_p50 | growth (AE) | SPiFiL (FPS) | aid | 1 | `done` | athena GPU 3 | L4 solta (_ft) | 13:33:44 | 22:54:26 | `done` | 0.7491 (L4_ft) | 0.7539 |
| 83 | g1_ae_fps_aid_s2_p50 | growth (AE) | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #10 | — | — | — | `wait` | — | — |
| 84 | g1_ae_fps_aid_s3_p50 | growth (AE) | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #34 | — | — | — | `wait` | — | — |
| 85 | g1_ae_fps_rs_s1_p50 | growth (AE) | SPiFiL (FPS) | remote_sensing | 1 | `done` | athena GPU 3 | L4 solta (_ft) | 18:45:35 | 21:55:39 | `done` | 0.5090 (L4_ft) | 0.5243 |
| 86 | g1_ae_fps_rs_s2_p50 | growth (AE) | SPiFiL (FPS) | remote_sensing | 2 | `done` | athena GPU 2 | L4 solta (_ft) | 20:10:59 | 23:45:11 | `done` | 0.5314 (L4_ft) | 0.4493 |
| 87 | g1_ae_fps_rs_s3_p50 | growth (AE) | SPiFiL (FPS) | remote_sensing | 3 | `done` | athena GPU 3 | L4 solta (_ft) | 06:53:12 | 10:39:08 | `done` | 0.4824 (L4_ft) | 0.4913 |
| 88 | g1_ae_fps_flw_s1_p50 | growth (AE) | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #13 | — | — | — | `wait` | — | — |
| 89 | g1_ae_fps_flw_s2_p50 | growth (AE) | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #37 | — | — | — | `wait` | — | — |
| 90 | g1_ae_fps_flw_s3_p50 | growth (AE) | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #61 | — | — | — | `wait` | — | — |
| 91 | g1_ae_fps_coco_s1_p50 | growth (AE) | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #58 | — | — | — | `wait` | — | — |
| 92 | g1_ae_fps_coco_s2_p50 | growth (AE) | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #82 | — | — | — | `wait` | — | — |
| 93 | g1_ae_fps_coco_s3_p50 | growth (AE) | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #106 | — | — | — | `wait` | — | — |
| 94 | g1_ae_fps_ham_s1_p50 | growth (AE) | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #130 | — | — | — | `wait` | — | — |
| 95 | g1_ae_fps_ham_s2_p50 | growth (AE) | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #154 | — | — | — | `wait` | — | — |
| 96 | g1_ae_fps_ham_s3_p50 | growth (AE) | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #178 | — | — | — | `wait` | — | — |
| 97 | g1_ae_fps_food_s1_p50 | growth (AE) | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #85 | — | — | — | `wait` | — | — |
| 98 | g1_ae_fps_food_s2_p50 | growth (AE) | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #109 | — | — | — | `wait` | — | — |
| 99 | g1_ae_fps_food_s3_p50 | growth (AE) | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #133 | — | — | — | `wait` | — | — |
| 100 | g1_simclr_fps_aid_s1_p50 | SimCLR | SPiFiL (FPS) | aid | 1 | `done` | athena GPU 3 | L4 solta (_ft) | 07:03:59 | 16:06:44 | `done` | 0.6739 (L4_ft) | 0.6705 |
| 101 | g1_simclr_fps_aid_s2_p50 | SimCLR | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #12 | — | — | — | `wait` | — | — |
| 102 | g1_simclr_fps_aid_s3_p50 | SimCLR | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #36 | — | — | — | `wait` | — | — |
| 103 | g1_simclr_fps_rs_s1_p50 | SimCLR | SPiFiL (FPS) | remote_sensing | 1 | `done` | athena GPU 3 | L4 solta (_ft) | 20:00:07 | 23:47:12 | `done` | 0.5134 (L4_ft) | 0.5102 |
| 104 | g1_simclr_fps_rs_s2_p50 | SimCLR | SPiFiL (FPS) | remote_sensing | 2 | `done` | athena GPU 3 | L4 solta (_ft) | 20:21:39 | 00:34:23 | `done` | 0.5275 (L4_ft) | 0.4785 |
| 105 | g1_simclr_fps_rs_s3_p50 | SimCLR | SPiFiL (FPS) | remote_sensing | 3 | `done` | athena GPU 3 | L4 solta (_ft) | 21:51:12 | 05:12:04 | `done` | 0.5033 (L4_ft) | 0.4438 |
| 106 | g1_simclr_fps_flw_s1_p50 | SimCLR | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #15 | — | — | — | `wait` | — | — |
| 107 | g1_simclr_fps_flw_s2_p50 | SimCLR | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #39 | — | — | — | `wait` | — | — |
| 108 | g1_simclr_fps_flw_s3_p50 | SimCLR | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #63 | — | — | — | `wait` | — | — |
| 109 | g1_simclr_fps_coco_s1_p50 | SimCLR | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #60 | — | — | — | `wait` | — | — |
| 110 | g1_simclr_fps_coco_s2_p50 | SimCLR | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #84 | — | — | — | `wait` | — | — |
| 111 | g1_simclr_fps_coco_s3_p50 | SimCLR | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #108 | — | — | — | `wait` | — | — |
| 112 | g1_simclr_fps_ham_s1_p50 | SimCLR | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #132 | — | — | — | `wait` | — | — |
| 113 | g1_simclr_fps_ham_s2_p50 | SimCLR | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #156 | — | — | — | `wait` | — | — |
| 114 | g1_simclr_fps_ham_s3_p50 | SimCLR | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #180 | — | — | — | `wait` | — | — |
| 115 | g1_simclr_fps_food_s1_p50 | SimCLR | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #87 | — | — | — | `wait` | — | — |
| 116 | g1_simclr_fps_food_s2_p50 | SimCLR | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #111 | — | — | — | `wait` | — | — |
| 117 | g1_simclr_fps_food_s3_p50 | SimCLR | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #135 | — | — | — | `wait` | — | — |
| 118 | g1_vicreg_fps_aid_s1_p50 | VICReg | SPiFiL (FPS) | aid | 1 | `done` | athena GPU 3 | L4 solta (_ft) | 10:58:41 | 22:59:07 | `done` | 0.6766 (L4_ft) | 0.6814 |
| 119 | g1_vicreg_fps_aid_s2_p50 | VICReg | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #14 | — | — | — | `wait` | — | — |
| 120 | g1_vicreg_fps_aid_s3_p50 | VICReg | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #38 | — | — | — | `wait` | — | — |
| 121 | g1_vicreg_fps_rs_s1_p50 | VICReg | SPiFiL (FPS) | remote_sensing | 1 | `done` | athena GPU 2 | L4 solta (_ft) | 20:00:08 | 23:03:19 | `done` | 0.5495 (L4_ft) | 0.5289 |
| 122 | g1_vicreg_fps_rs_s2_p50 | VICReg | SPiFiL (FPS) | remote_sensing | 2 | `done` | athena GPU 2 | L4 solta (_ft) | 20:21:51 | 23:13:12 | `done` | 0.4977 (L4_ft) | 0.4360 |
| 123 | g1_vicreg_fps_rs_s3_p50 | VICReg | SPiFiL (FPS) | remote_sensing | 3 | `done` | athena GPU 3 | L4 solta (_ft) | 22:07:28 | 03:52:13 | `done` | 0.4596 (L4_ft) | 0.4634 |
| 124 | g1_vicreg_fps_flw_s1_p50 | VICReg | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #17 | — | — | — | `wait` | — | — |
| 125 | g1_vicreg_fps_flw_s2_p50 | VICReg | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #41 | — | — | — | `wait` | — | — |
| 126 | g1_vicreg_fps_flw_s3_p50 | VICReg | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #65 | — | — | — | `wait` | — | — |
| 127 | g1_vicreg_fps_coco_s1_p50 | VICReg | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #62 | — | — | — | `wait` | — | — |
| 128 | g1_vicreg_fps_coco_s2_p50 | VICReg | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #86 | — | — | — | `wait` | — | — |
| 129 | g1_vicreg_fps_coco_s3_p50 | VICReg | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #110 | — | — | — | `wait` | — | — |
| 130 | g1_vicreg_fps_ham_s1_p50 | VICReg | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #134 | — | — | — | `wait` | — | — |
| 131 | g1_vicreg_fps_ham_s2_p50 | VICReg | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #158 | — | — | — | `wait` | — | — |
| 132 | g1_vicreg_fps_ham_s3_p50 | VICReg | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #182 | — | — | — | `wait` | — | — |
| 133 | g1_vicreg_fps_food_s1_p50 | VICReg | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #89 | — | — | — | `wait` | — | — |
| 134 | g1_vicreg_fps_food_s2_p50 | VICReg | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #113 | — | — | — | `wait` | — | — |
| 135 | g1_vicreg_fps_food_s3_p50 | VICReg | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #137 | — | — | — | `wait` | — | — |
| 136 | g1_ae_fps_reggs_s1_p50 | growth (AE) | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #202 | — | — | — | `wait` | — | — |
| 137 | g1_ae_fps_reggs_s2_p50 | growth (AE) | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #226 | — | — | — | `wait` | — | — |
| 138 | g1_ae_fps_reggs_s3_p50 | growth (AE) | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #250 | — | — | — | `wait` | — | — |
| 139 | g1_ae_fps_rlarv_s1_p50 | growth (AE) | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #157 | — | — | — | `wait` | — | — |
| 140 | g1_ae_fps_rlarv_s2_p50 | growth (AE) | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #181 | — | — | — | `wait` | — | — |
| 141 | g1_ae_fps_rlarv_s3_p50 | growth (AE) | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #205 | — | — | — | `wait` | — | — |
| 142 | g1_ae_fps_rprot_s1_p50 | growth (AE) | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #274 | — | — | — | `wait` | — | — |
| 143 | g1_ae_fps_rprot_s2_p50 | growth (AE) | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #298 | — | — | — | `wait` | — | — |
| 144 | g1_ae_fps_rprot_s3_p50 | growth (AE) | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #322 | — | — | — | `wait` | — | — |
| 145 | g1_simclr_fps_reggs_s1_p50 | SimCLR | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #204 | — | — | — | `wait` | — | — |
| 146 | g1_simclr_fps_reggs_s2_p50 | SimCLR | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #228 | — | — | — | `wait` | — | — |
| 147 | g1_simclr_fps_reggs_s3_p50 | SimCLR | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #252 | — | — | — | `wait` | — | — |
| 148 | g1_simclr_fps_rlarv_s1_p50 | SimCLR | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #159 | — | — | — | `wait` | — | — |
| 149 | g1_simclr_fps_rlarv_s2_p50 | SimCLR | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #183 | — | — | — | `wait` | — | — |
| 150 | g1_simclr_fps_rlarv_s3_p50 | SimCLR | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #207 | — | — | — | `wait` | — | — |
| 151 | g1_simclr_fps_rprot_s1_p50 | SimCLR | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #276 | — | — | — | `wait` | — | — |
| 152 | g1_simclr_fps_rprot_s2_p50 | SimCLR | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #300 | — | — | — | `wait` | — | — |
| 153 | g1_simclr_fps_rprot_s3_p50 | SimCLR | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #324 | — | — | — | `wait` | — | — |
| 154 | g1_vicreg_fps_reggs_s1_p50 | VICReg | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #206 | — | — | — | `wait` | — | — |
| 155 | g1_vicreg_fps_reggs_s2_p50 | VICReg | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #230 | — | — | — | `wait` | — | — |
| 156 | g1_vicreg_fps_reggs_s3_p50 | VICReg | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #254 | — | — | — | `wait` | — | — |
| 157 | g1_vicreg_fps_rlarv_s1_p50 | VICReg | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #161 | — | — | — | `wait` | — | — |
| 158 | g1_vicreg_fps_rlarv_s2_p50 | VICReg | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #185 | — | — | — | `wait` | — | — |
| 159 | g1_vicreg_fps_rlarv_s3_p50 | VICReg | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #209 | — | — | — | `wait` | — | — |
| 160 | g1_vicreg_fps_rprot_s1_p50 | VICReg | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #278 | — | — | — | `wait` | — | — |
| 161 | g1_vicreg_fps_rprot_s2_p50 | VICReg | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #302 | — | — | — | `wait` | — | — |
| 162 | g1_vicreg_fps_rprot_s3_p50 | VICReg | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #326 | — | — | — | `wait` | — | — |

Subtotal: 81

### 50% — Rede 3 (GeM+cat, final 1024) — 81 runs

_grupo A → B > método (AE → SimCLR → VICReg) > base > split_

| # | run | método | init | base | split | tag | máquina/GPU | camada atual | início | fim | SVM | κ val (camada final) | κ test (camada final) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 163 | g3_ae_fps_aid_s1_p50 | growth (AE) | SPiFiL (FPS) | aid | 1 | `doing` | athena GPU 2 | L3 congelada | 21:53:35 | — | `wait` | — | — |
| 164 | g3_ae_fps_aid_s2_p50 | growth (AE) | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #16 | — | — | — | `wait` | — | — |
| 165 | g3_ae_fps_aid_s3_p50 | growth (AE) | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #40 | — | — | — | `wait` | — | — |
| 166 | g3_ae_fps_rs_s1_p50 | growth (AE) | SPiFiL (FPS) | remote_sensing | 1 | `done` | athena GPU 3 | L6 solta (_ft) | 17:48:26 | 00:45:55 | `done` | 0.4935 (L6_ft) | 0.5265 |
| 167 | g3_ae_fps_rs_s2_p50 | growth (AE) | SPiFiL (FPS) | remote_sensing | 2 | `done` | athena GPU 2 | L6 solta (_ft) | 10:51:08 | 00:37:46 | `done` | 0.5434 (L6_ft) | 0.4832 |
| 168 | g3_ae_fps_rs_s3_p50 | growth (AE) | SPiFiL (FPS) | remote_sensing | 3 | `done` | athena GPU 3 | L6 solta (_ft) | 23:51:15 | 08:46:32 | `done` | 0.5029 (L6_ft) | 0.4924 |
| 169 | g3_ae_fps_flw_s1_p50 | growth (AE) | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #19 | — | — | — | `wait` | — | — |
| 170 | g3_ae_fps_flw_s2_p50 | growth (AE) | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #43 | — | — | — | `wait` | — | — |
| 171 | g3_ae_fps_flw_s3_p50 | growth (AE) | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #67 | — | — | — | `wait` | — | — |
| 172 | g3_ae_fps_coco_s1_p50 | growth (AE) | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #64 | — | — | — | `wait` | — | — |
| 173 | g3_ae_fps_coco_s2_p50 | growth (AE) | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #88 | — | — | — | `wait` | — | — |
| 174 | g3_ae_fps_coco_s3_p50 | growth (AE) | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #112 | — | — | — | `wait` | — | — |
| 175 | g3_ae_fps_ham_s1_p50 | growth (AE) | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #136 | — | — | — | `wait` | — | — |
| 176 | g3_ae_fps_ham_s2_p50 | growth (AE) | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #160 | — | — | — | `wait` | — | — |
| 177 | g3_ae_fps_ham_s3_p50 | growth (AE) | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #184 | — | — | — | `wait` | — | — |
| 178 | g3_ae_fps_food_s1_p50 | growth (AE) | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #91 | — | — | — | `wait` | — | — |
| 179 | g3_ae_fps_food_s2_p50 | growth (AE) | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #115 | — | — | — | `wait` | — | — |
| 180 | g3_ae_fps_food_s3_p50 | growth (AE) | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #139 | — | — | — | `wait` | — | — |
| 181 | g3_simclr_fps_aid_s1_p50 | SimCLR | SPiFiL (FPS) | aid | 1 | `doing` | athena GPU 3 | L4 congelada | 23:10:11 | — | `wait` | — | — |
| 182 | g3_simclr_fps_aid_s2_p50 | SimCLR | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #18 | — | — | — | `wait` | — | — |
| 183 | g3_simclr_fps_aid_s3_p50 | SimCLR | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #42 | — | — | — | `wait` | — | — |
| 184 | g3_simclr_fps_rs_s1_p50 | SimCLR | SPiFiL (FPS) | remote_sensing | 1 | `done` | athena GPU 2 | L6 solta (_ft) | 17:48:29 | 21:17:56 | `done` | 0.5359 (L6_ft) | 0.4376 |
| 185 | g3_simclr_fps_rs_s2_p50 | SimCLR | SPiFiL (FPS) | remote_sensing | 2 | `done` | athena GPU 2 | L6 solta (_ft) | 01:31:17 | 10:00:50 | `done` | 0.5491 (L6_ft) | 0.4499 |
| 186 | g3_simclr_fps_rs_s3_p50 | SimCLR | SPiFiL (FPS) | remote_sensing | 3 | `doing` | athena GPU 3 | L6 congelada | 04:02:46 | — | `wait` | — | — |
| 187 | g3_simclr_fps_flw_s1_p50 | SimCLR | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #21 | — | — | — | `wait` | — | — |
| 188 | g3_simclr_fps_flw_s2_p50 | SimCLR | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #45 | — | — | — | `wait` | — | — |
| 189 | g3_simclr_fps_flw_s3_p50 | SimCLR | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #69 | — | — | — | `wait` | — | — |
| 190 | g3_simclr_fps_coco_s1_p50 | SimCLR | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #66 | — | — | — | `wait` | — | — |
| 191 | g3_simclr_fps_coco_s2_p50 | SimCLR | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #90 | — | — | — | `wait` | — | — |
| 192 | g3_simclr_fps_coco_s3_p50 | SimCLR | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #114 | — | — | — | `wait` | — | — |
| 193 | g3_simclr_fps_ham_s1_p50 | SimCLR | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #138 | — | — | — | `wait` | — | — |
| 194 | g3_simclr_fps_ham_s2_p50 | SimCLR | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #162 | — | — | — | `wait` | — | — |
| 195 | g3_simclr_fps_ham_s3_p50 | SimCLR | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #186 | — | — | — | `wait` | — | — |
| 196 | g3_simclr_fps_food_s1_p50 | SimCLR | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #93 | — | — | — | `wait` | — | — |
| 197 | g3_simclr_fps_food_s2_p50 | SimCLR | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #117 | — | — | — | `wait` | — | — |
| 198 | g3_simclr_fps_food_s3_p50 | SimCLR | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #141 | — | — | — | `wait` | — | — |
| 199 | g3_vicreg_fps_aid_s1_p50 | VICReg | SPiFiL (FPS) | aid | 1 | `doing` | athena GPU 2 | L2 solta (_ft) | 00:48:39 | — | `wait` | — | — |
| 200 | g3_vicreg_fps_aid_s2_p50 | VICReg | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #20 | — | — | — | `wait` | — | — |
| 201 | g3_vicreg_fps_aid_s3_p50 | VICReg | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #44 | — | — | — | `wait` | — | — |
| 202 | g3_vicreg_fps_rs_s1_p50 | VICReg | SPiFiL (FPS) | remote_sensing | 1 | `done` | athena GPU 3 | L6 solta (_ft) | 17:51:42 | 00:25:07 | `done` | 0.4582 (L6_ft) | 0.4598 |
| 203 | g3_vicreg_fps_rs_s2_p50 | VICReg | SPiFiL (FPS) | remote_sensing | 2 | `done` | athena GPU 3 | L6 solta (_ft) | 07:21:13 | 13:31:17 | `done` | 0.6090 (L6_ft) | 0.4551 |
| 204 | g3_vicreg_fps_rs_s3_p50 | VICReg | SPiFiL (FPS) | remote_sensing | 3 | `doing` | athena GPU 3 | L6 congelada | 05:21:38 | — | `wait` | — | — |
| 205 | g3_vicreg_fps_flw_s1_p50 | VICReg | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #23 | — | — | — | `wait` | — | — |
| 206 | g3_vicreg_fps_flw_s2_p50 | VICReg | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #47 | — | — | — | `wait` | — | — |
| 207 | g3_vicreg_fps_flw_s3_p50 | VICReg | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #71 | — | — | — | `wait` | — | — |
| 208 | g3_vicreg_fps_coco_s1_p50 | VICReg | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #68 | — | — | — | `wait` | — | — |
| 209 | g3_vicreg_fps_coco_s2_p50 | VICReg | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #92 | — | — | — | `wait` | — | — |
| 210 | g3_vicreg_fps_coco_s3_p50 | VICReg | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #116 | — | — | — | `wait` | — | — |
| 211 | g3_vicreg_fps_ham_s1_p50 | VICReg | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #140 | — | — | — | `wait` | — | — |
| 212 | g3_vicreg_fps_ham_s2_p50 | VICReg | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #164 | — | — | — | `wait` | — | — |
| 213 | g3_vicreg_fps_ham_s3_p50 | VICReg | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #188 | — | — | — | `wait` | — | — |
| 214 | g3_vicreg_fps_food_s1_p50 | VICReg | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #95 | — | — | — | `wait` | — | — |
| 215 | g3_vicreg_fps_food_s2_p50 | VICReg | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #119 | — | — | — | `wait` | — | — |
| 216 | g3_vicreg_fps_food_s3_p50 | VICReg | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #143 | — | — | — | `wait` | — | — |
| 217 | g3_ae_fps_reggs_s1_p50 | growth (AE) | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #208 | — | — | — | `wait` | — | — |
| 218 | g3_ae_fps_reggs_s2_p50 | growth (AE) | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #232 | — | — | — | `wait` | — | — |
| 219 | g3_ae_fps_reggs_s3_p50 | growth (AE) | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #256 | — | — | — | `wait` | — | — |
| 220 | g3_ae_fps_rlarv_s1_p50 | growth (AE) | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #163 | — | — | — | `wait` | — | — |
| 221 | g3_ae_fps_rlarv_s2_p50 | growth (AE) | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #187 | — | — | — | `wait` | — | — |
| 222 | g3_ae_fps_rlarv_s3_p50 | growth (AE) | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #211 | — | — | — | `wait` | — | — |
| 223 | g3_ae_fps_rprot_s1_p50 | growth (AE) | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #280 | — | — | — | `wait` | — | — |
| 224 | g3_ae_fps_rprot_s2_p50 | growth (AE) | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #304 | — | — | — | `wait` | — | — |
| 225 | g3_ae_fps_rprot_s3_p50 | growth (AE) | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #328 | — | — | — | `wait` | — | — |
| 226 | g3_simclr_fps_reggs_s1_p50 | SimCLR | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #210 | — | — | — | `wait` | — | — |
| 227 | g3_simclr_fps_reggs_s2_p50 | SimCLR | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #234 | — | — | — | `wait` | — | — |
| 228 | g3_simclr_fps_reggs_s3_p50 | SimCLR | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #258 | — | — | — | `wait` | — | — |
| 229 | g3_simclr_fps_rlarv_s1_p50 | SimCLR | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #165 | — | — | — | `wait` | — | — |
| 230 | g3_simclr_fps_rlarv_s2_p50 | SimCLR | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #189 | — | — | — | `wait` | — | — |
| 231 | g3_simclr_fps_rlarv_s3_p50 | SimCLR | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #213 | — | — | — | `wait` | — | — |
| 232 | g3_simclr_fps_rprot_s1_p50 | SimCLR | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #282 | — | — | — | `wait` | — | — |
| 233 | g3_simclr_fps_rprot_s2_p50 | SimCLR | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #306 | — | — | — | `wait` | — | — |
| 234 | g3_simclr_fps_rprot_s3_p50 | SimCLR | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #330 | — | — | — | `wait` | — | — |
| 235 | g3_vicreg_fps_reggs_s1_p50 | VICReg | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #212 | — | — | — | `wait` | — | — |
| 236 | g3_vicreg_fps_reggs_s2_p50 | VICReg | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #236 | — | — | — | `wait` | — | — |
| 237 | g3_vicreg_fps_reggs_s3_p50 | VICReg | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #260 | — | — | — | `wait` | — | — |
| 238 | g3_vicreg_fps_rlarv_s1_p50 | VICReg | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #167 | — | — | — | `wait` | — | — |
| 239 | g3_vicreg_fps_rlarv_s2_p50 | VICReg | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #191 | — | — | — | `wait` | — | — |
| 240 | g3_vicreg_fps_rlarv_s3_p50 | VICReg | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #215 | — | — | — | `wait` | — | — |
| 241 | g3_vicreg_fps_rprot_s1_p50 | VICReg | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #284 | — | — | — | `wait` | — | — |
| 242 | g3_vicreg_fps_rprot_s2_p50 | VICReg | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #308 | — | — | — | `wait` | — | — |
| 243 | g3_vicreg_fps_rprot_s3_p50 | VICReg | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #332 | — | — | — | `wait` | — | — |

Subtotal: 81

### 50% — Rede 4 (Inception 2×2, final 1024) — 81 runs

_grupo A → B > método (AE → SimCLR → VICReg) > base > split_

| # | run | método | init | base | split | tag | máquina/GPU | camada atual | início | fim | SVM | κ val (camada final) | κ test (camada final) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 244 | g4_ae_fps_aid_s1_p50 | growth (AE) | SPiFiL (FPS) | aid | 1 | `doing` | athena GPU 3 | L3 congelada | 04:18:59 | — | `wait` | — | — |
| 245 | g4_ae_fps_aid_s2_p50 | growth (AE) | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #22 | — | — | — | `wait` | — | — |
| 246 | g4_ae_fps_aid_s3_p50 | growth (AE) | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #46 | — | — | — | `wait` | — | — |
| 247 | g4_ae_fps_rs_s1_p50 | growth (AE) | SPiFiL (FPS) | remote_sensing | 1 | `done` | athena GPU 2 | L6 solta (_ft) | 17:52:52 | 22:25:56 | `done` | 0.5001 (L6_ft) | 0.4988 |
| 248 | g4_ae_fps_rs_s2_p50 | growth (AE) | SPiFiL (FPS) | remote_sensing | 2 | `done` | athena GPU 2 | L6 solta (_ft) | 10:20:46 | 20:52:03 | `done` | 0.5196 (L6_ft) | 0.4521 |
| 249 | g4_ae_fps_rs_s3_p50 | growth (AE) | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #1 | — | — | — | `wait` | — | — |
| 250 | g4_ae_fps_flw_s1_p50 | growth (AE) | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #25 | — | — | — | `wait` | — | — |
| 251 | g4_ae_fps_flw_s2_p50 | growth (AE) | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #49 | — | — | — | `wait` | — | — |
| 252 | g4_ae_fps_flw_s3_p50 | growth (AE) | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #73 | — | — | — | `wait` | — | — |
| 253 | g4_ae_fps_coco_s1_p50 | growth (AE) | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #70 | — | — | — | `wait` | — | — |
| 254 | g4_ae_fps_coco_s2_p50 | growth (AE) | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #94 | — | — | — | `wait` | — | — |
| 255 | g4_ae_fps_coco_s3_p50 | growth (AE) | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #118 | — | — | — | `wait` | — | — |
| 256 | g4_ae_fps_ham_s1_p50 | growth (AE) | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #142 | — | — | — | `wait` | — | — |
| 257 | g4_ae_fps_ham_s2_p50 | growth (AE) | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #166 | — | — | — | `wait` | — | — |
| 258 | g4_ae_fps_ham_s3_p50 | growth (AE) | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #190 | — | — | — | `wait` | — | — |
| 259 | g4_ae_fps_food_s1_p50 | growth (AE) | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #97 | — | — | — | `wait` | — | — |
| 260 | g4_ae_fps_food_s2_p50 | growth (AE) | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #121 | — | — | — | `wait` | — | — |
| 261 | g4_ae_fps_food_s3_p50 | growth (AE) | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #145 | — | — | — | `wait` | — | — |
| 262 | g4_simclr_fps_aid_s1_p50 | SimCLR | SPiFiL (FPS) | aid | 1 | `doing` | athena GPU 3 | L1 solta (_ft) | 09:06:16 | — | `wait` | — | — |
| 263 | g4_simclr_fps_aid_s2_p50 | SimCLR | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #24 | — | — | — | `wait` | — | — |
| 264 | g4_simclr_fps_aid_s3_p50 | SimCLR | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #48 | — | — | — | `wait` | — | — |
| 265 | g4_simclr_fps_rs_s1_p50 | SimCLR | SPiFiL (FPS) | remote_sensing | 1 | `done` | athena GPU 2 | L6 solta (_ft) | 18:03:42 | 22:14:15 | `done` | 0.4241 (L6_ft) | 0.5129 |
| 266 | g4_simclr_fps_rs_s2_p50 | SimCLR | SPiFiL (FPS) | remote_sensing | 2 | `done` | athena GPU 3 | L6 solta (_ft) | 13:22:55 | 21:25:30 | `done` | 0.5312 (L6_ft) | 0.4549 |
| 267 | g4_simclr_fps_rs_s3_p50 | SimCLR | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #3 | — | — | — | `wait` | — | — |
| 268 | g4_simclr_fps_flw_s1_p50 | SimCLR | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #27 | — | — | — | `wait` | — | — |
| 269 | g4_simclr_fps_flw_s2_p50 | SimCLR | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #51 | — | — | — | `wait` | — | — |
| 270 | g4_simclr_fps_flw_s3_p50 | SimCLR | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #75 | — | — | — | `wait` | — | — |
| 271 | g4_simclr_fps_coco_s1_p50 | SimCLR | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #72 | — | — | — | `wait` | — | — |
| 272 | g4_simclr_fps_coco_s2_p50 | SimCLR | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #96 | — | — | — | `wait` | — | — |
| 273 | g4_simclr_fps_coco_s3_p50 | SimCLR | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #120 | — | — | — | `wait` | — | — |
| 274 | g4_simclr_fps_ham_s1_p50 | SimCLR | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #144 | — | — | — | `wait` | — | — |
| 275 | g4_simclr_fps_ham_s2_p50 | SimCLR | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #168 | — | — | — | `wait` | — | — |
| 276 | g4_simclr_fps_ham_s3_p50 | SimCLR | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #192 | — | — | — | `wait` | — | — |
| 277 | g4_simclr_fps_food_s1_p50 | SimCLR | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #99 | — | — | — | `wait` | — | — |
| 278 | g4_simclr_fps_food_s2_p50 | SimCLR | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #123 | — | — | — | `wait` | — | — |
| 279 | g4_simclr_fps_food_s3_p50 | SimCLR | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #147 | — | — | — | `wait` | — | — |
| 280 | g4_vicreg_fps_aid_s1_p50 | VICReg | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #2 | — | — | — | `wait` | — | — |
| 281 | g4_vicreg_fps_aid_s2_p50 | VICReg | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #26 | — | — | — | `wait` | — | — |
| 282 | g4_vicreg_fps_aid_s3_p50 | VICReg | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #50 | — | — | — | `wait` | — | — |
| 283 | g4_vicreg_fps_rs_s1_p50 | VICReg | SPiFiL (FPS) | remote_sensing | 1 | `done` | athena GPU 2 | L6 solta (_ft) | 18:21:04 | 21:57:03 | `done` | 0.4930 (L6_ft) | 0.5109 |
| 284 | g4_vicreg_fps_rs_s2_p50 | VICReg | SPiFiL (FPS) | remote_sensing | 2 | `done` | athena GPU 3 | L6 solta (_ft) | 10:40:16 | 15:55:17 | `done` | 0.5387 (L6_ft) | 0.4624 |
| 285 | g4_vicreg_fps_rs_s3_p50 | VICReg | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #5 | — | — | — | `wait` | — | — |
| 286 | g4_vicreg_fps_flw_s1_p50 | VICReg | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #29 | — | — | — | `wait` | — | — |
| 287 | g4_vicreg_fps_flw_s2_p50 | VICReg | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #53 | — | — | — | `wait` | — | — |
| 288 | g4_vicreg_fps_flw_s3_p50 | VICReg | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #77 | — | — | — | `wait` | — | — |
| 289 | g4_vicreg_fps_coco_s1_p50 | VICReg | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #74 | — | — | — | `wait` | — | — |
| 290 | g4_vicreg_fps_coco_s2_p50 | VICReg | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #98 | — | — | — | `wait` | — | — |
| 291 | g4_vicreg_fps_coco_s3_p50 | VICReg | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #122 | — | — | — | `wait` | — | — |
| 292 | g4_vicreg_fps_ham_s1_p50 | VICReg | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #146 | — | — | — | `wait` | — | — |
| 293 | g4_vicreg_fps_ham_s2_p50 | VICReg | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #170 | — | — | — | `wait` | — | — |
| 294 | g4_vicreg_fps_ham_s3_p50 | VICReg | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #194 | — | — | — | `wait` | — | — |
| 295 | g4_vicreg_fps_food_s1_p50 | VICReg | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #101 | — | — | — | `wait` | — | — |
| 296 | g4_vicreg_fps_food_s2_p50 | VICReg | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #125 | — | — | — | `wait` | — | — |
| 297 | g4_vicreg_fps_food_s3_p50 | VICReg | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #149 | — | — | — | `wait` | — | — |
| 298 | g4_ae_fps_reggs_s1_p50 | growth (AE) | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #214 | — | — | — | `wait` | — | — |
| 299 | g4_ae_fps_reggs_s2_p50 | growth (AE) | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #238 | — | — | — | `wait` | — | — |
| 300 | g4_ae_fps_reggs_s3_p50 | growth (AE) | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #262 | — | — | — | `wait` | — | — |
| 301 | g4_ae_fps_rlarv_s1_p50 | growth (AE) | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #169 | — | — | — | `wait` | — | — |
| 302 | g4_ae_fps_rlarv_s2_p50 | growth (AE) | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #193 | — | — | — | `wait` | — | — |
| 303 | g4_ae_fps_rlarv_s3_p50 | growth (AE) | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #217 | — | — | — | `wait` | — | — |
| 304 | g4_ae_fps_rprot_s1_p50 | growth (AE) | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #286 | — | — | — | `wait` | — | — |
| 305 | g4_ae_fps_rprot_s2_p50 | growth (AE) | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #310 | — | — | — | `wait` | — | — |
| 306 | g4_ae_fps_rprot_s3_p50 | growth (AE) | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #334 | — | — | — | `wait` | — | — |
| 307 | g4_simclr_fps_reggs_s1_p50 | SimCLR | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #216 | — | — | — | `wait` | — | — |
| 308 | g4_simclr_fps_reggs_s2_p50 | SimCLR | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #240 | — | — | — | `wait` | — | — |
| 309 | g4_simclr_fps_reggs_s3_p50 | SimCLR | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #264 | — | — | — | `wait` | — | — |
| 310 | g4_simclr_fps_rlarv_s1_p50 | SimCLR | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #171 | — | — | — | `wait` | — | — |
| 311 | g4_simclr_fps_rlarv_s2_p50 | SimCLR | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #195 | — | — | — | `wait` | — | — |
| 312 | g4_simclr_fps_rlarv_s3_p50 | SimCLR | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #219 | — | — | — | `wait` | — | — |
| 313 | g4_simclr_fps_rprot_s1_p50 | SimCLR | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #288 | — | — | — | `wait` | — | — |
| 314 | g4_simclr_fps_rprot_s2_p50 | SimCLR | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #312 | — | — | — | `wait` | — | — |
| 315 | g4_simclr_fps_rprot_s3_p50 | SimCLR | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #336 | — | — | — | `wait` | — | — |
| 316 | g4_vicreg_fps_reggs_s1_p50 | VICReg | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #218 | — | — | — | `wait` | — | — |
| 317 | g4_vicreg_fps_reggs_s2_p50 | VICReg | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #242 | — | — | — | `wait` | — | — |
| 318 | g4_vicreg_fps_reggs_s3_p50 | VICReg | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #266 | — | — | — | `wait` | — | — |
| 319 | g4_vicreg_fps_rlarv_s1_p50 | VICReg | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #173 | — | — | — | `wait` | — | — |
| 320 | g4_vicreg_fps_rlarv_s2_p50 | VICReg | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #197 | — | — | — | `wait` | — | — |
| 321 | g4_vicreg_fps_rlarv_s3_p50 | VICReg | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #221 | — | — | — | `wait` | — | — |
| 322 | g4_vicreg_fps_rprot_s1_p50 | VICReg | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #290 | — | — | — | `wait` | — | — |
| 323 | g4_vicreg_fps_rprot_s2_p50 | VICReg | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #314 | — | — | — | `wait` | — | — |
| 324 | g4_vicreg_fps_rprot_s3_p50 | VICReg | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #338 | — | — | — | `wait` | — | — |

Subtotal: 81

### 100% — Rede 2 (D=2048) — 81 runs

_grupo A → B > método (AE → SimCLR → VICReg) > base > split_

| # | run | método | init | base | split | tag | máquina/GPU | camada atual | início | fim | SVM | κ val (camada final) | κ test (camada final) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 325 | g2_ae_fps_aid_s1_p100 | growth (AE) | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #52 | — | — | — | `wait` | — | — |
| 326 | g2_ae_fps_aid_s2_p100 | growth (AE) | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #76 | — | — | — | `wait` | — | — |
| 327 | g2_ae_fps_aid_s3_p100 | growth (AE) | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #100 | — | — | — | `wait` | — | — |
| 328 | g2_ae_fps_rs_s1_p100 | growth (AE) | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #31 | — | — | — | `wait` | — | — |
| 329 | g2_ae_fps_rs_s2_p100 | growth (AE) | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #55 | — | — | — | `wait` | — | — |
| 330 | g2_ae_fps_rs_s3_p100 | growth (AE) | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #79 | — | — | — | `wait` | — | — |
| 331 | g2_ae_fps_flw_s1_p100 | growth (AE) | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #103 | — | — | — | `wait` | — | — |
| 332 | g2_ae_fps_flw_s2_p100 | growth (AE) | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #127 | — | — | — | `wait` | — | — |
| 333 | g2_ae_fps_flw_s3_p100 | growth (AE) | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #151 | — | — | — | `wait` | — | — |
| 334 | g2_ae_fps_coco_s1_p100 | growth (AE) | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #124 | — | — | — | `wait` | — | — |
| 335 | g2_ae_fps_coco_s2_p100 | growth (AE) | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #148 | — | — | — | `wait` | — | — |
| 336 | g2_ae_fps_coco_s3_p100 | growth (AE) | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #172 | — | — | — | `wait` | — | — |
| 337 | g2_ae_fps_ham_s1_p100 | growth (AE) | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #196 | — | — | — | `wait` | — | — |
| 338 | g2_ae_fps_ham_s2_p100 | growth (AE) | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #220 | — | — | — | `wait` | — | — |
| 339 | g2_ae_fps_ham_s3_p100 | growth (AE) | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #244 | — | — | — | `wait` | — | — |
| 340 | g2_ae_fps_food_s1_p100 | growth (AE) | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #175 | — | — | — | `wait` | — | — |
| 341 | g2_ae_fps_food_s2_p100 | growth (AE) | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #199 | — | — | — | `wait` | — | — |
| 342 | g2_ae_fps_food_s3_p100 | growth (AE) | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #223 | — | — | — | `wait` | — | — |
| 343 | g2_simclr_fps_aid_s1_p100 | SimCLR | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #342 | — | — | — | `wait` | — | — |
| 344 | g2_simclr_fps_aid_s2_p100 | SimCLR | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #366 | — | — | — | `wait` | — | — |
| 345 | g2_simclr_fps_aid_s3_p100 | SimCLR | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #390 | — | — | — | `wait` | — | — |
| 346 | g2_simclr_fps_rs_s1_p100 | SimCLR | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #225 | — | — | — | `wait` | — | — |
| 347 | g2_simclr_fps_rs_s2_p100 | SimCLR | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #249 | — | — | — | `wait` | — | — |
| 348 | g2_simclr_fps_rs_s3_p100 | SimCLR | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #273 | — | — | — | `wait` | — | — |
| 349 | g2_simclr_fps_flw_s1_p100 | SimCLR | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #297 | — | — | — | `wait` | — | — |
| 350 | g2_simclr_fps_flw_s2_p100 | SimCLR | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #321 | — | — | — | `wait` | — | — |
| 351 | g2_simclr_fps_flw_s3_p100 | SimCLR | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #345 | — | — | — | `wait` | — | — |
| 352 | g2_simclr_fps_coco_s1_p100 | SimCLR | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #414 | — | — | — | `wait` | — | — |
| 353 | g2_simclr_fps_coco_s2_p100 | SimCLR | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #438 | — | — | — | `wait` | — | — |
| 354 | g2_simclr_fps_coco_s3_p100 | SimCLR | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #462 | — | — | — | `wait` | — | — |
| 355 | g2_simclr_fps_ham_s1_p100 | SimCLR | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #486 | — | — | — | `wait` | — | — |
| 356 | g2_simclr_fps_ham_s2_p100 | SimCLR | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #510 | — | — | — | `wait` | — | — |
| 357 | g2_simclr_fps_ham_s3_p100 | SimCLR | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #534 | — | — | — | `wait` | — | — |
| 358 | g2_simclr_fps_food_s1_p100 | SimCLR | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #369 | — | — | — | `wait` | — | — |
| 359 | g2_simclr_fps_food_s2_p100 | SimCLR | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #393 | — | — | — | `wait` | — | — |
| 360 | g2_simclr_fps_food_s3_p100 | SimCLR | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #417 | — | — | — | `wait` | — | — |
| 361 | g2_vicreg_fps_aid_s1_p100 | VICReg | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #344 | — | — | — | `wait` | — | — |
| 362 | g2_vicreg_fps_aid_s2_p100 | VICReg | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #368 | — | — | — | `wait` | — | — |
| 363 | g2_vicreg_fps_aid_s3_p100 | VICReg | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #392 | — | — | — | `wait` | — | — |
| 364 | g2_vicreg_fps_rs_s1_p100 | VICReg | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #227 | — | — | — | `wait` | — | — |
| 365 | g2_vicreg_fps_rs_s2_p100 | VICReg | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #251 | — | — | — | `wait` | — | — |
| 366 | g2_vicreg_fps_rs_s3_p100 | VICReg | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #275 | — | — | — | `wait` | — | — |
| 367 | g2_vicreg_fps_flw_s1_p100 | VICReg | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #299 | — | — | — | `wait` | — | — |
| 368 | g2_vicreg_fps_flw_s2_p100 | VICReg | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #323 | — | — | — | `wait` | — | — |
| 369 | g2_vicreg_fps_flw_s3_p100 | VICReg | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #347 | — | — | — | `wait` | — | — |
| 370 | g2_vicreg_fps_coco_s1_p100 | VICReg | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #416 | — | — | — | `wait` | — | — |
| 371 | g2_vicreg_fps_coco_s2_p100 | VICReg | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #440 | — | — | — | `wait` | — | — |
| 372 | g2_vicreg_fps_coco_s3_p100 | VICReg | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #464 | — | — | — | `wait` | — | — |
| 373 | g2_vicreg_fps_ham_s1_p100 | VICReg | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #488 | — | — | — | `wait` | — | — |
| 374 | g2_vicreg_fps_ham_s2_p100 | VICReg | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #512 | — | — | — | `wait` | — | — |
| 375 | g2_vicreg_fps_ham_s3_p100 | VICReg | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #536 | — | — | — | `wait` | — | — |
| 376 | g2_vicreg_fps_food_s1_p100 | VICReg | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #371 | — | — | — | `wait` | — | — |
| 377 | g2_vicreg_fps_food_s2_p100 | VICReg | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #395 | — | — | — | `wait` | — | — |
| 378 | g2_vicreg_fps_food_s3_p100 | VICReg | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #419 | — | — | — | `wait` | — | — |
| 379 | g2_ae_fps_reggs_s1_p100 | growth (AE) | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #268 | — | — | — | `wait` | — | — |
| 380 | g2_ae_fps_reggs_s2_p100 | growth (AE) | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #292 | — | — | — | `wait` | — | — |
| 381 | g2_ae_fps_reggs_s3_p100 | growth (AE) | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #316 | — | — | — | `wait` | — | — |
| 382 | g2_ae_fps_rlarv_s1_p100 | growth (AE) | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #247 | — | — | — | `wait` | — | — |
| 383 | g2_ae_fps_rlarv_s2_p100 | growth (AE) | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #271 | — | — | — | `wait` | — | — |
| 384 | g2_ae_fps_rlarv_s3_p100 | growth (AE) | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #295 | — | — | — | `wait` | — | — |
| 385 | g2_ae_fps_rprot_s1_p100 | growth (AE) | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #340 | — | — | — | `wait` | — | — |
| 386 | g2_ae_fps_rprot_s2_p100 | growth (AE) | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #364 | — | — | — | `wait` | — | — |
| 387 | g2_ae_fps_rprot_s3_p100 | growth (AE) | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #388 | — | — | — | `wait` | — | — |
| 388 | g2_simclr_fps_reggs_s1_p100 | SimCLR | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #558 | — | — | — | `wait` | — | — |
| 389 | g2_simclr_fps_reggs_s2_p100 | SimCLR | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #582 | — | — | — | `wait` | — | — |
| 390 | g2_simclr_fps_reggs_s3_p100 | SimCLR | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #606 | — | — | — | `wait` | — | — |
| 391 | g2_simclr_fps_rlarv_s1_p100 | SimCLR | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #441 | — | — | — | `wait` | — | — |
| 392 | g2_simclr_fps_rlarv_s2_p100 | SimCLR | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #465 | — | — | — | `wait` | — | — |
| 393 | g2_simclr_fps_rlarv_s3_p100 | SimCLR | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #489 | — | — | — | `wait` | — | — |
| 394 | g2_simclr_fps_rprot_s1_p100 | SimCLR | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #630 | — | — | — | `wait` | — | — |
| 395 | g2_simclr_fps_rprot_s2_p100 | SimCLR | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #654 | — | — | — | `wait` | — | — |
| 396 | g2_simclr_fps_rprot_s3_p100 | SimCLR | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #678 | — | — | — | `wait` | — | — |
| 397 | g2_vicreg_fps_reggs_s1_p100 | VICReg | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #560 | — | — | — | `wait` | — | — |
| 398 | g2_vicreg_fps_reggs_s2_p100 | VICReg | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #584 | — | — | — | `wait` | — | — |
| 399 | g2_vicreg_fps_reggs_s3_p100 | VICReg | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #608 | — | — | — | `wait` | — | — |
| 400 | g2_vicreg_fps_rlarv_s1_p100 | VICReg | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #443 | — | — | — | `wait` | — | — |
| 401 | g2_vicreg_fps_rlarv_s2_p100 | VICReg | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #467 | — | — | — | `wait` | — | — |
| 402 | g2_vicreg_fps_rlarv_s3_p100 | VICReg | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #491 | — | — | — | `wait` | — | — |
| 403 | g2_vicreg_fps_rprot_s1_p100 | VICReg | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #632 | — | — | — | `wait` | — | — |
| 404 | g2_vicreg_fps_rprot_s2_p100 | VICReg | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #656 | — | — | — | `wait` | — | — |
| 405 | g2_vicreg_fps_rprot_s3_p100 | VICReg | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #680 | — | — | — | `wait` | — | — |

Subtotal: 81

### 100% — Rede 1 (compacta 4×4, final 1024) — 81 runs

_grupo A → B > método (AE → SimCLR → VICReg) > base > split_

| # | run | método | init | base | split | tag | máquina/GPU | camada atual | início | fim | SVM | κ val (camada final) | κ test (camada final) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 406 | g1_ae_fps_aid_s1_p100 | growth (AE) | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #346 | — | — | — | `wait` | — | — |
| 407 | g1_ae_fps_aid_s2_p100 | growth (AE) | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #370 | — | — | — | `wait` | — | — |
| 408 | g1_ae_fps_aid_s3_p100 | growth (AE) | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #394 | — | — | — | `wait` | — | — |
| 409 | g1_ae_fps_rs_s1_p100 | growth (AE) | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #229 | — | — | — | `wait` | — | — |
| 410 | g1_ae_fps_rs_s2_p100 | growth (AE) | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #253 | — | — | — | `wait` | — | — |
| 411 | g1_ae_fps_rs_s3_p100 | growth (AE) | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #277 | — | — | — | `wait` | — | — |
| 412 | g1_ae_fps_flw_s1_p100 | growth (AE) | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #301 | — | — | — | `wait` | — | — |
| 413 | g1_ae_fps_flw_s2_p100 | growth (AE) | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #325 | — | — | — | `wait` | — | — |
| 414 | g1_ae_fps_flw_s3_p100 | growth (AE) | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #349 | — | — | — | `wait` | — | — |
| 415 | g1_ae_fps_coco_s1_p100 | growth (AE) | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #418 | — | — | — | `wait` | — | — |
| 416 | g1_ae_fps_coco_s2_p100 | growth (AE) | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #442 | — | — | — | `wait` | — | — |
| 417 | g1_ae_fps_coco_s3_p100 | growth (AE) | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #466 | — | — | — | `wait` | — | — |
| 418 | g1_ae_fps_ham_s1_p100 | growth (AE) | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #490 | — | — | — | `wait` | — | — |
| 419 | g1_ae_fps_ham_s2_p100 | growth (AE) | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #514 | — | — | — | `wait` | — | — |
| 420 | g1_ae_fps_ham_s3_p100 | growth (AE) | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #538 | — | — | — | `wait` | — | — |
| 421 | g1_ae_fps_food_s1_p100 | growth (AE) | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #373 | — | — | — | `wait` | — | — |
| 422 | g1_ae_fps_food_s2_p100 | growth (AE) | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #397 | — | — | — | `wait` | — | — |
| 423 | g1_ae_fps_food_s3_p100 | growth (AE) | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #421 | — | — | — | `wait` | — | — |
| 424 | g1_simclr_fps_aid_s1_p100 | SimCLR | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #348 | — | — | — | `wait` | — | — |
| 425 | g1_simclr_fps_aid_s2_p100 | SimCLR | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #372 | — | — | — | `wait` | — | — |
| 426 | g1_simclr_fps_aid_s3_p100 | SimCLR | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #396 | — | — | — | `wait` | — | — |
| 427 | g1_simclr_fps_rs_s1_p100 | SimCLR | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #231 | — | — | — | `wait` | — | — |
| 428 | g1_simclr_fps_rs_s2_p100 | SimCLR | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #255 | — | — | — | `wait` | — | — |
| 429 | g1_simclr_fps_rs_s3_p100 | SimCLR | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #279 | — | — | — | `wait` | — | — |
| 430 | g1_simclr_fps_flw_s1_p100 | SimCLR | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #303 | — | — | — | `wait` | — | — |
| 431 | g1_simclr_fps_flw_s2_p100 | SimCLR | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #327 | — | — | — | `wait` | — | — |
| 432 | g1_simclr_fps_flw_s3_p100 | SimCLR | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #351 | — | — | — | `wait` | — | — |
| 433 | g1_simclr_fps_coco_s1_p100 | SimCLR | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #420 | — | — | — | `wait` | — | — |
| 434 | g1_simclr_fps_coco_s2_p100 | SimCLR | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #444 | — | — | — | `wait` | — | — |
| 435 | g1_simclr_fps_coco_s3_p100 | SimCLR | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #468 | — | — | — | `wait` | — | — |
| 436 | g1_simclr_fps_ham_s1_p100 | SimCLR | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #492 | — | — | — | `wait` | — | — |
| 437 | g1_simclr_fps_ham_s2_p100 | SimCLR | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #516 | — | — | — | `wait` | — | — |
| 438 | g1_simclr_fps_ham_s3_p100 | SimCLR | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #540 | — | — | — | `wait` | — | — |
| 439 | g1_simclr_fps_food_s1_p100 | SimCLR | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #375 | — | — | — | `wait` | — | — |
| 440 | g1_simclr_fps_food_s2_p100 | SimCLR | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #399 | — | — | — | `wait` | — | — |
| 441 | g1_simclr_fps_food_s3_p100 | SimCLR | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #423 | — | — | — | `wait` | — | — |
| 442 | g1_vicreg_fps_aid_s1_p100 | VICReg | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #350 | — | — | — | `wait` | — | — |
| 443 | g1_vicreg_fps_aid_s2_p100 | VICReg | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #374 | — | — | — | `wait` | — | — |
| 444 | g1_vicreg_fps_aid_s3_p100 | VICReg | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #398 | — | — | — | `wait` | — | — |
| 445 | g1_vicreg_fps_rs_s1_p100 | VICReg | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #233 | — | — | — | `wait` | — | — |
| 446 | g1_vicreg_fps_rs_s2_p100 | VICReg | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #257 | — | — | — | `wait` | — | — |
| 447 | g1_vicreg_fps_rs_s3_p100 | VICReg | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #281 | — | — | — | `wait` | — | — |
| 448 | g1_vicreg_fps_flw_s1_p100 | VICReg | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #305 | — | — | — | `wait` | — | — |
| 449 | g1_vicreg_fps_flw_s2_p100 | VICReg | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #329 | — | — | — | `wait` | — | — |
| 450 | g1_vicreg_fps_flw_s3_p100 | VICReg | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #353 | — | — | — | `wait` | — | — |
| 451 | g1_vicreg_fps_coco_s1_p100 | VICReg | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #422 | — | — | — | `wait` | — | — |
| 452 | g1_vicreg_fps_coco_s2_p100 | VICReg | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #446 | — | — | — | `wait` | — | — |
| 453 | g1_vicreg_fps_coco_s3_p100 | VICReg | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #470 | — | — | — | `wait` | — | — |
| 454 | g1_vicreg_fps_ham_s1_p100 | VICReg | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #494 | — | — | — | `wait` | — | — |
| 455 | g1_vicreg_fps_ham_s2_p100 | VICReg | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #518 | — | — | — | `wait` | — | — |
| 456 | g1_vicreg_fps_ham_s3_p100 | VICReg | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #542 | — | — | — | `wait` | — | — |
| 457 | g1_vicreg_fps_food_s1_p100 | VICReg | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #377 | — | — | — | `wait` | — | — |
| 458 | g1_vicreg_fps_food_s2_p100 | VICReg | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #401 | — | — | — | `wait` | — | — |
| 459 | g1_vicreg_fps_food_s3_p100 | VICReg | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #425 | — | — | — | `wait` | — | — |
| 460 | g1_ae_fps_reggs_s1_p100 | growth (AE) | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #562 | — | — | — | `wait` | — | — |
| 461 | g1_ae_fps_reggs_s2_p100 | growth (AE) | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #586 | — | — | — | `wait` | — | — |
| 462 | g1_ae_fps_reggs_s3_p100 | growth (AE) | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #610 | — | — | — | `wait` | — | — |
| 463 | g1_ae_fps_rlarv_s1_p100 | growth (AE) | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #445 | — | — | — | `wait` | — | — |
| 464 | g1_ae_fps_rlarv_s2_p100 | growth (AE) | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #469 | — | — | — | `wait` | — | — |
| 465 | g1_ae_fps_rlarv_s3_p100 | growth (AE) | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #493 | — | — | — | `wait` | — | — |
| 466 | g1_ae_fps_rprot_s1_p100 | growth (AE) | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #634 | — | — | — | `wait` | — | — |
| 467 | g1_ae_fps_rprot_s2_p100 | growth (AE) | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #658 | — | — | — | `wait` | — | — |
| 468 | g1_ae_fps_rprot_s3_p100 | growth (AE) | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #682 | — | — | — | `wait` | — | — |
| 469 | g1_simclr_fps_reggs_s1_p100 | SimCLR | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #564 | — | — | — | `wait` | — | — |
| 470 | g1_simclr_fps_reggs_s2_p100 | SimCLR | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #588 | — | — | — | `wait` | — | — |
| 471 | g1_simclr_fps_reggs_s3_p100 | SimCLR | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #612 | — | — | — | `wait` | — | — |
| 472 | g1_simclr_fps_rlarv_s1_p100 | SimCLR | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #447 | — | — | — | `wait` | — | — |
| 473 | g1_simclr_fps_rlarv_s2_p100 | SimCLR | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #471 | — | — | — | `wait` | — | — |
| 474 | g1_simclr_fps_rlarv_s3_p100 | SimCLR | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #495 | — | — | — | `wait` | — | — |
| 475 | g1_simclr_fps_rprot_s1_p100 | SimCLR | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #636 | — | — | — | `wait` | — | — |
| 476 | g1_simclr_fps_rprot_s2_p100 | SimCLR | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #660 | — | — | — | `wait` | — | — |
| 477 | g1_simclr_fps_rprot_s3_p100 | SimCLR | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #684 | — | — | — | `wait` | — | — |
| 478 | g1_vicreg_fps_reggs_s1_p100 | VICReg | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #566 | — | — | — | `wait` | — | — |
| 479 | g1_vicreg_fps_reggs_s2_p100 | VICReg | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #590 | — | — | — | `wait` | — | — |
| 480 | g1_vicreg_fps_reggs_s3_p100 | VICReg | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #614 | — | — | — | `wait` | — | — |
| 481 | g1_vicreg_fps_rlarv_s1_p100 | VICReg | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #449 | — | — | — | `wait` | — | — |
| 482 | g1_vicreg_fps_rlarv_s2_p100 | VICReg | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #473 | — | — | — | `wait` | — | — |
| 483 | g1_vicreg_fps_rlarv_s3_p100 | VICReg | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #497 | — | — | — | `wait` | — | — |
| 484 | g1_vicreg_fps_rprot_s1_p100 | VICReg | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #638 | — | — | — | `wait` | — | — |
| 485 | g1_vicreg_fps_rprot_s2_p100 | VICReg | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #662 | — | — | — | `wait` | — | — |
| 486 | g1_vicreg_fps_rprot_s3_p100 | VICReg | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #686 | — | — | — | `wait` | — | — |

Subtotal: 81

### 100% — Rede 3 (GeM+cat, final 1024) — 81 runs

_grupo A → B > método (AE → SimCLR → VICReg) > base > split_

| # | run | método | init | base | split | tag | máquina/GPU | camada atual | início | fim | SVM | κ val (camada final) | κ test (camada final) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 487 | g3_ae_fps_aid_s1_p100 | growth (AE) | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #352 | — | — | — | `wait` | — | — |
| 488 | g3_ae_fps_aid_s2_p100 | growth (AE) | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #376 | — | — | — | `wait` | — | — |
| 489 | g3_ae_fps_aid_s3_p100 | growth (AE) | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #400 | — | — | — | `wait` | — | — |
| 490 | g3_ae_fps_rs_s1_p100 | growth (AE) | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #235 | — | — | — | `wait` | — | — |
| 491 | g3_ae_fps_rs_s2_p100 | growth (AE) | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #259 | — | — | — | `wait` | — | — |
| 492 | g3_ae_fps_rs_s3_p100 | growth (AE) | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #283 | — | — | — | `wait` | — | — |
| 493 | g3_ae_fps_flw_s1_p100 | growth (AE) | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #307 | — | — | — | `wait` | — | — |
| 494 | g3_ae_fps_flw_s2_p100 | growth (AE) | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #331 | — | — | — | `wait` | — | — |
| 495 | g3_ae_fps_flw_s3_p100 | growth (AE) | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #355 | — | — | — | `wait` | — | — |
| 496 | g3_ae_fps_coco_s1_p100 | growth (AE) | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #424 | — | — | — | `wait` | — | — |
| 497 | g3_ae_fps_coco_s2_p100 | growth (AE) | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #448 | — | — | — | `wait` | — | — |
| 498 | g3_ae_fps_coco_s3_p100 | growth (AE) | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #472 | — | — | — | `wait` | — | — |
| 499 | g3_ae_fps_ham_s1_p100 | growth (AE) | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #496 | — | — | — | `wait` | — | — |
| 500 | g3_ae_fps_ham_s2_p100 | growth (AE) | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #520 | — | — | — | `wait` | — | — |
| 501 | g3_ae_fps_ham_s3_p100 | growth (AE) | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #544 | — | — | — | `wait` | — | — |
| 502 | g3_ae_fps_food_s1_p100 | growth (AE) | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #379 | — | — | — | `wait` | — | — |
| 503 | g3_ae_fps_food_s2_p100 | growth (AE) | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #403 | — | — | — | `wait` | — | — |
| 504 | g3_ae_fps_food_s3_p100 | growth (AE) | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #427 | — | — | — | `wait` | — | — |
| 505 | g3_simclr_fps_aid_s1_p100 | SimCLR | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #354 | — | — | — | `wait` | — | — |
| 506 | g3_simclr_fps_aid_s2_p100 | SimCLR | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #378 | — | — | — | `wait` | — | — |
| 507 | g3_simclr_fps_aid_s3_p100 | SimCLR | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #402 | — | — | — | `wait` | — | — |
| 508 | g3_simclr_fps_rs_s1_p100 | SimCLR | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #237 | — | — | — | `wait` | — | — |
| 509 | g3_simclr_fps_rs_s2_p100 | SimCLR | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #261 | — | — | — | `wait` | — | — |
| 510 | g3_simclr_fps_rs_s3_p100 | SimCLR | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #285 | — | — | — | `wait` | — | — |
| 511 | g3_simclr_fps_flw_s1_p100 | SimCLR | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #309 | — | — | — | `wait` | — | — |
| 512 | g3_simclr_fps_flw_s2_p100 | SimCLR | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #333 | — | — | — | `wait` | — | — |
| 513 | g3_simclr_fps_flw_s3_p100 | SimCLR | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #357 | — | — | — | `wait` | — | — |
| 514 | g3_simclr_fps_coco_s1_p100 | SimCLR | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #426 | — | — | — | `wait` | — | — |
| 515 | g3_simclr_fps_coco_s2_p100 | SimCLR | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #450 | — | — | — | `wait` | — | — |
| 516 | g3_simclr_fps_coco_s3_p100 | SimCLR | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #474 | — | — | — | `wait` | — | — |
| 517 | g3_simclr_fps_ham_s1_p100 | SimCLR | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #498 | — | — | — | `wait` | — | — |
| 518 | g3_simclr_fps_ham_s2_p100 | SimCLR | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #522 | — | — | — | `wait` | — | — |
| 519 | g3_simclr_fps_ham_s3_p100 | SimCLR | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #546 | — | — | — | `wait` | — | — |
| 520 | g3_simclr_fps_food_s1_p100 | SimCLR | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #381 | — | — | — | `wait` | — | — |
| 521 | g3_simclr_fps_food_s2_p100 | SimCLR | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #405 | — | — | — | `wait` | — | — |
| 522 | g3_simclr_fps_food_s3_p100 | SimCLR | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #429 | — | — | — | `wait` | — | — |
| 523 | g3_vicreg_fps_aid_s1_p100 | VICReg | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #356 | — | — | — | `wait` | — | — |
| 524 | g3_vicreg_fps_aid_s2_p100 | VICReg | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #380 | — | — | — | `wait` | — | — |
| 525 | g3_vicreg_fps_aid_s3_p100 | VICReg | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #404 | — | — | — | `wait` | — | — |
| 526 | g3_vicreg_fps_rs_s1_p100 | VICReg | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #239 | — | — | — | `wait` | — | — |
| 527 | g3_vicreg_fps_rs_s2_p100 | VICReg | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #263 | — | — | — | `wait` | — | — |
| 528 | g3_vicreg_fps_rs_s3_p100 | VICReg | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #287 | — | — | — | `wait` | — | — |
| 529 | g3_vicreg_fps_flw_s1_p100 | VICReg | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #311 | — | — | — | `wait` | — | — |
| 530 | g3_vicreg_fps_flw_s2_p100 | VICReg | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #335 | — | — | — | `wait` | — | — |
| 531 | g3_vicreg_fps_flw_s3_p100 | VICReg | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #359 | — | — | — | `wait` | — | — |
| 532 | g3_vicreg_fps_coco_s1_p100 | VICReg | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #428 | — | — | — | `wait` | — | — |
| 533 | g3_vicreg_fps_coco_s2_p100 | VICReg | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #452 | — | — | — | `wait` | — | — |
| 534 | g3_vicreg_fps_coco_s3_p100 | VICReg | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #476 | — | — | — | `wait` | — | — |
| 535 | g3_vicreg_fps_ham_s1_p100 | VICReg | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #500 | — | — | — | `wait` | — | — |
| 536 | g3_vicreg_fps_ham_s2_p100 | VICReg | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #524 | — | — | — | `wait` | — | — |
| 537 | g3_vicreg_fps_ham_s3_p100 | VICReg | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #548 | — | — | — | `wait` | — | — |
| 538 | g3_vicreg_fps_food_s1_p100 | VICReg | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #383 | — | — | — | `wait` | — | — |
| 539 | g3_vicreg_fps_food_s2_p100 | VICReg | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #407 | — | — | — | `wait` | — | — |
| 540 | g3_vicreg_fps_food_s3_p100 | VICReg | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #431 | — | — | — | `wait` | — | — |
| 541 | g3_ae_fps_reggs_s1_p100 | growth (AE) | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #568 | — | — | — | `wait` | — | — |
| 542 | g3_ae_fps_reggs_s2_p100 | growth (AE) | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #592 | — | — | — | `wait` | — | — |
| 543 | g3_ae_fps_reggs_s3_p100 | growth (AE) | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #616 | — | — | — | `wait` | — | — |
| 544 | g3_ae_fps_rlarv_s1_p100 | growth (AE) | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #451 | — | — | — | `wait` | — | — |
| 545 | g3_ae_fps_rlarv_s2_p100 | growth (AE) | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #475 | — | — | — | `wait` | — | — |
| 546 | g3_ae_fps_rlarv_s3_p100 | growth (AE) | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #499 | — | — | — | `wait` | — | — |
| 547 | g3_ae_fps_rprot_s1_p100 | growth (AE) | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #640 | — | — | — | `wait` | — | — |
| 548 | g3_ae_fps_rprot_s2_p100 | growth (AE) | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #664 | — | — | — | `wait` | — | — |
| 549 | g3_ae_fps_rprot_s3_p100 | growth (AE) | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #688 | — | — | — | `wait` | — | — |
| 550 | g3_simclr_fps_reggs_s1_p100 | SimCLR | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #570 | — | — | — | `wait` | — | — |
| 551 | g3_simclr_fps_reggs_s2_p100 | SimCLR | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #594 | — | — | — | `wait` | — | — |
| 552 | g3_simclr_fps_reggs_s3_p100 | SimCLR | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #618 | — | — | — | `wait` | — | — |
| 553 | g3_simclr_fps_rlarv_s1_p100 | SimCLR | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #453 | — | — | — | `wait` | — | — |
| 554 | g3_simclr_fps_rlarv_s2_p100 | SimCLR | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #477 | — | — | — | `wait` | — | — |
| 555 | g3_simclr_fps_rlarv_s3_p100 | SimCLR | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #501 | — | — | — | `wait` | — | — |
| 556 | g3_simclr_fps_rprot_s1_p100 | SimCLR | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #642 | — | — | — | `wait` | — | — |
| 557 | g3_simclr_fps_rprot_s2_p100 | SimCLR | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #666 | — | — | — | `wait` | — | — |
| 558 | g3_simclr_fps_rprot_s3_p100 | SimCLR | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #690 | — | — | — | `wait` | — | — |
| 559 | g3_vicreg_fps_reggs_s1_p100 | VICReg | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #572 | — | — | — | `wait` | — | — |
| 560 | g3_vicreg_fps_reggs_s2_p100 | VICReg | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #596 | — | — | — | `wait` | — | — |
| 561 | g3_vicreg_fps_reggs_s3_p100 | VICReg | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #620 | — | — | — | `wait` | — | — |
| 562 | g3_vicreg_fps_rlarv_s1_p100 | VICReg | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #455 | — | — | — | `wait` | — | — |
| 563 | g3_vicreg_fps_rlarv_s2_p100 | VICReg | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #479 | — | — | — | `wait` | — | — |
| 564 | g3_vicreg_fps_rlarv_s3_p100 | VICReg | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #503 | — | — | — | `wait` | — | — |
| 565 | g3_vicreg_fps_rprot_s1_p100 | VICReg | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #644 | — | — | — | `wait` | — | — |
| 566 | g3_vicreg_fps_rprot_s2_p100 | VICReg | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #668 | — | — | — | `wait` | — | — |
| 567 | g3_vicreg_fps_rprot_s3_p100 | VICReg | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #692 | — | — | — | `wait` | — | — |

Subtotal: 81

### 100% — Rede 4 (Inception 2×2, final 1024) — 81 runs

_grupo A → B > método (AE → SimCLR → VICReg) > base > split_

| # | run | método | init | base | split | tag | máquina/GPU | camada atual | início | fim | SVM | κ val (camada final) | κ test (camada final) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 568 | g4_ae_fps_aid_s1_p100 | growth (AE) | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #358 | — | — | — | `wait` | — | — |
| 569 | g4_ae_fps_aid_s2_p100 | growth (AE) | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #382 | — | — | — | `wait` | — | — |
| 570 | g4_ae_fps_aid_s3_p100 | growth (AE) | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #406 | — | — | — | `wait` | — | — |
| 571 | g4_ae_fps_rs_s1_p100 | growth (AE) | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #241 | — | — | — | `wait` | — | — |
| 572 | g4_ae_fps_rs_s2_p100 | growth (AE) | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #265 | — | — | — | `wait` | — | — |
| 573 | g4_ae_fps_rs_s3_p100 | growth (AE) | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #289 | — | — | — | `wait` | — | — |
| 574 | g4_ae_fps_flw_s1_p100 | growth (AE) | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #313 | — | — | — | `wait` | — | — |
| 575 | g4_ae_fps_flw_s2_p100 | growth (AE) | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #337 | — | — | — | `wait` | — | — |
| 576 | g4_ae_fps_flw_s3_p100 | growth (AE) | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #361 | — | — | — | `wait` | — | — |
| 577 | g4_ae_fps_coco_s1_p100 | growth (AE) | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #430 | — | — | — | `wait` | — | — |
| 578 | g4_ae_fps_coco_s2_p100 | growth (AE) | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #454 | — | — | — | `wait` | — | — |
| 579 | g4_ae_fps_coco_s3_p100 | growth (AE) | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #478 | — | — | — | `wait` | — | — |
| 580 | g4_ae_fps_ham_s1_p100 | growth (AE) | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #502 | — | — | — | `wait` | — | — |
| 581 | g4_ae_fps_ham_s2_p100 | growth (AE) | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #526 | — | — | — | `wait` | — | — |
| 582 | g4_ae_fps_ham_s3_p100 | growth (AE) | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #550 | — | — | — | `wait` | — | — |
| 583 | g4_ae_fps_food_s1_p100 | growth (AE) | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #385 | — | — | — | `wait` | — | — |
| 584 | g4_ae_fps_food_s2_p100 | growth (AE) | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #409 | — | — | — | `wait` | — | — |
| 585 | g4_ae_fps_food_s3_p100 | growth (AE) | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #433 | — | — | — | `wait` | — | — |
| 586 | g4_simclr_fps_aid_s1_p100 | SimCLR | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #360 | — | — | — | `wait` | — | — |
| 587 | g4_simclr_fps_aid_s2_p100 | SimCLR | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #384 | — | — | — | `wait` | — | — |
| 588 | g4_simclr_fps_aid_s3_p100 | SimCLR | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #408 | — | — | — | `wait` | — | — |
| 589 | g4_simclr_fps_rs_s1_p100 | SimCLR | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #243 | — | — | — | `wait` | — | — |
| 590 | g4_simclr_fps_rs_s2_p100 | SimCLR | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #267 | — | — | — | `wait` | — | — |
| 591 | g4_simclr_fps_rs_s3_p100 | SimCLR | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #291 | — | — | — | `wait` | — | — |
| 592 | g4_simclr_fps_flw_s1_p100 | SimCLR | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #315 | — | — | — | `wait` | — | — |
| 593 | g4_simclr_fps_flw_s2_p100 | SimCLR | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #339 | — | — | — | `wait` | — | — |
| 594 | g4_simclr_fps_flw_s3_p100 | SimCLR | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #363 | — | — | — | `wait` | — | — |
| 595 | g4_simclr_fps_coco_s1_p100 | SimCLR | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #432 | — | — | — | `wait` | — | — |
| 596 | g4_simclr_fps_coco_s2_p100 | SimCLR | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #456 | — | — | — | `wait` | — | — |
| 597 | g4_simclr_fps_coco_s3_p100 | SimCLR | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #480 | — | — | — | `wait` | — | — |
| 598 | g4_simclr_fps_ham_s1_p100 | SimCLR | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #504 | — | — | — | `wait` | — | — |
| 599 | g4_simclr_fps_ham_s2_p100 | SimCLR | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #528 | — | — | — | `wait` | — | — |
| 600 | g4_simclr_fps_ham_s3_p100 | SimCLR | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #552 | — | — | — | `wait` | — | — |
| 601 | g4_simclr_fps_food_s1_p100 | SimCLR | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #387 | — | — | — | `wait` | — | — |
| 602 | g4_simclr_fps_food_s2_p100 | SimCLR | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #411 | — | — | — | `wait` | — | — |
| 603 | g4_simclr_fps_food_s3_p100 | SimCLR | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #435 | — | — | — | `wait` | — | — |
| 604 | g4_vicreg_fps_aid_s1_p100 | VICReg | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #362 | — | — | — | `wait` | — | — |
| 605 | g4_vicreg_fps_aid_s2_p100 | VICReg | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #386 | — | — | — | `wait` | — | — |
| 606 | g4_vicreg_fps_aid_s3_p100 | VICReg | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #410 | — | — | — | `wait` | — | — |
| 607 | g4_vicreg_fps_rs_s1_p100 | VICReg | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #245 | — | — | — | `wait` | — | — |
| 608 | g4_vicreg_fps_rs_s2_p100 | VICReg | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #269 | — | — | — | `wait` | — | — |
| 609 | g4_vicreg_fps_rs_s3_p100 | VICReg | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #293 | — | — | — | `wait` | — | — |
| 610 | g4_vicreg_fps_flw_s1_p100 | VICReg | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #317 | — | — | — | `wait` | — | — |
| 611 | g4_vicreg_fps_flw_s2_p100 | VICReg | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #341 | — | — | — | `wait` | — | — |
| 612 | g4_vicreg_fps_flw_s3_p100 | VICReg | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #365 | — | — | — | `wait` | — | — |
| 613 | g4_vicreg_fps_coco_s1_p100 | VICReg | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #434 | — | — | — | `wait` | — | — |
| 614 | g4_vicreg_fps_coco_s2_p100 | VICReg | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #458 | — | — | — | `wait` | — | — |
| 615 | g4_vicreg_fps_coco_s3_p100 | VICReg | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #482 | — | — | — | `wait` | — | — |
| 616 | g4_vicreg_fps_ham_s1_p100 | VICReg | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #506 | — | — | — | `wait` | — | — |
| 617 | g4_vicreg_fps_ham_s2_p100 | VICReg | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #530 | — | — | — | `wait` | — | — |
| 618 | g4_vicreg_fps_ham_s3_p100 | VICReg | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #554 | — | — | — | `wait` | — | — |
| 619 | g4_vicreg_fps_food_s1_p100 | VICReg | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #389 | — | — | — | `wait` | — | — |
| 620 | g4_vicreg_fps_food_s2_p100 | VICReg | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #413 | — | — | — | `wait` | — | — |
| 621 | g4_vicreg_fps_food_s3_p100 | VICReg | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #437 | — | — | — | `wait` | — | — |
| 622 | g4_ae_fps_reggs_s1_p100 | growth (AE) | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #574 | — | — | — | `wait` | — | — |
| 623 | g4_ae_fps_reggs_s2_p100 | growth (AE) | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #598 | — | — | — | `wait` | — | — |
| 624 | g4_ae_fps_reggs_s3_p100 | growth (AE) | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #622 | — | — | — | `wait` | — | — |
| 625 | g4_ae_fps_rlarv_s1_p100 | growth (AE) | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #457 | — | — | — | `wait` | — | — |
| 626 | g4_ae_fps_rlarv_s2_p100 | growth (AE) | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #481 | — | — | — | `wait` | — | — |
| 627 | g4_ae_fps_rlarv_s3_p100 | growth (AE) | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #505 | — | — | — | `wait` | — | — |
| 628 | g4_ae_fps_rprot_s1_p100 | growth (AE) | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #646 | — | — | — | `wait` | — | — |
| 629 | g4_ae_fps_rprot_s2_p100 | growth (AE) | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #670 | — | — | — | `wait` | — | — |
| 630 | g4_ae_fps_rprot_s3_p100 | growth (AE) | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #694 | — | — | — | `wait` | — | — |
| 631 | g4_simclr_fps_reggs_s1_p100 | SimCLR | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #576 | — | — | — | `wait` | — | — |
| 632 | g4_simclr_fps_reggs_s2_p100 | SimCLR | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #600 | — | — | — | `wait` | — | — |
| 633 | g4_simclr_fps_reggs_s3_p100 | SimCLR | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #624 | — | — | — | `wait` | — | — |
| 634 | g4_simclr_fps_rlarv_s1_p100 | SimCLR | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #459 | — | — | — | `wait` | — | — |
| 635 | g4_simclr_fps_rlarv_s2_p100 | SimCLR | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #483 | — | — | — | `wait` | — | — |
| 636 | g4_simclr_fps_rlarv_s3_p100 | SimCLR | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #507 | — | — | — | `wait` | — | — |
| 637 | g4_simclr_fps_rprot_s1_p100 | SimCLR | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #648 | — | — | — | `wait` | — | — |
| 638 | g4_simclr_fps_rprot_s2_p100 | SimCLR | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #672 | — | — | — | `wait` | — | — |
| 639 | g4_simclr_fps_rprot_s3_p100 | SimCLR | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #696 | — | — | — | `wait` | — | — |
| 640 | g4_vicreg_fps_reggs_s1_p100 | VICReg | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #578 | — | — | — | `wait` | — | — |
| 641 | g4_vicreg_fps_reggs_s2_p100 | VICReg | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #602 | — | — | — | `wait` | — | — |
| 642 | g4_vicreg_fps_reggs_s3_p100 | VICReg | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #626 | — | — | — | `wait` | — | — |
| 643 | g4_vicreg_fps_rlarv_s1_p100 | VICReg | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #461 | — | — | — | `wait` | — | — |
| 644 | g4_vicreg_fps_rlarv_s2_p100 | VICReg | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #485 | — | — | — | `wait` | — | — |
| 645 | g4_vicreg_fps_rlarv_s3_p100 | VICReg | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #509 | — | — | — | `wait` | — | — |
| 646 | g4_vicreg_fps_rprot_s1_p100 | VICReg | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #650 | — | — | — | `wait` | — | — |
| 647 | g4_vicreg_fps_rprot_s2_p100 | VICReg | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #674 | — | — | — | `wait` | — | — |
| 648 | g4_vicreg_fps_rprot_s3_p100 | VICReg | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #698 | — | — | — | `wait` | — | — |

Subtotal: 81

### 75% — Rede 2 (D=2048) — 81 runs

_depois de 50% e 100%; mesma ordem_

| # | run | método | init | base | split | tag | máquina/GPU | camada atual | início | fim | SVM | κ val (camada final) | κ test (camada final) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 649 | g2_ae_fps_aid_s1_p75 | growth (AE) | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #412 | — | — | — | `wait` | — | — |
| 650 | g2_ae_fps_aid_s2_p75 | growth (AE) | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #436 | — | — | — | `wait` | — | — |
| 651 | g2_ae_fps_aid_s3_p75 | growth (AE) | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #460 | — | — | — | `wait` | — | — |
| 652 | g2_ae_fps_rs_s1_p75 | growth (AE) | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #319 | — | — | — | `wait` | — | — |
| 653 | g2_ae_fps_rs_s2_p75 | growth (AE) | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #343 | — | — | — | `wait` | — | — |
| 654 | g2_ae_fps_rs_s3_p75 | growth (AE) | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #367 | — | — | — | `wait` | — | — |
| 655 | g2_ae_fps_flw_s1_p75 | growth (AE) | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #391 | — | — | — | `wait` | — | — |
| 656 | g2_ae_fps_flw_s2_p75 | growth (AE) | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #415 | — | — | — | `wait` | — | — |
| 657 | g2_ae_fps_flw_s3_p75 | growth (AE) | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #439 | — | — | — | `wait` | — | — |
| 658 | g2_ae_fps_coco_s1_p75 | growth (AE) | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #484 | — | — | — | `wait` | — | — |
| 659 | g2_ae_fps_coco_s2_p75 | growth (AE) | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #508 | — | — | — | `wait` | — | — |
| 660 | g2_ae_fps_coco_s3_p75 | growth (AE) | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #532 | — | — | — | `wait` | — | — |
| 661 | g2_ae_fps_ham_s1_p75 | growth (AE) | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #556 | — | — | — | `wait` | — | — |
| 662 | g2_ae_fps_ham_s2_p75 | growth (AE) | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #580 | — | — | — | `wait` | — | — |
| 663 | g2_ae_fps_ham_s3_p75 | growth (AE) | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #604 | — | — | — | `wait` | — | — |
| 664 | g2_ae_fps_food_s1_p75 | growth (AE) | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #463 | — | — | — | `wait` | — | — |
| 665 | g2_ae_fps_food_s2_p75 | growth (AE) | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #487 | — | — | — | `wait` | — | — |
| 666 | g2_ae_fps_food_s3_p75 | growth (AE) | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #511 | — | — | — | `wait` | — | — |
| 667 | g2_simclr_fps_aid_s1_p75 | SimCLR | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #702 | — | — | — | `wait` | — | — |
| 668 | g2_simclr_fps_aid_s2_p75 | SimCLR | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #726 | — | — | — | `wait` | — | — |
| 669 | g2_simclr_fps_aid_s3_p75 | SimCLR | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #750 | — | — | — | `wait` | — | — |
| 670 | g2_simclr_fps_rs_s1_p75 | SimCLR | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #513 | — | — | — | `wait` | — | — |
| 671 | g2_simclr_fps_rs_s2_p75 | SimCLR | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #537 | — | — | — | `wait` | — | — |
| 672 | g2_simclr_fps_rs_s3_p75 | SimCLR | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #561 | — | — | — | `wait` | — | — |
| 673 | g2_simclr_fps_flw_s1_p75 | SimCLR | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #585 | — | — | — | `wait` | — | — |
| 674 | g2_simclr_fps_flw_s2_p75 | SimCLR | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #607 | — | — | — | `wait` | — | — |
| 675 | g2_simclr_fps_flw_s3_p75 | SimCLR | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #629 | — | — | — | `wait` | — | — |
| 676 | g2_simclr_fps_coco_s1_p75 | SimCLR | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #772 | — | — | — | `wait` | — | — |
| 677 | g2_simclr_fps_coco_s2_p75 | SimCLR | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #788 | — | — | — | `wait` | — | — |
| 678 | g2_simclr_fps_coco_s3_p75 | SimCLR | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #799 | — | — | — | `wait` | — | — |
| 679 | g2_simclr_fps_ham_s1_p75 | SimCLR | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #810 | — | — | — | `wait` | — | — |
| 680 | g2_simclr_fps_ham_s2_p75 | SimCLR | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #821 | — | — | — | `wait` | — | — |
| 681 | g2_simclr_fps_ham_s3_p75 | SimCLR | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #832 | — | — | — | `wait` | — | — |
| 682 | g2_simclr_fps_food_s1_p75 | SimCLR | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #651 | — | — | — | `wait` | — | — |
| 683 | g2_simclr_fps_food_s2_p75 | SimCLR | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #673 | — | — | — | `wait` | — | — |
| 684 | g2_simclr_fps_food_s3_p75 | SimCLR | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #695 | — | — | — | `wait` | — | — |
| 685 | g2_vicreg_fps_aid_s1_p75 | VICReg | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #704 | — | — | — | `wait` | — | — |
| 686 | g2_vicreg_fps_aid_s2_p75 | VICReg | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #728 | — | — | — | `wait` | — | — |
| 687 | g2_vicreg_fps_aid_s3_p75 | VICReg | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #752 | — | — | — | `wait` | — | — |
| 688 | g2_vicreg_fps_rs_s1_p75 | VICReg | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #515 | — | — | — | `wait` | — | — |
| 689 | g2_vicreg_fps_rs_s2_p75 | VICReg | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #539 | — | — | — | `wait` | — | — |
| 690 | g2_vicreg_fps_rs_s3_p75 | VICReg | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #563 | — | — | — | `wait` | — | — |
| 691 | g2_vicreg_fps_flw_s1_p75 | VICReg | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #587 | — | — | — | `wait` | — | — |
| 692 | g2_vicreg_fps_flw_s2_p75 | VICReg | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #609 | — | — | — | `wait` | — | — |
| 693 | g2_vicreg_fps_flw_s3_p75 | VICReg | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #631 | — | — | — | `wait` | — | — |
| 694 | g2_vicreg_fps_coco_s1_p75 | VICReg | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #774 | — | — | — | `wait` | — | — |
| 695 | g2_vicreg_fps_coco_s2_p75 | VICReg | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #789 | — | — | — | `wait` | — | — |
| 696 | g2_vicreg_fps_coco_s3_p75 | VICReg | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #800 | — | — | — | `wait` | — | — |
| 697 | g2_vicreg_fps_ham_s1_p75 | VICReg | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #811 | — | — | — | `wait` | — | — |
| 698 | g2_vicreg_fps_ham_s2_p75 | VICReg | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #822 | — | — | — | `wait` | — | — |
| 699 | g2_vicreg_fps_ham_s3_p75 | VICReg | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #833 | — | — | — | `wait` | — | — |
| 700 | g2_vicreg_fps_food_s1_p75 | VICReg | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #653 | — | — | — | `wait` | — | — |
| 701 | g2_vicreg_fps_food_s2_p75 | VICReg | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #675 | — | — | — | `wait` | — | — |
| 702 | g2_vicreg_fps_food_s3_p75 | VICReg | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #697 | — | — | — | `wait` | — | — |
| 703 | g2_ae_fps_reggs_s1_p75 | growth (AE) | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #628 | — | — | — | `wait` | — | — |
| 704 | g2_ae_fps_reggs_s2_p75 | growth (AE) | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #652 | — | — | — | `wait` | — | — |
| 705 | g2_ae_fps_reggs_s3_p75 | growth (AE) | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #676 | — | — | — | `wait` | — | — |
| 706 | g2_ae_fps_rlarv_s1_p75 | growth (AE) | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #535 | — | — | — | `wait` | — | — |
| 707 | g2_ae_fps_rlarv_s2_p75 | growth (AE) | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #559 | — | — | — | `wait` | — | — |
| 708 | g2_ae_fps_rlarv_s3_p75 | growth (AE) | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #583 | — | — | — | `wait` | — | — |
| 709 | g2_ae_fps_rprot_s1_p75 | growth (AE) | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #700 | — | — | — | `wait` | — | — |
| 710 | g2_ae_fps_rprot_s2_p75 | growth (AE) | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #724 | — | — | — | `wait` | — | — |
| 711 | g2_ae_fps_rprot_s3_p75 | growth (AE) | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #748 | — | — | — | `wait` | — | — |
| 712 | g2_simclr_fps_reggs_s1_p75 | SimCLR | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #843 | — | — | — | `wait` | — | — |
| 713 | g2_simclr_fps_reggs_s2_p75 | SimCLR | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #854 | — | — | — | `wait` | — | — |
| 714 | g2_simclr_fps_reggs_s3_p75 | SimCLR | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #865 | — | — | — | `wait` | — | — |
| 715 | g2_simclr_fps_rlarv_s1_p75 | SimCLR | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #717 | — | — | — | `wait` | — | — |
| 716 | g2_simclr_fps_rlarv_s2_p75 | SimCLR | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #739 | — | — | — | `wait` | — | — |
| 717 | g2_simclr_fps_rlarv_s3_p75 | SimCLR | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #761 | — | — | — | `wait` | — | — |
| 718 | g2_simclr_fps_rprot_s1_p75 | SimCLR | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #876 | — | — | — | `wait` | — | — |
| 719 | g2_simclr_fps_rprot_s2_p75 | SimCLR | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #887 | — | — | — | `wait` | — | — |
| 720 | g2_simclr_fps_rprot_s3_p75 | SimCLR | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #898 | — | — | — | `wait` | — | — |
| 721 | g2_vicreg_fps_reggs_s1_p75 | VICReg | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #844 | — | — | — | `wait` | — | — |
| 722 | g2_vicreg_fps_reggs_s2_p75 | VICReg | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #855 | — | — | — | `wait` | — | — |
| 723 | g2_vicreg_fps_reggs_s3_p75 | VICReg | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #866 | — | — | — | `wait` | — | — |
| 724 | g2_vicreg_fps_rlarv_s1_p75 | VICReg | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #719 | — | — | — | `wait` | — | — |
| 725 | g2_vicreg_fps_rlarv_s2_p75 | VICReg | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #741 | — | — | — | `wait` | — | — |
| 726 | g2_vicreg_fps_rlarv_s3_p75 | VICReg | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #763 | — | — | — | `wait` | — | — |
| 727 | g2_vicreg_fps_rprot_s1_p75 | VICReg | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #877 | — | — | — | `wait` | — | — |
| 728 | g2_vicreg_fps_rprot_s2_p75 | VICReg | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #888 | — | — | — | `wait` | — | — |
| 729 | g2_vicreg_fps_rprot_s3_p75 | VICReg | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #899 | — | — | — | `wait` | — | — |

Subtotal: 81

### 75% — Rede 1 (compacta 4×4, final 1024) — 81 runs

_depois de 50% e 100%; mesma ordem_

| # | run | método | init | base | split | tag | máquina/GPU | camada atual | início | fim | SVM | κ val (camada final) | κ test (camada final) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 730 | g1_ae_fps_aid_s1_p75 | growth (AE) | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #706 | — | — | — | `wait` | — | — |
| 731 | g1_ae_fps_aid_s2_p75 | growth (AE) | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #730 | — | — | — | `wait` | — | — |
| 732 | g1_ae_fps_aid_s3_p75 | growth (AE) | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #754 | — | — | — | `wait` | — | — |
| 733 | g1_ae_fps_rs_s1_p75 | growth (AE) | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #517 | — | — | — | `wait` | — | — |
| 734 | g1_ae_fps_rs_s2_p75 | growth (AE) | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #541 | — | — | — | `wait` | — | — |
| 735 | g1_ae_fps_rs_s3_p75 | growth (AE) | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #565 | — | — | — | `wait` | — | — |
| 736 | g1_ae_fps_flw_s1_p75 | growth (AE) | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #589 | — | — | — | `wait` | — | — |
| 737 | g1_ae_fps_flw_s2_p75 | growth (AE) | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #611 | — | — | — | `wait` | — | — |
| 738 | g1_ae_fps_flw_s3_p75 | growth (AE) | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #633 | — | — | — | `wait` | — | — |
| 739 | g1_ae_fps_coco_s1_p75 | growth (AE) | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #776 | — | — | — | `wait` | — | — |
| 740 | g1_ae_fps_coco_s2_p75 | growth (AE) | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #790 | — | — | — | `wait` | — | — |
| 741 | g1_ae_fps_coco_s3_p75 | growth (AE) | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #801 | — | — | — | `wait` | — | — |
| 742 | g1_ae_fps_ham_s1_p75 | growth (AE) | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #812 | — | — | — | `wait` | — | — |
| 743 | g1_ae_fps_ham_s2_p75 | growth (AE) | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #823 | — | — | — | `wait` | — | — |
| 744 | g1_ae_fps_ham_s3_p75 | growth (AE) | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #834 | — | — | — | `wait` | — | — |
| 745 | g1_ae_fps_food_s1_p75 | growth (AE) | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #655 | — | — | — | `wait` | — | — |
| 746 | g1_ae_fps_food_s2_p75 | growth (AE) | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #677 | — | — | — | `wait` | — | — |
| 747 | g1_ae_fps_food_s3_p75 | growth (AE) | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #699 | — | — | — | `wait` | — | — |
| 748 | g1_simclr_fps_aid_s1_p75 | SimCLR | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #708 | — | — | — | `wait` | — | — |
| 749 | g1_simclr_fps_aid_s2_p75 | SimCLR | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #732 | — | — | — | `wait` | — | — |
| 750 | g1_simclr_fps_aid_s3_p75 | SimCLR | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #756 | — | — | — | `wait` | — | — |
| 751 | g1_simclr_fps_rs_s1_p75 | SimCLR | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #519 | — | — | — | `wait` | — | — |
| 752 | g1_simclr_fps_rs_s2_p75 | SimCLR | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #543 | — | — | — | `wait` | — | — |
| 753 | g1_simclr_fps_rs_s3_p75 | SimCLR | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #567 | — | — | — | `wait` | — | — |
| 754 | g1_simclr_fps_flw_s1_p75 | SimCLR | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #591 | — | — | — | `wait` | — | — |
| 755 | g1_simclr_fps_flw_s2_p75 | SimCLR | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #613 | — | — | — | `wait` | — | — |
| 756 | g1_simclr_fps_flw_s3_p75 | SimCLR | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #635 | — | — | — | `wait` | — | — |
| 757 | g1_simclr_fps_coco_s1_p75 | SimCLR | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #778 | — | — | — | `wait` | — | — |
| 758 | g1_simclr_fps_coco_s2_p75 | SimCLR | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #791 | — | — | — | `wait` | — | — |
| 759 | g1_simclr_fps_coco_s3_p75 | SimCLR | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #802 | — | — | — | `wait` | — | — |
| 760 | g1_simclr_fps_ham_s1_p75 | SimCLR | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #813 | — | — | — | `wait` | — | — |
| 761 | g1_simclr_fps_ham_s2_p75 | SimCLR | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #824 | — | — | — | `wait` | — | — |
| 762 | g1_simclr_fps_ham_s3_p75 | SimCLR | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #835 | — | — | — | `wait` | — | — |
| 763 | g1_simclr_fps_food_s1_p75 | SimCLR | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #657 | — | — | — | `wait` | — | — |
| 764 | g1_simclr_fps_food_s2_p75 | SimCLR | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #679 | — | — | — | `wait` | — | — |
| 765 | g1_simclr_fps_food_s3_p75 | SimCLR | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #701 | — | — | — | `wait` | — | — |
| 766 | g1_vicreg_fps_aid_s1_p75 | VICReg | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #710 | — | — | — | `wait` | — | — |
| 767 | g1_vicreg_fps_aid_s2_p75 | VICReg | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #734 | — | — | — | `wait` | — | — |
| 768 | g1_vicreg_fps_aid_s3_p75 | VICReg | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #758 | — | — | — | `wait` | — | — |
| 769 | g1_vicreg_fps_rs_s1_p75 | VICReg | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #521 | — | — | — | `wait` | — | — |
| 770 | g1_vicreg_fps_rs_s2_p75 | VICReg | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #545 | — | — | — | `wait` | — | — |
| 771 | g1_vicreg_fps_rs_s3_p75 | VICReg | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #569 | — | — | — | `wait` | — | — |
| 772 | g1_vicreg_fps_flw_s1_p75 | VICReg | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #593 | — | — | — | `wait` | — | — |
| 773 | g1_vicreg_fps_flw_s2_p75 | VICReg | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #615 | — | — | — | `wait` | — | — |
| 774 | g1_vicreg_fps_flw_s3_p75 | VICReg | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #637 | — | — | — | `wait` | — | — |
| 775 | g1_vicreg_fps_coco_s1_p75 | VICReg | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #780 | — | — | — | `wait` | — | — |
| 776 | g1_vicreg_fps_coco_s2_p75 | VICReg | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #792 | — | — | — | `wait` | — | — |
| 777 | g1_vicreg_fps_coco_s3_p75 | VICReg | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #803 | — | — | — | `wait` | — | — |
| 778 | g1_vicreg_fps_ham_s1_p75 | VICReg | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #814 | — | — | — | `wait` | — | — |
| 779 | g1_vicreg_fps_ham_s2_p75 | VICReg | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #825 | — | — | — | `wait` | — | — |
| 780 | g1_vicreg_fps_ham_s3_p75 | VICReg | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #836 | — | — | — | `wait` | — | — |
| 781 | g1_vicreg_fps_food_s1_p75 | VICReg | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #659 | — | — | — | `wait` | — | — |
| 782 | g1_vicreg_fps_food_s2_p75 | VICReg | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #681 | — | — | — | `wait` | — | — |
| 783 | g1_vicreg_fps_food_s3_p75 | VICReg | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #703 | — | — | — | `wait` | — | — |
| 784 | g1_ae_fps_reggs_s1_p75 | growth (AE) | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #845 | — | — | — | `wait` | — | — |
| 785 | g1_ae_fps_reggs_s2_p75 | growth (AE) | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #856 | — | — | — | `wait` | — | — |
| 786 | g1_ae_fps_reggs_s3_p75 | growth (AE) | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #867 | — | — | — | `wait` | — | — |
| 787 | g1_ae_fps_rlarv_s1_p75 | growth (AE) | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #721 | — | — | — | `wait` | — | — |
| 788 | g1_ae_fps_rlarv_s2_p75 | growth (AE) | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #743 | — | — | — | `wait` | — | — |
| 789 | g1_ae_fps_rlarv_s3_p75 | growth (AE) | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #765 | — | — | — | `wait` | — | — |
| 790 | g1_ae_fps_rprot_s1_p75 | growth (AE) | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #878 | — | — | — | `wait` | — | — |
| 791 | g1_ae_fps_rprot_s2_p75 | growth (AE) | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #889 | — | — | — | `wait` | — | — |
| 792 | g1_ae_fps_rprot_s3_p75 | growth (AE) | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #900 | — | — | — | `wait` | — | — |
| 793 | g1_simclr_fps_reggs_s1_p75 | SimCLR | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #846 | — | — | — | `wait` | — | — |
| 794 | g1_simclr_fps_reggs_s2_p75 | SimCLR | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #857 | — | — | — | `wait` | — | — |
| 795 | g1_simclr_fps_reggs_s3_p75 | SimCLR | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #868 | — | — | — | `wait` | — | — |
| 796 | g1_simclr_fps_rlarv_s1_p75 | SimCLR | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #723 | — | — | — | `wait` | — | — |
| 797 | g1_simclr_fps_rlarv_s2_p75 | SimCLR | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #745 | — | — | — | `wait` | — | — |
| 798 | g1_simclr_fps_rlarv_s3_p75 | SimCLR | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #767 | — | — | — | `wait` | — | — |
| 799 | g1_simclr_fps_rprot_s1_p75 | SimCLR | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #879 | — | — | — | `wait` | — | — |
| 800 | g1_simclr_fps_rprot_s2_p75 | SimCLR | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #890 | — | — | — | `wait` | — | — |
| 801 | g1_simclr_fps_rprot_s3_p75 | SimCLR | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #901 | — | — | — | `wait` | — | — |
| 802 | g1_vicreg_fps_reggs_s1_p75 | VICReg | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #847 | — | — | — | `wait` | — | — |
| 803 | g1_vicreg_fps_reggs_s2_p75 | VICReg | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #858 | — | — | — | `wait` | — | — |
| 804 | g1_vicreg_fps_reggs_s3_p75 | VICReg | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #869 | — | — | — | `wait` | — | — |
| 805 | g1_vicreg_fps_rlarv_s1_p75 | VICReg | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #725 | — | — | — | `wait` | — | — |
| 806 | g1_vicreg_fps_rlarv_s2_p75 | VICReg | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #747 | — | — | — | `wait` | — | — |
| 807 | g1_vicreg_fps_rlarv_s3_p75 | VICReg | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #769 | — | — | — | `wait` | — | — |
| 808 | g1_vicreg_fps_rprot_s1_p75 | VICReg | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #880 | — | — | — | `wait` | — | — |
| 809 | g1_vicreg_fps_rprot_s2_p75 | VICReg | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #891 | — | — | — | `wait` | — | — |
| 810 | g1_vicreg_fps_rprot_s3_p75 | VICReg | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #902 | — | — | — | `wait` | — | — |

Subtotal: 81

### 75% — Rede 3 (GeM+cat, final 1024) — 81 runs

_depois de 50% e 100%; mesma ordem_

| # | run | método | init | base | split | tag | máquina/GPU | camada atual | início | fim | SVM | κ val (camada final) | κ test (camada final) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 811 | g3_ae_fps_aid_s1_p75 | growth (AE) | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #712 | — | — | — | `wait` | — | — |
| 812 | g3_ae_fps_aid_s2_p75 | growth (AE) | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #736 | — | — | — | `wait` | — | — |
| 813 | g3_ae_fps_aid_s3_p75 | growth (AE) | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #760 | — | — | — | `wait` | — | — |
| 814 | g3_ae_fps_rs_s1_p75 | growth (AE) | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #523 | — | — | — | `wait` | — | — |
| 815 | g3_ae_fps_rs_s2_p75 | growth (AE) | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #547 | — | — | — | `wait` | — | — |
| 816 | g3_ae_fps_rs_s3_p75 | growth (AE) | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #571 | — | — | — | `wait` | — | — |
| 817 | g3_ae_fps_flw_s1_p75 | growth (AE) | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #595 | — | — | — | `wait` | — | — |
| 818 | g3_ae_fps_flw_s2_p75 | growth (AE) | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #617 | — | — | — | `wait` | — | — |
| 819 | g3_ae_fps_flw_s3_p75 | growth (AE) | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #639 | — | — | — | `wait` | — | — |
| 820 | g3_ae_fps_coco_s1_p75 | growth (AE) | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #782 | — | — | — | `wait` | — | — |
| 821 | g3_ae_fps_coco_s2_p75 | growth (AE) | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #793 | — | — | — | `wait` | — | — |
| 822 | g3_ae_fps_coco_s3_p75 | growth (AE) | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #804 | — | — | — | `wait` | — | — |
| 823 | g3_ae_fps_ham_s1_p75 | growth (AE) | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #815 | — | — | — | `wait` | — | — |
| 824 | g3_ae_fps_ham_s2_p75 | growth (AE) | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #826 | — | — | — | `wait` | — | — |
| 825 | g3_ae_fps_ham_s3_p75 | growth (AE) | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #837 | — | — | — | `wait` | — | — |
| 826 | g3_ae_fps_food_s1_p75 | growth (AE) | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #661 | — | — | — | `wait` | — | — |
| 827 | g3_ae_fps_food_s2_p75 | growth (AE) | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #683 | — | — | — | `wait` | — | — |
| 828 | g3_ae_fps_food_s3_p75 | growth (AE) | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #705 | — | — | — | `wait` | — | — |
| 829 | g3_simclr_fps_aid_s1_p75 | SimCLR | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #714 | — | — | — | `wait` | — | — |
| 830 | g3_simclr_fps_aid_s2_p75 | SimCLR | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #738 | — | — | — | `wait` | — | — |
| 831 | g3_simclr_fps_aid_s3_p75 | SimCLR | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #762 | — | — | — | `wait` | — | — |
| 832 | g3_simclr_fps_rs_s1_p75 | SimCLR | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #525 | — | — | — | `wait` | — | — |
| 833 | g3_simclr_fps_rs_s2_p75 | SimCLR | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #549 | — | — | — | `wait` | — | — |
| 834 | g3_simclr_fps_rs_s3_p75 | SimCLR | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #573 | — | — | — | `wait` | — | — |
| 835 | g3_simclr_fps_flw_s1_p75 | SimCLR | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #597 | — | — | — | `wait` | — | — |
| 836 | g3_simclr_fps_flw_s2_p75 | SimCLR | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #619 | — | — | — | `wait` | — | — |
| 837 | g3_simclr_fps_flw_s3_p75 | SimCLR | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #641 | — | — | — | `wait` | — | — |
| 838 | g3_simclr_fps_coco_s1_p75 | SimCLR | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #783 | — | — | — | `wait` | — | — |
| 839 | g3_simclr_fps_coco_s2_p75 | SimCLR | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #794 | — | — | — | `wait` | — | — |
| 840 | g3_simclr_fps_coco_s3_p75 | SimCLR | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #805 | — | — | — | `wait` | — | — |
| 841 | g3_simclr_fps_ham_s1_p75 | SimCLR | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #816 | — | — | — | `wait` | — | — |
| 842 | g3_simclr_fps_ham_s2_p75 | SimCLR | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #827 | — | — | — | `wait` | — | — |
| 843 | g3_simclr_fps_ham_s3_p75 | SimCLR | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #838 | — | — | — | `wait` | — | — |
| 844 | g3_simclr_fps_food_s1_p75 | SimCLR | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #663 | — | — | — | `wait` | — | — |
| 845 | g3_simclr_fps_food_s2_p75 | SimCLR | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #685 | — | — | — | `wait` | — | — |
| 846 | g3_simclr_fps_food_s3_p75 | SimCLR | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #707 | — | — | — | `wait` | — | — |
| 847 | g3_vicreg_fps_aid_s1_p75 | VICReg | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #716 | — | — | — | `wait` | — | — |
| 848 | g3_vicreg_fps_aid_s2_p75 | VICReg | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #740 | — | — | — | `wait` | — | — |
| 849 | g3_vicreg_fps_aid_s3_p75 | VICReg | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #764 | — | — | — | `wait` | — | — |
| 850 | g3_vicreg_fps_rs_s1_p75 | VICReg | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #527 | — | — | — | `wait` | — | — |
| 851 | g3_vicreg_fps_rs_s2_p75 | VICReg | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #551 | — | — | — | `wait` | — | — |
| 852 | g3_vicreg_fps_rs_s3_p75 | VICReg | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #575 | — | — | — | `wait` | — | — |
| 853 | g3_vicreg_fps_flw_s1_p75 | VICReg | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #599 | — | — | — | `wait` | — | — |
| 854 | g3_vicreg_fps_flw_s2_p75 | VICReg | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #621 | — | — | — | `wait` | — | — |
| 855 | g3_vicreg_fps_flw_s3_p75 | VICReg | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #643 | — | — | — | `wait` | — | — |
| 856 | g3_vicreg_fps_coco_s1_p75 | VICReg | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #784 | — | — | — | `wait` | — | — |
| 857 | g3_vicreg_fps_coco_s2_p75 | VICReg | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #795 | — | — | — | `wait` | — | — |
| 858 | g3_vicreg_fps_coco_s3_p75 | VICReg | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #806 | — | — | — | `wait` | — | — |
| 859 | g3_vicreg_fps_ham_s1_p75 | VICReg | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #817 | — | — | — | `wait` | — | — |
| 860 | g3_vicreg_fps_ham_s2_p75 | VICReg | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #828 | — | — | — | `wait` | — | — |
| 861 | g3_vicreg_fps_ham_s3_p75 | VICReg | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #839 | — | — | — | `wait` | — | — |
| 862 | g3_vicreg_fps_food_s1_p75 | VICReg | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #665 | — | — | — | `wait` | — | — |
| 863 | g3_vicreg_fps_food_s2_p75 | VICReg | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #687 | — | — | — | `wait` | — | — |
| 864 | g3_vicreg_fps_food_s3_p75 | VICReg | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #709 | — | — | — | `wait` | — | — |
| 865 | g3_ae_fps_reggs_s1_p75 | growth (AE) | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #848 | — | — | — | `wait` | — | — |
| 866 | g3_ae_fps_reggs_s2_p75 | growth (AE) | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #859 | — | — | — | `wait` | — | — |
| 867 | g3_ae_fps_reggs_s3_p75 | growth (AE) | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #870 | — | — | — | `wait` | — | — |
| 868 | g3_ae_fps_rlarv_s1_p75 | growth (AE) | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #727 | — | — | — | `wait` | — | — |
| 869 | g3_ae_fps_rlarv_s2_p75 | growth (AE) | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #749 | — | — | — | `wait` | — | — |
| 870 | g3_ae_fps_rlarv_s3_p75 | growth (AE) | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #771 | — | — | — | `wait` | — | — |
| 871 | g3_ae_fps_rprot_s1_p75 | growth (AE) | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #881 | — | — | — | `wait` | — | — |
| 872 | g3_ae_fps_rprot_s2_p75 | growth (AE) | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #892 | — | — | — | `wait` | — | — |
| 873 | g3_ae_fps_rprot_s3_p75 | growth (AE) | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #903 | — | — | — | `wait` | — | — |
| 874 | g3_simclr_fps_reggs_s1_p75 | SimCLR | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #849 | — | — | — | `wait` | — | — |
| 875 | g3_simclr_fps_reggs_s2_p75 | SimCLR | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #860 | — | — | — | `wait` | — | — |
| 876 | g3_simclr_fps_reggs_s3_p75 | SimCLR | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #871 | — | — | — | `wait` | — | — |
| 877 | g3_simclr_fps_rlarv_s1_p75 | SimCLR | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #729 | — | — | — | `wait` | — | — |
| 878 | g3_simclr_fps_rlarv_s2_p75 | SimCLR | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #751 | — | — | — | `wait` | — | — |
| 879 | g3_simclr_fps_rlarv_s3_p75 | SimCLR | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #773 | — | — | — | `wait` | — | — |
| 880 | g3_simclr_fps_rprot_s1_p75 | SimCLR | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #882 | — | — | — | `wait` | — | — |
| 881 | g3_simclr_fps_rprot_s2_p75 | SimCLR | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #893 | — | — | — | `wait` | — | — |
| 882 | g3_simclr_fps_rprot_s3_p75 | SimCLR | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #904 | — | — | — | `wait` | — | — |
| 883 | g3_vicreg_fps_reggs_s1_p75 | VICReg | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #850 | — | — | — | `wait` | — | — |
| 884 | g3_vicreg_fps_reggs_s2_p75 | VICReg | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #861 | — | — | — | `wait` | — | — |
| 885 | g3_vicreg_fps_reggs_s3_p75 | VICReg | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #872 | — | — | — | `wait` | — | — |
| 886 | g3_vicreg_fps_rlarv_s1_p75 | VICReg | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #731 | — | — | — | `wait` | — | — |
| 887 | g3_vicreg_fps_rlarv_s2_p75 | VICReg | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #753 | — | — | — | `wait` | — | — |
| 888 | g3_vicreg_fps_rlarv_s3_p75 | VICReg | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #775 | — | — | — | `wait` | — | — |
| 889 | g3_vicreg_fps_rprot_s1_p75 | VICReg | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #883 | — | — | — | `wait` | — | — |
| 890 | g3_vicreg_fps_rprot_s2_p75 | VICReg | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #894 | — | — | — | `wait` | — | — |
| 891 | g3_vicreg_fps_rprot_s3_p75 | VICReg | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #905 | — | — | — | `wait` | — | — |

Subtotal: 81

### 75% — Rede 4 (Inception 2×2, final 1024) — 81 runs

_depois de 50% e 100%; mesma ordem_

| # | run | método | init | base | split | tag | máquina/GPU | camada atual | início | fim | SVM | κ val (camada final) | κ test (camada final) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 892 | g4_ae_fps_aid_s1_p75 | growth (AE) | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #718 | — | — | — | `wait` | — | — |
| 893 | g4_ae_fps_aid_s2_p75 | growth (AE) | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #742 | — | — | — | `wait` | — | — |
| 894 | g4_ae_fps_aid_s3_p75 | growth (AE) | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #766 | — | — | — | `wait` | — | — |
| 895 | g4_ae_fps_rs_s1_p75 | growth (AE) | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #529 | — | — | — | `wait` | — | — |
| 896 | g4_ae_fps_rs_s2_p75 | growth (AE) | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #553 | — | — | — | `wait` | — | — |
| 897 | g4_ae_fps_rs_s3_p75 | growth (AE) | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #577 | — | — | — | `wait` | — | — |
| 898 | g4_ae_fps_flw_s1_p75 | growth (AE) | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #601 | — | — | — | `wait` | — | — |
| 899 | g4_ae_fps_flw_s2_p75 | growth (AE) | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #623 | — | — | — | `wait` | — | — |
| 900 | g4_ae_fps_flw_s3_p75 | growth (AE) | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #645 | — | — | — | `wait` | — | — |
| 901 | g4_ae_fps_coco_s1_p75 | growth (AE) | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #785 | — | — | — | `wait` | — | — |
| 902 | g4_ae_fps_coco_s2_p75 | growth (AE) | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #796 | — | — | — | `wait` | — | — |
| 903 | g4_ae_fps_coco_s3_p75 | growth (AE) | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #807 | — | — | — | `wait` | — | — |
| 904 | g4_ae_fps_ham_s1_p75 | growth (AE) | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #818 | — | — | — | `wait` | — | — |
| 905 | g4_ae_fps_ham_s2_p75 | growth (AE) | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #829 | — | — | — | `wait` | — | — |
| 906 | g4_ae_fps_ham_s3_p75 | growth (AE) | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #840 | — | — | — | `wait` | — | — |
| 907 | g4_ae_fps_food_s1_p75 | growth (AE) | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #667 | — | — | — | `wait` | — | — |
| 908 | g4_ae_fps_food_s2_p75 | growth (AE) | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #689 | — | — | — | `wait` | — | — |
| 909 | g4_ae_fps_food_s3_p75 | growth (AE) | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #711 | — | — | — | `wait` | — | — |
| 910 | g4_simclr_fps_aid_s1_p75 | SimCLR | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #720 | — | — | — | `wait` | — | — |
| 911 | g4_simclr_fps_aid_s2_p75 | SimCLR | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #744 | — | — | — | `wait` | — | — |
| 912 | g4_simclr_fps_aid_s3_p75 | SimCLR | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #768 | — | — | — | `wait` | — | — |
| 913 | g4_simclr_fps_rs_s1_p75 | SimCLR | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #531 | — | — | — | `wait` | — | — |
| 914 | g4_simclr_fps_rs_s2_p75 | SimCLR | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #555 | — | — | — | `wait` | — | — |
| 915 | g4_simclr_fps_rs_s3_p75 | SimCLR | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #579 | — | — | — | `wait` | — | — |
| 916 | g4_simclr_fps_flw_s1_p75 | SimCLR | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #603 | — | — | — | `wait` | — | — |
| 917 | g4_simclr_fps_flw_s2_p75 | SimCLR | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #625 | — | — | — | `wait` | — | — |
| 918 | g4_simclr_fps_flw_s3_p75 | SimCLR | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #647 | — | — | — | `wait` | — | — |
| 919 | g4_simclr_fps_coco_s1_p75 | SimCLR | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #786 | — | — | — | `wait` | — | — |
| 920 | g4_simclr_fps_coco_s2_p75 | SimCLR | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #797 | — | — | — | `wait` | — | — |
| 921 | g4_simclr_fps_coco_s3_p75 | SimCLR | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #808 | — | — | — | `wait` | — | — |
| 922 | g4_simclr_fps_ham_s1_p75 | SimCLR | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #819 | — | — | — | `wait` | — | — |
| 923 | g4_simclr_fps_ham_s2_p75 | SimCLR | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #830 | — | — | — | `wait` | — | — |
| 924 | g4_simclr_fps_ham_s3_p75 | SimCLR | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #841 | — | — | — | `wait` | — | — |
| 925 | g4_simclr_fps_food_s1_p75 | SimCLR | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #669 | — | — | — | `wait` | — | — |
| 926 | g4_simclr_fps_food_s2_p75 | SimCLR | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #691 | — | — | — | `wait` | — | — |
| 927 | g4_simclr_fps_food_s3_p75 | SimCLR | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #713 | — | — | — | `wait` | — | — |
| 928 | g4_vicreg_fps_aid_s1_p75 | VICReg | SPiFiL (FPS) | aid | 1 | `wait` | na fila athena #722 | — | — | — | `wait` | — | — |
| 929 | g4_vicreg_fps_aid_s2_p75 | VICReg | SPiFiL (FPS) | aid | 2 | `wait` | na fila athena #746 | — | — | — | `wait` | — | — |
| 930 | g4_vicreg_fps_aid_s3_p75 | VICReg | SPiFiL (FPS) | aid | 3 | `wait` | na fila athena #770 | — | — | — | `wait` | — | — |
| 931 | g4_vicreg_fps_rs_s1_p75 | VICReg | SPiFiL (FPS) | remote_sensing | 1 | `wait` | na fila athena #533 | — | — | — | `wait` | — | — |
| 932 | g4_vicreg_fps_rs_s2_p75 | VICReg | SPiFiL (FPS) | remote_sensing | 2 | `wait` | na fila athena #557 | — | — | — | `wait` | — | — |
| 933 | g4_vicreg_fps_rs_s3_p75 | VICReg | SPiFiL (FPS) | remote_sensing | 3 | `wait` | na fila athena #581 | — | — | — | `wait` | — | — |
| 934 | g4_vicreg_fps_flw_s1_p75 | VICReg | SPiFiL (FPS) | flowers | 1 | `wait` | na fila athena #605 | — | — | — | `wait` | — | — |
| 935 | g4_vicreg_fps_flw_s2_p75 | VICReg | SPiFiL (FPS) | flowers | 2 | `wait` | na fila athena #627 | — | — | — | `wait` | — | — |
| 936 | g4_vicreg_fps_flw_s3_p75 | VICReg | SPiFiL (FPS) | flowers | 3 | `wait` | na fila athena #649 | — | — | — | `wait` | — | — |
| 937 | g4_vicreg_fps_coco_s1_p75 | VICReg | SPiFiL (FPS) | coconut_trees | 1 | `wait` | na fila athena #787 | — | — | — | `wait` | — | — |
| 938 | g4_vicreg_fps_coco_s2_p75 | VICReg | SPiFiL (FPS) | coconut_trees | 2 | `wait` | na fila athena #798 | — | — | — | `wait` | — | — |
| 939 | g4_vicreg_fps_coco_s3_p75 | VICReg | SPiFiL (FPS) | coconut_trees | 3 | `wait` | na fila athena #809 | — | — | — | `wait` | — | — |
| 940 | g4_vicreg_fps_ham_s1_p75 | VICReg | SPiFiL (FPS) | ham10000 | 1 | `wait` | na fila athena #820 | — | — | — | `wait` | — | — |
| 941 | g4_vicreg_fps_ham_s2_p75 | VICReg | SPiFiL (FPS) | ham10000 | 2 | `wait` | na fila athena #831 | — | — | — | `wait` | — | — |
| 942 | g4_vicreg_fps_ham_s3_p75 | VICReg | SPiFiL (FPS) | ham10000 | 3 | `wait` | na fila athena #842 | — | — | — | `wait` | — | — |
| 943 | g4_vicreg_fps_food_s1_p75 | VICReg | SPiFiL (FPS) | food101 | 1 | `wait` | na fila athena #671 | — | — | — | `wait` | — | — |
| 944 | g4_vicreg_fps_food_s2_p75 | VICReg | SPiFiL (FPS) | food101 | 2 | `wait` | na fila athena #693 | — | — | — | `wait` | — | — |
| 945 | g4_vicreg_fps_food_s3_p75 | VICReg | SPiFiL (FPS) | food101 | 3 | `wait` | na fila athena #715 | — | — | — | `wait` | — | — |
| 946 | g4_ae_fps_reggs_s1_p75 | growth (AE) | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #851 | — | — | — | `wait` | — | — |
| 947 | g4_ae_fps_reggs_s2_p75 | growth (AE) | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #862 | — | — | — | `wait` | — | — |
| 948 | g4_ae_fps_reggs_s3_p75 | growth (AE) | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #873 | — | — | — | `wait` | — | — |
| 949 | g4_ae_fps_rlarv_s1_p75 | growth (AE) | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #733 | — | — | — | `wait` | — | — |
| 950 | g4_ae_fps_rlarv_s2_p75 | growth (AE) | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #755 | — | — | — | `wait` | — | — |
| 951 | g4_ae_fps_rlarv_s3_p75 | growth (AE) | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #777 | — | — | — | `wait` | — | — |
| 952 | g4_ae_fps_rprot_s1_p75 | growth (AE) | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #884 | — | — | — | `wait` | — | — |
| 953 | g4_ae_fps_rprot_s2_p75 | growth (AE) | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #895 | — | — | — | `wait` | — | — |
| 954 | g4_ae_fps_rprot_s3_p75 | growth (AE) | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #906 | — | — | — | `wait` | — | — |
| 955 | g4_simclr_fps_reggs_s1_p75 | SimCLR | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #852 | — | — | — | `wait` | — | — |
| 956 | g4_simclr_fps_reggs_s2_p75 | SimCLR | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #863 | — | — | — | `wait` | — | — |
| 957 | g4_simclr_fps_reggs_s3_p75 | SimCLR | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #874 | — | — | — | `wait` | — | — |
| 958 | g4_simclr_fps_rlarv_s1_p75 | SimCLR | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #735 | — | — | — | `wait` | — | — |
| 959 | g4_simclr_fps_rlarv_s2_p75 | SimCLR | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #757 | — | — | — | `wait` | — | — |
| 960 | g4_simclr_fps_rlarv_s3_p75 | SimCLR | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #779 | — | — | — | `wait` | — | — |
| 961 | g4_simclr_fps_rprot_s1_p75 | SimCLR | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #885 | — | — | — | `wait` | — | — |
| 962 | g4_simclr_fps_rprot_s2_p75 | SimCLR | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #896 | — | — | — | `wait` | — | — |
| 963 | g4_simclr_fps_rprot_s3_p75 | SimCLR | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #907 | — | — | — | `wait` | — | — |
| 964 | g4_vicreg_fps_reggs_s1_p75 | VICReg | SPiFiL (FPS) | Eggs | 1 | `wait` | na fila athena #853 | — | — | — | `wait` | — | — |
| 965 | g4_vicreg_fps_reggs_s2_p75 | VICReg | SPiFiL (FPS) | Eggs | 2 | `wait` | na fila athena #864 | — | — | — | `wait` | — | — |
| 966 | g4_vicreg_fps_reggs_s3_p75 | VICReg | SPiFiL (FPS) | Eggs | 3 | `wait` | na fila athena #875 | — | — | — | `wait` | — | — |
| 967 | g4_vicreg_fps_rlarv_s1_p75 | VICReg | SPiFiL (FPS) | Larvae | 1 | `wait` | na fila athena #737 | — | — | — | `wait` | — | — |
| 968 | g4_vicreg_fps_rlarv_s2_p75 | VICReg | SPiFiL (FPS) | Larvae | 2 | `wait` | na fila athena #759 | — | — | — | `wait` | — | — |
| 969 | g4_vicreg_fps_rlarv_s3_p75 | VICReg | SPiFiL (FPS) | Larvae | 3 | `wait` | na fila athena #781 | — | — | — | `wait` | — | — |
| 970 | g4_vicreg_fps_rprot_s1_p75 | VICReg | SPiFiL (FPS) | Protozoan | 1 | `wait` | na fila athena #886 | — | — | — | `wait` | — | — |
| 971 | g4_vicreg_fps_rprot_s2_p75 | VICReg | SPiFiL (FPS) | Protozoan | 2 | `wait` | na fila athena #897 | — | — | — | `wait` | — | — |
| 972 | g4_vicreg_fps_rprot_s3_p75 | VICReg | SPiFiL (FPS) | Protozoan | 3 | `wait` | na fila athena #908 | — | — | — | `wait` | — | — |

Subtotal: 81

### Inception e residual na camada nova (rede 2, só growth/AE) — 162 runs

_depois do 75%; variante > grupo > % > base > split_

| # | run | método | init | base | split | tag | máquina/GPU | camada atual | início | fim | SVM | κ val (camada final) | κ test (camada final) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 973 | g2_ae_inc_fps_aid_s1_p50 | growth (AE) + inception | SPiFiL (FPS) | aid | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 974 | g2_ae_inc_fps_aid_s2_p50 | growth (AE) + inception | SPiFiL (FPS) | aid | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 975 | g2_ae_inc_fps_aid_s3_p50 | growth (AE) + inception | SPiFiL (FPS) | aid | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 976 | g2_ae_inc_fps_rs_s1_p50 | growth (AE) + inception | SPiFiL (FPS) | remote_sensing | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 977 | g2_ae_inc_fps_rs_s2_p50 | growth (AE) + inception | SPiFiL (FPS) | remote_sensing | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 978 | g2_ae_inc_fps_rs_s3_p50 | growth (AE) + inception | SPiFiL (FPS) | remote_sensing | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 979 | g2_ae_inc_fps_flw_s1_p50 | growth (AE) + inception | SPiFiL (FPS) | flowers | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 980 | g2_ae_inc_fps_flw_s2_p50 | growth (AE) + inception | SPiFiL (FPS) | flowers | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 981 | g2_ae_inc_fps_flw_s3_p50 | growth (AE) + inception | SPiFiL (FPS) | flowers | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 982 | g2_ae_inc_fps_coco_s1_p50 | growth (AE) + inception | SPiFiL (FPS) | coconut_trees | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 983 | g2_ae_inc_fps_coco_s2_p50 | growth (AE) + inception | SPiFiL (FPS) | coconut_trees | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 984 | g2_ae_inc_fps_coco_s3_p50 | growth (AE) + inception | SPiFiL (FPS) | coconut_trees | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 985 | g2_ae_inc_fps_ham_s1_p50 | growth (AE) + inception | SPiFiL (FPS) | ham10000 | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 986 | g2_ae_inc_fps_ham_s2_p50 | growth (AE) + inception | SPiFiL (FPS) | ham10000 | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 987 | g2_ae_inc_fps_ham_s3_p50 | growth (AE) + inception | SPiFiL (FPS) | ham10000 | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 988 | g2_ae_inc_fps_food_s1_p50 | growth (AE) + inception | SPiFiL (FPS) | food101 | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 989 | g2_ae_inc_fps_food_s2_p50 | growth (AE) + inception | SPiFiL (FPS) | food101 | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 990 | g2_ae_inc_fps_food_s3_p50 | growth (AE) + inception | SPiFiL (FPS) | food101 | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 991 | g2_ae_inc_fps_aid_s1_p100 | growth (AE) + inception | SPiFiL (FPS) | aid | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 992 | g2_ae_inc_fps_aid_s2_p100 | growth (AE) + inception | SPiFiL (FPS) | aid | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 993 | g2_ae_inc_fps_aid_s3_p100 | growth (AE) + inception | SPiFiL (FPS) | aid | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 994 | g2_ae_inc_fps_rs_s1_p100 | growth (AE) + inception | SPiFiL (FPS) | remote_sensing | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 995 | g2_ae_inc_fps_rs_s2_p100 | growth (AE) + inception | SPiFiL (FPS) | remote_sensing | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 996 | g2_ae_inc_fps_rs_s3_p100 | growth (AE) + inception | SPiFiL (FPS) | remote_sensing | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 997 | g2_ae_inc_fps_flw_s1_p100 | growth (AE) + inception | SPiFiL (FPS) | flowers | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 998 | g2_ae_inc_fps_flw_s2_p100 | growth (AE) + inception | SPiFiL (FPS) | flowers | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 999 | g2_ae_inc_fps_flw_s3_p100 | growth (AE) + inception | SPiFiL (FPS) | flowers | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1000 | g2_ae_inc_fps_coco_s1_p100 | growth (AE) + inception | SPiFiL (FPS) | coconut_trees | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1001 | g2_ae_inc_fps_coco_s2_p100 | growth (AE) + inception | SPiFiL (FPS) | coconut_trees | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1002 | g2_ae_inc_fps_coco_s3_p100 | growth (AE) + inception | SPiFiL (FPS) | coconut_trees | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1003 | g2_ae_inc_fps_ham_s1_p100 | growth (AE) + inception | SPiFiL (FPS) | ham10000 | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1004 | g2_ae_inc_fps_ham_s2_p100 | growth (AE) + inception | SPiFiL (FPS) | ham10000 | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1005 | g2_ae_inc_fps_ham_s3_p100 | growth (AE) + inception | SPiFiL (FPS) | ham10000 | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1006 | g2_ae_inc_fps_food_s1_p100 | growth (AE) + inception | SPiFiL (FPS) | food101 | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1007 | g2_ae_inc_fps_food_s2_p100 | growth (AE) + inception | SPiFiL (FPS) | food101 | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1008 | g2_ae_inc_fps_food_s3_p100 | growth (AE) + inception | SPiFiL (FPS) | food101 | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1009 | g2_ae_inc_fps_aid_s1_p75 | growth (AE) + inception | SPiFiL (FPS) | aid | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1010 | g2_ae_inc_fps_aid_s2_p75 | growth (AE) + inception | SPiFiL (FPS) | aid | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1011 | g2_ae_inc_fps_aid_s3_p75 | growth (AE) + inception | SPiFiL (FPS) | aid | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1012 | g2_ae_inc_fps_rs_s1_p75 | growth (AE) + inception | SPiFiL (FPS) | remote_sensing | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1013 | g2_ae_inc_fps_rs_s2_p75 | growth (AE) + inception | SPiFiL (FPS) | remote_sensing | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1014 | g2_ae_inc_fps_rs_s3_p75 | growth (AE) + inception | SPiFiL (FPS) | remote_sensing | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1015 | g2_ae_inc_fps_flw_s1_p75 | growth (AE) + inception | SPiFiL (FPS) | flowers | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1016 | g2_ae_inc_fps_flw_s2_p75 | growth (AE) + inception | SPiFiL (FPS) | flowers | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1017 | g2_ae_inc_fps_flw_s3_p75 | growth (AE) + inception | SPiFiL (FPS) | flowers | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1018 | g2_ae_inc_fps_coco_s1_p75 | growth (AE) + inception | SPiFiL (FPS) | coconut_trees | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1019 | g2_ae_inc_fps_coco_s2_p75 | growth (AE) + inception | SPiFiL (FPS) | coconut_trees | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1020 | g2_ae_inc_fps_coco_s3_p75 | growth (AE) + inception | SPiFiL (FPS) | coconut_trees | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1021 | g2_ae_inc_fps_ham_s1_p75 | growth (AE) + inception | SPiFiL (FPS) | ham10000 | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1022 | g2_ae_inc_fps_ham_s2_p75 | growth (AE) + inception | SPiFiL (FPS) | ham10000 | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1023 | g2_ae_inc_fps_ham_s3_p75 | growth (AE) + inception | SPiFiL (FPS) | ham10000 | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1024 | g2_ae_inc_fps_food_s1_p75 | growth (AE) + inception | SPiFiL (FPS) | food101 | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1025 | g2_ae_inc_fps_food_s2_p75 | growth (AE) + inception | SPiFiL (FPS) | food101 | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1026 | g2_ae_inc_fps_food_s3_p75 | growth (AE) + inception | SPiFiL (FPS) | food101 | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1027 | g2_ae_inc_fps_reggs_s1_p50 | growth (AE) + inception | SPiFiL (FPS) | Eggs | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1028 | g2_ae_inc_fps_reggs_s2_p50 | growth (AE) + inception | SPiFiL (FPS) | Eggs | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1029 | g2_ae_inc_fps_reggs_s3_p50 | growth (AE) + inception | SPiFiL (FPS) | Eggs | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1030 | g2_ae_inc_fps_rlarv_s1_p50 | growth (AE) + inception | SPiFiL (FPS) | Larvae | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1031 | g2_ae_inc_fps_rlarv_s2_p50 | growth (AE) + inception | SPiFiL (FPS) | Larvae | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1032 | g2_ae_inc_fps_rlarv_s3_p50 | growth (AE) + inception | SPiFiL (FPS) | Larvae | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1033 | g2_ae_inc_fps_rprot_s1_p50 | growth (AE) + inception | SPiFiL (FPS) | Protozoan | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1034 | g2_ae_inc_fps_rprot_s2_p50 | growth (AE) + inception | SPiFiL (FPS) | Protozoan | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1035 | g2_ae_inc_fps_rprot_s3_p50 | growth (AE) + inception | SPiFiL (FPS) | Protozoan | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1036 | g2_ae_inc_fps_reggs_s1_p100 | growth (AE) + inception | SPiFiL (FPS) | Eggs | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1037 | g2_ae_inc_fps_reggs_s2_p100 | growth (AE) + inception | SPiFiL (FPS) | Eggs | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1038 | g2_ae_inc_fps_reggs_s3_p100 | growth (AE) + inception | SPiFiL (FPS) | Eggs | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1039 | g2_ae_inc_fps_rlarv_s1_p100 | growth (AE) + inception | SPiFiL (FPS) | Larvae | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1040 | g2_ae_inc_fps_rlarv_s2_p100 | growth (AE) + inception | SPiFiL (FPS) | Larvae | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1041 | g2_ae_inc_fps_rlarv_s3_p100 | growth (AE) + inception | SPiFiL (FPS) | Larvae | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1042 | g2_ae_inc_fps_rprot_s1_p100 | growth (AE) + inception | SPiFiL (FPS) | Protozoan | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1043 | g2_ae_inc_fps_rprot_s2_p100 | growth (AE) + inception | SPiFiL (FPS) | Protozoan | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1044 | g2_ae_inc_fps_rprot_s3_p100 | growth (AE) + inception | SPiFiL (FPS) | Protozoan | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1045 | g2_ae_inc_fps_reggs_s1_p75 | growth (AE) + inception | SPiFiL (FPS) | Eggs | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1046 | g2_ae_inc_fps_reggs_s2_p75 | growth (AE) + inception | SPiFiL (FPS) | Eggs | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1047 | g2_ae_inc_fps_reggs_s3_p75 | growth (AE) + inception | SPiFiL (FPS) | Eggs | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1048 | g2_ae_inc_fps_rlarv_s1_p75 | growth (AE) + inception | SPiFiL (FPS) | Larvae | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1049 | g2_ae_inc_fps_rlarv_s2_p75 | growth (AE) + inception | SPiFiL (FPS) | Larvae | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1050 | g2_ae_inc_fps_rlarv_s3_p75 | growth (AE) + inception | SPiFiL (FPS) | Larvae | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1051 | g2_ae_inc_fps_rprot_s1_p75 | growth (AE) + inception | SPiFiL (FPS) | Protozoan | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1052 | g2_ae_inc_fps_rprot_s2_p75 | growth (AE) + inception | SPiFiL (FPS) | Protozoan | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1053 | g2_ae_inc_fps_rprot_s3_p75 | growth (AE) + inception | SPiFiL (FPS) | Protozoan | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1054 | g2_ae_res_fps_aid_s1_p50 | growth (AE) + residual | SPiFiL (FPS) | aid | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1055 | g2_ae_res_fps_aid_s2_p50 | growth (AE) + residual | SPiFiL (FPS) | aid | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1056 | g2_ae_res_fps_aid_s3_p50 | growth (AE) + residual | SPiFiL (FPS) | aid | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1057 | g2_ae_res_fps_rs_s1_p50 | growth (AE) + residual | SPiFiL (FPS) | remote_sensing | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1058 | g2_ae_res_fps_rs_s2_p50 | growth (AE) + residual | SPiFiL (FPS) | remote_sensing | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1059 | g2_ae_res_fps_rs_s3_p50 | growth (AE) + residual | SPiFiL (FPS) | remote_sensing | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1060 | g2_ae_res_fps_flw_s1_p50 | growth (AE) + residual | SPiFiL (FPS) | flowers | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1061 | g2_ae_res_fps_flw_s2_p50 | growth (AE) + residual | SPiFiL (FPS) | flowers | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1062 | g2_ae_res_fps_flw_s3_p50 | growth (AE) + residual | SPiFiL (FPS) | flowers | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1063 | g2_ae_res_fps_coco_s1_p50 | growth (AE) + residual | SPiFiL (FPS) | coconut_trees | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1064 | g2_ae_res_fps_coco_s2_p50 | growth (AE) + residual | SPiFiL (FPS) | coconut_trees | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1065 | g2_ae_res_fps_coco_s3_p50 | growth (AE) + residual | SPiFiL (FPS) | coconut_trees | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1066 | g2_ae_res_fps_ham_s1_p50 | growth (AE) + residual | SPiFiL (FPS) | ham10000 | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1067 | g2_ae_res_fps_ham_s2_p50 | growth (AE) + residual | SPiFiL (FPS) | ham10000 | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1068 | g2_ae_res_fps_ham_s3_p50 | growth (AE) + residual | SPiFiL (FPS) | ham10000 | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1069 | g2_ae_res_fps_food_s1_p50 | growth (AE) + residual | SPiFiL (FPS) | food101 | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1070 | g2_ae_res_fps_food_s2_p50 | growth (AE) + residual | SPiFiL (FPS) | food101 | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1071 | g2_ae_res_fps_food_s3_p50 | growth (AE) + residual | SPiFiL (FPS) | food101 | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1072 | g2_ae_res_fps_aid_s1_p100 | growth (AE) + residual | SPiFiL (FPS) | aid | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1073 | g2_ae_res_fps_aid_s2_p100 | growth (AE) + residual | SPiFiL (FPS) | aid | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1074 | g2_ae_res_fps_aid_s3_p100 | growth (AE) + residual | SPiFiL (FPS) | aid | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1075 | g2_ae_res_fps_rs_s1_p100 | growth (AE) + residual | SPiFiL (FPS) | remote_sensing | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1076 | g2_ae_res_fps_rs_s2_p100 | growth (AE) + residual | SPiFiL (FPS) | remote_sensing | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1077 | g2_ae_res_fps_rs_s3_p100 | growth (AE) + residual | SPiFiL (FPS) | remote_sensing | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1078 | g2_ae_res_fps_flw_s1_p100 | growth (AE) + residual | SPiFiL (FPS) | flowers | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1079 | g2_ae_res_fps_flw_s2_p100 | growth (AE) + residual | SPiFiL (FPS) | flowers | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1080 | g2_ae_res_fps_flw_s3_p100 | growth (AE) + residual | SPiFiL (FPS) | flowers | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1081 | g2_ae_res_fps_coco_s1_p100 | growth (AE) + residual | SPiFiL (FPS) | coconut_trees | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1082 | g2_ae_res_fps_coco_s2_p100 | growth (AE) + residual | SPiFiL (FPS) | coconut_trees | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1083 | g2_ae_res_fps_coco_s3_p100 | growth (AE) + residual | SPiFiL (FPS) | coconut_trees | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1084 | g2_ae_res_fps_ham_s1_p100 | growth (AE) + residual | SPiFiL (FPS) | ham10000 | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1085 | g2_ae_res_fps_ham_s2_p100 | growth (AE) + residual | SPiFiL (FPS) | ham10000 | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1086 | g2_ae_res_fps_ham_s3_p100 | growth (AE) + residual | SPiFiL (FPS) | ham10000 | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1087 | g2_ae_res_fps_food_s1_p100 | growth (AE) + residual | SPiFiL (FPS) | food101 | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1088 | g2_ae_res_fps_food_s2_p100 | growth (AE) + residual | SPiFiL (FPS) | food101 | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1089 | g2_ae_res_fps_food_s3_p100 | growth (AE) + residual | SPiFiL (FPS) | food101 | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1090 | g2_ae_res_fps_aid_s1_p75 | growth (AE) + residual | SPiFiL (FPS) | aid | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1091 | g2_ae_res_fps_aid_s2_p75 | growth (AE) + residual | SPiFiL (FPS) | aid | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1092 | g2_ae_res_fps_aid_s3_p75 | growth (AE) + residual | SPiFiL (FPS) | aid | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1093 | g2_ae_res_fps_rs_s1_p75 | growth (AE) + residual | SPiFiL (FPS) | remote_sensing | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1094 | g2_ae_res_fps_rs_s2_p75 | growth (AE) + residual | SPiFiL (FPS) | remote_sensing | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1095 | g2_ae_res_fps_rs_s3_p75 | growth (AE) + residual | SPiFiL (FPS) | remote_sensing | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1096 | g2_ae_res_fps_flw_s1_p75 | growth (AE) + residual | SPiFiL (FPS) | flowers | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1097 | g2_ae_res_fps_flw_s2_p75 | growth (AE) + residual | SPiFiL (FPS) | flowers | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1098 | g2_ae_res_fps_flw_s3_p75 | growth (AE) + residual | SPiFiL (FPS) | flowers | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1099 | g2_ae_res_fps_coco_s1_p75 | growth (AE) + residual | SPiFiL (FPS) | coconut_trees | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1100 | g2_ae_res_fps_coco_s2_p75 | growth (AE) + residual | SPiFiL (FPS) | coconut_trees | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1101 | g2_ae_res_fps_coco_s3_p75 | growth (AE) + residual | SPiFiL (FPS) | coconut_trees | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1102 | g2_ae_res_fps_ham_s1_p75 | growth (AE) + residual | SPiFiL (FPS) | ham10000 | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1103 | g2_ae_res_fps_ham_s2_p75 | growth (AE) + residual | SPiFiL (FPS) | ham10000 | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1104 | g2_ae_res_fps_ham_s3_p75 | growth (AE) + residual | SPiFiL (FPS) | ham10000 | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1105 | g2_ae_res_fps_food_s1_p75 | growth (AE) + residual | SPiFiL (FPS) | food101 | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1106 | g2_ae_res_fps_food_s2_p75 | growth (AE) + residual | SPiFiL (FPS) | food101 | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1107 | g2_ae_res_fps_food_s3_p75 | growth (AE) + residual | SPiFiL (FPS) | food101 | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1108 | g2_ae_res_fps_reggs_s1_p50 | growth (AE) + residual | SPiFiL (FPS) | Eggs | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1109 | g2_ae_res_fps_reggs_s2_p50 | growth (AE) + residual | SPiFiL (FPS) | Eggs | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1110 | g2_ae_res_fps_reggs_s3_p50 | growth (AE) + residual | SPiFiL (FPS) | Eggs | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1111 | g2_ae_res_fps_rlarv_s1_p50 | growth (AE) + residual | SPiFiL (FPS) | Larvae | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1112 | g2_ae_res_fps_rlarv_s2_p50 | growth (AE) + residual | SPiFiL (FPS) | Larvae | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1113 | g2_ae_res_fps_rlarv_s3_p50 | growth (AE) + residual | SPiFiL (FPS) | Larvae | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1114 | g2_ae_res_fps_rprot_s1_p50 | growth (AE) + residual | SPiFiL (FPS) | Protozoan | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1115 | g2_ae_res_fps_rprot_s2_p50 | growth (AE) + residual | SPiFiL (FPS) | Protozoan | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1116 | g2_ae_res_fps_rprot_s3_p50 | growth (AE) + residual | SPiFiL (FPS) | Protozoan | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1117 | g2_ae_res_fps_reggs_s1_p100 | growth (AE) + residual | SPiFiL (FPS) | Eggs | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1118 | g2_ae_res_fps_reggs_s2_p100 | growth (AE) + residual | SPiFiL (FPS) | Eggs | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1119 | g2_ae_res_fps_reggs_s3_p100 | growth (AE) + residual | SPiFiL (FPS) | Eggs | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1120 | g2_ae_res_fps_rlarv_s1_p100 | growth (AE) + residual | SPiFiL (FPS) | Larvae | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1121 | g2_ae_res_fps_rlarv_s2_p100 | growth (AE) + residual | SPiFiL (FPS) | Larvae | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1122 | g2_ae_res_fps_rlarv_s3_p100 | growth (AE) + residual | SPiFiL (FPS) | Larvae | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1123 | g2_ae_res_fps_rprot_s1_p100 | growth (AE) + residual | SPiFiL (FPS) | Protozoan | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1124 | g2_ae_res_fps_rprot_s2_p100 | growth (AE) + residual | SPiFiL (FPS) | Protozoan | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1125 | g2_ae_res_fps_rprot_s3_p100 | growth (AE) + residual | SPiFiL (FPS) | Protozoan | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1126 | g2_ae_res_fps_reggs_s1_p75 | growth (AE) + residual | SPiFiL (FPS) | Eggs | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1127 | g2_ae_res_fps_reggs_s2_p75 | growth (AE) + residual | SPiFiL (FPS) | Eggs | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1128 | g2_ae_res_fps_reggs_s3_p75 | growth (AE) + residual | SPiFiL (FPS) | Eggs | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1129 | g2_ae_res_fps_rlarv_s1_p75 | growth (AE) + residual | SPiFiL (FPS) | Larvae | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1130 | g2_ae_res_fps_rlarv_s2_p75 | growth (AE) + residual | SPiFiL (FPS) | Larvae | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1131 | g2_ae_res_fps_rlarv_s3_p75 | growth (AE) + residual | SPiFiL (FPS) | Larvae | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1132 | g2_ae_res_fps_rprot_s1_p75 | growth (AE) + residual | SPiFiL (FPS) | Protozoan | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1133 | g2_ae_res_fps_rprot_s2_p75 | growth (AE) + residual | SPiFiL (FPS) | Protozoan | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1134 | g2_ae_res_fps_rprot_s3_p75 | growth (AE) + residual | SPiFiL (FPS) | Protozoan | 3 | `wait` | — | — | — | — | `wait` | — | — |

Subtotal: 162

### Augmentation 1ª rodada (sem backprop, spf_grade da rede 2, 50%) — 108 runs

_por último; caso > base > split; sem camadas_

| # | run | método | init | base | split | tag | máquina/GPU | camada atual | início | fim | SVM | κ val (camada final) | κ test (camada final) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1135 | aug_base_aid_s1_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | aid | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1136 | aug_base_aid_s2_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | aid | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1137 | aug_base_aid_s3_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | aid | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1138 | aug_base_rs_s1_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | remote_sensing | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1139 | aug_base_rs_s2_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | remote_sensing | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1140 | aug_base_rs_s3_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | remote_sensing | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1141 | aug_base_flw_s1_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | flowers | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1142 | aug_base_flw_s2_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | flowers | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1143 | aug_base_flw_s3_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | flowers | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1144 | aug_base_coco_s1_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | coconut_trees | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1145 | aug_base_coco_s2_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | coconut_trees | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1146 | aug_base_coco_s3_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | coconut_trees | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1147 | aug_base_ham_s1_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | ham10000 | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1148 | aug_base_ham_s2_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | ham10000 | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1149 | aug_base_ham_s3_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | ham10000 | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1150 | aug_base_food_s1_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | food101 | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1151 | aug_base_food_s2_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | food101 | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1152 | aug_base_food_s3_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | food101 | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1153 | aug_base_reggs_s1_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | Eggs | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1154 | aug_base_reggs_s2_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | Eggs | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1155 | aug_base_reggs_s3_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | Eggs | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1156 | aug_base_rlarv_s1_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | Larvae | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1157 | aug_base_rlarv_s2_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | Larvae | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1158 | aug_base_rlarv_s3_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | Larvae | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1159 | aug_base_rprot_s1_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | Protozoan | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1160 | aug_base_rprot_s2_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | Protozoan | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1161 | aug_base_rprot_s3_p50 | spf_grade rede 2, baseline | SPiFiL (FPS) | Protozoan | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1162 | aug_fps_aid_s1_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | aid | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1163 | aug_fps_aid_s2_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | aid | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1164 | aug_fps_aid_s3_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | aid | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1165 | aug_fps_rs_s1_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | remote_sensing | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1166 | aug_fps_rs_s2_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | remote_sensing | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1167 | aug_fps_rs_s3_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | remote_sensing | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1168 | aug_fps_flw_s1_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | flowers | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1169 | aug_fps_flw_s2_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | flowers | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1170 | aug_fps_flw_s3_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | flowers | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1171 | aug_fps_coco_s1_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | coconut_trees | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1172 | aug_fps_coco_s2_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | coconut_trees | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1173 | aug_fps_coco_s3_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | coconut_trees | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1174 | aug_fps_ham_s1_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | ham10000 | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1175 | aug_fps_ham_s2_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | ham10000 | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1176 | aug_fps_ham_s3_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | ham10000 | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1177 | aug_fps_food_s1_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | food101 | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1178 | aug_fps_food_s2_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | food101 | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1179 | aug_fps_food_s3_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | food101 | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1180 | aug_fps_reggs_s1_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | Eggs | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1181 | aug_fps_reggs_s2_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | Eggs | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1182 | aug_fps_reggs_s3_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | Eggs | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1183 | aug_fps_rlarv_s1_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | Larvae | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1184 | aug_fps_rlarv_s2_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | Larvae | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1185 | aug_fps_rlarv_s3_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | Larvae | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1186 | aug_fps_rprot_s1_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | Protozoan | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1187 | aug_fps_rprot_s2_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | Protozoan | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1188 | aug_fps_rprot_s3_p50 | spf_grade rede 2, aug só no banco FPS | SPiFiL (FPS) | Protozoan | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1189 | aug_svm_aid_s1_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | aid | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1190 | aug_svm_aid_s2_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | aid | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1191 | aug_svm_aid_s3_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | aid | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1192 | aug_svm_rs_s1_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | remote_sensing | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1193 | aug_svm_rs_s2_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | remote_sensing | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1194 | aug_svm_rs_s3_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | remote_sensing | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1195 | aug_svm_flw_s1_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | flowers | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1196 | aug_svm_flw_s2_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | flowers | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1197 | aug_svm_flw_s3_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | flowers | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1198 | aug_svm_coco_s1_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | coconut_trees | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1199 | aug_svm_coco_s2_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | coconut_trees | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1200 | aug_svm_coco_s3_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | coconut_trees | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1201 | aug_svm_ham_s1_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | ham10000 | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1202 | aug_svm_ham_s2_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | ham10000 | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1203 | aug_svm_ham_s3_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | ham10000 | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1204 | aug_svm_food_s1_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | food101 | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1205 | aug_svm_food_s2_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | food101 | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1206 | aug_svm_food_s3_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | food101 | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1207 | aug_svm_reggs_s1_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | Eggs | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1208 | aug_svm_reggs_s2_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | Eggs | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1209 | aug_svm_reggs_s3_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | Eggs | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1210 | aug_svm_rlarv_s1_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | Larvae | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1211 | aug_svm_rlarv_s2_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | Larvae | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1212 | aug_svm_rlarv_s3_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | Larvae | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1213 | aug_svm_rprot_s1_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | Protozoan | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1214 | aug_svm_rprot_s2_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | Protozoan | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1215 | aug_svm_rprot_s3_p50 | spf_grade rede 2, aug só no SVM | SPiFiL (FPS) | Protozoan | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1216 | aug_2_aid_s1_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | aid | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1217 | aug_2_aid_s2_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | aid | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1218 | aug_2_aid_s3_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | aid | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1219 | aug_2_rs_s1_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | remote_sensing | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1220 | aug_2_rs_s2_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | remote_sensing | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1221 | aug_2_rs_s3_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | remote_sensing | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1222 | aug_2_flw_s1_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | flowers | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1223 | aug_2_flw_s2_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | flowers | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1224 | aug_2_flw_s3_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | flowers | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1225 | aug_2_coco_s1_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | coconut_trees | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1226 | aug_2_coco_s2_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | coconut_trees | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1227 | aug_2_coco_s3_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | coconut_trees | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1228 | aug_2_ham_s1_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | ham10000 | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1229 | aug_2_ham_s2_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | ham10000 | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1230 | aug_2_ham_s3_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | ham10000 | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1231 | aug_2_food_s1_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | food101 | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1232 | aug_2_food_s2_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | food101 | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1233 | aug_2_food_s3_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | food101 | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1234 | aug_2_reggs_s1_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | Eggs | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1235 | aug_2_reggs_s2_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | Eggs | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1236 | aug_2_reggs_s3_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | Eggs | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1237 | aug_2_rlarv_s1_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | Larvae | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1238 | aug_2_rlarv_s2_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | Larvae | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1239 | aug_2_rlarv_s3_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | Larvae | 3 | `wait` | — | — | — | — | `wait` | — | — |
| 1240 | aug_2_rprot_s1_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | Protozoan | 1 | `wait` | — | — | — | — | `wait` | — | — |
| 1241 | aug_2_rprot_s2_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | Protozoan | 2 | `wait` | — | — | — | — | `wait` | — | — |
| 1242 | aug_2_rprot_s3_p50 | spf_grade rede 2, aug nos dois | SPiFiL (FPS) | Protozoan | 3 | `wait` | — | — | — | — | `wait` | — | — |

Subtotal: 108

## Como este arquivo é atualizado

- Só o agente `fila_md` escreve aqui, a cada mensagem do orquestrador, rodando `gera_fila.py`.
- A cada atualização o arquivo é reescrito inteiro: hora no topo, contadores por seção e por tag, quem roda em cada GPU, resultados do SVM.
- Só entra o que está nos arquivos/logs; campo sem dado fica `—`.
