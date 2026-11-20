# Atenção sem parâmetros: triagem (10/11/2026)

Atenção sem parâmetros, modo (a), aplicada na leitura do mesmo encoder: 5 mecanismos em L5 e no final (o mapa da L6 antes do GeM), com α = 0,5. A receita roda sem TTA, em 1 split, com 50% dos dados e o sorteio 42, e só na validação.
YAML: `experiments/autoencoder/disc_md_atencao_triagem.yaml`, com 6 jobs (3 bases × 2 shards), todos OK.
Código: `eval/spf_grade.py` (`ATTN`, `Attn`, `att_layers` e as leituras `L_att_*`/`cat_att_*`) e `experiments/ray/schema.py` (validação de `amostra.atencao`).
Identidade: sem atenção, uma célula pequena (aid, 2 camadas, largura 16) dá CSVs iguais antes e depois da mudança. São 12 de 12 linhas e 35 de 35 linhas de diagnóstico, sem diferença fora das colunas de tempo.

## Tabela completa (leitura `cat`; Δ contra `cat` sem atenção, no mesmo braço e na mesma camada)
Receita: fps em aid e remote_sensing, fpsw@mapa_w0.5_1 em flowers. Na coluna random_z, o Δ é o do random_z com a mesma atenção.
O tempo por imagem é o `extract_s` da passada dividido pelo número de imagens (treino + validação). Nas linhas com atenção ele cobre a passada inteira do encoder com a atenção; na linha "(nenhum)" cobre a extração normal da camada.

| base | pos | mecanismo | receita κ | Δκ | F1 | ΔF1 | random_z κ | Δκ | F1 | ΔF1 | ms/img (rec/rz) | fonte receita ; random_z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aid | L5 | (nenhum) | 0.772 | +0.000 | 0.779 | +0.000 | 0.652 | +0.000 | 0.667 | +0.000 | 20.7/29.6 | aid_s0200_split1_pct50_sh1.csv:18 ; aid_s0200_split1_pct50_sh0.csv:18 |
| aid | L5 | sa | 0.766 | -0.006 | 0.772 | -0.007 | 0.643 | -0.009 | 0.658 | -0.009 | 3.0/4.9 | aid_s0200_split1_pct50_sh1.csv:22 ; aid_s0200_split1_pct50_sh0.csv:22 |
| aid | L5 | espacial | 0.768 | -0.003 | 0.775 | -0.004 | 0.646 | -0.006 | 0.661 | -0.006 | 3.6/4.8 | aid_s0200_split1_pct50_sh1.csv:26 ; aid_s0200_split1_pct50_sh0.csv:26 |
| aid | L5 | simam | 0.770 | -0.001 | 0.778 | -0.001 | 0.652 | -0.000 | 0.667 | -0.001 | 3.4/4.8 | aid_s0200_split1_pct50_sh1.csv:30 ; aid_s0200_split1_pct50_sh0.csv:30 |
| aid | L5 | nonlocal_g | 0.776 | +0.004 | 0.782 | +0.003 | 0.663 | +0.012 | 0.678 | +0.010 | 3.6/5.1 | aid_s0200_split1_pct50_sh1.csv:34 ; aid_s0200_split1_pct50_sh0.csv:34 |
| aid | L5 | multiescala | 0.776 | +0.004 | 0.782 | +0.003 | 0.655 | +0.003 | 0.671 | +0.004 | 3.7/5.1 | aid_s0200_split1_pct50_sh1.csv:38 ; aid_s0200_split1_pct50_sh0.csv:38 |
| aid | final | (nenhum) | 0.774 | +0.000 | 0.781 | +0.000 | 0.647 | +0.000 | 0.662 | +0.000 | 25.5/32.6 | aid_s0200_split1_pct50_sh1.csv:42 ; aid_s0200_split1_pct50_sh0.csv:42 |
| aid | final | sa | 0.774 | -0.000 | 0.781 | +0.001 | 0.625 | -0.021 | 0.640 | -0.021 | 4.0/5.3 | aid_s0200_split1_pct50_sh1.csv:46 ; aid_s0200_split1_pct50_sh0.csv:46 |
| aid | final | espacial | 0.783 | +0.009 | 0.791 | +0.010 | 0.636 | -0.011 | 0.652 | -0.010 | 4.3/5.5 | aid_s0200_split1_pct50_sh1.csv:50 ; aid_s0200_split1_pct50_sh0.csv:50 |
| aid | final | simam | 0.769 | -0.006 | 0.776 | -0.005 | 0.641 | -0.006 | 0.655 | -0.007 | 4.4/5.8 | aid_s0200_split1_pct50_sh1.csv:54 ; aid_s0200_split1_pct50_sh0.csv:54 |
| aid | final | nonlocal_g | 0.778 | +0.003 | 0.785 | +0.005 | 0.640 | -0.007 | 0.653 | -0.008 | 4.1/5.6 | aid_s0200_split1_pct50_sh1.csv:58 ; aid_s0200_split1_pct50_sh0.csv:58 |
| aid | final | multiescala | 0.776 | +0.001 | 0.782 | +0.002 | 0.631 | -0.016 | 0.644 | -0.018 | 4.4/5.0 | aid_s0200_split1_pct50_sh1.csv:62 ; aid_s0200_split1_pct50_sh0.csv:62 |
| remote_sensing | L5 | (nenhum) | 0.426 | +0.000 | 0.413 | +0.000 | 0.460 | +0.000 | 0.436 | +0.000 | 20.7/25.4 | remote_sensing_s0200_split1_pct50_sh1.csv:18 ; remote_sensing_s0200_split1_pct50_sh0.csv:18 |
| remote_sensing | L5 | sa | 0.412 | -0.014 | 0.396 | -0.017 | 0.410 | -0.050 | 0.399 | -0.037 | 2.9/4.7 | remote_sensing_s0200_split1_pct50_sh1.csv:22 ; remote_sensing_s0200_split1_pct50_sh0.csv:22 |
| remote_sensing | L5 | espacial | 0.438 | +0.012 | 0.414 | +0.002 | 0.468 | +0.007 | 0.442 | +0.006 | 2.9/4.6 | remote_sensing_s0200_split1_pct50_sh1.csv:26 ; remote_sensing_s0200_split1_pct50_sh0.csv:26 |
| remote_sensing | L5 | simam | 0.453 | +0.027 | 0.429 | +0.016 | 0.489 | +0.029 | 0.482 | +0.046 | 3.8/4.6 | remote_sensing_s0200_split1_pct50_sh1.csv:30 ; remote_sensing_s0200_split1_pct50_sh0.csv:30 |
| remote_sensing | L5 | nonlocal_g | 0.425 | -0.001 | 0.404 | -0.008 | 0.452 | -0.008 | 0.429 | -0.007 | 4.1/3.8 | remote_sensing_s0200_split1_pct50_sh1.csv:34 ; remote_sensing_s0200_split1_pct50_sh0.csv:34 |
| remote_sensing | L5 | multiescala | 0.412 | -0.014 | 0.397 | -0.016 | 0.431 | -0.029 | 0.415 | -0.020 | 4.2/3.1 | remote_sensing_s0200_split1_pct50_sh1.csv:38 ; remote_sensing_s0200_split1_pct50_sh0.csv:38 |
| remote_sensing | final | (nenhum) | 0.425 | +0.000 | 0.396 | +0.000 | 0.447 | +0.000 | 0.420 | +0.000 | 18.4/24.0 | remote_sensing_s0200_split1_pct50_sh1.csv:42 ; remote_sensing_s0200_split1_pct50_sh0.csv:42 |
| remote_sensing | final | sa | 0.433 | +0.008 | 0.411 | +0.016 | 0.410 | -0.036 | 0.392 | -0.028 | 3.0/3.2 | remote_sensing_s0200_split1_pct50_sh1.csv:46 ; remote_sensing_s0200_split1_pct50_sh0.csv:46 |
| remote_sensing | final | espacial | 0.412 | -0.013 | 0.385 | -0.010 | 0.433 | -0.014 | 0.413 | -0.007 | 2.9/3.8 | remote_sensing_s0200_split1_pct50_sh1.csv:50 ; remote_sensing_s0200_split1_pct50_sh0.csv:50 |
| remote_sensing | final | simam | 0.412 | -0.014 | 0.389 | -0.007 | 0.461 | +0.015 | 0.434 | +0.014 | 2.9/4.6 | remote_sensing_s0200_split1_pct50_sh1.csv:54 ; remote_sensing_s0200_split1_pct50_sh0.csv:54 |
| remote_sensing | final | nonlocal_g | 0.440 | +0.015 | 0.414 | +0.018 | 0.410 | -0.037 | 0.395 | -0.025 | 3.0/4.6 | remote_sensing_s0200_split1_pct50_sh1.csv:58 ; remote_sensing_s0200_split1_pct50_sh0.csv:58 |
| remote_sensing | final | multiescala | 0.426 | +0.001 | 0.406 | +0.010 | 0.404 | -0.043 | 0.392 | -0.028 | 3.7/4.7 | remote_sensing_s0200_split1_pct50_sh1.csv:62 ; remote_sensing_s0200_split1_pct50_sh0.csv:62 |
| flowers | L5 | (nenhum) | 0.598 | +0.000 | 0.680 | +0.000 | 0.533 | +0.000 | 0.630 | +0.000 | 18.3/19.4 | flowers_s0200_split1_pct50_sh0.csv:80 ; flowers_s0200_split1_pct50_sh0.csv:18 |
| flowers | L5 | sa | 0.589 | -0.009 | 0.674 | -0.006 | 0.510 | -0.024 | 0.609 | -0.021 | 4.0/4.1 | flowers_s0200_split1_pct50_sh0.csv:84 ; flowers_s0200_split1_pct50_sh0.csv:22 |
| flowers | L5 | espacial | 0.619 | +0.021 | 0.698 | +0.018 | 0.512 | -0.021 | 0.612 | -0.017 | 2.9/3.6 | flowers_s0200_split1_pct50_sh0.csv:88 ; flowers_s0200_split1_pct50_sh0.csv:26 |
| flowers | L5 | simam | 0.601 | +0.003 | 0.683 | +0.003 | 0.528 | -0.006 | 0.625 | -0.005 | 2.8/2.8 | flowers_s0200_split1_pct50_sh0.csv:92 ; flowers_s0200_split1_pct50_sh0.csv:30 |
| flowers | L5 | nonlocal_g | 0.595 | -0.003 | 0.678 | -0.002 | 0.540 | +0.006 | 0.636 | +0.006 | 2.9/2.9 | flowers_s0200_split1_pct50_sh0.csv:96 ; flowers_s0200_split1_pct50_sh0.csv:34 |
| flowers | L5 | multiescala | 0.595 | -0.003 | 0.680 | -0.000 | 0.488 | -0.045 | 0.593 | -0.037 | 2.9/2.9 | flowers_s0200_split1_pct50_sh0.csv:100 ; flowers_s0200_split1_pct50_sh0.csv:38 |
| flowers | final | (nenhum) | 0.628 | +0.000 | 0.704 | +0.000 | 0.546 | +0.000 | 0.641 | +0.000 | 20.3/18.9 | flowers_s0200_split1_pct50_sh0.csv:104 ; flowers_s0200_split1_pct50_sh0.csv:42 |
| flowers | final | sa | 0.625 | -0.003 | 0.703 | -0.001 | 0.522 | -0.024 | 0.622 | -0.019 | 3.2/2.9 | flowers_s0200_split1_pct50_sh0.csv:108 ; flowers_s0200_split1_pct50_sh0.csv:46 |
| flowers | final | espacial | 0.631 | +0.003 | 0.706 | +0.003 | 0.559 | +0.013 | 0.651 | +0.010 | 2.9/2.9 | flowers_s0200_split1_pct50_sh0.csv:112 ; flowers_s0200_split1_pct50_sh0.csv:50 |
| flowers | final | simam | 0.622 | -0.006 | 0.698 | -0.005 | 0.525 | -0.021 | 0.623 | -0.018 | 2.9/3.0 | flowers_s0200_split1_pct50_sh0.csv:116 ; flowers_s0200_split1_pct50_sh0.csv:54 |
| flowers | final | nonlocal_g | 0.619 | -0.009 | 0.696 | -0.007 | 0.546 | +0.000 | 0.641 | +0.000 | 3.0/4.0 | flowers_s0200_split1_pct50_sh0.csv:120 ; flowers_s0200_split1_pct50_sh0.csv:58 |
| flowers | final | multiescala | 0.619 | -0.009 | 0.699 | -0.005 | 0.531 | -0.015 | 0.628 | -0.013 | 4.2/2.9 | flowers_s0200_split1_pct50_sh0.csv:124 ; flowers_s0200_split1_pct50_sh0.csv:62 |

## Memória de pico
O pico foi 2673 MiB (`gpu_mib`), abaixo do teto de 2800. A atenção quase não aumenta esse valor: o pico vem da L1 com 200 px.

## Diversidade das features
Não foi medida. Os encoders só existem num diretório temporário dentro de `chain`: `artifacts_dir` fica vazio, e as features não são salvas. Para medir o posto efetivo e a correlação entre canais seria preciso refazer os encoders ou mudar o código para salvá-los, e isso não é barato.

## Leitura rápida
- `sa` e `multiescala` perdem em quase toda célula, até −0,05 κ no random_z.
- `espacial` e `simam` são os únicos com ganho somado na receita em L5, de +0,030 e +0,029 κ nas 3 bases. São um split só, e o ruído é de cerca de ±0,02.
- Vão para a confirmação em 3 splits: `espacial` e `simam`, em L5 e no final (`disc_md_atencao_conf.yaml`).

## Confirmação em 3 splits (`disc_md_atencao_conf.yaml`, 18 jobs OK, END 20:00:58)
Espacial e simam em L5 e no final, com α = 0,5, 50%, sorteio 42 e só validação. Leitura `cat`; o Δ é contra `cat` sem atenção no mesmo split, braço e camada. A coluna "splits com ganho" conta os splits com Δκ > 0. Para random_z, a régua é o random_z sem técnica, e as linhas dele com atenção são só controle.

| base | braço | pos | mecanismo | κ médio | Δκ médio | splits com ganho κ | F1 médio | ΔF1 médio | fontes (split 1..3) |
|---|---|---|---|---|---|---|---|---|---|
| aid | receita | L5 | (nenhum) | 0.767 | +0.000 | 0/3 | 0.776 | +0.000 | conf_aid_s0200_split1_pct50_sh1.csv:18, conf_aid_s0200_split2_pct50_sh1.csv:18, conf_aid_s0200_split3_pct50_sh1.csv:18 |
| aid | receita | L5 | espacial | 0.761 | -0.006 | 0/3 | 0.771 | -0.005 | conf_aid_s0200_split1_pct50_sh1.csv:22, conf_aid_s0200_split2_pct50_sh1.csv:22, conf_aid_s0200_split3_pct50_sh1.csv:22 |
| aid | receita | L5 | simam | 0.767 | -0.000 | 1/3 | 0.776 | -0.000 | conf_aid_s0200_split1_pct50_sh1.csv:26, conf_aid_s0200_split2_pct50_sh1.csv:26, conf_aid_s0200_split3_pct50_sh1.csv:26 |
| aid | receita | final | (nenhum) | 0.771 | +0.000 | 0/3 | 0.779 | +0.000 | conf_aid_s0200_split1_pct50_sh1.csv:30, conf_aid_s0200_split2_pct50_sh1.csv:30, conf_aid_s0200_split3_pct50_sh1.csv:30 |
| aid | receita | final | espacial | 0.771 | +0.000 | 1/3 | 0.781 | +0.001 | conf_aid_s0200_split1_pct50_sh1.csv:34, conf_aid_s0200_split2_pct50_sh1.csv:34, conf_aid_s0200_split3_pct50_sh1.csv:34 |
| aid | receita | final | simam | 0.771 | +0.000 | 2/3 | 0.780 | +0.000 | conf_aid_s0200_split1_pct50_sh1.csv:38, conf_aid_s0200_split2_pct50_sh1.csv:38, conf_aid_s0200_split3_pct50_sh1.csv:38 |
| aid | random_z | L5 | (nenhum) | 0.643 | +0.000 | 0/3 | 0.657 | +0.000 | conf_aid_s0200_split1_pct50_sh0.csv:18, conf_aid_s0200_split2_pct50_sh0.csv:18, conf_aid_s0200_split3_pct50_sh0.csv:18 |
| aid | random_z | L5 | espacial | 0.636 | -0.007 | 0/3 | 0.651 | -0.006 | conf_aid_s0200_split1_pct50_sh0.csv:22, conf_aid_s0200_split2_pct50_sh0.csv:22, conf_aid_s0200_split3_pct50_sh0.csv:22 |
| aid | random_z | L5 | simam | 0.641 | -0.002 | 1/3 | 0.654 | -0.003 | conf_aid_s0200_split1_pct50_sh0.csv:26, conf_aid_s0200_split2_pct50_sh0.csv:26, conf_aid_s0200_split3_pct50_sh0.csv:26 |
| aid | random_z | final | (nenhum) | 0.636 | +0.000 | 0/3 | 0.650 | +0.000 | conf_aid_s0200_split1_pct50_sh0.csv:30, conf_aid_s0200_split2_pct50_sh0.csv:30, conf_aid_s0200_split3_pct50_sh0.csv:30 |
| aid | random_z | final | espacial | 0.635 | -0.001 | 1/3 | 0.650 | -0.001 | conf_aid_s0200_split1_pct50_sh0.csv:34, conf_aid_s0200_split2_pct50_sh0.csv:34, conf_aid_s0200_split3_pct50_sh0.csv:34 |
| aid | random_z | final | simam | 0.631 | -0.005 | 0/3 | 0.646 | -0.005 | conf_aid_s0200_split1_pct50_sh0.csv:38, conf_aid_s0200_split2_pct50_sh0.csv:38, conf_aid_s0200_split3_pct50_sh0.csv:38 |
| remote_sensing | receita | L5 | (nenhum) | 0.484 | +0.000 | 0/3 | 0.453 | +0.000 | conf_remote_sensing_s0200_split1_pct50_sh1.csv:18, conf_remote_sensing_s0200_split2_pct50_sh1.csv:18, conf_remote_sensing_s0200_split3_pct50_sh1.csv:18 |
| remote_sensing | receita | L5 | espacial | 0.490 | +0.007 | 2/3 | 0.454 | +0.000 | conf_remote_sensing_s0200_split1_pct50_sh1.csv:22, conf_remote_sensing_s0200_split2_pct50_sh1.csv:22, conf_remote_sensing_s0200_split3_pct50_sh1.csv:22 |
| remote_sensing | receita | L5 | simam | 0.485 | +0.001 | 1/3 | 0.451 | -0.003 | conf_remote_sensing_s0200_split1_pct50_sh1.csv:26, conf_remote_sensing_s0200_split2_pct50_sh1.csv:26, conf_remote_sensing_s0200_split3_pct50_sh1.csv:26 |
| remote_sensing | receita | final | (nenhum) | 0.492 | +0.000 | 0/3 | 0.459 | +0.000 | conf_remote_sensing_s0200_split1_pct50_sh1.csv:30, conf_remote_sensing_s0200_split2_pct50_sh1.csv:30, conf_remote_sensing_s0200_split3_pct50_sh1.csv:30 |
| remote_sensing | receita | final | espacial | 0.476 | -0.015 | 0/3 | 0.443 | -0.016 | conf_remote_sensing_s0200_split1_pct50_sh1.csv:34, conf_remote_sensing_s0200_split2_pct50_sh1.csv:34, conf_remote_sensing_s0200_split3_pct50_sh1.csv:34 |
| remote_sensing | receita | final | simam | 0.482 | -0.010 | 1/3 | 0.447 | -0.012 | conf_remote_sensing_s0200_split1_pct50_sh1.csv:38, conf_remote_sensing_s0200_split2_pct50_sh1.csv:38, conf_remote_sensing_s0200_split3_pct50_sh1.csv:38 |
| remote_sensing | random_z | L5 | (nenhum) | 0.473 | +0.000 | 0/3 | 0.449 | +0.000 | conf_remote_sensing_s0200_split1_pct50_sh0.csv:18, conf_remote_sensing_s0200_split2_pct50_sh0.csv:18, conf_remote_sensing_s0200_split3_pct50_sh0.csv:18 |
| remote_sensing | random_z | L5 | espacial | 0.486 | +0.012 | 3/3 | 0.453 | +0.004 | conf_remote_sensing_s0200_split1_pct50_sh0.csv:22, conf_remote_sensing_s0200_split2_pct50_sh0.csv:22, conf_remote_sensing_s0200_split3_pct50_sh0.csv:22 |
| remote_sensing | random_z | L5 | simam | 0.499 | +0.026 | 3/3 | 0.480 | +0.031 | conf_remote_sensing_s0200_split1_pct50_sh0.csv:26, conf_remote_sensing_s0200_split2_pct50_sh0.csv:26, conf_remote_sensing_s0200_split3_pct50_sh0.csv:26 |
| remote_sensing | random_z | final | (nenhum) | 0.487 | +0.000 | 0/3 | 0.465 | +0.000 | conf_remote_sensing_s0200_split1_pct50_sh0.csv:30, conf_remote_sensing_s0200_split2_pct50_sh0.csv:30, conf_remote_sensing_s0200_split3_pct50_sh0.csv:30 |
| remote_sensing | random_z | final | espacial | 0.478 | -0.008 | 0/3 | 0.464 | -0.001 | conf_remote_sensing_s0200_split1_pct50_sh0.csv:34, conf_remote_sensing_s0200_split2_pct50_sh0.csv:34, conf_remote_sensing_s0200_split3_pct50_sh0.csv:34 |
| remote_sensing | random_z | final | simam | 0.483 | -0.003 | 1/3 | 0.465 | -0.000 | conf_remote_sensing_s0200_split1_pct50_sh0.csv:38, conf_remote_sensing_s0200_split2_pct50_sh0.csv:38, conf_remote_sensing_s0200_split3_pct50_sh0.csv:38 |
| flowers | receita | L5 | (nenhum) | 0.638 | +0.000 | 0/3 | 0.712 | +0.000 | conf_flowers_s0200_split1_pct50_sh0.csv:56, conf_flowers_s0200_split2_pct50_sh0.csv:56, conf_flowers_s0200_split3_pct50_sh0.csv:56 |
| flowers | receita | L5 | espacial | 0.632 | -0.006 | 2/3 | 0.709 | -0.003 | conf_flowers_s0200_split1_pct50_sh0.csv:60, conf_flowers_s0200_split2_pct50_sh0.csv:60, conf_flowers_s0200_split3_pct50_sh0.csv:60 |
| flowers | receita | L5 | simam | 0.645 | +0.006 | 3/3 | 0.717 | +0.005 | conf_flowers_s0200_split1_pct50_sh0.csv:64, conf_flowers_s0200_split2_pct50_sh0.csv:64, conf_flowers_s0200_split3_pct50_sh0.csv:64 |
| flowers | receita | final | (nenhum) | 0.654 | +0.000 | 0/3 | 0.725 | +0.000 | conf_flowers_s0200_split1_pct50_sh0.csv:68, conf_flowers_s0200_split2_pct50_sh0.csv:68, conf_flowers_s0200_split3_pct50_sh0.csv:68 |
| flowers | receita | final | espacial | 0.657 | +0.003 | 2/3 | 0.727 | +0.002 | conf_flowers_s0200_split1_pct50_sh0.csv:72, conf_flowers_s0200_split2_pct50_sh0.csv:72, conf_flowers_s0200_split3_pct50_sh0.csv:72 |
| flowers | receita | final | simam | 0.649 | -0.005 | 0/3 | 0.721 | -0.004 | conf_flowers_s0200_split1_pct50_sh0.csv:76, conf_flowers_s0200_split2_pct50_sh0.csv:76, conf_flowers_s0200_split3_pct50_sh0.csv:76 |
| flowers | random_z | L5 | (nenhum) | 0.549 | +0.000 | 0/3 | 0.640 | +0.000 | conf_flowers_s0200_split1_pct50_sh0.csv:18, conf_flowers_s0200_split2_pct50_sh0.csv:18, conf_flowers_s0200_split3_pct50_sh0.csv:18 |
| flowers | random_z | L5 | espacial | 0.524 | -0.025 | 0/3 | 0.620 | -0.021 | conf_flowers_s0200_split1_pct50_sh0.csv:22, conf_flowers_s0200_split2_pct50_sh0.csv:22, conf_flowers_s0200_split3_pct50_sh0.csv:22 |
| flowers | random_z | L5 | simam | 0.546 | -0.003 | 1/3 | 0.638 | -0.003 | conf_flowers_s0200_split1_pct50_sh0.csv:26, conf_flowers_s0200_split2_pct50_sh0.csv:26, conf_flowers_s0200_split3_pct50_sh0.csv:26 |
| flowers | random_z | final | (nenhum) | 0.549 | +0.000 | 0/3 | 0.641 | +0.000 | conf_flowers_s0200_split1_pct50_sh0.csv:30, conf_flowers_s0200_split2_pct50_sh0.csv:30, conf_flowers_s0200_split3_pct50_sh0.csv:30 |
| flowers | random_z | final | espacial | 0.550 | +0.001 | 1/3 | 0.643 | +0.001 | conf_flowers_s0200_split1_pct50_sh0.csv:34, conf_flowers_s0200_split2_pct50_sh0.csv:34, conf_flowers_s0200_split3_pct50_sh0.csv:34 |
| flowers | random_z | final | simam | 0.551 | +0.002 | 2/3 | 0.642 | +0.001 | conf_flowers_s0200_split1_pct50_sh0.csv:38, conf_flowers_s0200_split2_pct50_sh0.csv:38, conf_flowers_s0200_split3_pct50_sh0.csv:38 |

Leitura: na receita, nenhuma das 4 combinações (mecanismo × posição) ganha nas 3 bases. O melhor Δκ médio é +0,007 (espacial em L5, remote_sensing), e o pior é −0,015 (espacial no final, remote_sensing). Os ganhos da triagem eram ruído de 1 split.
O único sinal que se repete nos 3 splits é simam em L5 no remote_sensing, mas só no random_z (+0,026, 3 de 3 splits); na receita ele dá +0,001.
Recomendação: a atenção sem parâmetros no modo (a) não entra na receita. Se for testar de novo, o candidato é simam em L5 no modo (b), com o whitening estimado sobre o mapa já atendido.

Figura dos mapas (`mapas_atencao.png`): não foi feita. Os encoders só existem num diretório temporário dentro de `chain` (`artifacts_dir` fica vazio). Para desenhar os mapas seria preciso refazer uma célula inteira ou mudar o código para salvar o encoder.

## Combinação (pedido do usuário: atenção + TTA + 320 px)
YAMLs: `experiments/autoencoder/disc_md_atencao_combo.yaml` (200 px, 3 bases, 18 jobs OK, END 22:05) e `disc_md_atencao_combo320.yaml` (320 px, só aid e remote_sensing, receita fps sem saliência como em `pf_res320`, 12 jobs OK, END 00:18 de 11/11). Os dois rodam com 3 splits, 50%, sorteio 42 e só validação.
Código: a TTA agora passa pelo `Attn` e emite `L_tta_att_*`/`cat_tta_att_*` pareados (o `cat` usa as camadas anteriores também com TTA). A identidade com as opções desligadas continua valendo: 12 de 12 e 35 de 35 linhas iguais. A célula pequena com TTA e atenção deu 104 linhas, o número esperado.
Sem whitening em L4/L5 (só aid) ficou fora: é um mecanismo do `encdeep` (`z0~nw`), que usa outro tronco (5 camadas, kernels 5-5-5-3-3). Não liga junto com a receita L6.
O Δ é sempre contra a receita a 200 px (L6 `cat`) do mesmo split e braço. Na posição L5, a variante é o `cat` até L5.
Memória de pico: 2679 MiB a 200 px. A 320 px o CSV não grava `gpu_mib` (a coluna não está no cabeçalho), mas o teto de 2800 MiB do `gpucap` valeu e não houve nenhum OOM.

| base | braço | variante | κ médio | Δκ vs receita 200 px (L6 cat) | splits com ganho | F1 médio | ΔF1 | fontes (split 1..3) |
|---|---|---|---|---|---|---|---|---|
| aid | receita | receita | 0.771 | +0.000 | 0/3 | 0.779 | +0.000 | combo_aid_s0200_split1_pct50_sh1.csv:56, combo_aid_s0200_split2_pct50_sh1.csv:56, combo_aid_s0200_split3_pct50_sh1.csv:56 |
| aid | receita | +TTA | 0.844 | +0.073 | 3/3 | 0.850 | +0.071 | combo_aid_s0200_split1_pct50_sh1.csv:60, combo_aid_s0200_split2_pct50_sh1.csv:60, combo_aid_s0200_split3_pct50_sh1.csv:60 |
| aid | receita | +atenção espacial@L5 | 0.760 | -0.011 | 0/3 | 0.771 | -0.009 | combo_aid_s0200_split1_pct50_sh1.csv:40, combo_aid_s0200_split2_pct50_sh1.csv:40, combo_aid_s0200_split3_pct50_sh1.csv:40 |
| aid | receita | +atenção espacial@L5 +TTA | 0.829 | +0.058 | 3/3 | 0.835 | +0.055 | combo_aid_s0200_split1_pct50_sh1.csv:44, combo_aid_s0200_split2_pct50_sh1.csv:44, combo_aid_s0200_split3_pct50_sh1.csv:44 |
| aid | receita | +atenção espacial@final | 0.771 | +0.000 | 1/3 | 0.781 | +0.001 | combo_aid_s0200_split1_pct50_sh1.csv:64, combo_aid_s0200_split2_pct50_sh1.csv:64, combo_aid_s0200_split3_pct50_sh1.csv:64 |
| aid | receita | +atenção espacial@final +TTA | 0.840 | +0.069 | 3/3 | 0.846 | +0.066 | combo_aid_s0200_split1_pct50_sh1.csv:68, combo_aid_s0200_split2_pct50_sh1.csv:68, combo_aid_s0200_split3_pct50_sh1.csv:68 |
| aid | receita | +atenção simam@L5 | 0.767 | -0.004 | 1/3 | 0.776 | -0.004 | combo_aid_s0200_split1_pct50_sh1.csv:48, combo_aid_s0200_split2_pct50_sh1.csv:48, combo_aid_s0200_split3_pct50_sh1.csv:48 |
| aid | receita | +atenção simam@L5 +TTA | 0.835 | +0.064 | 3/3 | 0.840 | +0.061 | combo_aid_s0200_split1_pct50_sh1.csv:52, combo_aid_s0200_split2_pct50_sh1.csv:52, combo_aid_s0200_split3_pct50_sh1.csv:52 |
| aid | receita | +atenção simam@final | 0.771 | +0.000 | 2/3 | 0.780 | +0.000 | combo_aid_s0200_split1_pct50_sh1.csv:72, combo_aid_s0200_split2_pct50_sh1.csv:72, combo_aid_s0200_split3_pct50_sh1.csv:72 |
| aid | receita | +atenção simam@final +TTA | 0.843 | +0.072 | 3/3 | 0.848 | +0.069 | combo_aid_s0200_split1_pct50_sh1.csv:76, combo_aid_s0200_split2_pct50_sh1.csv:76, combo_aid_s0200_split3_pct50_sh1.csv:76 |
| aid | receita | +320 px | 0.785 | +0.014 | 3/3 | 0.793 | +0.014 | combo320_aid_s0320_split1_pct50_sh1.csv:56, combo320_aid_s0320_split2_pct50_sh1.csv:56, combo320_aid_s0320_split3_pct50_sh1.csv:56 |
| aid | receita | +320 px +TTA | 0.832 | +0.060 | 3/3 | 0.838 | +0.059 | combo320_aid_s0320_split1_pct50_sh1.csv:60, combo320_aid_s0320_split2_pct50_sh1.csv:60, combo320_aid_s0320_split3_pct50_sh1.csv:60 |
| aid | receita | tudo: 320 px + espacial@L5 + TTA | 0.825 | +0.054 | 3/3 | 0.831 | +0.052 | combo320_aid_s0320_split1_pct50_sh1.csv:44, combo320_aid_s0320_split2_pct50_sh1.csv:44, combo320_aid_s0320_split3_pct50_sh1.csv:44 |
| aid | receita | tudo: 320 px + espacial@final + TTA | 0.836 | +0.065 | 3/3 | 0.842 | +0.063 | combo320_aid_s0320_split1_pct50_sh1.csv:68, combo320_aid_s0320_split2_pct50_sh1.csv:68, combo320_aid_s0320_split3_pct50_sh1.csv:68 |
| aid | receita | tudo: 320 px + simam@L5 + TTA | 0.827 | +0.056 | 3/3 | 0.833 | +0.054 | combo320_aid_s0320_split1_pct50_sh1.csv:52, combo320_aid_s0320_split2_pct50_sh1.csv:52, combo320_aid_s0320_split3_pct50_sh1.csv:52 |
| aid | receita | tudo: 320 px + simam@final + TTA | 0.839 | +0.068 | 3/3 | 0.845 | +0.066 | combo320_aid_s0320_split1_pct50_sh1.csv:76, combo320_aid_s0320_split2_pct50_sh1.csv:76, combo320_aid_s0320_split3_pct50_sh1.csv:76 |
| aid | random_z | receita | 0.636 | +0.000 | 0/3 | 0.650 | +0.000 | combo_aid_s0200_split1_pct50_sh0.csv:56, combo_aid_s0200_split2_pct50_sh0.csv:56, combo_aid_s0200_split3_pct50_sh0.csv:56 |
| aid | random_z | +TTA | 0.764 | +0.128 | 3/3 | 0.774 | +0.123 | combo_aid_s0200_split1_pct50_sh0.csv:60, combo_aid_s0200_split2_pct50_sh0.csv:60, combo_aid_s0200_split3_pct50_sh0.csv:60 |
| aid | random_z | +atenção espacial@L5 | 0.636 | +0.000 | 2/3 | 0.651 | +0.001 | combo_aid_s0200_split1_pct50_sh0.csv:40, combo_aid_s0200_split2_pct50_sh0.csv:40, combo_aid_s0200_split3_pct50_sh0.csv:40 |
| aid | random_z | +atenção espacial@L5 +TTA | 0.763 | +0.127 | 3/3 | 0.772 | +0.122 | combo_aid_s0200_split1_pct50_sh0.csv:44, combo_aid_s0200_split2_pct50_sh0.csv:44, combo_aid_s0200_split3_pct50_sh0.csv:44 |
| aid | random_z | +atenção espacial@final | 0.634 | -0.001 | 1/3 | 0.649 | -0.001 | combo_aid_s0200_split1_pct50_sh0.csv:64, combo_aid_s0200_split2_pct50_sh0.csv:64, combo_aid_s0200_split3_pct50_sh0.csv:64 |
| aid | random_z | +atenção espacial@final +TTA | 0.766 | +0.130 | 3/3 | 0.775 | +0.125 | combo_aid_s0200_split1_pct50_sh0.csv:68, combo_aid_s0200_split2_pct50_sh0.csv:68, combo_aid_s0200_split3_pct50_sh0.csv:68 |
| aid | random_z | +atenção simam@L5 | 0.641 | +0.005 | 2/3 | 0.654 | +0.004 | combo_aid_s0200_split1_pct50_sh0.csv:48, combo_aid_s0200_split2_pct50_sh0.csv:48, combo_aid_s0200_split3_pct50_sh0.csv:48 |
| aid | random_z | +atenção simam@L5 +TTA | 0.765 | +0.129 | 3/3 | 0.774 | +0.124 | combo_aid_s0200_split1_pct50_sh0.csv:52, combo_aid_s0200_split2_pct50_sh0.csv:52, combo_aid_s0200_split3_pct50_sh0.csv:52 |
| aid | random_z | +atenção simam@final | 0.631 | -0.005 | 0/3 | 0.645 | -0.005 | combo_aid_s0200_split1_pct50_sh0.csv:72, combo_aid_s0200_split2_pct50_sh0.csv:72, combo_aid_s0200_split3_pct50_sh0.csv:72 |
| aid | random_z | +atenção simam@final +TTA | 0.767 | +0.131 | 3/3 | 0.776 | +0.126 | combo_aid_s0200_split1_pct50_sh0.csv:76, combo_aid_s0200_split2_pct50_sh0.csv:76, combo_aid_s0200_split3_pct50_sh0.csv:76 |
| aid | random_z | +320 px | 0.673 | +0.037 | 3/3 | 0.684 | +0.034 | combo320_aid_s0320_split1_pct50_sh0.csv:56, combo320_aid_s0320_split2_pct50_sh0.csv:56, combo320_aid_s0320_split3_pct50_sh0.csv:56 |
| aid | random_z | +320 px +TTA | 0.771 | +0.135 | 3/3 | 0.779 | +0.129 | combo320_aid_s0320_split1_pct50_sh0.csv:60, combo320_aid_s0320_split2_pct50_sh0.csv:60, combo320_aid_s0320_split3_pct50_sh0.csv:60 |
| aid | random_z | tudo: 320 px + espacial@L5 + TTA | 0.763 | +0.127 | 3/3 | 0.771 | +0.121 | combo320_aid_s0320_split1_pct50_sh0.csv:44, combo320_aid_s0320_split2_pct50_sh0.csv:44, combo320_aid_s0320_split3_pct50_sh0.csv:44 |
| aid | random_z | tudo: 320 px + espacial@final + TTA | 0.772 | +0.136 | 3/3 | 0.780 | +0.130 | combo320_aid_s0320_split1_pct50_sh0.csv:68, combo320_aid_s0320_split2_pct50_sh0.csv:68, combo320_aid_s0320_split3_pct50_sh0.csv:68 |
| aid | random_z | tudo: 320 px + simam@L5 + TTA | 0.764 | +0.129 | 3/3 | 0.772 | +0.122 | combo320_aid_s0320_split1_pct50_sh0.csv:52, combo320_aid_s0320_split2_pct50_sh0.csv:52, combo320_aid_s0320_split3_pct50_sh0.csv:52 |
| aid | random_z | tudo: 320 px + simam@final + TTA | 0.771 | +0.135 | 3/3 | 0.779 | +0.128 | combo320_aid_s0320_split1_pct50_sh0.csv:76, combo320_aid_s0320_split2_pct50_sh0.csv:76, combo320_aid_s0320_split3_pct50_sh0.csv:76 |
| remote_sensing | receita | receita | 0.492 | +0.000 | 0/3 | 0.459 | +0.000 | combo_remote_sensing_s0200_split1_pct50_sh1.csv:56, combo_remote_sensing_s0200_split2_pct50_sh1.csv:56, combo_remote_sensing_s0200_split3_pct50_sh1.csv:56 |
| remote_sensing | receita | +TTA | 0.539 | +0.047 | 2/3 | 0.524 | +0.065 | combo_remote_sensing_s0200_split1_pct50_sh1.csv:60, combo_remote_sensing_s0200_split2_pct50_sh1.csv:60, combo_remote_sensing_s0200_split3_pct50_sh1.csv:60 |
| remote_sensing | receita | +atenção espacial@L5 | 0.490 | -0.001 | 2/3 | 0.454 | -0.005 | combo_remote_sensing_s0200_split1_pct50_sh1.csv:40, combo_remote_sensing_s0200_split2_pct50_sh1.csv:40, combo_remote_sensing_s0200_split3_pct50_sh1.csv:40 |
| remote_sensing | receita | +atenção espacial@L5 +TTA | 0.541 | +0.050 | 2/3 | 0.514 | +0.055 | combo_remote_sensing_s0200_split1_pct50_sh1.csv:44, combo_remote_sensing_s0200_split2_pct50_sh1.csv:44, combo_remote_sensing_s0200_split3_pct50_sh1.csv:44 |
| remote_sensing | receita | +atenção espacial@final | 0.476 | -0.015 | 0/3 | 0.443 | -0.016 | combo_remote_sensing_s0200_split1_pct50_sh1.csv:64, combo_remote_sensing_s0200_split2_pct50_sh1.csv:64, combo_remote_sensing_s0200_split3_pct50_sh1.csv:64 |
| remote_sensing | receita | +atenção espacial@final +TTA | 0.539 | +0.047 | 3/3 | 0.517 | +0.058 | combo_remote_sensing_s0200_split1_pct50_sh1.csv:68, combo_remote_sensing_s0200_split2_pct50_sh1.csv:68, combo_remote_sensing_s0200_split3_pct50_sh1.csv:68 |
| remote_sensing | receita | +atenção simam@L5 | 0.485 | -0.007 | 1/3 | 0.451 | -0.008 | combo_remote_sensing_s0200_split1_pct50_sh1.csv:48, combo_remote_sensing_s0200_split2_pct50_sh1.csv:48, combo_remote_sensing_s0200_split3_pct50_sh1.csv:48 |
| remote_sensing | receita | +atenção simam@L5 +TTA | 0.536 | +0.044 | 2/3 | 0.510 | +0.051 | combo_remote_sensing_s0200_split1_pct50_sh1.csv:52, combo_remote_sensing_s0200_split2_pct50_sh1.csv:52, combo_remote_sensing_s0200_split3_pct50_sh1.csv:52 |
| remote_sensing | receita | +atenção simam@final | 0.482 | -0.010 | 1/3 | 0.447 | -0.012 | combo_remote_sensing_s0200_split1_pct50_sh1.csv:72, combo_remote_sensing_s0200_split2_pct50_sh1.csv:72, combo_remote_sensing_s0200_split3_pct50_sh1.csv:72 |
| remote_sensing | receita | +atenção simam@final +TTA | 0.539 | +0.048 | 3/3 | 0.521 | +0.062 | combo_remote_sensing_s0200_split1_pct50_sh1.csv:76, combo_remote_sensing_s0200_split2_pct50_sh1.csv:76, combo_remote_sensing_s0200_split3_pct50_sh1.csv:76 |
| remote_sensing | receita | +320 px | 0.525 | +0.033 | 2/3 | 0.495 | +0.036 | combo320_remote_sensing_s0320_split1_pct50_sh1.csv:56, combo320_remote_sensing_s0320_split2_pct50_sh1.csv:56, combo320_remote_sensing_s0320_split3_pct50_sh1.csv:56 |
| remote_sensing | receita | +320 px +TTA | 0.545 | +0.053 | 3/3 | 0.512 | +0.053 | combo320_remote_sensing_s0320_split1_pct50_sh1.csv:60, combo320_remote_sensing_s0320_split2_pct50_sh1.csv:60, combo320_remote_sensing_s0320_split3_pct50_sh1.csv:60 |
| remote_sensing | receita | tudo: 320 px + espacial@L5 + TTA | 0.571 | +0.079 | 3/3 | 0.537 | +0.078 | combo320_remote_sensing_s0320_split1_pct50_sh1.csv:44, combo320_remote_sensing_s0320_split2_pct50_sh1.csv:44, combo320_remote_sensing_s0320_split3_pct50_sh1.csv:44 |
| remote_sensing | receita | tudo: 320 px + espacial@final + TTA | 0.552 | +0.060 | 3/3 | 0.522 | +0.063 | combo320_remote_sensing_s0320_split1_pct50_sh1.csv:68, combo320_remote_sensing_s0320_split2_pct50_sh1.csv:68, combo320_remote_sensing_s0320_split3_pct50_sh1.csv:68 |
| remote_sensing | receita | tudo: 320 px + simam@L5 + TTA | 0.569 | +0.077 | 3/3 | 0.530 | +0.071 | combo320_remote_sensing_s0320_split1_pct50_sh1.csv:52, combo320_remote_sensing_s0320_split2_pct50_sh1.csv:52, combo320_remote_sensing_s0320_split3_pct50_sh1.csv:52 |
| remote_sensing | receita | tudo: 320 px + simam@final + TTA | 0.552 | +0.060 | 3/3 | 0.521 | +0.062 | combo320_remote_sensing_s0320_split1_pct50_sh1.csv:76, combo320_remote_sensing_s0320_split2_pct50_sh1.csv:76, combo320_remote_sensing_s0320_split3_pct50_sh1.csv:76 |
| remote_sensing | random_z | receita | 0.487 | +0.000 | 0/3 | 0.465 | +0.000 | combo_remote_sensing_s0200_split1_pct50_sh0.csv:56, combo_remote_sensing_s0200_split2_pct50_sh0.csv:56, combo_remote_sensing_s0200_split3_pct50_sh0.csv:56 |
| remote_sensing | random_z | +TTA | 0.501 | +0.015 | 2/3 | 0.476 | +0.011 | combo_remote_sensing_s0200_split1_pct50_sh0.csv:60, combo_remote_sensing_s0200_split2_pct50_sh0.csv:60, combo_remote_sensing_s0200_split3_pct50_sh0.csv:60 |
| remote_sensing | random_z | +atenção espacial@L5 | 0.486 | -0.001 | 2/3 | 0.453 | -0.012 | combo_remote_sensing_s0200_split1_pct50_sh0.csv:40, combo_remote_sensing_s0200_split2_pct50_sh0.csv:40, combo_remote_sensing_s0200_split3_pct50_sh0.csv:40 |
| remote_sensing | random_z | +atenção espacial@L5 +TTA | 0.514 | +0.027 | 2/3 | 0.485 | +0.019 | combo_remote_sensing_s0200_split1_pct50_sh0.csv:44, combo_remote_sensing_s0200_split2_pct50_sh0.csv:44, combo_remote_sensing_s0200_split3_pct50_sh0.csv:44 |
| remote_sensing | random_z | +atenção espacial@final | 0.478 | -0.008 | 0/3 | 0.464 | -0.001 | combo_remote_sensing_s0200_split1_pct50_sh0.csv:64, combo_remote_sensing_s0200_split2_pct50_sh0.csv:64, combo_remote_sensing_s0200_split3_pct50_sh0.csv:64 |
| remote_sensing | random_z | +atenção espacial@final +TTA | 0.513 | +0.026 | 2/3 | 0.485 | +0.019 | combo_remote_sensing_s0200_split1_pct50_sh0.csv:68, combo_remote_sensing_s0200_split2_pct50_sh0.csv:68, combo_remote_sensing_s0200_split3_pct50_sh0.csv:68 |
| remote_sensing | random_z | +atenção simam@L5 | 0.499 | +0.013 | 2/3 | 0.480 | +0.014 | combo_remote_sensing_s0200_split1_pct50_sh0.csv:48, combo_remote_sensing_s0200_split2_pct50_sh0.csv:48, combo_remote_sensing_s0200_split3_pct50_sh0.csv:48 |
| remote_sensing | random_z | +atenção simam@L5 +TTA | 0.509 | +0.022 | 2/3 | 0.480 | +0.014 | combo_remote_sensing_s0200_split1_pct50_sh0.csv:52, combo_remote_sensing_s0200_split2_pct50_sh0.csv:52, combo_remote_sensing_s0200_split3_pct50_sh0.csv:52 |
| remote_sensing | random_z | +atenção simam@final | 0.483 | -0.003 | 1/3 | 0.465 | -0.000 | combo_remote_sensing_s0200_split1_pct50_sh0.csv:72, combo_remote_sensing_s0200_split2_pct50_sh0.csv:72, combo_remote_sensing_s0200_split3_pct50_sh0.csv:72 |
| remote_sensing | random_z | +atenção simam@final +TTA | 0.506 | +0.020 | 2/3 | 0.481 | +0.015 | combo_remote_sensing_s0200_split1_pct50_sh0.csv:76, combo_remote_sensing_s0200_split2_pct50_sh0.csv:76, combo_remote_sensing_s0200_split3_pct50_sh0.csv:76 |
| remote_sensing | random_z | +320 px | 0.491 | +0.005 | 2/3 | 0.459 | -0.006 | combo320_remote_sensing_s0320_split1_pct50_sh0.csv:56, combo320_remote_sensing_s0320_split2_pct50_sh0.csv:56, combo320_remote_sensing_s0320_split3_pct50_sh0.csv:56 |
| remote_sensing | random_z | +320 px +TTA | 0.500 | +0.013 | 1/3 | 0.485 | +0.019 | combo320_remote_sensing_s0320_split1_pct50_sh0.csv:60, combo320_remote_sensing_s0320_split2_pct50_sh0.csv:60, combo320_remote_sensing_s0320_split3_pct50_sh0.csv:60 |
| remote_sensing | random_z | tudo: 320 px + espacial@L5 + TTA | 0.511 | +0.025 | 2/3 | 0.479 | +0.014 | combo320_remote_sensing_s0320_split1_pct50_sh0.csv:44, combo320_remote_sensing_s0320_split2_pct50_sh0.csv:44, combo320_remote_sensing_s0320_split3_pct50_sh0.csv:44 |
| remote_sensing | random_z | tudo: 320 px + espacial@final + TTA | 0.497 | +0.010 | 1/3 | 0.468 | +0.003 | combo320_remote_sensing_s0320_split1_pct50_sh0.csv:68, combo320_remote_sensing_s0320_split2_pct50_sh0.csv:68, combo320_remote_sensing_s0320_split3_pct50_sh0.csv:68 |
| remote_sensing | random_z | tudo: 320 px + simam@L5 + TTA | 0.513 | +0.027 | 2/3 | 0.487 | +0.022 | combo320_remote_sensing_s0320_split1_pct50_sh0.csv:52, combo320_remote_sensing_s0320_split2_pct50_sh0.csv:52, combo320_remote_sensing_s0320_split3_pct50_sh0.csv:52 |
| remote_sensing | random_z | tudo: 320 px + simam@final + TTA | 0.496 | +0.010 | 1/3 | 0.472 | +0.007 | combo320_remote_sensing_s0320_split1_pct50_sh0.csv:76, combo320_remote_sensing_s0320_split2_pct50_sh0.csv:76, combo320_remote_sensing_s0320_split3_pct50_sh0.csv:76 |
| flowers | receita | receita | 0.654 | +0.000 | 0/3 | 0.725 | +0.000 | combo_flowers_s0200_split1_pct50_sh0.csv:132, combo_flowers_s0200_split2_pct50_sh0.csv:132, combo_flowers_s0200_split3_pct50_sh0.csv:132 |
| flowers | receita | +TTA | 0.660 | +0.006 | 2/3 | 0.730 | +0.005 | combo_flowers_s0200_split1_pct50_sh0.csv:136, combo_flowers_s0200_split2_pct50_sh0.csv:136, combo_flowers_s0200_split3_pct50_sh0.csv:136 |
| flowers | receita | +atenção espacial@L5 | 0.632 | -0.021 | 0/3 | 0.709 | -0.016 | combo_flowers_s0200_split1_pct50_sh0.csv:116, combo_flowers_s0200_split2_pct50_sh0.csv:116, combo_flowers_s0200_split3_pct50_sh0.csv:116 |
| flowers | receita | +atenção espacial@L5 +TTA | 0.654 | -0.000 | 2/3 | 0.725 | +0.000 | combo_flowers_s0200_split1_pct50_sh0.csv:120, combo_flowers_s0200_split2_pct50_sh0.csv:120, combo_flowers_s0200_split3_pct50_sh0.csv:120 |
| flowers | receita | +atenção espacial@final | 0.657 | +0.003 | 2/3 | 0.727 | +0.002 | combo_flowers_s0200_split1_pct50_sh0.csv:140, combo_flowers_s0200_split2_pct50_sh0.csv:140, combo_flowers_s0200_split3_pct50_sh0.csv:140 |
| flowers | receita | +atenção espacial@final +TTA | 0.662 | +0.008 | 3/3 | 0.730 | +0.005 | combo_flowers_s0200_split1_pct50_sh0.csv:144, combo_flowers_s0200_split2_pct50_sh0.csv:144, combo_flowers_s0200_split3_pct50_sh0.csv:144 |
| flowers | receita | +atenção simam@L5 | 0.645 | -0.009 | 1/3 | 0.717 | -0.008 | combo_flowers_s0200_split1_pct50_sh0.csv:124, combo_flowers_s0200_split2_pct50_sh0.csv:124, combo_flowers_s0200_split3_pct50_sh0.csv:124 |
| flowers | receita | +atenção simam@L5 +TTA | 0.664 | +0.010 | 2/3 | 0.733 | +0.008 | combo_flowers_s0200_split1_pct50_sh0.csv:128, combo_flowers_s0200_split2_pct50_sh0.csv:128, combo_flowers_s0200_split3_pct50_sh0.csv:128 |
| flowers | receita | +atenção simam@final | 0.649 | -0.005 | 0/3 | 0.721 | -0.004 | combo_flowers_s0200_split1_pct50_sh0.csv:148, combo_flowers_s0200_split2_pct50_sh0.csv:148, combo_flowers_s0200_split3_pct50_sh0.csv:148 |
| flowers | receita | +atenção simam@final +TTA | 0.664 | +0.010 | 3/3 | 0.733 | +0.008 | combo_flowers_s0200_split1_pct50_sh0.csv:152, combo_flowers_s0200_split2_pct50_sh0.csv:152, combo_flowers_s0200_split3_pct50_sh0.csv:152 |
| flowers | random_z | receita | 0.549 | +0.000 | 0/3 | 0.641 | +0.000 | combo_flowers_s0200_split1_pct50_sh0.csv:56, combo_flowers_s0200_split2_pct50_sh0.csv:56, combo_flowers_s0200_split3_pct50_sh0.csv:56 |
| flowers | random_z | +TTA | 0.587 | +0.038 | 3/3 | 0.672 | +0.031 | combo_flowers_s0200_split1_pct50_sh0.csv:60, combo_flowers_s0200_split2_pct50_sh0.csv:60, combo_flowers_s0200_split3_pct50_sh0.csv:60 |
| flowers | random_z | +atenção espacial@L5 | 0.522 | -0.027 | 0/3 | 0.618 | -0.024 | combo_flowers_s0200_split1_pct50_sh0.csv:40, combo_flowers_s0200_split2_pct50_sh0.csv:40, combo_flowers_s0200_split3_pct50_sh0.csv:40 |
| flowers | random_z | +atenção espacial@L5 +TTA | 0.586 | +0.037 | 3/3 | 0.671 | +0.030 | combo_flowers_s0200_split1_pct50_sh0.csv:44, combo_flowers_s0200_split2_pct50_sh0.csv:44, combo_flowers_s0200_split3_pct50_sh0.csv:44 |
| flowers | random_z | +atenção espacial@final | 0.549 | -0.000 | 1/3 | 0.642 | +0.000 | combo_flowers_s0200_split1_pct50_sh0.csv:64, combo_flowers_s0200_split2_pct50_sh0.csv:64, combo_flowers_s0200_split3_pct50_sh0.csv:64 |
| flowers | random_z | +atenção espacial@final +TTA | 0.598 | +0.049 | 3/3 | 0.681 | +0.040 | combo_flowers_s0200_split1_pct50_sh0.csv:68, combo_flowers_s0200_split2_pct50_sh0.csv:68, combo_flowers_s0200_split3_pct50_sh0.csv:68 |
| flowers | random_z | +atenção simam@L5 | 0.546 | -0.003 | 1/3 | 0.638 | -0.004 | combo_flowers_s0200_split1_pct50_sh0.csv:48, combo_flowers_s0200_split2_pct50_sh0.csv:48, combo_flowers_s0200_split3_pct50_sh0.csv:48 |
| flowers | random_z | +atenção simam@L5 +TTA | 0.576 | +0.027 | 2/3 | 0.663 | +0.022 | combo_flowers_s0200_split1_pct50_sh0.csv:52, combo_flowers_s0200_split2_pct50_sh0.csv:52, combo_flowers_s0200_split3_pct50_sh0.csv:52 |
| flowers | random_z | +atenção simam@final | 0.551 | +0.002 | 2/3 | 0.642 | +0.001 | combo_flowers_s0200_split1_pct50_sh0.csv:72, combo_flowers_s0200_split2_pct50_sh0.csv:72, combo_flowers_s0200_split3_pct50_sh0.csv:72 |
| flowers | random_z | +atenção simam@final +TTA | 0.590 | +0.041 | 3/3 | 0.674 | +0.033 | combo_flowers_s0200_split1_pct50_sh0.csv:76, combo_flowers_s0200_split2_pct50_sh0.csv:76, combo_flowers_s0200_split3_pct50_sh0.csv:76 |

### Os ganhos somam?
- aid: não. Só a TTA já dá +0,073 (3 de 3 splits). Juntar atenção ou 320 px não melhora: 320 px + TTA dá +0,060, e tudo junto dá no máximo +0,068.
- remote_sensing: sim, em parte. A TTA dá +0,047, os 320 px dão +0,033, e 320 px + TTA dá +0,053. Com a atenção em L5 por cima, o ganho sobe para +0,079 (espacial) e +0,077 (simam), em 3 de 3 splits. Esses são os melhores números da rodada.
- flowers: a TTA dá +0,006. Com simam (L5 ou final) e TTA dá +0,010. A diferença está dentro do ruído.
- random_z (só controle): a TTA ajuda muito mais o aleatório (+0,128 em aid). A régua continua sendo o random_z sem técnica.
