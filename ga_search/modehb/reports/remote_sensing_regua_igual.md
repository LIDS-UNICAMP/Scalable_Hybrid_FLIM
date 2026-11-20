# remote_sensing — mesma régua SVM para os candidatos MO-DEHB e os encoders do discovery

Teste do split1 (159 imagens), SVM fitado no train do split1 (682 imagens, sem subamostragem). Todas as colunas usam C=100 (SVC linear libsvm ovo, `core.svm.fit_svm`).
O split1 do discovery (`splits_incremental/split1/data_descriptor_perc100.json`) tem as MESMAS listas train/val/test que `splits/split1.json` (conferido arquivo a arquivo).
Só o split1 foi avaliado: nossos modelos treinaram no train do split1, e o teste dos splits 2/3 pode se sobrepor a ele.

Leituras (`ga_search/modehb/svm_on_demand.py --readout ...`):
- (a) `cov_head_tta`: CovReadout (PCA64 + sqrtm + signed-sqrt/L2 por grupo) com média nas 8 vistas `tta_transforms(8)`, depois StandardScaler.
- (b) `cov_head_tta_pw`: o mesmo descritor sqrtm com TTA8, mas sem signed-sqrt/L2 e sem scaler. No lugar entra `pw_whiten` (PCA-whitening, shrinkage 0,1) + L2. A dimensão cai para n_train=682.
- (c) `gem_tta_pw`: a régua R-SPF do discovery. GeM p=3 do último bloco, média nas 8 vistas rot90×flip da imagem, `pw_whiten` (cópia de eval/svm.py l.381-389), sem scaler (eval/l1_fast.py l.144-147, spf_grade.py score_full l.1046-1049).
- (d) `cov_head`: sem TTA, com StandardScaler (a leitura histórica, agora com C=100).
- (e) "SVM original" = `core.svm.fit_svm` com os defaults (C=100, sem scaler, max_iter=-1). Nos nossos, roda sobre head(encoder(x)) (1280-d, ou 2048-d no lemonade c00098), igual ao monitor. Nos do discovery, roda sobre o GAP do último bloco (312-d), sem TTA e sem PW.
- "head + scaler": o mesmo embedding da (e), com StandardScaler. Os encoders do discovery não têm head, então o grupo head do cov_head deles também é esse GAP.
- Também existe `cat_tta_pw` (camadas concatenadas de trás para frente enquanto o total < 2000-d, spf_grade.py l.1662-1672). Está implementado, mas não foi rodado nesta grade.

Cada célula traz κ de teste e, entre parênteses, a dimensão da feature que vai ao SVM.

| cand | busca | params | (a) cov_head+TTA8 | (b) cov_head+TTA8+PW | (c) gem_tta_pw | (d) cov_head | (e) SVM original (head, sem scaler) | head + scaler | cov_head C=0,1 (test_svm.jsonl) |
|---|---|---|---|---|---|---|---|---|---|
| c00001 | hybrid | 20k | 0.522 (1421) | 0.517 (682) | 0.474 (14) | 0.510 (1421) | 0.467 (1280) | 0.432 (1280) | 0.524 |
| c00002 | hybrid | 237k | 0.579 (4263) | 0.481 (682) | 0.480 (98) | 0.589 (4263) | 0.493 (1280) | 0.521 (1280) | 0.589 |
| c00003 | hybrid | 958k | 0.528 (5440) | 0.549 (682) | 0.522 (392) | 0.528 (5440) | 0.505 (1280) | 0.480 (1280) | 0.528 |
| c00004 | hybrid | 1926k | 0.668 (7926) | 0.527 (682) | 0.463 (560) | 0.667 (7926) | 0.480 (1280) | 0.453 (1280) | 0.667 |
| c00010 | hybrid | 246k | 0.634 (4956) | 0.577 (682) | 0.594 (294) | 0.627 (4956) | 0.524 (1280) | 0.552 (1280) | 0.627 |
| c00001 | lemonade | 163k | 0.534 (4185) | 0.530 (682) | 0.423 (56) | 0.498 (4185) | 0.373 (1280) | 0.352 (1280) | 0.498 |
| c00002 | lemonade | 659k | 0.612 (6448) | 0.495 (682) | 0.460 (682) | 0.547 (6448) | 0.438 (1280) | 0.481 (1280) | 0.547 |
| c00003 | lemonade | 1244k | 0.593 (12086) | 0.584 (682) | 0.503 (308) | 0.617 (12086) | 0.486 (1280) | 0.500 (1280) | 0.617 |
| c00004 | lemonade | 2009k | 0.611 (10608) | 0.620 (682) | 0.564 (406) | 0.602 (10608) | 0.492 (1280) | 0.490 (1280) | 0.609 |
| c00012 | lemonade | 30k | 0.588 (3438) | 0.472 (682) | 0.470 (84) | 0.588 (3438) | 0.414 (1280) | 0.403 (1280) | 0.588 |
| c00076 | lemonade | 1322k | 0.575 (7456) | 0.451 (682) | 0.361 (448) | 0.555 (7456) | 0.264 (1280) | 0.319 (1280) | 0.555 |
| c00098 | lemonade | 531k | 0.633 (6313) | 0.614 (682) | 0.585 (168) | 0.637 (6313) | 0.598 (2048) | 0.570 (2048) | 0.637 |

| encoder discovery | (a) | (b) | (c) gem_tta_pw (nosso pipeline 224) | (d) cov_head | (e) SVM original (GAP 312, sem scaler) | GAP + scaler | gem_tta_pw no pipeline deles (200 px) |
|---|---|---|---|---|---|---|---|
| L0 | 0.616 (11532) | 0.646 (682) | 0.596 (312) | 0.635 (11532) | 0.570 (312) | 0.537 (312) | 0.554 |
| lejepa | 0.647 (11532) | 0.633 (682) | 0.577 (312) | 0.693 (11532) | 0.570 (312) | 0.565 (312) | 0.577 |
| cosface | 0.646 (11532) | 0.639 (682) | 0.581 (312) | 0.680 (11532) | 0.605 (312) | 0.579 (312) | 0.568 |
| simclr | 0.640 (11532) | 0.611 (682) | 0.547 (312) | 0.672 (11532) | 0.585 (312) | 0.640 (312) | 0.554 |
| lejepa_barlow | 0.682 (11532) | 0.675 (682) | 0.591 (312) | 0.686 (11532) | 0.571 (312) | 0.556 (312) | 0.592 |
| lejepa_vicreg | 0.652 (11532) | 0.584 (682) | 0.555 (312) | 0.642 (11532) | 0.467 (312) | 0.491 (312) | 0.583 |
| dino | 0.624 (11532) | 0.639 (682) | 0.514 (312) | 0.658 (11532) | 0.412 (312) | 0.462 (312) | 0.518 |
| barlow | 0.616 (11532) | 0.646 (682) | 0.596 (312) | 0.635 (11532) | 0.570 (312) | 0.537 (312) | 0.554 |

Números do discovery na régua R-SPF (teste 100%, n=3 splits): SPiFiL pequeno L0 0,529, +LeJEPA 0,586, +CosFace 0,563, +SimCLR 0,554; FMs AIMv2 0,746, DINOv2 0,719, ResNet-50 0,704, I-JEPA 0,698, ConvNeXtV2-pico 0,655.

## Ressalvas
- Validação da régua: rodei o código do discovery no pipeline deles (200 px letterbox, `_encode_pooled(gem_tta)` + `pw_whiten` + `fit_svm`) com os encoders deles, só no split1: L0 0,554, lejepa 0,577, cosface 0,568, simclr 0,554. Isso bate com as médias n=3 deles. O `_gem_tta_blocks` nosso é igual à referência (teste unitário).
- Na coluna (c) dos encoders deles, a diferença para o número deles vem só do pipeline de imagem: nosso Resize+CenterCrop 224 contra o letterbox 200 deles. O SPiFiL foi aprendido a 200 px.
- O `best_kappa.ckpt` do barlow é da época 0 (step 11) e dá números idênticos ao L0 em todas as leituras, ou seja, os pesos são os do L0.
- Dados brutos: `ga_search/modehb/reports/remote_sensing_regua_igual.jsonl`. W&B: runs external_eval com readout=gem_tta_pw em ophira-ai/progressive_mo_dehb, parents `<busca>-<cand>-f200` e `ext_ssl_pequeno-<enc>`. As outras leituras de cada encoder estão no config, em `other_readouts_C100`.
- `test_svm.jsonl` foi gerado com C=0,1. Ele aparece só como coluna de reprodução, porque a ordem é C=100 sempre. A coluna (d) com C=100 reproduz exatamente o valor de C=0,1 em 10 dos 12 candidatos. As exceções são hybrid c00001 (0,510 contra 0,524) e lemonade c00004 (0,602 contra 0,609).
- Os diretórios `artifacts/modehb_{hybrid,lemonade}` foram reiniciados durante a avaliação, e os candidatos f200 antigos agora estão em `artifacts/modehb_{hybrid,lemonade}_C01_final/`. Os κ de cov_head coincidem com test_svm.jsonl, então são os mesmos checkpoints.
- O fit do SVM leva ≤4,7 s em todas as leituras (682 amostras, fit_status=0 sempre). O que custa é a extração: TTA8 × CovReadout com eigh fp64.
