# Resultados consolidados: fila nova (crescimento / SimCLR / VICReg) + aleatório puro + modelos grandes

Atualizado em 15/11/2026, às 12:28 (athena). Todos os números são **κ de teste** com **50% dos dados de treino**.
- Leitura: camada final cheia → GeM + TTA → SVM linear (C=100), sem whitening e sem corte de dimensão.
- Entre parênteses: o número de splits prontos (de 3).
- O status ao vivo da fila fica em `growth_simclr_vicreg_fila.md`. Este arquivo é a foto consolidada.

## 1. remote_sensing (14 classes): todas as redes e métodos

| Rede (camada final → dim no SVM) | Crescimento (AE) | SimCLR | VICReg | Aleatório puro |
|---|---|---|---|---|
| Rede 1: [1024, 24, 24] → 1024 | 0,488 (3) | 0,477 (3) | 0,476 (3) | 0,478 (3) |
| Rede 1, última camada 1024 aleatória + decoder (teste à parte) | 0,535 (1) | — | — | — |
| Rede 2: [2048, 11, 11] → 2048 | 0,486 (3) | 0,466 (3) | 0,488 (3) | **0,501** (3) |
| Rede 3: [1024, 11, 11] → 1024 | 0,501 (3) | 0,444 (2) | 0,457 (2) | 0,487 (3) |
| Rede 4: [1024, 11, 11] → 1024 | 0,475 (2) | 0,484 (2) | 0,487 (2) | 0,496 (3) |
| DINOv2-g / AIMv2-3B / ResNet-50 / I-JEPA-g / ConvNeXtV2-pico | 0,710 / 0,743 / 0,702 / 0,648 / 0,629 | | | |

### Por split (κ test)

| Run | s1 | s2 | s3 |
|---|---|---|---|
| Rede 1 + AE | 0,524 | 0,449 | 0,491 |
| Rede 1 + SimCLR | 0,510 | 0,478 | 0,444 |
| Rede 1 + VICReg | 0,529 | 0,436 | 0,463 |
| Rede 2 + AE | 0,527 | 0,448 | 0,484 |
| Rede 2 + SimCLR | 0,536 | 0,449 | 0,414 |
| Rede 2 + VICReg | 0,557 | 0,424 | 0,483 |
| Rede 3 + AE | 0,526 | 0,483 | 0,492 |
| Rede 3 + SimCLR | 0,438 | 0,450 | — |
| Rede 3 + VICReg | 0,460 | 0,455 | — |
| Rede 4 + AE | 0,499 | 0,452 | — |
| Rede 4 + SimCLR | 0,513 | 0,455 | — |
| Rede 4 + VICReg | 0,511 | 0,462 | — |
| Aleatório rede 1 / 2 / 3 / 4 | 0,473 / 0,507 / 0,494 / 0,526 | 0,503 / 0,517 / 0,481 / 0,483 | 0,459 / 0,479 / 0,486 / 0,478 |

## 2. Outras bases: rede 2 + crescimento contra o aleatório e os grandes

| Base | SPiFiL rede 2 treinado | Aleatório puro rede 2 | DINOv2 | AIMv2 | I-JEPA | ResNet-50 | ConvNeXt-pico |
|---|---|---|---|---|---|---|---|
| Eggs | **0,854** (3): 0,850 / 0,846 / 0,866 | 0,806 | 0,981 | 0,982 | 0,965 | 0,949 | 0,938 |
| Protozoan | **0,845** (1): 0,845 | 0,700 | 0,943 | 0,952 | 0,924 | 0,814 | 0,830 |
| aid | **0,746** (3): 0,763 / 0,746 / 0,729 | 0,707 | 0,963 | 0,976 | 0,896 | 0,891 | 0,856 |
| Larvae | **0,727** (2): 0,780 / 0,673 | 0,463 | 0,974 | 0,973 | 0,877 | 0,879 | 0,873 |
| flowers | **0,625** (3): 0,629 / 0,617 / 0,628 | 0,563 | 0,977 | 0,981 | 0,940 | 0,882 | 0,889 |
| coconut_trees | rodando | **0,813** | — | — | — | 0,631 | 0,722 |
| ham10000 | rodando | **0,492** | — | — | — | 0,469 | — |
| food101 | rodando | adiado (SVM lento) | — | — | — | — | — |

- **aid, rede 1 + AE:** 0,754 (s1).
- **aid, rede 1 + SimCLR:** 0,670 (s1).
- **aid, rede 1 + VICReg:** 0,681 (s1).
- **ham10000:** os `split{1,2,3}.json` têm validation = train (100% de sobreposição). Lá só o teste vale. O split limpo é `splits/split1_modehb.json`, ainda sem decisão de uso.

## 3. Modelos grandes com 100% de treino (pedido da sessão claude_GA_JACI)

- CSV consolidado: `A_reports/november/grandes_pct100_svm.csv`
- Origem: `Scalable_Hybrid_FLIM_discovery/results/benchmark/<modelo>.csv`
- coconut_trees, κ test (média dos 3 splits): I-JEPA 0,785 · AIMv2 0,726 · ResNet-50 0,682 · DINOv2 0,674
- ham10000, κ test: AIMv2 0,727 · DINOv2 0,698 · I-JEPA 0,592 · ResNet-50 0,494
- ConvNeXt-pico em coconut e ham a 100%: ainda rodando (tmux `bench_cocoham_svm`).

## 4. Conclusões até agora

1. O treino ganha do aleatório em Larvae (+0,26), Eggs (+0,05), aid (+0,05) e flowers (+0,06). Em remote_sensing, empata ou perde.
2. A rede 2 aleatória (init He, sem treino) já é forte: no coconut passa o ConvNeXt-pico e o ResNet-50.
3. O SPiFiL fica 0,13–0,35 de κ abaixo de todos os modelos grandes.
4. A diferença entre métodos e entre redes é pequena (±0,03). O que pesa mais é a arquitetura: muitos canais no fim + GeM.
5. O ajuste fino final (Lk_ft) costuma piorar o teste em relação à camada congelada (Lk).

## 5. Onde ficam os pesos e os resultados (para recuperar)

### Pesos treinados (best ckpt por κ de validação)
- **Padrão:** `/exps/moliveira/artifacts/<família>/<base>_split<s>_pct<p>/<estágio>/flim-ssl/<id>/checkpoints/best_kappa.ckpt`
  - Exemplo: `/exps/moliveira/artifacts/g2_ae/remote_sensing_split1_pct50/L6_ft/flim-ssl/3d7bum0z/checkpoints/best_kappa.ckpt`
  - Estágios: `L1, L1_ft, …, L6, L6_ft` (rede 1: até `L4_ft`). `L{k}_grow` = camada nova antes do treino. `L{k}_flim` = encoder exportado no formato FLIM.
- **Famílias (athena):**
  - Rede 2: `g2_ae`, `g2_simclr`, `g2_vicreg`.
  - Redes 1, 3 e 4 com final ≥1024: `g1k_*`, `g3k_*`, `g4k_*`.
  - Teste da rede 1 com L4 aleatória: `g1x1024_ae`.
  - Aleatório puro: `rand_g1` … `rand_g4`, estágio `L0`.
- **Treinados no jaci, já trazidos para o athena:** `/exps/moliveira/artifacts_jaci/<família>/…`. Inclui g2_ae aid s1/s2 (o s3 e as bases coconut, Eggs, ham e Protozoan têm só treino parcial, morto quando o jaci foi liberado).
- **Versões velhas guardadas** (não usar nas tabelas):
  - `g1_*_final128`, `g3_*_final832`, `g4_*_final512` (antes da regra ≥1024);
  - `g2_ae_sem_whiten_between`;
  - `rand_g*_initpadrao` (init padrão do PyTorch, que apagava o sinal).
- **Encoders do aleatório puro** (init He, seed 0): `/exps/moliveira/encoders/rand_puro/g{1,2,3,4}/pesos` (formato FLIM) e `rand.ckpt`.
- **Encoder de partida do teste g1x1024:** `/exps/moliveira/encoders/g1_ae_L3` → `artifacts/g1_ae_final128/remote_sensing_split1_pct50/L3_flim`.

### YAMLs (repo `/dados/home/moliveira/Scalable_Hybrid_FLIM_discovery/experiments/autoencoder/`)
- Treino: `g{1,2,3,4}q_{ae,simclr,vicreg}.yaml`, `g1x1024_ae.yaml`, `rand_g{1,2,3,4}.yaml`.
- SVM: `eval_svm_<família>_{val,test}_gemtta.yaml` (a família g2 usa `eval_svm_g2_*`; as redes 1/3/4 também usam `eval_svm_g{1,3,4}_*_gemtta`, com family `g?k_*`).

### Resultados (CSV)
- **Resumo por run** (uma linha por SVM; use a última linha de cada run): `/exps/moliveira/queues/logs/g2_resultados.tsv`
- **Por estágio:** `Scalable_Hybrid_FLIM_discovery/results/svm_<família>_{val,test}_gemtta.csv` (ex.: `svm_g2_ae_test_gemtta.csv`, `svm_rand_g2_test_gemtta.csv`, `svm_g1x1024_ae_test_gemtta.csv`).
- **Do jaci:** `Scalable_Hybrid_FLIM_discovery/results_jaci/`
- **Modelos grandes:** `Scalable_Hybrid_FLIM_discovery/results/benchmark/{dinov2-giant,aimv2-3B-patch14-448,ijepa_vitg16_22k,convnextv2-pico-1k-224,resnet50.a1_in1k}.csv` (colunas dataset, split, pct, val_kappa, test_kappa).

### Outros reports do dia
- SSL antigos: `A_reports/november/svm_ssl_antigos_13nov.md`
- Limpeza do `fontes_filtros` + resultados: `A_reports/november/fontes_filtros_resultados_14nov.md`
- Barlow (em andamento): `A_reports/november/svm_ssl_barlow_14nov.md`
- Logs de derrubada/limpeza: `/exps/moliveira/queues/logs/{k1,k2,k3,k4}_*.md`, `apagados_*.txt`
