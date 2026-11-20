# SVM dos SSL da fila antiga (13/11)

**Resposta curta:** o SSL ajudou muito no **aid**: o κ de teste da camada final foi de 0,68 para 0,84 a 50% e de 0,74 para 0,86 a 100%, com SimCLR e CosFace. No **flowers** o ganho foi pequeno, de +0,02 a +0,05, e o κ fica perto de 0,65. O **8M LeJEPA** subiu bem em Eggs (0,85 → 0,90–0,93) e em Protozoan (0,81 → 0,86–0,88). Em Protozoan ele ficou **acima do ResNet-50 e do ConvNeXt-pico** (≈0,82). O **LeJEPA pequeno no aid não aprendeu nada**: o melhor checkpoint é a época 0. Todos ficam bem acima do aleatório puro e bem abaixo do DINOv2 e do AIMv2.

Como ler as tabelas:
- **Número oficial = camada final (L6) na largura cheia**, leitura `gem_tta` (GeM nas 8 rotações/espelhos) **sem whitening**. Está em negrito.
- A coluna "com pw" é a leitura antiga `gem_tta_pw`, com PCA-whitening. Aqui o pw **não cortou dimensão** em nenhum caso, porque o nº de imagens de treino é sempre maior que a largura. Por isso as duas leituras valem; a oficial é a sem whitening.
- Os YAMLs `*_catttapw_*` usam `embed_mode: gem_tta_pw`, ou seja, só a camada final, **sem** concatenar camadas. O nome engana: é a mesma leitura do `*_gemttapw_*`.
- L0 é o encoder intacto, antes do SSL. A época do L6 é a do checkpoint de menor val_ssl_loss, escolhido sem rótulo.

## Formato de cada modelo (do ckpt) e dimensão que entrou no SVM

| modelo | camadas conv [saída, entrada, k, k] | params encoder | dim SVM |
|---|---|---|---|
| SSL pequeno flowers (SimCLR) | [104,3,3,3] · [104,104,3,3]×2 · [104,104,5,5]×3 | 1,01 M | 104 |
| SSL pequeno aid (SimCLR, CosFace, LeJEPA) | [216,3,3,3] · [216,216,3,3]×2 · [216,216,5,5]×3 | 4,35 M | 216 |
| 8M LeJEPA (Eggs, Protozoan) | [256,3,5,5] · [256,256,5,5]×5 | 8,2 M | 256 |

Pools 2-2-2-2-1-1 a 200 px: o mapa final tem 11×11 nos pequenos.

## Athena — SSL pequenos (200 px, 6 camadas)

### SimCLR

| base | split | pct | camada | val κ sem whitening | test κ sem whitening | test F1 (sem pw; com pw se só houver pw) | dim SVM | val κ com pw | test κ com pw |
|---|---|---|---|---|---|---|---|---|---|
| flowers | 1 | 50 | L0 encoder intacto | 0.577 | 0.620 | 0.697 | 104 | 0.586 | 0.636 |
| flowers | 1 | 50 | **L6 final** (época 665) | **0.602** | **0.650** | 0.720 | 104 | 0.624 | 0.611 |
| flowers | 1 | 100 | L0 encoder intacto | 0.620 | 0.588 | 0.668 | 104 | 0.601 | 0.606 |
| flowers | 1 | 100 | **L6 final** (época 306) | **0.629** | **0.635** | 0.709 | 104 | 0.632 | 0.659 |
| flowers | 2 | 50 | L0 encoder intacto | 0.610 | 0.576 | 0.659 | 104 | 0.565 | 0.584 |
| flowers | 2 | 50 | **L6 final** (época 477) | **0.619** | **0.623** | 0.697 | 104 | 0.595 | 0.608 |
| flowers | 2 | 100 | L0 encoder intacto | 0.616 | 0.644 | 0.716 | 104 | 0.640 | 0.628 |
| flowers | 2 | 100 | **L6 final** (época 355) | **0.638** | **0.662** | 0.726 | 104 | 0.662 | 0.699 |
| aid | 1 | 50 | L0 encoder intacto | 0.710 | 0.677 | 0.690 | 216 | 0.732 | 0.688 |
| aid | 1 | 50 | **L6 final** (época 339) | **0.857** | **0.835** | 0.839 | 216 | 0.877 | 0.850 |
| aid | 1 | 100 | L0 encoder intacto | 0.752 | 0.739 | 0.747 | 216 | 0.751 | 0.734 |
| aid | 1 | 100 | **L6 final** (época 158) | **0.860** | **0.860** | 0.861 | 216 | 0.884 | 0.878 |

### CosFace

L0 é o mesmo encoder intacto do SimCLR (conferi: pesos iguais), então a linha L0 repete.

| base | split | pct | camada | val κ sem whitening | test κ sem whitening | test F1 (sem pw; com pw se só houver pw) | dim SVM | val κ com pw | test κ com pw |
|---|---|---|---|---|---|---|---|---|---|
| aid | 1 | 50 | L0 encoder intacto | 0.710 | 0.677 | 0.690 | 216 | 0.732 | 0.688 |
| aid | 1 | 50 | **L6 final** (época 280) | **0.848** | **0.825** | 0.829 | 216 | 0.872 | 0.844 |
| aid | 1 | 100 | L0 encoder intacto | 0.752 | 0.739 | 0.747 | 216 | 0.751 | 0.734 |
| aid | 1 | 100 | **L6 final** (época 176) | **0.875** | **0.862** | 0.864 | 216 | 0.889 | 0.877 |

### Modelos grandes e aleatório (mesma base/split/pct)

| base | split | pct | modelo | val κ | test κ | test F1 | dim |
|---|---|---|---|---|---|---|---|
| flowers | 1 | 50 | dinov2-giant | 0.979 | 0.973 | 0.978 | 1536 |
| flowers | 1 | 50 | aimv2-3B | 0.979 | 0.979 | 0.982 | 3072 |
| flowers | 1 | 50 | ijepa_vitg16_22k | 0.945 | 0.921 | 0.936 | 1408 |
| flowers | 1 | 50 | resnet50 | 0.881 | 0.860 | 0.888 | 2048 |
| flowers | 1 | 50 | convnextv2-pico | 0.890 | 0.884 | 0.907 | 512 |
| flowers | 1 | 50 | aleatório puro (random_z, mesma forma) | 0.357 (pw) | não rodou | — | 104 |
| flowers | 1 | 100 | dinov2-giant | 0.985 | 0.976 | 0.980 | 1536 |
| flowers | 1 | 100 | aimv2-3B | 0.985 | 0.979 | 0.982 | 3072 |
| flowers | 1 | 100 | ijepa_vitg16_22k | 0.954 | 0.930 | 0.943 | 1408 |
| flowers | 1 | 100 | resnet50 | 0.906 | 0.872 | 0.897 | 2048 |
| flowers | 1 | 100 | convnextv2-pico | 0.899 | 0.878 | 0.902 | 512 |
| flowers | 1 | 100 | aleatório puro (random_z, mesma forma) | 0.491 (pw) | não rodou | — | 104 |
| flowers | 2 | 50 | dinov2-giant | 0.960 | 0.988 | 0.990 | 1536 |
| flowers | 2 | 50 | aimv2-3B | 0.976 | 0.988 | 0.990 | 3072 |
| flowers | 2 | 50 | ijepa_vitg16_22k | 0.930 | 0.963 | 0.971 | 1408 |
| flowers | 2 | 50 | resnet50 | 0.884 | 0.912 | 0.930 | 2048 |
| flowers | 2 | 50 | convnextv2-pico | 0.903 | 0.894 | 0.913 | 512 |
| flowers | 2 | 50 | aleatório puro (random_z, mesma forma) | 0.422 (pw) | não rodou | — | 104 |
| flowers | 2 | 100 | dinov2-giant | 0.967 | 0.991 | 0.993 | 1536 |
| flowers | 2 | 100 | aimv2-3B | 0.976 | 0.985 | 0.987 | 3072 |
| flowers | 2 | 100 | ijepa_vitg16_22k | 0.930 | 0.957 | 0.966 | 1408 |
| flowers | 2 | 100 | resnet50 | 0.887 | 0.909 | 0.928 | 2048 |
| flowers | 2 | 100 | convnextv2-pico | 0.897 | 0.900 | 0.918 | 512 |
| flowers | 2 | 100 | aleatório puro (random_z, mesma forma) | 0.512 (pw) | não rodou | — | 104 |
| aid | 1 | 50 | dinov2-giant | 0.970 | 0.955 | 0.956 | 1536 |
| aid | 1 | 50 | aimv2-3B | 0.983 | 0.975 | 0.975 | 3072 |
| aid | 1 | 50 | ijepa_vitg16_22k | 0.895 | 0.892 | 0.893 | 1408 |
| aid | 1 | 50 | resnet50 | 0.912 | 0.884 | 0.884 | 2048 |
| aid | 1 | 50 | convnextv2-pico | 0.867 | 0.851 | 0.850 | 512 |
| aid | 1 | 50 | aleatório puro (random_z, mesma forma) | 0.576 (pw) | não rodou | — | 216 |
| aid | 1 | 100 | dinov2-giant | 0.977 | 0.970 | 0.971 | 1536 |
| aid | 1 | 100 | aimv2-3B | 0.990 | 0.979 | 0.979 | 3072 |
| aid | 1 | 100 | ijepa_vitg16_22k | 0.922 | 0.916 | 0.916 | 1408 |
| aid | 1 | 100 | resnet50 | 0.926 | 0.898 | 0.897 | 2048 |
| aid | 1 | 100 | convnextv2-pico | 0.890 | 0.888 | 0.887 | 512 |
| aid | 1 | 100 | aleatório puro (random_z, mesma forma) | 0.584 (pw) | não rodou | — | 216 |

## Jaci — LeJEPA

### 8M LeJEPA (6×256, k5)

| base | split | pct | camada | val κ sem whitening | test κ sem whitening | test F1 (sem pw; com pw se só houver pw) | dim SVM | val κ com pw | test κ com pw |
|---|---|---|---|---|---|---|---|---|---|
| Eggs | 1 | 50 | L0 encoder intacto | 0.851 | 0.850 | 0.865 | 256 | 0.869 | 0.857 |
| Eggs | 1 | 50 | **L6 final** (época 92) | **0.890** | **0.898** | 0.921 | 256 | 0.891 | 0.897 |
| Eggs | 2 | 50 | L0 encoder intacto | 0.809 | 0.834 | 0.888 | 256 | 0.836 | 0.822 |
| Eggs | 2 | 50 | **L6 final** (época 84) | **0.906** | **0.933** | 0.948 | 256 | 0.896 | 0.935 |
| Protozoan | 2 | 50 | L0 encoder intacto | — | — | 0.780 | 256 | 0.794 | 0.814 |
| Protozoan | 2 | 50 | **L6 final** (época 51) | **—** | **—** | 0.834 | 256 | 0.881 | 0.876 |
| Protozoan | 3 | 50 | L0 encoder intacto | — | — | 0.761 | 256 | 0.827 | 0.806 |
| Protozoan | 3 | 50 | **L6 final** (época 35) | **—** | **—** | 0.822 | 256 | 0.872 | 0.859 |

Protozoan não ganhou leitura sem whitening: o pw não cortou nada (n_treino 13086 ≫ 256).

### LeJEPA pequeno, aid (6×216)

| base | split | pct | camada | val κ sem whitening | test κ sem whitening | test F1 (sem pw; com pw se só houver pw) | dim SVM | val κ com pw | test κ com pw |
|---|---|---|---|---|---|---|---|---|---|
| aid | 1 | 50 | L0 encoder intacto | — | — | 0.701 | 216 | 0.732 | 0.688 |
| aid | 1 | 50 | **L6 final** (época 0) | **—** | **—** | 0.717 | 216 | 0.728 | 0.704 |
| aid | 2 | 50 | L0 encoder intacto | — | — | 0.735 | 216 | 0.699 | 0.720 |
| aid | 2 | 50 | **L6 final** (época 0) | **—** | **—** | 0.727 | 216 | 0.697 | 0.712 |
| aid | 3 | 50 | L0 encoder intacto | — | — | 0.727 | 216 | 0.725 | 0.716 |
| aid | 3 | 50 | **L6 final** (época 0) | **—** | **—** | 0.724 | 216 | 0.733 | 0.714 |

### Modelos grandes e aleatório (mesma base/split/pct)

| base | split | pct | modelo | val κ | test κ | test F1 | dim |
|---|---|---|---|---|---|---|---|
| Eggs | 1 | 50 | dinov2-giant | 0.980 | 0.980 | 0.976 | 1536 |
| Eggs | 1 | 50 | aimv2-3B | 0.984 | 0.984 | 0.981 | 3072 |
| Eggs | 1 | 50 | ijepa_vitg16_22k | 0.963 | 0.968 | 0.968 | 1408 |
| Eggs | 1 | 50 | resnet50 | 0.953 | 0.931 | 0.941 | 2048 |
| Eggs | 1 | 50 | convnextv2-pico | 0.936 | 0.944 | 0.943 | 512 |
| Eggs | 1 | 50 | aleatório puro (random_z, mesma forma) | 0.592 (pw) | não rodou | — | 256 |
| Eggs | 2 | 50 | dinov2-giant | 0.983 | 0.980 | 0.979 | 1536 |
| Eggs | 2 | 50 | aimv2-3B | 0.980 | 0.987 | 0.990 | 3072 |
| Eggs | 2 | 50 | ijepa_vitg16_22k | 0.968 | 0.957 | 0.956 | 1408 |
| Eggs | 2 | 50 | resnet50 | 0.931 | 0.959 | 0.964 | 2048 |
| Eggs | 2 | 50 | convnextv2-pico | 0.920 | 0.937 | 0.944 | 512 |
| Eggs | 2 | 50 | aleatório puro (random_z, mesma forma) | 0.569 (pw) | não rodou | — | 256 |
| Protozoan | 2 | 50 | dinov2-giant | 0.938 | 0.936 | 0.908 | 1536 |
| Protozoan | 2 | 50 | aimv2-3B | 0.949 | 0.947 | 0.925 | 3072 |
| Protozoan | 2 | 50 | ijepa_vitg16_22k | 0.916 | 0.917 | 0.879 | 1408 |
| Protozoan | 2 | 50 | resnet50 | 0.792 | 0.819 | 0.770 | 2048 |
| Protozoan | 2 | 50 | convnextv2-pico | 0.832 | 0.822 | 0.771 | 512 |
| Protozoan | 2 | 50 | aleatório puro (random_z, mesma forma) | 0.518 (pw) | não rodou | — | 256 |
| Protozoan | 3 | 50 | dinov2-giant | 0.943 | 0.946 | 0.934 | 1536 |
| Protozoan | 3 | 50 | aimv2-3B | 0.955 | 0.954 | 0.943 | 3072 |
| Protozoan | 3 | 50 | ijepa_vitg16_22k | 0.919 | 0.927 | 0.902 | 1408 |
| Protozoan | 3 | 50 | resnet50 | 0.811 | 0.807 | 0.761 | 2048 |
| Protozoan | 3 | 50 | convnextv2-pico | 0.819 | 0.831 | 0.793 | 512 |
| Protozoan | 3 | 50 | aleatório puro (random_z, mesma forma) | 0.525 (pw) | não rodou | — | 256 |
| aid | 1 | 50 | dinov2-giant | 0.970 | 0.955 | 0.956 | 1536 |
| aid | 1 | 50 | aimv2-3B | 0.983 | 0.975 | 0.975 | 3072 |
| aid | 1 | 50 | ijepa_vitg16_22k | 0.895 | 0.892 | 0.893 | 1408 |
| aid | 1 | 50 | resnet50 | 0.912 | 0.884 | 0.884 | 2048 |
| aid | 1 | 50 | convnextv2-pico | 0.867 | 0.851 | 0.850 | 512 |
| aid | 1 | 50 | aleatório puro (random_z, mesma forma) | 0.576 (pw) | não rodou | — | 216 |
| aid | 2 | 50 | dinov2-giant | 0.970 | 0.963 | 0.963 | 1536 |
| aid | 2 | 50 | aimv2-3B | 0.979 | 0.976 | 0.976 | 3072 |
| aid | 2 | 50 | ijepa_vitg16_22k | 0.908 | 0.897 | 0.897 | 1408 |
| aid | 2 | 50 | resnet50 | 0.895 | 0.888 | 0.887 | 2048 |
| aid | 2 | 50 | convnextv2-pico | 0.862 | 0.855 | 0.856 | 512 |
| aid | 2 | 50 | aleatório puro (random_z, mesma forma) | 0.540 (pw) | não rodou | — | 216 |
| aid | 3 | 50 | dinov2-giant | 0.970 | 0.970 | 0.969 | 1536 |
| aid | 3 | 50 | aimv2-3B | 0.981 | 0.977 | 0.976 | 3072 |
| aid | 3 | 50 | ijepa_vitg16_22k | 0.905 | 0.899 | 0.899 | 1408 |
| aid | 3 | 50 | resnet50 | 0.895 | 0.901 | 0.900 | 2048 |
| aid | 3 | 50 | convnextv2-pico | 0.864 | 0.863 | 0.863 | 512 |
| aid | 3 | 50 | aleatório puro (random_z, mesma forma) | 0.566 (pw) | não rodou | — | 216 |

O aleatório puro é o encoder `random_z` da mesma forma, sem treino. Ele só tem val, porque o spf_grade não roda teste, e a leitura dele é gem_tta_pw da camada final. Por isso a coluna de teste diz "não rodou".

## spf_grade (jaci)

- **disc_md_restr5m_k3_obj, ham10000:** terminou (7 ok, 41 pulados por estado, exit 0). Só tem val κ, sem teste. O melhor SPiFiL, em cat_tta_b, fica entre 0,51 e 0,53. Na camada final sozinha (L_tta) o melhor é 0,46, com d6 w216. O random_z fica entre 0,35 e 0,44.
- **disc_md_e07_on, coconut:** **travado, não terminou.** Os 2 jobs do coconut s1 (PIDs 524765 e 524766 no jaci) estão parados desde 01:20, ~8 h com ~30 s de CPU e ~3,3 GB de GPU. Não mexi neles. Só existe um parcial de 08/11, com val κ da camada final: fpsw 0,778 (dim 256) e random_z 0,569.

## Problemas e avisos

1. **O LeJEPA pequeno no aid não treinou.** O best é a época 0 nos 3 splits: a val_ssl_loss nunca melhorou. O L6 é praticamente o L0 (diferença ≤ 3e-5 nos pesos), então a tabela dele não mede o LeJEPA.
2. **Pontos relançados hoje no athena:** os `checkpoints/` dos 8 pontos apontam para os runs de hoje (conferi run-id por run-id). Antes de rodar, tirei do `execution_state.json` as chaves L6 do aid s1 do SimCLR e guardei cópia em `.bak_20261113`. O L0 velho pôde ficar, porque os pesos são idênticos aos de hoje. No aid s1, o SSL de hoje chegou à mesma época de best de ontem (158 e 339), ou seja, o treino é determinístico.
3. **Benchmark:** os grandes estão completos nos 7 conjuntos. O aleatório puro não tem teste em lugar nenhum.

## Arquivos

- athena, leitura sem whitening (YAMLs novos): `experiments/autoencoder/eval_svm_ssl_pequeno_{simclr,cosface}_{val,test}_gemtta_athena.yaml` → `results/svm_ssl_pequeno_{simclr,cosface}_{val,test}_gemtta_athena.csv`
- athena, leitura pw: `results/svm_ssl_pequeno_{simclr,cosface}_{test,val_pct50,val_pct100}_catttapw_athena.csv`
- jaci, leitura sem whitening (YAMLs novos): `eval_svm_ssl_8M_lejepa_{test,val_pct50}_gemtta_jaci.yaml` → `results/svm_ssl_8M_lejepa_{test,val_pct50}_gemtta_jaci.csv`
- jaci, leitura pw: `results/svm_ssl_8M_lejepa_{test,val_pct50}_catttapw_jaci.csv`, `results/svm_ssl_pequeno_lejepa_{test,val_pct50}_catttapw_jaci.csv`
- logs: athena `/exps/moliveira/queues/logs/sol.eval_svm_ssl_antigos_13nov_athena.log`; jaci `/dados/home/moliveira/exps/queues/logs/svm_jaci_1311/cadeia.out`
- Um SVM por vez em cada host, sem falha. Athena: GPU 2, 78 threads, teto de 2 GB (gpucap). Jaci: GPU 0, ~109 threads.
