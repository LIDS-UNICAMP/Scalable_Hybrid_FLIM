# Resultado de TESTE — os estágios do crescimento SPiFiL

Consolidação de `results/eval_growth_stages_pct5.csv` (42 linhas) e
`results/eval_growth_stages_pct50.csv` (40 linhas). **82 trabalhos no total.**

## Procedência — leia antes de olhar qualquer número

**Estes números são do conjunto de TESTE.** O classificador é o SVM oficial do
repositório, `fit_svm` (`src/utils/evaluate.py:372`) — a única `SVC` do projeto,
linear, `C=1e2`, `ovo`, `max_iter=-1` (convergido). As métricas saem da função
única `compute_metrics` (`src/metrics/classification.py:31`).

**Não misture com os números de validação.** O documento
[`docs/evolucao_dos_estagios_validacao.md`](evolucao_dos_estagios_validacao.md)
traz kappa da **sonda interna**, que roda a cada época durante o treino. É outro
conjunto (validação) e outro classificador (um `SVC` próprio,
`src/modules/autoencoder_flim_module.py:485`). **Os dois não são comparáveis** —
o próprio código avisa isso (`src/evaluate/eval_autoencoder.py:59-63`).

**Origem dos checkpoints:** `artifacts/spifil_growth/grid4/`, treino concluído —
18 braços (3 datasets × 3 splits × 2 porcentagens), 0 falhas. De cada estágio sai
o `checkpoints/best_kappa.ckpt`.

**Todos os embeddings são `flatten`.** A dimensão é `channels[-1] × 24 × 24` e
muda conforme o dataset e o estágio, porque a camada nova do SPiFiL entrega um
número de filtros diferente em cada dataset:

| dim | canais | onde aparece |
|---|---|---|
| 27648 | 48 | `stage1` e `stage2` de todos os datasets; **todos** os estágios de larvae |
| 25920 | 45 | eggs, `round1_stage3` e `round1_stage4` |
| 24192 | 42 | protozoan, `round1_stage3` e `round1_stage4` |

**O comando que gerou os CSVs:**

```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM

OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE \
  /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python \
  -m src.evaluate.eval_growth_stages --pct 5 --device cuda:0

OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE \
  /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python \
  -m src.evaluate.eval_growth_stages --pct 50 --device cuda:0
```

Sem `--out`, cada execução escreve em `results/eval_growth_stages_pct<PCT>.csv`.

---

## Legenda dos estágios

| rótulo | encoder | decoder | o que recebe gradiente |
|---|---|---|---|
| `stage1` | FLIM congelado | aleatório | só o decoder |
| `stage2` | destravado | vem do estágio 1 | tudo |
| `round1_stage3` | ganhou 1 camada SPiFiL; o resto congelado | ganhou 1 bloco novo; o resto congelado | só a camada nova + o bloco novo |
| `round1_stage4` | destravado | destravado | tudo |
| `round2_stage3` | ganhou a 2ª camada SPiFiL; o resto congelado | ganhou mais 1 bloco novo; o resto congelado | só a camada nova + o bloco novo |
| `round2_stage4` | destravado | destravado | tudo |

`stage1` e `stage2` rodam uma vez. Os estágios 3 e 4 se repetem a cada rodada de
crescimento.

**`round2_*` só existe em larvae.** Nos outros datasets o laço de crescimento
parou na rodada 1. Por isso o número de estágios muda de célula para célula:
eggs e protozoan têm 4 estágios, larvae tem 6.

---

## Sumário da rede por grau de profundidade

As formas abaixo foram **medidas** executando cada modelo com uma entrada
`[1, 3, 200, 200]`, não deduzidas. Entrada e saída têm a mesma forma: é um autoencoder,
e o alvo da reconstrução é a própria imagem de entrada.

Cada grau de profundidade é compartilhado por dois estágios — o congelado e o
destravado. O que muda entre eles não é a forma, é o que recebe gradiente.

### eggs — 3 camadas · estágios `stage1`, `stage2`

Canais `[3, 24, 32, 48]` · embedding flatten = 48 × 24 × 24 = **27.648** · 108.283 parâmetros, sendo 59.504 no encoder

| bloco | operação | entra | sai | params |
|---|---|---|---|---|
| `encoder.conv1` | Conv 5x5 + ReLU + MaxPool 3x3/2 | `3×200×200` | `24×99×99` | 1.824 |
| `encoder.conv2` | Conv 5x5 + ReLU + MaxPool 3x3/2 | `24×99×99` | `32×49×49` | 19.232 |
| `encoder.conv3` | Conv 5x5 + ReLU + MaxPool 3x3/2 | `32×49×49` | `48×24×24` | 38.448 |
| `decoder.blocks.0` | Upsample x2 + 2 Conv 3x3 residual | `48×24×24` | `32×48×48` | 24.704 |
| `decoder.blocks.1` | Upsample x2 + 2 Conv 3x3 residual | `32×48×48` | `24×96×96` | 12.960 |
| `decoder.blocks.2` | Upsample x2 + 2 Conv 3x3 residual | `24×96×96` | `24×192×192` | 10.464 |
| `interpolate` | bilinear para 200x200 | `24×192×192` | `24×200×200` | 0 |
| `decoder.to_image` | Conv 3x3 → logits | `24×200×200` | `3×200×200` | 651 |

O SVM lê o gargalo `48×24×24` achatado. O decoder lê o **mesmo mapa inteiro**, não o embedding — são dois consumidores independentes do mesmo tensor.

### eggs — 4 camadas · estágios `round1_stage3`, `round1_stage4`

Canais `[3, 24, 32, 48, 45]` · embedding flatten = 45 × 24 × 24 = **25.920** · 170.296 parâmetros, sendo 78.989 no encoder

| bloco | operação | entra | sai | params |
|---|---|---|---|---|
| `encoder.conv1` | Conv 5x5 + ReLU + MaxPool 3x3/2 | `3×200×200` | `24×99×99` | 1.824 |
| `encoder.conv2` | Conv 5x5 + ReLU + MaxPool 3x3/2 | `24×99×99` | `32×49×49` | 19.232 |
| `encoder.conv3` | Conv 5x5 + ReLU + MaxPool 3x3/2 | `32×49×49` | `48×24×24` | 38.448 |
| `encoder.conv4` | Conv 3x3 + ReLU · sem pool | `48×24×24` | `45×24×24` | 19.485 |
| `decoder.blocks.0` | Upsample x1 + 2 Conv 3x3 residual | `45×24×24` | `48×24×24` | 42.528 |
| `decoder.blocks.1` | Upsample x2 + 2 Conv 3x3 residual | `48×24×24` | `32×48×48` | 24.704 |
| `decoder.blocks.2` | Upsample x2 + 2 Conv 3x3 residual | `32×48×48` | `24×96×96` | 12.960 |
| `decoder.blocks.3` | Upsample x2 + 2 Conv 3x3 residual | `24×96×96` | `24×192×192` | 10.464 |
| `interpolate` | bilinear para 200x200 | `24×192×192` | `24×200×200` | 0 |
| `decoder.to_image` | Conv 3x3 → logits | `24×200×200` | `3×200×200` | 651 |

O SVM lê o gargalo `45×24×24` achatado. O decoder lê o **mesmo mapa inteiro**, não o embedding — são dois consumidores independentes do mesmo tensor.

### larvae — 3 camadas · estágios `stage1`, `stage2`

Canais `[3, 24, 32, 48]` · embedding flatten = 48 × 24 × 24 = **27.648** · 108.283 parâmetros, sendo 59.504 no encoder

| bloco | operação | entra | sai | params |
|---|---|---|---|---|
| `encoder.conv1` | Conv 5x5 + ReLU + MaxPool 3x3/2 | `3×200×200` | `24×99×99` | 1.824 |
| `encoder.conv2` | Conv 5x5 + ReLU + MaxPool 3x3/2 | `24×99×99` | `32×49×49` | 19.232 |
| `encoder.conv3` | Conv 5x5 + ReLU + MaxPool 3x3/2 | `32×49×49` | `48×24×24` | 38.448 |
| `decoder.blocks.0` | Upsample x2 + 2 Conv 3x3 residual | `48×24×24` | `32×48×48` | 24.704 |
| `decoder.blocks.1` | Upsample x2 + 2 Conv 3x3 residual | `32×48×48` | `24×96×96` | 12.960 |
| `decoder.blocks.2` | Upsample x2 + 2 Conv 3x3 residual | `24×96×96` | `24×192×192` | 10.464 |
| `interpolate` | bilinear para 200x200 | `24×192×192` | `24×200×200` | 0 |
| `decoder.to_image` | Conv 3x3 → logits | `24×200×200` | `3×200×200` | 651 |

O SVM lê o gargalo `48×24×24` achatado. O decoder lê o **mesmo mapa inteiro**, não o embedding — são dois consumidores independentes do mesmo tensor.

### larvae — 4 camadas · estágios `round1_stage3`, `round1_stage4`

Canais `[3, 24, 32, 48, 48]` · embedding flatten = 48 × 24 × 24 = **27.648** · 170.731 parâmetros, sendo 80.288 no encoder

| bloco | operação | entra | sai | params |
|---|---|---|---|---|
| `encoder.conv1` | Conv 5x5 + ReLU + MaxPool 3x3/2 | `3×200×200` | `24×99×99` | 1.824 |
| `encoder.conv2` | Conv 5x5 + ReLU + MaxPool 3x3/2 | `24×99×99` | `32×49×49` | 19.232 |
| `encoder.conv3` | Conv 5x5 + ReLU + MaxPool 3x3/2 | `32×49×49` | `48×24×24` | 38.448 |
| `encoder.conv4` | Conv 3x3 + ReLU · sem pool | `48×24×24` | `48×24×24` | 20.784 |
| `decoder.blocks.0` | Upsample x1 + 2 Conv 3x3 residual | `48×24×24` | `48×24×24` | 41.664 |
| `decoder.blocks.1` | Upsample x2 + 2 Conv 3x3 residual | `48×24×24` | `32×48×48` | 24.704 |
| `decoder.blocks.2` | Upsample x2 + 2 Conv 3x3 residual | `32×48×48` | `24×96×96` | 12.960 |
| `decoder.blocks.3` | Upsample x2 + 2 Conv 3x3 residual | `24×96×96` | `24×192×192` | 10.464 |
| `interpolate` | bilinear para 200x200 | `24×192×192` | `24×200×200` | 0 |
| `decoder.to_image` | Conv 3x3 → logits | `24×200×200` | `3×200×200` | 651 |

O SVM lê o gargalo `48×24×24` achatado. O decoder lê o **mesmo mapa inteiro**, não o embedding — são dois consumidores independentes do mesmo tensor.

### larvae — 5 camadas · estágios `round2_stage3`, `round2_stage4`

Canais `[3, 24, 32, 48, 48, 48]` · embedding flatten = 48 × 24 × 24 = **27.648** · 233.179 parâmetros, sendo 101.072 no encoder

| bloco | operação | entra | sai | params |
|---|---|---|---|---|
| `encoder.conv1` | Conv 5x5 + ReLU + MaxPool 3x3/2 | `3×200×200` | `24×99×99` | 1.824 |
| `encoder.conv2` | Conv 5x5 + ReLU + MaxPool 3x3/2 | `24×99×99` | `32×49×49` | 19.232 |
| `encoder.conv3` | Conv 5x5 + ReLU + MaxPool 3x3/2 | `32×49×49` | `48×24×24` | 38.448 |
| `encoder.conv4` | Conv 3x3 + ReLU · sem pool | `48×24×24` | `48×24×24` | 20.784 |
| `encoder.conv5` | Conv 3x3 + ReLU · sem pool | `48×24×24` | `48×24×24` | 20.784 |
| `decoder.blocks.0` | Upsample x1 + 2 Conv 3x3 residual | `48×24×24` | `48×24×24` | 41.664 |
| `decoder.blocks.1` | Upsample x1 + 2 Conv 3x3 residual | `48×24×24` | `48×24×24` | 41.664 |
| `decoder.blocks.2` | Upsample x2 + 2 Conv 3x3 residual | `48×24×24` | `32×48×48` | 24.704 |
| `decoder.blocks.3` | Upsample x2 + 2 Conv 3x3 residual | `32×48×48` | `24×96×96` | 12.960 |
| `decoder.blocks.4` | Upsample x2 + 2 Conv 3x3 residual | `24×96×96` | `24×192×192` | 10.464 |
| `interpolate` | bilinear para 200x200 | `24×192×192` | `24×200×200` | 0 |
| `decoder.to_image` | Conv 3x3 → logits | `24×200×200` | `3×200×200` | 651 |

O SVM lê o gargalo `48×24×24` achatado. O decoder lê o **mesmo mapa inteiro**, não o embedding — são dois consumidores independentes do mesmo tensor.

### protozoan — 3 camadas · estágios `stage1`, `stage2`

Canais `[3, 24, 30, 48]` · embedding flatten = 48 × 24 × 24 = **27.648** · 102.117 parâmetros, sendo 55.902 no encoder

| bloco | operação | entra | sai | params |
|---|---|---|---|---|
| `encoder.conv1` | Conv 5x5 + ReLU + MaxPool 3x3/2 | `3×200×200` | `24×99×99` | 1.824 |
| `encoder.conv2` | Conv 5x5 + ReLU + MaxPool 3x3/2 | `24×99×99` | `30×49×49` | 18.030 |
| `encoder.conv3` | Conv 5x5 + ReLU + MaxPool 3x3/2 | `30×49×49` | `48×24×24` | 36.048 |
| `decoder.blocks.0` | Upsample x2 + 2 Conv 3x3 residual | `48×24×24` | `30×48×48` | 22.620 |
| `decoder.blocks.1` | Upsample x2 + 2 Conv 3x3 residual | `30×48×48` | `24×96×96` | 12.480 |
| `decoder.blocks.2` | Upsample x2 + 2 Conv 3x3 residual | `24×96×96` | `24×192×192` | 10.464 |
| `interpolate` | bilinear para 200x200 | `24×192×192` | `24×200×200` | 0 |
| `decoder.to_image` | Conv 3x3 → logits | `24×200×200` | `3×200×200` | 651 |

O SVM lê o gargalo `48×24×24` achatado. O decoder lê o **mesmo mapa inteiro**, não o embedding — são dois consumidores independentes do mesmo tensor.

### protozoan — 4 camadas · estágios `round1_stage3`, `round1_stage4`

Canais `[3, 24, 30, 48, 42]` · embedding flatten = 42 × 24 × 24 = **24.192** · 161.391 parâmetros, sendo 74.088 no encoder

| bloco | operação | entra | sai | params |
|---|---|---|---|---|
| `encoder.conv1` | Conv 5x5 + ReLU + MaxPool 3x3/2 | `3×200×200` | `24×99×99` | 1.824 |
| `encoder.conv2` | Conv 5x5 + ReLU + MaxPool 3x3/2 | `24×99×99` | `30×49×49` | 18.030 |
| `encoder.conv3` | Conv 5x5 + ReLU + MaxPool 3x3/2 | `30×49×49` | `48×24×24` | 36.048 |
| `encoder.conv4` | Conv 3x3 + ReLU · sem pool | `48×24×24` | `42×24×24` | 18.186 |
| `decoder.blocks.0` | Upsample x1 + 2 Conv 3x3 residual | `42×24×24` | `48×24×24` | 41.088 |
| `decoder.blocks.1` | Upsample x2 + 2 Conv 3x3 residual | `48×24×24` | `30×48×48` | 22.620 |
| `decoder.blocks.2` | Upsample x2 + 2 Conv 3x3 residual | `30×48×48` | `24×96×96` | 12.480 |
| `decoder.blocks.3` | Upsample x2 + 2 Conv 3x3 residual | `24×96×96` | `24×192×192` | 10.464 |
| `interpolate` | bilinear para 200x200 | `24×192×192` | `24×200×200` | 0 |
| `decoder.to_image` | Conv 3x3 → logits | `24×200×200` | `3×200×200` | 651 |

O SVM lê o gargalo `42×24×24` achatado. O decoder lê o **mesmo mapa inteiro**, não o embedding — são dois consumidores independentes do mesmo tensor.

---

## Tabela 1 — 5% dos dados rotulados

Média ± desvio padrão amostral sobre os 3 splits, 4 casas decimais.

| dataset | classes | estágio | o que treina | dim do embedding | n_train (SVM) | n_test | kappa | acc | f1 |
|---|---|---|---|---|---|---|---|---|---|
| eggs | 9 | `stage1` | só o decoder | 27648 | 124 | 2557 | 0.6174 ± 0.0567 | 0.7130 ± 0.0535 | 0.6674 ± 0.0453 |
| eggs | 9 | `stage2` | tudo | 27648 | 124 | 2557 | **0.6749** ± 0.0120 | **0.7603** ± 0.0476 | **0.7152** ± 0.0020 |
| eggs | 9 | `round1_stage3` | camada nova + bloco novo | 25920 | 124 | 2557 | 0.6016 ± 0.0059 | 0.7110 ± 0.0483 | 0.6631 ± 0.0064 |
| eggs | 9 | `round1_stage4` | tudo | 25920 | 124 | 2557 | 0.6123 ± 0.0224 | 0.7208 ± 0.0640 | 0.6728 ± 0.0223 |
| larvae | 2 | `stage1` | só o decoder | 27648 | 87 | 1757 | 0.5648 ± 0.1012 | 0.7360 ± 0.0560 | 0.7806 ± 0.0516 |
| larvae | 2 | `stage2` | tudo | 27648 | 87 | 1757 | 0.5971 ± 0.1135 | 0.7562 ± 0.0592 | 0.7973 ± 0.0578 |
| larvae | 2 | `round1_stage3` | camada nova + bloco novo | 27648 | 87 | 1757 | 0.6180 ± 0.1028 | 0.7665 ± 0.0555 | 0.8079 ± 0.0522 |
| larvae | 2 | `round1_stage4` | tudo | 27648 | 87 | 1757 | 0.6495 ± 0.1233 | 0.7917 ± 0.0726 | 0.8240 ± 0.0626 |
| larvae | 2 | `round2_stage3` | camada nova + bloco novo | 27648 | 87 | 1757 | **0.6679** ± 0.1117 | 0.8057 ± 0.0685 | **0.8334** ± 0.0565 |
| larvae | 2 | `round2_stage4` | tudo | 27648 | 87 | 1757 | 0.6594 ± 0.1079 | **0.8062** ± 0.0698 | 0.8292 ± 0.0547 |
| protozoan | 7 | `stage1` | só o decoder | 27648 | 235 | 4786 | 0.6166 ± 0.0353 | 0.6409 ± 0.0415 | 0.6153 ± 0.0400 |
| protozoan | 7 | `stage2` | tudo | 27648 | 235 | 4786 | **0.6746** ± 0.0199 | **0.6990** ± 0.0205 | **0.6780** ± 0.0194 |
| protozoan | 7 | `round1_stage3` | camada nova + bloco novo | 24192 | 235 | 4786 | 0.6401 ± 0.0219 | 0.6905 ± 0.0248 | 0.6559 ± 0.0245 |
| protozoan | 7 | `round1_stage4` | tudo | 24192 | 235 | 4786 | 0.6449 ± 0.0220 | 0.6901 ± 0.0221 | 0.6569 ± 0.0219 |

*Negrito = maior valor daquela métrica dentro do dataset (comparação pela média, entre os estágios do mesmo dataset nesta tabela).*

## Tabela 2 — 50% dos dados rotulados

Média ± desvio padrão amostral sobre os splits disponíveis, 4 casas decimais.

| dataset | classes | estágio | o que treina | dim do embedding | n_train (SVM) | n_test | kappa | acc | f1 |
|---|---|---|---|---|---|---|---|---|---|
| eggs | 9 | `stage1` | só o decoder | 27648 | 1276 | 2557 | 0.8519 ± 0.0191 | 0.8985 ± 0.0120 | 0.8670 ± 0.0110 |
| eggs | 9 | `stage2` | tudo | 27648 | 1276 | 2557 | **0.9062** ± 0.0032 | **0.9245** ± 0.0095 | **0.9137** ± 0.0038 |
| eggs | 9 | `round1_stage3` | camada nova + bloco novo | 25920 | 1276 | 2557 | 0.8734 ± 0.0159 | 0.9116 ± 0.0043 | 0.8879 ± 0.0129 |
| eggs | 9 | `round1_stage4` | tudo | 25920 | 1276 | 2557 | 0.8881 ± 0.0225 | 0.9225 ± 0.0133 | 0.8985 ± 0.0194 |
| larvae | 2 | `stage1` | só o decoder | 27648 | 878 | 1757 | 0.8119 ± 0.0105 | 0.8992 ± 0.0094 | 0.9060 ± 0.0052 |
| larvae | 2 | `stage2` | tudo | 27648 | 878 | 1757 | 0.8593 ± 0.0336 | 0.9310 ± 0.0147 | 0.9297 ± 0.0168 |
| larvae | 2 | `round1_stage3` | camada nova + bloco novo | 27648 | 878 | 1757 | 0.8324 ± 0.0402 | 0.9156 ± 0.0263 | 0.9162 ± 0.0201 |
| larvae | 2 | `round1_stage4` | tudo | 27648 | 878 | 1757 | 0.8661 ± 0.0235 | 0.9344 ± 0.0115 | 0.9330 ± 0.0118 |
| larvae | 2 | `round2_stage3` | camada nova + bloco novo | 27648 | 878 **(n=2)** | 1757 | 0.8591 ± 0.0146 | **0.9386** ± 0.0086 | 0.9296 ± 0.0073 |
| larvae | 2 | `round2_stage4` | tudo | 27648 | 878 **(n=2)** | 1757 | **0.8687** ± 0.0120 | 0.9382 ± 0.0161 | **0.9344** ± 0.0060 |
| protozoan | 7 | `stage1` | só o decoder | 27648 | 2390 | 4786 | 0.8082 ± 0.0127 | 0.8067 ± 0.0169 | 0.7896 ± 0.0175 |
| protozoan | 7 | `stage2` | tudo | 27648 | 2390 | 4786 | **0.8587** ± 0.0062 | **0.8661** ± 0.0025 | **0.8511** ± 0.0055 |
| protozoan | 7 | `round1_stage3` | camada nova + bloco novo | 24192 | 2390 | 4786 | 0.8180 ± 0.0211 | 0.8386 ± 0.0185 | 0.8159 ± 0.0142 |
| protozoan | 7 | `round1_stage4` | tudo | 24192 | 2390 | 4786 | 0.8440 ± 0.0093 | 0.8518 ± 0.0015 | 0.8365 ± 0.0164 |

*Negrito = maior valor daquela métrica dentro do dataset (comparação pela média, entre os estágios do mesmo dataset nesta tabela).*

*Nos três melhores de larvae nesta tabela — `round2_stage4` em kappa e f1, `round2_stage3` em acc — a média vem de 2 splits, não de 3 (ver `(n=2)` acima).*

**A linha marcada `(n=2)`** — larvae, `round2_stage3` e `round2_stage4`, pct50 —
tem só os splits 1 e 2. O split 3 não chegou à rodada 2. Ali a média e o desvio
saem de 2 valores, não de 3.

### Sobre `n_train` e `n_test`

`n_train` é quantas imagens **rotuladas** o SVM recebe para treinar, e é o que a
porcentagem controla: 5% dá 124 (eggs), 87 (larvae) e 235 (protozoan); 50% dá
1276, 878 e 2390. Já `n_test` é fixo por dataset e **não muda com a
porcentagem** — 2557 em eggs, 1757 em larvae, 4786 em protozoan são os mesmos nas
duas tabelas.

---

## Apêndice — as 82 linhas individuais

Sem média, para conferência linha a linha.

### 5% — 42 linhas

| dataset | split | estágio | kappa | acc | f1 |
|---|---|---|---|---|---|
| eggs | 1 | `stage1` | 0.6130 | 0.6724 | 0.6669 |
| eggs | 1 | `stage2` | 0.6859 | 0.7075 | 0.7152 |
| eggs | 1 | `round1_stage3` | 0.5974 | 0.6557 | 0.6565 |
| eggs | 1 | `round1_stage4` | 0.5936 | 0.6521 | 0.6530 |
| eggs | 2 | `stage1` | 0.6762 | 0.7736 | 0.7129 |
| eggs | 2 | `stage2` | 0.6767 | 0.7737 | 0.7132 |
| eggs | 2 | `round1_stage3` | 0.5991 | 0.7455 | 0.6635 |
| eggs | 2 | `round1_stage4` | 0.6371 | 0.7786 | 0.6970 |
| eggs | 3 | `stage1` | 0.5631 | 0.6931 | 0.6223 |
| eggs | 3 | `stage2` | 0.6621 | 0.7999 | 0.7172 |
| eggs | 3 | `round1_stage3` | 0.6084 | 0.7317 | 0.6692 |
| eggs | 3 | `round1_stage4` | 0.6063 | 0.7318 | 0.6686 |
| larvae | 1 | `stage1` | 0.4663 | 0.6885 | 0.7307 |
| larvae | 1 | `stage2` | 0.4663 | 0.6885 | 0.7307 |
| larvae | 1 | `round1_stage3` | 0.5031 | 0.7067 | 0.7496 |
| larvae | 1 | `round1_stage4` | 0.5074 | 0.7090 | 0.7518 |
| larvae | 1 | `round2_stage3` | 0.5390 | 0.7282 | 0.7682 |
| larvae | 1 | `round2_stage4` | 0.5349 | 0.7259 | 0.7661 |
| larvae | 2 | `stage1` | 0.5596 | 0.7219 | 0.7774 |
| larvae | 2 | `stage2` | 0.6566 | 0.7824 | 0.8275 |
| larvae | 2 | `round1_stage3` | 0.6498 | 0.7763 | 0.8240 |
| larvae | 2 | `round1_stage4` | 0.7125 | 0.8214 | 0.8560 |
| larvae | 2 | `round2_stage3` | 0.7285 | 0.8307 | 0.8640 |
| larvae | 2 | `round2_stage4` | 0.7221 | 0.8403 | 0.8609 |
| larvae | 3 | `stage1` | 0.6685 | 0.7977 | 0.8338 |
| larvae | 3 | `stage2` | 0.6685 | 0.7977 | 0.8338 |
| larvae | 3 | `round1_stage3` | 0.7011 | 0.8163 | 0.8503 |
| larvae | 3 | `round1_stage4` | 0.7286 | 0.8448 | 0.8642 |
| larvae | 3 | `round2_stage3` | 0.7361 | 0.8582 | 0.8680 |
| larvae | 3 | `round2_stage4` | 0.7214 | 0.8524 | 0.8607 |
| protozoan | 1 | `stage1` | 0.5831 | 0.6024 | 0.5735 |
| protozoan | 1 | `stage2` | 0.6695 | 0.6811 | 0.6607 |
| protozoan | 1 | `round1_stage3` | 0.6247 | 0.6623 | 0.6362 |
| protozoan | 1 | `round1_stage4` | 0.6266 | 0.6665 | 0.6389 |
| protozoan | 2 | `stage1` | 0.6133 | 0.6848 | 0.6531 |
| protozoan | 2 | `stage2` | 0.6577 | 0.7214 | 0.6990 |
| protozoan | 2 | `round1_stage3` | 0.6305 | 0.7089 | 0.6833 |
| protozoan | 2 | `round1_stage4` | 0.6388 | 0.7104 | 0.6812 |
| protozoan | 3 | `stage1` | 0.6534 | 0.6355 | 0.6195 |
| protozoan | 3 | `stage2` | 0.6965 | 0.6946 | 0.6744 |
| protozoan | 3 | `round1_stage3` | 0.6651 | 0.7005 | 0.6482 |
| protozoan | 3 | `round1_stage4` | 0.6693 | 0.6935 | 0.6506 |

### 50% — 40 linhas

| dataset | split | estágio | kappa | acc | f1 |
|---|---|---|---|---|---|
| eggs | 1 | `stage1` | 0.8736 | 0.9020 | 0.8792 |
| eggs | 1 | `stage2` | 0.9035 | 0.9224 | 0.9101 |
| eggs | 1 | `round1_stage3` | 0.8625 | 0.9162 | 0.8771 |
| eggs | 1 | `round1_stage4` | 0.8624 | 0.9133 | 0.8762 |
| eggs | 2 | `stage1` | 0.8375 | 0.9083 | 0.8638 |
| eggs | 2 | `stage2` | 0.9056 | 0.9348 | 0.9133 |
| eggs | 2 | `round1_stage3` | 0.8661 | 0.9110 | 0.8842 |
| eggs | 2 | `round1_stage4` | 0.8978 | 0.9377 | 0.9088 |
| eggs | 3 | `stage1` | 0.8445 | 0.8850 | 0.8579 |
| eggs | 3 | `stage2` | 0.9097 | 0.9162 | 0.9177 |
| eggs | 3 | `round1_stage3` | 0.8917 | 0.9077 | 0.9022 |
| eggs | 3 | `round1_stage4` | 0.9041 | 0.9164 | 0.9106 |
| larvae | 1 | `stage1` | 0.8055 | 0.8989 | 0.9028 |
| larvae | 1 | `stage2` | 0.8703 | 0.9393 | 0.9351 |
| larvae | 1 | `round1_stage3` | 0.8449 | 0.9338 | 0.9225 |
| larvae | 1 | `round1_stage4` | 0.8902 | 0.9477 | 0.9451 |
| larvae | 1 | `round2_stage3` | 0.8488 | 0.9325 | 0.9244 |
| larvae | 1 | `round2_stage4` | 0.8603 | 0.9268 | 0.9301 |
| larvae | 2 | `stage1` | 0.8062 | 0.8900 | 0.9031 |
| larvae | 2 | `stage2` | 0.8216 | 0.9139 | 0.9108 |
| larvae | 2 | `round1_stage3` | 0.7875 | 0.8855 | 0.8937 |
| larvae | 2 | `round1_stage4` | 0.8432 | 0.9281 | 0.9216 |
| larvae | 2 | `round2_stage3` | 0.8695 | 0.9447 | 0.9347 |
| larvae | 2 | `round2_stage4` | 0.8772 | 0.9495 | 0.9386 |
| larvae | 3 | `stage1` | 0.8240 | 0.9089 | 0.9120 |
| larvae | 3 | `stage2` | 0.8861 | 0.9397 | 0.9431 |
| larvae | 3 | `round1_stage3` | 0.8649 | 0.9275 | 0.9325 |
| larvae | 3 | `round1_stage4` | 0.8649 | 0.9275 | 0.9325 |
| protozoan | 1 | `stage1` | 0.8219 | 0.8149 | 0.8029 |
| protozoan | 1 | `stage2` | 0.8523 | 0.8685 | 0.8526 |
| protozoan | 1 | `round1_stage3` | 0.7939 | 0.8174 | 0.8046 |
| protozoan | 1 | `round1_stage4` | 0.8547 | 0.8513 | 0.8519 |
| protozoan | 2 | `stage1` | 0.7969 | 0.8179 | 0.7962 |
| protozoan | 2 | `stage2` | 0.8647 | 0.8635 | 0.8557 |
| protozoan | 2 | `round1_stage3` | 0.8330 | 0.8515 | 0.8319 |
| protozoan | 2 | `round1_stage4` | 0.8398 | 0.8507 | 0.8384 |
| protozoan | 3 | `stage1` | 0.8057 | 0.7874 | 0.7698 |
| protozoan | 3 | `stage2` | 0.8591 | 0.8665 | 0.8449 |
| protozoan | 3 | `round1_stage3` | 0.8271 | 0.8470 | 0.8113 |
| protozoan | 3 | `round1_stage4` | 0.8376 | 0.8535 | 0.8193 |

---

## Nota de método

**`acc` é acurácia macro, não bruta.** `compute_metrics` chama
`multiclass_accuracy` com `average="macro"`
(`src/metrics/classification.py:31`), que é a média das acurácias por classe. A
acurácia bruta existe nos CSVs, na coluna separada `acc_raw`, e não foi trazida
para as tabelas acima.

**`ckpt_epoch`** registra em que época o `best_kappa.ckpt` foi selecionado.
Entre os 82 trabalhos, **23 foram selecionados na época 0** — ou seja, a
primeira época já foi a melhor pelo critério de seleção do treino. Os demais
59 vão da época 1 à 500.

**`predict_s`** é uma medição nova: o tempo de `predict` do SVM em segundos. Os
CSVs antigos do repositório não têm essa coluna; só `extract_s` e `fit_s`.

**Colunas constantes nos 82 trabalhos:** `method =
SVM_SPiFiL_growth_flatten_labcru`, `embed_mode = flatten`, `imagenet_norm =
False`, `max_iter = -1`, `fit_status = 0` (todos os ajustes convergiram sem
aviso). Nenhum `kappa`, `acc` ou `f1` nulo.
