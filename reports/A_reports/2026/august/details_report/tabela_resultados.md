# Estágios 1 a 4 do encoder FLIM + SPiFiL — teste e validação lado a lado, sem misturar

Tabelas de kappa, f1 e acurácia para os três datasets do projeto, com **5%** e com **50%**
dos dados rotulados para treinar o SVM.

Os números não foram recalculados aqui. Eles vêm de duas fontes que já estavam no disco, e
**as duas não são intercambiáveis** — leia a seção "Duas fontes" antes de comparar qualquer
coisa entre tabelas.

## Como ler os estágios

A regra que define o estágio está em `src/modules/autoencoder_flim_module.py:833`; os nomes
estão em `src/modules/autoencoder_flim_module.py:267-274`.

- **Estágio 1** — encoder FLIM congelado; só o decoder ResNet, que começa aleatório, treina.
- **Estágio 2** — encoder e decoder destravados, treinando juntos, a partir do melhor
  checkpoint do estágio 1.
- **Estágio 3** — uma camada SPiFiL nova foi recortada e acrescentada ao encoder; o encoder
  inteiro volta a ficar congelado (a camada nova inclusive) e só o decoder treina.
- **Estágio 4** — a mesma rodada de crescimento, agora com tudo destravado. Ele não tem
  número próprio no treinador: é registrado internamente como estágio 2
  (`artifacts/spifil_growth/grid3/eggs_split1_pct5/round1_stage4/run_metadata.json`,
  campo `"stage": 2`).

## Duas fontes

| Fonte | O que é | Conjunto | Braço | Estágios |
|---|---|---|---|---|
| `results/eval_autoencoder_lab.csv` e `results/eval_autoencoder_lab_flat.csv` | avaliação oficial, rodada depois do treino por `src/evaluate/eval_autoencoder.py:174` | **teste** | 48-d e flatten | 1 e 2 |
| `artifacts/spifil_growth/grid3/**/run_metadata.json` e os `wandb-summary.json` ao lado | sonda SVM interna, rodada a cada época durante o treino | **validação** | só flatten | 1, 2, 3 e 4 |

O avaliador oficial usa o único SVM do repositório (`src/utils/evaluate.py:437`) e a única
função de métrica (`src/metrics/classification.py:31`). A sonda interna usa um `SVC` próprio
(`src/modules/autoencoder_flim_module.py:485`), declarado e justificado em `:484`.

O próprio código avisa em `src/evaluate/eval_autoencoder.py:59-63` que os dois números não
são comparáveis. Por isso os estágios 3 e 4 **não** foram enfiados nas tabelas de teste: eles
têm tabela própria, de validação, onde os estágios 1 e 2 do mesmo grid aparecem junto para
servir de referência interna.

Cada célula é **média ± desvio-padrão das 3 splits**, com 4 casas decimais.

---

# 5% dos dados rotulados

## 1.1 — Teste, braço 48-d (`lab`, average pooling)

Protocolo atual: o SVM recebe um vetor de 48 dimensões.
Fonte: `results/eval_autoencoder_lab.csv` (18 linhas com `percentage=5`).

| Dataset | Métrica | Estágio 1<br>encoder congelado | Estágio 2<br>tudo destravado | Estágio 3<br>camada nova, congelado | Estágio 4<br>camada nova, destravado |
|---|---|---|---|---|---|
| **eggs (Helminth Eggs, 9 classes)** | kappa | 0.5380 ± 0.0926 | 0.5794 ± 0.0673 | - | - |
|  | f1 | 0.5451 ± 0.0759 | 0.5728 ± 0.0651 | - | - |
|  | acc | 0.5693 ± 0.0689 | 0.5974 ± 0.0544 | - | - |
| **larvae (Helminth Larvae, 2 classes)** | kappa | 0.7161 ± 0.0892 | 0.7588 ± 0.0531 | - | - |
|  | f1 | 0.8580 ± 0.0446 | 0.8794 ± 0.0266 | - | - |
|  | acc | 0.8516 ± 0.0468 | 0.8771 ± 0.0336 | - | - |
| **protozoan (Protozoan Cysts, 7 classes)** | kappa | 0.5567 ± 0.0044 | 0.5889 ± 0.0226 | - | - |
|  | f1 | 0.5405 ± 0.0546 | 0.5625 ± 0.0545 | - | - |
|  | acc | 0.5276 ± 0.0488 | 0.5468 ± 0.0430 | - | - |

Estágios 3 e 4 estão vazios porque **o crescimento SPiFiL nunca rodou neste braço** — todo
`run_metadata.json` do grid traz `"embed_mode": "flatten"`.

## 1.2 — Teste, braço flatten (`lab_flat`, 27.648 dimensões)

Fonte: `results/eval_autoencoder_lab_flat.csv` (18 linhas com `percentage=5`).

| Dataset | Métrica | Estágio 1<br>encoder congelado | Estágio 2<br>tudo destravado | Estágio 3<br>camada nova, congelado | Estágio 4<br>camada nova, destravado |
|---|---|---|---|---|---|
| **eggs (Helminth Eggs, 9 classes)** | kappa | 0.6174 ± 0.0567 | 0.7052 ± 0.0368 | - | - |
|  | f1 | 0.6674 ± 0.0453 | 0.7441 ± 0.0330 | - | - |
|  | acc | 0.7130 ± 0.0535 | 0.7771 ± 0.0493 | - | - |
| **larvae (Helminth Larvae, 2 classes)** | kappa | 0.5648 ± 0.1012 | 0.5721 ± 0.0863 | - | - |
|  | f1 | 0.7806 ± 0.0516 | 0.7844 ± 0.0441 | - | - |
|  | acc | 0.7360 ± 0.0560 | 0.7392 ± 0.0498 | - | - |
| **protozoan (Protozoan Cysts, 7 classes)** | kappa | 0.6166 ± 0.0353 | 0.6842 ± 0.0318 | - | - |
|  | f1 | 0.6153 ± 0.0400 | 0.6847 ± 0.0346 | - | - |
|  | acc | 0.6409 ± 0.0415 | 0.6946 ± 0.0380 | - | - |

Estágios 3 e 4 estão vazios porque **nenhum checkpoint de crescimento foi avaliado no
conjunto de teste**. Os treinos existem; a avaliação de teste deles não.

## 1.3 — Crescimento SPiFiL, validação, braço flatten

Fonte: `artifacts/spifil_growth/grid3/<dataset>_split<N>_pct5/`.
São 9 células (3 datasets × 3 splits), todas com estágio 1, estágio 2 e a rodada 1 completa.

| Dataset | Métrica | Estágio 1<br>encoder congelado | Estágio 2<br>tudo destravado | Estágio 3 (rodada 1)<br>camada nova, congelado | Estágio 4 (rodada 1)<br>camada nova, destravado |
|---|---|---|---|---|---|
| **eggs (Helminth Eggs, 9 classes)** | kappa | 0.6101 ± 0.0629 | 0.6773 ± 0.0148 | 0.5902 ± 0.0131 | 0.6607 ± 0.0736 |
|  | f1 | 0.6581 ± 0.0574 | 0.7117 ± 0.0096 | 0.6573 ± 0.0086 | 0.6876 ± 0.0773 |
|  | acc | 0.7086 ± 0.0708 | 0.7679 ± 0.0355 | 0.7282 ± 0.0409 | 0.7453 ± 0.0528 |
| **larvae (Helminth Larvae, 2 classes)** | kappa | 0.5378 ± 0.1281 | 0.5707 ± 0.1176 | 0.5729 ± 0.1263 | 0.6203 ± 0.1375 |
|  | f1 | 0.7665 ± 0.0657 | 0.7722 ± 0.0681 | 0.7847 ± 0.0645 | 0.7528 ± 0.0782 |
|  | acc | 0.7229 ± 0.0728 | 0.7352 ± 0.0701 | 0.7410 ± 0.0674 | 0.7180 ± 0.0791 |
| **protozoan (Protozoan Cysts, 7 classes)** | kappa | 0.6088 ± 0.0458 | 0.6741 ± 0.0272 | 0.6222 ± 0.0199 | 0.6371 ± 0.0205 |
|  | f1 | 0.6109 ± 0.0135 | 0.6706 ± 0.0194 | 0.6336 ± 0.0280 | 0.6511 ± 0.0105 |
|  | acc | 0.6407 ± 0.0176 | 0.6867 ± 0.0201 | 0.6579 ± 0.0247 | 0.6783 ± 0.0094 |

---

# 50% dos dados rotulados

## 2.1 — Teste, braço 48-d (`lab`, average pooling)

Fonte: `results/eval_autoencoder_lab.csv` (18 linhas com `percentage=50`).

| Dataset | Métrica | Estágio 1<br>encoder congelado | Estágio 2<br>tudo destravado | Estágio 3<br>camada nova, congelado | Estágio 4<br>camada nova, destravado |
|---|---|---|---|---|---|
| **eggs (Helminth Eggs, 9 classes)** | kappa | 0.7460 ± 0.0298 | 0.8118 ± 0.0097 | - | - |
|  | f1 | 0.7496 ± 0.0190 | 0.7882 ± 0.0209 | - | - |
|  | acc | 0.7349 ± 0.0285 | 0.7766 ± 0.0210 | - | - |
| **larvae (Helminth Larvae, 2 classes)** | kappa | 0.8578 ± 0.0030 | 0.9304 ± 0.0034 | - | - |
|  | f1 | 0.9289 ± 0.0015 | 0.9652 ± 0.0017 | - | - |
|  | acc | 0.9259 ± 0.0160 | 0.9694 ± 0.0104 | - | - |
| **protozoan (Protozoan Cysts, 7 classes)** | kappa | 0.6950 ± 0.0049 | 0.7823 ± 0.0058 | - | - |
|  | f1 | 0.6950 ± 0.0372 | 0.7884 ± 0.0270 | - | - |
|  | acc | 0.6605 ± 0.0301 | 0.7589 ± 0.0315 | - | - |

## 2.2 — Teste, braço flatten (`lab_flat`, 27.648 dimensões)

Fonte: `results/eval_autoencoder_lab_flat.csv` (18 linhas com `percentage=50`).

| Dataset | Métrica | Estágio 1<br>encoder congelado | Estágio 2<br>tudo destravado | Estágio 3<br>camada nova, congelado | Estágio 4<br>camada nova, destravado |
|---|---|---|---|---|---|
| **eggs (Helminth Eggs, 9 classes)** | kappa | 0.8519 ± 0.0191 | 0.9179 ± 0.0042 | - | - |
|  | f1 | 0.8670 ± 0.0110 | 0.9259 ± 0.0075 | - | - |
|  | acc | 0.8985 ± 0.0120 | 0.9343 ± 0.0167 | - | - |
| **larvae (Helminth Larvae, 2 classes)** | kappa | 0.8119 ± 0.0105 | 0.8759 ± 0.0125 | - | - |
|  | f1 | 0.9060 ± 0.0052 | 0.9380 ± 0.0062 | - | - |
|  | acc | 0.8992 ± 0.0094 | 0.9383 ± 0.0083 | - | - |
| **protozoan (Protozoan Cysts, 7 classes)** | kappa | 0.8082 ± 0.0127 | 0.8697 ± 0.0066 | - | - |
|  | f1 | 0.7896 ± 0.0175 | 0.8590 ± 0.0119 | - | - |
|  | acc | 0.8067 ± 0.0169 | 0.8687 ± 0.0158 | - | - |

## 2.3 — Crescimento SPiFiL, validação, braço flatten

Fonte: `artifacts/spifil_growth/grid3/<dataset>_split<N>_pct50/`.

| Dataset | Métrica | Estágio 1<br>encoder congelado | Estágio 2<br>tudo destravado | Estágio 3 (rodada 1)<br>camada nova, congelado | Estágio 4 (rodada 1)<br>camada nova, destravado |
|---|---|---|---|---|---|
| **eggs (Helminth Eggs, 9 classes)** | kappa | 0.8634 ± 0.0044 | 0.9162 ± 0.0110 | 0.8639 ± 0.0068 | 0.9064 ± 0.0220 |
|  | f1 | 0.8775 ± 0.0085 | 0.9223 ± 0.0016 | 0.8831 ± 0.0019 | 0.9118 ± 0.0120 |
|  | acc | 0.9130 ± 0.0212 | 0.9321 ± 0.0065 | 0.9169 ± 0.0136 | 0.9321 ± 0.0090 |
| **larvae (Helminth Larvae, 2 classes)** | kappa | 0.7927 ± 0.0081 | 0.8535 ± 0.0494 | 0.7980 ± 0.0532 | 0.8440 ± 0.0271 |
|  | f1 | 0.8964 ± 0.0040 | 0.9206 ± 0.0268 | 0.8990 ± 0.0266 | 0.9129 ± 0.0100 |
|  | acc | 0.8913 ± 0.0029 | 0.9201 ± 0.0278 | 0.8922 ± 0.0332 | 0.9130 ± 0.0158 |
| **protozoan (Protozoan Cysts, 7 classes)** | kappa | 0.8116 ± 0.0336 | 0.8696 ± 0.0072 | 0.7954 ± 0.0124 | 0.8376 ± 0.0189 |
|  | f1 | 0.7893 ± 0.0319 | 0.8520 ± 0.0164 | 0.7913 ± 0.0183 | 0.8228 ± 0.0190 |
|  | acc | 0.8045 ± 0.0214 | 0.8665 ± 0.0233 | 0.8222 ± 0.0212 | 0.8408 ± 0.0204 |

---

## Rodadas além da primeira

O laço para sozinho quando o kappa não melhora — `should_stop` em
`scripts/spifil_growth_loop.py:160-165`, com tolerância 0.01 e paciência 1. Por isso o número
de rodadas varia de célula para célula. Todas as 18 células têm a rodada 1; só estas cinco
foram além, e por isso as rodadas 2 e 3 não entram nas tabelas médias (não daria para tirar
média de 3 splits).

| Pct | Dataset | Split | Estágio | kappa (validação) |
|---|---|---|---|---|
| 5% | eggs | 2 | round2_stage3 | 0.6866 |
| 5% | eggs | 2 | round2_stage4 | 0.7280 |
| 5% | larvae | 1 | round2_stage3 | 0.5136 |
| 5% | larvae | 1 | round2_stage4 | 0.6081 |
| 5% | larvae | 1 | round3_stage3 | 0.6202 |
| 5% | larvae | 1 | round3_stage4 | 0.6280 |
| 5% | larvae | 2 | round2_stage3 | 0.5917 |
| 5% | larvae | 2 | round2_stage4 | 0.6256 |
| 5% | larvae | 3 | round2_stage3 | 0.7142 |
| 5% | larvae | 3 | round2_stage4 | 0.7129 |
| 50% | larvae | 2 | round2_stage3 | 0.8056 |
| 50% | larvae | 2 | round2_stage4 | 0.8142 |

Nenhuma célula chegou à rodada 4, que é o teto padrão (`scripts/spifil_growth_loop.py:367`).

---

## Cuidados na leitura

**Não compare a tabela de validação com as de teste.** São conjuntos diferentes e
implementações de SVM diferentes. Dentro da tabela 1.3 (ou 2.3) a comparação entre estágios é
válida, porque tudo ali foi medido do mesmo jeito.

**Não compare os dois braços entre si.** 48-d e flatten são embeddings diferentes; a diferença
entre eles não diz nada sobre o treino.

**O kappa e o par f1/acc do crescimento vêm de épocas diferentes.** O kappa é o
`best_val_svm_kappa`, da melhor época — é ele que escolhe o checkpoint salvo
(`ckpt_monitor: "probe/svm_kappa"`). Já f1 e acc só existem no resumo do W&B como
`probe/svm_f1` e `probe/svm_acc`, que são da **última** época. Medi a distância entre as duas
épocas nas 72 execuções: a mediana da diferença de kappa é 0.0012, mas o pior caso chega a
0.1454 (larvae split 1, 5%, estágio 4: última época 0.3354 contra melhor 0.4807). Onde essa
diferença for grande, f1 e acc não descrevem o modelo que ficou salvo.

**O estágio 3 congela a camada nova junto com o resto.** O congelamento é do encoder inteiro
(`src/models/models.py:675-676`), sem filtro por camada. No estágio 3 só o decoder aprende.

**O desvio-padrão vem de 3 amostras.** Com n=3 ele mostra a dispersão entre as splits, mas não
sustenta teste de significância.

ponytail: o desvio-padrão é o amostral simples sobre as 3 splits, sem intervalo de confiança e
sem teste de significância; as ferramentas para isso já existem em `statistics/tools/`.

ponytail: o apêndice traz kappa por split para todas as fontes, mas f1 e acc por split só para
os CSVs de teste — para o crescimento eles estão nos `wandb-summary.json` de cada estágio.

---

## Apêndice — valores por split

### 5%, teste, braço 48-d

| Dataset | Split | E1 kappa | E1 f1 | E1 acc | E2 kappa | E2 f1 | E2 acc |
|---|---|---|---|---|---|---|---|
| eggs | 1 | 0.4546 | 0.4789 | 0.4986 | 0.5083 | 0.5057 | 0.5386 |
| eggs | 2 | 0.6376 | 0.6279 | 0.6363 | 0.6420 | 0.6356 | 0.6460 |
| eggs | 3 | 0.5219 | 0.5286 | 0.5731 | 0.5881 | 0.5771 | 0.6075 |
| larvae | 1 | 0.6387 | 0.8193 | 0.8132 | 0.6980 | 0.8490 | 0.8383 |
| larvae | 2 | 0.8137 | 0.9068 | 0.9037 | 0.7959 | 0.8980 | 0.8957 |
| larvae | 3 | 0.6959 | 0.8479 | 0.8380 | 0.7825 | 0.8912 | 0.8972 |
| protozoan | 1 | 0.5529 | 0.5466 | 0.5338 | 0.5646 | 0.5549 | 0.5439 |
| protozoan | 2 | 0.5557 | 0.5918 | 0.5731 | 0.6094 | 0.6205 | 0.5911 |
| protozoan | 3 | 0.5616 | 0.4831 | 0.4760 | 0.5926 | 0.5123 | 0.5053 |

### 5%, teste, braço flatten

| Dataset | Split | E1 kappa | E1 f1 | E1 acc | E2 kappa | E2 f1 | E2 acc |
|---|---|---|---|---|---|---|---|
| eggs | 1 | 0.6130 | 0.6669 | 0.6724 | 0.6806 | 0.7197 | 0.7278 |
| eggs | 2 | 0.6762 | 0.7129 | 0.7736 | 0.7476 | 0.7816 | 0.8263 |
| eggs | 3 | 0.5631 | 0.6223 | 0.6931 | 0.6875 | 0.7311 | 0.7770 |
| larvae | 1 | 0.4663 | 0.7307 | 0.6885 | 0.4941 | 0.7448 | 0.7007 |
| larvae | 2 | 0.5596 | 0.7774 | 0.7219 | 0.5574 | 0.7764 | 0.7216 |
| larvae | 3 | 0.6685 | 0.8338 | 0.7977 | 0.6649 | 0.8320 | 0.7955 |
| protozoan | 1 | 0.5831 | 0.5735 | 0.6024 | 0.6501 | 0.6508 | 0.6674 |
| protozoan | 2 | 0.6133 | 0.6531 | 0.6848 | 0.6894 | 0.7199 | 0.7380 |
| protozoan | 3 | 0.6534 | 0.6195 | 0.6355 | 0.7131 | 0.6835 | 0.6783 |

### 5%, crescimento SPiFiL, validação (kappa da melhor época)

| Dataset | Split | E1 kappa | E2 kappa | E3 r1 kappa | E4 r1 kappa |
|---|---|---|---|---|---|
| eggs | 1 | 0.5780 | 0.6877 | 0.6012 | 0.6651 |
| eggs | 2 | 0.6826 | 0.6838 | 0.5937 | 0.7321 |
| eggs | 3 | 0.5697 | 0.6604 | 0.5758 | 0.5850 |
| larvae | 1 | 0.4443 | 0.4491 | 0.4443 | 0.4807 |
| larvae | 2 | 0.4852 | 0.5793 | 0.5776 | 0.6246 |
| larvae | 3 | 0.6839 | 0.6839 | 0.6969 | 0.7556 |
| protozoan | 1 | 0.5732 | 0.6708 | 0.6225 | 0.6446 |
| protozoan | 2 | 0.5927 | 0.6487 | 0.6022 | 0.6139 |
| protozoan | 3 | 0.6604 | 0.7027 | 0.6420 | 0.6527 |

### 50%, teste, braço 48-d

| Dataset | Split | E1 kappa | E1 f1 | E1 acc | E2 kappa | E2 f1 | E2 acc |
|---|---|---|---|---|---|---|---|
| eggs | 1 | 0.7360 | 0.7289 | 0.7085 | 0.8081 | 0.7654 | 0.7544 |
| eggs | 2 | 0.7796 | 0.7663 | 0.7651 | 0.8228 | 0.7931 | 0.7962 |
| eggs | 3 | 0.7225 | 0.7536 | 0.7310 | 0.8045 | 0.8062 | 0.7791 |
| larvae | 1 | 0.8599 | 0.9299 | 0.9176 | 0.9265 | 0.9633 | 0.9695 |
| larvae | 2 | 0.8543 | 0.9271 | 0.9444 | 0.9326 | 0.9663 | 0.9797 |
| larvae | 3 | 0.8593 | 0.9296 | 0.9157 | 0.9322 | 0.9661 | 0.9589 |
| protozoan | 1 | 0.6978 | 0.7189 | 0.6742 | 0.7861 | 0.8021 | 0.7616 |
| protozoan | 2 | 0.6894 | 0.7140 | 0.6813 | 0.7852 | 0.8057 | 0.7889 |
| protozoan | 3 | 0.6979 | 0.6521 | 0.6259 | 0.7756 | 0.7572 | 0.7261 |

### 50%, teste, braço flatten

| Dataset | Split | E1 kappa | E1 f1 | E1 acc | E2 kappa | E2 f1 | E2 acc |
|---|---|---|---|---|---|---|---|
| eggs | 1 | 0.8736 | 0.8792 | 0.9020 | 0.9158 | 0.9253 | 0.9265 |
| eggs | 2 | 0.8375 | 0.8638 | 0.9083 | 0.9227 | 0.9337 | 0.9535 |
| eggs | 3 | 0.8445 | 0.8579 | 0.8850 | 0.9152 | 0.9187 | 0.9229 |
| larvae | 1 | 0.8055 | 0.9028 | 0.8989 | 0.8670 | 0.9335 | 0.9351 |
| larvae | 2 | 0.8062 | 0.9031 | 0.8900 | 0.8902 | 0.9451 | 0.9477 |
| larvae | 3 | 0.8240 | 0.9120 | 0.9089 | 0.8706 | 0.9353 | 0.9320 |
| protozoan | 1 | 0.8219 | 0.8029 | 0.8149 | 0.8766 | 0.8639 | 0.8648 |
| protozoan | 2 | 0.7969 | 0.7962 | 0.8179 | 0.8636 | 0.8455 | 0.8551 |
| protozoan | 3 | 0.8057 | 0.7698 | 0.7874 | 0.8688 | 0.8676 | 0.8860 |

### 50%, crescimento SPiFiL, validação (kappa da melhor época)

| Dataset | Split | E1 kappa | E2 kappa | E3 r1 kappa | E4 r1 kappa |
|---|---|---|---|---|---|
| eggs | 1 | 0.8685 | 0.9036 | 0.8635 | 0.8811 |
| eggs | 2 | 0.8610 | 0.9208 | 0.8709 | 0.9216 |
| eggs | 3 | 0.8608 | 0.9241 | 0.8572 | 0.9163 |
| larvae | 1 | 0.7982 | 0.8874 | 0.8027 | 0.8735 |
| larvae | 2 | 0.7835 | 0.7969 | 0.7427 | 0.8202 |
| larvae | 3 | 0.7966 | 0.8763 | 0.8487 | 0.8383 |
| protozoan | 1 | 0.8412 | 0.8744 | 0.7996 | 0.8593 |
| protozoan | 2 | 0.7752 | 0.8613 | 0.7814 | 0.8245 |
| protozoan | 3 | 0.8184 | 0.8732 | 0.8052 | 0.8291 |
