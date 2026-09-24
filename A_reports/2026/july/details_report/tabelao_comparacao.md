# Tabelão — todos os braços lado a lado (Cohen κ)

**Data:** 2026-07-29 · Consolida os resultados espalhados por `report_2026-07-22.md`,
`report_2026-07-28.md` e `report_2026-07-29.md` numa comparação única.

**Métrica: Cohen's κ.** É a única comparável entre todos os braços — κ não tem variante
macro/weighted, então não sofre do problema de convenção que invalida comparar acurácia e F1
entre os CSVs "ours" (macro) e os do Felipe (global). Ver C3 em `check_sanity.md`.

| Braço | O que é | Fonte |
|:--|:--|:--|
| SVM + FLIM cru | SVM linear sobre encoder FLIM **sem** treino supervisionado | `data/reports_felipe/svm/` |
| MLP Felipe (frozen) | MLP 46.601 params, encoder congelado | `data/reports_felipe/flim_mlp/` |
| MLP Felipe (unfrozen) | MLP 46.601 params, fine-tune end-to-end | `data/reports_felipe/flim_mlp/` |
| Head Sigmoid | nossa cabeça 1.401 params, sem ativação de saída | `results/sigmoid2l_test_results.csv` |
| Head ReLU | idem + ReLU antes do Softmax | `results/relu2l_test_results.csv` |
| Head Softplus | idem + Softplus antes do Softmax | `results/softplus2l_test_results.csv` |

---

## Diagramas dos 6 braços

Todos partem do **mesmo encoder FLIM de 3 blocos**. As formas abaixo são medidas reais
(entrada 200×200), não estimativas:

```
entrada (3, 200, 200)
  │
  ├─ conv1: Conv2d(3→24, 5×5)  + ReLU + MaxPool(3×3, s2)  →  (24, 99, 99)    1.824 p
  ├─ conv2: Conv2d(24→32, 5×5) + ReLU + MaxPool(3×3, s2)  →  (32, 49, 49)   19.232 p
  └─ conv3: Conv2d(32→48, 5×5) + ReLU + MaxPool(3×3, s2)  →  (48, 24, 24)   38.448 p
                                                    ENCODER total  59.504 params
       (protozoan usa 24→30→48: 55.902 params)
```

**O ponto de divergência é o que se faz com o mapa `(48, 24, 24)`:**

| caminho | operação | dimensão vista pelo classificador |
|:--|:--|--:|
| A | `flatten` | 48 × 24 × 24 = **27.648-d** |
| B | `AdaptiveAvgPool2d(1)` → `flatten` | **48-d** |

Fator **576×**. O caminho B descarta toda a informação de *onde* cada ativação ocorre.
Quem usa A e quem usa B está marcado em cada diagrama.

### 1. SVM + FLIM cru — caminho A (27.648-d)

```
ENCODER FLIM (pesos do disco, NUNCA treinado de forma supervisionada)   ❄ congelado
  → (48, 24, 24) ──[A] flatten──> 27.648-d
      → SVC(kernel="linear", C=1e2, ovo)      C(C-1)/2 hiperplanos × 27.649 params
```
Nenhum gradiente em lugar nenhum: o encoder sai do disco e vai direto para o SVM.

### 2. MLP Felipe (frozen) — caminho B (48-d)

```
ENCODER FLIM ❄ congelado
  → (48, 24, 24) ──[B] GAP──> 48-d
      → Linear(48,256) → ReLU → Dropout(0.3)
      → Linear(256,128) → ReLU → Dropout(0.3)
      → Linear(128,C)  → logits → CrossEntropyLoss
                                          HEAD: 46.601 params (eggs)
```
Só a cabeça treina. 300 épocas, early stopping (patience 50), `weight_decay=1e-4`.

### 3. MLP Felipe (unfrozen) — caminho B (48-d)

```
ENCODER FLIM  treina (lr × 0.1, discriminativo)
  → (48, 24, 24) ──[B] GAP──> 48-d
      → [mesma cabeça de 46.601 params do braço 2] → logits → CrossEntropyLoss
                        TOTAL treinável: 59.504 + 46.601 = 106.105 params
```

### 4. Head Sigmoid (nossa) — caminho B (48-d)

```
ENCODER FLIM treina (lr uniforme 5e-4, weight_decay 5e-2)
  → (48, 24, 24) ──[B] GAP──> 48-d
      → Linear(48,24) → Sigmoid → Linear(24,C) → Softmax
      
                                          HEAD: 1.401 params (eggs)
                        TOTAL treinável: 59.504 + 1.401 = 60.905 params
```
100 épocas fixas, sem early stopping, sem augmentation.

### 5. Head ReLU (nossa) — caminho B (48-d)

```
      → Linear(48,24) → Sigmoid → Linear(24,C) → ReLU → Softmax
                                                   └─ zera todo logit ≤ 0
                                                      derivada = 0 → gradiente morto
                                          HEAD: 1.401 params (ReLU não tem parâmetros)
```
Idêntico ao braço 4 em tudo o mais. Foi este ReLU que matou 4 runs desde a inicialização.

### 6. Head Softplus (nossa) — caminho B (48-d)

```
      → Linear(48,24) → Sigmoid → Linear(24,C) → ✔ Softplus → Softmax
                                                   └─ log(1+e^z), sempre > 0
                                                      derivada = sigmoid(z), nunca 0
                                          HEAD: 1.401 params (Softplus não tem parâmetros)
```
Idêntico ao braço 5, trocando só a ativação de saída. `state_dict` igual aos braços 4 e 5.

**Procedência dos diagramas.** Os braços 4, 5 e 6 são leitura direta do código (`src/models/models.py`). Os braços 1, 2 e 3 são
**reconstruídos**: o pipeline do Felipe não está versionado neste repo. O `flatten` do braço 1
vem de `src/utils/evaluate.py:281` (o pipeline SVM do repo) e é corroborado pelo `test_time`
dos CSVs dele — 0,85 s por imagem em eggs, incompatível com 48-d. Que a MLP dele seja *pooled*
é inferência a partir de d≫n, não leitura de código.

---

## Tabelão 1 — Cohen κ médio ± dp sobre os 3 splits

| Dataset | pct | SVM + FLIM cru | MLP Felipe (frozen) | MLP Felipe (unfrozen) | Head Sigmoid | Head ReLU | Head Softplus |
|:--|--:|--:|--:|--:|--:|--:|--:|
| eggs | 5 | **0.585 ± 0.051** | 0.074 ± 0.019 | 0.531 ± 0.070 | — | 0.002 ± 0.005 | _0.001 ± 0.001_ |
| eggs | 75 | 0.870 ± 0.018 | 0.651 ± 0.039 | **0.919 ± 0.013** | 0.452 ± 0.391 | 0.560 ± 0.084 | _0.321 ± 0.416_ |
| larvae | 5 | 0.571 ± 0.121 | 0.655 ± 0.073 | **0.855 ± 0.008** | — | 0.199 ± 0.344 | _0.198 ± 0.343_ |
| larvae | 75 | 0.835 ± 0.013 | 0.826 ± 0.041 | **0.950 ± 0.020** | 0.929 ± 0.014 | _0.311 ± 0.538_ | 0.923 ± 0.009 |
| protozoan | 5 | **0.623 ± 0.025** | 0.214 ± 0.012 | 0.597 ± 0.057 | — | _0.014 ± 0.024_ | 0.020 ± 0.034 |
| protozoan | 75 | 0.837 ± 0.018 | 0.592 ± 0.019 | **0.910 ± 0.011** | 0.843 ± 0.023 | 0.380 ± 0.364 | _0.206 ± 0.004_ |

**negrito** = melhor da linha · _itálico_ = pior da linha · — = não existe esse braço nesse percentual


## Tabelão 2 — Cohen κ por split (75% dos dados)

| Dataset | split | SVM + FLIM cru | MLP Felipe (frozen) | MLP Felipe (unfrozen) | Head Sigmoid | Head ReLU | Head Softplus |
|:--|--:|--:|--:|--:|--:|--:|--:|
| eggs | 1 | 0.888 | 0.658 | **0.934** | 0.798 | _0.526_ | 0.790 |
| eggs | 2 | 0.869 | 0.687 | **0.913** | _0.028_ | 0.498 | 0.172 |
| eggs | 3 | 0.852 | 0.609 | **0.910** | 0.530 | 0.656 | _0.000_ |
| larvae | 1 | 0.835 | 0.857 | **0.934** | 0.918 | _0.000_ | 0.916 |
| larvae | 2 | 0.823 | 0.779 | 0.943 | **0.944** | _0.000_ | 0.932 |
| larvae | 3 | 0.849 | _0.842_ | **0.972** | 0.924 | 0.932 | 0.920 |
| protozoan | 1 | 0.855 | 0.572 | **0.898** | 0.828 | _0.000_ | 0.205 |
| protozoan | 2 | 0.818 | 0.594 | **0.915** | 0.869 | 0.415 | _0.203_ |
| protozoan | 3 | 0.839 | 0.610 | **0.918** | 0.832 | 0.725 | _0.210_ |


## Tabelão 3 — Ranking dos braços (κ médio nos 3 datasets)

| # | Braço | κ @5% | κ @75% | κ médio | pior caso | melhor caso |
|--:|:--|--:|--:|--:|--:|--:|
| 1 | MLP Felipe (unfrozen) | 0.661 | 0.926 | **0.793** | 0.450 | 0.972 |
| 2 | Head Sigmoid | — | 0.741 | **0.741** | 0.028 | 0.944 |
| 3 | SVM + FLIM cru | 0.593 | 0.847 | **0.720** | 0.454 | 0.888 |
| 4 | MLP Felipe (frozen) | 0.314 | 0.690 | **0.502** | 0.060 | 0.857 |
| 5 | Head Softplus | 0.073 | 0.483 | **0.278** | -0.000 | 0.932 |
| 6 | Head ReLU | 0.072 | 0.417 | **0.244** | -0.001 | 0.932 |

> A Cabeça Sigmoid não tem grade em 5%, então sua média é só de 75% — onde todos vão bem.
> Comparando **apenas a 75%**: MLP unfrozen 0.926 > SVM cru 0.847 > Sigmoid 0.741 > Softplus 0.483 > ReLU 0.417.


## Tabelão 4 — Os 10 piores e os 10 melhores runs individuais (κ, pct 5 e 75)

**Piores:**

| κ | Braço | Dataset | split | pct |
|--:|:--|:--|--:|--:|
| -0.001 | Head ReLU | eggs | 1 | 5 |
| -0.000 | Head Softplus | eggs | 3 | 5 |
| 0.000 | Head Softplus | eggs | 3 | 75 |
| 0.000 | Head Softplus | eggs | 2 | 5 |
| 0.000 | Head ReLU | protozoan | 1 | 75 |
| 0.000 | Head ReLU | protozoan | 3 | 5 |
| 0.000 | Head ReLU | protozoan | 1 | 5 |
| 0.000 | Head Softplus | larvae | 1 | 5 |
| 0.000 | Head ReLU | eggs | 3 | 5 |
| 0.000 | Head ReLU | larvae | 2 | 5 |

**Melhores:**

| κ | Braço | Dataset | split | pct |
|--:|:--|:--|--:|--:|
| 0.972 | MLP Felipe (unfrozen) | larvae | 3 | 75 |
| 0.944 | Head Sigmoid | larvae | 2 | 75 |
| 0.943 | MLP Felipe (unfrozen) | larvae | 2 | 75 |
| 0.934 | MLP Felipe (unfrozen) | larvae | 1 | 75 |
| 0.934 | MLP Felipe (unfrozen) | eggs | 1 | 75 |
| 0.932 | Head Softplus | larvae | 2 | 75 |
| 0.932 | Head ReLU | larvae | 3 | 75 |
| 0.924 | Head Sigmoid | larvae | 3 | 75 |
| 0.920 | Head Softplus | larvae | 3 | 75 |
| 0.918 | MLP Felipe (unfrozen) | protozoan | 3 | 75 |
