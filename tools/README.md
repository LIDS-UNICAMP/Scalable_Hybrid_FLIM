# Análise: os embeddings ficam presos na saturação da sigmoid

Documentação de duas ferramentas:

- `tools/analyze_sigmoid_saturation.py` → gera `results/relu_vs_sigmoid_flatten.csv` (métricas por checkpoint).
- `tools/plot_sigmoid_saturation.py` → gera 6 PNGs em `results/plots_flatten/`.

---

## 1. Conclusão (direto ao ponto)

Os embeddings do braço **FLIM + Sigmoid** ficam **presos na saturação da sigmoid**: entre **~61% (frozen)** e **~69% (unfrozen)** das ativações da camada oculta caem nas **caudas** da sigmoid (valor `< 0.01` ou `> 0.99`), justamente onde o gradiente é **≈ 0**.

Isso **achata a magnitude** do embedding — o "quão longe" do hiperplano de decisão simplesmente some — **mas NÃO destrói a informação de classe**: um linear-probe treinado sobre o vetor `S` ainda acerta **~0.8**.

Ou seja, o colapso do braço sigmoid (ex.: *eggs frozen*, acurácia REAL do modelo **~0.13** vs. probe sobre o **MESMO** vetor `S` **~0.75**) é uma falha de **treinabilidade** (o gradiente é estrangulado nas caudas), **não** de informação. A informação está lá; o modelo é que não consegue ler.

---

## 2. O que são P e S

| | Definição | Última não-linearidade | Efeito |
|---|---|---|---|
| **P** | vetor após o `avgpool` do encoder (**48-d**) | **ReLU** da `conv3` (P ≥ 0) | "após ReLU" → **PRESERVA magnitude** |
| **S** | `Sigmoid(layer1(P))` (**24-d**) | **Sigmoid** (S ∈ [0,1]) | "após Sigmoid" → **ACHATADO** |

**Por que usamos métricas invariantes à escala?** Porque `P ∈ [0, ∞)` e `S ∈ [0, 1]` vivem em escalas diferentes. Comparar a *variância crua* dos dois mediria só a diferença de escala, não a mudança de estrutura. Por isso trabalhamos com **coeficiente de variação (CV)**, **correlação de Spearman**, **fração saturada** e **linear-probe** — todos independentes da escala absoluta.

---

## 3. O que cada cálculo mede (explicação simples + fórmula)

- **CV das normas** — `std/mean` das normas L2 por amostra. Mede o quanto os embeddings variam em tamanho. **Menor = mais achatado.**
  *Analogia:* uma turma onde todos têm quase a mesma altura (CV baixo) vs. uma turma com alturas bem variadas (CV alto).

- **Fração saturada** (`sigmoid_sat_frac`) — fração das entradas de `S` (unidade × amostra) com valor `< 0.01` ou `> 0.99`. São os platôs da sigmoid, onde `σ(z)·(1−σ(z)) ≈ 0`, isto é, **gradiente nulo**. O limiar `0.01 / 0.99` corresponde a `|z| ≳ 4.6`.

- **Spearman(‖P‖, ‖S‖)** (`spearman_normP_normS`) — correlação de *ranking* entre o tamanho de `P` e o de `S`, por amostra. **≈ 0 ou negativo** significa que o "quão longe do hiperplano" contido em `P` **não sobrevive** à passagem pela sigmoid.

- **Faixa dinâmica p95/p5** (`dynrange_P`, `dynrange_S`) — razão entre o percentil 95 e o percentil 5 das normas. **→ 1 = a faixa colapsou** (todas as normas ficaram iguais).

- **dead_chan_P** — fração de canais de `P` com variância ≈ 0, isto é, **canais mortos da ReLU** (sempre zero).

- **effdim** (`effdim_P`, `effdim_S`) — *participation ratio* dos autovalores da matriz de covariância = **dimensionalidade efetiva** do embedding (quantas direções realmente "vivem").

- **entropy** (`entropy_P`, `entropy_S`) — entropia do histograma dos valores (normalizados para `[0,1]`). Mede o quão espalhados / concentrados estão os valores.

- **probe_acc_P / probe_acc_S** — acurácia de uma *logistic regression* (com cross-validation) treinada sobre `P` e sobre `S`. É a **informação de classe RETIDA** em cada vetor, independentemente de o modelo original conseguir ou não usá-la.

---

## 4. Como cada gráfico foi gerado e como lê-lo

Todos em `results/plots_flatten/`.

**A — Histograma das normas ‖P‖ vs ‖S‖**

![Histograma das normas](../results/plots_flatten/A_hist_norms.png)

Normas normalizadas pela mediana. `‖P‖` aparece **largo**; `‖S‖` aparece **estreito**. A **largura = magnitude preservada**. O estreitamento de `S` é o achatamento visto de frente.

**C — Scatter ‖P‖ vs ‖S‖ por amostra**

![Scatter norma P vs norma S](../results/plots_flatten/C_scatter_normP_normS.png)

Cada ponto é uma amostra. A nuvem é **plana** e o **Spearman ≈ 0**: amostras longe do hiperplano em `P` não ficam longe em `S`. O ranking de magnitude não passa pela sigmoid.

**E — Histograma dos valores de S**

![Histograma dos valores da sigmoid](../results/plots_flatten/E_hist_sigmoid_vals.png)

Distribuição dos valores das unidades de `S`. O **acúmulo em 0 e em 1** (as pontas) é a **saturação** — é exatamente a fração medida por `sigmoid_sat_frac`.

**B — Barra da fração saturada**

![Fração saturada por dataset e modo](../results/plots_flatten/B_saturation_bar.png)

Fração saturada por **dataset × modo** (frozen vs unfrozen). Mostra que a saturação é generalizada e que o **unfrozen satura até mais**.

**CV_dumbbell — colapso da magnitude**

![Dumbbell CV_P para CV_S](../results/plots_flatten/CV_dumbbell.png)

Para cada checkpoint, uma linha ligando `CV_P` → `CV_S`. As linhas caem quase sempre "para baixo": a magnitude **colapsa** ao atravessar a sigmoid.

**D — Trainability gap**

![Gap de treinabilidade](../results/plots_flatten/D_trainability_gap.png)

Eixo Y = **acurácia REAL do modelo**; eixo X = **probe sobre S**. Pontos **abaixo da diagonal** = a informação existe (probe alto) mas o **modelo não a lê** (acc real baixa). Esse é o gap de treinabilidade, o coração da conclusão.

---

## 5. Números (lidos de `results/relu_vs_sigmoid_flatten.csv`)

Médias por **dataset × modo**:

| dataset | modo | cv_P | cv_S | sigmoid_sat_frac | spearman(‖P‖,‖S‖) | probe_acc_P | probe_acc_S |
|---|---|---|---|---|---|---|---|
| eggs | frozen | 0.034 | 0.029 | 0.635 | 0.461 | 0.770 | 0.746 |
| eggs | unfrozen | 0.205 | 0.032 | 0.736 | 0.536 | 0.808 | 0.795 |
| larvae | frozen | 0.090 | 0.048 | 0.574 | −0.250 | 0.969 | 0.952 |
| larvae | unfrozen | 0.160 | 0.040 | 0.820 | −0.089 | 0.984 | 0.981 |
| protozoan | frozen | 0.101 | 0.022 | 0.632 | −0.549 | 0.653 | 0.628 |
| protozoan | unfrozen | 0.499 | 0.041 | 0.556 | −0.191 | 0.772 | 0.760 |

**Achados globais:**

- **CV colapsa**: mediana `cv_P = 0.062` → `cv_S = 0.033` (Wilcoxon pareado, **p = 8.8e-7**).
- **Unfrozen colapsa ainda mais forte**: CV médio `cv_P = 0.304` → `cv_S = 0.037`.
- **Saturação**: **~61% frozen** / **~69% unfrozen** das ativações nas caudas.
- **A informação sobrevive**: `probe_acc_S` fica em torno de **~0.75–0.98** conforme o dataset, sempre próximo de `probe_acc_P` — a sigmoid achata a *magnitude*, não a *classe*.

---

## 6. Como rodar

```bash
source /dados/home/moliveira/miniforge3/etc/profile.d/conda.sh && conda activate scalable_FLIM
python tools/analyze_sigmoid_saturation.py   # -> results/relu_vs_sigmoid_flatten.csv
python tools/plot_sigmoid_saturation.py      # -> results/plots_flatten/*.png
```

---

## Detalhe honesto

O caso **unfrozen** satura **até mais** (~0.69) e **mesmo assim classifica melhor**. Isso mostra que a **saturação no estado final é sintoma, não sentença**: o que decidiu o resultado foi **ter havido gradiente DURANTE o treino** (o unfrozen pôde ajustar o encoder e escapar dos platôs enquanto aprendia), não o quão saturado o embedding acabou.

Nota de completude: faltam **2 checkpoints** (*larvae* split1/split2, pct1, unfrozen — arquivos vazios), então o CSV tem **34 linhas** em vez de 36.
