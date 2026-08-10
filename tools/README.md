# `tools/` — experimentos versionados, auto-testes e análises

Duas classes de arquivo vivem aqui, e a diferença é o `.gitignore`:

| padrão | status | regra |
|---|---|---|
| `tools/diag_*.py` | **descartável, ignorado pelo git** | escrito para responder uma pergunta e morrer. [.gitignore:89](../.gitignore#L89) |
| todo o resto | **versionado e re-executável** | tem docstring de contrato, `--dry-run` ou `assert`, e escreve em caminho fixo |

> **Aviso comprado com sangue:** os `diag_*.py` não passam pelo histórico do git — quando somem, somem. Foi o que aconteceu com `diag_flim_gap_vs_flatten.py` e `diag_g2_flatten_curve.py`, que geraram a curva laranja do gráfico de protocolo. Sobraram só os JSONs de saída. Se um `diag_*` produziu resultado que você vai citar, **promova para `eval_*.py` antes de fechar o terminal.**

Todos rodam no mesmo ambiente:

```bash
source /dados/home/moliveira/miniforge3/etc/profile.d/conda.sh && conda activate scalable_FLIM
```

---

## Índice

| script | o que faz | saída |
|---|---|---|
| [`plot_comparacao_flim_protocolo.py`](plot_comparacao_flim_protocolo.py) | junta os CSVs dos dois `eval_*` + o CSV externo em 3 painéis de κ | `artifacts/plots/comparacao_flim_protocolo_original/comparacao_kappa_registrado_vs_reproducao.png` |
| [`check_probe_matches_evaluator.py`](check_probe_matches_evaluator.py) | auto-teste: o probe do treino e o avaliador oficial são o mesmo objeto | stdout (`assert`) |
| [`check_refactor_equivalence.py`](check_refactor_equivalence.py) | auto-teste: `fit_svm` / `compute_metrics` deduplicados não mudaram de comportamento | stdout (`assert`) |
| [`analyze_sigmoid_saturation.py`](analyze_sigmoid_saturation.py) | métricas de achatamento ReLU → Sigmoid, por checkpoint | `results/relu_vs_sigmoid_flatten.csv` |
| [`plot_sigmoid_saturation.py`](plot_sigmoid_saturation.py) | os 6 PNGs do achatamento | `results/plots_flatten/*.png` |
| [`analyze_unit_activations.py`](analyze_unit_activations.py) | quem acende: análise por neurônio das 24 unidades ocultas | `results/unit_activation_stats.csv` + `results/plots_units/U_*.png` |
| [`analyze_firing_relu_vs_sigmoid.py`](analyze_firing_relu_vs_sigmoid.py) | sigmoid ativa **menos** ou ativa **errado**? contrafactual ReLU sobre o mesmo `z` | `results/firing_relu_vs_sigmoid.csv` + `results/plots_units/F_*.png` |

---

# Parte I — O gráfico das curvas de κ

**Os dois `eval_*` que alimentam este gráfico moram em [`src/evaluate/`](../src/evaluate/), não aqui** — `eval_avg_pooling_48d.py` (GAP 48-d) e `eval_svm_flim_flatten.py` (conv3 achatado 27.648-d). Como rodar cada um: [`src/evaluate/README.md`](../src/evaluate/README.md). O diagrama dos três pipelines está em [`artifacts/plots/comparacao_flim_protocolo_original/pipelines.md`](../artifacts/plots/comparacao_flim_protocolo_original/pipelines.md).

Aqui fica só o plotter.

## `plot_comparacao_flim_protocolo.py` — o gráfico

```bash
conda run -n scalable_FLIM python tools/plot_comparacao_flim_protocolo.py
```

Sem flags. Lê e escreve **apenas** dentro de `artifacts/plots/comparacao_flim_protocolo_original/`; não toca em `src/`, `scripts/`, `configs/` nem no CSV registrado. Exige `comparacao_unified_svm_com_reproducao.csv` (gerado por `build_comparacao_csv.py`, que mora ao lado dos dados). A série verde é opcional: se o CSV 48-d não existir, o gráfico sai com duas curvas.

⚠️ **Nota de leitura das colunas.** `acc` não significa a mesma coisa nos três CSVs: no externo é acurácia crua, no laranja é `raw_acc`, e no verde é `multiclass_accuracy(average="macro")` — balanceada. Só **κ** é diretamente comparável entre as três séries; para acurácia use `acc_raw`.

---

# Parte II — Auto-testes

Nenhum usa pytest. São scripts de `assert`: ou imprimem OK, ou morrem.

```bash
python tools/check_probe_matches_evaluator.py
python tools/check_refactor_equivalence.py
```

**`check_probe_matches_evaluator.py`** — garante que `_encode_pooled(model.encoder, x) == model.embed(x)` elemento a elemento e que um mesmo `SVC` alimentado pelos dois devolve κ idêntico. Também fixa as formas: `avgpool2d → [B, 48]`, `flatten → [B, 27648]`. Rode depois de mexer no probe de `validation_step` / `_embeddings`.

**`check_refactor_equivalence.py`** — rode depois de tocar em `src/utils/evaluate.py:fit_svm`, `src/metrics/classification.py:compute_metrics` ou no probe SVM de `src/modules/autoencoder_flim_module.py`. Cada checagem confronta a implementação compartilhada contra a **antiga**, escrita à mão inline, para que uma mudança silenciosa de comportamento falhe alto em vez de ser absorvida pelo helper que ela deveria testar.

---

# Parte III — Análise: os embeddings ficam presos na saturação da sigmoid

```bash
python tools/analyze_sigmoid_saturation.py   # -> results/relu_vs_sigmoid_flatten.csv
python tools/plot_sigmoid_saturation.py      # -> results/plots_flatten/*.png
python tools/analyze_unit_activations.py     # -> results/unit_activation_stats.csv + plots_units/U_*.png
python tools/analyze_firing_relu_vs_sigmoid.py  # -> results/firing_relu_vs_sigmoid.csv + plots_units/F_*.png
```

`analyze_sigmoid_saturation.py` aceita `--pattern` (default `sigmoid2l_classhead_*`, varrido em `artifacts/classification_flim`), `--out` e `--image-size`. `plot_sigmoid_saturation.py` aceita `--csv`, `--test-csv` e `--out-dir`. Os dois `analyze_*` restantes não têm flags — os checkpoints representativos estão fixos no código (split1, pct75, frozen e unfrozen).

## 1. Conclusão (direto ao ponto)

Os embeddings do braço **FLIM + Sigmoid** ficam **presos na saturação da sigmoid**: entre **~61% (frozen)** e **~69% (unfrozen)** das ativações da camada oculta caem nas **caudas** da sigmoid (valor `< 0.01` ou `> 0.99`), justamente onde o gradiente é **≈ 0**.

Isso **achata a magnitude** do embedding — o "quão longe" do hiperplano de decisão simplesmente some — **mas NÃO destrói a informação de classe**: um linear-probe treinado sobre o vetor `S` ainda acerta **~0.8**.

Ou seja, o colapso do braço sigmoid (ex.: *eggs frozen*, acurácia REAL do modelo **~0.13** vs. probe sobre o **MESMO** vetor `S` **~0.75**) é uma falha de **treinabilidade** (o gradiente é estrangulado nas caudas), **não** de informação. A informação está lá; o modelo é que não consegue ler.

## 2. O que são P e S

| | Definição | Última não-linearidade | Efeito |
|---|---|---|---|
| **P** | vetor após o `avgpool` do encoder (**48-d**) | **ReLU** da `conv3` (P ≥ 0) | "após ReLU" → **PRESERVA magnitude** |
| **S** | `Sigmoid(layer1(P))` (**24-d**) | **Sigmoid** (S ∈ [0,1]) | "após Sigmoid" → **ACHATADO** |

**Por que usamos métricas invariantes à escala?** Porque `P ∈ [0, ∞)` e `S ∈ [0, 1]` vivem em escalas diferentes. Comparar a *variância crua* dos dois mediria só a diferença de escala, não a mudança de estrutura. Por isso trabalhamos com **coeficiente de variação (CV)**, **correlação de Spearman**, **fração saturada** e **linear-probe** — todos independentes da escala absoluta.

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

## 6. As duas análises por neurônio

**`analyze_unit_activations.py` — quem acende.** O plot E mostra que os valores de `S` se acumulam em 0 e 1, mas não diz **quais** unidades acendem nem **se** o padrão liga/desliga codifica a classe. Este script roda o test set num checkpoint representativo (split1, pct75, frozen e unfrozen) e classifica cada uma das 24 unidades em `STUCK_ON` / `STUCK_OFF` / `SWITCHING` / `RESPONSIVE`, com F-ANOVA da ativação entre classes e a matriz `M[24 × C]` de ativação média por classe.

**`analyze_firing_relu_vs_sigmoid.py` — ativa menos ou ativa errado.** Como os checkpoints da cabeça ReLU estão vazios, a comparação é **contrafactual sobre o mesmo sinal real**: extrai `z = layer1(P)` dos pesos treinados e avalia `S = Sigmoid(z)` contra `R = ReLU(z)`. O achado que fecha a questão está em `firing_agreement`: ReLU dispara com `z > 0` e Sigmoid passa de 0.5 com `z > 0`, então o **conjunto** de unidades que acende é o mesmo — o que difere é a **força graduada** (ReLU ilimitada, Sigmoid esmagada em (0,1)). Não é "ativa errado"; é magnitude.

## Detalhe honesto

O caso **unfrozen** satura **até mais** (~0.69) e **mesmo assim classifica melhor**. Isso mostra que a **saturação no estado final é sintoma, não sentença**: o que decidiu o resultado foi **ter havido gradiente DURANTE o treino** (o unfrozen pôde ajustar o encoder e escapar dos platôs enquanto aprendia), não o quão saturado o embedding acabou.

Nota de completude: faltam **2 checkpoints** (*larvae* split1/split2, pct1, unfrozen — arquivos vazios), então o CSV tem **34 linhas** em vez de 36.
