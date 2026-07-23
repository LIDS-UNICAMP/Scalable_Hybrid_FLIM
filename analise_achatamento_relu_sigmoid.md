# Análise de Achatamento ReLU → Sigmoid

Análise estatística, sobre pesos **já congelados** (sem experimento sintético), de quanto a
ativação **Sigmoid** da cabeça de classificação achata a magnitude do vetor que sai da última
**ReLU** do encoder. Fonte: `results/relu_vs_sigmoid_flatten.csv` (34 checkpoints) e
`results/sigmoid2l_test_results.csv` (acurácia real de teste).

---

## 1. Contexto

Para cada checkpoint passa-se o conjunto de teste pelo caminho `encoder → avgpool` e extraem-se
**dois vetores reais por amostra**:

- **P** — vetor logo após o `avgpool` (48-d). A última não-linearidade antes da cabeça é a **ReLU**
  da `conv3`, então `P ≥ 0` e **preserva magnitude**. É o "vetor após ReLU". Seu domínio é `[0, ∞)`.
- **S** — `Sigmoid(layer1(P))` (24-d), a ativação da camada oculta, **achatada em `[0, 1]`**.
  É o "vetor após Sigmoid".

O objetivo é medir **quanto S achata a magnitude de P**. Como `P ∈ [0, ∞)` e `S ∈ [0, 1]` vivem em
escalas diferentes, comparar variância crua seria injusto. Por isso todas as métricas centrais são
**invariantes a escala** (coeficiente de variação, faixa dinâmica p95/p5, correlação de Spearman,
dimensionalidade efetiva, entropia normalizada, acurácia de linear-probe).

---

## 2. Dicionário de colunas (`relu_vs_sigmoid_flatten.csv`)

| Coluna | Significado |
|---|---|
| `run` | Nome do checkpoint/experimento. |
| `dataset` | Conjunto: `eggs`, `larvae` ou `protozoan`. |
| `split` | Índice da divisão de dados (1–3). |
| `pct` | Percentual de rótulos usado no treino (1 ou 75). |
| `mode` | `frozen` (encoder congelado) ou `unfrozen` (encoder ajustado no fine-tuning). |
| `imagenet_norm` | Se a normalização ImageNet foi aplicada (aqui sempre `True`). |
| `N` | Número de amostras de teste avaliadas. |
| `cv_P` | Coeficiente de variação (`std/mean`) das **normas L2 por amostra** de P. **Menor = mais achatado.** |
| `cv_S` | Idem para S. **Menor = mais achatado.** |
| `cv_ratio_S_over_P` | `cv_S / cv_P`: quanto do spread **sobra** depois da sigmoid. **`<1` = achatou.** |
| `dynrange_P` | Faixa dinâmica = razão `p95/p5` das normas de P. **`→1` = colapsou.** |
| `dynrange_S` | Idem para S. **`→1` = colapsou.** |
| `spearman_normP_normS` | Correlação de Spearman entre `‖P‖` e `‖S‖` por amostra. **`≈0` ou negativa = o "quão longe" NÃO sobrevive** à sigmoid. |
| `sigmoid_sat_frac` | Fração das unidades sigmoides **saturadas** (`valor < 0.01` ou `> 0.99`). |
| `dead_chan_P` | Fração de canais de P com variância `~0` (**canais mortos** da ReLU). |
| `effdim_P` | Dimensionalidade efetiva de P (**participation ratio** dos autovalores da covariância). |
| `effdim_S` | Idem para S. |
| `entropy_P` | Entropia do histograma dos valores de P (normalizados para `[0,1]`). |
| `entropy_S` | Idem para S. |
| `probe_acc_P` | Acurácia de um **linear-probe** (regressão logística, validação cruzada) treinado sobre P. **Mede a informação de classe retida no vetor.** |
| `probe_acc_S` | Idem sobre S. |

---

## 3. Agregado por dataset × modo (média das colunas principais)

| dataset | mode | cv_P | cv_S | dynrange_P | dynrange_S | spearman | sat_frac | probe_P | probe_S |
|---|---|---|---|---|---|---|---|---|---|
| eggs | frozen | 0.034 | 0.029 | 1.106 | 1.099 | 0.461 | 0.635 | 0.770 | 0.746 |
| eggs | unfrozen | 0.205 | 0.032 | 1.904 | 1.097 | 0.536 | 0.736 | 0.808 | 0.795 |
| larvae | frozen | 0.090 | 0.048 | 1.321 | 1.160 | -0.250 | 0.574 | 0.969 | 0.952 |
| larvae | unfrozen | 0.160 | 0.040 | 1.613 | 1.119 | -0.089 | 0.820 | 0.984 | 0.981 |
| protozoan | frozen | 0.101 | 0.022 | 1.349 | 1.077 | -0.549 | 0.632 | 0.653 | 0.628 |
| protozoan | unfrozen | 0.499 | 0.041 | 2.945 | 1.142 | -0.191 | 0.556 | 0.772 | 0.760 |

Leitura rápida: em **todos** os grupos `cv_S < cv_P` e `dynrange_S < dynrange_P` — a sigmoid sempre
comprime a magnitude. O efeito é mais violento no `unfrozen` (onde P tem muito mais spread — ex.
protozoan `cv_P=0.499 → cv_S=0.041`). O Spearman fica em torno de 0 ou negativo, indicando que a
ordenação por "quão longe" de P **não sobrevive** em S. Mesmo assim, `probe_S` fica apenas ~0.01–0.02
abaixo de `probe_P`.

---

## 4. Tabela completa por checkpoint (34 linhas)

| dataset | split | pct | mode | cv_P | cv_S | cv_ratio | dynr_P | dynr_S | spearman | sat_frac | probe_P | probe_S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eggs | 1 | 1 | frozen | 0.0278 | 0.0387 | 1.393 | 1.079 | 1.137 | 0.355 | 0.404 | 0.7896 | 0.7485 |
| eggs | 1 | 75 | frozen | 0.0278 | 0.0365 | 1.312 | 1.079 | 1.129 | 0.518 | 0.638 | 0.7896 | 0.7474 |
| eggs | 2 | 1 | frozen | 0.0446 | 0.0332 | 0.745 | 1.144 | 1.116 | 0.692 | 0.646 | 0.7720 | 0.7306 |
| eggs | 2 | 75 | frozen | 0.0446 | 0.0224 | 0.503 | 1.144 | 1.057 | -0.039 | 0.599 | 0.7720 | 0.7364 |
| eggs | 3 | 1 | frozen | 0.0311 | 0.0203 | 0.652 | 1.094 | 1.070 | 0.595 | 0.746 | 0.7470 | 0.7548 |
| eggs | 3 | 75 | frozen | 0.0311 | 0.0241 | 0.777 | 1.094 | 1.085 | 0.644 | 0.775 | 0.7470 | 0.7591 |
| larvae | 1 | 1 | frozen | 0.0206 | 0.0367 | 1.776 | 1.064 | 1.119 | 0.427 | 0.501 | 0.9676 | 0.9516 |
| larvae | 1 | 75 | frozen | 0.0206 | 0.0399 | 1.933 | 1.064 | 1.153 | 0.661 | 0.485 | 0.9676 | 0.9562 |
| larvae | 2 | 1 | frozen | 0.1734 | 0.0370 | 0.213 | 1.656 | 1.119 | -0.982 | 0.505 | 0.9653 | 0.9397 |
| larvae | 2 | 75 | frozen | 0.1734 | 0.1138 | 0.656 | 1.656 | 1.354 | -0.985 | 0.653 | 0.9653 | 0.9431 |
| larvae | 3 | 1 | frozen | 0.0747 | 0.0116 | 0.155 | 1.243 | 1.038 | 0.182 | 0.578 | 0.9727 | 0.9710 |
| larvae | 3 | 75 | frozen | 0.0747 | 0.0508 | 0.680 | 1.243 | 1.178 | -0.800 | 0.724 | 0.9727 | 0.9528 |
| protozoan | 1 | 1 | frozen | 0.2097 | 0.0072 | 0.034 | 1.786 | 1.019 | -0.554 | 0.267 | 0.6617 | 0.6060 |
| protozoan | 1 | 75 | frozen | 0.2097 | 0.0245 | 0.117 | 1.786 | 1.084 | -0.704 | 0.688 | 0.6617 | 0.6055 |
| protozoan | 2 | 1 | frozen | 0.0303 | 0.0115 | 0.381 | 1.077 | 1.035 | -0.107 | 0.611 | 0.6358 | 0.6617 |
| protozoan | 2 | 75 | frozen | 0.0303 | 0.0443 | 1.459 | 1.077 | 1.162 | -0.400 | 0.898 | 0.6358 | 0.6400 |
| protozoan | 3 | 1 | frozen | 0.0617 | 0.0121 | 0.196 | 1.185 | 1.034 | -0.933 | 0.486 | 0.6605 | 0.6335 |
| protozoan | 3 | 75 | frozen | 0.0617 | 0.0336 | 0.544 | 1.185 | 1.128 | -0.597 | 0.845 | 0.6605 | 0.6225 |
| eggs | 1 | 1 | unfrozen | 0.0256 | 0.0210 | 0.822 | 1.074 | 1.075 | 0.341 | 0.712 | 0.7849 | 0.7376 |
| eggs | 1 | 75 | unfrozen | 0.4537 | 0.0472 | 0.104 | 3.374 | 1.155 | 0.003 | 0.782 | 0.9163 | 0.9253 |
| eggs | 2 | 1 | unfrozen | 0.0440 | 0.0262 | 0.596 | 1.141 | 1.046 | 0.954 | 0.704 | 0.7752 | 0.7517 |
| eggs | 2 | 75 | unfrozen | 0.0499 | 0.0321 | 0.644 | 1.165 | 1.099 | 0.872 | 0.588 | 0.7713 | 0.7376 |
| eggs | 3 | 1 | unfrozen | 0.0310 | 0.0208 | 0.672 | 1.094 | 1.073 | 0.629 | 0.775 | 0.7474 | 0.7576 |
| eggs | 3 | 75 | unfrozen | 0.6246 | 0.0420 | 0.067 | 3.578 | 1.136 | 0.418 | 0.854 | 0.8518 | 0.8576 |
| larvae | 1 | 75 | unfrozen | 0.0873 | 0.0436 | 0.499 | 1.286 | 1.133 | -0.408 | 0.843 | 0.9835 | 0.9818 |
| larvae | 2 | 75 | unfrozen | 0.3028 | 0.0650 | 0.214 | 2.312 | 1.205 | 0.653 | 0.935 | 0.9909 | 0.9898 |
| larvae | 3 | 1 | unfrozen | 0.0844 | 0.0126 | 0.149 | 1.278 | 1.041 | -0.477 | 0.553 | 0.9733 | 0.9693 |
| larvae | 3 | 75 | unfrozen | 0.1660 | 0.0395 | 0.238 | 1.576 | 1.096 | -0.123 | 0.949 | 0.9863 | 0.9841 |
| protozoan | 1 | 1 | unfrozen | 0.2158 | 0.0050 | 0.023 | 1.816 | 1.012 | -0.682 | 0.258 | 0.6703 | 0.6007 |
| protozoan | 1 | 75 | unfrozen | 0.9576 | 0.0708 | 0.074 | 4.587 | 1.242 | 0.093 | 0.666 | 0.8692 | 0.8742 |
| protozoan | 2 | 1 | unfrozen | 0.0319 | 0.0086 | 0.270 | 1.082 | 1.031 | 0.331 | 0.642 | 0.6436 | 0.6584 |
| protozoan | 2 | 75 | unfrozen | 0.4988 | 0.0728 | 0.146 | 3.265 | 1.253 | 0.161 | 0.634 | 0.8983 | 0.9047 |
| protozoan | 3 | 1 | unfrozen | 0.0617 | 0.0118 | 0.190 | 1.185 | 1.033 | -0.932 | 0.488 | 0.6621 | 0.6333 |
| protozoan | 3 | 75 | unfrozen | 1.2256 | 0.0770 | 0.063 | 5.733 | 1.280 | -0.118 | 0.646 | 0.8866 | 0.8884 |

> Observação: faltam `larvae split1/split2 pct1 unfrozen` porque os checkpoints correspondentes
> estavam vazios (daí 34 linhas em vez das 36 esperadas).

---

## 5. Achados globais

**Teste de Wilcoxon pareado (P vs S, por checkpoint):**

| Métrica | Mediana P | Mediana S | p-valor |
|---|---|---|---|
| CV (spread relativo da norma) | 0.062 | 0.033 | 8.8e-7 |
| probe_acc (informação de classe) | 0.787 | 0.756 | 3.6e-3 |

- A queda de CV é **grande e estatisticamente robusta** (`p ≈ 9e-7`): a sigmoid comprime a magnitude
  de forma sistemática.
- A queda de `probe_acc` é **estatisticamente significativa mas minúscula** (mediana 0.787 → 0.756,
  ~0.03): quase toda a informação de classe **permanece** em S.

**Médias globais:**

- **Unfrozen:** CV cai de `0.304 → 0.037` (**~8× menor**); faixa dinâmica cai de `2.22 → 1.12`.
- **Saturação:** ~61% das unidades saturadas no `frozen`, ~69% no `unfrozen`.
- **Spearman(‖P‖, ‖S‖) ≈ 0** globalmente: a ordenação por magnitude de P **não sobrevive** em S.

---

## 6. Interpretação dos plots (`results/plots_flatten/`)

### A — Distribuição das normas ‖P‖ vs ‖S‖
![](results/plots_flatten/A_hist_norms.png)

Dois histogramas sobrepostos das normas L2 por amostra. A distribuição de **‖P‖ é larga** (a
magnitude/energia varia bastante entre amostras — sinal preservado), enquanto **‖S‖ é estreita e
concentrada**. Essa estreiteza é a assinatura visual do **achatamento**: a sigmoid empurra tudo para
uma faixa apertada de magnitude.

### C — Dispersão ‖P‖ vs ‖S‖ por amostra
![](results/plots_flatten/C_scatter_normP_normS.png)

Cada ponto é uma amostra de teste. Se a magnitude de P sobrevivesse, veríamos uma reta crescente.
Em vez disso a **nuvem é achatada/horizontal** e o **Spearman ≈ 0** (frequentemente negativo): saber
"quão longe" uma amostra está em P **não permite** recuperar sua norma em S. O eixo de magnitude é
descartado pela sigmoid.

### E — Histograma dos valores das unidades sigmoides
![](results/plots_flatten/E_hist_sigmoid_vals.png)

Distribuição de todos os valores individuais das unidades de S. A massa **acumula-se nos extremos 0 e
1**, com pouca densidade no meio — o clássico regime de **saturação** da sigmoid (61% no frozen, 69%
no unfrozen). Unidades saturadas têm gradiente local `≈0`.

### B — Fração saturada por dataset × modo
![](results/plots_flatten/B_saturation_bar.png)

Barras da `sigmoid_sat_frac` agregada. Mostra que a saturação é **generalizada** em todos os datasets
e, notavelmente, tende a ser **maior no `unfrozen`** — o modo que, apesar disso, atinge a melhor
acurácia real (ver §7).

### CV — Dumbbell CV_P → CV_S
![](results/plots_flatten/CV_dumbbell.png)

Para cada checkpoint, uma linha ligando `cv_P` (esquerda) a `cv_S` (direita). As linhas **despencam**
quase sem exceção, com quedas dramáticas nos checkpoints `unfrozen pct75` (ex. protozoan
`1.226 → 0.077`). É a visão por-checkpoint do **colapso da magnitude**.

### D — Gap de treinabilidade
![](results/plots_flatten/D_trainability_gap.png)

Eixo x = `probe_acc_S` (acurácia de um linear-probe **treinável do zero** sobre S); eixo y =
**acurácia real** do modelo treinado (`test_accuracy`). Pontos **abaixo da diagonal `y = x`** são
checkpoints onde a informação de classe **existe** em S (probe alto) mas a `layer2` do modelo **não
consegue lê-la** (acurácia real baixa). É o diagnóstico central: o problema é de **treinabilidade**,
não de informação ausente.

---

## 7. Conclusão-chave

A sigmoid **achata a magnitude de forma dramática e verificável**: CV ~8× menor (`p ≈ 9e-7`), faixa
dinâmica `→1`, e Spearman(‖P‖,‖S‖) ≈ 0 — o eixo de "quão longe" praticamente **não sobrevive**.

**Porém, a informação de CLASSE sobrevive em S.** O linear-probe sobre S atinge ~0.75–0.98 conforme o
dataset, apenas **~0.01–0.02 abaixo** do probe sobre P.

Logo, o colapso do braço sigmoid é uma **falha de TREINABILIDADE, não de informação**. O caso mais
gritante: em `eggs frozen`, a acurácia **real** do modelo é **~0.125** (mean, 6 checkpoints), enquanto
um probe linear treinado **sobre o mesmíssimo S** chega a **~0.75**. A informação está lá; a `layer2`
do modelo nunca aprendeu a lê-la porque a **saturação estrangula o gradiente** durante o treino.

**Detalhe honesto:** o `unfrozen` tem saturação **até maior** (0.69 vs 0.61) e ainda assim vai
**melhor**. Ou seja, a saturação no **estado final** é **sintoma, não sentença** — o que de fato
importou foi **haver gradiente durante o treino** (o encoder ajustável abriu caminho para o
aprendizado), e não o quanto a sigmoid está saturada ao final.
