# Por que o SVM linear bate a MLP sobre o encoder FLIM congelado?

**Data:** 2026-07-29 · **git:** `06b7b1f` · **Gerado por:** investigação dirigida (Claude Code)
**Relatórios anteriores:** [`report_2026-07-22.md`](report_2026-07-22.md) · [`report_2026-07-28.md`](report_2026-07-28.md)

> **Revisão 2.** Corrige a interpretação da revisão 1, que tratava como fato estabelecido
> algo que era inferência sobre um pipeline externo. Ver §2.

---

## 1. A pergunta

A MLP é um separador não-linear e contém o classificador linear como caso particular.
Sobre os **mesmos pesos congelados** do encoder FLIM, ela deveria gerar hiperplanos
iguais ou melhores que os de um SVM **linear**. O observado é o contrário: a MLP
*frozen* satura em κ ≈ 0,65 (eggs) e 0,62 (protozoan) mesmo com 100% dos rótulos,
enquanto o SVM linear chega a κ ≈ 0,88 e 0,85.

Por quê? Existe explicação documentada?

---

## 2. O que mudou nesta revisão — e por quê

A **revisão 1** afirmava, como fato estabelecido, que o braço "SVM + FLIM" das tabelas
consumia o mapa `conv3` **achatado** (~27.648-d) enquanto a MLP consumia 48-d após
*global average pooling*, e concluía que o gap era "majoritariamente informacional,
~100% em pct ≥ 50".

**Essa afirmação foi longe demais.** Era uma **inferência** sobre um pipeline externo e
não versionado (`data/reports_felipe/`), apresentada com a confiança de uma leitura de
código. Duas coisas a derrubam:

1. **O `AdaptiveAvgPool2d(1)` é da cabeça, não do encoder.** Todo caminho que passa por
   uma cabeça de classificação — ou por `.encode()` — vê 48-d ou 1.280-d, *pooled*. Isso
   inclui **todos** os braços de destilação, o I-JEPA e o
   [svm_classification_flim.py](../../../../src/evaluate/svm_classification_flim.py). O
   `flatten` nesses casos só espreme `[C,1,1] → [C]`, exatamente como descrito por quem
   montou o pipeline.
2. **Evidência empírica contra a tese da dimensionalidade:** o **LeJEPA** é o único braço
   do figure que de fato recebe 27.648-d, e é o **pior de todos** (§7.1, achado I9).

O que **permanece verdadeiro e medido** é que existem *duas convenções distintas* no
repositório (§3), e que **qual delas o pipeline do Felipe usou continua desconhecido**
(LACUNA-1). Esta revisão trata isso como pergunta em aberto e organiza o diagnóstico em
(a) o que vale independentemente da resposta e (b) o que depende dela.

---

## 3. Fato medido: as duas convenções

Medido instanciando cada modelo e rodando um forward (entrada `1×3×200×200`), não lido de
docstring. O encoder é idêntico em FLIM e LeJEPA — só o init dos pesos difere:

```
entrada                    (1,  3, 200, 200)
conv1  Conv+ReLU+MaxPool   (1, 24,  99,  99)
conv2  Conv+ReLU+MaxPool   (1, 32,  49,  49)
conv3  Conv+ReLU+MaxPool   (1, 48,  24,  24)   <- feat_map final
```

Cada bloco é `Conv2d(5×5, padding=same) + ReLU + MaxPool2d(3×3, stride=2)`
([models.py:113-149](../../../../src/models/models.py#L113-L149)). Não há
`AdaptiveAvgPool2d` **dentro** do encoder.

| Braço | Consumidor do `feat_map` | Pooling? | Dimensão |
|:--|:--|:--:|--:|
| **FLIM** (`data/reports_felipe/svm/`) | pipeline **externo** | **LACUNA** | **?** |
| **LeJEPA** ([unified_eval.py:207,222](../../../../src/evaluate/unified_eval.py#L207), [svm.py:121,134](../../../../src/evaluate/svm.py#L121)) | `train_svm` → `flatten(start_dim=1)` | **NÃO** | **27.648** |
| **FLIM residual** ([svm_flim_residual.py:109,122](../../../../src/evaluate/svm_flim_residual.py#L109)) | idem | **NÃO** | ~41.472 / 46.080 |
| **Clássicos** kNN/RF/LGBM/GP/QDA ([classical_classifiers.py:190,206](../../../../src/evaluate/classical_classifiers.py#L190)) | `extract_features` → flatten | **NÃO** | **27.648** |
| Distill 4 — `conv_next_layers` 4×1×1 | proj head | sim, `AdaptiveAvgPool2d(1)` | 1.280 |
| Distill 3 — 3×3 BN2d | proj head | sim, `AdaptiveAvgPool2d(1)` | 1.280 |
| Distill 2 — 2L 1×1 48→256→1280 | proj head | sim, `AdaptiveAvgPool2d(1)` | 1.280 |
| Distill 1 — 1×1 BN2d | proj head | sim, `AdaptiveAvgPool2d(1)` | 1.280 |
| Distill linear — pool6→1728→1280 | proj head | sim, `AdaptiveAvgPool2d(6)` | 1.280 |
| I-JEPA ([ijepa_encoder.py:194](../../../../src/models/ijepa_encoder.py#L194)) | `x.mean(dim=1)` sobre patches | sim | 1.280 |
| Cabeças MLP / `svm_classification_flim.py` | `AdaptiveAvgPool2d(1)` → flatten | sim | 48 |

O repositório já documentava a dualidade em
[svm_distillation.py:179-181](../../../../src/evaluate/svm_distillation.py#L179-L181):
*"Unlike the FLIM SVM pipeline (which accesses .conv1/.conv2/.conv3 spatial tensors),
this function uses `student_model.encode()` to obtain the pooled 1-D embedding"*.

> **Armadilha de nomenclatura.** `svm_ijepa.py:104` e `ijepa_encoder.py` também chamam
> algo chamado `extract_features`, mas é o **método do `IJEPAEncoder`**, que faz
> `x.mean(dim=1)` — *pooled*. Não é a função de módulo de `src/utils/evaluate.py`, que
> achata. Nomes iguais para comportamentos opostos.

> **Nota:** o protozoan usa `conv2` com 30 canais (não 32), lido dos pesos reais via
> `override_arch_channels`. Não altera as dimensões acima: `conv3` é 48 e a grade é 24×24
> nos dois casos, logo D = 27.648 igual.

### 3.1 Consequência para o figure do artigo

A comparação de 8 modelos mistura três regimes: **LeJEPA a 27.648-d**, **destilação e
I-JEPA a 1.280-d**, e **FLIM com convenção desconhecida**. É um problema metodológico do
figure principal, independente da pergunta deste relatório, e deve ser resolvido antes do
camera-ready.

---

## 4. Tabela de resultados

Comparação **pareada e completa** — 54/54 células nos dois braços (3 datasets × 3 splits ×
6 pcts), ambos do mesmo pipeline externo, portanto **mesma convenção de métrica**
(verificado: `max|test_accuracy − test_recall_weighted| = 0,000000` → global/ponderada).
Δ = SVM_frozen − MLP_frozen.

| dataset | pct | SVM acc | MLP-F acc | Δacc | SVM κ | MLP-F κ | Δκ | MLP-U acc | MLP-U κ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| eggs | 1 | 0,174 | 0,069 | +0,105 | 0,128 | 0,010 | +0,118 | 0,047 | 0,003 |
| eggs | 5 | 0,751 | 0,666 | +0,086 | 0,585 | 0,074 | **+0,511** | 0,757 | 0,531 |
| eggs | 25 | 0,887 | 0,738 | +0,149 | 0,805 | 0,415 | +0,391 | 0,901 | 0,819 |
| eggs | 50 | 0,910 | 0,788 | +0,122 | 0,842 | 0,582 | +0,261 | 0,940 | 0,891 |
| eggs | 75 | 0,926 | 0,818 | +0,108 | 0,870 | 0,652 | +0,218 | 0,955 | 0,919 |
| eggs | 100 | 0,935 | 0,819 | +0,116 | 0,885 | 0,655 | +0,230 | 0,971 | 0,947 |
| larvae | 1 | 0,390 | 0,858 | −0,468 | 0,028 | 0,098 | −0,069 | 0,888 | 0,194 |
| larvae | 5 | 0,921 | 0,926 | −0,004 | 0,571 | 0,655 | −0,084 | 0,968 | 0,855 |
| larvae | 25 | 0,957 | 0,944 | +0,012 | 0,805 | 0,740 | +0,065 | 0,985 | 0,931 |
| larvae | 50 | 0,961 | 0,961 | −0,001 | 0,821 | 0,827 | −0,005 | 0,988 | 0,945 |
| larvae | 75 | 0,963 | 0,960 | +0,003 | 0,835 | 0,826 | +0,010 | 0,989 | 0,950 |
| larvae | 100 | 0,971 | 0,965 | +0,005 | 0,868 | 0,845 | +0,023 | 0,993 | 0,969 |
| protozoan | 1 | 0,208 | 0,050 | +0,158 | 0,145 | 0,006 | +0,139 | 0,067 | 0,019 |
| protozoan | 5 | 0,765 | 0,640 | +0,125 | 0,623 | 0,214 | **+0,409** | 0,780 | 0,597 |
| protozoan | 25 | 0,862 | 0,705 | +0,157 | 0,775 | 0,446 | +0,329 | 0,894 | 0,825 |
| protozoan | 50 | 0,888 | 0,752 | +0,136 | 0,816 | 0,548 | +0,268 | 0,928 | 0,880 |
| protozoan | 75 | 0,901 | 0,770 | +0,131 | 0,837 | 0,592 | +0,245 | 0,946 | 0,910 |
| protozoan | 100 | 0,907 | 0,781 | +0,126 | 0,847 | 0,616 | +0,231 | 0,960 | 0,933 |

Quatro fatos estruturais:

1. Sinal **unânime 3/3 splits em 12/12 células** de eggs+protozoan → não é artefato de média.
2. **Δκ ≫ Δacc** (κ +0,22..+0,51 contra acc +0,09..+0,16).
3. **larvae (2 classes) não tem o efeito** — Δ sempre menor que o dp entre splits.
4. O gap **não encolhe** com mais rótulos: a 100% ainda é Δκ ≈ +0,23.

> **larvae pct=1 não deve ser citado.** O Δacc = −0,468 é **falha do SVM**, não vitória da
> MLP: por split o SVM dá acc 0,159 / 0,127 / 0,884 (colapso na **minoritária**) e a MLP
> colapsa na **majoritária** (0,829 / 0,873 / 0,873). Ambos κ ≈ 0.

> **Braços "ours" (`sigmoid2l`, `relu2l`): use apenas κ.** `test_accuracy` deles é
> **acurácia balanceada** e `test_f1_weighted` é **F1 macro**
> ([classification.py:31-62](../../../../src/metrics/classification.py#L31-L62), default
> `average="macro"`) — não comparáveis em acc/F1 com os CSVs do Felipe. Cobertura:
> `sigmoid2l` só pct {1,75}; `relu2l` só pct {5,75} e só unfrozen.

---

## 5. Auditoria das duas cabeças

### 5.1 SVM

`SVC(kernel="linear", C=1e2, gamma="auto", decision_function_shape="ovo", max_iter=10000)`,
`class_weight=None`, `tol=1e-3`, `random_state=None`, `probability=False` —
[evaluate.py:236-252](../../../../src/utils/evaluate.py#L236-L252). `gamma`, `degree` e
`coef0` são **inertes** com kernel linear.

- `decision_function_shape` só muda o **shape do retorno**; o `SVC` **sempre** treina OvO.
  Hiperplanos reais (verificado via `coef_.shape`): eggs C(9,2) = **36**, protozoan
  C(7,2) = **21**, larvae **1**.
- **Sem `StandardScaler`** no caminho FLIM (grep exaustivo). A única exceção é
  [svm_distill_with_projection.py:197-200](../../../../src/evaluate/svm_distill_with_projection.py#L197-L200),
  braço 1280-d, cujo comentário admite *"StandardScaler é necessário com 1280 dims para
  convergência"*.
- **Sem tuning**: zero `GridSearchCV` / `RandomizedSearchCV` / `param_grid` no repo.
  `C=1e2` é valor herdado do notebook ancestral.
- 43 `ConvergenceWarning` em `logs/svm_frozen_20260612_144945.log`, **todos do braço
  proj-1280** — nenhum do caminho FLIM. Convergência do caminho FLIM: **LACUNA**.

### 5.2 MLP — são duas, e não devem ser confundidas

- **M1 — MLP ReLU do Felipe** (a das tabelas): externo. ~46.601 params (número
  **inferido**, ver §8), ReLU + Dropout 0.3, `wd=1e-4`, 300 épocas com early stopping, lr
  do encoder 10× menor. `epochs_trained` bate o **teto de 300** em eggs e protozoan para
  todo pct ≥ 25% (early stopping nunca dispara); no unfrozen para em 106–241.
- **M2 — cabeças do repo**: `MLPHead`
  ([models.py:304-326](../../../../src/models/models.py#L304-L326)) — **código morto, nunca
  produziu resultado**; e `TwoLayerSigmoidHead`
  ([models.py:351-389](../../../../src/models/models.py#L351-L389)) treinada por
  [classification_flim_module.py](../../../../src/modules/classification_flim_module.py):
  1.401 params (eggs), AdamW `lr=5e-4`, **`wd=5e-2` num único param group (decai também os
  biases de saída)**, 100 épocas fixas sem early stopping, `NLLLoss` sobre `log(softmax)`.
- **Sem tuning em nenhuma** (Ray é escalonador de jobs, nunca tuner).
- **Subtreino no regime baixo (M2):** eggs pct1 = 24 imagens = **1 step/época** → 100
  passos de gradiente no treino inteiro, 10 deles sob warmup.

### 5.3 Geometria do embedding pooled

De `results/relu_vs_sigmoid_flatten.csv` (runs `mode=frozen`), gerado por
[analyze_sigmoid_saturation.py:106-123](../../../../tools/analyze_sigmoid_saturation.py#L106-L123):

| dataset | sat_frac sigmoid | `cv_P` | `dead_chan_P` | `effdim_P` | `probe_acc_P` |
|:--|--:|--:|--:|--:|--:|
| eggs | 0,635 | 0,035 | **0,000** | **1,39** | **0,770** |
| larvae | 0,574 | 0,090 | **0,000** | **1,20** | **0,968** |
| protozoan | 0,632 | 0,101 | **0,000** | **1,01** | **0,653** |

- **Nenhum canal ReLU morto** → o encoder está saudável; o problema não é ele.
- **`effdim_P` (participation ratio) = 1,0–1,4**: após o GAP o vetor 48-d é dominado por
  **uma única direção** (magnitude global).
- **`probe_acc_P`** é `LogisticRegression(max_iter=2000, C=1.0)` **com `StandardScaler`**,
  5-fold CV, sobre `head.pool(enc(x)).flatten(1)`. **É convexo, com ótimo global** — logo
  0,770 / 0,653 / 0,968 são **tetos informacionais do embedding pooled**, não falhas de
  otimizador.
- Prova de que o encoder aí é FLIM-init **puro**: `probe_acc_P` é bit-a-bit idêntico entre
  pct=1 e pct=75 dentro de cada split (eggs: 0,7896 / 0,7720 / 0,7470). Com
  `freeze_encoder=True` o encoder nunca é atualizado.

### 5.4 O que é simétrico (hipóteses descartadas)

Mesmo `train` (JSONs verificados por hash), mesmo transform `_build_test(200)`, mesma
`imagenet_norm=True`, **sem augmentation em nenhum** (`V_train=1`), encoder **sem
BatchNorm e sem Dropout** (→ `eval()` vs `train()` é no-op), **sem scaler em nenhum dos
dois braços**, `class_weight=None` nos dois.

---

## 6. Evidências da literatura

- **Viés implícito do GD.** Soudry, Hoffer, Nacson, Gunasekar, Srebro — *The Implicit Bias
  of Gradient Descent on Separable Data*, JMLR 19(70), 2018,
  [arXiv:1710.10345](https://arxiv.org/abs/1710.10345). GD em dados separáveis converge à
  direção de margem máxima, mas **logaritmicamente devagar** (~O(1/log t)). **Ressalva: a
  teoria prevê melhora lenta, não platô duro.** Complementos: Lyu & Li ICLR 2020
  ([arXiv:1906.05890](https://arxiv.org/abs/1906.05890)); Nacson et al. AISTATS 2019
  ([arXiv:1803.01905](https://arxiv.org/abs/1803.01905)); Rosset/Zhu/Hastie NIPS 2003
  (hinge e CE têm o **mesmo** limite de margem — enfraquece "hinge é intrinsecamente
  melhor").
- **Linear probing.** Alain & Bengio ([arXiv:1610.01644](https://arxiv.org/abs/1610.01644));
  Razavian et al. CVPRW 2014 ([arXiv:1403.6382](https://arxiv.org/abs/1403.6382) — SVM
  linear sobre features CNN congeladas batendo sistemas ajustados); Tian et al. ECCV 2020
  ([arXiv:2003.11539](https://arxiv.org/abs/2003.11539)); Kumar et al. ICLR 2022 **LP-FT**
  ([arXiv:2202.10054](https://arxiv.org/abs/2202.10054)).
- **Contra-evidência, citada deliberadamente.** He et al. **MAE**, CVPR 2022
  ([arXiv:2111.06377](https://arxiv.org/abs/2111.06377)): *"linear probing misses the
  opportunity of pursuing strong but non-linear features"*, com cabeça não-linear
  **melhor** (73,5 → 79,1). Zaiem et al. 2024
  ([arXiv:2308.14456](https://arxiv.org/abs/2308.14456)): cabeças de probing maiores mudam
  o ranking. **A literatura de SSL reporta cabeças não-lineares empatando ou ganhando,
  nunca perdendo 20 pontos de κ.**
- **Margem com poucas amostras.** Bartlett & Shawe-Taylor 1999 (MIT Press); Koltchinskii &
  Panchenko, Ann. Statist. 30(1), 2002; Grønlund/Kamma/Larsen ICML 2020. Explica gap a
  1–25%, **não** o gap persistente a 100%.
- **Escala e condicionamento.** LeCun/Bottou/Orr/Müller, *Efficient BackProp*, LNCS 7700
  (entradas não centradas → Hessiano mal condicionado → platô); Ioffe & Szegedy ICML 2015;
  docs do sklearn (*"MLP is sensitive to feature scaling"*). GAP após ReLU dá features
  estritamente não-negativas, média positiva grande.
- **Bias-variance / double descent.** Belkin et al. PNAS 116(32), 2019.
- **Neural collapse.** Papyan/Han/Donoho PNAS 117(40), 2020; Yang et al. NeurIPS 2022 (sob
  desbalanceamento, minoritárias colapsam entre si → degradaria **ambos** os braços
  igualmente).
- **Desbalanceamento.** Cao et al. **LDAM**, NeurIPS 2019
  ([arXiv:1906.07413](https://arxiv.org/abs/1906.07413)). Hinge zera para pontos além da
  margem; CE nunca zera e sob desbalanceamento a majoritária domina o gradiente.
- **OvO vs softmax.** Fürnkranz JMLR 2, 2002 (cada subproblema binário é mais simples e usa
  só uma fração dos dados); Hsu & Lin IEEE TNN 13(2), 2002; Rifkin & Klautau JMLR 5, 2004
  (contra).
- **SVM vs MLP em small data.** Goddard & Shamir IEEE Access 10, 2022; Fernández-Delgado et
  al. JMLR 15, 2014; Grinsztajn et al. NeurIPS 2022
  ([arXiv:2207.08815](https://arxiv.org/abs/2207.08815)).

---

## 7. Diagnóstico

### 7.1 O que vale INDEPENDENTE da convenção do Felipe

Estes achados não dependem de LACUNA-1 e podem ser escritos hoje:

| # | Achado | Evidência |
|:--|:--|:--|
| I1 | O embedding **pooled 48-d** tem teto linear convexo de **0,770 / 0,653 / 0,968** | `probe_acc_P`, LogReg + StandardScaler, ótimo global (§5.3) |
| I2 | **H3 refutada.** A MLP frozen **supera** esse teto em 2 de 3 datasets (+0,049 eggs, +0,128 protozoan) | se capacidade extra fosse penalidade de variância, ficaria abaixo |
| I3 | O gap é de **ajuste, não de generalização**. A MLP frozen não chega a 83% no próprio treino de eggs; gap train−test = **0,010–0,020** | a pct=100 `validation ≡ training` (hash verificado) → `val_*` é métrica de treino |
| I4 | **Δκ ≫ Δacc**, e o erro se concentra nas minoritárias (1,56% em eggs, 0,82% em protozoan) | §4; `class_weight=None` nos dois braços |
| I5 | **larvae não tem o efeito** — e é justamente onde o teto pooled já é 0,968 | §4 + §5.3 |
| I6 | O gap **não encolhe** de 25% a 100% | §4 → mata margem-em-poucos-dados como causa *dominante* |
| I7 | Descongelar leva a **train κ = 1,000** | a capacidade da cabeça nunca foi o gargalo (LP-FT) |
| I8 | O figure de 8 modelos **mistura convenções** (27.648 / 1.280 / desconhecida) | §3 e §3.1 |
| I9 | **Dimensão alta não compra desempenho.** O LeJEPA é o único a 27.648-d e é o **pior** (κ 0,18–0,38 contra 0,70–0,83 dos distill *pooled*) | §3 + README |

**I2 e I3 juntos são o resultado mais sólido do relatório:** a MLP frozen não está
subtreinada em relação ao que recebe, e não está com excesso de variância — ela está
**saturando a informação disponível no vetor que lhe é entregue**.

**I9 é a evidência que derruba a leitura da revisão 1.** Ressalva na direção contrária: os
pesos do LeJEPA vêm de um SSL que não funcionou, então as features são ruins na origem —
não é um teste limpo do efeito do pooling. Mas, como está, a evidência disponível **não
sustenta** a tese de que dimensionalidade explica o gap.

### 7.2 H1 — Artefato experimental (embedding diferente) — **CONDICIONAL, não resolvida**

**PRÓ:** as duas convenções existem e diferem por 576× (§3, medido); o teto convexo em
48-d fica 0,17–0,25 abaixo do SVM em eggs/protozoan e **empatado em larvae**, e esse padrão
de *headroom* acompanha o gap nos 3 datasets; os recortes são determinísticos
(`Resize(200)+CenterCrop(200)`), o que torna templates posicionais aprendíveis por um braço
achatado e invisíveis após GAP; `test_time` do SVM em eggs = **2170,63 s para 2557 imagens
(0,85 s/img)** contra 3,01 s da MLP — e um `SVC` linear em 48-d prediz 2557 amostras em
**0,20 s** (medido, pior caso com todos os pontos virando SV).

**CONTRA:** **I9** — quem realmente recebe 27.648-d (LeJEPA) vai pior de todos; nenhuma
prova documental do pipeline do Felipe (LACUNA-1); e o notebook ancestral
(`2_flim_classification.ipynb`, célula 47) mostra a `ClassificationModel` da linhagem
original achatando **também** no braço MLP, o que, se herdado, tornaria os dois braços
simétricos.

**Status: em aberto.** Depende de LACUNA-1.

### 7.3 H2 — Fenômeno de otimização — **peso real, e cresce se H1 cair**

Três sub-casos distintos:

- **H2-a — a MLP do Felipe, em pct alto.** Ela entrega **acima** do teto linear convexo do
  embedding pooled (I2). Se o SVM dela também for pooled 48-d, então **dois classificadores
  lineares sobre features idênticas diferem por 0,165** (SVC `C=1e2` OvO = 0,935 contra
  LogReg `C=1.0` padronizada = 0,770). Isso seria um achado forte sobre margem, OvO e
  regularização — e tornaria H2 a explicação central. Se o SVM for achatado, o contraste
  desaparece.
- **H2-b — regime de poucos rótulos: contribuição aditiva real.** Decompondo
  `Δκ(pct) = Δκ_base + Δκ_amostra(pct)`, com `Δκ_base` = valor em pct=100:

  | pct | eggs Δκ | excesso s/ 0,230 | protozoan Δκ | excesso s/ 0,231 |
  |--:|--:|--:|--:|--:|
  | 5 | 0,511 | **+0,281** | 0,409 | **+0,178** |
  | 25 | 0,391 | +0,161 | 0,329 | +0,098 |
  | 50 | 0,261 | +0,031 | 0,268 | +0,037 |
  | 75 | 0,218 | −0,012 | 0,245 | +0,014 |
  | 100 | 0,230 | 0,000 | 0,231 | 0,000 |

  O termo de amostra **decai monotonicamente a zero em pct ≈ 50** nos dois datasets, e em
  pct=5 chega a ser **maior** que o termo de base. É aqui que Soudry et al. e Bartlett &
  Shawe-Taylor se aplicam. **Este termo é robusto a LACUNA-1.**
- **H2-c — a cabeça do repo (`sigmoid2l`): H2 em estado puro e catastrófica.** Frozen em
  eggs pct75 dá test acc 0,109–0,176 (**nível do acaso, 1/9 = 0,111**) e κ ≈ 0, sobre
  features cujo probe linear **do mesmo tensor** dá 0,770. Controle interno perfeito.
  Culpados nomeáveis: `wd=5e-2` sobre 1.401 params incluindo os biases de saída; `NLLLoss`
  sobre saída pós-Softmax (gradiente achatado); 57–64% das unidades sigmoides saturadas;
  `ModelCheckpoint(monitor="val/kappa")` preso em ~0 selecionando checkpoint arbitrário.
  **Não é a cabeça das tabelas do artigo — é um bug separado.**

### 7.4 H3 — Capacidade extra só adiciona variância — **REFUTADA**

Ver I2 e I3. A MLP **supera** o teto linear do próprio embedding; o gap train−test é de
0,010–0,020 (variância excessiva se manifesta como train ≫ test, o que não ocorre);
`dead_chan_P = 0`; e a literatura (MAE, Zaiem) reporta cabeças não-lineares ganhando, nunca
perdendo 20 pontos de κ. **Esta refutação não depende de LACUNA-1.**

### 7.5 Perguntas-ponte

**Por que Δκ ≫ Δacc?** Dois mecanismos somados. *(i) Geométrico:* `effdim_P` ≈ 1 → após o
GAP os 48-d colapsam sobre uma direção, e só dá para *ordenar* classes nesse eixo. A
majoritária (65,4% eggs / 59,7% protozoan) fica separável; as minoritárias colapsam entre
si. Acurácia é dominada pela majoritária e cai pouco; κ desconta o acaso e despenca.
*(ii) Perda:* `class_weight=None` + CE dominada pela majoritária (Cao et al.) contra OvO com
36/21 binários **localmente balanceados** (Fürnkranz), cujo hinge zera para pontos fáceis.

**Por que larvae não tem o efeito?** Porque o teto pooled ali já é 0,968 — não há headroom.
É o controle negativo. **Ressalva:** larvae é simultaneamente binário *e* saturado no
pooled, então sozinho **não separa** a hipótese do pooling da hipótese OvO. Mata, isso sim,
*neural collapse* como causa (degradaria os dois braços igualmente) e desbalanceamento *per
se* (larvae tem minoritária de 12,7% e nada acontece).

**Por que o gap não encolhe a 100%?** Mata margem-em-poucos-dados e subtreino como causa
*dominante*; favorece um teto estrutural fixo, independente de rótulos.

**Por que descongelar resolve?** Com o encoder treinável, o gradiente reorganiza os 48
canais para que a informação de classe seja linearizável *após* a média espacial — o encoder
passa a codificar em **magnitude de canal** o que antes dependia da cabeça. Train κ = 1,000
no unfrozen contra 0,665 / 0,636 no frozen, e o unfrozen dispara early stopping (106–241
épocas) enquanto o frozen bate o teto de 300. Mecanismo do LP-FT (Kumar et al.).

---

## 8. Lacunas de dados

1. **LACUNA-1 (central, bloqueia H1).** O pipeline `svm`/`flim_mlp` do Felipe **não é
   versionado**. Os 54 CSVs têm **zero colunas de hiperparâmetro**;
   `data/reports_felipe/flim_mlp/checkpoints/` tem 216 subdiretórios **vazios**. **Não se
   sabe se o SVM dele achata ou poola** — e é dele que saem os números das tabelas. *O autor
   do pipeline informará; até lá, §7.2 fica em aberto.*
2. **Anomalia não explicada:** `test_time` do SVM em eggs = 2170,63 s (0,85 s/img),
   ~10.000× o de um `SVC` linear em 48-d medido aqui (0,20 s para 2557 amostras). Sob a
   hipótese *pooled*, isso precisa de outra explicação (hardware, contenção de CPU, ou algo
   mais no pipeline).
3. **Tensão a resolver:** se o SVM do Felipe for pooled 48-d, ele atinge 0,935 onde a LogReg
   convexa padronizada sobre o mesmo 48-d atinge 0,770. Duas soluções lineares sobre
   features idênticas não deveriam diferir tanto — ou o probe subestima (é 5-fold CV
   **dentro do test set**, com `C=1.0`, não `C=1e2`), ou há algo mais.
4. Os "46.601 params" da MLP do Felipe (`A_reports/july/check_sanity.md:182`) **não são
   rastreáveis** — o número é exatamente `MLPHead(48,256,9)`, ou seja foi **inferido**.
5. Convergência do SVM no caminho FLIM: **desconhecida**.
6. `sigmoid2l` só cobre pct {1,75}; `relu2l` só {5,75} e só unfrozen; **não existe braço
   frozen do `relu2l`**. Não existe M2 a 100%.
7. Não existe checkpoint da época 99 de nenhum run (`last.ckpt` tem a mesma época do
   `best_kappa.ckpt` nos 94 runs).
8. **LACUNA-2 (literatura):** não há estudo publicado de OvO-vs-softmax em "muitas classes,
   poucos exemplos por classe" sobre features profundas congeladas, nem SVM-vs-MLP em
   microscopia de parasitos com embeddings congelados. **Posicionável como contribuição.**
9. Divergência não resolvida: a `sigmoid2l` tem train acc = frequência da majoritária
   (0,655) mas **test** acc 0,109–0,176, com priors de train/test praticamente idênticos.
   Sugere que o `best_kappa.ckpt` avaliado **não é** o de fim de treino. Relevante só para
   H2-c.

---

## 9. Próximo experimento

**T1 — PRIORIDADE 1. SVM linear achatado vs pooled, mesmo encoder FLIM puro congelado.**
Fecha LACUNA-1 **sem depender do pipeline externo** e é o único teste que isola o pooling
do resto. Hoje não existe: `results/svm_softplus2l_results.csv` tem
`train_freeze_encoder=False` nas 18 linhas, e
[svm_classification_flim.py:73-91](../../../../src/evaluate/svm_classification_flim.py#L73-L91)
força `AdaptiveAvgPool2d(1)` — só dá o lado *pooled*.

Script standalone (não toca no repo): carrega
`data/to_mateus/model/ch24_32_48_a0.5_f5/<ds>/train<N>`, extrai `conv3` e roda
`SVC(kernel="linear", C=1e2, max_iter=10000)` **duas vezes**: (a) `flatten(1)` → 27.648-d;
(b) `AdaptiveAvgPool2d(1)` → 48-d.

**Predição falsificável:** (a) ≈ 0,93–0,94 e (b) ≈ 0,77–0,80 → o pooling explica o gap. Se
(b) ≈ 0,93, **H1 cai inteira** e H2 passa a ser a explicação central. Rodar `larvae`
primeiro (fit de ~6 s) como smoke test.

**T2 — PRIORIDADE 2. Escada de resolução espacial.** Mesmo script variando
`AdaptiveAvgPool2d(k)` com k = 1 (48-d), 2 (192-d), 4 (768-d), 8 (3072-d), 24 (= flatten).
Predição: curva monótona em eggs/protozoan e **plana** em larvae. Transforma o diagnóstico
em **contribuição positiva** — "quanta resolução espacial um classificador linear precisa
sobre features FLIM" — e provavelmente mostra que k=4 recupera quase tudo por 1/36 do custo.

**T3 — PRIORIDADE 3. Uniformizar a convenção do figure (§3.1).** Reavaliar o LeJEPA com a
mesma convenção dos demais, ou declarar a diferença explicitamente na legenda. Sem isso a
curva compara três regimes de feature.

**T4 — PRIORIDADE 4. Consertar a cabeça do repo (H2-c)**, um knob por vez sobre os 48-d
congelados: (i) `wd` 5e-2 → 0; (ii) grupo `no_decay` para biases; (iii) logits crus +
`CrossEntropyLoss` no lugar de Softmax + `NLLLoss`; (iv) ReLU no lugar da Sigmoid oculta.
**Alvo a bater: 0,770.** Não afeta as conclusões sobre as tabelas do artigo.

**Não vale a pena:** tunar `C`/`class_weight` do SVM ou a MLP do Felipe.

---

## 10. Correções a relatórios anteriores

**A [`report_2026-07-22.md`](report_2026-07-22.md) §5.4** afirma: *"Mesmo encoder, mesma
informação disponível — a diferença está no classificador, não na representação."* Essa
frase **não está verificada**. Se o SVM do Felipe achata, a informação disponível não é a
mesma; se poola, a frase se sustenta. Hoje é pergunta em aberto, não conclusão.

**A `review_code_22_jul_2026.md` §B1** conclui que *"as features são ótimas; a camada
sigmoide oculta é o que quebra"*. Está **correto para a cabeça do repo** (`sigmoid2l`,
H2-c) e é confirmado pelo probe de 0,770 sobre o mesmo tensor. Mas **não explica** o gap
SVM-vs-MLP-do-Felipe, que é o das tabelas do artigo — são dois fenômenos distintos com o
mesmo sintoma.

**A revisão 1 deste relatório** tratava a convenção de feature do braço do Felipe como fato
estabelecido e atribuía a H1 peso "dominante, ~100% do gap em pct ≥ 50". Corrigido em §2 e
§7.2.

---

## 11. Fontes

| Braço | Arquivo |
|:--|:--|
| SVM + FLIM | `data/reports_felipe/svm/report_svm_*.csv` (54 CSVs, externo) |
| FLIM + MLP (ReLU, Felipe) | `data/reports_felipe/flim_mlp/report_flim_mlp_*.csv` (54 CSVs, 2 linhas cada) |
| FLIM + MLP (Sigmoid, ours) | `results/sigmoid2l_test_results.csv` (34 linhas) |
| FLIM + MLP (ReLU-saída, ours) | `results/relu2l_test_results.csv` (18 linhas) |
| Probe linear / saturação | `results/relu_vs_sigmoid_flatten.csv` (34 linhas) |
| Formas dos tensores | medido por instanciação (§3), entrada `1×3×200×200` |
| Pesos FLIM | `data/to_mateus/model/ch24_32_48_a0.5_f5/<ds>/train<N>/models` |
