# Por que o SVM linear bate a MLP sobre o mesmo encoder FLIM congelado?

**Data:** 2026-07-29 · **git:** `503fb5a` · **Gerado por:** investigação dirigida (Claude Code)
**Relatórios anteriores:** [`report_2026-07-22.md`](report_2026-07-22.md) · [`report_2026-07-28.md`](report_2026-07-28.md)

> **Aviso de escopo.** Este relatório **corrige parcialmente** o relatório de 22/07 (ver §9)
> e qualifica o `review_code_22_jul_2026.md`. A premissa que sustentava aquelas leituras
> — "mesmo encoder, mesma informação disponível" — **é falsa**.

---

## 1. A pergunta

Uma MLP é um separador não-linear. Sobre os **mesmos pesos congelados** do encoder FLIM,
ela deveria, em princípio, ser capaz de representar qualquer hiperplano que um **SVM
linear** encontre — e potencialmente algo melhor. A classe de funções da MLP contém a
classe de funções do classificador linear. Ainda assim, nas tabelas do artigo o SVM vence
por margens grandes e sistemáticas (até **+0,51 de κ**).

**Por que isso acontece? Existe explicação específica e documentada?**

Existe. E ela **não** é sobre otimização, nem sobre capacidade.

---

## 2. Veredito central — a premissa da pergunta é falsa

**O SVM e a MLP não recebem o mesmo tensor.**

| Braço | O que entra no classificador | Dimensão (eggs) |
|:--|:--|--:|
| SVM + FLIM | mapa `conv3` **achatado** (`flatten`) | **~27.648** |
| FLIM + MLP (Felipe) | mesmo mapa **após `AdaptiveAvgPool2d(1)`** | **48** |

Fator **576×**. O *global average pooling* destrói toda a informação espacial: cada canal
vira um único escalar (sua média sobre a grade 24×24). O SVM opera sobre a grade inteira.

**Portanto o gap é majoritariamente informacional, não de otimização nem de capacidade.**
A MLP não "deixa de encontrar" o hiperplano do SVM — esse hiperplano **não existe** no
espaço que ela recebe.

### 2.1 A medição que fecha o caso

Mesmo encoder FLIM-init congelado, `pct=100`, média dos 3 splits:

| dataset | (a) probe linear **48-d pooled** | (b) MLP-frozen (Felipe) | (c) SVM (Felipe, flatten) | headroom (c)−(a) | gap (c)−(b) | (b)−(a) |
|:--|--:|--:|--:|--:|--:|--:|
| eggs (9 cl.) | **0,770** | 0,819 | 0,935 | **0,165** | **0,116** | **+0,049** |
| protozoan/cistos (7 cl.) | **0,653** | 0,781 | 0,907 | **0,254** | **0,126** | **+0,128** |
| larvae (2 cl.) | **0,968** | 0,965 | 0,971 | **0,003** | **0,005** | −0,003 |

A coluna (a) é `probe_acc_P` de `results/relu_vs_sigmoid_flatten.csv` (`mode=frozen`),
gerada por
[analyze_sigmoid_saturation.py:106-123](../../../../tools/analyze_sigmoid_saturation.py#L106-L123):
`LogisticRegression(max_iter=2000, C=1.0)` **com `StandardScaler`**, 5-fold CV, sobre
`head.pool(enc(x)).flatten(1)`.

Três consequências, em ordem de força:

1. **O problema é convexo e tem ótimo global.** A regressão logística encontra o melhor
   separador linear possível em 48-d. Logo **0,770 / 0,653 / 0,968 são tetos
   informacionais do embedding pooled**, não resultados de um otimizador que falhou.
2. **A MLP frozen está ACIMA desse teto em 2 de 3 datasets** (+0,049 em eggs, +0,128 em
   protozoan). Uma cabeça que **supera o ótimo global linear das próprias features** não
   está subtreinada em relação ao que recebe.
3. **O headroom prediz o gap nos 3 datasets, incluindo o zero de larvae.** Onde há
   headroom espacial (eggs 0,165 / protozoan 0,254), há gap (0,116 / 0,126). Onde não há
   (larvae 0,003), não há gap (0,005). Isto é **predição, não ajuste post-hoc**.

Os mesmos valores já constavam de `analise_achatamento_relu_sigmoid.md` §3 — confirmação
independente.

**Prova de que o encoder é FLIM-init puro:** `probe_acc_P` é bit-a-bit idêntico entre
`pct=1` e `pct=75` dentro de cada split (eggs: 0,7896 / 0,7720 / 0,7470). Com
`freeze_encoder=True` o encoder nunca é atualizado, então o probe mede sempre o mesmo
tensor.

### 2.2 Corroboração independente pelos tempos de inferência

Verificado nos CSVs do Felipe (`pct=100`, frozen, média dos 3 splits):

| dataset | N_test | SVM `test_time` | MLP `test_time` | SVM `training_time` |
|:--|--:|--:|--:|--:|
| eggs | 2557 | **2170,63 s** | 3,01 s | 3104,72 s |
| protozoan/cistos | 4786 | **102,43 s** | 4,67 s | 122,19 s |
| larvae | 1757 | 4,93 s | 1,85 s | 6,13 s |

São **0,85 s por imagem** só para *predizer* com um SVM **linear** em eggs. Um
`SVC(kernel="linear")` sobre 48 features prediria 2557 amostras em milissegundos. A ordem
de grandeza só fecha com d ≈ 2,8·10⁴.

**Este é o argumento mais valioso do relatório: ele é observável diretamente nos dados
publicados do Felipe, sem nenhum acesso ao código dele.**

### 2.3 Refutação de "a MLP do Felipe também achatava"

A objeção óbvia é que o notebook ancestral achatava nos dois braços. Ela **se refuta com
os próprios números de treino**:

Se `flim_mlp` consumisse o mapa achatado, a primeira camada seria `Linear(27648, h)`
≈ **7·10⁶ parâmetros** sobre **2555** amostras de treino. Regime d ≫ n: o problema é
**separável por construção**, e a acurácia de treino tenderia a **1,000**.

O observado é **train acc = 0,826 (eggs) / 0,792 (cistos)**, em platô, no teto de 300
épocas. Um modelo linear em 27.648-d **não pode** ficar em 0,826 no próprio conjunto de
treino. **Logo `flim_mlp` NÃO achata.**

> Nota metodológica: a `pct=100`, `validation ≡ training` — as listas são idênticas
> (verificado por hash nos 3 datasets × 3 splits). Portanto a coluna `val_*` dos CSVs do
> Felipe **é** a métrica de treino.

---

## 3. Tabela de resultados

Comparação **pareada e completa**: 54/54 células nos dois braços (3 datasets × 3 splits ×
6 pcts), ambos vindos do pipeline externo. Δ = SVM_frozen − MLP_frozen. Métrica
**global/ponderada** nos dois braços (verificado:
`max|test_accuracy − test_recall_weighted| = 0,000000`).

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

Fatos estruturais desta tabela:

- **O sinal é unânime em 3/3 splits nas 12/12 células** de eggs+protozoan. Não é artefato
  de média.
- **Δκ ≫ Δacc** em toda a extensão de eggs e protozoan.
- **larvae não tem o efeito.**
- **O gap não encolhe a 100%** (eggs +0,230, protozoan +0,231 de κ). Isto é decisivo — ver §6.3.

**Ressalva obrigatória sobre `larvae` pct=1.** O Δacc de −0,468 **não é vitória da MLP**:
é falha do SVM. Por split, o SVM dá acc = 0,159 / 0,127 / 0,884 (colapso na classe
**minoritária**), e a MLP colapsa na **majoritária** (0,829 / 0,873 / 0,873). Ambos têm
κ ≈ 0 — são dois modos de colapso, não um resultado. **Não citar esta célula
isoladamente.**

**Ressalva sobre os braços "ours"** (`sigmoid2l`, `relu2l`): use **apenas κ**. O
`test_accuracy` deles é **acurácia balanceada** e `test_f1_weighted` é **F1 macro**
([classification.py:31-62](../../../../src/metrics/classification.py#L31-L62), default
`average="macro"`), portanto **não são comparáveis** em acc/F1 com os CSVs do Felipe.
Cobertura: `sigmoid2l` só pct {1, 75}; `relu2l` só pct {5, 75} e só unfrozen.

---

## 4. Auditoria das duas cabeças

### 4.1 SVM

Configuração real, em
[evaluate.py:236-252](../../../../src/utils/evaluate.py#L236-L252):

```python
SVC(kernel="linear", C=1e2, gamma="auto",
    decision_function_shape="ovo", max_iter=10000)
# class_weight=None, tol=1e-3, random_state=None, probability=False
```

- `gamma` / `degree` / `coef0` são **inertes** com kernel linear.
- `decision_function_shape` muda apenas o **shape do retorno**; o `SVC` **sempre** treina
  OvO internamente. Hiperplanos reais (verificado via `coef_.shape`): eggs
  C(9,2) = **36**, protozoan C(7,2) = **21**, larvae **1**.
- **Sem `StandardScaler`** em todo o caminho FLIM (grep exaustivo). A única exceção do
  repo é
  [svm_distill_with_projection.py:197-200](../../../../src/evaluate/svm_distill_with_projection.py#L197-L200),
  no braço de destilação 1280-d, cujo comentário admite que "StandardScaler é necessário
  com 1280 dims para convergência".
- **Sem tuning algum**: zero ocorrências de `GridSearchCV` / `RandomizedSearchCV` /
  `param_grid` no repositório inteiro. `C=1e2` é valor herdado do notebook ancestral.
- **Capacidade efetiva em eggs: 36 hiperplanos × 27.648 dims ≈ 995k pesos de decisão.**

### 4.2 MLP — são duas, e não devem ser confundidas

**M1 — MLP ReLU do Felipe** (a que aparece nas tabelas do artigo). Pipeline externo,
~46.601 parâmetros (**número inferido, não medido** — ver LACUNA-3), ReLU + Dropout 0.3,
`wd=1e-4`, 300 épocas com early stopping, lr do encoder 10× menor. O `epochs_trained`
bate o **teto de 300** em eggs e cistos para todo pct ≥ 25% (o early stopping nunca
dispara); no unfrozen ela para entre 106 e 241 épocas.

**M2 — cabeças deste repositório**:

- `MLPHead` ([models.py:304-326](../../../../src/models/models.py#L304-L326)) — **código
  morto, nunca produziu resultado publicado**.
- `TwoLayerSigmoidHead` ([models.py:351-389](../../../../src/models/models.py#L351-L389)),
  treinada por
  [classification_flim_module.py](../../../../src/modules/classification_flim_module.py):
  1.401 parâmetros (eggs), AdamW `lr=5e-4`, **`wd=5e-2` em um único param group — decai
  também os biases de saída**, 100 épocas fixas sem early stopping, `NLLLoss` sobre
  `log(softmax)`.

**Nenhuma das duas tem tuning.** O Ray, no repo, é escalonador de jobs — nunca tuner.

**Subtreino no regime baixo (M2):** eggs pct=1 são 24 imagens = **1 step por época** →
**100 passos de gradiente no treino inteiro**, 10 deles ainda sob warmup.

**A cabeça do repo é um caso à parte e catastrófico.** `sigmoid2l` frozen em eggs pct=75
dá test acc **0,109–0,176** (nível do acaso: 1/9 = 0,111) e κ ≈ 0 — **sobre features cujo
probe linear do mesmo tensor dá 0,770**. É um controle interno perfeito: as features
prestam, a cabeça não. Culpados nomeáveis:

1. `wd=5e-2` sobre 1.401 params, incluindo os biases de saída;
2. `NLLLoss` aplicada à saída **pós-Softmax** (gradiente achatado);
3. 57–64% das unidades sigmoides saturadas;
4. `ModelCheckpoint(monitor="val/kappa")` preso em ~0, selecionando um checkpoint
   arbitrário (época 0 em vários runs).

**Geometria pós-GAP** (de `results/relu_vs_sigmoid_flatten.csv`, `mode=frozen`):

| métrica | eggs / protozoan / larvae | leitura |
|:--|:--|:--|
| `dead_chan_P` | 0,000 nos 3 | **encoder saudável**, nenhum canal morto |
| `effdim_P` (participation ratio) | **1,01 – 1,39** | o vetor 48-d é dominado por **uma única direção** |
| `cv_P` | 0,035 – 0,101 | baixa dispersão relativa entre canais |

`effdim_P` ≈ 1 é o achado geométrico central: depois do GAP, os 48 números colapsam sobre
essencialmente **um eixo de magnitude global**.

### 4.3 O que é simétrico entre os dois braços (hipóteses descartadas)

Verificado: mesmo `train` (JSONs conferidos), mesmo transform `_build_test(200)`, mesma
`imagenet_norm=True`, **sem augmentation em nenhum dos dois** (`V_train=1`), encoder **sem
BatchNorm e sem Dropout** (logo `eval()` vs `train()` é no-op), **sem scaler em nenhum dos
dois**, `class_weight=None` nos dois.

**Nenhuma dessas explica o gap.**

---

## 5. Evidências da literatura

> Referências verificadas por busca. As ressalvas estão explicitadas onde não houve
> confirmação por acesso direto ao texto.

### 5.1 A favor de "otimização" (e por que não basta)

- **Soudry, Hoffer, Nacson, Gunasekar, Srebro — *The Implicit Bias of Gradient Descent on
  Separable Data*, JMLR 19(70), 2018, arXiv:1710.10345.** GD sobre dados separáveis
  converge **na direção de margem máxima**, mas **logaritmicamente devagar** (~O(1/log t)).
  **Ressalva crítica: a teoria prevê melhora lenta, não platô duro.** Ela não explica um
  gap que persiste idêntico a 100% dos rótulos.
- Complementos: Lyu & Li, ICLR 2020 (arXiv:1906.05890) — estende a redes homogêneas;
  Nacson et al., AISTATS 2019 (arXiv:1803.01905) — taxas de convergência.
- **Rosset, Zhu & Hastie, NIPS 2003** — hinge e cross-entropy têm o **mesmo** limite de
  margem. Isto **enfraquece** o argumento popular de que "hinge é intrinsecamente melhor
  que CE".

### 5.2 Linear probing sobre features congeladas

- Alain & Bengio, arXiv:1610.01644 — probes lineares como instrumento de medida.
- **Razavian et al., CVPRW 2014, arXiv:1403.6382** — SVM linear sobre features CNN
  congeladas batendo sistemas ajustados. **Precedente direto** do fenômeno observado aqui.
- Tian et al., ECCV 2020, arXiv:2003.11539.
- **Kumar et al., ICLR 2022 — LP-FT, arXiv:2202.10054.** Mecanismo que explica a §6.4
  ("por que descongelar resolve").

### 5.3 Contra-evidência — a literatura que este relatório **deve** citar honestamente

- **He et al., MAE, CVPR 2022, arXiv:2111.06377** — literalmente: *"linear probing misses
  the opportunity of pursuing strong but non-linear features"*, com cabeça **não-linear
  melhor** (73,5 → 79,1).
- **Zaiem et al., 2024, arXiv:2308.14456** — cabeças de probing maiores **mudam o ranking**
  dos modelos avaliados.

**Leitura:** a literatura de SSL reporta cabeças não-lineares **empatando ou ganhando,
nunca perdendo 20 pontos de κ** sobre o mesmo embedding. Isto é evidência **contra** uma
explicação puramente representacional e **a favor** de artefato de pipeline — exatamente
o que a §2 estabelece.

### 5.4 Margem com poucas amostras

Bartlett & Shawe-Taylor 1999 (MIT Press); Koltchinskii & Panchenko, *Ann. Statist.* 30(1),
2002; Grønlund, Kamma & Larsen, ICML 2020. Explicam o gap em 1–25%, **não** o gap
persistente a 100%.

### 5.5 Escala, condicionamento e regime

- **LeCun, Bottou, Orr & Müller, *Efficient BackProp*, LNCS 7700** — entradas não centradas
  → Hessiano mal condicionado → platô de treino. Relevante porque **nenhum** dos braços usa
  scaler.
- Ioffe & Szegedy, ICML 2015; documentação do scikit-learn ("MLP is sensitive to feature
  scaling").
- Belkin et al., PNAS 116(32), 2019 — bias-variance / double descent.

### 5.6 Colapso e desbalanceamento

- **Papyan, Han & Donoho, PNAS 117(40), 2020**; Yang et al., NeurIPS 2022 — sob
  desbalanceamento, classes minoritárias colapsam entre si. **Mas isso degradaria os dois
  braços igualmente** → não explica um gap assimétrico.
- **Cao et al., LDAM, NeurIPS 2019, arXiv:1906.07413** — CE com `class_weight=None` é
  dominada pela classe majoritária.

### 5.7 OvO vs softmax

- **Fürnkranz, JMLR 2, 2002** — cada subproblema binário é mais simples e usa apenas uma
  fração dos dados, **localmente balanceada**.
- Hsu & Lin, IEEE TNN 13(2), 2002. Rifkin & Klautau, JMLR 5, 2004 (posição contrária).

### 5.8 SVM vs MLP em small data

Goddard & Shamir, IEEE Access 10, 2022; Fernández-Delgado et al., JMLR 15, 2014;
Grinsztajn et al., NeurIPS 2022, arXiv:2207.08815.

---

## 6. Diagnóstico — três hipóteses

### H1 — Artefato experimental (o embedding é diferente) — **PESO: dominante, ~100% do gap em pct ≥ 50**

**A favor:**

- Assimetria **literal no código**: `flatten` vs `AdaptiveAvgPool2d(1)`, fator 576×.
- O teto linear **convexo** em 48-d fica 0,17–0,25 abaixo do SVM em eggs/protozoan e
  **empatado em larvae**.
- `test_time` de 2170 s é **incompatível** com d = 48 e compatível com d ≈ 2,8·10⁴ (§2.2).
- O headroom **prediz** o gap em 3/3 datasets (§2.1).
- Recortes determinísticos `Resize(200) + CenterCrop(200)` → **alinhamento espacial** entre
  imagens → **templates posicionais** aprendíveis pelo flatten e **apagados** pelo GAP.
- `effdim_P` ≈ 1 após o GAP (§4.2).
- O próprio repositório **já documentava a assimetria**: `statistics/tools/compute_cost.md:47`.

**Contra:**

- A objeção "o notebook ancestral achatava nos dois braços" — **refutada** pelo argumento
  d ≫ n (§2.3).
- Ausência de prova documental do `flim_mlp` — **LACUNA-1**, real e não fechada.

### H2 — Fenômeno de otimização — **PESO: nulo no braço do Felipe em pct alto; real e aditivo em pct ≤ 25; TOTAL na cabeça do repo**

É preciso separar três sub-casos, que vinham sendo tratados como um só.

**H2-a — MLP do Felipe: NÃO é otimização.** Ela entrega **+0,049 (eggs) e +0,128
(protozoan) ACIMA do probe linear convexo** no mesmo embedding. Uma cabeça que **supera o
ótimo global linear das próprias features** não pode ser descrita como "falhou em encontrar
a solução do SVM". O gap de generalização dela é de apenas **0,010–0,020**: isto é
**saturação informacional**, não subajuste.

**H2-b — Regime de poucos rótulos: contribuição aditiva real.** Decompondo
`Δκ(pct) = Δκ_embedding + Δκ_amostra(pct)`, com `Δκ_embedding` fixado no valor a pct=100:

| pct | eggs Δκ | excesso sobre 0,230 | protozoan Δκ | excesso sobre 0,231 |
|--:|--:|--:|--:|--:|
| 5 | 0,511 | **+0,281** | 0,409 | **+0,178** |
| 25 | 0,391 | +0,161 | 0,329 | +0,098 |
| 50 | 0,261 | +0,031 | 0,268 | +0,037 |
| 75 | 0,218 | −0,012 | 0,245 | +0,014 |
| 100 | 0,230 | 0,000 | 0,231 | 0,000 |

O termo de amostra **decai monotonicamente até zero em pct ≈ 50** nos dois datasets — e em
pct=5 chega a ser **maior** que o termo de embedding. **É exatamente aqui que Soudry et al.
e Bartlett & Shawe-Taylor se aplicam**, e só aqui.

**H2-c — Cabeça do repo (`sigmoid2l`): H2 em estado puro e catastrófica.** Acurácia de
teste no nível do acaso sobre features que suportam 0,770 (§4.2). **Mas esta não é a cabeça
das tabelas do artigo** — é um bug separado, e **não deve ser usada como evidência sobre a
pergunta central**.

### H3 — Fenômeno de representação (a capacidade extra da MLP só adiciona variância) — **REFUTADA**

**Contra, ponto a ponto:**

- A MLP **supera** o probe linear no mesmo embedding (+0,049 / +0,128). Se fosse penalidade
  de variância, ficaria **abaixo**.
- **O gap de generalização é 0,010–0,020.** Variância excessiva se manifesta como
  train ≫ test. Não acontece. **Este é o argumento mais forte contra H3, sem ambiguidade.**
- `dead_chan_P = 0,000` — o encoder está saudável, não há colapso de features.
- A literatura (MAE, Zaiem) reporta cabeças não-lineares **ganhando**, nunca perdendo 20
  pontos de κ.

### 6.1 Por que Δκ ≫ Δacc?

**(i) Causa geométrica, dominante.** `effdim_P` ≈ 1: depois do GAP, os 48-d colapsam sobre
**uma única direção**. Nesse espaço só é possível *ordenar* as classes ao longo de um eixo.
A classe majoritária (**65,4% em eggs, 59,7% em protozoan**) fica separável nas
extremidades; as minoritárias (**1,56% em eggs, 0,82% em protozoan**) colapsam umas sobre
as outras no meio. A acurácia é dominada pela majoritária e cai pouco; **κ desconta o acaso
e despenca**. O flatten preserva as direções de baixa variância — que é onde as
minoritárias vivem.

**(ii) Causa de perda, secundária.** `class_weight=None` + CE dominada pela majoritária
(Cao et al.) do lado da MLP, contra 36 (eggs) / 21 (protozoan) subproblemas binários
**localmente balanceados** no OvO (Fürnkranz) e um hinge que zera para os pontos fáceis, do
lado do SVM.

### 6.2 Por que larvae não tem o efeito?

O teto pooled em larvae já é **0,968** — **não há headroom espacial a recuperar**. É o
**controle negativo perfeito** de H1.

**Ressalva honesta:** larvae é simultaneamente **binário** *e* **saturado no pooled**.
Sozinho, ele **não separa** H1 do mecanismo OvO (com 2 classes há 1 hiperplano e a vantagem
combinatória do OvO desaparece junto). Quem separa é o **probe 48-d em protozoan**: 7
classes, teto 0,653, e esse teto **já explica todo o gap** sem invocar OvO.

O que larvae **mata** definitivamente:

- **neural collapse como causa** — degradaria os dois braços igualmente;
- **desbalanceamento *per se*** — larvae tem minoritária de 12,7% e nada acontece.

### 6.3 Por que o gap não encolhe a 100%?

Isto **mata margem-em-poucos-dados e subtreino como causa dominante** — ambos deveriam
desaparecer com rótulos suficientes. Favorece um **teto estrutural fixo, independente da
quantidade de rótulos** — propriedade que **só o embedding** tem.

### 6.4 Por que descongelar o encoder resolve?

Porque descongelar **remove a restrição imposta pelo GAP**: o encoder passa a poder
codificar **em magnitude de canal** aquilo que antes vivia **em posição** — exatamente a
informação que o pooling destruía.

Evidência:

- train κ = **1,000** no unfrozen, contra 0,665 (eggs) / 0,636 (cistos) no frozen;
- o unfrozen **dispara early stopping** (106–241 épocas), enquanto o frozen **bate o teto de
  300** sem convergir.

Mecanismo descrito por **Kumar et al., LP-FT, ICLR 2022**.

---

## 7. Lacunas de dados

**LACUNA-1 — central e irredutível hoje.** O pipeline `flim_mlp` / `svm` do Felipe **não é
versionado**. Os CSVs têm **zero colunas de hiperparâmetro**;
`data/reports_felipe/flim_mlp/checkpoints/` tem **216 subdiretórios, todos vazios**. A
conclusão "MLP = pooled, SVM = flatten" é **inferência tripla** (d ≫ n + posição relativa ao
teto linear convexo + headroom prediz o gap 3/3) — forte, mas **indireta**.

> **Consequência para o artigo: a comparação deve ser declarada como heterogênea no
> extrator de features, em vez de afirmar "mesmo embedding".**

**LACUNA-2 — protocolo do probe.** `probe_acc_P` é 5-fold CV **dentro do test set**
(~2046 amostras por fold, contra 2555 do train split) com `LogisticRegression(C=1.0)`, não
`SVC(C=1e2)`. Regularização e conjunto diferem do braço do Felipe. Não muda a ordem de
grandeza, mas o teste **T1** elimina a ressalva por completo.

**LACUNA-3 — params da MLP do Felipe.** Os "46.601 params"
(`A_reports/july/check_sanity.md:182`) **não são rastreáveis**: o número é exatamente
`MLPHead(48, 256, 9)`, ou seja foi **inferido** assumindo que ele usa a mesma `MLPHead`
deste repo. **Tratar como hipótese, não como fato.**

**LACUNA-4 — convergência do SVM.** Há 43 `ConvergenceWarning` em
`logs/svm_frozen_20260612_144945.log`, mas **todos do braço proj-1280**; nenhum do caminho
FLIM. A convergência do SVM no caminho FLIM é **desconhecida**.

**LACUNA-5 — cobertura dos braços "ours".** `sigmoid2l` só cobre pct {1, 75}; `relu2l` só
{5, 75} e só unfrozen; **não existe braço frozen do `relu2l`**; **não existe M2 a 100%**.

**LACUNA-6 — checkpoints.** Não existe checkpoint da época 99 de nenhum run: `last.ckpt`
tem a mesma época do `best_kappa.ckpt` nos 94 runs.

**LACUNA-7 (literatura) — posicionável como contribuição.** Não há estudo publicado de
OvO-vs-softmax em regime "muitas classes, poucos exemplos por classe" sobre features
profundas congeladas, nem SVM-vs-MLP em microscopia de parasitos com embeddings congelados.

**Divergência não resolvida.** A auditoria reporta train acc da `sigmoid2l` = frequência da
majoritária (0,655), enquanto o **test** acc da mesma config é 0,109–0,176. Com priors de
train/test praticamente idênticos, um preditor constante daria 0,654 nos dois. Isto sugere
que o `best_kappa.ckpt` avaliado **não é** o de fim de treino. Relevante apenas para H2-c.

---

## 8. Próximo experimento

### T0 — **JÁ EXECUTADO, custo zero**

O teste que faltava já existia nos dados: `probe_acc_P` de
`results/relu_vs_sigmoid_flatten.csv`. Resultado **0,770 / 0,653 / 0,968**.

Ele **bifurca a conclusão**: prova **otimização** para a cabeça do repo (H2-c) **e**
**limite informacional do embedding** para o braço do Felipe (H1). **Não é "um ou outro" —
são dois artefatos distintos, em dois pipelines distintos.**

### T1 — PRIORIDADE 1: SVM linear, flatten vs pooled

SVM linear sobre o mapa **achatado** contra o mesmo SVM sobre o **pooled**, mesmo encoder
FLIM-init congelado, mesmo split. **Fecha a LACUNA-1 sem depender do Felipe.**

Hoje isto **não existe**: `results/svm_softplus2l_results.csv` tem
`train_freeze_encoder=False` nas 18 linhas, e
[svm_classification_flim.py:73-91](../../../../src/evaluate/svm_classification_flim.py#L73-L91)
(`_EncoderProbe`) **força `AdaptiveAvgPool2d(1)`** — só entrega o lado pooled.

Script standalone (~40 linhas) que carrega
`data/to_mateus/model/ch24_32_48_a0.5_f5/eggs/train1`, extrai `conv3`, e roda
`SVC(kernel="linear", C=1e2, max_iter=10000)` duas vezes:
**(a)** `flatten(1)` → 27.648-d; **(b)** `AdaptiveAvgPool2d(1)` → 48-d.

> **Predição falsificável: (a) ≈ 0,93–0,94 e (b) ≈ 0,77–0,80.**
> **Se (b) vier ≈ 0,93, H1 cai inteira e H2 volta ao centro.**

Rodar `larvae` primeiro como smoke test (fit de ~6 s). Considerar `LinearSVC` ou
`SGDClassifier(loss='hinge')` para cortar o custo do braço (a).

### T2 — PRIORIDADE 2: escada de resolução espacial

Mesmo script, variando `AdaptiveAvgPool2d(k)` com k = 1 (48-d), 2 (192-d), 4 (768-d),
8 (3072-d) e 24 (= flatten).

Predição: curva **monótona 0,77 → 0,93** em eggs/protozoan e **plana** em larvae.

Isto transforma o diagnóstico em **contribuição positiva** — *"quanta resolução espacial um
classificador linear precisa sobre features FLIM"* — e provavelmente mostra que **k=4
recupera quase tudo por 1/36 do custo**, resposta direta e prática aos 2170 s de `test_time`
da §2.2.

### T3 — PRIORIDADE 3: consertar a cabeça do repo (H2-c)

Um knob por vez, sobre os 48-d congelados:
(i) `wd` 5e-2 → 0;
(ii) param group `no_decay` para os biases;
(iii) logits crus + `CrossEntropyLoss` no lugar de Softmax + `NLLLoss`;
(iv) ReLU no lugar da Sigmoid **oculta**.

**Alvo a bater: 0,770** (o probe). Minutos por variante com as features pré-extraídas.
**Não afeta as conclusões sobre as tabelas do artigo** — faça para não publicar uma cabeça
quebrada.

### Não vale a pena

Tunar `C` / `class_weight` do SVM, ou a MLP do Felipe. **O teto de 0,770 do probe convexo
limita qualquer método linear em 48-d, e a MLP do Felipe já está acima dele.**

---

## 9. Correções a relatórios anteriores

**Corrige** a leitura da §5.4 de [`report_2026-07-22.md`](report_2026-07-22.md), que
afirmava:

> *"Mesmo encoder, mesma informação disponível — a diferença está no classificador, não na
> representação."*

**A informação disponível NÃO é a mesma.** O SVM recebe ~27.648-d e a MLP recebe 48-d. **A
frase deve ser revista** — no relatório e em qualquer texto derivado dela.

**Qualifica** o §B1 de `review_code_22_jul_2026.md`. A conclusão de que "as features são
ótimas; a camada sigmoide oculta é o que quebra" está **correta para a cabeça deste repo**
(`sigmoid2l`, H2-c), mas **não explica** o gap SVM-vs-MLP-do-Felipe — que é o gap das
tabelas do artigo. São dois problemas diferentes que estavam sendo tratados como um.

---

## 10. Resumo executivo

1. **A premissa da pergunta é falsa.** SVM e MLP não veem o mesmo tensor: flatten
   (~27.648-d) vs GAP (48-d), fator 576×.
2. **O teto é informacional, não de otimização.** O probe linear **convexo** sobre os 48-d
   dá 0,770 / 0,653 / 0,968 — e a MLP do Felipe está **acima** dele em 2 de 3 datasets.
3. **O headroom prediz o gap em 3/3 datasets**, incluindo o zero de larvae. Predição, não
   ajuste.
4. **H3 (variância) está refutada** pelo gap de generalização de 0,010–0,020.
5. **H2 é real, mas só em pct ≤ 25** (termo aditivo que zera em pct ≈ 50) e, separadamente,
   é **total** na cabeça deste repo — que **não é** a cabeça do artigo.
6. **LACUNA-1 continua aberta**: a inferência é tripla e forte, mas indireta. **T1 a fecha.**

---

## 11. Fontes

| Braço / evidência | Arquivo |
|:--|:--|
| SVM + FLIM (frozen) | `data/reports_felipe/svm/*.csv` |
| FLIM + MLP (ReLU, Felipe) — frozen e unfrozen | `data/reports_felipe/flim_mlp/*.csv` |
| Probe linear 48-d, `effdim_P`, `dead_chan_P`, `cv_P` | `results/relu_vs_sigmoid_flatten.csv` (`mode=frozen`) |
| Gerador do probe | [tools/analyze_sigmoid_saturation.py](../../../../tools/analyze_sigmoid_saturation.py) (linhas 106-123) |
| Confirmação independente dos valores do probe | `analise_achatamento_relu_sigmoid.md` §3 |
| Config do SVM | [src/utils/evaluate.py](../../../../src/utils/evaluate.py) (linhas 236-252) |
| `_EncoderProbe` (força GAP) | [src/evaluate/svm_classification_flim.py](../../../../src/evaluate/svm_classification_flim.py) (linhas 73-91) |
| Braço de destilação com `StandardScaler` | [src/evaluate/svm_distill_with_projection.py](../../../../src/evaluate/svm_distill_with_projection.py) (linhas 197-200) |
| `MLPHead` (código morto) | [src/models/models.py](../../../../src/models/models.py) (linhas 304-326) |
| `TwoLayerSigmoidHead` | [src/models/models.py](../../../../src/models/models.py) (linhas 351-389) |
| Treino da cabeça do repo | [src/modules/classification_flim_module.py](../../../../src/modules/classification_flim_module.py) |
| Default `average="macro"` das métricas "ours" | [src/metrics/classification.py](../../../../src/metrics/classification.py) (linhas 31-62) |
| FLIM + MLP (Sigmoid, ours) | `results/sigmoid2l_test_results.csv` |
| FLIM + MLP (ReLU, ours) | `results/relu2l_test_results.csv` |
| SVM sobre `softplus2l` (só unfrozen) | `results/svm_softplus2l_results.csv` |
| `ConvergenceWarning` (todos do braço proj-1280) | `logs/svm_frozen_20260612_144945.log` |
| Assimetria flatten/GAP já documentada | `statistics/tools/compute_cost.md:47` |
| Params inferidos da MLP do Felipe | `A_reports/july/check_sanity.md:182` |
| Pesos FLIM-init | `data/to_mateus/model/ch24_32_48_a0.5_f5/<dataset>/train<N>` |
