# Análise LeJEPA SSL — Dataset helminth-eggs

## 1. Tabela Completa: SVM Kappa por init × pct

Médias sobre 3–4 splits (n_splits indicado). Inicialização **ijepa** refere-se ao teacher I-JEPA puro (embeddings ViT, sem convNN).

| pct | flim | he | random | xavier | trunc_normal | ijepa (teacher) |
|-----|---------|-------|--------|--------|-------------|-----------------|
| 1%  | 0.0623 (±0.078) | 0.2481 (±0.034) | 0.2193 (±0.081) | 0.1687 (±0.027) | 0.2366 (±0.022) | 0.6611 (±0.028) |
| 5%  | 0.0558 (±0.067) | 0.4434 (±0.071) | 0.4883 (±0.230) | 0.3687 (±0.018) | 0.3361 (±0.176) | 0.8636 (±0.014) |
| 25% | 0.7331 (±0.045) | 0.3588 (±0.104) | 0.4004 (±0.124) | 0.4014 (±0.043) | 0.3502 (±0.129) | 0.9171 (±0.014) |
| 50% | 0.7687 (±0.024) | 0.2865 (±0.075) | 0.3273 (±0.109) | 0.3229 (±0.060) | 0.3137 (±0.075) | 0.9397 (±0.008) |
| 75% | 0.8107 (±0.018) | 0.3894 (±0.063) | 0.3145 (±0.034) | 0.3397 (±0.099) | 0.3654 (±0.117) | 0.9457 (±0.008) |
| 100%| 0.7309 (±0.195) | 0.3152 (±0.055) | 0.3401 (±0.044) | 0.3128 (±0.051) | 0.3142 (±0.058) | 0.9572 (±0.007) |

**Observações:**
- `flim` é a única inicialização que entrega kappa competitivo com o SVM (>0.70 a partir de pct=25%).
- `he`, `random`, `xavier`, `trunc_normal` ficam presos na faixa 0.28–0.49 independente do pct (plateau).
- O teacher ijepa supera qualquer inicialização LeJEPA em todos os pcts — gap mínimo de 0.13 (pct=75%, flim) e máximo de 0.81 (pct=5%, flim).

---

## 2. Tabela MLP: freeze vs unfreeze por init e pct

### MLP_freeze (backbone congelado, apenas cabeça linear treinada)

| pct | flim | he | random | xavier | trunc_normal |
|-----|------|----|----|-----|-------|
| 1%  | 0.000 | 0.000 | 0.010 | 0.004 | 0.000 |
| 5%  | 0.000 | 0.009 | 0.000 | 0.000 | 0.000 |
| 25% | 0.709 | 0.000 | 0.060 | 0.082 | 0.075 |
| 50% | 0.776 | 0.094 | 0.138 | 0.127 | 0.147 |
| 75% | 0.802 | 0.150 | 0.053 | 0.234 | 0.266 |
| 100%| 0.716 | 0.236 | 0.244 | 0.194 | 0.211 |

### MLP_unfreeze (backbone + cabeça fine-tuning end-to-end)

| pct | flim | he | random | xavier | trunc_normal |
|-----|------|----|----|-----|-------|
| 1%  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5%  | 0.000 | 0.000 | 0.000 | 0.170 | 0.000 |
| 25% | **0.832** | 0.784 | 0.766 | 0.762 | 0.746 |
| 50% | **0.838** | 0.826 | 0.825 | 0.810 | 0.845 |
| 75% | **0.885** | 0.849 | 0.847 | 0.853 | 0.851 |
| 100%| **0.929** | 0.896 | 0.892 | 0.893 | 0.900 |

**Síntese MLP:**
- MLP_freeze comporta-se como avaliação de features puras: só `flim` produz representations linearly separable (kappa >0.70 a partir de pct=25%). Todas as demais inicializações colapsam — kappa próximo a zero — exceto pequenas exceções estocásticas.
- MLP_unfreeze reverte o colapso: com pct≥25%, todas as inicializações convergem para kappa 0.76–0.93 porque o gradiente de classificação reajusta toda a rede.
- O melhor resultado absoluto do LeJEPA no dataset eggs é **MLP_unfreeze flim pct=100%: kappa = 0.9289** (±0.010), aproximando-se do teacher I-JEPA (0.9572, gap = 0.028).

---

## 3. Comparação: LeJEPA vs FLIM Supervisionado vs Teacher I-JEPA

Referências tiradas de `unified_svm_comparison.csv` (método SVM_FLIM = FLIM supervisionado normalizado):

| pct | LeJEPA SVM (flim) | LeJEPA MLP_unfreeze (flim) | FLIM Supervisionado (SVM) | Teacher I-JEPA (SVM) |
|-----|------------------|---------------------------|--------------------------|---------------------|
| 1%  | 0.062 | 0.000 | 0.128 | 0.661 |
| 5%  | 0.056 | 0.000 | 0.585 | 0.864 |
| 25% | 0.733 | 0.832 | 0.805 | 0.917 |
| 50% | 0.769 | 0.838 | 0.842 | 0.940 |
| 75% | 0.811 | 0.885 | 0.870 | 0.946 |
| 100%| 0.731 | **0.929** | 0.885 | 0.957 |

**Destaques:**
- Em pct=25%, o LeJEPA MLP_unfreeze (0.832) já supera levemente o FLIM supervisionado via SVM (0.805).
- Em pct=100%, o LeJEPA MLP_unfreeze (0.929) supera o FLIM supervisionado SVM (0.885) em +4.4 pontos de kappa.
- O teacher I-JEPA (SVM linear, sem fine-tuning) supera o LeJEPA MLP_unfreeze em todos os pcts, revelando que a destilação para o backbone FLIM-CNN perde informação expressiva.
- O gap teacher → LeJEPA_MLP_unfreeze_flim cai de 0.661 (pct=1%) para 0.028 (pct=100%), indicando que com dados suficientes o backbone LeJEPA recupera boa parte da capacidade do teacher.

---

## 4. Anomalia: flim pior em pct=1–5%, melhor em pct≥25%

### Dados brutos da inversão

| pct | flim SVM kappa | he SVM kappa | random SVM kappa | Melhor não-flim |
|-----|---------------|--------------|-----------------|-----------------|
| 1%  | **0.062** | 0.248 | 0.219 | he (0.248) |
| 5%  | **0.056** | 0.443 | 0.488 | random (0.488) |
| 25% | **0.733** | 0.359 | 0.400 | flim |
| 50% | **0.769** | 0.287 | 0.327 | flim |
| 75% | **0.811** | 0.389 | 0.315 | flim |
| 100%| **0.731** | 0.315 | 0.340 | flim |

### Interpretação

A inversão em pct=1–5% revela um fenômeno de **underfitting de representação localizada por domínio**:

**Hipótese principal — Viés de inicialização FLIM vs. generalização de features pré-treinadas:**

A inicialização FLIM deriva filtros aprendidos com supervisão fraca e morfologia biológica dos ovos (helminth eggs). Esses filtros são altamente especializados em capturar texturas finas e contornos de estruturas parasitárias. Quando o pré-treinamento SSL (LeJEPA) ajusta o backbone a partir dessa inicialização, a estrutura latente torna-se altamente adaptada ao espaço de features morfológicas de eggs. Com poucos dados (pct=1–5%), o SVM não tem exemplos suficientes para separar as classes nesse espaço de alta curvatura.

Em contraste, `he` e `random` inicializam o backbone com pesos genéricos. O pré-treinamento SSL sobre o conjunto de eggs é o único sinal, produzindo uma representação latente mais "planificada" e linearmente separável com poucos pontos. O hiperplano do SVM encontra separadores mais fáceis nesse espaço menos especializado.

**Evidência de suporte:**
- O std de kappa para flim em pct=100% é 0.195, o maior de todos — indicando colapso intermitente em alguns splits. Isso sugere que a superfície de loss do SVM em cima dos features FLIM-inicializados é mais acidentada.
- MLP_freeze flim a pct=1–5% também colapsa (kappa=0.000–0.000), enquanto MLP_freeze random tem kappa=0.010 a pct=1%. Confirmando que o problema não é só do SVM.
- MLP_unfreeze flim a pct=1–5% colapsa igualmente (kappa=0.000), demonstrando que com apenas 1–5% dos dados o backbone não tem gradiente suficiente para ajustar sua superfície de features, independentemente da inicialização.

**Threshold de virada:** A partir de pct=25% (~1/4 do dataset), o SVM finalmente tem pontos suficientes para traçar hiperplanos que aproveitam a riqueza representacional gerada pela inicialização FLIM. A vantagem inverte completamente e flim domina por +0.33–0.37 pontos sobre he/random/xavier.

---

## 5. Por que MLP_freeze colapsa para eggs mas não para protozoan?

### Dados comparativos MLP_freeze (kappa, pct=25%–100%)

| dataset | init | pct=25% | pct=50% | pct=75% | pct=100% |
|---------|------|---------|---------|---------|---------|
| **eggs**     | flim  | 0.709 | 0.776 | 0.802 | 0.716 |
| **eggs**     | he    | **0.000** | 0.094 | 0.150 | 0.236 |
| **eggs**     | random| 0.060 | 0.138 | 0.053 | 0.244 |
| **protozoan**| flim  | 0.320 | 0.401 | 0.396 | 0.487 |
| **protozoan**| he    | 0.197 | 0.088 | 0.067 | 0.019 |
| **protozoan**| random| 0.085 | 0.035 | 0.056 | 0.096 |

Nota: para protozoan, todas as inicializações também são fracas no MLP_freeze; a diferença não é que "não colapsam" — mas que o colapso para eggs é mais severo e mais uniforme.

### Raízes do colapso mais severo em eggs

**1. Número de classes (9 vs 7 vs 2):**
O dataset eggs possui **9 classes** (espécies de ovos helmínticos), enquanto protozoan tem 7 e larvae 2. Com backbone congelado, a cabeça linear de 9 classes tem mais fronteiras a separar no mesmo espaço de features. Features não estruturadas (he, random, xavier) produzem representações sem organização inter-classe suficiente para 9 separadores lineares simultâneos. Em protozoan (7 classes), o mesmo problema existe mas é mais leve.

**2. Intra-class variability morfológica:**
Os ovos de diferentes espécies helmínticas têm alta variabilidade intra-classe (ovos do mesmo tipo podem variar em tamanho e coloração por estágio de maturação) e baixa inter-class variability entre certas espécies. Um backbone congelado sem inicialização informada não organiza o espaço latente de forma que a cabeça linear consiga explorar essas diferenças sutis. A inicialização FLIM mitiga isso porque seus filtros já capturam texturas e gradientes relevantes para distinguir espécies.

**3. Contraste com MLP_unfreeze:**
Quando o backbone é desbloqueado (MLP_unfreeze), o gradiente de cross-entropy com 9 classes é suficiente para reestruturar o espaço de features — daí todas as inicializações convergirem para kappa >0.76 a pct≥25%. Isso confirma que o problema é a rigidez do backbone congelado, não limitação intrínseca do modelo.

**4. Por que flim não colapsa no MLP_freeze:**
Os filtros FLIM foram aprendidos por saliência supervisionada sobre imagens biológicas do mesmo domínio. Mesmo sem fine-tuning SSL, eles já produzem uma representação que preserva informação discriminativa sobre os 9 tipos de ovos. O LeJEPA com inicialização FLIM parte desse estado e, durante o pré-treinamento self-supervised, preserva parte dessa estrutura.

---

## 6. Data Efficiency: pct mínimo para kappa > 0.70 por método

| Método | init | Primeiro pct com kappa > 0.70 | kappa nesse pct |
|--------|------|-------------------------------|-----------------|
| SVM LeJEPA | flim | **25%** | 0.733 |
| SVM LeJEPA | he | nunca alcança | max 0.443 (pct=5%) |
| SVM LeJEPA | random | nunca alcança | max 0.488 (pct=5%) |
| SVM LeJEPA | xavier | nunca alcança | max 0.401 (pct=25%) |
| SVM LeJEPA | trunc_normal | nunca alcança | max 0.365 (pct=75%) |
| MLP_freeze | flim | **25%** | 0.709 |
| MLP_freeze | he | nunca alcança | max 0.236 (pct=100%) |
| MLP_freeze | random | nunca alcança | max 0.244 (pct=100%) |
| MLP_unfreeze | flim | **25%** | 0.832 |
| MLP_unfreeze | he | **25%** | 0.784 |
| MLP_unfreeze | random | **25%** | 0.766 |
| MLP_unfreeze | xavier | **25%** | 0.762 |
| MLP_unfreeze | trunc_normal | **25%** | 0.746 |
| Teacher I-JEPA SVM | ijepa | **1%** | 0.661 → **5%**: 0.864 |
| FLIM Supervisionado SVM | flim | **25%** | 0.805 |

**Conclusões de data efficiency:**

1. **MLP_unfreeze é o único modo onde todas as inicializações atingem kappa >0.70**, e o fazem ao mesmo threshold de pct=25%.

2. **SVM e MLP_freeze só atingem kappa >0.70 com inicialização flim** (pct=25%). Para qualquer outra inicialização, o SVM LeJEPA é inutilizável como classificador linear.

3. **O teacher I-JEPA é dramaticamente mais eficiente em dados:** já com pct=5% entrega kappa=0.864, enquanto o melhor LeJEPA no mesmo pct é 0.000 (MLP_unfreeze) ou 0.056 (SVM flim). A destilação para o backbone CNN perde completamente a capacidade de few-shot do ViT teacher.

4. **Pct=25% é o threshold crítico** para o dataset eggs em qualquer variante de LeJEPA que não seja teacher. Abaixo disso (1–5%), nenhum método LeJEPA (com nenhuma inicialização) supera kappa=0.50, e a maioria colapsa para kappa≈0.

5. **Comparação com FLIM supervisionado:** O FLIM supervisionado (SVM) também atinge >0.70 apenas em pct=25% (0.805). Portanto, o LeJEPA MLP_unfreeze flim é competitivo com o FLIM supervisionado a partir de pct=25% e o supera em pct=100% (0.929 vs 0.885 = +4.4 pontos kappa).

---

## Resumo Executivo

| Dimensão | Achado |
|----------|--------|
| Melhor configuração absoluta | MLP_unfreeze, init=flim, pct=100%: kappa=0.929 |
| Gap para teacher | 0.028 kappa (teacher=0.957) |
| Pct mínimo para kappa>0.70 | 25% (MLP_unfreeze, todas as inits; SVM apenas flim) |
| Anomalia low-shot | flim piora em pct=1–5% por espaço latente mais especializado/curvado |
| Colapso MLP_freeze | eggs (9 classes) mais suscetível que protozoan (7) por maior demanda de fronteiras lineares |
| Inicialização crítica | flim é condição necessária para SVM funcionar; para MLP_unfreeze, todas funcionam ≥pct=25% |
| Vantagem sobre FLIM sup. | LeJEPA MLP_unfreeze supera FLIM supervisionado em pct=100% (+4.4 kappa), mas é inferior em pct=5% |
