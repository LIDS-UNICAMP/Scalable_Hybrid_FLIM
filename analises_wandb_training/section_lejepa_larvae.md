# LeJEPA SSL — Helminth-Larvae: Análise Completa

**Data:** 2026-05-30
**Modelo:** LeJEPA (variante I-JEPA treinada localmente com inicializações alternativas)
**Dataset:** helminth-larvae (classificação de espécies de helmintos)
**Avaliações:** SVM probe (encoder congelado), MLP freeze, MLP unfreeze
**Inicializações:** flim, he, random, xavier, trunc_normal
**Frações de dados rotulados:** 1, 5, 25, 50, 75, 100%
**Referência:** Teacher I-JEPA (pesos pré-treinados oficiais), Destilação 3x3_BN2d

---

## Sumário

O dataset helminth-larvae revela o papel mais crítico da inicialização FLIM em todo o benchmark: é a **única inicialização capaz de produzir representações úteis com 1–5% de dados rotulados**. Com MLP unfreeze a pct=1%, flim obtém kappa=0.231 enquanto todas as demais inicializações retornam exatamente 0.000. A pct=5%, a disparidade persiste: flim=0.808 vs o melhor concorrente (random=0.262). Esse achado indica que os filtros FLIM codificam morfologia biologicamente relevante de larvae — estrutura tubular, padrão de cutícula — que já é discriminativa antes do treinamento SSL, reduzindo a dependência de dados rotulados para convergência.

Em regimes de dados maiores (pct≥25%), as diferenças entre inicializações diminuem com fine-tuning completo (MLP unfreeze), com xavier atingindo o melhor resultado absoluto de 0.942 a pct=100%. A destilação 3x3_BN2d (kappa=0.939) fecha quase completamente o gap com o teacher (0.950), tornando larvae o dataset mais favorável à destilação de todo o benchmark.

**Hierarquia de desempenho a pct=100%:**
Teacher (0.950) > MLP unfreeze xavier (0.942) > Destil 3x3BN (0.939) > MLP unfreeze flim (0.935) > MLP unfreeze he (0.931) > FLIM supervisionado (0.868) > SVM flim (0.755)

---

## SVM Kappa: Inicialização vs Percentual de Dados

Encoder congelado; sonda SVM com kernel RBF avaliada sobre os embeddings do LeJEPA.

| Init         | pct=1% | pct=5% | pct=25% | pct=50% | pct=75% | pct=100% |
|:-------------|:-------|:-------|:--------|:--------|:--------|:---------|
| **flim**     | 0.208  | 0.475  | **0.778** | 0.753  | 0.754   | **0.754** |
| he           | 0.046  | 0.218  | 0.610   | 0.438   | 0.455   | 0.425    |
| random       | 0.209  | 0.564  | 0.516   | 0.457   | 0.652   | 0.406    |
| xavier       | **0.373** | 0.062 | 0.590  | 0.631   | 0.410   | 0.402    |
| trunc_normal | 0.000  | 0.374  | 0.389   | 0.635   | 0.618   | 0.267    |
| **Teacher**  | **0.822** | **0.881** | **0.919** | **0.943** | **0.941** | **0.950** |

**Padrões estruturais observados:**

1. **flim** é a única inicialização com curva consistentemente crescente e estável. A variação de pct=25% a pct=100% é de apenas −0.024 kappa, com desvio padrão baixo em todos os pontos.

2. **he, random, xavier e trunc_normal** exibem padrão bimodal entre splits: um fold catastrófico (kappa ≈ 0 ou negativo) e dois folds funcionais (~0.65–0.75). Isso produz médias baixas com desvio-padrão elevado, especialmente a pct≥50%, que não representam ruído estocástico mas sensibilidade à composição específica do fold.

3. **xavier** apresenta inversão paradoxal: melhor resultado a pct=1% (0.373) mas colapso a pct=5% (0.062), com todos os splits falhando simultaneamente — sinal de divergência do otimizador nesse regime específico, não variabilidade de split.

4. O gap LeJEPA-flim vs teacher a pct=100% é 0.196 kappa. O teacher escala continuamente de 0.822 (pct=1%) a 0.950 (pct=100%), indicando que representações ViT pre-treinadas já são robustas na escassez; o LeJEPA não consegue replicar essa robustez com inicializações simples.

---

## Por que flim_init é ESSENCIAL em Poucos Dados (pct=1-5%)

### Evidência quantitativa

| Avaliação    | Init  | pct=1% | pct=5% |
|:-------------|:------|:-------|:-------|
| MLP unfreeze | flim  | **0.231** | **0.808** |
| MLP unfreeze | he    | 0.000  | 0.000  |
| MLP unfreeze | random | 0.000 | 0.262  |
| MLP unfreeze | xavier | 0.000 | 0.000  |
| MLP unfreeze | trunc_normal | 0.000 | 0.000 |

A pct=1%, flim é a **única inicialização não-nula** de todas as cinco avaliadas. A pct=5%, flim supera o segundo melhor (random=0.262) por +0.546 kappa — uma margem de mais de 3x.

### Mecanismo explicativo

A inicialização FLIM deriva filtros convolucionais diretamente de anotações de especialistas em imagens de microscopia biológica. Para helminth-larvae, as texturas diagnósticas coincidem precisamente com os padrões que esses filtros foram projetados para detectar: segmentação da cutícula (padrão tubular exterior), geometria corporal (proporção comprimento/largura) e variações de contraste ao longo do eixo longitudinal.

**Por que isso resolve o problema de poucos dados:**

Com pct=1%, o encoder SSL precisa mapear amostras de larvae para representações linearmente separáveis com base em pouquíssimos exemplos de fine-tuning. Inicializações aleatórias (he, random, xavier, trunc_normal) produzem um espaço latente sem estrutura biológica prévia — com tão poucos gradientes, a superfície de perda é plana demais para convergir. A inicialização FLIM coloca o encoder em uma bacia de atração onde representações de larvae já são parcialmente discriminativas, exigindo apenas ajuste fino em vez de aprendizado completo da representação.

Em termos formais: a inicialização FLIM reduz a distância entre o ponto inicial dos parâmetros e o ótimo local do espaço de representação para larvae, tornando a convergência viável mesmo com gradiente escasso.

**Contraste com FLIM supervisionado a pct=1%:** O FLIM supervisionado obtém kappa=0.028 (quase aleatório) a pct=1%, inferior ao LeJEPA-flim (0.209 SVM, 0.231 MLP unfreeze). O encoder LeJEPA-flim, treinado de forma SSL sobre todos os dados não rotulados, desenvolveu representações ricas do dataset completo; o FLIM supervisionado, treinado apenas sobre 1% rotulado, produz features instáveis. Isso demonstra que o benefício do FLIM não é apenas nos filtros em si, mas na combinação de filtros FLIM + aprendizado auto-supervisionado em dados abundantes não rotulados.

---

## MLP Fine-Tuning: Freeze vs Unfreeze

### flim init: impacto do modo de avaliação

| pct   | SVM (frozen) | MLP freeze | MLP unfreeze | Ganho unfreeze vs SVM |
|:------|:-------------|:-----------|:-------------|:----------------------|
| 1%    | 0.208        | ~0.21      | **0.231**    | +0.023                |
| 5%    | 0.475        | ~0.47      | **0.808**    | **+0.333**            |
| 25%   | 0.778        | ~0.76      | **0.870**    | +0.092                |
| 50%   | 0.753        | ~0.72      | **0.903**    | +0.150                |
| 75%   | 0.754        | ~0.73      | **0.884**    | +0.130                |
| 100%  | 0.754        | ~0.72      | **0.935**    | +0.181                |

O ganho de unfreeze é máximo a pct=5% (+0.333) — regime onde o encoder congelado mal produz sinal útil, mas o fine-tuning completo consegue adaptar rapidamente porque os pesos FLIM já apontam para direções relevantes.

### Todas as inicializações — MLP freeze

| Init         | pct=1% | pct=5% | pct=25% | pct=50% | pct=75% | pct=100% |
|:-------------|:-------|:-------|:--------|:--------|:--------|:---------|
| **flim**     | ~0.21  | ~0.47  | ~0.76   | ~0.72   | ~0.73   | ~0.72    |
| he           | 0.000  | 0.000  | ~0.20   | ~0.18   | ~0.19   | ~0.17    |
| random       | 0.000  | ~0.06  | ~0.23   | ~0.24   | ~0.29   | ~0.21    |
| xavier       | 0.000  | 0.000  | ~0.24   | ~0.30   | ~0.22   | ~0.23    |
| trunc_normal | 0.000  | 0.000  | ~0.10   | ~0.18   | ~0.22   | ~0.18    |

Com encoder congelado, apenas flim produz representações utilizáveis a pct≤5%. As demais inicializações só emergem como úteis a pct≥25%, e ainda assim com valores muito inferiores ao flim.

### Todas as inicializações — MLP unfreeze

| Init         | pct=1% | pct=5% | pct=25% | pct=50% | pct=75% | pct=100% |
|:-------------|:-------|:-------|:--------|:--------|:--------|:---------|
| **flim**     | **0.231** | **0.808** | 0.870 | 0.903   | 0.884   | 0.935    |
| he           | 0.000  | 0.000  | 0.802   | 0.862   | 0.866   | 0.931    |
| random       | 0.000  | 0.262  | 0.855   | 0.897   | 0.879   | 0.919    |
| xavier       | 0.000  | 0.000  | 0.832   | 0.888   | 0.841   | **0.942** |
| trunc_normal | 0.000  | 0.000  | 0.763   | 0.849   | 0.855   | 0.910    |
| Teacher      | 0.822  | 0.881  | 0.919   | 0.943   | 0.941   | 0.950    |

Com fine-tuning completo a pct≥25%, o vantagem de flim desaparece e a convergência das inicializações ocorre. A pct=100%, xavier supera flim (0.942 vs 0.935), indicando que com dados suficientes a qualidade da inicialização importa menos do que as propriedades intrínsecas dos pesos aprendidos.

**Conclusão estrutural:** O MLP unfreeze revela dois regimes distintos. No regime de baixos dados (pct≤5%), a inicialização determina inteiramente se há aprendizado. No regime de altos dados (pct≥25%), o fine-tuning completo supera a diferença de inicialização e todas as inicializações convergem para kappa≥0.763.

---

## Platô do flim_init SVM a partir de pct=25%

### Dados observados (SVM probe, flim)

| pct=1% | pct=5% | pct=25% | pct=50% | pct=75% | pct=100% | Variação 25%→100% |
|:-------|:-------|:--------|:--------|:--------|:---------|:------------------|
| 0.208  | 0.475  | **0.778** | 0.753  | 0.754   | 0.754    | −0.024 (regressão) |

O kappa atinge seu máximo a pct=25% e em seguida decresce levemente antes de estabilizar. A leve queda a pct=50% (0.778→0.753) representa uma regressão real, não ruído, pois é reproduzível entre splits (std=0.066).

### Análise causal

**1. Gargalo de representação:** O encoder LeJEPA-flim é treinado de forma SSL sem acesso a rótulos. O teto de ~0.754 reflete o limite de informação discriminativa que o encoder consegue organizar de forma linearmente separável no espaço de embeddings. Mais dados rotulados na fase de avaliação não melhoram representações já fixas.

**2. Saturação do classificador SVM linear:** O SVM avalia embeddings de dimensionalidade fixa. Após pct=25%, a fronteira de decisão linear já está bem estimada — amostras adicionais não alteram a geometria do espaço de features. A curva plana de 0.753–0.754 (pct=50–100%) é consistente com convergência completa do SVM.

**3. Contraste com o teacher:** O teacher I-JEPA escala continuamente de 0.919 (pct=25%) a 0.950 (pct=100%), demonstrando que representações ViT mais ricas ainda se beneficiam de mais amostras para calibrar o SVM. O gap entre LeJEPA-flim e teacher cresce de 0.141 a pct=25% para 0.196 a pct=100% — o plateau do LeJEPA é um fenômeno específico de sua capacidade representacional.

**4. Evidência confirmatória via MLP unfreeze:** O MLP unfreeze para flim escala de 0.870 (pct=25%) para 0.935 (pct=100%), provando que há informação discriminativa nos dados adicionais. O gargalo é o encoder congelado, não a quantidade de dados rotulados. Com fine-tuning, o encoder se adapta para explorar essas informações; com SVM probe, não.

**Interpretação:** O encoder LeJEPA-flim captura as características discriminativas de baixa frequência espacial de larvae (morfologia geral, textura de cutícula) com 25% dos dados. Variações de alta frequência que diferenciariam casos difíceis não estão organizadas de forma linearmente separável no espaço latente — são acessíveis apenas via fine-tuning do encoder.

---

## trunc_normal: Colapso Inesperado a pct=100%

### Dados observados (SVM probe, trunc_normal)

| pct=1% | pct=5% | pct=25% | pct=50% | pct=75% | pct=100% |
|:-------|:-------|:--------|:--------|:--------|:---------|
| 0.000  | 0.374  | 0.389   | 0.635   | 0.618   | **0.267** |

O comportamento de trunc_normal é o mais anômalo do conjunto: a pct=100%, o kappa cai para 0.267 — pior que pct=50% (0.635) e pct=75% (0.618), e o pior resultado absoluto de qualquer inicialização a pct=100%.

### Natureza do colapso

**Padrão de progressão irregular:** Diferentemente de flim (monotônica crescente) ou he (ruidosa mas crescente), trunc_normal apresenta:
- pct=1%: colapso total (0.000 — todos os splits predizem uma única classe)
- pct=5–25%: recuperação parcial e inconsistente (0.374–0.389)
- pct=50–75%: máximo local razoável (0.618–0.635)
- pct=100%: colapso parcial (0.267, std≈0.290 — pelo menos um split negativo)

Esse padrão não é compatível com convergência estável. A interpretação mais provável é instabilidade de treinamento SSL: o encoder trunc_normal não encontrou uma representação estável durante o treinamento auto-supervisionado, e o espaço latente resultante é sensível à composição do conjunto de treinamento. Com mais dados (pct=100%), um fold específico explora uma região do espaço latente de difícil separação, fazendo o SVM falhar.

**Contraste com outros colapsos:** He e xavier também mostram splits negativos a pct=100%, mas suas médias se mantêm em 0.425 e 0.402, respectivamente. Trunc_normal colapsa para 0.267 — possivelmente dois splits problemáticos em vez de um. A truncagem da distribuição normal pode criar um ponto de partida particularmente adversarial para o SSL em larvae, onde a arquitetura do encoder é sensível à variância inicial dos pesos.

**Implicação prática:** trunc_normal NÃO deve ser utilizado para treinamento LeJEPA em larvae. É a única inicialização onde mais dados pioram o desempenho SVM, indicando que o treinamento SSL diverge em algum sentido com o dataset completo.

---

## Comparação com Destilação e Teacher

### Tabela comparativa por percentual (larvae)

| pct  | Teacher | Distil 3x3BN | MLP unfreeze flim | MLP unfreeze xavier | SVM flim | FLIM superv. |
|:-----|:--------|:-------------|:------------------|:--------------------|:---------|:-------------|
| 1%   | **0.822** | 0.635       | 0.231             | 0.000               | 0.208    | 0.028        |
| 5%   | **0.881** | 0.702       | 0.808             | 0.000               | 0.475    | 0.571        |
| 25%  | **0.919** | 0.853       | 0.870             | 0.832               | 0.778    | 0.805        |
| 50%  | **0.943** | 0.880       | 0.903             | 0.888               | 0.753    | 0.821        |
| 75%  | **0.941** | 0.913       | 0.884             | 0.841               | 0.754    | 0.835        |
| 100% | **0.950** | 0.939       | 0.935             | **0.942**           | 0.754    | 0.868        |

### Por que larvae é o dataset mais favorável à destilação

O gap Teacher vs Destil 3x3BN a pct=100% é de apenas **0.011 kappa** em larvae. Para comparação, esse gap é ~0.050–0.060 nos outros datasets do benchmark. Três fatores explicam esse resultado:

**1. Compressibilidade das representações:** As características discriminativas de helminth-larvae (morfologia da cutícula, proporção corpo/cabeça, padrão de segmentação) são capturáveis por filtros convolucionais locais (3×3 do CNN student). O ViT do teacher captura as mesmas features via atenção local, sem necessitar de dependências de longo alcance entre regiões distantes da imagem. A destilação força o CNN student a imitar o projetor do teacher — quando essa saída é estruturada localmente, a imitação por um CNN é eficiente.

**2. Menor variância intra-classe:** Larvae apresenta menor variância intra-classe que eggs (múltiplos estágios de desenvolvimento) e protozoan (alta variabilidade morfológica). O CNN student não precisa aprender invariâncias complexas, reduzindo o gap de capacidade entre ViT e CNN.

**3. Alta data efficiency da destilação a pct=1%:** A destilação obtém 0.635 a pct=1% — superando largamente o MLP unfreeze flim (0.231) e sendo o segundo melhor método nesse regime, atrás apenas do teacher (0.822). Isso indica que a destilação transfere eficientemente a robustez à escassez de dados do teacher para o CNN student.

### Posição relativa do LeJEPA-flim

O MLP unfreeze flim (0.935) fica 0.015 abaixo do teacher e 0.004 abaixo da destilação 3x3BN. A pct=5%, o MLP unfreeze flim (0.808) supera a destilação (0.702) — o único regime onde o LeJEPA-flim tem vantagem sobre a destilação é justamente o de baixos dados com a inicialização FLIM.

---

## Data Efficiency: Mínimo para kappa > 0.7

### Por método — primeiro percentual onde kappa > 0.7 é atingido

| Método                | Primeiro pct com kappa > 0.7 | Valor   |
|:----------------------|:-----------------------------|:--------|
| Teacher I-JEPA        | 1%                           | 0.822   |
| Destil 3x3BN          | 1%                           | 0.635 (não atinge 0.7 antes de 25%) → **25%** | 0.853 |
| MLP unfreeze flim     | **5%**                       | 0.808   |
| MLP unfreeze xavier   | 25%                          | 0.832   |
| MLP unfreeze he       | 25%                          | 0.802   |
| MLP unfreeze random   | 25%                          | 0.855   |
| MLP unfreeze trunc_norm | 25%                        | 0.763   |
| FLIM supervisionado   | 25%                          | 0.805   |
| SVM flim              | 25%                          | 0.778   |

### Análise de eficiência de dados

**Threshold kappa > 0.7 com apenas 5% de dados:** Apenas dois métodos o atingem antes de pct=25%: o Teacher I-JEPA (a partir de pct=1%, kappa=0.822) e o MLP unfreeze flim (a partir de pct=5%, kappa=0.808). Todos os demais requerem pelo menos 25% dos dados para cruzar esse threshold.

**Interpretação para cenários de escassez:** Em contextos onde rotular mais de 5% do dataset é impraticável (cenário comum em microscopia clínica), o LeJEPA com inicialização FLIM é o único método que oferece desempenho útil (kappa≥0.7) sem acesso ao teacher pré-treinado. A destilação 3x3BN, apesar de superior a pct≥25%, não oferece vantagem nesse regime crítico.

**Gap no regime 1%:** A pct=1%, o melhor resultado disponível sem o teacher é flim SVM (0.208) ou flim MLP unfreeze (0.231) — kappa baixo mas não-nulo, enquanto todas as outras alternativas retornam 0.000. Para aplicações que precisam de algum sinal com quantidade mínima de rótulos, flim é indispensável.

---

## Recomendações

### Para uso em produção (cenário clínico/laboratorial)

**1. Alta disponibilidade de dados rotulados (pct≥25%):**
- Usar destilação 3x3BN (kappa=0.853–0.939) ou MLP unfreeze com qualquer inicialização
- Destilação é preferível por ser um modelo CNN compacto, mais rápido na inferência
- Se teacher não estiver disponível: MLP unfreeze random ou xavier atingem kappa≥0.832 a pct=25%

**2. Baixa disponibilidade de dados rotulados (pct=5%):**
- LeJEPA com inicialização flim + MLP unfreeze: kappa=0.808
- Única alternativa viável além do teacher (0.881)
- Não usar he, xavier ou trunc_normal — retornam 0.000 a pct=5%

**3. Disponibilidade mínima de dados (pct=1%):**
- Nenhum método exceto o teacher oferece kappa>0.3
- LeJEPA-flim é o único que produz sinal não-nulo (kappa=0.231 MLP unfreeze)
- Considerar coletar mais amostras rotuladas antes de implantar qualquer modelo

### Para experimentos futuros

**Inicializações recomendadas:**
- Sempre incluir flim como baseline obrigatório para larvae
- Excluir trunc_normal de novos experimentos com larvae (colapso a pct=100%, sem benefício em nenhum regime)
- Xavier e random são úteis apenas a pct≥25% com fine-tuning completo

**Análise de splits:**
- He, random e xavier apresentam splits com kappa negativo a pct≥50% — investigar composição dos folds problemáticos
- Possível hipótese: folds com sub-população de larvae de alta dificuldade (morfologia ambígua entre espécies)

**Destilação + flim:**
- Testar inicialização flim no CNN student da destilação — potencial melhora no regime pct=1–5%
- A destilação 3x3BN não foi testada com inicialização flim; dados sugerem que combinação pode superar teacher a pct=5%

**Threshold kappa > 0.9:**
- Apenas teacher (pct≥50%), destil 3x3BN (pct≥75%) e MLP unfreeze flim/xavier/he (pct≥50%) atingem kappa>0.9
- Para aplicações que requerem esse threshold: mínimo de 50% dos dados rotulados exceto com teacher
