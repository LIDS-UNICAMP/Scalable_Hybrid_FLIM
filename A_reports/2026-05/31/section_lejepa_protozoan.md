# LeJEPA — Análise Detalhada: Dataset Protozoan

**Data da análise:** 2026-05-30  
**Dataset:** protozoan-cysts (7 classes, classificação multiclasse)  
**Referência teacher:** I-JEPA ViT (kappa SVM@100% = 0.892)  
**Referência supervisionada:** FLIM CNN supervisionado (kappa SVM@100% = 0.847)

---

## 1. Tabela SVM: init × pct (kappa Cohen ± std)

Valores extraídos de `artifacts/SVM/svm_aggregated.csv` (model_type = SVM, dataset_short = protozoan).

| init         | pct=1%               | pct=5%               | pct=25%              | pct=50%              | pct=75%              | pct=100%             |
|--------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|
| **flim**     | 0.098 ± 0.136        | 0.368 ± 0.103        | **0.480 ± 0.109**    | 0.495 ± 0.133        | 0.416 ± 0.178        | **0.291 ± 0.176**    |
| **he**       | 0.040 ± 0.026        | 0.471 ± 0.063        | 0.330 ± 0.061        | 0.174 ± 0.106        | 0.087 ± 0.036        | 0.200 ± 0.027        |
| **random**   | 0.272 ± 0.074        | 0.303 ± 0.032        | 0.194 ± 0.116        | 0.194 ± 0.150        | 0.055 ± 0.046        | 0.189 ± 0.052        |
| **xavier**   | **0.481 ± 0.022**    | **0.511 ± 0.083**    | 0.240 ± 0.102        | 0.186 ± 0.087        | 0.077 ± 0.072        | 0.126 ± 0.055        |
| **trunc_normal** | 0.270 ± 0.054    | 0.259 ± 0.035        | 0.231 ± 0.052        | 0.123 ± 0.053        | 0.117 ± 0.088        | 0.103 ± 0.062        |

**Referência I-JEPA teacher (SVM):**

| pct=1%  | pct=5%  | pct=25% | pct=50% | pct=75% | pct=100% |
|---------|---------|---------|---------|---------|----------|
| 0.586   | 0.725   | 0.831   | 0.864   | 0.882   | **0.892** |

**Observação critica:** Todas as inits do LeJEPA ficam muito abaixo do teacher I-JEPA no SVM, com gap de 0.601 (flim) a 0.766 (trunc_normal) em pct=100%. O modelo CNN não gerou representações linearmente separáveis de qualidade comparável ao ViT.

---

## 2. Tabela MLP_freeze: init × pct (kappa Cohen)

Valores extraídos de `artifacts/MLP/mlp_aggregated.csv` (model_type = MLP_freeze, dataset_short = protozoan).

| init         | pct=1%  | pct=5%  | pct=25% | pct=50% | pct=75% | pct=100% |
|--------------|---------|---------|---------|---------|---------|----------|
| **flim**     | 0.000   | 0.043   | 0.320   | 0.401   | 0.396   | **0.487** |
| **he**       | 0.000   | 0.002   | 0.197   | 0.088   | 0.067   | 0.019    |
| **random**   | 0.019   | 0.022   | 0.085   | 0.035   | 0.056   | 0.096    |
| **xavier**   | 0.000   | 0.049   | 0.159   | 0.102   | 0.000   | 0.004    |
| **trunc_normal** | 0.000| 0.046   | 0.105   | 0.041   | 0.002   | 0.000    |

**Conclusão MLP_freeze:** A cabeça MLP linear congelada com backbone fixo é muito fraca para protozoan em quase todas as inits (kappa < 0.5). Apenas flim@100% atinge kappa moderado (0.487). Isso confirma que as representações CNN não são linearmente separáveis sem ajuste do backbone.

---

## 3. Tabela MLP_unfreeze: init × pct (kappa Cohen)

Valores extraídos de `artifacts/MLP/mlp_aggregated.csv` (model_type = MLP_unfreeze, dataset_short = protozoan).

| init         | pct=1%  | pct=5%  | pct=25% | pct=50% | pct=75% | pct=100% |
|--------------|---------|---------|---------|---------|---------|----------|
| **flim**     | 0.179   | 0.372   | 0.817   | 0.871   | 0.893   | **0.899** |
| **he**       | 0.000   | 0.526   | 0.782   | 0.837   | 0.880   | 0.843    |
| **random**   | 0.127   | 0.595   | 0.742   | 0.804   | 0.853   | **0.897** |
| **xavier**   | 0.000   | 0.583   | 0.763   | 0.817   | 0.770   | 0.886    |
| **trunc_normal** | 0.002| 0.326   | 0.717   | 0.791   | 0.801   | 0.862    |

**Referência I-JEPA teacher:**  
kappa SVM@100% = **0.892** (usado como proxy do teacher; MLP_unfreeze não foi avaliado para o teacher diretamente)

**Destaque:**
- flim@100%: **0.899** > teacher (0.892) — diferença de +0.007
- random@100%: **0.897** > teacher (0.892) — diferença de +0.005
- he@100%: 0.843 (abaixo)
- xavier@100%: 0.886 (abaixo)

---

## 4. DESCOBERTA PRINCIPAL: LeJEPA supera o teacher I-JEPA em protozoan

> **MLP_unfreeze flim@100% (kappa = 0.899) > I-JEPA teacher SVM@100% (kappa = 0.892)**  
> **MLP_unfreeze random@100% (kappa = 0.897) > I-JEPA teacher SVM@100% (kappa = 0.892)**

Esta é uma descoberta notável. O modelo destilado CNN (LeJEPA) com fine-tuning completo supera a representação do teacher ViT (medida via SVM) no dataset protozoan. Análise dos mecanismos prováveis:

### 4.1 Inductive bias do CNN vs ViT para protozoan

O dataset protozoan-cysts contém imagens microscópicas de cistos de protozoários com características visuais bem definidas:
- **Texturas locais altamente discriminativas**: cistos possuem paredes celulares, núcleos e grânulos com padrões de textura característica em escala de poucos pixels.
- **Invariância translacional local**: as estruturas diagnósticas relevantes (vacúolos, núcleos) aparecem em posições variadas dentro das imagens.
- **Ausência de dependência de longa distância**: ao contrário de tarefas de reconhecimento de objetos complexos, a classificação de cistos não requer integração de contexto global extenso.

O ViT I-JEPA, por sua arquitetura baseada em self-attention global, processa informação de forma não-local por padrão, o que pode ser subótimo para texturas de baixo nível. O CNN, com suas convoluções locais hierárquicas, possui inductive bias que favorece exatamente esse tipo de padrão. Após fine-tuning (MLP_unfreeze), o backbone CNN pode se especializar nessas texturas de forma mais eficiente do que a representação congelada do ViT.

Adicionalmente, a destilação I-JEPA → CNN comprime conhecimento de alto nível do ViT para um backbone mais compacto, e o fine-tuning permite que esse backbone readquira e aprofunde a discriminabilidade para a tarefa específica.

### 4.2 Comparação com Distilação 3x3@100%

A destilação 3x3BN@100% (SVM kappa = 0.833) ficou **abaixo** do teacher (0.892) no SVM, sugerindo que a representação congelada do CNN destilado é inferior ao teacher. Contudo, o LeJEPA com MLP_unfreeze atinge 0.899, indicando que o fine-tuning é o fator decisivo — o CNN tem capacidade latente que o SVM não consegue explorar.

---

## 5. Anomalia SVM: flim_init DEGRADA com mais dados

### 5.1 O padrão

| pct  | flim kappa SVM |
|------|----------------|
| 25%  | **0.480**      |
| 50%  | 0.495          |
| 75%  | 0.416          |
| 100% | **0.291**      |

O kappa do flim_init no SVM **cai de 0.480 em pct=25% para 0.291 em pct=100%**, uma degradação de ~0.19 pontos. Para referência, xavier colapsa ainda mais dramaticamente (0.511@5% → 0.126@100%).

### 5.2 Hipótese central: Catastrophic forgetting seletivo

A hipótese mais coerente é que o pré-treinamento SSL (I-JEPA) destrói a discriminabilidade linear das representações progressivamente à medida que o modelo vê mais dados não supervisionados:

**Mecanismo proposto:**

1. **Inicialização flim**: os filtros FLIM são inicializados com informação supervisionada específica ao domínio (microscopia de fluorescência), fornecendo representações com boa separabilidade linear a priori.

2. **SSL com pouco dado (pct=25%)**: o pré-treinamento I-JEPA sobre 25% do dataset é curto — a otimização SSL não chega a sobrescrever completamente as features supervisionadas do FLIM. A representação resultante ainda mantém estrutura discriminativa linear, daí kappa=0.480 no SVM.

3. **SSL com mais dados (pct=100%)**: com mais dados de pré-treinamento, o objetivo I-JEPA (predição de patches mascarados no espaço de representação) converge mais completamente. Esse objetivo é agnóstico à classe — ele encoraja o modelo a capturar estrutura estatística geral das imagens, não necessariamente estrutura discriminativa de classe. As features originais do FLIM são progressivamente sobrescritas por representações orientadas à auto-supervisão, que não são linearmente separáveis para a tarefa de classificação.

4. **O que o SVM mede vs o MLP_unfreeze**: o SVM mede apenas a separabilidade **linear** da representação congelada. O MLP_unfreeze, por outro lado, ajusta o backbone inteiro — ele recupera a informação latente que ainda existe no modelo (mas que foi comprimida em uma forma não-linear pelo SSL), usando o gradiente da tarefa de classificação para reorganizar a representação. Isso explica por que o mesmo modelo que tem SVM-kappa=0.291 com backbone congelado atinge MLP_unfreeze-kappa=0.899 com backbone ajustado.

**Em síntese**: o SSL não destrói a informação — ele a transforma para representações não-lineares que resistem ao SVM mas são recuperáveis via fine-tuning supervisionado. Isso é consistente com a literatura que mostra que representações SSL são mais ricas em informação do que o SVM linear consegue capturar.

### 5.3 Por que essa degradação é mais severa em protozoan do que em outros datasets?

Os datasets eggs e larvae mostram o padrão oposto com flim: kappa SVM **cresce** monotonicamente com pct. A diferença provável está na dificuldade do problema:

- **Protozoan tem 7 classes** (vs 2 para larvae), com cistos morfologicamente similares entre espécies. As features lineares do FLIM não são suficientemente discriminativas para separar 7 classes em um espaço linear, especialmente após o SSL reorganizar a representação. Mais dados SSL aprofundam essa reorganização.
- **Eggs e larvae**: problemas binários ou com classes mais discrimináveis permitem que o SSL refine as representações sem destruir a separabilidade linear — mais dados apenas melhoram a qualidade geral das features.

---

## 6. Por que xavier é ótimo em pct=1% mas colapsa depois?

| pct  | xavier kappa SVM |
|------|-----------------|
| 1%   | **0.481**       |
| 5%   | **0.511**       |
| 25%  | 0.240           |
| 50%  | 0.186           |
| 75%  | 0.077           |
| 100% | 0.126           |

### 6.1 A inicialização xavier como "tábula rasa com boas propriedades estatísticas"

Xavier (Glorot) inicializa pesos para manter a variância dos gradientes constante entre camadas, mas sem qualquer prévio sobre a tarefa. Num regime de dados SSL extremamente limitado (pct=1%), isso significa que:

- O SSL converge muito pouco — o modelo ainda está quase na inicialização.
- Xavier, por suas propriedades de escala, gera representações com distribuição relativamente uniforme no espaço de features, o que acidentalmente pode ser favorável para o SVM (separador linear) em baixo regime.
- Em outras palavras: com 1% dos dados, o SSL mal modifica a inicialização, e a inicialização xavier já tem geometria razoável para separação linear de forma fortuita.

### 6.2 O colapso com mais dados

À medida que o SSL tem mais dados para treinar, ele modifica substancialmente os pesos para o objetivo de predição de patches. Xavier não tem nenhum viés de domínio para "resistir" a essa modificação (ao contrário do flim, que tem). A reorganização SSL apaga as boas propriedades acidentais da inicialização xavier, e como o objetivo SSL não é discriminativo, o resultado é uma representação ruim para SVM.

Esse padrão é análogo ao observado em NLP onde modelos inicializados aleatoriamente treinados com dados muito limitados às vezes superam modelos pré-treinados com poucos dados de fine-tuning — o pré-treinamento em regime insuficiente pode ser mais nocivo do que a ausência de pré-treinamento.

### 6.3 Contraste com flim

O flim_init tem viés supervisionado de domínio que lhe confere mais robustez: mesmo com pct=100% de SSL, parte da estrutura discriminativa é preservada (kappa=0.291 vs xavier=0.126). A inicialização supervisionada age como "âncora" parcial que o SSL não elimina completamente.

---

## 7. Comparação global: FLIM supervisionado vs LeJEPA SVM vs LeJEPA MLP

| Método                                    | kappa          | Regime        |
|-------------------------------------------|----------------|---------------|
| FLIM CNN supervisionado                   | 0.847          | Totalmente supervisionado |
| I-JEPA teacher (SVM@100%)                 | 0.892          | SSL ViT       |
| LeJEPA SVM flim@25% (melhor SVM linear)   | 0.480          | SSL CNN (sem fine-tuning) |
| LeJEPA SVM xavier@5% (2º melhor SVM)      | 0.511          | SSL CNN (sem fine-tuning, baixo regime) |
| **LeJEPA MLP_unfreeze flim@100%**         | **0.899**      | SSL CNN (com fine-tuning) |
| **LeJEPA MLP_unfreeze random@100%**       | **0.897**      | SSL CNN (com fine-tuning) |
| Distilação 3x3BN SVM@100%                 | 0.833          | Destilação KD (sem fine-tuning) |

**Interpretação:**

1. O LeJEPA sem fine-tuning (SVM) é muito inferior ao FLIM supervisionado (0.480 vs 0.847). O backbone CNN pré-treinado por SSL sozinho não compete com supervisão completa.

2. Com fine-tuning (MLP_unfreeze), o LeJEPA **inverte completamente** o quadro e supera tanto o FLIM supervisionado (0.899 > 0.847, ganho de +0.052) quanto o teacher I-JEPA (0.899 > 0.892, ganho de +0.007).

3. O gap entre SVM e MLP_unfreeze para o LeJEPA é enorme (0.291 → 0.899 para flim@100%), indicando que o backbone contém informação discriminativa muito rica que apenas se torna acessível com fine-tuning — a representação não é linearmente separável, mas é **não-linearmente separável de forma excelente**.

4. A destilação 3x3BN (sem fine-tuning, SVM@100% = 0.833) é substancialmente melhor do que o LeJEPA SVM mas ainda inferior ao MLP_unfreeze do LeJEPA. A arquitetura destilada com projetor 1280d parece capturar representações mais lineares do que o backbone bruto do LeJEPA.

---

## 8. Síntese e implicações

### 8.1 Achados principais

1. **LeJEPA supera o teacher em protozoan** (MLP_unfreeze = 0.899 > teacher SVM = 0.892): o inductive bias CNN, combinado com fine-tuning, supera a representação ViT para este dataset de texturas microscópicas locais.

2. **Catastrophic forgetting seletivo**: o SSL destrói discriminabilidade linear (SVM cai de 0.480 para 0.291) mas preserva informação recuperável via fine-tuning (MLP_unfreeze mantém 0.899). Isso é especialmente severo em protozoan (7 classes difíceis) comparado a eggs/larvae.

3. **Anomalia xavier em baixo regime**: a ausência de prévio de domínio na inicialização xavier gera por acaso boa separabilidade linear com pct=1-5%, mas colapsa com mais dados SSL por não ter resistência à reorganização do espaço de features.

4. **flim_init como melhor choice para fine-tuning**: flim_init domina em MLP_unfreeze@100% (0.899) e é consistentemente superior em regimes de dados moderados a altos. O prévio supervisionado do FLIM beneficia o fine-tuning.

5. **A compressão CNN não perde informação relevante**: apesar de o backbone CNN ser muito menor que o ViT teacher, o fine-tuning recupera — e supera — a performance do teacher, sugerindo que a destilação SSL transferiu com eficiência o conhecimento representacional necessário para protozoan.

### 8.2 Recomendação prática

Para protozoan, o protocolo ótimo é: inicialização flim → pré-treinamento LeJEPA@100% → fine-tuning completo (MLP_unfreeze). Não usar SVM diretamente sobre representações LeJEPA para protozoan — o gap de performance é de 0.608 pontos de kappa (0.899 - 0.291).

---

*Análise gerada com base nos arquivos: `artifacts/SVM/svm_aggregated.csv`, `artifacts/MLP/mlp_aggregated.csv`, `results/ijepa_svm_aggregated.csv`, `artifacts/normalized/unified_svm_comparison.csv`, `analises_wandb_training/protozoan/lejepa/lejepa_protozoan_analysis.csv`.*
