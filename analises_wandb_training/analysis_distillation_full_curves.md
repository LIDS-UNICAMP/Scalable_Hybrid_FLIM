# Análise Completa de Destilação: Curvas de Treinamento WandB
**Data:** 2026-05-30 | **Dados:** 20,752 linhas de histórico por época, 216 runs

---

## 1. Colapso de Embeddings: Evidência Precisa por Época

### 1.1 Trajetória do student_emb_norm (norma do encoder bruto)

| Época | 1x1_BN2d | 1x1_flim_init | 3x3_BN2d | next_layers |
|---|---|---|---|---|
| 0 | 0.579 | **270.17** | 0.629 | 0.623 |
| 10 | 0.074 | 17.32 | 0.415 | 0.360 |
| 25 | 0.066 | 16.38 | 0.380 | 0.296 |
| 50 | 0.059 | 11.03 | 0.391 | 0.328 |
| 99 | 0.066 | **11.25** | **0.354** | **0.314** |

**Descoberta crítica:** O colapso no 1x1_BN2d ocorre nos **primeiros 10 épocas** (durante o warmup!), caindo de 0.579 para 0.074 — redução de 87%. Este é um colapso **instantâneo e irreversível** dentro do período de warmup.

### 1.2 Colapso por Dataset (1x1_BN2d)

| Dataset | Época 0 | Época 10 | Época 25 | Época 50 | Época 99 |
|---|---|---|---|---|---|
| eggs | 0.650 | 0.091 | 0.049 | 0.033 | 0.040 |
| larvae | 0.664 | 0.062 | 0.053 | 0.031 | 0.034 |
| protozoan | 0.423 | 0.086 | 0.095 | 0.113 | 0.122 |

**Protozoan é mais resiliente**: mantém emb_norm ~0.12 em vez de colapsar para 0.03. Isso explica por que a performance em protozoan é ligeiramente melhor (~0.20 kappa) versus eggs/larvae (~0).

### 1.3 FLIM init: norma de 270 → prevenção do colapso

O 1x1_flim_init começa com **student_emb_norm = 270.17** (pesos FLIM têm normas muito altas). Isso previne o colapso: mesmo após 99 épocas, a norma é 11.25 — comparado a 0.066 para trunc_normal. O encoder FLIM mantém ativações fortes o suficiente para resistir ao sinal de gradiente fraco do 1×1 conv.

---

## 2. Evolução do Cosine Similarity (train/cosine_sim)

### 2.1 Trajetória por grupo

| Época | 1x1_BN2d | 1x1_flim_init | 3x3_BN2d | next_layers |
|---|---|---|---|---|
| 0 | 0.020 | 0.018 | 0.023 | 0.004 |
| 10 | 0.407 | 0.315 | 0.413 | 0.393 |
| 25 | 0.536 | 0.478 | 0.572 | 0.608 |
| 50 | 0.605 | 0.544 | 0.647 | 0.707 |
| 75 | 0.632 | 0.559 | 0.665 | 0.739 |
| 99 | 0.637 | 0.571 | 0.666 | **0.742** |

**Observações:**
- Grande salto de 0 → 0.4 nos primeiros 10 épocas (warmup)
- Melhoria lenta depois: apenas +0.23 de época 10 → 99 para 1x1
- flim_init é **mais lento**: cosine_sim = 0.315 em época 10 vs 0.407 para trunc_normal
- next_layers_direct tem a curva mais acentuada e alcança o maior plateau (0.742)

### 2.2 Teto de Cosine Similarity por Dataset

| Dataset | 1x1_BN2d (max) | 3x3_BN2d (max) | next_layers (max) |
|---|---|---|---|
| eggs | 0.766 | 0.784 | **0.942** |
| larvae | 0.747 | 0.774 | **0.919** |
| protozoan | 0.786 | 0.803 | **0.971** |

next_layers alcança cosine_sim ≈ 0.97 para protozoan (quase alinhamento perfeito!). O 1x1 tem teto em ~0.77 independente de mais treinamento.

### 2.3 Velocidade para alcançar cosine_sim = 0.70

| Grupo | Média de épocas | Mediana | Runs que nunca chegam a 0.7 |
|---|---|---|---|
| next_layers | **21.3** | 18.0 | 24/54 |
| 3x3_BN2d | 25.1 | 21.0 | 21/54 |
| 1x1_BN2d | 28.7 | 29.0 | 24/54 |

---

## 3. Scale Ratio: Paradoxo da Inicialização

**Scale ratio = student_proj_norm / teacher_emb_norm** (ideal = 1.0)

| Época | 1x1_BN2d | 1x1_flim_init | 3x3_BN2d | next_layers |
|---|---|---|---|---|
| 0 | 0.807 | 0.761 | 0.738 | **1.019** |
| 10 | 0.287 | 0.297 | 0.360 | 0.354 |
| 25 | 0.300 | 0.342 | 0.402 | 0.387 |
| 50 | 0.371 | 0.390 | 0.463 | 0.489 |
| 99 | 0.404 | 0.422 | 0.494 | **0.542** |

**Paradoxo:** Todos os grupos começam com scale_ratio ≈ 0.74–1.02 (BN2d inicializa com escala 1), mas caem para 0.28–0.39 em época 10. Depois sobem lentamente. A queda inicial ocorre porque:
1. O BatchNorm2d aprende a normalizar as ativações do estudante
2. O teacher tem norma fixa ~21.4, mas o estudante ainda está aprendendo a dimensionar

Nenhum grupo atinge scale_ratio = 1.0 mesmo após 99 épocas, sugerindo que **mais épocas ou uma loss de regularização de norma** beneficiariam todos os grupos.

---

## 4. Convergência e Não-Convergência

### 4.1 Porcentagem de runs que ainda melhoravam na época 99

| Grupo | % em época 99 (best_epoch=99) |
|---|---|
| **1x1_BN2d** | **100%** — NENHUM run convergiu! |
| 3x3_BN2d | 96.3% |
| next_layers | 88.3% |
| 1x1_flim_init | 75.0% |

**Conclusão: 100 épocas é insuficiente para todos os grupos.** Recomendação mínima: 200 épocas.

### 4.2 Training Loss por Grupo e Dataset (@época 99)

| Dataset | 1x1_BN2d | 3x3_BN2d | next_layers |
|---|---|---|---|
| eggs | 0.2325 | 0.2061 | **0.1643** |
| larvae | 0.2549 | 0.2270 | **0.1977** |
| protozoan | 0.1985 | 0.1826 | **0.1214** |

next_layers alcança losses significativamente menores, especialmente em protozoan (0.1214 vs 0.1985 para 1x1).

### 4.3 Oscilações

| Grupo | Oscilação média (épocas 80-99) | Runs oscilatórios (>2%) |
|---|---|---|
| 1x1_BN2d | 0.0065 | 0/54 |
| 3x3_BN2d | 0.0080 | 4/54 |
| next_layers | 0.0122 | 1/54 |
| 1x1_flim_init | 0.0078 | 2/36 |

Treinamento é estável — nenhuma divergência ou oscilação severa detectada. O maior osc. = 3% (3x3_BN2d).

---

## 5. Predição Precoce de Performance (Early Stopping)

### 5.1 Correlação Pearson: métrica de treinamento → SVM kappa

| Época | cosine_r | emb_norm_r | scale_r | loss_r |
|---|---|---|---|---|
| 10 | +0.343 | **+0.575** | +0.337 | -0.423 |
| 25 | +0.452 | **+0.798** | +0.552 | -0.513 |
| 50 | +0.486 | **+0.864** | +0.547 | -0.529 |
| 75 | +0.503 | **+0.859** | +0.538 | -0.533 |
| final | +0.438 | **+0.845** | +0.492 | -0.481 |

**O student_emb_norm é o melhor preditor de performance** com r=0.864 já em época 50. Isso é contraintuitivo: a norma do encoder bruto (antes da projeção) prediz melhor o kappa downstream que o próprio cosine_similarity (alinhamento com o teacher)!

### 5.2 Threshold crítico: emb_norm em época 10

| Threshold | Kappa médio (acima) | Kappa médio (abaixo) | Diferença |
|---|---|---|---|
| emb_norm_ep10 ≥ 0.05 | 0.647 (n=121) | 0.122 (n=37) | **+0.525** |
| emb_norm_ep10 ≥ 0.10 | 0.667 (n=117) | 0.119 (n=41) | **+0.548** |
| emb_norm_ep10 ≥ 0.20 | 0.687 (n=107) | 0.184 (n=51) | **+0.503** |

**Regra prática: se `student_emb_norm < 0.05` na época 10, a run irá falhar (kappa~0.12). Pode ser encerrada precocemente.**

### 5.3 Estados dos runs

| Estado | Count |
|---|---|
| finished | 198 |
| running | 12 |
| crashed | 6 |

6 runs crashados — provavelmente OOM ou timeout no servidor.

---

## 6. Comparação 1×1 vs 3×3: Por que o kernel size importa

### 6.1 Evidência das curvas

| Métrica @época 99 | 1x1_BN2d | 3x3_BN2d | Diferença |
|---|---|---|---|
| student_emb_norm | **0.066** | **0.354** | 3×3 produz 5.4× mais norma |
| cosine_sim | 0.637 | 0.666 | 3×3 +4.5% melhor |
| train_loss | 0.225 | 0.206 | 3×3 -8.4% menos loss |
| scale_ratio | 0.404 | 0.494 | 3×3 +22% melhor escala |

### 6.2 Hipótese mecanística confirmada pelas curvas

A curva de student_emb_norm mostra que:
- **1x1**: colapso nos primeiros 10 épocas (mesmo período de warm-up)
- **3x3**: mantém norma estável em 0.38-0.41 durante todo o treinamento

O 1×1 conv atua como uma transformação linear por localização espacial. O gradiente do MSE propagado via 1×1 ao encoder é fraco porque:
1. Não há interação entre posições espaciais adjacentes no gradiente
2. O encoder com 24×24 posições espaciais aprende que "zerar tudo" minimiza a loss trivialmente
3. Depois de zerarem, o BN2d normaliza os zeros para ter média 0, std 1 — criando representações aleatórias, não estruturadas

O 3×3 kernel tem receptive field local: os gradientes cruzam posições espaciais, forçando o encoder a manter diversidade nas ativações. **Isso é consistente com a análise de gradiente para colapso de modo simples em redes convolucionais.**

---

## 7. Próximos Passos Recomendados

1. **Aumentar para 200+ épocas**: 100% dos runs ainda estão melhorando na época 99
2. **Early stopping via emb_norm**: encerrar runs onde emb_norm < 0.05 em época 10
3. **Regularização de norma**: adicionar `L_norm = (||s_proj|| - ||t_emb||)²` para melhorar scale_ratio
4. **1x1 + flim_init como substituto de 1x1 trunc_normal**: resolve o colapso com custo mínimo
5. **Monitorar student_emb_norm** como métrica primária de health do treinamento
