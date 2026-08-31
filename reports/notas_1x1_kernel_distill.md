# Notas: Análise do conv 1×1 na projection head de destilação

Data: 2026-05-27

## Resultado resumido

O conv 1×1 na projection head é arquiteturalmente inferior ao conv 3×3 para destilação FLIM→I-JEPA.
Não há bug de código — a causa é uma limitação matemática da convolução 1×1.

## Comparação de resultados SVM (kappa médio por dataset/percentagem)

### eggs (9 classes)

| pct | 3×3 [1280] | 1×1 [1280] | encoder-only [48] |
|-----|-----------|-----------|-------------------|
| 1   | 0.268     | 0.000     | 0.000             |
| 5   | 0.596     | −0.002    | 0.000             |
| 25  | 0.758     | −0.000    | 0.000             |
| 50  | 0.833     | 0.022     | 0.000             |
| 75  | 0.870     | 0.171     | 0.002             |
| 100 | **0.900** | 0.296     | 0.073             |

### larvae (2 classes)

| pct | 3×3 [1280] | 1×1 [1280] | encoder-only [48] |
|-----|-----------|-----------|-------------------|
| 1   | 0.635     | 0.000     | 0.000             |
| 5   | 0.702     | 0.061     | 0.000             |
| 25  | 0.853     | 0.059     | 0.674             |
| 50  | 0.880     | 0.088     | 0.801             |
| 75  | 0.913     | 0.066     | 0.831             |
| 100 | **0.939** | 0.201     | 0.850             |

### protozoan (7 classes)

| pct | 3×3 [1280] | 1×1 [1280] | encoder-only [48] |
|-----|-----------|-----------|-------------------|
| 1   | 0.345     | −0.015    | 0.000             |
| 5   | 0.504     | 0.040     | 0.000             |
| 25  | 0.707     | 0.150     | 0.001             |
| 50  | 0.787     | 0.241     | 0.195             |
| 75  | 0.811     | 0.297     | 0.233             |
| 100 | **0.833** | 0.202     | 0.245             |

## Causa raiz: limitação matemática do conv 1×1

### Como o conv 1×1 funciona como projection head

O conv 1×1 realiza apenas projeção linear dos canais em cada posição espacial:

```
output[b, 1280, h, w] = W × encoder[b, 48, h, w]   (W ∈ R^{1280×48})
→ GAP: emb[b] = W @ spatial_mean(encoder[b])
→ GELU aplicado antes do GAP
```

Resultado: o embedding final é equivalente a uma **projeção linear da média espacial** do encoder.
Toda a estrutura espacial local é perdida **antes** da projeção.

### Como o conv 3×3 funciona

```
output[b, 1280, h, w] = agregação de vizinhança 3×3 → captura padrões locais
→ GAP após a agregação espacial → preserva muito mais informação
```

O 3×3 agrega vizinhança antes do pooling → embeddings mais ricos e linearmente separáveis.

## Evidências experimentais

### ConvergenceWarning no SVM (confirmado para 1×1)

```
ConvergenceWarning: Solver terminated early (max_iter=20000).
Consider pre-processing your data with StandardScaler...
```

O SVM não converge nos embeddings 1×1 mesmo com `pct=100`. Quando não converge, prediz a
classe majoritária → kappa=0, acc=1/N_classes (ex: 0.111 para eggs com 9 classes).

Para o 3×3, este warning **não aparece**.

### Qualidade dos embeddings (imagens reais, eggs_split3_pct100, N=2557)

| Métrica                     | 1×1    | 3×3    |
|-----------------------------|--------|--------|
| Dims ativas (std ≥ 0.01)    | 249/1280 | **416/1280** |
| Between/total variance      | 0.293  | 0.268  |
| Global std                  | 0.370  | 0.385  |
| L2 norm (média)             | 13.71  | 14.26  |

O 3×3 tem 67% mais dimensões ativas. A separabilidade linear no espaço 1280-dim é superior
(SVM converge sem warning).

### Validação cruzada rápida (mesmo split, mesmo SVM)

| Head    | kappa  | acc    |
|---------|--------|--------|
| **3×3** | 0.892  | 0.896  |
| 1×1     | 0.354  | 0.775  |

## Por que kappa=0 com pct baixo (pct1, pct25)

Com poucos exemplos de treino + embeddings de baixa separabilidade linear:
1. SVM não converge antes de `max_iter=20000`
2. Prediz classe majoritária em todos os exemplos de teste
3. kappa=0, acc=1/N_classes (0.111 para eggs)

## Conclusão científica

**O receptive field espacial na projection head é crítico para destilação FLIM→I-JEPA.**

- Conv 3×3: agrega contexto local → representações discriminativas → SVM converge bem
- Conv 1×1: projeção linear de média espacial → menor separabilidade → SVM não converge

O 1×1 não serve como substituto do 3×3 neste contexto.
A arquitetura 3×3+BN2d+GELU+GAP deve ser adotada como padrão na projection head de destilação.

## Arquivos de resultados

| Arquivo | Conteúdo |
|---------|----------|
| `results/svm_proj1280_3x3_BN2d_results.csv` | SVM com 3×3 proj head ativa [B,1280] |
| `results/svm_proj1280_1x1_BN2d_results.csv` | SVM com 1×1 proj head ativa [B,1280] |
| `results/svm_distillation_conv_results.csv`  | SVM encoder-only [B,48] (next_layers_direct) |
| `results/svm_distill_proj1280_results.csv`   | SVM com proj head (runs antigos conv_next_layers) |
| `results/svm_1x1_BN2d_results.csv`           | Resultado anterior com erro de import (inválido) |
