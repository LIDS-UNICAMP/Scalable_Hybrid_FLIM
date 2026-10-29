# Discovery híbrido SPiFiL — log de decisões

Branch `discovery_hybrid_spifil`. CNN, entrada LAB, treino só com decoder (sem rótulo), um eixo por candidato, crescer camada a camada.

## Alvo (I-JEPA a 50%, κ teste, média de 3 splits — report_results_only_spifil.md:209)

| dataset | I-JEPA | melhor SPiFiL | fonte | gap |
|---|---|---|---|---|
| eggs | 0.9397 | 0.9094 (flim_L3) | :206 | 0.030 |
| larvae | 0.9433 | 0.9164 (flim_avg_L3) | :347 | 0.027 |
| protozoan | 0.8636 | 0.8587 (bp3, grid4 backprop) | :195 | 0.005 |

Aceite: κ teste ≥ I-JEPA nos 3 datasets a 50% (média dos splits rodados). Teste só no comparativo final.

## Painel / plano

- Decisão (crescer, aceitar eixo): κ de **validação** do SVM oficial (`core/svm.py:143` + `core/metrics.py:46`). Hoje `eval/svm.py:694` fixa `_loader("test")` → ver pergunta Q1.
- Crescimento: adiciona Lk+1 enquanto κ val melhora (> +0.002); para na 1ª queda.
- Rodada de busca: 50%, split 1, 3 datasets. Confirmação (candidato que bate o melhor): splits 1-3.
- Parada imediata: NaN/Inf na loss ou OOM. κ val abaixo do melhor em 2 checagens seguidas → para o candidato.
- GPU 3 apenas, `max_per_gpu` 2.

## Eixos (ordem; convolução primeiro)

1. Convolução — dilation 2 em L4–L5 sem pooling (R6 h2: L4–L5 pioram no flatten; dilation aumenta campo receptivo sem perder resolução). Precisa de código: dilation fixo `[1,1]` em flim/spifil.py:624-651.
2. Kernel / largura — 7x7 em L1, ou 32/48/64 filtros (R6 h1 já fixou 5x5 24/32/48 como melhor conhecido).
3. Objetivo sem rótulo — MAE por superpixel no lugar da BCE pura (R6 h3: BCE satura em ~0,468).
4. Fine-tune ponta a ponta só com reconstrução após layerwise (R6 h4: backprop grid4 é o melhor em protozoan).
5. Embedding multi-escala concat (R6 h5).
6. BN (runs layerwise5_bn em andamento) e 7. geo_aug (layerwise5_aug em andamento) — reusar o resultado, não rodar de novo.
8. Profundidade adaptativa por pct — fora do escopo (só 50%).

## Entradas

### 2026-10-29 — plano inicial
- R1 (baseline reproduzível): runner growth layerwise, 3 camadas, 5x5, 24/32/48, DISF/50, paper_classes, sem trava N≤D, flatten. É o flim_L3 (melhor eggs) dentro do runner novo; mede-se κ val e teste para ancorar.
- R2 (1º eixo): R1 + L4 com dilation 2 e sem pooling, cresce enquanto κ val sobe.
- Porquê: o flim_L3 é o melhor SPiFiL reproduzível neste runner; convolução é o eixo 1 pedido; dilation é a hipótese R6 mais barata que mexe só em L4+.

### 2026-10-29 — respostas do usuário ao plano (substituem as leituras padrão acima)
- **Critério de decisão:** métricas de validação **sem classificação**, só do autoencoder (perda de reconstrução em val, `val_recon_loss`). O SVM no teste roda só de vez em quando, como checagem. Por isso não se mexe em `eval/svm*` nem se liga `_svm_probe`. Q1 fechada.
  - Crescer Lk+1 enquanto a val_recon_loss da nova camada fica abaixo da anterior. Comparar losses só dentro da mesma função de perda: quando o eixo 3 trocar a loss, a âncora passa a ser o SVM teste de checagem.
- **Épocas:** `max_epochs` 200 por camada (antes 1000). A BCE satura em ~0,468.
- **Splits:** os 3 splits em toda rodada (não só o split 1).
- **BN e geo_aug:** aproveitar os runs layerwise5_bn e layerwise5_aug do usuário (eixos 6 e 7), sem rodar de novo.
- **GPU:** 3 (a máquina não tem GPU 4).

### 2026-10-29 — usuário: SVM na validação para decidir e extrair as leis
- **Decisão:** κ do SVM oficial medido na **validação**, com o mesmo `fit_svm` e `compute_metrics` de sempre. A chave `eval_set: val` (default `test`) entra em `eval/svm.py` com aval do usuário. Para cada camada e cada eixo vale a regra: κ val sobe = fica. A val_recon_loss do autoencoder vira sinal auxiliar (escolha do best dentro da camada).
- **Leis de escala:** saem da tabela κ val por camada × eixo (largura, kernel, dilation, profundidade, loss) × dataset. O teste fica só para conferir de vez em quando e para o comparativo final com o I-JEPA.
