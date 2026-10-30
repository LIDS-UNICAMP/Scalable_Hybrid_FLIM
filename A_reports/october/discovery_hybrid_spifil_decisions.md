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

### 2026-10-30 — usuário: modelo leve
- **Limite:** o modelo é lightweight, com **no máximo 1M de parâmetros** (encoder mais qualquer projetora ou preditora usada para o embedding).
- **Projetora/preditora:** nunca pode ter mais parâmetros que o encoder.
- **Decoder:** fica fora da conta. Ele só serve para treinar e não vai para o SVM. Se o usuário quiser que o decoder também conte, rever.
- **Referência:** o encoder do R1 (5x5, 24/32/48) tem ~59k parâmetros. O A2 deve checar largura × kernel contra o teto.

### 2026-10-30 — usuário: mais camadas e teto de 5M
- **Profundidade:** sem limite de 3 camadas. O runner já aceita N camadas quando o YAML traz `layer_channels` e `layer_pools` com N itens. O limite de 5 era só o default de `LAYERWISE_SPECS`.
- **Teto de parâmetros:** sobe de 1M para **5M** (encoder mais projetora). A projetora continua sem poder ser maior que o encoder, e o decoder segue fora da conta.
- **R2 (A0) reescrito antes de rodar:** passou de "L4 com dilation 2" para um crescimento profundo de 8 camadas.
  - Canais: 24/32/48/48/64/64/64/64.
  - Kernels: 5,5,5,5 nas L1–L4 e 3 nas L5–L8.
  - Pool: 2 nas L1–L4 e nenhum nas L5–L8.
  - Dilation: 1 nas L1–L4 e 2,2,4,4 nas L5–L8.
  - Encoder com ~256k parâmetros.
  - Hipótese: no histórico as L4–L5 pioravam porque, sem pool, o campo receptivo não crescia. A dilation faz ele crescer sem encolher o mapa. O SVM em cada Lk mostra até onde vale crescer.

### 2026-10-30 — usuário: REGRA QUE PREVALECE — o SPiFiL tem que contribuir no treino
- Toda camada nasce dos kernels gerados pelo SPiFiL, nunca de inicialização aleatória. O treino com o decoder ajusta a partir deles.
- **Prova obrigatória:** o melhor candidato de cada frente ganha um controle idêntico com `random_layer_classic: true` (mesma arquitetura, filtros aleatórios). O SPiFiL só "contribui" se ganhar do controle no κ val. Esse controle entra na fila da GPU 3 após o R1.
- Um ganho de κ que venha de os filtros se afastarem dos kernels SPiFiL (fine-tune longo, lr alta) não vale. Por isso cada frente mede a similaridade entre os pesos treinados e os do grow.

### 2026-10-30 — usuário: encoder pequeno, profundo, com SPiFiL + backprop juntos
- **Norte:** encoder pequeno e profundo, sempre nascendo do SPiFiL e sempre testando backprop junto com o SPiFiL.
- **Largura:** a escada de largura do A2 (w96–w256, até 1,7M) foi cancelada. O w64 termina e entra só como ponto da lei de largura.
- **A2 muda de frente:** passa a cuidar de "SPiFiL + backprop". Cria a chave `layerwise_backprop: <escala de lr>`, com que as camadas antigas treinam com lr reduzido em vez de congelar. Hoje isso existe só no método ijepa_grow (growth.py:255-258).
- **Variante:** um fine-tune ponta a ponta só com reconstrução depois da última camada.
- **Por quê:** no histórico, o SPiFiL + backprop (grid4) foi o melhor SPiFiL a partir de 25% em eggs e protozoan (report:108-109). O SPiFiL dá o ponto de partida e o backprop ajusta o conjunto. A similaridade de cosseno aos kernels do grow vigia se o SPiFiL continua presente.
