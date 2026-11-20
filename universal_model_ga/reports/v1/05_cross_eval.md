# 05 — Avaliação cruzada (rede universal, só SPiFiL)

Mesma topologia em todas as bases; SPiFiL ajustado só no treino de cada base; SVM original (C=100) por base.
Sem backprop/SSL. Leitura por célula escolhida pela **validação** entre gap1280 e so2_4096_pca_tta8.
Ranking: menor arrependimento máximo de validação, depois maior κ_val médio. **O teste não seleciona.**

Completas (5 bases): 0; parciais: 0.

## Ranking (validação)

| # | arquitetura (fonte) | κ_val médio | regret máx (val) | val flowers | val remote_sensing | val ham10000 | val aid | val coconut_trees | κ_teste médio |
|---|---|---|---|---|---|---|---|---|---|

## Teste (só reportado)

| arquitetura (fonte) | teste flowers | teste remote_sensing | teste ham10000 | teste aid | teste coconut_trees | leituras (val) |
|---|---|---|---|---|---|---|
