# Estágio 1: grades X1, X2, X3 (superpixels, ponto, redundância) — Fase com backprop

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `disc_md_x1_*`, `disc_md_x1s_*`, `disc_md_x2_{centroid,geodesic,mean,random,union}_*`, `disc_md_x3_{both,corr,cos}{80,90,95}_{rep,norep}_*` (runner `growth`); SVM em `disc_md_eval_x*_val_gemttapw`
- **Alternativa:** X1: px por superpixel (50, 200, 800, 3200) × filtros da L1 (32, 64, 128) × tamanho (200, 320, 640); X2: ponto do superpixel; X3: corte de redundância (cos, corr, ambos; 0,8/0,9/0,95; com e sem reposição)
- **O que testa:** lei dos superpixels e dos kernels (S3/U1 do prompt das leis) (fase com backprop, pausada em 07/11 a pedido do usuário)
- **Como:** treino camada a camada com backprop; leitura `gem_tta_pw`; validação
- **Bases:** 12 bases
- **Resultados:** a única tabela publicada dessa fase é a de superpixels da camada 1 (parcial, 28 de 117): ver [`03_superpixels_l1.md`](03_superpixels_l1.md). A lei tamanho × superpixels saiu depois pelo `l1fast` sem treino: [`09_lei_tamanho_superpixels.md`](09_lei_tamanho_superpixels.md)
- **Estado:** pausado (pesos em `/exps/moliveira/spifil_growth/disc_md_x*`)
- **Decisão:** superpixels fixados em 50 pelo usuário

## Fontes
- [`results_log.md`](../multi_dataset/results_log.md)
- [`pausa_treinos.md`](../multi_dataset/pausa_treinos.md)
- [`CONTEXT.md`](../multi_dataset/CONTEXT.md)
