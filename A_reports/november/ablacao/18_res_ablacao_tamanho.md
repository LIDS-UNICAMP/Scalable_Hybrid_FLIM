# res_* (ablação de tamanho) — Fase com backprop

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `disc_md_res_s<tam>_n<sp>_d<prof>_{a,b}` (31 YAMLs, runner `growth`) e `disc_md_eval_res_s<tam>_val_gemttapw` (10 SVMs, um por estágio, só depois do treino)
- **Alternativa:** tamanho 64 a 1280 px × nº de superpixels × profundidade 3–5, braços a/b
- **O que testa:** relação SPiFiL × tamanho da imagem de entrada (fase com backprop, pausada em 07/11 a pedido do usuário)
- **Como:** lote 32×1; validação
- **Bases:** várias
- **Resultados:** — (nenhuma fonte publica número desta grade)
- **Estado:** pausado
- **Decisão:** —

## Fontes
- comentários do topo de `experiments/autoencoder/disc_md_res_*.yaml`
- [`pausa_treinos.md`](../multi_dataset/pausa_treinos.md)
