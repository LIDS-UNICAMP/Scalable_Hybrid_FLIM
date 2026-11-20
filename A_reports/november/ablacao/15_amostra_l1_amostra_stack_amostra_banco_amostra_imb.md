# amostra_l1_*, amostra_stack_*, amostra_banco_*, amostra_imb_* — Imagens por classe

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `amostra_l1_*`, `amostra_stack_*`, `amostra_banco_*`, `amostra_imb_*` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** k imagens por classe × largura (L1), pilha, banco sem classe, H3
- **O que testa:** partes 1–4 antigas
- **Como:** controles na mesma largura
- **Bases:** aid, flowers, hlarv
- **Resultados:** "mais imagens por classe" ficou de fora (07/11)
- **Estado:** concluído
- **Decisão:** —

## Fontes
- [`results_log.md`](../multi_dataset/results_log.md)
- [`fila.md`](../multi_dataset/fila.md)
