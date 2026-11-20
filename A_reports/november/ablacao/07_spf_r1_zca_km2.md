# spf_r1 — PCA/ZCA nos filtros

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `spf_r1` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** `zca_km2` (Coates e Ng)
- **O que testa:** ZCA + K-means
- **Como:** idem
- **Bases:** flowers, Psub
- **Resultados:** +0,015
- **Estado:** concluído
- **Decisão:** descartado

## Fontes
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
