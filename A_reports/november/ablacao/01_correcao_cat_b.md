# correção cat_b — Whitening entre camadas

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** correção `cat_b` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** leitura sem PCA-whitening
- **O que testa:** —
- **Como:** —
- **Bases:** —
- **Resultados:** `cat_b` não mede o whitening entre camadas
- **Estado:** —
- **Decisão:** nota de leitura

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`relatorio_07nov.md`](../multi_dataset/relatorio_07nov.md)
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
