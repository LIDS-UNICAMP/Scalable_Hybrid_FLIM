# sol_p6 PCANet — Encoders alternativos

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `sol_p6` PCANet (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** filtros 8/7, kernel 7
- **O que testa:** —
- **Como:** idem
- **Bases:** 12 bases
- **Resultados:** Lsub 0,586; Esub 0,752; rs 0,450; abaixo nas demais
- **Estado:** concluído
- **Decisão:** descartado

## Fontes
- [`results_log.md`](../multi_dataset/results_log.md)
- [`auditoria.md`](fontes/auditoria.md)
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
