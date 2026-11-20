# spf_r1 — Diversidade de filtros e seleção

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `spf_r1` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** `spifil` original, `spifil_dd`
- **O que testa:** método com rótulo (regime B)
- **Como:** idem
- **Bases:** flowers, Psub
- **Resultados:** `spifil` +0,099 ranking, +0,037 (64), pior −0,072; `spifil_dd` +0,027
- **Estado:** concluído
- **Decisão:** fora (usa rótulo)

## Fontes
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
- [`rodada2_sol_p3.md`](fontes/rodada2_sol_p3.md)
- [`auditoria.md`](fontes/auditoria.md)
- [`auditoria_rodada2_jaci.md`](fontes/auditoria_rodada2_jaci.md)
- [`results_log.md`](../multi_dataset/results_log.md)
