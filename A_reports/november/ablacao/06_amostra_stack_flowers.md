# amostra_stack_flowers — Diversidade de filtros e seleção

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `amostra_stack_flowers` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** DPP
- **O que testa:** seleção por DPP
- **Como:** contra 1 aleatório
- **Bases:** flowers
- **Resultados:** +0,028 (64 L1); +0,031 (64 L3); +0,044 (128 L3); +0,024 (256 L3)
- **Estado:** parcial
- **Decisão:** não levado à grade

## Fontes
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
- [`rodada2_sol_p3.md`](fontes/rodada2_sol_p3.md)
- [`auditoria.md`](fontes/auditoria.md)
- [`auditoria_rodada2_jaci.md`](fontes/auditoria_rodada2_jaci.md)
- [`results_log.md`](../multi_dataset/results_log.md)
