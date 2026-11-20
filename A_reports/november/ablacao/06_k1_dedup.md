# K1 dedup (treino com backprop) — Diversidade de filtros e seleção

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** K1 dedup (treino com backprop) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** tirar cópias do banco
- **O que testa:** cópias acima de 10%
- **Como:** treino real
- **Bases:** pcysts, heggs, hlarv
- **Resultados:** pcysts +0,118; heggs +0,028; hlarv −0,010
- **Estado:** concluído
- **Decisão:** regra candidata (fora do regime sem backprop)

## Fontes
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
- [`rodada2_sol_p3.md`](fontes/rodada2_sol_p3.md)
- [`auditoria.md`](fontes/auditoria.md)
- [`auditoria_rodada2_jaci.md`](fontes/auditoria_rodada2_jaci.md)
- [`results_log.md`](../multi_dataset/results_log.md)
