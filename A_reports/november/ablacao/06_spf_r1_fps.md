# spf_r1 — Diversidade de filtros e seleção

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `spf_r1` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** `fps` (FPS nos patches)
- **O que testa:** filtros diferentes entre si
- **Como:** 3 splits × 2 sorteios; 64/256; 8 aleatórios
- **Bases:** flowers, Psub (+ `r3`)
- **Resultados:** +0,071 (64) e +0,072 (256) em 7 bases, pior −0,017
- **Estado:** concluído
- **Decisão:** entra na receita

## Fontes
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
- [`rodada2_sol_p3.md`](fontes/rodada2_sol_p3.md)
- [`auditoria.md`](fontes/auditoria.md)
- [`auditoria_rodada2_jaci.md`](fontes/auditoria_rodada2_jaci.md)
- [`results_log.md`](../multi_dataset/results_log.md)
