# sol_p3 (Jaci) — Diversidade de filtros e seleção

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `sol_p3` (Jaci) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** f2 (B1 FPS em 1−corr), f3 (B2 baixa redundância), f4 cobertura, f5 (B3 visual + funcional), f6 (f3 + estabilidade), K-medoids
- **O que testa:** diversidade pelas respostas
- **Como:** 3 splits × 2 sorteios; 64/256; L3 `cat`; 3 aleatórios
- **Bases:** Esub, Lsub, Psub, aid, flowers, heggs
- **Resultados:** f − fps em 72 células: f2 −0,001; f3 +0,001; f4 −0,023; f5 −0,004; f6 −0,051. F1: f3 +0,007, f5 +0,004. aid f3 +0,017/+0,028. K-medoids não ganha e custa 3,5×
- **Estado:** concluído
- **Decisão:** não substitui `fps`; f3 sinal em aid

## Fontes
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
- [`rodada2_sol_p3.md`](fontes/rodada2_sol_p3.md)
- [`auditoria.md`](fontes/auditoria.md)
- [`auditoria_rodada2_jaci.md`](fontes/auditoria_rodada2_jaci.md)
- [`results_log.md`](../multi_dataset/results_log.md)
