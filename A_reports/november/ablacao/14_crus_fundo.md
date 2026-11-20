# crus@fundo — Fundo e detecção

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `crus@fundo` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** patches só de fundo, sem máscara
- **O que testa:** —
- **Como:** `r1`
- **Bases:** 7 bases
- **Resultados:** −0,058 (ranking); −0,074
- **Estado:** concluído
- **Decisão:** descartado

## Fontes
- [`results_log.md`](../multi_dataset/results_log.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
- [`prompt_multiescala_resumo.md`](fontes/prompt_multiescala_resumo.md)
