# spf_h (+ spf_h3) — Fundo e detecção

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `spf_h` (+ `spf_h3`) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** banco misto, só objeto, só fundo [oráculo]
- **O que testa:** hipótese do fundo
- **Como:** 3 splits × 2 sorteios; 64/256; L3; 19 braços; 4 aleatórios
- **Bases:** Psub, Esub, Lsub, Larvae
- **Resultados:** só fundo até −0,13 (IC < 0 em 7 de 8); só objeto até +0,066, nenhum IC > 0
- **Estado:** concluído
- **Decisão:** fundo atrapalha; objeto sozinho não basta

## Fontes
- [`results_log.md`](../multi_dataset/results_log.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
- [`prompt_multiescala_resumo.md`](fontes/prompt_multiescala_resumo.md)
