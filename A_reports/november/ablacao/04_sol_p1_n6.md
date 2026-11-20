# sol_p1_n6 (Jaci) — Normalizações

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `sol_p1_n6` (Jaci) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** N6 = z-score por patch + whitening entre camadas (λ 0,001; 0,01; 0,1)
- **O que testa:** N1 com N5
- **Como:** idem `sol_p1`
- **Bases:** idem
- **Resultados:** κ cai em todas as células: Esub 64 `fps` 0,694→0,473; Psub 256 0,819→0,644; aid 256 0,705→0,609; heggs 256 0,929→0,874
- **Estado:** concluído
- **Decisão:** descartado

## Fontes
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`prompt_sol_resumo.md`](fontes/prompt_sol_resumo.md)
