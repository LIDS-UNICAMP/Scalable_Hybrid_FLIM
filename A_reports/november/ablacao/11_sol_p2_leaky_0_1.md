# sol_p2 leaky 0,1 — Polaridade e ativação

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `sol_p2` `leaky` 0,1 (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** Leaky ReLU
- **O que testa:** —
- **Como:** idem
- **Bases:** idem
- **Resultados:** empata; só Lsub 64 `fpsw` 0,617→0,657
- **Estado:** concluído
- **Decisão:** descartado

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
