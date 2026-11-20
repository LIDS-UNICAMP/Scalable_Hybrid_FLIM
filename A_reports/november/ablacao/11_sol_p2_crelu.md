# sol_p2 crelu — Polaridade e ativação

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `sol_p2` `crelu` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** R+ e R− (dobra canais)
- **O que testa:** —
- **Como:** idem
- **Bases:** idem
- **Resultados:** `fps` 256→512 canais +0,007 a +0,028 (Psub 0,813→0,841; aid 0,704→0,724), mas aleatório também sobe (aid 0,742, flowers 0,644)
- **Estado:** concluído
- **Decisão:** ganho por dimensão

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
