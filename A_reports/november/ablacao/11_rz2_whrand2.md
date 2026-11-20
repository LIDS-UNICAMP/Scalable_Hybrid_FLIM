# rz2, whrand2 — Polaridade e ativação

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `rz2`, `whrand2` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** R+ e R− só no aleatório
- **O que testa:** metade das direções
- **Como:** 12 pares
- **Bases:** —
- **Resultados:** `rz2` − `random_z` −0,001; `whrand2` − `whrand` −0,009
- **Estado:** concluído
- **Decisão:** sem efeito

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
