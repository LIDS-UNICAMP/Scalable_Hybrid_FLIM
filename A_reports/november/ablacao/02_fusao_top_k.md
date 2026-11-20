# Fusão Top-K (B) — Saliência

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** Fusão Top-K (B) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** fusão de 3 mapas
- **O que testa:** B contra `gmr` fora da amostra
- **Como:** 432 imagens
- **Bases:** Lsub
- **Resultados:** IoU 0,657 vs 0,643; em heggs/pcysts perde
- **Estado:** parado pelo usuário
- **Decisão:** descartado

## Fontes
- [`saliencia.md`](../multi_dataset/saliencia.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`fila.md`](../multi_dataset/fila.md)
