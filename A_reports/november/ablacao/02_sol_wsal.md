# sol_wsal (saliência) — Saliência

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `sol_wsal` (saliência) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** saliência + whitening
- **O que testa:** soma?
- **Como:** ver tema 1
- **Bases:** 12 bases
- **Resultados:** +0,02 a +0,025 sobre `fps`; Lsub 64: 0,648→0,824; Larvae 256: 0,744→0,855; t = 0 ou 0,5, g = 1
- **Estado:** concluído
- **Decisão:** soma

## Fontes
- [`saliencia.md`](../multi_dataset/saliencia.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`fila.md`](../multi_dataset/fila.md)
