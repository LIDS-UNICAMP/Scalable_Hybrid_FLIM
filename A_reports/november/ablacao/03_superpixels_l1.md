# Superpixels L1 — Pontos (sementes)

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** Superpixels L1 (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** 800 vs 50 superpixels
- **O que testa:** número de superpixels
- **Como:** camada 1, 200 px
- **Bases:** 7 bases
- **Resultados:** heggs 0,744 (800) vs 0,842 (50); pcysts 0,589 vs 0,743; 50 fixado pelo usuário (custa 0,029 na média)
- **Estado:** parcial (28 de 117)
- **Decisão:** 50 DISF

## Fontes
- [`multiescala.md`](../multi_dataset/multiescala.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`results_log.md`](../multi_dataset/results_log.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
