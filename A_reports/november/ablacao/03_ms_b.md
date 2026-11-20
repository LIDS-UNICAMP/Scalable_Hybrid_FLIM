# ms_b — Pontos (sementes)

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `ms_b` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** centroide, SIFT filtrado, objeto sem rótulo, `fps@geodesic`
- **O que testa:** localização com patch cru
- **Como:** 3 splits × 5 sorteios; 64; L3 `cat`; aleatório congelado (`ms_ctl`)
- **Bases:** flowers, Psub, Esub, Lsub, aid
- **Resultados:** média Δ: centroide +0,077; SIFT +0,060; objeto +0,085; FPS +0,087 (5 de 5 bases > 0)
- **Estado:** concluído
- **Decisão:** FPS o mais estável

## Fontes
- [`multiescala.md`](../multi_dataset/multiescala.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`results_log.md`](../multi_dataset/results_log.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
