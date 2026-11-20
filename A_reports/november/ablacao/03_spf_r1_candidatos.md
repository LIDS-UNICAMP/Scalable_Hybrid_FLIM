# spf_r1 (candidatos) — Pontos (sementes)

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `spf_r1` (candidatos) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** `crus@centroid`, `@geodesic`, `@sift`, `@sift_nms`, `@sift_sp`, `@uniao`, `@obj`, `@sal`, `@l1obj`
- **O que testa:** de onde tirar o patch
- **Como:** 3 splits × 2 sorteios; 64/256; L1–L3; 8 aleatórios
- **Bases:** flowers, Psub
- **Resultados:** ranking: `crus@obj` +0,126; `@centroid` +0,105; `@geodesic` +0,097; `@sift_nms` +0,097; `@sift_sp` +0,108
- **Estado:** concluído
- **Decisão:** diferenças pequenas

## Fontes
- [`multiescala.md`](../multi_dataset/multiescala.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`results_log.md`](../multi_dataset/results_log.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
