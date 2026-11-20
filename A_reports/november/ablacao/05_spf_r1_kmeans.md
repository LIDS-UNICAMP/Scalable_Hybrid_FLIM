# spf_r1 — Agrupamento

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `spf_r1` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** K-means (`km_w`, `km_w_m`)
- **O que testa:** cluster como kernel
- **Como:** 3 splits × 2 sorteios; 64/256
- **Bases:** flowers, Psub
- **Resultados:** empata com patch sorteado (+0,004); vs `fps`: −0,028 (64), −0,044 (256); 53% de patches planos vs 2%
- **Estado:** concluído
- **Decisão:** descartado

## Fontes
- [`agrupamento_auditoria.md`](../multi_dataset/agrupamento_auditoria.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`prompt_clustering_resumo.md`](fontes/prompt_clustering_resumo.md)
