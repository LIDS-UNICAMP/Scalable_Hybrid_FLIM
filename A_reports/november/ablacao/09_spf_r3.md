# spf_r3 (Jaci) — Largura, profundidade, kernel e tamanho da entrada

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `spf_r3` (Jaci) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** sobreviventes do `r2`
- **O que testa:** 1 base por família
- **Como:** 3 splits × 2 sorteios; 64/256; L3
- **Bases:** flowers, Psub, heggs, hlarv, aid
- **Resultados:** `fps@sift` entre os 5
- **Estado:** concluído
- **Decisão:** → `r4`

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`pool_foco.md`](fontes/pool_foco.md)
- [`results_log.md`](../multi_dataset/results_log.md)
- [`law_size_superpixels.md`](../multi_dataset/law_size_superpixels.md)
