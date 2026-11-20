# spf_sal_fl — Saliência

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `spf_sal_fl` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** máscara (triangle, otsu, t0,75) e peso, sobre `cos`, `fps`, `crus@sift_nms`
- **O que testa:** saliência como máscara ou peso
- **Como:** 3 splits × 2 sorteios; 64/256; L1–L3; `random_z`, `orth`, `whrand`
- **Bases:** flowers
- **Resultados:** 6 de 120 células passam, todas 64; `fpsw@mapa_peso` L2 cat Δ +0,053 [+0,031; +0,075]; `fps` sem mapa já dá +0,047
- **Estado:** concluído
- **Decisão:** mérito pequeno

## Fontes
- [`saliencia.md`](../multi_dataset/saliencia.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`fila.md`](../multi_dataset/fila.md)
