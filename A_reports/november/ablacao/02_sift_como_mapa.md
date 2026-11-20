# SIFT como mapa — Saliência

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** SIFT como mapa (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** densidade de keypoints (a–d)
- **O que testa:** SIFT diz onde está o objeto?
- **Como:** 269 imagens com máscara
- **Bases:** 9 bases
- **Resultados:** perde do `gmr` em IoU nas 9; AP em 7–8; revocação 0,32 vs 0,78
- **Estado:** concluído
- **Decisão:** descartado (SIFT só para pontos)

## Fontes
- [`saliencia.md`](../multi_dataset/saliencia.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`fila.md`](../multi_dataset/fila.md)
