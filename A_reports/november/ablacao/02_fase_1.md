# Fase 1 (saliency/batch.py) — Saliência

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** Fase 1 (`saliency/batch.py`) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** 11 mapas clássicos + fusões A, B, C, D
- **O que testa:** qual mapa usar, sem rótulo
- **Como:** 270 imagens (30 por base); Q interno, estabilidade; IoU só para avaliar
- **Bases:** 9 bases
- **Resultados:** `gmr`: Q 0,725, estab. 0,898, 0,33 s, IoU 0,781; A: Q 0,737, 37 s
- **Estado:** concluído
- **Decisão:** `gmr` fixo

## Fontes
- [`saliencia.md`](../multi_dataset/saliencia.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`fila.md`](../multi_dataset/fila.md)
