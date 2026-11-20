# ext_piloto — Bases novas

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `ext_piloto` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** `fps` vs `random_z`
- **O que testa:** custo e sinal
- **Como:** split 1 × s42; 64; L3
- **Bases:** coco, food101
- **Resultados:** coco: 0,740 vs 0,675; food101: 0,17 vs 0,15, 3 h 23 min por job
- **Estado:** concluído
- **Decisão:** coco entra; food101 caro

## Fontes
- [`fila.md`](../multi_dataset/fila.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`relatorio_07_08nov.md`](../multi_dataset/relatorio_07_08nov.md)
- [`pool_foco.md`](fontes/pool_foco.md)
