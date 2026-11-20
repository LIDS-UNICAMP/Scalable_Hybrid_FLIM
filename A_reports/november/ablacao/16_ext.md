# *_ext (18 YAMLs) — Bases novas

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `*_ext` (18 YAMLs) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** todos os experimentos em coco
- **O que testa:** repetir as ablações
- **Como:** split 1 em boa parte
- **Bases:** coco
- **Resultados:** coco L6: 64 0,785 vs 256 0,667; melhor 0,792 (L, 1 split); aleatório 0,511
- **Estado:** parcial
- **Decisão:** ver temas 8 e 9

## Fontes
- [`fila.md`](../multi_dataset/fila.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`relatorio_07_08nov.md`](../multi_dataset/relatorio_07_08nov.md)
- [`pool_foco.md`](fontes/pool_foco.md)
