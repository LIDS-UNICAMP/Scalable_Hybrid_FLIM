# fpsw@mapa_peso (ranking) — Saliência

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `fpsw@mapa_peso` (ranking) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** peso pelo mapa
- **O que testa:** contra `fps`
- **Como:** célula em 2 splits, medida no 3º
- **Bases:** 5 bases
- **Resultados:** +0,174 (ranking); vs `fps`: +0,043 [+0,019; +0,067] (64), +0,015 [−0,013; +0,043] (256)
- **Estado:** concluído
- **Decisão:** candidato

## Fontes
- [`saliencia.md`](../multi_dataset/saliencia.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`fila.md`](../multi_dataset/fila.md)
