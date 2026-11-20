# fps@sift — Pontos (sementes)

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `fps@sift` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** FPS sobre pontos SIFT
- **O que testa:** SIFT como ponto
- **Como:** `r1`–`r3`, `spfdeep_*`
- **Bases:** várias
- **Resultados:** `crus@sift` +0,087 (64, 7 bases); detecção 36% vs 25% nas larvas; entre os 5 do `r3`
- **Estado:** concluído
- **Decisão:** alternativa ao medoide

## Fontes
- [`multiescala.md`](../multi_dataset/multiescala.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`results_log.md`](../multi_dataset/results_log.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
