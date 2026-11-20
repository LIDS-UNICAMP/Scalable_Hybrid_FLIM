# Com × sem (if/else) — Saliência

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** Com × sem (if/else) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** receita − `fps` sem saliência
- **O que testa:** quando ligar
- **Como:** `sol_wsal` L6 cat
- **Bases:** 12 bases
- **Resultados:** aid −0,010/−0,014; rs +0,006/−0,015; flowers +0,022/+0,020; larvas +0,03 a +0,14; ovos +0,006 a +0,033; protoz. +0,04 com 64, −0,002 a −0,016 com 256
- **Estado:** concluído
- **Decisão:** regra do usuário: sem saliência em aid e rs

## Fontes
- [`saliencia.md`](../multi_dataset/saliencia.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`fila.md`](../multi_dataset/fila.md)
