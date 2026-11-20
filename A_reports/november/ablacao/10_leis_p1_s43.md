# leis_p1_s43 — Leis de construção (largura por camada)

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `leis_p1_s43` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** idem, s43
- **O que testa:** estabilidade da ordem
- **Como:** idem
- **Bases:** aid, flowers, rs
- **Resultados:** Spearman 42 vs 43: aid 0,94; flowers 0,64; rs −0,07; A 512 r0,71 em 1º (aid 0,786, flowers 0,644)
- **Estado:** concluído
- **Decisão:** ordem estável só em aid

## Fontes
- [`leis_p0_p1.md`](fontes/leis_p0_p1.md)
- [`plano.md`](fontes/plano.md)
- [`relatorio_07_08nov.md`](../multi_dataset/relatorio_07_08nov.md)
- [`fila.md`](../multi_dataset/fila.md)
