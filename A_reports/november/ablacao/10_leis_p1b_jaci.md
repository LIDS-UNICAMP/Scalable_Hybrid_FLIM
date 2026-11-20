# leis_p1b_jaci — Leis de construção (largura por camada)

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `leis_p1b_jaci` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** idem
- **O que testa:** bases menores
- **Como:** split 1 × s42
- **Bases:** Lsub, Esub, heggs, hlarv
- **Resultados:** `cat`: receita ≥ `fps` (+0,007 a +0,032), +0,15 a +0,39 sobre aleatório; leis ±0,03; L: A r0,71 melhor em 3/4 (Esub 0,865 vs 0,739 com r0,5)
- **Estado:** concluído
- **Decisão:** começar largo, reduzir devagar

## Fontes
- [`leis_p0_p1.md`](fontes/leis_p0_p1.md)
- [`plano.md`](fontes/plano.md)
- [`relatorio_07_08nov.md`](../multi_dataset/relatorio_07_08nov.md)
- [`fila.md`](../multi_dataset/fila.md)
