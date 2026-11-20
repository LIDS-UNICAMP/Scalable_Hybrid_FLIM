# pf_leis_off — Leis de construção (largura por camada)

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `pf_leis_off` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** A 512 r0,71, B late, 256×6 × `cat`/`cat_ms_eq`
- **O que testa:** lei + leitura multiescala
- **Como:** split 1 × 2 sorteios; `fps`; aleatórios puros
- **Bases:** aid, rs
- **Resultados:** —
- **Estado:** na fila (Jaci, fila3)
- **Decisão:** —

## Fontes
- [`leis_p0_p1.md`](fontes/leis_p0_p1.md)
- [`plano.md`](fontes/plano.md)
- [`relatorio_07_08nov.md`](../multi_dataset/relatorio_07_08nov.md)
- [`fila.md`](../multi_dataset/fila.md)
