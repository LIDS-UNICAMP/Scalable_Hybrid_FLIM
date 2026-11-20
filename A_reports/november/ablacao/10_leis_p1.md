# leis_p1 (Athena) — Leis de construção (largura por camada)

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `leis_p1` (Athena) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** 17 leis: A geométrica, B posição, C orçamento (P = 8,2M)
- **O que testa:** forma da rede contra 256×6
- **Como:** split 1 × s42; L6 `cat` e L; `fps`, `fpsw@mapa_w0.5_1`; `random_z`, `orth`
- **Bases:** aid, flowers, rs
- **Resultados:** aid: A 512 r0,71 0,788 vs âncora 0,786 (1,5× params); flowers 0,659 vs 0,656; rs: as 17 passam a âncora 0,432 (melhor C 0,492), mas o aleatório sobe junto
- **Estado:** concluído
- **Decisão:** sem ganho com mesmo orçamento

## Fontes
- [`leis_p0_p1.md`](fontes/leis_p0_p1.md)
- [`plano.md`](fontes/plano.md)
- [`relatorio_07_08nov.md`](../multi_dataset/relatorio_07_08nov.md)
- [`fila.md`](../multi_dataset/fila.md)
