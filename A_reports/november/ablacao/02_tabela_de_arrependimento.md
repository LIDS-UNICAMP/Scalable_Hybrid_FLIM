# Tabela de arrependimento — Saliência

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** Tabela de arrependimento (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** 104 receitas fixas
- **O que testa:** receita única (minimax)
- **Como:** `sol_wsal`, 5 famílias
- **Bases:** 12 bases
- **Resultados:** `fpsw@mapa_w0.5_1` L6 256: pior arrependimento −0,029; vs aleatório: ovos +0,199, larvas +0,290, protoz. +0,210, aid+rs +0,028, flowers +0,067; família de fora: mesma receita em 4 de 5
- **Estado:** concluído
- **Decisão:** receita única

## Fontes
- [`saliencia.md`](../multi_dataset/saliencia.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`fila.md`](../multi_dataset/fila.md)
