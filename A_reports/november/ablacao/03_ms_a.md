# ms_a — Pontos (sementes)

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `ms_a` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** S1–S8 (Gauss, DoG, LoG, região ∝ escala, várias escalas, fusão)
- **O que testa:** pontos/filtros multiescala
- **Como:** 3 splits × 5 sorteios; 64; L3 `cat`; aleatório congelado
- **Bases:** flowers, Psub, Esub, Lsub, aid
- **Resultados:** média Δ: S8 +0,077; S5 +0,080; geodésico +0,066; S4 −0,024 (pior); Gauss = cru
- **Estado:** concluído
- **Decisão:** empate com geodésico

## Fontes
- [`multiescala.md`](../multi_dataset/multiescala.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`results_log.md`](../multi_dataset/results_log.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
