# sol_p1 — Normalizações

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `sol_p1` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** N2 global, N3 por canal, N7 mediana/MAD
- **O que testa:** outras normalizações
- **Como:** 3 splits × 2 sorteios; 64/256; L3 `cat`; `crus@geodesic`, `fps`, `fpsw@mapa_peso`; 3 aleatórios
- **Bases:** Esub, Psub, Lsub, aid, flowers, heggs
- **Resultados:** vs N0: N2 −0,089 (4/36 células); N3 −0,003 (12/36); N7 −0,236 (0/36); N2 derruba larvas −0,29
- **Estado:** concluído
- **Decisão:** N0 fica

## Fontes
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`prompt_sol_resumo.md`](fontes/prompt_sol_resumo.md)
