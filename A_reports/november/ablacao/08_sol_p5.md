# sol_p5 — Leituras

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `sol_p5` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** GAP, GMP, GeM p 2/4/6, SPP, `ms` (3 escalas), covariância
- **O que testa:** pooling
- **Como:** 3 splits × 2 sorteios; 64/256; L3; 3 braços; 3 aleatórios
- **Bases:** Esub, Psub, Lsub, aid, flowers, heggs
- **Resultados:** média 6 bases (`fpsw@mapa_w0_1`): `cat` 0,770 (256)/0,710 (64); `cat_ms` 0,783/0,739; GeM p fixo empata; GAP/GMP 0,65–0,72; SPP piora (>1300-d); `L_cov` 0,706→0,755 (256), 0,655→0,762 (64)
- **Estado:** concluído
- **Decisão:** `cat_ms` candidata; GAP, GMP, SPP fora

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`plano.md`](fontes/plano.md)
- [`pool_foco.md`](fontes/pool_foco.md)
- [`auditoria.md`](fontes/auditoria.md)
