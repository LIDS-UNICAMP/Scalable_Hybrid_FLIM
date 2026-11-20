# leitura_a2 — Leituras

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `leitura_a2` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** A0 GeM p3, A1 GeM p2, A2 `cat_ms_eq` (mesma dimensão por PCA de treino)
- **O que testa:** ganho é escala ou dimensão?
- **Como:** 3 splits × 2 sorteios; 64/256; L3; `fps`; 3 aleatórios
- **Bases:** Esub, Lsub, Psub, aid, flowers, rs
- **Resultados:** `cat_ms_eq` − `cat`, fps 256: +0,009, −0,013, +0,029, +0,020, −0,002, +0,030 (4/6 bases, 3/5 famílias); fps 64: 5/6 bases; A1 ±0,03 sem direção
- **Estado:** concluído
- **Decisão:** **passa o critério** (escala); GeM p2 encerrado

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`plano.md`](fontes/plano.md)
- [`pool_foco.md`](fontes/pool_foco.md)
- [`auditoria.md`](fontes/auditoria.md)
