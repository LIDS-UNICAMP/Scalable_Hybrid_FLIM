# spf_hz (Jaci) — Fundo e detecção

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `spf_hz` (Jaci) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** fundo zerado pelo label [oráculo]
- **O que testa:** teto se soubesse onde está o objeto
- **Como:** 3 splits × 2 sorteios; 64/256; `spifil`, `crus`, `zca2`
- **Bases:** Psub, Esub, Lsub, Larvae, Eggs, Protozoan
- **Resultados:** `crus` +0,086 (teria +0,226); `spifil` +0,082 (+0,216): ~38% do caminho; larvas 46–67%; ovos 1–18%
- **Estado:** concluído
- **Decisão:** detectar daria +0,14 de κ

## Fontes
- [`results_log.md`](../multi_dataset/results_log.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
- [`prompt_multiescala_resumo.md`](fontes/prompt_multiescala_resumo.md)
