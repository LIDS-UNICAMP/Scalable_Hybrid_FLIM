# P4 fundo F1–F4 (spf_fundo) — Fundo e detecção

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** P4 fundo F1–F4 (`spf_fundo`) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** cor sólida, textura, periferia, transformações
- **O que testa:** robustez ao fundo
- **Como:** splits 1–2; 64/256; 3 camadas; 400 treino/200 val
- **Bases:** 9 bases
- **Resultados:** κ retido, cor sólida: `fpsw` 0,99, `fps` 0,83, `random_z` 0,59; textura: 0,64 / 0,37 / 0,29; energia/área: 2,11 / 1,07 / 0,83
- **Estado:** concluído (2 splits)
- **Decisão:** `fpsw@mapa_peso` o mais robusto

## Fontes
- [`results_log.md`](../multi_dataset/results_log.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
- [`prompt_multiescala_resumo.md`](fontes/prompt_multiescala_resumo.md)
