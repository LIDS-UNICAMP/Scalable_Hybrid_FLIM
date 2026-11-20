# Detecção 12 bases (spf_deteccao) — Fundo e detecção

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** Detecção 12 bases (`spf_deteccao`) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** energia na camada 3 dentro do objeto
- **O que testa:** quem acende no objeto
- **Como:** split 1, 4 imagens por base
- **Bases:** 9 com máscara
- **Resultados:** `fpsw@mapa_peso` acende no objeto em 6 de 6 bases de fundo real (1,3–3,4× a área)
- **Estado:** concluído (sinal)
- **Decisão:** saliência guia a detecção

## Fontes
- [`results_log.md`](../multi_dataset/results_log.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
- [`prompt_multiescala_resumo.md`](fontes/prompt_multiescala_resumo.md)
