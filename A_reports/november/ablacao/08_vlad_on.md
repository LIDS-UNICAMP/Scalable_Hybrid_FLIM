# vlad_on — Leituras

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `vlad_on` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** idem com saliência
- **O que testa:** idem
- **Como:** idem; `fpsw@mapa_w0.5_1`
- **Bases:** flowers, coco
- **Resultados:** coco: L 0,788, vlad32 0,689, `cat` 0,630; flowers: `cat` 0,630, vlad 0,604
- **Estado:** concluído
- **Decisão:** VLAD encerrado

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`plano.md`](fontes/plano.md)
- [`pool_foco.md`](fontes/pool_foco.md)
- [`auditoria.md`](fontes/auditoria.md)
