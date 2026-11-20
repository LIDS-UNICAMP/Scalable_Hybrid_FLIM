# vlad_off — Leituras

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `vlad_off` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** VLAD 16/32 na última camada (`_eq`)
- **O que testa:** leitura de cenas
- **Como:** split 1 × s42/43; 256; L6; `fps`; `random_z`, `orth`
- **Bases:** aid, rs
- **Resultados:** aid: `cat` 0,776, vlad32 0,669, L 0,630; rs: vlad32 0,462, `cat` 0,456
- **Estado:** concluído
- **Decisão:** ganha do GeM da última, não do `cat`

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`plano.md`](fontes/plano.md)
- [`pool_foco.md`](fontes/pool_foco.md)
- [`auditoria.md`](fontes/auditoria.md)
