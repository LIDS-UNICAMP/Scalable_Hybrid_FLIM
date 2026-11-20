# spf_r1 — Agrupamento

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `spf_r1` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** BoVW SIFT (`bovw_w@sift`, `bovw_sil@sift`), `fpsd@sift`, `cosd@sift`
- **O que testa:** descritor SIFT 128-d
- **Como:** idem
- **Bases:** flowers, Psub
- **Resultados:** `fpsd@sift` − `fps@sift` = −0,031 (256); `bovw_w` +0,046 ranking
- **Estado:** concluído
- **Decisão:** descritor não ajuda

## Fontes
- [`agrupamento_auditoria.md`](../multi_dataset/agrupamento_auditoria.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`prompt_clustering_resumo.md`](fontes/prompt_clustering_resumo.md)
