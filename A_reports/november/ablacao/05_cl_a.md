# cl_a — Agrupamento

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `cl_a` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** C3–C9: grafo Louvain (`gr`, `gr_m`), SIFT + cluster, híbrido (`hib`), união, K-means com saliência
- **O que testa:** agrupamento em rodada barata
- **Como:** 3 splits × 5 sorteios; 64; L3 `cat`; SLIC 400; z-score por patch; aleatório de `cl_ctl`
- **Bases:** flowers, Psub, Esub, Lsub, aid
- **Resultados:** melhor: Esub 0,482, Lsub 0,438, Psub 0,522, aid 0,506, flowers 0,479; vs aleatório comum e geodésico: −0,09 a −0,12
- **Estado:** concluído
- **Decisão:** descartado nesta forma

## Fontes
- [`agrupamento_auditoria.md`](../multi_dataset/agrupamento_auditoria.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`prompt_clustering_resumo.md`](fontes/prompt_clustering_resumo.md)
