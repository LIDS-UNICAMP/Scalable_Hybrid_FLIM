# sol_hebb2_piloto (Fila 2) — SoftHebb

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `sol_hebb2_piloto` (Fila 2) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** E04 viés por frequência, E05 reforço de inativos
- **O que testa:** homeostase
- **Como:** split 1 × s42
- **Bases:** flowers, aid, rs, Esub, Lsub, Psub
- **Resultados:** banco fica redundante (vizinho 0,77–0,84); E04 random 64 L3 0,519→0,597 (puxado por Lsub); E05 quase não age
- **Estado:** concluído
- **Decisão:** não resolve

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`results_log.md`](../multi_dataset/results_log.md)
- [`hebb_4_1_auditoria.md`](fontes/hebb_4_1_auditoria.md)
- [`plano.md`](fontes/plano.md)
- [`pool_foco.md`](fontes/pool_foco.md)
- [`prompt_hebb_resumo.md`](fontes/prompt_hebb_resumo.md)
