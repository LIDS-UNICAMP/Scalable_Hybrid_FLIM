# sol_p6 (braço SoftHebb) — SoftHebb

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `sol_p6` (braço SoftHebb) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** SoftHebb do aleatório (regime C)
- **O que testa:** encoder alternativo
- **Como:** 3 splits × 2 sorteios; 64/256; 3 camadas; 300 passos; inv_temp 4
- **Bases:** 12 bases
- **Resultados:** Eggs 0,742; Esub 0,803; Psub 0,774; aid 0,690; flowers 0,590
- **Estado:** concluído
- **Decisão:** ver tema 13

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`results_log.md`](../multi_dataset/results_log.md)
- [`hebb_4_1_auditoria.md`](fontes/hebb_4_1_auditoria.md)
- [`plano.md`](fontes/plano.md)
- [`pool_foco.md`](fontes/pool_foco.md)
- [`prompt_hebb_resumo.md`](fontes/prompt_hebb_resumo.md)
