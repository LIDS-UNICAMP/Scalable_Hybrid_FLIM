# sol_hebb3_conf — SoftHebb

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `sol_hebb3_conf` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** idem
- **O que testa:** confirmação
- **Como:** 3 splits × 5 sorteios (n = 15 por célula); 64/256
- **Bases:** Eggs, Larvae, aid, flowers, rs
- **Resultados:** 256 `cat`: Eggs +0,026, Larvae +0,052, aid +0,064, flowers +0,036, rs +0,026; F1 +0,021 a +0,060; acc +0,020 a +0,059; 64: só aid +0,073 e rs +0,053
- **Estado:** concluído
- **Decisão:** **confirmado** (256, 3 camadas, sem whitening nem saliência)

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`results_log.md`](../multi_dataset/results_log.md)
- [`hebb_4_1_auditoria.md`](fontes/hebb_4_1_auditoria.md)
- [`plano.md`](fontes/plano.md)
- [`pool_foco.md`](fontes/pool_foco.md)
- [`prompt_hebb_resumo.md`](fontes/prompt_hebb_resumo.md)
