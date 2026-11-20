# Auditoria 4.1 — SoftHebb

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** Auditoria 4.1 (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** código contra Journé et al. 2023
- **O que testa:** implementação certa?
- **Como:** teste numpy, lr = 0, BN
- **Bases:** —
- **Resultados:** regra igual (5,6e-17); diferenças: inv_temp 4 vs 1/0,65, passo por neurônio vs por tensor, lr igual nas camadas, norma inicial √2 vs ~25; efeito vem do peso, não da BN (26% vs 1,6%)
- **Estado:** concluído
- **Decisão:** receita dos autores → `hebb3`

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`results_log.md`](../multi_dataset/results_log.md)
- [`hebb_4_1_auditoria.md`](fontes/hebb_4_1_auditoria.md)
- [`plano.md`](fontes/plano.md)
- [`pool_foco.md`](fontes/pool_foco.md)
- [`prompt_hebb_resumo.md`](fontes/prompt_hebb_resumo.md)
