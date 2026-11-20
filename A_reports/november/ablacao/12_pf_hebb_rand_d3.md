# pf_hebb_rand_d3 — SoftHebb

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `pf_hebb_rand_d3` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** aleatório puro (H00) na rede SoftHebb
- **O que testa:** controle
- **Como:** idem
- **Bases:** aid, rs, flowers, coco
- **Resultados:** —
- **Estado:** rodando (Athena)
- **Decisão:** —

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`results_log.md`](../multi_dataset/results_log.md)
- [`hebb_4_1_auditoria.md`](fontes/hebb_4_1_auditoria.md)
- [`plano.md`](fontes/plano.md)
- [`pool_foco.md`](fontes/pool_foco.md)
- [`prompt_hebb_resumo.md`](fontes/prompt_hebb_resumo.md)

## Atualização (revisão de completude, 08/11)
- **Terminou no Athena às 21:02 UTC (08/11), sem falhas.** Ainda sem comparação: as versões que ele controla (`pf_hebb_d3_off` no Jaci, 1 de 2 jobs; `pf_hebb_d3_on` no Athena) estão rodando.
- Fonte extra: histórico da conversa (sessão 990a33ea, mensagens do assistente), 08/11 21:02 UTC.
