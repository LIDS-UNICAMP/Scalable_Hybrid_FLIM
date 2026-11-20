# sol_hebb1 (Fila 1, E00–E03) — SoftHebb

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `sol_hebb1` (Fila 1, E00–E03) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** início random, `fps`, `crus@geodesic`; 0/30/90/300 passos; restrito (bola 0,5\|w0\|)
- **O que testa:** SoftHebb melhora o SPiFiL?
- **Como:** 3 splits × 2 sorteios; 64/256; 3 camadas; `gem_pw`
- **Bases:** 12 bases
- **Resultados:** `fps` 256 L3: A 0,732; C30 0,708; C90 0,703; C300 0,704; restrito 0,721; A ≥ C em 9 de 12
- **Estado:** concluído
- **Decisão:** não melhora (implementação antiga)

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`results_log.md`](../multi_dataset/results_log.md)
- [`hebb_4_1_auditoria.md`](fontes/hebb_4_1_auditoria.md)
- [`plano.md`](fontes/plano.md)
- [`pool_foco.md`](fontes/pool_foco.md)
- [`prompt_hebb_resumo.md`](fontes/prompt_hebb_resumo.md)
