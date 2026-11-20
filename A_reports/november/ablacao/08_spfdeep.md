# spfdeep_* — Leituras

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `spfdeep_*` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** `cat` vs `L`
- **O que testa:** concatenar camadas
- **Como:** 8 camadas, lidas em cada uma
- **Bases:** 6 bases
- **Resultados:** `fps` 64 `cat` L8 +0,094 vs `L` +0,035; só a última cai com a profundidade
- **Estado:** concluído
- **Decisão:** `cat` na receita

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`plano.md`](fontes/plano.md)
- [`pool_foco.md`](fontes/pool_foco.md)
- [`auditoria.md`](fontes/auditoria.md)
