# L vs cat em coco (vlad_on, pf_res128_on) — Leituras

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `L` vs `cat` em coco (`vlad_on`, `pf_res128_on`) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** só a última camada
- **O que testa:** a concatenação dilui?
- **Como:** split 1 × 2 sorteios; 64/256
- **Bases:** coco
- **Resultados:** 128 px: L 0,792 (256) vs `cat` 0,639; aleatório L 0,511
- **Estado:** concluído
- **Decisão:** em aberto: leitura por base

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`plano.md`](fontes/plano.md)
- [`pool_foco.md`](fontes/pool_foco.md)
- [`auditoria.md`](fontes/auditoria.md)
