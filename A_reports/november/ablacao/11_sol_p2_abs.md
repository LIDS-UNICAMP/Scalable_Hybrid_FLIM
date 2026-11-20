# sol_p2 abs — Polaridade e ativação

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `sol_p2` `abs` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** \|R\|
- **O que testa:** módulo
- **Como:** 3 splits × 2 sorteios; 64/256; L3 `cat`; 3 braços; 3 aleatórios
- **Bases:** Esub, Psub, Lsub, aid, flowers, heggs
- **Resultados:** perde: `fps` 256 Lsub 0,698→0,500; Psub 0,813→0,710
- **Estado:** concluído
- **Decisão:** descartado

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
