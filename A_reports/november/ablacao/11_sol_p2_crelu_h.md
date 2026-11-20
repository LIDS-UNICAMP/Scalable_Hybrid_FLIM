# sol_p2 crelu_h — Polaridade e ativação

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `sol_p2` `crelu_h` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** mesmo orçamento
- **O que testa:** —
- **Como:** idem
- **Bases:** idem
- **Resultados:** igual ou abaixo do `fps`
- **Estado:** concluído
- **Decisão:** ReLU fica

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
