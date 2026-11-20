# spf_sal_par — Saliência

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `spf_sal_par` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** idem + oráculo pelo label
- **O que testa:** idem nos parasitos
- **Como:** idem
- **Bases:** Psub, Esub, Lsub, Larvae
- **Resultados:** ganho do mapa: Lsub +0,064; Larvae +0,041; Esub +0,013; Psub −0,001; máscara e peso empatam (+0,022 a +0,028)
- **Estado:** concluído
- **Decisão:** só para larvas (na época)

## Fontes
- [`saliencia.md`](../multi_dataset/saliencia.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`fila.md`](../multi_dataset/fila.md)
