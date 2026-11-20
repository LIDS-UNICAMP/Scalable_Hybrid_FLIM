# spfcombo_w2 (Jaci) — Whitening entre camadas

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `spfcombo_w2` (Jaci) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** idem
- **O que testa:** as 6 bases que faltavam
- **Como:** mesmo protocolo do `spfcombo_w`
- **Bases:** Eggs, Larvae, Protozoan, pcysts, hlarv, rs
- **Resultados:** L6 256: Eggs 0,845 (aleat. 0,681); Larvae 0,785 (0,533); Protozoan 0,849 (0,651); pcysts 0,920 (0,800); hlarv 0,966 (0,961); rs 0,506 (0,504)
- **Estado:** concluído
- **Decisão:** passa do aleatório em 11 de 12 bases; rs empata; resolve Eggs (+0,16)

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`relatorio_07nov.md`](../multi_dataset/relatorio_07nov.md)
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
