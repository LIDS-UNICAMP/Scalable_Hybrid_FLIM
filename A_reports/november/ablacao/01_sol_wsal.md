# sol_wsal (Athena) — Whitening entre camadas

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `sol_wsal` (Athena) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** whitening + saliência
- **O que testa:** os dois juntos pela 1ª vez
- **Como:** 3 splits × s42; 64/256; L2–L6; 9 braços (`fps`, `fps@combo`, `fpsw@mapa_*`, `fps@foco_*`); 4 aleatórios
- **Bases:** 12 bases
- **Resultados:** 24 de 24 células acima do melhor aleatório em L3 e L6 (rs empata); `fps@combo` igual ao `spfcombo_w` até a 3ª casa; dicionário SIFT soma só +0,003
- **Estado:** concluído
- **Decisão:** base da receita única

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`relatorio_07nov.md`](../multi_dataset/relatorio_07nov.md)
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
