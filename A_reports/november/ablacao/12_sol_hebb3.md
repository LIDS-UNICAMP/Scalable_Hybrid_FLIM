# sol_hebb3 (piloto) — SoftHebb

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `sol_hebb3` (piloto) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** H01 fps fixo, H03 só calibração, H05 fps + SoftHebb dos autores
- **O que testa:** receita fiel
- **Como:** split 1 × s42; 64/256; 3 camadas
- **Bases:** Eggs, Larvae, aid, flowers, rs
- **Resultados:** H05−H01 `cat`: 64 κ 0,585→0,625, F1 0,663→0,690, acc 0,661→0,694; 256 κ 0,642→0,673, F1 0,699→0,721, acc 0,690→0,711; Larvae 256 −0,057; H03 ≈ H01
- **Estado:** concluído
- **Decisão:** passa em `gem_pw` → confirmação

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`results_log.md`](../multi_dataset/results_log.md)
- [`hebb_4_1_auditoria.md`](fontes/hebb_4_1_auditoria.md)
- [`plano.md`](fontes/plano.md)
- [`pool_foco.md`](fontes/pool_foco.md)
- [`prompt_hebb_resumo.md`](fontes/prompt_hebb_resumo.md)
