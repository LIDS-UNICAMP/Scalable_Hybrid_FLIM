# sol_p6 Scattering — Encoders alternativos

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `sol_p6` Scattering (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** Morlet, ordens 0–2, J 2–4, L 8 (243 a 1251-d)
- **O que testa:** encoder sem parâmetros
- **Como:** 3 splits × 2 sorteios; configuração escolhida em 2 splits; aleatório de `ms_ctl` (64)
- **Bases:** 12 bases
- **Resultados:** Eggs 0,739; heggs 0,957; hlarv 0,980; pcysts 0,926; Larvae 0,704; aid 0,616; flowers 0,555; rs 0,421
- **Estado:** concluído
- **Decisão:** forte em fundo preto; SPiFiL ganha em 6 de 9

## Fontes
- [`results_log.md`](../multi_dataset/results_log.md)
- [`auditoria.md`](fontes/auditoria.md)
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
