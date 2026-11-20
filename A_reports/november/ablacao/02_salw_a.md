# salw_a (Jaci) — Saliência

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `salw_a` (Jaci) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** peso `w = clip((S−t)/(1−t))^g` na seleção (`fpsw@mapa_w*`) e no foco dos superpixels (`fps@foco_w*`)
- **O que testa:** corte t e força g
- **Como:** 3 splits × 2 sorteios; 64/256; L3 `cat`; 26 braços; `xavier`, `random_z`
- **Bases:** flowers, Psub, Esub, Lsub, aid, Larvae
- **Resultados:** todos ganham do `fps` (+0,015 a +0,034); Lsub 64: 0,726 vs 0,582; nada em aid/flowers 256; foco vence em 11/12 células, peso em 8–9/12; t 0,25–0,5 melhor; otsu entre os piores
- **Estado:** concluído
- **Decisão:** `fps@foco_w0.25_4` ou `fpsw@mapa_w0.5_1`

## Fontes
- [`saliencia.md`](../multi_dataset/saliencia.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`fila.md`](../multi_dataset/fila.md)
