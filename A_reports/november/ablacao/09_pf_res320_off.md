# pf_res320_off — Largura, profundidade, kernel e tamanho da entrada

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `pf_res320_off` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** entrada 320 px
- **O que testa:** detalhe em sensoriamento remoto
- **Como:** split 1 × 2 sorteios; 256; L6 `cat`; `fps`
- **Bases:** aid, rs
- **Resultados:** aid 0,801/0,794 vs 0,786 (+0,015); rs 0,533/0,486 vs 0,432 (+0,054 a +0,10); aleatório 0,692–0,700 e 0,449–0,481
- **Estado:** concluído (piloto)
- **Decisão:** promissor; falta 3 splits

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`pool_foco.md`](fontes/pool_foco.md)
- [`results_log.md`](../multi_dataset/results_log.md)
- [`law_size_superpixels.md`](../multi_dataset/law_size_superpixels.md)
