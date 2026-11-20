# spfdeep_cedo — Largura, profundidade, kernel e tamanho da entrada

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `spfdeep_cedo` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** 1–8 camadas, pools no começo
- **O que testa:** profundidade
- **Como:** 3 splits × s42; 64/256; 6 braços; 4 aleatórios
- **Bases:** Psub, flowers, Esub, Lsub, heggs, aid
- **Resultados:** `fps@sift` 4/6 boas (+0,088); `fps` 64 `cat` sobe até L8 (+0,094); `fps` 256 `cat` pico L3 (+0,052)
- **Estado:** concluído
- **Decisão:** ir fundo com `cat`

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`pool_foco.md`](fontes/pool_foco.md)
- [`results_log.md`](../multi_dataset/results_log.md)
- [`law_size_superpixels.md`](../multi_dataset/law_size_superpixels.md)
