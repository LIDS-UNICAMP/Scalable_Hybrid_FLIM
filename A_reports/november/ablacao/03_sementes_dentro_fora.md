# Sementes dentro/fora — Pontos (sementes)

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** Sementes dentro/fora (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** medoide, SIFT, `sift_nms`, `obj`
- **O que testa:** onde os pontos caem (máscara só mede)
- **Como:** split 1, 4 imagens por base
- **Bases:** 9 bases
- **Resultados:** medoide 25–49% no objeto (2,4–4× o acaso); com corte de saliência 70–98%, mas 9–23 pontos por imagem
- **Estado:** concluído
- **Decisão:** diagnóstico

## Fontes
- [`multiescala.md`](../multi_dataset/multiescala.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`results_log.md`](../multi_dataset/results_log.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
