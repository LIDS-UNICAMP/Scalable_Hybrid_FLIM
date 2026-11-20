# porclasse_piloto — Imagens por classe

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `porclasse_piloto` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** k = 1, 5, 10, 15, 30 por classe (B) vs mesmo total sem classe (A)
- **O que testa:** escolher por classe ajuda?
- **Como:** split 1 × s42; 64; L6 + whitening
- **Bases:** flowers
- **Resultados:** B: 0,647/0,626/0,659/0,689/0,656; A: 0,619/0,653/0,652/0,680/0,677; controle 0,601–0,610
- **Estado:** concluído
- **Decisão:** B e A trocam de lugar; SPiFiL +0,04 a +0,09 sobre controle

## Fontes
- [`results_log.md`](../multi_dataset/results_log.md)
- [`fila.md`](../multi_dataset/fila.md)
