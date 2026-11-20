# base_* (linha de base, receita oficial) — Fase com backprop

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `disc_md_base_s200`, `disc_md_base_s320`, `disc_md_base_s640`, `disc_md_base_dedup_s200` (runner `growth`)
- **Alternativa:** receita oficial completa nos 12 datasets (raw, sem máscara)
- **O que testa:** linha de base (fase com backprop, pausada em 07/11 a pedido do usuário)
- **Como:** 4 camadas 32/48/64/1280, 200 épocas, 50 superpixels, 3 splits; 200 px (helminth_eggs, pcysts sem cópias, Protozoan, 3 subgrupos, hlarv), 320 px (Eggs, flowers), 640 px (Larvae, aid, rs); 36 runs
- **Bases:** 12 bases
- **Resultados:** — (braços prontos listados em `/exps/moliveira/queues/base_prontos.txt`; sem número nas fontes)
- **Estado:** pausado
- **Decisão:** —

## Fontes
- [`fila.md`](../multi_dataset/fila.md)
- [`pausa_treinos.md`](../multi_dataset/pausa_treinos.md)
