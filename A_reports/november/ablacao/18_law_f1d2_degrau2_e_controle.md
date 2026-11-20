# law_f1d2 degrau 2 e law_ctl_rand (controle aleatório) — Fase com backprop

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `disc_md_law_f1d2_*` (36 YAMLs), `disc_md_law_ctl_rand*` (incl. s43/s44); SVM `disc_md_law_eval_f1d2`, `disc_md_law_eval_ctl`, `_ctl2`
- **Alternativa:** 5 perdas finalistas + só reconstrução, 3 splits; aleatório com mesma forma e largura real
- **O que testa:** SPiFiL contra aleatório em B0 (fase com backprop, pausada em 07/11 a pedido do usuário)
- **Como:** B0, 60 épocas por estágio; flowers, heggs, hlarv, Eggs
- **Bases:** flowers, heggs, hlarv, Eggs
- **Resultados:** controle B0 (s42, 3 splits): aleatório − SPiFiL: flowers +0,004; heggs +0,001; hlarv +0,026; agrupado +0,010 (erro 0,009, 9 pares). Degrau 2: sem número
- **Estado:** controle s42 concluído; degrau 2 e controle cheio parciais (pendentes em `/exps/moliveira/queues/law_pendentes.txt`)
- **Decisão:** motivo da pausa: o SPiFiL não saiu na frente do aleatório; antes de SSL e lei de escala, melhorar o SPiFiL

## Fontes
- [`results_log.md`](../multi_dataset/results_log.md)
- [`fila.md`](../multi_dataset/fila.md)
- [`pausa_treinos.md`](../multi_dataset/pausa_treinos.md)
