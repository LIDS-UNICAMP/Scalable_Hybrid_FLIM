# law_f0 (ruído entre sementes) — Fase com backprop

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `disc_md_law_f0_s{42,43,44}_*`; SVM `disc_md_law_eval_f0`
- **Alternativa:** 3 sementes por célula
- **O que testa:** régua de ruído para a lei (fase com backprop, pausada em 07/11 a pedido do usuário)
- **Como:** B0, Weak-SIGReg 0,01, 60 épocas por estágio; 3 splits
- **Bases:** flowers, heggs, hlarv
- **Resultados:** κ médio / desvio entre sementes / entre splits: flowers 0,524 / 0,026 / 0,051; heggs 0,938 / 0,004 / 0,005; hlarv 0,959 / 0,016 / 0,004; agrupado 0,018
- **Estado:** concluído
- **Decisão:** pendente (flowers passa de 0,02, onde o plano mandava tornar o sorteio de imagens um fator)

## Fontes
- [`results_log.md`](../multi_dataset/results_log.md)
