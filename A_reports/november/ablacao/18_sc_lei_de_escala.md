# sc_*, scbig_* (lei de escala largura × cabeça × profundidade) — Fase com backprop

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `disc_md_sc_w{1,2,4}_{c640,c1280,d4,d5,d6}_{spf,rand}`, `disc_md_scbig_w{1,2,4}_c2560_{spf,rand}` (runner `growth`)
- **Alternativa:** largura w1/w2/w4 × cabeça 640/1280/2560 × profundidade 4–6; SPiFiL contra aleatório
- **O que testa:** lei de escala (fase com backprop, pausada em 07/11 a pedido do usuário)
- **Como:** 6 famílias; leitura `gem_tta_pw`
- **Bases:** 6 famílias
- **Resultados:** só as réplicas w1 (helminth_eggs, split 1, 50%, L3): c1280 0,899; d6 0,876; d5 0,869; d4 0,861; c640 0,859; c2560 0,857; média 0,870, desvio 0,016, faixa 0,041 (mistura GPU e época da sonda)
- **Estado:** parado em 07/11 (era piloto; pesos em `/exps/moliveira/spifil_growth/disc_md_sc*`)
- **Decisão:** não entra na lei sem indicador de regime

## Fontes
- [`results_log.md`](../multi_dataset/results_log.md)
- [`fila.md`](../multi_dataset/fila.md)
- [`pausa_treinos.md`](../multi_dataset/pausa_treinos.md)
