# gauss (Jaci) — Saliência

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `gauss` (Jaci) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** peso por gaussiana ajustada ao mapa (k = 1, 2, 3)
- **O que testa:** gaussiana contra corte 0,5
- **Como:** 3 splits × s42; 64/256; L6; `fpsw@mapa_gauss*`, `fps@foco_gauss*`; 4 aleatórios
- **Bases:** 12 bases
- **Resultados:** rodando
- **Estado:** rodando
- **Decisão:** —

## Fontes
- [`saliencia.md`](../multi_dataset/saliencia.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`fila.md`](../multi_dataset/fila.md)

## Atualização (revisão de completude, 08/11)
- **Origem:** pedido do usuário (08/11 13:11 UTC): trocar o corte 0,5 por um mapa gaussiano, forte onde o objeto está e caindo até a borda; se não coubesse no SPiFiL sem backprop, guardar para o treino com backprop. Coube: implementado (`gauss()` em `eval/spf_grade.py`, chave `pesos.gauss`; centro e covariância saem do mapa `gmr`; teste com objeto sintético deu peso ~1 no meio e 0,001 na borda).
- **Andamento:** 14 de 72 jobs (18:29 UTC); 20 de 72 (20:42 UTC). Sem número ainda.
- Fontes extras: [`fila.md`](../multi_dataset/fila.md); histórico da conversa (sessão 990a33ea, mensagens do assistente).
