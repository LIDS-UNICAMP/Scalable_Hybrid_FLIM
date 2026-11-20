# spf_r4 (Jaci) — Largura, profundidade, kernel e tamanho da entrada

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `spf_r4` (Jaci) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** kernel 3/5/7; 128/200 px; 4 camadas; 32–256 filtros
- **O que testa:** eixos que faltavam
- **Como:** 3 splits × s42
- **Bases:** mesmas 5
- **Resultados:** kernel 3 e entrada 200 ganham ou empatam em 4 e 5 de 5; largura cresce monotônica
- **Estado:** parcial (58/60)
- **Decisão:** kernel 3–5

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`pool_foco.md`](fontes/pool_foco.md)
- [`results_log.md`](../multi_dataset/results_log.md)
- [`law_size_superpixels.md`](../multi_dataset/law_size_superpixels.md)

## Atualização (revisão de completude, 08/11)
- **Job travado (08/11, 20:42 UTC):** o último job (`flowers`, split 1, segunda metade) roda há 18h30 com 1 núcleo e o CSV não muda desde a meia-noite (horário do Athena); a outra metade fechou em 1h40. Ele segura a fila: o `r4` não fecha (59 de 60) e o `r5` não começa. Opções dadas: (1) matar só esse processo (recomendado; refaz depois a partir das 69 linhas salvas) ou (2) deixar. **Aguarda decisão do usuário.**
- Fonte extra: histórico da conversa (sessão 990a33ea, mensagens do assistente), 08/11 20:42–21:15 UTC.
