# porclasse_jaci — Imagens por classe

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `porclasse_jaci` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** idem, com a receita e `fps`
- **O que testa:** grade
- **Como:** 3 splits × s42; 64/256; 4 braços; 4 aleatórios
- **Bases:** flowers, Lsub, rs, Esub, Psub, aid, coco
- **Resultados:** —
- **Estado:** rodando (Jaci)
- **Decisão:** —

## Fontes
- [`results_log.md`](../multi_dataset/results_log.md)
- [`fila.md`](../multi_dataset/fila.md)

## Atualização (revisão de completude, 08/11)
- **Origem:** pedido do usuário (08/11 01:50 UTC): bases de alta variabilidade precisam de 10 a 15 imagens por classe para compor os kernels; e (11:32 UTC) "esse é o próximo passo". Relançado no Jaci com teto depois de passar de 15 GB por GPU no Athena.
- **Andamento:** 5 de 21 (18:29 UTC); 6 de 21 (20:42 UTC). Sem número ainda.
- Fonte extra: histórico da conversa (sessão 990a33ea, mensagens do assistente).
