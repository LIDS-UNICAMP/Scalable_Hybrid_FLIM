# leis_p1b_s43_jaci, leis_p1_s43_jaci — Leis de construção (largura por camada)

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `leis_p1b_s43_jaci`, `leis_p1_s43_jaci` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** idem, s43
- **O que testa:** bases menores, larvas e ovos
- **Como:** split 1 × s43
- **Bases:** Lsub, Esub, heggs, hlarv; Larvae, Eggs
- **Resultados:** —
- **Estado:** na fila (Jaci, fila3) / rodando
- **Decisão:** —

## Fontes
- [`leis_p0_p1.md`](fontes/leis_p0_p1.md)
- [`plano.md`](fontes/plano.md)
- [`relatorio_07_08nov.md`](../multi_dataset/relatorio_07_08nov.md)
- [`fila.md`](../multi_dataset/fila.md)

## Atualização (revisão de completude, 08/11)
- **08/11 20:29 UTC:** o sorteio 43 das bases menores entrou; **20:43 UTC** a proteção derrubou-o (GPU do Jaci a 95 GB; subiu ~30 GB em 14 min), nenhum dos 8 jobs tinha terminado; voltou ao começo da fila com vaga de 63 GB.
- Fonte extra: histórico da conversa (sessão 990a33ea, mensagens do assistente), 08/11 20:42–20:44 UTC.
