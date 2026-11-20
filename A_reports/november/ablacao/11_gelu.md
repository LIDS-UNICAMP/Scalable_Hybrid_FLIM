# gelu (Athena) — Polaridade e ativação

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `gelu` (Athena) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** GELU vs ReLU na receita
- **O que testa:** ativação
- **Como:** 3 splits × s42; 256; L6; `fps`, `fpsw@mapa_w0.5_1`; `orth`, `random_z` com GELU
- **Bases:** Larvae, Eggs, Protozoan, aid, flowers, rs
- **Resultados:** rodando
- **Estado:** rodando
- **Decisão:** —

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)

## Atualização (revisão de completude, 08/11)
- **Origem:** pedido do usuário (08/11 16:52 UTC): testar GELU no melhor modelo.
- **Andamento:** 2 de 36 enviados (18:29 UTC). Sem número ainda.
- Fonte extra: histórico da conversa (sessão 990a33ea, mensagens do assistente).
