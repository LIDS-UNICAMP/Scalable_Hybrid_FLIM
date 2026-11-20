# spfcombo_w (Jaci) — Whitening entre camadas

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `spfcombo_w` (Jaci) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** ZCA dos canais entre camadas (N5)
- **O que testa:** se branquear o mapa entre camadas ajuda
- **Como:** 3 splits × s42; 64/256; L3–L6; `fps@combo`, `cos@combo`; leitura `cat` e `cat_b`; 4 aleatórios com o mesmo whitening
- **Bases:** Esub, Lsub, Psub, aid, flowers, heggs
- **Resultados:** 256, L3, sem → com: Esub 0,821→0,873; Lsub 0,613→0,769; Psub 0,813→0,859; aid 0,714→0,732; flowers 0,588→0,643; heggs 0,931→0,944. Contra melhor aleatório (L6, 256): Esub +0,095, Lsub +0,22, Psub +0,10, aid +0,041, flowers +0,006
- **Estado:** concluído
- **Decisão:** entra na receita

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`relatorio_07nov.md`](../multi_dataset/relatorio_07nov.md)
- [`sol_auditoria.md`](../multi_dataset/sol_auditoria.md)
