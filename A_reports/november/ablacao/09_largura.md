# largura (Jaci) — Largura, profundidade, kernel e tamanho da entrada

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `largura` (Jaci) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** 32/64/128/256
- **O que testa:** coco vs Protozoan lado a lado
- **Como:** 3 splits × s42/43; `fps`, `fpsw@mapa_w0.5_1`; 4 aleatórios
- **Bases:** coco, Protozoan
- **Resultados:** rodando
- **Estado:** rodando
- **Decisão:** —

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`pool_foco.md`](fontes/pool_foco.md)
- [`results_log.md`](../multi_dataset/results_log.md)
- [`law_size_superpixels.md`](../multi_dataset/law_size_superpixels.md)

## Atualização (revisão de completude, 08/11)
- **Origem:** pedido do usuário (08/11 12:47–12:50 UTC) depois do coco preferir 64 filtros: achar a relação entre filtros e tamanho da imagem. YAML `disc_md_largura_jaci.yaml` (criado no Jaci a partir do `sol_wsal`, trazido ao repositório).
- **Andamento:** 2 de 12 jobs (18:29 e 20:42 UTC). Sem número ainda.
- Fonte extra: histórico da conversa (sessão 990a33ea, mensagens do assistente).
