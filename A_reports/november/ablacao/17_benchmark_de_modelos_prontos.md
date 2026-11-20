# Benchmark de modelos prontos — Benchmark

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** Benchmark de modelos prontos (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** 12 modelos (Athena), 6 (Jaci)
- **O que testa:** régua de cima
- **Como:** —
- **Bases:** —
- **Resultados:** DINOv2 já medido (ex.: flowers 0,973, rs 0,708)
- **Estado:** na fila (por último)
- **Decisão:** —

## Fontes
- [`relatorio_07_08nov.md`](../multi_dataset/relatorio_07_08nov.md)
- [`results_log.md`](../multi_dataset/results_log.md)

## Atualização (revisão de completude, 08/11)
- **Lista (18 modelos congelados + SVM, 14 bases):** Athena (12): AIMv2 large 224/336/448 px; DINOv2 small, base, large (518 px); ConvNeXt 22k base, large, xlarge; I-JEPA vith14 1k, vith16 1k (448 px), vith14 22k. Jaci (6): `ijepa_vitg16_22k` (224 px), `dinov2_giant` (518 px), `convnext_xxlarge_clip` (256 px), `aimv2_huge` 224/336/448 px.
- **Agenda:** por último nas duas máquinas, só depois da fila do SPiFiL de cada uma (sessão `bench_athena` armada; fila do Jaci armada). Teto no Athena corrigido de 36 GB para 15 GB por GPU. Carregadores validados nas duas máquinas (pesos: Athena 15 GB em `/exps/moliveira/hf_cache`; Jaci 23 GB em `/dados/home/moliveira/exps/hf`).
- **Pendente do usuário:** `aimv2_large_native` (224 px, tamanho original ou pular).
- Benchmark da fase 1 (4 modelos, já medido): [`17_benchmark_fase1_4_modelos.md`](17_benchmark_fase1_4_modelos.md).
- Fonte extra: histórico da conversa (sessão 990a33ea, mensagens do assistente), 08/11 15:12–15:23 UTC.
