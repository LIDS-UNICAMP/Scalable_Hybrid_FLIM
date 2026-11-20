# Tamanho × arquitetura (só leitura) — Largura, profundidade, kernel e tamanho da entrada

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** Tamanho × arquitetura (só leitura) (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** largura e profundidade vs tamanho nativo e nº de treino
- **O que testa:** imagem pequena pede rede estreita?
- **Como:** `sol_wsal`, `spfdeep_esp`, `r4`, coco
- **Bases:** 12 + coco
- **Resultados:** 256 vence em 12 de 12 (Protozoan, 64 px, +0,118); ρ tamanho × largura +0,41 (p 0,17); ganho L6−L3 vs nº de treino ρ +0,67 (p 0,02)
- **Estado:** concluído
- **Decisão:** 256 e L6; pouco treino → L3–L4

## Fontes
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`spifil_vs_aleatorio.md`](../multi_dataset/spifil_vs_aleatorio.md)
- [`pool_foco.md`](fontes/pool_foco.md)
- [`results_log.md`](../multi_dataset/results_log.md)
- [`law_size_superpixels.md`](../multi_dataset/law_size_superpixels.md)
