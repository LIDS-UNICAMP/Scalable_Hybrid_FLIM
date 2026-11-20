# disc_md_ham_jaci — Bases novas

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `disc_md_ham_jaci` (YAMLs em `experiments/autoencoder/disc_md_*.yaml`; código em `eval/spf_grade.py`, `eval/alt_encoders.py`, `saliency/`, conforme o caso)
- **Alternativa:** receita vs `fps`
- **O que testa:** base nova (dermatologia)
- **Como:** 3 splits × 3 sorteios; 256; `cat`
- **Bases:** HAM10000
- **Resultados:** L6 receita κ 0,461 ± 0,009, `fps` 0,468 ± 0,021 (pareado −0,007 ± 0,026); melhor aleatório 0,340; parcial: F1 0,518/0,532, acc 0,485/0,489
- **Estado:** concluído
- **Decisão:** empate; SPiFiL +0,12 sobre aleatório

## Fontes
- [`fila.md`](../multi_dataset/fila.md)
- [`decisoes_spifil.md`](../multi_dataset/decisoes_spifil.md)
- [`relatorio_07_08nov.md`](../multi_dataset/relatorio_07_08nov.md)
- [`pool_foco.md`](fontes/pool_foco.md)
