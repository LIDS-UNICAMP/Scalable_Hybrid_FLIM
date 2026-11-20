# Benchmark da fase 1 (4 modelos) — Benchmark

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** benchmark do prompt das leis (Estágio 1, K1–K5); CSVs em `results/benchmark/*.csv`
- **Alternativa:** Pico, I-JEPA g/16, DINOv2 giant, AIMv2 3B (pesos oficiais do Hugging Face)
- **O que testa:** régua de cima a 50%
- **Como:** κ de validação a 50%, média de 3 splits; 6 frações medidas
- **Bases:** 12 bases
- **Resultados:** DINOv2 giant: Eggs 0,980; Esub 0,989; Larvae 0,968; Lsub 0,981; Protozoan 0,939; Psub 0,945; aid 0,970; flowers 0,973; heggs 0,987; hlarv 0,994; pcysts 0,966; rs 0,708. AIMv2 3B: aid 0,981, flowers 0,978, rs 0,688 (heggs, hlarv, pcysts parciais). I-JEPA g/16: rs 0,603, flowers 0,937. Pico: rs 0,601, flowers 0,888
- **Estado:** concluído (DINOv3 pendente do login do usuário)
- **Decisão:** DINOv2 giant é a régua de cima usada em todo o catálogo

## Fontes
- [`results_log.md`](../multi_dataset/results_log.md)
- [`pausa_treinos.md`](../multi_dataset/pausa_treinos.md)
- [`CONTEXT.md`](../multi_dataset/CONTEXT.md)
