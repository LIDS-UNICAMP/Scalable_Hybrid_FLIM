# Pausa dos treinos (07/11/2026)

Motivo: o SPiFiL nao saiu na frente do aleatorio. Antes de SSL e lei de escala, melhorar o SPiFiL. Pausa pedida pelo usuario.

## Estado salvo
- Plano: plano/PLANO_FINAL.md; fila: fila.md; resultados: results_log.md.
- Pesos: /exps/moliveira/spifil_growth/ (disc_md_law_*, disc_md_base*, disc_md_sc_*, disc_md_x*, disc_md_l1fast).
- Fila da lei: F0, F1 degrau 1 e controle B0 completos, com SVM (results/svm_disc_md_law_eval_*.csv). F1 degrau 2 e controle cheio parciais: pendentes em /exps/moliveira/queues/law_pendentes.txt.
- Linha de base: bracos prontos em /exps/moliveira/queues/base_prontos.txt.
- Benchmark: Pico, I-JEPA g/16, DINOv2 giant e AIMv2 3B nas 6 fracoes; DINOv3 pendente de login.

## Para retomar
Export: LD_LIBRARY_PATH do env, RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0, PYTHONPATH=/exps/moliveira/queues/gpucap, GPU_CAP_MIB=5000 (640 px: 30000), PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
Listas por faixa: /exps/moliveira/queues/law_lanes/. Rodar so os YAMLs de law_pendentes.txt.
Atencao: skip: state nao sobrevive a kill: braco interrompido recomeca; confira "pulado(s)" e nao sobrescreva ckpt pronto.
Teto: 35 GB do moliveira por GPU, crispim preservado. best_by_probe false nos treinos novos.

## Por que parou
Em B0 sem resposta clara: aleatorio menos SPiFiL +0,010 (erro 0,009). Em flowers, sem treino, SPiFiL perde do aleatorio normalizado (random_z) nas 8 celulas de k=1 e 2; mais imagens por classe ate 20 nao ajudam com o seletor atual.
