# R_yamls: ablacao de tamanho em YAML (06/11)

Worktree /dados/home/moliveira/Scalable_Hybrid_FLIM_discovery. Nenhum .py editado, nada lancado, nada commitado.

## 1. YAMLs
- 34 de treino + 2 opcionais (Eggs 896): experiments/autoencoder/disc_md_res_*.yaml (36 arquivos).
- 10 de eval: experiments/autoencoder/disc_md_eval_res_s<NNNN>_val_gemttapw.yaml. Nome da secao 8 e dos irmaos, NAO o prefixo disc_md_res_ do pedido: se preferir o outro, e so renomear.
- Conferencia campo a campo contra res_yamls.tsv: 34 de 34 certos (nome, work_dir, wandb igual a X1, datasets, N, n, d, larguras, ipc, lote, acumulado, num_workers 8, GPU).
- GPU no YAML: 600 px para cima alterna 2/3 (5 e 5); abaixo alterna 0/1 (12 e 12).
- Opcionais: disc_md_res_s0896_n0128_d5_b, disc_md_res_s0896_n1004_d3_b. Fora das filas. Se o usuario quiser, somar Eggs em grid.datasets do eval s0896.

## 2. Validacao
- Treino: 36 de 36 passam preflight e --dry-run. accumulate_grad_batches aparece nos comandos de s1280 (9), b8a4 (9), s0896_b (15).
- Eval: 10 de 10 passam preflight (0 erros, 4 checagens puladas), mas o --dry-run dos 10 PARA em "nenhum job casou": ainda nao ha checkpoint. Esperado, nao e aprovacao.
- Teste real de 1 eval: s1280 apontado para os checkpoints curtos gerou 3 jobs (L1, L2, L3). stages com L4/L5 ausentes foi ignorado sem erro (eval.py:435), o que fecha o "nao testado" da secao 8. Os outros 9 nao tiveram esse teste.

## 3. Filas
- /exps/moliveira/queues/res_lote1.txt: 20 YAMLs. /exps/moliveira/queues/res_lote2.txt: 14. Ordem por N crescente: controles de lote antes de 896 e 1280.
- Cuidado: --gpu-ids sobrepoe o YAML. Linha de 600 px para cima so nas GPUs 2 e 3.
- Cuidado: as lanes atuais usam GPU_CAP_MIB=12000. Isso derruba 1280 (treino pede ~29,7 GB).

## 4. Medida acima de 640: INCOMPLETA
Pedido: 3 pontos. Medido: parte de 1. O prazo de 25 min do coordenador cortou a rodada as 05:48.

| ponto | estado | pico |
|---|---|---|
| Larvae 1280 n=128 L5, lote 8x4 | L1, L2, L3 completos; L4 cortado no meio; L5 nunca rodou | reservado 29.702-29.704, alocado 28.836-28.838 MiB; 0 OOM, 0 retries |
| Larvae 1280 n=2048 L3, lote 8x4 | NAO MEDIDO | - |
| Larvae 896 L5, lote 16x2 | NAO MEDIDO | - |

- Placa: pico 45.245 MiB (limite 46.000). moliveira: 40.246 (limite 41.000), somando lk_feats.py de outro agente, que dividia a GPU 2.
- Outros usuarios na placa: 4.938 a 8.698 MiB. O processo do crispim (1512118) segue vivo. Nao identifiquei quem subiu os 3,7 GB.
- Modelo da secao 6: 27.250 alocado; medido 28.838 (+6%).
- Inferencia, nao medida: o pico igual em L1, L2 e L3 indica que quem manda e o pedaco da sonda (64 imagens), nao a profundidade. Entao L5 deve dar parecido. n=2048 somaria so ~800 MiB no worker Ray, e 896 deve ficar abaixo de 1280.
- Nenhum YAML teve lote alterado: nenhum ponto estourou.

Diferencas da rodada: limit_train_batches 16 (b2_real usava 40), para caber no prazo. 1 split, 2 epocas, warmup 1. Comecou com lk_feats.py ainda na GPU 2, por ordem do coordenador. O codigo ja tinha a correcao float16 da sonda.
Tempo: ~6 a 7 min por estagio a 1280, com o encurtamento.

Falta: medir 1280 n=2048 e 896 L5, e L4/L5 de 1280, antes de esses YAMLs chegarem na fila. Estao no fim de res_lote1.txt.

## 5. Sonda (best_by_probe)
Os 36 YAMLs de treino estao com `model.init_args.best_by_probe: true`, como em X1 (epoch_probe: gem). Trocar numa linha, a partir do worktree:

    sed -i 's/best_by_probe: true/best_by_probe: false/' experiments/autoencoder/disc_md_res_*.yaml

Limpeza: /exps/moliveira/tmp_res apagada. O --dry-run de eval com checkpoints ESCREVE jobs/*.yaml no work_dir, apesar do que diz launch.py:86; apaguei os 3 que meu teste criou em /exps/moliveira/eval/disc_md_eval_res_s1280_val_gemttapw. Dados crus da medida: law/res_meas.tsv.
