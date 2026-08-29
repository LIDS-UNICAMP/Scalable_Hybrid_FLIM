# Como rodar os distills — os 24 experimentos, e quando usar cada um

> Escopo: a matriz **4 students x 3 losses x 2 estados do teacher**. Nao interpreta
> resultado. Irmao de [how_to_work_spifil.md](how_to_work_spifil.md), que cobre o outro
> experimento.

---

## A pergunta que a matriz responde

Tres eixos, um por dimensao do comando:

| eixo | valores | pergunta |
|---|---|---|
| **student** | `--distill_1..4` | quanta capacidade a cabeca de projecao precisa ter? |
| **loss** | `mse`, `cos`, `kd` | o aluno copia a POSICAO do embedding, a DIRECAO, ou o rotulo? |
| **teacher** | `frozen`, `unfrozen` | deixar o professor se mexer ajuda? |

### Os quatro students

Sao arquiteturas que ja existiam no repo; as flags so as nomeiam. Os numeros foram medidos
(encoder FLIM + cabeca, `requires_grad`) e conferem com `statistics/tools/compute_cost.csv`.

| flag | params | modulo | cabeca |
|---|---|---|---|
| `--distill_1` | 123.504 | `distillation_onelayer_module` | `OneLayer1x1ConvDistillationProjectionHead` |
| `--distill_2` | 402.544 | `distillation_twolayer_module` | `TwoLayer1x1ConvBN2dDistillationProjectionHead` |
| `--distill_3` | 615.024 | `distillation_onelayer_module` | `OneLayerConvDistillationProjectionHead` |
| `--distill_4` | 889.200 | `distillation_conv_module` | `ConvDistillationProjectionHead` |

A flag e VALIDADA contra o modulo: `--distill_2` no modulo errado recusa apontando o certo.
Ela tambem fixa o `--proj-kernel` do `d1`/`d3`, entao nao passe os dois.

**Os numeros acima sao do encoder `eggs`.** No protozoan a `layer2` tem 30 canais em vez de
32, o encoder cai de 59.504 para 55.902 e os totais viram 119.902 / 398.942 / 611.422 /
885.598. Como todo comando roda `--dataset all`, cada um treina os tres tamanhos.

### As tres losses

| tag | flags | o que minimiza |
|---|---|---|
| `mse` | `--fine_tune False --loss_mse` | `MSE(proj(enc(x)), teacher(x))` — distancia absoluta |
| `cos` | `--fine_tune False --loss_cos` | `1 - cos(...)` — so o angulo, escala livre |
| `kd` | `--fine_tune True` | `(1-a)*CE(rotulo) + a*KL(professor)` — supervisionado |

`kd` NAO leva flag de loss de embedding, e e o unico que constroi cabeca de classificacao.
Ele usa `--kd_alpha` / `--kd_temperature` (default 0.7 / 4.0) — **nao** `--alpha` /
`--temperature`, que sao de outro tipo (`hybrid`) e seriam ignoradas em silencio.

---

## Os 24 comandos

Um experimento por comando. Cada um varre os 3 datasets x 3 splits x 6 percentuais
internamente, um modelo na GPU por vez.

### distill 1 — 123.504 params · onelayer, proj-kernel 1 · GPU 0

```bash
tmux new -d -s distill1_all_mse_frozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_onelayer_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 0 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill1_all_mse_frozen --distill_1 --init_flim --fine_tune False --loss_mse --teacher_frozen --wandb 2>&1 | tee logs/distill1_all_mse_frozen.log"

tmux new -d -s distill1_all_mse_unfrozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_onelayer_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 0 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill1_all_mse_unfrozen --distill_1 --init_flim --fine_tune False --loss_mse --teacher_unfrozen --wandb 2>&1 | tee logs/distill1_all_mse_unfrozen.log"

tmux new -d -s distill1_all_cos_frozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_onelayer_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 0 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill1_all_cos_frozen --distill_1 --init_flim --fine_tune False --loss_cos --teacher_frozen --wandb 2>&1 | tee logs/distill1_all_cos_frozen.log"

tmux new -d -s distill1_all_cos_unfrozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_onelayer_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 0 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill1_all_cos_unfrozen --distill_1 --init_flim --fine_tune False --loss_cos --teacher_unfrozen --wandb 2>&1 | tee logs/distill1_all_cos_unfrozen.log"

tmux new -d -s distill1_all_kd_frozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_onelayer_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 0 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill1_all_kd_frozen --distill_1 --init_flim --fine_tune True --teacher_frozen --wandb 2>&1 | tee logs/distill1_all_kd_frozen.log"

tmux new -d -s distill1_all_kd_unfrozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_onelayer_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 0 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill1_all_kd_unfrozen --distill_1 --init_flim --fine_tune True --teacher_unfrozen --wandb 2>&1 | tee logs/distill1_all_kd_unfrozen.log"
```

### distill 2 — 402.544 params · twolayer · GPU 1

```bash
tmux new -d -s distill2_all_mse_frozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_twolayer_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 1 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill2_all_mse_frozen --distill_2 --init_flim --fine_tune False --loss_mse --teacher_frozen --wandb 2>&1 | tee logs/distill2_all_mse_frozen.log"

tmux new -d -s distill2_all_mse_unfrozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_twolayer_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 1 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill2_all_mse_unfrozen --distill_2 --init_flim --fine_tune False --loss_mse --teacher_unfrozen --wandb 2>&1 | tee logs/distill2_all_mse_unfrozen.log"

tmux new -d -s distill2_all_cos_frozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_twolayer_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 1 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill2_all_cos_frozen --distill_2 --init_flim --fine_tune False --loss_cos --teacher_frozen --wandb 2>&1 | tee logs/distill2_all_cos_frozen.log"

tmux new -d -s distill2_all_cos_unfrozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_twolayer_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 1 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill2_all_cos_unfrozen --distill_2 --init_flim --fine_tune False --loss_cos --teacher_unfrozen --wandb 2>&1 | tee logs/distill2_all_cos_unfrozen.log"

tmux new -d -s distill2_all_kd_frozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_twolayer_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 1 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill2_all_kd_frozen --distill_2 --init_flim --fine_tune True --teacher_frozen --wandb 2>&1 | tee logs/distill2_all_kd_frozen.log"

tmux new -d -s distill2_all_kd_unfrozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_twolayer_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 1 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill2_all_kd_unfrozen --distill_2 --init_flim --fine_tune True --teacher_unfrozen --wandb 2>&1 | tee logs/distill2_all_kd_unfrozen.log"
```

### distill 3 — 615.024 params · onelayer, proj-kernel 3 · GPU 2

```bash
tmux new -d -s distill3_all_mse_frozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_onelayer_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 2 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill3_all_mse_frozen --distill_3 --init_flim --fine_tune False --loss_mse --teacher_frozen --wandb 2>&1 | tee logs/distill3_all_mse_frozen.log"

tmux new -d -s distill3_all_mse_unfrozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_onelayer_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 2 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill3_all_mse_unfrozen --distill_3 --init_flim --fine_tune False --loss_mse --teacher_unfrozen --wandb 2>&1 | tee logs/distill3_all_mse_unfrozen.log"

tmux new -d -s distill3_all_cos_frozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_onelayer_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 2 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill3_all_cos_frozen --distill_3 --init_flim --fine_tune False --loss_cos --teacher_frozen --wandb 2>&1 | tee logs/distill3_all_cos_frozen.log"

tmux new -d -s distill3_all_cos_unfrozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_onelayer_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 2 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill3_all_cos_unfrozen --distill_3 --init_flim --fine_tune False --loss_cos --teacher_unfrozen --wandb 2>&1 | tee logs/distill3_all_cos_unfrozen.log"

tmux new -d -s distill3_all_kd_frozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_onelayer_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 2 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill3_all_kd_frozen --distill_3 --init_flim --fine_tune True --teacher_frozen --wandb 2>&1 | tee logs/distill3_all_kd_frozen.log"

tmux new -d -s distill3_all_kd_unfrozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_onelayer_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 2 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill3_all_kd_unfrozen --distill_3 --init_flim --fine_tune True --teacher_unfrozen --wandb 2>&1 | tee logs/distill3_all_kd_unfrozen.log"
```

### distill 4 — 889.200 params · conv · GPU 3

```bash
tmux new -d -s distill4_all_mse_frozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_conv_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 3 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill4_all_mse_frozen --distill_4 --init_flim --fine_tune False --loss_mse --teacher_frozen --wandb 2>&1 | tee logs/distill4_all_mse_frozen.log"

tmux new -d -s distill4_all_mse_unfrozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_conv_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 3 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill4_all_mse_unfrozen --distill_4 --init_flim --fine_tune False --loss_mse --teacher_unfrozen --wandb 2>&1 | tee logs/distill4_all_mse_unfrozen.log"

tmux new -d -s distill4_all_cos_frozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_conv_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 3 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill4_all_cos_frozen --distill_4 --init_flim --fine_tune False --loss_cos --teacher_frozen --wandb 2>&1 | tee logs/distill4_all_cos_frozen.log"

tmux new -d -s distill4_all_cos_unfrozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_conv_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 3 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill4_all_cos_unfrozen --distill_4 --init_flim --fine_tune False --loss_cos --teacher_unfrozen --wandb 2>&1 | tee logs/distill4_all_cos_unfrozen.log"

tmux new -d -s distill4_all_kd_frozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_conv_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 3 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill4_all_kd_frozen --distill_4 --init_flim --fine_tune True --teacher_frozen --wandb 2>&1 | tee logs/distill4_all_kd_frozen.log"

tmux new -d -s distill4_all_kd_unfrozen "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.distillation_conv_module --dataset all --split 1,2,3 --percentage 1,5,25,50,75,100 --gpu 3 --cpus-per-experiment 4 --max-concurrent-per-gpu 1 --num-workers 4 --run-name distill4_all_kd_unfrozen --distill_4 --init_flim --fine_tune True --teacher_unfrozen --wandb 2>&1 | tee logs/distill4_all_kd_unfrozen.log"
```
---

## As flags de recurso — o que cada uma controla

Tudo no argparse, nada em variavel de ambiente: a linha diz sozinha onde e com quanto roda.

| flag | controla | default |
|---|---|---|
| `--gpu N` | `devices=[N]` no Trainer | `None` = primeira GPU visivel |
| `--cpus-per-experiment N` | teto de threads de CPU do processo | `None` = o do ambiente |
| `--max-concurrent-per-gpu N` | **celulas da grade em voo ao mesmo tempo** | `1` = uma por vez |
| `--num-workers N` | processos do dataloader | `4` |

### `--max-concurrent-per-gpu` e o que voce provavelmente quer

Um comando com `--dataset all --split 1,2,3 --percentage 1,5,25,50,75,100` e **54 treinos**.
Com `N=1` eles rodam em SERIE, no proprio processo: um modelo na GPU por vez. E o default, e
e o que se quer para nao disputar memoria.

Com `N>1` cada celula vira um subprocesso e N ficam em voo. Nesse caminho o
`--cpus-per-experiment` e DIVIDIDO entre os filhos (12 com N=3 da 4 em cada), senao N filhos
com o teto cheio multiplicariam a carga por N.

### Sobre o teto de CPU

O docstring de `scripts/spifil_growth_loop.py` afirma que esse teto so da por variavel de
ambiente, porque o OpenBLAS le `OMP_NUM_THREADS` no import. Isso vale para a rota ingenua,
mas aqui a flag funciona: o `threadpoolctl` redimensiona pool ja criado. Sao tres frentes —
`threadpool_limits` (pools BLAS/OpenMP do import), `torch.set_num_threads` (intra-op do
torch) e `os.environ` (workers e subprocessos, que nascem depois).

---

## Quanto tempo, e como dimensionar

Cada comando = 54 treinos em serie. Conte com **muitas horas** por comando.

Se isso for inviavel, o caminho e **reduzir a grade**, nao aumentar o paralelismo:
`--percentage 5,50` (os dois percentuais que o resto da tese usa) derruba de 54 para 18
celulas, um terco do tempo.

Para dimensionar o teto de CPU: `nucleos totais / processos simultaneos`. Com 96 logicos e um
experimento sozinho, 4 e conservador de proposito — a maquina costuma ja estar com outros
treinos. Confira com `cat /proc/loadavg` antes.

**O pico de memoria de GPU por student NAO foi medido.** Se algo estourar, o
`--max-concurrent-per-gpu` e onde se ajusta sem tocar em mais nada.

---

## Tres armadilhas que mudam a leitura do resultado

**1. `--teacher_unfrozen` treina o teacher de verdade.** Nao ha `.detach()` no caminho do
embedding (`src/models/distillation.py`, as tres losses), e o teacher entra no otimizador com
`lr x 0.1`. E deliberado: detachar tornaria a flag um no-op na faixa `--fine_tune False`.
Mas medido, com `--loss_cos` o gradiente que chega no teacher e ~3x o que chega na projetora
do aluno, o que admite **colapso trivial** — aluno e professor convergindo um para o outro.
Se um braco `cos_unfrozen` der loss otima e kNN ruim, e o primeiro lugar para olhar.

**2. O MLP vestigial entra no otimizador.** O `self.student` carrega um
`multi_layer_perceptron` de 4.825.600 params (`src/models/lejepa_flim.py:77`) que nunca e
usado no caminho KD, mas esta com gradiente ligado. A tabela conta 123k-889k; o otimizador ve
~4,9M. Afeta memoria e tempo dos 24.

**3. `--num-classes` e ignorada nestes tres modulos** — `num_classes` e resolvido por dataset
dentro do laco (eggs=9, larvae=2, protozoan=7, conferindo com `scripts/constants.py:101`). A
flag continua declarada so porque os lancadores Ray a passam.

---

## Depois do treino

```bash
python -m src.evaluate.svm_distill_with_projection --run-filter distill1_all --output-csv svm_distill1_all_results
```

Sem `--output-csv` o nome sai `svm_proj1280_<run-filter>_results.csv`. A fila completa de
avaliacao esta em `scripts/regen_svm_queue.sh`.

---

## Nao confundir com os lancadores Ray

`scripts/distillation_conv_ray.py` e o que gerou tudo que esta em `results/` hoje: ele varre a
grade por `--proj-type` e faz o pinning por `CUDA_VISIBLE_DEVICES`. Os comandos deste
documento sao a rota do MODULO DIRETO, que existe para experimento controlado — a matriz de
24 celulas com flags explicitas na linha.

As duas rotas coexistem de proposito: sem `--gpu`, o `devices=1` pega a primeira GPU visivel,
que e a que o Ray pinou por ambiente.

**Pendencia conhecida:** `scripts/distillation_conv_ray.py:589` acrescenta `--proj-type` a
todo comando, e o `distillation_conv_module` nunca declarou essa flag. Como o roteamento
manda `conv_next_layers` para ele, essa rota morre com `unrecognized arguments`. E anterior a
esta matriz e nao foi corrigido — o conserto e uma linha, no lancador ou no modulo.
