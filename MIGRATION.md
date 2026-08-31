# MIGRATION.md — o mapa do antes e do depois

**Para que serve este arquivo.** Quando um comando que você sabia de cor parar de
funcionar, a resposta está aqui. Este documento não explica a refatoração — ele
traduz. Você chega com o comando antigo, o caminho antigo ou o `class_path`
antigo, e sai com o equivalente de hoje.

**Data do levantamento:** 2026-08-30. Base: commit `f43435d chore: snapshot do
estado ANTES da refatoracao estrutural`.

---

## 0. As três coisas que você precisa saber antes de qualquer tabela

### 0.1 NADA foi commitado, NADA foi apagado

Toda a refatoração está no *index* e no *worktree*. `git log` não mostra quase
nada (só renomeações de `docs/` e `tools/` feitas em commits anteriores); o grosso
está em `git status`. Se quiser ver as renomeações você mesmo:

    git diff --cached -M20% --name-status --diff-filter=R

O `-M20%` é obrigatório. Com o limiar padrão (50%) o git deixa de enxergar
`configs/model/lejepa_custom_cnn.yaml -> configs/model/lejepa/_variants/custom_cnn.yaml`
como renomeação: o arquivo ganhou um cabeçalho de quarentena de 44 linhas e a
similaridade caiu para **22%** (`R022`). Sem `-M20%` ele aparece como
delete + add e você perde o rastro.

Nenhum arquivo antigo foi removido. `src/`, `scripts/`, `tools/`,
`check_experiments/`, `statistics/tools/` e `analysis_flim_distill/` continuam no
disco. A deleção de qualquer um deles depende de aprovação sua — a lista está na
seção 9.

### 0.2 A árvore nova, em uma olhada

| pacote | .py | linhas | o que é |
|---|---:|---:|---|
| `core/` | 31 | 3.100 | blocos, dados, métricas, constantes, W&B |
| `flim/` | 9 | 2.248 | FLIM/SPiFiL (arquitetura, crescimento, enxerto) |
| `methods/` | 41 | 6.124 | os LightningModules: `autoencoder`, `classification`, `distillation`, `lejepa` |
| `eval/` | 23 | 9.346 | sondas e avaliadores (inclui `eval/svm_variants/`) |
| `experiments/` | 26 | 7.280 | motor de grade Ray, `gen_configs`, `ckpt/`, `oneoff/` |
| `analysis/` | 35 | 9.342 | estatística, plots, checagens, análise de destilação |
| `train.py` | 1 | — | o antigo `src/main.py`, agora na raiz |

### 0.3 As TRÊS convenções de linha de comando da árvore nova

Isto é o que mais confunde na hora de rodar. Não é uma convenção só:

| onde | como se passa parâmetro | exemplo |
|---|---|---|
| `eval/` e `eval/svm_variants/` | **as flags continuam funcionando**, traduzidas por `eval.svm.cli_kwargs` (`eval/svm.py:75`) | `python -m eval.mlp --config x.yaml --ckpt-selection=best` |
| `analysis/`, `experiments/ckpt/`, `experiments/gen_configs.py` | **as flags SUMIRAM**. O `__main__` chama a função sem argumento; o parâmetro virou argumento nomeado de função | `python -c "from analysis.stats.wilcoxon_acc import wilcoxon_acc; wilcoxon_acc(alpha=0.01)"` |
| `experiments/oneoff/` | **argparse mantido, igual ao antigo** | `python -m experiments.oneoff.run_missing_mlp --mode freeze` |

> **Forma normal de uso.** Para rodar com os defaults — que e o caso comum — use
> `python -m <modulo>`, ex.: `python -m analysis.stats.wilcoxon_acc`. O `__main__`
> de cada script chama a funcao sem argumento, entao isso ja roda. O `python -c`
> das tabelas abaixo so e necessario para SOBREPOR um parametro pontualmente; para
> os dois scripts com muitos parametros (`heatmap_stages`, `plot_continuity`) a
> forma prevista e o YAML de analise (`config='meu.yaml'`), nao o `python -c`.



Duas exceções dentro da segunda linha:

* `experiments/ckpt/validate_stripped.py:48` continua posicional:
  `python -m experiments.ckpt.validate_stripped <orig.ckpt> <stripped.ckpt>`.
* `analysis/activations/heatmap_stages.py` e
  `analysis/plots/plot_continuity_spifil_hybrid.py` (14 e 16 parâmetros) aceitam
  um YAML pelo parâmetro `config=`, mas **não pela linha de comando** — só
  chamando a função.

Confirmado por `grep -rn "sys.argv" analysis/ --include=*.py` → vazio, e
`grep -rln argparse experiments/oneoff/` → 2 arquivos.

---

## 1. TABELA (a) — caminho antigo → caminho novo

Gerada de `git diff --cached -M20% --name-status --diff-filter=R` (232
renomeações staged) mais `git log --diff-filter=R --name-status -M20%` (as de
commits anteriores). As três linhas do `gen_configs.py` não saem do git (fusão
3→1) e estão marcadas.

### 1.1 Entrypoint e ambiente

| antigo | novo | git |
|---|---|---|
| `src/main.py` | `train.py` | R100 |
| `Dockerfile` | `env/Dockerfile` | R100 |
| `environment.yml` | `env/environment.yml` | R100 |
| `requirements.txt` | `env/requirements.txt` | R100 |
| `setup_env.sh` | `env/setup_env.sh` | R100 |

### 1.2 `src/evaluate/` → `eval/svm_variants/` (10 arquivos, 3.510 linhas)

Todos R100 (byte-idênticos no momento do `git mv`; os imports foram religados
depois).

| antigo | novo |
|---|---|
| `src/evaluate/eval_autoencoder.py` | `eval/svm_variants/eval_autoencoder.py` |
| `src/evaluate/eval_avg_pooling_48d.py` | `eval/svm_variants/eval_avg_pooling_48d.py` |
| `src/evaluate/eval_svm_flim_flatten.py` | `eval/svm_variants/eval_svm_flim_flatten.py` |
| `src/evaluate/svm_classification_flim.py` | `eval/svm_variants/svm_classification_flim.py` |
| `src/evaluate/svm_distill_with_projection.py` | `eval/svm_variants/svm_distill_with_projection.py` |
| `src/evaluate/svm_distillation.py` | `eval/svm_variants/svm_distillation.py` |
| `src/evaluate/svm_distillation_conv.py` | `eval/svm_variants/svm_distillation_conv.py` |
| `src/evaluate/svm_flim_residual.py` | `eval/svm_variants/svm_flim_residual.py` |
| `src/evaluate/svm_ijepa.py` | `eval/svm_variants/svm_ijepa.py` |
| `src/evaluate/svm_real_flim.py` | `eval/svm_variants/svm_real_flim.py` |

Dois destes têm história anterior, registrada no `git log`:
`tools/eval_avg_pooling_48d.py` → `src/evaluate/eval_avg_pooling_48d.py` (R058) e
`tools/eval_svm_flim_flatten.py` → `src/evaluate/eval_svm_flim_flatten.py` (R091),
commit `c814648`. Mais atrás ainda: `tools/eval_48d_norm_off.py` →
`tools/eval_avg_pooling_48d.py` (R095, commit `d198061`).

### 1.3 `src/evaluate/` → `eval/` (os 13 com destino na árvore-alvo)

Estes **não** são `git mv` — são arquivos novos com origem preservada, porque o
conteúdo foi religado no mesmo passo. Fonte: relatório E-EVAL §3.

| antigo | novo | linhas |
|---|---|---:|
| `src/utils/evaluate.py` + `src/evaluate/svm.py` | `eval/svm.py` | 842 |
| `src/evaluate/mlp.py` | `eval/mlp.py` | 687 |
| `src/evaluate/ray_mlp.py` + `ray_mlp_queue.py` (a parte **não-ray**) | `eval/ray_queue.py` | 478 |
| `src/evaluate/unified_eval.py` | `eval/unified_eval.py` | 909 |
| `src/evaluate/wandb_resolver.py` | `eval/wandb_resolver.py` | 640 |
| `src/evaluate/eval_plotter.py` | `eval/eval_plotter.py` | 767 |
| `src/evaluate/eval_growth_stages.py` | `eval/growth_stages.py` | 589 |
| `src/evaluate/tsne_analysis.py` | `eval/tsne.py` | 389 |
| `src/evaluate/classical_classifiers.py` | `eval/classical_classifiers.py` | 268 |
| `src/evaluate/__main__.py` | `eval/__main__.py` | 63 |
| `src/evaluate/constants.py` (6 nomes) | `eval/constants.py` | 130 |

A parte **ray** de `ray_mlp.py`/`ray_mlp_queue.py` foi para
`experiments/ray/runners/eval.py`, `experiments/ray/execution_state.py` e
`experiments/ray/gpu_slot_scheduler.py`.

`eval/knn.py` **não existe** e não foi criado: o kNN já mora em dois lugares
(`src/evaluate/classical_classifiers.py:81-86` e
`core/mixins/knn_kappa_probe_mixin.py:75-114`) e criar o arquivo seria duplicar.

### 1.4 `tools/`, `check_experiments/`, `statistics/tools/`, `analysis_flim_distill/` → `analysis/` (29 renomeações)

| antigo | novo |
|---|---|
| `tools/heatmap_stages.py` | `analysis/activations/heatmap_stages.py` |
| `tools/analyze_firing_relu_vs_sigmoid.py` | `analysis/activations/relu_vs_sigmoid.py` |
| `tools/analyze_sigmoid_saturation.py` | `analysis/activations/saturation.py` |
| `tools/analyze_unit_activations.py` | `analysis/activations/unit_activations.py` |
| `tools/plot_comparacao_flim_protocolo.py` | `analysis/plots/plot_comparacao_flim_protocolo.py` |
| `tools/plot_continuity_spifil_hybrid.py` | `analysis/plots/plot_continuity_spifil_hybrid.py` |
| `tools/plot_partial_train_spifil_hybrid.py` | `analysis/plots/plot_partial_train_spifil_hybrid.py` |
| `tools/plot_sigmoid_saturation.py` | `analysis/plots/plot_sigmoid_saturation.py` |
| `src/utils/plot_svm_results.py` | `analysis/plots/plot_svm_results.py` |
| `tools/check_ckpt_slim.py` | `analysis/checks/check_ckpt_slim.py` |
| `tools/check_probe_matches_evaluator.py` | `analysis/checks/check_probe_matches_evaluator.py` |
| `tools/check_refactor_equivalence.py` | `analysis/checks/check_refactor_equivalence.py` |
| `tools/keep_only_best_ckpt.py` | `analysis/checks/keep_only_best_ckpt.py` |
| `check_experiments/__init__.py` | `analysis/checks/__init__.py` |
| `check_experiments/_common.py` | `analysis/checks/_common.py` |
| `check_experiments/check_ssl.py` | `analysis/checks/check_ssl.py` |
| `check_experiments/fine_results.py` | `analysis/checks/fine_results.py` |
| `check_experiments/fine_tune.py` | `analysis/checks/fine_tune.py` |
| `scripts/check_spifil_growth.py` | `analysis/checks/check_spifil_growth.py` |
| `analysis_flim_distill/aggregate_nonorm_compare.py` | `analysis/checks/aggregate_nonorm_compare.py` |
| `analysis_flim_distill/distill_destroys_flim.py` | `analysis/distill/distill_destroys_flim.py` |
| `analysis_flim_distill/embedding_analysis.py` | `analysis/distill/embedding_analysis.py` |
| `analysis_flim_distill/ruler_mismatch.py` | `analysis/distill/ruler_mismatch.py` |
| `statistics/tools/measure_compute_cost.py` | `analysis/stats/compute_cost.py` |
| `statistics/tools/wilcoxon_acc.py` | `analysis/stats/wilcoxon_acc.py` |
| `statistics/tools/wilcoxon_equivalence.py` | `analysis/stats/wilcoxon_equivalence.py` |
| `statistics/tools/wilcoxon_f1.py` | `analysis/stats/wilcoxon_f1.py` |
| `statistics/tools/wilcoxon_flim_init.py` | `analysis/stats/wilcoxon_flim_init.py` |
| `statistics/tools/wilcoxon_kappa.py` | `analysis/stats/wilcoxon_kappa.py` |

**Onde os relatórios são escritos não mudou.** Os cinco `wilcoxon_*` escreviam ao
lado do próprio `.py`; depois do move isso gravaria dentro de um pacote Python.
Foram reancorados em `REPO/statistics/tools/`, que é exatamente onde o sexto
(`compute_cost.py`) já escrevia. Os `.csv`/`.md` continuam saindo em
`statistics/tools/`.

### 1.5 `scripts/` → `experiments/` (7 renomeações + 1 fusão)

| antigo | novo | git |
|---|---|---|
| `scripts/strip_teacher_from_ckpt.py` | `experiments/ckpt/strip_teacher_from_ckpt.py` | R100 |
| `scripts/strip_teacher_only.py` | `experiments/ckpt/strip_teacher_only.py` | R100 |
| `scripts/validate_stripped_ckpt.py` | `experiments/ckpt/validate_stripped.py` | R100 |
| `scripts/verify_finetune_weights.py` | `experiments/ckpt/verify_finetune.py` | R100 |
| `scripts/check_distill_conv_status.py` | `experiments/oneoff/check_distill_conv_status.py` | R100 |
| `scripts/retry_protozoan_experiment.py` | `experiments/oneoff/retry_protozoan_experiment.py` | R100 |
| `scripts/run_missing_mlp.py` | `experiments/oneoff/run_missing_mlp.py` | R100 |
| `scripts/generate_mlp_configs.py` | `experiments/gen_configs.py::generate_mlp_configs()` | **fora do git** |
| `scripts/download_organmnist3d.py` | `experiments/gen_configs.py::download_organmnist3d()` | **fora do git** |
| `scripts/normalize_reports.py` | `experiments/gen_configs.py::normalize_reports()` | **fora do git** |

As três últimas são uma fusão 3→1 (1.224 linhas). `git mv` não casa 3 origens
com 1 destino, e a similaridade cairia para ~11% — abaixo até do `-M20%`. As
origens continuam intactas em `scripts/`.

### 1.6 Configs

| antigo | novo | n |
|---|---|---:|
| `configs/data/percentage/<ds>_split_N/100/lejepa_line.yaml` | `configs/dataset/<ds>/split_N.yaml` | 9 (R100) |
| `configs/model/lejepa_line_flim_{eggs,larvae,protozoan}_train{1,2,3}.yaml` | `configs/model/lejepa/flim/<ds>/split_N.yaml` | 9 (R087-R088) |
| `configs/model/lejepa_line_{he,random,trunc_normal,xavier}_protozoan_train{1,2,3}.yaml` | `configs/model/lejepa/<init>/protozoan-cysts/split_N.yaml` | 12 (R063-R065) |
| `configs/model/lejepa_{resnet50,real_sigreg,simple_sigreg,custom_cnn}.yaml` | `configs/model/lejepa/_variants/<nome>.yaml` | 4 (R092/R093/R092/**R022**) |
| `configs/evaluate/mlp/<modo>/<ds>/split_N/pct_P/<run_id>.yaml` | `configs/generated/mlp/<modo>/<ds>/<run_id>.yaml` | 112 (R100) |

O achatamento dos MLP (`split_N/pct_P/` sai do caminho) não perde informação:
`split` e `percentage` continuam **dentro** do YAML.

**Só 9 dos 54 configs de `configs/data/percentage/` foram movidos** — os `100/`.
Os outros 45 ficaram onde estavam; a porcentagem hoje entra pela linha de comando
(`--data.init_args.percentage=<P>`), não por arquivo.

**Configs de modelo que existem agora e não têm origem** (foram gerados, não
movidos): `configs/model/autoencoder/flim/<ds>/split_N.yaml` (9),
`configs/model/classification/...` (9) e `configs/model/distillation/...` (18).

### 1.7 Documentos

39 renomeações R100 para `reports/`: todo o `A_reports/`, `metrics_distillation/`
e os `.md` soltos da raiz (`report.md`, `distill_details.md`,
`distillation_model_architecture.md`, `notas_1x1_kernel_distill.md`,
`analise_achatamento_relu_sigmoid.md`, `review_code_22_jul_2026.md`,
`storage_report_hawk.md`). Padrão: `<qualquer_coisa>.md` → `reports/<mesmo caminho>`.

---

## 2. TABELA (b) — comando antigo → comando novo

**Esta é a tabela do dia a dia.** Todo comando novo roda da raiz do repositório,
com o python do env `scalable_FLIM`
(`/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python`).

### 2.1 Treino direto (LightningCLI)

| antigo | novo |
|---|---|
| `python src/main.py fit --config <...>` | `python train.py fit --config <...>` |

A forma canônica do comando novo tem **exatamente três** `--config`, sempre nesta
ordem:

    python train.py fit \
      --config configs/default.yaml \
      --config configs/dataset/<dataset>/split_<N>.yaml \
      --config configs/model/<metodo>/<init>/<dataset>/split_<N>.yaml \
      --data.init_args.percentage=<P>

`<metodo>` ∈ `lejepa | autoencoder | classification | distillation`.
`<init>` ∈ `flim | he | xavier | random | trunc_normal`.

### 2.2 As grades Ray — os 6 lançadores viraram UM comando

| antigo | novo |
|---|---|
| `python scripts/run_ssl_ray.py ...` | `python -m experiments.ray.launch experiments/lejepa/init_ablation.yaml` |
| `python scripts/spifil_growth_loop.py ...` | `python -m experiments.ray.launch experiments/lejepa/growth_grid4.yaml` |
| `python scripts/autoencoder_flim_ray.py ...` | `python -m experiments.ray.launch experiments/<...>.yaml` (YAML ainda não escrito) |
| `python scripts/classification_flim_ray.py ...` | `python -m experiments.ray.launch experiments/<...>.yaml` (YAML ainda não escrito) |
| `python scripts/distillation_ray.py ...` | `python -m experiments.ray.launch experiments/distillation/<...>.yaml` |
| `python scripts/distillation_conv_ray.py ...` | `python -m experiments.ray.launch experiments/distillation/conv_1x1_grid.yaml` — **ver seção 8, família sem `method` válido** |
| `python -m src.evaluate.ray_mlp_queue --mode freeze ...` | `python -m experiments.ray.launch experiments/lejepa/eval_mlp_freeze.yaml` |

Flags úteis do lançador novo: `--dry-run` (imprime os comandos, não executa) e
`--set chave=valor` (sobrescreve uma chave do YAML só naquela invocação).

O `2>&1 | tee logs/...` **sai do comando**: `experiments/ray/launch.py:251-257`
grava `<log_dir>/<name>_<timestamp>.log` sozinho.

#### Mapa flag → chave de YAML (vale para os seis)

| flag antiga | chave nova |
|---|---|
| `--datasets` | `grid.datasets` |
| `--splits` | `grid.splits` |
| `--pcts` / `--percentages` | `grid.percentages` |
| `--inits` | `grid.init` |
| `--gpus` / `--gpu-ids` | `resources.gpu_ids` |
| `--num-gpus N` | `resources.gpu_ids: [0..N-1]` (o novo escolhe a GPU **física**; o antigo não sabia) |
| `--max-concurrent-per-gpu` | `resources.max_per_gpu` |
| `--cpus-per-experiment` | `resources.cpus_per_experiment` |
| `--num-workers` | `resources.num_workers` — **chave morta hoje**, ninguém consome (seção 8) |
| `--work-dir` | `output.work_dir` |
| `--wandb` / `--wandb-project` | `wandb.enabled` / `wandb.project` |
| `--resume` | `skip: state` |
| `--skip-existing --check-wandb` | `skip: wandb` |
| `--ignore-existing` | `skip: none` |
| `--max-rounds` | `runner_args.max_rounds` |
| `--embed-mode` | `runner_args.embed_mode` |
| `--mode freeze` (avaliação) | `runner_args.freeze: true` |
| `--run-prefix` | **sem equivalente no schema** (seção 8) |
| `--batch-size` / `--multicrop` / `--3lproj` (sufixos de nome do SSL) | **sem equivalente no schema** (seção 8) |
| `2>&1 \| tee ...` | some (log automático) |

#### Comandos completos, lado a lado

Crescimento SPiFiL:

    # ANTES
    python scripts/spifil_growth_loop.py \
      --work-dir artifacts/spifil_growth/grid4 --percentages 5 50 \
      --gpus 1 2 3 --max-concurrent-per-gpu 3 --cpus-per-experiment 10 \
      --max-rounds 2 --num-workers 2 --wandb --wandb-project phd_thesis_grid4 \
      2>&1 | tee logs/spifil_growth_$(date +%Y%m%d_%H%M%S).log

    # DEPOIS
    python -m experiments.ray.launch experiments/lejepa/growth_grid4.yaml

Ablação de inicialização (SSL):

    # ANTES
    python scripts/run_ssl_ray.py \
      --inits flim he xavier random trunc_normal \
      --datasets helminth-eggs helminth-larvae protozoan-cysts \
      --splits 1 2 3 --pcts 1 5 25 50 75 100 \
      --gpu-ids 0 1 2 3 --max-concurrent-per-gpu 2 --cpus-per-experiment 8 --resume

    # DEPOIS
    python -m experiments.ray.launch experiments/lejepa/init_ablation.yaml

Fila de MLP congelado:

    # ANTES
    python -m src.evaluate.ray_mlp_queue --mode freeze --num-gpus 1 \
      --max-concurrent-per-gpu 4 --cpus-per-experiment 4 --resume

    # DEPOIS
    python -m experiments.ray.launch experiments/lejepa/eval_mlp_freeze.yaml

### 2.3 Treino por módulo (o `python -m src.modules.*`)

| antigo | novo |
|---|---|
| `python -m src.modules.autoencoder_flim_module --dataset eggs --split 1 --percentage 50 ...` | `python train.py fit --config configs/default.yaml --config configs/dataset/helminth-eggs/split_1.yaml --config configs/model/autoencoder/flim/helminth-eggs/split_1.yaml --data.init_args.percentage=50` |
| `python -m src.modules.distillation_conv_module --distill_4 --init_flim --fine_tune False --loss_cos --teacher_unfrozen` | `python train.py fit ... --config configs/model/distillation/...` com os `init_args` da tabela 2.4 |
| `python -m src.modules.lejepa_line_module ...` | `python train.py fit --config configs/model/lejepa/<init>/<ds>/split_N.yaml` |

### 2.4 As flags de destilação → `init_args`

As flags-fachada de `src/models/distillation.py` (hoje espelhadas em
`methods/distillation/cli.py`) viram chaves de `model.init_args` no YAML.

| flag antiga | `init_args` novo | default |
|---|---|---|
| `--init_flim` | `encoder_init: flim` | — |
| `--init_random` | `encoder_init: random` | — |
| `--fine_tune True` | `fine_tune: true` + `distillation_type: kd_hybrid` | — |
| `--fine_tune False --loss_cos` | `fine_tune: false` + `distillation_type: direct_cosine` | — |
| `--fine_tune False --loss_mse` | `fine_tune: false` + `distillation_type: direct` | — |
| `--teacher_frozen` | `teacher_frozen: true` | — |
| `--teacher_unfrozen` | `teacher_frozen: false` | — |
| `--kd_temperature` | `kd_temperature` | `4.0` |
| `--kd_alpha` | `kd_alpha` | `0.7` |
| `--distill_1` (123.504 params) | `class_path: methods.distillation.DistillationOneLayerModule` + `proj_kernel: 1` | — |
| `--distill_2` (402.544 params) | `class_path: methods.distillation.DistillationTwoLayerModule` | — |
| `--distill_3` (615.024 params) | `class_path: methods.distillation.DistillationOneLayerModule` + `proj_kernel: 3` | — |
| `--distill_4` (889.200 params) | `class_path: methods.distillation.DistillationConvModule` | — |
| `--arch-json` | `arch_json: <caminho>` (explícito no config do dataset/split) | derivado |
| `--flim-weights-path` | `flim_weights_path: <caminho>` (idem) | derivado |
| `--proj-type` | **não existe como `init_args`** — a escolha do student É o `class_path` | — |

Os pares mutuamente exclusivos (`--init_flim`/`--init_random`,
`--teacher_frozen`/`--teacher_unfrozen`) deixam de poder se contradizer: no YAML
`encoder_init` é um enum e `teacher_frozen` é um booleano.

**A armadilha que sumiu:** `derive_flim_paths` só preenchia campo vazio, então
reusar o mesmo objeto `args` entre iterações de uma grade congelava os caminhos
FLIM do primeiro dataset e as células seguintes rodavam com o FLIM errado,
caladas. No YAML cada `configs/dataset/<ds>/split_N.yaml` carrega os dois
caminhos explícitos e nenhum estado atravessa células.

### 2.5 Avaliação e sondas (`eval/`) — as flags continuam iguais

| antigo | novo |
|---|---|
| `python -m src.evaluate.svm` | `python -m eval.svm` |
| `python -m src.evaluate.mlp --config <yaml>` | `python -m eval.mlp --config <yaml> --ckpt-selection=best` |
| `python -m src.evaluate.unified_eval --mode all` | `python -m eval.unified_eval --mode all` |
| `python -m src.evaluate` | `python -m eval` |
| `python -m src.evaluate.eval_growth_stages` | `python -m eval.growth_stages` |
| `python -m src.evaluate.tsne_analysis` | `python -m eval.tsne` |
| `python -m src.evaluate.classical_classifiers` | `python -m eval.classical_classifiers` |
| `python -m src.evaluate.svm_distillation` | `python -m eval.svm_variants.svm_distillation` |
| `python -m src.evaluate.svm_distillation_conv` | `python -m eval.svm_variants.svm_distillation_conv` |
| `python -m src.evaluate.svm_distill_with_projection` | `python -m eval.svm_variants.svm_distill_with_projection` |
| `python -m src.evaluate.svm_ijepa` | `python -m eval.svm_variants.svm_ijepa` |
| `python -m src.evaluate.svm_real_flim` | `python -m eval.svm_variants.svm_real_flim` |
| `python -m src.evaluate.svm_flim_residual` | `python -m eval.svm_variants.svm_flim_residual` |
| `python -m src.evaluate.svm_classification_flim` | `python -m eval.svm_variants.svm_classification_flim` |
| `python -m src.evaluate.eval_autoencoder` | `python -m eval.svm_variants.eval_autoencoder` |
| `python -m src.evaluate.eval_svm_flim_flatten` | `python -m eval.svm_variants.eval_svm_flim_flatten` |
| `python -m src.evaluate.eval_avg_pooling_48d` | `python -m eval.svm_variants.eval_avg_pooling_48d` |

Duas mudanças de flag em `eval/mlp.py`, e são **novas**, não renomeações:
`--ckpt-selection=<best|last>` e `--wandb`. O contrato com
`experiments/ray/runners/eval.py:200-215` exige as duas. `last` levanta
`NotImplementedError` de propósito (`eval/mlp.py:538`) em vez de avaliar o
checkpoint errado calado.

Três flags que **se perderam** na conversão:
`--csv` perdeu o alias (`--csv`/`--out` viraram só `--out`), os `choices=` dos
parsers sumiram (um valor inválido agora falha mais tarde e com mensagem pior), e
`ap.error()` virou `raise SystemExit()` — mesma mensagem, exit code muda de 2
para 1 e some o `usage:`.

### 2.6 Análise — aqui as flags SUMIRAM

| antigo | novo |
|---|---|
| `python statistics/tools/wilcoxon_acc.py --alpha 0.01` | `python -c "from analysis.stats.wilcoxon_acc import wilcoxon_acc; wilcoxon_acc(alpha=0.01)"` |
| `python statistics/tools/wilcoxon_{f1,kappa,equivalence,flim_init}.py` | idem, módulo `analysis.stats.wilcoxon_<nome>` |
| `python statistics/tools/measure_compute_cost.py` | `python -m analysis.stats.compute_cost` (sem flag: `compute_cost(skip_cpu=False)`) |
| `python tools/heatmap_stages.py --dataset eggs --split 1 ...` | `python -c "from analysis.activations.heatmap_stages import heatmap_stages; heatmap_stages(dataset='eggs', split=1)"` ou `heatmap_stages(config='meu.yaml')` |
| `python tools/plot_continuity_spifil_hybrid.py --by epoch ...` | `python -c "from analysis.plots.plot_continuity_spifil_hybrid import plot_continuity; plot_continuity(by='epoch')"` |
| `python tools/plot_partial_train_spifil_hybrid.py` | `python -m analysis.plots.plot_partial_train_spifil_hybrid` |
| `python tools/analyze_sigmoid_saturation.py` | `python -m analysis.activations.saturation` |
| `python tools/analyze_unit_activations.py` | `python -m analysis.activations.unit_activations` |
| `python tools/analyze_firing_relu_vs_sigmoid.py` | `python -m analysis.activations.relu_vs_sigmoid` |
| `python tools/plot_sigmoid_saturation.py` | `python -m analysis.plots.plot_sigmoid_saturation` |
| `python tools/plot_comparacao_flim_protocolo.py` | `python -m analysis.plots.plot_comparacao_flim_protocolo` |
| `python src/utils/plot_svm_results.py` | `python -m analysis.plots.plot_svm_results` |
| `python check_experiments/check_ssl.py --update-wandb` | `python -c "from analysis.checks.check_ssl import check_ssl; raise SystemExit(check_ssl(update_wandb=True))"` |
| `python check_experiments/fine_tune.py` | `python -m analysis.checks.fine_tune` |
| `python check_experiments/fine_results.py` | `python -m analysis.checks.fine_results` |
| `python tools/check_ckpt_slim.py <ckpt>` | `python -c "from analysis.checks.check_ckpt_slim import check_ckpt_slim; check_ckpt_slim(ckpt=('<ckpt>',))"` |
| `python tools/keep_only_best_ckpt.py --apply` | `python -c "from analysis.checks.keep_only_best_ckpt import keep_only_best; keep_only_best(apply=True)"` |
| `python tools/check_probe_matches_evaluator.py` | `python -m analysis.checks.check_probe_matches_evaluator` |
| `python tools/check_refactor_equivalence.py` | `python -m analysis.checks.check_refactor_equivalence` |
| `python scripts/check_spifil_growth.py` | `python -m analysis.checks.check_spifil_growth` |
| `python analysis_flim_distill/embedding_analysis.py` | `python -m analysis.distill.embedding_analysis` |
| `python analysis_flim_distill/ruler_mismatch.py` | `python -m analysis.distill.ruler_mismatch` |
| `python analysis_flim_distill/distill_destroys_flim.py` | `python -m analysis.distill.distill_destroys_flim` |
| `python analysis_flim_distill/aggregate_nonorm_compare.py` | `python -m analysis.checks.aggregate_nonorm_compare` |

Regra prática: **se o comando antigo tinha flag, hoje é `python -c "from ...
import <fn>; <fn>(<parametro>=<valor>)"`**. Sem flag, `python -m <modulo>` basta.
O nome do parâmetro é o nome da flag com `-` virando `_` e sem os dois traços.

### 2.7 Utilitários de checkpoint e geração de config

| antigo | novo |
|---|---|
| `python scripts/strip_teacher_from_ckpt.py --glob ... --apply` | `python -c "from experiments.ckpt.strip_teacher_from_ckpt import strip_teacher_from_ckpt; raise SystemExit(strip_teacher_from_ckpt(apply=True))"` |
| `python scripts/strip_teacher_only.py ...` | `python -c "from experiments.ckpt.strip_teacher_only import strip_teacher_only; ..."` |
| `python scripts/validate_stripped_ckpt.py A.ckpt B.ckpt` | `python -m experiments.ckpt.validate_stripped A.ckpt B.ckpt` (posicional, igual) |
| `python scripts/verify_finetune_weights.py` | `python -m experiments.ckpt.verify_finetune` |
| `python scripts/generate_mlp_configs.py [--dry-run]` | `python -m experiments.gen_configs` (ou `python -c "from experiments.gen_configs import generate_mlp_configs; generate_mlp_configs(dry_run=True)"`) |
| `python scripts/normalize_reports.py` | `python -c "from experiments.gen_configs import normalize_reports; normalize_reports()"` |
| `python scripts/download_organmnist3d.py` | `python -c "from experiments.gen_configs import download_organmnist3d; download_organmnist3d()"` |
| `python scripts/run_missing_mlp.py --mode freeze` | `python -m experiments.oneoff.run_missing_mlp --mode freeze` (**argparse mantido**) |
| `python scripts/retry_protozoan_experiment.py <flags>` | `python -m experiments.oneoff.retry_protozoan_experiment <mesmas flags>` |
| `python scripts/check_distill_conv_status.py` | `python -m experiments.oneoff.check_distill_conv_status` |

Uma mudança de comportamento deliberada em `download_organmnist3d()`: `out_dir`
era `"data/organmnist3d"` relativo ao **cwd** e virou
`f"{PROJECT_ROOT}/data/organmnist3d"`. Rodando da raiz o valor é idêntico; de
qualquer outro diretório, agora acerta.

---

## 3. TABELA (c) — `class_path` antigo → novo

91 substituições em 91 YAMLs de `configs/`. O alvo é sempre a forma curta
re-exportada pelo `__init__.py` do pacote, nunca o caminho do módulo interno.

| `class_path` antigo | `class_path` novo | ocorrências |
|---|---|---:|
| `src.data_modules.parasite_data_module_lejepa_splited.ParasiteLejepaDataModuleSplited` | `core.data.ParasiteDataModule` | 54 |
| `src.data_modules.parasite_to_lejepa.ParasiteLejepaDataModule` | `core.data.ParasiteDataModule` | 1 |
| `src.data_modules.LejepaDataModule` | `core.data.FolderDataModule` | 2 |
| `src.modules.lejepa_line_module.LejepaLineModule` | `methods.lejepa.LejepaLineModule` | 29 |
| `src.modules.LeJEPAModule` | `methods.lejepa.LeJEPAModule` | 4 |
| `src.modules.ClassificationFinetuneModule` | `methods.classification.ClassificationFinetuneModule` | 1 |
| `src.modules.LeJEPACNNModule` | **NÃO REESCRITO** — quarentena, ver bug (i) | 1 |

### Os módulos novos que não têm antecessor em YAML

Estes aparecem em configs que foram **gerados**, não movidos:

| `class_path` | onde |
|---|---|
| `methods.autoencoder.AutoEncoderFlimModule` | `configs/model/autoencoder/flim/<ds>/split_N.yaml` |
| `methods.classification.ClassificationFlimModule` | `configs/model/classification/...` |
| `methods.distillation.DistillationModule` | `configs/model/distillation/...` |
| `methods.distillation.DistillationOneLayerModule` | idem (`--distill_1` e `--distill_3`) |
| `methods.distillation.DistillationTwoLayerModule` | idem (`--distill_2`) |
| `methods.distillation.DistillationConvModule` | idem (`--distill_4`) |
| `experiments.run_metadata_callback.RunMetadataCallback` | callback dos configs de modelo |

### Sobrou `class_path: src.*` em dois lugares — de propósito

    $ grep -rn "class_path: src\." configs/ config.yaml

* `configs/model/lejepa/_variants/custom_cnn.yaml:4` (prosa, dentro do cabeçalho
  de quarentena) e `:50` (o valor real). Ver bug (i).
* `config.yaml` na raiz do repositório, 2 ocorrências. **Não foi reescrito de
  propósito**: é um dump do `SaveConfigCallback` do Lightning (primeira linha
  `# lightning.pytorch==2.6.1`), com valores já resolvidos de uma execução
  concreta, e ninguém o lê — não existe `--config config.yaml` em lugar nenhum do
  repositório. Quem for apagar `src/` deve **apagar o `config.yaml` da raiz
  junto**, não reescrevê-lo.

### Teste que prova o mapa

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python <script de checagem>
    # 271 YAMLs varridos, 141 class_path verificados
    # 1 xfail esperado (custom_cnn.yaml), 0 vermelhos inesperados

O teste importa o módulo e confere que o símbolo existe e é `callable`. Ele **não**
confere se os `init_args` de cada YAML batem com a assinatura da classe nova — se
alguma classe migrada renomeou um parâmetro, o teste passa e o treino quebra no
jsonargparse.

---

## 4. O que mudou de COMPORTAMENTO e o que NÃO mudou

### 4.1 As cinco convenções de `acc` foram PRESERVADAS — nada foi unificado

Decisão sua, aplicada literalmente. Nenhum número muda.

| # | convenção | onde é definida | quem usa hoje |
|---|---|---|---|
| 1 | macro / balanceada (`multiclass_accuracy`, default macro do torchmetrics) | `src/metrics/classification.py:58`, hoje `core/metrics.py:74` | **todo o `eval/`** (8 chamadas, todas `core.metrics.compute_metrics`) |
| 2 | micro (`sklearn.accuracy_score`, sonda kNN/SVM de validação) | `src/modules/lejepa_line_module.py:295` | `methods/lejepa/lejepa_line_module.py` |
| 3 | micro (`torchmetrics.Accuracy(task="multiclass")`, default micro) | `src/modules/classifier_module.py:84` | `methods/classification/` |
| 4 | micro (`test_accuracy`) | `src/evaluate/svm_classification_flim.py:161` | só `eval/svm_variants/svm_classification_flim.py` |
| 5 | balanceada (`test_accuracy_balanced`) | `src/evaluate/svm_classification_flim.py:164` | idem |

As convenções 4 e 5 convivem no MESMO dicionário
(`_metrics_both_conventions`, `src/evaluate/svm_classification_flim.py:157-166`,
hoje `eval/svm_variants/svm_classification_flim.py:154-162`), de propósito e com
nome explícito, para comparabilidade com `data/reports_felipe/svm/`. O bloco
segue byte-idêntico ao original.

**Achado colateral, e é uma sexta convenção de fato:** `acc_raw` (acurácia micro
crua) convive com o `acc` balanceado em três arquivos —
`eval/svm_variants/eval_autoencoder.py:257`,
`eval/svm_variants/eval_svm_flim_flatten.py:257` e
`eval/svm_variants/eval_avg_pooling_48d.py:207`. As duas vão para o CSV.
Preservado intacto nos três.

`grep -rn "accuracy_score\|balanced_accuracy\|cohen_kappa_score\|f1_score\|Accuracy(" eval/*.py`
→ **nenhuma linha**. Zero métrica calculada na mão dentro de `eval/`.

### 4.2 O `run_name` reproduz as fórmulas históricas byte a byte

`experiments/ray/paths.py:170` (`run_name`). **São SEIS fórmulas, não uma.**
Uniformizar teria quebrado o link entre o nome do run e os pesos em disco — que é
exatamente o que você proibiu.

| # | família | fórmula | vocabulário de dataset |
|---|---|---|---|
| 1 | `lejepa` (SSL) | `lejepa_line_{ds}_split_{s}_pct_{p}_model_{init}` | **LONGO** (`helminth-eggs`) |
| 2 | `autoencoder` | `{pre}ae_resnet_flim_{ds}_split{s}_pct{p}_{stage1_frozen\|stage2_fine_tune}{_lab}{_flat}` | curto (`eggs`) |
| 3 | `classification` | `{pre}classhead_{ds}_split{s}_pct{p}_{sigmoid2l\|relu2l\|softplus2l}{_frozen}` | curto |
| 4 | `distillation` | `distillation_{ds}_split{s}_pct{p}_model{dist_type}` | curto |
| 5 | `distillation_conv` | 9 ramos de `proj_type` + `{_flim_init}` + `{_no_imagenet_norm}`, com 2 `return` antecipados | curto |
| 6 | `growth` (runner) | `spifil_growth_{basename(work_dir)}_{ds}_split{s}_pct{p}_{stage}` | curto |

Origens históricas: `check_experiments/_common.py:66-70` (1),
`scripts/autoencoder_flim_ray.py:303-309` (2),
`scripts/classification_flim_ray.py:220-232` (3),
`scripts/distillation_ray.py:141` (4),
`scripts/distillation_conv_ray.py:329-368` (5),
`scripts/spifil_growth_loop.py:209` + `scripts/constants.py:248` (6).

**Prova:** 1878 nomes reais testados, 1878 bateram, **0 divergiram**. As fontes
foram diretórios de `artifacts/`, 1315 arquivos `run_metadata.json`, 555 pares
braço × estágio de `artifacts/spifil_growth/` e 1921 display names do W&B em
`configs/wandb_update/ids_wandb.json`. O método não é circular: a função gerou
2754 nomes a partir do produto cartesiano dos eixos, e depois se perguntou se
cada nome real estava nesse conjunto.

Detalhes que a uniformização teria apagado, e que foram preservados:

* A cauda `model...` da família 4 **não** é o eixo `init`, é o
  `distillation_type` (`direct`/`hybrid`), apesar da grafia idêntica.
* A família 5 tem dois `return` antecipados: as cabeças `*_flim_frozen` forçam a
  divisão de normalização dentro do filho, então o nome delas **nunca** leva
  `_no_imagenet_norm`.
* A família 1 é a única com dataset LONGO e com `_` antes de
  `split`/`pct`/`model`. É a grafia que `core/wandb.py:51-53` casa; trocar por
  `split1_pct5` quebraria `resolve_run` além de quebrar o link com os pesos.
* Os sufixos `_lab` (sem normalização ImageNet) e `_flat` (`embed_mode=flatten`)
  seguem literais em `experiments/ray/paths.py:142-144`. As 8 combinações de
  (stage, imagenet_norm, embed_mode) dão 8 nomes distintos, zero colisão.

`method` virou parâmetro obrigatório de `run_name` (sem default). Um default
silencioso daria o nome de **outra família** a quem esquecesse de passar — nome de
família errada é diretório de checkpoint errado.

### 4.3 O `embed_mode` default continua `None` / `avgpool2d`

`methods/autoencoder/autoencoder_flim_module.py` ganhou `embed_mode` como
`init_arg`, com o valor `Optional[str] = None`. `None` significa "herda a global,
em tempo de chamada, exatamente como hoje" (`src/utils/evaluate.py:249,266`, cuja
base é `DEFAULT_EMBED_MODE = "avgpool2d"`, `:243`).

**Por que não foi fixado em `flatten`**, mesmo sendo `flatten` o que o crescimento
usa: `flatten` é default **do experimento** (`scripts/spifil_growth_loop.py:426`),
não do módulo. Fixá-lo no módulo teria três efeitos proibidos: mudaria o número
de qualquer chamador atual que não seta a chave (48-d → 27.648-d), passaria por
cima do rebind de `src/evaluate/eval_growth_stages.py:391`, e reinterpretaria em
silêncio um `.ckpt` antigo da família `_lab` (treinado em `avgpool2d`) recarregado
sem a chave.

O runner de crescimento (`experiments/ray/runners/growth.py:145`) carrega
`DEFAULT_EMBED_MODE = "flatten"` e o emite **sempre explícito** em `:228`
(`--model.init_args.embed_mode=...`), então nada muda na prática para o grid.

### 4.4 A `class Head` do autoencoder NÃO virou `MLPHead`

`Head` (origem `src/modules/autoencoder_flim_module.py:165-182`) e
`core/blocks/mlp_head.py:30-52` (`MLPHead`) são **cabeças diferentes**, não a
mesma com nomes distintos.

| | `Head` | `MLPHead` |
|---|---:|---:|
| params (eggs, 45 → 9) | **414** | **45.577** |

Uma é linear; a outra tem duas camadas ocultas (256, 128). Trocar mudaria o
número de parâmetros por 110× e invalidaria todo `.ckpt` existente. **Não foi
trocada.**

### 4.5 O `StudentClassificationHead` da destilação também NÃO virou `MLPHead`

`src/models/distillation.py:797` vs `core/blocks/mlp_head.py:30`.

| in_features | num_classes | `StudentClassificationHead` | `MLPHead` | fator |
|---:|---:|---:|---:|---:|
| 1280 | 2 (larvae) | **2.562** | **361.090** | 141× |

Além do tamanho, as chaves do `state_dict` mudariam
(`cls_head.classifier.0.weight` etc.), invalidando todo `.ckpt` de destilação. E
`MLPHead.forward` começa com `self.pool(x)`, que o `StudentClassificationHead`
não faz. **Não foi trocado.**

### 4.6 O que mudou de fato, e você precisa saber

1. **As flags de `analysis/`, `experiments/ckpt/` e `experiments/gen_configs.py`
   deixaram de existir.** Ver seção 0.3 e 2.6.
2. **`--ckpt-selection` e `--wandb` são flags NOVAS de `eval/mlp.py`**, exigidas
   pelo contrato com o runner. `--ckpt-selection=last` levanta
   `NotImplementedError` (`eval/mlp.py:538`).
3. **`configs/model/classifier.yaml`**: a chave `freeze_encoder:` virou `freeze:`
   (consequência do conserto do bug (iii)).
4. **Os `configs/generated/mlp/` foram achatados**: sumiram os níveis
   `split_N/pct_P/` do caminho. `split` e `percentage` continuam dentro do YAML.
5. **`download_organmnist3d()` ancora `out_dir` em `PROJECT_ROOT`**, não mais no
   cwd.
6. **Os `.csv`/`.md` dos wilcoxon continuam em `statistics/tools/`** — o move
   teria partido a saída em dois lugares e isso foi corrigido de propósito.
7. **`experiments/ray/launch.py --dry-run` com `skip: wandb` ESCREVE em disco**
   (reescreve `configs/wandb_update/ids_wandb.json`, via
   `experiments/ray/skip.py:129` → `core/wandb.py:48`), apesar de `launch.py:86`
   prometer o contrário. Pior: o caminho do cache é fixo, não depende de
   `(entity, project)` — um dry-run com `--wandb-project X` sobrescreve o cache
   que os relatórios de `flim-ssl` leem. Contorno enquanto não for consertado:
   `--set skip=state`.

---

## 5. Os cinco bugs pré-existentes que a refatoração desenterrou

Nenhum deles nasceu aqui. Todos existiam antes.

### (i) `LeJEPACNNModule` — classe que nunca existiu — **EM QUARENTENA**

`configs/model/lejepa/_variants/custom_cnn.yaml:50` (antes
`configs/model/lejepa_custom_cnn.yaml:6`) declara
`class_path: src.modules.LeJEPACNNModule`.

Esse símbolo **não existe em lugar nenhum**. `src/modules/__init__.py:19-29`
exporta exatamente quatro nomes: `LeJEPAModule`, `ClassificationFinetuneModule`,
`LeJEPAFLIMModule`, `LejepaLineModule`. `graphify explain "LeJEPACNNModule"`
devolve "No node matching". O arquivo **nasceu quebrado no commit inicial** e
`wandb/` (59 diretórios) não tem um único run dele.

O símbolo mais próximo que existe é `LeJEPACNNModel`
(`src/models/custom_cnn.py:124`), mas **não é uma troca de nome**: `LeJEPACNNModel`
é `nn.Module`, não `LightningModule`, e o `class_path` é consumido pelo
LightningCLI. Além disso 5 dos 11 `init_args` do YAML (`lam`, `sigreg_type`, `lr`,
`weight_decay`, `warmup_epochs`) não existem na assinatura de `LeJEPACNNModel` —
são parâmetros de treino, que moram no Module.

**Estado: NÃO consertado, movido e marcado.** O arquivo ganhou um cabeçalho de
quarentena de 44 linhas (foi ele que derrubou a similaridade para `R022`) e é o
único `xfail` esperado do teste de `class_path`. Consertar de verdade exige
escrever um `LeJEPACNNModule` (LightningModule) espelhando `LeJEPAFLIMModule`.

### (ii) `--proj-type` injetado num módulo que não o define — **NÃO consertado**

`scripts/distillation_conv_ray.py:590` faz
`cmd += ["--proj-type", proj_type]` **incondicionalmente**. Mas
`src/modules/distillation_conv_module.py` — o módulo escolhido para o default
`conv_next_layers`, roteado em `:568` — não define esse argumento e usa
`parse_args()` (`:654`), não `parse_known_args()`. Pelo código, uma run com o
`proj_type` default morre com "unrecognized arguments".

**Estado: NÃO consertado.** O arquivo está na lista de deleção (seção 9) — o
sucessor é `experiments/ray/launch.py`, onde a escolha do student é o
`model.class_path`, não uma flag. Não foi verificado por execução.

### (iii) `freeze_encoder` sombreado por parâmetro homônimo — **CONSERTADO**

`src/modules/classifier_module.py:63` importa a função `freeze_encoder`; a
assinatura de `__init__` (`:79-81`) declara um parâmetro `freeze_encoder: bool`.
Dentro de `__init__` o nome resolve para o **parâmetro bool**, então a linha
`freeze_encoder(self.encoder)` chama `True(...)` e levanta `TypeError`. A classe
**nunca instanciou** com o default, e o ramo `unfreeze_norms` era código morto.

**Estado: CONSERTADO em `methods/classification/classification_finetune_module.py`**
— o parâmetro virou `freeze`, o import volta a ser alcançável.
`configs/model/classifier.yaml` acompanhou (`freeze_encoder: true` → `freeze: true`).
A origem em `src/modules/classifier_module.py` fica com o bug (`src/` é read-only
nesta fase).

Prova: instanciação real com os `init_args` do YAML — 53/53 tensores não-norm
com `requires_grad=False`, 106/106 norms com `requires_grad=True`, controle com
`unfreeze_norms=False` dá 0/106, `configure_optimizers` devolve 1 grupo com
`freeze=True` e 2 com `freeze=False`.

**Atenção ao commitar:** a mudança no YAML e a do `class_path` (tabela c) **têm que
entrar no mesmo commit**. Só o YAML → jsonargparse recusa a chave `freeze`
desconhecida da classe antiga; só o `class_path` → a classe nova recebe
`freeze_encoder`, que ela não tem mais.

### (iv) `init_ckpt` mudava congelamento sem carregar peso — **CONSERTADO**

Na origem, o bloco `if args.init_ckpt:` (que faz `torch.load` +
`load_state_dict(strict=False)` + log) vivia dentro do `main()` do
`src/modules/autoencoder_flim_module.py`, e não dentro da classe. Sob YAML não
existe mais `main()` — `init_ckpt: <path>` mudaria a política de congelamento sem
carregar peso nenhum.

**Estado: CONSERTADO.** Virou o método `_load_init_ckpt` em
`methods/autoencoder/autoencoder_flim_module.py` (`:400-453`), chamado do
`__init__`, conferido contra `main:1033-1069` da origem.

Ressalva registrada: num estágio 3 congelado o `torch.load` do `init_ckpt` roda
**duas vezes** — uma na política de freeze (com `try/except`) e outra em
`_load_init_ckpt` (sem guarda).

### (v) `("vit_")` sem vírgula fazia `resnet50` nunca construir — **CONSERTADO**

`src/models/encoders.py:43`: `is_vit = any(k in arch for k in ("vit_"))`.
`("vit_")` **não é uma tupla, é uma string** — falta a vírgula. O `any()` itera os
caracteres `'v','i','t','_'`, e `'t'` casa com `"resnet50"`. Resultado:
`is_vit=True` e o kwarg `dynamic_img_size=True` (só de ViT) vai para o `ResNet`,
que morre com
`TypeError: ResNet.__init__() got an unexpected keyword argument 'dynamic_img_size'`.

**Estado: CONSERTADO em `core/blocks/timm_encoder.py`** — virou `"vit_" in arch`,
que é o comportamento pretendido e idêntico ao da tupla correta. A origem em
`src/models/encoders.py:43` fica com o bug.

---

## 6. Dívida deixada de propósito, e o porquê

### 6.1 `_encode_pooled` e a global `EMBED_MODE`

`_encode_pooled` veio para `eval/svm.py:309` (origem `src/utils/evaluate.py:254`),
e o padrão de **rebind da global** continua vivo: três lugares reescrevem
`_ev.EMBED_MODE` em runtime (`src/evaluate/eval_growth_stages.py:391`,
`src/evaluate/unified_eval.py:731`, `src/modules/autoencoder_flim_module.py:980`).

Por que ficou: matar o rebind agora mudaria a dimensão do embedding (48-d vs
27.648-d) de quem não seta a chave, e o rebind ainda é o contrato do avaliador de
crescimento. A saída limpa é o `init_arg` `embed_mode`, que já existe e é emitido
explicitamente pelo runner novo — a global morre sozinha quando o último chamador
antigo sair.

### 6.2 `methods/distillation/cli.py` — nasce condenado

Arquivo de transição, 15 KB, cópia byte-exata das funções de CLI de
`src/models/distillation.py` (`derive_flim_paths`, `_str2bool`, `STUDENTS`,
`add_student_flags`, `resolve_student`, `add_distill_flags`, `distill_run_tags`,
`resolve_distill_flags`). **Não é importado por nenhum entrypoint** — nasce sem
consumidor, por desenho.

Existe só para que a tradução flag → `init_args` (tabela 2.4) tenha uma fonte
única e verificável enquanto os dois mundos convivem. Quando todos os YAMLs de
destilação estiverem escritos, ele some.

### 6.3 Os 3 scripts de `analysis/distill/` resolvem contra OUTRO checkout

| file:line | conteúdo |
|---|---|
| `analysis/distill/embedding_analysis.py:52` | `_ROOT = "/dados/home/moliveira/scalable_FLIM_self_supervised"` |
| `analysis/distill/ruler_mismatch.py:35` | idem |
| `analysis/distill/distill_destroys_flim.py:58` | idem |

Os `OUT_DIR` que gravam os `.json` **lá** também estão intocados
(`embedding_analysis.py:78`, `ruler_mismatch.py:50`, `distill_destroys_flim.py:88`).

Por que ficou: os três continuam lendo o `src/` de junho e escrevendo no outro
repositório. Desmontar o `src/` daqui **não os quebra**. "Consertar" o caminho
mudaria números já publicados nos `.md` de investigação. Não é decisão de
refatoração — é sua.

(Como o `_ROOT` é absoluto, ele é imune à mudança de profundidade do move.)

### 6.4 `resources.num_workers` — chave morta

`experiments/ray/schema.py:175` valida a chave, e **ninguém a consome**: nem
`experiments/ray/runners/train.py:78-97`, nem `growth.stage_cmd`, nem
`eval.build_cmd`. O `--num-workers 2` do comando antigo ia para o trainer
(`scripts/spifil_growth_loop.py:424`).

Duas saídas, nenhuma tomada: ou vira `--data.init_args.num_workers=` no
`train.build_cmd`, ou sai do schema.

### 6.5 A família `distillation_conv` tem fórmula de nome e nenhum `method` a alcança

`experiments/ray/paths.py:336` registra
`"distillation_conv": _name_distillation_conv`, com as 9 fórmulas históricas de
nome da grade conv. Mas `experiments/ray/schema.py:283-287` valida `method` contra
as pastas de `methods/`, e `methods/` tem `autoencoder`, `classification`,
`distillation`, `lejepa` — **não há `distillation_conv`**.

Consequência concreta: um experimento conv sai com o nome da família MLP
(`distillation_eggs_split1_pct100_modeldirect`) em vez de
`distillation_eggs_split1_pct100_1x1_BN2d_1280_one_layer`. **Diretório de
checkpoint e display name diferentes dos 594 runs conv já gravados.**

Três saídas possíveis: nasce `methods/distillation_conv/`, ou
`_name_distillation_conv` vira variante de `distillation` acionada por
`variant["proj_type"]`, ou vira código morto. Nenhuma foi tomada.

### 6.6 Dívidas menores, registradas

* **`experiments/ray/runners/train.py:64-70` tem os 5 posicionais de `run_name`
  TROCADOS** (`init` no lugar de `dataset`, etc.) — bug pré-existente,
  independente do `run_name`. `grid.py:60` usa a ordem certa. Hoje isso já produz
  nome diferente do que a grade calculou: `train.py` grava o checkpoint num
  diretório e `skip.py` procura em outro. **Merece verificação antes de qualquer
  corrida.**
* **`scripts/run_ssl_ray.py:239`** ainda faz
  `from check_experiments._common import ...`, e `check_experiments/` está vazia.
  Esse import **quebra na próxima execução do launcher**. O conserto é uma linha:
  `from analysis.checks._common import ...`.
* **`eval/growth_stages.py:291`** importa `analysis.plots` — `eval/` depende de
  `analysis/`, o que quebra "análise é folha". A dependência é real, não é detalhe
  de import: `_wandb_val` usa `_pt.PROJECT`, `_pt.fetch()`, `_pt.RUN_RE`,
  `_pt.grid_dir()` e `_pt.CSV_DIR` — a lógica de BUSCAR RUN NO W&B mora dentro de
  um módulo de PLOT. O conserto de verdade é tirar `fetch`/`RUN_RE`/`CSV_DIR`/
  `grid_dir` do plot.
* **`eval/svm_variants/` importa de `experiments/`** em 6 linhas
  (`svm_real_flim.py:98-99`, `eval_svm_flim_flatten.py:94-95`,
  `eval_avg_pooling_48d.py:78-79`) — `from experiments.ray.paths import arch_json,
  flim_weights_path` e `from experiments.constants import PARASITE_DIR`. Já
  estavam assim antes; é a direção de dependência errada.
* **`RUNNER_ARGS["growth"]` não tem `imagenet_norm`** (`schema.py:55-66`), embora
  `growth.GROW_KNOBS` o liste (`growth.py:153`). Um YAML com
  `runner_args.imagenet_norm: false` é recusado pelo schema antes de chegar ao
  runner. A ablação `_lab` vai precisar disso.
* **`--run-prefix` e os sufixos de CLI do SSL** (`_bs150`, `_mc8g8l`, `_3lproj`)
  são reproduzíveis por `paths.run_name(..., prefix=, suffix=)`, mas **não há
  campo no `schema.Grid`** para eles. A grade não sabe montá-los sozinha.
* **12 configs de protozoan não-flim têm `arch_json` ABSOLUTO** apontando para
  `/dados/home/moliveira/scalable_FLIM_self_supervised/...`. O equivalente
  relativo **já existe neste repositório** e é byte-idêntico (`cmp` confirma). A
  correção é reescrever os 12 para o relativo; não foi feita.
* **A ordem de import pandas/PIL está pinada** em `experiments/gen_configs.py`
  (numpy e PIL antes de pandas) com um comentário de 8 linhas. Não é capricho:
  neste env `import pandas` carrega o `libstdc++` do sistema e depois
  `from PIL import Image` morre com `GLIBCXX_3.4.29 not found`. **É bug do
  ambiente, não do código** — `python scripts/normalize_reports.py` já falha hoje
  no passo de plot, e `import src.evaluate.ray_mlp` já não importa hoje.
* **`medmnist` não está instalado** no env — `download_organmnist3d()` bate no
  `SystemExit` na primeira execução real. O script está efetivamente morto.

---

## 7. O que ainda NÃO foi apagado

**Nada foi apagado.** Toda deleção depende da sua aprovação. O inventário, com
contagens medidas hoje:

| o que | tamanho medido | por que ainda está aí |
|---|---|---|
| Os 6 lançadores Ray | **6.463 linhas** | sucessor é `experiments/ray/launch.py`, mas nem toda família tem YAML escrito ainda |
| `src/` inteiro | **18.521 linhas** em 66 `.py` rastreados | o grosso já tem sucessor em `core/`, `flim/`, `methods/`, `eval/`, `experiments/`, `analysis/`; 10 arquivos de `src/evaluate/` (3.510 linhas) foram movidos mas a origem continua |
| `src/utils/constant.py` | **33 linhas** | zero consumidores |
| `configs/fine_tune/SVM/` | **56 YAMLs** | órfãos: nenhum leitor, nenhum gerador |
| Configs genéricos do lejepa | **8 arquivos** | ambíguos por construção; mover seria escolher vencedor |
| Diretórios vazios sob `configs/evaluate/mlp/` | **112** | `git mv` esvazia mas não remove |
| `configs/data/percentage/` (o que sobrou) | **45 YAMLs** + 9 dirs vazios | só os `100/` foram movidos |

Detalhamento dos 6.463: `scripts/run_ssl_ray.py` (873),
`scripts/autoencoder_flim_ray.py` (1.094),
`scripts/classification_flim_ray.py` (850), `scripts/distillation_ray.py` (744),
`scripts/distillation_conv_ray.py` (1.181), `scripts/spifil_growth_loop.py` (528),
`src/evaluate/ray_mlp.py` (525), `src/evaluate/ray_mlp_queue.py` (668).

Detalhamento dos 8 configs genéricos do lejepa:

* 4 duplicatas flim, sem nenhuma referência, byte-idênticas (ignorando
  comentários) aos per-split: `configs/model/lejepa_line_flim.yaml`,
  `lejepa_line_flim_split_1.yaml`, `_split_2.yaml`, `_split_3.yaml`. O comentário
  em `:2-3` ("overridden at runtime by run_experiments.py") é obsoleto:
  `run_experiments.py` resolve por `_FLIM_YAMLS` e nunca abre esses arquivos.
* 4 genéricos não-flim: `configs/model/lejepa_line_{he,xavier,random,trunc_normal}.yaml`.
  Cada um servia **6 células** (eggs + larvae × 3 splits) — exatamente as 24
  células que faltam na grade nova (a grade completa é 5 inits × 3 datasets × 3
  splits = 45; existem 21).

Sobre `configs/fine_tune/SVM/`: verificação dupla. Buscas que voltaram **zero**:
`configs/fine_tune`, `fine_tune/SVM`, `SVM/lejepa`, `svm_config`, `FINE_TUNE`.
Não existe gerador (`yaml.dump` aparece em 3 lugares no repo, nenhum escreve ali)
e não existe leitor (`src/evaluate/svm.py` não tem `--config` nem
`yaml.safe_load`). São 56 de 216 esperados — padrão de geração interrompida.

Outras pastas que ficaram para trás, esvaziadas mas não removidas:

* `check_experiments/` — **vazia** (só `__pycache__`). Ainda assim
  `scripts/run_ssl_ray.py:239` importa dela.
* `tools/` — sobraram `__init__.py` e `README.md`, que agora descreve uma pasta
  que não existe mais.
* `statistics/tools/` — 14 `.csv`/`.md` + 2 `README`. **Continuam sendo o destino
  de escrita** dos 6 scripts de `analysis/stats/`; não são candidatos a deleção.
* `analysis_flim_distill/` — 3 `.json`, 4 `.md` de investigação, `plots/*.png`.
  Dois comentários vivos apontam para eles (`src/evaluate/svm_distillation.py:215`,
  `src/models/distillation.py:849`).
* `experiments/oneoff/` (3 arquivos): nenhum é importado por linha de código
  alguma, e nenhum foi editado por conteúdo desde o commit inicial (45 dias).
  São os candidatos legítimos. **Ressalva antes de aprovar:**
  `retry_protozoan_experiment.py:104` contém o único `_PROTOZOAN_30CH_ARCH` do
  repositório (a arquitetura 30 canais escrita em código) e as duas funções que a
  materializam em disco (`:146`, `:169`, esta última escreve 9 YAMLs em
  `configs/model/`). Se esses JSONs/YAMLs não estiverem versionados, apagar o
  arquivo perde a receita.

### A ordem dos commits importa — armadilha registrada

Se `git rm --cached -r configs/generated/` entrar **no mesmo commit** que a
renomeação, o commit resultante vira "112 deleções em `configs/evaluate/`" e
**nenhum `R`** — `git log --diff-filter=R` não vê nada e este documento perde o
registro do move. São dois commits, nesta ordem:

    # 1) preserva o rename
    git add -A configs .gitignore && git commit -m "<sua mensagem>"
    # 2) só então tira os 112 do indice (os arquivos ficam no disco)
    git rm --cached -r configs/generated/ && git commit -m "<sua mensagem>"

Diretórios vazios (comando para quando você aprovar):

    find configs/evaluate -type d -empty -delete

---

## 8. Se você não achou o que procurava aqui

1. Os relatórios completos dos agentes estão em
   `<scratchpad>/refactor_reports/` — 78 arquivos, um por frente de trabalho.
   Os mais consultáveis: `Y-CLASSPATH.md` (os 91 `class_path`), `E1-E4.md` (os
   quatro YAMLs de experimento com o comando antigo equivalente no cabeçalho),
   `MD-17.md` (flag → `init_args` da destilação), `FIX-RUNNAME.md` (as 6 fórmulas
   de nome de run e a prova dos 1878 nomes).
2. O grafo do repositório responde busca estrutural:
   `graphify query "<pergunta>"`, `graphify explain "<no>"`,
   `graphify affected "<no>"`.
3. O comando que reconstrói a tabela (a) sozinho:
   `git diff --cached -M20% --name-status --diff-filter=R`.
