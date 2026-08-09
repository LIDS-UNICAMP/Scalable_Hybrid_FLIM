# scalable_FLIM_self_supervised — Execution Guide

> **Installation**: see [INSTALL.md](INSTALL.md)

---

## Activate environment

```bash
conda activate scalable_FLIM
# or, if using uv:
source .venv/bin/activate
```

---

## W&B Metadata Cache

All evaluation scripts use a local cache (`configs/wandb_update/ids_wandb.json`) instead of querying W&B on every run.

```bash
# Build / refresh the cache from W&B (run once before first evaluation)
python -m src.utils.wandb_cache --update

# Show cache status
python -m src.utils.wandb_cache
```

Pass `--wandb-update` to any evaluation script to refresh the cache inline before running.

---

## SSL Pre-training

Run one experiment manually:

```bash
python src/main.py fit \
  --config configs/default.yaml \
  --config configs/data/percentage/helminth-eggs_split_1/100/lejepa_line.yaml \
  --config configs/model/lejepa_line_xavier.yaml \
  --trainer.accelerator=gpu
```

Swap configs for other datasets / splits / percentages / initialisers:

| Axis | Values |
|---|---|
| dataset | `helminth-eggs`, `protozoan-cysts`, `helminth-larvae` |
| split | `1`, `2`, `3` |
| pct | `1`, `5`, `25`, `50`, `75`, `100` |
| init | `lejepa_line_{xavier,random,he,flim,trunc_normal}.yaml` |

Run the full grid automatically (sequential):

```bash
python run_experiments.py
```

Run the full grid in parallel across multiple GPUs (Ray):

```bash
# Preview queue without running
python scripts/run_ssl_ray.py --dry-run

# Run all missing experiments (2 GPUs × 3 slots each)
python scripts/run_ssl_ray.py --num-gpus 2 --max-concurrent-per-gpu 3 --cpus-per-experiment 4

# Run only trunc_normal experiments
python scripts/run_ssl_ray.py --inits trunc_normal --num-gpus 1 --max-concurrent-per-gpu 3

# Resume after interruption
python scripts/run_ssl_ray.py --resume
```

#### Running with a different batch size

Use `--batch-size` to override the value in the data YAML (default: 32).
Because this is a **new experiment** (not a replacement), use `--ignore-existing`
to bypass the local-checkpoint check, and optionally `--pcts` to limit scope.

```bash
# Preview: trunc_normal, all 3 datasets, pct=100%, batch=256
python scripts/run_ssl_ray.py \
  --inits trunc_normal \
  --datasets helminth-eggs helminth-larvae protozoan-cysts \
  --pcts 100 \
  --batch-size 256 \
  --ignore-existing \
  --num-gpus 2 \
  --max-concurrent-per-gpu 3 \
  --dry-run

# Run for real (remove --dry-run)
python scripts/run_ssl_ray.py \
  --inits trunc_normal \
  --datasets helminth-eggs helminth-larvae protozoan-cysts \
  --pcts 100 \
  --batch-size 256 \
  --ignore-existing \
  --num-gpus 2 \
  --max-concurrent-per-gpu 3
```

W&B identification: the run name gets a `_bs<N>` suffix and a `bs<N>` tag,
so runs with different batch sizes are distinct and filterable in the W&B UI:

```
lejepa_line_helminth-eggs_split_1_pct_100_model_trunc_normal_bs256   # tag: bs256
lejepa_line_helminth-eggs_split_1_pct_100_model_trunc_normal          # original (bs32)
```

#### `run_ssl_ray.py` flags reference

| Flag | Default | Description |
|---|---|---|
| `--inits` | all | Initialisations to run (`xavier random he flim trunc_normal`) |
| `--datasets` | all | Datasets to run |
| `--pcts` | all | Percentages to run, e.g. `--pcts 100` or `--pcts 25 50 100` |
| `--batch-size` | YAML value | Override `batch_size` in the data config |
| `--ignore-existing` | off | Queue even if a local checkpoint already exists |
| `--num-gpus` | 2 | Number of GPUs (uses IDs 0..N-1) |
| `--max-concurrent-per-gpu` | 10 | Parallel experiments per GPU |
| `--cpus-per-experiment` | 4 | CPU cores per experiment |
| `--resume` | off | Skip experiments already recorded as OK in the state file |
| `--fail-fast` | off | Stop dispatching after the first failure |
| `--dry-run` | off | Print the queue plan and exit without running |

Experiment naming convention:

```
lejepa_line_<dataset>_split_<1|2|3>_pct_<percentage>_model_<init>[_bs<batch>]

# Examples:
lejepa_line_helminth-eggs_split_1_pct_100_model_trunc_normal
lejepa_line_protozoan-cysts_split_2_pct_5_model_trunc_normal
lejepa_line_helminth-larvae_split_3_pct_100_model_flim_bs256
```

`trunc_normal` applies `init_weights_vit_timm` from timm (std=0.02, clip=[-0.04, +0.04] for `nn.Linear`).

---

## MLP Fine-tuning Queue (Ray, slot-based)

### Generate YAML configs (once, or after adding new runs)

```bash
python scripts/generate_mlp_configs.py
```

### Run queue

```bash
# Preview — print plan without executing
python -m src.evaluate.ray_mlp_queue --dry-run

# Run all experiments (both freeze and unfreeze, 2 GPUs × 10 slots)
python -m src.evaluate.ray_mlp_queue

# Resume after interruption
python -m src.evaluate.ray_mlp_queue --resume

# Refresh W&B cache inline, then run
python -m src.evaluate.ray_mlp_queue --wandb-update

# Protozoan only
python -m src.evaluate.ray_mlp_queue --dataset-group protozoan

# Eggs, freeze mode only
python -m src.evaluate.ray_mlp_queue --mode freeze --dataset-group eggs

# Custom GPU profile
python -m src.evaluate.ray_mlp_queue --num-gpus 1 --max-concurrent-per-gpu 5
```

### CLI reference

| Argument | Default | Description |
|---|---|---|
| `--num-gpus` | `2` | Physical GPUs to use |
| `--max-concurrent-per-gpu` | `10` | Max experiments per GPU simultaneously |
| `--cpus-per-experiment` | `4` | CPU cores per experiment |
| `--mode` | `all` | `freeze`, `unfreeze`, or `all` |
| `--dataset-group` | `all` | `all`, `eggs`, `protozoan`, `larvae` |
| `--state-file` | `results/queue_state.json` | Resume state |
| `--resume` | off | Skip already-completed experiments |
| `--fail-fast` | off | Stop after first failure |
| `--wandb` | off | Enable W&B metric logging |
| `--wandb-update` | off | Refresh W&B cache before running |
| `--dry-run` | off | Print queue plan and exit |
| `--experiment-filter` | — | Substring filter on run_id / name |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, or `WARN` |

### Output layout

```
results/
├── queue_state.json
├── ray_mlp_queue_results.csv
└── ray_finetune_queue/
    └── ray_finetune_lejepa_line_<dataset>_split_<N>_<init>_pct<pct>_<id>/
        ├── config/config.yaml
        ├── logs/train.log
        ├── metrics/test_metrics.json
        └── weights/model_final.pth
```

---

## Retry Missing / Broken Runs

`retry_models_again.py` audits the full experiment matrix
(`run_manifest.csv` + local artifact search), identifies unresolved runs, and
retries them through the same Ray GPU-slot queue used by `ray_mlp_queue.py`.

### Typical cluster run

```bash
# Step 1 — audit only (no execution, fast)
python retry_models_again.py --report-only

# Step 2 — dry-run: confirm exactly what will be submitted
python retry_models_again.py --dry-run

# Step 3 — execute, log results to W&B (standard cluster command)
python retry_models_again.py \
    --num-gpus 2 \
    --max-concurrent-per-gpu 10 \
    --wandb
```

> `--wandb` enables W&B metric logging for each retried MLP experiment.
> The W&B **metadata cache** (`ids_wandb.json`) is used automatically — no live API call at startup.
> Run-ids that have local checkpoints but no W&B SSL entry (e.g. protozoan runs trained offline) are injected automatically and will execute normally.

### Resume after interruption

```bash
python retry_models_again.py \
    --num-gpus 2 \
    --max-concurrent-per-gpu 10 \
    --wandb
# state is saved to results/retry_again_queue_state.json after each experiment
# re-running the same command automatically skips already-completed ones
```

### Force W&B cache refresh before running

```bash
python retry_models_again.py \
    --num-gpus 2 \
    --max-concurrent-per-gpu 10 \
    --wandb \
    --wandb-update
```

### Save audit report to file

```bash
python retry_models_again.py \
    --report-only \
    --report-out results/retry_report.txt
```

### CLI reference

| Argument | Default | Description |
|---|---|---|
| `--num-gpus` | `2` | Physical GPUs for the retry queue |
| `--max-concurrent-per-gpu` | `10` | Max experiments running per GPU |
| `--cpus-per-experiment` | `4` | CPU cores per experiment (DataLoader workers) |
| `--state-file` | `results/retry_again_queue_state.json` | Resume state (auto-skips completed) |
| `--wandb` | off | Log MLP training metrics to W&B for each retried run |
| `--wandb-update` | off | Refresh `ids_wandb.json` from live W&B API before running |
| `--dry-run` | off | Print queue plan without submitting any Ray tasks |
| `--report-only` | off | Audit and classify only, skip execution entirely |
| `--report-out` | — | Write full audit + retry report to this file |
| `--manifest` | `artifacts/run_manifest.csv` | Manifest to audit |

### How it decides what to retry

```
run_manifest.csv
      │
      ▼
audit() — for each expected SSL run (3 datasets × 3 splits × 6 pcts × 4 inits = 216):
      │
      ├─ search results/ray_finetune*/          → resolved_local
      ├─ search artifacts_view/MLP/             → resolved_artifacts_view
      ├─ weights dir exists but empty           → broken_partial_artifact  ← retry
      └─ not found anywhere                     → missing_after_full_search ← retry
```

Only `broken_partial_artifact` and `missing_after_full_search` are submitted to the queue.

### Retry classification categories

| Category | Action |
|---|---|
| `resolved_local` | Already has weights — skipped |
| `resolved_artifacts_view` | Found in `artifacts_view/MLP/` — skipped |
| `missing_after_full_search` | Not found anywhere — **retried** |
| `broken_partial_artifact` | Weights dir is empty — **retried** |
| `name_mismatch_or_parse_issue` | Parse failure, investigated manually |
| `manifest_only_no_weights` | In manifest but no run_id resolved |

### W&B behaviour for retried runs

- **SSL encoder run-ids** (e.g. protozoan runs trained without W&B): injected automatically from local checkpoints. The SSL entry does not need to be in W&B.
- **MLP fine-tuning run** (`--wandb`): a new W&B run is created for each retried experiment under the project `flim-ssl`, named:
  ```
  X_finetune_lejepa_line_<dataset>_split_<N>_model_<init>[_<run_id>]
  ```
- **Cache**: `ids_wandb.json` is read at startup. If the file does not exist, it is created automatically by fetching from W&B once.

---

## Knowledge Distillation — I-JEPA → FLIM CNN

Treina o encoder FLIM CNN (student) com um teacher I-JEPA congelado (ViT-H/14, 1280-dim).
Ver arquitetura completa em [distillation_model_architecture.md](distillation_model_architecture.md).

### Variantes de proj head

Cada variante é selecionada com `--proj-type`. As de **init aleatório** podem ser rodadas
**sem** ou **com** FLIM (basta acrescentar `--flim-init`); as marcadas como *FLIM intrínseco*
sempre treinam com o encoder FLIM inicializado, independente da flag.

| `--proj-type` | Arquitetura | Params | FLIM init | Convenção de nome |
|---|---|---|---|---|
| `conv_next_layers` | 4× 1×1 convs 48→128→256→512→1280 | ~889K | opcional (`--flim-init`) | `..._next_layers_<dist>` (+ `_flim_init`) |
| `3x3_bn2d_1280` | Conv 3×3 + BN2d (48→1280) | ~615K | opcional (`--flim-init`) | `..._3x3_BN2d_1280_one_layer` (+ `_flim_init`) |
| `1x1_bn2d_1280` | Conv 1×1 + BN2d (48→1280) | ~124K | opcional (`--flim-init`) | `..._1x1_BN2d_1280_one_layer` (+ `_flim_init`) |
| `2l_1x1_bn2d_256_1280` | 2× 1×1 + BN2d (48→256→1280) | ~402K | opcional (`--flim-init`) | `..._2l_1x1_BN2d_256_1280` (+ `_flim_init`) |
| `2l_1x1_init_flim_256_1280` | igual acima | ~402K | **intrínseco** | `..._2l_1x1_init_flim_256_1280` |
| `3x3_init_flim_1280` | head 3×3 | ~615K | **intrínseco** | `..._3x3_BN2d_1280_one_layer_init_flim` |
| `next_layers_init_flim` | head conv_next_layers | ~889K | **intrínseco** | `..._next_layers_init_flim_<dist>` |

FLIM encoder: **60K** | I-JEPA teacher: **632M**

### Resultados SVM — kappa médio entre splits

| Método | Params | Eggs | Larvae | Protozoan |
|---|---|---|---|---|
| FLIM | 60K | 0.686 | 0.655 | 0.674 |
| LeJEPA | 60K | 0.319 | 0.381 | 0.184 |
| I-JEPA | 632M | 0.881 | 0.909 | 0.796 |
| Distil 4×1×1 | 889K | 0.738 | 0.833 | 0.706 |
| Distil 3×3 BN | 615K | 0.704 | 0.820 | 0.664 |
| Distil 1×1 BN | 124K | — | — | — |

Plots: [`artifacts/plots/plots_compare_to_flim/`](artifacts/plots/plots_compare_to_flim/)

### Rodar o grid — variante conv (recomendada)

```bash
# Preview
python scripts/distillation_conv_ray.py --dry-run

# Grid completo: 3 datasets × 3 splits × 6 pcts = 54 runs
python scripts/distillation_conv_ray.py --num-gpus 1 --max-concurrent-per-gpu 1

# Retry após falha parcial — pula concluídos, refaz só os que falharam no W&B
python scripts/distillation_conv_ray.py \
    --num-gpus 1 --max-concurrent-per-gpu 1 \
    --retry --skip-existing --check-wandb --wandb-update
```

> Use sempre `--max-concurrent-per-gpu 1` para evitar OOM (o teacher I-JEPA ocupa ~5 GB por processo).

### Rodar todas as destilações — com e sem FLIM

O script roda **um `--proj-type` por invocação**. Para cobrir todas as arquiteturas nas duas
condições (init aleatório = *sem FLIM*, e *com FLIM*), itere sobre os proj-types. As heads de
init aleatório ganham a versão FLIM com `--flim-init`; as `*_init_flim*` já são FLIM intrínseco.

```bash
mkdir -p logs

# 1) SEM FLIM — heads de init aleatório (trunc_normal)
for PT in conv_next_layers 3x3_bn2d_1280 1x1_bn2d_1280 2l_1x1_bn2d_256_1280; do
  python scripts/distillation_conv_ray.py \
      --proj-type "$PT" \
      --num-gpus 1 --max-concurrent-per-gpu 1 \
      --skip-existing --check-wandb --wandb-update \
      2>&1 | tee "logs/distill_${PT}_noflim_$(date +%Y%m%d_%H%M%S).log"
done

# 2) COM FLIM — mesmas heads, agora com --flim-init (encoder FLIM pré-treinado)
for PT in conv_next_layers 3x3_bn2d_1280 1x1_bn2d_1280 2l_1x1_bn2d_256_1280; do
  python scripts/distillation_conv_ray.py \
      --proj-type "$PT" --flim-init \
      --num-gpus 1 --max-concurrent-per-gpu 1 \
      --skip-existing --check-wandb --wandb-update \
      2>&1 | tee "logs/distill_${PT}_flim_$(date +%Y%m%d_%H%M%S).log"
done

# 3) COM FLIM intrínseco — proj-types que já forçam o init FLIM
for PT in 2l_1x1_init_flim_256_1280 3x3_init_flim_1280 next_layers_init_flim; do
  python scripts/distillation_conv_ray.py \
      --proj-type "$PT" \
      --num-gpus 1 --max-concurrent-per-gpu 1 \
      --skip-existing --check-wandb --wandb-update \
      2>&1 | tee "logs/distill_${PT}_$(date +%Y%m%d_%H%M%S).log"
done
```

Cada invocação cobre o grid padrão **3 datasets × 3 splits × 6 pcts × 2 dist-types**. Restrinja
com `--datasets`, `--splits`, `--percentages` ou `--distillation-types` quando quiser um subconjunto.
Para sobreviver a desconexões, rode os três blocos dentro de uma sessão tmux
(`tmux new -s distill_all`, cole os `for`-loops, depois `Ctrl+b d` para destacar).

| Flag | Efeito |
|---|---|
| `--skip-existing` | Pula runs com `run_metadata.json` local `status=ok` |
| `--check-wandb` | Só pula se W&B também confirmar `finished`; re-fila `failed/crashed` |
| `--retry` | Manifest salvo em `run_manifest_conv_retry.csv` |

### Verificar status dos experimentos

```bash
python scripts/check_distill_conv_status.py
```

Cruza 4 fontes: processo do SO, W&B, metadata local e checkpoint.

### Relatório de disco + W&B (verificar possíveis divergências)

Levanta, para **cada** experimento em `artifacts/distillation/`, o espaço ocupado, os
checkpoints `best`/`last` (caminho + tamanho) e o estado do run no W&B, consolidando tudo
numa coluna `category` (`done` / `divergent` / `failed` / `missing`). Útil para **verificar
possíveis divergências** entre o que o W&B reporta (`finished`/`failed`/`crashed`/`not_found`)
e o status local (`run_metadata.json`).

```bash
python scripts/report_distill_disk_wandb.py            # disco + W&B (default)
python scripts/report_distill_disk_wandb.py --no-wandb # só disco (offline, rápido)
```

Gera **dois CSVs**, por default dentro de `artifacts/distillation/`:

| Arquivo | Conteúdo |
|---|---|
| `artifacts/distillation/distill_disk_wandb_report.csv` | relatório completo — uma linha por experimento |
| `artifacts/distillation/distill_divergences.csv` | só os experimentos divergentes (coluna `divergence_reason`) |

> Um `category=divergent` (ex.: `wandb=failed but local=ok`) indica que o run treinou e salvou
> checkpoint localmente, mas o W&B marcou falha/crash — vale revisar antes de re-enfileirar.

Sobrescreva os caminhos com `--csv` e `--divergences-csv` se precisar.

### Convenção de nome

```
distillation_<dataset>_split<N>_pct<P>_next_layers_direct
```

### Layout de saída

```
artifacts/distillation/
  distillation_<dataset>_split<N>_pct<P>_next_layers_direct/
    checkpoints/
      best-*.ckpt
      last.ckpt
    run_metadata.json
  run_manifest_conv.csv
  run_manifest_conv_retry.csv
```

### SVM dos modelos destilados (conv)

```bash
# Roda em tmux (demora)
tmux new-session -d -s svm_conv "python -m src.evaluate.svm_distillation_conv 2>&1 | tee logs/svm_conv_$(date +%Y%m%d_%H%M%S).log"

# Filtrar por run
python -m src.evaluate.svm_distillation_conv --run eggs_split1

# → results/svm_distillation_conv_results.csv
```

O I-JEPA **não é carregado** durante o SVM — só os pesos `student.*` são extraídos do checkpoint.

---

## AutoEncoder não-supervisionado — encoder FLIM + decoder ResNet

Nenhum rótulo entra na loss. O encoder FLIM é treinado só reconstruindo a própria entrada, e
o decoder ResNet é descartado no fim. A pergunta que o experimento responde é se o encoder
**sai melhor do que entrou**.

### Como o veredito é medido

Um probe SVM one-vs-one linear (`C=1e2`, sem scaler) sobre o embedding pooled de 48-d,
ajustado no train rotulado e pontuado **só na validação**. O test nunca é tocado.

A baseline não vem de CSV: `on_fit_start` roda o probe no encoder FLIM **intocado, antes do
primeiro passo de gradiente**, no mesmo run, mesmo split, mesma seed, mesmo dataloader, mesmo
probe. Isso é deliberado — os CSVs de baseline do repo pontuam o SVM no **test**
(`src/evaluate/svm.py:123-137`), enquanto este probe pontua na **validação**, então os dois
números não são comparáveis. Medindo dentro do run, o veredito fica sólido:

```
Δκ = val/svm_kappa (melhor época)  −  baseline/svm_kappa (época −1, FLIM puro)
```

| Δκ | Leitura |
|---|---|
| > 0 | reconstrução é auto-supervisão útil para este encoder pequeno |
| ≈ 0 | o embedding FLIM já está saturado |
| < 0 | reconstrução puxa o embedding para informação de baixo nível (textura, cromaticidade a/b, fundo) — o risco conhecido |

Os dois números ficam lado a lado em `run_metadata.json` (`best_val_svm_kappa` e
`baseline_flim_svm`) e no W&B a baseline é replicada como série plana, para o painel mostrar
de relance se o treino subiu ou desceu em relação ao ponto de partida.

### Seleção e parada

| | Valor |
|---|---|
| Checkpoint | `best_kappa.ckpt`, monitor `val/svm_kappa` (max) |
| EarlyStopping | `val/svm_kappa` (max), `patience=50`, `strict=False` |
| Teto de épocas | `--max-epochs 1000` |

Os dois olham `val/svm_kappa`, **não** `val/recon_loss` — o melhor *encoder* é o entregável,
não a menor reconstrução. O `strict=False` existe porque com `--svm-probe-every > 1` a kappa
não é logada nas épocas em que o probe é pulado, e um callback estrito abortaria o run em vez
de esperar a próxima.

⚠️ **`val/recon_loss` quase não se move, e isso é esperado.** BCE sobre alvo contínuo tem piso
de entropia: a loss mínima alcançável não é 0, é `−p·log p − (1−p)·log(1−p)` do próprio alvo,
≈ **0.466** para o LAB destes datasets. Os runs pousam em 0.465–0.469, ou seja, o sinal real
de reconstrução são ~0.002 em cima de uma constante. Não leia essa curva como convergência, e
não a use como critério de parada.

### Rodar o grid

```bash
# Preview — mostra o plano, não executa nada
python scripts/autoencoder_flim_ray.py --dry-run

# Grid completo: 3 datasets × {5%, 75%} × 3 splits = 18 runs
python scripts/autoencoder_flim_ray.py \
    --num-gpus 4 --max-concurrent-per-gpu 5 \
    --num-workers 8 --max-epochs 1000 --patience 50 \
    --wandb-update

# Retry após falha parcial — pula concluídos, refaz só os que falharam
python scripts/autoencoder_flim_ray.py \
    --retry --skip-existing --check-wandb --wandb-update
```

Em tmux, num comando só:

```bash
tmux new -s ae_flim "cd /dados/home/moliveira/Scalable_Hybrid_FLIM && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python scripts/autoencoder_flim_ray.py --num-gpus 4 --max-concurrent-per-gpu 5 --num-workers 8 --max-epochs 1000 --patience 50 --wandb-update 2>&1 | tee /tmp/ae_flim_$(date +%F_%H%M).log; exec bash"
```

Use o path direto do python do env, **não** `conda activate`: o shell que o tmux abre não tem
o hook do conda carregado e a sessão morre em silêncio.

### Grade

3 datasets × {5%, 75%} × 3 splits = **18 runs**.

`percentage` seleciona um `data_descriptor_perc{p}.json` diferente, que reparticiona **train
e validação as duas** — então cada percentual tem a sua própria baseline, medida numa
validação diferente. O percentual não entra na loss de reconstrução; ele decide quantas
imagens rotuladas o *probe* enxerga.

Arquitetura por dataset: eggs/larvae `ch24_32_48`, protozoan `ch24_30_48`. Os **pesos** FLIM
sempre vêm da árvore `ch24_32_48_a0.5_f5` (a única com subdiretório `models/`); para
protozoan as contagens reais de kernel lá são 24/30/48, que `get_actual_channels_from_weights`
recupera no load.

### Convenção de nome

```
encoder_decoder_FLIM_<dataset>_split<N>_pct<P>
```

### Layout de saída

```
artifacts/autoencoder_resnet_init_flim/
  encoder_decoder_FLIM_<dataset>_split<N>_pct<P>/
    checkpoints/
      best_kappa.ckpt      ← o encoder entregável
      last.ckpt
    run_metadata.json      ← best_val_svm_kappa + baseline_flim_svm lado a lado
    wandb/
  run_manifest.csv
  run_manifest_retry.csv
```

---

## Classical Classifiers Evaluation (kNN / RF / LightGBM / GP / QDA)

Evaluates frozen LeJEPA encoder embeddings with classical classifiers.
Runs all classifiers for every available experiment (dataset × split × pct × init),
saving results incrementally — safe to interrupt and resume.

```bash
# Run all (auto-resumes from results/classical_classifiers_results.csv if it exists)
python -m src.evaluate.classical_classifiers

# → results/classical_classifiers_results.csv
```

Classifiers: kNN (k=5, 10, 15), QDA, Random Forest (200 trees), LightGBM, Gaussian Process (skipped if n\_train > 2000).

After running, aggregate and plot:

```bash
# Aggregate into per-init mean/std table
python scripts/aggregate_classical_results.py

# Filter to a specific init
python scripts/aggregate_classical_results.py --init trunc_normal

# Custom input/output paths
python scripts/aggregate_classical_results.py \
  --input results/classical_classifiers_results.csv \
  --output results/classical_aggregated_custom.csv

# Generate plots (classifiers_per_init/ and inits_per_classifier/)
python scripts/plot_classical_classifiers.py

# Only merge plots with custom font sizes
python scripts/plot_classical_classifiers.py --mode merge \
  --subtitle-fontsize 36 --tick-fontsize 42 --legend-fontsize 20 --legend-ncol 4

# → artifacts/plots/classical_classifiers/
```

### If the CSV is already generated (skip model loading)

If `results/classical_classifiers_results.csv` already exists, skip the
evaluation step and go straight to aggregation and plotting:

```bash
# 1. Aggregate
python scripts/aggregate_classical_results.py
# → results/classical_classifiers_aggregated.csv

# 2. Plot
python scripts/plot_classical_classifiers.py --mode merge \
  --metrics kappa f1 \
  --subtitle-fontsize 36 --tick-fontsize 42 \
  --legend-fontsize 20 --legend-ncol 4
# → artifacts/plots/classical_classifiers/plots_compare_to_flim/merge_plots/
```

---

## SVM Evaluation (frozen encoder features)

```bash
# Use cached W&B metadata (default)
python -m src.evaluate.svm

# Refresh W&B cache first
python -m src.evaluate.svm --wandb-update

# → results/svm_results.csv
```

---

## MLP Evaluation (sequential, single process)

```bash
# All modes
python -m src.evaluate.mlp --mode all

# Freeze only
python -m src.evaluate.mlp --mode freeze

# Unfreeze only, protozoan dataset
python -m src.evaluate.mlp --mode unfreeze --dataset protozoan

# Single config
python -m src.evaluate.mlp --config configs/evaluate/mlp/freeze/helminth-eggs/split_1/pct_100/7rkcbbnk.yaml

# Refresh W&B cache first
python -m src.evaluate.mlp --mode all --wandb-update

# → results/mlp_results.csv
```

---

## Unified Evaluation Pipeline

Runs SVM and MLP evaluations for all datasets in one command and generates publication-style plots.

```bash
# Evaluate everything
python -m src.evaluate.unified_eval --model all --dataset all

# SVM only, eggs
python -m src.evaluate.unified_eval --model svm --dataset eggs

# MLP freeze, protozoan, pct 25
python -m src.evaluate.unified_eval --model mlp_freeze --dataset protozoan --pct 25

# Preview (no GPU work)
python -m src.evaluate.unified_eval --model all --dataset all --dry-run
```

| `--model` | What runs |
|---|---|
| `svm` | Linear SVM on frozen encoder features |
| `mlp_freeze` | MLP with frozen encoder |
| `mlp_unfreeze` | MLP with unfrozen encoder |
| `all` | svm → mlp_unfreeze → mlp_freeze |

### Artifact layout

```
artifacts/
  SVM/{dataset}/lejepa_pct_{pct}/
  MLP/{dataset}/lejepa_pct_{pct}/
  plots/{dataset}/
  run_manifest.csv
```

---

## t-SNE Analysis

Gera plots 2D t-SNE do test set para dois modelos:

- **FLIM** — encoder do student distilado (`3x3_BN2d_1280_one_layer`), embedding 48-dim via GAP
- **LeJEPA trunc_normal** — encoder SSL com inicialização trunc_normal, embedding 48-dim via GAP

```bash
# Gera todos os plots (3 datasets × 3 splits × 6 percentagens = 108 plots)
python -m src.evaluate.tsne_analysis

# Só o modelo FLIM
python -m src.evaluate.tsne_analysis --model flim

# Só o LeJEPA
python -m src.evaluate.tsne_analysis --model lejepa

# Dataset específico
python -m src.evaluate.tsne_analysis --dataset eggs

# Split e percentagem específicos
python -m src.evaluate.tsne_analysis --dataset larvae --split 1 --pct 100

# Limitar amostras por plot (mais rápido, para testes)
python -m src.evaluate.tsne_analysis --max-samples 500

# Regar plots já existentes
python -m src.evaluate.tsne_analysis --force
```

### Estrutura de saída

```
artifacts/TSNE_analysis/
  eggs/
    FLIM/
      split1_pct1.png
      split1_pct5.png
      split1_pct25.png
      ...
    lejepa/
      pct1/
        split1.png  split2.png  split3.png
      pct5/
      pct25/
      pct50/
      pct75/
      pct100/
  larvae/
    FLIM/
    lejepa/
  protozoan/
    FLIM/
    lejepa/
```

---

## Cleanup

```bash
python scripts/clear_wandb_cache.py
```

### Encolher checkpoints inchados (teacher de ~2,5 GB)

> ⚠️ **Use isto se encontrar algum `best.ckpt` perdido com ~2,5 GB.** Esse tamanho é sinal
> de que o teacher I-JEPA congelado (ViT-H/14, 1280-dim) foi salvo dentro do `state_dict` —
> ele é inútil para inferência/SVM e pode entupir o disco. Um checkpoint saudável (só o
> student + proj head) fica em torno de **~20 MB**.

`scripts/strip_teacher_from_ckpt.py` é **não-destrutivo por padrão**: lê `best.ckpt` e grava
`best.student.ckpt` ao lado, mantendo só `student.*` / `proj_kd.*` + hyperparams. Os
avaliadores (`svm_distillation_conv` / `svm_distill_with_projection`) carregam o `.student.ckpt`
sem nenhuma mudança.

```bash
# 1) dry-run — lista o que faria, não grava nada:
python scripts/strip_teacher_from_ckpt.py \
    --glob 'artifacts/distillation/*/checkpoints/best*.ckpt'

# 2) gravar de fato os .student.ckpt (mantém os originais):
python scripts/strip_teacher_from_ckpt.py \
    --glob 'artifacts/distillation/*/checkpoints/best*.ckpt' --apply

# alternativa: só os best referenciados pelo run_metadata.json
python scripts/strip_teacher_from_ckpt.py --from-metadata --apply
```

Para **sobrescrever in-place** e recuperar o disco (DESTRUTIVO — o teacher é perdido, mas é
redownloadável para um treino novo); a escrita é atômica (`.tmp` + replace):

```bash
python scripts/strip_teacher_from_ckpt.py \
    --glob 'artifacts/distillation/*/checkpoints/best*.ckpt' --apply --overwrite
```
