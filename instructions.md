# instructions.md — how to navigate this repository

**Audience: Claude (or any agent) opening this repo cold.**
This is a *navigation* document, not a spec. It describes the tree **as it exists on
disk**, verified by running the commands quoted here. Where `spec_refactor.md`
disagrees with this file, **the disk wins** — the spec is the intent, several points
of it diverged during implementation.

> **This is a snapshot.** The refactor is being carried out by several agents at
> once, and files moved *while this document was being written*. Every import-boundary
> claim in §4 and §7 comes with the exact `grep` that produced it — **run the grep, do
> not trust the number.** If a count disagrees, the tree is right and this file is old.

Environment: conda env `scalable_FLIM` at
`/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python`. Always invoke that
absolute interpreter, never a bare `pip`. Run everything from the repo root.

---

## 0. Thirty-second orientation

| Question | Answer |
|---|---|
| Where do I start a training? | `train.py` — the single LightningCLI entrypoint (was `src/main.py`) |
| Where do I start a grid of trainings? | `python -m experiments.ray.launch <experiment.yaml>` |
| Where does a model class live? | `methods/<method>/` — `lejepa`, `autoencoder`, `classification`, `distillation` |
| Where is the FLIM encoder? | `flim/` — reachable only through `from flim import build` |
| What is shared by all methods? | `core/` — constants, data, blocks, mixins, losses, metrics, wandb, CLI |
| Where are the probes / offline evaluation? | `eval/` |
| Where are the figures and statistics? | `analysis/` |
| What is dead? | `src/`, `scripts/`, `tools/` — **legacy, but NOT yet deletable**: 13 files still import `src.*`. See §7 |

---

## 1. The tree

### Live packages

| Path | Role | Import boundary |
|---|---|---|
| `train.py` | The one training entrypoint. `pyrootutils.setup_root(...)` then `CustomLightningCLI`. | — |
| `core/` | Shared by every method: `constants.py`, `data/`, `blocks/`, `mixins/`, `losses/`, `metrics.py`, `wandb.py`, `custom_lightning_cli.py` | Imports nothing from the project except itself |
| `flim/` | The FLIM encoder. `arch.py`, `encoder.py`, `weights.py`, `spifil.py`, `flim_residual_encoder.py`, `pyift_strategy.py`, `build.py` | **`from flim import build` is the only door.** `flim/__init__.py` exports exactly `build` |
| `methods/` | One package per method: `lejepa/`, `autoencoder/`, `classification/`, `distillation/`. Each `__init__.py` re-exports the classes so YAML `class_path` stays short (`methods.lejepa.LejepaLineModule`, not `src.modules.<long_file>.<Class>`) | May import `core`, `flim` (via `build`) |
| `eval/` | Probes: `svm.py`, `mlp.py`, `classical_classifiers.py`, `unified_eval.py`, `growth_stages.py`, `tsne.py`, `eval_plotter.py`, `ray_queue.py`, `wandb_resolver.py`, plus `svm_variants/` (10 one-per-experiment probes) | Consumer layer — reaches into `methods.*` and `flim.*` internals on purpose. See §4 |
| `experiments/` | Orchestration: `ray/` (the engine), `ckpt/`, `oneoff/`, `gen_configs.py`, `constants.py`, `run_metadata_callback.py` | May import everything; **nothing imports it back** (one documented exception, §4) |
| `analysis/` | `stats/`, `activations/`, `distill/`, `plots/`, `checks/`. Figures and statistics | **Leaf: nothing imports it.** §4 R2 |
| `configs/` | `default.yaml` + `dataset/<ds>/split_N.yaml` + `model/<method>/<init>/<ds>/split_N.yaml` + `generated/` | Data |

### `flim`'s only public function

```python
# flim/build.py
def build(arch_json: str, init: str, weights_path: Optional[str] = None,
          in_channels: int = 3) -> Encoder
```
Only `init="flim"` is resolved here (it loads the FLIM weights). `he`, `xavier`,
`trunc_normal`, `random` come back with torch's default init — `core/blocks/init.py`
applies the init function.

### Config layout

```
configs/default.yaml                                        # seed, trainer, callbacks, logger
configs/dataset/<dataset>/split_<N>.yaml                    # 3 datasets x 3 splits = 9
configs/model/<method>/<init>/<dataset>/split_<N>.yaml      # 81 grid cells
configs/generated/                                          # derived, regenerable (see §6, Trap C)
```

Datasets: `helminth-eggs`, `helminth-larvae`, `protozoan-cysts`. Splits: `1 2 3`.
Percentages: `1 5 25 50 75 100`.

The `init` axis is **not uniform across methods** — do not assume a full product:

| method | inits present on disk | cells |
|---|---|---|
| `lejepa` | `flim`, `he`, `random`, `trunc_normal`, `xavier` | 45 |
| `distillation` | `flim`, `trunc_normal` | 18 |
| `classification` | `flim` | 9 |
| `autoencoder` | `flim` | 9 |

Also under `configs/model/`: 10 loose legacy YAMLs (`lejepa_line_*.yaml`,
`classifier.yaml`, `lejepa.yaml`) and `configs/model/lejepa/_variants/` (4 files —
`_variants` sits at the *init* level but is **not** an init). 95 files total.

Two directory traps:
- `configs/evaluate/mlp/` is a deep tree of **empty directories**. Its 112 YAMLs
  were moved to `configs/generated/mlp/`. Anything pointing at `configs/evaluate/`
  points at nothing.
- `.gitignore` lists `configs/generated/`, but its 112 files are still **tracked**
  (`git ls-files configs/generated | wc -l` → 112). Git only ignores untracked
  paths, so the ignore line currently has no effect.

---

## 2. Running one training — the three-config protocol

```bash
python train.py fit \
  --config configs/default.yaml \
  --config configs/dataset/helminth-eggs/split_1.yaml \
  --config configs/model/lejepa/flim/helminth-eggs/split_1.yaml \
  --data.init_args.percentage=25
```

**Validate without training** — append `--print_config`. It resolves and prints the
merged config and exits; it starts no trainer and touches no GPU:

```bash
python train.py fit \
  --config configs/default.yaml \
  --config configs/dataset/helminth-eggs/split_1.yaml \
  --config configs/model/lejepa/flim/helminth-eggs/split_1.yaml \
  --data.init_args.percentage=25 --print_config
```

### Why three configs

Each file owns exactly one concern, and LightningCLI merges them left to right
(later `--config` wins):

| config | owns | changes with |
|---|---|---|
| `configs/default.yaml` | `seed_everything`, `trainer`, callbacks, W&B logger, `use_wandb` | never (per run) |
| `configs/dataset/<ds>/split_N.yaml` | `data.class_path` + `data.init_args` — parasite name, split, batch size, loader, image size | the dataset/split axis |
| `configs/model/<m>/<init>/<ds>/split_N.yaml` | `model.class_path` + `model.init_args` — arch json, FLIM weights path, `encoder_init`, lr, loss hparams | the method/init axis |

A single merged file would be a 3-way cartesian product on disk (81+ duplicated
trainer blocks). Three orthogonal files keep it 9 + 81.

### Why `percentage` comes in from outside

`percentage` is a **grid axis, not a property of the dataset**. The same
`helminth-eggs/split_1` dataset is trained at 1%, 5%, 25%, 50%, 75% and 100% —
those are six runs of the same dataset config. Baking it into
`configs/dataset/` would mean 54 dataset files instead of 9, and the split
definition would stop being the single source of truth for what "split 1" is.

Confirmed: **no file under `configs/dataset/` or `configs/model/` mentions
`percentage`**. It arrives as `--data.init_args.percentage=<pct>` on the CLI — and
that is exactly what the grid launcher emits (see §3).

```bash
# verify: should print nothing
grep -rn 'percentage' configs/dataset/ configs/model/
```

(`configs/data/percentage/<ds>_split_N/<pct>/lejepa_line.yaml` also exists — 45
overlay files from the older workflow. The current launcher does not use them.)

### Every method, validated

All five of these were run with `--print_config` and exited 0:

```bash
for m in lejepa/flim lejepa/random autoencoder/flim classification/flim \
         distillation/flim distillation/trunc_normal; do
  python train.py fit --config configs/default.yaml \
    --config configs/dataset/helminth-eggs/split_1.yaml \
    --config configs/model/$m/helminth-eggs/split_1.yaml \
    --data.init_args.percentage=25 --print_config >/dev/null && echo "OK $m" || echo "FAIL $m"
done
```

Note on distillation: its model YAML **replaces** `default.yaml`'s callback list on
purpose. `default.yaml` monitors `val/kappa_with_proj` and `val/kappa_no_proj`, which
distillation modules never log — a missing monitor kills the fit. Distillation
monitors `val/knn_kappa` and `val/loss` instead.

---

## 3. Running a grid

```bash
python -m experiments.ray.launch experiments/lejepa/init_ablation.yaml --dry-run
```

`--dry-run` runs the full preflight, prints every command it would launch, and exits.
It writes no log file and no `work_dir`.

**One caveat, and it matters:** a YAML with `skip: wandb` hits W&B and **rewrites
`configs/wandb_update/ids_wandb.json` even under `--dry-run`**. For a pure plan:

```bash
python -m experiments.ray.launch experiments/lejepa/init_ablation.yaml --dry-run --set skip=state
```

All four experiment YAMLs on disk pass `--dry-run` today (verified):

| file | runner | grid points |
|---|---|---|
| `experiments/lejepa/init_ablation.yaml` | `train` | 270 |
| `experiments/lejepa/eval_mlp_freeze.yaml` | `eval` | 54 jobs (globbed, not cartesian) |
| `experiments/distillation/conv_1x1_grid.yaml` | `train` | 3 |
| `experiments/autoencoder/growth_grid4.yaml` | `growth` | 18 |

These directories hold **only** the YAML — no `__init__.py`, no Python. They are data.

### The CLI — exactly one positional and four flags

```
python -m experiments.ray.launch <config.yaml> [--dry-run] [--fail-fast]
                                 [--gpu-ids 0 1 ...] [--set KEY=VALUE ...]
```
`--set` takes a dotted path and parses the right-hand side with `yaml.safe_load`
(`5` → int, `[5,50]` → list, `true` → bool). Repeatable. Precedence is
CLI flag > YAML > schema default; overrides are written into a temp copy of the YAML
so preflight checks the grid that will actually run.

`experiments/ray/launch.py:118` says it plainly: *"As quatro flags autorizadas e o
caminho do YAML. Nao acrescente uma quinta."*

### The experiment YAML schema

**`experiments/ray/schema.py` is the authority.** Plain dataclasses, no pydantic.
Unknown keys at any level are a hard error (`_check_keys`, `schema.py:71`).

Top level (`Experiment`, `schema.py:254`):

| key | type | required | default | meaning |
|---|---|---|---|---|
| `name` | str | **yes** | — | run-name prefix + log filename; no `/`, no whitespace |
| `method` | str | **yes** | — | must be an existing dir under `methods/`; selects the model-config path |
| `runner` | str | **yes** | — | `train` \| `growth` \| `eval` |
| `resources` | map | **yes** | — | GPUs and parallelism |
| `output` | map | **yes** | — | `work_dir` (required) and `log_dir` (default `logs`) |
| `grid` | map | no | all axes `None` | cartesian axes; `None` = use the full valid set |
| `overrides` | flat dict | no | `{}` | each pair becomes a literal `--<key>=<value>` for LightningCLI. **Nested mappings are rejected** |
| `runner_args` | dict | no | `{}` | runner-specific knobs, validated against `RUNNER_ARGS[runner]` |
| `wandb` | map | no | disabled | `enabled` / `project` / `entity`; `project` mandatory when enabled |
| `skip` | str | no | `state` | `none` \| `state` \| `wandb` — how "already ran" is decided |

`grid` (`schema.py:142`): `init` (`flim he xavier random trunc_normal`), `datasets`
(the 3 long names), `splits` (`1 2 3`), `percentages` (`1 5 25 50 75 100`). Each
present axis must be a non-empty list of valid values.

`resources` (`schema.py:163`): `gpu_ids` (**required**, list[int]); `max_per_gpu`
(default `1`; int = same limit everywhere, or a dict `{0: 7, 1: 3}` for heterogeneous
slots, keys matching `gpu_ids` exactly); `max_total_per_gpu` (default `None` = no
quota; a per-GPU total for the whole queue, `0` removes that GPU);
`cpus_per_experiment` (default `4`); `num_workers` (default `4`, **validated but
currently not translated into any override by any runner**); `ray_address`
(default `None` = local).

`runner_args` allowed keys (`RUNNER_ARGS`, `schema.py:48`):
- `train` — **none.** Everything hparam-ish goes in `overrides`.
- `growth` — 19 keys: `max_rounds`, `kappa_tolerance`, `rounds_patience`,
  `embed_mode`; the SPiFiL grow knobs (`out_channels`, `kernel_size`, `pool_stride`,
  `n_superpixels`, `n_images`, `image_size`, `seed`, `spifil_in_feature`,
  `spifil_in_image`, `impurities`, `one_per_class`, `random_layer`,
  `random_layer_classic`); and the stage-chain shape `head_finetune` /
  `unfrozen_after_stage_two` (**mutually exclusive**).
- `eval` — `probe` (`mlp`/`svm`), `freeze` (true = only `freeze/`, false = only
  `unfreeze/`, absent = both), `ckpt_selection` (`best`/`last`; **only `best` is
  implemented** — `last` raises `NotImplementedError` in `eval/mlp.py`).

### What the engine actually emits

The dry-run proves the launcher builds the same three-config command from §2:

```
python train.py fit --config <abs>/configs/default.yaml \
  --config configs/dataset/helminth-eggs/split_1.yaml \
  --config configs/model/lejepa/flim/helminth-eggs/split_1.yaml \
  --data.init_args.percentage=5 \
  --trainer.logger.init_args.name=lejepa_line_helminth-eggs_split_1_pct_5_model_flim \
  --trainer.devices=1 --trainer.max_epochs=1000
```

### The rest of `experiments/ray/`

| file | role |
|---|---|
| `grid.py` | `build(exp) -> (cells, skipped)`: product over the four axes, resolves paths, asks `skip.should_skip`, returns cells + skipped-with-reason |
| `paths.py` | **The single owner of path resolution.** The package's only two config f-strings live here, plus `run_name` (see §5) |
| `preflight.py` | 16 read-only checks over the raw YAML and the expanded grid; prints all failures at once, `SystemExit(1)`. Also runnable: `python -m experiments.ray.preflight <yaml>` |
| `skip.py` | The one answer to "has this cell already run?" for the three `skip:` modes |
| `execution_state.py` | JSON of per-experiment outcomes for resume; today only `runners/eval.py` uses it |
| `gpu_slot_scheduler.py` | Hands out GPU ids by concurrency slot. This is why Ray runs `num_gpus=0` and each child sets its own `CUDA_VISIBLE_DEVICES` |
| `runners/train.py` | One grid point = one `train.py` subprocess. `build_cmd` is pure and shared by dry-run and production |
| `runners/eval.py` | MLP/SVM probes over trained checkpoints; the plan is a **glob** of `configs/generated/mlp/**/*.yaml` **filtered** by the grid axes, not a cartesian product |
| `runners/growth.py` | SPiFiL growth protocol; one arm per (dataset, split, percentage) pinned to one GPU end to end |
| `../run_metadata_callback.py` | Lightning `Callback` declared in YAML; writes each stage's `run_metadata.json` — the contract file growth's stopping rule and `skip.py` read |

---

## 4. Boundary rules — and the exact grep that checks each one

These are what keep the repo from becoming spaghetti again. **Each rule below is
stated with its real scope: three of the five have documented exceptions on disk.
Do not "fix" the exceptions — verify against the command, not against the prose.**

### R1 — `methods/*` touches FLIM only through `from flim import build`

```bash
grep -rn "from flim\.[a-z]" --include='*.py' methods/ core/
# EXPECTED: empty
```

Holds. The five allowed lines are all exactly `from flim import build`:

```
methods/lejepa/lejepa_flim_model.py
methods/autoencoder/autoencoder.py
methods/autoencoder/autoencoder_flim.py
methods/autoencoder/autoencoder_classifier.py
methods/classification/classification_flim_module.py
```

**Exception, deliberate:** `eval/` does reach into `flim.arch`, `flim.encoder`,
`flim.weights`, `flim.flim_residual_encoder` — 8 lines across `eval/mlp.py`,
`eval/tsne.py` and 4 `svm_variants/`. `eval/` is a consumer layer, not a method.

### R2 — `analysis/` is a leaf

```bash
grep -rnE "^[[:space:]]*(from|import)[[:space:]]+analysis[.[:space:]]" --include='*.py' \
  core/ flim/ methods/ eval/ experiments/ train.py
```

**Empty today — the rule holds.** Note the `[[:space:]]*` prefix: it is there on
purpose. Until recently the single violation was a *function-local* import inside
`eval/growth_stages.py`, which a `^from`-anchored scan misses entirely. Scan indented
lines too, or you will certify a leaf that is not one.

`analysis/` may import `core`, `flim`, `methods`, `eval` and `experiments` freely.
Nothing may import it back.

### R3 — nothing in `core/`, `flim/`, `methods/` imports `experiments/`

```bash
grep -rn "from experiments\|import experiments" --include='*.py' core/ flim/ methods/
# EXPECTED: empty
```

Holds today.

**Exception, on disk:** `eval/svm_variants/` breaks it in 3 files —
`svm_real_flim.py`, `eval_avg_pooling_48d.py`, `eval_svm_flim_flatten.py` all do
`from experiments.ray.paths import arch_json, flim_weights_path` and
`from experiments.constants import PARASITE_DIR`. That is the only inbound edge into
`experiments/`. Verify with:

```bash
grep -rn "from experiments" --include='*.py' eval/
```

### R4 — one edge between method packages

The only cross-package edge is the **distillation teacher**: the four distillation
modules import the LeJEPA model, and `teacher_constants.py` imports its backbone.

```bash
grep -rn "from methods\." --include='*.py' methods/ | grep -v "from methods\.\([a-z_]*\)" \
  ; grep -rn "^from methods\.\|^from \.\." --include='*.py' methods/
```

Simplest exact check — list every cross-package line and confirm it names only
`lejepa`:

```bash
grep -rn "from methods\.lejepa" --include='*.py' methods/distillation/
grep -rn "ijepa_encoder" --include='*.py' methods/distillation/
```

The six real edges — all `methods/distillation` -> `methods/lejepa`, nothing else:

```
methods/distillation/distillation_module.py:65            from methods.lejepa import LeJEPAFLIMModel
methods/distillation/distillation_conv_module.py:67       from methods.lejepa import LeJEPAFLIMModel
methods/distillation/distillation_one_layer_module.py:79  from methods.lejepa import LeJEPAFLIMModel
methods/distillation/distillation_two_layer_module.py:82  from methods.lejepa import LeJEPAFLIMModel
methods/distillation/teacher_constants.py:33              from ..lejepa.ijepa_encoder import IJEPAEncoder
methods/distillation/frozen_teacher.py:37                 from ..lejepa.ijepa_encoder import IJEPAEncoder
```

`LeJEPAFLIMModel` is the distillation **teacher**; `IJEPAEncoder` is that teacher's
backbone. Everything else matching `from methods.<x>` is a package importing itself.

### R5 — the `constants.py` direction

Local may import global. Global never imports local. Local never imports another
package's local.

```bash
for f in core/constants.py flim/constants.py eval/constants.py \
         experiments/constants.py analysis/constants.py \
         methods/distillation/teacher_constants.py; do
  echo "== $f"; grep -nE "^\s*(from|import) " "$f"; done
```

One false positive to expect: `experiments/constants.py:35` matches, but that line is
a usage example **inside the module docstring**, not an import.

Actual graph:
```
core/constants.py           stdlib only (Path) — the root
  ├── experiments/constants.py   from core.constants import PROJECT_ROOT
  └── analysis/constants.py      from core.constants import DATASETS, METRICS, PERCENTAGES
flim/constants.py           zero imports
eval/constants.py           zero imports (self-declared "arquivo PLANO")
methods/distillation/teacher_constants.py   imports ..lejepa.ijepa_encoder.IJEPAEncoder
                                            — NOT a flat data file; it is the R4 edge
```
No cycles.

### R6 — `methods/` -> `eval/`: one edge, and it is temporary

```bash
grep -rn "from eval\.\|import eval\." --include='*.py' methods/ core/ flim/
```

One line today:
`methods/autoencoder/autoencoder_flim_module.py:120: from eval.svm import EMBED_MODES, _encode_pooled`

This is a **model module importing the probe layer** — the direction the rest of the
tree avoids. It is deliberate and flagged in place: the module's in-training SVM probe
embeds with the official evaluator's own function rather than a look-alike, because
two identical implementations are one drift away from disagreeing. It also means
`eval.svm.EMBED_MODE` governs the probe for free (see `embed_mode` in §5). The comment
above it carries a `ponytail:` marker — `_encode_pooled` has no settled home yet.
Do not add a second edge in this direction.

---

## 5. What you must not touch

### The five `acc` conventions — deliberate, do not unify

`core/metrics.py:74` defines the canonical `acc` as **macro / balanced** accuracy.
Five modules deliberately diverge, and the divergence is written into the source as a
comment in each case. The sharpest one: **`val/acc` is the same W&B key logged with
two different reductions.**

| convention | key(s) | reduction | where |
|---|---|---|---|
| A | `train/acc`, `val/acc` | **micro** (torchmetrics `Accuracy` object, default average) | `methods/classification/classification_finetune_module.py:143,156` |
| B | `train/acc` (micro inline), `val/acc` (**macro** via `compute_metrics`) | mixed | `methods/classification/classification_flim_module.py:187,215` |
| C | `train/kd_acc`, `val/kd_acc` | **micro**, inline `(logits.argmax(1)==y).float().mean()`, never `compute_metrics` | all 4 `methods/distillation/*_module.py` |
| D | `{stage_prefix}/head_acc`, `{stage_prefix}/svm_acc` | **macro** via `compute_metrics`; the key name is itself computed from `_stage_prefix` | `methods/autoencoder/autoencoder_flim_module.py:793,820` |
| E | `val/svm_acc`, `val/svm_acc_with_proj` | **micro** via `sklearn.accuracy_score`, from a **KNN** probe (not an SVM, despite the name) | `methods/lejepa/lejepa_line_module.py:333,341` |

Consequence for analysis: crossing a curve from (A)/(C)/(E) with a number from
`core.metrics.compute_metrics` compares **micro with macro**. Eval-side CSVs sidestep
this by emitting both columns: `acc` (macro) and `acc_raw` (micro).

No module uses `train_acc`/`val_acc` as a *key* — those are only Python attribute
names. Every logged key is slash-separated.

### `run_name` — reproduces historical formulas byte for byte

`experiments/ray/paths.py:170-227`, and its own docstring is the warning:

> *"ESTE E O UNICO PONTO DO PACOTE QUE NAO FOI UNIFORMIZADO, E E DE PROPOSITO. …
> Uma formula nova, por mais uniforme que fosse, deixaria TODO checkpoint ja gravado
> inalcancavel pelo motor novo e faria `skip` reportar a grade inteira como pendente.
> … Nao 'conserte' isto de volta para uma f-string so."*

`run_name` decides **two** things at once: the artifact directory
(`<work_dir>/<run_name>/checkpoints/`) and the W&B display name. Change it and you
lose the link to every weight already on disk.

Six branches, one per historical family, each a separate function citing its source:
`_name_lejepa` (`:230`), `_name_autoencoder` (`:239`), `_name_classification`
(`:253`), `_name_distillation` (`:267`), `_name_distillation_conv` (`:277`, a 9-branch
chain with two early `return`s that bypass the `_no_imagenet_norm` suffix on purpose),
`_name_growth` (`:313`). Registry at `:330` — *"nao e registry: e um `if` escrito como
dict"*.

Two traps inside it:
- `_name_distillation` produces `...model{dist_type}` — the `model...` tail is the
  **distillation type**, not the init, despite matching the lejepa spelling.
- The lejepa spelling is matched by a regex, `_CANONICAL_RE` in `core/wandb.py:51-53`.
  Changing one without the other silently breaks W&B lookup.
- `experiments/ray/runners/growth.py:229` builds its W&B display name **by hand**,
  not through `paths.run_name`. Known divergence.

### `embed_mode` defaults — they differ on purpose

Allowed values: `("avgpool2d", "flatten")` (`eval/svm.py:306`).

| where | default |
|---|---|
| `eval/svm.py:298` `DEFAULT_EMBED_MODE` | `avgpool2d` |
| `eval/unified_eval.py:650` | `avgpool2d` |
| `experiments/ray/paths.py:249`, `experiments/run_metadata_callback.py:215` | `avgpool2d` |
| `methods/autoencoder/autoencoder_flim_module.py:263` | **`None`** — inherits the module global |
| `experiments/ray/runners/growth.py:145` | **`flatten`** |
| `eval/growth_stages.py:126`, `eval/svm_variants/eval_svm_flim_flatten.py:112` | **`flatten`** |

The `flatten` in the growth path is a documented deliberate divergence
(`growth.py:139-142`): that experiment's kappa is measured on the flattened last
feature map, and the value is therefore **always emitted explicitly** in the child
command, never by omission.

### `class Head` and `StudentClassificationHead` are NOT `MLPHead`

| class | file | body | state_dict keys |
|---|---|---|---|
| `Head` | `methods/autoencoder/autoencoder_flim_module.py:173` | GAP → Flatten → Dropout(0.2) → **one** Linear | `head.fc.weight`, `head.fc.bias` |
| `StudentClassificationHead` | `methods/distillation/student_classification_head.py:37` | a single `nn.Linear` | `cls_head.head.*`, `teacher_cls_head.head.*` |
| `MLPHead` | `core/blocks/mlp_head.py:30` | GAP → Sequential of **three** Linear + ReLU + Dropout(0.3) | `classifier.{0,3,6}.*` |

Swapping either for `MLPHead` renames **every** key, changes the parameter count and
the head architecture — every `.ckpt` with `head.*` / `cls_head.*` stops loading. It
would also break `experiments/run_metadata_callback.py:209-211`, which reads
`pl_module.head.fc.in_features` directly (`MLPHead` has no `.fc`).

`MLPHead` *is* used legitimately elsewhere (`core/blocks/classification_model.py:42`,
`methods/autoencoder/autoencoder_classifier.py:58`) — different checkpoint lineages.

### Weights on disk

- **`/exps/scalable_hybrid_FLIM_weights` does not exist on this machine.** `/exps`
  exists but holds unrelated directories. `grep -rI "scalable_hybrid_FLIM_weights"`
  over the whole repo returns nothing — it is not referenced by any code or config.
  Treat it as a path belonging to the other host in the athena/jaci pair (see the
  `sync_check_*.log` files at the root), **not** as something to resolve locally.
- The FLIM weights the code actually loads are repo-relative, from
  `core/constants.py:151-160`:
  `data/to_mateus/model/ch24_32_48_a0.5_f5/<dataset>/train1/{architecture.json,models}`.
  Documented invariant there: for `protozoan` the arch tree (`ch24_30_48`) and the
  weights tree (`ch24_32_48`) **diverge**; the two dicts are not redundant.
- Training artifacts land under `artifacts/<work_dir>/<run_name>/checkpoints/`.
  Rule from the owner: keep `best`, drop `last` — unless a run only has `last`.

---

## 6. Traps that already cost time and will bite again

### Trap A — `parents[N]` breaks in silence when a file changes depth

A file computing the repo root with `Path(__file__).resolve().parents[N]` does not
fail when moved — it silently resolves to the wrong directory, and every downstream
path is wrong.

```bash
grep -rn "parents\[" --include='*.py' . --exclude-dir=.git --exclude-dir=wandb --exclude-dir=artifacts
```

14 hits today, **all correct**: `parents[1]` in `core/constants.py:53` (1 level down),
`parents[2]` in `experiments/ray/schema.py:36` and in 8 `analysis/*/*.py` files
(2 levels down). Two hits in `analysis/activations/heatmap_stages.py:129,133` walk a
*checkpoint* path, not the repo root — outside the trap.

The same computation also hides behind `.parent.parent`, which this grep misses — e.g.
`analysis/stats/wilcoxon_acc.py:113-114` and `flim/spifil.py:257`. Grep both spellings
when you move a file.

The historical record is preserved in a comment at
`analysis/activations/heatmap_stages.py:79`: *"analysis/activations/ esta a 2 niveis
da raiz do repo (o arquivo veio de tools/, que era 1)."* That is exactly the move that
would have broken it.

**When you add a file, do not recompute the root.** Do
`from core.constants import PROJECT_ROOT`. Only `train.py` uses
`pyrootutils.setup_root`, and only because it is the entrypoint. If you must use
`parents[N]`, write the depth comment next to it, the way `schema.py:35` and
`core/constants.py:50-52` do.

### Trap B — `save_hyperparameters()` needs the *word* `super` in `__init__`

Lightning's `_get_init_args`
(`.../lightning/pytorch/utilities/parsing.py:92-96`) reads `__class__` out of the
`__init__` frame:

```python
if "__class__" not in local_vars or frame.f_code.co_name != "__init__":
    return None, {}
```

CPython only creates that implicit `__class__` closure cell when the name `super`
appears **lexically** in the method. No `super` → no cell → `save_hyperparameters()`
returns silently empty → `hparams` is `{}` → every config's hyperparameters are lost,
with no error.

This is why two modules use the 2-argument form: they *skip* the parent constructor
on purpose but still write `super`:

```python
# methods/lejepa/lejepa_line_module.py:127-132
# super de 2 argumentos: pula a construcao dos dois pais — esta classe monta o
# proprio model e tem a propria tupla ENCODER_INITS — e, por mencionar o nome
# `super`, mantem a cell implicita `__class__` no frame, sem a qual o
# save_hyperparameters abaixo sai vazio e as 30 configs perdem os hparams.
super(LeJEPAModule, self).__init__()
```

Rewriting that as `pl.LightningModule.__init__(self)` compiles `__class__` away and
empties `hparams`. All 11 files calling `save_hyperparameters()` are clean today:

```bash
grep -rln "save_hyperparameters" --include='*.py' core/ methods/ eval/
```

### Trap C — the config generator's cleanup loop deleted siblings

`scripts/generate_mlp_configs.py:117-122` deleted **every** sibling `.yaml` in the
destination directory on each write. That was safe while the destination was
`<mode>/<ds>/split_N/pct_P/` (one run_id per directory). After the path was flattened
to `<mode>/<ds>/` — split and percentage became keys *inside* the YAML — each
directory hosts ~18 run_ids, so that loop would have deleted 17 of them per file
written, collapsing 582 YAMLs to 8 while printing `[DEL]` as if normal.

Current code, `experiments/gen_configs.py:187-195`:

```python
removed = 0
for target_dir in sorted({os.path.dirname(p) for p in planned}):
    os.makedirs(target_dir, exist_ok=True)
    for old_yaml in sorted(os.listdir(target_dir)):
        old_path = os.path.join(target_dir, old_yaml)
        if old_yaml.endswith(".yaml") and old_path not in planned:
            os.remove(old_path)
```

Safe **only** because of `old_path not in planned`, and only because the whole grid is
built into `planned` (lines 170-179) *before* any write. That ordering is the
correctness precondition — interleave build and delete and the guard degrades back to
the original bug.

`experiments/gen_configs.py` has **no argparse** (deliberate; each function has named
defaults). `python -m experiments.gen_configs` runs `generate_mlp_configs()` with
`dry_run=False` — i.e. the **deleting** path, with no flag to opt out. The safe form:

```bash
python -c "from experiments.gen_configs import generate_mlp_configs as f; f(dry_run=True)"
```

---

## 7. Legacy: `src/`, `scripts/`, `tools/` — still load-bearing

Not yet deleted, and **not yet deletable**. `src/` has 63 `.py`, `scripts/` 26,
`tools/` 1 (only `__init__.py` + `README.md`).

```bash
grep -rn "from src\.\|import src\.\|from scripts\.\|from tools\." --include='*.py' \
  core/ flim/ methods/ eval/ experiments/ analysis/ train.py
```

31 import lines across 13 files today — **and the number is falling by the hour**:

| package | files importing `src.*` | notes |
|---|---|---|
| `analysis/` | 12 | the whole remainder: `distill/` (3), `activations/` (4), `checks/` (3), `stats/` (1), `plots/` (1) |
| `experiments/` | 1 | `oneoff/run_missing_mlp.py` |
| `core/`, `flim/`, `methods/`, `eval/` | **0** | clean |

`methods/` was cleared during this session: `methods/autoencoder/autoencoder_flim_module.py:120`
used to read `from src.utils.evaluate import EMBED_MODES, _encode_pooled` and now reads
`from eval.svm import ...` (see R6 in §4). That was the last edge out of a model module
into the legacy tree.

**`src/` still cannot be deleted** — 12 `analysis/` files and one `experiments/oneoff/`
script import it. Run the grep above before assuming otherwise.

`analysis/checks/check_refactor_equivalence.py` imports `src.*` **on purpose** — it
pins old-vs-new behaviour and must keep both sides reachable. It is not a leftover.

Also live: `flim/spifil.py:257` reads `scripts/spifil_paper_seeds.json` — a **data**
file under `scripts/`, not code.

One config still points at the legacy namespace:
`configs/model/lejepa/_variants/custom_cnn.yaml` uses
`class_path: src.modules.LeJEPACNNModule`.

Other legacy at the root: `config.py`, `config.yaml`, `run_experiments.py`, `test.py`,
`retry_models_again.py`, and the directories `analysis_flim_distill/`,
`check_experiments/`, `statistics/`, `analises_wandb_training/`, `new_hope_jepa/`
(all five contain zero `.py` — their contents were merged into `analysis/`).

---

## 8. Where to look before you grep

### 1. graphify — but it is stale

A knowledge graph lives in `graphify-out/`. **It is out of date.**
`graphify-out/graph.json` was built **2026-08-26**; the structural refactor landed
**2026-08-30**. Checked: the graph contains **zero** references to `methods/`,
`core/`, `flim/`, `eval/` and thousands to `src/models/models.py`,
`src/modules/*`. Querying it today will send you to files that moved.

Rebuild first — it takes about 9 seconds, is local, and uses no LLM:

```bash
graphify update .
```

Then query it (binary at `~/.local/bin/graphify`):

```bash
graphify query "<question>"        # BFS traversal from the question
graphify explain "<node>"          # the node and its neighbourhood
graphify affected "<node>"         # what breaks if I change this
graphify path "A" "B"
graphify god-nodes
```

Cite the node's `source_location` when you answer from the graph.

### 2. The refactor reports

Every piece of this refactor was written up by the agent that did it. The reports are
in this session's scratchpad:

```
/tmp/claude-1033/-dados-home-moliveira-Scalable-Hybrid-FLIM/<session-id>/scratchpad/refactor_reports/
```

~75 files. Useful entry points by topic: `FIX-RUNNAME.md` (the run_name formulas),
`FIX-IMPORTS.md`, `FIX-SCHEMA.md`, `MK-EMBEDMODE.md`, `E-EXP.md` / `E1-E4.md`
(the Ray engine), `E-EVAL.md` / `E-EVAL-VAR.md` (`eval/` and `svm_variants/`),
`E-ANA.md` (`analysis/`), `Y-*.md` (the config YAML generation), `MD-*.md` /
`ML-*.md` / `MA-*.md` (the method packages), `C1-INV.md` (the invariant audit).

### 3. `spec_refactor.md`

156 KB of intent at the repo root. Read it for **why**, never for **where** — several
of its decisions changed during implementation. Confirm every path it names with
`ls` before acting on it.
