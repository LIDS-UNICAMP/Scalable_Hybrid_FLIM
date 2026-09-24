# EXPERIMENTS.md — como reproduzir cada grupo de experimento

Guia de reprodução. Para navegar o repositório, veja `instructions.md`; para traduzir
um comando antigo no novo, veja `MIGRATION.md`.

> **Aviso de verificação.** Este documento foi escrito a partir do estado do disco e
> dos relatórios da refatoração. Os comandos de grade e de célula única seguem o
> protocolo que foi testado durante a refatoração (`--dry-run` e `--print_config`),
> mas **os comandos deste arquivo não foram re-executados um a um**. Rode sempre
> `--dry-run` antes da primeira vez. O que é inferência está marcado como tal.

---

## 0. O eixo central: com FLIM vs sem FLIM

A pergunta que atravessa o repositório é se inicializar o encoder com FLIM bate as
inicializações clássicas. Estes são os pares prontos para lançar, reconstruídos do que
**já foi treinado** (2.587 runs no cache do W&B mais os checkpoints em disco):

| Grupo | COM FLIM | SEM FLIM | runs |
|---|---|---|---|
| **LeJEPA SSL** | `lejepa/init_flim.yaml` | `lejepa/init_no_flim.yaml` | 545 × 1.252 |
| **Destilação** | `distillation/all_init_flim.yaml` | `distillation/all_init_no_flim.yaml` | 1.299 × 794 |
| **Crescimento SPiFiL** | `autoencoder/growth_*.yaml` (9 famílias) | — ¹ | 893 |
| **Classhead** | `classification/classhead_flim.yaml` | — ² | 204 |

O braço sem FLIM do LeJEPA agrega os quatro: `he` (397), `xavier` (338), `random` (338),
`trunc_normal` (179). Na destilação, o único não-FLIM que rodou é `trunc_normal`.

¹ O crescimento não tem contrafactual de init — o autoencoder só aceita `flim`. O
controle dele é outro: `growth_g5_random.yaml`, que usa `random_layer` (troca só os
**valores** dos pesos no fim, preservando a norma filtro a filtro).

² Idem: o módulo levanta `ValueError` sem pesos FLIM.

```
python -m experiments.ray.launch experiments/lejepa/init_flim.yaml     --dry-run
python -m experiments.ray.launch experiments/lejepa/init_no_flim.yaml  --dry-run
```

> `skip: state` em todos: relançar pula o que já tem checkpoint em disco, então dá para
> rodar por cima do que já existe sem repetir trabalho.

---

## 0.1. Os dois jeitos de rodar

### Uma célula (um experimento só)

```
python train.py fit \
  --config configs/default.yaml \
  --config configs/dataset/<dataset>/split_<N>.yaml \
  --config configs/model/<metodo>/<init>/<dataset>/split_<N>.yaml \
  --data.init_args.percentage=<pct>
```

São **três** `--config` e o LightningCLI empilha nessa ordem: o último vence. O global
traz trainer e callbacks; o de dataset traz o parasita e o split; o de método traz a
classe, os pesos FLIM e os hiperparâmetros.

**`percentage` não está no config de dataset, e isso é de propósito.** Ele é eixo da
grade, não propriedade do dataset — por isso vem por fora. Um config de dataset sozinho
**não instancia**: falha alto dizendo que falta `percentage`, em vez de cair num default
silencioso.

Trocar `--print_config` por `fit` imprime a configuração resolvida sem treinar. Use isso
para conferir antes de queimar GPU.

### Uma grade inteira

```
python -m experiments.ray.launch experiments/<metodo>/<nome>.yaml --dry-run
python -m experiments.ray.launch experiments/<metodo>/<nome>.yaml
```

`--dry-run` imprime cada comando montado e sai sem executar nada. O pré-voo roda
**antes** do `ray.init()` e nenhum job é submetido se ele falhar — ele nomeia o arquivo
culpado e lista os caminhos que faltam, um por linha.

Flags do launcher (as únicas quatro que sobreviveram no repositório inteiro):

| flag | para quê |
|---|---|
| `--dry-run` | imprime e sai |
| `--fail-fast` | para no primeiro erro |
| `--gpu-ids 0 1` | sobrepõe `resources.gpu_ids` |
| `--set chave=valor` | sobrepõe qualquer chave do YAML (`--set grid.percentages=[5,50]`) |

**`skip: wandb` consulta a rede.** Se você não quer isso, rode com `--set skip=state`,
que decide pelo estado em disco.

Autoridade do schema: `experiments/ray/schema.py`. Chave desconhecida em `runner_args`
é **erro**, não aviso.

---

## 1. LeJEPA SSL — a ablação de inicialização

**Pergunta:** a inicialização FLIM do encoder bate as inicializações clássicas no
pré-treino auto-supervisionado?

**Configs:** 45 células — `configs/model/lejepa/<init>/<dataset>/split_N.yaml`, com
`init ∈ {flim, he, xavier, random, trunc_normal}`. A grade está **completa**: as 24
células que faltavam (os inits não-flim de eggs e larvae) foram geradas a partir dos
genéricos que o launcher antigo usava, preservando os hiperparâmetros de cada init.

> Atenção: `trunc_normal` usa `lr: 5.0e-4` e `weight_decay: 1.0e-5`; os outros três
> não-flim usam `3.0e-3` e `5.0e-2`. Não é engano — é o experimento como sempre foi.

**Grade inteira:**
```
python -m experiments.ray.launch experiments/lejepa/init_ablation.yaml --dry-run
```
270 pontos (5 inits × 3 datasets × 3 splits × 6 percentuais).

**Uma célula:**
```
python train.py fit \
  --config configs/default.yaml \
  --config configs/dataset/helminth-eggs/split_1.yaml \
  --config configs/model/lejepa/flim/helminth-eggs/split_1.yaml \
  --data.init_args.percentage=50
```

**Saída:** `artifacts/lejepa/init_ablation/`. Nome do run:
`lejepa_line_<dataset>_split_<N>_pct_<P>_model_<init>`.

**Variantes fora da grade** (`configs/model/lejepa/_variants/`): `real_sigreg.yaml` e
`simple_sigreg.yaml` trocam o eixo `sigreg_type`; `resnet50.yaml` troca o encoder.
**`custom_cnn.yaml` está em QUARENTENA** — aponta para uma classe que nunca existiu, e
o cabeçalho do arquivo explica com as evidências. Não use.

**Avaliar:** seção 5.

---

## 2. Autoencoder FLIM + crescimento SPiFiL

**Pergunta:** crescer o encoder FLIM camada a camada, guiado por superpixels, melhora a
representação em relação a parar no encoder base?

**Configs:** 9 células — `configs/model/autoencoder/flim/<dataset>/split_N.yaml`.
**Só existe o init `flim`**, e isso é correto: o módulo levanta `ValueError` sem pesos
FLIM, então as outras 36 células seriam experimentos que abortam antes da primeira época.

**Grade:**
```
python -m experiments.ray.launch experiments/autoencoder/growth_grid4.yaml --dry-run
```
18 pontos, 144 comandos de estágio — o runner encadeia `treina → cresce → treina`, e
cada estágio recebe o `best_kappa.ckpt` do anterior via `init_ckpt`.

**Uma célula (só o estágio 1, sem crescer):**
```
python train.py fit \
  --config configs/default.yaml \
  --config configs/dataset/helminth-eggs/split_1.yaml \
  --config configs/model/autoencoder/flim/helminth-eggs/split_1.yaml \
  --data.init_args.percentage=50
```

**Saída:** `artifacts/spifil_growth/<name>/<parasita>_split<N>_pct<P>/<estagio>/`.
Nome do run: `spifil_growth_<name>_<parasita>_split<N>_pct<P>_<estagio>`.

> **O `name` do YAML vira o diretório dos pesos.** Trocá-lo quebra o link com os
> checkpoints já gravados. Ele deve ser o basename do `work_dir` (`grid4`, não
> `spifil_growth_grid4` — há uma guarda contra o prefixo duplicado, mas não conte com ela).

### As famílias já rodadas — todas têm experiment YAML

Cada família de `artifacts/spifil_growth/` tem agora o seu YAML em
`experiments/autoencoder/`. Os knobs foram **reconstruídos das flags nos logs de cada
corrida** (`logs/<familia>.log`), não do `run_metadata.json` — este grava só os hparams
do estágio de treino, e os knobs de `grow` não aparecem nele.

| Família | Datasets | pcts | Knobs que a definem | YAML |
|---|---|---|---|---|
| `grid3` | eggs, larvae, protozoan | 5, 50 | baseline, sem controle | `growth_grid3.yaml` |
| `grid4` | eggs, larvae, protozoan | 5, 50 | baseline | `growth_grid4.yaml` |
| `g5_in_image` | eggs, larvae, protozoan | 5, 50 | `spifil_in_image`, `impurities`, `one_per_class` | `growth_g5_in_image.yaml` |
| `g5_in_feature` | eggs, larvae, protozoan | 5, 50 | `spifil_in_feature`, `impurities`, `one_per_class` | `growth_g5_in_feature.yaml` |
| `g5_head` | eggs, larvae, protozoan | 5, 50 | `head_finetune` + `spifil_in_image` + `impurities` + `one_per_class` | `growth_g5_head.yaml` |
| `g5_random` | eggs, larvae, protozoan | 5, 50 | `random_layer` + `head_finetune` + `one_per_class` | `growth_g5_random.yaml` |
| `g5_head_larvae` | larvae | 5, 50 | mesma do `g5_head`, restrita | `growth_g5_head_larvae.yaml` |
| `g5_head_repro` | larvae | 5, 50 | replicação do anterior | `growth_g5_head_repro.yaml` |
| `g5_head_nsp450` | larvae | 5, 50 | `impurities`, `one_per_class`, `n_superpixels: 450` ⚠ | `growth_g5_head_nsp450.yaml` |
| `_debug` | — | — | descartável | — |

⚠ **O `450` do `g5_head_nsp450` vem do NOME, não do log** — o `--n-superpixels` não
aparece nas flags registradas. Confirme antes de rodar.

O par `g5_in_image` × `g5_in_feature` é a comparação central: a mesma coisa com a
segmentação por superpixel feita na **imagem** ou na **grade de features**.

```
python -m experiments.ray.launch experiments/autoencoder/growth_g5_in_feature.yaml --dry-run
```

Os 19 knobs aceitos estão em `RUNNER_ARGS["growth"]` (`experiments/ray/schema.py`):
`max_rounds`, `kappa_tolerance`, `rounds_patience`, `embed_mode`, `kernel_size`,
`out_channels`, `pool_stride`, `n_superpixels`, `n_images`, `image_size`, `seed`,
`spifil_in_feature`, `spifil_in_image`, `impurities`, `one_per_class`, `random_layer`,
`random_layer_classic`, `head_finetune`, `unfrozen_after_stage_two`.

> Os YAMLs reconstruídos herdaram `resources` do `growth_grid4.yaml` (GPUs 1-3,
> 3 slots cada). Ajuste ao que estiver livre na máquina antes de lançar.

> **Os dois controles aleatórios são perguntas diferentes.** `random_layer` é o controle
> **pareado**: roda o pipeline inteiro e troca só os valores dos pesos no fim,
> preservando a norma filtro a filtro. `random_layer_classic` é o controle **clássico**:
> `kaiming_normal_`, sem checkpoint, sem dado, sem superpixel, sem orçamento. Não são
> intercambiáveis.

> **Orçamento de covariância.** O crescimento recusa com código de saída 3 quando
> `N <= D` (`D = in_channels × kernel²`). Isso vale também para `random_layer`, que é o
> mesmo caminho; só `random_layer_classic` escapa, porque não ajusta nada.

**Avaliar o crescimento:** `python -m eval.growth_stages` (aceita `pct=`).

---

## 3. Destilação

**Pergunta:** um aluno pequeno consegue herdar a representação do professor I-JEPA, e
qual cabeça de projeção preserva melhor a informação?

**Configs:** 18 células — `configs/model/distillation/<init>/<dataset>/split_N.yaml`,
com `init ∈ {flim, trunc_normal}`.

**O `class_path` é quem escolhe o aluno.** As quatro variantes vivem em
`methods/distillation/` e são selecionadas por `overrides` no experiment YAML, não por
flag:

| Módulo | Cabeça |
|---|---|
| `DistillationModule` | linear (base) |
| `DistillationConvModule` | convolucional |
| `DistillationOneLayerModule` | uma camada (1×1 ou 3×3, por `proj_kernel`) |
| `DistillationTwoLayerModule` | duas camadas 1×1 + BatchNorm2d |

As antigas flags `--distill_1..4` **não existem mais**: o `class_path` já é a escolha, e
`proj_kernel: 1|3` desambigua as duas que dividem o mesmo módulo.

### As 13 configurações já rodadas — todas têm experiment YAML

Reconstruídas do **argv real** gravado em `artifacts/distillation/**/wandb-metadata.json`.
Não é inferência de nome: é a linha de comando que rodou.

**Grupo A — família `distill4_*`** (`DistillationConvModule`, ex-`--distill_4 --init_flim`),
todas com eggs+larvae+protozoan, splits 1–3, pcts 1/5/25/50/75/100:

| YAML | loss | professor | runs |
|---|---|---|---|
| `distill4_cos_frozen.yaml` | cosseno | congelado | 48 |
| `distill4_cos_unfrozen.yaml` | cosseno | descongelado | 134 |
| `distill4_mse_frozen.yaml` | MSE | congelado | 2 |
| `distill4_mse_unfrozen.yaml` | MSE | descongelado | 14 |
| `distill4_kd_frozen.yaml` | KD híbrido (`fine_tune=True`) | congelado | 22 |
| `distill4_kd_unfrozen.yaml` | KD híbrido | descongelado | 135 |

**Grupo B — as famílias por `--distillation-type`:**

| YAML | init | cabeça | pcts | runs |
|---|---|---|---|---|
| `direct_trunc_normal.yaml` | trunc_normal | — (os **4** módulos) | todos | **619** |
| `direct_flim.yaml` | flim | one_layer + two_layer | todos | 450 |
| `direct_flim_1x1.yaml` | flim | 1×1 | todos | 236 |
| `direct_flim_1x1_frozen.yaml` | flim | 1×1, **encoder congelado** | todos | 222 |
| `direct_trunc_normal_1x1.yaml` | trunc_normal | 1×1 | todos | 139 |
| `cosine_flim_bn2d.yaml` | flim | 1×1 BN2d 1280 | **1, 75** | 36 |
| `cosine_trunc_normal_bn2d.yaml` | trunc_normal | 1×1 BN2d 1280 | **1, 75** | 36 |

`direct_trunc_normal` é a maior família (619 runs) e foi rodada nos **quatro** módulos —
troque `model.class_path` no `overrides` para reproduzir cada um.

```
python -m experiments.ray.launch experiments/distillation/distill4_cos_frozen.yaml --dry-run
```

**Uma célula:**
```
python train.py fit \
  --config configs/default.yaml \
  --config configs/dataset/helminth-eggs/split_1.yaml \
  --config configs/model/distillation/flim/helminth-eggs/split_1.yaml \
  --data.init_args.percentage=100 \
  --model.class_path=methods.distillation.DistillationOneLayerModule
```

**A leitura dos resultados** está em `reports/distill_details.md`, que documenta seis
grupos pelo **sufixo do run** (`2l_1x1_init_flim_256_1280_no_imagenet_norm`,
`1x1_BN2d_1280_one_layer_flim_init_no_imagenet_norm`, `3x3_BN2d_1280_one_layer_init_flim`,
`2l_1x1_init_flim_256_1280`, `next_layers_direct`, `pct75_modeldirect`) com módulo, cabeça
e contagem de parâmetros. Esses sufixos são recortes dos YAMLs acima, não famílias à parte.

**Saída:** `artifacts/distillation/<name>/`.

> Os YAMLs reconstruídos usam `gpu_ids: [0,1,2,3]`, `max_per_gpu: 2`. Ajuste ao que
> estiver livre. E `alpha: 0.5` / `temperature: 4.0` vieram do argv — confira se é o que
> você quer antes de relançar.

> **Cada YAML carrega o `num_classes` do seu próprio dataset** — eggs 9, larvae 2,
> protozoan 7. Isso importa: o conserto desse bug vivia no antigo `_run_one`, não no
> módulo. Um valor único para todos faz o bug do `--dataset all` voltar pela porta do YAML.

> **Contagem de parâmetros tem armadilhas conhecidas:** há um MLP vestigial e o encoder
> aparece duplicado no `state_dict`. São propriedades do checkpoint gravado, não sujeira
> a limpar. `reports/distill_details.md` tem a seção sobre isso.

---

## 4. Classificação (classhead)

**Pergunta:** ReLU ou sigmoid na cabeça de classificação sobre o encoder FLIM?

**Configs:** 9 células — `configs/model/classification/flim/<dataset>/split_N.yaml`.

**Uma célula:**
```
python train.py fit \
  --config configs/default.yaml \
  --config configs/dataset/helminth-eggs/split_1.yaml \
  --config configs/model/classification/flim/helminth-eggs/split_1.yaml \
  --data.init_args.percentage=100
```

**Saída:** `artifacts/classification_flim/`. Nome do run:
`classhead_<parasita>_split<N>_pct<P>_<variante>`.

O fine-tune supervisionado é outra classe, `methods.classification.ClassificationFinetuneModule`,
com config em `configs/model/classifier.yaml`. **A chave dele mudou de nome**:
`freeze_encoder` virou `freeze`, porque o nome antigo sombreava a função importada e a
classe levantava `TypeError` na construção — ela nunca instanciou. Está consertada.

---

## 5. Sondas de avaliação

Todas em `eval/`. Elas **não treinam**: rodam sobre checkpoints já gravados.

### MLP (freeze / unfreeze)

```
python -m experiments.ray.launch experiments/lejepa/eval_mlp_freeze.yaml --dry-run
```
54 jobs. Os configs de entrada são gerados, ficam em
`configs/generated/mlp/{freeze,unfreeze}/<dataset>/<run_id>.yaml` e estão no
`.gitignore` — regere com `python -m experiments.gen_configs`.

### SVM

```
python -m eval.svm
```

As **10 variantes por experimento** vivem em `eval/svm_variants/`: `svm_distillation`,
`svm_distillation_conv`, `svm_ijepa`, `svm_distill_with_projection`,
`svm_classification_flim`, `eval_svm_flim_flatten`, `eval_autoencoder`,
`svm_real_flim`, `eval_avg_pooling_48d`, `svm_flim_residual`. Cada uma roda com
`python -m eval.svm_variants.<nome>`.

> **`svm_classification_flim` grava `acc` micro E balanceada, e `f1` weighted E macro,
> de propósito** — é para comparabilidade com `data/reports_felipe/svm/`. Três outras
> gravam `acc` balanceada e `acc_raw` micro no mesmo CSV. Não unifique nenhuma delas.

### Outras

| Sonda | Comando |
|---|---|
| Classificadores clássicos (QDA, RF, LGBM, GP, kNN) | `python -m eval.classical_classifiers` |
| Avaliação unificada | `python -m eval.unified_eval` |
| Estágios de crescimento | `python -m eval.growth_stages` |
| t-SNE | `python -m eval.tsne` |

---

## 6. Análise e estatística

Uso normal: `python -m <modulo>`, que roda com os defaults.

| Grupo | Módulos |
|---|---|
| `analysis.stats` | `wilcoxon_acc`, `wilcoxon_f1`, `wilcoxon_kappa`, `wilcoxon_equivalence`, `wilcoxon_flim_init`, `compute_cost` |
| `analysis.activations` | `heatmap_stages`, `relu_vs_sigmoid`, `saturation`, `unit_activations` |
| `analysis.distill` | `embedding_analysis`, `ruler_mismatch`, `distill_destroys_flim` |
| `analysis.plots` | `plot_comparacao_flim_protocolo`, `plot_continuity_spifil_hybrid`, `plot_partial_train_spifil_hybrid`, `plot_sigmoid_saturation`, `plot_svm_results` |
| `analysis.checks` | `check_ssl`, `check_ckpt_slim`, `keep_only_best_ckpt`, `check_refactor_equivalence`, `check_probe_matches_evaluator`, `check_spifil_growth`, … |

Exemplos:
```
python -m analysis.stats.wilcoxon_kappa
python -m analysis.activations.relu_vs_sigmoid
python -m analysis.plots.plot_continuity_spifil_hybrid
```

Para sobrepor um parâmetro:
```
python -c "from analysis.stats.wilcoxon_acc import wilcoxon_acc; wilcoxon_acc(alpha=0.01)"
```

`heatmap_stages` (14 parâmetros) e `plot_continuity_spifil_hybrid` (16) aceitam um YAML:
passe `config='meu.yaml'`.

> **`analysis/distill/` roda contra outro checkout.** Os três módulos resolvem `src.*`
> de `/dados/home/moliveira/scalable_FLIM_self_supervised` (último commit de junho de
> 2026) e **escrevem os `.json` lá, não aqui**. É deliberado: religá-los mudaria números
> já publicados.

> **`analysis.checks.keep_only_best_ckpt` APAGA checkpoint.** A regra é "sempre o best,
> nunca o last"; onde só existe `last`, o `last` fica. Ele tem `--skip-recent-min` para
> não tocar em run vivo.

---

## 7. Armadilhas que já custaram caro

1. **`percentage` fora do config de dataset** — é eixo da grade. Config de dataset
   sozinho não instancia, e falha alto de propósito.
2. **`skip: wandb` vai à rede.** Use `--set skip=state` para decidir por disco.
3. **`configs/model/lejepa/_variants/custom_cnn.yaml` está em quarentena** — aponta para
   uma classe que nunca existiu. Nasceu quebrado no commit inicial e nunca rodou.
4. **O `name` do growth vira o diretório dos pesos.** Trocar quebra o link com os
   checkpoints gravados.
5. **`num_classes` por dataset nos YAMLs de destilação** (9 / 2 / 7).
6. **Protozoan usa bases FLIM diferentes** para arquitetura e para pesos: arch em
   `ch24_30_48`, pesos em `ch24_32_48`. Não unifique.
7. **Um arquivo que calcula a raiz do repo com `parents[N]` quebra em silêncio** ao
   mudar de profundidade — roda e escreve no lugar errado, sem erro.
8. **`save_hyperparameters()` só funciona se o `__init__` mencionar `super`.** Trocar
   por `pl.LightningModule.__init__(self)` esvazia o `hparams` e o módulo não treina.

---

## 8. Estado do repositório

`src/`, `scripts/` e `tools/` ainda existem como **legado** e ainda não foram apagados.
O inventário de deleção está em `MIGRATION.md`, seção final. O ponto de retorno de toda
a refatoração é o commit `f43435d`.

O grafo do `graphify` está **desatualizado** — foi construído contra o layout antigo e
não conhece `core/`, `flim/`, `methods/`, `eval/`. Rode `graphify update .` antes de
consultá-lo.
