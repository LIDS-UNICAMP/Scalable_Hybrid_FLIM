# Refatoração do Scalable_Hybrid_FLIM — árvore-alvo, launchers Ray unificados e YAML de experimento (multiagente)

**Repositório-alvo:** `Scalable_Hybrid_FLIM` (o repositório de código). Este arquivo é **só a spec** —
ele descreve o que fazer lá, não é código e não roda nada.

**Natureza da tarefa:** reorganização estrutural + deduplicação + eliminação de `argparse`. Não é
redesenho, não é reescrita, não é "aproveitar para melhorar". É mover, fundir e apagar.

**Comportamento numérico não pode mudar.** Isto é uma movimentação pura. Se um número mudar depois do
refactor, **o refactor está errado, não o número**. Qualquer divergência numérica é achado bloqueante e
sobe para o usuário antes de qualquer coisa ser apagada.

---

## 0.1 Regras inegociáveis

1. **`argparse` é proibido por padrão.** Texto do usuário: *"o objetivo tambem é evitar ao maximo usar
   argparse, fica muito ruim! argparses poluem os comandos do cmd, so use se for impossivel de fazer
   sem!"* Regra operacional: toda configuração vira **YAML** / `init_args` do **LightningCLI**. Nenhum
   entrypoint novo nasce com `argparse`. `argparse` só sobrevive onde for **comprovadamente
   impossível** eliminá-lo — e nesse caso o agente responsável **justifica por escrito no relatório**,
   com `arquivo:linha` e a razão técnica concreta ("impossível" não é opinião, é demonstração). Sem
   justificativa escrita, a flag é migrada para YAML.

2. **Primeiro passo obrigatório e bloqueante: commit do estado atual.** Texto do usuário: *"vamos
   refatorar o repositorio, mas antes faça um commit e alerte dizendo que é antes da refatoração"*.
   Antes de mover, criar, renomear ou apagar **qualquer** arquivo: commitar o estado atual do
   repositório com mensagem sinalizando explicitamente que é o snapshot **antes da refatoração**, e
   **alertar o usuário** de que esse commit foi feito e qual é o hash. Nada começa antes disso — nenhum
   agente de escrita é despachado enquanto esse commit não existir. Regra do repositório: **nunca
   adicionar Claude/Anthropic como co-autor** nem mencioná-los na mensagem; autor e committer são o
   usuário.

3. **Entregável obrigatório: `instructions.md`, EM INGLÊS.** Texto do usuário: *"Eu quero que voce crie
   um instructions.md ... explicando em english como navegar por esse repositorio, escrito com o
   objetivo de ajudar o claude"*. É um documento de navegação do repositório, escrito **em inglês**,
   com o objetivo declarado de orientar o Claude: onde cada coisa mora, o que é entrypoint, o que é
   config, como se roda um treino, o que não se toca.

4. **Entregável obrigatório: arquivo de referência de migração (mapa antigo → novo).** Texto do
   usuário: *"de um modo geral voce deve ter um arquivo de referencia dizendo o que mudou para que
   quando eu quiser executar e voce se perder, voce check o passado para me dizer como era o passado"*.
   É um mapa consultável **caminho antigo → caminho novo**, **comando antigo → comando novo**,
   `class_path` antigo → `class_path` novo. Quando um comando antigo parar de funcionar, a resposta é
   encontrada nesse arquivo, não por adivinhação. Ele é atualizado por quem move o código, na hora em
   que move.

5. **Há experimentos rodando agora — não mexa neles.** Texto do usuário: *"vai ter alguns experimentos
   rodando nesse momento, nao mexa neles"*. Concretamente, e sem exceção:
   - **não matar processos** (nada de `kill`, `pkill`, `scancel`, restart de container);
   - **não mexer em GPUs ocupadas**, não realocar, não "liberar" memória;
   - **não apagar checkpoints, logs ou artifacts de runs vivos**;
   - **não alterar arquivos que um processo em execução esteja lendo** (config, `.py` importado por um
     processo vivo, split JSON, arquivo de pesos em uso).

   **Antes de qualquer deleção ou movimentação**, checar o que está vivo: `nvidia-smi`,
   `ps aux | grep python`, e o estado dos runs no **W&B**. O que estiver em uso — ou cuja associação a
   um run vivo for **incerta** — não é candidato a deleção. Incerto nunca vira apagável.

6. **Regra ponytail: o refactor MOVE e APAGA código.** Linhas líquidas devem **cair**. Sem registry,
   sem plugin system, sem camada de config nova, sem classe base "para depois", sem wrapper, sem
   abstração especulativa. Se uma necessidade já é atendida por código existente, o agente diz isso em
   uma linha e segue. Atalho deliberado leva comentário `# ponytail:` nomeando o teto.

7. **As árvores deste documento são NORMATIVAS.** Texto do usuário: *"entao eu tou dando toda
   estreutura para voce, siga esse tipo de extrutura"*. Isso significa:
   - **1 arquivo = 1 classe pública** (ou 1 grupo coeso de funções);
   - **nome do arquivo = snake_case do nome da classe**;
   - **não renomear, não reagrupar, não "melhorar" a árvore**.

   Agente que discordar de um nome ou de um agrupamento **reporta a discordância no relatório e segue a
   árvore como está**. Ele não muda a estrutura por conta própria.

8. **Regra do `__init__.py`.** Trecho preservado verbatim do documento original:

   > O `__init__.py` é o que paga a conta
   > Nome de arquivo longo (`one_layer_1x1_conv_distillation_projection_head.py`, 47 chars) só dói se o
   > import for longo. Com re-export no `__init__.py`, o `class_path` fica mais curto que hoje:

   ```
   # hoje
   class_path: src.modules.lejepa_line_module.LejepaLineModule
   # depois
   class_path: methods.lejepa.LejepaLineModule
   ```

   Consequência operacional: **todo `__init__.py` de pacote re-exporta as classes públicas da pasta**, e
   o `class_path` do YAML **sempre** usa a forma curta. Nome de arquivo comprido é aceitável e desejável
   — quem paga o custo é o `__init__.py`, não o usuário que escreve o YAML.

---

## 0.2 Conflito já identificado pelo usuário — resolução normativa

Texto original, preservado:

> **Um conflito na sua árvore**
> Na sua proposta, `core/cli.py` é o `CustomLightningCLI`. Mas os grupos de argparse acima também querem
> morar em `core/cli.py`. São duas coisas diferentes:
> * `core/cli.py` → LightningCLI, serve só `train.py`, fala YAML.
> * `core/args.py` → grupos de argparse, servem os outros ~60 entrypoints.

**Resolução (a decisão do usuário prevalece: são dois arquivos).**

**(a) Solução adotada — manter os dois arquivos separados, com `core/args.py` marcado como ZONA DE
EXTINÇÃO.** `core/cli.py` fica sendo o `CustomLightningCLI` e só isso. `core/args.py` recebe, no topo do
arquivo, um aviso explícito de que é **código condenado**: nada novo entra ali, e **cada flag migrada
para YAML apaga uma função de lá**. O arquivo encolhe monotonicamente a cada agente que converte um
entrypoint. Quando os ~60 entrypoints virarem 2, `core/args.py` **morre**. A regra 1 (argparse proibido
por padrão) é o motor dessa extinção; `core/args.py` é apenas onde os sobreviventes esperam a vez.

**(b) Por que NÃO unificar num arquivo só.** São **públicos e formatos diferentes**: `core/cli.py`
atende o LightningCLI e fala **YAML** (`class_path` / `init_args`, tipagem via Lightning); `core/args.py`
atende scripts soltos e fala **linha de comando** (`--flag valor`, `add_argument_group`). Juntar os dois
cria um arquivo com duas razões para mudar, dois estilos de parsing e importação cruzada desnecessária —
exatamente a camada de config nova que a regra 6 proíbe. Além disso, unificar **preservaria** o argparse
ao dar a ele um lar respeitável, quando o objetivo declarado é matá-lo. Separado, o argparse fica
visivelmente isolado e visivelmente encolhendo.

**(c) Critério objetivo para declarar `core/args.py` deletável.** `core/args.py` pode ser apagado quando
**nenhum import restante** apontar para ele — verificado por busca em todo o repositório por
`from core.args`, `import core.args`, `core.args.` e qualquer referência por string (configs,
launchers, `.sh`, notebooks, YAMLs, documentação de comandos). Zero ocorrências fora do próprio arquivo
= deletável. Uma ocorrência viva = continua vivo, e o relatório nomeia qual é, com `arquivo:linha`. A
deleção final ainda passa pela regra 5 (checar processos vivos) e por aprovação explícita do usuário.

---

## 0.3 Regra multiagente (obrigatória)

Seguindo o `CLAUDE.md` deste repositório:

- **Decompor por granularidade antes de executar.** Quebrar a refatoração até a menor unidade
  autocontida e dar **um subprompt para cada agente**. Um agente, uma responsabilidade, de preferência
  um arquivo.
- **Tudo que é independente vai num único dispatch paralelo** — uma mensagem só, com todos os agentes
  daquele grupo. Cada seção de agente declara explicitamente a que grupo paralelo pertence.
- **Sequencial só quando existe dependência real**: o subprompt consome o output de outro, ou os dois
  escreveriam o **mesmo arquivo**. Fora esses dois casos, é paralelo.
- **Nenhum agente ocioso esperando irmão** — sem gargalo de espera. Onde a partição puder ser feita por
  arquivo de destino, faça: agentes disjuntos por construção nunca colidem.
- **Toda afirmação de relatório carrega `arquivo:linha`.** Sem âncora é palpite, e palpite vai numa
  **seção separada, marcada como tal** — nunca misturado com o que foi verificado.

---

## Diagnóstico — o que já foi verificado (não re-descobrir)

Tudo abaixo já foi medido no repositório. É contexto de entrada, não tarefa de
investigação: nenhum agente deve gastar passo re-contando linhas, re-grepando
duplicações ou re-descobrindo bugs que já estão listados aqui.

### 1. Os quatro nomes — resposta direta

- **`dataloader.py` não deve existir.** `DataLoader` é do torch, e
  `train/val/test_dataloader()` já são métodos do `LightningDataModule`. Um
  arquivo separado só para isso é rung 3 da escada (ponytail): já existe na
  stdlib/dependência instalada, não se escreve de novo.
- **`data_module.py` é UM arquivo, não quatro.** Dos **58 configs** que declaram
  `data:`, **55** usam a mesma classe (`ParasiteLejepaDataModuleSplited`), **2**
  usam `LejepaDataModule` e **1** usa `ParasiteLejepaDataModule` — e essa última
  é a mesma coisa **sem multicrop**.

**Consequência prática:** a fusão dos 3 datamodules em
`core/data/parasite_data_module.py` + `core/data/folder_data_module.py` cobre os
58 configs. O "sem multicrop" vira um flag / `init_args`, não uma classe própria.

### 2. Os sete arquivos que orquestram Ray

O repo tem **SETE** arquivos que orquestram Ray, **~5.900 linhas**, com as mesmas
funções copiadas:

| Arquivo | Linhas |
|---|---|
| `scripts/run_ssl_ray.py` | 873 |
| `scripts/distillation_ray.py` | 744 |
| `scripts/distillation_conv_ray.py` | 1181 |
| `scripts/autoencoder_flim_ray.py` | 1094 |
| `scripts/classification_flim_ray.py` | 850 |
| `scripts/spifil_growth_loop.py` | 526 |
| `src/evaluate/ray_mlp_queue.py` | 668 |
| **Total** | **~5.900** |

### 3. Duplicações medidas

| Símbolo duplicado | Em quantos arquivos |
|---|---|
| `_log(msg, level)` | 7 |
| `class GpuSlotScheduler` | 4 |
| `_arch_json` / `_flim_weights_path` / `_split_json` / `_run_name` | 4 |
| `_query_wandb_state` / `_should_skip_experiment` | 3 |
| `validate_experiment` / `validate_output_dir` / `build_experiment_grid` | 3 |
| `class ExecutionState` | 2 |

### 4. Dois protocolos de execução incompatíveis

**A) `scripts/run_ssl_ray.py:404`** —
`python src/main.py fit --config <global> --config <dataset> --config <model>`
(LightningCLI). **CORRETO, é o alvo.**

`scripts/run_ssl_ray.py:404-408` já faz exatamente o que se quer:

```python
cmd = [sys.executable, "src/main.py", "fit",
       "--config", DEFAULT_CONFIG_YAML,   # global
       "--config", exp["data_config"],    # dataset
       "--config", exp["model_config"]]   # model
```

**B) os demais** — `python -m src.modules.<módulo>` + ~15 flags argparse.
**Deve ser eliminado.**

### 5. Gerência de GPU (manter exatamente)

`src/evaluate/ray_mlp_queue.py:322` faz `ray.init(num_gpus=0)` e cada task recebe
`gpu_id` e fixa `CUDA_VISIBLE_DEVICES`.

**NÃO usar `@ray.remote(num_gpus=1)`** — isso tira do usuário a escolha de QUAL
GPU e impede mais de um job por GPU (`max_per_gpu`).

### 6. Vocabulário divergente a canonizar

Os três launchers já têm vocabulários diferentes pra mesma coisa:

| Conceito | `run_ssl_ray` | `spifil_growth_loop` | `distillation_conv_ray` |
|---|---|---|---|
| quais GPUs | `--gpu-ids` | `--gpus 1 2 3` | `--gpu-slots` |
| porcentagens | `--pcts` | `--percentages` | `--percentages` |
| pular feito | `--resume` / `--ignore-existing` | — | `--skip-existing` / `--check-wandb` / `--retry` |

> "É por isso que você redigita a linha inteira: não há um nome canônico pra nada."

Canonização:

- quais GPUs : `--gpu-ids` | `--gpus` | `--gpu-slots` → canônico: **`gpu_ids`**
- percentuais: `--pcts` | `--percentages` → canônico: **`percentages`**
- pular feito: `--resume` | `--ignore-existing` | `--skip-existing` | `--check-wandb` | `--retry` → canônico: **`skip`**

**Decisão do usuário — a partir de hoje isso acaba.** Não se trata de escolher um nome de flag
melhor: o objetivo é **não ter flag nenhuma**. A partir de agora **toda a concentração vai para a
nova proposta de YAML** — o *experiment YAML* é o único lugar onde `gpu_ids`, `percentages` e `skip`
existem, como **chaves do YAML**, nunca como `--flag` na linha de comando. Consequências práticas,
válidas para todo agente deste documento:

- **Nenhum nome canônico novo vira flag de CLI.** `gpu_ids`, `percentages` e `skip` são chaves de
  `resources:`, `axes:`/`grid:` e do campo `skip:`. A única exceção é o punhado de overrides do
  `launch.py` (`--dry-run`, `--fail-fast`, `--gpu-ids`, `--set`), que existem só para sobrepor o YAML
  pontualmente, com precedência CLI > YAML > default do schema.
- **Nenhum entrypoint novo nasce com `argparse`.** Se um agente sentir vontade de acrescentar uma
  flag, a resposta é uma chave no experiment YAML, ou nada.
- **O vocabulário divergente não é traduzido, é apagado.** Não construir camada de compatibilidade
  que aceite `--pcts`/`--gpu-slots`/`--retry`: os seis launchers antigos vão para a lista de deleção,
  e quem precisar do nome antigo consulta o arquivo de migração (`MIGRATION.md`), que registra
  comando antigo → experiment YAML equivalente.
- **Critério de pronto:** `grep -rn "add_argument" experiments/ methods/ core/ flim/` só retorna as
  poucas flags do `launch.py` justificadas por escrito. Qualquer outra ocorrência é regressão.

### 7. Bugs reais já encontrados

- **`configs/model/lejepa_custom_cnn.yaml`** aponta pra `src.modules.LeJEPACNNModule`,
  que **não existe** — nem em `src/modules/__init__.py`, nem em arquivo nenhum
  (só numa docstring). Esse config quebra na instanciação. É exatamente o que o
  `test_class_paths.py` pega. **Consertar.**
- **`configs/model/lejepa_line_he_protozoan_train1.yaml`** tem
  `arch_json: /dados/home/moliveira/scalable_FLIM_self_supervised/...`
  (absoluto, só roda numa máquina).
- **`model/lejepa/{he,xavier,random,trunc_normal}/{helminth-eggs,helminth-larvae}/`**
  não existem hoje (são o caso especial de protozoan em `run_ssl_ray.py:113`) —
  **24 configs faltando**.

### 8. O que a pasta de configs paga

Duas f-strings substituem todo o código de resolução de config:

```
f"configs/dataset/{dataset}/split_{split}.yaml"
f"configs/model/{method}/{init}/{dataset}/split_{split}.yaml"
```

Isso apaga:

- `_FLIM_YAMLS` — dict de 9 entradas escrito à mão em `run_experiments.py:57-67`
- `_model_config()` e `_data_config()` em `run_ssl_ray.py:113`
- `_arch_json` / `_split_json` / `_flim_weights_path` replicados em 4 ray scripts
- `DATASET_LONG_TO_SHORT` no caminho de resolução — a pasta usa um vocabulário
  (`helminth-eggs`), some o alias

### 9. Inventário do que some (O que apagar quando terminar)

- `scripts/{run_ssl_ray,distillation_ray,distillation_conv_ray,autoencoder_flim_ray,classification_flim_ray,spifil_growth_loop}.py`
- `src/evaluate/ray_mlp_queue.py` (vira `experiments/ray/runners/eval.py`)
- Os blocos `if __name__ == "__main__"` com argparse dentro de `src/modules/*.py`
- Os helpers argparse dentro de `src/models/distillation.py:279-497`
  (`add_student_flags`, `resolve_student`, `add_distill_flags`,
  `resolve_distill_flags`, `derive_flim_paths`, `distill_run_tags`) — as flags
  viram `init_args` do módulo
- O `from constants import ...` com hack de `sys.path` (existe só porque os
  launchers rodam de dentro de `scripts/`)

---

## Árvore-alvo — NORMATIVA

Esta é a estrutura que o usuário desenhou. Ela é copiada aqui **verbatim**. Nenhum agente
renomeia arquivo, funde pasta, cria camada ou "melhora" a árvore. Discordância vira item de
relatório, nunca alteração silenciosa.

### Árvore geral do repositório

```
repo/
├── train.py                    # = src/main.py (LightningCLI já monta o método pelo class_path)
├── run_experiments.py          # 2º entrypoint: grid runner que chama train.py
│
├── core/                       # compartilhado por TODOS os métodos
│   ├── constants.py            # ÚNICO — funde config.py + src/utils/constant.py
│   │                           #   + scripts/constants.py + src/evaluate/constants.py
│   ├── data/
│   │   ├── data_module.py      # src/data_modules/parasite*.py, lejepa.py
│   │   ├── datasets.py         # src/data_modules/datasets/*
│   │   └── splits.py           # data/create_splits.py
│   ├── transforms.py           # src/transforms/multicrop.py
│   ├── blocks.py               # src/models/models.py  (proj heads, encoders genéricos)
│   ├── mixins.py               # FrozenTeacherCheckpointMixin, KnnKappaProbeMixin
│   ├── metrics.py              # src/metrics/{classification,linear_probe}.py
│   ├── cli.py                  # CustomLightningCLI (swap wandb→csv)
│   └── wandb.py                # wandb_cache.py + get_names_wandb.py
│
├── flim/                       # não é método: encoder que os métodos importam
│   ├── encoder.py              # src/models/{encoders,custom_cnn}.py + arch_json/weights loader
│   ├── pyift_strategy.py       # + Dockerfile.pyift (roda isolado)
│   └── spifil.py               # scripts/spifil_{grow,growth_loop,resnet_graft}.py
│
├── methods/                    # 1 pasta = models.py + module.py + loss.py
│   ├── lejepa/                 # lejepa.py, lejepa_flim.py, lejepa_{,flim,line}_module.py,
│   │                           #   losses/{lejepa_loss,epps_pulley}.py
│   ├── autoencoder/            # SSL — autoencoder_resnet.py + autoencoder_flim_module.py
│   ├── distillation/           # 4 módulos + models/distillation.py
│   │                           #   ⚠ importa methods/lejepa (professor = LeJEPAFLIMModel)
│   ├── classification/         # classification_flim_module.py + classifier_module.py (fine-tune)
│   ├── dino_v2/                # a entrar
│   └── byol/                   # a entrar
│
├── eval/                       # 9,1k linhas — probe, não treino (tem __main__.py, mantém)
│   ├── svm.py  mlp.py  knn.py  classical_classifiers.py
│   ├── ray_queue.py            # ray_mlp{,_queue}.py
│   ├── unified_eval.py  wandb_resolver.py  eval_plotter.py
│   └── growth_stages.py  tsne.py
│
├── experiments/                # scripts/ (13,7k) — orquestração, não biblioteca
│   ├── ray/                    # *_ray.py launchers
│   ├── ckpt/                   # strip_teacher*, validate_stripped, verify_finetune
│   ├── gen_configs.py          # generate_mlp_configs.py, download_*, normalize_reports
│   └── oneoff/                 # retry_*, run_missing_*, check_*_status  (descartáveis)
│
├── analysis/                   # tools/ + statistics/ + analysis_flim_distill/ + check_experiments/
│   ├── stats/                  # wilcoxon_{acc,f1,kappa,equivalence,flim_init}, compute_cost
│   ├── activations/            # relu_vs_sigmoid, saturation, unit_activations, heatmap_stages
│   ├── distill/                # embedding_analysis, ruler_mismatch, distill_destroys_flim
│   └── plots/                  # todos os plot_*.py + src/utils/plot_svm_results.py
│
├── configs/                    # árvore atual (261 YAMLs), NÃO 1 por método
│   ├── method/{lejepa,dino_v2,byol,autoencoder,distillation}.yaml
│   ├── data/percentage/<parasito>_split_N/*.yaml
│   ├── evaluate/mlp/{freeze,unfreeze}/<parasito>/*.yaml
│   └── default.yaml
│
├── data/                       # splits JSON + to_mateus/ (pesos + arch_json FLIM)
├── reports/                    # A_reports/ + os .md soltos da raiz + metrics_distillation/
└── env/                        # Dockerfile, environment.yml, requirements.txt, setup_env.sh
```

### `core/` — compartilhado por TODOS os métodos

```
core/
├── constants.py                          (sem classe — só dados)
├── custom_lightning_cli.py             → class CustomLightningCLI
├── metrics.py                          → def compute_metrics
├── wandb.py                            → def resolve_run, def cached_history
│
├── data/
│   ├── __init__.py                       re-exporta as 7 classes abaixo
│   ├── parasite_data_module.py         → class ParasiteDataModule      ← funde 3 datamodules (55 configs)
│   ├── folder_data_module.py           → class FolderDataModule        ← src/data_modules/lejepa.py (2 configs)
│   ├── parasite_dataset.py             → class ParasiteDataset         ← DatasetParasite
│   ├── multi_view_dataset.py           → class MultiViewDataset        ← ParasiteLejepaMultiViewDataset
│   ├── multi_crop_dataset.py           → class MultiCropDataset
│   ├── folder_dataset.py               → class FolderDataset           ← LejepaDataset
│   ├── multi_crop_transform.py         → class MultiCropTransform      ← src/transforms/multicrop.py
│   ├── loaders.py                      → def pil_loader, def ift_lab_loader
│   ├── transforms.py                   → def build_aug, def build_test  (tira o `_` do nome)
│   └── splits.py                       → def create_splits             ← data/create_splits.py
│
├── blocks/
│   ├── __init__.py
│   ├── mlp_head.py                     → class MLPHead
│   ├── two_layer_sigmoid_head.py       → class TwoLayerSigmoidHead
│   ├── two_layer_softplus_head.py      → class TwoLayerSoftplusHead
│   ├── classification_model.py         → class ClassificationModel
│   ├── sigmoid_classification_model.py → class SigmoidClassificationModel
│   ├── timm_encoder.py                 → def build_encoder, def get_embed_dim
│   └── init.py                         → def freeze_encoder, unfreeze_encoder,
│                                          init_weights_{he,xavier,trunc_normal}
└── mixins/
    ├── __init__.py
    ├── frozen_teacher_checkpoint_mixin.py → class FrozenTeacherCheckpointMixin
    └── knn_kappa_probe_mixin.py           → class KnnKappaProbeMixin
                                             + def knn_kappa_probe, _extract_probe_features
```

### `flim/` — não é método: encoder que os métodos importam

```
flim/
├── __init__.py                           expõe só `build`
├── build.py                            → def build(arch_json, init, weights_path, in_channels)
│                                          ÚNICA porta que methods/* usa
├── encoder.py                          → class Encoder
├── flim_residual_encoder.py            → class FLIMResidualEncoder
│                                          + _StashConv, _ResidualConv3 (privadas, ficam)
│                                          + def build_flim_residual_encoder
├── arch.py                             → def parse_architecture, get_channels_from_arch,
│                                          get_actual_channels_from_weights, override_arch_channels,
│                                          build_encoder_from_arch, build_decoder_from_arch
├── weights.py                          → def get_bias, get_weights, shift_weights,
│                                          load_FLIM_encoder, load_FLIM_encoder_from_arch_dict
├── pyift_strategy.py                   → class PyIFTStrategy  + Dockerfile.pyift
└── spifil.py                           → def grow, def graft   ← scripts/spifil_*.py
```

### `methods/lejepa/`

```
methods/lejepa/
├── __init__.py                           from .lejepa_line_module import LejepaLineModule ...
├── projection_head.py                  → class ProjectionHead
├── projection_head_hawk.py             → class ProjectionHeadHawk
├── lejepa_model.py                     → class LeJEPAModel
├── lejepa_flim_model.py                → class LeJEPAFLIMModel
├── lejepa_cnn_model.py                 → class LeJEPACNNModel
├── simple_cnn_model.py                 → class SimpleCNNModel
├── ijepa_encoder.py                    → class IJEPAEncoder
│                                          + _SelfAttention, _Block, _MLP1, _MLP2,
│                                            _PatchEmbeddings, _Embeddings, _Encoder,
│                                            _AttentionWrapper, _AttentionOutput, _IJepaViT
├── simple_sigreg.py                    → class SimpleSIGReg
├── real_sigreg.py                      → class RealSIGReg        (Epps-Pulley)
├── invariance_loss.py                  → def invariance_loss
├── lejepa_module.py                    → class LeJEPAModule
├── lejepa_flim_module.py               → class LeJEPAFLIMModule(LeJEPAModule)
└── lejepa_line_module.py               → class LejepaLineModule(LeJEPAFLIMModule)   ← 30 configs
```

### `methods/autoencoder/`

```
methods/autoencoder/
├── __init__.py
├── residual_up_block.py                → class ResidualUpBlock
├── resnet_decoder.py                   → class ResNetDecoder
├── autoencoder_flim.py                 → class AutoEncoderFLIM
├── autoencoder.py                      → class AutoEncoder
├── autoencoder_classifier.py           → class AutoEncoderClassifier
└── autoencoder_flim_module.py          → class AutoEncoderFlimModule   ← 1314 linhas hoje
                                           a `class Head` interna some (= core.blocks.MLPHead)
```

### `methods/distillation/`

```
methods/distillation/
├── __init__.py
├── frozen_teacher.py                                  → class FrozenTeacher
├── distillation_projection_head.py                    → class DistillationProjectionHead
├── conv_distillation_projection_head.py               → class ConvDistillationProjectionHead
├── one_layer_conv_distillation_projection_head.py     → class OneLayerConvDistillationProjectionHead
├── one_layer_1x1_conv_distillation_projection_head.py → class OneLayer1x1ConvDistillationProjectionHead
├── two_layer_1x1_conv_bn2d_distillation_projection_head.py
│                                                      → class TwoLayer1x1ConvBN2dDistillationProjectionHead
├── student_classification_head.py                     → class StudentClassificationHead
├── kl_distillation_loss.py                            → class KLDistillationLoss
├── mse_distillation_loss.py                           → class MSEDistillationLoss
├── cosine_distillation_loss.py                        → class CosineDistillationLoss
├── kd_loss.py                                         → def kd_loss
├── teacher_input.py                                   → def prepare_teacher_input
├── distillation_module.py                             → class DistillationModule
├── distillation_conv_module.py                        → class DistillationConvModule
├── distillation_one_layer_module.py                   → class DistillationOneLayerModule
├── distillation_two_layer_module.py                   → class DistillationTwoLayerModule
└── cli.py                              → def add_student_flags, resolve_student,
                                           add_distill_flags, resolve_distill_flags,
                                           derive_flim_paths, distill_run_tags
                                           (argparse — hoje mora dentro de models/distillation.py)
```

### `methods/classification/`, `methods/dino_v2/`, `methods/byol/`

```
methods/classification/
├── __init__.py
├── classification_flim_module.py       → class ClassificationFlimModule
└── classification_finetune_module.py   → class ClassificationFinetuneModule

methods/dino_v2/                          a escrever
├── __init__.py
├── dino_v2_model.py                    → class DINOv2Model
├── dino_head.py                        → class DINOHead
├── dino_loss.py                        → class DINOLoss
└── dino_v2_module.py                   → class DINOv2Module

methods/byol/                             a escrever
├── __init__.py
├── byol_model.py                       → class BYOLModel        (online + target EMA)
├── byol_predictor.py                   → class BYOLPredictor
└── byol_module.py                      → class BYOLModule
```

### `experiments/ray/` — padronização do Ray (multi-GPU)

```
experiments/ray/
├── __init__.py
├── gpu_slot_scheduler.py    → class GpuSlotScheduler      (4 cópias )
├── execution_state.py       → class ExecutionState        (2 cópias )
├── paths.py                 → def arch_json, flim_weights_path, split_json, run_name
├── grid.py                  → def build_grid, validate_experiment, validate_output_dir
├── skip.py                  → def query_wandb_state, should_skip
├── train_worker.py          → @ray.remote def run_train(exp)
├── eval_worker.py           → @ray.remote def run_eval(exp)     ← ray_mlp_queue não é treino
└── launch.py                → CLI única
```

padronizacao do ray proposta onde ele deve ser capaz de rodar em multiplas gpus.

```python
# experiments/ray/train_worker.py
@ray.remote                                # ← sem num_gpus
def run_train(exp: dict, gpu_id: int, cpus: int = 4) -> dict:
    import os, subprocess, sys
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": str(gpu_id),
           "OMP_NUM_THREADS":      str(cpus)}
    cmd = [sys.executable, "train.py", "fit",
           "--config", "configs/default.yaml",
           "--config", exp["dataset_config"],
           "--config", exp["model_config"],
           f"--data.init_args.percentage={exp['pct']}",
           f"--trainer.logger.init_args.name={exp['run_name']}",
           "--trainer.devices=1"]          # 1 = a única visível, que é a gpu_id
    return _run(cmd, env=env, exp=exp, gpu_id=gpu_id)
```

### `configs/` — organização dos YAMLs

```
configs/
├── default.yaml                                    # GLOBAL
│
├── dataset/                                        # 57 → 9
│   ├── helminth-eggs/
│   │   ├── split_1.yaml
│   │   ├── split_2.yaml
│   │   └── split_3.yaml
│   ├── helminth-larvae/
│   │   └── split_{1,2,3}.yaml
│   └── protozoan-cysts/
│       └── split_{1,2,3}.yaml
│
├── model/                                          # método → init → dataset → split
│   ├── lejepa/
│   │   ├── flim/
│   │   │   ├── helminth-eggs/split_{1,2,3}.yaml    ← eram lejepa_line_flim_eggs_train{1,2,3}
│   │   │   ├── helminth-larvae/split_{1,2,3}.yaml
│   │   │   └── protozoan-cysts/split_{1,2,3}.yaml
│   │   ├── he/
│   │   │   ├── helminth-eggs/split_{1,2,3}.yaml    ← HOJE NÃO EXISTEM
│   │   │   ├── helminth-larvae/split_{1,2,3}.yaml  ← HOJE NÃO EXISTEM
│   │   │   └── protozoan-cysts/split_{1,2,3}.yaml
│   │   ├── xavier/   random/   trunc_normal/       ← mesma forma, mesmos buracos
│   │   └── _variants/                              # fora da grade, o glob não pega
│   │       ├── resnet50.yaml  real_sigreg.yaml  simple_sigreg.yaml
│   │       └── custom_cnn.yaml                     ← o quebrado (LeJEPACNNModule não existe)
│   │
│   ├── autoencoder/   distillation/   classification/
│   └── byol/          dino_v2/                     # entram com a mesma forma
│
└── generated/                                      # SAÍDA — .gitignore
    └── mlp/{freeze,unfreeze}/<dataset>/<run_id>.yaml
```

### `experiments/` — 1 pasta por método + o motor Ray

```
Estrutura proposta

experiments/
├── lejepa/                          # espelha methods/lejepa
│   ├── init_ablation.yaml
│   ├── flim_full.yaml
│   └── growth_grid4.yaml
├── distillation/
│   └── conv_1x1_grid.yaml
├── autoencoder/    classification/    byol/    dino_v2/
└── ray/                             # o motor, agnóstico de método
    ├── launch.py                    # ÚNICA entrada: recebe o .yaml
    ├── schema.py  grid.py  paths.py  skip.py
    ├── gpu_slot_scheduler.py  execution_state.py
    └── runners/{train,growth,eval}.py

python -m experiments.ray.launch experiments/lejepa/growth_grid4.yaml

```

---

## Mapa de dispatch global — quem roda com quem

Cada onda é **uma única mensagem com todos os seus agentes em paralelo**. Só existe onda nova
quando há dependência real: o agente consome o output do anterior, ou escreve o mesmo arquivo.
Nenhum agente fica ocioso esperando irmão.

| Onda | Grupos / agentes | Depende de | Dispatch |
|---|---|---|---|
| **W0** | `G0` — commit de segurança "antes da refatoração" + alerta ao usuário | nada | sequencial, **bloqueante** |
| **W1** | `C1-INV` (inventário de constantes, read-only), `C2..C27` (core), `F1..F8` (flim), `Y1` (inventário de configs, read-only), `R1` (schema do experiment YAML), `E-ANA`, `E-MISC` | W0 | **um dispatch paralelo** — arquivos de destino disjuntos por construção |
| **W1b** | `C1-GLOBAL`, `C1-FLIM`, `C1-EVAL`, `C1-EXP`, `C1-ANA`, `C1-M<método>` (um por arquivo de constantes), depois `C1-CALL` (um por consumidor) e `C1-GATE` | `C1-INV` | **um dispatch paralelo** por camada; `C1-GATE` sequencial no fim |
| **W2** | `M-LEJEPA` (ML-00..ML-10 + ML-CHAIN), `M-AE`, `M-CLS`, `R2..R9`, `Y2..Yn` (um agente por dataset), `Y-VAR`, `Y-GEN`, `P1..P7`, `P9` | W1 | **um dispatch paralelo** |
| **W3** | `M-DISTILL` (professor = `LeJEPAFLIMModel`), `R10` (`launch.py`), `P8` (agregador do pré-voo), `E1..E4` (os quatro experiment YAMLs), `E-EVAL`, `E-EXP` | W2 | **um dispatch paralelo** |
| **W4** | `Y-CLASSPATH` (reescrita dos `class_path` em todos os YAMLs), `B1` (bug `LeJEPACNNModule`), `B2` (caminho absoluto), `V1` (verificação de contrato `flim`/`methods`) | W3 | paralelo, exceto `Y-CLASSPATH` que escreve todos os YAMLs sozinho |
| **W5** | **Portão de validação**: pré-voo verde em todos os experiment YAMLs, `--dry-run` reproduzindo os comandos antigos, 1 experimento pequeno ponta a ponta batendo o número da árvore velha | W4 | sequencial |
| **W6** | `D1` (`instructions.md`, inglês), `D2` (`MIGRATION.md`), `D3` (`INDEX.md`), `M-NEW` (`dino_v2`, `byol`) | W5 | **um dispatch paralelo** |
| **W7** | Agentes de deleção, um por alvo | W5 **+ aprovação explícita do usuário** | paralelo, só depois do portão |

Regra de partição: **um agente por arquivo de destino**. Dois agentes nunca editam o mesmo
arquivo. Quando dois subprompts caem no mesmo arquivo, ou eles viram um agente só, ou o arquivo
foi mal decomposto — nesse caso o inventário está errado e é ele que se conserta, não a divisão.

Sequência obrigatória e o porquê de cada aresta:
- `M-DISTILL` depois de `M-LEJEPA` — o professor é `LeJEPAFLIMModel`; é a única aresta entre métodos.
- `Y-CLASSPATH` depois de `core/` e `methods/` — o destino do `class_path` precisa existir.
- `R10` depois de `R2..R9` — o `launch.py` importa todos os outros.
- `P8` depois de `P1..P7` — o agregador chama todas as checagens.
- Deleção depois do portão — e nunca sem aprovação.

---

## Grupo C — construir `core/`

**Grupo paralelo. Todos os agentes disparados numa única mensagem; nenhum depende de irmão.**
A partição é **por arquivo de destino**, o que torna os agentes disjuntos por construção: dois
agentes nunca escrevem no mesmo arquivo. Cada agente cria o arquivo novo e move o código para
dentro dele; **nenhum agente apaga a origem** nesta fase — a remoção das definições antigas é um
passo posterior, com gate. Toda afirmação de relatório vem com `file:line`; sem âncora é chute, e
chute vai numa seção separada marcada como tal.

### Constantes — DUAS camadas: uma global, várias locais

**Decisão do usuário, normativa:** vai existir **constants global** e **constants local**. A
anotação `# ÚNICO` da árvore passa a significar **único global**, não "único no repositório
inteiro". Isto **refina** a árvore, não a altera: `core/constants.py` continua exatamente onde
está e continua sendo o único arquivo de constantes compartilhadas; os `constants.py` locais são
adições autorizadas explicitamente pelo usuário, um por pacote, sem pasta nova e sem camada nova.

```
core/constants.py            ← GLOBAL   (o único compartilhado)
flim/constants.py            ← LOCAL    (só flim usa)
methods/<método>/constants.py← LOCAL    (só aquele método usa; só se existir de fato)
eval/constants.py            ← LOCAL    (ex-src/evaluate/constants.py, o que não subir)
experiments/constants.py     ← LOCAL    (ex-scripts/constants.py, o que não subir)
analysis/constants.py        ← LOCAL    (INIT_ORDER, METHOD_COLOR, METHOD_LABEL, METHOD_HATCH,
                                         DATASET_LABEL, METRIC_LABELS, METRICS — dicionário de plot)
```

**A fusão das 4 fontes vira TRIAGEM, não empilhamento.** `config.py`, `src/utils/constant.py`,
`scripts/constants.py` e `src/evaluate/constants.py` não são despejados num arquivo só: cada nome é
classificado antes de mudar de lugar. Empilhar tudo no global recria o god-module que o refactor
está desmontando — só que com endereço melhor.

#### A regra de decisão (mecânica, contável, sem gosto pessoal)

Conte os **pacotes de topo** que consomem o nome (`core`, `flim`, `methods`, `eval`, `experiments`,
`analysis`):

| Consumidores | Onde o nome mora | Por quê |
|---|---|---|
| **2 ou mais pacotes** | `core/constants.py` (**global**) | é contrato entre pacotes |
| **1 pacote, 2+ arquivos** | `<pacote>/constants.py` (**local**) | ninguém de fora precisa saber |
| **1 arquivo só** | **não é constante de módulo** — fica no próprio arquivo, no topo | promover a arquivo é rung 1 da escada |

Empate ou dúvida honesta: **fica local**. É trivial promover local → global depois (um `git mv` de
linha); é caro devolver um nome que já virou contrato público.

#### As quatro regras de import (o que impede isso de virar um novelo)

1. **Local pode importar global.** `from core.constants import IMAGE_SIZE` é sempre válido.
2. **Global NUNCA importa local.** `core/constants.py` não tem um único `import` de pacote.
   Verificável: `grep -n "^from\|^import" core/constants.py` só pode casar com stdlib, e de
   preferência nada.
3. **Local nunca importa local de outro pacote.** `analysis/constants.py` não importa
   `eval/constants.py`. Se dois locais precisam do mesmo nome, esse nome é **global** por definição
   da tabela acima — sobe, não atravessa.
4. **Proibido shadowing.** Um nome que existe no global **não pode ser redefinido** num local. Se o
   pacote precisa de um valor diferente, o **nome muda** e carrega o prefixo do pacote
   (`ANALYSIS_IMAGE_SIZE`), para que ninguém leia `IMAGE_SIZE` num arquivo e receba outro valor do
   que leu no arquivo ao lado. Dois valores com o mesmo nome é a origem exata do bug que se está
   apagando.

Regra transversal, válida para **todo** arquivo de constantes, global ou local: **plano, sem classe,
sem lógica, sem valor computado, sem `os.environ`, sem leitura de disco** — só nomes e dados. E o
padrão `from constants import ...` apoiado em hack de `sys.path` morre: os únicos imports válidos
passam a ser `from core.constants import ...` e `from <pacote>.constants import ...`.

#### C1-INV — inventário e classificação (read-only, roda PRIMEIRO, sozinho)
Lê as quatro fontes (`config.py`, `src/utils/constant.py`, `scripts/constants.py`,
`src/evaluate/constants.py`) mais toda constante solta em `IMAGE_SIZE`-style espalhada por scripts, e
produz **uma linha por nome**: valor(es), `file:line` de cada definição, consumidores com
`file:line`, pacote de topo de cada consumidor, contagem de pacotes, e o **destino calculado** pela
tabela (global / local `<pacote>` / fica no arquivo).
**O que NÃO faz:** não move nada, não edita nada, não escolhe vencedor de colisão.
**Problema esperado:** nomes iguais com valores diferentes entre as fontes.
**Solução:** **tabela de colisões** — uma linha por nome, os 4 valores (ou "ausente") e os
consumidores de cada definição. Colisão **idêntica** é fundida direto, sem perguntar. Colisão
**divergente** vira **pergunta ao usuário**: o agente para naquele nome, mantém o comportamento
atual e registra a pergunta. Unificar valores divergentes muda números produzidos — essa decisão
não é do agente.
**Entrega:** a tabela de classificação, a tabela de colisões, o mapa definição → consumidores e a
lista de nomes pendentes de decisão. Tudo o que vem depois consome este relatório.

#### C1-GLOBAL, C1-FLIM, C1-EVAL, C1-EXP, C1-ANA, C1-M\<método\> — um agente por arquivo de destino
**Dispatch paralelo único**, depois de `C1-INV`. Arquivos disjuntos por construção: cada agente
escreve **um** `constants.py` e mais nada.
- **C1-GLOBAL** → `core/constants.py`: só os nomes classificados como global (2+ pacotes).
- **C1-FLIM** → `flim/constants.py`; **C1-EVAL** → `eval/constants.py`; **C1-EXP** →
  `experiments/constants.py`; **C1-ANA** → `analysis/constants.py` (os dicionários de plot);
  **C1-M\<método\>** → `methods/<método>/constants.py`, **um agente por método e só se o inventário
  provar que o método tem constante própria**. Pacote sem nome local **não ganha arquivo vazio**.
**O que NÃO fazem:** não reescrevem consumidor (isso é do C1-CALL), não criam nome novo, não
renomeiam nada exceto o prefixo exigido pela regra 4, e não movem nome que o `C1-INV` deixou
pendente de decisão do usuário.

#### C1-CALL — reescrita dos consumidores (paralelo, um agente por arquivo consumidor)
Cada agente troca os imports de **um** arquivo consumidor para o endereço novo (global ou local do
próprio pacote). Partição por arquivo consumidor: dois agentes nunca tocam o mesmo arquivo.
**O que NÃO faz:** não muda valor, não muda nome, não aproveita para "limpar" o arquivo.

#### C1-GATE — verificação (sequencial, fecha o bloco)
Roda e reporta:
- `grep -rn "^from\|^import" core/constants.py` → só stdlib ou vazio (regra 2).
- `grep -rn "from \(flim\|eval\|experiments\|analysis\|methods\)\..*constants import" ` fora do
  próprio pacote → **vazio** (regras 1 e 3).
- Interseção de nomes entre o global e cada local → **vazia** (regra 4, sem shadowing).
- `grep -rn "sys.path" ` no código movido → **vazio**.
- Nenhum `constants.py` contém `def`, `class`, `os.environ` ou chamada de função.
- Nome definido e não consumido por ninguém → lista de **candidatos a deleção**, que não são
  apagados aqui: vão para o portão de deleção com o resto.

### C2 — `core/custom_lightning_cli.py`
Move `CustomLightningCLI` para o arquivo novo, com o swap de logger wandb → csv. Serve **só o
`train.py`** e fala YAML.
**O que NÃO faz:** não toca em argparse — grupos de argparse são outro arquivo, outro dono; e não
adiciona flag, subcomando ou opção que não exista hoje.
**Entrega:** o arquivo, a origem `file:line` da classe e a confirmação de que o logger csv é o
default sem nenhuma flag nova na linha de comando.

### C3 — `core/metrics.py`
Funde `src/metrics/classification.py` e `src/metrics/linear_probe.py` numa **única** `compute_metrics`.
Uma função, sem classe, sem objeto de configuração, sem explosão de keywords "para depois".
**O que NÃO faz:** não inventa métrica nova e não muda convenção silenciosamente.
**Problema esperado:** as duas fontes podem divergir em esquema de averaging (`macro`/`weighted`) e
em indexação de rótulo (0 ou 1). **Solução:** divergência é achado, não detalhe a ser aparado — o
agente escolhe a convenção canônica, **declara qual foi e o que ela quebra**, e deixa o caso
duvidoso como pergunta.
**Entrega:** o arquivo, a tabela das duas convenções antigas e a assinatura única final.

### C4 — `core/wandb.py`
Funde `wandb_cache.py` e `get_names_wandb.py` em duas funções: `resolve_run` e `cached_history`.
Cache com o que já existe na stdlib/dependência instalada; nada de classe de cache própria.
**O que NÃO faz:** não muda o formato do cache em disco já gravado e não adiciona camada de retry
que não existe hoje.
**Entrega:** o arquivo, as duas assinaturas e a lista de chamadores atuais dos dois módulos antigos
(para o passo de reescrita posterior).

---

### Subgrupo `core/data/` — um agente por arquivo

Mesma regra: arquivos de destino disjuntos, todos no mesmo dispatch do Grupo C.

**C5 — `core/data/parasite_data_module.py` → `class ParasiteDataModule`.** Funde os **3 datamodules
de parasita** num só (os 55 configs que usam `ParasiteLejepaDataModuleSplited`, o que usa
`ParasiteLejepaDataModule` — a mesma coisa sem multicrop — e o irmão restante). A diferença entre
eles vira **parâmetro do `__init__`**, não subclasse. Não cria `dataloader.py`: `train/val/test_dataloader()`
já são métodos do `LightningDataModule`. **NÃO** reescreve YAML — isso é do agente de `class_path`.
**Entrega:** a classe, a tabela "campo do config antigo → argumento novo" e a lista dos 3 nomes
antigos que morrem.

**C6 — `core/data/folder_data_module.py` → `class FolderDataModule`.** Ex-`src/data_modules/lejepa.py`
(2 configs). Datamodule de pasta genérica, sem nada específico de parasita. **NÃO** funde com C5 —
são dois casos de uso distintos e a fusão viraria um `if` no `__init__`.
**Entrega:** a classe e os 2 configs que a apontam.

**C7 — `core/data/parasite_dataset.py` → `class ParasiteDataset`** (← `DatasetParasite`). Move o
dataset, renomeando a classe. **NÃO** muda a lógica de leitura, de rótulo nem de split.
**Entrega:** a classe e o diff de nomes (antigo → novo).

**C8 — `core/data/multi_view_dataset.py` → `class MultiViewDataset`** (← `ParasiteLejepaMultiViewDataset`).
Move e renomeia. **NÃO** absorve `MultiCropDataset` — são datasets diferentes e continuam separados.
**Entrega:** a classe, o número de views que ela produz e quem a consome hoje.

**C9 — `core/data/multi_crop_dataset.py` → `class MultiCropDataset`.** Move o dataset de multi-crop
para o arquivo próprio. **NÃO** implementa a transform: a transform é C11.
**Entrega:** a classe e a dependência explícita em `MultiCropTransform`.

**C10 — `core/data/folder_dataset.py` → `class FolderDataset`** (← `LejepaDataset`). Move e renomeia
o dataset de pasta. **NÃO** reimplementa `ImageFolder` se o comportamento já for o do torchvision —
se for, o relatório diz isso e propõe a deleção.
**Entrega:** a classe e o veredito "é ou não é `ImageFolder` com outro nome".

**C11 — `core/data/multi_crop_transform.py` → `class MultiCropTransform`** (← `src/transforms/multicrop.py`).
Move a transform de multicrop. **NÃO** mexe em `build_aug`/`build_test` (arquivo de C13).
**Entrega:** a classe e os parâmetros de crop que hoje vêm de constante (para cruzar com `C1-INV`).

**C12 — `core/data/loaders.py` → `def pil_loader`, `def ift_lab_loader`.** Duas funções de leitura de
imagem, uma PIL e uma IFT/lab. **NÃO** cria classe, registry nem despachante — quem escolhe o loader
é o dataset, por argumento.
**Entrega:** as duas funções e a lista de datasets que passam a importar daqui.

**C13 — `core/data/transforms.py` → `def build_aug`, `def build_test`.** Move as pipelines de
augmentation e de teste, **tirando o `_` do início do nome** (eram privadas, viram públicas porque
YAML e datasets as chamam). **NÃO** muda a composição de transforms nem os valores de normalização.
**Entrega:** as duas funções, os nomes antigos com `_` e a confirmação de que a composição é idêntica.

**C14 — `core/data/splits.py` → `def create_splits`** (← `data/create_splits.py`). Move a geração de
splits para dentro do pacote. **NÃO** carrega argparse junto: se o arquivo de origem tinha `main` com
argparse, a função pura vem para cá e o entrypoint fica de fora (`experiments/`).
**Entrega:** a função pura, a assinatura e o que sobrou de argparse na origem.

**C15 — `core/data/__init__.py`.** Re-exporta as **7 classes** (`ParasiteDataModule`, `FolderDataModule`,
`ParasiteDataset`, `MultiViewDataset`, `MultiCropDataset`, `FolderDataset`, `MultiCropTransform`) para
que o `class_path` fique curto: `core.data.ParasiteDataModule`. **NÃO** re-exporta funções (`build_aug`,
`pil_loader`, `create_splits`) — essas se importam pelo módulo, para o `__init__` não virar fachada de tudo.
**Entrega:** o arquivo e a lista exata de nomes públicos do pacote.

#### Problema e solução do subgrupo `core/data/`

**Problema:** fundir 3 datamodules em 1 muda o `class_path` de **58 configs**.
**Solução recomendada — C16, agente dedicado de reescrita de `class_path`:** um único agente, guiado
por um **mapa antigo→novo com uma linha por classe**, aplica a troca em `configs/` com **um único
`sed`/script** e não com edição manual arquivo a arquivo. Ele **não** interpreta YAML nem
"aproveita para arrumar" outro campo. A conferência é um **teste que instancia cada `class_path`
declarado** — o mesmo teste que pega o bug do `configs/model/lejepa_custom_cnn.yaml` apontando para
`src.modules.LeJEPACNNModule`, que não existe em lugar nenhum. Esse teste é o **gate de fim do Grupo C**:
o mapa é conhecido a priori pela árvore de destino, então C16 dispara junto com os irmãos; só a
execução do teste espera os arquivos existirem.
**Entrega:** o mapa antigo→novo, o comando único aplicado, a contagem de configs tocados e a saída
do teste de instanciação.
**Alternativa descartada:** alias de compatibilidade ou subclasse vazia com o nome antigo. É dívida —
o refactor existe para **apagar** código, não para manter dois nomes vivos apontando para a mesma classe.

---

### Subgrupo `core/blocks/` — um agente por arquivo

**C17 — `core/blocks/mlp_head.py` → `class MLPHead`.** Move a cabeça MLP genérica de
`src/models/models.py`. **NÃO** parametriza profundidade "para depois": só os argumentos que algum
config usa hoje. **Entrega:** a classe e os configs que a instanciam.

**C18 — `core/blocks/two_layer_sigmoid_head.py` → `class TwoLayerSigmoidHead`.** Move a cabeça de duas
camadas com sigmoid. **NÃO** funde com a softplus (C19) num único bloco com argumento `activation` —
são duas classes citadas por `class_path` distinto e a fusão quebraria o eixo de comparação.
**Entrega:** a classe e a origem `file:line`.

**C19 — `core/blocks/two_layer_softplus_head.py` → `class TwoLayerSoftplusHead`.** Idem C18, versão
softplus. **NÃO** compartilha base abstrata com C18. **Entrega:** a classe e a origem `file:line`.

**C20 — `core/blocks/classification_model.py` → `class ClassificationModel`.** Move o modelo de
classificação genérico (encoder + head). **NÃO** embute escolha de encoder: o encoder chega pronto,
por argumento. **Entrega:** a classe e a assinatura de composição.

**C21 — `core/blocks/sigmoid_classification_model.py` → `class SigmoidClassificationModel`.** Move a
variante sigmoid. **NÃO** vira subclasse "por economia" se o corpo divergir do de C20 — se for
idêntico exceto pela head, o relatório propõe a deleção em vez de mover.
**Entrega:** a classe e o veredito "duplicata de `ClassificationModel` ou não".

**C22 — `core/blocks/timm_encoder.py` → `def build_encoder`, `def get_embed_dim`.** Duas funções
finas sobre `timm`. **NÃO** reimplementa nada que `timm` já dá; `get_embed_dim` lê o atributo do
modelo em vez de manter tabela de dimensões.
**Entrega:** as duas funções e a lista de arquiteturas `timm` efetivamente usadas nos configs.

**C23 — `core/blocks/init.py` → `def freeze_encoder`, `def unfreeze_encoder`,
`def init_weights_he`, `def init_weights_xavier`, `def init_weights_trunc_normal`.**
Este é o arquivo que **faz o eixo `init: [flim, he, xavier, random, trunc_normal]` existir**.
**Problema:** hoje a escolha de init está espalhada em `if/elif` dentro dos módulos.
**Solução:** **um único dicionário nome→função** dentro de `init.py`, consumido pelo YAML via
`init_args`. Sem registry, sem plugin, sem decorator de registro, sem `if/elif` em módulo nenhum.
`flim` e `random` não são funções deste arquivo: `flim` entra pela porta do pacote `flim` (F2) e
`random` é o default do torch — o dicionário mapeia esses dois nomes para "não faz nada aqui" e o
relatório diz explicitamente onde cada um é resolvido.
**O que NÃO faz:** não altera nenhum módulo de método para consumir o dicionário — isso é do grupo
que reescreve `methods/`.
**Entrega:** o arquivo, o dicionário com as 5 chaves e o mapa "chave → onde ela é efetivamente aplicada".

**C24 — `core/blocks/__init__.py`.** Re-exporta as classes de bloco para encurtar `class_path`.
**NÃO** re-exporta as funções de init (elas são acessadas por `init_args` com caminho explícito, e
uma fachada aqui esconderia o dicionário único de C23).
**Entrega:** o arquivo e a lista de nomes públicos.

---

### Subgrupo `core/mixins/` — um agente por arquivo

**C25 — `core/mixins/frozen_teacher_checkpoint_mixin.py` → `class FrozenTeacherCheckpointMixin`.**
Move o mixin que congela o professor e ajusta o que entra no checkpoint. **NÃO** muda a chave nem o
formato do state dict salvo — checkpoints já gravados têm que continuar carregando.
**Entrega:** a classe, os módulos que a herdam hoje e a confirmação de compatibilidade do state dict.

**C26 — `core/mixins/knn_kappa_probe_mixin.py` → `class KnnKappaProbeMixin` + `def knn_kappa_probe`,
`def _extract_probe_features`.** Move o probe kNN/κ de validação. A função pública e a privada de
extração ficam **no mesmo arquivo do mixin** — não viram utilitário solto. **NÃO** duplica cálculo de
κ: importa de `core/metrics.py` (C3).
**Entrega:** a classe, as duas funções e a confirmação de que κ vem de `compute_metrics`.

**C27 — `core/mixins/__init__.py`.** Re-exporta os dois mixins. **NÃO** exporta as funções internas
do probe. **Entrega:** o arquivo e os dois nomes públicos.

---

## Grupo F — construir `flim/`

**Grupo paralelo, disparado na MESMA mensagem que o Grupo C. Nenhum agente F depende de irmão F nem
de agente C.** Um agente por arquivo, destinos disjuntos.

`flim/` **não é um método**: é o encoder que os métodos importam. Ele não tem `module.py`, não tem
loss, não aparece em `methods/`.

**F1 — `flim/__init__.py`.** Expõe **só `build`**. Nada mais: nem `Encoder`, nem `arch`, nem
`weights`, nem `PyIFTStrategy`. **O que NÃO faz:** não re-exporta por conveniência — a superfície
mínima é o que sustenta a regra de contrato lá embaixo.
**Entrega:** o arquivo com um único nome público e a declaração explícita do contrato.

**F2 — `flim/build.py` → `def build(arch_json, init, weights_path, in_channels)`.**
A **única porta** que `methods/*` usa. Lê o arch_json, resolve o init, carrega os pesos e devolve o
encoder pronto. É também o ponto onde o valor `flim` do eixo `init` de C23 é honrado.
**O que NÃO faz:** não define arquitetura, não parseia arch nem carrega peso por conta própria —
delega para `arch.py` (F5) e `weights.py` (F6); e não aceita `**kwargs`.
**Entrega:** a assinatura final, o fluxo interno em 4 linhas e a lista dos módulos de `methods/` que
passarão a chamar `from flim import build`.

**F3 — `flim/encoder.py` → `class Encoder`.** Funde `src/models/encoders.py` e
`src/models/custom_cnn.py` e absorve o loader de arch_json/weights que hoje vive junto deles.
**O que NÃO faz:** não é importado por `methods/*` — só por `build.py`; e não mantém as duas classes
antigas vivas como alias.
**Entrega:** a classe, as duas origens `file:line` e a lista de atributos públicos que `build` usa.

**F4 — `flim/flim_residual_encoder.py` → `class FLIMResidualEncoder`, `_StashConv`, `_ResidualConv3`,
`def build_flim_residual_encoder`.** Move o encoder residual FLIM inteiro. `_StashConv` e
`_ResidualConv3` **continuam privadas e continuam neste arquivo** — não sobem para `core/blocks/`,
porque só este encoder as usa.
**O que NÃO faz:** não expõe as privadas no `__init__.py` do pacote.
**Entrega:** a classe, as duas privadas, a função de construção e a origem de cada uma.

**F5 — `flim/arch.py` → `def parse_architecture`, `get_channels_from_arch`,
`get_actual_channels_from_weights`, `override_arch_channels`, `build_encoder_from_arch`,
`build_decoder_from_arch`.** Tudo que lê e interpreta o arch_json fica aqui. Seis funções, nenhuma
classe.
**O que NÃO faz:** não abre arquivo de peso (isso é F6) e não é importado por `methods/*`.
**Entrega:** as seis assinaturas e o mapa "função → quem chama" (esperado: só `build.py`, `encoder.py`
e `flim_residual_encoder.py`).

**F6 — `flim/weights.py` → `def get_bias`, `get_weights`, `shift_weights`, `load_FLIM_encoder`,
`load_FLIM_encoder_from_arch_dict`.** Tudo que lê os pesos FLIM de `data/to_mateus/` fica aqui.
**O que NÃO faz:** não interpreta arch (F5) e não decide init (C23) — só carrega e transforma tensor.
**Entrega:** as cinco assinaturas, o formato de arquivo de peso esperado e a diferença entre as duas
funções de `load_FLIM_encoder*`.

**F7 — `flim/pyift_strategy.py` → `class PyIFTStrategy` (+ `Dockerfile.pyift`).**
**Problema:** `pyift` roda isolado em Docker; quem não tem o container não consegue nem importar o
pacote se o import for de topo.
**Solução:** este arquivo **só define a interface e o comando**; o `import pyift` fica **dentro da
função** (import tardio). Assim `import flim` funciona numa máquina sem o container, e só quebra —
com mensagem clara — na hora em que a estratégia é de fato executada.
**O que NÃO faz:** não importa `pyift` no topo do módulo, em hipótese alguma, e não tenta instalar
nada.
**Entrega:** a classe, a linha exata do import tardio e o comando Docker que ela dispara.

**F8 — `flim/spifil.py` → `def grow`, `def graft`.** Vem de `scripts/spifil_grow.py`,
`scripts/spifil_growth_loop.py` e `scripts/spifil_resnet_graft.py`.
**Problema:** `spifil_growth_loop.py` é ao mesmo tempo **biblioteca** (grow/graft) e **launcher Ray**.
**Solução:** cortar em dois. A lógica de crescimento vem para `flim/spifil.py` **pura — sem Ray, sem
argparse**; a orquestração multi-round vai para `experiments/ray/runners/growth.py` (fora deste grupo).
**Critério de corte, literal:** *se a linha importa `ray` ou `argparse`, não pertence a `flim/`.*
**O que NÃO faz:** não escreve `experiments/ray/runners/growth.py` — apenas entrega o bloco recortado
para o agente dono daquele arquivo.
**Entrega:** as duas funções puras, o bloco de orquestração recortado com origem `file:line` e a
confirmação de que `grep -n "ray\|argparse" flim/spifil.py` não retorna nada.

---

## Regra de contrato entre C, F e os métodos

`build()` de `flim/build.py` é a **única fronteira**. `methods/*` **nunca** importa `flim.encoder`,
`flim.arch`, `flim.weights`, `flim.flim_residual_encoder` nem `flim.pyift_strategy` diretamente. A
única linha permitida é:

```
from flim import build
```

Do lado de `core/`, a fronteira equivalente são os `__init__.py` dos subpacotes (C15, C24, C27): o
`class_path` do YAML aponta para o pacote, não para o módulo interno.

### V1 — agente de verificação de contrato (sequencial, roda no fim)

Único agente que espera os Grupos C e F terminarem. Roda:

```
grep -rn "from flim\." methods/
```

e **falha** se aparecer qualquer coisa além de `from flim import build`. Roda também o teste de
instanciação de `class_path` de C16 e o `grep` de Ray/argparse de F8.
**O que NÃO faz:** não conserta as violações que encontrar — lista cada uma com `file:line` e devolve
para o agente dono daquele arquivo.
**Entrega:** as três saídas de verificação, a lista de violações e a **contagem líquida de linhas**
dos Grupos C e F. Se ela não for negativa, alguma coisa foi adicionada que não deveria.

---

## Grupo M — construir `methods/` (dispatch paralelo, uma pasta por agente-líder, um arquivo por sub-agente)

**Regra da pasta.** `1 pasta = models.py + module.py + loss.py`, expandido em **1 arquivo por classe
pública**. O `__init__.py` re-exporta tudo e é ele que paga a conta do nome longo: com o re-export,
o `class_path` do YAML fica **mais curto que hoje** — `methods.lejepa.LejepaLineModule` no lugar de
`src.modules.lejepa_line_module.LejepaLineModule`. Nome de arquivo comprido só dói se o import for
comprido; com `__init__.py` ele não é.

**Move puro, não redesign.** Nenhum agente deste grupo reescreve lógica, renomeia atributo, troca
default, funde classes parecidas ou "melhora" alguma coisa enquanto move. Se um número mudar depois
do refactor, quem está errado é o refactor, não o número. Toda decisão que não seja mecânica sobe
como pergunta, não vira invenção.

**A árvore-alvo é normativa.** Ela lista exatamente os arquivos e as classes que devem existir. Nenhum
agente cria arquivo fora dela e nenhum agente elimina arquivo dela por achar que dá pra unificar.

### Ordem de dispatch do Grupo M

| Sub-grupo | Agentes | Depende de | Dispatch |
|---|---|---|---|
| **M-LEJEPA** | ML-00 … ML-10, ML-CHAIN | nada | **um dispatch paralelo, um agente por arquivo** |
| **M-AE** | MA-00 … MA-06 | nada (pasta disjunta) | **paralelo, na mesma mensagem de M-LEJEPA** |
| **M-CLS** | MC-00 … MC-02 | nada (pasta disjunta) | **paralelo, na mesma mensagem de M-LEJEPA** |
| **gate LeJEPA** | ML-GATE | M-LEJEPA | sequencial |
| **M-DISTILL** | MD-00 … MD-17 | ML-GATE | paralelo entre si, **depois** de M-LEJEPA |
| **verificador de arestas** | MD-EDGE | M-DISTILL | sequencial |
| **M-NEW** | MN-DINO, MN-BYOL | refactor verde | paralelo entre si |

M-LEJEPA, M-AE e M-CLS são três pastas disjuntas: nenhum agente das três toca um arquivo de outra,
então saem todas na mesma dispatch. A única sequência real do grupo é **M-LEJEPA → M-DISTILL**, porque
o professor da destilação é `LeJEPAFLIMModel` e essa é uma dependência de verdade, não de calendário.

---

### M-LEJEPA — `methods/lejepa/` (um agente por arquivo, paralelo entre si)

**Problema.** `LejepaLineModule` é a raiz de **30 configs** — é a classe de maior blast radius do
refactor inteiro. Qualquer erro no caminho dela quebra metade da grade de experimentos.
**Solução.** A cadeia de herança `LeJEPAModule → LeJEPAFLIMModule → LejepaLineModule` é movida por
**UM agente só** (ML-CHAIN): mesma cadeia = mesma dependência, não se paraleliza. Ela é a **primeira
a ser movida e a primeira a ser testada**. Heads, losses, encoders e models são folhas disjuntas e
saem todos em paralelo com ela, na mesma mensagem.

**ML-00 — `__init__.py`**
Escreve os re-exports da pasta (`LejepaLineModule`, `LeJEPAFLIMModule`, `LeJEPAModule`, models, heads,
losses), na ordem da árvore. Não define nada, não importa de fora de `methods/lejepa` exceto o que a
árvore mandar. Entrega: o arquivo que faz `methods.lejepa.LejepaLineModule` resolver.
Não faz: nenhum `import *`, nenhum símbolo que não esteja na árvore.

**ML-01 — `projection_head.py` → `ProjectionHead`**
Move a classe do arquivo de models atual do LeJEPA para o arquivo próprio, verbatim. Não faz: não a
substitui por `core.blocks.MLPHead` (é head do método, não bloco genérico) nem mexe em dimensões.
Entrega: arquivo com uma classe e seus imports mínimos.

**ML-02 — `projection_head_hawk.py` → `ProjectionHeadHawk`**
Idem ML-01 para a variante Hawk. Não faz: não funde com `ProjectionHead` por serem parecidas — a
árvore pede dois arquivos e o `class_path` do YAML é quem escolhe.
Entrega: arquivo com uma classe.

**ML-03 — `lejepa_model.py` → `LeJEPAModel`**
Move o model base do LeJEPA (hoje em `src/models/lejepa.py`). Não faz: não toca no forward, no
`embed_dim`, nem em nome de atributo lido por checkpoint. Entrega: model isolado, importável.

**ML-04 — `lejepa_flim_model.py` → `LeJEPAFLIMModel`**
Move a variante FLIM (hoje em `src/models/lejepa_flim.py`) e troca a construção do encoder pela porta
única `flim.build`. Não faz: não recria loader de `arch_json`/pesos dentro do método. Entrega: model
FLIM que constrói o encoder por `flim.build` e nada mais.
Este é o símbolo que a destilação vai importar — a assinatura pública dele é congelada aqui.

**ML-05 — `lejepa_cnn_model.py` → `LeJEPACNNModel`**
Move a variante CNN e **investiga o bug do config** (ver bloco abaixo). Não faz: não cria classe nova
por conta própria. Entrega: o model movido + um relatório de evidência com `file:line` sobre o
`LeJEPACNNModule` que o config referencia.

**ML-06 — `simple_cnn_model.py` → `SimpleCNNModel`**
Move a CNN simples usada como baseline. Não faz: não a manda para `core/blocks` (é model de método,
não bloco compartilhado) e não mexe em canais. Entrega: arquivo com uma classe.

**ML-07 — `ijepa_encoder.py` → `IJEPAEncoder` + privadas**
Move `IJEPAEncoder` **junto com** `_SelfAttention`, `_Block`, `_MLP1`, `_MLP2`, `_PatchEmbeddings`,
`_Embeddings`, `_Encoder`, `_AttentionWrapper`, `_AttentionOutput`, `_IJepaViT` — todas ficam privadas
no mesmo arquivo, um arquivo só. Não faz: não promove nenhuma privada a arquivo próprio, não exporta
nenhuma delas no `__init__.py`, não troca por `timm`. Entrega: o ViT do I-JEPA em um arquivo fechado.

**ML-08 — `simple_sigreg.py` → `SimpleSIGReg`**
Move a loss SIGReg simplificada (hoje em `losses/lejepa_loss.py`). Não faz: não unifica com
`RealSIGReg`, não muda redução nem escala. Entrega: uma loss, um arquivo.

**ML-09 — `real_sigreg.py` → `RealSIGReg` (Epps-Pulley)**
Move o teste de Epps-Pulley (hoje em `losses/epps_pulley.py`). Não faz: não reimplementa a estatística,
não troca constantes numéricas, não "estabiliza" nada. Entrega: a loss real, verbatim, com as mesmas
constantes.

**ML-10 — `invariance_loss.py` → `def invariance_loss`**
Move a função de invariância para arquivo próprio. É função, continua função — não vira classe.
Entrega: um arquivo, uma função, sem estado.

**ML-CHAIN — `lejepa_module.py`, `lejepa_flim_module.py`, `lejepa_line_module.py`**
Um agente só para os três, nesta ordem: `LeJEPAModule`, depois `LeJEPAFLIMModule(LeJEPAModule)`,
depois `LejepaLineModule(LeJEPAFLIMModule)`. Move `training_step`/`validation_step`/
`configure_optimizers` como estão e troca os imports dos models/losses pelos novos caminhos.
Não faz: não achata a herança, não move hiperparâmetro para outro nível da cadeia, não mexe em
`save_hyperparameters`, não deleta argparse aqui (isso é regra global, mas a mudança de comportamento
é decisão do gate). Entrega: a cadeia completa importável por `methods.lejepa`.
Prioridade: **começa por `LejepaLineModule`** — é o alvo das 30 configs.

**ML-GATE — o teste primeiro (sequencial, depois de todo o M-LEJEPA)**
Instancia `methods.lejepa.LejepaLineModule` a partir de **uma** das 30 configs reais, com
`class_path` já reescrito, e compara com a instanciação equivalente pelo caminho antigo. Um check
runnable, `assert`, sem framework. Entrega: verde/vermelho + a lista dos 30 YAMLs cujo `class_path`
precisa ser reescrito. Vermelho aqui bloqueia M-DISTILL.

**Problema e solução — `LeJEPACNNModel` vs `LeJEPACNNModule` (dono: ML-05)**
A árvore-alvo tem `lejepa_cnn_model.py → class LeJEPACNNModel` (um **model**), mas
`configs/model/lejepa_custom_cnn.yaml` aponta para `src.modules.LeJEPACNNModule` (um **module**), que
não existe em lugar nenhum do repositório — nem em `src/modules/__init__.py`, nem em arquivo algum;
só aparece numa docstring. Esse config quebra na instanciação e é exatamente o que `test_class_paths.py`
pega. ML-05 investiga a docstring que cita o nome, junta a evidência com `file:line` e **só então**
decide, explicitamente, entre:
- **(a)** o config estava errado e deve apontar para o model existente sob a nova árvore
  (`methods.lejepa.LeJEPACNNModel`), ou
- **(b)** falta de fato um `LeJEPACNNModule`, e ele precisa ser escrito espelhando `LeJEPAFLIMModule`.

Se a evidência for ambígua, **pergunta ao usuário** e para — não inventa classe para fazer um YAML
passar. A escolha entre (a) e (b) muda o que roda naquele experimento; não é detalhe de import.

---

### M-AE — `methods/autoencoder/` (um agente por arquivo, paralelo entre si e com M-LEJEPA)

**MA-00 — `__init__.py`**
Re-exporta `AutoEncoderFlimModule`, `AutoEncoderFLIM`, `AutoEncoder`, `AutoEncoderClassifier`,
`ResNetDecoder`, `ResidualUpBlock`. Não define nada. Entrega: `methods.autoencoder.<Classe>` resolvendo
para todas.

**MA-01 — `residual_up_block.py` → `ResidualUpBlock`**
Extrai o bloco de upsampling de dentro do arquivo de autoencoder atual. Não faz: não generaliza para
`core/blocks` (é bloco do decoder deste método) e não mexe em normalização/ativação.
Entrega: um bloco, um arquivo.

**MA-02 — `resnet_decoder.py` → `ResNetDecoder`**
Move o decoder, que passa a importar `ResidualUpBlock` do arquivo do MA-01. Não faz: não muda a
sequência de canais nem o número de estágios. Entrega: decoder isolado.

**MA-03 — `autoencoder_flim.py` → `AutoEncoderFLIM`**
Move a variante FLIM e faz o encoder vir de `flim.build`. Não faz: não carrega `arch_json`/pesos por
conta própria. Entrega: model FLIM com uma única porta de entrada para o encoder.

**MA-04 — `autoencoder.py` → `AutoEncoder`**
Move o autoencoder base (hoje em `src/models/autoencoder_resnet.py`). Não faz: não funde com a variante
FLIM. Entrega: model base isolado.

**MA-05 — `autoencoder_classifier.py` → `AutoEncoderClassifier`**
Move o classificador que roda sobre o encoder do autoencoder. Não faz: não duplica o head — se o head
for MLP genérico, importa `core.blocks.MLPHead`. Entrega: classificador com head vindo de `core/blocks`.

**MA-06 — `autoencoder_flim_module.py` → `AutoEncoderFlimModule`**
Move o módulo de treino e **apaga a `class Head` interna**, que é idêntica a `core.blocks.MLPHead` —
o módulo passa a importar de lá. Não faz: não reescreve as steps, não mexe em loss, não mantém cópia
local do head. Entrega: o módulo enxuto + o relatório de linhas (antes/depois).

**Problema e solução — 1314 linhas em um módulo (dono: MA-06)**
O corte é **mecânico e já está dado pela árvore**: os blocos viram arquivos próprios (MA-01, MA-02),
os models saem para os seus (MA-03, MA-04, MA-05) e a `Head` some para `core.blocks.MLPHead`. O que
tem que sobrar em `autoencoder_flim_module.py` é `training_step`, `validation_step` e
`configure_optimizers` — mais nada. Se depois do corte o arquivo ainda passar de ~300 linhas, o
excedente é quase certamente uma de duas coisas: **argparse**, que vai fora (regra global do refactor:
flag vira `init_args` no YAML), ou **avaliação embutida**, que vai para `eval/`. MA-06 nomeia o
excedente linha a linha e o encaminha para um desses dois destinos; não deixa "sobra" dentro do módulo.

---

### M-DISTILL — `methods/distillation/` (paralelo interno, **depois** de M-LEJEPA)

**MD-00 — `__init__.py`**
Re-exporta os 4 modules, o `FrozenTeacher`, as 5 projection heads, o `StudentClassificationHead`, as
3 losses de classe e as 2 funções (`kd_loss`, `prepare_teacher_input`). Não re-exporta `cli.py`.
Entrega: `class_path` curto para os 4 modules.

**MD-01 — `frozen_teacher.py` → `FrozenTeacher`**
Move o wrapper do professor. Não faz: não reimplementa o congelamento de pesos se
`core.mixins.FrozenTeacherCheckpointMixin` já resolve o carregamento do checkpoint — usa o mixin.
Entrega: professor congelado, sem lógica de checkpoint duplicada.

**MD-02 — `distillation_projection_head.py` → `DistillationProjectionHead`**
Move a head linear de destilação, verbatim. Não faz: não parametriza, não funde com as irmãs.
Entrega: uma classe, um arquivo.

**MD-03 — `conv_distillation_projection_head.py` → `ConvDistillationProjectionHead`**
Move a head convolucional, verbatim. Não faz: não altera kernel, stride ou padding. Entrega: idem.

**MD-04 — `one_layer_conv_distillation_projection_head.py` → `OneLayerConvDistillationProjectionHead`**
Move a head conv de uma camada, verbatim. Não faz: não a expressa como caso particular da MD-03.
Entrega: idem.

**MD-05 — `one_layer_1x1_conv_distillation_projection_head.py` → `OneLayer1x1ConvDistillationProjectionHead`**
Move a head 1x1 de uma camada, verbatim. Não faz: não troca o 1x1 por parâmetro. Entrega: idem.

**MD-06 — `two_layer_1x1_conv_bn2d_distillation_projection_head.py` → `TwoLayer1x1ConvBN2dDistillationProjectionHead`**
Move a head 1x1 de duas camadas com `BatchNorm2d`, verbatim. Não faz: não mexe em `eps`/`momentum` da
BN nem na ordem conv-bn-ativação. Entrega: idem.

**MD-07 — `student_classification_head.py` → `StudentClassificationHead`**
Move o head de classificação do aluno. Não faz: não substitui por `core.blocks.MLPHead` sem provar que
são idênticos — se forem, reporta e o import muda; se não forem, fica. Entrega: head + o veredito da
comparação.

**MD-08 — `kl_distillation_loss.py` → `KLDistillationLoss`**
Move a loss KL. Não faz: não mexe em temperatura, redução ou no fator `T^2`. Entrega: uma loss, um arquivo.

**MD-09 — `mse_distillation_loss.py` → `MSEDistillationLoss`**
Move a loss MSE. Não faz: não unifica com a KL nem com a cosseno. Entrega: idem.

**MD-10 — `cosine_distillation_loss.py` → `CosineDistillationLoss`**
Move a loss de cosseno. Não faz: não muda normalização nem sinal. Entrega: idem.

**MD-11 — `kd_loss.py` → `def kd_loss`**
Move a função de destilação clássica para arquivo próprio. Continua função. Não faz: não vira classe,
não vira método de um dos modules. Entrega: uma função pura.

**MD-12 — `teacher_input.py` → `def prepare_teacher_input`**
Move o preparo do input do professor (resize/normalização/canais). Não faz: não muda interpolação nem
estatísticas de normalização — isso mudaria os embeddings do professor. Entrega: uma função pura.

**MD-13 — `distillation_module.py` → `DistillationModule`**
Move o módulo base de destilação (hoje em `src/models/distillation.py`), **sem** os grupos de argparse
das linhas 279-497, que vão para MD-17. Não faz: não altera as steps nem o acoplamento com o professor.
Entrega: módulo base limpo de CLI.

**MD-14 — `distillation_conv_module.py` → `DistillationConvModule`**
Move a variante conv. Não faz: não funde com o base por herança nova nem muda a head default —
a head é escolhida pelo `class_path` no YAML. Entrega: módulo isolado.

**MD-15 — `distillation_one_layer_module.py` → `DistillationOneLayerModule`**
Move a variante de uma camada. Não faz: não deduplica contra MD-14/MD-16. Entrega: módulo isolado.

**MD-16 — `distillation_two_layer_module.py` → `DistillationTwoLayerModule`**
Move a variante de duas camadas. Não faz: idem. Entrega: módulo isolado.

**MD-17 — `cli.py` → `add_student_flags`, `resolve_student`, `add_distill_flags`, `resolve_distill_flags`, `derive_flim_paths`, `distill_run_tags`**
Recorta as linhas 279-497 de `src/models/distillation.py` e as cola aqui **sem mudar uma vírgula**, para
que o corte do módulo seja um move puro. Não faz: não conserta as flags, não renomeia argumento, não é
importado por nenhum entrypoint novo. Entrega: o arquivo de transição + a tabela `flag → init_args`
que vai ser usada para matá-lo.

**Problema e solução — dependência entre métodos**
`methods/distillation` **importa** `methods/lejepa`: o professor é `LeJEPAFLIMModel`. Essa é a **única
aresta entre métodos** que o refactor aceita, e ela fica **explícita**. Consequências:
- M-DISTILL roda **depois** de M-LEJEPA e depois do ML-GATE verde. Dependência real; as duas pastas
  não vão em paralelo.
- O import é o mais estreito possível: `from methods.lejepa import LeJEPAFLIMModel`, e nada mais —
  nem module, nem loss, nem head do LeJEPA.

**MD-EDGE — verificador de arestas (sequencial, no fim de M-DISTILL)**
Roda `grep -rn "from methods\." methods/` e **falha** se aparecer qualquer aresta além de
`from methods.lejepa import LeJEPAFLIMModel` dentro de `methods/distillation/`. Não faz: não conserta
o import sozinho — reporta `file:line` e para. Entrega: verde/vermelho e a lista de arestas encontradas.

**Problema e solução — o `cli.py` da destilação**
A árvore mantém `methods/distillation/cli.py`, mas a regra global do refactor é **matar argparse**
(argparse polui os comandos do terminal; só sobrevive se for impossível sem). Resolução normativa:
`cli.py` nasce como **zona de transição**. Ele existe apenas para que o corte de
`src/models/distillation.py` seja um **MOVE puro**, sem mudança de comportamento. Em seguida, cada flag
vira `init_args` do módulo no YAML e a função correspondente é apagada. **Alvo final:
`methods/distillation/cli.py` deletado.** Enquanto existir, **nenhum entrypoint novo pode importá-lo** —
quem importa `cli.py` é código legado em vias de morrer, não código novo.

**Problema e solução — as 5 projection heads**
São 5 classes quase iguais, e a tentação é fundir tudo em uma classe parametrizada por string. **Não
fundir.** A árvore do usuário é normativa e o `class_path` do YAML já é o seletor de head — trocar isso
por um parâmetro `head_type="one_layer_1x1"` move a escolha do lugar declarativo (YAML) para dentro do
código, que é o contrário do refactor. Isso fica registrado aqui para que nenhum agente "melhore" a
árvore por conta própria; MD-02 … MD-06 recebem essa proibição no próprio subprompt.

---

### M-CLS — `methods/classification/` (paralelo com M-LEJEPA e M-AE)

**MC-00 — `__init__.py`**
Re-exporta `ClassificationFlimModule` e `ClassificationFinetuneModule`. Não define nada.
Entrega: `class_path` curto para os dois modules de classificação.

**MC-01 — `classification_flim_module.py` → `ClassificationFlimModule`**
Move o módulo de classificação com encoder FLIM (hoje `src/modules/classification_flim_module.py`), com
o encoder vindo de `flim.build` e o head de `core.blocks`. Não faz: não mistura com o caminho de
fine-tune. Entrega: módulo isolado.

**MC-02 — `classification_finetune_module.py` → `ClassificationFinetuneModule`**
Move o antigo `classifier_module.py` (o de fine-tune) e o **renomeia** para o nome da árvore. Não faz:
não muda o que a classe faz — o rename é só de arquivo/símbolo. Entrega: módulo renomeado + a lista dos
YAMLs cujo `class_path` aponta para o nome antigo e precisam ser atualizados no mesmo commit.

---

### M-NEW — `methods/dino_v2/` e `methods/byol/` (a escrever; paralelo entre si, **depois** do refactor verde)

**Regra.** Estes são os dois métodos que **provam que a arquitetura funciona**. O critério é direto: se
escrever um método novo exigir tocar em `experiments/ray/`, em `train.py` ou em `core/cli.py`, a
arquitetura falhou e o refactor não terminou. Cada um dos dois agentes só pode fazer três coisas:
1. criar sua pasta em `methods/`;
2. criar seus YAMLs em `configs/model/<método>/<init>/<dataset>/split_N.yaml`;
3. criar seu experiment YAML em `experiments/<método>/`.

**Zero linha fora disso.** Se um deles precisar de uma quarta coisa, ele **para e reporta** — a
necessidade da quarta coisa é o resultado do experimento, não um obstáculo a contornar.

**MN-DINO — `methods/dino_v2/`**
Escreve `__init__.py` (re-exporta `DINOv2Module` e os demais), `dino_v2_model.py` → `DINOv2Model`,
`dino_head.py` → `DINOHead`, `dino_loss.py` → `DINOLoss` (centering + sharpening), `dino_v2_module.py`
→ `DINOv2Module`. Reusa `core/blocks` (encoder timm, heads), `core/mixins.KnnKappaProbeMixin` para a
métrica online e `flim.build` quando a inicialização for FLIM. Não faz: não escreve encoder próprio,
não escreve probe próprio, não toca em `experiments/ray/`. Entrega: pasta + YAMLs + experiment YAML.

**MN-BYOL — `methods/byol/`**
Escreve `__init__.py`, `byol_model.py` → `BYOLModel` (rede online + rede target com EMA),
`byol_predictor.py` → `BYOLPredictor`, `byol_module.py` → `BYOLModule` (o passo de EMA vive no module).
Reusa `core/blocks`, `core/mixins.KnnKappaProbeMixin` e `flim.build`. Não faz: não adiciona dependência
nova para BYOL (o EMA é meia dúzia de linhas), não duplica o probe kNN/κ. Entrega: pasta + YAMLs +
experiment YAML.

**Possível solução para o custo.** O custo de escrever DINOv2 e BYOL cai para quase nada se eles
consumirem o que já existe: `core/blocks` (heads e encoders timm), `core/mixins` (`KnnKappaProbeMixin`
para a métrica online, `FrozenTeacherCheckpointMixin` quando houver target congelado) e `flim.build`
para a inicialização FLIM. Se DINOv2 ou BYOL precisarem de um bloco que não existe, esse bloco **nasce
em `core/blocks/`** — compartilhado desde o primeiro dia — e **não** dentro da pasta do método. Bloco
novo escondido dentro de um método é a semente da próxima duplicação.

---

## Grupo Y — reorganizar `configs/` (261 YAMLs)

A árvore-alvo (colada acima, verbatim) é **normativa**: nenhum agente deste grupo a altera, discute
ou "melhora". A regra que ela codifica, em uma frase por nível:

- `configs/default.yaml` — global, um único arquivo.
- `configs/dataset/<dataset>/split_N.yaml` — **57 arquivos viram 9** (3 datasets × 3 splits).
- `configs/model/<método>/<init>/<dataset>/split_N.yaml` — método → init → dataset → split, nesta
  ordem, sempre.
- `configs/model/<método>/_variants/` — tudo que está **fora da grade**, isto é, o que o glob da
  grade não pega. Um YAML só vai para `_variants/` se ele genuinamente não tem uma célula
  `(método, init, dataset, split)`; `_variants/` não é depósito de conveniência.
- `configs/generated/` — **SAÍDA**, não fonte. Vai para o `.gitignore`, no formato
  `mlp/{freeze,unfreeze}/<dataset>/<run_id>.yaml`.

### As DUAS únicas f-strings de resolução (nada de dict à mão)

```
dataset_config = f"configs/dataset/{dataset}/split_{split}.yaml"
model_config   = f"configs/model/{method}/{init}/{dataset}/split_{split}.yaml"
```

Não existe uma terceira. Não existe dicionário auxiliar, tabela de alias, nem `if` por dataset no
caminho de resolução. **Se um desses arquivos não existir, FALHE — imprimindo o caminho esperado.**
Não cair em config genérico, não cair em "o mais parecido", não completar campo faltante com default:
o caso especial de protozoan em `scripts/run_ssl_ray.py:113-121` é exatamente o comportamento que
este refactor está matando, e reintroduzi-lo em qualquer forma anula o grupo Y inteiro.

### Decomposição atômica — quem roda com quem

| Agente | Escopo | Depende de | Dispatch |
|---|---|---|---|
| **Y1** | inventário e mapa antigo→novo | nada | sequencial, **primeiro** |
| **Y2..Yn** | migração, um agente por dataset | Y1 | **um dispatch paralelo, 3 agentes** |
| **Y-VAR** | variantes fora da grade | Y1 | paralelo com Y2..Yn |
| **Y-GEN** | configs gerados + `.gitignore` | Y1 | paralelo com Y2..Yn |
| **Y-CLASSPATH** | reescrita dos `class_path` | Y2..Yn, Y-VAR, Y-GEN **e** `core/` + `methods/` já existindo | sequencial, **último** |

Y2..Yn, Y-VAR e Y-GEN são **disjuntos por construção** — cada um toca um conjunto de arquivos que
nenhum outro toca — e por isso vão em uma única mensagem, em paralelo. Nenhum deles espera por um
irmão. Só Y1 (antes) e Y-CLASSPATH (depois) são sequenciais, e ambos por dependência real de dados.

---

### Y1 — inventário e mapa antigo→novo

Roda **sozinho e primeiro**; todos os outros agentes deste grupo consomem a saída dele.

Lista os 261 YAMLs e classifica **cada um** em exatamente uma das cinco classes:

| Classe | Significa |
|---|---|
| `dataset` | vira `configs/dataset/<dataset>/split_N.yaml` |
| `model-na-grade` | tem célula `(método, init, dataset, split)` completa |
| `model-variante` | fora da grade → `_variants/` |
| `gerado` | é saída de script → `configs/generated/` |
| `órfão` | ninguém referencia, ou não se encaixa em nenhuma das quatro |

**Saída:** uma tabela/CSV com quatro colunas — `caminho_antigo, caminho_novo, classe, evidência`.
`evidência` é `file:line` de quem lê ou escreve aquele YAML, ou a chave do YAML que determinou a
classe. Sem âncora, a linha é palpite, e palpites vão numa seção separada marcada como tal.

**Nada é movido nesta fase.** Y1 é read-only. Se dois YAMLs mapeiam para o mesmo `caminho_novo`,
isso é uma colisão: Y1 reporta o par e **não** escolhe um vencedor sozinho.

---

### Y2..Yn — migração por dataset (paralelo, um agente por dataset)

Um agente para `helminth-eggs`, um para `helminth-larvae`, um para `protozoan-cysts`. Cada agente
move **apenas** os YAMLs classificados por Y1 como pertencentes ao seu dataset, e nada mais.

- Movimentação sempre com **`git mv`** — preserva histórico. Nunca `cp` + `rm`, nunca reescrever o
  arquivo no destino.
- O conteúdo do YAML não muda nesta fase. Só o caminho. (`class_path` é problema do Y-CLASSPATH;
  caminho absoluto é problema do agente de `arch_json`.)
- Partição por dataset torna os agentes disjuntos por construção. Se dois agentes disputarem o mesmo
  arquivo, isso é bug na classificação do Y1 — reporte a Y1, não resolva no braço.

Cada agente reporta: quantos moveu, quantos do seu dataset ficaram para trás e por quê.

---

### Y-VAR — variantes fora da grade

Move para `configs/model/lejepa/_variants/`:

- `resnet50.yaml`
- `real_sigreg.yaml`
- `simple_sigreg.yaml`
- `custom_cnn.yaml`

`custom_cnn.yaml` é **o quebrado**: aponta para `LeJEPACNNModule`, classe que não existe em nenhum
arquivo do repositório. Duas saídas aceitáveis, e o agente **reporta explicitamente qual escolheu**:

1. **movido e consertado** — se, e somente se, existir no repo uma classe que é inequivocamente a
   destinatária, com `file:line` como prova;
2. **movido e marcado como quarentena** — comentário no topo do YAML dizendo que ele não instancia,
   e entrada no relatório. Esta é a saída padrão quando (1) exige adivinhação.

O agente não inventa a classe faltante e não apaga o arquivo.

---

### Y-GEN — configs gerados

`configs/evaluate/mlp/{freeze,unfreeze}/<parasito>/*.yaml` deixam de ser fonte versionada e viram
**saída** em `configs/generated/mlp/{freeze,unfreeze}/<dataset>/<run_id>.yaml`.

O agente:

1. adiciona `configs/generated/` ao `.gitignore`;
2. aponta o gerador — `generate_mlp_configs.py`, que neste refactor vira `experiments/gen_configs.py`
   — para escrever em `configs/generated/`;
3. **reporta quantos YAMLs saem do controle de versão** (número exato, não estimativa), e confirma
   que o gerador regenera cada um deles.

Se algum YAML de `evaluate/` não for reproduzível pelo gerador, ele **não** sai do versionamento:
vira item de relatório, porque removê-lo do git perderia informação.

---

### Y-CLASSPATH — reescrita dos `class_path` (SEQUENCIAL)

Roda **depois** que `core/` e `methods/` já existem — antes disso o destino do rename não é
importável e o teste abaixo não pode passar.

Aplica o mapa antigo→novo de classes em **todos** os YAMLs. Exemplo canônico:

```
src.modules.lejepa_line_module.LejepaLineModule  →  methods.lejepa.LejepaLineModule
```

Regra de execução: **um script único, um diff, um teste**. O script faz a substituição em lote sobre
o mapa; o diff é revisado de uma vez; o teste importa e **instancia todos** os `class_path` de todos
os YAMLs, falhando com o caminho do YAML e o `class_path` ofensor. Não é um agente por arquivo — a
transformação é a mesma em todos, e n agentes aqui só produziriam n diffs para revisar.

---

## Os 24 configs que faltam — problema e soluções possíveis

**Fato:** `model/lejepa/{he,xavier,random,trunc_normal}/{helminth-eggs,helminth-larvae}/` **não
existem hoje**. Só a célula `protozoan-cysts` foi preenchida, e a ausência das outras é justamente o
que o caso especial em `scripts/run_ssl_ray.py:113` está compensando em runtime. Traduzido para a
árvore-alvo, é um buraco de grade de tamanho exato:

**4 inits × 2 datasets × 3 splits = 24 arquivos.**

Enquanto o buraco existir, a f-string `model_config` falha para 24 combinações legítimas — e é
precisamente por isso que ela **deve** falhar, em vez de cair num genérico.

### S1 (recomendada) — gerar os 24 a partir de um template

Um agente gerador lê, para cada célula faltante, um YAML **já existente da mesma célula da grade** —
mesmo método, mesmo dataset, mesmo split, init `flim` — troca **apenas a chave de init** e escreve o
arquivo novo no caminho da grade.

- **Por que é a recomendada:** é mecânico (uma chave muda, o resto é cópia), auditável por diff (todo
  arquivo gerado difere do seu template em uma linha; qualquer diff maior é bug do gerador) e fecha a
  grade — a f-string passa a valer **sempre**, e o caso especial de protozoan morre sem substituto.
- **Custo:** 24 arquivos novos versionados.
- **Ganho:** zero lógica de resolução em runtime. A grade fica simétrica e o pré-voo passa a poder
  afirmar "toda célula existe" como invariante, não como exceção.

### S2 — declarar o buraco no `exclude:` do experiment YAML

Não gera arquivo nenhum. A grade simplesmente não passa por essas 24 células, porque elas estão
listadas em `exclude:`.

- **Correto quando:** o usuário **não quer** rodar esses experimentos. Nesse caso os 24 arquivos
  seriam lixo versionado que nunca é lido.
- **Custo:** a grade continua assimétrica — `protozoan-cysts` tem inits que os outros dois datasets
  não têm — e quem lê a árvore precisa do `exclude:` para entender por quê.
- **Barato, honesto, e não reintroduz resolução em runtime.** É a segunda melhor opção, não uma
  opção ruim.

### S3 (rejeitada) — herança / `defaults` no YAML resolvendo init em runtime

Um YAML base por dataset, com o init resolvido por `defaults:`/herança na hora de carregar.

**Rejeitada.** Reintroduz exatamente a camada de resolução que este refactor está apagando: volta a
existir lógica escondida entre o nome da célula e o arquivo que de fato é lido, e o caminho impresso
num erro deixa de ser o caminho do arquivo. Trocaria 24 arquivos explícitos por uma indireção que
ninguém consegue auditar por `ls`. Registrada aqui para que não seja reproposta.

### Decisão

**A escolha entre S1 e S2 é do usuário** — ela depende de uma informação que o refactor não tem (se
esses 24 experimentos vão ser rodados ou não), e portanto nenhum agente a toma sozinho.

**Independente da escolha, o pré-voo LISTA os 24 caminhos faltantes**, um por linha, com o caminho
completo esperado. Sob S1 a lista é a ordem de serviço do gerador e esvazia; sob S2 a lista é a
prova de que o `exclude:` cobre exatamente esses 24 e nenhum outro.

---

## O `arch_json` absoluto — problema e solução

**Fato:** `configs/model/lejepa_line_he_protozoan_train1.yaml` tem

```
arch_json: /dados/home/moliveira/scalable_FLIM_self_supervised/...
```

Caminho absoluto, apontando para o home de uma máquina específica. O config só roda ali, e falha em
qualquer outro lugar por um motivo que não tem nada a ver com o experimento.

**Solução:** um agente varre **TODOS** os YAMLs procurando valores que começam com `/` e converte
cada um para caminho relativo à raiz do repositório.

Onde o arquivo apontado **não existir** no repositório, o agente **NÃO inventa** — não escolhe o
arquivo de nome parecido, não cria um placeholder, não comenta a chave. Ele reporta o caminho
absoluto original, o YAML onde está, e **pede o arquivo** ao usuário.

**Regra permanente:** caminho absoluto em config é **erro de lint**, checado pelo pré-voo
(**checagem 16**). E é um erro **DIFERENTE** de "caminho inexistente": um diz que o config não é
portátil, o outro diz que o alvo sumiu. Um caminho relativo apontando para um arquivo que existe está
correto; um caminho absoluto apontando para um arquivo que existe **naquela máquina** continua sendo
erro. As duas checagens têm mensagens distintas e nunca são fundidas.

---

## O que a reorganização paga

As duas f-strings substituem todo o código de resolução de config. Isso apaga:

- **`_FLIM_YAMLS`** — dict de 9 entradas escrito à mão em `run_experiments.py:57-67`.
- **`_model_config()` e `_data_config()`** em `run_ssl_ray.py:113`.
- **`_arch_json` / `_split_json` / `_flim_weights_path`** replicados em 4 ray scripts.
- **`DATASET_LONG_TO_SHORT` no caminho de resolução** — a pasta passa a usar **um** vocabulário
  (`helminth-eggs`), e o alias some.

### Risco e mitigação

`DATASET_LONG_TO_SHORT` pode ser usado em **OUTROS lugares** além da resolução de config: nomes de
run no W&B, nomes de pasta de artifacts, rótulos em plots, chaves de CSV de resultados. Apagá-lo
inteiro renomearia runs e pastas silenciosamente — e isso quebra continuidade histórica sem que
nenhum teste reclame.

**Mitigação:** um agente roda

```
grep -rn "DATASET_LONG_TO_SHORT"
```

e classifica **cada** ocorrência em "caminho de resolução de config" ou "outro uso". Ele remove
**somente** as do caminho de resolução. Todas as demais viram **item explícito de relatório**, com
`file:line` e o que aquele uso produz (nome de run, nome de pasta, rótulo), para o usuário decidir
depois. Nenhuma ocorrência é removida por semelhança.

---

## Critério de aceitação do grupo Y

- [ ] `configs/dataset/` tem **exatamente 9 arquivos**.
- [ ] Nenhum YAML fora de `_variants/` e `generated/` deixa de casar com uma das duas f-strings.
- [ ] `grep -rn "^\s*arch_json: /" configs/` não retorna nada.
- [ ] Todo `class_path` de todo YAML é **importável** (o teste do Y-CLASSPATH instancia todos).

---

## TAREFA — unificar os launchers Ray sob um "experiment YAML" (multiagente)

**Repositório-alvo:** `Scalable_Hybrid_FLIM`
**Natureza da tarefa:** deduplicação + uma entrada única. Tudo o que se pede aqui já existe
no repositório — só existe **sete vezes, em sete arquivos**. O trabalho é deixar **um** de
cada e apontar todo mundo para ele.

**Regra ponytail, inegociável:** esta tarefa só *remove* código de orquestração. Nenhuma
dependência nova, nenhum registry, nenhum plugin, nenhuma camada de config além do próprio
YAML. Atalho deliberado leva comentário `# ponytail:` nomeando o teto.

**Comportamento não muda.** O comando montado hoje por `scripts/run_ssl_ray.py:404-408` é o
alvo; ele já está certo. Se um número mudar depois do refactor, o refactor está errado.

**Multiagente, obrigatório.** São muitas tarefas independentes, não uma — nunca em passe
único. Cada grupo abaixo é despachado como **uma única mensagem com todos os seus agentes em
paralelo**; um agente por subprompt, um agente por arquivo. Nada é sequencial exceto onde um
grupo genuinamente consome a saída do anterior (R1 → {R2…R9, R11, E1…E4} → R10 → T1).
Nenhum agente fica ocioso esperando um irmão.

**Aviso operacional:** vai ter alguns experimentos rodando nesse momento, não mexa neles.

---

### 0. Contexto do repositório (verificado, não re-descobrir)

O repo tem SETE arquivos que orquestram Ray, ~5.900 linhas, com as mesmas
funções copiadas:

```
  scripts/run_ssl_ray.py             (873)
  scripts/distillation_ray.py        (744)
  scripts/distillation_conv_ray.py  (1181)
  scripts/autoencoder_flim_ray.py   (1094)
  scripts/classification_flim_ray.py (850)
  scripts/spifil_growth_loop.py      (526)
  src/evaluate/ray_mlp_queue.py      (668)
```

Duplicações medidas:

```
  _log(msg, level)                                        em 7
  class GpuSlotScheduler                                  em 4
  _arch_json / _flim_weights_path / _split_json / _run_name em 4
  _query_wandb_state / _should_skip_experiment            em 3
  validate_experiment / validate_output_dir / build_experiment_grid em 3
  class ExecutionState                                    em 2
```

Existem DOIS protocolos de execução incompatíveis:

```
  A) scripts/run_ssl_ray.py:404 — `python src/main.py fit --config <global>
     --config <dataset> --config <model>` (LightningCLI). CORRETO, é o alvo.
  B) os demais — `python -m src.modules.<módulo>` + ~15 flags argparse.
     Deve ser eliminado.
```

     --config <dataset> --config <model>` (LightningCLI). CORRETO, é o alvo.
  B) os demais — `python -m src.modules.<módulo>` + ~15 flags argparse.
     Deve ser eliminado.

Vocabulário divergente a canonizar:

```
  quais GPUs : --gpu-ids | --gpus | --gpu-slots   → canônico: gpu_ids
  percentuais: --pcts | --percentages             → canônico: percentages
  pular feito: --resume | --ignore-existing | --skip-existing | --check-wandb
               | --retry                          → canônico: skip
```

O diagnóstico que o executor precisa carregar antes do prompt: os três launchers já têm
vocabulários diferentes pra mesma coisa.

| Conceito | run_ssl_ray | spifil_growth_loop | distillation_conv_ray |
|---|---|---|---|
| quais GPUs | `--gpu-ids` | `--gpus 1 2 3` | `--gpu-slots` |
| porcentagens | `--pcts` | `--percentages` | `--percentages` |
| pular feito | `--resume` / `--ignore-existing` | — | `--skip-existing` / `--check-wandb` / `--retry` |

É por isso que você redigita a linha inteira: não há um nome canônico pra nada.



```python
cmd = [sys.executable, "src/main.py", "fit",
       "--config", DEFAULT_CONFIG_YAML,   # global
       "--config", exp["data_config"],    # dataset
       "--config", exp["model_config"]]   # model
```

Padronização do Ray proposta, onde ele deve ser capaz de rodar em múltiplas GPUs:

```
experiments/ray/
├── __init__.py
├── gpu_slot_scheduler.py    → class GpuSlotScheduler      (4 cópias )
├── execution_state.py       → class ExecutionState        (2 cópias )
├── paths.py                 → def arch_json, flim_weights_path, split_json, run_name
├── grid.py                  → def build_grid, validate_experiment, validate_output_dir
├── skip.py                  → def query_wandb_state, should_skip
├── train_worker.py          → @ray.remote def run_train(exp)
├── eval_worker.py           → @ray.remote def run_eval(exp)     ← ray_mlp_queue não é treino
└── launch.py                → CLI única
```

---

### 1. Objetivo

Um arquivo YAML por experimento descreve a corrida inteira (grade + recursos +
wandb + saída). O launcher Ray recebe SÓ esse arquivo. O usuário para de
redigitar a linha de comando gigante.

O comando que ele digita hoje, e que deve virar YAML:

```bash
    python scripts/spifil_growth_loop.py \
      --work-dir artifacts/spifil_growth/grid4 --percentages 5 50 \
      --gpus 1 2 3 --max-concurrent-per-gpu 3 --cpus-per-experiment 10 \
      --max-rounds 2 --num-workers 2 --wandb --wandb-project phd_thesis_grid4 \
      2>&1 | tee logs/spifil_growth_$(date +%Y%m%d_%H%M%S).log
```

Entrada única depois do refactor:

```bash
python -m experiments.ray.launch experiments/lejepa/growth_grid4.yaml
```

---

### 2. Estrutura a criar

```
experiments/
├── <method>/*.yaml        # 1 pasta por método, espelhando methods/
└── ray/
    ├── launch.py                 # entrada única: recebe caminho do YAML
    ├── schema.py                 # dataclass do experimento + validação
    ├── grid.py                   # produto cartesiano da seção `grid`
    ├── paths.py                  # resolve dataset/model config (f-strings)
    ├── skip.py                   # query_wandb_state, should_skip
    ├── gpu_slot_scheduler.py     # movido de run_ssl_ray.py:334 (4 cópias → 1)
    ├── execution_state.py        # movido de run_ssl_ray.py:301
    └── runners/
        ├── train.py              # 1 subprocess por ponto da grade
        ├── growth.py             # multi-round (spifil): treina → cresce → repete
        └── eval.py               # MLP/SVM sobre checkpoint (não passa por train.py)
```

Layout final dos YAMLs, espelhando `methods/`:

```
experiments/
├── lejepa/                          # espelha methods/lejepa
│   ├── init_ablation.yaml
│   ├── flim_full.yaml
│   └── growth_grid4.yaml
├── distillation/
│   └── conv_1x1_grid.yaml
├── autoencoder/    classification/    byol/    dino_v2/
└── ray/                             # o motor, agnóstico de método
    ├── launch.py                    # ÚNICA entrada: recebe o .yaml
    ├── schema.py  grid.py  paths.py  skip.py
    ├── gpu_slot_scheduler.py  execution_state.py
    └── runners/{train,growth,eval}.py
```

---

### 3. Decomposição em agentes — grupo R

Um agente por arquivo de `experiments/ray/`. Arquivos disjuntos por construção: nenhum par
de agentes edita o mesmo arquivo. O único acoplamento real é o schema — por isso **R1 é
declarado primeiro, sozinho**, e todo o resto vai num **único dispatch paralelo**.

| Agente | Arquivo que ele cria | Depende de | Dispatch |
|---|---|---|---|
| **R1** | `experiments/ray/schema.py` | nada | **sequencial, primeiro e sozinho** |
| **R2** | `experiments/ray/grid.py` | R1 (schema) | **onda paralela única (13 agentes)** |
| **R3** | `experiments/ray/paths.py` | R1 (schema) | mesma onda paralela |
| **R4** | `experiments/ray/skip.py` | R1 (schema) | mesma onda paralela |
| **R5** | `experiments/ray/gpu_slot_scheduler.py` | nada (move puro) | mesma onda paralela |
| **R6** | `experiments/ray/execution_state.py` | nada (move puro) | mesma onda paralela |
| **R7** | `experiments/ray/runners/train.py` | R1 (schema) | mesma onda paralela |
| **R8** | `experiments/ray/runners/growth.py` | R1 (schema) | mesma onda paralela |
| **R9** | `experiments/ray/runners/eval.py` | R1 (schema) | mesma onda paralela |
| **R11** | `experiments/ray/__init__.py` + `runners/__init__.py` | nada | mesma onda paralela |
| **E1** | `experiments/lejepa/growth_grid4.yaml` | R1 (schema) | mesma onda paralela |
| **E2** | `experiments/lejepa/init_ablation.yaml` | R1 (schema) | mesma onda paralela |
| **E3** | `experiments/distillation/conv_1x1_grid.yaml` | R1 (schema) | mesma onda paralela |
| **E4** | `experiments/lejepa/eval_mlp_freeze.yaml` | R1 (schema) | mesma onda paralela |
| **R10** | `experiments/ray/launch.py` | R1…R9, R11 | **sequencial, depois da onda** |
| **T1** | `tests/test_experiment_yaml.py` | R1 + E1…E4 | sequencial, por último |

Ordem de despacho, três mensagens no total:
`R1` → `{R2, R3, R4, R5, R6, R7, R8, R9, R11, E1, E2, E3, E4}` numa só mensagem → `R10` → `T1`.
R5, R6 e R11 não dependem do schema e poderiam sair junto com R1; ficam na onda por
simplicidade de despacho, não por dependência.

#### R1 — `experiments/ray/schema.py` (dataclass do experimento + validação)

Cobre TODO o schema abaixo. Nada de dict solto circulando pelo resto do pacote: o launcher
lê o YAML, instancia a dataclass, e é a dataclass que viaja.

Schema do experiment YAML:

```yaml
name:     str                   # identifica a corrida; prefixo dos run_names
method:   str                   # DEVE ser uma pasta existente em methods/
runner:   train | growth | eval

grid:                           # produto cartesiano; qualquer chave omitida
  init:        [flim, he, xavier, random, trunc_normal]   # usa o default do método
  datasets:    [helminth-eggs, helminth-larvae, protozoan-cysts]
  splits:      [1, 2, 3]
  percentages: [1, 5, 25, 50, 75, 100]

resources:
  gpu_ids:             [int]    # GPUs FÍSICAS. obrigatório, sem default mágico
  max_per_gpu:         int = 1
  cpus_per_experiment: int = 4
  num_workers:         int = 4
  ray_address:         str | null

overrides:                      # repassado literal ao train.py como --<k>=<v>
  trainer.max_epochs: 300
  model.init_args.lr: 3.0e-3

runner_args:                    # SÓ o que é específico do runner escolhido
  ...                           # validado contra o runner; chave desconhecida = erro

output:
  work_dir: str                 # obrigatório
  log_dir:  str = logs          # launcher grava <log_dir>/<name>_<timestamp>.log
                                # sozinho — o `tee` sai da linha de comando

wandb:
  enabled: bool = false
  project: str | null
  entity:  str | null

skip: none | state | wandb      # default: state
```

`method` é validado com `os.path.isdir("methods/<method>")` e falha imprimindo a lista de
métodos existentes. `runner` é validado contra `{train, growth, eval}`. Chave desconhecida
em `runner_args` é erro, não aviso.

#### R2 — `experiments/ray/grid.py` (produto cartesiano da seção `grid`)

Produto cartesiano de `grid`, com qualquer chave omitida caindo no default do método.
Absorve `build_experiment_grid` / `validate_experiment` / `validate_output_dir` (3 cópias
hoje). `itertools.product` — não escrever laço aninhado à mão.

#### R3 — `experiments/ray/paths.py` (resolve dataset/model config)

Resolução de caminho (as duas ÚNICAS f-strings; nada de dict à mão):

```python
  dataset_config = f"configs/dataset/{dataset}/split_{split}.yaml"
  model_config   = f"configs/model/{method}/{init}/{dataset}/split_{split}.yaml"
```

Se um desses arquivos não existir, FALHE com o caminho esperado impresso.
Não caia em config genérico — o caso especial de protozoan em
scripts/run_ssl_ray.py:113-121 é exatamente o que deve morrer.

Absorve também `_arch_json` / `_flim_weights_path` / `_split_json` / `_run_name`
(4 cópias hoje).

#### R4 — `experiments/ray/skip.py` (`query_wandb_state`, `should_skip`)

Unifica `_query_wandb_state` / `_should_skip_experiment` (3 cópias hoje) e o vocabulário
`--resume | --ignore-existing | --skip-existing | --check-wandb | --retry` no único campo
`skip: none | state | wandb`.

#### R5 — `experiments/ray/gpu_slot_scheduler.py`

MOVIDO de `scripts/run_ssl_ray.py:334` (4 cópias → 1). **REUSAR como está**:
`pick_gpu / acquire / release / has_free_slot / status_line`. Ele já é correto. Este agente
não redesenha nada — é `git mv` com ajuste de import, e nada mais.

#### R6 — `experiments/ray/execution_state.py`

Movido de `scripts/run_ssl_ray.py:301` (2 cópias → 1). Mesmo tratamento de R5: move puro.

#### R7 — `experiments/ray/runners/train.py` (1 subprocess por ponto da grade)

Comando montado pelo runner `train` (idêntico para TODO método):

```python
    cmd = [sys.executable, "train.py", "fit",
           "--config", "configs/default.yaml",
           "--config", dataset_config,
           "--config", model_config,
           f"--data.init_args.percentage={pct}",
           f"--trainer.logger.init_args.name={run_name}",
           "--trainer.devices=1",
           *[f"--{k}={v}" for k, v in overrides.items()]]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id),
           "OMP_NUM_THREADS": str(cpus_per_experiment)}
```

Passar o env via subprocess(env=...), NUNCA mutando os.environ do worker Ray —
o processo é reusado entre tasks e o valor vaza para a próxima.

Forma da task Ray (sem `num_gpus`):

```python
# experiments/ray/train_worker.py
@ray.remote                                # ← sem num_gpus
def run_train(exp: dict, gpu_id: int, cpus: int = 4) -> dict:
    import os, subprocess, sys
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": str(gpu_id),
           "OMP_NUM_THREADS":      str(cpus)}
    cmd = [sys.executable, "train.py", "fit",
           "--config", "configs/default.yaml",
           "--config", exp["dataset_config"],
           "--config", exp["model_config"],
           f"--data.init_args.percentage={exp['pct']}",
           f"--trainer.logger.init_args.name={exp['run_name']}",
           "--trainer.devices=1"]          # 1 = a única visível, que é a gpu_id
    return _run(cmd, env=env, exp=exp, gpu_id=gpu_id)
```

`--trainer.devices=1` significa "a única GPU visível", que é a `gpu_id` — nunca "a GPU 1".

#### R8 — `experiments/ray/runners/growth.py` (multi-round, spifil)

Treina → cresce → repete. Consome `runner_args`: `max_rounds`, `kappa_tolerance`,
`rounds_patience`, `embed_mode`. Cada round é uma chamada ao mesmo comando de R7; o runner
não monta comando próprio, ele reusa o de `train.py`.

#### R9 — `experiments/ray/runners/eval.py` (MLP/SVM sobre checkpoint)

Não passa por `train.py`. É o `src/evaluate/ray_mlp_queue.py` reencarnado. Consome
`runner_args`: `probe`, `freeze`, `ckpt_selection`. Mantém o `ray.init(num_gpus=0)` +
`CUDA_VISIBLE_DEVICES` por task que já está certo em `ray_mlp_queue.py:322`.

#### R10 — `experiments/ray/launch.py` (entrada única)

Recebe o caminho do YAML. Depende de R1…R9 e R11.

CLI do launcher:

```
    python -m experiments.ray.launch <caminho.yaml> [--dry-run] [--fail-fast]
                                     [--gpu-ids 0 1] [--set chave=valor ...]
```

Precedência: flag de CLI > YAML > default do schema. --dry-run imprime cada
comando montado e sai sem executar nada.

O `launch` chama o pré-voo (`experiments/ray/preflight.py`) **obrigatoriamente antes de
`ray.init()`**. Nenhum job é submetido se o pré-voo falhar.

#### R11 — `experiments/ray/__init__.py`

Arquivo de pacote, mais `experiments/ray/runners/__init__.py`. Vazios. Nenhuma camada de
re-export, nenhum `__all__`.

#### T1 — `tests/test_experiment_yaml.py`

O único check runnable da tarefa; ver critério de aceitação 5.

---

### 4. Os quatro exemplos (criar os quatro)

Quatro agentes, quatro arquivos disjuntos, **um único dispatch paralelo** (E1…E4, junto com
a onda de R2…R9/R11). Nenhum depende de outro; todos dependem só do schema de R1.

#### E1 — `experiments/lejepa/growth_grid4.yaml` — tradução exata do comando de hoje

```yaml
name:   spifil_growth_grid4
method: lejepa
runner: growth
grid:
  init:        [flim]
  datasets:    [helminth-eggs, helminth-larvae, protozoan-cysts]
  splits:      [1, 2, 3]
  percentages: [5, 50]
resources:
  gpu_ids:             [1, 2, 3]
  max_per_gpu:         3
  cpus_per_experiment: 10
  num_workers:         2
runner_args:
  max_rounds:      2
  kappa_tolerance: 0.01
  rounds_patience: 1
  embed_mode:      flatten
output:
  work_dir: artifacts/spifil_growth/grid4
wandb:
  enabled: true
  project: phd_thesis_grid4
skip: wandb
```

#### E2 — `experiments/lejepa/init_ablation.yaml` — a grade de 5 inits

```yaml
name:   lejepa_init_ablation
method: lejepa
runner: train
grid:
  init:        [flim, he, xavier, random, trunc_normal]
  datasets:    [helminth-eggs, helminth-larvae, protozoan-cysts]
  splits:      [1, 2, 3]
  percentages: [1, 5, 25, 50, 75, 100]
resources: {gpu_ids: [0, 1, 2, 3], max_per_gpu: 2, cpus_per_experiment: 8}
overrides: {trainer.max_epochs: 1000}
output:    {work_dir: artifacts/lejepa/init_ablation}
wandb:     {enabled: true, project: flim-ssl}
skip: wandb
```

#### E3 — `experiments/distillation/conv_1x1_grid.yaml`

```yaml
name:   distill_conv_1x1
method: distillation
runner: train
grid:
  init:        [flim]
  datasets:    [helminth-eggs]
  splits:      [1, 2, 3]
  percentages: [100]
runner_args: {}
overrides:
  model.init_args.proj_type:   one_layer_1x1
  model.init_args.alpha:       0.5
  model.init_args.temperature: 4.0
  model.init_args.freeze_encoder: true
resources: {gpu_ids: [2, 5], max_per_gpu: 2, cpus_per_experiment: 10}
output:    {work_dir: artifacts/distillation/conv_1x1}
wandb:     {enabled: true, project: distill_conv}
```

#### E4 — `experiments/lejepa/eval_mlp_freeze.yaml` — runner eval

```yaml
name:   mlp_freeze_lejepa
method: lejepa
runner: eval
grid:
  datasets:    [helminth-eggs, helminth-larvae, protozoan-cysts]
  splits:      [1, 2, 3]
  percentages: [1, 5, 25, 50, 75, 100]
runner_args:
  probe:  mlp
  freeze: true
  ckpt_selection: best_model_kappa_with_proj
resources: {gpu_ids: [0], max_per_gpu: 4, cpus_per_experiment: 4}
output:    {work_dir: artifacts/MLP/freeze}
skip: wandb
```

---

### 5. O que apagar quando terminar

- scripts/{run_ssl_ray,distillation_ray,distillation_conv_ray,
  autoencoder_flim_ray,classification_flim_ray,spifil_growth_loop}.py
- src/evaluate/ray_mlp_queue.py  (vira experiments/ray/runners/eval.py)
- Os blocos `if __name__ == "__main__"` com argparse dentro de src/modules/*.py
- Os helpers argparse dentro de src/models/distillation.py:279-497
  (add_student_flags, resolve_student, add_distill_flags, resolve_distill_flags,
   derive_flim_paths, distill_run_tags) — as flags viram init_args do módulo
- O `from constants import ...` com hack de sys.path (existe só porque os
  launchers rodam de dentro de scripts/)

Deleção é a última onda, depois de T1 passar. Nada é apagado antes de o launcher novo rodar
os quatro YAMLs em `--dry-run`.

---

### 6. Restrições

- NÃO criar sistema de plugin/registry para descobrir métodos. `method:` é o
  nome da pasta em methods/; validar com os.path.isdir e falhar com a lista.
- NÃO adicionar dependência nova. ray, pyyaml e lightning já estão no projeto.
- NÃO usar ray.tune. O GpuSlotScheduler existente já resolve, e tune tiraria o
  controle de qual GPU física é usada.
- REUSAR o GpuSlotScheduler de scripts/run_ssl_ray.py:334 como está
  (pick_gpu / acquire / release / has_free_slot / status_line). Ele já é correto.
- Um runner novo = um arquivo em runners/. Nada mais deve mudar.

---

### 7. Critérios de aceitação

1. `python -m experiments.ray.launch experiments/lejepa/growth_grid4.yaml --dry-run`
   imprime 18 comandos (3 datasets × 3 splits × 2 percentages) e sai com 0.
2. Cada comando impresso tem exatamente 3 `--config` e nenhuma flag de argparse
   específica de método.
3. Trocar `method: lejepa` por `method: distillation` num YAML muda apenas o
   caminho do model config — zero mudança em experiments/ray/.
4. `grep -rn "num_gpus=1" experiments/` não retorna nada.
5. Um test_experiment_yaml.py que, para cada .yaml em experiments/**/:
   valida contra o schema, confere que methods/<method>/ existe, e que todo
   (init, dataset, split) da grade tem o configs/model/... correspondente
   no disco. Falha listando os caminhos faltantes.

---

### 8. Expansões

#### Problemas previstos e soluções

**Divergência entre `grid:` (esta tarefa) e `axes:` (a tarefa do pré-voo).**
O documento define os dois formatos para a mesma coisa: `grid` tem quatro chaves fixas
(`init`, `datasets`, `splits`, `percentages`); `axes` é dict aberto, com `path` (templates
`.format(**ponto, method=method)`), `map` (eixo → chave de override do LightningCLI) e
`exclude` (buracos declarados). Duas leituras do mesmo YAML é o começo de um terceiro
launcher. Recomendação: **`axes` é o formato canônico e o schema aceita `grid:` como açúcar
sintático traduzido para `axes` na leitura** (`datasets`→`dataset`, `splits`→`split`,
`percentages`→`percentage`, `path` e `map` preenchidos com os defaults das duas f-strings de
R3). Motivo: `axes` já cobre eixos específicos de método (`alpha`) e buracos de grade, coisas
que `grid` não expressa e que a ablação de distillation vai pedir; e a tradução é umas dez
linhas em `schema.py`, contra reescrever os quatro exemplos e todo o texto do pré-voo. A
alternativa honesta — reescrever E1…E4 em `axes` e apagar `grid` do schema — é mais limpa a
longo prazo e mais barata se os quatro YAMLs forem os únicos que existem hoje. **Decisão
final do usuário**; até ela sair, escreva o schema com um só caminho de leitura, nunca dois.

**Retomada de corrida interrompida.**
`skip: none | state | wandb`. `state` lê `execution_state.py` (arquivo local no `work_dir`):
barato, instantâneo, funciona offline e sem token de W&B. `wandb` consulta o servidor: caro
(uma query por ponto da grade), lento, mas correto entre máquinas — é o único modo válido
quando a mesma grade foi rodada em outro host. Recomendação: **`state` como default** (é o
default do schema) e **`wandb` para grades longas** ou compartilhadas, como nos exemplos E1,
E2 e E4. Modos de falha, ambos silenciosos: `state` mente se o `work_dir` for apagado ou
movido (ele reexecuta tudo, caro mas seguro); `wandb` mente se o run crashou depois de logar
as primeiras métricas (ele pula um experimento que nunca terminou, e isso é perda de dado).
Com `skip: wandb`, exigir estado terminal do run — não a mera existência dele.

**Worker Ray reusado vazando estado.**
O `env` do subprocess resolve `CUDA_VISIBLE_DEVICES` e `OMP_NUM_THREADS`, mas não resolve o
que já foi importado no processo. Regra adicional: **proibir qualquer `import torch` no
processo do launcher e nos workers Ray** — importar torch e tocar CUDA cria um contexto que
segura centenas de MB na GPU 0 pelo tempo inteiro da corrida, em cada worker, e some do
`nvidia-smi` como "processo do launcher". Solução: todo uso de torch acontece dentro do
subprocess `train.py`; o worker Ray só monta lista de strings e chama `subprocess.run`. Um
`grep -n "import torch" experiments/` sem resultado é o teste.

**Um job trava e segura o slot.**
Um `train.py` que trava num deadlock de dataloader segura o slot de GPU para sempre e a
grade nunca termina. Solução: **timeout por job no runner** (`subprocess.run(..., timeout=T)`),
`T` configurável em `resources` (por exemplo `job_timeout_s`), com o slot devolvido no
`finally` — `release` nunca pode estar no caminho feliz. O job estourado é marcado como falha
no `ExecutionState`, não como concluído, para que a retomada o pegue de novo. Marcar com
`# ponytail: timeout fixo por job; por-runner só se um runner legítimo passar do teto.`

**Log por corrida.**
O `tee` sai da linha de comando: o launcher grava `<log_dir>/<name>_<timestamp>.log` sozinho,
e cada job grava `<work_dir>/<run_name>.log`. Isso resolve o stdout entrelaçado de N workers,
que é ilegível por construção quando 9 subprocessos escrevem no mesmo descritor. Solução:
**cada job escreve no SEU arquivo** (stdout e stderr redirecionados para ele no
`subprocess.run`), e o log do launcher recebe **só as linhas de estado**: submetido,
terminou, falhou, pulado — uma linha por evento, com `run_name`, `gpu_id` e duração. Unifica
os sete `_log(msg, level)` de hoje num só.

**`--dry-run` como contrato de teste.**
`--dry-run` não é conveniência de debug, é a superfície de teste da tarefa. Os critérios de
aceitação 1 e 2 são exatamente `--dry-run` + `grep -c -- --config`: sem mock, sem framework,
sem fixture, sem GPU. Solução: o runner monta o comando numa função pura
(`build_cmd(exp, point) -> list[str]`) que `--dry-run` imprime e o caminho normal executa —
nunca dois caminhos de montagem, senão o dry-run testa código que não roda em produção.
O check runnable de T1 é isso mais a checagem de disco do critério 5.

---

## TAREFA — escrever o pré-voo do experiment YAML (multiagente)

**Restrição do ambiente, herdada do original:** *"vai ter alguns experimentos rodando nesse
momento, não mexa neles."* O pré-voo é read-only sobre o disco: ele só lê YAML, faz `stat` de
caminhos e importa módulos. Não escreve, não move, não mata processo, não toca em `work_dir`.

---

### Objetivo

Criar **UM** arquivo: `experiments/ray/preflight.py`.

Ele expande a grade do experiment YAML e confere que TUDO existe em disco.
É chamado obrigatoriamente por `experiments/ray/launch.py` **ANTES** de `ray.init()`.
Nenhum job é submetido se o pré-voo falhar.

**Requisito de saída (do usuário, literal):** *"ele imprime qual foi arquivo ou file que tá
falhando e quais tão faltando."* O relatório tem as duas metades e nenhuma delas é opcional:
ele **nomeia o arquivo culpado** (o YAML de model/dataset que quebrou, com a checagem que ele
violou) **e lista os que faltam** (um caminho esperado por linha, todos, sem `...` e sem
truncar). Um relatório que diz "24 configs faltando" sem imprimir os 24 caminhos não cumpre a
tarefa.

---

### Schema que ele valida

```yaml
name:   str                      # obrigatório
method: str                      # obrigatório; deve ser pasta em methods/
runner: train | growth | eval    # obrigatório

axes:                            # dict ABERTO — toda chave é eixo do produto
  init:       [flim, he]         # cartesiano. Valores devem ser listas não-vazias.
  dataset:    [helminth-eggs]
  split:      [1, 2, 3]
  percentage: [5, 50]
  alpha:      [0.3, 0.5]         # eixos específicos do método são livres

path:                            # templates .format(**ponto, method=method)
  dataset: "configs/dataset/{dataset}/split_{split}.yaml"
  model:   "configs/model/{method}/{init}/{dataset}/split_{split}.yaml"

map:                             # eixo → chave de override do LightningCLI
  percentage: data.init_args.percentage
  alpha:      model.init_args.alpha

exclude:                         # buracos declarados da grade
  - {init: he, dataset: helminth-larvae}

resources: {gpu_ids: [...], ...}
output:    {work_dir: ...}
```

---

### Funções

```python
def expand(exp: dict) -> list[dict]
    Produto cartesiano de `axes`, removendo os pontos que casam com algum
    dict de `exclude` (casa se TODOS os pares do dict batem com o ponto).

def resolve(exp: dict, point: dict) -> dict
    Devolve o ponto + "dataset_config", "model_config" (templates de `path`
    formatados com **point e method=exp["method"]) e "overrides"
    ({flag: point[eixo] for eixo, flag in exp["map"].items()}).

def preflight(exp_path: str, root: str = ".") -> list[dict]
    Roda TODAS as checagens, agrega os erros, e:
      - se houver erro: imprime o relatório agrupado e levanta SystemExit(1)
      - se não: devolve a lista de pontos resolvidos, pronta pro launcher
```

---

### Decomposição em agentes — grupo P

Cada checagem é uma **função pura e independente**: recebe `exp` (e, da 12 em diante, a lista
de pontos resolvidos) e devolve uma lista de erros. Ninguém escreve no mesmo lugar que ninguém.
Por isso a partição é **uma checagem — ou um bloco coeso de checagens — por agente**, e P1..P7
mais P9 saem **num único dispatch paralelo**. Só P8 é sequencial, porque consome a saída de
todos.

| Agente | Escopo | Depende de | Dispatch |
|---|---|---|---|
| **P1** | `expand()` — cartesiano + `exclude` | nada | **paralelo (dispatch único)** |
| **P2** | `resolve()` — configs + overrides | nada (contrato de `expand` é o dict do ponto) | **paralelo (dispatch único)** |
| **P3** | Checagens de **estrutura** 1–5 | nada | **paralelo (dispatch único)** |
| **P4** | Checagens de **coerência da grade** 6–11 | nada | **paralelo (dispatch único)** |
| **P5** | Checagens de **disco** 12–13 | contrato de `resolve` | **paralelo (dispatch único)** |
| **P6** | Checagens de **disco** 14–15 | contrato de `resolve` | **paralelo (dispatch único)** |
| **P7** | Checagem de **disco** 16 | contrato de `resolve` | **paralelo (dispatch único)** |
| **P9** | Fixtures de teste (`test_experiment_yaml.py`) | nada — parte dos 3 bugs reais, que já estão em disco | **paralelo (dispatch único)** |
| **P8** | Agregador + relatório + `SystemExit(1)` + `__main__` | **todos** (P1–P7) | sequencial, depois do dispatch |

P1..P7 e P9 são funções/arquivos disjuntos por construção: nenhum agente edita a região de
outro, nenhum agente espera um irmão. Se a partição colocar dois agentes na mesma função, isso
é bug da partição, não motivo para serializar.

#### P1 — `expand()`

Produto cartesiano de `axes`, removendo os pontos que casam com algum dict de `exclude`. A
regra de casamento é a do original e não admite atalho: **casa se TODOS os pares do dict batem
com o ponto**. `{init: he, dataset: helminth-larvae}` remove só os pontos que são `he` **e**
`helminth-larvae` — não remove todo `he`, não remove toda `helminth-larvae`. Um `exclude` com
uma chave só é legítimo e remove um plano inteiro da grade.

#### P2 — `resolve()`

Devolve o ponto acrescido de três coisas:
- `"dataset_config"` e `"model_config"`: os templates de `path` formatados com `**point` e
  `method=exp["method"]`;
- `"overrides"`: `{flag: point[eixo] for eixo, flag in exp["map"].items()}`.

O ponto original permanece intacto no dict devolvido — a checagem 10 e o relatório dependem de
saber de qual ponto da grade cada caminho veio.

#### P3 — Checagens de ESTRUTURA (1–5)

```
  1. Campos obrigatórios presentes: name, method, runner, path, axes.
  2. runner ∈ {train, growth, eval}.
  3. methods/<method>/ existe e é diretório. Se não, listar os que existem.
  4. configs/default.yaml existe.
  5. resources.gpu_ids presente e não-vazio.  output.work_dir presente.
```

Nota de execução para a 3: o "listar os que existem" é parte da checagem, não enfeite — o erro
imprime o `method` pedido e o `ls methods/` ao lado, porque o caso real é typo.

#### P4 — Checagens de COERÊNCIA DA GRADE (6–11)

```
  6. Todo valor de `axes` é lista não-vazia.
  7. Todo placeholder dos templates de `path` é um eixo de `axes` ou a
     palavra `method`. Placeholder órfão = erro, citando o template.
  8. Toda chave de `map` é um eixo de `axes`.
  9. Toda chave de `exclude` é um eixo de `axes`.
 10. Eixo morto: eixo que não aparece em nenhum template de `path` nem em
     `map`. É erro — multiplica a grade sem mudar nada do que roda.
 11. Grade vazia após `exclude` = erro.
```

Os placeholders da 7 saem do próprio template com `string.Formatter().parse(tpl)` — não com
regex caseira. A 10 tem uma ressalva de falso positivo tratada adiante em
*"Falso positivo do eixo morto"*, e a implementação dela **tem** que incorporar essa ressalva.

#### P5 — Checagens de DISCO 12/13

```
 12. Cada dataset_config resolvido existe.
 13. Cada model_config resolvido existe.
```

Cada erro carrega o caminho **esperado** e o ponto da grade que o gerou. É a checagem que
produz a lista de 24 do fixture, e é ela que precisa imprimir os 24 caminhos, um por linha.

#### P6 — Checagens de DISCO 14/15

```
 14. Cada model_config (deduplicado) é YAML parseável e tem model.class_path.
 15. Cada class_path é importável: importlib.import_module(módulo) e
     hasattr(mod, classe). Reportar módulo E classe no erro.
```

A 14 é pré-requisito da 15 no mesmo arquivo: YAML que não parseia, ou sem `model.class_path`,
não gera também um erro de import — gera **um** erro, o da 14, e a 15 pula aquele arquivo e diz
que pulou. Custo e mitigação da 15 estão em *"Checagem 15 é a mais cara"*.

#### P7 — Checagem de DISCO 16

```
 16. Nos init_args de cada model_config, as chaves arch_json e
     flim_weights_path: o caminho deve existir em disco E ser relativo à raiz
     do repo. Caminho absoluto é erro separado de caminho inexistente.
```

"Erro separado" é literal: são duas mensagens distintas, contadas separadamente no resumo.
`/dados/home/moliveira/...` que por acaso existe na máquina atual continua sendo erro — o
problema é ser absoluto, não ser inexistente.

#### P8 — O agregador e o relatório

Roda **TODAS** as checagens, **SEMPRE**; nunca para no primeiro erro. Agrupa por categoria
(estrutura / grade / disco), imprime o relatório no formato definido adiante, e levanta
`SystemExit(1)`. Se não houver erro, devolve a lista de pontos resolvidos, pronta pro launcher.
É também quem expõe o `__main__` do comando isolado.

#### P9 — As fixtures de teste

Escreve `test_experiment_yaml.py`, que chama `python -m experiments.ray.preflight` (ou a função
`preflight()` direto) e **usa os três bugs reais do repo como casos** — eles já estão em disco,
não é preciso inventar fixture sintético. Um caso por bug, cada um afirmando a checagem
correspondente (15, 16, 13) e afirmando que o relatório contém o caminho culpado. Sem
framework, sem fixtures elaboradas: `assert` e pronto. O teste roda sem Ray e sem GPU.

---

### Bugs reais do repo que ele TEM que pegar (usar como fixture)

```
 - configs/model/lejepa_custom_cnn.yaml aponta para src.modules.LeJEPACNNModule,
   que não existe em nenhum arquivo do repo → checagem 15.
 - configs/model/lejepa_line_he_protozoan_train1.yaml tem
   arch_json: /dados/home/moliveira/scalable_FLIM_self_supervised/...
   (absoluto, só roda numa máquina) → checagem 16.
 - model/lejepa/{he,xavier,random,trunc_normal}/{helminth-eggs,helminth-larvae}/
   não existem hoje (são o caso especial de protozoan em run_ssl_ray.py:113)
   → checagem 13, e o relatório deve listar os 24.
```

Os 24 são `4 init × 2 dataset × 3 split`. Se o relatório imprimir "24 faltando" sem os 24
caminhos, o fixture falha — é exatamente o requisito do usuário.

---

### Expansões — o que este documento acrescenta ao original

#### Formato do relatório e soluções

**Formato do relatório de erro.** Agrupado por categoria e, dentro dela, por checagem. Cada
bloco de checagem traz: o número e o nome da checagem, a contagem entre parênteses, e os
caminhos **esperados** impressos **um por linha**, sem elisão. Onde o erro é sobre um arquivo
que existe mas está errado, a linha nomeia o arquivo culpado e a chave dentro dele. Sempre uma
saída só — todas as checagens numa passada — e uma linha de resumo no fim.

Exemplo renderizado, com os três bugs reais na mesma corrida:

```
preflight: experiments/ray/exp_lejepa_growth.yaml
  method=lejepa  runner=growth  grade=24 pontos

ESTRUTURA .................................................. ok (5/5)
GRADE ...................................................... ok (6/6)
DISCO ...................................................... 26 erros

  [13] model_config resolvido não existe (24)
    configs/model/lejepa/he/helminth-eggs/split_1.yaml
    configs/model/lejepa/he/helminth-eggs/split_2.yaml
    configs/model/lejepa/he/helminth-eggs/split_3.yaml
    configs/model/lejepa/he/helminth-larvae/split_1.yaml
    configs/model/lejepa/he/helminth-larvae/split_2.yaml
    configs/model/lejepa/he/helminth-larvae/split_3.yaml
    configs/model/lejepa/xavier/helminth-eggs/split_1.yaml
    configs/model/lejepa/xavier/helminth-eggs/split_2.yaml
    configs/model/lejepa/xavier/helminth-eggs/split_3.yaml
    configs/model/lejepa/xavier/helminth-larvae/split_1.yaml
    configs/model/lejepa/xavier/helminth-larvae/split_2.yaml
    configs/model/lejepa/xavier/helminth-larvae/split_3.yaml
    configs/model/lejepa/random/helminth-eggs/split_1.yaml
    configs/model/lejepa/random/helminth-eggs/split_2.yaml
    configs/model/lejepa/random/helminth-eggs/split_3.yaml
    configs/model/lejepa/random/helminth-larvae/split_1.yaml
    configs/model/lejepa/random/helminth-larvae/split_2.yaml
    configs/model/lejepa/random/helminth-larvae/split_3.yaml
    configs/model/lejepa/trunc_normal/helminth-eggs/split_1.yaml
    configs/model/lejepa/trunc_normal/helminth-eggs/split_2.yaml
    configs/model/lejepa/trunc_normal/helminth-eggs/split_3.yaml
    configs/model/lejepa/trunc_normal/helminth-larvae/split_1.yaml
    configs/model/lejepa/trunc_normal/helminth-larvae/split_2.yaml
    configs/model/lejepa/trunc_normal/helminth-larvae/split_3.yaml
    template: configs/model/{method}/{init}/{dataset}/split_{split}.yaml
    (existe hoje só o caso especial de protozoan — run_ssl_ray.py:113)

  [15] class_path não importável (1)
    arquivo:  configs/model/lejepa_custom_cnn.yaml
    class_path: src.modules.LeJEPACNNModule
    módulo src.modules importou, mas não tem o atributo LeJEPACNNModule

  [16] caminho absoluto em init_args (1)
    arquivo: configs/model/lejepa_line_he_protozoan_train1.yaml
    chave:   arch_json
    valor:   /dados/home/moliveira/scalable_FLIM_self_supervised/...
    deve ser relativo à raiz do repo

preflight: 24 pontos, 26 erros em 3 checagens
```

O resumo final é sempre `preflight: N pontos, M erros em K checagens` — `N` é o tamanho da
grade após `exclude`, `M` o total de erros, `K` quantas checagens distintas produziram erro.
Sem erro, a mesma linha sai com `0 erros` e o processo segue.

**Checagem 15 é a mais cara** — é a única que importa módulos de verdade, e importar módulo de
modelo puxa a árvore inteira de dependências. Três soluções, e a recomendação é **usar as três
juntas**:

- (a) **deduplicar por `class_path` antes de importar** — a grade de 24 pontos costuma apontar
  para 1 ou 2 classes; sem dedup são 24 imports do mesmo módulo;
- (b) **importar só o MÓDULO e usar `hasattr`**, sem instanciar — instanciar exigiria pesos e
  GPU, que é justamente o que o pré-voo não pode depender de ter;
- (c) **cachear o resultado por `class_path` dentro da corrida** — um dict `class_path → ok/erro`
  vale para o resto da execução, inclusive entre checagens.

**Risco registrado:** import com efeito colateral — um módulo que aloca CUDA, inicializa
contexto, ou reserva memória já no `import`. Isso transformaria o pré-voo, que deveria ser
barato, em algo que compete com os experimentos rodando. **Mitigação:** rodar o pré-voo
**antes de qualquer `import torch` do launcher**, com o `preflight()` chamado no topo de
`launch.py`, antes de `ray.init()` e antes dos imports pesados. Se ainda assim um módulo alocar
CUDA no import, isso é achado do pré-voo e vira issue — não vira exceção na checagem.

**Falso positivo do "eixo morto" (checagem 10).** Um eixo pode ser legitimamente consumido pelo
`runner_args` em vez de aparecer em `path` ou em `map` — nesse caso ele muda o que roda, e
marcá-lo como morto seria erro do pré-voo, não do YAML. **Solução:** a checagem 10 considera
também as chaves referenciadas por `runner_args`; o eixo só é morto se não aparecer em nenhum
template de `path`, **nem** em `map`, **nem** em `runner_args`. Se depois disso ele ainda não
for usado por ninguém, é erro — ele multiplica a grade sem mudar nada do que roda. Isso está
dito explicitamente para a checagem 10 não virar uma armadilha que obriga o usuário a inventar
um `map` falso só para calar o pré-voo.

**Pré-voo como comando isolado.** `python -m experiments.ray.preflight <caminho.yaml>` roda
sozinho, **sem Ray e sem GPU**, e é exatamente o que `test_experiment_yaml.py` chama. Uma
implementação, dois consumidores: o teste e o launcher. Não existe cópia da lógica no teste,
não existe versão "leve" separada — o que o launcher executa é o que o teste executa.

**Ordem de execução para não afogar o usuário.** Estrutura (1–5) primeiro: se `method` não
existe ou `axes` está ausente, não adianta expandir a grade — todo caminho resolvido sairia
lixo e o relatório viraria centenas de erros derivados de um só. **Mas o pré-voo ainda reporta
TUDO que conseguir avaliar** com o que tem, e diz explicitamente **quais checagens não puderam
rodar por causa de um erro anterior**, nomeando o erro que as bloqueou:

```
ESTRUTURA .................................................. 1 erro
  [3] methods/lejepaa/ não existe ou não é diretório
      existem: methods/flim/  methods/lejepa/  methods/simclr/

GRADE ...................................................... ok (6/6)
DISCO ...................................................... não avaliado
  checagens 12–16 puladas: dependem de [3]

preflight: 24 pontos, 1 erro em 1 checagem, 5 checagens puladas
```

Regra dura: **nunca imprimir "0 erros" quando alguma checagem foi pulada.** Ou o pré-voo
avaliou tudo e passou, ou ele diz o que ficou de fora. Um "ok" que esconde cinco checagens não
executadas é pior que o erro original, porque manda o usuário submeter os jobs.

---

## Grupo E — `eval/`, `experiments/`, `analysis/` (dispatch paralelo, três agentes-líderes)

Os três destinos são disjuntos por construção: nenhum agente deste grupo escreve num arquivo
que outro agente do grupo também escreve. Logo, **um único dispatch paralelo com três
agentes-líderes** (E-EVAL, E-EXP, E-ANA) mais o agente de movimentação pura (E-MISC), que não
depende de nenhum deles. Nenhum agente do Grupo E espera outro.

Toda afirmação de qualquer relatório carrega `arquivo:linha`. Sem âncora é palpite, e palpite
vai numa seção separada, marcada como tal.

### E-EVAL — `eval/`

Destino: `eval/` (9,1k linhas). **É probe, não treino.** Tem `__main__.py` e **MANTÉM** o
`__main__.py` — é o único lugar da árvore nova onde um `__main__` de módulo é esperado.

Arquivos sob responsabilidade deste agente:

| Arquivo destino | Origem |
|---|---|
| `svm.py`, `mlp.py`, `knn.py`, `classical_classifiers.py` | probes atuais de `src/evaluate/` |
| `ray_queue.py` | funde `ray_mlp.py` + `ray_mlp_queue.py` (a parte que **não** é ray) |
| `unified_eval.py`, `wandb_resolver.py`, `eval_plotter.py` | `src/evaluate/` |
| `growth_stages.py`, `tsne.py` | `src/evaluate/` |

**Nota crítica — leia antes de mover qualquer coisa.**
`src/evaluate/ray_mlp_queue.py` **NÃO fica em `eval/`.** Ele vira
`experiments/ray/runners/eval.py`, porque não é treino nem probe: é **orquestração**. Está na
lista de deleção do refactor exatamente com esse destino.

**Conflito aparente, e sua resolução.** A árvore nova tem `eval/ray_queue.py` e
`experiments/ray/runners/eval.py` ao mesmo tempo, e os dois nascem do mesmo par de arquivos.
Não é duplicação; é uma separação por responsabilidade:

- `eval/ray_queue.py` guarda a **lógica de probe** — o que treina o MLP/SVM sobre embeddings e
  devolve os números. Roda **in-process**, chamável direto de um teste ou de um notebook, sem
  cluster.
- `experiments/ray/runners/eval.py` guarda a **orquestração** — monta comandos, pede slot de
  GPU ao `GpuSlotScheduler`, submete jobs, coleta estado. Nunca calcula uma métrica.

**Critério de corte, mecânico e não negociável: se o arquivo importa `ray`, é orquestração e
vai para `experiments/`.** Se não importa, é probe e fica em `eval/`. O agente aplica esse
critério linha a linha na partição de `ray_mlp.py` / `ray_mlp_queue.py` e reporta a fronteira
exata (quais funções foram para cada lado, com `arquivo:linha` de origem).

Verificação que este agente deixa: `grep -rn "^import ray\|^from ray" eval/` não retorna nada.

### E-EXP — `experiments/`

Destino: `experiments/` (13,7k linhas vindas do antigo `scripts/`). **Orquestração, não
biblioteca** — nada em `core/`, `flim/`, `methods/` ou `eval/` pode importar de `experiments/`.

| Subpasta / arquivo | Conteúdo |
|---|---|
| `ray/` | os `*_ray.py` launchers |
| `ckpt/` | `strip_teacher*`, `validate_stripped`, `verify_finetune` |
| `gen_configs.py` | funde `generate_mlp_configs.py` + `download_*` + `normalize_reports` |
| `oneoff/` | `retry_*`, `run_missing_*`, `check_*_status` — descartáveis |

**Problema.** `oneoff/` é lixo com data de validade. Scripts escritos para uma corrida
específica, que ninguém vai rodar de novo, e que apodrecem virando dependência acidental do
resto do repositório assim que alguém os importa "só pra reaproveitar aquela função".

**Solução (a que este agente executa).** Mover para `experiments/oneoff/` com um `README.md` de
**uma linha**:

> scripts descartáveis; nada em `experiments/` ou `methods/` pode importar daqui.

E um **agente verificador** que roda `grep -rn "oneoff" --include=*.py` sobre a árvore nova e
**falha** se qualquer arquivo fora de `experiments/oneoff/` importar de lá. Essa verificação
entra no pré-voo como regra permanente, não como checagem de uma vez só.

**Segunda opção, mais barata.** Apagar os que não rodam há N meses — checar `git log` por
arquivo para a data do último commit e do último uso. **Mediante aprovação explícita do
usuário**, e passando pelo portão de deleção descrito adiante. O agente traz a lista com datas;
não apaga por conta própria.

### E-ANA — `analysis/`

Destino: `analysis/`, que funde `tools/` + `statistics/` + `analysis_flim_distill/` +
`check_experiments/`.

| Subpasta | Conteúdo |
|---|---|
| `stats/` | `wilcoxon_{acc,f1,kappa,equivalence,flim_init}`, `compute_cost` |
| `activations/` | `relu_vs_sigmoid`, `saturation`, `unit_activations`, `heatmap_stages` |
| `distill/` | `embedding_analysis`, `ruler_mismatch`, `distill_destroys_flim` |
| `plots/` | todos os `plot_*.py` + `src/utils/plot_svm_results.py` |

**Problema.** Os scripts de análise são **os que mais usam argparse** no repositório inteiro.
São também os que menos se encaixam no LightningCLI: não treinam nada, não têm `model:` nem
`data:`, e forçá-los no LightningCLI seria construir um framework para não usar outro.

**Solução.** Análise **não passa pelo LightningCLI**. Para esses arquivos, a substituição do
argparse é:

1. Uma função com **defaults nomeados** na assinatura (`def wilcoxon_acc(runs_dir="reports/A_reports", alpha=0.05, out="reports/wilcoxon_acc.md"):`), mais
2. Um `if __name__ == "__main__":` de **três linhas** que chama essa função com os defaults.

Se a lista de parâmetros passar de **~5**, aí sim um YAML de análise, lido com `yaml.safe_load`
e desempacotado na mesma função. Nada além disso.

**Restrição explícita: não criar um segundo framework de CLI.** Sem parser próprio, sem
registry de comandos, sem decorator `@analysis`, sem classe base de análise. A regra
"argparse é proibido" não é licença para escrever o substituto do argparse.

### E-MISC — `data/`, `reports/`, `env/`

**Um agente, movimentação pura, `git mv`.** Nenhum conteúdo de arquivo é editado; nenhum import
é reescrito neste agente (imports quebrados por estes movimentos são reportados, não
consertados aqui).

- `data/` — splits JSON + `to_mateus/` (pesos e `arch_json` FLIM).
- `reports/` — `A_reports/` + os `.md` soltos da raiz + `metrics_distillation/`.
- `env/` — `Dockerfile`, `environment.yml`, `requirements.txt`, `setup_env.sh`.

Usar `git mv` e não `mv` + `git add`: as tabelas (a) e (b) do `MIGRATION.md` são geradas a
partir de `git log --diff-filter=R`, e um move que o git não registrou como rename não aparece
lá. Este agente entrega a lista de renames que produziu.

---

## Grupo D — entregáveis de documentação (paralelo entre si; dependem do refactor estar verde)

Os três são disjuntos (arquivos diferentes) e rodam **num único dispatch paralelo**. Todos
dependem, em sequência, do refactor ter passado no pré-voo — documentar uma árvore que ainda vai
mudar produz documentação errada no dia seguinte.

### D1 — `instructions.md` (EM INGLÊS)

Pedido literal do usuário: *"crie um instructions.md explicando em english como navegar por esse
repositorio, escrito com o objetivo de ajudar o claude"*. Portanto: **o arquivo é escrito em
inglês**, na raiz do repositório, e o leitor-alvo é um agente, não uma pessoa.

Conteúdo obrigatório, nesta ordem:

1. **Os dois entrypoints** — `train.py` (LightningCLI; treina um modelo) e `run_experiments.py`
   (grid runner; chama `train.py`). Uma frase para cada dizendo **quando usar qual**.
2. **O mapa de pastas, uma linha cada**: `core`, `flim`, `methods`, `eval`, `experiments`,
   `analysis`, `configs`, `data`, `reports`, `env`.
3. **As DUAS f-strings de resolução de config**, verbatim:
   `f"configs/dataset/{dataset}/split_{split}.yaml"` e
   `f"configs/model/{method}/{init}/{dataset}/split_{split}.yaml"`.
   Com a regra: se o arquivo não existe, falha imprimindo o caminho esperado; nunca cai em
   config genérico.
4. **Como rodar um experimento do zero**: escrever o experiment YAML em `experiments/<method>/`
   → `python -m experiments.ray.launch <caminho.yaml> --dry-run` → conferir os comandos →
   `launch` sem a flag.
5. **A regra "argparse é proibido; configuração vive em YAML."** Com a exceção registrada de
   `analysis/` (defaults nomeados + `__main__` de 3 linhas).
6. **A regra "1 arquivo = 1 classe pública; `__init__.py` re-exporta; `class_path` usa a forma
   curta."**
7. **Onde procurar quando algo não existe mais** → `MIGRATION.md` (D2), com o caminho exato.

**Estilo, e isto é requisito, não gosto:** escrito para um agente ler, não para um humano
navegar. Frases curtas. Caminhos exatos, nunca "the model folder". Zero prosa motivacional,
zero "welcome to the repository", zero explicação de o que é SSL. Se uma frase não muda o que o
agente faz na próxima ação, ela sai.

### D2 — `MIGRATION.md`, o arquivo de referência de migração

Pedido literal do usuário: *"voce deve ter um arquivo de referencia dizendo o que mudou para que
quando eu quiser executar e voce se perder, voce check o passado para me dizer como era o
passado"*.

Local: **`MIGRATION.md` na raiz**. Três tabelas, nesta ordem:

- **(a) arquivo antigo → arquivo novo.** Toda linha do refactor que moveu ou apagou um arquivo.
- **(b) `class_path` antigo → `class_path` novo.** Ex.: `src.modules.LeJEPAFLIMModule` →
  a forma curta sob `methods/`. É a tabela que o agente consulta quando um YAML velho não
  instancia.
- **(c) comando antigo → comando novo.** À esquerda, a linha de comando gigante com flags
  argparse; à direita, o experiment YAML equivalente. **Com pelo menos o caso do
  `spifil_growth_loop.py` traduzido** — o comando de hoje (`--work-dir artifacts/spifil_growth/grid4 --percentages 5 50 --gpus 1 2 3 --max-concurrent-per-gpu 3 --cpus-per-experiment 10 --max-rounds 2 --num-workers 2 --wandb --wandb-project phd_thesis_grid4`) virando o YAML de grade correspondente, campo a campo. É esse o caso que prova que a tabela serve para alguma coisa.

**Regra permanente:** nenhum `git rm` / `git mv` do refactor entra sem uma linha em
`MIGRATION.md`. Um move sem linha na tabela é um move não feito.

**Solução para manter o arquivo honesto:** gerar as tabelas (a) e (b) a partir de
`git log --diff-filter=R --name-status`, **não à mão**. Tabela escrita à mão desatualiza no
primeiro rename que alguém esquece; tabela derivada do log não tem como mentir. A tabela (c) é
manual por natureza — são poucos comandos, e a tradução exige julgamento.

### D3 — atualizar o `INDEX.md` deste repositório de prompts

Escopo diferente dos outros dois: este agente mexe **neste repositório de prompts**, não no
repositório de código. Para cada `.md` novo criado por este refactor, uma linha em `INDEX.md`,
no formato exato:

    - [caminho](caminho) — descrição curta

na seção do diretório correspondente. Sem reordenar as linhas existentes, sem reescrever
descrições que já estão lá.

---

## Grupo B — os dois bugs (paralelo entre si; sequenciais só em relação ao pré-voo que os detecta)

B1 e B2 tocam arquivos diferentes e rodam **em paralelo, num dispatch só**. A única dependência
é para trás: os dois são achados do pré-voo (checagens 15 e 16), então o pré-voo existe antes
deles. Depois disso, nenhum dos dois espera o outro.

### B1 — `LeJEPACNNModule` inexistente

`configs/model/lejepa_custom_cnn.yaml` aponta para `src.modules.LeJEPACNNModule`, que **não
existe** — nem em `src/modules/__init__.py`, nem em arquivo nenhum do repositório. A única
ocorrência do nome está numa docstring. O config quebra na instanciação, e é exatamente o que a
checagem de `class_path` importável pega (`test_class_paths.py` / checagem 15 do pré-voo).

**Consertar.** Três soluções possíveis, com recomendação:

- **(a) O config aponta para a classe certa sob a nova árvore** (`methods.lejepa.LeJEPACNNModel`
  ou o module equivalente). **Recomendada** — se a docstring provar qual era a intenção
  original. É a correção de menor diff e não inventa código.
- **(b) Escrever de fato o `LeJEPACNNModule`**, espelhando `LeJEPAFLIMModule`. Só se o
  experimento for necessário, e o usuário confirmar que é.
- **(c) Apagar o config e o experimento junto** — se nada nem ninguém o referencia. Checar o
  W&B por runs que o usaram antes de propor isto; um run histórico que usou o config é motivo
  para preservá-lo.

**O agente traz a evidência ANTES de escolher** (a docstring com `arquivo:linha`, o resultado do
grep pelo nome em todo o repositório, o resultado da busca no W&B) e **não inventa classe**.
Escrever uma classe nova para satisfazer um config quebrado, sem saber se o experimento existe,
é a pior das três saídas.

### B2 — caminho absoluto em config

`configs/model/lejepa_line_he_protozoan_train1.yaml` tem
`arch_json: /dados/home/moliveira/scalable_FLIM_self_supervised/...` — absoluto, roda numa
máquina só.

Conserto, em três passos:

1. **Relativizar à raiz do repositório** o caminho desse arquivo.
2. **Varrer TODOS os YAMLs** pelo mesmo padrão — `arch_json` e `flim_weights_path` com valor
   começando em `/`. Um caso encontrado quase nunca é o único.
3. **Virar erro de lint permanente no pré-voo** (checagem 16): caminho absoluto é um erro
   **separado** de caminho inexistente, com mensagem própria. O bug não pode voltar pela porta
   dos fundos no próximo config escrito à mão.

---

## Portão de deleção (SEQUENCIAL, e só com aprovação explícita do usuário)

**Nada é apagado** antes de todos os quatro:

1. O **pré-voo passa** em todos os experiment YAMLs.
2. O **`--dry-run` do launcher reproduz os comandos antigos** — mesma semântica, não
   necessariamente mesma string.
3. **UM experimento pequeno roda ponta a ponta na árvore nova e bate o número com a árvore
   velha.**
4. **O usuário aprova a lista** de deleção, item por item. Imprimir a lista e **parar** até a
   resposta chegar.

### Trava de segurança — há experimentos rodando neste momento

**Há experimentos rodando agora. Não mexer neles.** Antes de qualquer deleção:

- Checar processos vivos: `ps aux | grep python` e `nvidia-smi`.
- Checar runs ativos no W&B.
- **Qualquer arquivo que um processo vivo esteja lendo fica congelado até o run terminar.** Não
  importa que ele esteja na lista aprovada: a lista é revalidada contra os processos vivos no
  momento da deleção, não no momento da aprovação.

**Incerto nunca vira apagável.** Se o agente não consegue provar que nenhum processo vivo usa o
arquivo, ele vai para a lista "incerto", que não é deletada.

### Agentes de deleção — um por alvo, paralelos entre si

Depois do portão, e só depois, um dispatch paralelo com um agente por alvo. Os alvos são
disjuntos, então nenhum agente espera outro:

| Agente | Alvo |
|---|---|
| **H1** | os 6 scripts ray: `scripts/{run_ssl_ray, distillation_ray, distillation_conv_ray, autoencoder_flim_ray, classification_flim_ray, spifil_growth_loop}.py` |
| **H2** | `src/evaluate/ray_mlp_queue.py` (já vive como `experiments/ray/runners/eval.py`) |
| **H3** | os blocos `if __name__ == "__main__"` com argparse dentro de `src/modules/*.py` |
| **H4** | os helpers argparse em `src/models/distillation.py:279-497` (`add_student_flags`, `resolve_student`, `add_distill_flags`, `resolve_distill_flags`, `derive_flim_paths`, `distill_run_tags`) — as flags viraram `init_args` do módulo |
| **H5** | o `from constants import ...` com hack de `sys.path` (existe só porque os launchers rodavam de dentro de `scripts/`) |

Cada agente confirma, antes de apagar, que o destino novo do seu alvo já existe e já foi
exercitado pelo `--dry-run`. Cada agente escreve sua linha em `MIGRATION.md` (D2). Nenhum agente
apaga um arquivo que outro agente da lista também toca — se a partição colocar dois agentes no
mesmo arquivo, isso é bug da partição e vira relatório, não deleção.

---

## Critérios de aceitação globais do refactor

Cada item é um comando executável. Item que não é comando não é critério de aceitação.

1. `grep -rn "argparse" methods/ core/ flim/` → **vazio**. Exceções só as registradas por
   escrito (nenhuma prevista nessas três pastas).
2. `grep -rn "argparse" analysis/` → só os `__main__` de 3 linhas permitidos, cada um com
   justificativa registrada; qualquer parser completo é falha.
3. `grep -rn "num_gpus=1" experiments/` → **vazio**.
4. `grep -rn "sys.path" ` → **vazio** no código movido.
5. Todo `class_path` de todo YAML é **importável**: para cada `.yaml` em `configs/`,
   `importlib.import_module(módulo)` e `hasattr(mod, classe)`. Falha reportando módulo E classe.
6. `python -m experiments.ray.launch experiments/lejepa/growth_grid4.yaml --dry-run` imprime
   **18 comandos** (3 datasets × 3 splits × 2 percentages) e sai com 0.
7. Cada comando impresso tem exatamente **3 `--config`** e **nenhuma flag de argparse específica
   de método**.
8. `git diff --stat` do refactor inteiro: a **contagem de linhas do repositório CAIU**. Se não
   caiu, alguma coisa foi adicionada que não devia.
9. **Um experimento pequeno roda ponta a ponta na árvore nova e reproduz o número da árvore
   antiga.** Este é o critério que autoriza o portão de deleção.
10. `grep -rn "oneoff" --include=*.py` não retorna nenhum import vindo de fora de
    `experiments/oneoff/`.
11. `grep -rn "^import ray\|^from ray" eval/` → **vazio** (o critério de corte do E-EVAL).
12. Nenhum YAML tem `arch_json` ou `flim_weights_path` com valor absoluto (checagem 16 do
    pré-voo, agora permanente).
13. `MIGRATION.md` tem uma linha para cada rename de `git log --diff-filter=R --name-status` e
    para cada `git rm` do refactor. Divergência entre as duas listas é falha.
14. **Constantes, as duas camadas:** `grep -rn "^from\|^import" core/constants.py` → só stdlib ou
    vazio (global não importa local); nenhum import de `<pacote>.constants` vindo de fora do próprio
    pacote (local não atravessa); interseção de nomes entre `core/constants.py` e cada
    `<pacote>/constants.py` → **vazia** (sem shadowing); nenhum `constants.py` contém `def`,
    `class`, `os.environ` ou chamada de função.
