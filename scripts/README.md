# `scripts/` — launchers, relatórios e manutenção

Tudo aqui roda como script de topo, **sempre a partir da raiz do repositório**:

```bash
source /dados/home/moliveira/miniforge3/etc/profile.d/conda.sh && conda activate scalable_FLIM
python scripts/<arquivo>.py
```

Nunca `python -m scripts.<arquivo>`: não existe `scripts/__init__.py`, e cada arquivo depende de ser executado como script para que `scripts/` entre no `sys.path[0]`.

## `constants.py` é a fonte única

Todo valor que aparece em dois ou mais arquivos desta pasta mora em [`constants.py`](constants.py) — datasets, splits, percentuais, caminhos dos pesos FLIM, entity/projeto W&B, níveis de log, truncamento de stderr. Importe assim:

```python
from constants import DATASETS, NUM_CLASSES, PARASITE_DIR
```

O cabeçalho de `constants.py` documenta o que **não** foi unificado e por quê. Antes de "consertar" uma divergência, leia lá: várias são intencionais (o protozoan tem `ch24_30_48` na arquitetura e `ch24_32_48` nos pesos; cada launcher tem seu próprio default de `--max-concurrent-per-gpu` por tamanho de modelo; `flim-ssl` e `journal_02_2026_hybrid_FLIM` são projetos W&B diferentes).

**Duas armadilhas ao usar `constants.py`:**

1. **`autoencoder_flim_ray.py` insere o próprio diretório no `sys.path`** antes de importar de `constants`. Isso é obrigatório: quatro arquivos de fora (`src/evaluate/eval_avg_pooling_48d.py`, `src/evaluate/eval_svm_flim_flatten.py`, `src/evaluate/svm_real_flim.py`, `tools/check_probe_matches_evaluator.py`) importam `_arch_json`/`_flim_weights_path` dele, e nesse caminho o `sys.path[0]` é a raiz do repo, não `scripts/`. Não remova aquele bloco.
2. **Não use uma *função* de `constants.py` dentro de um corpo `@ray.remote`.** O cloudpickle serializa função por referência e o worker não tem `scripts/` no path — `ModuleNotFoundError: No module named 'constants'`. Valores simples (`str`, `int`, `dict`) são copiados por valor e passam sem problema. Isso já quebrou uma vez, em `retry_protozoan_experiment.run_ssl_experiment`.

---

## Índice

| script | o que faz |
|---|---|
| **Launchers Ray (treino)** | |
| [`run_ssl_ray.py`](run_ssl_ray.py) | fila SSL LeJEPA — a grade principal de pré-treino |
| [`autoencoder_flim_ray.py`](autoencoder_flim_ray.py) | autoencoder não supervisionado, protocolo de 2 estágios |
| [`distillation_ray.py`](distillation_ray.py) | destilação I-JEPA → FLIM |
| [`distillation_conv_ray.py`](distillation_conv_ray.py) | destilação `next_layers`, com grade de proj heads |
| [`classification_flim_ray.py`](classification_flim_ray.py) | Experimento 3: encoder FLIM + cabeça de classificação |
| [`retry_protozoan_experiment.py`](retry_protozoan_experiment.py) | re-treino SSL só de protozoan-cysts |
| [`run_missing_mlp.py`](run_missing_mlp.py) | fine-tune MLP de `run_ids` específicos |
| [`train_autoencoder_flim.py`](train_autoencoder_flim.py) | entrypoint de UMA run de autoencoder (o que o launcher invoca) |
| **Crescimento SPiFiL (uma camada por rodada)** | |
| [`spifil_growth_loop.py`](spifil_growth_loop.py) | protocolo completo: estágio 1/2, depois cresce+congela / solta, até o κ parar |
| [`spifil_grow.py`](spifil_grow.py) | UMA rodada: corta uma camada SPiFiL do backbone treinado e grava em formato FLIM |
| [`spifil_resnet_graft.py`](spifil_resnet_graft.py) | referência: enxerta blocos SPiFiL no meio de uma ResNet-18 |
| [`check_spifil_growth.py`](check_spifil_growth.py) | check do crescimento: shape do decoder, regra de parada, layout dos pesos |
| **Status e auditoria** | |
| [`check_distill_conv_status.py`](check_distill_conv_status.py) | status da grade `next_layers_direct`: disco × W&B |
| [`report_distill_disk_wandb.py`](report_distill_disk_wandb.py) | relatório de disco + W&B de toda a destilação |
| [`report_distill_divergences.py`](report_distill_divergences.py) | filtra as divergências do relatório acima |
| [`verify_finetune_weights.py`](verify_finetune_weights.py) | auditoria: pesos locais × YAML × W&B |
| [`test_dataset.py`](test_dataset.py) | smoke test dos 3 datamodules |
| **Normalização e agregação** | |
| [`normalize_reports.py`](normalize_reports.py) | CSVs legados → schema canônico; escreve `unified_svm_comparison.csv` |
| [`aggregate_classical_results.py`](aggregate_classical_results.py) | agrega os classificadores clássicos por split |
| [`eval_sigmoid2l_test.py`](eval_sigmoid2l_test.py) | métricas de teste das runs `sigmoid2l_` |
| [`build_classhead_report.py`](build_classhead_report.py) | tabelas markdown comparando cabeças de classificação |
| **Plots** | |
| [`plot_comparison_flim.py`](plot_comparison_flim.py) | FLIM × LeJEPA × I-JEPA × destilação |
| [`plot_classical_classifiers.py`](plot_classical_classifiers.py) | curvas dos clássicos (kNN/QDA/RF/LGBM/GP) |
| [`plot_parameters_vs_metrics.py`](plot_parameters_vs_metrics.py) | nº de parâmetros × métrica |
| [`plot_real_flim_vs_distill4.py`](plot_real_flim_vs_distill4.py) | FLIM cru × Distill 4 × FLIM registrado |
| [`plot_svm_vs_mlp_pct.py`](plot_svm_vs_mlp_pct.py) | SVM × MLP ao longo do % de rótulos |
| **Manutenção de checkpoints** | |
| [`strip_teacher_only.py`](strip_teacher_only.py) | remove só as chaves `teacher.*` |
| [`strip_teacher_from_ckpt.py`](strip_teacher_from_ckpt.py) | reescreve o ckpt guardando só student + proj |
| [`validate_stripped_ckpt.py`](validate_stripped_ckpt.py) | prova que o student sobreviveu idêntico |
| [`clear_wandb_cache.py`](clear_wandb_cache.py) | apaga runs offline do W&B e checkpoints órfãos |
| **Dados e configs** | |
| [`generate_mlp_configs.py`](generate_mlp_configs.py) | gera todos os YAML de avaliação do MLP |
| [`download_organmnist3d.py`](download_organmnist3d.py) | baixa OrganMNIST3D e fatia em PNG 2D |
| **Shell** | |
| [`regen_svm_queue.sh`](regen_svm_queue.sh) | regenera os CSVs de SVM e replota |
| [`run_protozoan_ssl.sh`](run_protozoan_ssl.sh) | laço bash sobre splits × pcts × inits do protozoan |

---

## Launchers Ray

Os cinco compartilham a mesma anatomia: montam uma grade `dataset × split × percentage`, escalonam com um `GpuSlotScheduler` que faz pinning explícito de `CUDA_VISIBLE_DEVICES`, disparam cada experimento como subprocess e gravam um manifest CSV no fim.

**Ray não gerencia GPU aqui.** `RAY_INIT_KWARGS` passa `num_gpus=0` e o pinning é manual. Ao conectar num cluster existente com `--ray-address`, os scripts removem `num_gpus` do dict — o Ray rejeita esse kwarg nesse modo.

```bash
# sempre comece pelo dry-run: ele imprime a grade sem gastar GPU
python scripts/distillation_conv_ray.py --dry-run
python scripts/classification_flim_ray.py --dry-run

# rodada real, uma GPU, 3 experimentos por GPU
python scripts/distillation_conv_ray.py --num-gpus 1 --max-concurrent-per-gpu 3

# recortes
python scripts/distillation_conv_ray.py --datasets eggs --splits 1 --percentages 100
python scripts/autoencoder_flim_ray.py --stage 1 --embed-modes avgpool2d flatten
python scripts/run_ssl_ray.py --inits flim trunc_normal --pcts 5 75

# rodada longa: solte em tmux e guarde o log
tmux new-session -d -s distill \
  "python scripts/distillation_conv_ray.py 2>&1 | tee logs/distill_$(date +%Y%m%d_%H%M%S).log"
```

Flags comuns: `--datasets --splits --percentages --num-gpus --max-concurrent-per-gpu --cpus-per-experiment --ray-address --dry-run --fail-fast --log-level`. Os defaults de `--num-gpus` e `--max-concurrent-per-gpu` **diferem por script de propósito** (o autoencoder cabe 1 por GPU, o SSL cabe 10) — não os unifique.

⚠️ **`--dry-run` reescreve o manifest.** O caminho de dry-run chama `_write_manifest([], skipped, ...)`, então `artifacts/*/run_manifest*.csv` fica só com o cabeçalho. Os manifests não são versionados; se o histórico importar, copie antes.

⚠️ **`retry_protozoan_experiment.py --dry-run` reescreve 9 YAML** em `configs/model/` antes de checar o flag, porque `_ensure_protozoan_model_configs()` roda primeiro. Os arquivos atuais apontam para `/dados/home/moliveira/scalable_FLIM_self_supervised/`, um checkout diferente — rodar o dry-run sobrescreve isso.

## Status e auditoria

```bash
python scripts/check_distill_conv_status.py            # grade next_layers_direct
python scripts/report_distill_disk_wandb.py --no-wandb # só disco, não consulta W&B
python scripts/report_distill_divergences.py
python scripts/verify_finetune_weights.py --dataset eggs --mode freeze --verbose
python scripts/test_dataset.py                         # smoke test dos datamodules
```

## Normalização, agregação e plots

A cadeia de resultados tem uma ordem obrigatória — os plots leem o que o `normalize_reports.py` escreve:

```
results/*.csv  +  data/reports_felipe/
        │
        ▼  normalize_reports.py
artifacts/normalized/unified_svm_comparison.csv
        │
        ▼  plot_comparison_flim.py · plot_parameters_vs_metrics.py · plot_svm_vs_mlp_pct.py
artifacts/plots/**.png
```

```bash
python scripts/normalize_reports.py           # -> artifacts/normalized/
python scripts/plot_comparison_flim.py
python scripts/plot_classical_classifiers.py
python scripts/plot_svm_vs_mlp_pct.py
python scripts/plot_real_flim_vs_distill4.py
python scripts/plot_parameters_vs_metrics.py
bash  scripts/regen_svm_queue.sh 0            # regenera os CSVs de SVM na GPU 0 e replota
```

Os scripts de plot aceitam um conjunto grande de flags de estilo (`--legend-fontsize`, `--linewidth`, `--fig-width-per-dataset`, `--dpi`, …). Esses valores **divergem entre scripts de propósito** — ordem de `METRICS`, textos de `METRIC_LABEL`, paletas, dpi 150 vs 200. Mudar qualquer um muda o PNG, então eles ficam locais em cada arquivo, não em `constants.py`.

## Manutenção de checkpoints

Os checkpoints de destilação carregam o teacher congelado e passam de 2 GB. Dois scripts encolhem, um valida. **Todos são no-op sem `--apply`** — o default é dry-run.

```bash
python scripts/strip_teacher_only.py --min-gb 1.5           # lista o que faria
python scripts/strip_teacher_only.py --min-gb 1.5 --apply   # executa
python scripts/strip_teacher_from_ckpt.py --from-metadata --apply
python scripts/validate_stripped_ckpt.py <original.ckpt> <stripped.student.ckpt>
python scripts/clear_wandb_cache.py
```

⚠️ Os globs de `strip_teacher_*.py` e os caminhos de `clear_wandb_cache.py` são **relativos ao cwd**, não à raiz do repo. Rodados de outro diretório eles silenciosamente não encontram nada — ou, pior, encontram outra coisa. Rode da raiz. Isso é deliberado: absolutizar faria `--apply` começar a reescrever checkpoints de qualquer lugar.

---

## Problemas conhecidos, ainda abertos

Registrados durante a centralização das constantes e **não** consertados, porque consertar mudaria comportamento:

1. **Os relatórios de destilação não se encontram.** `report_distill_disk_wandb.py` grava em `artifacts/distillation/distill_disk_wandb_report.csv`; `report_distill_divergences.py` lê `results/distill_disk_wandb_report.csv` por default. Encadeados sem flags explícitas, o segundo nunca acha o primeiro. Passe `--in` na mão.
2. **`validate_stripped_ckpt.py` não põe a raiz no `sys.path`** e faz `from src.evaluate...`. Executado como `python scripts/validate_stripped_ckpt.py`, morre com `ModuleNotFoundError: No module named 'src'`.
3. **`plot_classical_classifiers.py` quebra com `--out` relativo** ou fora do repo: quatro `print(...relative_to(PROJECT_ROOT))` levantam `ValueError`. Use caminho absoluto dentro do repo.
4. **`example_distill_hawk.py` é standalone e não pertence a este pipeline** — usa o dataset Corel e VGG16, não os parasitos. Precisa de `./images/corel` relativo ao cwd.
5. **A docstring de `run_ssl_ray.py` ainda se chama `retry_ssl_missing.py`**, nome antigo do arquivo.
6. **`ALL_DATASETS` e `ALL_SPLITS` em `plot_classical_classifiers.py` são código morto** — definidos, nunca lidos.
7. **`test_dataset.py` usa dois esquemas de split.** `data/to_modules/splits/` só tem `split_0`; `new_split_parasito/*/splits_incremental/` só tem `1, 2, 3`. Daí as duas constantes `SPLIT_LEGACY = 0` e `SPLIT = 1`. Também vale olhar: train e validation reportam o mesmo tamanho (9094), o que merece investigação.
