# `src/evaluate/` — como rodar cada avaliação

Todo módulo aqui é executável como `python -m src.evaluate.<módulo>`, sempre **a partir da raiz do repositório**, com o ambiente ativo:

```bash
source /dados/home/moliveira/miniforge3/etc/profile.d/conda.sh && conda activate scalable_FLIM
```

Quem não é executável (`constants.py`, `eval_plotter.py`, `wandb_resolver.py`) é biblioteca — importado pelos outros, nunca rodado direto.

**Regra permanente:** existe **um** `fit_svm` e **um** `compute_metrics`, em [`src/utils/evaluate.py`](../utils/evaluate.py) e [`src/metrics/classification.py`](../metrics/classification.py). Todo módulo desta pasta os importa; nenhum reescreve. Depois de mexer nesses dois, rode [`tools/check_refactor_equivalence.py`](../../tools/check_refactor_equivalence.py).

---

## Índice

| módulo | o que avalia | saída |
|---|---|---|
| [`unified_eval.py`](unified_eval.py) | pipeline unificado: SVM + MLP freeze + MLP unfreeze | `results/` + plots |
| [`svm.py`](svm.py) | SVM linear sobre encoders SSL congelados | `results/svm_results.csv` |
| [`mlp.py`](mlp.py) | fine-tune de cabeça MLP sobre encoders pré-treinados | `results/mlp_results.csv` |
| [`ray_mlp.py`](ray_mlp.py) | o mesmo MLP, paralelo via Ray (1 experimento por GPU) | `results/ray_mlp_results.csv` |
| [`ray_mlp_queue.py`](ray_mlp_queue.py) | fila com slots: vários experimentos **na mesma** GPU | idem, com state file resumível |
| [`eval_avg_pooling_48d.py`](eval_avg_pooling_48d.py) | **FLIM cru, GAP 48-d**, LAB cru, solver convergido | `artifacts/plots/comparacao_flim_protocolo_original/eval_48d_norm_off.csv` |
| [`eval_svm_flim_flatten.py`](eval_svm_flim_flatten.py) | **FLIM cru, conv3 achatado 27.648-d** — o protocolo original | `.../eval_svm_flim_flatten.csv` |
| [`eval_autoencoder.py`](eval_autoencoder.py) | encoder do **autoencoder treinado**, um braço de peso (`lab` / `lab_flat`) por rodada | `results/eval_autoencoder_<weights>.csv` |
| [`svm_real_flim.py`](svm_real_flim.py) | FLIM cru sob o protocolo do "Distill 4" | `results/real_FLIM_results.csv` |
| [`svm_classification_flim.py`](svm_classification_flim.py) | sonda linear **depois** do treino supervisionado (Experimento 3) | `results/svm_relu2l_results.csv` |
| [`svm_distillation.py`](svm_distillation.py) | students destilados, embedding cru do encoder | `results/svm_distill_*.csv` |
| [`svm_distillation_conv.py`](svm_distillation_conv.py) | idem, só runs `next_layers_direct`, proj head cortada → [B, 48] | `results/svm_distillation_conv_results.csv` |
| [`svm_distill_with_projection.py`](svm_distill_with_projection.py) | idem, proj head **mantida** → [B, 1280] | `results/svm_proj1280_*.csv` |
| [`svm_flim_residual.py`](svm_flim_residual.py) | encoders FLIM residuais montados à mão (só eggs) | `results/svm_flim_residual_eggs.csv` |
| [`svm_ijepa.py`](svm_ijepa.py) | embeddings I-JEPA (ViT-H/14) | `results/ijepa_svm_results.{csv,json}` |
| [`classical_classifiers.py`](classical_classifiers.py) | kNN / QDA / RF / LightGBM / GP sobre embeddings congelados | `results/classical_classifiers_results.csv` |
| [`tsne_analysis.py`](tsne_analysis.py) | t-SNE 2D do test set (FLIM destilado vs LeJEPA) | PNGs |

---

## Pipeline unificado

```bash
# tudo
python -m src.evaluate.unified_eval --model all --dataset all

# recortes
python -m src.evaluate.unified_eval --model svm --dataset eggs
python -m src.evaluate.unified_eval --mode freeze --split 1 --pct 100
python -m src.evaluate.unified_eval --dry-run            # lista o que rodaria
python -m src.evaluate.unified_eval --resume             # continua de onde parou
```

`python -m src.evaluate` (sem módulo) delega para o `main()` daqui.

Flags: `--model --mode --dataset --split --pct --init --embed-mode --entity --project --output-root --max-runs --resume --restart --resume_id --dry-run --verbose`.

⚠️ **Caveat conhecido:** o `unified_eval` reagrupa séries de origens diferentes num CSV só. Antes de comparar duas linhas dele, confira se as duas saíram do mesmo protocolo — a coluna `method` não garante isso sozinha.

---

## SVM e MLP sobre encoders SSL

```bash
python -m src.evaluate.svm                      # -> results/svm_results.csv
python -m src.evaluate.svm --wandb-update       # e escreve de volta no W&B

python -m src.evaluate.mlp --mode all
python -m src.evaluate.mlp --mode freeze
python -m src.evaluate.mlp --mode unfreeze --dataset protozoan
python -m src.evaluate.mlp --config configs/evaluate/mlp/freeze/helminth-eggs/split_1/pct_100/7rkcbbnk.yaml
```

O `mlp.py` descobre os YAML em `configs/evaluate/mlp/{freeze,unfreeze}/` sozinho; `--config` roda um só.

**Paralelizando o MLP.** Duas estratégias, escolha pela ocupação da GPU:

```bash
# um experimento por GPU
python -m src.evaluate.ray_mlp --num-gpus 4 --mode all

# vários experimentos na MESMA GPU (slots), resumível
python -m src.evaluate.ray_mlp_queue --dry-run
python -m src.evaluate.ray_mlp_queue --num-gpus 1 --max-concurrent-per-gpu 5
python -m src.evaluate.ray_mlp_queue --resume
python -m src.evaluate.ray_mlp_queue --dataset-group protozoan
python -m src.evaluate.ray_mlp_queue --mode freeze --dataset-group eggs
python -m src.evaluate.ray_mlp_queue --wandb-update
```

A fila mantém um state file (`--state-file`) e faz pinning explícito de `CUDA_VISIBLE_DEVICES` por task.

---

## Encoder FLIM cru — as curvas de protocolo

Estes dois respondem à mesma pergunta por caminhos diferentes: **o encoder FLIM em disco reproduz a curva de κ do CSV externo (`data/reports_felipe/svm/`)?** Usam os mesmos pesos, os mesmos dados e o mesmo SVM; divergem em **uma linha** — como o mapa conv3 vira vetor.

| | entrada | features | `max_iter` | κ @75% protozoan |
|---|---|---|---|---|
| avaliador oficial de hoje | ImageNet-norm | GAP 48-d | 10000 | 0.3247 |
| 🟢 `eval_avg_pooling_48d` | LAB[0,1] cru | GAP 48-d | −1 | 0.7209 |
| 🟠 `eval_svm_flim_flatten` | LAB[0,1] cru | flatten 27.648-d | −1 | 0.8302 |
| 🔵 CSV externo (não versionado) | (?) | (?) ~10⁴-d | desconhecido | 0.8371 |

Encoder: FLIM cru, **sem checkpoint** — o mesmo caminho que o estágio 1 do autoencoder congela. Os dois **importam** `train_svm`, `extract_features`, `_encode_pooled` e `compute_metrics`; nada é reescrito. O gráfico das três curvas sai de [`tools/plot_comparacao_flim_protocolo.py`](../../tools/plot_comparacao_flim_protocolo.py).

### `eval_avg_pooling_48d` — GAP 48-d

Isola o efeito da dimensão: adota as correções de entrada e de solver, mantendo as 48 dimensões do lado oficial.

```bash
# valida arch JSON, pesos e splits sem ajustar SVM nenhum
python -m src.evaluate.eval_avg_pooling_48d --dry-run

OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE CUDA_VISIBLE_DEVICES=0 \
  conda run -n scalable_FLIM --no-capture-output \
  python -m src.evaluate.eval_avg_pooling_48d --dataset protozoan --splits 1 2 3 --percentages 1 5 25 50 75 100
```

| flag | default | nota |
|---|---|---|
| `--dataset` | `protozoan` | **um por vez**, e o CSV abre em modo `"w"` — rodar eggs depois de protozoan **sobrescreve** |
| `--splits` | `1 2 3` | |
| `--percentages` | `1 5 25 50 75 100` | |
| `--max-iter` | `-1` | convergido; um teto positivo fica declarado na coluna `max_iter` |
| `--num-workers` | `8` | |
| `--out` | `.../eval_48d_norm_off.csv` | nome herdado de antes do rename; o plotter aponta pra ele |
| `--dry-run` | — | |

Custo: ~15 s por split. Barato em tempo, caro em iterações — 727 k no pct100, que é exatamente por que o `max_iter=10000` do avaliador oficial destrói esta configuração.

### `eval_svm_flim_flatten` — conv3 achatado 27.648-d

As 27.648 dimensões são 48 × 24 × 24: o mapa conv3 inteiro, sem pooling. A troca é feita rebindando `src.utils.evaluate.EMBED_MODE = "flatten"`, a chave que o próprio [`_encode_pooled`](../utils/evaluate.py#L251) lê.

Reconstrói o eval de dois scripts `tools/diag_*.py` que a regra do `.gitignore` apagou sem passar pelo histórico do git. Sobraram só as saídas, em `artifacts/plots/comparacao_flim_protocolo_original/dados_brutos/` — e é contra elas que `--compare` confere.

```bash
python -m src.evaluate.eval_svm_flim_flatten --dry-run

# as 54 células, conferindo contra os JSONs da curva já plotada
OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE CUDA_VISIBLE_DEVICES=0 \
  conda run -n scalable_FLIM --no-capture-output \
  python -m src.evaluate.eval_svm_flim_flatten --datasets larvae eggs protozoan --compare

# só as células que faltam
python -m src.evaluate.eval_svm_flim_flatten --datasets protozoan --splits 3 --percentages 75 100 --compare
```

| flag | default | nota |
|---|---|---|
| `--datasets` | os três | aceita **vários numa rodada**; o CSV é único, com coluna `dataset` |
| `--splits` | `1 2 3` | |
| `--percentages` | `1 5 25 50 75 100` | |
| `--max-iter` | `-1` | |
| `--num-workers` | `8` | |
| `--out` | `.../eval_svm_flim_flatten.csv` | |
| `--compare` | — | confere cada célula contra `dados_brutos/*.json`, imprime `dk` por linha e MAE no fim |
| `--dry-run` | — | valida arquivos e conta as células de referência |

**Estado da verificação:** 52 das 54 células conferidas, **todas com `Δκ = 0.000000`** — larvae 18/18, eggs 18/18, protozoan 16/18 (a rodada foi interrompida antes de `sp3 pct75/pct100`). `n_sv` e `n_iter_max` batem célula a célula. Como o CSV só é escrito no fim, a rodada interrompida **não deixou arquivo**.

Custo: ~2 min de fit por split em protozoan pct100. O `torch.cat` acumulativo do `train_svm` oficial domina o tempo nessa dimensão.

### `svm_real_flim` — FLIM cru sob o protocolo do Distill 4

Mesma pergunta, outro protocolo de referência: quanto o FLIM real entrega sob **exatamente** o pipeline que produziu `results/svm_distill_proj1280_results.csv`.

```bash
python -m src.evaluate.svm_real_flim --datasets protozoan eggs larvae --splits 1 2 3 --imagenet-norm
```

Flags: `--datasets --splits --percentages --imagenet-norm --num-workers --out-dir --csv-stem`.

---

## Destilação

Os três compartilham o scanner de `artifacts/distillation/` (run com `run_metadata.json` + ao menos um `.ckpt`); mudam onde cortam o modelo:

| módulo | corte | embedding |
|---|---|---|
| `svm_distillation` | encoder cru, sem proj head | conforme o student |
| `svm_distillation_conv` | proj head cortada, só runs `next_layers_direct` | `[B, 48]` |
| `svm_distill_with_projection` | proj head **mantida** | `[B, 1280]` |

```bash
python -m src.evaluate.svm_distillation --run eggs_split1
python -m src.evaluate.svm_distillation_conv
python -m src.evaluate.svm_distillation_conv --run eggs_split1
python -m src.evaluate.svm_distill_with_projection --run-filter 1x1_BN2d --only-ok

# rodada longa: solte em tmux e guarde o log
tmux new-session -d -s svm_conv \
  "python -m src.evaluate.svm_distillation_conv 2>&1 | tee logs/svm_conv_$(date +%Y%m%d_%H%M%S).log"
```

Flags comuns: `--run --run-filter --output-csv --artifacts-dir --wandb-update --no-imagenet-norm --only-ok`.

⚠️ **Contagem de parâmetros dos students tem armadilha** (MLP vestigial no `state_dict`, encoder duplicado, protozoan com 30 canais em vez de 32). Ver [`distill_details.md`](../../distill_details.md) antes de reportar número de parâmetros.

---

## Baselines e visualização

```bash
# sonda linear depois do treino supervisionado (Experimento 3)
python -m src.evaluate.svm_classification_flim --pattern 'relu2l_*'
python -m src.evaluate.svm_classification_flim --run <run_id> --no-imagenet-norm

# encoders FLIM residuais, só eggs
python -m src.evaluate.svm_flim_residual --flim_residual_assessment

# I-JEPA ViT-H/14
python -m src.evaluate.svm_ijepa --dataset eggs --split 1 --pct 100
python -m src.evaluate.svm_ijepa --aggregate-only     # só reagrega o que já existe

# kNN / QDA / RF / LightGBM / GP — salva incremental, resume sozinho
python -m src.evaluate.classical_classifiers

# t-SNE 2D do test set
python -m src.evaluate.tsne_analysis --model flim --dataset eggs --split 1 --pct 100
python -m src.evaluate.tsne_analysis --model lejepa --perplexity 30 --max-samples 3000 --force
```

---

## Auto-testes

Não usam pytest — são scripts de `assert`: ou imprimem OK, ou morrem.

```bash
python tools/check_refactor_equivalence.py    # depois de mexer em fit_svm / compute_metrics / probe SVM
python tools/check_probe_matches_evaluator.py # o probe do treino e o avaliador são o mesmo objeto
```

---

## Ao ler os CSVs

- **`acc` não é sempre a mesma acurácia.** `compute_metrics` devolve `multiclass_accuracy(average="macro")` — balanceada. Os CSVs que trazem `acc_raw` / `raw_acc` é que têm a acurácia crua. O CSV externo do FLIM registra a crua na coluna `acc`. **κ é a única métrica diretamente comparável entre todas as séries.**
- **`f1` idem**: o repositório calcula macro; o CSV externo registra weighted. Não são comparáveis.
- **`fit_status` importa.** `fit_status_ = 1` significa que o solver bateu no teto de iterações e parou num ponto arbitrário do caminho de otimização — o resultado passa a depender da ordem das linhas de treino. Isso passou meses despercebido porque nada salvava a coluna. Hoje `fit_svm_with_diagnostics` grava `fit_status`, `n_iter_max`, `n_iter_sum` e `n_sv` em toda rodada. Confira antes de citar qualquer κ.
