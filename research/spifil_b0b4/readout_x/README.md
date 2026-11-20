# readout_x — backbone congelado × cabeça sem treino

`eval_backbone_head.py` mede κ/F1 de **validação** e de **teste** com a SVM original do repositório (`core.svm.fit_svm`: SVC linear, C=100, ovo, max_iter=-1, **sem scaler**). A SVM é reajustada no treino para cada representação. Nada é treinado e nada é ajustado em val/teste: a PCA vê só pixels do treino. O teste serve só para o relatório (`never_feeds_fitness`). O embedding é extraído em fp32, sem autocast, sem TF32 e sem TTA.

| cabeça (jsonl `head`) | o que é | dim |
|---|---|---|
| `gap1280` | embedding oficial: `head(encoder(x))` (growth/modehb, com a dimensão da própria cabeça: 1280 ou 2560); nos híbridos de outubro, o GAP do L4 (= `results/svm_disc_bp01w64c1280_test_avg.csv`) | 1280 (2560 no c00306) |
| `so2_1280_pca` | `Norm(P[μ, σ, triu(sqrtm Cov)])` do mapa do último bloco; P = PCA dos canais (só treino), k=48 → 48+48+1176 = 1272, mais 8 zeros | 1280 |
| `so2_4096_pca` | idem, k=88 → 88+88+3916 = 4092, mais 4 zeros | 4096 |
| `so2_*_rnd` | controle: mesmo cálculo com P = projeção ortonormal fixa semeada | 1280 / 4096 |
| `xb_1280_pca`, `xb_4096_pca` | cross-cov entre todos os pares de blocos. O mapa maior é reamostrado para a grade do menor e cada bloco passa pela sua PCA (só treino); kp = ⌊√(D/n_pares)⌋, mais zeros até D | 1280 / 4096 |
| `so2_4096_pca_tta8` (só com `--tta8`) | `so2_4096_pca` com TTA8 só na avaliação: média dos vetores das 8 vistas rot90 × flip, com a mesma PCA e a mesma SVM do treino sem TTA (`svm_from_head`) | 4096 |

Norm é signed-sqrt seguido de L2 sobre o vetor inteiro. A sqrtm usa eigh em fp64. Os pixels da PCA são `--n-px` (64) por imagem de treino, sorteados com semente por imagem, o que torna o resultado independente do batch.

## Comando (athena)

O ambiente é torch 2.2, e o script não usa nenhuma API exclusiva da 2.7. Use a GPU 3 ou 2. O pico de memória é pequeno: com batch 16, menos de 8 GB. Rode **antes** o `--dry-run` e confira se todo `[ITEM]` aparece como OK. Uma linha `[PRESET] ... PULADO` indica que o glob não encontrou o nó; nesse caso passe o caminho com `--item`.

```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8 \
  python research/spifil_b0b4/readout_x/eval_backbone_head.py --preset table --dry-run

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8 \
  nice -n 10 python research/spifil_b0b4/readout_x/eval_backbone_head.py --preset table \
  --device cuda:0 --batch-size 16 --num-workers 8 \
  --out research/spifil_b0b4/readout_x/results_athena.jsonl \
  [--nossl-b1 <.../remote_sensing_fix5_nossl/.../nodes/<B1>/<f>/backbone.pt>]
# só a tabela, a partir do jsonl:
python research/spifil_b0b4/readout_x/eval_backbone_head.py --table \
  --out research/spifil_b0b4/readout_x/results_athena.jsonl
```

- Um backbone fora do preset entra com `--item id=NOME,kind=growth|modehb|hybrid,ckpt=CAMINHO` (pode repetir). `--only <substr>` filtra os ids.
- A retomada é idempotente. Rodar de novo com o mesmo `--out` pula todo backbone que já tem as 7 cabeças, pela chave (backbone_id, ckpt, max_images, head).
- `--code-root` vale por padrão `/dados/home/moliveira/exps/growth/code`. É ele que fornece `ga_search.backbones`, e é importado antes do repo.
- `--max-images N` serve **só** para o smoke.

## Backbones da tabela (`--preset table`)

| id | tipo | checkpoint |
|---|---|---|
| RS_B0_seed01_c00306 | modehb | `artifacts/modehb_spifil/remote_sensing/state/seed42/candidates/c00306/f100/encoder.pt`; se faltar, o growth `remote_sensing_fix4/**/nodes/seed01` (conversão bitwise do c00306) |
| RS_B1_fix5_B1g0c00_f45 | growth | `<growth>/artifacts/hmodehb_b0b4/growth/*/remote_sensing_fix5/**/nodes/B1g0c00/f45/backbone.pt` |
| RS_B2_fix4_B2g1c02_f15 / _f45 | growth | `.../growth/*/remote_sensing_fix4/**/nodes/B2g1c02/{f15,f45}/backbone.pt` (ex.: `athena_gpu3/remote_sensing_fix4/seed42/...`) |
| RS_B1_fix5_nossl | growth | B1 **sem LeJEPA** (`remote_sensing_fix5_nossl`): passar com `--nossl-b1` quando o caminho chegar; sem ele, o script tenta um glob opcional |
| FL_B0_seed01 | growth | `.../growth/*/flowers*/**/nodes/seed01/backbone.pt` |
| HYB_oct_{eggs,larvae,protozoan}_s{1,2,3}_pct100 | hybrid | `artifacts/spifil_growth/disc_bp01w64c1280/<ds>_split<s>_pct100/L4/checkpoints/best_kappa.ckpt` |

Origem dos dados de cada tipo:

- growth: `growth_state.json:config_full.run.data`.
- modehb: `config_resolved.yaml:data`.
- híbridos: `ParasiteDataset` (ift_lab, imagenet_norm=False) com os descritores da discovery. O md5 é conferido contra `Scalable_Hybrid_FLIM_discovery`.

A coluna `split_name` registra o split usado em cada linha.

## Armadilha: validação dos híbridos em pct100

Nos descritores `data_descriptor_perc100.json` da discovery, `validation` é **a mesma lista** que `train`, nos três datasets e nos três splits. Por isso o κ_val dos híbridos em pct100 é κ **no treino**. Essas linhas saem com `val_is_train=true` e levam † na tabela. Para eles, só o κ_test é informativo.

## Testes

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q research/spifil_b0b4/readout_x/test_eval_backbone_head.py
```

Os testes verificam:

- as dimensões exatas 1280/4096;
- que a PCA é ajustada só com o treino;
- o determinismo (bitwise) e a independência do batch;
- o protocolo da SVM.


## Flags extras (default = comportamento antigo)

- `--heads h1 h2 ...`: calcula só essas cabeças.
- `--tta8`: acrescenta `so2_4096_pca_tta8`.
- `--data-cache DIR`: materializa os splits em memmap fp32 (valores idênticos), compartilhado entre processos.

GA (melhores por validação das buscas modehb): `select_ga_best.py` → `ga_best_list.json`; `run_ga_best.py --jobs 8 --threads 4` → `results/ga_best_test.jsonl`; `make_ga_best_table.py` → `results/ga_best_tabela.md`.
