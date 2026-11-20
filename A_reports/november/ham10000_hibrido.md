# HAM10000 híbrido (SPiFiL + backprop) — preparado, não lançado (09/11)

**Resumo:** são dois experimentos. No athena roda o modelo oficial `disc_bp01w64c1280_bg1` sem mudar nada. No jaci roda a forma da receita SPiFiL do HAM (6 camadas de 256) com o mesmo backprop, mais um controle aleatório puro.

## O pipeline (o que já existe)
- Entrada: `python -m experiments.ray.launch <yaml>`, runner `growth` (`experiments/ray/runners/growth.py`, ramo `layerwise`).
- Cada braço monta a rede uma camada por vez. Para cada camada Lk:
  1. o `flim.spifil.grow` cria a camada nova a partir de patches de superpixel;
  2. o `train.py fit` treina a rede com `AutoEncoderFlimModule` e um decoder novo.
     A camada nova aprende com lr cheio; as antigas aprendem com lr × 0,1 (`layerwise_backprop`).
- Perda: reconstrução BCE com `recon_bg_weight 1.0`, ou seja, o fundo pesa igual. São 200 épocas.
- Escolha do modelo: com `best_by_probe: false`, o checkpoint `best` e o early stopping olham a `val/loss`. Nenhum rótulo entra nessa escolha, e o teste também não.
- O checkpoint fica em `<work_dir>/<braço>/Lk/checkpoints/best_kappa.ckpt`. O nome é antigo, mas o arquivo é o best por val/loss. É um symlink para a subpasta do W&B.
- O `train.py` não roda teste no fim. O teste só sai depois, na avaliação SVM, que é um passo separado.
- O 50% vem de `percentages: [50]`, que vira `--data.init_args.percentage=50` e lê `splits_incremental/splitN/data_descriptor_perc50.json`.

## A receita SPiFiL é a mesma coisa que o oficial?
**Não.**
- O oficial tem 4 camadas: 32/48/64/1280, kernels 5/5/5/1.
- A receita do HAM tem 6 camadas de 256, kernel 5, fps sem rótulo e whitening entre camadas.

Por isso ficaram dois arquivos.

## O que cabe da receita no pipeline híbrido (jaci)
| Receita | No YAML | Lacuna |
|---|---|---|
| 6 camadas × 256, k5, pools 2-2-2-2-1-1 | `layer_channels/kernels/pools` | nenhuma |
| 50 superpixels DISF, 200 px | `n_superpixels: 50`, `superpixels: disf`, `image_size: 200`, `no_crop` | nenhuma |
| fps **sem rótulo** | o `grow` do repo | **lacuna.** O `grow` escolhe 1 imagem por classe e ordena os patches por Fisher, que usa o rótulo de treino. Não há fps sem rótulo no `grow`. O fps vive só no `spf_grade` (runner `eval`), que não grava pesos. |
| whitening entre camadas | `whiten: true` | **lacuna parcial.** O `whiten` do `grow` é um ZCA usado só para escolher os patches. Ele não entra no forward entre as camadas. |

Não escrevi código de modelo novo. Fechar essas lacunas exigiria fazer o `spf_grade` exportar pesos no formato FLIM, e isso é código novo.

## YAMLs
| Arquivo | Máquina | Jobs | O que compara |
|---|---|---|---|
| `experiments/autoencoder/hyb_ham10000_oficial.yaml` | athena, GPUs 0–3, 1 por GPU | 3 (splits 1–3) | o seu método de backprop, como ele é, no HAM |
| `experiments/autoencoder/hyb_ham10000_jaci.yaml` | jaci, GPU 0 | 1 (split 1, piloto) | forma da receita + backprop |
| `experiments/autoencoder/hyb_ham10000_jaci_conf.yaml` | jaci | 3 (splits 1–3) | o mesmo, confirmação |
| `experiments/autoencoder/hyb_ham10000_jaci_rand.yaml` | jaci | 3 (splits 1–3) | controle: mesma rede e treino, pesos kaiming puros (`random_layer_classic`) |

Todos usam seed 42, 50% dos dados e `skip: state`. Cada job faz 4 ou 6 pares de grow + treino em sequência.

## Memória e tempo (estimativa, nada medido)
- Oficial: o mesmo modelo usou cerca de 3,1 GB por run (ver o YAML `_pcts`). A 200 px, coloque `GPU_CAP_MIB=6000` por job, bem abaixo do teto de 15 GB.
  Comando: `PYTHONPATH=/exps/moliveira/queues/gpucap GPU_CAP_MIB=6000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Jaci: 256 canais em todas as camadas pesam umas 3–4 vezes o oficial. A estimativa é de 8 a 12 GB.
  Hoje a GPU 0 do jaci tem cerca de 14 GB livres (83 de 98 GB ocupados). Por isso fica 1 job por vez, e vale medir no piloto.
- Tempo: não medido. O HAM a 50% tem cerca de 3,5 mil imagens de treino, com 200 épocas por camada e early stopping. Use o piloto para medir.

## Métricas registradas
- No W&B (projeto `phd_thesis_layerwise`), por camada Lk: `train/loss`, `val/loss` (reconstrução), lr e o `epoch_metrics` do módulo.
- A sonda SVM está desligada no treino (`evaluate.SVM: false`). O κ, o F1 e a acurácia balanceada saem depois, numa avaliação SVM separada, uma por vez.
  A validação serve para escolher. O teste é reportado à parte e nunca entra em escolha.

## Bloqueios e o que sincronizar
- **O athena não tem o HAM10000.** Não existe `/dados/home/moliveira/datasets/organized/ham10000`, e o dry-run pula os 3 pontos por falta do split JSON. É preciso trazer do jaci:
  - `datasets/organized/ham10000` (352 MB);
  - `exps/image_cache/ham10000` (1,2 GB), que no athena vai para `/exps/moliveira/image_cache/`.
- `core/constants.py` do athena ganhou a linha `"ham10000": ("orig", 7)`, que o jaci já tinha.
- Para o jaci (rsync seu): os 3 `experiments/autoencoder/hyb_ham10000_jaci*.yaml`. O código do jaci já roda: o dry-run passou no preflight com 3 pontos em `_conf` e em `_rand`.
