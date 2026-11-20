# K: benchmark de modelos congelados (K1 I-JEPA, K2 ConvNeXt V2 Pico, K4 DINOv2, K5 AIMv2)

**Estado:** codigo pronto, pesos oficiais baixados, teste em CPU ok, fila `disc_md_bench` lancada (espera `impuro_grid` fechar). K3 DINOv3 nao foi tocado.

## Arquivos
Worktree `/dados/home/moliveira/Scalable_Hybrid_FLIM_discovery`, so arquivos novos: `benchmark/common.py`, `convnextv2.py`, `ijepa.py`, `dinov2.py`, `aimv2.py`. Nenhum .py existente mudou. Sem commit.
Embeddings: `/exps/moliveira/benchmark/emb/<modelo>/<dataset>.npz`. Resultados: `results/benchmark/<modelo>.csv` no worktree. Pesos: `/exps/moliveira/benchmark/hf`. Trava do SVM: `/exps/moliveira/queues/svm.lock` (as filas de SVM do treino devem usar a mesma).
Logs: `/exps/moliveira/queues/logs/bench.log` e `bench.status`.

## Modelos (so repositorios oficiais, revisao fixada)
| | repositorio | revisao | como carrega | dim | params | entrada |
|---|---|---|---|---|---|---|
| K2 | facebook/convnextv2-pico-1k-224 | 90108c4f | transformers 4.45.2, ConvNextV2Model, pooler_output | 512 | 8,6M | 224, crop .875 |
| K1 | facebook/ijepa_vitg16_22k | 2306205e | safetensors em `_IJepaViT` do repo, strict, media dos tokens | 1408 | 1011,4M | 224, crop .9 |
| K4 | facebook/dinov2-giant | 611a9d42 | transformers 4.45.2, Dinov2Model, token CLS | 1536 | 1136,5M | 518, crop 1.0 |
| K5 | apple/aimv2-3B-patch14-448 | 3501a916 | safetensors renomeado para o ViT do timm, media dos tokens | 3072 | 2723,0M | 448, crop 1.0 |

Espelhos `timm/` e o .tar parcial do I-JEPA foram apagados de /exps/moliveira/benchmark. O timm so fornece arquitetura (K5) e receita de pre-processamento; nenhum peso vem dele.

Conferencia contra a implementacao de referencia (`transformers` 5.22.1, env novo uv em `/exps/moliveira/bench_ref`, CPU, fp32, 4 imagens reais; env scalable_FLIM intocado):
- K1 contra `IJepaModel`: erro relativo 9,7e-07, cosseno minimo 0,99999994.
- K5 contra `Aimv2VisionModel`: erro relativo 3,2e-07, cosseno minimo 0,9999999.
Isso compara com o codigo do transformers, nao com o codigo original de pesquisa dos autores (facebookresearch/ijepa, apple/ml-aim), que nao foi rodado.

trust_remote_code: usado 1 vez, so num teste de carga em CPU de apple/aimv2-3B-patch14-448. O `modeling_aimv2.py` do repo nao casa com os nomes do safetensors: carregou tudo aleatorio, sem erro, e deu NaN. Abandonado; a fila nao usa trust_remote_code.

## Para decidir
1. **Pico nao e mais so autossupervisionado.** O unico Pico do `facebook/` no HF e o `-1k-224`: FCMAE + ajuste com rotulo em ImageNet-1k. O D1 pedia `convnextv2_pico.fcmae`, que so existe no espelho timm/. Muda bastante: Larvae, 50%, validacao, media de 3 splits: 0,876 (oficial) contra 0,636 (espelho, antes de apagar).
2. **Pre-processamento:** segui o brief (padrao do timm, corte central). D1 secao 1 dizia letterbox. O corte tira area das raw: Larvae mantem em media 54% (10% piores: 19%), Eggs 78%, Protozoan 85%, flowers 72%. Medido com Pico oficial em Larvae a 50%: corte 0,876, letterbox 0,841. Nao prejudicou esse modelo; os outros nao foram medidos.
3. **AIMv2 tem 3072-d**, acima da regra de leitura < 2000-d; o brief pediu embedding nativo. K4 usa 518 do timm; o preprocessor do repo HF usa 224.
4. Precisao: fp32 com TF32 nas matmuls da GPU, para caber no tempo. Efeito no embedding nao medido.
5. "acc" e a balanceada de `compute_metrics`.

## Medidas de CPU (Pico oficial, Larvae_subgroup e Larvae)
Caminho inteiro ok: extracao, 6 percentuais, CSV com treino/val/teste, recall por classe, tempos. Imagem cortada de aid abre com `LOAD_TRUNCATED_IMAGES`.
Treino a 1% sem alguma classe (a CSV reporta em `classes_missing_train`): Eggs_subgroup s1 e s3, Protozoan_subgroup s1-3, remote_sensing s1-3; a 5%, remote_sensing s1 e s2. Nenhum caso de classe unica; o ramo `one_class` foi testado com dados falsos.

## Regra de GPU
Guarda dentro da extracao, a cada 5 s, por PID: soma do moliveira na GPU 0 acima de 36.000 MiB mata o proprio run e a fila nao segue (testado com teto de 1 MiB, rc 137, sem tocar a GPU). Pico por PID sai em bench.log e bench.status.

## Fila
`disc_md_bench` lancada 01:38 de 06/11. Espera `impuro_grid` (previsao ~02:40: o treino de L4 comecou ~00:11 e os estagios levam ~2h30). Depois, na GPU 0, um modelo por vez: Pico, I-JEPA, DINOv2, AIMv2. Em paralelo, 1 processo de SVM em CPU, sob a trava, `nice -n 10`: 50% de todos os modelos, depois 1, 5, 25, 75, 100%. Relancar o mesmo comando retoma: pula .npz e linhas de CSV prontas.
12 datasets (a conta de D1, 96.239 imagens), 3 splits, 6 percentuais: 216 SVMs por modelo. Tempo do SVM em Protozoan a 3072-d nao medido; solver sem teto, como o oficial.
**Ainda nao exercitado:** o caminho na GPU (so rodou em CPU). Falha aparece como `FAIL` ou `KILL` em bench.status e a fila para.
Ver estado: `tail -5 /exps/moliveira/queues/logs/bench.status`.

Parametros ficam como constantes nos arquivos de benchmark/, sem argparse e sem YAML: o caminho YAML exige mexer em schema.py e nos runners, proibido enquanto as filas rodam.
Sem kappa do run real ainda. Da fumaca em CPU, mesmos pesos oficiais do Pico, validacao a 50%: Larvae 0,876, Larvae_subgroup 0,834.
