# Custo computacional dos 8 modelos oficiais

Resposta ao Revisor 3: FLOPs, tempo de inferência e consumo de memória por modelo, batch=1, uma imagem por forward.

| Modelo | Params | TFLOP/img | Tempo GPU (ms/img) | Tempo CPU (ms/img) | Pesos (MB) | Pico VRAM (MB) | Pesos medidos |
|---|---:|---:|---:|---:|---:|---:|---|
| FLIM (59.504) | 59.504 | 0.000705 (704.8 MFLOP) | 0.267 ± 0.008 | 2.912 ± 0.305 | 0.23 | 7.0 | checkpoint real |
| LeJEPA (59.504) | 59.504 | 0.000705 (704.8 MFLOP) | 0.271 ± 0.027 | 3.005 ± 0.171 | 0.23 | 7.0 | checkpoint real |
| I-JEPA (632M) | 630.762.240 | 0.3332 | 25.49 ± 0.27 | 496.34 ± 104.51 | 2406.17 | 2486.2 | checkpoint real |
| Distill 4 (889K) | 889.200 | 0.001656 (1.66 GFLOP) | 0.596 ± 0.035 | 6.063 ± 1.560 | 3.39 | 135.9 | arquitetura (ckpt ausente) |
| Distill 3 (615K) | 615.024 | 0.001342 (1.34 GFLOP) | 0.385 ± 0.006 | 4.124 ± 1.103 | 2.35 | 10.8 | arquitetura (ckpt ausente) |
| Distill 1 (123K) | 123.504 | 0.000776 (775.5 MFLOP) | 0.383 ± 0.007 | 4.025 ± 1.999 | 0.47 | 131.9 | arquitetura (ckpt ausente) |
| Distill 2 (402K) | 402.544 | 0.001096 (1.10 GFLOP) | 0.460 ± 0.009 | 4.236 ± 0.184 | 1.54 | 133.5 | arquitetura (ckpt ausente) |
| Distill 1 - FLIM init (123K) | 123.504 | 0.000776 (775.5 MFLOP) | 0.385 ± 0.011 | 3.519 ± 0.062 | 0.47 | 131.9 | checkpoint real |

Comparando os dois extremos: o I-JEPA (ViT-H/14, teacher) custa 430x mais FLOPs, 67x mais tempo de GPU, 124x mais tempo de CPU (medianas de latência) e 5108x mais memória de pesos que o Distill 1, o student de 123K parâmetros destilado a partir dele.

## Metodologia

- Hardware: GPU NVIDIA RTX A6000 (uma única GPU, ociosa e dedicada à medição); CPU Intel(R) Xeon(R) Gold 5220R CPU @ 2.20GHz, 48 threads PyTorch.
- Software: PyTorch 2.2.2, CUDA 12.1, float32, `model.eval()` + `torch.no_grad()`, seed 0, cuDNN benchmark desligado.
- Batch size 1 em todas as medições. O número reportado é por imagem.
- Resolução de entrada por modelo: 3x200x200 para FLIM, LeJEPA e as quatro variantes Distill (é o input do pipeline: LAB via `ift_lab` + normalização ImageNet); 3x224x224 para o I-JEPA, resolução nativa do ViT-H/14 (`IJEPAEncoder.IMAGE_SIZE`). A diferença de resolução é intrínseca aos modelos e está refletida nos FLOPs.
- Repetições, com `torch.cuda.synchronize()` antes e depois de cada iteração cronometrada: FLIM (59.504), LeJEPA (59.504), Distill 4 (889K), Distill 3 (615K), Distill 1 (123K), Distill 2 (402K), Distill 1 - FLIM init (123K): 100 iterações cronometradas na GPU (warmup 20) e 50 na CPU (warmup 10); I-JEPA (632M): 50 iterações cronometradas na GPU (warmup 10) e 20 na CPU (warmup 5). Reporta-se média ± desvio padrão amostral sobre as iterações cronometradas; o CSV traz também a mediana.
- FLOPs: `torch.utils.flop_counter.FlopCounterMode`, que conta multiplicação e soma separadamente (FLOPs = 2 x MACs) em convoluções, matmuls e atenção. Não contabiliza BatchNorm, GELU/ReLU, pooling nem softmax, que são desprezíveis frente aos termos multiplicativos. A coluna `macs` do CSV traz FLOPs/2 para comparação com trabalhos que reportam MACs.
- Memória: duas grandezas distintas. "Pesos" = params x 4 bytes (float32), o custo estático de armazenar o modelo. "Pico VRAM" = `torch.cuda.max_memory_allocated()` durante um forward com batch=1, com `reset_peak_memory_stats()` antes de cada medição e descontando a alocação persistente do contexto CUDA (handles de cuBLAS/cuDNN, pré-aquecidos antes do laço); inclui pesos residentes na GPU, tensor de entrada, ativações intermediárias e workspaces temporários de convolução. Não inclui o contexto CUDA em si (algumas centenas de MB), que é custo do runtime e igual para todos os modelos.
- Grafo medido: apenas o extrator de features, isto é, o que produz o embedding entregue ao SVM linear. O SVM não entra na conta: com kernel linear e `decision_function_shape="ovo"` o custo é o produto `n_pares x dim_embedding` (no pior caso, eggs com 9 classes sobre features de 27.648 dimensões, cerca de 2 MFLOP), abaixo de 1% do custo do encoder.

Grafo exato medido em cada linha:

| Modelo | Grafo | Fonte dos pesos |
|---|---|---|
| FLIM (59.504) | encoder FLIM -> flatten (27.648d) | `data/to_mateus/model/ch24_32_48_a0.5_f5/eggs/train1/models` |
| LeJEPA (59.504) | encoder FLIM -> flatten (27.648d) | `logs/flim-ssl/rxe7zmgk/checkpoints/best.ckpt` |
| I-JEPA (632M) | ViT-H/14 -> mean-pool (1280d) | `/dados/home/moliveira/.cache/huggingface/hub/models--facebook--ijepa_vith14_1k/snapshots/f157467ea509bc356ff9f61fd3c0d840eec5e04e/model.safetensors` |
| Distill 4 (889K) | encoder FLIM -> proj 4x conv1x1 -> GAP (1280d) | `arquitetura (checkpoint ausente)` |
| Distill 3 (615K) | encoder FLIM -> proj conv3x3 -> GAP (1280d) | `arquitetura (checkpoint ausente)` |
| Distill 1 (123K) | encoder FLIM -> proj conv1x1 -> GAP (1280d) | `arquitetura (checkpoint ausente)` |
| Distill 2 (402K) | encoder FLIM -> proj 2x conv1x1 -> GAP (1280d) | `arquitetura (checkpoint ausente)` |
| Distill 1 - FLIM init (123K) | encoder FLIM congelado -> proj conv1x1 -> GAP (1280d) | `artifacts/distillation/distillation_eggs_split1_pct100_1x1_BN2d_1280_flim_frozen/checkpoints/best_loss.ckpt` |

## Ressalvas

- Os checkpoints das quatro variantes Distill com encoder treinável (Distill 1, 2, 3 e 4) não estão mais no disco: `artifacts/distillation/<run>/checkpoints/` está vazio para essas runs (apenas as runs `*_flim_frozen` preservaram `.ckpt`). Essas linhas foram medidas sobre a arquitetura reinstanciada com init `trunc_normal`. FLOPs, latência, pico de memória e contagem de parâmetros dependem só da topologia da rede, não do valor numérico dos pesos, portanto os números são idênticos aos que os checkpoints originais produziriam.
- O pico de VRAM das cabeças de projeção com conv 1x1 para 1280 canais é dominado por um workspace temporário que a cuDNN reserva para o algoritmo que escolhe, não pelas ativações. Repetindo a medição com `torch.backends.cudnn.enabled = False`, o pico cai assim: Distill 4 (889K): 135.9 -> 28.4 MB; Distill 1 (123K): 131.9 -> 25.5 MB; Distill 2 (402K): 133.5 -> 26.5 MB; Distill 1 - FLIM init (123K): 131.9 -> 25.5 MB. Nessa mesma configuração o encoder isolado já pica em 25.2 MB, ou seja, as cabeças de projeção 1x1 acrescentam menos de 1 MB de ativações: os ~130 MB observados são workspace da cuDNN, não dado. Os valores com cuDNN desabilitada estão na coluna `peak_vram_cudnn_disabled_MB` do CSV; note que para as linhas sem cabeça 1x1 esse número é maior que o pico com cuDNN, porque o fallback im2col das convoluções 5x5 do encoder consome mais memória que o algoritmo da cuDNN. Em resumo, o pico de VRAM é dependente de biblioteca e de GPU; a coluna de pesos é a grandeza estável.
- As medições usam a arquitetura do par eggs/larvae (canais 24-32-48, 59.504 parâmetros no encoder). Para `protozoan` a seleção de kernels FLIM produz 30 canais na conv2 em vez de 32 (55.902 parâmetros no encoder), o que reduz ligeiramente FLOPs e latência. Essa variante não foi medida em separado.
- FLIM e LeJEPA entregam ao SVM o mapa conv3 achatado (48x24x24 = 27.648 dimensões), não um vetor pooled, seguindo `src/utils/evaluate.py::extract_features`. Isso não altera os FLOPs de convolução, apenas a dimensão vista pelo SVM.
- Latência de CPU medida com 48 threads (padrão do PyTorch nesta máquina, igual ao número de núcleos físicos); em CPU de 1 thread ou em hardware embarcado os valores absolutos mudam, mas a razão entre modelos se mantém. A GPU usada estava livre durante a medição, mas a CPU do nó é compartilhada com outros processos, o que explica o desvio padrão alto na coluna de CPU do I-JEPA; o CSV traz também a mediana (`cpu_ms_median`, `gpu_ms_median`), mais robusta a essa contenção.
- Gerado por `statistics/tools/measure_compute_cost.py`; dados brutos em `statistics/tools/compute_cost.csv`.
