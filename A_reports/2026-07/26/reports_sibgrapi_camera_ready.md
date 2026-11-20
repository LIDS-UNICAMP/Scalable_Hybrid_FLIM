# SIBGRAPI camera-ready: custo computacional e significância estatística

**Data:** 2026-07-25 · **git:** `ddd61d7` · **Escopo:** consolidação das medições pedidas pelo Revisor 3

Este documento reúne as duas medições solicitadas pelo Revisor 3 e as ressalvas que as
acompanham. Não é o texto do rebuttal: é o registro interno dos números, da metodologia e dos
pontos frágeis, para servir de base ao rebuttal e à edição do artigo.

Os dois pedidos atendidos:

1. **Custo computacional** dos 8 modelos oficiais: parâmetros, FLOPs, latência de inferência em
   GPU e CPU, memória de pesos e pico de VRAM.
2. **Significância estatística** das diferenças de desempenho contra o FLIM, com teste pareado
   não paramétrico e correção para múltiplas comparações, nas três métricas do artigo
   (F1 ponderado, acurácia, kappa de Cohen).

Modelos avaliados, com o rótulo usado no artigo: FLIM (59.504), LeJEPA (59.504), I-JEPA (632M),
Distill 4 (889K), Distill 3 (615K), Distill 1 (123K), Distill 2 (402K) e
Distill 1 - FLIM init (123K).

---

## 1. Custo computacional

Fonte: `statistics/tools/compute_cost.csv`, gerado por `statistics/tools/measure_compute_cost.py`.
Batch = 1, uma imagem por forward, float32, `model.eval()` + `torch.no_grad()`.

| Modelo | Params | TFLOP/img | GPU (ms/img) | CPU (ms/img) | Pesos (MB) | Pico VRAM (MB) | Pico VRAM sem cuDNN (MB) | Resolução | Pesos medidos |
|:--|--:|--:|--:|--:|--:|--:|--:|:--:|:--|
| FLIM (59.504) | 59.504 | 0,000705 | 0,267 ± 0,008 | 2,912 ± 0,305 | 0,227 | 7,04 | 25,21 | 3×200×200 | checkpoint real |
| LeJEPA (59.504) | 59.504 | 0,000705 | 0,271 ± 0,027 | 3,005 ± 0,171 | 0,227 | 7,04 | 25,21 | 3×200×200 | checkpoint real |
| I-JEPA (632M) | 630.762.240 | 0,33325 | 25,486 ± 0,270 | 496,34 ± 104,51 | 2406,17 | 2486,24 | 2486,24 | 3×224×224 | checkpoint real |
| Distill 4 (889K) | 889.200 | 0,001656 | 0,596 ± 0,035 | 6,063 ± 1,560 | 3,392 | 135,91 | 28,40 | 3×200×200 | **arquitetura (ckpt ausente)** |
| Distill 3 (615K) | 615.024 | 0,001342 | 0,385 ± 0,006 | 4,124 ± 1,103 | 2,346 | 10,76 | 27,34 | 3×200×200 | **arquitetura (ckpt ausente)** |
| Distill 2 (402K) | 402.544 | 0,001096 | 0,460 ± 0,009 | 4,236 ± 0,184 | 1,536 | 133,49 | 26,54 | 3×200×200 | **arquitetura (ckpt ausente)** |
| Distill 1 (123K) | 123.504 | 0,000776 | 0,383 ± 0,007 | 4,025 ± 1,999 | 0,471 | 131,86 | 25,47 | 3×200×200 | **arquitetura (ckpt ausente)** |
| Distill 1 - FLIM init (123K) | 123.504 | 0,000776 | 0,385 ± 0,011 | 3,519 ± 0,062 | 0,471 | 131,86 | 25,47 | 3×200×200 | checkpoint real |

Valores absolutos de FLOPs para leitura direta: 704,8 MFLOP (FLIM, LeJEPA), 775,5 MFLOP
(Distill 1 e Distill 1 - FLIM init), 1,096 GFLOP (Distill 2), 1,342 GFLOP (Distill 3),
1,656 GFLOP (Distill 4), 333,25 GFLOP (I-JEPA). Latência reportada como média ± desvio padrão
amostral; medianas (`gpu_ms_median`, `cpu_ms_median`) estão no CSV.

Razões entre os extremos, tomadas do CSV. Contra o **Distill 1 (123K)**, o student destilado a
partir do I-JEPA, o teacher custa 430× mais FLOPs, 67× mais tempo de GPU, 124× mais tempo de CPU
(medianas) e 5108× mais memória de pesos. Contra o **FLIM (59.504)**, encoder do artigo, o
I-JEPA custa 473× mais FLOPs, 96× mais tempo de GPU e 10.600× mais parâmetros.

### 1.1 Metodologia

- Hardware: uma GPU NVIDIA RTX A6000, ociosa e dedicada à medição; CPU Intel Xeon Gold 5220R
  @ 2,20 GHz, 48 threads PyTorch.
- Software: PyTorch 2.2.2, CUDA 12.1, float32, seed 0, cuDNN benchmark desligado.
- Batch = 1 em todas as medições; o número reportado é por imagem.
- Repetições, com `torch.cuda.synchronize()` antes e depois de cada iteração cronometrada:
  100 iterações em GPU (warmup 20) e 50 em CPU (warmup 10) para os sete modelos leves;
  50 em GPU (warmup 10) e 20 em CPU (warmup 5) para o I-JEPA.
- FLOPs por `torch.utils.flop_counter.FlopCounterMode`, com a convenção **FLOPs = 2 × MACs**
  (multiplicação e soma contadas separadamente) em convoluções, matmuls e atenção. BatchNorm,
  GELU/ReLU, pooling e softmax não entram. A coluna `macs` do CSV traz FLOPs/2 para comparação
  com trabalhos que reportam MACs.
- Memória em duas grandezas distintas: "Pesos" = params × 4 bytes (custo estático de
  armazenamento); "Pico VRAM" = `torch.cuda.max_memory_allocated()` durante um forward com
  batch = 1, com `reset_peak_memory_stats()` antes de cada medição e descontando a alocação
  persistente do contexto CUDA. Inclui pesos residentes, tensor de entrada, ativações e
  workspaces de convolução; não inclui o contexto CUDA em si.
- Grafo medido: apenas o extrator de features, isto é, o que produz o embedding entregue ao SVM
  linear. O SVM linear não entra na conta: no pior caso (eggs, 9 classes, features de 27.648
  dimensões) custa cerca de 2 MFLOP, abaixo de 1% do custo do encoder.

Grafo exato por linha: encoder FLIM → flatten 27.648d (FLIM, LeJEPA); ViT-H/14 → mean-pool 1280d
(I-JEPA); encoder FLIM → cabeça de projeção → GAP 1280d (as quatro variantes Distill, com a
cabeça variando entre conv 1×1, 2× conv 1×1, conv 3×3 e 4× conv 1×1; no Distill 1 - FLIM init o
encoder está congelado).

### 1.2 Ressalvas do custo computacional

**Quatro dos oito modelos foram medidos a partir da arquitetura, não de checkpoint carregado.**
Os `.ckpt` das runs Distill 1, Distill 2, Distill 3 e Distill 4 não existem mais no disco
(`artifacts/distillation/<run>/checkpoints/` está vazio para essas runs; só as runs
`*_flim_frozen` preservaram checkpoint). Essas quatro linhas foram medidas sobre a arquitetura
reinstanciada com init `trunc_normal`. FLOPs, latência, pico de memória e contagem de parâmetros
dependem apenas da topologia da rede, não do valor numérico dos pesos, portanto os números são
idênticos aos que os checkpoints originais produziriam. Os quatro medidos com peso real são
FLIM, LeJEPA, I-JEPA e Distill 1 - FLIM init.

**O pico de VRAM das cabeças de projeção 1×1 → 1280 está inflado por workspace da cuDNN, não por
ativação.** Os ~132 a 136 MB observados em Distill 1, Distill 2, Distill 4 e
Distill 1 - FLIM init são workspace temporário do algoritmo escolhido pela cuDNN. Repetindo a
medição com `torch.backends.cudnn.enabled = False` o pico cai para 25,5 a 28,4 MB, e nessa mesma
configuração o encoder isolado já pica em 25,2 MB: as cabeças 1×1 acrescentam menos de 1 MB de
ativações. As duas leituras estão na tabela (colunas `peak_vram_MB` e
`peak_vram_cudnn_disabled_MB` do CSV). Para as linhas sem cabeça 1×1 o valor sem cuDNN é o maior
dos dois, porque o fallback im2col das convoluções 5×5 do encoder consome mais memória que o
algoritmo da cuDNN. Conclusão prática: o pico de VRAM é dependente de biblioteca e de GPU; a
coluna de **pesos** é a grandeza estável e é ela que deve sustentar qualquer afirmação de
footprint no artigo.

**As resoluções de entrada diferem entre modelos.** 3×200×200 para FLIM, LeJEPA e as quatro
variantes Distill (input do pipeline: LAB via `ift_lab` + normalização ImageNet); 3×224×224 para
o I-JEPA, resolução nativa do ViT-H/14 (`IJEPAEncoder.IMAGE_SIZE`). A diferença é intrínseca aos
modelos e afeta diretamente os FLOPs, por isso está reportada linha por linha.

Outras ressalvas:

- As medições usam a arquitetura do par eggs/larvae (canais 24-32-48, 59.504 parâmetros no
  encoder). Em `protozoan` a seleção de kernels FLIM produz 30 canais na conv2 em vez de 32
  (55.902 parâmetros), o que reduz ligeiramente FLOPs e latência. Essa variante não foi medida
  em separado.
- FLIM e LeJEPA entregam ao SVM o mapa conv3 achatado (48×24×24 = 27.648 dimensões), não um
  vetor pooled, seguindo `src/utils/evaluate.py::extract_features`. Isso não altera os FLOPs de
  convolução, apenas a dimensão vista pelo SVM.
- Latência de CPU medida com 48 threads. Em CPU de 1 thread ou em hardware embarcado os valores
  absolutos mudam, mas a razão entre modelos se mantém. A CPU do nó é compartilhada com outros
  processos, o que explica o desvio padrão alto na linha do I-JEPA; a mediana no CSV é mais
  robusta a essa contenção.

---

## 2. Significância estatística

### 2.1 Desenho

- Teste: Wilcoxon pareado de sinais (signed-rank), bilateral, `zero_method="wilcox"`, p **exato**
  (n efetivo ≤ 25), via `scipy.stats.wilcoxon`.
- Baseline único: **FLIM (59.504)** (`SVM_FLIM`). Desenho **um-contra-todos**: família de
  **7 comparações**, α = 0,05.
- Unidade pareada: célula `(dataset, fração de pré-treino)`, 3 datasets × 6 frações
  (1, 5, 25, 50, 75, 100%) = **n = 18 pares** por comparação. O pareamento usa a chave
  `(dataset, fração)` explicitamente e é verificado antes de cada teste.
- Correção de multiplicidade: **Holm** (principal) e **Bonferroni** (referência conservadora),
  via `statsmodels.stats.multitest.multipletests`.
- Tamanho de efeito: correlação rank-biserial. IC95% da mediana da diferença por bootstrap
  percentil, 10.000 reamostragens das 18 células, semente 42.
- Convenção de sinal em todas as tabelas: **diferença = modelo − FLIM**. Positivo significa que
  o modelo supera o FLIM; negativo, que o FLIM supera o modelo. O `r` rank-biserial segue a mesma
  convenção.
- Filtro obrigatório: `SVM_LeJEPA` aparece no CSV unificado com cinco inicializações
  (flim, he, random, trunc_normal, xavier). Somente `init == trunc_normal` entra no artigo e no
  teste.
- `scikit-posthocs` **não** foi usado. Seu `posthoc_wilcoxon` é all-vs-all (28 comparações com 8
  modelos), enquanto o desenho pedido é um-contra-todos (7 comparações contra o FLIM). A família
  de correção seria outra e o ajuste de p ficaria mais conservador sem necessidade.

### 2.2 Tabela-resumo cruzando as três métricas

Mediana da diferença (modelo − FLIM) e veredito após Holm em cada métrica. `sig` marca
p Holm < 0,05; `ns` marca ausência de significância. As contagens são células em que o modelo
supera o FLIM, de 18.

| Modelo | F1: mediana (Holm) | F1: células | Acurácia: mediana (Holm) | Acc: células | Kappa: mediana (Holm) | Kappa: células | Leitura |
|:--|--:|:--:|--:|:--:|--:|:--:|:--|
| I-JEPA (632M) | +0,0185 (ns) | 13/18 | +0,0175 (ns) | 13/18 | **+0,1039 (sig)** | 18/18 | superior só em kappa |
| Distill 4 (889K) | −0,0225 (ns) | 3/18 | −0,0087 (ns) | 4/18 | **+0,0461 (sig)** | 13/18 | **inverte de sinal entre métricas** |
| Distill 3 (615K) | −0,0523 (ns) | 3/18 | −0,0328 (ns) | 4/18 | +0,0130 (ns) | 10/18 | indistinguível do FLIM |
| Distill 2 (402K) | −0,1029 (ns) | 3/18 | −0,0883 (ns) | 3/18 | −0,0758 (ns) | 6/18 | indistinguível do FLIM |
| Distill 1 - FLIM init (123K) | −0,1568 (ns) | 3/18 | −0,1795 (ns) | 3/18 | −0,0844 (ns) | 7/18 | indistinguível do FLIM |
| LeJEPA (59.504) | **−0,4021 (sig)** | 3/18 | **−0,3194 (sig)** | 3/18 | **−0,4352 (sig)** | 2/18 | FLIM superior nas 3 métricas |
| Distill 1 (123K) | **−0,4955 (sig)** | 2/18 | **−0,4407 (sig)** | 2/18 | **−0,6070 (sig)** | 0/18 | FLIM superior nas 3 métricas |

Contagem de comparações que sobrevivem a Holm: 2 de 7 em F1, 2 de 7 em acurácia, 4 de 7 em
kappa. Sob Bonferroni os mesmos conjuntos se mantêm.

Os dois resultados robustos, presentes nas três métricas, são: o FLIM supera o LeJEPA e supera o
Distill 1 (123K). Todos os demais vereditos dependem da métrica.

### 2.3 F1 ponderado

| Modelo | Mediana dif. | IC95% (mediana) | W | p bruto | p Holm | p Bonferroni | r | Sig. Holm |
|:--|--:|--:|--:|--:|--:|--:|--:|:--:|
| Distill 1 (123K) | −0,4955 | [−0,5978; −0,4407] | 4,0 | 5,34e-05 | 3,74e-04 | 3,74e-04 | −0,953 | sim |
| LeJEPA (59.504) | −0,4021 | [−0,4592; −0,1915] | 13,0 | 6,71e-04 | 0,0040 | 0,0047 | −0,848 | sim |
| Distill 1 - FLIM init (123K) | −0,1568 | [−0,1872; −0,0149] | 35,0 | 0,0268 | 0,1342 | 0,1879 | −0,591 | não |
| I-JEPA (632M) | +0,0185 | [−0,0102; +0,0384] | 51,0 | 0,1415 | 0,5661 | 0,9906 | +0,404 | não |
| Distill 3 (615K) | −0,0523 | [−0,1053; −0,0139] | 51,0 | 0,1415 | 0,5661 | 0,9906 | −0,404 | não |
| Distill 4 (889K) | −0,0225 | [−0,0582; −0,0068] | 51,0 | 0,1415 | 0,5661 | 0,9906 | −0,404 | não |
| Distill 2 (402K) | −0,1029 | [−0,1899; −0,0465] | 51,0 | 0,1415 | 0,5661 | 0,9906 | −0,404 | não |

Mediana de F1 do FLIM nas 18 células: 0,9044.

> **As quatro linhas com W = 51 e p bruto = 0,1415 não são erro de transcrição.** O teste exato
> depende apenas de (W, n), logo comparações com o mesmo W recebem o mesmo p. O I-JEPA chega a
> W = 51 pelo lado W− (13 vitórias do modelo contra 5 do FLIM); Distill 2, 3 e 4 chegam ao mesmo
> W pelo lado W+ (3 vitórias do modelo contra 15 do FLIM, justamente as três de maior
> magnitude). A magnitude do efeito difere entre as quatro linhas; o p, não.

Médias das diferenças, que divergem em sinal das medianas em três casos: I-JEPA +0,0893
[+0,0062; +0,1872]; Distill 4 +0,0183 [−0,0472; +0,0998]; Distill 3 −0,0089
[−0,0808; +0,0785]; Distill 2 −0,0564 [−0,1388; +0,0432]. A divergência vem do regime de dado
escasso descrito em 3.2.

### 2.4 Acurácia

| Modelo | Mediana dif. | IC95% (mediana) | Média dif. | W | p bruto | p Holm | p Bonferroni | r | Sig. Holm |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| Distill 1 (123K) | −0,4407 | [−0,4821; −0,3684] | −0,3947 | 4,0 | 5,34e-05 | 3,74e-04 | 3,74e-04 | −0,953 | sim |
| LeJEPA (59.504) | −0,3194 | [−0,4075; −0,1846] | −0,2596 | 15,0 | 1,05e-03 | 0,0063 | 0,0073 | −0,825 | sim |
| Distill 1 - FLIM init (123K) | −0,1795 | [−0,2320; −0,0182] | −0,1196 | 29,0 | 0,0120 | 0,0602 | 0,0842 | −0,661 | não |
| I-JEPA (632M) | +0,0175 | [−0,0061; +0,0442] | +0,0841 | 48,0 | 0,1084 | 0,4335 | 0,7587 | +0,439 | não |
| Distill 2 (402K) | −0,0883 | [−0,1311; −0,0425] | −0,0383 | 51,0 | 0,1415 | 0,4335 | 0,9906 | −0,404 | não |
| Distill 3 (615K) | −0,0328 | [−0,0793; −0,0132] | −0,0021 | 52,0 | 0,1540 | 0,4335 | 1,0000 | −0,392 | não |
| Distill 4 (889K) | −0,0087 | [−0,0429; −0,0030] | +0,0265 | 53,0 | 0,1674 | 0,4335 | 1,0000 | −0,380 | não |

Nas cinco comparações não significativas o |mediana da diferença| não passa de 0,180, e em 4 das
5 o p bruto já não passaria de 0,05 mesmo sem correção. A exceção é
Distill 1 - FLIM init (p bruto 0,0120), que perde a significância apenas após Holm.

### 2.5 Kappa de Cohen

| Modelo | Mediana dif. | IC95% (mediana) | W | p bruto | p Holm | p Bonferroni | r | Sig. Holm |
|:--|--:|--:|--:|--:|--:|--:|--:|:--:|
| I-JEPA (632M) | +0,1039 | [+0,0759; +0,2004] | 0,0 | 7,63e-06 | 5,34e-05 | 5,34e-05 | +1,000 | sim |
| Distill 4 (889K) | +0,0461 | [+0,0064; +0,0909] | 23,0 | 0,0047 | 0,0190 | 0,0332 | +0,731 | sim |
| Distill 3 (615K) | +0,0130 | [−0,0176; +0,0743] | 53,0 | 0,1674 | 0,5021 | 1,0000 | +0,380 | não |
| Distill 1 - FLIM init (123K) | −0,0844 | [−0,1070; +0,0623] | 58,0 | 0,2462 | 0,5021 | 1,0000 | −0,322 | não |
| Distill 2 (402K) | −0,0758 | [−0,1046; +0,0177] | 64,0 | 0,3692 | 0,5021 | 1,0000 | −0,251 | não |
| LeJEPA (59.504) | −0,4352 | [−0,5572; −0,2071] | 5,0 | 7,63e-05 | 3,81e-04 | 5,34e-04 | −0,942 | sim |
| Distill 1 (123K) | −0,6070 | [−0,7066; −0,5575] | 0,0 | 7,63e-06 | 5,34e-05 | 5,34e-05 | −1,000 | sim |

Kappa médio nas 18 células: FLIM 0,6715; I-JEPA 0,8621; Distill 4 0,7569; Distill 3 0,7296;
Distill 2 0,6650; Distill 1 - FLIM init 0,6541; LeJEPA 0,2946; Distill 1 0,1042. Kappa mediano
do FLIM: 0,8104.

O IC95% bootstrap da mediana exclui zero exatamente nas 4 comparações que sobrevivem a Holm,
em acordo com os p ajustados.

---

## 3. Ressalvas metodológicas

### 3.1 Ausência de significância não é equivalência

Com n = 18 pares, teste bilateral exato e família de 7 hipóteses corrigida por Holm, o menor p
bruto alcançável é 7,63e-06 e o p mais baixo da família é confrontado com α/7 = 0,0071. O teste
só detecta efeito quando o sinal da diferença é quase uniforme ao longo das 18 células.
Diferenças moderadas mas inconsistentes entre frações de pré-treino ficam indetectáveis. Todas
as linhas marcadas "não" acima são **inconclusivas**, não demonstradamente equivalentes ao FLIM.
Os IC95% bootstrap da mediana delimitam a faixa de efeitos ainda compatível com os dados e devem
acompanhar qualquer afirmação de paridade no artigo.

### 3.2 O efeito se concentra em `pretrained_pct = 1`

Para 6 dos 7 modelos (todos exceto o I-JEPA), **todas** as células em que o modelo supera o FLIM
estão na fração de 1% de pré-treino, e de 5% em diante o FLIM ganha em todas as células. Como o
signed-rank pesa posto e não magnitude, o teste agregado sobre as 18 células dilui exatamente o
regime de dado escasso onde os modelos destilados ganham. É essa concentração que produz a
discordância entre mediana e média das diferenças em Distill 2, 3 e 4: a mediana é negativa (o
FLIM ganha no caso típico) enquanto a média fica próxima de zero ou positiva, porque os ganhos em
1% são grandes o bastante para compensar quinze perdas pequenas.

O que está reportado nas seções 2.3 a 2.5 é o **teste agregado sobre as 18 células**, que é o que
o revisor pediu. Nenhuma análise estratificada por fração foi executada, e nenhuma conclusão
nova sobre o regime de 1% é feita aqui. O registro fica como observação metodológica: o desenho
agregado subestima a vantagem em dado escasso, e o artigo não deve usar o teste agregado para
falar sobre o regime de 1%.

### 3.3 Distill 4 (889K) inverte de sinal entre métricas: não resolvido

O Distill 4 tem mediana **negativa** em F1 (−0,0225) e em acurácia (−0,0087), ambas não
significativas, e mediana **positiva e significativa** em kappa (+0,0461, p Holm 0,0190,
r = +0,731). As contagens de células acompanham a inversão: 3/18 vitórias em F1, 4/18 em
acurácia, 13/18 em kappa. Além disso, dentro do kappa o Distill 4 inverte o sinal no dataset
`protozoan` (mediana −0,0115, 1 de 6 células a favor), enquanto é positivo nas 6 células de
`eggs` e nas 6 de `larvae`.

Duas leituras possíveis, e **nenhuma das duas está verificada**:

1. **Substantiva.** Kappa corrige o acordo ao acaso, e os datasets são desbalanceados (eggs com
   9 classes, protozoan com 7, larvae com 2). Um classificador que acerta melhor as classes
   minoritárias pode subir em kappa sem subir em acurácia ou F1 ponderado, que são dominados
   pelas classes de maior suporte. Nesse cenário o resultado em kappa é real e informativo.
2. **Artefato de agregação.** Os CSVs de origem dos 8 modelos vêm de scripts distintos, um por
   modelo (ver seção 5 e `metrics_distillation/data_provenance.md`). Uma diferença de agregação
   ou de definição de métrica entre esses scripts produziria exatamente essa dissociação entre
   kappa e as outras duas métricas.

Consequência prática: **não afirmar no artigo que o Distill 4 supera o FLIM em kappa** sem esta
ressalva explícita, e de preferência sem resolver antes qual das duas leituras vale. A mesma
cautela se aplica, em grau menor, ao Distill 3 (615K), que passa de 3/18 e 4/18 vitórias em F1 e
acurácia para 10/18 em kappa sem atingir significância em nenhuma métrica.

### 3.4 I-JEPA: 18/18 em kappa, 13/18 nas outras duas

O I-JEPA vence em **18 de 18 células em kappa** (r = +1,0, p Holm 5,34e-05), mas em apenas 13 de
18 em acurácia e em F1, com mediana de +0,0175 em acurácia e +0,0185 em F1, ambos com IC95%
cruzando zero e sem significância após Holm. Na análise exploratória por dataset o I-JEPA troca
de sinal em `protozoan` tanto em F1 (mediana −0,0321) quanto em acurácia (−0,0290), mantendo
sinal positivo nas 6 células de `protozoan` em kappa.

O enquadramento honesto e favorável ao artigo: **630 milhões de parâmetros não compram diferença
estatisticamente detectável em acurácia nem em F1 sobre um encoder de 59 mil parâmetros**, ao
custo de 473× mais FLOPs (430× contra o Distill 1) e cerca de 96× mais tempo de inferência em
GPU. A vantagem do I-JEPA se manifesta em kappa, onde é inequívoca, e o artigo deve dizer isso
nesses termos, não afirmar paridade global.

### 3.5 Variabilidade entre splits não entra no teste

Cada célula pareada já é a **média sobre os 3 splits** registrada em
`artifacts/normalized/unified_svm_comparison.csv`. O Wilcoxon trata a heterogeneidade de dataset
e de fração de pré-treino como única fonte de variação pareada; a variância entre splits é
absorvida na média e não contribui para o teste. Isso reduz o ruído das unidades pareadas, mas
significa que os p-valores não refletem a incerteza de amostragem dos splits.

### 3.6 A análise por dataset é exploratória

As três `.md` de Wilcoxon incluem tabelas secundárias com o teste dentro de cada dataset
(n = 6 frações). Com n = 6 o menor p bilateral exato alcançável é 0,0312, o poder é mínimo e
nenhuma correção de multiplicidade foi aplicada. Essas tabelas servem apenas para inspecionar se
o sinal do efeito é consistente entre datasets ou vem de um deles. Não sustentam conclusão
isolada e não devem entrar no artigo como inferência.

---

## 4. Verificação de consistência entre os relatórios

Os quatro relatórios do Grupo 1 foram conferidos linha a linha contra os respectivos `.csv`. Os
valores conferem. Dois pontos de atenção, ambos de forma e não de número:

- O filtro `init == trunc_normal` para `SVM_LeJEPA` está documentado explicitamente no cabeçalho
  de `wilcoxon_kappa.md` e nos docstrings dos três scripts, mas não no corpo de
  `wilcoxon_f1.md` nem de `wilcoxon_acc.md`. O filtro foi aplicado nos três (os scripts o fazem);
  o registro é que a documentação não é uniforme.
- `wilcoxon_f1.md` e `wilcoxon_kappa.md` usam acentuação; `wilcoxon_acc.md` foi gerado sem
  acentos. Diferença puramente cosmética.

A dissociação entre kappa e as outras duas métricas em Distill 4 e Distill 3 (seção 3.3) é a
única inconsistência de substância encontrada, e ela é uma propriedade dos dados de entrada, não
uma discordância entre os relatórios.

---

## 5. Índice de arquivos e reprodução

### 5.1 Arquivos

| Arquivo | Conteúdo |
|:--|:--|
| `statistics/tools/measure_compute_cost.py` | mede params, FLOPs, latência GPU/CPU e VRAM dos 8 modelos |
| `statistics/tools/compute_cost.csv` | saída bruta do custo computacional (inclui `macs`, medianas de latência e `peak_vram_cudnn_disabled_MB`) |
| `statistics/tools/compute_cost.md` | relatório do custo computacional |
| `statistics/tools/wilcoxon_f1.py` / `.csv` / `.md` | Wilcoxon um-contra-todos em F1 ponderado |
| `statistics/tools/wilcoxon_acc.py` / `.csv` / `.md` | Wilcoxon um-contra-todos em acurácia |
| `statistics/tools/wilcoxon_kappa.py` / `.csv` / `.md` | Wilcoxon um-contra-todos em kappa de Cohen |
| `statistics/tools/README.md` | índice do diretório |
| `artifacts/normalized/unified_svm_comparison.csv` | CSV pareado de entrada dos três testes |
| `metrics_distillation/data_provenance.md` | procedência de cada CSV de origem (treino → checkpoint → SVM eval → CSV) |
| `scripts/README_plot_comparison_flim.md`, seção 8 | mapeamento rótulo do artigo → chave `method` → CSV de origem |

### 5.2 Comandos

O ambiente conda `scalable_FLIM` exige o prefixo `LD_LIBRARY_PATH` apontando para as libs do env,
sob pena de `GLIBCXX_3.4.29 not found`.

```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM

# custo computacional (requer GPU; --skip-cpu omite a latência de CPU)
CUDA_VISIBLE_DEVICES=1 \
LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib \
/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python \
    statistics/tools/measure_compute_cost.py

# Wilcoxon nas três métricas (só CPU, segundos cada)
for M in f1 acc kappa; do
  LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib \
  /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python \
      statistics/tools/wilcoxon_${M}.py
done
```

Cada script reescreve seu próprio par `.csv` + `.md` no mesmo diretório. Os três scripts de
Wilcoxon aceitam `--csv`, `--baseline` e `--alpha`; `measure_compute_cost.py` aceita `--skip-cpu`.

Versões: scipy 1.17.1, statsmodels 0.14.6, numpy 1.26.4, pandas 3.0.3, PyTorch 2.2.2, CUDA 12.1.
O `statsmodels` foi instalado no env `scalable_FLIM` para estas análises.
