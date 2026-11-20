# Auditoria inicial do SPiFiL para a comparação multiescala

Leitura em 07/11/2026, 15:15 a 15:25 UTC. Worktree `Scalable_Hybrid_FLIM_discovery`, branch `discovery`, HEAD `0de1838`, com muito código ainda não commitado. SPiFiL em `4845021`, sem mudança de código.

Convenções: `SPiFiL:` = `/dados/home/moliveira/SPiFiL/src/spifil/`. `SVA` = `spifil_vs_aleatorio.md`. As linhas de `eval/spf_grade.py` valem para a versão de 15:00:45, com 838 linhas (sha256 `52bf75f54d66`). Às 15:26 o arquivo passou a 942 linhas com os braços S1-S8: nele, procure pelo nome da função.

O grafo não tem `spf_grade`, `pw_whiten` nem os outros arquivos novos, e `graphify update` escreveria em `graphify-out/`. Por isso li esses arquivos direto e usei grep nos nomes que o grafo não achou.

`flim/scale_space.py` e os braços S1-S8 entraram depois desta leitura e ficam fora da auditoria. `flim/filter_diversity.py` apareceu às 15:16 e também não foi auditado.

## 1. Arquitetura atual e pipeline de inicialização

O SPiFiL vive em três camadas:

| Camada | Onde | Papel |
|---|---|---|
| Biblioteca | `SPiFiL:` | superpixel, semente, pontuação, seleção e banco de filtros |
| Fachada | `flim/spifil.py` | `grow` acrescenta uma camada e grava no formato FLIM (`flim/spifil.py:778`, `:680`) |
| Grade sem backprop | `eval/spf_grade.py` | compara o conteúdo do kernel entre braços (`eval/spf_grade.py:18-23`) |

Rede da grade, sem gradiente:

| Item | Valor | Fonte |
|---|---|---|
| Entrada | 3 bandas LAB em [0,1], 200 px, letterbox | `core/data/loaders.py:49-55`, `core/data/transforms.py:33-40`, `disc_md_spf_r1.yaml:15-16` |
| Bloco | conv com padding `k//2`, ReLU, MaxPool 3 sem padding | `flim/arch.py:99-110` |
| Kernel | 5 | `disc_md_spf_r1.yaml:29` |
| Larguras | 64 e 256, iguais em todas as camadas | `disc_md_spf_r1.yaml:28`, `eval/spf_grade.py:748` |
| Profundidade | 3, ou 8 em `disc_md_spfdeep_cedo` | `disc_md_spf_r1.yaml:30`, `disc_md_spfdeep_cedo.yaml` (`depth: 8`) |
| Pool | stride 2; em `cedo`, 2-2-2-2-1-1-1-1 | `disc_md_spf_r1.yaml:31`, `disc_md_spfdeep_cedo.yaml` |
| Mapa a 200 px | 99, 49, 24, 11 | `eval/spf_tabela.py:60` |

Na biblioteca, da imagem ao peso (`Learner.fit`, `SPiFiL:learner.py:392`):

1. Converte para Lab, roda o superpixel e tira uma semente por região (`SPiFiL:preparation.py:73-83`).
2. Recorta um patch por semente (`SPiFiL:patches.py:86-92`).
3. Pontua com Fisher sobre Mahalanobis e ranqueia por classe (`SPiFiL:learner.py:368-371`, `SPiFiL:scoring.py:95-109`).
4. Reparte `out_channels // n_classes` por classe (`SPiFiL:allocation.py:53`).
5. Escolhe por rank e diversidade (`SPiFiL:selection.py:155-194`).
6. Transforma os patches escolhidos em conv (`SPiFiL:nn/builder.py:119-129`).

Não há K-means no aprendizado de filtros: o filtro é o próprio patch normalizado. O único K-means está dentro do SLIC do skimage, com máscara (`skimage/segmentation/slic_superpixels.py:40,44,60`), e a grade usa DISF. O superpixel não tem RNG no código do SPiFiL; num teste em CPU, o DISF deu hash idêntico em 2 processos, numa imagem sintética.

Na grade, um job é (dataset, split, tamanho, fatia) e uma célula é (braço, larguras, kernel, sorteio) (`eval/spf_grade.py:20-21`):

1. Sorteia até 500 imagens do treino, sem olhar classe (`eval/spf_grade.py:813`).
2. `sources` prepara o LAB, o DISF, os pontos e o corte de fundo (`eval/spf_grade.py:324-356`).
3. `chain` constrói a rede camada a camada. Em cada uma, mede média e desvio, pede as direções `u` ao braço, dobra em conv, grava e remonta o encoder (`eval/spf_grade.py:665-698`).
4. Cada camada é lida por GeM, whitening e SVM (`eval/spf_grade.py:702`, `:715-726`).

Onde a grade difere da biblioteca:

| Item | Biblioteca | Grade | Fonte |
|---|---|---|---|
| Superpixel | SLIC, 100 | DISF, 50 | `SPiFiL:learner.py:211,221`; `disc_md_spf_r1.yaml:17,21` |
| Kernel | 3 | 5 | `SPiFiL:config.py:25`; `disc_md_spf_r1.yaml:29` |
| Imagens | as do dataset | 500 sorteadas; o braço `spifil` usa só `k` por classe, com `k >= 3` | `eval/spf_grade.py:813`, `:626`, `:636` |
| Rótulo | sim, por classe | só no braço `spifil` e nos oráculos | `SPiFiL:selection.py:120-122`; `eval/spf_grade.py:41-44` |
| Z-score | média e desvio dos filtros escolhidos | média e desvio de todas as sementes medoide | `SPiFiL:nn/builder.py:119-122`; `eval/spf_grade.py:669`, `:673` |
| Pool | padding 1 | sem padding | `SPiFiL:nn/blocks.py:185-187`; `flim/arch.py:110` |
| Máscara | opcional | desligada | `eval/spf_grade.py:327`, `flim/spifil.py:903` |

O LAB tem dois caminhos. Superpixel, semente e SIFT usam o `LabNorm` do SPiFiL sobre a imagem já em letterbox (`eval/spf_grade.py:327-331`). Os patches e a rede usam `ift_lab_loader` (`eval/spf_grade.py:328`, `flim/spifil.py:563`). Não medi a diferença.

## 2. Seleção de pontos e de patches

Tudo isto escolhe onde recortar. Na grade, o braço é `metodo@pontos`, e sem `@` vale medoide (`eval/spf_grade.py:25`, `:646`, `:649`).

| Ponto | O que faz | Rótulo? | Fonte |
|---|---|---|---|
| `medoid` | pixel com a cor mais perto da média do superpixel; é o default | não | `SPiFiL:superpixels/centers.py:87` |
| `centroid` | média das coordenadas; pode cair fora da região | não | `SPiFiL:superpixels/centers.py:52-57` |
| `geodesic` | centro geodésico da forma; exige PyIFT | não | `SPiFiL:superpixels/centers.py:21-49` |
| `sift` | keypoints mais fortes da imagem, até 50 | não | `eval/spf_grade.py:207`, `:210` |
| `sift_sp` | keypoint mais forte de cada superpixel | não | `eval/spf_grade.py:208`, `:210` |
| `sift_nms` | mantém um keypoint por vizinhança; raio de 14,1 px com 50 pontos a 200 px | não | `eval/spf_grade.py:192-200`, `:209` |
| `uniao` | medoide mais `sift_sp` | não | `eval/spf_grade.py:212` |
| `dic` | dicionário SIFT: um keypoint real por palavra | não | `eval/spf_grade.py:292-295` |
| `combo`, `combo_s*` | união de geodesic, centroid, sift_nms, obj e dic; `_s*` tira um | não | `eval/spf_grade.py:296-306` |
| `obj`, `fundo` | medoides de variância local alta e baixa, corte por Otsu | não | `eval/spf_grade.py:221-225`, `:342`, `:346` |
| `sal` | superpixel cuja cor foge da cor da borda | não | `eval/spf_grade.py:233-248` |
| `l1obj` | energia de um banco L1 provisório de 64 filtros | não | `eval/spf_grade.py:310-321` |
| `mapa_*`, `nms_*` | corte ou peso por mapa de saliência | não | `eval/spf_grade.py:270-280` |
| `m_obj`, `m_fundo`, `m_nms` | oráculos: filtram pela máscara, só diagnóstico | sim | `eval/spf_grade.py:41-44`, `:280`, `:350` |

`obj` não acha objeto: ele acha textura. Quem usa o rótulo verdadeiro é `m_obj`. A biblioteca não tem seleção de pontos de objeto: a máscara só restringe o superpixel (`SPiFiL:preparation.py:74`).

Nome parecido, coisa diferente: `uniao` é medoide + SIFT, mas `SeedPoints("union")` são os 3 centros (`flim/seed_points.py:49-50`).

Depois do ponto vem o método, que escolhe entre os patches candidatos. Antes dele, `Ctx.keep` tira janela chapada e repetida e sorteia até 5000 linhas (`eval/spf_grade.py:429-442`, `eval/l1_sem_rotulo.py:71`).

| Método | O que faz | Fonte |
|---|---|---|
| `crus` | os primeiros patches da ordem sorteada | `eval/l1_sem_rotulo.py:180-181` |
| `fps` | farthest point em `abs(cos)`, a partir da linha 0 | `eval/l1_sem_rotulo.py:184-189` |
| `fpsw` | FPS com a saliência multiplicando a distância | `eval/spf_grade.py:486-493` |
| `fpsd`, `cosd`, `bovw_*` | a mesma ideia sobre o descritor SIFT de 128-d | `eval/spf_grade.py:551-554` |
| `qr`, `cos` | QR pivotado; poda por cosseno > 0,9 com sinal | `eval/l1_sem_rotulo.py:197-199`, `:228-233` |
| `km_*`, `hdbscan*` | agrupam; devolvem representante ou média | `eval/spf_grade.py:545-550` |
| `zca2*`, `zca_km2`, `pca*` | direções em espaço branqueado ou PCA | `eval/spf_grade.py:500-511`, `:533-537` |
| `spifil`, `spifil_dd` | o `grow` oficial; `_dd` tira cópias | `eval/spf_grade.py:621-639`, `:682-684` |

O FPS não existe na biblioteca. O mais perto é `DiversitySelector` com `alpha=0`, que começa no melhor rank (`SPiFiL:selection.py:169-170`).

## 3. Datasets

Contei com `ParasiteDataset` às 15:17 UTC, em splits 1 a 3 e percentuais 1, 5, 10, 25, 50, 75 e 100. A resolução vem de 100 imagens sorteadas (`random.Random(0)`), lendo só o cabeçalho.

| Dataset | Pasta | Classes | Imagens | Menor classe | Maior classe |
|---|---|---|---|---|---|
| Eggs | `raw/orig` | 9 | 12.691 | H.diminuta 83 | impurezas 9.815 (77,3%) |
| Protozoan | `raw/orig` | 7 | 37.376 | B.hominis 309 | impurezas 28.528 (76,3%) |
| Larvae | `raw/orig` | 2 | 1.598 | S.stercoralis 247 | impurezas 1.351 (84,5%) |
| aid | `orig` | 30 | 10.000 | BaseballField 220 | Pond e Viaduct 420 (4,2%) |
| Protozoan_subgroup | `images` | 7 | 7.919 | E.histolytica 16 | impurezas 4.457 (56,3%) |
| Eggs_subgroup | `images` | 9 | 3.974 | E.vermicularis 48 | impurezas 2.579 (64,9%) |
| flowers | `orig` | 5 | 2.746 | sunflower 495 | dandelion 646 (23,5%) |
| protozoan_cysts | `orig` | 7 | 9.568 | E.histolytica 78 | impurezas 5.716 (59,7%) |
| remote_sensing | `orig` | 14 | 1.000 | River 20 | City 173 (17,3%) |
| helminth_eggs | `orig` | 9 | 5.112 | H.diminuta 80 | impurezas 3.344 (65,4%) |
| helminth_larvae | `orig` | 2 | 3.514 | Strongyloides 446 | impurezas 3.068 (87,3%) |
| Larvae_subgroup | `images` | 2 | 741 | S.stercoralis 137 | impurezas 604 (81,5%) |

Classes e nomes vêm do `manifest.json`. Os 3 subgrupos não têm manifest: a classe sai do prefixo do nome (`config.py:116-126`, `core/data/parasite_dataset.py:119`), e a tabela usa os nomes de `mapa_ids.csv`. A pasta vem de `core/constants.py:69-82`.

Divisões, sempre treino/validação/teste a 100%, e treino a 50%:

| Dataset | Split 1 | Split 2 | Split 3 | Treino a 50% (s1/s2/s3) | Menor classe no treino a 50% |
|---|---|---|---|---|---|
| Eggs | 8.885/1.903/1.903 | igual | igual | 4.448 | 30 |
| Protozoan | 26.162/5.607/5.607 | igual | igual | 13.086 | 109 |
| Larvae | 1.118/240/240 | igual | igual | 560 | 87 |
| aid | 6.996/1.502/1.502 | igual | igual | 3.498 | 77 |
| Protozoan_subgroup | 5.532/1.190/1.197 | 5.542/1.186/1.191 | 5.559/1.187/1.173 | 2.766/2.791/2.764 | 7/5/5 |
| Eggs_subgroup | 2.782/593/599 | 2.764/607/603 | 2.787/598/589 | 1.377/1.397/1.390 | 16/21/18 |
| flowers | 1.922/412/412 | igual | igual | 963 | 174 |
| protozoan_cysts | 6.698/1.435/1.435 | igual | igual | 3.351 | 27 |
| remote_sensing | 682/159/159 | 708/140/152 | 699/147/154 | 344/357/351 | 9/5/8 |
| helminth_eggs | 3.580/766/766 | igual | igual | 1.791 | 28 |
| helminth_larvae | 2.460/527/527 | igual | igual | 1.230 | 156 |
| Larvae_subgroup | 519/107/115 | 507/118/116 | 522/109/110 | 246/256/265 | 46/47/53 |

Resolução e máscara. "Preto" é a fração de pixels com `max(RGB) <= 8` no arquivo original, antes do letterbox. Área é a fração de pixels da máscara. Ambos são medianas das 100 imagens.

| Dataset | Largura mín/med/máx | Altura mín/med/máx | Preto | Máscara em disco | Área |
|---|---|---|---|---|---|
| Eggs | 160/368/944 | 192/384/1040 | 0,000 | `raw/label`, 12.691 | 0,248 |
| Protozoan | 48/112/240 | 48/112/240 | 0,000 | `raw/label`, 37.376 | 0,126 |
| Larvae | 192/992/2576 | 144/756/2336 | 0,000 | `raw/label`, 1.598 | 0,094 |
| aid | 600 | 600 | 0,000 | não | - |
| Protozoan_subgroup | 200 | 200 | 0,114 | `masks`, 7.919 | 0,108 |
| Eggs_subgroup | 200 | 200 | 0,190 | `masks`, 3.974 | 0,203 |
| flowers | 159/320/500 | 211/240/417 | 0,002 | não | - |
| protozoan_cysts | 200 | 200 | 0,973 | `label`, 9.568 | 0,027 |
| remote_sensing | 640 | 640 | 0,000 | não | - |
| helminth_eggs | 200 | 200 | 0,978 | `label`, 5.112 | 0,022 |
| helminth_larvae | 200 | 200 | 0,981 | `label`, 3.514 | 0,019 |
| Larvae_subgroup | 200 | 200 | 0,355 | `masks`, 741 | 0,045 |

Nos 9 com máscara, todo arquivo tem máscara de mesmo nome, e as 100 da amostra têm o tamanho da imagem. O modelo não as lê: `config.py:107` fixa `masks: None`. A porta de diagnóstico `MASK_DIR` só cobre 6 datasets; `helminth_eggs`, `helminth_larvae` e `protozoan_cysts` têm máscara em disco e ficam fora dos oráculos (`eval/spf_grade.py:124-125`).

O que a contagem confirmou:

- Em todos os 12, nenhum arquivo falta, nenhum nome repete, e treino, validação e teste não se cruzam.
- Validação e teste não mudam com o percentual. O treino de um percentual está contido no seguinte, e 100% é igual a `splits/split{n}.json`.
- `raw/` está em uso. `core/constants.py:70-72` aponta para `raw/orig`, os nomes dos splits existem lá, e 6 de 6 imagens de `/exps/moliveira/image_cache/<dataset>/200_nocrop` batem byte a byte com o pipeline em Eggs, Protozoan, Larvae, Eggs_subgroup e aid.
- O cache tem 8 tamanhos, de 64 a 640 `nocrop`, em todos os 12, com o número certo de arquivos. Larvae tem também 896 e 1280. Não há cache com crop.

O que merece cuidado:

- **Os 12 não são independentes.** São 3 famílias de parasitos mais aid, flowers e remote_sensing. `mapa_ids.csv` liga cada subgrupo ao raw e à versão de fundo preto: Eggs_subgroup tem 3.974 de 12.691 do Eggs e 3.974 de 5.112 do helminth_eggs; Protozoan_subgroup tem 7.919 de 37.376 e 7.919 de 9.568; Larvae_subgroup tem 741 de 1.598 e 741 de 3.514. Os pixels não são iguais: 0 de 100 idênticos.
- **`*_cysts` e `helminth_*` são quase só preto**, cerca de 97% do quadro. Os outros parasitos têm fundo real.
- **A escala do objeto varia muito depois do letterbox.** A 200 px, Protozoan sobe de cerca de 112 px, e Larvae desce de cerca de 992 px, sem nenhuma normalização de tamanho do objeto.
- **Os 3 splits não são folds.** São reamostragens: os testes se cruzam, por exemplo em 275 de 1.903 entre s1 e s2 de Eggs.
- **Classes muito raras** deixam κ e acurácia ruidosos: em Protozoan_subgroup a menor classe tem 2 a 4 imagens na validação e 1 a 5 no teste; em remote_sensing, 2 a 4 e 1 a 3.
- **A 1%, no split 1, falta classe no treino:** Protozoan_subgroup 6 de 7, Eggs_subgroup 7 de 9, remote_sensing 11 de 14 (13 de 14 a 5%). Larvae_subgroup tem 3 imagens.
- **500 imagens nem sempre existem a 50%:** Larvae_subgroup e remote_sensing têm menos, e a grade usa todas (`eval/spf_grade.py:813`).
- **Só remote_sensing é agrupado** (568 grupos em 1.000 imagens, `split_report.json`). Por isso os tamanhos variam entre splits. Os subgrupos também variam, e não têm `split_report.json`.
- **2 rótulos conflitam em Protozoan_subgroup.** `000001_00000274.png` e `000001_00000564.png` entram como E.coli pelo prefixo, mas o mapa diz E.histolytica (`mapa_ids.csv:937`, `:1222`). Essa classe só tem 16 imagens.
- **O id da impureza muda:** Eggs 8, Protozoan 6, Larvae 1, helminth_eggs 5, protozoan_cysts 5, helminth_larvae 0. `IMPURITY` só cobre os 6 de `eval/spf_grade.py:122`, então as colunas de impureza ficam vazias nos outros.
- **Existem mais datasets em `organized/`** (coconut_trees, fer2013, food101, 6 medmnist). Estão fora dos 12.
- **`pre/` tem outra numeração.** Só 1.555 de 12.691 nomes de Eggs, 1.910 de 37.376 de Protozoan e 1.351 de 1.598 de Larvae coincidem com `raw/orig`.

## 4. Normalização, aplicação e propagação

Na grade, a mesma dobra vale para todos os braços. É isso que torna a comparação justa: só muda `u`.

| Passo | Conta | Fonte |
|---|---|---|
| Referência | patches de todas as sementes medoide, nos mapas do próprio braço | `eval/spf_grade.py:669` |
| Média e desvio | por dimensão; desvio populacional + 0,01 | `eval/spf_grade.py:673`, `:109` |
| Direção | `u` com norma 1 | `eval/spf_grade.py:687` |
| Peso | `K = u / desvio` | `eval/spf_grade.py:688` |
| Bias | `b = -K . média` | `eval/spf_grade.py:689` |

A conv então mede `u . z(x)`, com `z(x)` sendo o z-score da janela. O peso final `K` não tem norma 1, e cada filtro não tem média zero (`SPiFiL:nn/builder.py:124-129`).

Isso vale também para o controle aleatório: a direção é sorteada, mas média e desvio vêm dos dados (`eval/spf_grade.py:561-571`, `:680-681`). Ele não é "sem dados".

Um formato fixo posto em `u` não sai igual no peso, porque cada posição e banda é dividida pelo seu desvio (`eval/spf_grade.py:688`). Uma Gaussiana entregue como `u` sai como Gaussiana dividida pelo desvio.

Com uma exceção, nada normaliza entre camadas. A exceção é `entre_camadas`: média e ZCA dos canais nas sementes, dobradas na conv seguinte (`eval/spf_grade.py:594-613`, `:670-676`, `:690-691`).

Como o filtro chega à rede:

1. `write_weights` grava `conv{n}-kernels.npy` e `conv{n}-bias.txt` (`eval/spf_grade.py:694-695`, `flim/spifil.py:680-698`).
2. `grown_arch` acrescenta a camada; stride 1 vira sem pool (`flim/spifil.py:701-729`).
3. `build` remonta o encoder (`eval/spf_grade.py:698`, `flim/build.py:38-56`).

Propagação: na camada seguinte, a imagem passa pelo encoder pronto, a semente é reposicionada na grade nova por conta proporcional, e o patch sai dos mapas (`eval/l1_sem_rotulo.py:113-120`, `flim/spifil.py:476-491`). Sementes que caem no mesmo pixel viram linhas repetidas; `Ctx.keep` as tira da escolha, mas elas seguem em `ref` e entram na média e no desvio (`eval/spf_grade.py:434-436`, `:673`).

A dimensão do patch é canais vezes 25: 75 na L1, 1.600 na L2 de 64 e 6.400 na L2 de 256.

Largura real pode sair menor que a pedida, por exemplo PCA acima do posto. A grade então roda `twin_controls` naquela largura exata (`eval/spf_grade.py:831-833`). Na L1 de 256 há mais filtros que dimensões (256 contra 75).

## 5. Pooling e classificação

Dentro da rede só há MaxPool 3 sem padding, com stride do YAML (`flim/arch.py:109-110`, `eval/spf_grade.py:616-618`).

A leitura da grade é única:

| Passo | Valor | Fonte |
|---|---|---|
| Pool global | GeM com p=3 | `eval/spf_grade.py:107`, `eval/l1_amostras.py:62`, `eval/svm.py:359-361` |
| Whitening | PCA no treino, shrinkage 0,1 da variância média, depois L2 | `eval/svm.py:365-373` |
| Classificador | SVC linear, C=100, `ovo`, sem limite de iterações, sem `class_weight`, sem scaler | `core/svm.py:143-186` |
| Métricas | κ, `acc`, F1 macro | `core/metrics.py:72-76` |
| Conjuntos | treino e validação; o teste não sobe | `eval/l1_amostras.py:59`, `eval/l1_fast.py:80` |

Saem até 4 linhas por camada:

- `L`: só a camada atual. Não tem teto de dimensão (`eval/spf_grade.py:715`).
- `cat`: concatena da última para trás enquanto a soma fica abaixo de 2000-d (`eval/spf_grade.py:717-724`).
- `soma`: as camadas somadas depois de L2, com `amostra.soma` (`eval/spf_grade.py:725-726`).
- `<nome>_b`: sem `pw_whiten`, com `entre_camadas` (`eval/spf_grade.py:712-713`).

`acc` é acurácia balanceada, com uma ressalva. Com torchmetrics 1.9.0, `average` tem default `macro`. Num teste com classes desbalanceadas deu 0,5833, igual ao `balanced_accuracy_score`, contra 0,93 de acerto simples. Só diverge se uma classe for prevista sem existir em `y_true`: 0,600 contra 0,900. Em validação e teste dos 12 datasets, nos 3 splits, toda classe aparece, então aqui `acc` é balanceada.

O SVM não pesa classes, e as impurezas são 56% a 87% em 9 dos 12 datasets (item 3). Por isso a grade guarda `imp_recall`, `par_recall`, `imp_kappa` e `par_kappa` (`eval/spf_grade.py:585-590`).

Existem outras leituras que a grade não usa: `avgpool2d`, `flatten`, `spp`, `l2flat`, `gem_dense`, `avgpool2d_fg`, `gem_fg`, `gem_tta` (`eval/svm.py:309-310`).

Divergências:

- O docstring de `_encode_pooled` só cita `avgpool2d` e `flatten`; o código tem 9 modos (`eval/svm.py:316-323`, `:309`).
- O docstring da grade diz "abaixo de 2000-d" para a leitura (`eval/spf_grade.py:47-48`), mas o teto só vale para `cat`.
- Retomada: `expected` conta `cat` em toda camada (`eval/spf_grade.py:763`). Com largura de 1.000 ou mais, `cat` não sai, a célula nunca fecha e é refeita a cada relançamento. Com 64 e 256 não acontece.

## 6. Experimentos anteriores e resultados

Leitura às 15:23 UTC. SVA muda sozinho: `### Tabela` estava na linha 2398 neste momento e cresce. Ache cada número pela chave `| base | real | método | largura | camadas | leitura`. O SVA se dizia atualizado às 15:23, com 11.345 leituras (SVA:56).

Protocolo comum: 50%, 200 px, `no_crop`, 50 superpixels DISF, kernel 5, GeM + whitening + SVM, κ de validação (SVA:3, SVA:56, `disc_md_spf_r1.yaml`).

Base de comparação: o melhor controle da célula, na mesma largura real, escolhido pelo maior κ médio (`eval/spf_tabela.py:24-26`). Controles: xavier, he, orth, tnorm, random_z, whrand, rz2, whrand2 (`disc_md_spf_r1.yaml:33`).

Critério fixado antes: Δ ≥ +0,03 com intervalo acima de 0 em pelo menos 3 de 5 bases: flowers, helminth_eggs, helminth_larvae, Protozoan_subgroup, aid (SVA:9, `eval/spf_tabela.py:29`, `:48`).

**Resposta curta: nenhum método cumpre o critério, e hoje ele não pode ser cumprido.** Só 2 das 5 bases têm dados. O máximo é 1 base boa (SVA:64-76).

Comparação direta, κ de validação, 3 camadas, leitura `cat`:

| Base | Método | Largura | κ | Melhor aleatório | Δκ | IC 95% | Splits |
|---|---|---|---|---|---|---|---|
| flowers | spifil | 256 | 0,560 | rz2 | -0,059 | [-0,210; +0,092] | 3 |
| flowers | spifil | 64 | 0,554 | orth | +0,006 | [-0,051; +0,063] | 3 |
| flowers | crus | 256 | 0,586 | rz2 | -0,032 | [-0,158; +0,093] | 3 |
| flowers | crus | 64 | 0,523 | orth | -0,025 | [-0,108; +0,059] | 3 |
| flowers | fps | 256 | 0,613 | rz2 | -0,006 | [-0,162; +0,151] | 3 |
| flowers | fps | 64 | 0,566 | orth | +0,018 | [-0,048; +0,085] | 3 |
| Protozoan_subgroup | spifil | 256 | 0,773 | tnorm | +0,043 | [+0,021; +0,065] | 3 |
| Protozoan_subgroup | spifil | 64 | 0,654 | he | +0,004 | [-0,054; +0,061] | 3 |
| Protozoan_subgroup | crus | 256 | 0,778 | tnorm | +0,049 | [+0,030; +0,067] | 3 |
| Protozoan_subgroup | crus | 64 | 0,729 | he | +0,078 | [+0,033; +0,124] | 3 |
| Protozoan_subgroup | fps | 256 | 0,813 | tnorm | +0,083 | [+0,063; +0,104] | 3 |
| Protozoan_subgroup | fps | 64 | 0,750 | he | +0,100 | [+0,068; +0,131] | 3 |
| Eggs_subgroup | spifil | 256 | 0,756 | orth | -0,013 | [-0,049; +0,022] | 3 |
| Eggs_subgroup | spifil | 64 | 0,687 | orth | +0,034 | [-0,053; +0,121] | 3 |
| Eggs_subgroup | crus | 256 | 0,722 | orth | -0,047 | [-0,121; +0,027] | 3 |
| Eggs_subgroup | crus | 64 | 0,667 | orth | +0,013 | [-0,073; +0,100] | 3 |
| Larvae_subgroup | spifil | 256 | 0,624 | random_z | +0,147 | [-0,567; +0,861] | 2 |
| Larvae_subgroup | spifil | 64 | 0,578 | he | +0,184 | [-0,124; +0,492] | 3 |
| Larvae_subgroup | crus | 256 | 0,669 | orth | +0,270 | [-0,023; +0,564] | 3 |
| Larvae_subgroup | crus | 64 | 0,571 | he | +0,134 | [-0,792; +1,060] | 2 |
| Larvae | spifil | 256 | 0,714 | random_z | +0,258 | [-0,175; +0,691] | 2 |
| Larvae | spifil | 64 | 0,561 | he | +0,214 | [-0,376; +0,803] | 2 |
| Larvae | crus | 256 | 0,549 | random_z | +0,093 | [-0,106; +0,292] | 2 |
| Larvae | crus | 64 | 0,581 | orth | +0,228 | [-0,874; +1,330] | 2 |

Com 2 splits o intervalo não informa quase nada. Com 3 ele ainda é largo, porque tem 2 graus de liberdade. Em Larvae, `fps` só tem 1 split, e em Eggs_subgroup também (64 filtros).

O que já se sabe:

- **Patch sorteado fica acima do SPiFiL em Protozoan_subgroup:** `crus` 0,729 contra `spifil` 0,654 em 64 filtros. Em 256 eles ficam perto (0,778 e 0,773), e `fps` passa os dois (0,813).
- **Em flowers, `spifil`, `crus` e `fps` não ganham do aleatório na `cat` de 3 camadas.** Com 64 filtros e 2 camadas há intervalo acima de 0: `fpsw@mapa_peso` 0,588, Δ +0,053 [+0,031; +0,075], e `fps` 0,582, Δ +0,047 [+0,020; +0,074].
- **Ranking** (Δ com célula escolhida em 2 splits e medida no 3º): crus@geodesic +0,113, crus@obj +0,096, crus@sift_nms +0,082, spifil +0,054, fps +0,054, zca2 -0,039 (SVA:64-107). Essa média inclui bases fora do critério; `bases boas` não (`eval/spf_tabela.py:154-155`).
- **Sementes** (escrito às 13:55): medoide cai em objeto de 25% a 49% das vezes, contra 6% a 20% de área (SVA:25).
- **Fundo:** banco só de fundo perde até -0,13 de κ; só de objeto ganha até +0,066 (SVA:26).
- **Impureza:** o recall fica entre 0,85 e 0,93, e quem paga pelo fundo são as outras classes, 0,37 contra 0,57 (SVA:27).
- **Fundo zerado, 1 split:** o aleatório sobe de 0,66 para 0,85, e a vantagem encolhe ou inverte (SVA:28).
- **Mais imagens por classe não vencem `random_z`** em flowers, L1 de ~255 filtros: spifil k=1 0,573, k=10 0,546, random_z 0,578 (`spifil_amostras.md:130`, `:132`, `:138`). A seção `Resultados` desse arquivo está vazia, e a tabela só tem flowers.
- **Saliência:** o mapa escolhido é `gmr` (`saliencia.md:3`). Em flowers, ele muda Δκ em +0,01 ou menos na média (`saliencia.md:85`).
- **Lei de superpixels:** não há lei única, R² 0,01 (`law_size_superpixels.md:3`). Regra prática: 2 classes, n = 100; resto, n = 12 (`law_size_superpixels.md:40`).
- **Treinado (B0):** aleatório fica +0,010 acima do SPiFiL, erro-padrão 0,009, 9 pares (`results_log.md:120`).

Cobertura dos 12. Linhas contadas nos CSVs `results/svm_disc_md_spf*` a 200 px, às 15:24. L1 é o melhor κ de **teste** da lei, 64 filtros, com GeM + TTA, escolhido pela validação (`law_size_superpixels.md:5`, `:12-23`). É só relato: não decide nada. As duas últimas colunas são κ de validação a 50%, com os nomes do log (`results_log.md:11-24`).

| Dataset | Linhas na grade | Situação na grade | L1, teste | Pico | DINOv2 giant |
|---|---|---|---|---|---|
| Eggs | 71 | só fundo zerado, 1 split | 0,641 | 0,932 | 0,980 |
| Protozoan | 0 | sem κ | 0,407 (1 split) | 0,823 | 0,939 |
| Larvae | 1.203 | 2 splits | 0,616 | 0,876 | 0,968 |
| aid | 0 | sem κ | 0,586 | 0,864 | 0,970 |
| Protozoan_subgroup | 5.186 | 3 splits | 0,674 | 0,859 | 0,945 |
| Eggs_subgroup | 1.889 | 3 splits, parcial | 0,695 | 0,924 | 0,989 |
| flowers | 4.765 | 3 splits | 0,529 | 0,888 | 0,973 |
| protozoan_cysts | 0 | sem κ | 0,824 | 0,898 | 0,966 |
| remote_sensing | 0 | sem κ | 0,425 | 0,601 | 0,708 |
| helminth_eggs | 0 | sem κ | 0,880 | 0,952 | 0,987 |
| helminth_larvae | 0 | sem κ | 0,935 | 0,966 | 0,994 |
| Larvae_subgroup | 1.721 | 2 a 3 splits, parcial | 0,676 | 0,834 | 0,981 |

Na lei, SPiFiL contra aleatório na L1 dá +0,058 de κ [0,005; 0,122] na média dos 12, no teste e só no split 1 (`law_size_superpixels.md:42`). Esse número não é comparável à grade: a lei usa um só aleatório, com n = 100, e a grade usa o melhor de vários controles. Em Protozoan_subgroup, L1 de 64, a lei diz +0,17 e a grade diz -0,039.

Buracos:

- **Metade dos 12 não tem κ na grade**, e Eggs só tem fundo zerado.
- **`Eggs` e `Protozoan` em raw com fundo real** dependem de `disc_md_spf_h_raw`, que não estava entre os launchers às 15:27 (`ps`: combo_a, deep_cedo, h, r1, sal_par).
- **`spf_sobreviventes.json` não existe**, e r2 a r5 dependem dele (`eval/spf_tabela.py:47`, `eval/spf_grade.py:734`).
- **Profundidade e Combo só têm 3 bases, 1 split** (SVA, seções `### Profundidade` e `### Combo`).
- **Comparar os 12 pede mudar `BASES`**, que hoje lista 5 (`eval/spf_tabela.py:48`).
- **Os documentos mudaram de estado.** `fila.md:16` diz "Rodando agora", mas `pausa_treinos.md:3` pausou. O cabeçalho de `results_log.md:3` diz 07:40, com arquivo de 14:46.

## 7. O que já existe de Gaussian, DoG, LoG, SIFT e multiescala

**Resposta curta: como filtro de rede, nada. Só existe SIFT, e ele só diz onde recortar.**

| Item | Na biblioteca | No caminho do modelo | Fora do modelo |
|---|---|---|---|
| Gaussian | não; só `sigma` do SLIC, default 0, desligado (`SPiFiL:superpixels/slic.py:36`) | não | augmentation (`core/data/transforms.py:50`, `:79`, `core/data/train_aug.py:118`, `core/data/multi_crop_transform.py:70`), saliência (`saliency/fusion.py:130-133`, `saliency/maps_a.py:57`, `saliency/maps_b.py:92`, `:110`, `:133`), teste de robustez (`eval/spf_fundo.py:186`) |
| DoG | não | não achei | só dentro do SIFT do OpenCV |
| LoG | não | não achei | não achei |
| SIFT | não | sim, `cv2.SIFT_create(contrastThreshold=0.02)`, OpenCV 4.11.0 (`eval/spf_grade.py:127`) | repetibilidade em `eval/spf_fundo.py:357` |
| Seleção multiescala | não | não achei | não achei |

Busca: `graphify query` não trouxe implementação; devolveu `scale()` do NAS e SIGReg. Depois usei grep por `gauss`, `dog`, `laplac`, `gabor`, `scale.?space`, `multi.?scale`, `multi.?res`, `pyramid`, `sift` em `flim`, `eval`, `core`, `saliency`, `tools`, `experiments`, `methods` e `ga_search`. Na biblioteca rodei `gauss|dog|log[^a-z]|laplac|sift|pyramid|scale|sigma`.

Falso positivo: `laplacian` em `saliency/fusion.py:358-359` é o laplaciano de um grafo de regiões, não um filtro LoG. `GaussianMixture` em `eval/l1_sem_rotulo.py:176` é agrupamento.

O SIFT da grade joga a escala fora. O detector procura em várias escalas por dentro, mas `keypoints` só guarda `pt`, `response` e o descritor (`eval/spf_grade.py:182-189`). `size`, `octave` e `angle` não são usados. O patch sai sempre com o mesmo `kernel` na posição do ponto, e a imagem do SIFT é só a banda L (`eval/spf_grade.py:207`). Um ponto de escala grande vira uma janela 5x5 igual às outras.

O que chega mais perto de multiescala hoje:

| Mecanismo | O que é | Fonte |
|---|---|---|
| Camadas com pool | cada camada vê uma região maior; é o único que está ativo | `eval/spf_tabela.py:60` |
| `dilation` | existe na arquitetura, sempre 1 na grade | `flim/spifil.py:701-702`, `eval/spf_grade.py:693`, `SPiFiL:config.py:27` |
| Tamanho da imagem | um job por tamanho, sem combinar | `eval/l1_fast.py:75`, `experiments/ray/runners/eval.py:482` |
| Leitura `spp` | pirâmide espacial 1x1 + 2x2 + 4x4 de pooling; fora da grade | `eval/svm.py:353-355` |
| Gabor em várias escalas | só no mapa `gbvs` de saliência | `saliency/maps_a.py:170-178` |
| `multires` do NAS | busca por resolução de entrada, não filtro | `ga_search/prepare_multires_grid.py` |

O encaixe natural de um braço novo é `METHODS` ou `CONTROLS`: basta entregar `u`, e a dobra, a largura real, a leitura e o pareamento vêm de `chain` (`eval/spf_grade.py:680-697`).

## Novos experimentos necessários

O que faltava para responder ao pedido: pirâmide Gaussiana, DoG, LoG e escala de keypoint (`flim/scale_space.py`), candidatos com escala na grade (`amostra.multiescala` em `eval/spf_grade.py`) e os braços S1 a S8. Protocolo, critério e tabelas em `multiescala.md`.
