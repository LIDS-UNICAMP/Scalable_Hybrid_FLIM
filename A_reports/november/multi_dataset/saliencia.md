# Saliência sem rótulo: 11 mapas em 270 imagens

**Resposta:** o mapa que segue para o SPiFiL é o `gmr`. Ele quase empata com o melhor por imagem (Q 0,725 contra 0,737), é mais estável (0,898 contra 0,864) e custa 110 vezes menos (0,33 s contra 37 s). Saliente aqui quer dizer "se destaca". Não quer dizer "é o objeto": os casos de falha abaixo mostram a diferença.

Amostra: 30 imagens de treino por base (split 1, 50%), uma de cada classe por vez, em Eggs, Larvae, Protozoan (raw), nos 3 subgrupos, flowers, aid e remote_sensing. Código: `saliency/batch.py`, `experiments/saliency/lote.yaml`. Saídas por imagem em `/exps/moliveira/saliency/<base>/<imagem>/`: original, 11 mapas, heatmaps, máscaras por limiar, `mapas.npz`, `scores.csv` (score, posição, Top-K, seleção), `candidatos.csv`. Tabelas completas e precisão-revocação por limiar em `/exps/moliveira/saliency/resumo.md` e `limiares.csv`. Figuras: `saliencia/<base>.png` (melhor, mediano e pior caso) e `saliencia/casos_de_falha.png`.

## Comparação sem label

Q é o score interno (forma do mapa, contraste, concordância e estabilidade). Estabilidade: o pipeline inteiro é refeito em 4 imagens mexidas de leve, com a escolha refeita em cada uma. Coerência: continuidade espacial (Moran) e 1 - fragmentação. Custo: segundos por imagem, com a máquina carregada; vale como ordem de grandeza.

| candidato | Q | estabilidade | continuidade | 1 - fragmentação | s/imagem | s em 200 px |
|---|---|---|---|---|---|---|
| A melhor mapa selecionado | 0,737 | 0,864 | 0,978 | 0,963 | 36,9 | 16,1 |
| B fusão Top-K (K = 3) | 0,680 | 0,905 | 0,984 | 0,946 | 36,9 | 16,1 |
| C fusão regional | 0,698 | 0,871 | 0,981 | 0,956 | 37,3 | 16,2 |
| D média simples | 0,607 | 0,979 | 0,993 | 0,843 | 5,7 | 3,1 |
| `gmr` fixo | 0,725 | 0,898 | 0,977 | 0,935 | 0,33 | 0,05 |

A, B e C custam 37 s porque a seleção precisa dos 11 mapas e da estabilidade deles. D é o mais estável porque média muda pouco, e é o mais espalhado. A escolha de A troca de mapa entre imagens mexidas: em remote_sensing a estabilidade dele cai para 0,62.

| estabilidade | A | B | C | D | gmr |
|---|---|---|---|---|---|
| Eggs | 0,968 | 0,976 | 0,939 | 0,994 | 0,937 |
| Eggs_subgroup | 0,970 | 0,981 | 0,960 | 0,997 | 0,971 |
| Larvae | 0,835 | 0,901 | 0,817 | 0,985 | 0,828 |
| Larvae_subgroup | 0,847 | 0,907 | 0,836 | 0,985 | 0,886 |
| Protozoan | 0,933 | 0,955 | 0,933 | 0,994 | 0,931 |
| Protozoan_subgroup | 0,980 | 0,982 | 0,974 | 0,995 | 0,975 |
| flowers | 0,846 | 0,864 | 0,832 | 0,970 | 0,892 |
| aid | 0,779 | 0,813 | 0,756 | 0,953 | 0,817 |
| remote_sensing | 0,616 | 0,763 | 0,787 | 0,940 | 0,847 |

## Com label: só avaliação, nada foi escolhido por isto

180 imagens, de Eggs, Larvae, Protozoan (`raw/label`) e dos subgrupos (`masks/`, ligadas a `raw/*/label`). IoU, Dice, precisão e revocação no limiar de Otsu; AP varre todos os limiares.

| candidato | AP | IoU | Dice | precisão | revocação |
|---|---|---|---|---|---|
| A | 0,909 | 0,787 | 0,862 | 0,885 | 0,864 |
| B | 0,925 | 0,795 | 0,870 | 0,879 | 0,888 |
| C | 0,912 | 0,774 | 0,856 | 0,870 | 0,872 |
| D | 0,939 | 0,751 | 0,840 | 0,766 | 0,978 |
| gmr | 0,915 | 0,781 | 0,860 | 0,885 | 0,859 |

| IoU | A | B | C | D | gmr |
|---|---|---|---|---|---|
| Eggs | 0,869 | 0,875 | 0,859 | 0,889 | 0,854 |
| Eggs_subgroup | 0,886 | 0,884 | 0,872 | 0,912 | 0,881 |
| Larvae | 0,616 | 0,623 | 0,609 | 0,563 | 0,612 |
| Larvae_subgroup | 0,677 | 0,671 | 0,627 | 0,650 | 0,675 |
| Protozoan | 0,791 | 0,818 | 0,798 | 0,689 | 0,785 |
| Protozoan_subgroup | 0,883 | 0,899 | 0,877 | 0,801 | 0,877 |

O Q interno não reproduz o label: D tem o pior Q e o melhor AP, porque cobre quase tudo (revocação 0,98, precisão 0,77). Por imagem, Q e AP têm correlação de postos entre 0,52 e 0,71. Q serve para separar mapa ruim de mapa razoável, não para ordenar os bons.

## Como o mapa foi escolhido

Regra final: o candidato mais barato com Q a até 0,03 do melhor, na média das 270 imagens. Dá `gmr` no geral e em 8 de 9 bases (em aid dá A). O resultado não depende do 0,03: qualquer margem entre 0,012 e 0,085 dá `gmr`.

**A regra mudou no meio.** A primeira era "o mais barato a 1 erro-padrão do melhor", e deu A em 8 de 9 bases. Ela não servia: A é o máximo por imagem do próprio score, logo fica acima de qualquer mapa fixo por construção. Troquei pela margem depois de ver a saída, que já trazia o IoU. A razão da troca é interna e `gmr` é o primeiro mapa fixo em Q, mas fica o registro.

Cortes da máscara na fase 2: `triangle`, `otsu`, `t0.75`, porque cobrem em média 35%, 23% e 14% da imagem. Escolhidos só pela cobertura.

## Correções no meio

- **Faixas de borda.** Os subgrupos trazem faixas pretas e brancas embutidas: em Larvae_subgroup, cerca de metade da imagem é faixa. O `gmr` lê a borda como fundo, então a imagem inteira virava "saliente". `content_box` agora corta as faixas antes dos mapas. Vi a falha na figura. O IoU do `gmr` em Larvae_subgroup foi de 0,18 para 0,68, e em Protozoan_subgroup de 0,70 para 0,88.
- **2 px a mais no corte, e a costura.** A transição da faixa para a imagem tem 1 a 2 px e ainda invertia o mapa; a costura entre faixa preta e branca segurava o corte antes da hora. Sem label, o `gmr` em Larvae_subgroup subiu em Q de 0,671 para 0,728. Antes de aplicar os 2 px também olhei o IoU de 6 imagens: foi um uso do label fora do combinado, e fica registrado.

## Escala ao tamanho da imagem

O que mede pixel escala com o lado maior sobre 400 px (tamanho dos artigos): segmentação e borda de `region_contrast`, dilatação e blur de `bms`, blur de `mser`, `ks` de `selective_search`, escalas de `gbvs`. `gmr` e fusão regional usam 200 superpixels com pelo menos 64 px cada. Lado acima de 640 px é reduzido antes de tudo (Larvae chega a 2528). Em imagens abaixo de 150 px, as duas escalas de `gbvs` viram uma só.

## Sucesso e falha

- **Vai bem:** objeto compacto em fundo liso. Eggs, Eggs_subgroup e Protozoan_subgroup têm IoU entre 0,85 e 0,88 no `gmr`. Em flowers, a flor isolada sai limpa.
- **Objeto fino que encosta na borda:** o `gmr` usa a borda como fundo e inverte o mapa. Larvae 000002_00001474 tem AP 0,04; Larvae_subgroup 000002_00002782 tem IoU 0,00.
- **Destaque que não é o alvo:** na flor sobre neve (flowers, pior caso), A acende a faixa escura de pedra, não a flor. Em Protozoan_subgroup, D acende também os detritos pequenos.
- **Sem objeto único:** aid e remote_sensing têm Q perto de 0,58, e os mapas mudam com pouca perturbação. Lá o mapa aponta para o que difere, e isso nem sempre tem relação com a classe.
- **Mapa frágil:** Eggs 000009_00001054 tem estabilidade 0,04 no `gmr`, sem nada visível que avise.

## Fase 2

`gmr` como máscara e como peso, sobre `cos`, `fps` e `crus@sift_nms`. YAMLs: `experiments/autoencoder/disc_md_spf_sal_fl.yaml` e `disc_md_spf_sal_par.yaml`. Resultado em `spifil_vs_aleatorio.md`, seção "Saliência sem rótulo". flowers não tem label, logo não tem oráculo. Quase todo flowers rodou com o corte anterior das faixas: a versão final muda 1 das 30 imagens da amostra.

flowers fechou: 6 de 120 células passam no critério, todas com 64 filtros e nenhuma com 256. A mais forte é `fpsw@mapa_peso` (L2 cat, Δκ +0,053, IC [+0,031; +0,075]), a única que passa quando a célula é escolhida em 2 splits e medida no 3º. O mérito é pequeno: o `fps` sem mapa já dá +0,047 nessa célula, e na média das células o mapa muda Δκ em +0,01 ou menos. Com 120 células e 3 splits, algumas passam por acaso.

## Fase 2: resultado nos parasitos (07/11, 18:10 UTC)

240 comparações, 3 splits cada; médias simples sobre a tabela de `spifil_vs_aleatorio.md`, sem teste estatístico. Ganho = Δκ com mapa menos Δκ sem mapa.

| base | Δκ sem mapa | Δκ com mapa | ganho |
|---|---|---|---|
| Larvae_subgroup | +0,121 | +0,185 | +0,064 |
| Larvae | +0,129 | +0,170 | +0,041 |
| Eggs_subgroup | +0,010 | +0,023 | +0,013 |
| flowers | −0,014 | −0,008 | +0,006 |
| Protozoan_subgroup | +0,050 | +0,049 | −0,001 |

- Máscara e peso empatam: ganho de +0,022 a +0,028.
- Com `fps`, 256 filtros e L3 concatenada, o ganho nas larvas some (+0,007 e −0,001).
- Leitura: a saliência não entra no método final, salvo como opção para bases parecidas com larvas.

## Três bases novas, com `label/` (07/11, 20:50 UTC)

90 imagens, 30 por base, em `/exps/moliveira/saliency_helminth` (`iou_label.csv`). Corte Otsu. O `resumo.md` dessa pasta não foi gerado: `summary` espera `limiares.csv`, que só existe para bases de `MASK_DIR`. Os mapas estão completos.

| IoU | A | B | C | D | gmr | área do objeto |
|---|---|---|---|---|---|---|
| helminth_eggs | 0,498 | 0,534 | 0,545 | 0,406 | 0,621 | 88% |
| helminth_larvae | 0,649 | 0,540 | 0,383 | 0,457 | 0,493 | 40% |
| protozoan_cysts | 0,490 | 0,525 | 0,551 | 0,413 | 0,711 | 91% |

Nessas bases o objeto enche o recorte, e os mapas fazem o contrário do que foram feitos para fazer: procuram algo que se destaca do fundo. B perde do `gmr` em duas das três. Por isso a fusão B não vale como padrão sem medir base a base.

## Fusão Top-K (B) em lote, por imagem (07/11, 21h)

**Resposta:** o mapa B agora sai igual ao de antes e cerca de 7 vezes mais rápido. Fora da amostra, em Larvae_subgroup, ele ganha do `gmr` por pouco: IoU 0,657 contra 0,643. As outras bases esperam o lote no jaci.

**Ressalva:** B foi escolhida vendo o IoU das 180 imagens da amostra. Por isso a medida justa é a de fora dessa amostra, que é a desta seção.

Código: `saliency/topk_lote.py` gera o cache e `saliency/topk_iou.py` mede. YAMLs: `experiments/saliency/topk.yaml` (athena) e `topk_jaci.yaml` (jaci, 12 bases).

### Por que ficou rápido, sem mudar o mapa

- **`gbvs` sem sistema linear.** As duas cadeias de Markov do GBVS são reversíveis, então o ponto de equilíbrio sai em fórmula fechada: a soma das linhas na primeira cadeia e `a·(F a)` na segunda. Antes eram 24 `solve` 1369×1369 por chamada. A fórmula bate com o `solve` até 1e-12, inclusive no mapa de dois valores, e custa 115 vezes menos.
- **Somas por blocos de 128 linhas.** O resultado é igual bit a bit e o tempo caiu de 53 s para 3,4 s em 12 chamadas. O `falloff` passou a ficar em cache.
- **Estabilidade:** as imagens mexidas e as máscaras de visibilidade são feitas uma vez, não uma vez por mapa (`fusion._moves`). A estabilidade continua inteira no Q.
- **Só B:** o lote não refaz C, D nem as características finais nas 4 imagens mexidas.
- **Memória:** nesta máquina, cada página nova custa cerca de 140 µs no kernel. O lote manda o glibc reter a memória (`MALLOC_MMAP_THRESHOLD_`, `MALLOC_TRIM_THRESHOLD_`).
- **Paralelo:** um processo por imagem (`multiprocessing`, spawn), retomável. O arquivo só aparece quando está completo.

Igualdade contra `mapas.npz`, em 10 imagens das 9 bases a 640 px: diferença máxima 0,0 em 9 imagens e 6e-8 em remote_sensing (um passo de float32). Guardar em float16 soma até 2,4e-4.

Velocidade, caminho antigo contra o novo no mesmo processo, em flowers a 200 px: 29,6 s contra 3,9 s por imagem, 7,5 vezes. O lote de Larvae_subgroup no athena levou 0,67 s por imagem com 4 workers, com a carga entre 75 e 116. A 640 px, em remote_sensing, o ganho é menor: 111,8 s contra 59,5 s, 1,9 vezes, também com diferença 0,0. Ali o maior custo passa a ser `region_contrast`, mas o lote roda a 200 px, a entrada do SPiFiL.

### Faixas pretas

**Resposta:** o corte falhava em 17 imagens, todas de larva fina, e agora acerta as 17. Fora delas, nenhuma imagem muda.

Varri as imagens de treino e validação a 200 px, sem máscara. "Falha" aqui quer dizer que a borda do recorte ainda tem 3 ou mais linhas ou colunas saturadas.

A causa: depois de tirar 2 px de cada lado, uma larva fina deixa só de 9 a 15 px vivos, abaixo do mínimo de 16, e o corte desistia. Agora `content_box` aceita uma tira de 8 a 15 px se duas coisas valem ao mesmo tempo: o recorte está mais de 95% vivo e encosta na borda no outro eixo. Nas 18 tiras de larva o recorte fica 100% vivo. Num objeto sobre fundo já apagado, fica no máximo 77% vivo, ou o objeto não encosta na borda.

| base | imagens | falha antes | falha depois | mudaram |
|---|---|---|---|---|
| Larvae | 1.591 | 13 | 0 | 14 |
| Larvae_subgroup | 739 | 4 | 0 | 4 |
| Eggs, Eggs_subgroup, Protozoan, Protozoan_subgroup, aid, remote_sensing | 72.682 | 0 | 0 | 0 |
| flowers | 2.738 | 3 | 3 | 0 |
| helminth_eggs | 5.094 | 645 | 645 | 0 |
| helminth_larvae | 3.502 | 1.376 | 1.376 | 0 |
| protozoan_cysts | 9.536 | 2.079 | 2.079 | 0 |

A 14ª imagem de Larvae tinha só 2 px de resíduo e foi corrigida junto. Em flowers, os 3 casos são fundo escuro da foto, não faixa.

**Fica em aberto:** em helminth_eggs, helminth_larvae e protozoan_cysts, o preto não é faixa: o fundo já vem apagado na própria base. Nelas o corte faz hoje uma tira horizontal na altura do objeto e deixa a largura inteira. Isso não é faixa nem fundo inteiro. Não mexi porque mudaria 3.600 imagens que o `gmr` dos braços `mapa_*` já usa. O mais limpo seria não cortar nessas 3 bases, mas a decisão é do usuário.

Uma primeira versão, só com o mínimo de 8 px, recortava o objeto nessas 3 bases (3.636 imagens). Descartei pela figura dos recortes, sem usar máscara.

### Avaliação fora da amostra

Imagens que são treino ou validação nos 3 splits, ou seja, nenhuma é teste em split algum, e fora das 30 da amostra de `batch`. B lido do cache a 200 px; `gmr` feito na hora sobre o mesmo recorte. Corte de Otsu; o AP varre os limiares.

| base | mapa | AP | IoU | Dice | precisão | revocação | imagens |
|---|---|---|---|---|---|---|---|
| Larvae_subgroup | B | 0,811 | 0,657 | 0,768 | 0,754 | 0,843 | 432 |
| Larvae_subgroup | gmr | 0,803 | 0,643 | 0,755 | 0,766 | 0,781 | 432 |

B ganha do `gmr` em IoU por +0,014, com mais revocação (+0,06) e um pouco menos de precisão. Na amostra de 30 a ordem era a inversa (0,671 contra 0,675). A diferença é pequena nos dois casos. As outras bases dependem do lote no jaci: `python -m saliency.topk_iou` grava em `<out>/iou.csv`.

### Custo do lote

Treino e validação dos 3 splits, nas 12 bases: 95.892 imagens. No athena, com carga de 75 a 116, B custa de 2,7 a 4 s por imagem por worker a 200 px. Com 64 workers, isso dá no máximo cerca de 1,7 h para tudo. No jaci deve sair menos: lá o caminho antigo custa 5,4 s por imagem, contra 37 s aqui.

## SIFT como mapa de objeto (07/11, 22h)

**Resposta:** não serve como mapa. A ideia está certa pela metade: os keypoints caem no objeto 2 a 4,5 vezes mais do que o acaso. Mas o mapa de densidade feito com eles perde do `gmr` em IoU nas 9 bases com máscara, e em AP em 7 ou 8 delas. Nas 1 ou 2 em que "ganha" em AP, o objeto enche quase todo o recorte: um mapa constante já tira AP igual à área.

Ideia testada: os objetos ficam onde há muitos keypoints SIFT, então a densidade de keypoints, sem rótulo e sem backprop, diria onde está o objeto. Cada keypoint vira uma mancha (gaussiana de pico 1, sigma = metade do `size` dele). Quatro versões: (a) só conta os keypoints; (b) pesa cada mancha pela força do keypoint (`response`); (c) igual a (b) com mancha de tamanho fixo, 5% do lado; (d) (b) vezes `gmr`. Todas vão para [0,1]. Detector igual ao de `spf_grade` (`contrastThreshold=0.02`, banda L do `LabNorm`). Mesmas imagens, mesmo recorte (`content_box`) e mesmo lado do `gmr` de `batch` (até 640 px).

Amostra: as 180 imagens de `/exps/moliveira/saliency` com máscara e as 90 de `/exps/moliveira/saliency_helminth` (máscara em `label/`), todas de treino. Uma imagem de Larvae_subgroup ficou de fora: o `content_box` mudou depois do lote e o recorte não bate mais com o `gmr` guardado. Mapa sem corte de Otsu (constante) conta IoU 0, como em `topk_iou`. Por isso o `gmr` de helminth_eggs e helminth_larvae aparece abaixo da tabela das 20:50, que tirava essas imagens; imagem a imagem os valores são iguais.

Código: `saliency/sift_density.py`. Saída por imagem e `sift.csv` em `/exps/moliveira/saliency_sift/`. Figuras: `sift_densidade/<base>.png` (original com keypoints, densidade (b), `gmr`, produto (d); melhor, mediana e pior AP de (b) onde há máscara, as 3 primeiras imagens onde não há).

**Ressalva:** as quatro versões e os dois tamanhos de mancha foram escolhidos e julgados nesta mesma amostra. Como todas perderam, isso não muda a conclusão; se uma tivesse ganho, precisaria de confirmação fora da amostra.

| AP | gmr | B | D | sift_a | sift_b | sift_c | sift_d |
|---|---|---|---|---|---|---|---|
| Eggs | 0,974 | 0,986 | 0,985 | 0,725 | 0,749 | 0,633 | 0,922 |
| Eggs_subgroup | 0,989 | 0,989 | 0,994 | 0,723 | 0,741 | 0,641 | 0,911 |
| Larvae | 0,768 | 0,809 | 0,845 | 0,405 | 0,415 | 0,515 | 0,725 |
| Larvae_subgroup | 0,829 | 0,842 | 0,888 | 0,377 | 0,378 | 0,335 | 0,618 |
| Protozoan | 0,950 | 0,949 | 0,956 | 0,634 | 0,626 | 0,607 | 0,888 |
| Protozoan_subgroup | 0,989 | 0,990 | 0,984 | 0,717 | 0,697 | 0,646 | 0,925 |
| helminth_eggs | 0,941 | 0,983 | 0,931 | 0,955 | 0,950 | 0,935 | 0,948 |
| helminth_larvae | 0,698 | 0,956 | 0,929 | 0,577 | 0,578 | 0,559 | 0,597 |
| protozoan_cysts | 0,979 | 0,971 | 0,882 | 0,974 | 0,969 | 0,975 | 0,980 |

| IoU (Otsu) | gmr | B | D | sift_a | sift_b | sift_c | sift_d |
|---|---|---|---|---|---|---|---|
| Eggs | 0,854 | 0,875 | 0,889 | 0,442 | 0,437 | 0,421 | 0,444 |
| Eggs_subgroup | 0,881 | 0,884 | 0,912 | 0,413 | 0,386 | 0,404 | 0,425 |
| Larvae | 0,612 | 0,623 | 0,563 | 0,267 | 0,254 | 0,269 | 0,285 |
| Larvae_subgroup | 0,691 | 0,684 | 0,665 | 0,201 | 0,172 | 0,228 | 0,199 |
| Protozoan | 0,785 | 0,818 | 0,689 | 0,420 | 0,390 | 0,476 | 0,458 |
| Protozoan_subgroup | 0,877 | 0,899 | 0,801 | 0,516 | 0,497 | 0,478 | 0,545 |
| helminth_eggs | 0,536 | 0,542 | 0,403 | 0,158 | 0,127 | 0,127 | 0,106 |
| helminth_larvae | 0,427 | 0,540 | 0,457 | 0,106 | 0,101 | 0,156 | 0,066 |
| protozoan_cysts | 0,711 | 0,525 | 0,413 | 0,161 | 0,127 | 0,257 | 0,097 |

Placar contra o `gmr`, nas 9 bases: em AP, (a) e (b) ganham em 1 (helminth_eggs), (c) em 0, (d) em 2 (helminth_eggs, protozoan_cysts por 0,001); em IoU, nenhuma ganha em nenhuma. Média das 269 imagens: precisão 0,70 e revocação 0,32 em (b), contra 0,86 e 0,78 no `gmr`.

"Quantas vezes o acaso": a fração de keypoints dentro do objeto, dividida pela área do objeto no recorte.

| base | keypoints por imagem | área do objeto | keypoints no objeto | vezes o acaso |
|---|---|---|---|---|
| Eggs | 540 | 28% | 54% | 2,0 |
| Eggs_subgroup | 147 | 28% | 60% | 2,2 |
| Larvae | 1.210 | 11% | 43% | 3,8 |
| Larvae_subgroup | 93 | 12% | 40% | 3,5 |
| Protozoan | 101 | 14% | 51% | 3,6 |
| Protozoan_subgroup | 74 | 13% | 56% | 4,5 |
| helminth_eggs | 9 | 88% | 96% | 1,1 |
| helminth_larvae | 5 | 40% | 88% | 2,2 |
| protozoan_cysts | 19 | 91% | 100% | 1,1 |

Pesar pela força quase não muda a fração (diferença de até 2 pontos).

**Por que a pista boa vira mapa ruim.** Pense em pegadas na areia: há mais pegadas onde o objeto está, mas elas ficam na borda e nas manchas dele, não no meio liso. E metade das pegadas está no fundo: sujeira, bolhas, outras células, folhas e grama em flowers. A densidade acende em poucos pontos quentes e deixa o resto do objeto frio (revocação 0,32). O AP não depende de como o mapa é esticado, e mesmo assim cai de 0,90 (`gmr`) para 0,68. Multiplicar pelo `gmr` (d) só piora o `gmr`: fica com a precisão dele e perde metade da revocação.

Leitura para o SPiFiL: SIFT serve para escolher **pontos** (já é o que `sift_nms` e `sift_sp` fazem em `spf_grade`), não para desenhar **onde está o objeto**. Nenhuma versão bateu o `gmr` na maioria das 9 bases, então não há integração nem YAML `disc_md_salw_sift_jaci`.

Custo, com a carga da máquina entre 85 e 95: mediana 0,03 s por imagem para detectar e desenhar a densidade (as 4 versões). A média é 1,1 s porque aid (600 px, 4.100 keypoints) leva 6,8 s e remote_sensing (640 px, 3.200 keypoints) 3,8 s: a soma exata das manchas cresce com keypoints × altura × largura. O `gmr` custa de 0,04 a 0,86 s nas mesmas imagens.

## Fusão Top-K em lote: parada (07/11, 22:40 UTC)

O usuário mandou parar o lote da fusão B, apagar o que ele gerou e seguir com o `gmr`. Apagados: `/exps/moliveira/saliency_b` (cache de Larvae_subgroup e `iou.csv`), no jaci `/dados/home/moliveira/exps/saliency_b`, e os arquivos `saliency/topk_lote.py`, `saliency/topk_iou.py`, `experiments/saliency/topk.yaml` e `topk_jaci.yaml`. Ficam `saliency/batch.py`, `fusion.py` e `maps_a.py`: o `gmr` usa o corte de faixas novo e o `gbvs` rápido, e o mapa sai igual. Último número medido: Larvae_subgroup, fora da amostra, B IoU 0,657 contra `gmr` 0,643. A opção `saliency.cache` de `spf_grade` fica sem uso.
