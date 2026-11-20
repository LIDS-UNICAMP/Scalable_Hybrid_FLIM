# L_input: SPiFiL e tamanho da imagem de entrada

**Resposta curta:** aumentar a imagem faz duas coisas ao mesmo tempo: mostra mais detalhe e deixa o objeto maior em pixels, enquanto o filtro continua do mesmo tamanho. A hipótese principal é que o tamanho ótimo depende de quanto do quadro o objeto ocupa, e isso varia mais de 10 vezes entre datasets. Logo, não deve existir um tamanho único. Isso ainda é geometria e previsão: nenhum kappa desta grade existe.

A grade: 63 células, 189 braços, 268 horas de slot (cerca de 67 h em 4 slots, 87 h com margem de 30%), pico estimado de 33,0 GB, 34 YAMLs de treino e 10 de eval.

Só leitura e CPU. Nada foi lançado. Nada escrito no repositório.
Busca: grafo primeiro; grep só para valores de YAML e chaves de config, que não têm nó no grafo.
Scripts e dados: pasta `<scratchpad>/law/` (`plan.py` regenera tudo; `res_yamls.tsv` é a lista da seção 8).

## 1. Quantidades físicas

**O que o filtro enxerga.** Conv 5x5 com `padding` + `MaxPool2d(3, stride=2)` (`flim/spifil.py:476`). Conta em pixels da imagem já redimensionada:

| camada | filtro cobre | campo com o pool | passo | grade das sementes a 200 / 320 / 640 / 1280 |
|---|---|---|---|---|
| L1 | 5 | 7 | 2 | 200² / 320² / 640² / 1280² |
| L2 | 15 | 19 | 4 | 99² / 159² / 319² / 639² |
| L3 | 35 | 43 | 8 | 49² / 79² / 159² / 319² |
| L4 (5x5) | 75 | 91 | 16 | 24² / 39² / 79² / 159² |
| L5 (5x5) | 155 | 187 | 32 | 11² / 19² / 39² / 79² |

- Cada camada dobra o campo: `campo(d) = 3·2^(d+1) − 5`. A L4 oficial é 1x1 e não aumenta nada: o modelo oficial enxerga 43 px.
- **X1 e X1s só medem L1: campo de 7 px.** Eles não veem a escala do objeto em nenhum tamanho. Por isso a grade daqui vai até L3, L4 e L5.
- O campo acima é o teórico. O efetivo deve ser menor, por isso a razão ótima tem que sair do dado.

**Tamanho nativo e objeto.** Varredura completa do cabeçalho em Eggs, Protozoan, Larvae e flowers; amostra nos outros. Objeto estimado sem máscara, sem abrir `label/` nem `masks/`, 48 imagens por dataset.

| dataset | lado maior: mediana (p5 a p95) | objeto / quadro | área do objeto | objeto a 200 / 640 px | como medi |
|---|---|---|---|---|---|
| Eggs raw | 432 (256 a 896) | 0,50 a 0,60 | 26% | 100 / 320 | contraste com a borda |
| Protozoan raw | 112 (64 a 176) | 0,40 a 0,43 | 14% | 80 / 256 | contraste com a borda |
| Larvae raw, comprimento | 1.328 (656 a 2.064) | ~0,70 | | 140 / 448 | contraste com a borda |
| Larvae raw, espessura | | ~0,045 | | 9 / 29 | forma de helminth_larvae (abaixo) |
| helminth_eggs | 200 | 0,16 (0,12 a 0,32 por classe) | 2% | 32 / 102 | exato: não preto |
| protozoan_cysts | 200 | 0,18 | 2,3% | 36 / 115 | exato: não preto |
| helminth_larvae, verme | 200 | espessura 0,04, comprimento 0,62 | 1,7% | 8 x 124 | exato: não preto |
| flowers | 320 (240 a 500) | ~0,46 | 26% | 92 / 294 | contraste, fraco |
| aid, remote_sensing | 600, 640 | sem objeto único | | | cena |

Três correções ao que estava assumido:
- **"Objeto é 3% do quadro" não vale para raw.** Vale para helminth_* e cysts (2%). Em Eggs raw é 26%: o recorte é justo em volta do ovo. O comentário em `flim/spifil.py` ("o fundo é ~97%") só vale nesse tipo, e os 1.249 px de R4 batem com ele (inferência minha: não abri as máscaras).
- **Larvae raw:** altura e largura medianas são 864 e 968 (de onde vem "~900"), mas o letterbox usa o lado maior: mediana **1.328**.
- **Eggs raw "até 1.344":** é o máximo (medi 1.360). A mediana é 432 e só 18% das imagens têm 640 ou mais.

**Razão objeto / campo em L3 (43 px).** Esta é a quantidade da hipótese H1.

| | 96 | 128 | 200 | 320 | 448 | 640 | 896 | 1280 |
|---|---|---|---|---|---|---|---|---|
| Eggs | 1,1 | 1,5 | 2,3 | 3,7 | 5,2 | 7,4 | | |
| Protozoan | 0,9 | 1,2 | 1,9 | | | | | |
| helminth_eggs | 0,4 | 0,5 | 0,7 | 1,2 | | 2,4 | | |
| Larvae, espessura | | 0,1 | 0,2 | 0,3 | 0,5 | 0,7 | 0,9 | 1,3 |
| Larvae, comprimento | | 2,1 | 3,3 | 5,2 | 7,3 | 10 | 15 | 21 |
| flowers | 1,0 | 1,4 | 2,1 | 3,4 | 4,8 | | | |

Pista: no modelo oficial antigo (tipo helminth, 200 px) o objeto tem 32 a 40 px e o campo 43: razão perto de 1. Nos tamanhos de D1, Eggs fica em 3,7 e Protozoan em 1,9: o campo não cobre o objeto. Isso não prova nada, mas é a razão de ir abaixo de 200.

**Quanto detalhe se perde ao reduzir.** Erro de pixel ao reduzir para N e voltar, em % da variância da imagem:

| | 96 | 128 | 200 | 320 | 448 | 640 | 896 | 1280 |
|---|---|---|---|---|---|---|---|---|
| Eggs | 2,5 | 1,6 | 0,7 | 0,2 | 0 | | | |
| Protozoan | 1,1 | 0 | | | | | | |
| Larvae | 8,5 | 6,2 | 4,0 | 2,4 | 1,4 | 0,7 | 0,3 | 0,1 |
| aid | 21,6 | 16,6 | 9,8 | 5,4 | 2,8 | 0 | | |
| flowers | 8,4 | 6,4 | 3,6 | 0,7 | 0,4 | 0 | | |
| helminth_eggs | 9,3 | 7,1 | 0 | | | | | |

Leitura: Eggs perde menos de 1% já a 200. aid é quem mais tem a perder. Larvae perde pouco acima de 640 em média, mas o verme é fino, então essa média pode enganar.

**Imagens ampliadas (lado maior menor que N).** O letterbox sempre leva o lado maior a N (`core/data/transforms.py:33`). Acima do nativo não entra informação nova: só o objeto cresce.

| dataset | % ampliadas em cada tamanho da grade |
|---|---|
| Larvae | 128 a 320: 0 / 448: 1 / 640: 4 / 896: 19 / 1280: **46** |
| Eggs | até 200: 0 / 320: 22 / 448: **54** / 640: **82** |
| flowers | até 200: 0 / 320: 21 / 448: **63** |
| aid | 0 em todos (topo 600, não 640) |
| Protozoan | 64: 2 / 96: 27 / 128: **64** / 200: **97** |
| helminth_eggs | até 200: 0 / 320 e 640: **100** (de propósito, controle) |

Consequência para D1: Protozoan a 200 px é 97% ampliado (mediana x1,8), e aid a 640 é 100% ampliado (x1,07).

**Pixels por superpixel / área do filtro.** Com área 800: 32 em L1, 3,6 em L2, 0,65 em L3, 0,14 em L4, 0,03 em L5. Com máscara o superpixel tinha ~25 px (R4): exatamente a área do filtro L1. Sem máscara ele ficou 32 vezes maior.

**Superpixels no objeto, pela área** (é um piso, ver "Pontos no objeto"). Área 800: Eggs 5 a 128 px, 33 a 320, 133 a 640; Protozoan 3 a 128, 7 a 200. Com n = 128 fixo: Eggs 33 e Protozoan 18, em qualquer N.

**Superpixel contra o filtro** (DISF real, letterbox, Eggs, Larvae, flowers, aid, helminth_eggs; 2 a 3 imagens por ponto):
- Área real é desigual: pedindo 800 px, a mediana fica entre 190 e 590 px (~85 em helminth_eggs) e o p90 perto de 1.200 a 1.900.
- O ponto (medoid) fica a **0 a 4 px da borda do superpixel em qualquer área**, de 50 a 3.200. Motivo: o medoid é o pixel de cor mais parecida com a média da região, não o centro dela (`spifil/superpixels/centers.py:87`). Geodesic e centroid são geométricos e devem se comportar diferente.
- Filtro L1 toca a borda em 96 a 100% dos pontos com área 50 e em 33 a 75% com área 3.200. Filtro L2: de 100% a 69 a 99%.
- Contraste do recorte no ponto, contra um recorte ao acaso: cerca de 2 a 4 vezes em Eggs e Larvae, 1,1 a 1,5 em aid e flowers. Em helminth_eggs não se aplica (recorte ao acaso cai no preto).
- A área fixa, isso quase não muda com N (Eggs, área 800, L1 toca a borda: 71 / 75 / 63% a 200 / 320 / 640).

**Pontos no objeto** (fundo preto, exato): o objeto cobre 2% da área, mas recebe **~20% dos pontos**. A n = 50 são ~11 pontos no objeto, não 1. O DISF concentra pontos onde há conteúdo.

## 2. O que eu supus e os dados derrubaram

- **"O tamanho nativo prevê a classe" (kappa 0,62):** artefato de amostra balanceada. No split 1, treino para validação, só com altura e largura: Eggs 0,09, Protozoan 0,12, Larvae 0,16, flowers 0,05. O letterbox apaga tamanho absoluto, mas isso pesa pouco na métrica. Fora da grade. Ainda assim a mediana por classe em Eggs vai de 256 a 928, então vale olhar entre parasitos.
- **"Superpixels no objeto = n x fração de área":** errado por 10 vezes (seção 1). Essa conta não pode virar lei; tem que contar pontos de verdade.
- **"O ponto cai no miolo liso e o filtro vira detector de cor":** errado, o ponto fica quase na borda. Isso enfraquece a razão "pixels por superpixel / área do filtro".
- **Estimadores que falharam:** LoG pega sujeira em fundo real (Eggs 25 px, contra ~180 px pelo contraste); autocorrelação falha com objeto pequeno; contraste não serve nos subgrupos (faixas pretas) e não mede espessura de verme (emenda sujeira). Resolução por espectro não vale em fundo preto (deu 56 em helminth_eggs, contra 7% de perda a 128), por isso uso o erro de pixel.

## 3. Hipóteses

Definições: **x** = objeto em px / campo. **A** = pixels por superpixel. **S** = pontos no objeto.

| | hipótese | previsão que pode falhar |
|---|---|---|
| H1 | Existe razão ótima objeto / campo, x*, igual entre datasets | κ contra log x cai em uma curva só. N*(d) = x*·campo(d) / (objeto/quadro) |
| H2 | O que manda é A / área do filtro | Braço P plano em N, braço C piora. **Prior baixo com medoid:** o ponto toca a borda em qualquer A. Com geodesic pode valer: X2 já tem esse dado |
| H3 | O que manda é S, pontos no objeto | Braço C plano em N, braço P muda |
| H4 | O que manda é detalhe real | κ sobe enquanto o erro de pixel cai e para quando ele some. Ampliar não ajuda |
| H5 | Colisão de sementes limita a camada funda | Em todas as células n é menor que a grade (seção 1), mas as sementes se concentram no objeto, então colidem antes: R4 viu 50 virarem ~20 em 24². Filtros distintos caem em N pequeno. Lê-se em `conv*-kernels.npy`, sem treino |
| H6 | Lei composta: dobrar N pede +1 camada | Melhor profundidade sobe 1 a cada dobra de N. `d* ≈ log2((objeto/x* + 5)/3) − 1` |
| H0 | Nula: N não importa | Tudo empata: fica o menor N, que é o mais barato |

Previsão de N* com x* = 1 (a escrever antes de ver o resultado):

| | L3 (43) | L4 (91) | L5 (187) |
|---|---|---|---|
| Eggs | 86 | 182 | 374 |
| Protozoan | 108 | 228 | 468 |
| flowers | 93 | 198 | 407 |
| helminth_eggs | 269 | 569 | |
| Larvae pelo comprimento | 61 | 130 | 267 |
| Larvae pela espessura | 956 | | |

O teste mais forte é Larvae: se a escala que manda é a espessura, ele quer mais de 900 px; pelo comprimento, menos de 300.

## 4. A grade

**Dois braços que se cruzam em 320 px, n = 128:**
- **Braço P (pixels por superpixel constantes):** A = 800, n = N²/800. Mesmo superpixel em pixels, então o filtro vê o mesmo.
- **Braço C (número constante):** n = 128. Mesmo superpixel em relação ao objeto.
- Os dois juntos separam os efeitos: P testa H2, C testa H3.

**Tamanhos por dataset:**

| dataset | tamanhos | papel |
|---|---|---|
| Larvae | 128, 200, 320, 448, 640, 896, 1280 | único com nativo real acima de 640 |
| Eggs | 96, 128, 200, 320, 448, 640 | objeto grande, previsão de N* baixo |
| aid | 128, 200, 320, 448, 600 | cena, muito detalhe fino |
| flowers | 96, 128, 200, 320, 448 | imagem natural |
| helminth_eggs | 96, 128, 200, 320, 640 | objeto pequeno; 320 e 640 são só ampliação: **mede geometria pura** |
| Protozoan | 64, 96, 128, 200 | nativo baixo; abaixo de 200 |

Acima de 640 só Larvae tem imagens de verdade. Eggs a 896 fica **fora da conta principal**: 94% ampliado, 45 horas de slot (seção 8, opcional).

**Profundidade:** o braço P vai a L3. O braço C vai a L3 abaixo de 128 px, L4 de 128 a 319 e L5 de 320 em diante, para o campo caber na imagem. L4 e L5 aqui são 5x5, não a 1x1 oficial. Largura `[32, 48, 64, 64, 64]`.
Como cada braço lê todos os estágios, um run dá vários valores de x, e pares como (320, L2) e (640, L3) têm x parecido com N diferente: é isso que separa H1 de H4.

**Controles:**
- **P+** (Eggs, aid, flowers; 128 px, n = 20, `images_per_class: 6`): em N pequeno o braço P quase não tem candidatos para escolher. P+ repõe os candidatos. Se P+ ganhar de P, a perda era falta de candidatos, não superpixel.
- **Lote** (helminth_eggs, 200 px, n = 50): 32x1 contra 8x4 contra 8x1.
- Larvae a 128 no braço P usa `images_per_class: 2`: com n = 20 a L3 pediria 32 filtros por classe e encolheria.

**Medida:** 20 épocas por estágio, warmup 2, 50%, 3 splits, `no_crop`, sem máscara, `gem_tta_pw` em validação, um SVM por estágio. Teste nunca. Leitura de 32 a 64 dimensões.
**Não decidir pela sonda por época.** Em X1 ela dá 0,00 / 0,66 / 0,19 no mesmo ponto de helminth_eggs entre splits.

## 5. Custo

Modelo de D1 + 5 s por época. Conferido contra época medida em X1 (L1): Eggs +29%, Protozoan +32%, helminth_eggs −3%, flowers −4%. Ou seja, folgado nos caros; L2 a L5 e N acima de 320 não têm medida. Sem margem:

| lote | dataset | células | braços | horas de slot |
|---|---|---|---|---|
| 1 | helminth_eggs | 11 | 33 | 21,5 |
| 1 | flowers | 10 | 30 | 12,3 |
| 1 | Protozoan | 7 | 21 | 17,1 |
| 1 | Larvae | 13 | 39 | 55,7 |
| 2 | aid | 10 | 30 | 76,1 |
| 2 | Eggs | 12 | 36 | 85,8 |
| | **total** | **63** | **189** | **268,5** |

- SVM: mais 22 h em série, ao lado do treino (1 por vez, flock, nice).
- 4 slots: lote 1 em ~27 h, lote 2 em ~40 h, total ~67 h. Com 30% de margem, ~87 h (3,6 dias).
- É mais que todo o Estágio 1 (198 h). Sugestão: lote 1 antes de X2. Ele já responde H1 a H5; o lote 2 confirma.
- Eggs e aid são caros por leitura de arquivo: N pequeno não fica mais barato neles (87 e 120 ms por imagem).

## 6. Lote e memória

**Controle do lote: lote efetivo 32 em todos os tamanhos.**

| N | `batch_size` | `accumulate_grad_batches` | efetivo |
|---|---|---|---|
| até 640 | 32 | 1 | 32 |
| 896 | 16 | 2 | 32 |
| 1280 | 8 | 4 | 32 |

- **O acumulado já existe por YAML:** vai em `overrides` como `trainer.accumulate_grad_batches`, sem código (`experiments/ray/runners/train.py:103`). Conferido com `--print_config` e dry-run. Nenhum YAML usava antes: é o primeiro uso.
- É a mesma conta do lote 32: sem BatchNorm, dropout 0, agendador por época (`autoencoder_flim_module.py:1385`). Mesmo número de passos e mesma lr em todo N. Só o último lote incompleto difere.
- Lote mínimo 8 respeitado.
- O controle 8x4 tem que dar igual a 32x1. Se não der, há bug e os tamanhos 896 e 1280 não valem.
- 8x1 mostra quanto o lote muda sozinho. O `_b10` de B2 mede o mesmo em X1 (lote 10). Os YAMLs de X1 a 640 ainda não têm `batch_size`, ou seja, 32 do config; mas L1 = 64 com lote 32 a 640 dá ~40 GB pelo modelo, e o `_b10` indica que B2 pretende baixar. Se baixar, a comparação de tamanhos em X1 e X1s fica misturada com lote até esses controles fecharem.

**Por que L1 = 32 e não 64:** é a única largura que chega a 1280. A sonda codifica 64 imagens de cada vez, número fixo no código (`autoencoder_flim_module.py:908`), então baixar o lote não baixa esse pico. A 1280 com L1 = 64 ele sozinho passa de 41 GB.

**Modelo de memória**, ajustado em `s1/b2_real.tsv` (15 linhas, erro de −4 a +9%):
- Worker Ray: `538 + 16 bytes x (n x classes)²` MiB (D1 dizia 11 bytes).
- train.py, alocado: `cache + max(462·B·(w/32)·(N/640)²·f, 22.000·(w/128)·(N/640)²)` MiB, em que cache = `min(2048, treino)·3·N²·2` bytes e `f` vem de B1 (1,14 / 1,21 / 1,31 em L2 / L3 / L4; 1,35 em L5 é chute).
- Reservado = 1,17 x alocado, mais 346 MiB.

| N | pior célula | soma estimada (MiB) | base |
|---|---|---|---|
| até 200 | aid, L4 | 3.900 | escala de b2 |
| 320 | aid, L5 | 8.350 | escala de b2 |
| 448 | aid, L5 | 15.300 | escala de b2 |
| 600 | aid, n = 450 | 27.000 | escala de b2 |
| 640 | Eggs, L5 | 29.900 | medido até L1; L5 é chute |
| 896 | Larvae, L5 | 26.800 | **extrapolado** |
| 1280 | Larvae, n = 2048 | 33.000 | **extrapolado** |

Teto: 41.000 MiB de moliveira por GPU. Tudo cabe, mas **nada acima de 640 foi medido**. Antes de lançar 896 e 1280, rodar 3 pontos curtos como `b2_real.py`: Larvae 1280 n = 2048 lote 8, Larvae 1280 n = 128 com L5, Larvae 896 lote 16.
De 600 para cima: 1 run por GPU, só GPUs 2 e 3. SVM de 896 e 1280 fora da GPU que estiver treinando tamanho grande (~12 GB estimados, 33 + 12 passa do teto).

## 7. O que cada resultado significa

Empate: 0,02 na média de 3 splits (regra de D1). **Larvae é mais ruidoso: abaixo de ~0,05 não decide** (240 de validação). Conta aproximada do erro padrão da média: Larvae 0,05; flowers e helminth_eggs 0,02; Eggs, aid e Protozoan 0,01.

| se acontecer | significa | vira regra |
|---|---|---|
| Curvas de κ contra x caem juntas; mesmo x com N diferente empata | H1 vale | `N = x*·campo(d) / (objeto/quadro)` |
| Mesmo x, N maior ganha | Razão importa, detalhe também | Subir N até o erro de pixel sumir; depois, profundidade |
| N* de Larvae alto, N* de Eggs baixo | Manda a menor escala do objeto | Medir espessura, não bbox |
| Todos os N* perto de 200 a 320, apesar do objeto | Não há lei de razão (H1 cai) | Tamanho fixo, simples |
| Braço C plano, braço P muda | H3 vale | n fixo, independente de N |
| Braço P plano, braço C muda | H2 vale, contra o prior | n ∝ N², A fixo |
| Os dois planos | Superpixel não interage com N | n pelo X1; N por custo |
| P+ ganha de P a 128 px | Perda era falta de candidatos | Subir `images_per_class` em imagem pequena |
| helminth_eggs melhora a 320 ou 640 (100% ampliado) | Ganho é geometria pura | Ampliar é barato e útil: objeto pequeno pede ampliar |
| helminth_eggs não melhora ao ampliar | Ganho de N é informação (H4) | Nunca passar do nativo |
| Protozoan melhor a 96 do que a 200 | Ampliar atrapalha | D1 a 200 está errado para Protozoan |
| L4 e L5 ganham de L3 em N alto, e a melhor camada sobe com N | H6 vale | Lei composta: dobrar N, +1 camada |
| Profundidade não ajuda em nenhum N | Campo não é o limite; o pooling global já resolve | H1 cai junto |
| Filtros distintos caem quando n passa das posições | H5 vale | Limitar n pelas posições da camada |
| 8x4 difere de 32x1 | Bug | Parar: 896 e 1280 não valem |
| 8x1 difere de 32x1 em mais de 0,02 | Lote pesa sozinho | X1 e X1s não comparam tamanhos com lotes diferentes |
| Tudo empata | H0 | Menor N que empata |

**Como achar N\*:** por dataset e profundidade, N* é o **menor** N que empata com o melhor em validação. Depois, ajustar log N* contra log(objeto/quadro). Inclinação −1 confirma a razão. A lei só passa se, deixando um dataset de fora, ela acertar o N* dele a um degrau da escada de tamanhos.
São só 5 datasets com objeto medido (aid não tem objeto único): serve para falsear, não para cravar expoente.

**Sinais parciais de X1**, sonda, L1, 2 a 3 splits, larguras diferentes entre níveis: não decidem nada.
- flowers: 0,534 a 200 px contra 0,524 a 320. Plano até aqui.
- Fundo real (Eggs_subgroup): n menor ganha (0,30 com 800 pontos, 0,50 com 50, 0,54 com 12).
- Fundo preto: helminth_eggs e helminth_larvae preferem n maior (0,45 contra 0,29; 0,78 contra 0,55, com 800 contra 50). protozoan_cysts não mostra isso (0,22 contra 0,26).
- Não sei por que o fundo preto oscila entre splits: falta de pontos no objeto eu já descartei.
- Ainda não há SVM de validação de X1. Só existe o CSV de teste, que não abri.

## 8. Lista exata de YAMLs

**Molde:** copiar `experiments/autoencoder/disc_md_x1_s640_p3200_w128.yaml` e trocar:
- `name: spifil_growth_<nome>`, `work_dir: /exps/moliveira/spifil_growth/<nome>`, `grid.datasets`, `image_size`, `n_superpixels`.
- `layerwise: d`, `layer_channels:` os d primeiros de `[32, 48, 64, 64, 64]`, `layer_kernels:` d vezes 5, `layer_pools:` d vezes 2.
- Tirar o comentário "so L1" de `layerwise`.
- Em `overrides`, sempre `data.init_args.batch_size: B`; se acumulado > 1, `trainer.accumulate_grad_batches: a`.
- Se ipc > 1, `images_per_class: ipc` em `runner_args`.
- `resources.gpu_ids`: 2 ou 3 de 600 px para cima; 0 ou 1 abaixo. `max_per_gpu: 1`.

| nome (`experiments/autoencoder/<nome>.yaml`) | datasets | N | n | d | ipc | B | acum | braço |
|---|---|---|---|---|---|---|---|---|
| disc_md_res_s0064_n0128_d3_a | Protozoan | 64 | 128 | 3 | 1 | 32 | 1 | C |
| disc_md_res_s0096_n0012_d3_a | helminth_eggs, flowers, Protozoan | 96 | 12 | 3 | 1 | 32 | 1 | P |
| disc_md_res_s0096_n0128_d3_a | helminth_eggs, flowers, Protozoan | 96 | 128 | 3 | 1 | 32 | 1 | C |
| disc_md_res_s0128_n0020_d3_a | helminth_eggs, flowers, Protozoan | 128 | 20 | 3 | 1 | 32 | 1 | P |
| disc_md_res_s0128_n0020_d3_i2_a | Larvae | 128 | 20 | 3 | 2 | 32 | 1 | P |
| disc_md_res_s0128_n0020_d3_i6_a | flowers | 128 | 20 | 3 | 6 | 32 | 1 | P+ |
| disc_md_res_s0128_n0128_d4_a | helminth_eggs, flowers, Protozoan, Larvae | 128 | 128 | 4 | 1 | 32 | 1 | C |
| disc_md_res_s0200_n0050_d3_a | helminth_eggs, flowers, Protozoan, Larvae | 200 | 50 | 3 | 1 | 32 | 1 | P |
| disc_md_res_s0200_n0050_d3_b8a1_a | helminth_eggs | 200 | 50 | 3 | 1 | 8 | 1 | lote |
| disc_md_res_s0200_n0050_d3_b8a4_a | helminth_eggs | 200 | 50 | 3 | 1 | 8 | 4 | lote |
| disc_md_res_s0200_n0128_d4_a | helminth_eggs, flowers, Protozoan, Larvae | 200 | 128 | 4 | 1 | 32 | 1 | C |
| disc_md_res_s0320_n0128_d5_a | helminth_eggs, flowers, Larvae | 320 | 128 | 5 | 1 | 32 | 1 | P e C |
| disc_md_res_s0448_n0128_d5_a | flowers, Larvae | 448 | 128 | 5 | 1 | 32 | 1 | C |
| disc_md_res_s0448_n0251_d3_a | flowers, Larvae | 448 | 251 | 3 | 1 | 32 | 1 | P |
| disc_md_res_s0640_n0128_d5_a | helminth_eggs, Larvae | 640 | 128 | 5 | 1 | 32 | 1 | C |
| disc_md_res_s0640_n0512_d3_a | helminth_eggs, Larvae | 640 | 512 | 3 | 1 | 32 | 1 | P |
| disc_md_res_s0896_n0128_d5_a | Larvae | 896 | 128 | 5 | 1 | 16 | 2 | C |
| disc_md_res_s0896_n1004_d3_a | Larvae | 896 | 1004 | 3 | 1 | 16 | 2 | P |
| disc_md_res_s1280_n0128_d5_a | Larvae | 1280 | 128 | 5 | 1 | 8 | 4 | C |
| disc_md_res_s1280_n2048_d3_a | Larvae | 1280 | 2048 | 3 | 1 | 8 | 4 | P |
| disc_md_res_s0096_n0012_d3_b | Eggs | 96 | 12 | 3 | 1 | 32 | 1 | P |
| disc_md_res_s0096_n0128_d3_b | Eggs | 96 | 128 | 3 | 1 | 32 | 1 | C |
| disc_md_res_s0128_n0020_d3_b | aid, Eggs | 128 | 20 | 3 | 1 | 32 | 1 | P |
| disc_md_res_s0128_n0020_d3_i6_b | aid, Eggs | 128 | 20 | 3 | 6 | 32 | 1 | P+ |
| disc_md_res_s0128_n0128_d4_b | aid, Eggs | 128 | 128 | 4 | 1 | 32 | 1 | C |
| disc_md_res_s0200_n0050_d3_b | aid, Eggs | 200 | 50 | 3 | 1 | 32 | 1 | P |
| disc_md_res_s0200_n0128_d4_b | aid, Eggs | 200 | 128 | 4 | 1 | 32 | 1 | C |
| disc_md_res_s0320_n0128_d5_b | aid, Eggs | 320 | 128 | 5 | 1 | 32 | 1 | P e C |
| disc_md_res_s0448_n0128_d5_b | aid, Eggs | 448 | 128 | 5 | 1 | 32 | 1 | C |
| disc_md_res_s0448_n0251_d3_b | aid, Eggs | 448 | 251 | 3 | 1 | 32 | 1 | P |
| disc_md_res_s0600_n0128_d5_b | aid | 600 | 128 | 5 | 1 | 32 | 1 | C |
| disc_md_res_s0600_n0450_d3_b | aid | 600 | 450 | 3 | 1 | 32 | 1 | P |
| disc_md_res_s0640_n0128_d5_b | Eggs | 640 | 128 | 5 | 1 | 32 | 1 | C |
| disc_md_res_s0640_n0512_d3_b | Eggs | 640 | 512 | 3 | 1 | 32 | 1 | P |

**Sufixo:** `_a` é o lote 1 (20 YAMLs), `_b` é o lote 2 (14). São YAMLs separados de propósito: `work_dir` próprio, e relançar o mesmo YAML com outros datasets esbarraria em `skip: state`, que não pula growth pronto. Dentro de cada um, o dataset barato vem primeiro.

**Opcional, fora da conta (decisão do usuário):** Eggs a 896, 94% ampliado, +45,5 horas de slot, pico 34.800 MiB. Seriam 2 YAMLs a mais: `disc_md_res_s0896_n0128_d5_b` e `disc_md_res_s0896_n1004_d3_b`, só Eggs, lote 16x2.

**Eval, 10 YAMLs:** copiar `disc_md_eval_x1_s640_val_gemttapw.yaml`. Nome `disc_md_eval_res_s<NNNN>_val_gemttapw`. Trocar `name`, `work_dir`, `results_csv`, `grid.datasets`, `family: disc_md_res_s<NNNN>_*`, `image_size: N`, `stages: [L1, L2, L3, L4, L5]`. Manter `embed_mode: gem_tta_pw`, `eval_set: val`, `no_crop: true`.
O glob pega `_a` e `_b`. Tamanhos: 64, 96, 128, 200, 320, 448, 600, 640, 896, 1280; datasets de cada um conforme a tabela acima.
Não testado: `stages` com camada que não existe em todo braço (o filtro de `eval/growth_stages.py:297` deve só ignorar). Conferir no primeiro eval.

**Conferido:** `python -m experiments.ray.launch --dry-run` nos 34, a partir de cópias em `<scratchpad>/law/yamls/`: 34 de 34 ok, e `accumulate_grad_batches=4` aparece nos 9 comandos de treino do último YAML (`s1280`). Os 10 de eval não foram testados: ainda não há checkpoint. Decoder aceita qualquer N (`resnet_decoder.py:86-87`), testado em 64 a 640 com d até 5. Nenhum tamanho desses rodou treino.

## 9. Limites

- Objeto: 48 imagens por dataset. Contraste é pseudo-recorte só de análise, **sem validação contra verdade**; a faixa da tabela é só a diferença entre dois jeitos de medir (diâmetro equivalente e caixa). A espessura de Larvae raw vem da forma de helminth_larvae, não de medida direta.
- Geometria de superpixel com 2 a 3 imagens por ponto, LAB aproximado, não o `LabNorm`.
- Memória e velocidade acima de 640 extrapoladas. L5 sem medida. Velocidade de L2 a L5 assumida igual à de L1.
- Com n fixo em todas as camadas, a razão por camada não é testada. n por camada precisa de código (R4); deixei fora.
- Largura fixa `[32, 48, 64, ...]`. Se X1 achar que L1 maior ganha muito, a grade fica medindo o tamanho numa largura pior: reavaliar antes do lote 2.
- A = 800 e n = 128 vêm do ponto central de D1. Se X1 apontar outro ponto, mudar `A0`, `NC` em `plan.py` e regenerar.
- Dimensão da leitura muda com a camada (32 a 64), o que entra na comparação entre camadas.
