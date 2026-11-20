## VLAD na ultima camada (`vlad<K>_eq`)

**O que e:** em vez de espremer o mapa da ultima camada num vetor so (GeM), cada posicao do mapa vira um "descritor"
(o vetor de canais dela). Um k-means sem rotulo, fitado so no treino, acha K "palavras"; cada imagem soma o quanto
seus descritores diferem da palavra mais proxima. Depois: normaliza por palavra, raiz com sinal e L2.

- Codigo: `eval/spf_grade.py`, funcao `vlad` + bloco `books` em `readouts`; opt-in por `amostra.leituras: [vlad16, vlad32]`.
- Codebook: 32 posicoes sorteadas por imagem de treino (semente 0), `MiniBatchKMeans(K, random_state=0, n_init=3)`.
- So na ultima camada da celula. Cru tem `K x C` dimensoes (256 canais: 4096 e 8192), entao sai so o `vlad<K>_eq`:
  PCA fitada no treino ate a dimensao do `cat` de GeM (ou do `L` se nao houver `cat`); `raw_dim` guarda `K x C`.
  O cru so sai como linha `vlad<K>` quando fica abaixo de MAX_DIM. Custo entra em `extract_s`, `gpu_mib`, `rss_mib`.
- Identidade: `readouts` antigo x novo em tensor fixo (gem, gap, gmp, spp, pol, gem_p2, ms, cov): bit-identico.
- Fumaca (20 imagens de aid, encoder falso 64-96-200 canais, CPU 2 threads): vlad16 raw_dim 3200, vlad32 6400,
  vetores com norma 1; ~2,5-2,9 s por chamada incluindo k-means; pico RSS ~1 GB.

| YAML | bases | braco | controles | jobs |
|---|---|---|---|---|
| `disc_md_vlad_off(_jaci)` | aid, remote_sensing | fps (sem saliencia) | random_z, orth | 4 (6 celulas por base) |
| `disc_md_vlad_on(_jaci)` | flowers, coconut_trees | fpsw@mapa_w0.5_1 | random_z, orth | 4 (6 celulas por base) |

Receita L6, 256, entre_camadas, split 1, draws [42, 43], leituras gem (L, cat), vlad16_eq, vlad32_eq. Resultados em
`/exps/moliveira/results_sol/pool_foco/`. Nao lancados.


## Pool F1-F4: resolucao, leis + leitura, SoftHebb (08/11, nao lancado)

Meta: subir o kappa de validacao em aid, flowers, remote_sensing e coconut_trees sem backprop, sem modelo pre-treinado e sem rotulo ou mascara na construcao dos filtros; flowers e remote_sensing acima de 0,70 (DINOv2: ~0,97 e ~0,71). Teste nunca entra.

YAMLs em `experiments/autoencoder/disc_md_pf_*.yaml` (repo discovery). Cada um tem um gemeo `_jaci` (paths em /dados/home/moliveira/exps, `gpu_ids [0]`, `max_per_gpu 8`). Athena: /exps/moliveira, `gpu_ids [0,1,2,3]`, `max_per_gpu 1`; lancar com `GPU_CAP_MIB=4200`. Resultados em `<raiz>/results_sol/pool_foco/svm_disc_md_pf_*`; fontes em `<raiz>/fontes_filtros/pf_<item>`, todas novas. Piloto: split 1, sorteios 42 e 43.

Saliencia (o if/else): **SAL-OFF** = aid e remote_sensing, braco `fps`, sem chave `saliency`. **SAL-ON** = flowers e coconut_trees, `fpsw@mapa_w0.5_1` (receita) e, no F2/F3, `fps@foco_w0.5_1` (gmr, `pesos {t: [0.5], g: [1], foco: 400}`).

Controles aleatorios puros: `random_z` e `orth` (spf_grade) com a mesma arquitetura, leitura e protocolo, sem nenhuma tecnica. Eles passam pelo mesmo z-score e pelo mesmo `entre_camadas` da receita, como em sol_wsal, leis_p1 e gelu, e nunca usam pontos de saliencia (so o medoide). No SoftHebb o controle e o YAML `hebb_rand`: init kaiming, lr nunca aplicado.

### Tabela

Jobs = `python -m experiments.ray.launch <yaml> --dry-run` (0 erros no preflight nos 22); o `_jaci` tem o mesmo numero. Custo: tempo de celula estimado de leis_p1, sol_wsal_ext e hebb3_conf (fit + select + extract), com parede no Athena a 1 job por GPU. **Ordem de grandeza, nao medida.**

| item | hipotese | comparacao (pareada, mesmo sorteio) | YAML (Athena; + `_jaci`) | jobs | celulas | custo estimado |
|---|---|---|---|---|---|---|
| F1a | aid (600 px) e remote_sensing (640 px) perdem detalhe a 200 px; 320 recupera | fps 256x6 a 320 contra o mesmo a 200 (celula lei A w0 256 r1.0 do F2-OFF) e contra random_z/orth a 320 | `disc_md_pf_res320_off.yaml` | 4 | 12 | ~7 h de celula, ~2,5 h de parede |
| F1b | idem, 448 px | fps a 448 contra 320 e 200 | `disc_md_pf_res448_off.yaml` | 4 | 12 | ~12 h, ~4,5 h; **risco de passar de 4200 MiB** (LAB de 500 imagens a 448 ~1,2 GB na GPU; nenhum spf_grade L6 rodou acima de 200 px) |
| F1c | coconut (90 px) ampliado a 200 so interpola; 128 fica mais perto do nativo | fpsw@mapa_w0.5_1, larguras 64 e 256, a 128 contra o mesmo a 200 em `disc_md_largura_jaci` (Jaci, sorteios 42/43) | `disc_md_pf_res128_on.yaml` | 2 | 12 | ~2 h, ~1 h (+ cache) |
| F2-OFF | leis early-heavy (A 512 r0.71 d6 = 512-512-256-256-128-128; B late = 512-512-256-256-64-32) + multiescala igualada somam ganho | 3 arquiteturas (as 2 leis + ancora 256x6 = lei A w0 256 r1.0) x `cat` e `cat_ms_eq` (`leituras [gem, ms]`, `eq: true`); fps contra random_z/orth | `disc_md_pf_leis_off.yaml` | 4 | 36 | ~19 h, ~7,5 h |
| F2-ON | idem em flowers e coconut, com saliencia | mesmas 3 x 2; fpsw@mapa_w0.5_1 e fps@foco_w0.5_1 contra random_z/orth | `disc_md_pf_leis_on.yaml` | 4 | 48 | ~30 h, ~8 h |
| F3-OFF L3 | SoftHebb (receita hebb3) bate fps fixo em aid/RS tambem a 512 e com TTA | H05 (fps_w<W> p300) contra H01 (fps_w<W>_c1) e H03 (p0); W 256/512; gem_pw, cat_gem_pw, gem_tta_pw (+ cat_gem_tta_pw) | `disc_md_pf_hebb_d3_off.yaml` | 2 | 8 (24 redes) | ~1 h, <1 h |
| F3-OFF L4 | idem, L4 | idem, L4 contra L3 | `disc_md_pf_hebb_d4_off.yaml` | 2 | 8 (24 redes) | ~1,5 h, ~1 h |
| F3-ON L3 | idem em flowers/coconut, init nos pontos da saliencia | H05 = fps@foco_w0.5_1_w<W> p300 contra H01 = ..._c1 | `disc_md_pf_hebb_d3_on.yaml` | 2 | 8 (24 redes) | ~2 h, ~1,5 h |
| F3-ON L4 | idem | idem | `disc_md_pf_hebb_d4_on.yaml` | 2 | 8 (24 redes) | ~2,5 h, ~1,5 h |
| F3-ctl L3 | aleatorio puro (H00) na mesma rede SoftHebb | random_w<W>_c1 contra H01; random_w<W> p0 contra H03; 4 bases | `disc_md_pf_hebb_rand_d3.yaml` | 4 | 16 (48 redes) | ~2 h, ~1 h |
| F3-ctl L4 | idem | idem | `disc_md_pf_hebb_rand_d4.yaml` | 4 | 16 (48 redes) | ~2,5 h, ~1 h |
| F4 | — | **pulado**: `disc_md_largura_jaci.yaml` (so no Jaci, rodando agora) ja cobre coconut com 32/64/128/256 na receita, 3 splits, sorteios 42/43, fps e fpsw@mapa_w0.5_1, random_z/orth/xavier/tnorm | — | 0 | 0 | 0 |

Total no Athena: 34 jobs. Ordem sugerida: F3 e F3-ctl primeiro (baratos, sinal rapido), depois F1, e F2 por ultimo.

### Detalhes que mudam a leitura

- **Fontes por tamanho.** O `.pt` de `amostra.fontes` e `<base>_split<k>_d<sorteio>.pt`, sem o tamanho no nome. Por isso cada resolucao tem o seu YAML e a sua pasta: dois tamanhos no mesmo YAML leriam o LAB do outro.
- **Superpixels**: `l1_superpixels: [50]` em todo tamanho, ou seja, 50 sementes por imagem a 128, 200, 320 e 448 px; a area de cada superpixel cresce com a imagem. `foco: 400` e so a grade densa dos pontos `foco_*`, e so em SAL-ON.
- **Mapa final em L6**, com pool 2-2-2-2-1-1: 7 px a 128, 11 a 200, 19 a 320 e 27 a 448.
- **Leis**: 12,74M e 11,97M params de conv contra 8,21M da ancora, entao a lei ganha com mais parametros (ja sabido).
- **Leituras do F2**: `[gem, ms]` + `eq` da `L`, `cat`, `L_ms`, `cat_ms`, `L_ms_eq` e `cat_ms_eq`, cada uma tambem com `_b`. O `gem_p2` do leitura_a2 saiu: nao entra no `cat_ms_eq` e so somaria fits de SVM.
- **F3**: `hebb_init.depth` e um inteiro, entao L3 e L4 ficam em YAMLs separados. A 3a e a 4a camada repetem o ultimo lr e o ultimo inv_temp (0,005 e 0,65). `cat_gem_pw` sai sozinho de `gem_pw` (`score_reads`). `gem_tta_pw` existe em `eval/alt_encoders.embed` (4 rotacoes x flip). Em L3 w256, o F3-OFF repete celulas fps de hebb3_conf como ancora barata, agora com TTA.
- **hebb_rand**: com `steps: [0]`, a celula `random_w<W>` grava o p0 duas vezes (linhas identicas); deduplique na leitura. `steps: []` quebraria a figura do fim do job (`durations(spec)[-1]`).
- **Dry-run escreve** `work_dir/jobs/*.yaml`: os de /exps/moliveira/eval/disc_md_pf_* ficaram (sao regravados no lancamento). Os dos gemeos `_jaci` criados no Athena em /dados/home/moliveira/exps/eval/ foram apagados.
- GELU fica de fora (disc_md_gelu rodando).

### Cache de imagem

| base | tamanho | Athena (/exps/moliveira/image_cache) | Jaci (/dados/home/moliveira/exps/image_cache) |
|---|---|---|---|
| aid | 320 / 448 | existe (10000; 3,0 / 5,7 GB) | **falta** (so 200) |
| remote_sensing | 320 / 448 | existe (1000; 0,30 / 0,58 GB) | **falta** (so 200) |
| coconut_trees | 128 | **falta** (so 200) | **falta** (so 200) |

Sem cache, o `ParasiteDataset` avisa e le a imagem crua: o job roda, so mais lento. Para construir (nao construido aqui): `tools/build_image_cache.py <raiz> <processos> 1 coconut_trees:128`. O 200 de coconut levou ~1,5 min (13587 imagens, 1,6 GB), entao o 128 deve dar ~1,5 min e ~0,65 GB. No Jaci, o F1 precisa ainda de `aid:320,448` (~8,7 GB) e `remote_sensing:320,448` (~0,9 GB); a outra saida e rodar o F1 so no Athena.

### Criterios para avancar (piloto → confirmacao 3 splits x 5 sorteios)

Kappa de validacao, media dos sorteios 42 e 43, na leitura do item (`cat`, `cat_ms_eq`, `cat_gem_pw` ou `gem_tta_pw`).

1. **Contra o aleatorio pareado**: metodo − melhor controle puro do mesmo YAML (mesma arquitetura e leitura) ≥ +0,02 na media e positivo nos dois sorteios. Se nao passar, o ganho e da arquitetura e nao do filtro: registrar e nao avancar como metodo.
2. **Contra a ancora pareada**: F1 contra o mesmo braco a 200 px; F2 lei contra 256x6 na mesma leitura, e `cat_ms_eq` contra `cat`; F3 H05 contra H01. Precisa de ≥ +0,01 na media, com o mesmo sinal nos dois sorteios.
3. **Contra o melhor conhecido da base** (aid 0,775; flowers 0,690; remote_sensing 0,535; coconut 0,785): ≥ +0,01 vai para confirmacao. Cruzar 0,70 em flowers ou remote_sensing vai direto para confirmacao.
4. **Parar**: se o ganho sobre a ancora for < 0,005 nos dois sorteios, ou se o aleatorio subir junto (a margem do item 1 encolher). F1b para tambem se estourar o teto de GPU: registrar o pico e nao reduzir `images` sem pedir.
5. **Combinar** so depois da confirmacao: os vencedores de F1, F2 e F3 da mesma base viram um YAML de combinacao, com os mesmos controles puros.

## Resultado do VLAD sem saliência (`vlad_off`, athena, END 19:38 UTC rc=0; split 1, sorteios 42 e 43, receita 256 × 6)
κ de validação, mesma dimensão final (940):

| base | L (GeM, última camada) | cat (GeM, 6 camadas) | vlad16_eq | vlad32_eq |
|---|---|---|---|---|
| aid, fps | 0,630 | **0,776** | 0,654 | 0,669 |
| aid, melhor aleatório | 0,383 | 0,643 | 0,341 | 0,387 |
| remote_sensing, fps | 0,418 | 0,456 | 0,457 | **0,462** |
| remote_sensing, melhor aleatório | 0,262 | 0,451 | 0,267 | 0,324 |

- O VLAD da última camada ganha do GeM da última camada (aid +0,039; remote_sensing +0,044), mas perde da concatenação em aid (−0,107) e empata em remote_sensing.
- O VLAD aumenta a vantagem do SPiFiL sobre o aleatório na última camada: em aid, +0,28 contra +0,25 no GeM.
- Próximo passo possível: VLAD concatenado às camadas (`cat` + VLAD) com a mesma dimensão final. Hoje o VLAD sai só da última camada.

## VLAD com saliência (`vlad_on`, athena, END 20:11 UTC rc=0; split 1, sorteios 42 e 43, receita 256 × 6, `fpsw@mapa_w0.5_1`)

| base | L | cat | vlad16_eq | vlad32_eq |
|---|---|---|---|---|
| coconut_trees, receita | **0,788** | 0,630 | 0,643 | 0,689 |
| coconut_trees, melhor aleatório | 0,589 | 0,564 | 0,425 | 0,494 |
| flowers, receita | 0,583 | **0,630** | 0,604 | 0,604 |
| flowers, melhor aleatório | 0,291 | 0,534 | 0,332 | 0,363 |

- coconut_trees: a última camada sozinha (L, 256-d) é muito melhor que a concatenação (cat, 940-d): 0,788 contra 0,630. Isso explica o "64 filtros melhor que 256" visto antes: o problema é a leitura concatenada, não a largura. O VLAD fica entre os dois.
- flowers: `cat` continua sendo o melhor; o VLAD perde 0,026.
- Veredito: o VLAD da última camada não substitui o `cat` em nenhuma das 4 bases; ganha só do GeM da última camada (exceto em coconut_trees). A linha VLAD fica encerrada como leitura principal.
- Achado novo: em coconut_trees, ler só a última camada evita a perda da concatenação. Vale testar se a leitura deve depender da base (regra sem rótulo, p.ex. pela resolução nativa).

## coconut_trees com entrada de 128 px (`pf_res128_on`, athena, END 20:29 UTC rc=0; split 1, sorteios 42 e 43, receita L6)

| leitura | receita 64 | receita 256 | melhor aleatório 64 | melhor aleatório 256 |
|---|---|---|---|---|
| L (última camada) | 0,789 | **0,792** | 0,424 | 0,511 |
| cat (6 camadas) | 0,778 | 0,639 | 0,720 | 0,548 |

- Com 128 px, o resultado fica igual ao de 200 px (L 0,788): reduzir a resolução não ajuda nem atrapalha o coconut_trees.
- Confirmação: com 256 filtros, o problema está no `cat` (0,639), e não na largura; com L, 64 e 256 empatam (0,79).
- Com L, o SPiFiL fica +0,28 a +0,37 acima do aleatório; com cat, só +0,06 a +0,09. A concatenação dilui a vantagem do SPiFiL nessa base.

## Resolução de 320 px sem saliência (`pf_res320_off`, athena, END 20:35 UTC rc=0; split 1, sorteios 42 e 43, receita L6 `cat`, fps)
- aid: 0,801 / 0,794 a 320 px, contra 0,786 a 200 px (`sol_wsal`, split 1, seed 42): **+0,015**. O melhor aleatório a 320 px fica em 0,692 a 0,700.
- remote_sensing: 0,533 / 0,486 a 320 px, contra 0,432 a 200 px: **+0,054 a +0,10** (os dois sorteios variam muito). O melhor aleatório a 320 px fica em 0,449 a 0,481.
- Leitura (exploratória, 1 split): a resolução maior ajuda nas duas bases de sensoriamento remoto e aumenta a vantagem sobre o aleatório em remote_sensing. Falta o 448 px (na fila automática) e a confirmação com 3 splits.
