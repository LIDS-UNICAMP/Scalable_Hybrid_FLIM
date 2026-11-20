# P4: arquitetura escalável da CNN SPiFiL híbrida (B0 a B3) e baselines

Plano. Nada foi implementado, treinado ou lançado. Sem GPU: as contas abaixo rodaram em CPU, e os pesos oficiais só passaram pela RAM.
Busca: grafo primeiro. O grafo não tem nó para chave de YAML, número de log nem linha de CSV: para esses usei grep e leitura direta.
Scripts das contas: `<scratchpad>/p4/` (`family.py`, `real2.py`, `table.py`, `distinct.py`, `effnet_load.py`, `hub2.py`).

## 0. Resposta curta

**A família B0 a B3 é o modelo oficial de hoje, crescido com três botões: mais camadas (d), mais filtros (w) e imagem maior (r). A cada degrau o custo dobra.**

Cinco coisas mudam o plano:

1. **A conta do EfficientNet não vale aqui.** Lá, custo ≈ d·w²·r². Aqui uma camada a mais custa só 3%, porque ela roda na grade pequena. O custo real é ≈ w^1,8 · r². Por isso a profundidade não entra no orçamento: ela segue a imagem, para o campo de visão acompanhar.
2. **O teto de 5M não aperta.** O B3 tem de 0,73M a 1,92M de parâmetros. O que aperta é a oferta de candidatos do SPiFiL: 50 superpixels dão só 50 × classes candidatos. Com 2 classes são 100.
3. **A cabeça 1×1 é o elo fraco.** De 1280 filtros, a parte que nasce do SPiFiL vai de 8% (2 classes) a 100% (30 classes). Em helminth_eggs, sem contar as cópias, sobram 13%. E o backprop quase não mexe nessa camada, então o sorteio fica para sempre. A fração tem de ser registrada, e subir exige código novo (seção 12).
4. **7 dos 12 datasets não têm imagem maior que 200 px.** Neles ampliar não traz informação nova, então o orçamento vai todo para w (tabela 6.3).
5. **O benchmark atual é mais fraco que a nossa leitura.** Nós usamos GeM + TTA + PCA-whitening; os modelos do CSV usam vetor cru. Para ser justo, a leitura tem de ser igual, ou a diferença tem de ser medida.

EfficientNet é só baseline. Os pesos oficiais `google/efficientnet-b0..b3` carregam no env pelo `transformers`, sem chave faltando. DINOv3 não carrega hoje: falta login e o `transformers` do env não tem o modelo.

## 1. O que são d, w e r

Pense num prédio: **d** é o número de andares, **w** é quantas pessoas cabem em cada andar, **r** é o tamanho do terreno.

| botão | o que é | chave no YAML | B0 |
|---|---|---|---|
| **d** | nº de convoluções espaciais (kernel maior que 1). São 3 fixas com pool, mais `m` blocos 3×3 com soma residual. d = 3 + m | `layerwise` = d + 1; listas `layer_*` | 3 |
| **w** | multiplicador de largura de `[32, 48, 64]` | `layer_channels` | 1,0 |
| **r** | lado da imagem depois do letterbox, em px | `image_size` | 200 |

**Fixos em toda a família:**
- 50 superpixels DISF (`n_superpixels: 50`, `superpixels: disf`).
- Kernels 5, 5, 5 nas três primeiras camadas, 3 nos blocos, 1 na cabeça. Pool 3 com passo 2 nas três primeiras.
- **Cabeça c = 1280.** Ela não escala com w, por três motivos: (a) mais filtros seriam mais sorteio, (b) a leitura fica igual entre B0 e B3 e abaixo de 2000-d, (c) o mapa da cabeça já tem 6× mais números que a imagem, e aumentá-lo só facilita copiar na reconstrução (`agentes/L_classic.md`, tabela da seção 3, Vincent). É um eixo próprio: a grade `disc_md_sc_*_c{640,1280,2560}` já mede isso.
- Leitura GeM + TTA de 8 vistas + PCA-whitening + SVM linear C=100 (`embed_mode: gem_tta_pw`; `eval/svm.py:366`, `core/svm.py:143`).

Blocos ficam depois da L3, na grade pequena. Motivo: é onde a soma cabe de graça. A soma exige mesma largura, kernel ímpar e sem pool (`methods/autoencoder/mechanisms.py:70-75`), e é a forma que a grade `disc_md_sc_*_d*` já roda.

## 2. O B0 de partida

**B0 é a forma do modelo oficial `disc_bp01w64c1280_bg1`, a 200 px, sem máscara, sem residual, com 50 superpixels.**

```
L1 conv 5×5, 32 → pool   L2 conv 5×5, 48 → pool   L3 conv 5×5, 64 → pool   L4 conv 1×1, 1280
grade 200 → 99 → 49 → 24     200.944 parâmetros     704 milhões de MACs     campo de visão 43 px
```

Por quê: é o único ponto com histórico. A lei do outubro já usa 200.944 (`A_reports/november/scaling_law.md`).
Um MAC é uma multiplicação com uma soma. 1 MAC = 2 FLOPs.

Onde o custo mora, em B0: L1 14%, **L2 53%**, L3 26%, cabeça 7%. Esse é o motivo de o modelo ter poucos parâmetros e muita conta: a 5×5 roda em imagem quase inteira.

## 3. Como contar parâmetros e FLOPs

Com grades g₁ = r e g₍ₗ₊₁₎ = ⌊(gₗ − 3)/2⌋ + 1, ou seja g₂, g₃ e g₄ depois de cada pool:

```
P   = 75·w1 + 25·w1·w2 + 25·w2·w3 + 9·m·w3² + w3·c        (pesos)
      + w1 + w2 + w3 + m·w3 + c                            (bias)
      + m·w3                                               (ganho da soma, 1 por filtro)

MAC = r²·75·w1 + g₂²·25·w1·w2 + g₃²·25·w2·w3 + g₄²·(9·m·w3² + w3·c)

FLOPs = 2·MAC           por imagem, 1 vista, só o encoder
FLOPs com TTA = 8 × isso
Treino por imagem ≈ 3 × (MAC encoder + MAC decoder)        (ida + volta)
```

ReLU, pool e GeM ficam de fora: são pequenos e `FlopCounterMode` não os conta.

**Conferido contra o encoder real** (`Encoder` + `apply_mechanisms`, com `FlopCounterMode`, já usado em `analysis/stats/compute_cost.py:296`). Bateu exato nos 9 casos, por exemplo:

| w | m | c | r | P fórmula | P real | MAC fórmula | MAC real |
|---|---|---|---|---|---|---|---|
| 32, 48, 64 | 0 | 1280 | 200 | 200.944 | 200.944 | 703,9M | 703,9M |
| 36, 54, 72 | 1 | 1280 | 240 | 288.902 | 288.902 | 1.298,8M | 1.298,8M |
| 57, 85, 114 | 2 | 1280 | 336 | 749.490 | 749.490 | 6.168,1M | 6.168,1M |

**O que o decoder custa.** Ele é jogado fora depois, então não conta nos 5M, mas pesa no treino: é 2,4× o encoder em MACs e 4,8× em parâmetros no B0.

**Expoentes locais em B0** (contas com a divisão de custo acima):
- r: 2,0.
- w: 0,136 + 2·(0,535 + 0,262) + 0,067 = **1,8** (a L1 cresce linear, porque a entrada tem 3 canais; a cabeça também, porque c fica fixo).
- Bloco a mais: **+3%** (21M de 704M).

Lei de orçamento que substitui `α·β²·γ² ≈ 2`:

```
β^1,8 · γ² ≈ 2     por degrau, com d fora da conta
```

Regra geral: nunca use essa aproximação para o número final. Ela só propõe; quem aprova é a fórmula exata de MAC.

## 4. Como arredondar para filtros inteiros

**A regra: toda largura tem de ser múltiplo do número de classes.** O SPiFiL dá `largura // classes` filtros para cada classe e joga o resto fora (`SPiFiL/src/spifil/allocation.py:53`). Isso já acontece sem aviso: em helminth_eggs, 32 / 48 / 64 viram 27 / 45 / 63 (`/exps/moliveira/queues/logs/sc.log`, "32 pedidos -> 27 filtros").

Passos, nesta ordem:

1. **Alvo:** T = 703,9M × 2^φ.
2. **Largura:** `w_l = C · max(1, round(w⁰_l · β^φ / C))`, com C = classes. Depois force w1 ≤ w2 ≤ w3.
3. **Cabeça:** múltiplo de C mais perto de 1280, sempre abaixo de 2000.
4. **Imagem:** múltiplo de 8 mais perto de 200·γ^φ, sem passar do teto do dataset.
5. **Blocos:** `m` da seção 6.
6. **Acerto fino:** calcule o MAC exato. Fora de ±6% do alvo, mexa r em passos de 8 px. Se r estiver no teto, some C em w2 e w3, onde está o custo.
7. **Confira os limites** da seção 5. Se nada mais absorver o orçamento, **a família para ali**. Não se inventa modelo só para fechar a tabela.

Exemplo, 9 classes, B2: β² = 1,47 dá 47, 71, 94. Múltiplos de 9: 45, 72, 90. 200·γ² = 283 vira 280. MAC exato 2.818M, 4,00× o alvo. Pronto.

B0 fica em r = 200 sempre, sem acerto fino, porque é a âncora. Ele varia de −15% (14 classes) a +12% (30 classes). **Em todo gráfico, o eixo é o custo real, nunca o nominal.**

## 5. Limites reais

| limite | de onde vem | regra | o que acontece hoje |
|---|---|---|---|
| **L1 ≥ classes** | allocator: `0 // classes` não dá filtro | w1 ≥ C; melhor 2C | só aid aperta: 32 dá 30 filtros, 1 por classe |
| **Candidatos por filtro** | Q = 50 · C · ipc por camada; medido N=450 com 9 classes (`sc.log`) | k = w_l/C ≤ 50 · ipc | com 2 classes, w3 acima de 100 pede `images_per_class` ≥ 2. Fora da última camada e dos blocos, o que faltar **some sem aviso**: o `fill_random` só liga nesses dois (`growth.py:403-404, 416-417`) |
| **Escolha de verdade** | se k chega perto de 50, todo candidato entra | manter k ≤ 25 · ipc quando der | Larvae B2: k = 47 de 50. Aí não há seleção, só uso |
| **N contra D** | nota de Fisher precisa N > D; D = k²·C_in | `budget_check: false` já está nos YAMLs (`flim/spifil.py:327`) | N/D de 0,05 (2 classes) a 2 (30 classes) em L2 e L3. Largura piora: é sinal de que a nota enfraquece, não trava |
| **1×1 majoritariamente aleatória** | Q ≪ 1280 | fração é registrada; ver abaixo | 8% a 100% nominal; 13% real em helminth_eggs |
| **Cópias** | sementes colidem e o fundo preto repete | regime recortado: `redundancy_cos: 0.9999`, `redundancy_replace: true`, ipc 3 se faltar (`agentes/L_kernels.md`) | ver medidas abaixo |
| **Residual só com mesma largura** | `can_sum` (`mechanisms.py:70`) | blocos sempre na largura w3, sem pool | largura exata vem de `out_channels=0` (`growth.py:414`) |
| **Campo de visão** | campo = 43 + 16·m px | campo ≤ r é o único limite duro. Relatar x = objeto / campo | ver seção 6.4. BagNet e a nossa pista dizem que campo não é tudo |
| **Grade final** | pool extra deixaria 11² a 200 px | sem 4ª camada com pool abaixo de r = 320 | 50 sementes numa grade 11² colidem demais (`agentes/L_input.md`, H5) |
| **Memória** | teto 35 GB por GPU | lote efetivo 32; lote físico ≥ 8 | B3 ≈ 12 GB em lote 32 (seção 6.5). Não aperta |
| **Teto 5M** | regra do projeto | P ≤ 5M | B3 usa de 15% a 38%. Só chegaria perto em w ≈ 5 (5,15M com m = 2), custo acima de 16× |
| **Leitura < 2000-d** | regra do projeto | c ≤ 1920 | c = 1280 |

**Cópias medidas agora** (filtros iguais por cosseno > 0,9999, conv kernels reais, split 1, 50%):

| run | L1 | L2 | L3 | bloco L4 | cabeça |
|---|---|---|---|---|---|
| helminth_eggs, sem poda (`sc_w1_c1280`) | 10 de 27 | 21 de 45 | 32 de 63 | | 172 de 450 SPiFiL, 1280 no total |
| helminth_eggs, com poda (`base_dedup_s200`) | 27 de 27 | 45 de 45 | 63 de 63 | | 172 de 172, 1280 no total |
| helminth_eggs, w = 2, sem poda (`sc_w2_d4`) | 22 de 63 | 32 de 90 | 86 de 126 | 86 de 126 | |
| protozoan_cysts, com poda | 28 de 28 | 42 de 42 | 62 de 63 | | |

Leitura: a poda resolve o tronco, e a cabeça continua faltando filtro. **Nos quadros a cabeça a 50 superpixels ainda não foi medida:** sai de graça, do `labels1.txt`, quando a grade `sc` terminar em flowers e aid.

**Por que a fração da cabeça importa.** Na 1×1 o cosseno entre início e fim do treino é 0,97 (`A_reports/october/discovery_act_drift.md:30,47`): o que nasce sorteado morre sorteado. Em L2 e L3 não: 0,59 e 0,54.

**O alerta contrário.** Encher a 1×1 com patch cru do SPiFiL **piorou** na sonda sem treino: −0,049 e −0,076 (`agentes/L_classic.md`, tabela 1.1). Branqueado com dois sinais empatou ou ganhou. Mais SPiFiL cru na cabeça não é melhora garantida; isso precisa de braço próprio (seção 8).

## 6. B1, B2 e B3 a partir de φ

Sem dado ainda, parto de um **prior**: metade do orçamento para w, metade para r. Isso dá `β = 2^(0,5/1,8) = 1,21` e `γ = 2^0,25 = 1,19`. **As tabelas abaixo são o ponto de partida, não o resultado.** Quando a descoberta medir as parcelas s_w e s_r de cada dataset (com s_w + s_r = 1), basta trocar por `β = 2^(s_w/1,8)` e `γ = 2^(s_r/2)` e rodar o mesmo arredondamento.

**Profundidade:** `m = 0, 1, 1, 2`. Duas regras chegam nesse número:
- `d = round(3 · 1,2^φ)` dá 3, 4, 4, 5, com o α = 1,2 do EfficientNet.
- Manter campo / r perto de 0,215, a razão de B0: precisa de 51, 61, 72 px, e os blocos dão 59, 59, 75.

Atenção: hoje mais camadas pioraram (`discovery_insights_board.md:2009`: L4 0,875; d5 0,857; d6 0,851). Os braços `sc_*_d*` respondem isso. Se a profundidade continuar perdendo, a família fica com m = 0 e nada mais muda.

**Referência oficial do EfficientNet** (lida do `config.json` de cada repo): largura 1,0 / 1,0 / 1,1 / 1,2; profundidade 1,0 / 1,1 / 1,2 / 1,4; r 224 / 240 / 260 / 300. Custo real deles: 1× / 1,78× / 2,58× / 4,75×. Ou seja, o B3 oficial nem chega a 8×. Os números α = 1,2, β = 1,1, γ = 1,15 são do artigo (Tan & Le 2019), de memória, não reli.

### 6.1 Imagem cresce (Eggs, Larvae, flowers, remote_sensing, aid)

Formato: larguras / m / r. Custo em × de 703,9M.

| classes (datasets) | B0 | B1 | B2 | B3 |
|---|---|---|---|---|
| 2 (Larvae) | 32,48,64 / 0 / 200 | 38,58,78 / 1 / 240 | 48,70,94 / 1 / 280 | 58,86,114 / 2 / 320 |
| custo, parâmetros | 1,00×, 201 mil | 2,09×, 327 mil | 4,15×, 454 mil | 8,10×, 756 mil |
| 5 (flowers) | 30,50,65 / 0 / 200 | 40,60,80 / 1 / 232 | 45,70,95 / 1 / 280 | 55,85,115 / 2 / 328 |
| custo, parâmetros | 1,00×, 206 mil | 2,09×, 345 mil | 3,99×, 453 mil | 8,19×, 753 mil |
| 9 (Eggs) | 36,45,63 / 0 / 200 | 36,54,81 / 1 / 240 | 45,72,90 / 1 / 280 | 54,90,117 / 2 / 320 |
| custo, parâmetros | 1,02×, 196 mil | 1,93×, 325 mil | 4,00×, 436 mil | 8,14×, 787 mil |
| 14 (remote_sensing) | 28,42,70 / 0 / 200 | 42,56,84 / 1 / 232 | 42,70,98 / 1 / 280 | 56,84,112 / 2 / 328 |
| custo, parâmetros | 0,85×, 196 mil | 2,08×, 352 mil | 3,88×, 461 mil | 8,11×, 727 mil |
| 30 (aid) | 30,60,60 / 0 / 200 | 30,60,90 / 1 / 240 | 60,60,90 / 1 / 280 | 60,90,120 / 2 / 312 |
| custo, parâmetros | 1,12×, 216 mil | 1,98×, 373 mil | 4,19×, 420 mil | 8,35×, 826 mil |

Cabeça: múltiplo de C mais perto de 1280 (1278 com 9 classes, 1274 com 14, 1290 com 30).

**Larvae B3 quebra a oferta:** w3 = 114 passa de Q = 100. Precisa de `images_per_class: 2`, que vale para todas as camadas.

### 6.2 Quanto custa cada um, 9 classes (real, medido em CPU)

| modelo | parâmetros encoder | GMAC 1 vista | GMAC com TTA | decoder (só treino) | treino GMAC por imagem |
|---|---|---|---|---|---|
| B0 | 196.011 | 0,72 | 5,8 | 0,94M | 7,4 |
| B1 | 324.828 | 1,36 | 10,9 | 1,34M | 13,0 |
| B2 | 435.960 | 2,82 | 22,5 | 1,57M | 25,9 |
| B3 | 786.735 | 5,73 | 45,9 | 2,42M | 51,3 |

### 6.3 Imagem presa em 200 px (Protozoan, subgrupos, helminth_*, protozoan_cysts)

Esses 7 datasets têm lado nativo de 200 ou menos. Ampliar não traz informação (`agentes/L_input.md`, tabela de ampliadas: Protozoan a 200 px já é 97% ampliado). Então tudo vai para w:

| classes (datasets) | B1 | B2 | B3 |
|---|---|---|---|
| 2 (Larvae_subgroup, helminth_larvae) | 48,70,94 / m1: 2,10×, 454 mil | 70,98,130 / m1: 4,03×, 816 mil | 96,142,190 / m2: 8,19×, 1,92M |
| 7 (Protozoan, Protozoan_subgroup, protozoan_cysts) | 49,70,91 / m1: 2,10×, 441 mil | 70,98,133 / m1: 4,07×, 834 mil | 98,140,189 / m2: 8,17×, 1,90M |
| 9 (Eggs_subgroup, helminth_eggs) | 45,72,90 / m1: 2,03×, 436 mil | 72,99,126 / m1: 4,10×, 801 mil | 99,144,180 / m2: 8,26×, 1,83M |

**Com 2 classes essa regra esbarra na oferta:** w3 = 130 e 190 passam de Q = 100, e pedem ipc 2 a 3. Em helminth_larvae ipc 3 ajuda (+0,047). Em quadros ele custou −0,03 a −0,08, em simulação de L1 a largura 64 (`agentes/L_kernels.md`), e Larvae_subgroup é quadro. **Se ipc alto piorar lá, a família de Larvae_subgroup para em B1.** É um caso de regra compartilhada falhando, e entra como resultado.

**Teste barato antes de fechar o teto:** um braço de helminth_eggs com r crescendo mesmo assim. É a geometria pura de `L_input` (100% ampliado). Sinal de L1 a 50 superpixels, validação: 0,843 a 200 px, 0,851 a 320, 0,795 a 640. Plano, mas L1 enxerga 7 px, então não decide.

### 6.4 Campo de visão

Campo = 43, 59, 59, 75 px, em r = 200, 240, 280, 320: fica entre 21% e 25% do quadro. Logo x = objeto / campo fica parado na família: perto de 2,3 em Eggs, 2,1 em flowers, 1,9 em Protozoan, 0,7 em helminth_eggs, 0,8 em protozoan_cysts (frações de `L_input`).

Se H1 de `L_input` achar x* perto de 1, Eggs e flowers precisam de campo que 3×3 não dá barato. Saída, só com r ≥ 320: trocar um bloco por uma 4ª camada 5×5 com pool (campo 91). Já roda por YAML, mas sai sem residual.

No ramo da seção 6.3, o campo cresce em px e a imagem não: x cai de 0,7 para 0,43 em helminth_eggs. Ali a profundidade é sobra, não necessidade.

### 6.5 Memória

Modelo de `agentes/L_input.md` (seção 6, erro de −4 a +9%), reservado somado, lote 32, 2048 imagens na sonda:

| modelo | C = 9, seção 6.1 | ramo 200 px |
|---|---|---|
| B0 | 3,4 GB | 3,4 GB |
| B1 | 4,7 GB | 4 a 5 GB |
| B2 | 7,5 GB | 5 a 6 GB |
| B3 | 11,3 a 12,2 GB | 7,7 GB |

Âncoras medidas com w1 = 32, lote 32 (`agentes/B1.md`, tabela 3): 3,3 GB a 200 px, 7,7 GB a 320, 28,2 GB a 640. Largura acima de 32 e blocos residuais **não foram medidos**: é o modelo linear em w1. Um braço só de r a 8× (r = 560, w1 = 36) dá ~24 GB. Tudo em lote 32: a família não precisa de acúmulo, e isso é bom para as perdas (seção 7).

## 7. Como d, w e r mexem no SPiFiL e nas perdas

| | no SPiFiL | nas perdas |
|---|---|---|
| **w sobe** | Quota por classe sobe, com os mesmos candidatos: menos seleção. N/D cai (D cresce com a largura de entrada). Em quadros, abrir largura rendeu mais que escolher (+0,118, L1, `L_kernels`). Em recortados, mais cópias: 63 pedidos, 22 distintos | Dimensão de z muda só nas camadas do tronco, c fica fixa. `weak_sigreg` usa sketch K=16, então não muda com C (`losses/weak_sigreg.py:41`). Hipótese: filtros mais largos somam mais entradas, z fica maior e a perda sobe. Olhar `reg_ratio` |
| **d sobe** | Mais uma rodada do `grow`, com os mesmos 50 pontos e a mesma grade. Na 3×3, D = 9·w3, então N/D melhora (0,62 contra 0,33 a 9 classes). O bloco nasce com ganho 0,1: o filtro SPiFiL entra pela porta dos fundos, 2% da saída (`agentes/Y2.md`, seção 4) | Mais um estágio, e `reg_loss` age em cada um. Sem atalho cada camada reforça o brilho; a perda de variância pode segurar isso, é hipótese. +1 bloco no decoder custa pouco. +60 épocas por estágio: é o que pesa no tempo, não o FLOP |
| **r sobe** | N não muda: 50 são 50. A área por superpixel vai de 800 a 2.048 px a 320. O filtro cobre menos da região. Pontos no objeto: ~11 de 50, quase parado (`L_input`). Grade funda maior: menos colisão (24² a 200, 39² a 320), **não medido na cabeça**. Cópias de L1 **não caem** com r: helminth_eggs fica entre 0,43 e 0,65 de 64 a 640 px (`results/svm_disc_md_l1fast_*`) | Reconstrução é média por pixel, não muda de escala. Na prática r só ameaça o lote: cada lote físico é o que a perda vê, e acúmulo não aumenta isso. Abaixo de 8, ela nem roda (`autoencoder_flim_module.py:166`). Com lote 8, K=16 passa de N−1: use `k: 6` no bloco, que já chega à perda (`:447`). VICRegL paga por posição: custo cresce com r² |

Mais duas ligações que valem medir:
- **A perda muda o que o SPiFiL vê na camada seguinte.** O `grow` lê o encoder já treinado. Se a perda espalha z, a L4 pode ter mais candidatos distintos. Medir é de graça: filtros distintos em `conv*-kernels.npy`, com e sem `reg_loss`.
- **O piso da reconstrução é fixo na família.** Mapa da cabeça / imagem fica em ~6× de B0 a B3, porque c não cresce. É mais um motivo para não escalar c.

## 8. Regime de treino: SPiFiL + self-supervised

**Todo modelo nasce do SPiFiL e treina sem rótulo.**

- **Início:** `grow` camada a camada, 50 superpixels. Aleatório só completa, e a fração sai registrada. `random_layer: true` existe só como controle.
- **Treino:** autoencoder camada a camada, `layerwise_backprop: 0.1`, 60 épocas por estágio, como em `disc_md_sc_w2_d6_spf.yaml`.
- **Perda:** reconstrução + λ·R(z), z = GeM do topo do estágio. R vem de `losses/`, na ordem de `agentes/R8.md` (2.5): Weak-SIGReg, SIGReg, VICReg, Barlow. VICRegL e UR-JEPA só se os anteriores ajudarem. CorInfoMax fica de fora.
- **λ, médio para baixo:** `{0,01; 0,001}`. λ = 0,1 fica fora: passa de 14 a 60× a reconstrução no fim do estágio (`R8.md`, 2.3).
- **Um λ por família.** Escolha em B0, na validação, e leve para B1 a B3. Troque só se `reg_ratio` sair da faixa em algum estágio. Aviso de J1: em L2 a perda veio 18 a 50× maior que em L4 (6 passos, é aviso, não medida). λ por estágio seria chave nova, e só entra se isso se confirmar.
- **Decisão pela validação com SVM.** `best_by_probe` escolhe o checkpoint na sonda, e a própria sonda diverge do SVM em braço aleatório (`agentes/J1.md`, item 3).

**Honestidade sobre rótulo.** "Sem rótulo" vale para o gradiente. Mas o SPiFiL usa o rótulo de C × ipc imagens para escolher filtros (`flim/spifil.py`, `train_samples`), e o checkpoint é escolhido com rótulo de validação. Isso tem de sair escrito na tabela final.

**Braços da cabeça, só na descoberta, em B0 e B3:**

| braço | o que muda | pergunta | precisa |
|---|---|---|---|
| H0 | c = 1280, SPiFiL + resto aleatório | referência | só YAML |
| H1 | c = Q, sem nenhum aleatório | completar faz falta? | só YAML |
| H2 | `seed_point: union` | triplicar candidatos (N 294 contra 100, `J1.md`) sobe a fração sem custo? Vale para todas as camadas | só YAML |
| H3 | ipc só na cabeça | mais candidatos sem mexer em L1 | código, item 5 |
| H4 | branco + dois sinais na cabeça | SPiFiL que não perde na sonda | código, item 6, e decisão do usuário |
| Controle | `random_layer: true` | SPiFiL ganha do sorteio? | só YAML |

Forma do YAML, irmão de `disc_md_sc_w2_d6_spf.yaml` (exemplo, **não criado**):

```yaml
name:   spifil_growth_disc_md_b1_c09_r240
method: autoencoder
runner: growth
grid:
  init: [flim]
  datasets: [Eggs]
  splits: [1, 2, 3]
  percentages: [50]
runner_args:
  layerwise:  5
  impurities: true
  image_size: 240
  no_crop: true
  image_cache_dir: /exps/moliveira/image_cache
  layer_channels: [36, 54, 81, 81, 1278]
  layer_kernels:  [5, 5, 5, 3, 1]
  layer_pools:    [2, 2, 2, 1, 1]
  fill_random_last: true
  superpixels: disf
  n_superpixels: 50
  budget_check: false
  layerwise_backprop: 0.1
  reg_loss: {name: weak_sigreg, lambda: 0.01}
overrides:
  model.init_args.residual_gain: 0.1
  model.init_args.recon_bg_weight: 1.0
  model.init_args.epoch_probe: gem
  model.init_args.best_by_probe: true
```

Um YAML por degrau e por grupo de classes, porque `layer_channels` depende de C. Datasets de mesmo C e mesmo ramo dividem YAML. São 8 grupos, cerca de 30 YAMLs, mais os braços. Recortados levam ainda `redundancy_cos: 0.9999` e `redundancy_replace: true`.

## 9. Baselines: pesos, se carregam, custo

Regra do usuário: pesos oficiais do Hugging Face, repos dos autores, sem espelho `timm/`.

| modelo | repo oficial | carrega no env? | parâmetros | dim | r | GMAC por vista | ms por imagem |
|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | `google/efficientnet-b0` | **sim**, 0 chaves faltando | 4,01M | 1280 | 224 | 0,38 | a medir |
| EfficientNet-B1 | `google/efficientnet-b1` | **sim** | 6,51M | 1280 | 240 | 0,69 | a medir |
| EfficientNet-B2 | `google/efficientnet-b2` | **sim** | 7,70M | 1408 | 260 | 0,99 | a medir |
| EfficientNet-B3 | `google/efficientnet-b3` | **sim** | 10,70M | 1536 | 300 | 1,83 | a medir |
| ConvNeXt V2 Pico | `facebook/convnextv2-pico-1k-224` | já rodou, 216 células | 8,55M | 512 | 224 | 1,37 | 1 a 8 |
| I-JEPA g/16 | `facebook/ijepa_vitg16_22k` | já rodou, 216 | 1.011M | 1408 | 224 | 202 | 12 a 22 |
| DINOv2 giant | `facebook/dinov2-giant` | já rodou, 216 | 1.136M | 1536 | 518 | 1.783 | 185 a 314 |
| AIMv2 3B | `apple/aimv2-3B-patch14-448` | já rodou, 216 | 2.723M | 3072 | 448 | 2.940 | 181 a 308 |
| DINOv3 ViT-H+ | `facebook/dinov3-vith16plus-pretrain-lvd1689m` | **não**, ver abaixo | 841M | 1280 | 224 | 172 | a medir |
| DINOv3 ViT-7B | `facebook/dinov3-vit7b16-pretrain-lvd1689m` | **não** | 6.716M | 4096 | 224 | 1.363 | a medir |

GMAC de EfficientNet e Pico: `FlopCounterMode` em CPU. ViTs: mesma contagem, na arquitetura do timm em device `meta`. ms por imagem: `extract_ms_per_img` em `results/benchmark/*.csv`.

**EfficientNet: como carregar.**
- `transformers` 4.45.2 tem `EfficientNetModel`. Testei os 4 repos, com revisão fixa, só em RAM (139 MB no total): `missing 0, unexpected 0`, e `last_hidden_state` dá o mapa para o GeM. Revisões: b0 `3d8d75d0`, b1 `92124c4c`, b2 `faaffeb7`, b3 `130db530`. Licença apache-2.0, sem gate. Só `pytorch_model.bin`, sem safetensors.
- **timm 1.0.27 também carrega**, e testei `tf_efficientnet_b0.in1k` em RAM: todas as chaves bateram. Mas os pesos vêm de `timm/`, o espelho que o usuário vetou. Use timm só para a arquitetura em `pretrained=False`, no treino do zero.
- **Armadilha do pré-processamento.** `extract` tira o pré-processamento do timm (`benchmark/common.py:91`). Para os pesos do `google/` isso está errado: o repo usa desvio 0,4785 / 0,4733 / 0,4743, e o timm 0,229 / 0,224 / 0,225. Em B0 a B3 use o `EfficientNetImageProcessor` do próprio repo.
- Baixe com `HF_HOME` em `/exps`, nunca em `/dados`.

**DINOv3: três travas.** O repo é gated (`manual`, confirmado). O env não está logado (`huggingface-cli whoami`: "Not logged in"). E `transformers` 4.45.2 não tem `DINOv3ViTModel`. O timm 1.0.27 tem a arquitetura `vit_*_dinov3`, então o caminho é o mesmo do AIMv2: baixar o safetensors oficial e renomear (`benchmark/aimv2.py`). Sugiro ViT-H+ primeiro (3,4 GB). O 7B tem 26,9 GB, e seu vetor de 4096-d é mais largo que o resto. **Depende do login do usuário.**

**Pico segue sendo o único do benchmark com ajuste por rótulo no ImageNet**, e o EfficientNet entra igual: é treinado com rótulo. Os outros são auto-supervisionados em muito mais dado.

**Custo, comparado com B0 a B3 (GMAC por vista):**

| par | nosso | EfficientNet | razão de custo | razão de parâmetros |
|---|---|---|---|---|
| B0 | 0,72 | 0,38 | 1,9× nosso | 20× deles |
| B1 | 1,36 | 0,69 | 2,0× | 20× |
| B2 | 2,82 | 0,99 | 2,8× | 18× |
| B3 | 5,73 | 1,83 | 3,1× | 14× |

Ou seja: **nosso modelo gasta mais conta com muito menos parâmetros.** A comparação por parâmetros e a comparação por FLOPs vão contar histórias opostas. As duas têm de aparecer.

## 10. Protocolo justo

**Três regimes, nesta ordem de prioridade.**

| regime | o que é | rótulo no treino | quem roda | prioridade |
|---|---|---|---|---|
| **A. Congelado** | pesos oficiais parados, mesmo SVM | ImageNet (EfficientNet, Pico); não (DINO, I-JEPA, AIMv2) | EfficientNet B0-B3 + benchmark, 12 datasets × 6 frações × 3 splits | **obrigatório** |
| **B. Do zero, mesmo regime** | EfficientNet aleatória, reconstrução + mesma perda e λ, depois congelado + SVM | não | EfficientNet-B0 e B3, a 50%, 12 datasets, 3 splits | **recomendado** |
| **C. Do zero, com rótulo** | EfficientNet aleatória, entropia cruzada, sem pré-treino | sim | EfficientNet-B0, 3 datasets de confirmação, frações 1, 5, 50, 100 | opcional, só como teto |

O regime B é o mais justo: mesma entrada, mesmo alvo, só muda quem inicializa e a arquitetura. Ele responde "SPiFiL e arquitetura leve valem mais que uma rede moderna, sem pré-treino?". O regime C usa rótulo, que o nosso modelo nunca vê no gradiente: fica numa linha separada e nunca entra no aceite.

**Regras iguais para todos:**

1. **Mesmos dados:** mesmos splits, val e teste fixos, por nome de arquivo (`benchmark/common.py:127`).
2. **Mesmo SVM:** `core.svm.fit_svm`, linear, C=100, sob flock. Sem scaler.
3. **Mesma leitura, em três linhas:**
   - `cru`: o vetor oficial do modelo. Já existe para o benchmark.
   - `pw`: cru + `pw_whiten`. CPU, a partir dos `.npz`, sem extrair de novo.
   - `tta_pw`: média das 8 vistas + `pw_whiten`. Em CNN, GeM no mapa (`last_hidden_state`), como no nosso.
4. **Melhor leitura escolhida na validação**, por modelo. Teste só para relatar.
5. **Cada modelo com sua imagem e sua cor.** Nós LAB, eles RGB, com o pré-processamento do próprio repo (o erro antigo do I-JEPA com LAB, em `results_log.md`, não se repete).
6. **Linha extra com mesma resolução:** EfficientNet também em r igual ao nosso (B0: 0,34 GMAC a 200 px; B3: 0,86). Separa ganho de resolução de ganho de modelo.
7. **Mesmo número de passadas no regime B:** a cadeia B0 faz 4 estágios × 60 = 240 passadas; o EfficientNet recebe 240 épocas. Mesma regra de checkpoint, mesma leitura.
8. **Teste nunca decide nada.** Múltiplas sementes: splits 1 a 3; mais `seed` se a diferença ficar abaixo de ~0,02 (`L_classic`) ou ~0,05 em Larvae (`L_input`).
9. **Aceite:** κ ≥ melhor do benchmark − 0,01, relatando a diferença real. Melhor do benchmark = `tta_pw` se existir, senão a melhor leitura que rodou, com isso dito na tabela.

**O caso caro.** TTA em DINOv2 e AIMv2 custa cerca de 53 h cada (96.239 imagens × ~250 ms × 8). Proposta: `pw` em todos; `tta_pw` em Pico, I-JEPA, EfficientNet e DINOv3 H+; DINOv2 e AIMv2 só nos datasets onde a diferença para o nosso ficar abaixo de 0,02. Isso favorece a nós nos outros casos, e deve vir declarado.

## 11. O que comparar

Por dataset × fração, média e desvio de 3 splits:

| grupo | medida | fonte |
|---|---|---|
| Qualidade | κ, F1, acurácia, de validação e de teste | `compute_metrics` |
| Tamanho | parâmetros do encoder; decoder à parte | fórmula ou `count_params` |
| Conta | MAC e FLOPs por vista; com TTA | fórmula; `FlopCounterMode` para EfficientNet |
| Tempo | ms por imagem na extração; GPU-horas de treino; segundos de SVM | CSV, `elapsed_s` em `run_metadata.json` |
| Memória | pico por run | hoje não é gravado: item 7 da seção 12 |
| Origem | fração SPiFiL por camada, por filtro e por parâmetro; cópias | `labels1.txt` / largura (450 de 1280 conferido) |
| Rótulo | quantas imagens rotuladas tocaram o modelo | tabela fixa |

**Generalização entre frações.**
- Curva κ contra fração nas 6 frações, em escala log.
- Área sob a curva, a mesma para todos.
- **Retenção:** κ(5%) / κ(100%). É o número que diz se o SPiFiL ajuda com pouco dado, como a literatura prevê (`L_new_papers.md`, seção 3).
- **Matriz cruzada:** SSL em 100% sem rótulo, SVM em 1, 5 e 25%, por `svm_train_pct` (`schema.py`). Já há irmãos em `eval_svm_disc_bp01w64c1280_bg1_ssl100_svm*`. Só vale para os nossos e o regime B.

Fração baixa tem casos especiais (split 1, de `dinov2-giant.csv`): Larvae_subgroup a 1% tem 3 imagens; Eggs_subgroup, Protozoan_subgroup e remote_sensing a 1% não têm todas as classes. O SPiFiL ali fica com menos classes e menos candidatos, a largura real cai, e a fração SPiFiL da cabeça despenca. **Registre a largura real, não a pedida.**

## 12. O que é só YAML e o que exige código

**Só YAML, nada de Python:**
- d, w, r, c: `layerwise`, `layer_channels`, `layer_kernels`, `layer_pools`, `image_size`.
- Soma: `residual_gain`; `drop_path` se quiser.
- Oferta do SPiFiL: `n_superpixels`, `images_per_class` (global), `seed_point: union` (global), `redundancy_*`, `fill_random_last`.
- Controle aleatório: `random_layer`.
- Perdas: `reg_loss`, com `k` para lote pequeno. Lote: `batch_size`, `accumulate_grad_batches`.
- Leitura e frações: `embed_mode: gem_tta_pw`, `stages`, `eval_set`, `svm_train_pct`.
- Medir r novo: `tools/build_image_cache.py`, que já existe. O cache tem 64, 96, 128, 200, 256, 320, 448 e 640; **240, 232, 280, 312 e 328 precisam ser gerados**. Se quiser evitar isso, troque para 256 e 320 e reaplique o passo 6.
- Fração SPiFiL: `labels1.txt` já dá, offline.

**Código novo (lista, nada escrito):**

| # | o quê | onde | tamanho | sem isso |
|---|---|---|---|---|
| 1 | EfficientNet congelado: arquivo com 4 modelos, e `extract` aceitar transformação do próprio repo em vez do nome timm | `benchmark/efficientnet.py`, `benchmark/common.py:91` | pequeno | sem regime A |
| 2 | Leitura igual: `pw_whiten` no `_cell`; extração com 8 vistas; GeM no mapa em CNN | `benchmark/common.py` | pequeno | comparação fica cru contra `tta_pw` |
| 3 | DINOv3 oficial no ViT do timm, renomeando chaves. Depende de login | `benchmark/dinov3.py`, molde `aimv2.py` | pequeno | sem DINOv3 |
| 4 | EfficientNet do zero com reconstrução: encoder timm com `forward_features` + decoder. `ResNetDecoder` hoje lê o arch FLIM (`resnet_decoder.py:65-74`). Reusa `_reg_term` e o SVM | novo módulo em `methods/autoencoder/` | médio | sem regime B |
| 5 | ipc por camada: `layer_images_per_class`, em `RUNNER_ARGS` e em `run_arm`. Hoje é global (`GROW_KNOBS`, `growth.py:151`) | `schema.py`, `growth.py` | pequeno | sem H3; subir ipc mexe na L1 |
| 6 | Banco branco + dois sinais só na cabeça: `layer_bank` de `L_classic.md` seção 5. Dois sinais dobram a entrada da camada seguinte | `flim/spifil.py`, junto do `fill_random` (`:971`) | médio | sem H4. **Precisa decisão** (seção 14) |
| 7 | Pico de memória no `run_metadata.json`: `max_memory_reserved` ao fim do estágio. Hoje nada grava isso (grep vazio em `methods/` e no callback) | `experiments/run_metadata_callback.py` | mínimo | memória só por `nvidia-smi` |
| 8 | Tabela offline de custo: `architecture.json` + `labels1.txt` para parâmetros, MACs, campo, fração. Lacuna citada em `fila.md` | `analysis/stats/`, reusa `compute_cost.py:290,296` | pequeno | conta à mão |

**Só se o dado pedir:** λ por estágio (seção 8).
**Regime C** é o menos claro: `ClassificationFinetuneModule` já aceita `arch` do timm, `checkpoint_path=None` e `freeze=False`, então deve bastar config nova, mas `configs/model/classification/` só tem `flim`. Não testei.

## 13. Custo

Conta grosseira. Base: helminth_eggs B0 a 50% gasta ~22 s por época com 1.791 imagens, ou 12,3 ms por imagem e por época (`epoch_metrics.csv`, `sc_w1_c1280`). Eggs e Protozoan raw podem ser mais caros (`L_input`, seção 5); há cache de imagem.

Tempo relativo da cadeia, estimado: B0 1×, B1 1,5×, B2 2×, B3 3,5×. Vem de só dois pontos: época de L3 de 21 s (w1) para 31 s (w2), em GPU dividida, mais 4 / 5 / 5 / 6 estágios. A família soma ~8× o B0.

| escopo, 12 datasets, 3 splits | B0 | B0 a B3 | dias em 4 slots |
|---|---|---|---|
| 6 frações | 424 h | ~3.400 h | ~35 |
| 5, 50, 100% | 257 h | ~2.050 h | ~21 |
| só 50% | 83 h | ~660 h | ~7 |

**Conclusão: a grade cheia não cabe.** Sugestão: família inteira só a 50%; as outras 5 frações em B0 e no melhor B de cada dataset. A divisão descoberta / confirmação / externa corta mais.

| baseline | custo estimado |
|---|---|
| Regime A, EfficientNet | ~13 min por modelo por vista (96.239 imagens × ~8 ms, tirado do pior caso do Pico); cerca de 8 h nos 4 modelos, com TTA. SVM ~22 min por modelo e por leitura, ~4,5 h em série. Disco ~2 GB por leitura |
| `pw` nos 4 já prontos | só CPU, ~1,5 h de SVM |
| `tta_pw` no Pico e I-JEPA | ~1 h e ~3,6 h de GPU |
| Regime B, EfficientNet-B0 e B3 | ~83 h + ~170 h (B3 a 2×, chute), a 50%: ~2,6 dias em 4 slots |
| Regime C | ~20 h por dataset e por modelo, chute; não medi |

## 14. Decisões do usuário e limites

**Precisa de resposta:**
1. **Filtro branco com dois sinais, ou PCA de patches, ainda é "SPiFiL"?** Ele vem dos patches das sementes, mas não é mais um patch. Se não valer, H4 cai, e a cabeça fica com SPiFiL cru + resto aleatório. (Mesma pergunta de `L_classic.md`, seção 5.)
2. **DINOv3:** login no HF e aceite do gate, ou ele sai da tabela. Sugiro ViT-H+.
3. **Regime C entra?** Usa rótulo. Sugiro só como teto, em poucos datasets.
4. **Teto de r em 200 px para os 7 datasets:** aceita, ou prefere o teste de ampliação antes?

**Limites deste plano:**
- B1 a B3 são prior: metade para w, metade para r. Nenhum foi treinado. A divisão real vem da descoberta.
- Nenhum κ aqui prova que algum B é melhor. A grade `sc` ainda roda: só helminth_eggs passou de L1, e `results/` não tem SVM dela.
- Memória acima de w1 = 32 e com blocos residuais é modelo, não medida.
- Tempo de B1 a B3 sai de dois pontos, com GPU dividida.
- Cabeça em quadros a 50 superpixels: ainda sem medida. Colisões em r maior na cabeça também.
- Medidas de cópia: só helminth_eggs e protozoan_cysts, split 1, 50%.
- `law_size_superpixels` não existe com esse nome. Usei `results/svm_disc_md_l1fast_*`, leitura de validação, só L1. Ali κ fica plano ou cai com r, sem prova para camadas fundas.
- Com r preso a 200 em aid ou remote_sensing a família perde resolução: o braço só de r em B3 (r ≈ 560) cobre isso. O oficial a 640 custa 7,3 GMAC, 10,4× B0: acima do B3. Se esses datasets precisarem de 640, a família deles não começa em 200.
- O custo de B0 tem 53% na L2. Trocar 5×5 por 3×3 ali cortaria muito, mas k5 rendeu mais que k3 em L_classic (0,875 → ~0,900). Está fora de d, w, r; fica registrado.
