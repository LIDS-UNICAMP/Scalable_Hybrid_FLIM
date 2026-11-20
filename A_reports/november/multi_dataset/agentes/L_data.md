# L_data: o que os dados já medidos dizem sobre uma lei de montagem

Retrato de 06/11, ~05:15 UTC, com 53 runs `disc_md_x1_*` prontos (a grade segue rodando). Só leitura e CPU com `nice -n 10`; nada foi editado no repositório. Scripts e features nesta pasta (seção 7).

## 0. Resumo

1. **A sonda por época tem um bug, e ele explica o sobe e desce.** O treino da sonda passa por `float16` e a validação não. Em imagem de fundo preto isso desloca todas as features de um lado só. O mesmo checkpoint que registrou κ=0 dá κ=0,87 com a sonda corrigida.
2. **Só 3 datasets são afetados:** `helminth_larvae`, `helminth_eggs` e `protozoan_cysts`, os que têm 96–98% de fundo preto. Os outros passam ilesos. Classe rara e ordem de arquivo **não** explicam.
3. **A diferença treino/validação nesses 3 é falsa:** registrada 0,33 em média, real −0,007.
4. **Não existe "melhor época" na L1.** Nos datasets sãos a curva é plana, porque em 20 épocas os pesos mal saem do SPiFiL (mudança de 0,3% a 10%).
5. **Não há lei de potência em N.** Com 3 ou 4 valores de N, qualquer expoente serve. Dá para dizer o seguinte, tudo no split antigo:
   - largura: plana com gem em eggs (2×10⁵ a 6,3×10⁵);
   - profundidade: 4 camadas é o pico em eggs, com evidência fraca;
   - a 1x1 e a leitura rendem +0,04 a +0,05 cada, bem mais que largura;
   - dados: dobrar tira ~35% do erro, sem saturar até 50%.
6. **Benchmark:** nenhuma propriedade pedida (imagens, classes, desbalanceamento, resolução) explica o κ. A maior distância medida até o nosso antigo é −0,17 em Larvae_subgroup (fundo real).
7. **Faltam medidas baratas:** 21 evals de SVM sobre pesos que já existem preencheriam os maiores buracos (seção 6).

**Decisão que cabe a quem coordena:** corrigir a sonda antes de ler os `epoch_metrics` desses 3 datasets. É uma linha, mas é código em uso pela grade, então não mexi.

---

## 1. De onde vem cada número (leia antes das tabelas)

| marca | o que é | origem | compara com raw/ novo? |
|---|---|---|---|
| **[A]** | `results/svm_disc_*` | split antigo: `parasite_name: helminth-eggs_split_N` no `config.yaml` de cada run, pasta `new_split_parasito` (`config.py:72-90`). Imagens de fundo preto, 200x200. perc50 = 1276/1279/2557 (R1 §1.1) | **não** |
| **[S]** | `results/svm_main_*_impuro_*` | os splits do worktree `impuro` são symlink para `organized/*_subgroup`. `n_train` por split = [246, 256, 265], igual ao benchmark | só com o `_subgroup` do benchmark |
| **[R]** | benchmark Eggs, Protozoan, Larvae | `config.get_single_parasite_paths` devolve `.../raw/orig` | é a régua nova |
| **[F]** | `r1_val`, `r2_val` | leitura flatten (até 235.224-d), proibida na receita | **não**, nem com [A] gem/avg |

Nenhum número medido veio de `pre/`: o split antigo usa `new_split_parasito`, e `Eggs`, `Protozoan` e `Larvae` leem `raw/orig`. Então a marca que importa é **[A]**: **toda a seção 2 é [A]**, exceto onde está dito.

N é a contagem de parâmetros do encoder, calculada da arquitetura real de cada célula (`*_grow/architecture.json`). Bate com os 200.944 do relatório.

Intervalo: IC95 t sobre 3 splits, 2 graus de liberdade (multiplicador 4,30). Com 3 splits ele é largo de propósito: é o que os dados permitem. Bootstrap sobre 3 splits só tem 10 reamostras distintas, por isso não usei.

---

## 2. Pergunta 1: κ de validação contra tamanho

**Resposta curta:** os dados não sustentam uma lei de potência em N. Eles sustentam quatro fatos: largura quase não ajuda com gem, profundidade além de 4 piora em eggs, a 1x1 e a leitura rendem muito mais que largura, e dado é o eixo mais forte e não satura até 50%.

### 2.1 Largura, 4 camadas, 200 épocas [A]

| dataset, leitura | pontos (N → κ) | κ por dobra de N, IC95 | leitura |
|---|---|---|---|
| eggs, gem | 200.944 → 0,8985 e 0,9020 (dois treinos); 358.368 → 0,9014; 631.008 → 0,8998 | **−0,0001** [−0,008; +0,008] | **plano** |
| eggs, avg | 122.224 → 0,8676; 200.944 → 0,8797 e 0,8782; 358.368 → 0,8849; 631.008 → 0,8987 | **+0,0123** [+0,006; +0,018] | sobe, sem saturar |
| eggs, avg na L3 (sem 1x1) | 59.504 → 0,8190; 117.744 → 0,8417 e 0,8416; 234.208 → 0,8730; 465.888 → 0,8830 | **+0,0227** [+0,007; +0,039] | sobe, freando |
| larvae, gem, 200 ép. | 200.944 → 0,9096; 358.368 → 0,9192 | Δ +0,010 [−0,068; +0,087] | indefinido |
| larvae, gem, 60 ép. | 3.558 → 0,8532; 200.944 → 0,9034 | Δ **+0,050** [+0,030; +0,071], ou +0,0086 por dobra [+0,005; +0,012] | 56x mais params rendem 0,05 |
| larvae, avg | 122.224 → 0,9117; 200.944 → 0,9173 | +0,008 [−0,012; +0,028] | indefinido |
| protozoan, avg | 118.622 → 0,8283; 188.468 → 0,8370 e 0,8405 | +0,016 [−0,096; +0,127] | indefinido; só 2 valores de N |

Protozoan não tem nenhum eixo de largura com gem: as duas famílias têm o mesmo N.

**Onde satura:**
- Eggs com gem: já em N ≈ 2×10⁵ não há ganho até 6,3×10⁵. Não sabemos onde começa, porque não existe ponto gem abaixo de 2×10⁵ em eggs.
- Eggs com avg: não satura até 6,3×10⁵.
- O gem "come" o ganho de largura que o avg mostra. É o mesmo padrão de HAL §4.14 (a leitura absorve ganhos do treino).
- Larvae e protozoan: **não dá para dizer**.
- **Com a leitura final (gem_tta_pw) a curva de largura nunca foi medida.** Só existe na largura base. É o buraco mais importante deste eixo.

### 2.2 O ajuste e(N) = E + A·N^(−α) não é identificável

Perfil em α, com efeito de split:

| eixo | faixa de α (95%) | teto de κ compatível |
|---|---|---|
| eggs avg L4 | 0,02 a 2,21 (bate na borda) | de 0,89 até 1,0 |
| eggs avg L3 | 0,02 a 1,34 (bate na borda) | de 0,88 até 1,0 |
| eggs gem L4 | 0,02 a 6,00 (bate na borda) | 0,89 a 0,90 |

Com 3 ou 4 valores de N, qualquer expoente serve. **Não publique α nem N\* a partir disso.** Sem o piso E, a potência pura dá α = 0,15 [0,07; 0,23] em eggs avg, 0,10 [0,07; 0,14] em larvae (60 ép.) e ~0 em eggs gem.

A reta em log₂N com os pontos menores prevê bem o maior ponto (erro −0,003 em avg L4, +0,003 em gem L4, +0,015 em avg L3). Serve para interpolar, mas o maior ponto está só uma dobra à frente. Nada foi medido entre 7,4×10⁵ e o teto de 5×10⁶, então qualquer afirmação perto de 5M é extrapolação de 3 dobras.

### 2.3 Profundidade [A]

| eixo | pontos | Δ pareado, IC95 |
|---|---|---|
| eggs, gem, 60 ép., última camada | 4 → 0,8749 (N 200.944); 5 → 0,8567 (N 395.600); 6 → 0,8512 (N 743.888) | 5−4: **−0,018** [−0,036; −0,000]; 6−4: −0,024 [−0,095; +0,047] |
| eggs, avg, 60 ép. | 4 → 0,8305; 5 → 0,8248; 6 → 0,8199 | 5−4: −0,006 [−0,054; +0,043]; 6−4: −0,011 [−0,033; +0,012] |
| estágios da mesma rede, eggs | L1 0,161; L2 0,730; L3 0,842; L4 0,880 | sobe até L4 |
| estágios, larvae | L1 0,842; L2 0,902; L3 0,900; L4 0,917 | quase plano desde L2 |
| estágios, protozoan | L1 0,233; L2 0,569; L3 0,786; L4 0,837 | sobe até L4 |
| R2 flatten, eggs **[F]** | L3 0,894; L4 0,873; L5 0,857; L6 0,854; L7 0,838; L8 0,817 | cai; mas k vira 3 a partir de L5 e a leitura muda de dimensão: não é eixo limpo |

Ressalvas:
- A queda com profundidade só foi medida em **eggs, a 60 épocas**, e só 5−4 com gem exclui zero, e por pouco. A 60 épocas a camada nova treina menos.
- Em d5/d6 os pools ficam só nas 3 primeiras camadas, então a queda **não** vem de encolher o mapa.
- Nos estágios, profundidade, N e dimensão da leitura mudam juntos (32 → 48 → 64 → 1280).
- Larvae satura em profundidade bem antes (L2) de eggs e protozoan. É o único indício de que a profundidade certa depende do dataset.

**N sozinho não é a variável certa.** Em eggs a 60 épocas, dobrar N por profundidade tira κ; a 200 épocas, dobrar N por largura não muda nada com gem.

### 2.4 O que rende mais que largura [A]

| mudança | Δκ pareado, IC95 | custo |
|---|---|---|
| leitura gem → gem_tta_pw | eggs **+0,044** [+0,028; +0,059]; larvae +0,051 [−0,001; +0,104]; protozoan **+0,052** [+0,033; +0,070] | zero parâmetros |
| 1x1 de 1280 sobre a L3 (avg) | eggs **+0,038** [+0,020; +0,056]; larvae +0,018 [−0,008; +0,043]; protozoan **+0,051** [+0,028; +0,075] | +83 mil parâmetros |
| 200 contra 60 épocas (gem) | eggs +0,024 [−0,003; +0,050]; larvae +0,006 [−0,024; +0,037]; protozoan +0,013 [−0,012; +0,038] | 3,3x o tempo |
| manter o mapa 49x49 (`res49`, sem o 3º pool), eggs | −0,028 (HAL §4.5; pool [2,2,1,1] conferido) | — |

Curiosidade não explicada: em eggs, 100 épocas não ganham de 60 (−0,002 [−0,006; +0,002]), mas 200 ganham. Só uma família, então não tiro lei daí.

### 2.5 Número de imagens [A], validação

n de treino: 1% = 24/17/46 (eggs/larvae/protozoan), 5% = 124/87/235, 50% = 1276/878/~2392.

| dataset | κ a 1% | κ a 5% | κ a 50% | β global em e ~ n^(−β), IC95 | corte de erro por dobra de dados |
|---|---|---|---|---|---|
| eggs | 0,278 | 0,659 | 0,942 | **0,64** [0,61; 0,68] | **36%** |
| larvae | 0,600 | 0,797 | 0,961 | **0,62** [0,37; 0,86] | **35%** |
| protozoan | 0,552 (1 split) | sem ponto | 0,926 | sem IC | — |

- **Não satura até 50%.** A inclinação até aumenta: eggs β = 0,46 de 1 para 5%, 0,76 de 5 para 50%.
- Comparando eixos em eggs: dobrar dados tira 36% do erro; dobrar parâmetros tira ~10% com avg e ~0% com gem.
- Ressalva: 1% e 5% usam `gem_tta_pw_aug8`, e 50% usa `gem_tta_pw`. A leitura não é idêntica. Com o encoder fixo em 100% (`ssl100`, rótulos de 1 a 5%), β = 0,75 [0,60; 0,91] em eggs e 0,48 [0,15; 0,81] em larvae, mesma ordem de grandeza.
- Validação a 25, 75 e 100% não existe no pipeline [A]; só teste. Não ajustei com teste.
- **[S] Larvae_subgroup**, validação, gem_tta_pw: 25% 0,545; 50% 0,651; 75% 0,577; 100% 0,685. Não é monótono, com n_val ≈ 107 e erro-padrão ~0,07. Isso é ruído; não dá inclinação.

**A lei conjunta e(N, D) é impossível com o que existe.** Nenhuma célula varia N e D juntos: todos os percentuais têm N = 200.944.

### 2.6 O que o `disc_md_x1` diz sobre escala: quase nada, ainda

Só há a L1. `p0050_w032` contra `p0800_w064` muda duas coisas ao mesmo tempo, e cada dataset tem um tamanho só (s200, s320 ou s640), então resolução fica confundida com dataset. **O eixo SPiFiL × resolução não tem nenhum dado utilizável até agora.**

---

## 3. Pergunta 2: benchmark

Só colunas `val_*` e `train_*`; as `test_*` foram descartadas na leitura do CSV. Completo: ConvNeXt V2 pico (8,55M) e I-JEPA ViT-G/16 22k (1,01B), 12 datasets × 3 splits, 50%. DINOv2-giant (1,14B): **só Eggs**. DINOv3 e AIMv2: nada.

**Resposta curta:** nenhuma das propriedades pedidas explica o κ. A distância até o nosso antigo só é medível direto em Larvae_subgroup, e lá é grande: −0,17.

| dataset | n treino | classes | maior/menor | lado mediano | fundo preto | I-JEPA | ConvNeXt | I-JEPA − ConvNeXt |
|---|---|---|---|---|---|---|---|---|
| Eggs [R] | 4448 | 9 | 118 | 372 | 0% | 0,970 | 0,932 | +0,039 |
| helminth_eggs | 1791 | 9 | 42 | 200 | 97% | 0,972 | 0,952 | +0,020 |
| Eggs_subgroup | 1388 | 9 | 54 | 200 | 22% | 0,963 | 0,924 | +0,039 |
| Protozoan [R] | 13086 | 7 | 92 | 96 | 0% | 0,921 | 0,823 | **+0,098** |
| protozoan_cysts | 3351 | 7 | 73 | 200 | 96% | 0,925 | 0,898 | +0,026 |
| Protozoan_subgroup | 2774 | 7 | 248 | 200 | 11% | 0,926 | 0,859 | +0,067 |
| Larvae [R] | 560 | 2 | 5,5 | 890 | 0% | 0,850 | 0,876 | −0,026 |
| helminth_larvae | 1230 | 2 | 6,9 | 200 | 98% | 0,965 | 0,966 | −0,001 |
| Larvae_subgroup | 256 | 2 | 4,4 | 200 | 43% | 0,818 | 0,834 | −0,016 |
| aid | 3498 | 30 | 1,9 | 600 | 0% | 0,903 | 0,864 | +0,038 |
| flowers | 963 | 5 | 1,3 | 277 | 2% | 0,937 | 0,888 | +0,049 |
| remote_sensing | 351 | 14 | 8,7 | 640 | 0% | 0,603 | 0,601 | +0,003 |

Contagens de classe: R2 §2. Fundo preto: medido em 120 imagens de treino por dataset (`law/bgfrac.py`).

**Sustentado por dados:**
- Sozinha, nenhuma propriedade prevê κ. Spearman contra I-JEPA: n de treino +0,40 (p=0,20), imagens por classe +0,47 (p=0,13), classes +0,07, desbalanceamento +0,34, lado −0,34. Permutação, 4000 sorteios.
- Fora da amostra (leave-one-out): melhor R² = 0,13 para I-JEPA e 0,22 para ConvNeXt, com erro médio de ~0,06 em κ. Não serve para prever nada.
- São 12 pontos, e 9 deles são 3 famílias de parasitas que dividem imagens (R3: subgrupos herdam o split do pai). Na prática há bem menos que 12 observações independentes.
- **A família do dataset importa mais que o tamanho.** Protozoan: 13.086 contra 2.774 imagens dá o mesmo κ (0,921 e 0,926). Eggs: idem (0,970 e 0,963).
- **Larvae é a exceção:** helminth_larvae (fundo preto) 0,965 contra Larvae_subgroup (fundo real) 0,818. Os objetos do subgrupo são um subconjunto do pai (R3), mas o n de treino é 5x menor, então fundo e n estão misturados.
- O modelo 120x maior **não** ajuda nas larvas: nas três, I-JEPA fica empatado ou abaixo do ConvNeXt de 8,5M.
- DINOv2-G em Eggs: 0,9804. Contra I-JEPA, +0,010 [−0,015; +0,035]: não separável. Mas é a régua mais alta de Eggs até agora, com 518 px.

**Hipótese (não sustentada ainda):**
- A vantagem do modelo grande cresce com o n de treino: ρ = +0,66, p = 0,022. Mas são 27 testes, e não sobrevive a correção. Vale testar, não vale citar.
- `remote_sensing` em 0,60 nos dois modelos pode vir de 25 imagens por classe, de grupos no split (568 grupos em 1.000 imagens, R2) ou do encolhimento de 640 para 224. Nada aqui separa as três causas.
- Larvae raw (lado ~890) pode sofrer ao encolher para 224, com a larva fina sumindo. Não medido: DINOv2 a 518 px ainda não rodou em Larvae.

### 3.1 Distância até o nosso antigo

| comparação | nosso | benchmark | Δ | confiança |
|---|---|---|---|---|
| **Larvae_subgroup [S]**, val, gem_tta_pw, mesmo split | 0,651 | I-JEPA 0,818; ConvNeXt 0,834 | **−0,167** [−0,330; −0,004]; **−0,183** [−0,365; −0,001] | direta, pareada por split |
| Eggs_subgroup [S], 50% | 0,833 (teste) | I-JEPA antigo 0,951 (teste) | −0,118 | reportado em R1 §1.4; teste; I-JEPA antigo |
| Larvae_subgroup [S], 50% | 0,588 (teste) | I-JEPA antigo 0,750 (teste) | −0,162 | idem |
| fundo preto [A] contra benchmark, val | 0,942 / 0,961 / 0,926 | 0,972 / 0,965 / 0,925 | −0,030 / −0,004 / +0,001 | **indicativo**: mesmas imagens, splits diferentes |

**A maior distância conhecida é nos `_subgroup`, com fundo real, e sobretudo em larvae.** O que se chamava de empate com o I-JEPA valia só para fundo preto (R1 §1.4). Em raw/ (Eggs, Protozoan, Larvae) e nos naturais (aid, flowers, remote_sensing) **não existe κ final nosso**, então a distância é desconhecida. Eggs_subgroup e Protozoan_subgroup não têm CSV de validação nosso.

A régua também subiu: o I-JEPA antigo (ViT-H, teste) marcava 0,940 / 0,943 / 0,864; o novo (ViT-G 22k, validação, outro split) marca 0,972 / 0,965 / 0,925. Validação e teste, com splits distintos, então é só um sinal, mas o alvo novo é mais alto que o do "empate" de outubro.

### 3.2 O aceite de 0,01 não é mensurável em todo dataset

Erro-padrão da média de 3 splits no κ-alvo, só por amostragem da validação:

| dataset | n validação | erro-padrão da média | dá para ver 0,01? |
|---|---|---|---|
| Larvae_subgroup | 111 | 0,040 | não |
| Larvae | 240 | 0,025 | não |
| remote_sensing | 149 | 0,025 | não |
| helminth_larvae | 527 | 0,010 | não |
| flowers | 412 | 0,008 | mal |
| Eggs_subgroup, Protozoan_subgroup, protozoan_cysts | 599 a 1435 | 0,005 a 0,006 | mal |
| helminth_eggs, aid, Eggs, Protozoan | 766 a 5607 | 0,003 a 0,005 | sim |

O desvio observado entre splits bate com esse erro teórico (Larvae_subgroup 0,054 a 0,070 contra 0,069–0,072; helminth_larvae 0,015 a 0,018 contra 0,017). Quase toda a variação entre splits é sorte de quais imagens caíram na validação. Comparar por split, pareado, reduz isso; não elimina.

---

## 4. Pergunta 3: épocas, ruído e o sobe e desce

**Resposta curta:** o sobe e desce de `helminth_larvae` e a diferença treino/validação em `helminth_eggs` e `protozoan_cysts` são efeito da sonda, não do modelo.

### 4.1 Como as 2048 imagens são escolhidas

- `_probe_images` (`methods/autoencoder/autoencoder_flim_module.py:1191-1207`) pega as primeiras 2048 de `train_dataloader()`, que tem `shuffle=True` (`core/data/parasite_data_module.py:265-266`). É um sorteio simples, sem estratificar e sem reposição.
- O sorteio acontece **uma vez por estágio**, e as mesmas imagens valem para todas as épocas. Sorteio, portanto, não causa pulo entre épocas.
- Treino e validação usam o mesmo transform, porque V=1 (`core/data/multi_view_dataset.py:115`).
- **Abaixo de 2048 não há sorteio nenhum:** `helminth_larvae` (1230), `helminth_eggs` (1791), `Eggs_subgroup`, `Larvae_subgroup`, `flowers` e `remote_sensing` usam o treino inteiro. Sorteiam: `Protozoan_subgroup` (74%), `protozoan_cysts` (61%), `aid` (59%), `Eggs` (46%) e `Protozoan` (16%).

### 4.2 O bug

- Treino da sonda: imagens guardadas em `float16` (`autoencoder_flim_module.py:1201`).
- Validação da sonda: imagens em `float32` (`:919`).

Pense em pesar a turma de treino numa balança que marca 100 g a menos, e a de validação numa balança certa. Em foto comum os erros de arredondamento são variados e se cancelam. No fundo preto, todo píxel tem o mesmo valor, `LAB_BLACK` = (0, 119, 136)/255 (`core/data/transforms.py:30`), então todos erram igual: −1,14e-4 no canal a, −1,30e-4 no canal b.

O erro é minúsculo, mas as features dessas imagens quase não variam (desvio relativo 0,3–0,9%, contra 14–36% nos outros). A sonda divide cada coluna pelo seu desvio, e o erro vira um salto enorme: até **59 desvios** numa coluna. A validação inteira cai fora da região que a sonda aprendeu.

### 4.3 As provas

**(a) Mesmo checkpoint, sonda em CPU.** Treino bate nos 15 runs (diferença mediana 0,004, máxima 0,021).

| run (s200_p0800_w064) | época | κ val registrado | reproduzido com o bug | **κ val limpo** | κ treino limpo |
|---|---|---|---|---|---|
| helminth_larvae s1 | 12 | 0,814 | 0,753 | **0,836** | 0,818 |
| helminth_larvae s2 | 3 | 0,822 | 0,818 | **0,852** | 0,808 |
| helminth_larvae s3 | 2 | 0,000 | 0,000 | **0,868** | 0,882 |
| helminth_eggs s1 | 16 | 0,002 | 0,001 | **0,541** | 0,489 |
| helminth_eggs s2 | 15 | 0,663 | 0,640 | **0,729** | 0,744 |
| helminth_eggs s3 | 16 | 0,191 | 0,057 | **0,718** | 0,754 |
| protozoan_cysts s1 | 19 | 0,133 | 0,120 | **0,626** | 0,628 |
| protozoan_cysts s2 | 14 | 0,545 | 0,531 | **0,523** | 0,513 |
| protozoan_cysts s3 | 2 | 0,114 | 0,012 | **0,565** | 0,557 |
| *controle* Eggs_subgroup s1 | 19 | 0,535 | 0,528 | 0,524 | 0,547 |
| *controle* Protozoan_subgroup s1 | 19 | 0,488 | 0,481 | 0,483 | 0,498 |
| *controle* flowers s1 | 4 | 0,548 | 0,542 | 0,536 | 0,561 |
| *controle* Larvae_subgroup s1 | 3 | 0,531 | 0,471 | 0,469 | 0,422 |

Média limpa: `helminth_larvae` 0,852, `helminth_eggs` 0,662, `protozoan_cysts` 0,571. Diferença treino − validação limpa: média −0,007 (de −0,051 a +0,036), contra 0,33 registrada. Nos controles, bug e limpo diferem no máximo 0,006.

**Onde a reprodução não fecha, dito às claras:** a validação registrada bate em até 0,06 nos 7 runs estáveis, mas diverge nos instáveis: 0,134 em eggs s3, 0,102 em cysts s3, 0,061 em larvae s1 e **0,713** em `helminth_larvae` w032 s2 (registrado 0,713, reproduzido 0; limpo 0,850). É o esperado pelo item (b): nesse regime, diferença de aritmética entre GPU e CPU já muda o resultado, e o checkpoint é justamente a época de maior κ da série. Não testei em GPU.

**(b) O sobe e desce reproduzido.** Só existe o peso da melhor época, então andei em linha reta do peso inicial até ele (`helminth_larvae` s1). É um caminho aproximado, não o caminho real do treino.

| fração do caminho | 0 | 0,1 | 0,2 | 0,3 | 0,4 | 0,5 | 0,6 | 0,7 | 0,8 | 0,9 | 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **com o bug** | 0,607 | 0,770 | 0,000 | 0,000 | 0,000 | 0,100 | 0,000 | 0,000 | 0,051 | 0,409 | 0,753 |
| **limpo** | 0,828 | 0,817 | 0,836 | 0,846 | 0,836 | 0,836 | 0,836 | 0,836 | 0,836 | 0,836 | 0,836 |

Desvio: 0,309 com o bug, 0,007 limpo. Ruído de 0,1% de um desvio-padrão nos pesos, 8 sorteios: com o bug κ vai de 0,666 a 0,861 (desvio 0,077); limpo, de 0,836 a 0,846 (0,003). Pesos praticamente iguais dão κ muito diferente, então o pulo não vem de aprendizado.

**(c) Outro otimizador.** Com `LogisticRegression` do sklearn, a queda se repete: `helminth_larvae` s3 0,000 contra 0,817; `helminth_eggs` s3 0,027 contra 0,694; `protozoan_cysts` s1 0,141 contra 0,576. Controles: 0,450/0,458, 0,462/0,460, 0,530/0,533.

**(d) Só onde há fundo preto.** κ exatamente 0 só aparece em massa nos 3 datasets de 96–98% de preto, mais o caso à parte de 4.6. Runs com salto pequeno escapam, como `protozoan_cysts` s2.

Nota: simulando TF32 na conv, a assimetria some (κ 0,876 em larvae s3), mas o registro mostra o colapso. Então a conv na GPU aparentemente não arredonda a entrada assim. É inferência; não medi na GPU.

### 4.4 Classe rara e ordem de arquivo: não explicam

**Classe rara.**
- Em `helminth_larvae` e `helminth_eggs` não há amostragem: entra o treino inteiro.
- O colapso cai numa classe **qualquer**, não na rara nem na maior. `helminth_eggs` s1 põe 765 de 766 imagens na classe 4 (52 imagens reais), e o recall da classe maior vai de 0,93 para 0. `helminth_eggs` s3 põe 593 na classe 1 (51 imagens reais). Só em `helminth_larvae`, binário, a classe escolhida é a maior.
- Nas bases sorteadas, 12 sorteios de 2048 dão desvio de 0,006 a 0,017 no κ limpo, e nenhuma classe faltou em 48 sorteios (4 runs × 12). A menor ficou com 3 imagens em `Protozoan_subgroup` e 9 em `protozoan_cysts`.

**Ordem de arquivo.** O `shuffle=True` a descarta. Mas é uma mina: o JSON está em ordem de classe (6 trocas em 3351 linhas). Sem o shuffle, as 2048 primeiras de `protozoan_cysts` s1 perdem a classe 6 inteira, e o κ cai de 0,625 para 0,447.

### 4.5 Melhor época

Datasets sãos, família `p0800_w064`, média entre splits, 20 épocas:

| dataset | melhor época | melhor por split | κ época 0 | κ melhor | κ fim | melhor − fim | treino fim |
|---|---|---|---|---|---|---|---|
| Eggs (2 splits) | 11 | 11, 9 | 0,285 | 0,298 | 0,289 | 0,008 | 0,352 |
| Eggs_subgroup | 19 | 1, 15, 4 | 0,495 | 0,501 | 0,501 | 0,000 | 0,572 |
| Protozoan | 15 | 15, 17, 18 | 0,173 | 0,191 | 0,190 | 0,002 | 0,208 |
| Protozoan_subgroup | 19 | 19, 17, 5 | 0,525 | 0,539 | 0,539 | 0,000 | 0,556 |
| flowers | 12 | 4, 8, 3 | 0,503 | 0,520 | 0,515 | 0,005 | 0,568 |
| Larvae_subgroup | 14 | 3, 0, 18 | 0,429 | 0,440 | 0,392 | 0,048 | 0,469 |

- **Não há melhor época real.** A melhor época muda de split para split, e melhor − fim fica em até 0,008 (Larvae_subgroup é ruído: erro-padrão ~0,12).
- A época 0 já tem quase todo o κ: o ganho até o fim fica entre −0,04 e +0,017.
- A razão é simples: os pesos da L1 mudam de 0,3% a 10% (mediana 2,3%, 16 runs de split 1), com cosseno ≥ 0,9994 contra o início (`law/drift.py`). A L1 é o próprio SPiFiL. A `val_loss` também para: cai no máximo 0,005 depois da época 5.
- Atenção a pegar o máximo de 20 épocas: com ruído de 0,005 a 0,008, o máximo fica ~0,01 acima da média só por sorte.
- Isso vale **só para a L1 com 20 épocas**. Nas redes de 4 camadas, 200 épocas ainda ajudam eggs (seção 2.4).
- Nos 3 datasets com bug, os registros não dizem a melhor época. O caminho limpo acima sugere que também é plano.

**Os números citados no pedido:** 0,545 é a média dos máximos de cada split (0,814, 0,822, 0), e "época ~5" é a média das épocas (12, 3, 0). Nenhuma época real tem isso. A curva média tem pico 0,356 na época 12 e termina em 0,180.

### 4.6 Quão ruidosa é a sonda

| fonte | tamanho no κ |
|---|---|
| arredondamento nos pesos (sonda limpa) | 0,003 |
| sorteio das 2048 | 0,006 a 0,017 |
| entre épocas (datasets sãos; Larvae_subgroup 0,039) | 0,005 a 0,008 |
| amostra da validação, teórico | 0,016 (Protozoan) a 0,12 (Larvae_subgroup) |
| entre splits, observado | 0,022 a 0,054 |

- Eggs_subgroup, flowers e Larvae_subgroup: a variação entre splits cabe no ruído de amostra.
- Protozoan (0,034 contra 0,016) e Protozoan_subgroup (0,054 contra 0,021): ~2x acima.
- Nos 3 com bug, usando κ limpo: `helminth_larvae` 0,016 (cabe), `protozoan_cysts` 0,052 (2,7x), `helminth_eggs` 0,105 (4x). Em eggs s1 o treino também é baixo (0,489), então não é ruído da sonda. **Hipótese:** o sorteio do SPiFiL muda os filtros de cada split.

**A sonda limpa também tem limites.** Use como sinal relativo, não como qualidade do modelo:
- **Não converge em 100 passos.** Com 1000 passos, o κ de validação sobe de 0,002 a 0,035. Em Larvae_subgroup o treino vai de 0,42 para 0,54.
- **Depende muito da L2.** Com L2 10x maior (`PROBE_L2`, linha 171): `helminth_larvae` s2 0,852 → 0,694; Larvae_subgroup 0,469 → 0,051; flowers só 0,536 → 0,511.
- **κ=0 não prova feature ruim.** `Larvae_subgroup` w032 s1 dá exatamente 0 mesmo limpo e convergido (treino 46 contra 200), mas sobe para 0,295 com L2 de 1e-6. Esse caso não é o bug do fp16.
- O L1 do `x1` não compara com κ final: no pipeline antigo, a L1 sozinha já dava 0,16 / 0,84 / 0,23.

### 4.7 O que isso afeta

- **Afetado:** todo `val_kappa` por época e `best_kappa.ckpt` nos 3 datasets, porque `best_by_probe` escolhe por esse κ (`autoencoder_flim_module.py:742`), e esse peso alimenta o `grow` seguinte (`experiments/ray/runners/growth.py:419`). A frase da J1 de que a sonda "segue o SVM oficial de perto" não vale para eles.
- **Não afetado:** o SVM oficial. Não há `float16` em `eval/svm.py` nem em `eval/growth_stages.py`.
- **Dano na L1 deve ser pequeno:** os pesos mal se movem, e o κ limpo é plano ao longo do caminho. Nos estágios seguintes pode ser maior. Isso é inferência; não medi.

Conserto possível (não aplicado): arredondar também o lado da validação em `:919`, com `x.half().float()`. Medido em `helminth_larvae` s3: κ 0,876, sem custo de memória. Guardar em `uint8` é a outra saída, sem perda enquanto a imagem sair de uint8 (não vale com `strong_aug`), e custa metade do fp16: 2,5 GB contra 5 GB a 640 px. Trocar por `float32` puro custaria 10 GB a 640 px.

---

## 5. Lei de montagem: o que é fato e o que é aposta

**Sustentado por dados** (todos no split antigo [A], salvo indicação):

| # | fato | força |
|---|---|---|
| 1 | N sozinho não é a variável: largura e profundidade dão efeitos diferentes com o mesmo N | boa, só eggs |
| 2 | Largura plana com gem em eggs, de 2×10⁵ a 6,3×10⁵ | boa, só eggs |
| 3 | A mesma largura ainda rende com avg: +0,012 por dobra | boa, só eggs |
| 4 | 1x1 larga no fim: +0,04 a +0,05 em eggs e protozoan, por +83 mil parâmetros | boa |
| 5 | Leitura final: +0,044 a +0,052, sem parâmetro nenhum | boa |
| 6 | Dados: e ~ n^(−0,6), sem saturar até 50% | boa em eggs; leitura mista |
| 7 | Encolher 56x custa 0,05 em larvae | boa, um par de pontos, 60 épocas |
| 8 | 5 camadas pior que 4 em eggs | **fraca**: 60 épocas, IC encosta em zero |
| 9 | Nenhuma propriedade do dataset prevê o κ do benchmark | boa, novo protocolo, 12 pontos dependentes |
| 10 | Na L1, 20 épocas quase não movem o SPiFiL | boa, novo protocolo |
| 11 | Aceite de 0,01 não é mensurável em pelo menos 4 datasets com 3 splits | boa, novo protocolo |

**Hipóteses (precisam de medida):**

| # | hipótese | por que está aqui |
|---|---|---|
| a | Com a leitura final, largura também é plana | gem sugere; nunca medido |
| b | A profundidade certa depende do dataset | larvae satura na L2, eggs e protozoan na L4; profundidade misturada com N e leitura |
| c | Escala composta ao estilo EfficientNet | **nenhum par de eixos foi cruzado**, e resolução não tem dado |
| d | Cabe alcançar o benchmark com ≤5M | nada medido acima de 7,4×10⁵; a largura sozinha não indica chegar lá |
| e | O problema central é o fundo real, não o tamanho | a distância se concentra nos `_subgroup`; n e fundo misturados |
| f | A vantagem do modelo grande cresce com n | p=0,022 sem correção, entre 27 testes |
| g | Variação entre splits em eggs vem do sorteio do SPiFiL | não testado |

**O que os dados sugerem para a lei:** mexer primeiro no que rende (dados, leitura, 1x1 larga) e tratar largura e profundidade como ajuste fino. É orientação, não lei: quase tudo veio de fundo preto e split antigo, e a régua nova é outra.

---

## 6. O que medir, por custo

1. **Corrigir a sonda** antes de ler `epoch_metrics` dos 3 datasets.
2. **12 evals de SVM em pesos prontos:** `disc_sl_w025_e60` e `disc_sl_w050_e60` (larvae e protozoan, 3 splits) têm L4 treinada e nunca foram avaliados. Junto com o que existe, dão a primeira curva real de largura: 4 pontos em larvae, 3 em protozoan.
3. **9 evals com a leitura final** em `disc_g20_e_w96`, `disc_g20_e_w128` e `disc_g20_l_w96`: respondem à hipótese (a).
4. **Mesmo dataset em 2 ou mais resoluções:** hoje cada dataset tem um tamanho só, então o eixo SPiFiL × resolução está vazio.
5. **Um cruzamento largura × dados** (por exemplo w96 a 5% e 25%): sem ele, a lei conjunta é impossível.
6. **Mais splits ou validação maior** em Larvae, Larvae_subgroup e remote_sensing, ou um aceite pareado por split: com 3 splits, 0,01 não é visível ali.

---

## 7. Limites e arquivos

**Limites deste trabalho**
- Sem GPU: toda reprodução foi em CPU. Só existe o peso da melhor época de cada run.
- Reproduzi 15 runs, não todos. Protozoan, Eggs, aid e remote_sensing não entraram na reprodução.
- Teste: nenhuma coluna `test_*` de `disc_md` ou do benchmark entrou em conta (descartadas na leitura). Um `head` inicial do CSV, só para ver o formato, mostrou algumas dessas colunas na tela; nada dali foi usado. Os únicos números de teste aqui são históricos, marcados, de R1 e HAL.
- `A_reports/november/scaling_law.md` e `A_reports/october/report_halloween_october.md` só existem no worktree, não na main.
- `remote_sensing` s640 split 1: o CSV parou em 3 épocas às 01:17, sem `run_metadata.json`. Há um processo `remote_sensing_split1` recente, mas não verifiquei se é a mesma run.

**Scripts, todos nesta pasta:** `load_em.py`, `em_table.py`, `q3_epochs.py`, `q3_final.py` (épocas); `extract2.py`, `probe_test.py`, `validate.py`, `path.py`, `sens.py`, `w032.py`, `drift.py` (reprodução da sonda); `arch.py`, `q1_points.py`, `q1_fit.py`, `q1_data.py` (lei); `q2.py`, `q2b.py`, `q2c.py`, `bgfrac.py` (benchmark). Features em `feats/`.
