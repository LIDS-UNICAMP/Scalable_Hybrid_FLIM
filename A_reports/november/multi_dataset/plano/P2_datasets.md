# P2: datasets, papéis, frações, ruído e vazamento

07/11/2026. Só plano: sem GPU, sem treino, nada lançado. Teste não entrou em nenhuma conta (limites na seção 9).

**Resposta curta:** são **5 famílias, não 6**, porque `remote_sensing` é em boa parte feito de imagens do `aid`. A divisão que recomendo é 7 datasets em descoberta e 5 em confirmação, mais 2 inéditos na validação externa. Se o usuário não aprovar os inéditos, vale 7 / 3 / 2 só com os 12. A descoberta é igual nos dois casos, então ela pode começar sem esperar essa resposta.

## 1. Decisão

| Pergunta | Decisão | Por quê |
|---|---|---|
| Quantas famílias? | **5**: Eggs (3), Protozoan (3), Larvae (3), Cena aérea (`aid` + `remote_sensing`), `flowers` | 453 das 1.000 imagens de `remote_sensing` são imagens do `aid` (seção 2) |
| Descoberta | Eggs, Larvae, `flowers`: **7 datasets, 3 famílias** | Cobre 2 a 9 classes, 200 a 1.328 px, razão 1,3 a 118. Com 3 famílias dá para deixar uma de fora e ver se a lei se mantém |
| Confirmação | Protozoan + Cena aérea: **5 datasets, 2 famílias** (opção B) ou só Protozoan: **3 datasets** (opção A) | Testa o que a descoberta não viu: 112 px, 37 mil imagens, razão 248, 30 classes |
| Validação externa | `coconut_trees` + `medmnist_bloodmnist` (opção B) ou `aid` + `remote_sensing` (opção A) | Só imagens novas têm teste limpo. Os 12 já têm números de teste calculados (seção 8) |
| Opção recomendada | **B, com aprovação do usuário. Sem resposta, vale A** | B é desvio do pedido: entram 2 datasets fora dos 12. O prazo é o início da confirmação |
| Irmãs em papéis diferentes? | **Nunca.** Família inteira no mesmo papel, contada como 1 unidade na estatística | Teste do subgrupo cai ~70% no treino da base (seção 8) |
| Frações | As 6 são medidas onde for possível, mas **só células L entram no ajuste da lei** (seção 6) | Abaixo de 5 imagens por classe o regime é outro, e o ruído a 1% passa de 0,10 em quase tudo |
| `remote_sensing` a 1% e 5% | **Fora** | 16 imagens para 14 classes; faltam classes em todos os splits a 1% e em 2 de 3 a 5% |
| Splits novos ou com piso por classe? | **Não.** Ficam os splits registrados | Mudar invalida 288 linhas de benchmark (4 datasets afetados, 72 linhas cada), e o ganho é só em células que já estão fora |
| Quantos splits? | **3**. `Larvae` e `Larvae_subgroup` não decidem nada sozinhos | Com 3 splits eles só enxergam 0,14 e 0,17 (seção 7) |
| Aceite de 0,01 | **Reportar a diferença, não afirmar que passou** | Com 3 splits nenhum dataset enxerga 0,01; os melhores enxergam 0,02 |
| Grade `sc` que está rodando | Só as linhas da descoberta entram na lei | 3 dos 6 datasets dela (`Protozoan_subgroup`, `aid`, `remote_sensing`) não são da descoberta |

## 2. Correção: `aid` e `remote_sensing` são irmãos

Pense em dois álbuns de fotos que parecem de pessoas diferentes, mas metade das fotos de um é cópia do outro. Foi isso que achei.

- Comparei miniaturas 16x16 de todas as imagens: 453 de `remote_sensing` batem com NCC ≥ 0,98 em uma imagem do `aid`. NCC mede quão iguais duas imagens são; 1 é igual.
- Conferi 40 desses pares a 128 px: NCC mediano **0,9989**, mínimo 0,9896. É a mesma foto, ampliada de 600 para 640.
- São 453 fontes diferentes, sem repetição. As outras 537 têm NCC mediano 0,605: não vêm do `aid`.
- Vazamento real: no split 1, 54 de 66 imagens do teste de `remote_sensing` que vêm do `aid` estão no treino do `aid`.

**Dois enganos dos relatórios anteriores caem junto:**

1. **Os 568 grupos não são grupos.** Imagens do "mesmo grupo" têm NCC mediano 0,007, ou seja, nada em comum. O grupo é só o número do arquivo (`tools/dataset_std/remote_sensing.py:34,64`), e números repetem entre fontes. Isso não vaza nada, mas por isso as classes somem a 1%: a estratificação usa a classe do grupo, não a da imagem (`tools/make_splits.py:80`).
2. **O κ preso em ~0,7 deve vir do rótulo.** O rótulo é a classe mais comum das caixas de um detector (`remote_sensing.py:19`), não a cena. Nas 453 imagens com fonte no `aid`, 364 (80%) têm rótulo compatível com a cena, pelo meu mapeamento de nomes (como `Pond` para `Lake`); 89 discordam, como rio rotulado `Grassland`. Inferência minha: nenhum modelo passa de 0,73, e parte do erro é rótulo, não modelo. Isso derruba a hipótese "grupos no split" de `L_data`.

## 3. Critérios

| Critério | Tipo | Regra |
|---|---|---|
| Família inteira | obrigatório | Irmãs no mesmo papel |
| Descoberta com dois domínios | obrigatório | Microscopia e imagem natural, senão não há como achar lei compartilhada |
| Externa sem o laboratório da descoberta | obrigatório | Fora as 3 famílias de parasitas, que vêm da mesma fonte |
| Força de medida | obrigatório | Cada papel tem ao menos 1 dataset que enxerga 0,03 com 3 splits |
| Variedade na descoberta | forte | Pelo menos 3 famílias, para deixar uma de fora |
| Faixa na descoberta | forte | Pegar classes, resolução e desbalanceamento amplos; a confirmação deve testar além da borda |
| Custo | fraco | A família cara não vai na descoberta, que roda a maior parte dos experimentos. Protozoan tem 38 mil imagens de treino a 100%, Eggs 15 mil, Larvae 4 mil |

Com os 12, isso quase decide sozinho. A externa precisa ser natural e a descoberta também precisa de uma natural, então a externa é Cena aérea e `flowers` fica na descoberta. `flowers` sozinho na externa seria 1 dataset, com 0,042 de resolução. Sobram 6 jeitos de dividir Eggs, Protozoan e Larvae. Tiro 4: os 3 com Protozoan na descoberta (custo; um deles ainda deixa a confirmação só com Larvae, que quase não enxerga diferença) e o de Larvae + `flowers` na descoberta, que não enxerga 0,03 e não tem desbalanceamento alto. Ficam 2, que são as opções A e C.

**Sem máscara:** nenhum critério usa máscara, e Eggs, Protozoan e Larvae entram com `raw/`, como o usuário decidiu.

### Os 12, medidos

| dataset | família | domínio | imagens | classes | maior/menor | lado (mediana) | val | papel (A e B) |
|---|---|---|---|---|---|---|---|---|
| Eggs | Eggs | microscopia, fundo real | 12.691 | 9 | 118 | 432 | 1.903 | descoberta |
| helminth_eggs | Eggs | microscopia, fundo preto | 5.112 | 9 | 42 | 200 | 766 | descoberta |
| Eggs_subgroup | Eggs | microscopia, fundo real | 3.974 | 9 | 54 | 200 | ~600 | descoberta |
| Larvae | Larvae | microscopia, fundo real | 1.598 | 2 | 5,5 | 1.328 | 240 | descoberta |
| helminth_larvae | Larvae | microscopia, fundo preto | 3.514 | 2 | 6,9 | 200 | 527 | descoberta |
| Larvae_subgroup | Larvae | microscopia, fundo real | 741 | 2 | 4,4 | 200 | ~111 | descoberta |
| flowers | flowers | natural | 2.746 | 5 | 1,3 | 320 | 412 | descoberta |
| Protozoan | Protozoan | microscopia, fundo real | 37.376 | 7 | 92 | 112 | 5.607 | confirmação |
| protozoan_cysts | Protozoan | microscopia, fundo preto | 9.568 | 7 | 73 | 200 | 1.435 | confirmação |
| Protozoan_subgroup | Protozoan | microscopia, fundo real | 7.919 | 7 | 248 | 200 | ~1.190 | confirmação |
| aid | Cena aérea | cena | 10.000 | 30 | 1,9 | 600 | 1.502 | B: confirmação. A: externa |
| remote_sensing | Cena aérea | cena | 1.000 | 14 | 8,7 | 640 | ~150 | B: confirmação. A: externa |

Fundo preto e fundo real são **curvas separadas**, nunca misturadas, mesmo no mesmo papel.

## 4. As opções

| | **A: só os 12** | **B: 12 + 2 inéditos** | **C: só os 12, descoberta menor** |
|---|---|---|---|
| Descoberta | Eggs, Larvae, flowers | Eggs, Larvae, flowers | Eggs, flowers |
| Confirmação | Protozoan | Protozoan, Cena aérea | Protozoan, Larvae |
| Externa | Cena aérea | coconut_trees, bloodmnist | Cena aérea |
| Datasets | 7 / 3 / 2 | 7 / 5 / 2 | 4 / 6 / 2 |
| Famílias | 3 / 1 / 1 | 3 / 2 / 2 | 2 / 2 / 1 |
| Imagens de treino a 100% | 21 mil / 38 mil / 8 mil | 21 mil / 46 mil / 21 mil | 17 mil / 42 mil / 8 mil |
| Descoberta: classes, lado | 2 a 9; 200 a 1.328 | igual | 5 a 9; 200 a 432 |
| Confirmação: classes, lado | 7; 112 a 200 | 7 a 30; 112 a 640 | 2 a 7; 112 a 1.328 |
| Confirmação tem imagem natural? | **não** | sim | **não** |
| Dá para deixar 1 família de fora? | sim | sim | fraco: sobra 1 |
| Teste da externa nunca calculado? | **não** | **sim** | **não** |
| Desvio do pedido | nenhum no conjunto; 7/3/2 no lugar de 3/3/2 | **+2 datasets fora dos 12**; recupera 3/3/2 quase em famílias (3/2/2) | nenhum no conjunto; 4/6/2 |

- **A** respeita o pedido à risca. O ponto fraco: a confirmação é uma família só, e a "externa" não é inédita.
- **B** é a única com externa de verdade. Custa um desvio e mais runs.
- **C** é mais barata, mas perde Larvae da descoberta, justo o teste mais forte de escala do objeto (`L_input` seção 3), e fica com 2 famílias para achar a lei.
- **Descartada, 5/4/3** (Eggs + Cena aérea, Protozoan + flowers, Larvae): é a mais próxima da proporção 3:3:2, mas a "externa" seria do mesmo laboratório e a mais ruidosa. Um teste que não enxerga 0,06 aprova qualquer lei.

**Por que A e B dividem a mesma descoberta:** a decisão sobre inéditos só é necessária antes da confirmação. Nada na descoberta precisa esperar.

## 5. Datasets ainda não vistos

Hash: nenhuma imagem deles repete pixel de outro dataset. `coconut_trees` não tem par com `aid` nem `remote_sensing` (melhor NCC 0,95).

| dataset | imagens, classes, razão | lado | val | duplicatas exatas entre treino e val+teste (split 1) | problema | veredito |
|---|---|---|---|---|---|---|
| **coconut_trees** | 13.587, 2, 3,1 | 90 | 2.038 | 0 | Telhas vizinhas de 2 mosaicos (`upper`, `lower`) caem em treino e teste; sem coordenada no nome | **usar** |
| **medmnist_bloodmnist** | 17.092, 8, 2,7 | 28 | 2.564 | 10 | Só 28 px | **usar** |
| fer2013 | 35.887, 7, 16 | 48 | 5.383 | 753 | Cinza, 1.516 grupos de cópias, quase-duplicatas em 13,5% | só limpando |
| food101 | 101.000, 101, 1,0 | 512 | 15.150 | 29 | **Largura menor que 101 dá 0 filtros** (abaixo); 70.700 no treino | não, inteiro |
| medmnist_dermamnist | 10.015, 7, 58 | 28 | 1.502 | 2 | Mesma lesão repetida; sem id no disco (`configs/organize_datasets.yaml:109`) | não |
| medmnist_octmnist | 109.309, 4, 5,8 | 28 | 16.396 | 3.293 | 7.891 cópias exatas; paciente sem id | não |
| medmnist_organsmnist | 25.211, 11, 5,2 | 28 | 3.783 | 0 | Fatias vizinhas: 128 de val+teste têm quase-par no treino | não |
| medmnist_retinamnist | 1.600, 5, 7,8 | 128 | 240 | 0 | Val pequena; classes em ordem; 61% de quase-pares (pode ser falso positivo: fundo de olho se parece) | não |
| medmnist_breastmnist | 780, 2, 2,7 | 28 | 118 | 1 | Val mínima | não |

**Por que `food101` trava:** o SPiFiL divide os filtros igualmente entre as classes: `out_channels // n_classes` (`SPiFiL/src/spifil/allocation.py:53`). 64 filtros em 101 classes dá 0 por classe. E o SPiFiL não pode ser alterado.

**O que falta para usar coconut_trees e bloodmnist:**

1. Uma linha para cada em `ORGANIZED` (`core/constants.py:68`). Schema, preflight, cache e benchmark leem dali (`experiments/ray/schema.py:257`, `benchmark/common.py:41`).
2. Rodar o benchmark nos 2. AIMv2 e DINOv2 levam ~250 ms por imagem cada: cerca de 2 h no coconut e 2,5 h no blood. O SVM é de minutos.
3. Montar o cache de imagens.
4. Os splits **já existem e estão corretos**: 3 splits, 6 frações, aninhados, val e teste fixos, todas as classes a 1%.
5. Rodar B0 a B3 nas 6 frações e 3 splits: 72 runs por dataset, 144 no total, por semente.

**Ruído:** ainda não há benchmark nos dois, então a régua deles não foi medida. A val de 2.038 e 2.564 sugere algo perto de Eggs e `aid`; é estimativa.

**Limite honesto de B:** os dois têm imagem pequena. Abaixo de 100 px a lei só vê ampliação, sem detalhe novo. Eles testam domínio e largura, mas **não** testam o eixo de resolução. Se isso pesar, há duas saídas, ambas mais trabalhosas: bloodmnist em 224 px (precisa baixar e adaptar o padronizador; não testei), ou `food101` reduzido a 30 classes (precisa de manifest e splits novos). Nos dois casos a escolha é do usuário.

**Em `food101`, a régua pode ser injusta:** até onde lembro do artigo do DINOv2, Food-101 está entre as fontes dos dados de treino dele. Não conferi. Para nós isso só deixaria o alvo mais difícil, não mais fácil.

## 6. Frações

**O que já está certo.** Conferi os 21 datasets, 3 splits, 6 frações: val e teste iguais em todas as frações, frações aninhadas, treino sem sobreposição com val e teste. As classes vêm do manifest quando há (`config.py:117`), senão do prefixo.

**Regra das células:**

- **L, entra na lei:** toda classe tem ao menos 5 imagens de treino, nos 3 splits.
- **R, só relato:** todas as classes presentes, mas alguma com menos de 5. Roda, aparece no gráfico, não entra no ajuste.
- **X, fora:** falta classe em algum split, ou menos de 10 imagens.

Na célula: papel, imagens de treino, e entre parênteses o **menor** número de imagens de uma classe no pior split.

| dataset | 1% | 5% | 25% | 50% | 75% | 100% | pontos L |
|---|---|---|---|---|---|---|---|
| Eggs | R 96 (1) | R 452 (3) | L 2.228 (15) | L 4.448 (30) | L | L | 4 |
| helminth_eggs | R 40 (1) | R 184 (3) | L 897 (14) | L 1.791 (28) | L | L | 4 |
| Eggs_subgroup | **X** 33-35 (0) | R 145-150 (1) | L 681-702 (9) | L ~1.390 (16) | L | L | 4 |
| Larvae | R 12 (2) | L 57 (9) | L 281 (44) | L 560 | L | L | 5 |
| helminth_larvae | R 26 (4) | L 124 (16) | L 615 | L 1.230 | L | L | 5 |
| Larvae_subgroup | **X** 3-7 (1) | R 25-31 (4) | L 124-136 (23) | L ~255 | L | L | 4 |
| flowers | R 22 (4) | L 99 (18) | L 482 | L 963 | L | L | 5 |
| Protozoan | R 268 (3) | L 1.315 (11) | L 6.546 | L 13.086 | L | L | 5 |
| protozoan_cysts | R 72 (1) | R 339 (3) | L 1.678 (14) | L 3.351 | L | L | 4 |
| Protozoan_subgroup | **X** 56-59 (0) | R ~280 (1) | R ~1.380 (2) | L ~2.770 (5) | L (8) | L (9) | **3** |
| aid | R 84 (2) | L 364 (8) | L 1.761 | L 3.498 | L | L | 5 |
| remote_sensing | **X** 16 (0) | **X** 41 (0) | R ~180 (2) | L ~350 (5) | L (10) | L (14) | **3** |
| coconut_trees | L 96 (24) | L | L | L | L | L | 6 |
| medmnist_bloodmnist | L 123 (9) | L | L | L | L | L | 6 |

**Classes que somem a 1%.**

- `Eggs_subgroup`: faltam 2 classes no split 1 e 1 no split 3.
- `Protozoan_subgroup`: faltam 1, 1 e 2 classes. `E.histolytica` tem só 16 imagens no total, e uma classe chega a ter 1 imagem no teste.
- `remote_sensing`: presentes 11, 12 e 11 de 14.

O κ é calculado sobre todas as classes da val, então classe ausente é erro garantido. Essas células medem falta de classe, não a lei.

**`remote_sensing` a 1%: 16 imagens.** Uma por classe, e nem isso. O benchmark inteiro fica entre 0,21 e 0,39. Isso é poucos exemplos, um regime diferente. Fica fora, e 5% também. A lei dele tem só 3 pontos: dá para medir a inclinação, não um piso. O mesmo vale para `Protozoan_subgroup`.

**Frações aninhadas têm um custo.** A de 5% contém a de 1%, então um sorteio ruim contamina a curva inteira daquele split. Os 18 pontos de um dataset **não** são 18 observações independentes, mas 3 curvas. O ajuste tem que tratar o split como efeito aleatório.

**Duas armadilhas novas que mudam a leitura da curva:**

1. **A largura muda sem aviso.** O SPiFiL só divide filtros entre classes presentes na fração (`flim/spifil.py:514,534`). Em `remote_sensing`, 64 filtros pedidos viram 56 com 14 classes, 52 com 13, 60 com 12 e 55 com 11. E no `aid` (30 classes), 32 e 48 pedidos dão os mesmos 30 filtros: na grade `sc`, `w1` 32 e 48 dão a mesma largura de L1 nesse dataset. **Na lei, a largura é a real (`architecture.json`), nunca a pedida.**
2. **Menos dados também é menos passos.** Com épocas fixas e lote 32, a 1% `Larvae` dá 1 passo por época e a 100% dá 35; `Protozoan` vai de 9 a 818. A curva de dados mistura duas coisas. Controle barato: `helminth_eggs` e `flowers` a 5% com os mesmos passos de 100%. Quem define é o plano de treino; aqui só fica o aviso.

**O benchmark já tem as 6 frações:** 4 modelos, 216 linhas `ok` cada (`results/benchmark/*.csv`), mais completo que o `results_log.md` indica. Só falta DINOv3.

## 7. Ruído e diferença mínima detectável

A régua aqui é a menor diferença que dá para ver em 80% das vezes, com 5% de alarme falso. Com 3 splits ela vale **3,1 vezes** o desvio da diferença pareada, porque 3 pontos quase não dão certeza. Com 5 splits vale 1,66 vezes, com 10 vale 0,99.

O desvio vem das diferenças pareadas entre os 4 modelos do benchmark, por split, só na validação. Ele é um limite por cima para modelos parecidos entre si.

| dataset | val | desvio (25-100%) | **enxerga, 3 splits** | 5 splits | 10 splits | 3 splits a 5% | 3 splits a 1% | desvio entre braços nossos (L1) |
|---|---|---|---|---|---|---|---|---|
| Protozoan | 5.607 | 0,007 | **0,021** | 0,011 | 0,007 | 0,054 | 0,059 | sem dado |
| aid | 1.502 | 0,007 | **0,021** | 0,011 | 0,007 | 0,049 | 0,104 | 0,015 |
| protozoan_cysts | 1.435 | 0,007 | **0,021** | 0,012 | 0,007 | 0,053 | 0,102 | 0,035 |
| helminth_eggs | 766 | 0,008 | **0,025** | 0,013 | 0,008 | 0,074 | 0,150 | 0,025 |
| Eggs | 1.903 | 0,011 | **0,032** | 0,017 | 0,010 | 0,022 | 0,153 | 0,040 |
| Protozoan_subgroup | ~1.190 | 0,013 | 0,040 | 0,022 | 0,013 | 0,105 | 0,185 | 0,034 |
| flowers | 412 | 0,014 | 0,042 | 0,022 | 0,013 | 0,051 | 0,146 | 0,041 |
| Eggs_subgroup | ~600 | 0,016 | 0,049 | 0,026 | 0,016 | 0,081 | 0,215 | 0,045 |
| helminth_larvae | 527 | 0,019 | 0,060 | 0,032 | 0,019 | 0,093 | 0,227 | 0,036 |
| remote_sensing | ~150 | 0,031 | 0,097 | 0,052 | 0,031 | 0,137 | 0,132 | 0,045 |
| Larvae | 240 | 0,046 | **0,143** | 0,077 | 0,046 | 0,236 | 0,676 | 0,181 |
| Larvae_subgroup | ~111 | 0,056 | **0,173** | 0,093 | 0,056 | 0,526 | 0,575 | 0,191 |

**O que isso muda:**

- **O aceite de 0,01 não dá para provar em nenhum dataset com 3 splits.** `L_data` 3.2 dizia que dá em 4; ele olhou o erro-padrão da média, e aqui é a diferença pareada com só 2 graus de liberdade. Os dois estão certos, sobre perguntas diferentes.
- **A 1%, quase nada é separável:** abaixo de 0,10 só em Protozoan. A 5% a régua piora pouco nos grandes e muito em Larvae.
- **Sementes não resolvem isso.** Dois treinos iguais em Eggs diferem de 0,0001 a 0,0035 (`L_data` 2.1), bem menos que entre splits. Quase todo o ruído vem de quais imagens caíram na validação. Semente a mais só encolhe a parte pequena. Ressalva: a semente escolhe as imagens do SPiFiL (`flim/spifil.py:530`), e isso pode pesar mais em fração baixa.
- **Na descoberta, quem decide é `helminth_eggs`, Eggs e, com peso menor, `flowers`, `Eggs_subgroup` e `helminth_larvae`.** Larvae e `Larvae_subgroup` só dão direção e medem resolução.
- **A grade `sc` usa `Larvae_subgroup`, o dataset mais ruidoso.** Nas próximas grades, trocar por `helminth_larvae`.
- **Teto:** a 100%, 7 dos 12 têm melhor κ ≥ 0,98. `Larvae_subgroup` chega a 1,000 no AIMv2 com ~110 de validação. Ali a régua não separa quem empata de quem ganha.

Com 5 famílias, a lei compartilhada tem 5 unidades, não 12. Isso serve para derrubar uma regra, não para cravar um expoente.

## 8. Vazamento

Os datasets treinam separados, então nenhum modelo vê imagem de outro dataset. O risco é de outro tipo: confirmar a lei em imagens que já moldaram a lei e achar que é prova nova.

| o que medi | resultado | regra |
|---|---|---|
| Pixel idêntico entre os 21 datasets (410.720 imagens) | **0** | Hash **não** acha irmãs: cada versão tem outro recorte. Quem liga as irmãs é `mapa_ids.csv` |
| `aid` e `remote_sensing` | 453 cópias (seção 2) | Mesmo papel |
| Subgrupo e versão antiga (`helminth_*`, `protozoan_cysts`) | **Split herdado: 100% no mesmo conjunto**, nos 3 splits (3.974, 7.919 e 741 de cada) | Não são dois datasets independentes. A diferença entre eles é efeito do fundo, um experimento pareado de graça |
| Subgrupo e base (split 1) | **Sorteios independentes.** Teste do subgrupo no treino da base: Eggs 411/599, Protozoan 854/1.197, Larvae 78/115 | Nunca treinar numa e medir na outra |
| Duplicatas exatas dentro de um dataset, cruzando treino e val+teste | Eggs 1, aid 1, flowers 1 a 2 | Ignorar. Peso desprezível |
| Quase-duplicatas internas (NCC ≥ 0,98) | aid 0,1%, flowers 0,9%, remote_sensing 0 | Ignorar |
| Mesma imagem em famílias de parasitas diferentes | 0 pixel idêntico nas 3 versões. A 96 px o melhor par dá 0,985, contra 0,9989 das cópias reais do `aid`. Compatível com fundo liso; não olhei par a par | **Não provado por completo:** ver abaixo |

**Limite do detector.** Testei o detector nos pares verdadeiros do `mapa_ids.csv`: o NCC deles é igual ao de pares ao acaso, porque cada versão muda recorte e escala. Ou seja, entre versões diferentes ele não enxerga nada. Por isso a pergunta de R2 sobre imagens "fora do mapa" (1.138, 1.649 e 2.773) segue aberta. Não muda o plano, porque a família inteira fica junta. Entre famílias, o que tenho é "nenhum recorte idêntico". Mesma lâmina ou mesma sessão não dá para medir: não há metadado.

**Regras práticas:**

1. Nada aprendido num dataset vai para uma irmã: pesos, banco de kernels, cache de PCA ou estatística.
2. Comparação entre irmãs é sempre por dataset, e a família conta como 1 unidade.
3. Rótulo: `Protozoan_subgroup` usa o prefixo, e 2 imagens divergem de `mapa_ids` (R2 seção 5). Fica como está. O efeito é mínimo, mas `E.histolytica` tem 16 imagens, não 18.
4. Os YAMLs de cada papel listam só datasets daquele papel, com o papel no nome. Sem código novo.

### O que já foi visto: nenhum dos 12 é virgem

| exposição | onde | efeito |
|---|---|---|
| Teste calculado nos 12 | `results/benchmark/*.csv` (`test_*`), `results/svm_disc_md_l1fast_*.csv`, `results_log.md` | Teste dos 12 não é inédito |
| Medida da camada 1 no teste, pedida pelo usuário, para a lei tamanho x superpixels | `CONTEXT.md` (06/11 19:30) | Se os 50 superpixels saíram dela, o teste dos 12 ajudou a escolher um parâmetro. **Não achei** `law_size_superpixels` como arquivo para confirmar |
| Grade `sc` e `base` | `fila.md` | Tocam 1 dataset de cada família em split 1, 50% |
| Modelo antigo | `disc_bp01w64c1280_bg1` | Medido em eggs, larvae e protozoan, no split antigo (`L_data` seção 1) |

**Regra daqui para frente:** antes da confirmação, a lei fica congelada e escrita, com a previsão de cada dataset. A confirmação decide pela validação. O teste abre uma vez, no final. Tudo acima vai declarado no relatório. Na opção A, chamar `aid` e `remote_sensing` de "validação externa" exige esta ressalva.

## 9. Pendências e limites

**Para o usuário decidir:**

1. Aprovar B (+2 datasets, ~4,5 h de benchmark, 144 runs por semente)? Só é preciso antes da confirmação. Sem resposta, vale A.
2. A externa precisa testar resolução? Se sim, bloodmnist 224 ou `food101` com 30 classes, com trabalho a mais.
3. Confirmar "5 famílias, não 6". O dado é forte, mas a contagem de 6 veio do usuário.

**Limites:**

- O desvio da seção 7 sai de 3 splits, e os splits compartilham imagens, então subestima um pouco. Os pares entre modelos são dependentes.
- A coluna "braços nossos" vem da camada 1 sem treino, e mistura superpixels, leitura e SPiFiL contra aleatório. Serve de ordem de grandeza. Não há medida equivalente do modelo completo.
- O ruído entre sementes foi visto só em Eggs, em 3 pares.
- Dos 453 pares `aid`-`remote_sensing`, conferi 40 a 128 px.
- `retinamnist` e `breastmnist`: quase-pares podem ser falsos. `coconut_trees`: não medi vizinhança entre telhas, só duplicata.
- Corte de 5 imagens por classe é escolha minha, não resultado. Recomendo ajustar a lei com e sem as células R.
- **Teste:** nenhuma coluna `test_*` entrou em conta. Dois `print` de formato mostraram na tela uma linha de teste (Eggs, ConvNeXt no benchmark; Eggs, 64 px no `l1fast`). Não usei.

**Como medi:** CPU, `nice 19`, 6 processos, sem GPU. Scripts em `<scratchpad>/p2ds/` (`hashall.py`, `an1.py` a `an5.py`). Consultei o grafo primeiro; ele não resolveu `ORGANIZED` (nó ambíguo) nem `UniformAllocator`, que está no pacote SPiFiL, então usei grep nesses dois.
