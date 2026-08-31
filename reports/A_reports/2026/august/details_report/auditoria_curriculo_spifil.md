# Auditoria — o código implementa o currículo de crescimento FLIM + SPiFiL?

Data: 2026-08-21. Auditoria de leitura, feita sobre a árvore de trabalho (com as edições
ainda não commitadas). Nenhum arquivo de código foi alterado durante a auditoria.

## Resposta curta

O código faz **quase tudo** o que o currículo descreve. Dos 8 pontos verificados, 5 estão
implementados, 2 estão parciais e 1 depende de como se lê a palavra "na frente".

As duas diferenças que importam de verdade:

1. **No estágio 3 a camada nova também fica congelada.** O currículo diz que a camada SPiFiL
   nova deve treinar enquanto o resto do encoder está travado. O código congela o encoder
   inteiro, e a camada nova está dentro dele. Só o decoder treina.
2. **Os superpixels nunca são recalculados em profundidade.** Só as sementes (as coordenadas)
   são reprojetadas. E com o padrão `--pool-stride 1` essa reprojeção devolve exatamente as
   mesmas coordenadas em toda rodada — ela funciona, mas não move nada.

## O currículo que foi verificado

Ponto de partida, um encoder e um decoder: `y_enc = encoder(x)`, `y_dec = decoder(y_enc)`.
O encoder é carregado do FLIM; o decoder não.

```
estágio 1:  y_enc congelado (pesos FLIM);  y_dec destravado
estágio 2:  y_enc destravado;              y_dec destravado
```

Depois do estágio 2, uma camada FLIM nova é acrescentada, construída a partir de um SPiFiL
recém-calculado — os superpixels precisam ser reposicionados por reprojeção nas camadas
seguintes, porque o posicionamento antigo não vale na profundidade nova. Cada camada FLIM
nova ganha uma camada de decoder equivalente, com inicialização aleatória.

```
estágio 3:  y_enc congelado;    camada_FLIM_nova treinando;  decoder_novo (aleatório) treinando;  y_dec destravado
estágio 4:  y_enc destravado;   camada_FLIM_nova treinando;  decoder_novo treinando;              y_dec destravado
```

Depois do estágio 4 as peças novas são absorvidas (`y_enc' = y_enc + camada_FLIM_nova`,
`y_decoder' = camada_decoder + y_dec`) e o passo de crescimento se repete.

---

## O que o modelo é, concretamente

Esta seção descreve o modelo que os oito pontos auditam. Os shapes abaixo foram **medidos**
executando o modelo com uma entrada de teste, não deduzidos no papel.

### Dois caminhos que saem do mesmo lugar e não se encontram

O ponto que mais confunde: **o decoder nunca vê o embedding.** São dois consumidores
independentes do mesmo mapa de features.

```
                                    ┌─→ decoder → reconstrução [B, 3, 200, 200]  ← a perda vive aqui
imagem → encoder → [B, 48, 24, 24] ─┤
                                    └─→ redução → embedding → SVM                ← a métrica vive aqui
```

O `forward` prova isso em uma linha (`src/models/autoencoder_resnet.py:155`):

```python
def forward(self, x):
    return self.decoder(self.encoder(x))
```

O decoder recebe `self.encoder(x)`, que é o mapa `[B, 48, 24, 24]` inteiro, com as 24×24
posições preservadas. O `embed` é um método **separado** (`:158`) que só o avaliador e a sonda
chamam. Trocar entre `avgpool2d` e `flatten` **não muda uma vírgula da reconstrução nem da
perda** — muda só o que o SVM recebe.

### O caminho da reconstrução, com os shapes medidos

Entrada `[2, 3, 200, 200]`, dataset eggs:

| Etapa | Shape na saída | O que acontece |
|---|---|---|
| entrada | `[2, 3, 200, 200]` | imagem LAB já em [0, 1] |
| `conv1` | `[2, 24, 99, 99]` | Conv 5×5 + ReLU + MaxPool 3×3 stride 2 |
| `conv2` | `[2, 32, 49, 49]` | idem |
| `conv3` | `[2, 48, 24, 24]` | idem — **o gargalo** |
| decoder bloco 1 | `[2, 32, 48, 48]` | upsample nearest ×2 + 2 convs 3×3, residual interno |
| decoder bloco 2 | `[2, 24, 96, 96]` | idem |
| decoder bloco 3 | `[2, 24, 192, 192]` | idem (o último mantém a largura) |
| `interpolate` | `[2, 24, 200, 200]` | três upsamples ×2 dão 192, não 200; o ajuste é no fim |
| `to_image` | `[2, 3, 200, 200]` | conv 3×3 final, emite **logits** |

O encoder encolhe 200 → 99 → 49 → 24 porque cada `MaxPool2d(3, stride=2)` faz `(n−3)/2 + 1`.
A convolução não encolhe nada: o padding é `kernel//2`, então 5×5 com padding 2 mantém o
tamanho (`src/models/models.py:133`).

### O que é reconstruído

A **própria imagem de entrada**, `[B, 3, 200, 200]`. É um autoencoder puro: o alvo é a
entrada, sem transformação. `_target_from_input` (`src/modules/autoencoder_flim_module.py:324-332`)
só faz `x.clamp(0, 1)` quando `imagenet_norm=False`, que é o caso de todas as execuções do
grid.

O decoder emite **logits**, não pixels — a sigmoide está dentro do `BCEWithLogitsLoss`
(`src/modules/autoencoder_flim_module.py:230`, `:339-342`). Para visualizar a reconstrução o
código aplica `torch.sigmoid` à mão (`:684`).

O rótulo é descartado de propósito no passo de treino (`:345`):

```python
views, _ = batch  # labels deliberately discarded — training is unsupervised
```

Duas decisões de projeto que mudam o que o experimento mede:

- **Não existe skip connection cruzando o gargalo** (`src/models/autoencoder_resnet.py:23-26`).
  O residual mora dentro de cada bloco do decoder e nunca volta ao encoder. Um skip tipo U-Net
  deixaria o decoder reconstruir contornando o embedding — exatamente a pressão que se quer
  testar.
- **O upsample é `nearest` + conv, não `ConvTranspose2d`** (`src/models/autoencoder_resnet.py:43-49`).
  Kernel 5×5 com stride 2 não é divisível, e a transposta nesse caso produz o artefato de
  tabuleiro de xadrez.

### O embedding — e qual modo é escolhido

O SVM nunca vê o decoder. Ele recebe a saída do encoder reduzida de um dos dois jeitos
(`src/utils/evaluate.py:274-276`):

```python
if mode == "flatten":
    return out.flatten(start_dim=1)                 # [B, 48*24*24] = [B, 27648]
return F.adaptive_avg_pool2d(out, 1).flatten(1)     # [B, 48]
```

Os dois saem do **mesmo** mapa `[B, 48, 24, 24]`; e 48 × 24 × 24 = 27.648.

Quem escolhe é a flag `--embed-mode`, e o padrão **não é o mesmo em todo lugar**:

| Onde | Padrão | Linha |
|---|---|---|
| o treinador, quando chamado direto | `avgpool2d` | `src/modules/autoencoder_flim_module.py:768` |
| a fila do autoencoder (`autoencoder_flim_ray.py`) | rodou os **dois** braços | `scripts/autoencoder_flim_ray.py:304` (`_lab` e `_lab_flat`) |
| o laço de crescimento (`spifil_growth_loop.py`) | **`flatten`** | `scripts/spifil_growth_loop.py:379` |

A troca no laço de crescimento é uma exceção deliberada e documentada
(`scripts/spifil_growth_loop.py:147`). A consequência prática está nas tabelas de resultado:
**todo o grid de crescimento é flatten**, então ele não tem contraparte no braço de 48
dimensões.

A flag é global para o processo: `src/modules/autoencoder_flim_module.py:844` faz
`_ev.EMBED_MODE = args.embed_mode`, o que governa tanto a sonda interna quanto o avaliador
oficial.

### O que muda quando o encoder cresce

Uma rodada de crescimento acrescenta `conv4`. Como o padrão é `--pool-stride 1`, ele **não
encolhe** o mapa — a grade continua 24×24. Só o número de canais muda, e varia por dataset
porque o FLIM produz `n_classes × kernels_por_marcador` filtros, limitado pelo teto:

| Dataset | Classes | Canais antes | Canais depois | Embedding avgpool | Embedding flatten |
|---|---|---|---|---|---|
| eggs | 9 | `[3, 24, 32, 48]` | `[3, 24, 32, 48, 45]` | 48 → **45** | 27.648 → **25.920** |
| larvae | 2 | `[3, 24, 32, 48]` | `[3, 24, 32, 48, 48]` | 48 → **48** | 27.648 → **27.648** |
| protozoan | 7 | `[3, 24, 30, 48]` | `[3, 24, 30, 48, 42]` | 48 → **42** | 27.648 → **24.192** |

Duas coisas para notar:

- **O protozoan é diferente desde o começo:** a segunda camada tem 30 canais, não 32. Por isso
  nada no decoder é fixo em 32 — as larguras saem da `architecture.json`
  (`src/models/autoencoder_resnet.py:16-19`).
- **Crescer pode diminuir o embedding.** Eggs sai de 48 para 45 dimensões. O decoder ganha um
  bloco (de 3 para 4), mas com escala 1: ele processa sem fazer upsample. A reconstrução
  continua `[B, 3, 200, 200]` nos dois casos — verificado por execução, e também garantido
  pelo teste em `scripts/check_spifil_growth.py:104-117`.

### Onde cada peça vive

| Peça | Arquivo | O que faz |
|---|---|---|
| Encoder (montagem) | `src/models/models.py:112-146` | monta `conv1..convN` a partir da `architecture.json`: Conv + ReLU + MaxPool |
| Encoder (forward) | `src/models/models.py:165-166` | roda os blocos em ordem crescente |
| Bloco do decoder | `src/models/autoencoder_resnet.py:52` | upsample nearest ×escala, 2 convs 3×3, residual interno |
| Decoder | `src/models/autoencoder_resnet.py:85-125` | espelha o encoder ao contrário, `interpolate`, `to_image` |
| Modelo completo | `src/models/autoencoder_resnet.py:132` | junta os dois, guarda `embed_dim`, tem o `AdaptiveAvgPool2d(1)` |
| Perda | `src/modules/autoencoder_flim_module.py:339-342` | `BCEWithLogitsLoss(logits, alvo)` |
| Alvo da reconstrução | `src/modules/autoencoder_flim_module.py:324-332` | é a própria entrada, com clamp em [0, 1] |
| Passo de treino | `src/modules/autoencoder_flim_module.py:344-347` | descarta o rótulo |
| Redução para embedding | `src/utils/evaluate.py:254-276` | `avgpool2d` ou `flatten` |

---

## Tabela de veredictos

| # | Ponto do currículo | Veredicto | Prova | Nota |
|---|---|---|---|---|
| 1 | Estágio 1 carrega encoder do FLIM e congela; só decoder recebe gradiente | **IMPLEMENTADO** | `src/modules/autoencoder_flim_module.py:211`, `:220-221`, `src/models/models.py:674-676` | Otimizador recebe lista filtrada, não `parameters()` inteiro (`:706`) |
| 2 | Estágio 2 destrava o encoder; os dois treinam | **IMPLEMENTADO** (por construção) | `src/modules/autoencoder_flim_module.py:833`, `scripts/spifil_growth_loop.py:291` | Não existe chamada de "destravar"; é um processo novo onde `freeze_encoder()` não é chamado |
| 3 | Camada FLIM nova é acrescentada **na frente** da atual | **IMPLEMENTADO** — mas no fim, não no começo | `scripts/spifil_grow.py:375`, `:324-326`, `src/models/models.py:165-166` | Ver "a ambiguidade do na frente" |
| 4 | Kernels vêm de um SPiFiL **recalculado**, não do nível da imagem | **IMPLEMENTADO** | `scripts/spifil_grow.py:287`, `:370-371` | Patches saem das features do encoder treinado, a cada rodada, sem cache |
| 5 | Superpixels são **reprojetados** nas camadas seguintes | **PARCIAL** | `scripts/spifil_grow.py:283`, `:223-238` contra `:278-281` | Sementes reprojetadas sim; segmentação nunca refeita |
| 6 | Cada camada FLIM nova ganha uma camada de decoder nova e aleatória | **IMPLEMENTADO** | `src/models/autoencoder_resnet.py:106,111,116`, `src/modules/autoencoder_flim_module.py:905-913` | O decoder velho **não** se perde: os índices são deslocados |
| 7 | Encoder anterior congelado no 1º estágio pós-crescimento, destravado no 2º | **PARCIAL** | `scripts/spifil_growth_loop.py:310-311` | A alternância existe; mas a camada nova congela junto (`src/models/models.py:675-676`) |
| 8 | Absorção acontece e o laço recomeça do modelo absorvido | **IMPLEMENTADO** | `scripts/spifil_growth_loop.py:301,308,310-311`, `src/modules/autoencoder_flim_module.py:914` | Não há função `absorb()`; a absorção é o repasse de checkpoint |

---

## Ponto 3 — a ambiguidade do "na frente"

Em português "na frente" pode significar duas coisas opostas, e o código escolhe uma delas
sem ambiguidade nenhuma:

- **O que o código faz:** a camada nova vira `conv{N+1}` e roda **por último**, depois de
  todas as antigas. Ela come as features das camadas velhas.
- **A outra leitura possível:** inserir antes, perto da entrada.

Nenhum caminho do código insere no índice 0 nem desloca as camadas do encoder. `layer1`
continua sendo `layer1` para sempre. Se a intenção era "à frente no fluxo" (mais fundo), está
certo. Se era "antes da entrada", é uma divergência total.

Três confirmações independentes de que a camada nova é a mais profunda:

1. Os patches que o SPiFiL recorta vêm da **saída** do encoder treinado
   (`scripts/spifil_grow.py:287`), não da imagem — logo a camada nova consome as antigas.
2. `--freeze-spifil-layer` congela `blocks[f"conv{n_layers}"]`, com o comentário "Last block
   only = the grafted SPiFiL layer" (`src/modules/autoencoder_flim_module.py:222-228`).
3. O doc diz "SPiFiL corta a camada N+1" (`spifil_growth.md:28`).

Do lado do **decoder** o inverso acontece e está correto: o bloco novo entra no índice 0
(`src/models/autoencoder_resnet.py:111`), que é o espelho de um append no encoder.

## Pontos 4 e 5 — o que exatamente é recalculado

Duas coisas diferentes entram no SPiFiL a cada rodada, vindas de fontes diferentes
(`scripts/spifil_grow.py:271-290`):

| Campo do `LayerState` | De onde vem | Grade | Linha |
|---|---|---|---|
| `features` (de onde os kernels são recortados) | saída do **encoder treinado atual** | 48×24×24 | `scripts/spifil_grow.py:287` |
| `seeds` (onde recortar) | medoides de superpixel da **imagem LAB crua**, reescalados | 24×24 | `scripts/spifil_grow.py:283` |
| `superpixel_labels` | mapa de rótulos da imagem crua, sem reescalar | 200×200 | `scripts/spifil_grow.py:289` |

Como o recorte de patch lê só `features` + `seeds.coords`
(`SPiFiL/src/spifil/patches.py:86-92`), **todo byte de kernel vem das ativações do encoder
atual.** Os superpixels da imagem decidem apenas *onde* amostrar. Isso é o ponto 4, e está
certo: cada rodada é um processo separado que refaz `preparation.prepare` do zero
(`scripts/spifil_grow.py:278-281`), sem cache, sem banco reaproveitado, sem `.npy` de
sementes salvo.

A reprojeção existe e roda toda rodada. `rescale()` (`scripts/spifil_grow.py:223-238`)
remapeia as coordenadas proporcionalmente para a grade do encoder atual, lida ao vivo em
`scripts/spifil_grow.py:198-201`:

```python
return Seeds(coords=seeds.coords * new // old, labels=seeds.labels, ranks=seeds.ranks, grid=tuple(grid))
```

É uma função escrita à mão porque a API nativa da biblioteca (`Seeds.project`,
`SPiFiL/src/spifil/coords.py:53-56`) só faz divisão inteira por stride, e a cadeia de
`MaxPool2d(3, stride=2)` do encoder vai 200 → 99 → 49 → 24, onde 200/24 não é inteiro
(`spifil_growth.md:278-286`).

Duas ressalvas que impedem um "sim" limpo no ponto 5:

1. **A partição de superpixel é sempre no nível da imagem.** `preparation.prepare` é chamado
   sobre as imagens cortadas, nunca sobre features. Só os medoides viajam. É deliberado e
   documentado (`scripts/spifil_grow.py:272-277`, `spifil_grafting.md:36-39`).
2. **Com o padrão `--pool-stride 1` a grade nunca muda entre rodadas**
   (`scripts/spifil_grow.py:358-359`). `trunk.grid` fica em 24 para sempre, então
   `rescale(200 → 24)` devolve coordenadas idênticas em toda rodada. A reprojeção está viva e
   correta, mas depois da rodada 1 não move nada. Ela só faz trabalho de verdade com
   `--pool-stride 2`.

### Uma hipótese que foi levantada e se mostrou falsa

O título do doc diz "cortada do backbone já treinado"
(`spifil_growth.md:1`), o que levantou a suspeita de que o código estivesse fatiando
camadas de uma rede pré-treinada em vez de calcular SPiFiL. **Não é isso.** "Cortada do
backbone" quer dizer que os filtros são recortes dos mapas de features que o backbone
treinado produz. É um SPiFiL de verdade, refeito a cada rodada, sobre ativações ao vivo
(`scripts/spifil_grow.py:287`, `SPiFiL/src/spifil/patches.py:86-92`). Nenhuma camada é
transplantada de rede pré-treinada em `spifil_grow.py`.

## Ponto 7 — o congelamento é tudo-ou-nada

Não existe congelamento por camada em lugar nenhum do repositório. Varrendo os 40 usos de
`requires_grad` em `src/` e `scripts/`, só aparecem dois formatos: congelar o modelo inteiro,
e congelar o último bloco (`src/modules/autoencoder_flim_module.py:226-228`) — que é o
**contrário** do que o currículo pede, porque adiciona a camada nova ao conjunto congelado em
vez de tirá-la. Não há grupos de parâmetro com `lr=0`.

Vale dizer: **isso é deliberado no projeto, não um esquecimento.** O
`spifil_growth.md:28` tabela o estágio 3 como "congelado + uma camada nova";
`spifil_growth.md:60-62` dá a razão ("o bloco novo do decoder é aleatório de novo. Toda
rodada reintroduz exatamente o problema que o estágio 1 resolveu"); e
`spifil_growth.md:96-110` argumenta que deixar o gradiente de um decoder aleatório bater
em filtros recém-recortados é justamente o que se quer evitar. O repositório e o currículo
discordam de propósito neste ponto.

## Qual mecanismo governa o congelamento em execução

Quem manda é o **`requires_grad`**. O filtro do otimizador é consequência dele, não um
interruptor independente — e os dois não podem discordar, porque o filtro lê `requires_grad`
na hora de montar o otimizador.

Ordem dos eventos, tudo no mesmo processo:

1. `__init__` monta o modelo, carrega o FLIM (`:211`) e põe `requires_grad=False` no encoder
   se a flag estiver ligada (`:220-221`).
2. O Lightning chama `configure_optimizers` **depois**, e a lista do AdamW sai dos flags já
   definidos (`:706`).

No estágio 1 o encoder fica duplamente excluído: nenhum gradiente é acumulado, e nenhum
estado de otimizador é criado para ele. O `.eval()` não participa disso — em `_embeddings()`
ele só alterna em volta da sonda SVM e restaura o modo anterior (`:386-394`). Não há cirurgia
em `param_groups` nem hook de freeze.

Fragilidade de mão única que vale nomear: como a lista é materializada uma vez, um destravar
**depois** da construção do otimizador produziria gradientes que nunca seriam aplicados.
Nada neste caminho faz isso.

Duas armadilhas clássicas **não** existem aqui:

- **Sem BatchNorm no encoder.** Ele é só `Conv2d + ReLU + MaxPool2d`
  (`src/models/models.py:138-146`), então não há estatística correndo por baixo de um encoder
  "congelado". BatchNorm existe só no decoder (`src/models/autoencoder_resnet.py:68,70`), que
  treina nos dois estágios.
- **Sem weight decay em parâmetro congelado.** O `weight_decay=5e-2` do AdamW só atinge a
  lista filtrada (`:705-708`).

## O que uma rodada de crescimento executa de fato

Antes das rodadas, uma vez por braço:

1. `stage1` — `--freeze-encoder`, sem `--init-ckpt`. Encoder inteiro congelado (FLIM cru),
   decoder aleatório (`scripts/spifil_growth_loop.py:290`).
2. `stage2` — sem freeze, `--init-ckpt stage1/checkpoints/best_kappa.ckpt`
   (`scripts/spifil_growth_loop.py:291`).

Depois, cada rodada são **3 processos**:

3. `round{r}_grow` — carrega o checkpoint anterior (`scripts/spifil_grow.py:370-372`),
   congela aquele encoder e põe em `.eval()` (`:195-197`), recalcula os superpixels no LAB e
   reprojeta as sementes (`:271-290`), recorta **uma** camada SPiFiL sem backprop (`:143`), e
   escreve um diretório de pesos FLIM completo — `conv1..conv{n-1}` copiados do diretório
   anterior (`:302-304`) mais o `conv{n}` novo (`:306`) e a `architecture.json` crescida
   (`:311`).
4. `arch, weights` passam a apontar para o diretório novo
   (`scripts/spifil_growth_loop.py:308`).
5. `round{r}_stage3` — `--freeze-encoder --init-ckpt <melhor anterior>`
   (`scripts/spifil_growth_loop.py:310`). Encoder inteiro congelado.
6. `round{r}_stage4` — sem freeze, a partir do melhor do estágio 3
   (`scripts/spifil_growth_loop.py:311`). Internamente é registrado como estágio 2 — não
   existe `stage=4` no treinador (`src/modules/autoencoder_flim_module.py:833`).
7. O kappa dos dois estágios é lido do `run_metadata.json`
   (`scripts/spifil_growth_loop.py:315-317`); só o do estágio 4 entra na regra de parada.
8. Checagem de parada (`scripts/spifil_growth_loop.py:318`).

Orçamento impresso em `scripts/spifil_growth_loop.py:427`: `2 + 2 * max_rounds` processos de
treino por braço.

## Ponto 8 — como a absorção acontece

Não há função `absorb()`. A absorção é o repasse de checkpoint, por dois canais que carregam
coisas diferentes:

- **No disco (diretório de pesos):** os originais **não treinados**.
  `scripts/spifil_grow.py:304` copia `conv1..conv{n-1}` do diretório anterior, mais a camada
  SPiFiL recém-recortada. `src/modules/autoencoder_flim_module.py:215` carrega tudo isso no
  `__init__`.
- **Por cima (checkpoint):** os pesos **treinados**.
  `src/modules/autoencoder_flim_module.py:896` monta o módulo primeiro, e só então `:914`
  sobrepõe o `state_dict` remapeado com `strict=False`. As chaves de encoder que batem
  (`conv1..conv{n-1}`) são sobrescritas com os valores treinados; `conv{n}` não existe no
  checkpoint, então permanece no seu init SPiFiL.

É essa sobreposição que é a absorção.

Cadeia completa de arquivos:

```
stage1/checkpoints/best_kappa.ckpt
      └─lido como --init-ckpt por→ stage2
stage2/checkpoints/best_kappa.ckpt
      └─lido como --ckpt por→ round1_grow
      └─lido como --init-ckpt por→ round1_stage3
round1_grow/{architecture.json, conv1..conv4-*.npy/.txt}
      └─lido como --arch-json/--flim-weights-path por→ round1_stage3 E round1_stage4
      └─lido como fonte de cópia por→ round2_grow
round1_stage3/checkpoints/best_kappa.ckpt
      └─lido como --init-ckpt por→ round1_stage4
round1_stage4/checkpoints/best_kappa.ckpt
      └─lido como --ckpt por→ round2_grow
      └─lido como --init-ckpt por→ round2_stage3
```

O carregamento é só de pesos, por decisão de projeto — otimizador, scheduler e contador de
época **não** são restaurados (`src/modules/autoencoder_flim_module.py:750-754`, com a razão
em `:888-895`).

### O lado do decoder

O decoder treinado **sobrevive**. O módulo é reconstruído do zero (todos os blocos
aleatórios) e depois o `state_dict` anterior é aplicado com os índices deslocados por `delta`
(`src/modules/autoencoder_flim_module.py:905-913`): o bloco velho `i` cai no bloco novo
`i+delta`, e o bloco novo 0 — o up-block do gargalo, que corresponde à camada de encoder nova
— fica no init aleatório.

A aritmética de forma fecha: o bloco velho `i` espelha a camada de encoder `N_old − i`; o
bloco novo `i+1` espelha `N_new − (i+1) = N_old − i`. Mesma camada, mesmas larguras. O
`to_image` mantém o nome porque a largura dele é `channels[1]`, que o crescimento não toca
(`src/modules/autoencoder_flim_module.py:902-903`).

## Controle do laço

- `--max-rounds`, padrão 4 (`scripts/spifil_growth_loop.py:367`).
- Três saídas possíveis: teto de rodadas atingido; kappa estagnado via `should_stop`
  (`scripts/spifil_growth_loop.py:160-165`, tolerância 0.01 e paciência 1 nos padrões); ou
  `spifil_grow` devolvendo `EXIT_BUDGET = 3` (`scripts/spifil_growth_loop.py:302-305`).
- `BudgetExhausted` (`scripts/spifil_grow.py:114`) governa o **orçamento de covariância de
  Mahalanobis**, não época e não tempo de GPU: a métrica inverte uma covariância `D×D` com
  `D = in_channels * kernel²`; `D` cresce a cada rodada enquanto `N` (o número de sementes)
  não cresce, e quando `N ≤ D` a estratégia se recusa (`scripts/spifil_grow.py:137-142`) em
  vez de deixar a regularização de Ledoit-Wolf manter a matriz invertível mas sem informação.

---

## O que teria que mudar (se o currículo é a referência)

Em ordem de impacto. **Nada disso foi feito** — é lista, não patch.

1. **Congelamento parcial no estágio 3.** Hoje não existe mecanismo nenhum. Precisaria de um
   filtro por índice de camada em `freeze_encoder` (`src/models/models.py:672`), para travar
   `conv1..convN` e deixar `conv{N+1}` solto. É a única mudança que altera o resultado do
   treino.
2. **Decidir o que "reprojetar" significa.** Se for "mover as sementes", já está feito. Se for
   "re-segmentar na profundidade nova", a biblioteca SPiFiL não oferece isso —
   `SPiFiL/src/spifil/preparation.py:74-75` é o único ponto onde o algoritmo de superpixel
   roda em todo o pacote. Seria código novo, e o próprio `spifil_grafting.md:125-127`
   argumenta que fazer isso seria "outro método".
3. **Confirmar a direção do ponto 3.** Se "na frente" for perto da entrada, é uma reescrita
   grande. Se for "mais fundo", não há nada a fazer.

---

## Achados operacionais (não são o currículo, mas quebram coisa)

**1. A fila do Ray quebrou com as edições não commitadas.**
`scripts/autoencoder_flim_ray.py:285` ainda declara
`_STAGE_CKPT = {1: "best_recon.ckpt", 2: "best_kappa.ckpt"}`, mas o módulo na árvore de
trabalho agora devolve `best_kappa` para **todos** os estágios
(`src/modules/autoencoder_flim_module.py:295`) — o ramo do estágio 1 com
`("…/val_recon_loss", "min", "best_recon")` foi apagado. O estágio 1 nunca escreve
`best_recon.ckpt`, e o guard em `scripts/autoencoder_flim_ray.py:614` vai falhar com
"stage-1 checkpoint not found" em toda célula. O `spifil_growth_loop.py:105`
(`BEST_CKPT_FILENAME = "best_kappa.ckpt"`) está consistente com o módulo e não é afetado.

**2. O checkpoint do estágio 1 é escolhido por uma métrica que é constante mais ruído.**
No estágio 1 o encoder não se move, então o embedding é idêntico toda época e o
`probe/svm_kappa` varia só com o ruído do próprio SVM. O código diz isso em
`src/modules/autoencoder_flim_module.py:288-293`, e `scripts/autoencoder_flim_ray.py:283-284`
ainda documenta a intenção antiga e mais sólida ("Stage 1 selects its checkpoint on
val/recon_loss"). Consequência: o decoder de onde o estágio 2 parte é escolhido por uma nota
que é matematicamente independente do decoder.

**3. O caminho do checkpoint é reconstruído por convenção, nunca verificado.**
`scripts/spifil_growth_loop.py:286` fixa `<estágio>/checkpoints/best_kappa.ckpt`, enquanto o
módulo grava o caminho autoritativo em `run_metadata.json`
(`src/modules/autoencoder_flim_module.py:1038`). O `ModelCheckpoint` do Lightning versiona em
colisão: um estágio re-executado no mesmo diretório gera `best_kappa-v1.ckpt` e deixa o velho
no lugar. O laço então alimentaria o checkpoint **velho** no estágio seguinte e no `grow`, em
silêncio — `--init-ckpt` só reclama quando o arquivo **falta**
(`src/modules/autoencoder_flim_module.py:817-818`), nunca quando ele está velho.

**4. Perda silenciosa do decoder com um `--init-ckpt` malformado.**
`src/modules/autoencoder_flim_module.py:898` faz
`torch.load(...).get("state_dict", {})` e `:914` carrega com `strict=False`. Um checkpoint sem
a chave `state_dict` devolve `{}`, e o estágio 2 treina um decoder **aleatório** registrando
apenas `missing=N unexpected=0` em nível INFO (`:915-919`). Nada garante que alguma coisa foi
de fato carregada.

**5. `--freeze-spifil-layer` é repassado para todos os estágios, inclusive os pré-crescimento.**
`scripts/spifil_growth_loop.py:227-228` acrescenta a flag sem condição em `_trainer_cmd`, que
serve do estágio 1 ao 4. No estágio 2 (modelo ainda de 3 camadas)
`src/modules/autoencoder_flim_module.py:226` resolve "o último bloco" como `conv3` — uma
camada **FLIM** original, não um enxerto SPiFiL — e congela ela, de modo que o estágio 2
deixa de ser "o modelo inteiro solto". Inofensivo no padrão (`False`), latente sempre que a
ablação for rodada.

**6. `delta` é inferido, nunca conferido.**
`src/modules/autoencoder_flim_module.py:905-907` calcula `delta` como o número de camadas
novas menos a contagem de índices distintos de bloco de decoder no checkpoint. Nunca é
verificado se está entre 0 e 1. Um checkpoint de um modelo **mais profundo** produziria delta
negativo e remaparia em índices errados ou inexistentes, com o único sinal sendo as contagens
`missing`/`unexpected` do log. Não é alcançável pelo `spifil_growth_loop.py`, que sempre
encadeia para frente.

**7. `check_spifil_growth.py` não testa nada disso.**
Ele cobre três coisas, e as três batem com o código: forma do decoder crescido (`:104-127`),
`should_stop` (`:130-143`), e o round-trip de layout de peso FLIM (`:146-167`). Nada nele
toca estado de congelamento, `requires_grad`, o par estágio 3 / estágio 4, `--init-ckpt`, o
remap de índice do decoder, ou o repasse de peso entre rodadas. Em particular o `delta`
(`src/modules/autoencoder_flim_module.py:905-912`) — o único ponto onde o decoder treinado
pode ser descartado em silêncio, e que `spifil_growth.md:377-399` aponta como uma carga
que falha *sem erro* — **não tem teste nenhum**.

**8. `spifil_growth.md` está correto no conteúdo, mas com as linhas defasadas.**
Todas as afirmações de protocolo que conferi batem. As referências de linha, não: por
exemplo `spifil_growth_loop.py:246-267` → hoje `299-321`; `:248` → `301`; `:130-131` →
`227-228`; `autoencoder_flim_module.py:810-812` → `834-838`; `:872-895` → `897-919`.

---

## Ressalvas de método

- Nenhuma execução foi feita. Tudo aqui é leitura de código na árvore de trabalho de
  2026-08-21. Onde a auditoria depende de comportamento de biblioteca (versionamento do
  `ModelCheckpoint`, por exemplo), isso está dito no achado.
- O grafo do graphify (`graphify-out/`, construído em 2026-08-18) localizou os nós, mas está
  levemente defasado: `scripts/spifil_growth_loop.py` cresceu ~170 linhas depois daquela
  construção. Todos os números de linha deste relatório foram reconferidos direto nos
  arquivos, não tirados do grafo.
- O pacote `spifil` foi lido em `/dados/home/moliveira/SPiFiL` (instalação editable) só para
  entender a API; ele não faz parte deste repositório.
