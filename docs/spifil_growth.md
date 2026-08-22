# Crescimento SPiFiL — uma camada por rodada, cortada do backbone já treinado

Notas de raciocínio do código que já está escrito e em disco:
[`scripts/spifil_grow.py`](../scripts/spifil_grow.py) (uma rodada),
[`scripts/spifil_growth_loop.py`](../scripts/spifil_growth_loop.py) (o laço e a regra de
parada) e [`scripts/check_spifil_growth.py`](../scripts/check_spifil_growth.py) (a
verificação runnable). Companheiro de [`spifil_grafting.md`](spifil_grafting.md), que
explica o SPiFiL em si — este aqui explica **por que** o protocolo tem a forma que tem.

**O que foi medido e o que não foi.** Medido: a rodada 1 do `grow` em eggs/split 1/pct 50
com `--n-images 60 --n-superpixels 100` (`N=5962`, `D=432`, 48 canais pedidos → 45
filtros), a forma da reconstrução do modelo crescido, e as três verificações de
`check_spifil_growth.py` — as três rodadas nesta sessão, `ALL CHECKS PASSED`.
**Não medido: κ de rodada nenhuma.** Nenhum treino completo do protocolo rodou ainda.
Tudo o que este documento diz sobre κ subir, empatar ou parar é o que o código **fará**,
não o que ele fez.

---

## 1. O protocolo

Quatro estágios, e os dois últimos se repetem — uma camada por rodada.

| Estágio | Encoder | Decoder | O que muda |
|---|---|---|---|
| 1 | congelado (FLIM cru) | aleatório, aprende sozinho | nada no encoder |
| 2 | solto | parte do estágio 1 | ponta a ponta |
| 3 | congelado + **uma camada nova** | ganha um bloco novo (aleatório) | SPiFiL corta a camada N+1 |
| 4 | solto | parte do estágio 3 | ponta a ponta, modelo crescido |

Repete 3–4, uma camada por rodada, até κ parar de melhorar
([`spifil_growth_loop.py:246-267`](../scripts/spifil_growth_loop.py)).

O `grow` da rodada *r* come o checkpoint do estágio 4 da rodada *r−1* — e o do estágio 2
na rodada 1 (`spifil_growth_loop.py:248`). Esse encadeamento é o protocolo inteiro: a
camada tem que sair do backbone treinado, nunca do backbone original de três camadas.

**Cuidado com o vocabulário.** "Estágio 4" é nome de protocolo, não variável do código. O
treinador só conhece três valores em `stage`
([`autoencoder_flim_module.py:269-274`](../src/modules/autoencoder_flim_module.py)):
`2` quando o encoder está solto, `3` quando está congelado **e** veio de `--init-ckpt`,
`1` quando está congelado sem checkpoint. Uma run de estágio 4 é, para o módulo,
`stage=2` e leva a tag `stage2_fine_tune`. O que separa o estágio 2 do estágio 4 no
W&B é o nome da run (`stage2` contra `round1_stage4`) e o `parent_run`, não o campo
`stage`.

---

## 2. Por que o decoder aprende primeiro, contra um encoder congelado

**Porque um decoder aleatório produz um gradiente grande e sem sentido.**

Se o encoder estiver solto enquanto o decoder ainda é ruído, esse gradiente cai em cima
dos kernels FLIM. E os kernels FLIM não são inicialização: são um **prior** montado a
partir de patches de imagem real. O ruído os destrói antes de ter chance de melhorá-los.

O estágio 1 leva o decoder até o ponto em que o gradiente dele carrega informação sobre a
imagem. Só então vale a pena aplicar esse gradiente no encoder — e isso é o estágio 2.

O mesmo argumento é a razão do estágio 3 recongelar: **o bloco novo do decoder é
aleatório de novo.** Toda rodada reintroduz exatamente o problema que o estágio 1
resolveu, um nível acima.

O próprio treinador avisa quando alguém pula essa ordem
([`autoencoder_flim_module.py:810-812`](../src/modules/autoencoder_flim_module.py)):
estágio 2 sem `--init-ckpt` é "encoder solto contra decoder aleatório".

---

## 3. Por que a camada nova sai do backbone treinado, e não das imagens

**Porque um filtro SPiFiL é literalmente um patch do mapa de features em que ele foi
fitado.** Cortar da imagem crua produziria outra primeira camada.

O que se quer é a camada que vem **depois** do que o modelo já aprendeu. Então os patches
têm que sair do encoder no estado em que o estágio 2 (ou a rodada anterior) o deixou.

É por isso que o passo de crescimento lê um **checkpoint**, e não uma pasta de pesos:

```python
module = AutoEncoderFlimModule.load_from_checkpoint(args.ckpt, map_location=args.device)
trunk = Trunk(module.model.encoder, args.image_size, args.device)
```
([`spifil_grow.py:370-373`](../scripts/spifil_grow.py))

A pasta `--flim-weights-path` continua sendo passada, mas só para **copiar** as camadas
antigas para o diretório de saída (`write_weights`, `spifil_grow.py:293-311`). Quem doa os
patches é o encoder do checkpoint.

Consequência prática: o `Trunk` congela e põe em `.eval()` o encoder antes de qualquer
coisa (`spifil_grow.py:196-197`). Um gradiente no backbone entre o fit e o uso invalida o
banco em silêncio — o comentário no código diz isso com todas as letras.

---

## 4. Por que o crescimento alterna congelado e solto

**Porque cada rodada muda duas coisas ao mesmo tempo:** o encoder ganha uma camada e o
decoder ganha um bloco.

Soltar as duas juntas significa deixar o gradiente do bloco novo (aleatório) do decoder
bater no encoder inteiro — inclusive nos filtros recém-cortados, que ainda não foram
usados por ninguém.

A metade congelada da rodada paga pelo bloco novo do decoder. A metade solta então move
tudo junto. Alternar separa duas perguntas que de outro jeito se misturam:

- estágio 3: *aprender a inverter o espaço de features novo*;
- estágio 4: *melhorar o espaço de features*.

---

## 5. κ é o único sinal de seleção — e a ressalva

`monitor_spec` devolve `("probe/svm_kappa", "max", "best_kappa")` em **todos** os
estágios ([`autoencoder_flim_module.py:282-295`](../src/modules/autoencoder_flim_module.py)).
Uma chave só, e o prefixo `probe/` é o mesmo em todo estágio e toda rodada — então cada
estágio de cada rodada cai na **mesma curva comparável** no W&B, em vez de uma curva por
estágio. A identidade do estágio vive no nome da run e na config, não na chave da métrica.

Isto foi uma instrução deliberada e **substitui** a seleção anterior, que nos estágios
congelados usava `val_recon_loss`.

**A ressalva, que o próprio repositório carrega no comentário.** Num estágio congelado o
embedding é idêntico byte a byte a cada época. κ então só se mexe pelo ruído de ajuste do
próprio SVM — este repositório mediu **0.0675 de dispersão** em permutações da mesma
matriz de features (protozoan/split1/pct5,
[`autoencoder_flim_module.py:897-902`](../src/modules/autoencoder_flim_module.py)).

Logo: o κ de um estágio congelado é uma **medida do ponto de partida da rodada**, não uma
curva de treino, e qual época vence é quase arbitrário. Leia o κ dos estágios 1 e 3 como
"onde esta rodada começou". A comparação entre rodadas é feita com o κ do estágio 4 — que
é o único que o laço guarda (seção 6).

---

## 6. A regra de parada, exatamente como está no código

```python
def should_stop(kappas, tolerance, patience):
    fails = 0
    for r in range(1, len(kappas)):
        fails = 0 if kappas[r] > max(kappas[:r]) + tolerance else fails + 1
    return fails >= patience
```
([`spifil_growth_loop.py:96-101`](../scripts/spifil_growth_loop.py))

Lendo em português:

- `kappas[0]` é o κ do **estágio 2** (a baseline sem crescimento). `kappas[r]` é o κ do
  **estágio 4** da rodada *r*. O κ do estágio 3 vai para a tabela do resumo, mas **não**
  entra em `kappas` (`spifil_growth_loop.py:261-263`).
- Uma rodada **melhora** quando seu κ de estágio 4 bate o melhor de *todas* as rodadas
  anteriores por **mais** que `--kappa-tolerance` (default `0.01`, espelhando
  `KAPPA_TOLERANCE` em
  [`autoencoder_flim_module.py:137`](../src/modules/autoencoder_flim_module.py)). Empatar
  dentro da tolerância não é ganho.
- `fails` conta rodadas ruins **consecutivas** — uma melhora zera o contador. Uma rodada
  ruim seguida de uma boa não para nada.
- O laço para quando `fails >= --rounds-patience` (default `1`).

O laço também para em mais dois casos:

- o `grow` sai com **código 3** — orçamento de covariância esgotado (seção 7),
  `spifil_growth_loop.py:249-251`; isso não é falha, é o método dizendo que acabou;
- `--max-rounds` é atingido (default `4`).

Qualquer outro código de saída, ou um `run_metadata.json` faltando ou com
`best_val_svm_kappa=null`, **aborta** o laço em vez de chutar número
(`spifil_growth_loop.py:154-163`).

A verificação (b) de `check_spifil_growth.py` cobre 8 casos dessa função, incluindo o
empate dentro da tolerância, a regressão e a recuperação depois de uma rodada ruim.

---

## 7. O orçamento de covariância — N e D

A métrica de Mahalanobis inverte uma covariância `D x D` com
`D = in_channels * kernel_size²`.

**`D` cresce a cada rodada** porque `in_channels` é a largura da camada em que a nova se
apoia. **`N` (as sementes) não cresce.** Em algum momento os dois se cruzam.

`spifil_grow.py` imprime os dois toda rodada e **recusa** quando `N <= D`:

```python
n, d = len(patches), int(patches.feats.shape[1])
print(f"[grow] N={n} D={d}")
if n <= d:
    raise BudgetExhausted(...)
```
([`spifil_grow.py:132-143`](../scripts/spifil_grow.py))

**Por que recusar em vez de avisar.** Porque o shrinkage de Ledoit-Wolf mantém a métrica
*definida*, e isso não é a mesma coisa que informativa. Perto do limiar ela deriva
silenciosamente para uma identidade escalada — o Fisher score continua saindo, com o mesmo
shape, sem nenhum aviso. Um número errado que parece certo é pior que uma parada.

**Números medidos na rodada 1** (eggs, split 1, pct 50, `--n-images 60
--n-superpixels 100`):

```
[grow] N=5962 D=432
```

`D = 432` é `48 canais de entrada × 3²`. A margem é de ~14×, confortável. Na rodada
seguinte `in_channels` passa a ser 45 (seção 11), então `D = 405` — ele só dispara de
verdade se a camada nova for muito mais larga que a anterior.

**A única alavanca que aumenta `N` é `--n-images`.** As sementes saturam em
`--n-superpixels` por imagem: 60 × 100 = 6000, e saíram 5962. Pedir mais superpixels por
imagem não resolve indefinidamente; pedir mais imagens, sim. (O default do `grow` é
`--n-images 200`; a medição acima usou 60.)

---

## 8. Duas decisões que foram tomadas explicitamente

### 8.1 Os filtros SPiFiL descongelam no estágio 4? **Sim, por padrão.**

Depois de gravados nos arquivos de peso FLIM, eles são pesos de conv comuns de uma camada
FLIM comum — indistinguíveis dos kernels FLIM que o estágio 2 já ajusta desde sempre.

Congelar só eles faria da última camada a única congelada do modelo, e o estágio 4
deixaria de significar "o modelo inteiro solto". Isso quebra a comparação entre estágios,
que é o único motivo de todos eles reportarem a mesma métrica.

A escolha oposta existe como flag visível, `--freeze-spifil-layer`, para a ablação que
mantém o banco exatamente como o SPiFiL cortou
([`autoencoder_flim_module.py:755-760`](../src/modules/autoencoder_flim_module.py); o
congelamento em si está em `:222-228`, no último bloco do encoder). O laço repassa a flag
a todos os treinos (`spifil_growth_loop.py:130-131`).

**Contraste com [`scripts/spifil_resnet_graft.py`](../scripts/spifil_resnet_graft.py):**
ali o congelamento é proposital (`:285-291`), porque nada a jusante estava sendo treinado
contra os filtros — o script monta o modelo e para. Aqui há um decoder inteiro treinando
contra eles, todas as épocas. São situações diferentes, e por isso o default é diferente.

### 8.2 Por que a camada nova é gravada como arquivos de peso FLIM, e não como um `nn.Sequential` salvo

**Porque assim o modelo crescido não é um modelo novo.**

O `grow` escreve `conv{N+1}-kernels.npy` + `conv{N+1}-bias.txt` e um `architecture.json`
com `nlayers` incrementado (`spifil_grow.py:293-341`). A partir daí:

- [`build_encoder_from_arch`](../src/models/models.py) (models.py:113) remonta
  `conv1..conv{N+1}` do mesmo JSON;
- [`ResNetDecoder`](../src/models/autoencoder_resnet.py) (autoencoder_resnet.py:85)
  espelha esse mesmo JSON, lendo `pooling.stride` como fator de upsample (`:112`) e
  forçando o tamanho da imagem num `interpolate` final (`:124-125`);
- [`load_FLIM_encoder`](../src/models/models.py) (models.py:555) carrega todas as camadas
  sem saber que a última veio do SPiFiL;
- `get_actual_channels_from_weights` (models.py:84) abre `conv{n}-bias.txt` de 1 a N —
  por isso o diretório de saída é **completo**, com as camadas antigas copiadas, e não só
  a nova.

O estágio 3 é portanto o **treinador que já existia**, apontado para um diretório novo.
Sem fork, sem classe de modelo nova, sem branch em nenhum lugar.

`flim_kernels()` (`spifil_grow.py:209-220`) é o inverso exato de `shift_weights`
(models.py:48) — o layout `.npy` do FLIM tem o canal variando mais rápido, e o
`permute(0, 2, 3, 1) → reshape → .T` reproduz isso. A verificação (c) de
`check_spifil_growth.py` prova o ida-e-volta elemento a elemento:
`shift_weights(flim_kernels(W)) == W`. Sem ela o erro seria silencioso, porque o *shape*
bate nos dois layouts.

---

## 9. O problema da projeção das sementes

Esta é a única parte de método de verdade do `grow`, e vale ler devagar.

**O SPiFiL calcula os superpixels UMA vez, na imagem LAB.** Só as **sementes** viajam para
as camadas seguintes, normalmente via `Seeds.project(stride)`, que é divisão inteira de
coordenada.

Aqui isso não funciona. O `MaxPool2d(3, stride=2, padding=0)` do encoder FLIM leva:

```
200 -> 99 -> 49 -> 24
```

e `200/24` não é inteiro. **Não existe stride inteiro que mapeie o grid LAB no grid de
features.**

`rescale` reposiciona as sementes proporcionalmente:

```python
return Seeds(coords=seeds.coords * new // old, labels=..., ranks=..., grid=tuple(grid))
```
([`spifil_grow.py:223-238`](../scripts/spifil_grow.py))

A conta é exata nas duas pontas: `0 → 0` e `199 * 24 // 200 = 23`.

**A consequência.** Sementes que caem no mesmo pixel de feature viram **candidatas
duplicadas**, em vez de se fundirem numa só como `project` faria. É um estado que o
próprio `Seeds.project` admite — o docstring dele diz "*duplicates and all*"
(`spifil/types.py:108`) para o caso `stride <= 1` — e que o seletor por diversidade
descarta depois: o passo guloso maximiza distância ao que já foi escolhido, e uma
duplicata tem distância zero.

**Por que `Cropped` reaplica `Resize` + `CenterCrop` antes do LAB.** Porque os superpixels
têm que ser calculados em cima de **exatamente a imagem que o encoder enxerga**
(`spifil_grow.py:146-166`). Se o recorte viesse depois, as sementes ficariam em cima de
pixels que foram cortados fora — coordenadas válidas apontando para conteúdo que o
encoder nunca viu. A máscara usa interpolação `NEAREST` pelo motivo de sempre: máscara é
rótulo, e interpolar rótulo inventa classe.

---

## 10. Por que o crescimento é uma `FitStrategy` nova, e não um patch no SPiFiL

**Porque o SPiFiL já expõe essa costura, e a política de parada é deste experimento, não
da biblioteca.**

O SPiFiL injeta cada etapa como um `Protocol` com um default de uma linha. O default do
laço de camadas é literalmente isto:

```python
class SequentialStrategy:
    def run(self, learn: Learner) -> None:
        for layer in range(1, learn.n_layers + 1):
            learn.fit_layer(layer)
```
(`spifil/strategies.py:30-41`; o docstring do módulo diz que "as variantes interessantes
de pesquisa são *ordenações*, não pipelines novos")

`GrowOneLayer` mora em [`scripts/spifil_grow.py:118-143`](../scripts/spifil_grow.py),
**neste** repositório, e é injetada por `strategy=`. O que ela acrescenta ao laço é o
relatório `N/D` e a recusa da seção 7. **O repositório do SPiFiL não é modificado em
nada.**

Cada rodada monta um `Learner` cujo `ArchSpec` carrega exatamente **uma** `LayerSpec`
(`spifil_grow.py:385-393`) — uma camada por rodada, nunca duas. Por isso "a camada nova" é
simplesmente `learn.n_layers`, sem contabilidade extra.

---

## 11. Por que não há adapter e não há `after_graft`

**Porque nada a jusante exige uma contagem de canais.** No graft de ResNet o `layer3`
esperava exatamente 128 canais de entrada, o que obrigava a variável `need` e a ponte
1×1 quando o SPiFiL entregava menos. Aqui a variável `need` e o adapter simplesmente
desaparecem: o decoder é construído *depois*, do JSON.

Em compensação o decoder tem que ser dimensionado pelo que o SPiFiL **produziu**, lido do
modelo, nunca pelo que foi **pedido**:

```python
block = learn.model.block(learn.n_layers)
got = int(block.out_channels)
```
([`spifil_grow.py:404-410`](../scripts/spifil_grow.py))

O `UniformAllocator` dá `out_channels // n_classes` filtros por classe e joga o resto
fora. Em eggs, que tem **9 classes**, os **48 canais pedidos viraram 45 filtros** — número
medido na rodada 1, e o script imprime a diferença.

Um decoder dimensionado pelo valor pedido **não falharia**. Ele erraria calado.

---

## 12. As três mudanças que o repositório teve que absorver

Nenhuma delas é cosmética: sem elas o protocolo roda e dá número errado.

### 12.1 `_encode_pooled` não pode mais andar até `conv3` na mão

[`src/utils/evaluate.py:254-276`](../src/utils/evaluate.py) percorria
`conv1 → conv2 → conv3` escrito à mão. Agora percorre `model.n_layers` camadas, com
default 3:

```python
for n in range(1, getattr(model, "n_layers", 3) + 1):
    out = getattr(model, f"conv{n}")(out)
```

Sem isso a sonda κ mediria um embedding de 3 camadas enquanto um modelo de 4 camadas
treinava — **em silêncio**, porque a forma continuaria válida.

### 12.2 `--init-ckpt` não pode usar `load_from_checkpoint`

Um modelo crescido tem chaves de encoder novas **e**, o que é pior, **todos os índices de
bloco do decoder deslocados de um**.

`ResNetDecoder` monta `self.blocks` do mais profundo para o mais raso —
`for n in range(n_layers, 0, -1)`
([autoencoder_resnet.py:111-116](../src/models/autoencoder_resnet.py)) — então o bloco 0 é
sempre o up-block do gargalo. Crescer de 3 para 4 faz o antigo `decoder.blocks.0` virar o
novo `decoder.blocks.1`.

Um `strict=False` puro, baseado em nome, bateria em incompatibilidade de shape em **todos**
os blocos do decoder e jogaria fora, sem erro, o decoder treinado inteiro. O carregador
atual desloca os índices por `delta` e reporta missing/unexpected
([`autoencoder_flim_module.py:872-895`](../src/modules/autoencoder_flim_module.py)):

```python
pre = "model.decoder.blocks."
delta = module.model.encoder.n_layers - len({k[len(pre):].split(".")[0] for k in state if k.startswith(pre)})
```

`to_image` mantém o nome porque a largura dele é `channels[1]`, que o crescimento não
toca. E o carregamento é **só pesos**, de propósito: `trainer.fit(ckpt_path=...)`
restauraria também otimizador, scheduler e contador de época, e a rodada nova "convergiria"
na cauda do cosseno sem ter saído do lugar.

### 12.3 `--freeze-encoder` + `--init-ckpt` deixaram de ser mutuamente exclusivos

Antes o parser exigia escolher um dos dois. **O estágio 3 é os dois ao mesmo tempo**, então
a checagem saiu e `stage` virou 1 / 2 / 3
([`autoencoder_flim_module.py:809`](../src/modules/autoencoder_flim_module.py)):

```python
stage = 2 if not args.freeze_encoder else (3 if args.init_ckpt else 1)
```

### 12.4 O branch `dataset == "protozoan"` saiu

Ele trocava o JSON lido do disco pelo dicionário fixo `PROTOZOAN_FLIM_ARCH`
(models.py:611). Isso teria substituído **silenciosamente** uma arquitetura crescida de 4
camadas por uma de 3.

A remoção é preservadora de comportamento, e isso foi verificado antes de tirar: o
dicionário fixo e o JSON do protozoan em disco divergem apenas em `nkernels_per_marker` /
`nkernels_per_image` — campos que nenhum builder lê — e `build_encoder_from_arch` produz
um encoder string-idêntico a partir de qualquer um dos dois. Além disso
`override_arch_channels` sobrescreve `noutput_channels` logo em seguida com o que os
`conv{n}-bias.txt` dizem
([`autoencoder_flim_module.py:195-203`](../src/modules/autoencoder_flim_module.py)).

---

## 13. O que isto NÃO faz

**Não há conexão residual dentro da camada crescida.** O bloco novo é
`Conv2d → ReLU → MaxPool opcional`, sem skip (`build_encoder_from_arch`, models.py:113).
Crescer o encoder é portanto **mudança de arquitetura**, não só mudança de peso — e κ
comparado entre rodadas tem que ser reportado como tal. Uma rodada que ganha κ ganhou
com uma camada a mais, não com os mesmos parâmetros melhor ajustados.

**Não há pooling na camada nova, por default.** `--pool-stride 1`
(`spifil_grow.py:358-359`), o que faz o tipo virar `"none"` no JSON e o upsample
correspondente do decoder virar fator 1, um no-op. Dois motivos: o grid já é 24×24, e todo
pooling ao mesmo tempo destrói detalhe espacial **e** funde sementes — derrubando `N`
exatamente quando `D` está subindo (seção 7). `--pool-stride 2` está disponível e o
decoder o espelha corretamente; a verificação (a) de `check_spifil_growth.py` confirma a
forma `(2, 3, 200, 200)` nas duas variantes e também na arquitetura de 3 camadas não
crescida, para provar que crescer não mexeu na garantia.

**Não treina nada por backprop no `grow`.** O `grow` é um passo sem gradiente; quem treina
é o estágio 3 logo depois.

**Não mede nada além de κ.** `val_recon_loss` continua sendo logada, mas não seleciona
mais checkpoint em estágio nenhum.

---

## 14. Ambiente — os dois repositórios, reconciliados

Os dois repositórios documentam setups próprios. Eles precisam ser **reconciliados**, não
fundidos em silêncio.

- **Repositório de treino** (`Scalable_Hybrid_FLIM`): conda env `scalable_FLIM` em
  `/dados/home/moliveira/miniforge3/envs/scalable_FLIM`, Python 3.11, torch 2.2 + cu121 —
  conforme [`INSTALL.md`](../INSTALL.md), [`environment.yml`](../environment.yml) e
  [`setup_env.sh`](../setup_env.sh).
- **SPiFiL** (`/dados/home/moliveira/SPiFiL`): o `README.md` documenta
  `pip install -e .` **ou** `uv sync` a partir de um clone; `pyproject.toml` pede
  `requires-python >=3.10`.

**A reconciliação em vigor:** o `spifil` está instalado **editable dentro do env conda**
(`pip install -e /dados/home/moliveira/SPiFiL`) e já é importável de lá. Verificado nesta
sessão:

```
$ /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -c "import spifil; print(spifil.__file__)"
/dados/home/moliveira/SPiFiL/src/spifil/__init__.py     # spifil 1.0.0, Python 3.11.15
```

Nenhum caminho novo é recomendado aqui. Este é o que está em uso.

### Conflitos, ditos na cara

**(a) `README_UV.md` não existe mais.** Foi fundido no `README.md` deste repositório no
commit `c8ff208` *"docs(readme): funde o setup com uv no README unico"*. Qualquer coisa
que ainda aponte para ele está velha. (O SPiFiL nunca teve um `README_UV.md`.)

**(b) O backend DISF do SPiFiL vem de um wheel vendorizado `cp312`**
(`vendor/pyift-0.1-cp312-cp312-linux_x86_64.whl`), que sob Python 3.11 não instala —
`[tool.uv.sources]` e o marcador `python_version == '3.12'` cuidam para que ele
simplesmente não seja escolhido. **Mas há uma nuance medida nesta sessão:** este env já
tem um `pyift` **cp311**, instalado pelo passo 3 do `INSTALL.md` deste repositório (wheel
do Google Drive), e `import pyift.pyift` funciona, com `DISF`, `Circular`,
`CreateMImageFromNumPy` e `CreateImageFromNumPy` todos presentes. Ou seja: o
`require_pyift()` do SPiFiL provavelmente passaria aqui. Isso **não** foi testado ponta a
ponta, e não muda nada na prática — `build_layer0` instancia `SLIC()` fixo
(`spifil_grow.py:279`), então o crescimento usa SLIC de qualquer jeito. Vale saber antes
de alguém concluir que DISF é impossível neste env.

**(c) O SPiFiL usa *dependency groups* do PEP 735.** `uv sync --group disf` funciona;
`pip install "spifil[disf]"` **não existe** — não é um extra, é um group, e o comentário
no `pyproject.toml` explica por quê (um extra publicado resolveria `pyift` do PyPI, que é
um projeto sem relação com o mesmo nome).

**(d) Os passos manuais de conda do `INSTALL.md` nunca rodam
`pip install -r requirements.txt`**, ao contrário do `setup_env.sh`, que faz
`uv pip install --python "$ENV_PY" -r requirements.txt` (`setup_env.sh:92`). Quem seguir a
receita manual fica com o env incompleto.

---

## 15. Comandos

Todos a partir da raiz do repositório, com o python do env por caminho absoluto. **Sem
wrapper `.sh` e sem variável de ambiente de espécie alguma** — tudo o que o processo filho
precisa entra como argumento explícito, para que `--dry-run` mostre o plano inteiro.

Configuração usada abaixo: `eggs`, split 1, percentage 50, arquitetura e pesos de partida
em `data/to_mateus/model/ch24_32_48_a0.5_f5/eggs/train1/`, tudo escrito em
`artifacts/spifil_growth/eggs_s1_p50`.

### Estágio 1 — encoder congelado, só o decoder aprende

```bash
/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.autoencoder_flim_module --dataset eggs --split 1 --percentage 50 --arch-json data/to_mateus/model/ch24_32_48_a0.5_f5/eggs/train1/architecture.json --flim-weights-path data/to_mateus/model/ch24_32_48_a0.5_f5/eggs/train1/models --run-name stage1 --output-dir artifacts/spifil_growth/eggs_s1_p50/stage1 --freeze-encoder
```

Escreve `artifacts/spifil_growth/eggs_s1_p50/stage1/` com `checkpoints/best_kappa.ckpt`,
`checkpoints/last.ckpt` e `run_metadata.json`. O estágio 2 consome o `best_kappa.ckpt`.

### Estágio 2 — tudo solto, ponta a ponta

```bash
/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.autoencoder_flim_module --dataset eggs --split 1 --percentage 50 --arch-json data/to_mateus/model/ch24_32_48_a0.5_f5/eggs/train1/architecture.json --flim-weights-path data/to_mateus/model/ch24_32_48_a0.5_f5/eggs/train1/models --run-name stage2 --output-dir artifacts/spifil_growth/eggs_s1_p50/stage2 --init-ckpt artifacts/spifil_growth/eggs_s1_p50/stage1/checkpoints/best_kappa.ckpt
```

Escreve `.../stage2/`. O `best_kappa.ckpt` daqui é o backbone de onde a rodada 1 corta a
camada nova, e o κ daqui é a baseline contra a qual a regra de parada compara.

### Estágio 3 — dois comandos: primeiro cresce, depois treina congelado

```bash
/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python scripts/spifil_grow.py --ckpt artifacts/spifil_growth/eggs_s1_p50/stage2/checkpoints/best_kappa.ckpt --arch-json data/to_mateus/model/ch24_32_48_a0.5_f5/eggs/train1/architecture.json --flim-weights-path data/to_mateus/model/ch24_32_48_a0.5_f5/eggs/train1/models --dataset eggs --split 1 --percentage 50 --out artifacts/spifil_growth/eggs_s1_p50/round1_grow
```

Escreve `round1_grow/` com `architecture.json` (`nlayers: 4`), `conv1..conv4-kernels.npy`,
`conv1..conv4-bias.txt` e `spifil_bundle/`. Imprime `[grow] N=... D=...` e a diferença
entre canais pedidos e filtros obtidos. Sai **3** se o orçamento acabou. Para reproduzir os
números medidos, acrescente `--n-images 60 --n-superpixels 100`.

```bash
/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.autoencoder_flim_module --dataset eggs --split 1 --percentage 50 --arch-json artifacts/spifil_growth/eggs_s1_p50/round1_grow/architecture.json --flim-weights-path artifacts/spifil_growth/eggs_s1_p50/round1_grow --run-name round1_stage3 --output-dir artifacts/spifil_growth/eggs_s1_p50/round1_stage3 --freeze-encoder --init-ckpt artifacts/spifil_growth/eggs_s1_p50/stage2/checkpoints/best_kappa.ckpt
```

Note que `--arch-json` e `--flim-weights-path` agora apontam para `round1_grow/`, enquanto
`--init-ckpt` ainda aponta para o `stage2` — os pesos velhos vêm do checkpoint, a camada
nova vem do diretório.

### Estágio 4 — tudo solto, modelo crescido

```bash
/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.modules.autoencoder_flim_module --dataset eggs --split 1 --percentage 50 --arch-json artifacts/spifil_growth/eggs_s1_p50/round1_grow/architecture.json --flim-weights-path artifacts/spifil_growth/eggs_s1_p50/round1_grow --run-name round1_stage4 --output-dir artifacts/spifil_growth/eggs_s1_p50/round1_stage4 --init-ckpt artifacts/spifil_growth/eggs_s1_p50/round1_stage3/checkpoints/best_kappa.ckpt
```

O `best_kappa.ckpt` daqui é o backbone do `round2_grow`, e o κ daqui é o que a regra de
parada compara.

### O protocolo inteiro, num comando só

```bash
/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python scripts/spifil_growth_loop.py --dataset eggs --split 1 --percentage 50 --arch-json data/to_mateus/model/ch24_32_48_a0.5_f5/eggs/train1/architecture.json --flim-weights-path data/to_mateus/model/ch24_32_48_a0.5_f5/eggs/train1/models --work-dir artifacts/spifil_growth/eggs_s1_p50 --max-rounds 2
```

Acrescente `--dry-run` para ver o plano completo (8 comandos com `--max-rounds 2`) sem
executar nada. Os comandos das seções acima **foram copiados dessa saída**, não escritos à
mão.

### A verificação

```bash
/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python scripts/check_spifil_growth.py
```

Sem pytest, sem `tests/`, sem dependência nova. Uma linha por verificação, morre no
primeiro assert que falhar.

---

## 16. Mapa de fontes

| O que | Onde |
|---|---|
| Uma rodada de crescimento, `GrowOneLayer`, `rescale`, `flim_kernels` | [`scripts/spifil_grow.py`](../scripts/spifil_grow.py) |
| O laço, `should_stop`, os subprocessos | [`scripts/spifil_growth_loop.py`](../scripts/spifil_growth_loop.py) |
| As três verificações runnable | [`scripts/check_spifil_growth.py`](../scripts/check_spifil_growth.py) |
| Estágios, `monitor_spec`, o load de `--init-ckpt` | [`src/modules/autoencoder_flim_module.py`](../src/modules/autoencoder_flim_module.py) |
| `build_encoder_from_arch`, `shift_weights`, `load_FLIM_encoder` | [`src/models/models.py`](../src/models/models.py) |
| `ResNetDecoder`, o espelho do JSON, o `out_size` | [`src/models/autoencoder_resnet.py`](../src/models/autoencoder_resnet.py) |
| A sonda κ, `_encode_pooled` | [`src/utils/evaluate.py`](../src/utils/evaluate.py) |
| O SPiFiL em si, e o graft do qual este script descende | [`docs/spifil_grafting.md`](spifil_grafting.md), [`scripts/spifil_resnet_graft.py`](../scripts/spifil_resnet_graft.py) |
| `FitStrategy` / `SequentialStrategy`, a costura usada | `spifil/strategies.py` |
| `Seeds.project`, "duplicates and all" | `spifil/types.py:108` |
