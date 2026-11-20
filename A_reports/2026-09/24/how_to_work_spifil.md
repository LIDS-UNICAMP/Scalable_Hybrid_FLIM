# Como funciona o SPiFiL — explicação simples, com as referências no código

> Nota de escopo: isto é a mecânica do crescimento camada a camada. Não interpreta
> resultado de treino. Complementa [spifil_growth.md](spifil_growth.md), que descreve o
> protocolo de rodadas; aqui o foco é **de onde saem os pesos da camada nova**.

---

## A ideia, em uma frase

**Um filtro SPiFiL não é aprendido — é um pedaço recortado do próprio mapa de ativações.**

Nenhum gradiente, nenhuma época. Você passa as imagens pelo encoder que já tem, olha o mapa
de features que sai, recorta janelinhas em lugares escolhidos, e essas janelinhas *viram* os
pesos da camada nova.

---

## O filtro é um carimbo

Pense num carimbo. Você recorta um pedacinho característico — a borda de um ovo de helminto,
digamos — e a partir daí o filtro acende onde a imagem se parece com aquele pedaço.

Convolução é produto interno: `w·x + b` mede o quanto a janela atual `x` se parece com o
carimbo `w`. Se o carimbo *é* um exemplar real, ativação alta significa "achei outro parecido
com aquele".

É por isso que o encoder tem que estar treinado antes. `scripts/spifil_grow.py:42-48` diz
direto:

> Um filtro SPiFiL e um patch recortado das ativacoes. Recortar da imagem crua daria a
> primeira camada de novo. O que se quer e a camada que vem DEPOIS do que o modelo ja
> aprendeu.

---

## A matemática são duas linhas

Em `~/SPiFiL/src/spifil/nn/builder.py:120-130` (o pacote `spifil`, repo irmão):

```python
zscored = (feats - mean) / stdev
zscored = zscored / ||zscored||         # norma unitária
kernels = zscored / stdev               # ← o peso
bias    = -(kernels * mean).sum(dim=1)  # ← o viés
```

O `bias` negativo cancela a média: `w·x + b` = `w·(x − mean)`. Ou seja, o filtro mede
similaridade **com os dados centrados**. Isso é um matched filter clássico, não uma rede
treinando.

---

## Como escolhe *onde* recortar

Recortar em qualquer lugar daria filtros redundantes. O SPiFiL:

1. **Segmenta em superpixels** (SLIC) — regiões homogêneas
2. **Pega o medoide** de cada região — o pixel mais central, virando a "semente"
3. **Recorta** uma janela 3×3 em torno de cada semente, com todos os 48 canais →
   **432 números por patch**
4. **Rankeia por classe** (critério de Fisher) e seleciona os melhores
5. `UniformAllocator` divide igual entre as classes: pediu 48 filtros com 9 classes →
   `48 // 9 * 9 = ` **45**

Esse 45 é o número que aparece em `noutput_channels` nos `round1_grow/architecture.json`.

Cadeia no pacote `spifil`: `patches.py:38` (`extract_patches`) → `learner.py:349`
(`per_class_ranks`) → `learner.py:445-447` (allocator + selector) → `nn/builder.py:82`
(`build_filters`).

---

## `(feat)` vs `(img)` — a diferença é só onde o SLIC roda

A hipótese natural seria "um recorta da imagem, outro das features". **Não é isso.** Os dois
recortam das features. Prova em `scripts/spifil_grow.py:437` — a linha está **fora** do `if`:

```python
features = encoder_features(...)   # ← sempre, nos dois braços
if in_feature:
    seeds, regions = seeds_in_features(features, ...)      # SLIC na grade 24×24
else:
    prepared = preparation.prepare(data, SLIC(), ..., color=LabNorm())  # SLIC na imagem LAB
    seeds = [rescale(image.seeds, grid) for image in prepared]          # reprojeta 200→24
```

| | `(img)` — default, do artigo | `(feat)` |
|---|---|---|
| valores do filtro | features 48×24×24 | features 48×24×24 |
| onde o SLIC segmenta | imagem LAB 200×200 | grade de features 24×24 |
| sementes | reprojetadas 200→24 (`rescale`, `:276`) | já nascem em 24×24 |
| normalização p/ o SLIC | `LabNorm()` do SPiFiL | `band / max(\|band\|)` por imagem (`:407`) |
| teto de superpixels | não | `area // MIN_POSITIONS_PER_SUPERPIXEL` (`:415`) |

`--spifil-in-image` é flag decorativa: `args.spifil_in_image` nunca é lido —
`in_image` é o ramo `else` de `not in_feature`.

Docstring correspondente: `scripts/spifil_grow.py:76-78`.

---

## A sequência de uma rodada

Do laço `scripts/spifil_growth_loop.py:299` (`run_arm`) — cada passo é um **processo
separado** do SO:

```
stage1 → stage2 → grow₁ → stage3₁ → stage4₁ → grow₂ → ...
```

O `grow` (`scripts/spifil_grow.py:544-590`) **não treina nada**:

1. `AutoEncoderFlimModule.load_from_checkpoint` — carrega o encoder do estágio anterior
2. `Trunk(...)` (`:233`) — congela ele (`eval()` + `requires_grad_(False)`)
3. `build_layer0(...)` (`:424`) — features + superpixels + sementes
4. `learn.fit()` — recorta e seleciona os 45 patches
5. `write_weights(...)` (`:454`) — grava `conv4-kernels.npy` e `conv4-bias.txt`,
   **copiando as camadas 1-3 junto** (um diretório só com `conv4-*` quebraria o laço
   `1..N` do loader)
6. `grown_arch(...)` (`:475`) — incrementa `nlayers` e acrescenta a chave `layer4` no JSON

Depois o `stage3` treina **só a camada nova** — `src/models/models.py:672-684`:

```python
def freeze_encoder(model, except_last: bool = False) -> None:
    encoder = model.encoder if hasattr(model, "encoder") else model
    for param in encoder.parameters():
        param.requires_grad = False              # congela tudo
    if except_last:
        for param in encoder.blocks[f"conv{encoder.n_layers}"].parameters():
            param.requires_grad = True           # solta só a última
```

Chamado por `_freeze_for_growth` (`src/modules/autoencoder_flim_module.py:152-162`), que
também congela os blocos antigos do decoder e o `to_image`. O optimizer só respeita o que já
foi decidido: `filter(lambda p: p.requires_grad, self.parameters())` (`:837`).

E o truque que faz tudo funcionar: o módulo recarrega a camada 4 pelo **mesmo
`load_FLIM_encoder` de sempre** (`models.py:555`), sem saber que ela veio do SPiFiL. O
`Encoder` deixou de ser fixo em 3 camadas — `models.py:158-165` itera `arch["nlayers"]`
sobre um `nn.ModuleDict`.

---

## A restrição que decide se a rodada acontece

Para estimar os filtros é preciso **mais patches que dimensões**: N > D, com
D = 3×3×48 = **432**. Se não fecha, `GrowOneLayer` recusa e sai com código 3
(`scripts/spifil_grow.py:170-182`, `EXIT_BUDGET`):

```python
def run(self, learn: Learner) -> None:
    layer = learn.n_layers
    patches = learn.state[layer - 1].patches
    n, d = len(patches), int(patches.feats.shape[1])
    if n <= d:
        raise BudgetExhausted(...)
    learn.fit_layer(layer)
```

É aí que os dois braços divergem na prática: em 24×24 a máscara cobre ~3,2 % do quadro,
dando mediana de 12 posições úteis por imagem. O braço `(img)` chega a N=12.313; o `(feat)`
empaca em N=185 contra D=432. Documentado em `scripts/spifil_grow.py:379-397`.

---

## Onde ficam os artefatos

`<work-dir>/<dataset>_split<N>_pct<P>/<estágio>` — ex.:

```
artifacts/spifil_growth/g5_in_image/eggs_split1_pct5/
  ├── stage1/         checkpoints/best_kappa.ckpt
  ├── stage2/         checkpoints/best_kappa.ckpt
  ├── round1_grow/    architecture.json, conv{1..4}-*, spifil_bundle/   ← sem checkpoint
  ├── round1_stage3/  checkpoints/best_kappa.ckpt
  └── round1_stage4/  checkpoints/best_kappa.ckpt
```

O `round*_grow/` não tem checkpoint — e é por isso que `src/evaluate/eval_growth_stages.py`
filtra os estágios avaliáveis pela existência de `checkpoints/best_kappa.ckpt`.

### Os arms

O nome do arm é o basename do `--work-dir` (`spifil_growth_loop.py:122`), não uma constante
no código. Configurações em `README.md:796-810`:

| arm | flag distintiva |
|---|---|
| `g5_in_image` | `--spifil-in-image` (o default; a flag só torna explícito) |
| `g5_in_feature` | `--spifil-in-feature` |
| `g5_head` | `--spifil-in-image --head-finetune` |
| `g5_finetune` | `--spifil-in-image --unfrozen-after-stage-two` |
| `grid3`, `grid4` | sem flag de ablação (defaults) |

Os `g5_*` compartilham `--percentages 5 50 --one-per-class --impurities --pool-stride 2
--max-rounds 2`. Diferença visível no `layer4` do `architecture.json`: `g5_*` tem
`pooling.type = "max_pool"` stride 2 (grade 24×24 → 11×11); `grid3`/`grid4` têm
`pooling.type = "none"`.

`grid3` vs `grid4` é geração de código, não flag: `git_sha` `3f69d95` contra `1bf1ba0`.
O `grid3` rodou **antes** de `f2e5acf` ("estagio 3 — cresce uma camada e treina so ela"),
ou seja, antes da política de congelamento existir.

---

## Os scripts — onde ficam e quem chama quem

Sete arquivos. Só **um** você chama à mão para treinar; os outros ou são chamados por ele,
ou rodam depois, sobre o que ele deixou em disco.

| # | Script | Papel | Como roda |
|---|---|---|---|
| 1 | `scripts/spifil_growth_loop.py` | **orquestrador** — o único que você chama para treinar | `python3 scripts/spifil_growth_loop.py …` |
| 2 | `src/modules/autoencoder_flim_module.py` | treinador de **um** estágio | chamado pelo (1) como subprocesso |
| 3 | `scripts/spifil_grow.py` | **corta** a camada nova; não treina | chamado pelo (1) como subprocesso |
| 4 | `scripts/check_spifil_growth.py` | verificação do protocolo | `python scripts/check_spifil_growth.py` |
| 5 | `src/evaluate/eval_growth_stages.py` | SVM de **teste** sobre os encoders | `python -m src.evaluate.eval_growth_stages …` |
| 6 | `tools/plot_continuity_spifil_hybrid.py` | curvas de continuidade | `python tools/plot_continuity_spifil_hybrid.py …` |
| 7 | `tools/heatmap_stages.py` | heatmap de atenção por estágio | `python tools/heatmap_stages.py …` |

Os (2) e (3) **nunca** são chamados à mão num experimento normal — o laço monta a linha de
comando dos dois. Chamar o (3) direto só serve para depurar um corte isolado.

---

### Passo 1 — treinar (o único comando obrigatório)

```bash
tmux new -d -s g5_img "python3 scripts/spifil_growth_loop.py \
  --work-dir artifacts/spifil_growth/g5_in_image \
  --percentages 5 50 --spifil-in-image --one-per-class --impurities --pool-stride 2 \
  --gpus 1 2 3 --max-concurrent-per-gpu 2 --cpus-per-experiment 8 \
  --max-rounds 2 --num-workers 2 --wandb --wandb-project phd_thesis_grid4 \
  2>&1 | tee logs/g5_in_image.log"
```

`--work-dir` é o que define a família: o basename dele vira o nome do experimento
(`_exp()`, `spifil_growth_loop.py:122`) e entra no nome da run do W&B. **Família nova =
work-dir novo**, nada mais.

Antes de rodar de verdade, `--dry-run` imprime o plano inteiro sem executar nada — 144
comandos para uma grade de 18 braços.

### Passo 2 — o que o laço faz sozinho

`main()` (`spifil_growth_loop.py:397`) expande a grade em 18 braços (3 datasets × 3 splits ×
2 percentuais) e joga cada um num `ThreadPoolExecutor`, com GPU fixa do início ao fim.

`run_arm()` (`:302`) é a receita de um braço, e cada passo é **um processo separado do SO**:

| passo | quem monta o comando | o que executa |
|---|---|---|
| `stage1` | `_trainer_cmd` (`:223`) | `python -m src.modules.autoencoder_flim_module --freeze-encoder …` |
| `stage2` | `_trainer_cmd` | idem, sem `--freeze-encoder`, com `--init-ckpt` do stage1 |
| `round1_grow` | `_grow_cmd` (`:259`) | `python scripts/spifil_grow.py --ckpt <stage2> --out <round1_grow>` |
| `round1_stage3` | `_trainer_cmd` | treinador, `--freeze-encoder`, mas com o `--arch-json` do grow |
| `round1_stage4` | `_trainer_cmd` | treinador destravado |

O acoplamento entre grow e estágio seguinte é uma linha só (`:345`): o diretório que o grow
escreveu vira o `--arch-json` e o `--flim-weights-path` do estágio 3. É por isso que o grow
copia as camadas antigas junto — o loader percorre `1..N` e um diretório só com `conv4-*`
quebraria.

**Duas listas governam o repasse de flags:**

- `TRAINER_FWD` (`:169`) — o que chega no treinador
- `GROW_FWD` (`:176`) — o que chega no `spifil_grow.py`

`_fwd()` (`:213`) só emite a flag se o valor não for `None`, e é por isso que todo repasse é
declarado com `default=None` no laço: o default de verdade mora no argparse do filho, para
não haver dois defaults divergindo. **Flag booleana não cabe nessas listas** — `_fwd`
emitiria `--minha-flag False`; booleana precisa de um `if` explícito em `_grow_cmd`
(`:271-280`), que é como `--spifil-in-feature`, `--impurities` e `--random-layer` são
passadas.

Quer acrescentar uma flag nova que só o grow precisa ver? Com valor: dois toques (declarar no
argparse do laço sem default, e pôr o nome em `GROW_FWD`). Booleana: três linhas, com o `if`.

### Passo 3 — avaliar no teste

O laço decide parada por kappa de **validação**; a métrica que vai para tabela é de **teste**,
e sai daqui:

```bash
python -m src.evaluate.eval_growth_stages --pct 5  --family g5_in_image --device cuda:0
python -m src.evaluate.eval_growth_stages --pct 50 --family g5_in_image --device cuda:0
```

`--family` aceita várias de uma vez e não tem `choices` travado: quem valida é o disco
(`eval_growth_stages.py:189`, `main`). Sem `--family`, o default é `grid4` e o CSV sai no
caminho curto de sempre. `--dry-run` lista os trabalhos; `--skip-existing` retoma um CSV
parcial sem recalcular o que já está lá.

Custo: o gargalo é o `fit` do libsvm, não a GPU. Em `pct50`/protozoan um único fit já levou
56 min (27.648 dimensões, `C=100`, `max_iter=-1`).

### Passo 4 — os gráficos

```bash
# trajetória contínua, uma pasta por família, série por época vinda do W&B
python tools/plot_continuity_spifil_hybrid.py --by epoch --fetch --family g5_in_image --pct 5 50

# eixo dos estágios, famílias sobrepostas, do CSV do avaliador (teste)
python tools/plot_continuity_spifil_hybrid.py --by stage --family g5_in_image g5_in_feature --pct 5 50
```

Os dois modos leem fontes **diferentes**: `--by epoch` usa os CSVs por época baixados do W&B
(`--fetch` rebaixa), e pontua na **validação**; `--by stage` usa o CSV do passo 3 e pontua no
**teste**. Não são continuação um do outro.

Saída em `artifacts/analysis_continuidade/<família>/` — no modo `stage` com mais de uma
família, a pasta é a comparação (`<f1>_vs_<f2>/`).

**Rode da raiz do repo.** De dentro de `tools/` o `out_shapes()` acha 6 formas em vez de 14 e
a legenda perde a forma dos estágios crescidos, calada — o `arch_json` deles é caminho
relativo e o `except` engole o erro.

### Passo 5 — verificação (opcional, mas é o que trava o protocolo)

```bash
python scripts/check_spifil_growth.py
```

É a única verificação executável do crescimento. Cobre as quatro peças que não são óbvias por
leitura — a principal sendo que **o decoder crescido continua devolvendo a imagem no tamanho
certo**: se a camada nova entra com um pooling que o `ResNetDecoder` não espelha, a
reconstrução muda de tamanho e o BCE explode. Testa as duas variantes que o grow pode
produzir (pooling `none`/stride 1 e `max_pool`/stride 2) e também a arquitetura de 3 camadas
não crescida, para provar que crescer não quebrou a garantia.

---

## De onde vêm as linhas da tabela

`src/evaluate/eval_growth_stages.py` produz os CSVs:

| linha da tabela | CSV |
|---|---|
| `hybrid_FLIM` | `results/eval_growth_stages_pct{5,50}.csv` (família `grid4`) |
| `hybrid_FLIM (feat)` | `results/eval_growth_stages_g5_in_feature_pct{5,50}.csv` |
| `hybrid_FLIM (img)` | `results/eval_growth_stages_g5_in_image_pct{5,50}.csv` |

Métrica: SVM linear sobre as features achatadas (`EMBED_MODE = "flatten"`,
`channels[-1]*24*24`), `train_svm` de `src/utils/evaluate.py:437` (`C=1e2`, ovo), kappa de
**teste** via `src/metrics/classification.py:31`. O `best_val_svm_kappa` que o laço usa para
decidir parada é de **validação** — os dois não devem bater, e o próprio script avisa disso.
