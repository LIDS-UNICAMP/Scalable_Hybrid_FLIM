# Comparativo κ — braços `lab` × `lab_flat`, estágio 1 (encoder congelado) × estágio 2 (autoencoder ajustado)

**Métrica reportada: κ (Cohen's kappa), coluna `kappa` dos CSVs. Maior é melhor.**

κ é a métrica escolhida porque é a única diretamente comparável entre as séries deste
repositório — `acc` é macro em uns CSVs e crua em outros, e `f1` é macro aqui e weighted no CSV
externo ([`src/evaluate/README.md`](../src/evaluate/README.md#L240-L241)).

Todos os valores de κ são **transcritos verbatim** dos CSVs: sem média, sem reagrupamento, sem
re-arredondamento. Nenhuma célula ausente foi preenchida por inferência. **A única coluna
calculada é a `Δκ`** (`κ_estágio2 − κ_estágio1`, 4 casas), e está marcada como tal.

---

## As duas fontes — e por que elas trazem as quatro variantes

Os dois CSVs foram gerados por:

```bash
python -m src.evaluate.eval_autoencoder --weights lab_flat --out results/eval_autoencoder_lab_flat.csv
python -m src.evaluate.eval_autoencoder --weights lab      --out results/eval_autoencoder_lab.csv
# rodada complementar de 2026-08-11, que fechou o bloco protozoan/pct=50 do braço achatado:
python -m src.evaluate.eval_autoencoder --weights lab_flat --datasets protozoan --percentages 50 \
  --stages 1 2 --out results/eval_autoencoder_lab_flat_protozoan_pct50.csv
```

Sem `--stages`, então **cada arquivo traz os dois estágios**: o default do flag é `None` e
`_runs()` aceita toda linha `status == "ok"` do manifesto daquele braço
([`eval_autoencoder.py`](../src/evaluate/eval_autoencoder.py#L124-L138)) — os runs
`..._stage1_frozen_*` e `..._stage2_fine_tune_*`. Logo as quatro variantes saem de dois arquivos:

| # | variante | `stage` | `dim` | CSV de origem |
|---|---|---|---|---|
| 1 | Baseline `lab_flat` — encoder **congelado**, conv3 achatado | 1 | 27.648 | `results/eval_autoencoder_lab_flat.csv` |
| 2 | Baseline `lab` — encoder **congelado**, GAP | 1 | 48 | `results/eval_autoencoder_lab.csv` |
| 3 | Autoencoder `lab_flat` — encoder **ajustado** (fine-tune) | 2 | 27.648 | `results/eval_autoencoder_lab_flat.csv` |
| 4 | Autoencoder `lab` — encoder **ajustado** (fine-tune) | 2 | 48 | `results/eval_autoencoder_lab.csv` |

**Por que o estágio 1 é o baseline:** ele congela o encoder, então o SVM enxerga exatamente as
features do FLIM cru que inicializaram o autoencoder. Isso não é interpretação — está verificado
contra o avaliador independente do FLIM cru: nas 6 células em que os dois coexistem (`protozoan`
× 3 splits × 2 pct), o κ do estágio 1 `lab` e o κ de
`artifacts/plots/comparacao_flim_protocolo_original/eval_48d_norm_off.csv` são **bit-idênticos
nos 16 dígitos**, com os mesmos `n_train` (235 e 2390). Detalhe no
[apêndice](#apêndice--a-conferência-contra-o-avaliador-do-flim-cru).

Por isso **`Δκ` = quanto o autoencoder acrescentou** sobre o FLIM cru, com a mesma entrada, o
mesmo split e o mesmo SVM.

**Recorte:** percentuais `5` e `50`, os únicos presentes nos dois CSVs. Linhas em
`dataset × split` porque essa é a granularidade que os CSVs têm — eles não guardam nenhuma
linha agregada por dataset, e agregar exigiria calcular médias.

**Cobertura:** os dois braços têm as 36 linhas esperadas (3 datasets × 3 splits × 2 pct × 2
estágios). Nenhuma célula vazia — as 5 linhas de `protozoan`/pct = 50 que faltavam ao braço
achatado foram avaliadas em 2026-08-11 pela rodada complementar acima, ver
[Como o bloco faltante foi fechado](#como-o-bloco-faltante-foi-fechado).

**Sanidade do solver:** todas as 72 linhas têm `fit_status = 0`, `max_iter = -1` (solver
convergido) e `imagenet_norm = False` (LAB cru). Nenhum κ citado veio de ajuste truncado.

---

## Tabelas

**Negrito = melhor κ entre os dois braços, dentro de cada abordagem** — um negrito entre as duas
colunas de estágio 1 (baseline) e outro entre as duas de estágio 2 (autoencoder). Onde só um dos
braços existe, ele é trivialmente o melhor daquela abordagem e sai em negrito.

### Tabela 1 — κ, percentage = 5%

Fontes: coluna `kappa` de `results/eval_autoencoder_lab_flat.csv` e
`results/eval_autoencoder_lab.csv`, linhas `stage == 1` e `stage == 2`.

| dataset | split | s1 `lab_flat` | s1 `lab` | s2 `lab_flat` | s2 `lab` | Δκ `lab_flat` | Δκ `lab` |
|---|---|---|---|---|---|---|---|
| eggs | 1 | **0.6129509210586548** | 0.4545564651489258 | **0.6806322336196899** | 0.5082675218582153 | +0.0677 | +0.0537 |
| eggs | 2 | **0.6762145757675171** | 0.6376086473464966 | **0.7475530505180359** | 0.6419921517372131 | +0.0713 | +0.0044 |
| eggs | 3 | **0.5630757212638855** | 0.5219123959541321 | **0.6875293254852295** | 0.5880502462387085 | +0.1245 | +0.0661 |
| larvae | 1 | 0.4662896394729614 | **0.6386788487434387** | 0.4941027760505676 | **0.6979770660400391** | +0.0278 | +0.0593 |
| larvae | 2 | 0.5595971941947937 | **0.8136687874794006** | 0.5574283003807068 | **0.795947253704071** | −0.0022 | −0.0177 |
| larvae | 3 | 0.6684563159942627 | **0.6959288120269775** | 0.6648523807525635 | **0.7824781537055969** | −0.0036 | +0.0865 |
| protozoan | 1 | **0.5830865502357483** | 0.5528641939163208 | **0.6500614881515503** | 0.5645862817764282 | +0.0670 | +0.0117 |
| protozoan | 2 | **0.6133423447608948** | 0.5557061433792114 | **0.6893922090530396** | 0.609352707862854 | +0.0760 | +0.0536 |
| protozoan | 3 | **0.6533938646316528** | 0.5615551471710205 | **0.7131029367446899** | 0.5926387310028076 | +0.0597 | +0.0311 |

### Tabela 2 — κ, percentage = 50%

Mesmas duas fontes.

| dataset | split | s1 `lab_flat` | s1 `lab` | s2 `lab_flat` | s2 `lab` | Δκ `lab_flat` | Δκ `lab` |
|---|---|---|---|---|---|---|---|
| eggs | 1 | **0.8736264109611511** | 0.7359811663627625 | **0.9157859086990356** | 0.8081185817718506 | +0.0422 | +0.0721 |
| eggs | 2 | **0.8375082612037659** | 0.7795658707618713 | **0.9226958751678467** | 0.8227815628051758 | +0.0852 | +0.0432 |
| eggs | 3 | **0.8445422649383545** | 0.7224652767181396 | **0.9151576161384583** | 0.8045452833175659 | +0.0706 | +0.0821 |
| larvae | 1 | 0.8055294752120972 | **0.8598628044128418** | 0.8669687509536743 | **0.9265109300613403** | +0.0614 | +0.0666 |
| larvae | 2 | 0.8062162399291992 | **0.8543435335159302** | 0.8902029991149902 | **0.9325987100601196** | +0.0840 | +0.0783 |
| larvae | 3 | 0.824020504951477 | **0.8593077063560486** | 0.8706033229827881 | **0.93218994140625** | +0.0466 | +0.0729 |
| protozoan | 1 | **0.8219481706619263** | 0.6978076696395874 | **0.8765825033187866** | 0.7860798835754395 | +0.0546 | +0.0883 |
| protozoan | 2 | **0.7969300746917725** | 0.689412534236908 | **0.8635572195053101** | 0.7851977944374084 | +0.0666 | +0.0958 |
| protozoan | 3 | **0.805691659450531** | 0.6979272365570068 | **0.8688174486160278** | 0.7755645513534546 | +0.0631 | +0.0776 |

---

## Quanto melhorou depois do autoencoder

Contagem sobre as células em que estágio 1 e estágio 2 coexistem (nenhuma média envolvida):

| braço | pct | células que sobem | Δκ mínimo | Δκ máximo |
|---|---|---|---|---|
| `lab` (48-d) | 5 | **8 de 9** | −0.0177 | +0.0865 |
| `lab` (48-d) | 50 | **9 de 9** | +0.0432 | +0.0958 |
| `lab_flat` (27.648-d) | 5 | **7 de 9** | −0.0036 | +0.1245 |
| `lab_flat` (27.648-d) | 50 | **9 de 9** | +0.0422 | +0.0852 |

Ou seja: **o autoencoder melhora 33 das 36 células.** Em pct = 50 o ganho é uniforme
— 18 de 18 células, sempre entre +0.042 e +0.096 de κ. Em pct = 5 é maior no melhor caso
(+0.1245, eggs split 3 no braço achatado) mas irregular: as 3 únicas quedas do relatório estão
todas aí, e todas em `larvae` (−0.0177 no `lab` split 2; −0.0022 e −0.0036 no `lab_flat` splits
2 e 3, ambas na terceira casa decimal).

Sobre qual braço é melhor, que é a leitura das colunas em negrito: **não há vencedor único.**
`lab_flat` ganha em `eggs` e `protozoan` (12 de 12 células em cada, nos dois estágios e nos dois
percentuais), `lab` ganha em `larvae` (12 de 12). A ordenação entre braços é a mesma no estágio 1
e no estágio 2 nas 18 células — o autoencoder desloca os dois braços para cima sem inverter quem
está na frente. Com o bloco `protozoan`/pct = 50 agora preenchido, a partição por dataset é
limpa: **36 de 36 linhas** seguem a divisão eggs+protozoan → `lab_flat`, larvae → `lab`.

Um alerta de interpretação: estes números são a **sonda SVM post-hoc sobre o checkpoint salvo**,
não a curva de κ registrada durante o treino. As duas coisas discordam nos runs deste braço — o
κ de treino pica no fim do warmup —, então não compare uma linha daqui com `stage{N}/svm_kappa`
do wandb ([`eval_autoencoder.py`](../src/evaluate/eval_autoencoder.py#L62)).

---

## Como o bloco faltante foi fechado

Até 2026-08-11 este relatório trazia 5 células `-` — `protozoan`/pct = 50 no braço `lab_flat`,
splits 1 e 2 (estágios 1 e 2) e split 3 (estágio 2). **A causa não era falta de treino.** Os 6
runs correspondentes já existiam completos: `status = ok` em
`artifacts/autoencoder_resnet_init_flim/run_manifest.csv` e checkpoints no disco
(`best_recon.ckpt` nos estágios 1, `best_kappa.ckpt` nos estágios 2). O que faltou foi a passada
do **avaliador**.

A rodada original de `--weights lab_flat` gravou 31 linhas e parou às 19:00 de 2026-08-10, logo
depois da primeira célula cara do arquivo (`protozoan`/split 3/pct 50/estágio 1, `fit_s = 50.1`
contra ~1.7 s nas células de pct = 5). As 31 linhas são **prefixo exato** da ordem do manifesto,
e o laço de avaliação não tem `try`/`except`
([`eval_autoencoder.py`](../src/evaluate/eval_autoencoder.py#L230-L279)) — nenhuma linha pode ter
falhado e sido pulada em silêncio. Foi corte do processo, não erro de célula: as 5 posições
seguintes do manifesto são exatamente as 5 que faltavam, na ordem.

O bloco foi reavaliado com:

```bash
OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE CUDA_VISIBLE_DEVICES=1 \
  conda run -n scalable_FLIM --no-capture-output \
  python -m src.evaluate.eval_autoencoder --weights lab_flat \
    --datasets protozoan --percentages 50 --stages 1 2 \
    --out results/eval_autoencoder_lab_flat_protozoan_pct50.csv
```

⚠️ Saída em **arquivo separado**: o script abre o `--out` em modo `"w"`
([`eval_autoencoder.py`](../src/evaluate/eval_autoencoder.py#L226)) e reusar o caminho principal
apagaria as 31 linhas já obtidas. As 5 linhas novas foram então anexadas a
`results/eval_autoencoder_lab_flat.csv`, que passou a ter as 36.

**Conferência de reprodutibilidade.** O filtro seleciona 6 runs, não 5: `protozoan`/split 3/pct
50/estágio 1 já estava no CSV e foi reavaliado junto. Os dois valores são idênticos nos 16
dígitos — `0.805691659450531`, com o mesmo `n_sv = 776` e `fit_status = 0`. A re-rodada
reproduz o pipeline bit a bit; as 5 células novas entram na mesma escala das 31 antigas.

---

## Apêndice — a conferência contra o avaliador do FLIM cru

Um terceiro CSV, `artifacts/plots/comparacao_flim_protocolo_original/eval_48d_norm_off.csv`
(método `SVM_FLIM_48d_labcru_conv`), avalia o **FLIM cru** com GAP 48-d por um caminho de código
independente. Ele cobre só `protozoan`, mas nos 6 pontos em comum bate dígito por dígito com o
estágio 1 `lab`:

| dataset | split | pct | FLIM cru (CSV externo) | Estágio 1 `lab` |
|---|---|---|---|---|
| protozoan | 1 | 5 | 0.5528641939163208 | 0.5528641939163208 |
| protozoan | 2 | 5 | 0.5557061433792114 | 0.5557061433792114 |
| protozoan | 3 | 5 | 0.5615551471710205 | 0.5615551471710205 |
| protozoan | 1 | 50 | 0.6978076696395874 | 0.6978076696395874 |
| protozoan | 2 | 50 | 0.689412534236908 | 0.689412534236908 |
| protozoan | 3 | 50 | 0.6979272365570068 | 0.6979272365570068 |

É o que valida o desenho deste relatório: o estágio 1 **é** o FLIM cru, então `Δκ` mede o
autoencoder e não uma diferença de protocolo. Serve também de verificação de que os dois
avaliadores compartilham dados, SVM e métrica, como a docstring de
[`eval_autoencoder.py`](../src/evaluate/eval_autoencoder.py#L25-L33) afirma.

Por que esse CSV externo não virou coluna das tabelas: ele cobre apenas `protozoan`
(`eval_avg_pooling_48d.py` roda um dataset por vez e abre o CSV em modo `"w"` — a última rodada
sobrescreveu eggs e larvae, [`src/evaluate/README.md`](../src/evaluate/README.md#L121)), e o
equivalente achatado 27.648-d nunca chegou a existir como CSV — `eval_svm_flim_flatten.py` só
escreve no fim da rodada e a rodada foi interrompida
([`src/evaluate/README.md`](../src/evaluate/README.md#L160)). Esses números do braço achatado
existem como **JSON** em `artifacts/plots/comparacao_flim_protocolo_original/dados_brutos/`
(`g1/pct*.json` = protozoan, `g2_eggs.json`, `g2_larvae.json`, todos com `"dim": 27648`); não
foram transcritos porque o pedido restringe o relatório aos CSVs. Como o estágio 1 já cobre o
papel de baseline nos dois braços, nada falta às tabelas por causa disso.
