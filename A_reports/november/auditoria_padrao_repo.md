# Auditoria: o repositório ainda segue o próprio padrão?

Data: 2026-11-08 · host athena · repo `Scalable_Hybrid_FLIM_discovery` · somente leitura (nada editado, nada lançado).
Fontes: `instructions.md`, `MIGRATION.md`, `README.md`, grafo `graphify update .` (7853 nós), três auditorias paralelas + checagem manual.
O grafo respondeu quem importa `fit_svm`/`compute_metrics`; não respondeu onde há `SVC(`/`cohen_kappa_score` soltos nem monkeypatch — isso saiu de grep.

## Resposta curta

**O núcleo segue; a borda nova não.** `core/`, `flim/`, `methods/` e o motor `experiments/ray/` continuam no desenho. O SVM e a métrica são únicos (`core/svm.py:143 fit_svm`, `core/metrics.py:46 compute_metrics`) e todo o código novo passa por eles. Mas o trabalho de outubro/novembro cresceu **dentro de `eval/`** como uma família de scripts que se importam entre si (`spf_grade` ↔ `l1_*` ↔ `alt_encoders` ↔ `saliency`). Neles vivem modelo, camada, leitura de imagem, cache e caminho fixo, tudo junto. E **nada disso está no git**.

Analogia: a casa (core/flim/methods/ray) está arrumada; o puxadinho no quintal (`eval/spf_*`, `eval/l1_*`, `benchmark/`, `saliency/`, `ga_search/`) virou uma segunda casa sem planta.

## O padrão pretendido (em 6 linhas)

| peça | onde mora |
|---|---|
| modelo / encoder | `methods/<metodo>/` (LightningModule) e `flim/` (só por `from flim import build`) |
| blocos, perdas, dados, métricas | `core/blocks`, `core/losses`, `core/data` (`ParasiteDataset`, `transforms`, `loaders`), `core/metrics.py`, `core/svm.py` |
| sondas / avaliação | `eval/` (camada consumidora; flags via `eval.svm.cli_kwargs`, convenção documentada) |
| experimento | YAML `experiments/<metodo>/<nome>.yaml` → `experiments/ray/schema.py` (`RUNNER_ARGS`) → `experiments/ray/runners/{train,growth,eval}.py` → `python -m experiments.ray.launch <yaml>` |
| análise | `analysis/` (folha: ninguém importa) |
| caminho antigo | `src/`, `scripts/`, `tools/` — não estender |

O runner `eval` abre um **subprocesso novo por job** (`experiments/ray/runners/eval.py:569`: `python -m eval.<source> --config ...`). Por isso, qualquer mudança em `eval/spf_grade.py` vale no **próximo job de uma grade que já está rodando**. Daí a regra deste plano: nenhuma mudança de comportamento, e todo nome importado continua importável.

## Desvios encontrados

### Altos

| # | onde | o que quebra |
|---|---|---|
| A1 | `eval/spf_grade.py`, `eval/alt_encoders.py`, `eval/l1_*.py`, `eval/spf_*.py`, `eval/ms_tabela.py`, `benchmark/`, `saliency/`, `flim/{scale_space,filter_diversity,kernel_redundancy}.py`, `core/data/train_aug.py`, **607** `experiments/autoencoder/disc_md_*.yaml` | **não versionados** (`??`; 630 arquivos não rastreados no repo). Sem commit não há "antes" para comparar bit a bit, nem como voltar atrás. É o risco nº 1. |
| A2 | `eval/spf_grade.py` (1499 linhas) | um arquivo, seis papéis: fontes de pontos/saliência (:189-566), aprendizado de filtros (`Ctx` :628, `PatchCtx` :702, `METHODS` :919, `control` :945), camadas/ativações (`Abs`/`ACTS` com GELU :1000-1005, `map_whitener` :979, `folded`/`whitened` :1110-1117, `law_channels` :1137, `chain` :1196, ~135 linhas montando a pilha conv), leituras (`gem` :1015, `covariance` :1019, `POOLS` :1032, `vlad` :1038, `readouts` :1047), SVM (`score_full` :961) e grade/CLI (:1331-1499). Código de **modelo** dentro de uma sonda. |
| A3 | `eval/alt_encoders.py:131 Scattering`, `:181 HebbBlock` (SoftHebb), `:286 PCANet` | três `nn.Module` (encoders) morando em `eval/` em vez de `methods/` ou `flim/`. |
| A4 | `eval/spf_deteccao.py:100-118` | **monkeypatch**: troca `spf_grade.build` e `spf_grade.background_cut` em tempo de execução para espiar o encoder. Mover/renomear `build`, `background_cut` ou `chain` quebra isto **em silêncio**. |
| A5 | `eval/alt_encoders.py:491` | grava `sys.modules["__main__"].Source` para que os `.pt` salvos se des-picklem: o caminho do módulo da classe `Source` (`spf_grade.py:189`) **está gravado nos artefatos**. Mover `Source` invalida caches em disco. |
| A6 | acoplamento entre sondas | `spf_grade.py:139` sobrescreve `l1_sem_rotulo.MAX_WIDTH = 4096` ao ser importado; `spf_grade` importa `_paths`/`_short` **privados** de `experiments.ray.runners.growth`; `saliency/batch.py` ↔ `spf_grade` são circulares (contornado com import local). Consumidores de `spf_grade`: `spf_fundo.py:65`, `spf_deteccao.py:43,46`, `alt_encoders.py:73,75`, `spf_figuras.py:53`, `saliency/sift_density.py:67`, `saliency/batch.py:222`. |

### Médios

| # | onde | o que quebra |
|---|---|---|
| M1 | caminhos fixos `/exps/moliveira`: `spf_grade.py:140` (`SURVIVORS`), `spf_tabela.py:47`, `spf_deteccao.py:51,64,67`, `ms_tabela.py:46`, `l1_fast.py:59` (`SVM_LOCK`), `benchmark/common.py:48,50,51`, `benchmark/lista.py:40-41`, `saliency/sift_density.py:47`; `/dados/home/moliveira`: `ms_tabela.py:202`, `ga_search/run_ga.py:55`, `flim/spifil.py`, `flim/seed_points.py`, `flim/filter_diversity.py`, `core/constants.py`, `analysis/stats/wilcoxon_*.py` (6+6+6+3) | roda só neste host; quebra na Jaci sem editar fonte. O lugar certo é `runner_args`/`output` do YAML ou `core/constants.py` (um só). |
| M2 | segundo carregador de imagens: `eval/l1_fast.py:78-83 load_images` (reusado por `l1_amostras.py:52`, `l1_sem_rotulo.py:62`, `spf_grade.py:125`, `alt_encoders.py:72`); `spf_fundo.py:106` monta outro igual; `spf_grade.py:201-204` lê máscara com `imread`+`lru_cache`, `:317` `cv2.imread`, caches próprios `.npy`/`.pt` (`:306`, `:554-561`); `benchmark/common.py:177`, `saliency/sift_density.py:87,144,147` | leitura e cache reimplementados ao lado de `core/data`. Eles **usam** `ParasiteDataset`, então não é uma cópia completa, mas o cache é paralelo. |
| M3 | `ga_search/spifil_ext.py:32,39,73`, `ga_search/spifil_residual.py:25` | blocos de rede (`ActivationBlock`, `BudgetSequential`, `ResidualBlock`) fora de `flim/`/`core/blocks`. `ga_search/` não tem `__init__.py`. |
| M4 | argparse fora do padrão: `ga_search/{run_ga:30,run_scaling:30,analyze_laws:42,plot_laws:20,progressive_monitor:8}.py`; `sys.argv` cru: `benchmark/lista.py:149`, `spf_deteccao.py:189`, `sift_density.py:171`, `ms_tabela.py:248`, `spf_tabela.py:410`; `saliency/batch.py:323` lê YAML próprio (`experiments/saliency/lote.yaml`) **sem** schema nem launcher | fogem da regra "experimento = YAML + launcher". |
| M5 | 607 `disc_md_*.yaml` (430 growth, 177 eval), copiados à mão | repetem ~20 chaves de `runner_args` + o mesmo bloco `overrides`; o par `amostra_banco_aid`/`_flowers` difere só em nome, dataset, `draws` e `channels`. Todas as chaves de topo estão em `RUNNER_ARGS`, **mas** `amostra` e `l1_grow` são dicts livres: as chaves internas (`survivors`, `twin_controls`, `shards`, `widths`, `methods`...) não são validadas por ninguém. |
| M6 | `methods/autoencoder/autoencoder_flim_module.py:1059` `SVC(` próprio | segundo SVM. Exceção **já anotada** (`ponytail:` em :1055) por causa do print/tqdm do `fit_svm`; a anotação aponta para `src/utils/evaluate.py`, que já não é a casa (é `core/svm.py`). |
| M7 | kappa local: `methods/lejepa/lejepa_line_module.py:65,294`, `core/mixins/knn_kappa_probe_mixin.py:91,120`, `eval/svm_variants/svm_classification_flim.py:60,159`; `methods/transformer/classifier.py:29,42` `LinearSVC` | fere "uma métrica". São anteriores ao trabalho recente (não é drift novo). |
| M8 | `analysis/distill/{ruler_mismatch:47,distill_destroys_flim:75,embedding_analysis:57}.py`, `analysis/stats/compute_cost.py:169` | ainda importam `src.metrics.classification.compute_metrics` (caminho antigo). Já documentado em MIGRATION §6.3. |

### Baixos

- `score_full` (`spf_grade.py:968-973`) calcula `val_accuracy` e recall por classe à mão, ao lado de `compute_metrics`.
- `pw_whiten` mora em `eval/svm.py:365` e é usado como pré-processamento comum. Já está documentado como R6 (`_encode_pooled`) e não tem casa definida.
- `imagenet_norm` está em `GROW_KNOBS` (`runners/growth.py:153`) mas não em `RUNNER_ARGS`, então nunca pode ser ligado (chave morta).
- Os comentários de `schema.py:54-69` apontam defaults em `scripts/spifil_grow.py` (legado).
- Nomes repetidos:
  - `chain` em `spf_grade` e `l1_sem_rotulo`
  - `cells` em `spf_grade`, `l1_amostras` e `alt_encoders`
  - `features` em `l1_fast` e `l1_amostras`
  - `extract` em `benchmark/common` e `alt_encoders`
  - `freeze` em `spf_tabela` e `ms_tabela`, onde é duplicata real
- `benchmark/`, `saliency/` e `ga_search/` não têm `__init__.py` e só funcionam como pacote-namespace via `python -m`.
- Os `cli_kwargs(sys.argv)` dos `eval/*.py` novos **não** são desvio, porque essa é a convenção documentada de `eval/` (MIGRATION §0.3).
- Código morto em `spf_grade`: nenhuma função nunca referenciada, numa checagem só por nome.

### O que está certo (não mexer)

- Todo o SVM novo passa por `core.svm.fit_svm` + `core.metrics.compute_metrics`: `spf_grade.py:964`, `l1_fast.py:147-151`, `benchmark/common.py:150`, `ga_search/progressive.py:157-174`. `eval/svm.py:394 train_svm` é só um invólucro.
- Os runners leem só chaves declaradas e não há chave declarada sem uso (exceto `imagenet_norm`, acima).
- R1, R3 e R5 de `instructions.md` §4 continuam valendo para `core/flim/methods`.

## Plano de refatoração em fases

Princípio: cada fase **move**, não reescreve. O módulo antigo continua exportando os mesmos nomes (`from novo import *` ou reexport explícito), então jobs em andamento e os monkeypatches de A4 continuam funcionando. Nenhuma abstração nova.

**Verificação padrão (vale para toda fase).** Rodar **uma célula mínima com seed fixa** antes e depois da fase e comparar com `cmp`:

- Exemplo: um YAML `disc_md_spf_*` com `datasets: [flowers]`, `splits: [1]`, `draws: [42]`, `widths: [32]`, `images: 50`, `depth: 1`, com `work_dir`/`results_csv` num diretório de teste.
- Comparar o CSV de resultado e o `.pt` de fontes.
- Também `python -m experiments.ray.launch <yaml> --dry-run --set skip=state` em 3-4 YAMLs de cada `source`. A lista de comandos tem que ser igual.
- E `python -c "import eval.spf_grade, eval.alt_encoders, eval.spf_deteccao, eval.spf_fundo, eval.spf_figuras, saliency.batch"`.

### Fase 0: versionar (pré-requisito, zero código)

- Commitar na `main`, sem branch, o estado atual dos arquivos não rastreados da lista A1, em commits separados por pasta.
- Sem isso não há linha de base para comparar.
- Verificação: `git status` sem `??` nessas pastas e `git diff HEAD` vazio.

### Fase 1: caminhos fixos para um lugar só (M1)

- Os caminhos `/exps/moliveira/...` e `/dados/home/moliveira/...` dos `.py` viram constantes em `core/constants.py`, com o **mesmo valor**. O env var `SPF_SURVIVORS` continua.
- Cada arquivo passa a importar a constante. Não muda valor nenhum.
- Depois, num segundo passo e só se ele quiser, `SURVIVORS` e `SVM_LOCK` entram como chave de `runner_args` + `RUNNER_ARGS`.
- Verificação: célula mínima bit-idêntica e `grep -rn "/exps/moliveira\|/dados/home/moliveira" --include=*.py eval benchmark saliency ga_search flim` vazio fora de `core/constants.py`.

### Fase 2: o carregador de imagens vai para `core/data` (M2)

- `eval/l1_fast.py:load_images` vai para `core/data/loaders.py`, que já existe e tem 55 linhas.
- `l1_fast` reexporta o nome. Os 5 importadores seguem funcionando sem mudança.
- `spf_fundo.py:106` passa a chamar o mesmo carregador **só se** a saída for idêntica. Se não for, fica como está, com um comentário.
- Verificação: hash dos tensores carregados para um split antes e depois, e célula mínima.

### Fase 3: blocos de modelo saem de `spf_grade` (A2, parte camada)

- `Abs`, `ACTS`, `gem`, `covariance`, `vlad`, `POOLS`, `half_pool`, `map_whitener`, `renorm` vão para **um** arquivo novo, `flim/readouts.py`.
- `spf_grade.py` faz `from flim.readouts import ...` com os mesmos nomes no seu namespace.
- **Não** mover `build`, `background_cut`, `chain` nem `Source` (A4/A5).
- Verificação: célula mínima com `l1_readouts` cobrindo `gem_pw`, `vlad` e um `_eq`, mais `act: gelu`, bit-idêntica.

### Fase 4: encoders alternativos para `methods/` (A3)

- `Scattering`, `HebbBlock` e `PCANet` vão para `methods/alt_encoders/` (um `__init__.py` + um arquivo).
- `eval/alt_encoders.py` reexporta os nomes. O hack `__main__.Source` não muda nesta fase.
- Verificação: uma célula `source: alt_encoders` por encoder, com seed fixa, CSV idêntico.

### Fase 5: aprendizado de filtros sai de `spf_grade` (A2, parte grande)

- `Ctx`, `PatchCtx`, `METHODS`, `RANDOM`, `control` e as ~30 funções de :628-958 vão para `flim/filter_bank.py`. É uma família só, o que fica coerente com `flim/spifil.py`.
- `spf_grade` reexporta tudo, porque `alt_encoders` importa `METHODS`/`Ctx` dele.
- Só fazer **depois** que as grades `disc_md_spf_*` em andamento terminarem: é o maior bloco e o que mais tem consumidores.
- Verificação: uma célula por método de `METHODS` (`depth: 1`, `widths: [32]`), CSV idêntico.

### Fase 6: desfazer os acoplamentos frágeis (A4, A5, A6)

Feita com as grades paradas, porque muda importação de verdade.

- (a) `l1_sem_rotulo.MAX_WIDTH = 4096` vira o valor no próprio `l1_sem_rotulo`, se ninguém mais depender de 256. Senão, um parâmetro do job.
- (b) `_paths` e `_short` de `runners/growth` passam a públicos em `experiments/ray/paths.py`, que já é o dono dos caminhos.
- (c) `spf_deteccao` deixa o monkeypatch e passa a pedir o encoder por um retorno opcional de `chain`. É a única mudança de assinatura do plano: o default não muda nada.
- (d) `Source` só se move com um alias de pickle (`spf_grade.Source = ...`). Senão, os `.pt` antigos param de abrir.
- Verificação: `spf_deteccao` gera as mesmas figuras com `cmp` nos PNG, e `stored_sources` reabre um `.pt` antigo.

### Fase 7: YAMLs `disc_md_*` (M5), opcional

- Sem gerador novo. Validar só as chaves internas de `amostra` e `l1_grow` em `_check_eval_args` (schema), com a lista de chaves que os 177 YAMLs já usam, para que erro de digitação pare no preflight.
- Rodar `--dry-run` nos 607: nenhum pode ser recusado.
- Os YAMLs de rodadas encerradas podem ir para `experiments/autoencoder/_arquivo/` **se ele quiser**. O `skip: state` procura por `work_dir`, não pelo caminho do YAML. Confirmar isso antes.

### Fora deste plano (dívida antiga, já registrada)

- M6, M7 e M8: são anteriores ao trabalho recente e estão anotados em MIGRATION/instructions. M6 se resolve quando `fit_svm` ganhar um modo silencioso.
- `ga_search/` (M3, M4): tem as próprias leis e argparse. Merece uma auditoria própria antes de mexer.

## Ordem segura enquanto há grades rodando

As fases 0-4 podem entrar com as grades em andamento, porque só movem código e reexportam. As fases 5-6 entram com as grades `spf_grade`/`alt_encoders` paradas. A fase 7 é independente.
