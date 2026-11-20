# Cadeia de crescimento B0→B4: status da implementação

_17/11/2026. Tudo foi feito numa cópia (`scratchpad/growth`), só em CPU. A árvore viva e os processos não foram tocados._

## Entrega
- `growth.patch`, md5 `ea4787acd50b5867b9e24c43adc0ed1f`. Contém os patches a2, a3 e a4 mais a integração. O `git apply --check` passa na árvore viva limpa. O patch **não foi aplicado**.
- `fila_growth.json`: fila com 4 jobs, RS e Flowers, um par na athena GPU 3 (torch 2.2) e outro na Jaci GPU 0 (torch 2.7). Cada job traz um comando `smoke_antes`.
- Aplicação dos patches na ordem a2, a3, a4: **nenhum conflito**. O único arquivo compartilhado, o `monitor_report`, ficou como o a4 o deixou.

## Integração
- **`ga_search/backbones` (A3), mudanças retrocompatíveis**
  - `LayerSpec` ganhou `flavor` (`convblock` | `spifil_ops`) e `pool_padding`. Com o valor padrão, esses campos ficam fora do hash, então os hashes antigos não mudam.
  - `build.make_spifil_block` monta exatamente o `SpifilOpsBlock` de produção.
  - O `_fill_block` passou a reconhecer `conv.depthwise`.
- **`tests/backbones/_helpers.py` foi renomeado para `_bb_helpers.py`.** Os dois `_helpers` (A3 e A4) colidiam quando a suíte inteira roda junta.
- **`ga_search/hmodehb/growth_chain.py`**
  - Converte os checkpoints de produção em `Backbone`, com igualdade bit a bit verificada (c00306 RS e c00003 Flowers).
  - Inclui a semente "SPiFiL pequeno" com os pesos importados. No RS ela tem 1.383.536 params e entra como **B2**; a cabeça é nova.
  - Plano de morfismos exatos: widen/head_widen, deepen_identity, deepen_residual_zero e add_skip_zero. O plano é decodificado de um θ∈[0,1]⁴ via DE rand1/bin.
  - Os canais novos recebem SPiFiL pelo `KernelInitializer` do A2 (`init_method`: spifil0/1/2), com as colunas de saída em zero.
  - `protect` padrão `lowlr`, aplicado só aos elementos herdados (`ElementProtect`); os novos treinam com lr cheio.
  - Depois do morfismo: AE local só nos blocos novos → LeJEPA+SIGReg → WiSE com âncora no passo 0 (α=0 carrega a âncora bit a bit) → κ ORIG-val.
  - Toda decisão passa por `FV.check_decisions`.
  - Aceitação: κ_filho ≥ κ_pai + δ, com δ=0 por padrão. A flag `require_improvement` recusa α=0.
  - Seleção por faixa: pais na frente de Pareto (κ, FLOPs 2·MAC); triagem em f15 e promoção de 1/η para f45.
  - Teto de 5,5M em todo filho; a cabeça ≥1280 nunca encolhe.
  - Linhagem em `lineage.jsonl`: pai, op, Δκ, rel-L2 de preservação, params e FLOPs.
  - Robustez: flock (exit 3), deriva de config (exit 4), `cand_key` com o sha256 do pai, adoção de `result.json` no resume, OOM 10×300 s.
- **`growth_test_channel.py`** roda em subprocesso, calcula o SVM ORIG no teste e grava `test_orig.jsonl` com `never_feeds_fitness`. O controlador nunca lê esse arquivo, e um teste estático garante isso.
- **`config/hmodehb_b0b4/growth.yaml`**: os padrões são 4 pais, 8 filhos, 2 gerações, η=3, f15/f45.

## Testes
- `tests/hmodehb/test_growth_chain.py`: **24 passed**. Cobrem:
  - preservação ≤1e-5 no embedding 1280;
  - α=0 igual ao pai bit a bit;
  - a regra de aceitação;
  - a linhagem;
  - o teto;
  - kill -9 e resume;
  - flock e deriva;
  - o e2e real.
- **E2E real**: RS, CPU, f1, 60 imagens de treino e 40 de validação, B0→B1→B2, 133 s.
  - Seeds convertidas: c00306 (κ 0,3629) e c00001.
  - B1 (widen, 168 filtros SPiFiL0): preservação 3,9e-8, κ 0,3964 e 0,3982, α=0,25.
  - B2: aceitos por α=0 com κ igual ao do pai.
  - Isto valida o código, não é resultado científico.
- **Suíte inteira da cópia**: **2676 passed, 4 skipped, 0 falhas** (8 min 24 s).

## Limitações declaradas
- **α=0 só é igual ao pai bit a bit nos estados.** Na função, deepen e skip são exatos bit a bit. O widen fica em rel ≤1e-7, por causa da ordem de soma da conv; no e2e o κ(α=0) foi igual ao do pai.
- **Os canais novos de dwsep, inception e depthwise usam a init do PyTorch**, registrada como fallback. O SPiFiL só é aplicado a conv densa e pointwise.
- **Promoção f15→f45 refaz o treino a partir do morfismo**, em vez de continuar o de f15.
- **A semente "SPiFiL pequeno" é medida com a cabeça nova**, que não passou por treino.
- **As estimativas da fila são extrapoladas e marcadas [H]**, a partir do A2 escalado por FLOPs: RS de ~1,7 h/geração em B0→B1 até ~7,5 h em B3→B4, com pico de ~14 a ~38 GB. O `result.json` grava o tempo e o pico reais, e o smoke mede os dois.
- **No athena, os checkpoints de seed precisam existir** (rsync da Jaci). Os que faltam são pulados com aviso.
- **torch 2.2**: não foi testado. Só a varredura estática foi feita, por isso o smoke é obrigatório.

## growth_fix4 (17/11/2026): leitura fp32, adopted_as_parent, planner por família, `ssl: none`
_Feito numa cópia (scratchpad), só CPU. Nada aplicado na árvore viva, nenhum processo tocado._

- **Entrega:** `growth_fix4.diff` (incremental sobre `/dados/home/moliveira/exps/growth/code`, já com v2+mem+mem2+mem3+fix do `run_search.py:488`), md5 em `growth_fix4.diff.md5`. `git apply --check` e `patch --dry-run` passam na árvore viva.
- **(a) Leitura determinística.** `run_search._embed` só liga o autocast se `embed_amp` (atributo, padrão True = histórico inalterado). O `GrowthEvaluator` põe `embed_amp=False` e cada `score` roda em `fp32_reading` (autocast off + TF32 off em cuDNN/matmul, restaurados na saída). O canal de teste usa a mesma leitura. C=100, sem scaler, cabeça ≥1280 inalterados. `embed_precision="fp32_noautocast_notf32"` vai para o `result.json` (sementes e filhos), para a linhagem e para a `fitness_spec`.
  - **Atenção:** isso **muda a `fitness_sig`**. Um estado/`result.json` lido em FP16 não é adotado (`FitnessSigMismatch`). A cadeia do athena precisa recomeçar num `--out` novo, com as sementes re-pontuadas em fp32.
- **(b) Aceitação.** No `_run_job`, o pai é re-pontuado no mesmo processo e na mesma leitura, antes do treino do filho. A decisão `decide()` dá:
  - `adopted_as_parent` quando a preservação é exata e α=0. O filho é o pai: não entra na faixa, não é promovido para f_high e não conta como rejeição. Isso cobre o flowers B1g0c00 (0,6656 contra 0,6718) e o RS B1g0c00 (κ idêntico 0,5719, antes `accepted=true`).
  - `accepted` / `rejected` pela regra κ_filho ≥ κ_pai_re-pontuado + δ.
  - `failed`.
  - A linhagem grava `decision`, `kappa_val_orig` (filho), `kappa_parent_rescored` (= `parent_kappa_val_orig`), `parent_kappa_val_orig_member`, `embed_precision` e `family`.
- **(c) Planner.** `plan_children` gera n_children planos válidos, com os pais em round-robin. O k-ésimo filho de cada pai começa por uma família diferente (`widen_b{j}`, `head_widen`, `widen_all`, `deepen_residual_zero@a`, `add_skip_zero@j`, `deepen_identity@a`) e completa a faixa com widens encadeados (fator ≤2,5 por op, bisseção até o alvo θ[2] dentro da faixa, alargando junto as camadas amarradas por residual identity). Planos com a mesma forma no mesmo pai são pulados. Um pai esgotado cede a vaga a outro. "Sem plano" só aparece quando nenhum pai/família cabe, e vem com o motivo por família (ex.: "já acima da faixa", "DBN partido", "passa do teto").
  - Causa do "sem plano" antigo: o tiny (21k params) sorteava deepen/skip que quase não somam params e estourava `max_ops`; o planner antigo falhava em ~3/8 dos θ e só tentava uma vez por filho.
  - Genomas reais para B1 (8 filhos): flowers c00001 (tiny) → 5 planos (4–5 widens encadeados); c00003 → 3 planos (o widen do bloco inception+DBN é inexato e o motivo fica registrado). RS c00001 → ≥4; c00306 → ≥3. São 16/16 filhos válidos nas duas bases.
- **`ssl: none`** (desligado por padrão; o padrão continua `lejepa`, sem deriva de config): pula o SSL global e fica só o AE local nos blocos novos, para a ablação. O resultado grava `ssl`/`ssl_skipped`.
- **Testes:** `tests/hmodehb/test_growth_fix4.py`: **15 passed** (34 s). `test_growth_chain.py` + `test_growth_mem.py`: **37 passed** (incluindo o e2e real RS), 0 skip. Cobertura:
  - (a) α=0 dá embedding bit a bit e κ idêntico (3 morfismos exatos), a leitura é repetível e a precisão é declarada;
  - (b) regras de `decide`, incluindo os dois casos reais;
  - (c) genomas reais RS/flowers com ≥2 planos por pai e 8/8 válidos; impossível com motivo;
  - `ssl: none`;
  - controlador ponta a ponta com decisão e os dois κ.
  - Em `test_growth_chain.py` só mudou a asserção de `accepted` em `test_lineage_records` (passou a considerar `adopted_as_parent`).

## growth_fix5 (17/11/2026): estimador de pico, filtro infeasible_mem, falha rápida de OOM de capacidade
_Feito numa cópia (scratchpad), só CPU. Nada aplicado na árvore viva, nenhum processo tocado._

- **Entrega:** `growth_fix5.diff` (incremental sobre `/dados/home/moliveira/exps/growth/code`, já com v2+mem+mem2+mem3+fix4+pmodel.cpu()), md5 `635b7ea86372ac42293a742959d1a182` em `growth_fix5.diff.md5`. `git apply --check -p1` e `patch --dry-run` passam na árvore viva.
- **(1) Estimador** `estimate_peak_gb` / `plan_memory`. Conta, por bloco e na resolução de entrada dele, N·C·H·W·bytes, com N = V·B no SSL e B no AE. Entram:
  - as entradas salvas pelo checkpoint;
  - o maior bloco recomputado: 3 cópias (conv, y do _ActPool, grad), mais a norma (BN em bf16; DBN em fp32 sob bf16, 5 cópias), mais o intermediário do dwsep e o extra de um bloco não fundível;
  - a projeção da cabeça;
  - c0 = 2 GB.
  Os coeficientes ficam em `MEM_EST`. Calibração com genomas reconstruídos localmente:
  - c00306 widen 0/1→376: estimado **46,2 GB** (medido 45,3–47,2);
  - widen 295: **37,4 GB** (medido 40,6, −8%);
  - c00306 widen 0/1→455: **54,8 GB** (> 47,3). Este é o B1g0c00 RS que o planner novo gera a partir de θ/seeds determinísticos, e é o OOM de 21,77 GiB (DBN do bloco 1 sobe para fp32).
  - **Não calibrado:** B1g0c03 (pai tiny, 14,3 GB) e flowers B1g0c00 (33,6 GB). Os genomas exatos desses nós do athena não estão aqui. Com os pais locais, o planner dá outros planos (tiny head_widen ≈ 36 GB, dominado pela cabeça 14→1280 em 112²; flowers widen_b0→188 ≈ 19 GB). O teste fica em skip até trazer o `growth_spec.json`/`backbone.pt` desses nós. **[H]** O pico medido é `max_memory_reserved` do job inteiro (cache do alocador, fase SPiFiL), não só do SSL.
- **(2) Filtro.** `growth.mem.max_gpu_gb` (null = total da GPU − 3 GB via torch.cuda; sem GPU, o filtro fica desligado). Um plano com pico estimado acima do teto vira `infeasible_mem`, com o motivo (pico ssl/ae e teto) registrado na linhagem (`node_id=<nó>_mem<k>`, `decision=infeasible_mem`). O planner passa para a próxima família/pai. Se nada couber, vem "sem plano" com o motivo.
- **(3) Falha rápida.** `oom_is_capacity` lê a mensagem do OOM. É capacidade quando (pedido + uso deste processo > total − `oom_capacity_margin_gib`=1,0) ou (GPU sem outros processos e pedido > livre). Nesse caso: `failed/oom_capacidade` na hora, sem retries. Na contenção, os 10×300 s continuam.
- **(4) Fatiamento do BN no recompute: NÃO feito.** Fatiar o apply do BN do cuDNN (kernel fundido) por um apply manual não é bit a bit. No DBN, o grad do afim/W soma sobre M, e fatiar muda a ordem da soma.
- **Atenção no resume:** as chaves novas `growth.mem.max_gpu_gb` e `growth.mem.oom_capacity_margin_gib` mudam o config_sha. Para retomar um `--out` existente, use `--allow-drift growth.mem.max_gpu_gb --allow-drift growth.mem.oom_capacity_margin_gib`. A fitness_sig não muda.
- **Testes:** `tests/hmodehb/test_growth_fix5.py`: 11 passed, 1 skip (calibração tiny/flowers). fix4+chain+mem+fix5: **63 passed, 1 skipped** (2 min 28 s).

## growth_fix6 (17/11/2026): a promoção f15→f45 continua o checkpoint em vez de refazer o treino
_Feito numa cópia (scratchpad), só CPU. Nada foi aplicado na árvore viva e nenhum processo foi tocado._

- **Entrega:** `growth_fix6.diff` é incremental sobre `/dados/home/moliveira/exps/growth/code` (v2+mem+mem2+mem3+fix4+pmodel.cpu+fix5). O md5 está em `growth_fix6.diff.md5`. `git apply --check -p1` e `patch --dry-run` passam na árvore viva. Arquivos: `ga_search/hmodehb/growth_chain.py`, `ga_search/modehb/ssl_trainers.py` (opt-in, padrão inalterado) e `tests/hmodehb/test_growth_fix6.py` (novo).
- **Motivação:** no athena, o RS B2g1c02 deu 0,620 em f15 e caiu para 0,593 em f45, porque o f45 era refeito do morfismo e o ganho se perdia.
- **O que é salvo em f_low** (`nodes/<nó>/f15/resume_state.pt`, gravação atômica, sha256 no `result.json`):
  - pesos treinados **PRÉ-WiSE**;
  - âncora (o morfismo, antes de qualquer treino);
  - máscaras do protect;
  - estado do SSL: AdamW, RNG torch CPU/CUDA, gerador das views, gerador do loader e passo global.
  O `result.json` de f15 também grava `resume_state_has_optimizer` e `resume_state_has_rng`. Só é salvo quando fid=fid_low, fid_low≠fid_high, `promotion.mode=resume` e o SSL está ligado. Tamanho: cerca de 4× os parâmetros em fp32 (≈90 MB num filho de 5,5M). O arquivo não é apagado.
- **Promoção (`_promote` → `_run_resume_job`):**
  - O pai é re-pontuado no mesmo processo (fix4 b).
  - Os pesos PRÉ-WiSE de f15 são carregados na arquitetura de f15 e o SSL continua por **f_high − f_low = 30 épocas** (`promotion.extra_epochs: fid_diff`; a alternativa `ssl_diff` dá ssl(45) − ssl(15) = 20).
  - O lr é o mesmo e é fixo (o trainer não tem schedule). AdamW e RNGs são restaurados. O AE local **não** é repetido.
  - Resultado: o nó termina com AE 5 + SSL 10+30 = 45 épocas. É diferente do f45 do zero (AE 15 + SSL 30), e isso fica declarado em `ae_epochs`/`ssl_epochs`.
  - Sem AdamW salvo (`promotion.save_optimizer: false`): o AdamW é novo e o fato fica registrado em `promotion.optimizer = "restarted (AdamW não salvo em f_low)"` e `optimizer_resumed=false`.
  - O WiSE em f45 escolhe α entre a **âncora = morfismo (pai)** e o treinado-f45.
- **Seleção do nó pela VALIDAÇÃO:** se κ_val(f45) < κ_val(f15), o nó fica com f15 (checkpoint, κ e α de f15). Empate fica com f45. O teste nunca é lido.
  - Registrado em `result.json` e na linhagem: `selected_fid`, `kappa_val_orig_f_low`, `wise_alpha_f_low`, `kappa_val_orig_f_high`, `wise_alpha_f_high`, `checkpoint_f_high` (o blend de f45 fica salvo mesmo quando perde), `promotion{mode, from_fid, from_cand_key, extra_ssl_epochs, optimizer_resumed, rng_resumed}` e `selection`.
  - O membro da faixa ganha `selected_fid`.
  - A decisão (`decide`) usa o κ e o α selecionados contra o pai re-pontuado.
  - O κ de f15 é o gravado no f15 (leitura fp32 determinística; não é re-pontuado).
- **Quando refaz do morfismo (registrado em `promotion.reason`):** `promotion.mode: retrain`, `ssl: none`, f15 sem `resume_state` (resultado anterior ao fix6) ou `resume_state` com sha256 diferente.
- **Compatibilidade de retomada:**
  - A cand_key de f15 não muda (f15 antigos são adotados).
  - A cand_key da retomada em f45 inclui a cand_key de f15, então um f45 antigo refeito não é adotado como retomada. No fallback de retrain, vale a cand_key histórica.
  - A **fitness_sig não muda**.
  - A chave nova `growth.promotion` muda o config_sha. Para retomar um `--out` existente, use `--allow-drift growth.promotion`. Os f15 antigos não têm `resume_state`, então a promoção deles cai no retrain (registrado). Para ter retomada desde o início, recomece num `--out` novo.
- **ssl_trainers.train_global_ssl:** kwargs novos `return_state=False` e `resume=None`. Com os valores padrão, o comportamento é o histórico (testado: `return_state` não muda o treino). Na retomada, a forward-sonda do `build_opt` e a init dos módulos extra não mexem nas running stats de BN nem nas RNGs. Projector e decoder são restaurados.
- **Bit a bit:** em CPU com seed fixa, f_low + retomada == treino contínuo (AE(f_low) + SSL(f_high − ae(f_low))): pesos, AdamW, passo, κ do WiSE, α e modelo final. **[H]** Em CUDA não é garantido bit a bit, porque há kernels não determinísticos; a semântica é a mesma.
- **Testes:**
  - `tests/hmodehb/test_growth_fix6.py`: **11 passed** (50 s). Cobrem trainer lejepa/simclr retomado == contínuo; padrão inalterado; filho f_low→f_high == contínuo bit a bit (com AdamW), e sem AdamW reinicia e registra; sha errado recusado; regra melhor-de-dois (inclui 0,620/0,593); controlador in-process (retomada, campos, f_low forçado vence pela val, 2ª execução adota tudo); fallbacks de retrain registrados; deriva `growth.promotion` e fitness_sig.
  - chain + fix4 + fix5 + mem: **63 passed, 1 skipped** (3 min 44 s; o skip é a calibração tiny/flowers do fix5). O kill -9/resume do chain (fid 1→2, eta=1) agora passa pela retomada.
  - `tests/modehb/test_ssl_trainers.py` + `test_external_seed.py` + `tests/hmodehb/test_h4_e2e.py`: **57 passed**.
- md5 `38cce06eadc9249f36ecb5bbbfdfc32a` em `growth_fix6.diff.md5`.

## growth_fix7 (17/11/2026): velocidade da fase "grow". O fit SPiFiL NÃO é o gargalo.
_Feito numa cópia (scratchpad), só CPU na Jaci, com o ambiente de threads do athena (OPENBLAS=2, OMP=8, MKL=8, cap do run_search). Nada foi aplicado na árvore viva e nenhum processo foi tocado._

- **Entrega:** `growth_fix7_speed.diff`, incremental sobre `/dados/home/moliveira/exps/growth/code` (v2+mem+mem2+mem3+fix4+fix5+fix6). O md5 `6238b362ba6398e432df254aa6c7532c` está em `growth_fix7_speed.diff.md5`. `git apply --check -p1` passa na árvore viva. Arquivos: `ga_search/hmodehb/growth_chain.py` e `tests/hmodehb/test_growth_fix7.py` (novo).
- **Reprodução fiel do "[grow] N=2000 D=936":** flowers c00003 (k3), com o bloco 1 alargado para 104 e +60 canais no bloco 2 pelo `NewFilterInit` real (spifil0 → fit_ops_block → fit_dense), ipc=4, 20 doadores × 100 sementes.
  - **realize_plan inteiro: 1,34 s** (cProfile).
  - **fit_dense (scorer Fisher/Mahalanobis + alocação + DiversitySelector + build_filters): 0,11 s.**
  - Resto: SLIC 0,36 s, encoder_features 0,31 s, gc 0,2 s.
  - Não há laço por canal (o caminho do widen é denso; o `fit_depthwise` não é usado pelo `NewFilterInit`, que cai no init PyTorch para conv agrupada). Também não há eigh/cdist relevante.
  - Conclusão: nada na biblioteca spifil precisa ser envolvido. Nada foi mudado no SPiFiL.
- **SVM (C=100, libsvm linear, max_iter=-1, sem scaler; inalterada)** nos embeddings fp32 reais do pai flowers (1922×1280 / 412): **1 fit = 0,70 s**. No log do athena, o SVM imprime "SVM fit:"/`[SVM_FITNESS]`. Se o último print é o `[grow]`, a SVM não está rodando.
- **[H] Onde está a parada do athena:** depois do último `[grow]` vêm o resto do `realize_plan`, o `MO.finalize`/preservação (CPU, 8 imagens) e o `train_child` (AE local + SSL), que **não imprime nada**. No smoke da Jaci, depois do `[grow] N=5600 D=27` veio direto o treino. Não consegui reproduzir horas em nenhuma parte medível na Jaci. É preciso a pilha real do processo do athena.
- **Mudanças (resultado idêntico):**
  1. **SIGUSR1 → pilha de todas as threads** (`faulthandler.register` no `main`): `kill -USR1 <pid>` escreve no log onde o processo está, sem py-spy/ptrace e sem parar o processo. Para os processos JÁ rodando no athena (sem o handler), um SIGUSR1 os MATA. Use `py-spy dump --pid` via `pip install --user py-spy` se `/proc/sys/kernel/yama/ptrace_scope` for 0. Senão, `top -H -p <pid>` + `cat /proc/<pid>/task/<tid>/wchan`.
  2. **WiSE:** as 5 leituras (GPU) seguem em sequência; as 5 SVMs rodam em paralelo em threads (`GrowthEvaluator.score_many`; o `fit` do libsvm do sklearn é `nogil`). Workers = `svm_workers`/`svm_ram_gb`, igual ao `run_search`; `svm_blas_threads` = 1. Mesma ordem e mesmo desempate. Medido nos embeddings reais: 5 sequenciais **3,14 s** → paralelo **0,82 s (3,85×)**, kappa/f1/quality **bit a bit iguais**.
  3. **Cache do re-score do pai** (`score_parent`) por (sha256 do checkpoint, embed_precision, fitness_sig, n_val, n_train) no processo. Um pai é re-pontuado uma vez por processo, não uma vez por filho. Os carimbos/réguas são restaurados. `result.json` ganha `parent_rescore_cached` dentro do sv (a fitness_sig não muda). Só no `_run_job`; o `_run_resume_job` continua re-pontuando.
  4. `cache_size` do libsvm: **não mudado**. Com n_train ≤ 4000, a matriz de kernel de cada par ovo já cabe nos 200 MB padrão, então não há ganho, e a SVM do repo fica intocada.
- **Testes:** `tests/hmodehb/test_growth_fix7.py`: 6 passed (score_many == sequencial bit a bit; concorrência real; workers=1; `_wise_select` novo == laço antigo em alpha/kappas; cache do pai; SIGUSR1). fix7 + fix4 + fix6: **32 passed** (75 s).

## growth_fix8 (17/11/2026): fase muda do train_child + "MKL ERROR DLASCL". O dado NÃO é o gargalo.
_Feito numa cópia (scratchpad), só CPU na Jaci: a GPU da Jaci estava cheia (96,9/97,9 GB, 138 MB livres; o 1º perfil na GPU deu OOM no AE). Nada foi aplicado na árvore viva e nenhum processo foi tocado._

- **Entrega:** `growth_fix8_data.diff`, incremental sobre `/dados/home/moliveira/exps/growth/code` (v2+mem..mem3+fix4..fix7). O md5 `de15ce69a84d42b012deb001d4f78800` está em `growth_fix8_data.diff.md5`. `git apply --check -p1` e `patch --dry-run` passam na árvore viva. Arquivos: `ga_search/modehb/ssl_trainers.py`, `ga_search/modehb/local_ae.py`, `ga_search/modehb/external_seed.py`, `ga_search/abstract_init/spifil0.py`, `ga_search/hmodehb/growth_chain.py` e `tests/hmodehb/test_growth_fix8.py` (novo).
- **Perfil real (flowers, CPU, 8 threads, ambiente de threads do athena):** filho de 381k params (inception 14 batch_dbn, dwsep 145 zca, conv 145) vindo do pai hybrid:c00001 por `propose_plan` + `realize_plan` com o spifil0 real. Lotes de 64, `max_images_per_epoch=640` (10 passos/época).
  - **Carregamento:** os dados já estão em RAM (`TensorDataset`, `ram_cache` padrão), então não há decode por época. Espera do loader = **0,07 s em 10 passos (7 ms/passo)**. Collate `TensorDataset`+`default_collate` = 9,4 ms/lote; com `__getitems__` em lote = 9,8 ms. Não há ganho, então **não mudei o loader** (a ordem e as seeds ficam intactas).
  - **make_views (4×64×224):** 0,26 s/passo na CPU, já vetorizado (affine_grid/grid_sample no device de `x`, que é a GPU no athena).
  - **Passo:** o tempo é compute. AE: 0,16 a 0,43 passos/s; SSL: 70 s/passo (0,014 passos/s) na CPU. Dado + vistas somam < 0,5 % do passo.
  - **Conclusão:** a hipótese "augment/make_views na CPU, num_workers=0 ou decode" está **refutada** para os dados.
  - **[H] O que sobra para explicar "1 thread a 100 % + GPU 0↔30–80 %"** são sincronizações GPU→CPU por passo. A thread principal faz espera ativa (spin) em cada sync, o que aparece como 100 % de CPU, e a GPU fica ociosa entre kernels pequenos. Por passo do SSL havia:
    - 9·V = 36 cópias H2D síncronas dos sorteios do `make_views`;
    - 1 `isfinite(...).all()` por parâmetro;
    - 2 `nonzero` por parâmetro protegido no `ElementProtect` (protect=lowlr);
    - `eigh` float64 por grupo de 16 canais do DBN (com checagem de info);
    - os micro-lotes de 16 do `_ActPool`.
    - No AE local havia ainda 1 `float(norm)` por parâmetro por passo.
  - **Não medido em GPU** (sem memória livre na Jaci). Para confirmar no athena: `py-spy dump --native` ou `kill -USR1` (fix7).
- **Mudanças (todas bit a bit na CPU, testadas contra cópias literais do código histórico):**
  1. `make_views`: mesmos 9 sorteios por vista, mesmo gerador CPU, mesma ordem e mesmas formas, levados ao device numa ÚNICA cópia (pinned, non_blocking) em vez de 36. As vistas são bit a bit iguais e o estado do gerador termina igual, então não há mudança de distribuição nem de RNG. `result["augment_device"]` (SSL) e `augment_device` no `result.json` (train_child e retomada).
  2. `train_global_ssl`: a checagem de gradiente não finito passou a fazer 1 sincronização por passo (antes, 1 por parâmetro). Os nomes só são procurados se houver algo não finito, e a decisão de pular é a mesma. O kwarg novo `progress` imprime 1 linha por época com perda, tempo, passos/s, espera do loader e passos pulados.
  3. `train_local_ae`: as normas de gradiente saem numa sincronização (mesmos valores, mesma ordem, mesma regra do 1º não finito). O kwarg `progress` imprime 1 linha por época.
  4. `ElementProtect`: snapshot do tensor inteiro + `torch.where`, sem `nonzero`. A conta por elemento mascarado continua `old + scale·(new − old)`, então o resultado é bit a bit igual.
  5. `growth_chain`: passa `progress="[ae] bloco j"` e `"[ssl] N ép."`. O log agora mostra, por exemplo, `[ae] bloco 1 época 2/2 passos=10 perda=5.8136 t=54.1s passos/s=0.185 pulados=0`.
  - **Não feito:** batch de grupos do `eigh` do DBN (não seria bit a bit; cusolver em lote ≠ por matriz) e troca do loader (sem ganho medido).
- **"Intel MKL ERROR: Parameter 4/5 … DLASCL" (48×) logo após "[grow] N=5600 D=720":**
  - **Origem:** é o `torch.linalg.eigh(torch.cov(x.T))` do ZCA do `GrowOneLayer.run` (`flim/spifil.py`). O spifil0 usa `whiten=True` por padrão, e D=720 é a dimensão do patch impressa na linha anterior. Os patches (features do prefixo do PAI nas imagens SPiFiL) chegam com NaN/Inf. O divide-and-conquer do LAPACK chama DLASCL para escalar e desescalar cada subproblema com norma NaN (parâmetro 4 = CFROM, 5 = CTO), o que dá 2 linhas por subproblema; 48 linhas ≈ 24 subproblemas para D=720.
  - O MKL do athena (torch 2.2) imprime e devolve NaN em silêncio, e o ZCA/filtros NaN seguem adiante. Na Jaci (torch 2.7, oneMKL) só `svd`/`svdvals` com Inf reproduzem as mesmas 2 mensagens; `eigh` devolve NaN calado.
  - **Checagem:** `spifil0.check_finite_patches` roda no `FoldedGrowOneLayer.run` ANTES do `super().run` (portanto antes do eigh). Com patch não finito, levanta `InfeasibleCandidate("spifil0_nonfinite_patches", details={layer, N, D, nonfinite_values, nonfinite_rows, nan, inf})` e imprime `[grow] RECUSADO (fix8): ...`. O `NewFilterInit` já captura a exceção e registra `fallback="InfeasibleCandidate: spifil0_nonfinite_patches"` (init PyTorch nos canais novos, como em qualquer falha do fit). Com entrada finita nada muda.
  - **[H]** A causa a montante (que bloco do pai solta NaN/Inf nas imagens SPiFiL: DBN/ZCA ou overflow) fica para o próximo diagnóstico.
- **Passos/s antes → depois (CPU):** iguais dentro do ruído (make_views 264,6 → 260,1 ms; o AE e o SSL são compute). O ganho esperado é **só na GPU**: por passo de SSL saem 35 cópias síncronas + ~2·P nonzero + (P − 1) syncs do isfinite (P = nº de parâmetros); por passo de AE saem P − 1 syncs. **Não medido em GPU (Jaci cheia): speedup em GPU = [H], a medir no athena com as linhas `[ssl] ... passos/s=`.**
- **Testes:** `tests/hmodehb/test_growth_fix8.py`: 14 passed (4 s). Regressão: test_ssl_trainers + test_local_ae + test_external_seed + fix4 + fix6 + fix7 + fix8 = **95 passed** (1 min 43 s). O fix6 cobre a retomada bit a bit, que segue valendo com o make_views novo.

## growth_v31_hotfix_ae_diverge (18/11/2026): AE local divergindo nos blocos alargados
_Feito numa cópia (scratchpad), só em CPU na Jaci. Nada foi aplicado na árvore viva e nenhum processo foi tocado._

- **Entrega:** `growth_v31_hotfix_ae_diverge.diff`, incremental sobre `/dados/home/moliveira/exps/growth/code` (v3 + v3.1 b1–b7 + hotfix). O md5 está em `growth_v31_hotfix_ae_diverge.diff.md5`. `git apply --check -p1` passa na árvore viva.
- **Arquivos:** `ga_search/modehb/local_ae.py`, `ga_search/hmodehb/growth_chain.py` e `tests/hmodehb/test_growth_v31_ae_diverge.py` (novo).
- **Reprodução:** script em `calib/repro_ae_diverge.py`, curvas em `calib/{before,final}_*.json`. Mesmo laço do `train_child`: `realize_plan` com spifil0 real, ElementProtect lowlr, lr_ae 1e-3, batch 64, fid 1 (1 época de 10 passos), alvo pixel.
  - **RS c00306, widen 0,1 → 455:** bloco 0 foi de 0,399 → **2,57** (passo 2) e terminou em 0,48 (razão 2,69). Bloco 1 foi de 0,593 → **2,98** e terminou em 0,155 (razão 1,61).
  - **tiny c00001, widens encadeados (18/32 → 80 → 200 → 500):** bloco 1 foi de 20,96 → **6017** → … → 139,7 (razão 38,5).
- **Causa (medida):** o pico está no **passo 2**, ou seja, no 1º passo do Adam do **decoder aleatório**.
  - A entrada do decoder z tem RMS de ~8 a 16, e a 1ª conv do decoder tem fan-in 9·C (C ≈ 450–500).
  - O Adam move cada peso em ~lr com sinal coerente, então a reconstrução anda ~lr·9C·|z|.
  - **Não é a init SPiFiL.** O W&B mostra o mesmo pico com o SPiFiL congelado.
  - A norma de gradiente do decoder no passo 1 é 2,4e3 (tiny bloco 1).
- **O que cada opção fez (tiny bloco 1 e RS bloco 0, razão):**

  | opção | tiny bloco 1 | RS bloco 0 |
  |---|---|---|
  | `grad_clip=1` | 37,8 (não resolve: o Adam ignora a escala do gradiente) | — |
  | `warmup` 5 passos | 1,34 | — |
  | só `dec_in_norm` | 0,38 → 4,08, razão 3,38 | 3,38 |
  | `dec_in_norm` + `warmup` | 0,74 (ruidosa) | 0,81 |
  | **`dec_in_norm` + `lr_scale_fanin` (padrão)** | **0,380 → 0,029, razão 0,25** | **0,373 → 0,056, razão 0,32** (bloco 1: 0,574 → 0,044, razão 0,24) |

- **Correção (padrão novo; mexe só no decoder descartável; o bloco, o alvo, o lr do bloco e o ElementProtect não mudam):**
  - `ae_dec_in_norm: true`: o decoder lê `(z − μ)/σ`, com μ e σ por canal FIXOS (buffers) do lote-sonda. É uma reparametrização da 1ª conv do decoder.
  - `ae_lr_scale_fanin: true`: o lr dos pesos do decoder é multiplicado por `sqrt(144/fan_in)` quando fan_in > 144.
  - Opções desligadas: `ae_grad_clip` (null) e `ae_warmup_steps` (0).
- **Guarda `ae_diverge_guard: 1.0`:** usa razão e tendência, nunca o valor absoluto. Função: `local_ae.ae_guard_verdict`. Dispara se:
  - a perda não for finita;
  - (média da época / perda inicial) > guarda **e** a perda estiver subindo (média maior que a da época anterior; na 1ª época, o último quarto maior que o primeiro quarto);
  - ou, no fim, (último quarto da última época / perda inicial) > guarda.

  "Começa alto e cai" passa (por exemplo, 10,85 → 3,28). Quando dispara, o bloco volta **bit a bit** ao estado pré-AE (o morfismo, incluindo os buffers), e o result.json registra `ae_logs[j].ae_diverged/ae_restored` e `ae_diverged_blocks`. O filho segue sem AE naquele bloco. `null` desliga a guarda.
- **Smoke:** sempre que `recon_ratio > 1`, o log imprime `[ae] ALERTA (hotfix ae_diverge)` e o result.json ganha `ae_alerts`.
  - `ae_smoke_strict: true` faz o filho falhar (ChainError). **Use `--gset ae_smoke_strict=true` nos smokes da v3.**
  - O padrão é `false`. Um padrão automático "estrito em fid 1" quebrava o `test_growth_fix4` do controlador.
- **Bit a bit:** com as opções desligadas (sem kwargs), `train_local_ae` é idêntico ao da árvore viva (conferido contra o arquivo vivo). A guarda ligada sem disparar não muda nada (testado no `train_child` inteiro).
- **Deriva:** as chaves novas em `GROWTH_DEFAULTS` mudam o config_sha. Para retomar um `--out` existente, use `--allow-drift`. A fitness_sig não muda. **O padrão muda a ciência do AE (otimização do decoder), e isso está declarado aqui.** Para voltar ao histórico: `--gset ae_dec_in_norm=false --gset ae_lr_scale_fanin=false --gset ae_diverge_guard=null`.
- **Testes:** `test_growth_v31_ae_diverge.py`: **6 passed**. Regressão (`test_local_ae`, fix4, fix6, fix8, v31_speed, v31_covgate e o teste novo): **80 passed** (2 min 21 s).
