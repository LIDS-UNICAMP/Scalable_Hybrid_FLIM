# Status dos mecanismos (cadeia B0→B4 e GA) × literatura

Data: 17/11/2026, ~21h (jaci). Auditoria só de leitura.

**Fontes de código**
- GA: `/dados/home/moliveira/Scalable_Hybrid_FLIM/ga_search/modehb/*`.
- Cadeia de crescimento: `/dados/home/moliveira/exps/growth/code/ga_search/{hmodehb,backbones}/*`, que reaproveita `ga_search/modehb` daquela árvore.

**Fontes de "uso real"**
- Configs: `ga_search/config/modehb/{search_v3,search_v3_lemonade,pnas_cs}.yaml` e `ga_search/config/hmodehb_b0b4/{growth,base}.yaml`.
- Estados dos runs em `artifacts/modehb_hybrid/*/state/seed42/search_state.json` (625 candidatos, 11 bases) e `artifacts/modehb_lemonade/*` (408 candidatos, 7 bases).
- Nós da cadeia:
  - `research/spifil_b0b4/growth/calib/*/nodes/*/result.json` (4 nós B1 de calibração);
  - `research/spifil_b0b4/growth/summaries/` (RS B0→B1→B2 do athena);
  - `artifacts/hmodehb_b0b4/growth_smoke` (smoke de hoje, que caiu em OOM em B1).

**Limites desta auditoria**
- O PNAS-CS e os runs completos da cadeia rodam no **athena**, que não resolve a partir da jaci. Os `result.json` deles não foram lidos.
- Sobre o PNAS-CS, só os yaml e o código foram lidos.

## 1. Tabela

Legenda:
- **Gene** = varia por indivíduo na busca.
- **Opção** = chave de config fixa por run.
- **Herdado** = só aparece porque veio na semente B0 do GA.

| # | Mecanismo | Implementado? (arquivo:linha) | Gene ou opção? | Usado na cadeia B0→B4? | Usado no GA (hybrid / LEMONADE / PNAS-CS)? |
|---|---|---|---|---|---|
| 1 | Whitening (ZCA, DBN, batch_zca, batch_dbn) | Sim. `modehb/ops_norm_act.py:187` (ZCAWhitening), `:258` (DecorrelatedBatchNorm), `:318` (make_norm); ajuste em `run_search.py:1062` (`_fit_zca`). Morfismo exato só em eval: `backbones/morphisms.py:21-24, 242-259` | **Gene** por camada `norm ∈ {none, batch, zca, dbn, batch_zca, batch_dbn}` | **Só herdado**. A semente RS seed01 tem `batch_dbn` no bloco 1. Camadas novas (`deepen_*`) entram com `norm=none`, e o ZCA não é re-ajustado no filho. A régua v2 tem `whiten: none` | **Sim.** Nas camadas ativas, ~80% têm alguma norm e ~60% são ZCA/DBN (hybrid: zca 462, dbn 451, batch_dbn 461, batch_zca 442, batch 485, none 561). A semente LEMONADE `seed_B` usa zca |
| 2 | SoftHebb | Sim. `modehb/softhebb.py:74` (update), `:186` (`fit_softhebb`); chamado em `run_search.py:934-939`, entre o SPiFiL e o AE local | **Gene** booleano `softhebb` por camada | **Não.** `LayerSpec` (`backbones/spec.py:66`) não tem o campo, e o crescimento nunca chama o SoftHebb. Só sobrevive o efeito já "assado" nos pesos da semente | **Sim.** hybrid 1415/2862 camadas (49%), LEMONADE 774/2084 (37%). Config fixa: 1 época, lr 0,01, T=1 |
| 3 | Inception | Sim. `ops_norm_act.py:358` (InceptionConv); SPiFiL por ramo em `spifil_ops.py:54`; morfismo widen por ramo em `morphisms.py:19, 129-146` | **Gene** `operator ∈ {conv, dwsep, inception}` | **Parcial.** Só existe se a semente for Inception, e então é alargado. Os canais novos de Inception e dwsep **não recebem SPiFiL**: caem em init PyTorch (`growth_chain.py:663-669`; visto no calib RS B1g0c03). `deepen_residual_zero` sempre cria `conv` | **Sim.** hybrid 947/2862 camadas (33%), LEMONADE 573/2084 (27%) |
| 4 | Concatenação da camada anterior na próxima (DenseNet / skip-concat) | **Fora da busca.** `ga_search/spifil_residual.py:18-33` (`ResidualBlock = cat([x, act(conv(x))])`), usado por `ga_search/evaluate.py`; grade E1/E2/E3 em `experiments/ray/runners/spifil_dense.py` + `configs/model/spifil_e*_*.yaml` (não versionados). Não existe em `ops_norm_act` nem em `backbones` | Nenhum (nem gene nem opção) | **Não** (não há morfismo de concat) | **Não.** O mais próximo é a leitura `cat_tta_pw` (concat de GeM por bloco, `svm_on_demand.py:95-125`), que é só leitura de avaliação |
| 5 | Covariância para detectar e repor filtros ruins/redundantes | **Só diagnóstico.** `modehb/init.py:291-330` (`kernel_diagnostics`: `filter_redundancy` = fração de pares com \|cos\|>0,9, `dead_fraction`, erank), logado em `run_search.py:1044-1054`; `embedding_eval.py:58, 355` só emite um WARN de redundância do embedding. **Não há substituição** de filtros | Nenhum | **Não** | **Não.** Só aparece nas métricas do W&B |
| 6 | Residual | Sim. GA: flag `residual` no ConvBlock (`ops_norm_act.py:408`). Crescimento: `morphisms.py:463` (`deepen_residual_zero`), `:491` (`add_skip_zero`), `:429` (`deepen_identity`); spec `RESIDUALS = none/identity/project/gated` (`spec.py:39`) | **Gene** `residual` (bool) no GA. No crescimento é uma **operação** sorteada (`ops_weights`: widen 0,55, deepen_res0 0,20, add_skip 0,15, deepen_id 0,10; `growth_chain.py:78`) | **Sim, é o principal.** RS: B0 (0,534 teste) → B1 = `deepen_residual_zero` silu (0,577) → B2 = mais um `deepen_residual_zero` (0,592 em f15 / **0,529** em f45). Os nós de calibração usam widen e `deepen_residual_zero` | **Sim.** Gene no hybrid (1123/2862 = 39%) e no LEMONADE (35%). Operador `residual` nos morfismos do LEMONADE (`search_v3_lemonade.yaml:220`) e `add_residual` no PNAS-CS (`pnas_cs.yaml:217`) |
| 7 | Variação do SSL (init sempre SPiFiL) | Sim: `ssl_trainers.py:261`, com SimCLR (NT-Xent), LeJEPA (inv + RealSIGReg), AE, mais `weak_sigreg`; `modehb.py:94-95` define `SSLS = simclr/lejepa/ae` e `REGS = none/sigreg/weak_sigreg`. **VICReg, Barlow e DINO não existem** em `ga_search` | **Gene** global `ssl` × `reg` no GA. No crescimento é **opção** `ssl: lejepa \| none` (a ablação `fix5_nossl` está pendente) | **Só LeJEPA+SIGReg**, curto (10 épocas em f15; +30 com o fix6). Init dos canais novos = `spifil0`, com fallback PyTorch em dwsep/inception (item 3) | **Sim.** hybrid: lejepa 293, simclr 170, ae 162; LEMONADE: 279/68/61; PNAS-CS fixa lejepa+sigreg. `init=spifil` em 100% dos candidatos (`random_baseline: false`) |
| 8 | Data augmentation durante o SSL | Sim, **fixa**: `ssl_trainers.py:77-86` (`AugConfig`): RandomResizedCrop escala (0,3–1,0), razão 3/4–4/3, hflip 0,5, vflip 0,5, brilho ±0,2, contraste ±0,2, jitter por canal 0,05 (LAB). V = 4 vistas (LeJEPA) ou 2 (SimCLR) | **Nenhum.** `aug_cfg` nunca é passado (só o default). Não varia por base, por fase nem por gene | Sim, com o default e as mesmas 4 vistas | Sim, com o mesmo default em todos os runs. Não há multi-crop global/local, blur, solarize nem jitter de cor forte |
| 9 | Cabeça ≠ GAP (2ª ordem, cov, momentos) | Sim. `pooling_head.py:79` (EmbeddingHead: gap, gem, gem_learn, meanstd, **moments**, combo), `:321` (**CovReadout**: sqrtm de cov k×k por bloco + PCA); `readout_x/eval_backbone_head.py` (so2 / cross-block) | **Gene** `pooling ∈ {gap, gem, moments}`. A leitura `cov_head` é **opção da fitness** (`svm_fitness.readout: cov_head, cov_k 64, scaler standard`) | **Sim, cabeça `moments` 2560-d** (herdada da semente RS). Decisão por régua ORIG `head` (`fitness_v2.py:27`); a cov só vai para `aux`. `readout_x` (2ª ordem) foi avaliado depois do fato | **Sim.** Gene: moments 224 / gem 217 / gap 184 (hybrid). **A fitness da busca do GA é a `cov_head`** (2ª ordem), enquanto o ranking reportado é ORIG `head` (o próprio código registra ρ(val COV, teste ORIG) = 0,09) |
| 10 | TTA no teste | Sim. `pooling_head.py:180` (rot90 × hflip, até 8), `:464` (`encode_tta`); leituras `cov_head_tta`, `gem_tta_pw`, `cat_tta_pw` em `svm_on_demand.py:162-180`; ablação `ablations/tta.yaml` | **Opção**, travada em `tta: 1` | **Não.** A `fitness_v2` exige `tta = 1` (REQUIRED, `fitness_v2.py:27`), e o canal de teste também usa tta 1 | **Não.** `tta.search: 1`; o monitor de teste (`monitor_report.py:168/175`) usa `head` e `cov_head` **sem** TTA. Só os resultados antigos do SPiFiL discovery (`gem_tta_pw`) usaram TTA |
| 11 | 100% dos dados nas 5 bases prioritárias | Configs `datasets/{flowers, remote_sensing, ham10000, aid, coconut_trees}.yaml`: `pcts: [100]`, split 1. Teste do monitor com o treino **inteiro** (n_train 1922/682/4129/6996/9511) | Opção | **Só RS e flowers** (fila `growth/fila_growth.json`). ham, aid e coconut não estão na fila do crescimento | **Hybrid:** as 5 bases têm estado, nenhuma `done` (todas no bracket 0, exceto o LEMONADE RS, que terminou); LEMONADE aid/coconut quase vazios. **Mas na busca:** (a) 1 época = ≤2048 imagens (`max_images_per_epoch`), ou seja ~21% do coconut e ~29% do aid por época; (b) a SVM da fitness usa `max_train: 4000` (subamostra em aid, coconut e ham); (c) o SPiFiL vê 1–32 imagens por classe |

### Fatos que mudam a leitura da tabela

**1. A cadeia ainda não sobe com segurança, e o SSL tende a destruir.**
- Nos 4 nós de calibração, o WiSE escolheu **α = 0** (o filho é o pai; SSL piora). Exemplo: flowers B1 com κ_val 0,666 em α=0 contra 0,590 em α=1.
- No RS, f45 < f15 (0,529 contra 0,592 no teste). Por isso nasceu o fix6 (retomada).
- O AE local dos blocos novos sai com `effective_rank_ratio` ≈ 0,01, isto é, colapso da reconstrução local.

**2. Desalinhamento de régua no GA.** A seleção usa `cov_head` + StandardScaler, enquanto o resultado reportado é `head` sem scaler. A cadeia já corrigiu isso (régua ORIG em tudo).

**3. A 2ª ordem ajuda a semente, mas não o crescimento.** Na `readout_x` (SVM original):
- flowers B0: GAP 0,635 → 2ª ordem 0,784 (teste);
- parasitas: +3 a 6 pontos;
- RS B0: 0,534 → 0,605;
- RS B1/B2 crescidos: a 2ª ordem **não** ajuda (0,577 → 0,555/0,577; B2 f15 0,592 → 0,550).

## 2. Literatura

### 2.1 Crescimento progressivo e morfismos
- **Net2Net** (Chen, Goodfellow, Shlens, ICLR 2016, arXiv:1511.05641) e **Network Morphism** (Wei et al., ICML 2016, arXiv:1603.01670) são a base do `widen`, do `deepen_identity` e da preservação exata.
  - Os dois mostram que preservar a função só garante o **ponto de partida**. O ganho vem de treinar depois, e um pai bem treinado + morfismo + treino curto tende a ficar perto do pai.
  - Net2Wider precisa de **ruído/quebra de simetria** nos neurônios replicados. Aqui a simetria é quebrada pelo SPiFiL nos novos canais com saída zero, o que é correto.
- **LEMONADE** (Elsken, Metzen, Hutter, ICLR 2019, arXiv:1804.09081): morfismos + herança lamarckiana + multiobjetivo, com "approximate morphisms" para **encolher**.
  - Já está no GA (deepen, widen, residual, kernel), sem os operadores de remoção ou compressão.
- **Splitting Steepest Descent** (Liu, Wang, Wu, Liu, NeurIPS 2019, arXiv:1910.02366) e **Firefly** (Wu, Liu, Stone, Liu, NeurIPS 2020, arXiv:2102.08574): escolhem **onde** crescer pela curvatura ou pelo gradiente (o neurônio cuja divisão mais reduz a perda).
  - Aqui o "onde" é sorteado (θ + DE) e testado depois, o que é caro e explica a taxa alta de α=0.
- **GradMax** (Evci et al., ICLR 2022, arXiv:2201.05125): inicializa os pesos novos com saída zero e entrada pelo SVD que **maximiza a norma do gradiente** dos novos pesos.
  - É exatamente a lacuna do nosso `*_zero`. Com saída zero, o gradiente dos pesos de entrada também é zero no passo 0. Só a saída aprende primeiro, e com `lowlr`/WiSE o filho volta ao pai.
- **AutoGrow** (Wen et al., AAAI 2020, arXiv:1906.02909): cresce profundidade enquanto houver ganho e **para na saturação**; relata que init com "zero" ou identidade + treino curto entre crescimentos pode estagnar.
  - A política "Rapid growth" (crescer cedo, treinar bastante no fim) supera crescer devagar.
- **Plasticidade:**
  - Ash & Adams (NeurIPS 2020, arXiv:1910.08475): o *warm start* gera um gap de generalização, e o *shrink-and-perturb* recupera.
  - **ReDo / dormant neurons** (Sokar et al., ICML 2023, arXiv:2302.12902) e Continual Backprop (Dohare et al., Nature 2024): reinicializar unidades **dormentes** restaura plasticidade. É o análogo direto do item 5.
- **Compound scaling** (EfficientNet, Tan & Le 2019, arXiv:1905.11946) é a base do PNAS-CS. **MO-DEHB** (Awad et al., arXiv:2305.04502).

### 2.2 Pooling de 2ª ordem
- **B-CNN** (Lin, RoyChowdhury, Maji, ICCV 2015, arXiv:1504.07889): pooling bilinear (produto externo) com signed-sqrt + L2. Forte em fine-grained (flowers, aves).
- **MPN-COV** (Li, Xie, Wang, Gao, ICCV 2017, arXiv:1703.08050): a potência de matriz ½ da covariância é o fator decisivo, mais que a cov crua.
- **iSQRT-COV** (Li et al., CVPR 2018, arXiv:1712.01034): sqrtm por Newton-Schulz, treinável fim a fim e rápida em GPU.
- **DeepO2P / matrix backprop** (Ionescu et al., ICCV 2015, arXiv:1509.07838) e "Statistically-motivated second-order pooling" (Yu & Salzmann, ECCV 2018, arXiv:1801.07492).
  - Mostram que a 2ª ordem é mais útil quando **treinada junto** com o backbone e com redução de dimensão antes (PCA/1×1).
  - Isso explica por que a `cov_head` sem treino ajuda a semente, mas não o B1/B2: os blocos novos foram treinados para a cabeça `moments`, não para a covariância.

### 2.3 SSL
- **LeJEPA / SIGReg** (Balestriero & LeCun, arXiv:2511.08544, nov/2025): embeddings isotrópico-gaussianos minimizam o risco downstream. SIGReg = projeções 1D + teste de Epps-Pulley, com um único λ.
  - Estável de ResNets a ViT-g, mas os resultados são com **multi-crop** (vistas globais + locais) e treino longo.
- **VICReg** (Bardes, Ponce, LeCun, ICLR 2022, arXiv:2105.04906) e **Barlow Twins** (Zbontar et al., ICML 2021, arXiv:2103.03230): regularizam variância e covariância do embedding.
  - É o mesmo princípio de "decorrelacionar" que ZCA/DBN fazem nos mapas. São baratos e funcionam com batches pequenos e redes pequenas.
- **SimCLR** (Chen et al., ICML 2020, arXiv:2002.05709) mostra que **crop + color jitter forte** é a combinação crítica: sem cor, a rede resolve a tarefa pela histograma.
- **DINO** (Caron et al., ICCV 2021, arXiv:2104.14294) e **SwAV** (arXiv:2006.09882) introduzem multi-crop 2×224 + N×96.
- O nosso `AugConfig` é **fraco** nos dois eixos: brilho/contraste ±0,2 e jitter 0,05, sem multi-crop. Isso combina com o SSL que "não ensina nada novo" e com o WiSE devolvendo α=0.
- **W-MSE** (Ermolov et al., ICML 2021, arXiv:2007.06346): SSL por **branqueamento** do embedding. É a ponte natural entre o nosso ZCA e o SSL.

### 2.4 DenseNet / Inception
- **DenseNet** (Huang et al., CVPR 2017, arXiv:1608.06993): a concatenação reaproveita features e dá gradiente direto às camadas antigas.
  - Para crescimento, a concatenação **é naturalmente um morfismo**: canais novos com peso zero no consumidor, sem mexer nos antigos. O SPiFiL já pensado como "C_l = C_{l-1} + N_l" (`spifil_residual.py`) é exatamente isso.
- **Inception / GoogLeNet** (Szegedy et al., CVPR 2015, arXiv:1409.4842): multiescala por ramos e 1×1 de redução. Já é gene. Falta SPiFiL nos canais novos dos ramos no crescimento.

### 2.5 SoftHebb e whitening
- **SoftHebb** (Journé et al., ICLR 2023, arXiv:2209.11883): soft-WTA Hebbiano sem feedback, até 5 camadas, com linear probe (CIFAR-10 80,3%, STL-10 76,2%).
  - Depende de **whitening da entrada** e de **camadas largas com pooling triangular**. É competitivo com poucos rótulos, mas fraco quando seguido de backprop agressivo, que apaga os filtros.
- **ZCA / whitening:**
  - Coates & Ng (AISTATS 2011) mostram que o whitening é o fator mais importante para features não supervisionadas rasas;
  - Kessy, Lewin, Strimmer (Am. Stat. 2018, arXiv:1512.00809) mostram que o ZCA é o whitening que menos distorce a entrada.
- **Decorrelated BN** (Huang et al., CVPR 2018, arXiv:1804.08450): ZCA por grupo dentro da rede; melhora condicionamento e convergência.
  - No crescimento, DBN e ZCA só são exatos em eval (running stats). O artigo recalcula o branqueamento no treino, o que muda a função do pai ao alargar.
  - Isso é coerente com os OOMs e as inexatidões já registradas (DBN do bloco 1 do RS).

### 2.6 O que a literatura sugere para "B0 + resíduo, saturação B0→B4 sem piorar"
1. **Garantir monotonicidade por construção.** O nosso esquema (morfismo + WiSE com α=0 + aceitação κ_filho ≥ κ_pai) já garante "não piorar" na **validação**. No teste, o RS mostra B2 f45 < B1, porque o WiSE/seleção sobreajusta a validação (159 imagens).
   - Recomendação: aceitar só com margem δ > ruído (bootstrap da val) e usar várias seeds.
2. **Fazer os blocos novos aprenderem.**
   - Init com gradiente não nulo (GradMax ou Firefly) em vez de saída zero pura.
   - Alternativa: ReZero/SkipInit (Bachlechner et al., arXiv:2003.04887; De & Smith, arXiv:2002.10444), com escala α inicial = 0 aprendível e lr cheio só nela.
   - Hoje a saída zero + `lowlr` + 10 épocas leva a α=0 e estagnação.
3. **Crescer menos vezes, treinar mais em cada fase** (AutoGrow, "rapid growth") e **retomar em vez de refazer** (o fix6 vai nessa direção).
4. **Escolher onde crescer por sinal**, não por sorteio: splitting/Firefly (curvatura), GradMax (gradiente) ou a covariância dos mapas (canais com alta redundância ou dormentes → split/reinit via SPiFiL). Isso une os itens 5 e 6.
5. **Contra a saturação:** shrink-and-perturb ou ReDo nos filtros dormentes do pai antes de crescer (Ash & Adams; Sokar et al.). Sem isso, o pai treinado vira um mínimo do qual o filho não sai.
6. **Readout coerente com o crescimento:** se a 2ª ordem é o melhor readout da semente, crescer **com a cabeça de covariância treinável** (iSQRT-COV) em vez de moments, e decidir na mesma régua.

## 3. Combinações que faltam testar (prioridade)

| P | O quê | Por quê | Custo e onde |
|---|---|---|---|
| 1 | **Ablação `ssl: none`** (RS_B1_fix5_nossl, pendente) + **augment forte e multi-crop** no SSL do crescimento (cor ±0,4, blur, 2 globais + 4 locais 96 px) | Os 4 nós de calibração deram α=0, e o SSL atual piora. É preciso saber se o problema é o SSL ou a augment fraca (SimCLR, DINO, LeJEPA) | Baixo. `AugConfig` já aceita dict; falta só passar `aug_cfg` em `growth_chain._global_ssl` |
| 2 | **SPiFiL nos canais novos de dwsep e Inception** (hoje usam init PyTorch) | Viola "init sempre SPiFiL", e o GA tem 33% de camadas Inception | Médio, em `NewFilterInit` (`growth_chain.py:663-669`) |
| 3 | **Init dos novos com gradiente** (GradMax/SVD ou ReZero α aprendível) em vez de saída zero pura | Saída zero + lowlr leva a α=0 e estagnação (GradMax, AutoGrow) | Médio, em `morphisms.deepen_residual_zero` / `widen` |
| 4 | **Operador de concat DenseNet** como morfismo (`add_dense_concat`: novos N_l concatenados, consumidor com colunas zero) | É morfismo exato e não mexe nos filtros SPiFiL antigos (alinha com "backprop soma, nunca desfaz") | Médio. `spifil_residual.py` já tem o bloco; falta spec/morfismo |
| 5 | **Detecção e reposição de filtros por covariância** (pares \|cos\|>0,9 ou dormentes → re-fit SPiFiL nos resíduos; ReDo/RePr) como operação do crescimento e mutação do GA | O diagnóstico já existe (`init.py:291`); falta agir | Médio |
| 6 | **Régua única no GA**: fitness ORIG `head` (como na cadeia) ou cabeça cov **treinável** (iSQRT) decidida na mesma régua | ρ(COV-val, ORIG-teste) = 0,09: hoje o GA otimiza uma coisa e reporta outra | Baixo (config), mas muda a busca |
| 7 | **TTA 8 (rot × flip) no teste** em todos os aceitos da cadeia e na frente do GA (coluna separada, sem tocar o SVM C=100) | Já implementado; ganho barato e invariante a orientação (RS, flowers) | Baixo |
| 8 | **VICReg/Barlow** como 4º SSL (gene) e **W-MSE** com o ZCA | Decorrelacionar é coerente com ZCA/DBN e SIGReg; barato em batch 64 | Médio, em `ssl_trainers` |
| 9 | **SoftHebb no crescimento** (canais novos: SPiFiL → SoftHebb → AE) e **whitening (ZCA) nas camadas novas** | Hoje SoftHebb e whitening só entram pela semente | Médio |
| 10 | **Cadeia nas 5 bases** (ham, aid, coconut além de RS e flowers) e **época = treino inteiro** (`max_images_per_epoch: null`) + SVM da fitness sem `max_train` nas bases grandes | "100%" hoje é 100% do split, mas a busca vê ≤2048 imagens por época e ≤4000 na SVM | Alto (GPU) |
| 11 | **Aceitação com margem estatística** (δ = 1 desvio do bootstrap da val) + 3 seeds | B2 f45 < B1 no teste mesmo aceito na val | Baixo |

Fontes web consultadas: [LeJEPA arXiv:2511.08544](https://arxiv.org/abs/2511.08544), [SoftHebb arXiv:2209.11883](https://arxiv.org/abs/2209.11883v2), [GradMax arXiv:2201.05125](https://arxiv.org/abs/2201.05125v2), [Firefly NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/fdbe012e2e11314b96402b32c0df26b7-Abstract.html). As demais citações (IDs arXiv) são de memória e não foram reconferidas nesta sessão.
