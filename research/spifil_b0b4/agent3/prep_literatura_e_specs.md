# Agente 3 — Preparação: literatura, auditoria do crescimento e especificações B0–B4

> Fase de **preparação** (MISSION.md, seção Agente 3). Este documento cobre só literatura e desenho. Nada aqui foi treinado nem rodado em GPU, e nenhum código de produção foi alterado.
> Data: 16/11/2026. Tudo o que for número de custo é **estimativa analítica** (script descartável em scratchpad, conv densa k3, entrada 224, sem BN), não medição.
> Rótulos usados: **[paper]** = lido no texto do artigo; **[código]** = lido no repo (arquivo:linha); **[proposta]** = desenho deste agente, ainda não testado; **[hipótese]** = afirmação a verificar.

---

## 0. Resumo executivo

1. **No repo, a preservação exata de função só existe para blocos "planos"** (`conv` denso, `groups=1`, `norm=none`, sem skip) e para camadas inseridas como identidade/ReLU ou como residual zero com SiLU/GELU. Qualquer bloco com BN/ZCA/DBN, residual, dwsep, depthwise ou inception **interrompe a cadeia**. A partir dele, `grow_blocks` re-inicializa (SPiFiL + AE) tudo o que vem depois. Ver §2.
2. **A cabeça 1280 nunca é herdada.** `EmbeddingHead` é recriada com pesos aleatórios a cada filho (`run_search.py:700-702`). Por isso o embedding 1280 do filho no passo 0 **não** é o do pai, mesmo quando o encoder é preservado exatamente. O `preservation_error` atual mede só os canais antigos ao fim do prefixo, em 8 imagens de validação. Proponho herdar a cabeça com colunas zero para os canais novos, o que é exato com qualquer pooling, porque o conv 1×1 vem antes do pooling.
3. **Incompatibilidades entre o código atual e a missão:**
   - os orçamentos do PNAS-CS são 0,5/1/2/3,5/5,5M, mas a missão pede 0,3/0,8/1,5/3/5,5M;
   - `cap_factor=1.05` permite 5,775M, e a missão proíbe passar de 5,5M;
   - o campo `flops` gravado é na verdade **MACs** (`run_search.py:334-360`), então FLOPs = 2×.
4. **A cabeça 1280 é a maior parcela do orçamento em B0.** Com GAP/GeM, a cabeça custa ≈ C_last×1280 parâmetros (112 canais ⇒ 145k = metade do B0). `moments` custa 1/4 disso, e uma cabeça bilinear compacta com 1×1 para 50 canais + triângulo superior (1275+5 = 1280) custa ~1/20. Como a troca de cabeça muda o orçamento do encoder, toda comparação de cabeças precisa de **dois controles**: mesmo encoder, e mesmo total de parâmetros.
5. Proponho uma família de referência B0→B4 **plana e contida** (o filho contém o pai), com custo calculado (§4). Também especifico BackboneSpec/GrowthSpec (§3), o DAG de linhagens (§5), os experimentos A–H (§6) e as cabeças (§7).

---

## 1. Literatura (7 artigos)

Os textos foram baixados do arXiv/PMLR e lidos por extração local (pypdf). As citações literais estão entre aspas.

### 1.1 Net2Net — Chen, Goodfellow, Shlens (ICLR 2016, arXiv 1511.05641)

**Princípio [paper]**
- O aluno maior começa computando **a mesma função** do professor, por meio de "function-preserving transformations".
- **Net2WiderNet** usa um mapeamento aleatório g de replicação de unidades. Os pesos de entrada são copiados, e os pesos de saída de cada réplica são divididos pelo fator de replicação 1/|{x | g(x)=g(j)}|. Cada unidade precisa ser replicada pelo menos uma vez. Adiciona-se "a small amount of noise" às réplicas para quebrar a simetria, o que deixa de ser exato.
- **Net2DeeperNet** insere uma camada inicializada como identidade. Isso só vale se φ(Iφ(v)) = φ(v), o que é verdade para ReLU e falso, por exemplo, para a sigmoide.
- Em conv, o kernel identidade é o próprio filtro identidade.
- Com BatchNorm, é preciso ajustar escala e viés para desfazer a normalização, o que "requires running forward propagation through the network on training data".
- O Deeper não precisa de ruído.
- Os autores reconhecem que a fatoração geral de camadas não lineares (ReLU + BN) é difícil, por isso ficam na inserção de identidade.

**Aplicação a B0–B4 SPiFiL**
- O repo **não** usa replicação. Usa a variante "zero-pad do consumidor" (também citada pelo LEMONADE): os filtros novos são ajustados pelo SPiFiL e as colunas do consumidor que leem esses canais começam em zero.
- Vantagem: os canais novos trazem **informação nova** (protótipos SPiFiL) em vez de cópias, e não é preciso ruído.
- Desvantagem: no passo 0 os canais novos não influenciam nada a jusante. O gradiente que chega aos filtros novos vindo do consumidor é zero até as colunas saírem de zero. As colunas recebem gradiente porque os canais novos têm ativação ≠ 0.
- O Deeper com ReLU é exato no repo porque a entrada vem de um ReLU (≥ 0) e o max-pool preserva o sinal.
- **[proposta]** Oferecer as duas variantes no `GrowthSpec.operation`:
  - `widen_zero`: o que existe hoje, exato;
  - `widen_replicate`: Net2Net com divisão pelo fator, exato sem ruído e com ruído controlado como opção.
  - Comparar as duas no experimento A.

**Limitações**
- Só aumenta a capacidade.
- A exatidão se perde com ruído, com BN em modo treino, com ativações não idempotentes e com `cut`.
- Os autores observam que, contra uma baseline forte com BN, a vantagem de convergência diminui. O artigo não comprova que a função preservada leve a um mínimo melhor, só a um treino mais rápido.

### 1.2 LEMONADE — Elsken, Metzen, Hutter (ICLR 2019, arXiv 1804.09081)

**Princípio [paper]**
- Busca evolutiva multiobjetivo com **herança lamarckiana**: o filho herda os pesos treinados do pai.

Operadores de morfismo exato (NM):
1. inserir um bloco Conv-BN-ReLU com conv identidade e BN inicializado com a média e variância móveis, o que dá identidade (ReLU idempotente sobre entrada de ReLU);
2. aumentar filtros com zero-pad do conv seguinte, ou Net2Wider;
3. adicionar skip:
   - por concatenação, com zero-pad a jusante;
   - por soma como **combinação convexa (1−λ)x + λy com λ aprendível iniciado em 0**.

Morfismos aproximados (ANM), que reduzem capacidade:
- (i) remover uma camada ou um skip;
- (ii) podar 1/2 ou 1/4 dos filtros de um conv;
- (iii) trocar conv por depthwise-separable.

Seleção:
- Os objetivos são separados em **f_exp** (erro, caro) e **f_cheap** (parâmetros etc.).
- Os pais são amostrados com probabilidade inversamente proporcional a uma **KDE sobre f_cheap**, e uma segunda amostragem escolhe quais filhos são treinados.

**Aplicação**
- O repo implementa a herança (prefixo reutilizável) e os morfismos `deepen/widen/residual/kernel` (`lemonade.py:124-170`).
- Ficam de fora:
  - os ANMs de redução, exceto o `cut` de `external_seed`;
  - o skip por combinação convexa com λ=0, que seria a forma **exata** de "adicionar residual a uma camada existente";
  - a KDE (o repo usa MO-DEHB).
- **[proposta]** Para B0–B4:
  - `add_skip_convex` (exato) para o experimento E;
  - `prune` e `to_dwsep` (ANM) só para linhagens de **redução** (controle OFA, §5).

**Limitações**
- NMs só aumentam a capacidade, então a população incha sem ANM.
- ANMs aumentam o erro inicial.
- A herança cria **dependência de caminho**: o filho fica perto do ancestral. Por isso a missão exige o controle herança × reinicialização na mesma arquitetura.
- Os resultados foram em CIFAR-10 supervisionado. Não há evidência no regime SSL/SPiFiL de poucos dados.

### 1.3 EfficientNet — Tan & Le (ICML 2019, PMLR v97 tan19a)

**Princípio [paper]**
- Escalar só uma dimensão (profundidade d, largura w ou resolução r) **satura rapidamente**: "accuracy gain quickly saturate"; ResNet-1000 ≈ ResNet-101.
- A proposta é o compound scaling: d = α^φ, w = β^φ, r = γ^φ, sujeito a α·β²·γ² ≈ 2, de modo que os FLOPs crescem ≈ 2^φ.
- α, β, γ saem de um **grid search pequeno em B0 com φ=1** (α=1,2; β=1,1; γ=1,15) e depois ficam fixos para B1–B7.
- A base B0 foi encontrada por NAS.

**Aplicação**
- Na missão a resolução é fixa (224), então γ=1. Os FLOPs passam a escalar ≈ d·w², e o vínculo vira α·β² ≈ 2.
- **[código]** `pnas_cs.fit_level` (`pnas_cs.py:287`) **não impõe** α·β² ≈ 2. Ele escolhe φ para acertar o **orçamento de parâmetros** (busca em φ seguida de bisseção na largura). Na prática, (α, β) só definem a *direção* log α / log β, o que é equivalente e mais adequado a orçamentos por parâmetros.
- **[proposta]** Seguir o EfficientNet:
  1. fazer o grid de (α, β) **apenas na transição B0→B1, por validação**;
  2. fixar a direção para B2–B4;
  3. registrar a direção ótima por domínio.

  Isso reduz o custo de busca e é o braço "compound" do experimento C.

**Limitações**
- A base B0 domina o resultado e foi achada por NAS supervisionado.
- O benefício da profundidade diminui.
- FLOPs não são latência.
- Com resolução fixa perde-se um eixo, que o próprio artigo mostra ser útil.
- Os resultados são do regime ImageNet supervisionado. No nosso regime (poucos dados, SSL, SVM linear), a saturação pode chegar bem antes **[hipótese]**.

### 1.4 Once-for-All — Cai et al. (ICLR 2020, arXiv 1908.09791)

**Princípio [paper]**
- Treina-se uma rede grande e depois **progressive shrinking**: os sub-redes menores compartilham pesos com a rede grande.
- A elasticidade é aplicada nesta ordem: resolução (sempre), kernel, profundidade e largura.
- **Kernel elástico**: o centro do 7×7 serve de 5×5 e o do 5×5 de 3×3, por meio de *matrizes de transformação* (25×25 e 9×9).
- **Profundidade elástica**: mantêm-se as **primeiras D** camadas de cada unidade e pulam-se as últimas.
- **Largura elástica**: os canais são ordenados por importância (**norma L1** dos pesos) e os mais importantes são mantidos.
- Usa destilação da rede maior.
- O espaço tem > 10^19 sub-redes.

**Aplicação**
- É a direção oposta ao crescimento: B4 → B3 → … → B0.
- **[código]** `keep_filters` (`external_seed.py:343`) já faz algo parecido com a ordenação de canais do OFA, mas usa a norma L2 das colunas do consumidor em vez da L1 dos filtros.
- **[proposta]** Criar uma **linhagem-controle "shrink"**:
  1. treinar B4;
  2. extrair B3..B0 por primeiras-D camadas por estágio + top-canais;
  3. comparar com a linhagem crescida de mesmo orçamento.

  Isso responde se crescer de baixo para cima é melhor do que podar de cima para baixo. A cabeça 1280 é refeita, ou herdada nas colunas mantidas.

**Limitações**
- Exige treinar uma super-rede com PS + KD, o que é caro.
- Os sub-redes extraídas sem re-treino herdam pesos treinados para o contexto da rede grande.
- O SPiFiL não foi pensado para pesos compartilhados.
- Não preserva função.

### 1.5 FPN — Lin et al. (CVPR 2017, arXiv 1612.03144)

**Princípio [paper]**
- Uma via **top-down** com upsampling ×2 (vizinho mais próximo) é somada elemento a elemento a **conexões laterais 1×1** sobre os mapas bottom-up C2..C5.
- O começo é um 1×1 sobre C5, e um 3×3 "suaviza" cada nível somado.
- A dimensão é fixa (**d = 256**) em todos os níveis, e as camadas extras **não têm não linearidade** ("minor impacts").
- O objetivo são mapas semanticamente fortes em todas as escalas a custo marginal.

**Aplicação**
- O Agente 1 investiga por que mapas intermediários com covariância vão bem e o 1280 nem sempre.
- **[proposta]** Uma cabeça **FPN-lite global**:
  - laterais 1×1 → d=128 nos 3 estágios finais (pós-pool);
  - soma top-down **sem** o 3×3, que é o que custa;
  - GeM por nível, concatenação de 384 dimensões e linear → 1280.
- Custo: +0,5–0,6M parâmetros e +30–70M MACs (§7). É o braço "multiescala" do experimento H.

**Limitações**
- Foi desenhado para detecção densa. Na classificação global, a via top-down injeta semântica em mapas que vão ser pooled de qualquer forma, então a vantagem não está garantida **[hipótese]**.
- O 3×3 em resoluções altas é caro.
- A linear 384→1280 é uma *expansão*: não cria informação, só se ajusta ao SVM linear.

### 1.6 Bilinear CNN — Lin, RoyChowdhury, Maji (ICCV 2015, arXiv 1504.07889)

O PDF atual dessa ID é a extensão TPAMI, que inclui o conteúdo do ICCV.

**Princípio [paper]**
- A imagem é representada pelo **produto externo das features de dois streams, somado (pooled) sobre as posições**.
- Isso captura interações locais de forma invariante à translação. É uma representação **orderless** de textura, treinável de ponta a ponta.
- O caso simétrico (f_A = f_B) equivale ao pooling de segunda ordem (O2P).
- Usa normalização **raiz quadrada com sinal + L2**.
- A dimensão é quadrática: **262k** para VGG com 512 canais. As features são muito redundantes e podem ser reduzidas em uma ordem de grandeza sem perda.
- Cita o pooling bilinear entre camadas diferentes (Liu et al.).

**Aplicação**
- O `CovReadout` do repo (`pooling_head.py:321`) já é uma leitura de segunda ordem: PCA para k=64, covariância, sqrtm, triângulo superior, depois signed-sqrt + L2 por grupo, com dimensão ≈ 17.920.
- A missão proíbe usá-lo como **substituto silencioso** da cabeça 1280.
- **[proposta]** Fazer a segunda ordem **dentro** da cabeça, com dimensão fixa 1280 (§7 `bilin50`):
  - 1×1 → 50 canais;
  - covariância 50×50 com o triângulo superior (1275) + 5 médias;
  - signed-sqrt + L2.
- Variante assimétrica entre estágios (abstrato × intermediário reamostrado) como braço exploratório.

**Limitações**
- Dimensão quadrática, por isso a redução para 50 canais é forte.
- Depende de normalização.
- A sqrtm melhora, mas exige `eigh`, que já é o gargalo no `CovReadout`.
- O SVM linear **sem scaler** (R1/R2) é sensível à escala, então a normalização tem que fazer parte da cabeça e ser registrada.

### 1.7 Compact Bilinear Pooling — Gao et al. (CVPR 2016, arXiv 1511.06062)

**Princípio [paper]**
- O pooling bilinear é um kernel polinomial de 2ª ordem, e por isso pode ser aproximado por mapas explícitos de baixa dimensão:
  - **Random Maclaurin (RM)**: 2cd parâmetros aleatórios fixos;
  - **Tensor Sketch (TS)**: 2c parâmetros (hash + sinal), custo O(hw(c + d log d)) via FFT.
- Com d ≈ 8k, a perda é de "only 2% of the bilinear feature dimension" com desempenho ≈ igual (redundância de 98%). Antes do fine-tune a perda é de 0,6–1,9 pp, e ela some depois do fine-tune.
- TS é mais lento que o bilinear puro, por causa da constante do FFT.
- PCA para a mesma dimensão é pior que TS.

**Aplicação**
- **[proposta]** `ts1280`: TS por posição com d = 1280, sum-pool, signed-sqrt + L2.
  - Não tem parâmetros treináveis e custa ≈ 20M MACs a 14×14.
  - É a cabeça mais barata em parâmetros que atinge exatamente 1280.
  - Variante: 1×1 → 128 canais antes do TS.

**Limitações**
- d = 1280 sobre C = 300–400 canais corresponde a 0,7–1,4% de C², **abaixo** da faixa em que o artigo mostra paridade (a curva de erro sobe para d pequeno).
- O sketch é aleatório, então exige semente fixa e registrada.
- FFT em bf16 não é confiável, então deve rodar em fp32 (regra HPC da missão).

---

## 2. Auditoria do que já existe (`ga_search/modehb/`)

### 2.1 `external_seed.py`

| Peça | O que faz | Preserva função? | Condições / falhas conhecidas |
|---|---|---|---|
| `widen_block` (`:357`) | Copia os filtros mantidos (zero-pad central se o kernel cresce); filtros novos = `new_weight` (SPiFiL) ou zeros; colunas de **entrada** novas = 0 | **Exato** nos canais mantidos (com `keep` = todos e kernel crescendo em passo par) | Só para `_plain` (`:325`): `Conv2d`, `groups==1`, com bias, `norm` Identity, `skip` None. O kernel não pode diminuir (`:372`). Com `keep` < todos (*cut*) **não** é exato. |
| `keep_filters` (`:343`) | Escolhe os filtros com maior norma L2 das colunas do consumidor | Não (redução) | Análogo à ordenação L1 do OFA. Não há re-treino dos que ficam. |
| `identity_block` (`:416`) | Conv com centro = I, bias 0, ReLU | **Exato** se a entrada for ≥ 0 | `_nonneg` (`:331`) exige ReLU e ausência de skip no bloco anterior. Também exige `norm=none` e `pool=none`. |
| `residual_zero_block` (`:427`) | y = act(conv(x)) + x com conv = 0, act ∈ {SiLU, GELU} | **Exato** (act(0)=0) e treinável (act′(0)=0,5) | `cin==cout`, skip identidade, `pool=none`, `norm=none`. |
| `grow_blocks` (`:444`) | Monta o filho camada a camada pelo `layer_map`: cópia literal (qualquer tipo de bloco, se os genes **e a entrada** são iguais) / widen / identity / residual_zero | Exato até a **primeira camada não mapeável** | Para no primeiro bloco não plano cuja entrada ou largura mudou. **Tudo depois é re-ajustado do zero** (SPiFiL + AE, `run_search._grow_fp:1027`). Exemplo: um BN no estágio 2 seguido de widen no estágio 1 perde a herança dos estágios 2–4. |
| `preservation_error` (`:517`) | max\|filho − pai\| nos canais antigos, ao fim do prefixo fp | — | Usa 8 imagens de validação (`run_search.py:997`) e só o máximo absoluto (o `rel_err` é dividido pelo máximo do pai). Retorna `None` após *cut*. **Não mede o embedding 1280** nem as camadas re-ajustadas. |
| `ElementProtect` (`:552`) | Multiplica o passo dos elementos herdados por `scale` (1 / 0,01 / 0) | Exato como regra de LR para Adam/AdamW/SGD+momentum (os momentos só dependem do gradiente) | **`freeze` não preserva a função após o treino.** As colunas novas do consumidor são "novas" e treinam com LR cheio. Quando saem de zero, a saída dos canais antigos da camada seguinte muda. Só vale como preservação no passo 0. |

**Limitação central (widen só em blocos planos)**: a exatidão hoje se restringe à classe "conv denso + ReLU + max/avg-pool". Mas extensões **exatas** são possíveis **[proposta, não implementado]**:
- **BN** (`conv→BN→act`): copiar γ, β, `running_mean` e `running_var` dos canais mantidos.
  - Em `eval()` é exato.
  - Em modo treino também é exato, porque as estatísticas por canal dos canais antigos só dependem dos canais antigos.
  - Os canais novos podem ter qualquer valor, já que as colunas a jusante são zero.
- **Residual com skip identidade**: alargar entrada e saída com o **mesmo** `in_map`, ou seja, o mesmo conjunto mantido/novo na mesma ordem. O skip leva x_novo para a saída nova e os canais antigos ficam intactos. Isso exige `keep` alinhado entre a entrada e a saída do bloco.
- **Residual com skip `project`** (1×1): colunas novas = 0. Exato.
- **dwsep / depthwise**: depthwise novos para os canais de entrada novos (arbitrário); pointwise com colunas novas = 0 e filtros novos via SPiFiL. Exato.
- **Inception**: widen por ramo (1×1/3×3/5×5/pool→1×1), com colunas novas = 0 em cada ramo. Exato, mas a partição `inception_split` das larguras precisa ser mantida.
- **ZCA/DBN**: estender a matriz de branqueamento em bloco-diagonal [W 0; 0 I] só é exato com o branqueamento **congelado** (buffer em `eval`). Em modo treino o branqueamento é recalculado sobre todos os canais, então **não** é exato.
- **SoftHebb**: depende da regra. Tratar como não exato até que haja teste.

### 2.2 `pnas_cs.py`

| Peça | O que faz | Observação |
|---|---|---|
| Orçamentos (`:58`) | `DEFAULT_BUDGETS` = 0,5/1/2/3,5/5,5M; `cap_factor` = 1,05 | **Diverge da missão** (B0–B4 = 0,3/0,8/1,5/3/5,5M, proibido > 5,5M). Na configuração dos experimentos: `budgets: [300000, 800000, 1500000, 3000000, 5500000]` e `cap_factor: 1.0`. |
| `stages` / `slots` / `stage_additions` (`:229`) | Estágio = camadas até o próximo pool. O stem nunca se repete. A profundidade é inserida antes da camada que faz pool. | As adições são **monótonas** em `d_mult` (o filho contém o pai), o que é uma boa propriedade para morfismo. |
| `scale_genome` (`:252`) | d por estágio = round(n·d); todas as larguras × w (≤ `cmax` = 1024) | A largura é uniforme por estágio (sem largura por estágio). O "stage-aware" (experimento D) precisa de multiplicadores por estágio. |
| `fit_level` (`:287`) | Busca em φ + bisseção na largura até atingir o orçamento de **parâmetros** (tolerância de 10%) | Não impõe α·β² ≈ 2 nem alvo de FLOPs. Os parâmetros contam o encoder + `EmbeddingHead` (`strata.param_estimate`), o que está de acordo com a missão. |
| `layer_map` (`:385`) | Alinha pai e filho; aceita "identity" / "residual_zero" nas camadas inseridas | Recusa mudança de operador, encolhimento e residual adicionado a camada existente. |
| `mutate` | `add_layer` (identidade), `add_residual` (zero, SiLU), `widen` (estágio × fator), `pooling` | `pooling` de estágio (max↔avg) **não** preserva a função. `pooling` final só troca a cabeça, que é sempre nova de qualquer forma. |
| `plain_grid=True` | Mantém as famílias da grade planas + ReLU | É exatamente o que garante a cadeia P0→P4 exata. Deve ser mantido nas linhagens "morfismo puro". |
| `fit_scaling_law` | κ = a + b·log10(params) + ponto de saturação (ganho por duplicação < 0,01) | Pode ser reutilizado nas curvas de saturação (§6). |

### 2.3 `lemonade.py`

- `DEFAULT_SEED` (`:39`) usa **`norm="batch"`** (RegNet 40-80-152-152-152-312).
  - Com `function_preserving=True`, nenhum widen ou deepen sobre essa semente é exato: os blocos não são planos e `identity_block` exige `norm=none`.
  - Sobra a cópia literal do prefixo.
- `function_preserving=False` (`:50`) é o padrão: só a herança de prefixo (`reusable_prefix`, `:112`). A camada alterada **e todas as seguintes** são re-ajustadas.
- O morfismo `residual` (`:150`) liga `residual=True` numa camada existente, o que muda a função (y = act(...) + skip(x)). **Não é o morfismo exato do LEMONADE** (combinação convexa com λ=0).
- O morfismo `kernel` (`:157`) inclui *diminuir* o kernel, que não é exato. Só k → k+2m é exato (zero-pad).
- Não há ANMs (poda, remoção, troca por dwsep) nem a KDE em f_cheap.

### 2.4 `pooling_head.py` + cabeça em `run_search.py`

- `EmbeddingHead` (`:79`) faz conv 1×1 → pooling ∈ {gap, gem, gem_learn, meanstd, moments, combo}, com dimensão final **exata** = `embed_dim` (≥ 1280).
- O 1×1 roda **na resolução cheia** do último mapa, em pedaços com checkpoint (`CHUNK_ELEMS`).
  - Com `gap`, o 1×1 e a média comutam (ambos lineares). Fazer o GAP antes daria o mesmo resultado matemático com HW× menos MACs na cabeça. O código não explora isso, e a diferença numérica fica na ordem de fp32.
- `moments` com curtose usa largura 320 (4 blocos), ou seja, **1/4 dos parâmetros** de GAP/GeM para o mesmo 1280.
- `CovReadout` (`:321`) é uma leitura externa de 2ª ordem, rotulada (PCA por bloco, sqrtm, signed-sqrt + L2, concatenada com a cabeça). Não tem dimensão fixa (≈ 17.920 no P4 do RS). É **auxiliar**, nunca a fitness ORIG.
- `readout_embed` `cat`/`cat_head` faz GeM de todos os blocos concatenados (hypercolumn). É multiescala, mas tem dimensão variável e não passa pela cabeça 1280.
- **A cabeça é recriada aleatoriamente em todo filho** (`run_search.py:700-702`, `torch.manual_seed(seed)`). A âncora do WiSE (α=0) é então "encoder herdado + cabeça aleatória". Isso afeta a leitura de "frequência de α=0" que o Agente 1 está auditando.

### 2.5 Custos registrados

- `cost/flops` = `inference_macs` (`run_search.py:334-360`). Conta só Conv2d/Linear, sem bias, pooling ou operações elemento a elemento. **Reportar FLOPs = 2×MACs** e dizer explicitamente qual dos dois.
- Exemplo do inventário: o PNAS-CS P4 c00011 (5,20M parâmetros, 17,44 "GFLOPs") tem 17,44 G **MACs**.

---

## 3. Especificações

### 3.1 `BackboneSpec` (v1) — modelo autocontido Bk

```yaml
backbone_spec:
  schema: backbone_spec/v1
  id: "<dataset>/<Bk>/<hash12>"           # hash = sha256 do bloco 'arch' + 'head' canonicalizado
  dataset: remote_sensing                  # base canônica; split e pct registrados
  split: 1
  budget_class: B2                         # B0 ≤300k | B1 ≤800k | B2 ≤1.5M | B3 ≤3M | B4 ≤5.5M (duro, sem tolerância acima)
  input: {resolution: 224, color: lab01|rgb, norm: none|imagenet, crop_protocol: "<id>"}
  arch:
    stem:          {kernel: 5, width: 32, conv: conv, norm: none, act: relu, pool: {type: max, size: 3, stride: 2, pad: 1},
                    init: spifil0, local_learning: {ae: pixel|feature|masked|none, schedule: joint|decoder_first}}
    stages:                                  # 0..3 estágios; nem todos obrigatórios
      - name: local                          # local | intermediate | abstract
        n_layers: 2
        width: 64                            # ou lista por camada
        kernel: 3
        conv: conv|dwsep|depthwise|inception|pointwise
        norm: none|batch|batch_zca|batch_dbn
        act: relu|relu6|silu|gelu
        residual: none|identity|project|convex_lambda0
        softhebb: false
        pool: {type: max|avg|none, stride: 2}   # o pool fecha o estágio
        init: spifil0|spifil1|spifil2|spifil3|spifil4|random   # contrato do Agente 2 (KernelInitializer.fit)
        local_learning: {ae: pixel, epochs: 5, protect: full|lowlr|freeze}
  head:
    embed_dim: 1280                          # SEMPRE 1280
    type: gap|gem|gem_learn|meanstd|moments|combo|multiscale_gem|fpn_lite|concat_compress|bilinear_lowrank|tensor_sketch
    taps: [abstract]                         # estágios lidos pela cabeça (multiescala: [intermediate, abstract], ...)
    proj: {kind: conv1x1|linear_after_pool|none, width: 1280, bn: false, act: none}
    norm: none|signed_sqrt_l2|l2         # faz parte da cabeça, conta na leitura ORIG
    random_seed: 0                           # TS/RM: semente do sketch
  extras_not_counted:                        # registrados à parte, NUNCA no orçamento nem na leitura ORIG
    decoders: {params: 0}
    ssl_projector: {params: 0}
  cost:                                      # medido (não estimado) no build
    params_encoder: 0
    params_head: 0
    params_total: 0                          # = encoder + head; verificação: params_total ≤ budget(Bk)
    macs: 0                                  # hook de run_search (Conv2d+Linear)
    flops: 0                                 # = 2*macs (+ ops de cabeça não lineares documentadas)
    latency_ms: {device: "<gpu>", bs: 1|64, dtype: fp32, tta: 1}
    peak_mem_gb_train: 0
    ckpt_size_mb: 0
  provenance:
    lineage_id: "<L-id>"
    parent: "<backbone_spec id>|null"
    growth_spec: "<growth_spec id>|null"
    checkpoint: "<path>"                     # encoder + head + spec (autocontido; sem dependência do pai)
    checkpoint_sha256: "<hex>"
    config_sha256: "<hex>"
    seeds: {init: 0, ae: 0, ssl: 0}
    fidelity_epochs: 0
    wise_alpha: 0.0
```

Regras:
1. `params_total` é **medido** no modelo construído (o SPiFiL pode construir menos filtros que o genoma, `run_search._grow_fp:976`).
2. Ultrapassar o orçamento invalida o modelo. Não há reparo silencioso: o reparo gera um novo spec com um novo hash.
3. O checkpoint é autocontido: `torch.load` do arquivo sozinho reconstrói o encoder e a cabeça.

### 3.2 `GrowthSpec` (v1) — uma aresta do DAG

```yaml
growth_spec:
  schema: growth_spec/v1
  id: "<parent_hash12>-><child_hash12>:<op>"
  parent: "<backbone_spec id>"
  child:  "<backbone_spec id>"
  operation:                                 # uma operação atômica ou lista ordenada
    - op: widen_zero|widen_replicate|deepen_identity|deepen_residual_zero|add_skip_convex|kernel_grow|
          compound|stage_widen|stage_deepen|head_widen|head_swap|prune_l1|to_dwsep|reinit_twin
      where: {stage: intermediate, layer: 3}
      params: {factor: 1.5, new_init: spifil2, noise_std: 0.0}
  inheritance:
    head: inherit_zero_cols|reinit             # [proposta] inherit_zero_cols preserva o 1280 no passo 0
    per_layer:                                 # um por camada do filho
      - {child_layer: 0, parent_layer: 0, origin: inherited|fp_widen|fp_identity|fp_residual_zero|fp_cut|refit|new,
         shape_parent: [32, 3, 5, 5], shape_child: [48, 3, 5, 5],
         in_map: [0, 1, 2], keep: [0, ..., 31],          # índices; -1 = canal novo
         n_inherited_elements: 2400, n_new_elements: 1200,
         new_init: spifil2, new_init_meta: {n_candidates: 0, rank: 0}}
  protect: {mode: full|lowlr|freeze, scale: 0.01, scope: inherited_elements}
  params: {parent_total: 0, child_total: 0, delta: 0, child_budget_class: B3}
  macs:   {parent: 0, child: 0}
  preservation:                               # medido ANTES de qualquer treino, model.eval(), fp32, determinístico
    data: {split: val, n_images: 64, indices_sha256: "<hex>"}    # NUNCA teste
    exact_claimed: true|false                 # true só se toda op for exata (§2)
    feat_max_abs: 0.0                         # canais antigos, ao fim do último bloco fp
    feat_rel_l2: 0.0                          # ||Δ||/||pai||
    per_layer_max_abs: [ ... ]
    emb_cos_min: 1.0                          # embedding 1280 pai × filho (só se head=inherit_zero_cols)
    emb_rel_l2: 0.0
    kappa_val_step0: {parent: 0.0, child: 0.0, delta: 0.0}   # SVM R1 (C=100, sem scaler)
    tolerance: {rel: 1.0e-5}                  # fp32; bf16 nunca usado para esta verificação
    verdict: exact|approx|not_preserving
  twin:                                       # controle obrigatório (experimento G)
    reinit_twin_id: "<backbone_spec id>"      # mesma arquitetura do filho, pesos sem herança
```

Transformações de forma por operação (com W = pesos conv `[cout, cin, k, k]`, b = bias):

| op | Filho | Exato? | Parâmetros adicionais |
|---|---|---|---|
| `widen_zero` (atual) | W_ℓ: `[c+n, cin, k, k]` (n linhas SPiFiL); W_{ℓ+1}: `[c′, c+n, k′, k′]`, colunas novas = 0 | Sim (blocos planos; BN/residual/dwsep com as extensões de §2.1) | n·(cin·k²+1) + c′·n·k′² |
| `widen_replicate` (Net2Net) | Linhas novas = cópia de g(j); colunas a jusante divididas pela contagem de réplicas | Sim sem ruído; não com `noise_std` > 0 | igual |
| `deepen_identity` | Insere `[c, c, k, k]` com centro = I, ReLU, entrada ≥ 0 | Sim | c²k² + c |
| `deepen_residual_zero` | Insere y = x + SiLU(0·x) | Sim | c²k² + c |
| `add_skip_convex` [proposta] | y = (1−λ)·f(x) + λ·g(x), λ = 0 aprendível | Sim | 1 (+ projeção se cin ≠ cout) |
| `kernel_grow` | k → k+2m com zero-pad central | Sim | (k+2m)² − k² por filtro |
| `head_widen` [proposta] | 1×1 `[1280, c+n]`, colunas novas = 0 | Sim para qualquer pooling (o 1×1 vem antes do pooling) | 1280·n |
| `compound` / `stage_*` | Composição das operações acima | Exato se todas forem exatas | — |
| `prune_l1` / `to_dwsep` / `head_swap` / `reinit_twin` | Redução / troca | Não | — |

---

## 4. Família de referência B0→B4 (plana e contida)

Template:
- stem conv k5 + max-pool s2 (224 → 112);
- 3 estágios conv k3 + ReLU, cada um fechado por max-pool s2 (112 → 56 → 28 → 14);
- norm none;
- cabeça GeM 1280 no mapa 14×14.

Cada Bk+1 contém Bk (larguras e número de camadas por estágio não decrescem), então a cadeia é **toda exata** com as operações atuais (`widen_zero`, `deepen_identity`), mais `head_widen` quando implementado.

| Bk | stem | local (n×w) | intermediário | abstrato | params encoder | params cabeça GeM | **total** | GMACs (224) |
|---|---|---|---|---|---|---|---|---|
| B0 | 24 | 1×48 | 1×80 | 1×112 | 128k | 145k | **0,272M** | 0,42 |
| B1 | 32 | 1×64 | 2×112 | 2×176 | 655k | 227k | **0,882M** ⚠ | 1,31 |
| B2 | 32 | 2×64 | 2×128 | 2×256 | 1,165M | 329k | **1,494M** | 2,27 |
| B3 | 40 | 2×80 | 3×160 | 3×288 | 2,575M | 370k | **2,945M** | 4,61 |
| B4 | 48 | 2×96 | 3×208 | 3×400 | 4,717M | 513k | **5,231M** | 7,69 |

- ⚠ B1 passa de 800k com GeM. Com `moments` cai para 0,712M, ou basta reduzir o abstrato para 160. Isso mostra que **a escolha da cabeça muda a classe de porte**.
- A semente externa "SPiFiL pequeno" (40-80-152-152-152-312, k3) tem ≈ 0,98M só de encoder. Com GeM 1280 dá ≈ 1,38M, então é **B2**, não B1.
- Variante B4d (2×96, 3×192, 4×352): 5,365M com GeM. Mesma ordem de MACs, mais profundidade no estágio abstrato.
- A maior parte dos MACs vem do stem e do estágio local (resolução 112/56). Crescer em profundidade no estágio local é caro em FLOPs e barato em parâmetros. Crescer no estágio abstrato é o inverso. Isso justifica o experimento D (por estágio) com os dois objetivos da fitness (κ × FLOPs).

---

## 5. DAG de linhagens

Cada nó é um `BackboneSpec` (dataset, Bk, hash, init, fidelidade, checkpoint) e cada aresta um `GrowthSpec`. Gêmeos sem herança (`reinit_twin`) são nós irmãos ligados por uma aresta de controle tracejada.

```
            ┌──────────── L-size (tamanho; cadeia exata) ───────────┐
 seed B0 ──widen/deepen──▶ B1 ──compound──▶ B2 ──stage_deepen(abs)──▶ B3 ──compound──▶ B4
   │  ╎                     ╎                 ╎                          ╎                ╎
   │  └╌twin╌▶ B1'(reinit)  └╌twin╌▶ B2'      └╌twin╌▶ B3'                └╌twin╌▶ B4'     (G)
   │
   ├── L-val   (a cada nível, o filho de melhor κ_val entre {widen, deepen, compound, stage_*}; o teste fica fora)
   ├── L-flops (mesmo, mas o objetivo é κ_val por GMAC; Pareto local)
   ├── L-div   (operadores estruturais: dwsep / inception / residual, preservação aproximada declarada)
   ├── L-expl  (init abstrata SPiFiL-1..4 nos filtros novos; ramos com head multiescala)
   └── L-shrink(controle OFA): B4 treinado ──prune_l1 + primeiras-D──▶ B3s ─▶ B2s ─▶ B1s ─▶ B0s
```

- O registro é um JSONL append-only (`lineage.jsonl`) com uma linha por nó e uma por aresta, contendo o hash do spec e o sha do checkpoint. A visualização é gerada a partir desse arquivo.
- **Seleção do ancestral**: sempre por κ_val (R1, C=100, sem scaler) na fidelidade máxima disponível ≤ à do filho (`inherit_fidelity: le_child`). O teste nunca entra (R2/missão).

---

## 6. Experimentos estruturais A–H

**Protocolo comum:**
- fitness = κ_val ORIG (embedding 1280, `core.svm.fit_svm`, C=100, sem scaler);
- relatar também F1_val, rank efetivo do embedding, params, MACs/FLOPs, latência (bs = 1 e 64, fp32), pico de memória e tamanho do checkpoint;
- teste só no relatório final, isolado;
- **3 seeds** (init/AE/SSL) por braço;
- datasets na ordem RS → Flowers → HAM10000, depois Eggs/Larvae/Protozoan (raw e pre separados), com AID e Coconut como controles;
- mesma receita SSL (LeJEPA+SIGReg), mesma fidelidade e mesmo WiSE em todos os braços de um experimento;
- critério de "benefício": Δκ_val médio > 0 com IC95% (t, n = 3) que não cruza 0 **e** Δ ≥ 0,01;
- caso contrário, registrar como nulo ou negativo.

| Exp | Pergunta | Controle | Tratamento(s) | Variável independente | Fixos | Notas |
|---|---|---|---|---|---|---|
| **A** Largura | Alargar ajuda e com qual inicialização dos canais novos? | Pai Bk (mesma fidelidade total) | `widen_zero` ×{1,25; 1,5; 2} uniforme; `widen_replicate` (Net2Net) ×1,5 | Fator e tipo de widen | Profundidade, cabeça (com params re-medidos), init dos canais herdados | Mede preservação no passo 0 e Δκ após o treino |
| **B** Profundidade | Profundidade ótima por domínio | Pai Bk | `deepen_identity` +1/+2 por estágio; `deepen_residual_zero` | Nº de camadas / estágio | Largura, cabeça | Parasitas 2–4 blocos; Flowers 3–6; RS moderada |
| **C** Width+depth | Compound é melhor que um eixo só ao mesmo orçamento? | A e B ao **mesmo** params_total (±3%) | `compound` com direção (α, β) por grid em B0→B1 (validação) | Direção log α/log β | Orçamento, resolução 224 | Replica a Fig. 3 do EfficientNet no nosso regime |
| **D** Por estágio | Onde crescer: local, intermediário ou abstrato? | Crescimento uniforme (C) | `stage_widen`/`stage_deepen` em 1 estágio por vez, ao mesmo params | Estágio | Orçamento | Comparar também por FLOPs (local = caro em MACs) |
| **E** Residual | Residual ajuda em profundidade maior? | `deepen_identity` (plano) | `deepen_residual_zero` (SiLU); `add_skip_convex` λ = 0 | Tipo de conexão | Params (±1%) | O residual quebra o widen exato hoje (§2.1). Registrar. |
| **F** Init abstrata | SPiFiL-1..4 nos filtros/camadas novas supera o SPiFiL-0? | Novos via SPiFiL-0 | SPiFiL-1/2/3/4 (Agente 2) nos mesmos slots | Inicializador | Arquitetura e operação | **Depende do gate do Agente 2.** Também testar `random` como piso. |
| **G** Herança × sem | Herdar melhora sobre re-inicializar a **mesma** arquitetura? | `reinit_twin`: SPiFiL + AE + SSL do zero | Herança fp com `protect` ∈ {full, lowlr, freeze} | Herança / protect | Arquitetura e orçamento de passos **total** (o pai já gastou passos: comparar também com o gêmeo treinado com passos pai+filho) | Controle de compute justo. Sem ele, a herança "vence" só por ter treinado mais. |
| **H** Cabeça multiescala × original | Fundir mapas intermediários no 1280 resolve o "COV vai bem, 1280 nem sempre"? | Cabeça GeM 1280 original | `multiscale_gem`, `fpn_lite`, `bilinear_lowrank`, `tensor_sketch`, `moments` | Tipo de cabeça | **Dois controles**: (i) mesmo encoder (params diferem); (ii) mesmo params_total (encoder re-ajustado) | O `CovReadout` só como referência auxiliar, nunca como braço ORIG |

- **Curvas de saturação**: para cada linhagem L-size, κ_val × log10(params) e κ_val × log10(MACs) em B0..B4, ajustadas por `pnas_cs.fit_scaling_law` (ganho por duplicação < 0,01 = saturação).
- **Prioridade sugerida** (custo × informação): G e A em RS e Flowers primeiro (respondem se a herança vale), depois H, C/D, B/E, e F por último (gate do Agente 2).

---

## 7. Opções de cabeça 1280 (custo estimado)

Exemplo B2 (último mapa 256 × 14×14; saídas dos estágios 64@56², 128@28², 256@14²). Os MACs da cabeça foram calculados como no código atual (1×1 na resolução cheia).

| Cabeça | Como chega a 1280 | Params cabeça (B0 / B2 / B4) | MACs cabeça B2 | Preservável no crescimento? | Notas |
|---|---|---|---|---|---|
| `gap` (atual) | 1×1 → 1280, média | 145k / 329k / 513k | 64M (0,3M com GAP→linear, mesma matemática) | Sim (`head_widen`) | Mais simples. Parâmetros pesam no B0. |
| `gem` (atual) | 1×1 → 1280, GeM p = 3 | 145k / 329k / 513k | 65M | Sim | Padrão atual dos genomas |
| `meanstd` | 1×1 → 640, μ‖σ | 72k / 164k / 257k | 32M | Sim | |
| `moments` (atual) | 1×1 → 320, μ‖σ‖skew‖kurt | 36k / 82k / 128k | 16M | Sim | Escalas heterogêneas e SVM sem scaler: precisa de normalização por grupo **[hipótese]** |
| `multiscale_gem` | GeM nos 3 estágios (448 dimensões) → linear 1280 | 308k / 575k / 902k | 1,6M | Parcial (colunas novas = 0 na linear) | Estoura o orçamento em B3/B4 sem reduzir o encoder |
| `multiscale_proj` | 1×1 → 320 por estágio, GeM, concatenação de 4 | 78k / 144k / 231k | 112M | Sim por ramo | 1×1 em 56² é caro |
| `fpn_lite` | Laterais 1×1 → 128, soma top-down (sem 3×3), GeM por nível, 384 → 1280 | 524k / 551k / 583k | 45M | Sim (laterais e linear com colunas zero) | Com o 3×3 de suavização os MACs ficam ≈ 2,5G. Não recomendado. |
| `concat_compress` | GeM concat → **PCA/linear fixa** → 1280 | 0 treináveis (PCA) / ≈ como `multiscale_gem` (linear) | ~1M | Não (PCA é refeita) | Se ΣC < 1280 é expansão. Ajustar a PCA só no treino. |
| `bilinear_lowrank` | 1×1 → 50, cov 50×50 com triângulo superior (1275) + 5 médias, sqrt com sinal + L2 | 6k / 13k / 20k | 3M (+ sqrtm opcional: eigh 50×50 por imagem) | Sim no 1×1 | 2ª ordem *dentro* do 1280. Alternativa honesta ao CovReadout. |
| `tensor_sketch` | TS d = 1280 por posição, sum-pool, sqrt com sinal + L2 | 0 treináveis (2C de hash) | ≈ 20M (FFT, fp32) | Trivial (o sketch não depende de C) | d/C² ≈ 0,8–2% < faixa validada no artigo |
| `tensor_sketch_red` | 1×1 → 128 + TS | 14k / 33k / 53k | 26M | Sim | |

Observações:
- Toda normalização (sqrt com sinal, L2) é **parte da cabeça** e é registrada no `BackboneSpec.head.norm`, porque o SVM ORIG não usa scaler (R2).
- As cabeças sem parâmetros (TS, PCA) liberam 150–500k parâmetros para o encoder ao mesmo Bk. Por isso o controle (ii) "mesmo params_total" do experimento H é obrigatório.

---

## 8. Riscos e pendências para a fase de execução

1. **Gate do Agente 2**: o experimento F e as linhagens L-expl dependem de SPiFiL-1..4. Os experimentos A–E, G e H podem rodar com SPiFiL-0.
2. **Implementações necessárias (código isolado, opt-in, testes de ponta a ponta antes da fila, R9)**:
   1. `head_widen` com colunas zero;
   2. widen exato para BN, residual, dwsep e inception (§2.1);
   3. `add_skip_convex`;
   4. `widen_replicate`;
   5. cabeças `bilinear_lowrank` / `tensor_sketch` / `fpn_lite` / `multiscale_gem`;
   6. orçamentos B0–B4 com `cap_factor=1.0`;
   7. FLOPs = 2×MACs explícitos;
   8. `preservation` com 64 imagens de validação, rel-L2, cosseno do 1280 e κ_val no passo 0;
   9. teste unitário de cada morfismo: `max_abs < 1e-5·escala` em `eval()`, fp32.
3. **Justiça de compute (G)**: o filho herdado já "contém" os passos do pai. Os gêmeos precisam de dois orçamentos: o mesmo do filho e o total pai+filho.
4. **Resolução 224 fixa**: o crescimento por resolução do EfficientNet fica fora do escopo por definição da missão. Isso deve ser registrado como limitação das conclusões de scaling.
5. **WiSE com cabeça aleatória (α=0)**: com `head_widen`, a âncora passa a ser a cabeça herdada. Isso muda o significado de α=0 e deve ser uma variante registrada, não uma mudança silenciosa.
