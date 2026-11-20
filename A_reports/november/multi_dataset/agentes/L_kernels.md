# L_kernels: quais kernels SPiFiL ajudam e quais atrapalham (só L1)

## Resposta curta

**Quase nenhum kernel atrapalha sozinho: o que atrapalha é gastar vaga com cópia.**
Nos datasets recortados (objeto sobre fundo preto), metade do banco é a mesma peça repetida: um quadradinho preto.
Tirar as cópias e repor com outro candidato sobe κ em +0,14 (protozoan_cysts) e +0,04 (helminth_eggs), com 3 de 3 splits positivos.
Em helminth_larvae, que não tem o que repor, precisa de 3 imagens por classe: +0,03.
Isso já funciona por YAML, sem código novo. Ainda é simulação sem treino da regra exata; o treino real de uma versão próxima (x3) apontou na mesma direção.

Nos datasets de quadro inteiro nenhuma regra que testei bateu a escolha atual do SPiFiL. Lá, o melhor é deixar como está.

Três coisas que eu esperava que importassem não importam:
- **O backprop não conserta kernel ruim.** Em L1 ele quase não mexe: giro mediano de 0,24°.
- **A nota Fisher não diz quem é útil** dentro do banco escolhido: correlação −0,01 [−0,11; 0,11].
- **A classe de origem não diz nada:** tirar os kernels de uma classe é igual a tirar um grupo sorteado.

Tudo aqui é treino + validação. O teste não foi lido, nem o CSV de teste que já existe em `results/`.

## O que eu medi

- 7 datasets, 3 splits, 50%: os 21 runs de `disc_md_x1_s200_p0800_w064`. O pedido era "poucos datasets", mas ficou barato o bastante para rodar todos.
- **O truque que barateou tudo:** em L1 cada coluna da leitura GeM depende de um kernel só. Zerar o kernel é tirar a coluna, sem novo forward. Por run foram ~2.400 ajustes de classificador.
- `kernel_redundancy.json` não existe em x1 (só em x3). Para saber de onde veio cada kernel, refiz o `grow` offline sem editar nada.
  Bateu com o `L1_grow`: diferença até 5e-5 em 17 runs (CPU) e 0,0 nos outros 4 (GPU).
- A leitura é GeM + PCA-whitening com shrinkage + L2 + classificador linear, ajustada no treino, medida na validação.
  Para as milhares de ablações usei regressão logística C=100, que acompanha o SVM oficial: correlação 0,987, erro médio 0,026 em 63 pontos.
  Os números finais de regra são do SVM oficial (`core/svm.py`, linear, C=100). Sem TTA.
- Os dados têm **dois regimes**, e quase tudo muda entre eles:
  - **Recortado** (helminth_eggs, helminth_larvae, protozoan_cysts): 97 a 98% dos pixels são preto puro.
  - **Quadro inteiro** (Eggs_subgroup, Protozoan_subgroup, Protozoan, Larvae_subgroup): só a faixa do letterbox é preta, 11 a 42%.

## 1. Quanto cada kernel se move no backprop

**Quase nada. O kernel inicial é, na prática, o kernel final.**

| | mediana | 95% dos kernels até | máximo |
|---|---|---|---|
| Mudança relativa do peso | 1,5% | 9,9% | 17% |
| Giro da direção | 0,24° | 2,7° | 9,9° |

- 88% do movimento é só encolher a norma, sem mudar a direção. Parece weight decay: em Protozoan, lr 5e-4 × wd 0,05 × 7.200 passos dá no máximo 18%, e o medido foi 9,8%. Não conferi o agendamento do lr, então é inferência.
- A resposta do kernel fica igual: correlação mediana entre a coluna GeM antes e depois de 0,9999.
- O ganho de κ do backprop em L1 não se distingue de zero: +0,007 [−0,003; 0,019] nos recortados, +0,011 [−0,010; 0,033] nos quadros.
- Kernels que disparam no preto giram mais (1,3 a 1,5° contra 0,2°), mas seguem longe de virar outra coisa.
- Ressalva: são 20 épocas, e o best caiu na época 2 a 5 em 7 dos 21 runs (ver bug abaixo). Mais épocas podem mover mais; não medi.

## 2. Utilidade de cada kernel na validação

**A utilidade é real e dá para medir; quase nenhum kernel é prejudicial.**

Método: 800 sorteios por run, cada um tirando um grupo aleatório de metade dos kernels distintos. A regressão do κ dá a contribuição média de cada kernel. Cópias exatas contam como um kernel só.

| Dataset | kernels distintos | ajudam (>2 erros-padrão) | atrapalham (<−2 erros-padrão) | confiabilidade entre metades da validação | zerar 1 kernel (mediana de \|Δκ\|) |
|---|---|---|---|---|---|
| Eggs_subgroup | 63 | 46% | 5,9% | 0,65 | 0,010 |
| Protozoan_subgroup | 62 | 51% | 2,2% | 0,88 | 0,012 |
| Protozoan | 62 | 57% | 2,2% | 0,95 | 0,006 |
| helminth_eggs | 31 | 80% | 0% | 0,84 | 0,009 |
| helminth_larvae | 26 | 62% | 2,9% | 0,71 | 0,018 |
| protozoan_cysts | 26 | 90% | 0% | 0,93 | 0,008 |
| Larvae_subgroup | 63 | 25% | 6,3% | 0,52 | 0,021 |

- "Atrapalham" fica em 2,2% sem Larvae_subgroup, igual ao esperado por acaso (2,3%). **Não há evidência de kernel que prejudique.** Os fracos só não somam.
- Larvae_subgroup tem 107 a 118 imagens de validação. Trate como ruído; ficou fora dos efeitos de regra.
- Zerar um kernel só é pouco informativo: o efeito fica no tamanho do ruído. Por isso grupos funcionam melhor que um por um.

**Ruído que precisa ficar na cabeça:** por run, diferença abaixo de ~0,03 não significa nada.
Trocar 1 kernel moveu κ em até ±0,04, e em Eggs_subgroup split 1 uma diferença de 1e-5 nos pesos (CPU contra GPU) moveu a leitura rápida 0,019.
Grupo aleatório do mesmo tamanho de uma classe varia ±0,011 a ±0,019 (±0,09 a ±0,11 nos datasets de 2 classes, onde o grupo é metade do banco).

**Classe de origem: sem sinal.** Tirar uma classe fica dentro do ruído de grupos aleatórios (|z| médio 0,6 a 1,2), e a classe importante num split não é a mesma no outro: concordância entre splits de −0,30 a +0,26.

**Controle aleatório** (leitura oficial, validação):

| Dataset | SPiFiL | aleatório centrado | aleatório bias 0 (`random_layer`) |
|---|---|---|---|
| Eggs_subgroup | 0,652 | 0,536 | 0,510 |
| Protozoan_subgroup | 0,642 | 0,543 | 0,414 |
| Protozoan | 0,247 | 0,100 | 0,092 |
| Larvae_subgroup | 0,630 | 0,529 | 0,485 |
| helminth_eggs | 0,824 | 0,833 | 0,823 |
| helminth_larvae | 0,885 | 0,875 | 0,916 |
| protozoan_cysts | 0,681 | 0,731 | 0,689 |

- Quadro inteiro: SPiFiL ganha do aleatório por +0,116 [0,086; 0,145].
- Recortado: empate, −0,016 [−0,052; 0,021]. **O SPiFiL de hoje não acrescenta nada ali**, e o motivo é a seção 4.
- Dos 63 kernels saem só ~5 dimensões úteis depois do GeM (posto efetivo). Nos recortados o SPiFiL fica em 2 a 3; o aleatório em 3,8 a 5,7.

## 3. O que prevê utilidade antes do treino

Correlação de Spearman dentro de cada run, média por dataset, IC 95% hierárquico (sorteia dataset, depois split). Ruído por permutação: |ρ| ≈ 0,07.

| Propriedade | todos | recortado | quadro | Leitura |
|---|---|---|---|---|
| **Utilidade medida no treino** (mesma ablação, kernels iniciais, sem validação) | **+0,60** [0,47; 0,73] | +0,69 | +0,54 | melhor sinal; precisa de rótulo do treino |
| **Ativação média alta** (GeM) | **−0,42** [−0,54; −0,30] | −0,44 | −0,40 | kernel sempre ligado rende pouco; sem rótulo |
| Separa classes sozinho (F no treino) | +0,34 [0,16; 0,49] | +0,38 | +0,31 | estável |
| Dispara no objeto | +0,34 [0,17; 0,48] | +0,19 | +0,45 | forte só em quadro |
| Perto do centro da imagem | −0,31 [−0,45; −0,18] | −0,25 | −0,36 | estável |
| Correlação de ativação com o vizinho | −0,27 [−0,41; −0,13] | −0,22 | −0,31 | estável; sem rótulo |
| Frequência | +0,17 [0,08; 0,26] | +0,15 | +0,18 | fraco |
| Energia (contraste do patch) | −0,13 [−0,21; −0,03] | −0,09 | −0,16 | fraco, e tirar por energia não ajuda |
| Peso no canal L | −0,10 [−0,43; 0,24] | **+0,43** | **−0,51** | inverte entre regimes: não serve de regra |
| **Cosseno com o vizinho** | +0,02 [−0,07; 0,12] | +0,06 | 0,00 | **zero** |
| **Nota Fisher** | −0,01 [−0,11; 0,11] | 0,00 | −0,01 | **zero** |
| Rank na classe | +0,05 [−0,07; 0,17] | −0,03 | +0,12 | zero |

- Correlações abaixo de ~0,2 não sobrevivem à correção por 33 propriedades.
- **Cosseno entre pesos não mede redundância; correlação de ativação mede.** Isso importa para a chave: `redundancy_cos` só serve para cópia exata.
- Fisher e rank foram medidos só entre kernels já escolhidos, um grupo estreito. Não digo que Fisher é inútil na seleção inteira: sortear candidatos perde −0,064 [−0,106; −0,030] da seleção do SPiFiL nos quadros.
- Em filtros aleatórios a ativação média não prevê utilidade (−0,10 e +0,21). A regra é sobre kernels SPiFiL, não uma lei geral de filtros.
- **Em dataset nunca visto:** treinei a previsão em 6 datasets e testei no 7º. Utilidade no treino sozinha dá ρ=0,60; ativação média + correlação, sem rótulo, dá 0,42; todas as 32 propriedades fixas dão 0,46. Mais propriedades quase não acrescentam.

## 4. Regra candidata

### Kernel ruim: cópia

**Definição, medida antes do treino:** o patch é idêntico a outro já escolhido. Na prática é o preto puro.

**Por que acontece:** a nota Fisher é "distância às outras classes ÷ distância à própria" (`SPiFiL/src/spifil/scoring.py:40-46`).
Patch preto tem dezenas de gêmeos idênticos na própria classe, então a distância à própria classe despenca e a nota sobe: em helminth_eggs e protozoan_cysts, 1,31 a 1,46 contra ~1,0 dos outros.
A lista de candidatos da classe enche de preto, e o `DiversitySelector` não tem o que escolher. Em helminth_eggs split 1, 42 dos 63 filtros são o mesmo.
Em helminth_larvae a causa é outra: a nota do preto não se destaca (0,87 a 1,17), mas são 2 classes com 32 vagas para 50 candidatos, então quase todo candidato entra.
Mais superpixels não resolve: com 800 superpixels (p0050) os recortados continuam com ~metade do banco em cópias.

**Efeito de tirar as cópias e repor**, com o `grow` real (`redundancy_cos: 0.9999`, `redundancy_replace: true`), leitura oficial:

| Dataset | cópias tiradas | repostas | κ antes | κ depois | Δ | splits |
|---|---|---|---|---|---|---|
| protozoan_cysts | 37 | 37 | 0,668 | 0,810 | **+0,142** | +0,12 +0,17 +0,13 |
| helminth_eggs | 32 | 32 | 0,813 | 0,850 | **+0,037** | +0,07 +0,03 +0,01 |
| helminth_larvae | 38 | **0** | 0,888 | 0,873 | −0,015 | sem ter o que repor |
| Protozoan | 1 | 1 | 0,254 | 0,260 | +0,006 | ruído |
| Eggs_subgroup | 0,3 | 0,3 | 0,641 | 0,628 | −0,013 | ruído |
| Protozoan_subgroup | 1,3 | 1,3 | 0,645 | 0,627 | −0,018 | ruído |
| Larvae_subgroup | 1 | 1 | 0,522 | 0,489 | −0,033 | ruído |

Intervalo pareado por imagem (bootstrap, 3 splits, versão simulada da mesma regra): protozoan_cysts +0,155 [0,135; 0,176]; helminth_eggs +0,044 [0,021; 0,069].
Nos quadros a mesma simulação deu −0,014 [−0,027; −0,003] e −0,015 [−0,029; −0,001]: pode haver um custo pequeno e real. **Por isso a regra vale só para o regime recortado**, não para todos.

**Guarde um preto.** Sozinho ele fica entre o percentil 30 e 80 de utilidade (provavelmente mede o tamanho do objeto). Tirar todos deu −0,006 contra deixar um, dentro do ruído: não há ganho em tirar o último.

**Evidência com treino real:** o braço x3 (cos 0,8 + reposição) já treinou. Leitura oficial, kernels treinados:
- helminth_eggs: 0,824 → 0,856, com menos filtros (52 a 57, em vez de 63).
- Eggs_subgroup: 0,652 → 0,636, dentro do ruído, mas sem ganho.

### Kernel bom: só serve para decidir quem sai

Se for preciso cortar kernels, a ordem importa. Tirando o pior quartil, contra tirar ao acaso (sem Larvae_subgroup):

| Critério | ganho sobre tirar ao acaso, 25% | 50% |
|---|---|---|
| Oráculo (usa a validação; é o teto) | +0,033 [0,023; 0,043] | +0,078 [0,067; 0,089] |
| **Utilidade no treino** | **+0,023** [0,010; 0,035] | **+0,062** [0,039; 0,080] |
| Ativação média + correlação (sem rótulo) | +0,015 [0,002; 0,027] | +0,039 [0,017; 0,058] |
| Fisher | +0,001 [−0,014; 0,015] | −0,002 [−0,021; 0,018] |
| Energia | +0,001 [−0,016; 0,017] | −0,032 [−0,055; −0,010] |

- Tirar 25% ao acaso custa −0,029; pela utilidade no treino custa −0,006. Ou seja, **um quarto do banco sai quase de graça**, com 70% do ganho do oráculo.
- **Mas não vira seletor melhor.** Usar utilidade no treino para escolher do zero, entre todos os candidatos, perde nos quadros: −0,059 [−0,109; −0,013]. Mesmo com poda de redundância: −0,027 [−0,054; −0,001]. Nos recortados não passa da deduplicação simples (+0,061 contra +0,061).
- Ativação média baixa como filtro de candidatos foi pior ainda (−0,095 nos quadros). Regra derrubada.

### Quando falta candidato para repor

helminth_larvae tem só 2 imagens no pool, ou 19 a 30 patches distintos para 64 vagas. A saída é `images_per_class: 3`, chave que já existe. Simulei nos 21 runs, leitura oficial:

| Dataset | hoje (1 imagem) | sem cópias, 1 imagem | sem cópias, 3 imagens | 3 contra 1 imagem, sem cópias |
|---|---|---|---|---|
| helminth_larvae | 0,888 | 0,873 | **0,919** | **+0,047** [0,009; 0,087] |
| helminth_eggs | 0,813 | 0,857 | 0,849 | −0,009 [−0,029; 0,011] |
| protozoan_cysts | 0,668 | **0,823** | 0,792 | −0,031 [−0,048; −0,014] |
| Eggs_subgroup | 0,641 | 0,627 | 0,594 | −0,032 [−0,062; −0,002] |
| Protozoan | 0,267 | 0,271 | 0,231 | −0,040 [−0,052; −0,030] |
| Protozoan_subgroup | 0,645 | 0,630 | 0,548 | **−0,081** [−0,104; −0,059] |

**Mais imagens não é melhor por si só.** Só ajuda quando faltam candidatos; nos quadros piora bastante. Não medi por quê.

### Como escolher antes de inicializar

1. **Conte as cópias no banco.** A separação entre regimes é limpa: 44 a 70% do banco nos recortados, 0 a 3% nos quadros. Bom corte: mais de 10%.
2. **Acima de 10%:** tire as cópias e reponha. Se faltar candidato, use 3 imagens por classe.
3. **Abaixo de 10%:** não mexa. A seleção atual do SPiFiL foi a melhor que encontrei.
4. **Se precisar de um banco menor:** corte pela utilidade no treino, nunca pelo Fisher.

Sinal acima do ruído existe nos passos 1, 2 e 4. Para "achar um kernel melhor que os do SPiFiL" em quadro inteiro, **não achei sinal**.

**Largura pesa mais que escolha.** Usar todos os ~350 candidatos em vez de 63 dá +0,118 [0,071; 0,179] nos quadros (leitura rápida, sem Larvae_subgroup), bem mais que qualquer regra. Isso é tema da lei de tamanho, mas muda a prioridade: nos quadros, abrir a largura de L1 promete mais do que escolher melhor.


## 5. Proposta para testar num treino real (nada foi criado)

**Nenhuma chave nova.** `redundancy_cos`, `redundancy_replace` (`experiments/ray/schema.py:93`) e `images_per_class` (`:79`) já existem. Rodei o `grow` real com essas chaves nos 21 runs, sem treino: funcionam.

Dois YAMLs irmãos de `experiments/autoencoder/disc_md_x3_cos80_rep_s200.yaml`, só com estas diferenças:

```yaml
# disc_md_k1_dedup_s200.yaml: tira so copias exatas, com reposicao
name: spifil_growth_disc_md_k1_dedup_s200
grid:
  datasets: [helminth_eggs, protozoan_cysts]
runner_args:
  redundancy_cos: 0.9999     # era 0.8; so copia exata (cosseno entre pesos nao mede mais que isso)
  redundancy_replace: true
output:
  work_dir: /exps/moliveira/spifil_growth/disc_md_k1_dedup_s200
overrides:
  model.init_args.best_by_probe: false   # enquanto a sonda float16 nao for corrigida; cai em val_recon_loss
```

```yaml
# disc_md_k1_dedup_ipc3_s200.yaml: igual ao de cima, para quem nao tem o que repor
name: spifil_growth_disc_md_k1_dedup_ipc3_s200
grid:
  datasets: [helminth_larvae]
runner_args:
  redundancy_cos: 0.9999
  redundancy_replace: true
  images_per_class: 3
output:
  work_dir: /exps/moliveira/spifil_growth/disc_md_k1_dedup_ipc3_s200
overrides:
  model.init_args.best_by_probe: false
```

**Custo:** 9 runs de L1, 20 épocas, ~4 minutos cada.
**Comparação:** contra x1, split a split, com a leitura oficial `gem_tta_pw` na validação.
**Passa se:** a média dos 3 splits subir pelo menos +0,03 em protozoan_cysts, helminth_eggs e helminth_larvae. Abaixo disso é ruído.
**Atenção ao comparar:** x1 foi selecionado com a sonda quebrada e o k1 não seria. O ideal é rodar também um x1 com `best_by_probe: false`, ou corrigir a sonda antes.

Não proponho chave nova de regra por utilidade: ela só serve para encolher banco, e encolher L1 quase não economiza parâmetro.


## 6. O que falta para L2 a L4

**O método vale, faltam os dados.** Com as camadas anteriores congeladas, cada coluna GeM do estágio k ainda depende de um kernel só do estágio k. O truque da ablação barata continua servindo, estágio por estágio.

Falta, nesta ordem:
1. **Corrigir a sonda float16.** Sem isso o best das camadas seguintes sai errado justo nos recortados.
2. **Checkpoints.** x1 só tem L1. É preciso uma cadeia `layerwise: 4` sobre a L1 sem cópias, com o par x1 para comparar.
3. **Provar que ganho em L1 chega em L4.** É a pergunta mais importante, e só o treino em cadeia responde: aqui medi a leitura da própria L1.
4. **Contar cópias em L2 a L4.** A região preta continua constante depois de L1, então espero o mesmo acúmulo. Não medi.
5. **Reposição em L4.** Lá todos os candidatos já entram (Y4), então não há quem repor: tirar cópia aumenta os aleatórios de `fill_random`. Precisa de pool maior só nessa camada.
6. **Limiar por camada.** `redundancy_cos` é um número só para o YAML inteiro. Serve para cópia exata; para redundância parcial não, e cosseno nem prevê utilidade.
7. **Refazer o movimento do backprop.** "O kernel não se move" foi medido em L1, com 20 épocas.
8. **Trocar as propriedades.** Com patch de D=800 a 1200, cor e frequência perdem sentido; ativação média e correlação continuam valendo. Medir utilidade kernel a kernel com 1280 filtros fica caro: lá precisa ser por grupo mesmo.


## Bug que achei no caminho: a sonda por época mede κ errado nos recortados

**O que é:** `_probe_images` guarda as imagens de treino em float16 (`methods/autoencoder/autoencoder_flim_module.py:1201`), mas a validação fica em float32.
Em kernels que quase só veem preto a coluna tem desvio ~1e-5, e o arredondamento do float16 vira um deslocamento de 30 a 43 desvios entre treino e validação. A sonda colapsa numa classe.

**Prova:** mesmos pesos, mesma função `_probe_logits`, só trocando a precisão:

| Run | treino float32 | treino float16 | CSV do run |
|---|---|---|---|
| helminth_larvae split 3 | 0,876 | **0,000** | 0,000 |
| helminth_eggs split 1 | 0,528 | **0,139** | 0,002 |

**Alcance:** 5 dos 9 runs recortados têm κ registrado errado por 0,47 a 0,88 (helminth_eggs 1 e 3, helminth_larvae 3, protozoan_cysts 1 e 3). Os quadros passam quase ilesos.
**Consequência:** `best_by_probe` escolhe checkpoint com um κ errado, e as curvas de M1 estão falsas nesses runs.
Em L1 isso quase não muda os pesos, porque eles quase não se movem. Em camadas que se movem pode importar. **Quem usa `epoch_metrics.csv` dos recortados precisa saber disso.** Não mexi no código.

## Erro meu que corrigi no caminho

A primeira rodada usou regressão logística C=1 sobre colunas padronizadas e deu "aleatório ganha do SPiFiL por +0,34". Era falso.
Colunas colineares, com C pequeno, perdem as direções fracas, e isso pune o SPiFiL. Detectei porque Larvae_subgroup dava κ 0,07 contra 0,5 da sonda.
Calibrei contra o SVM oficial e refiz tudo. Nada neste relatório vem da rodada antiga (`/exps/moliveira/law_kernels/abl`, que pode ser ignorada).

## Limites

- **Utilidade aqui é para a leitura de L1, não para o modelo fundo.** Um kernel fraco em L1 pode alimentar bem a L2. Só o treino em cadeia responde isso.
- Só microscopia: os 7 datasets de x1 são parasitas. aid, flowers e remote_sensing ficaram de fora.
- Só 50%, 200 px, 5x5, 64 filtros, seed 42. O caso de 3 imagens por classe é simulado.
- Regime recortado tem 3 datasets; o IC hierárquico entre eles é largo (+0,061 [−0,013; 0,154]) porque helminth_larvae não pôde repor. Os intervalos por imagem são mais apertados, mas valem só para esses datasets.
- Regras simuladas não treinaram: a reposição usa kernel inicial. Dá para confiar nisso em L1 porque o backprop quase não muda nada, mas não é prova.
- Fiz a escolha de leitura e de regras olhando a validação; o teste de dataset deixado de fora protege parte disso, não tudo.
- A linha "dispara no objeto" usa uma estimativa minha sem máscara (mediana de uma faixa dentro da caixa não preta). Não validei contra máscara, de propósito.
- Em Protozoan o treino foi cortado em 4.000 imagens, para caber no tempo.

## Arquivos

Scripts em `/tmp/claude-1033/-dados-home-moliveira-Scalable-Hybrid-FLIM/990a33ea-4327-4228-ac99-2136fb213621/scratchpad/law/`: `lk_replay.py`, `lk_feats.py`, `lk_props.py`, `lk_abl.py`, `lk_prune.py`, `lk_rules.py`, `lk_rule_defs.py`, `lk_evalbank.py`, `lk_an1.py` a `lk_an6.py`, `lk_boot.py`, `lk_boot2.py`, `lk_probe.py`, `lk_probe2.py`.
Dados em `/exps/moliveira/law_kernels/` (features, propriedades, ablação em `abl2/`, regras em `rules/` e `rules3/`).
Nenhum `.py` do worktree, do SPiFiL ou dos dados foi alterado. GPU: sempre com `GPU_CAP_MIB=6000`, a placa mais livre e guarda de 40 GB.
