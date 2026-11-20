# Curvas de treino no W&B dos melhores modelos do GA: a init SPiFiL causa picos?

Fonte: W&B `ophira-ai/progressive_mo_dehb`, 260 runs baixados pela API (nenhum offline). O `flim-ssl` não tem runs dessas 5 bases e ficou de fora. São **89 modelos** nas 5 bases (flowers 14, remote_sensing 42, ham10000 13, aid 11, coconut_trees 9). A seleção (`models_selected.json`) junta três critérios: o melhor por κ de validação em cada (busca × estrato), os 3 melhores de cada base e os modelos de `readout_x/results/ga_best_tabela.md`. Cada modelo foi seguido pela **cadeia de fidelidades** f5→f15→f45→f100→f200. A promoção retoma o encoder e a cabeça do checkpoint anterior, com Adam novo. O AE local e o fit SPiFiL só rodam no f5, e o SSL continua nas fidelidades seguintes.
Cadeia de crescimento (B0→B2, `growth/calib` e `growth/summaries`): os `result.json` têm só o início e o fim do AE por bloco (`ae_logs`) e o `ssl_final_loss`. Não há curva nem run W&B, então essa cadeia entra só como resumo.

Arquivos: `metricas_por_modelo.csv` (todas as métricas), `controle_random_vs_spifil_rs.csv`, `crescimento_ae_resumo.csv` e `figures/curvas_<base>.png` (os 3 melhores candidatos de cada base). Em cada figura, a coluna 1 mostra o AE por bloco, a 2 o SSL na cadeia (total, inv e SIGReg) e a 3 as normas de gradiente e o κ por fidelidade. A figura `figures/controle_spifil_vs_random_rs.png` traz o controle. Scripts: `select_models.py`, `select_extra.py`, `fetch_wandb.py` e `analyze.py`.

## Critério

Cada série (perda de reconstrução do AE por bloco, perda total do SSL, inv, SIGReg e ‖g‖) foi medida assim:
- **Pico inicial** = máx(primeiros max(3, 10%) pontos depois do 1º) / valor inicial. **Grande se > 2×**, moderado entre 1,2× e 2×, ausente se ≤ 1,2×. Também medimos pico/mediana. O valor inicial é a referência porque a pergunta é se o treino *se afasta* do ponto de partida (a init). Dobrar a perda logo após começar significa que o gradiente empurrou o modelo para um lugar pior que a init. Abaixo de 1,2× o efeito fica no nível do ruído de mini-batch (o cv das diferenças do SSL é ~0,12).
- **Junção decoder→bloco** (só no AE `decoder_first`): na 1ª época só o decoder treina (bloco SPiFiL com lr=0). Na 2ª o bloco entra. Medimos máx(3 passos após a entrada)/média(3 antes). **Esse é o teste direto de "a init SPiFiL é ruim"**: se ela fosse ruim, a perda subiria quando o backprop passa a mexer nos pesos SPiFiL.
- **Oscilação**: cv = std(diff)/|mediana| e fração de reversões de sinal.
- **Convergência**: inclinação relativa nos últimos 30%, fim/x0 e fim/mín. Converge se fim ≤ 1,1·mín ou inclinação ≤ 0. Diverge se fim > 1,5·mín com inclinação > 0.
- **Salto na promoção**: 1ª época da fidelidade nova / última época da anterior.

## Resultado por base

| base | n | AE: modelos com pico > 2× | AE pico (mediana / máx) | AE junção decoder→bloco (máx) | AE fim/x0 (mediana) | SSL pico inicial máx | SSL fim/x0 (mediana) | salto na promoção (mediana / máx) | SSL converge |
|---|---|---|---|---|---|---|---|---|---|
| flowers | 14 | 11 | 5,0× / 509× | 1,15× | 0,05 | 0,99 (nenhum) | 0,64 | 0,97 / 2,2× | 14/14 |
| remote_sensing | 42 | 28 | 13,8× / 327× | 2,34× (1 caso) | 0,20 | 1,11 (nenhum > 1,2) | 0,58 | 1,02 / 3,2× | 42/42 |
| ham10000 | 13 | 8 | 3,0× / 1524× | 1,33× | 0,11 | 0,83 (nenhum) | 0,44 | 1,43 / 2,4× | 13/13 |
| aid | 11 | 4 | 0,96× / 995× | 1,22× | 0,12 | 0,79 (nenhum) | 0,62 | 1,26 / 13× (SSL `ae`, perda ~0,003) | 11/11 |
| coconut_trees | 9 | 7 | 19× / 1538× | 1,48× | 0,03 | 1,23 (1 moderado) | 0,48 | 1,06 / 1,3× | 9/9 |

### AE local: há picos, e alguns são enormes, mas não vêm da init SPiFiL
- **O pico é sempre no passo 1**, o 1º update do Adam, e 42 de 89 modelos passam de 2× (em 17 modelos o pico de algum bloco passa de 100×; máx. 1538×). Ele aparece sobretudo nos **blocos profundos** (2–5), onde a escala das features é maior: o x0 mediano do bloco com pico é 1,19, contra 0,46 nos blocos sem pico. A norma de gradiente do bloco sobe junto (ex.: 2,8 → 190).
- **Prova de que não é a init:** nos 34 modelos `decoder_first`, o bloco SPiFiL fica **congelado (lr=0) na 1ª época**, e mesmo assim 17 deles têm o mesmo pico no passo 1. O pico vem do **decoder aleatório** dando o primeiro passo do Adam (lr 1e-3) sobre features de grande magnitude. Exemplo: ham10000 hybrid c00043, bloco 2, com x0 = 516 → 786 578 no passo 1, com o bloco congelado.
- **Quando o backprop passa a mexer nos pesos SPiFiL** (junção decoder→bloco), a perda praticamente não se move: mediana 1,04×, 75% ≤ 1,19×, máx. 2,34× (remote_sensing lemonade c00107, 1 caso, que depois desce). Os modelos `joint` têm mais picos (25/30) que os `decoder_first` (17/34): o aquecimento do decoder suaviza o transiente.
- **Recuperação:** volta abaixo do valor inicial em 1–54 passos (mediana poucos passos). O AE termina em fim/x0 ≈ 0,03–0,20, ou seja, cai de 5 a 30×. Só 1/89 não volta ao x0: remote_sensing lemonade_C01_final c00014, bloco 3, `pixel`/`joint`/`lowlr`, fim/x0 = 4,0. Mesmo esse modelo tem κ 0,584.
- A oscilação é alta (cv mediano 2,6; ~44% de reversões), mas vem do alvo `masked` (máscaras aleatórias) e do lote pequeno, não de instabilidade de tendência.

### SSL: não há pico inicial, e o modelo continua convergindo
- Pico inicial / valor inicial: mediana 0,57 e máximo 1,23 (coconut hybrid c00004, moderado). **Nenhum dos 89 passa de 2×**, e só 1 passa de 1,2×. A perda desce desde o 1º ponto, e inv e SIGReg descem juntos.
- 89/89 convergem e nenhum diverge. A inclinação final é ≤ 0 e o fim/x0 fica em 0,04–0,99 (mediana ~0,55). A oscilação é baixa: cv das diferenças ~0,12.
- O único "dente" no SSL é o **salto na promoção de fidelidade**, quando o treino é retomado com um otimizador Adam novo. Mediana ~1,0–1,4×. Os maiores valores são 13× (aid c00038, SSL `ae`, perda absoluta ~0,003→0,06) e 2–3×, e todos se desfazem em <1 época e terminam abaixo do nível anterior. É um artefato do reinício do otimizador, não da init.
- O κ de validação nunca cai ao longo da cadeia: em 39/39 cadeias, κ_final ≥ κ_f5 (ganho mediano +0,007, máx. +0,076). Essa medida tem um viés: só os modelos que melhoram são promovidos.
- Norma do gradiente do encoder no SSL: pico inicial > 2× em só 4/89 (máx. 6,5×, em remote_sensing v3_old c00042, init aleatória).

### Controle histórico (remote_sensing, buscas v3/v3_old, f5): só referência
Comparação com init aleatória He (24 runs) contra SPiFiL (32 runs): pico do AE > 2× em **37,5% (aleatória) contra 31% (SPiFiL)**. Medianas: 1,13× contra 0,93×. No SSL, a mediana do pico é 0,78 contra 0,85, e nenhum passa de 2× em nenhum dos dois grupos. O transiente do AE aparece com qualquer init, então não é uma assinatura do SPiFiL.

### Cadeia de crescimento (sem curva)
Os 18 blocos de AE em `growth/` terminam com fim/início entre 0,002 e 0,13 e estabilidade relativa (std da última época / perda final) entre 0,06 e 0,23, sem nenhum passo pulado. Todos convergem. O pico não pode ser medido (não há curva salva). Um detalhe: no remote_sensing B1g0c03, a perda inicial cresce com a profundidade (0,47 → 28 no bloco 5), o que é coerente com a explicação "escala das features × decoder aleatório".

## Conclusão: a init SPiFiL causa picos?

| base | AE local | SSL |
|---|---|---|
| flowers | Há pico no passo 1 (11/14 > 2×, até 509×), causado pelo decoder aleatório. Ao liberar os pesos SPiFiL, ≤ 1,15×. **Não é a init.** Converge (fim/x0 0,05). | **Não.** Sem pico (máx. 0,99×). Converge. |
| remote_sensing | Pico no passo 1 em 28/42 (até 327×), igual com init aleatória. Junção ≤ 2,34× (1 caso). **Não é a init.** Converge, exceto 1 bloco (c00014). | **Não.** Máx. 1,11×. Converge 42/42. |
| ham10000 | Pico em 8/13 (até 1524×, com o bloco congelado). Junção ≤ 1,33×. **Não é a init.** | **Não.** Máx. 0,83×. Converge. |
| aid | Menos picos (4/11). Junção ≤ 1,22×. **Não é a init.** | **Não.** Máx. 0,79×. Converge (só saltos de reinício na promoção). |
| coconut_trees | Pico em 7/9 (até 1538×) nos blocos profundos. Junção ≤ 1,48×. **Não é a init.** | **Não.** Máx. 1,23× (moderado, 1 caso). Converge. |

**Resposta curta:** no SSL, o treino segue convergindo desde o 1º passo, sem pico inicial (nenhum > 2×; mediana 0,57× do valor inicial) e sem divergência nos 89 modelos. No AE local há um pico grande (> 2×, às vezes > 100×) no 1º passo dos blocos profundos. Esse pico aparece também com o bloco SPiFiL congelado e com init aleatória, e some em poucos passos. Ele vem do decoder aleatório diante da escala grande das features, **não de uma init SPiFiL ruim**. Quando o backprop passa a mexer nos pesos SPiFiL, a perda sobe no máximo ~1,0–1,5× (um caso com 2,3×). Se quisermos eliminar o transiente, basta normalizar a entrada do decoder ou fazer warm-up de lr no decoder. Isso não exige mudar a init.

## Tabela por modelo
x0 = 1º valor logado. "recupera em" = passos até voltar ≤ x0. Na coluna de oscilação, cv = std(diff)/|mediana| e rev = fração de reversões. "(tabela)" = modelo de `ga_best_tabela.md`. Init: `inherited+spifil` = camadas herdadas de outro candidato mais camadas SPiFiL novas; `random_he` = só as buscas v3 antigas.

### flowers

| modelo (busca/cand/fid) | estrato · params | κ val GA | init | AE: alvo/sched | AE pico ini (pior bloco): x pico/x0 · blocos>2× · recupera em | AE junção (decoder→bloco) | AE fim/x0 | AE converge | SSL método: pico ini/x0 | SSL oscil. cv·rev | SSL fim/x0 · incl. final | salto na promoção (máx) | SSL converge | κ na cadeia |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid/c00029/f5 (tabela) | tiny · 0.23M | 0.775 | spifil | masked/joint | **GRANDE** 36.9× · 2/2 · 8 passos | - | 0.50 | sim | simclr: nao 0.77 | 0.05 · 0.00 | 0.76 · - | - | sim | f5:0.775 |
| hybrid/c00004/f200 | huge · 1.89M | 0.772 | spifil | pixel/joint | **GRANDE** 428.3× · 1/4 · 14 passos | - | 0.04 | sim | lejepa: nao 0.92 | 0.57 · 0.54 | 0.07 · -0.001 | 1.04× | sim | f5:0.729 > f15:0.742 > f45:0.735 > f100:0.750 > f200:0.772 |
| hybrid/c00020/f5 (tabela) | huge · 2.49M | 0.763 | spifil | feature/joint | **GRANDE** 3.8× · 2/4 · 3 passos | - | 0.18 | sim | ae+sigreg: nao 0.36 | 0.14 · 0.00 | 0.66 · - | - | sim | f5:0.763 |
| lemonade/c00003/f15 (tabela) | medium · 1.31M | 0.757 | spifil | pixel/decoder_first | nao 1.1× · 0/6 · - | 1.05× | 0.05 | sim | lejepa: nao 0.95 | 0.04 · 0.00 | 0.68 · -0.021 | 0.89× | sim | f5:0.750 > f15:0.757 |
| hybrid/c00003/f45 (tabela) | medium · 0.89M | 0.757 | spifil | pixel/joint | **GRANDE** 2.7× · 1/3 · 1 passos | - | 0.02 | sim | lejepa: nao 0.76 | 0.04 · 0.56 | 0.67 · -0.001 | 0.97× | sim | f5:0.738 > f15:0.745 > f45:0.757 |
| hybrid/c00002/f200 (tabela) | small · 0.68M | 0.748 | spifil | pixel/joint | **GRANDE** 508.7× · 2/4 · 34 passos | - | 0.65 | sim | lejepa: nao 0.55 | 0.15 · 0.55 | 0.28 · -0.002 | 2.18× | sim | f5:0.735 > f15:0.735 > f45:0.744 > f100:0.735 > f200:0.748 |
| hybrid/c00028/f5 | over · 3.63M | 0.738 | spifil | masked/joint | **GRANDE** 465.2× · 5/6 · 10 passos | - | 0.24 | sim | simclr+weak_sigreg: nao 0.93 | 0.01 · 0.00 | 0.95 · - | - | sim | f5:0.738 |
| lemonade/c00004/f15 (tabela) | huge · 2.18M | 0.726 | spifil | pixel/decoder_first | **GRANDE** 2.7× · 2/6 · 1 passos | 1.01× | 0.04 | sim | lejepa: nao 0.94 | 0.03 · 0.00 | 0.61 · -0.030 | 0.96× | sim | f5:0.717 > f15:0.726 |
| lemonade/c00013/f5 (tabela) | small · 0.48M | 0.705 | spifil | pixel/decoder_first | **GRANDE** 204.8× · 1/4 · 21 passos | 1.10× | 0.05 | sim | lejepa: nao 0.55 | 0.16 · 0.00 | 0.56 · - | - | sim | f5:0.705 |
| lemonade/c00001/f45 (tabela) | tiny · 0.18M | 0.702 | spifil | pixel/decoder_first | nao 0.4× · 0/3 · - | 1.15× | 0.02 | sim | lejepa: nao 0.59 | 1.06 · 0.46 | 0.09 · -0.004 | 1.16× | sim | f5:0.693 > f15:0.690 > f45:0.702 |
| spifil/c00003/f15 | tiny · 0.24M | 0.675 | spifil | pixel/- | **GRANDE** 3.8× · 1/2 · 1 passos | - | 0.03 | sim | lejepa: nao 0.53 | 0.50 · 0.18 | 0.28 · -0.025 | 0.92× | sim | f5:0.641 > f15:0.675 |
| spifil/c00046/f5 | medium · 0.82M | 0.647 | spifil | pixel/- | **GRANDE** 6.1× · 1/5 · 1 passos | - | 0.03 | sim | ae+weak_sigreg: nao 0.90 | 0.02 · 0.00 | 0.95 · - | - | sim | f5:0.647 |
| spifil/c00006/f5 | small · 0.56M | 0.621 | spifil | pixel/- | **GRANDE** 9.3× · 2/3 · 1 passos | - | 0.03 | sim | lejepa: nao 0.74 | 0.19 · 0.00 | 0.61 · - | - | sim | f5:0.621 |
| spifil/c00042/f5 | huge · 2.81M | 0.470 | spifil | feature/- | nao 1.1× · 0/7 · - | - | 0.11 | sim | lejepa_weak: nao 0.99 | 0.00 · 0.00 | 0.99 · - | - | sim | f5:0.470 |

### remote_sensing

| modelo (busca/cand/fid) | estrato · params | κ val GA | init | AE: alvo/sched | AE pico ini (pior bloco): x pico/x0 · blocos>2× · recupera em | AE junção (decoder→bloco) | AE fim/x0 | AE converge | SSL método: pico ini/x0 | SSL oscil. cv·rev | SSL fim/x0 · incl. final | salto na promoção (máx) | SSL converge | κ na cadeia |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lemonade/c00028/f5 (tabela) | huge · 1.82M | 0.611 | inherited+spifil | pixel/decoder_first | **GRANDE** 80.1× · 1/2 · 6 passos | 0.84× | 0.13 | sim | lejepa: nao 0.38 | 0.14 · 0.00 | 0.67 · - | - | sim | f5:0.611 |
| hybrid_C01_final/c00020/f45 (tabela) | huge · 2.76M | 0.605 | spifil | feature/joint | **GRANDE** 15.4× · 2/4 · 3 passos | - | 0.41 | sim | ae+sigreg: nao 0.35 | 0.35 · 0.27 | 0.15 · -0.046 | 2.75× | sim | f5:0.605 > f15:0.598 > f45:0.605 |
| hybrid_C01_final/c00020/f5 | huge · 2.76M | 0.605 | spifil | feature/joint | **GRANDE** 15.4× · 2/4 · 3 passos | - | 0.41 | sim | ae+sigreg: nao 0.35 | 0.18 · 0.00 | 0.57 · - | - | sim | f5:0.605 |
| hybrid/c00020/f5 (tabela) | huge · 2.38M | 0.599 | spifil | feature/joint | **GRANDE** 12.1× · 1/4 · 2 passos | - | 0.38 | não* | ae+sigreg: nao 0.27 | 0.32 · 0.00 | 0.37 · - | - | sim | f5:0.599 |
| hybrid_C01_final/c00039/f5 | medium · 0.90M | 0.596 | spifil | masked/joint | **GRANDE** 38.1× · 2/5 · 2 passos | - | 0.26 | não* | lejepa_weak: nao 0.20 | 0.60 · 0.00 | 0.42 · - | - | sim | f5:0.596 |
| lemonade/c00107/f15 (tabela) | medium · 1.26M | 0.596 | spifil | pixel/decoder_first | **GRANDE** 10.8× · 1/6 · 1 passos | 2.34× | 0.06 | sim | lejepa: nao 0.50 | 0.08 · 0.00 | 0.57 · -0.058 | - | sim | f15:0.596 |
| hybrid/c00009/f5 (tabela) | tiny · 0.13M | 0.591 | spifil | pixel/joint | **GRANDE** 54.5× · 1/2 · 6 passos | - | 0.14 | sim | lejepa: nao 0.55 | 0.05 · 0.00 | 0.54 · - | - | sim | f5:0.591 |
| lemonade_C01_final/c00081/f15 (tabela) | huge · 1.78M | 0.591 | inherited+spifil | pixel/decoder_first | **GRANDE** 89.5× · 2/3 · 5 passos | 0.97× | 0.04 | sim | lejepa: nao 0.42 | 0.10 · 0.00 | 0.58 · -0.030 | - | sim | f15:0.591 |
| lemonade_C01_final/c00061/f15 | small · 0.39M | 0.585 | spifil | feature/decoder_first | nao 0.8× · 0/2 · - | 0.97× | 0.19 | sim | simclr+sigreg: nao 0.77 | 0.06 · 0.18 | 0.52 · -0.026 | 1.04× | sim | f5:0.571 > f15:0.585 |
| lemonade_C01_final/c00014/f5 | medium · 1.32M | 0.584 | spifil | pixel/joint | **GRANDE** 327.5× · 2/4 · não volta | - | 4.03 | não* | lejepa: nao 0.23 | 0.37 · 0.00 | 0.43 · - | - | sim | f5:0.584 |
| lemonade_C01_final/c00105/f45 | tiny · 0.09M | 0.583 | spifil | feature/joint | nao 0.9× · 0/2 · - | - | 0.05 | sim | lejepa: nao 0.88 | 0.11 · 0.32 | 0.34 · -0.009 | - | sim | f45:0.583 |
| hybrid/c00027/f5 | medium · 0.98M | 0.579 | spifil | feature/joint | nao 0.8× · 0/2 · - | - | 0.24 | sim | simclr+sigreg: nao 0.73 | 0.06 · 0.00 | 0.81 · - | - | sim | f5:0.579 |
| hybrid_C01_final/c00010/f200 (tabela) | tiny · 0.25M | 0.577 | spifil | pixel/decoder_first | nao 0.8× · 0/2 · - | 1.34× | 0.08 | sim | lejepa: nao 0.43 | 0.25 · 0.53 | 0.19 · -0.003 | 2.63× | sim | f5:0.564 > f15:0.551 > f45:0.571 > f100:0.564 > f200:0.577 |
| spifil/c00306/f100 | tiny · 0.20M | 0.572 | spifil | masked/- | **GRANDE** 3.3× · 2/2 · 1 passos | - | 0.28 | não* | ae+weak_sigreg: nao 0.92 | 0.53 · 0.60 | 0.08 · -0.003 | - | sim | f100:0.572 |
| hybrid_C01_final/c00058/f45 (tabela) | small · 0.37M | 0.570 | spifil | masked/decoder_first | **GRANDE** 86.3× · 1/2 · 2 passos | 1.09× | 0.35 | sim | ae+sigreg: nao 0.37 | 0.63 · 0.32 | 0.16 · -0.017 | 3.21× | sim | f5:0.564 > f15:0.564 > f45:0.570 |
| hybrid/c00015/f5 (tabela) | medium · 1.48M | 0.570 | spifil | pixel/joint | **GRANDE** 59.4× · 2/3 · 1 passos | - | 1.01 | sim | lejepa: nao 0.47 | 0.12 · 0.00 | 0.61 · - | - | sim | f5:0.570 |
| lemonade/c00054/f15 | small · 0.74M | 0.569 | spifil | masked/joint | **GRANDE** 22.6× · 2/6 · 2 passos | - | 0.18 | não* | lejepa: nao 0.82 | 0.07 · 0.00 | 0.59 · -0.012 | 0.90× | sim | f5:0.569 > f15:0.569 |
| lemonade/c00054/f45 (tabela) | small · 0.74M | 0.569 | spifil | masked/joint | **GRANDE** 22.6× · 2/6 · 2 passos | - | 0.18 | não* | lejepa: nao 0.82 | 0.05 · 0.44 | 0.52 · -0.002 | 0.95× | sim | f5:0.569 > f15:0.569 > f45:0.569 |
| spifil/c00301/f100 | medium · 1.11M | 0.557 | spifil | masked/- | **GRANDE** 16.5× · 2/3 · 2 passos | - | 0.09 | sim | lejepa_weak: nao 0.22 | 0.85 · 0.31 | 0.09 · -0.001 | - | sim | f100:0.557 |
| lemonade/c00012/f45 (tabela) | tiny · 0.03M | 0.556 | spifil | pixel/joint | **GRANDE** 3.9× · 1/2 · 1 passos | - | 0.25 | não* | lejepa: nao 0.77 | 0.17 · 0.29 | 0.15 · -0.010 | 0.92× | sim | f5:0.549 > f15:0.537 > f45:0.556 |
| hybrid/c00026/f5 | small · 0.79M | 0.542 | spifil | feature/decoder_first | nao 0.9× · 0/3 · - | 0.94× | 0.26 | não* | ae: nao 0.13 | 1.01 · 0.00 | 0.20 · - | - | sim | f5:0.542 |
| spifil/c00004/f5 | small · 0.34M | 0.535 | spifil | pixel/- | **GRANDE** 2.4× · 1/2 · 1 passos | - | 0.10 | sim | lejepa: nao 0.38 | 0.37 · 0.00 | 0.36 · - | - | sim | f5:0.535 |
| hybrid/c00006/f5 (tabela) | small · 0.32M | 0.528 | spifil | pixel/joint | **GRANDE** 39.6× · 1/3 · 17 passos | - | 0.61 | sim | lejepa: nao 0.52 | 0.09 · 0.00 | 0.77 · - | - | sim | f5:0.528 |
| strat_prefix_2130/c00003/f100 | medium · 0.90M | 0.517 | spifil | pixel/- | **GRANDE** 52.4× · 3/3 · 4 passos | - | 0.13 | sim | lejepa: nao 0.40 | 0.74 · 0.54 | 0.08 · -0.002 | 1.00× | sim | f5:0.441 > f15:0.492 > f45:0.504 > f100:0.517 |
| v3/c00008/f5 | small · 0.63M | 0.514 | random_he | pixel/- | **GRANDE** 8.1× · 1/2 · 2 passos | - | 0.20 | sim | lejepa: nao 0.69 | 0.09 · 0.00 | 0.66 · - | - | sim | f5:0.514 |
| strat_prefix_2130/c00016/f5 | huge · 1.87M | 0.507 | spifil | pixel/- | **GRANDE** 71.2× · 2/3 · 8 passos | - | 0.36 | sim | lejepa: nao 0.67 | 0.12 · 0.00 | 0.66 · - | - | sim | f5:0.507 |
| strat_prefix_2130/c00009/f5 | tiny · 0.10M | 0.501 | spifil | pixel/- | nao 0.8× · 0/2 · - | - | 0.10 | sim | lejepa: nao 0.28 | 0.70 · 0.00 | 0.36 · - | - | sim | f5:0.501 |
| spifil/c00099/f15 | huge · 3.05M | 0.493 | spifil | feature/- | nao 1.0× · 0/4 · - | - | 0.23 | sim | ae: nao 0.15 | 0.85 · 0.38 | 0.23 · -0.026 | - | sim | f15:0.493 |
| v3/c00006/f5 | tiny · 0.24M | 0.487 | random_he | pixel/- | moderado 1.6× · 0/2 · 1 passos | - | 0.08 | não* | lejepa: nao 0.71 | 0.08 · 0.00 | 0.68 · - | - | sim | f5:0.487 |
| strat_preHybrid_2325/c00002/f5 | small · 0.45M | 0.487 | spifil | pixel/- | **GRANDE** 19.1× · 2/3 · 2 passos | - | 0.18 | sim | lejepa: nao 0.38 | 0.24 · 0.00 | 0.61 · - | - | sim | f5:0.487 |
| v3_old/c00008/f5 | small · 0.32M | 0.487 | random_he | pixel/- | nao 1.1× · 0/2 · - | - | 0.08 | sim | lejepa: nao 0.75 | 0.07 · 0.00 | 0.69 · - | - | sim | f5:0.487 |
| spifil/c00021/f15 | over · 14.78M | 0.486 | spifil | masked/- | **GRANDE** 165.0× · 2/8 · 9 passos | - | 0.29 | não* | simclr+weak_sigreg: nao 0.84 | 0.04 · 0.00 | 0.61 · -0.013 | 1.00× | sim | f5:0.411 > f15:0.486 |
| strat_prefix_2130/c00038/f5 | small · 0.35M | 0.483 | spifil | feature/- | nao 1.0× · 0/3 · - | - | 0.19 | sim | simclr+weak_sigreg: nao 0.78 | 0.03 · 0.00 | 0.87 · - | - | sim | f5:0.483 |
| lemonade_seeded/c00001/f100 | tiny · 0.16M | 0.480 | spifil | pixel/decoder_first | nao 0.5× · 0/3 · - | 1.35× | 0.04 | sim | lejepa: nao 0.42 | 1.45 · 0.48 | 0.03 · -0.002 | 1.25× | sim | f5:0.466 > f15:0.467 > f45:0.466 > f100:0.480 |
| v3_old/c00004/f5 | tiny · 0.05M | 0.478 | random_he | pixel/- | nao 0.6× · 0/2 · - | - | 0.10 | não* | lejepa: nao 0.86 | 0.04 · 0.00 | 0.79 · - | - | sim | f5:0.478 |
| strat_fid120_1415/c00001/f5 | tiny · 0.02M | 0.475 | spifil | pixel/- | nao 1.0× · 0/2 · - | - | 0.29 | sim | lejepa: nao 0.89 | 0.01 · 0.00 | 0.83 · - | - | sim | f5:0.475 |
| v3/c00015/f5 | huge · 2.08M | 0.466 | spifil | pixel/- | **GRANDE** 17.6× · 2/3 · 3 passos | - | 0.38 | não* | lejepa: nao 0.77 | 0.02 · 0.00 | 0.84 · - | - | sim | f5:0.466 |
| v3_old/c00042/f5 | huge · 1.87M | 0.459 | random_he | pixel/- | **GRANDE** 66.1× · 1/5 · 8 passos | - | 0.21 | não* | lejepa: nao 0.97 | 0.05 · 0.00 | 0.67 · - | - | sim | f5:0.459 |
| strat_preHybrid_2325/c00001/f5 | tiny · 0.02M | 0.439 | spifil | pixel/- | nao 1.0× · 0/2 · - | - | 0.29 | sim | lejepa: nao 0.89 | 0.02 · 0.00 | 0.82 · - | - | sim | f5:0.439 |
| v3_old/c00018/f5 | over · 8.97M | 0.439 | spifil | pixel/- | **GRANDE** 104.7× · 4/5 · 12 passos | - | 0.44 | não* | lejepa: nao 1.11 | 0.04 · 0.00 | 0.81 · - | - | sim | f5:0.439 |
| v3_old/c00046/f5 | medium · 1.12M | 0.436 | spifil | pixel/- | **GRANDE** 4.1× · 2/3 · 2 passos | - | 0.12 | sim | lejepa: nao 0.73 | 0.08 · 0.00 | 0.80 · - | - | sim | f5:0.436 |
| v3/c00018/f5 | over · 8.97M | 0.398 | spifil | pixel/- | **GRANDE** 119.4× · 4/5 · 13 passos | - | 0.43 | sim | lejepa: nao 1.06 | 0.01 · 0.00 | 0.81 · - | - | sim | f5:0.398 |

### ham10000

| modelo (busca/cand/fid) | estrato · params | κ val GA | init | AE: alvo/sched | AE pico ini (pior bloco): x pico/x0 · blocos>2× · recupera em | AE junção (decoder→bloco) | AE fim/x0 | AE converge | SSL método: pico ini/x0 | SSL oscil. cv·rev | SSL fim/x0 · incl. final | salto na promoção (máx) | SSL converge | κ na cadeia |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid/c00034/f100 | small · 0.47M | 0.621 | spifil | masked/decoder_first | **GRANDE** 3.2× · 1/2 · 1 passos | 1.19× | 0.17 | não* | simclr+weak_sigreg: nao 0.80 | 0.01 · 0.60 | 0.85 · -0.001 | 1.07× | sim | f5:0.601 > f15:0.601 > f45:0.601 > f100:0.621 |
| lemonade/c00061/f5 (tabela) | medium · 1.05M | 0.614 | inherited+spifil | feature/joint | nao 1.0× · 0/1 · - | - | 0.12 | sim | ae: nao 0.12 | 1.50 · 1.00 | 0.25 · - | - | sim | f5:0.614 |
| hybrid/c00043/f15 | medium · 1.35M | 0.612 | spifil | masked/decoder_first | **GRANDE** 1523.7× · 2/3 · 28 passos | 1.15× | 0.09 | sim | ae+sigreg: nao 0.43 | 0.13 · 0.18 | 0.33 · -0.159 | 1.15× | sim | f5:0.612 > f15:0.612 |
| hybrid/c00043/f45 (tabela) | medium · 1.35M | 0.612 | spifil | masked/decoder_first | **GRANDE** 1523.7× · 2/3 · 28 passos | 1.15× | 0.09 | sim | ae+sigreg: nao 0.43 | 0.25 · 0.37 | 0.28 · -0.002 | 2.37× | sim | f5:0.612 > f15:0.612 > f45:0.612 |
| hybrid/c00034/f45 (tabela) | small · 0.47M | 0.601 | spifil | masked/decoder_first | **GRANDE** 3.2× · 1/2 · 1 passos | 1.19× | 0.17 | não* | simclr+weak_sigreg: nao 0.80 | 0.01 · 0.54 | 0.87 · -0.001 | 1.04× | sim | f5:0.601 > f15:0.601 > f45:0.601 |
| lemonade/c00048/f15 (tabela) | small · 0.40M | 0.598 | inherited+spifil | pixel/joint | **GRANDE** 693.0× · 2/3 · 54 passos | - | 0.15 | sim | lejepa: nao 0.54 | 0.12 · 0.18 | 0.55 · -0.013 | 1.22× | sim | f5:0.577 > f15:0.598 |
| hybrid/c00020/f5 (tabela) | huge · 2.53M | 0.586 | spifil | feature/joint | **GRANDE** 4.4× · 1/4 · 2 passos | - | 0.26 | sim | ae+sigreg: nao 0.31 | 0.17 · 0.00 | 0.67 · - | - | sim | f5:0.586 |
| hybrid/c00025/f5 (tabela) | tiny · 0.17M | 0.586 | spifil | masked/joint | **GRANDE** 3.0× · 1/3 · 1 passos | - | 0.07 | sim | simclr: nao 0.83 | 0.03 · 0.00 | 0.87 · - | - | sim | f5:0.586 |
| lemonade/c00070/f100 | huge · 1.53M | 0.585 | spifil | feature/decoder_first | nao 0.8× · 0/4 · - | 1.03× | 0.11 | sim | lejepa: nao 0.17 | 0.20 · 0.65 | 0.33 · -0.004 | 1.57× | sim | f5:0.574 > f15:0.565 > f45:0.583 > f100:0.585 |
| lemonade/c00044/f5 (tabela) | tiny · 0.30M | 0.584 | spifil | pixel/decoder_first | **GRANDE** 2.9× · 1/3 · 1 passos | 1.33× | 0.01 | sim | simclr: nao 0.83 | 0.10 · 0.00 | 0.70 · - | - | sim | f5:0.584 |
| lemonade/c00070/f45 (tabela) | huge · 1.53M | 0.583 | spifil | feature/decoder_first | nao 0.8× · 0/4 · - | 1.03× | 0.11 | sim | lejepa: nao 0.17 | 0.25 · 0.63 | 0.41 · -0.003 | 1.57× | sim | f5:0.574 > f15:0.565 > f45:0.583 |
| hybrid_C01_2049/c00001/f45 | tiny · 0.02M | 0.526 | spifil | pixel/decoder_first | nao 1.0× · 0/2 · - | 1.02× | 0.16 | sim | lejepa: nao 0.69 | 0.11 · 0.32 | 0.44 · -0.006 | 1.43× | sim | f5:0.518 > f15:0.498 > f45:0.526 |
| hybrid_fullepoch_1555/c00001/f45 | tiny · 0.02M | 0.478 | spifil | pixel/decoder_first | nao 1.0× · 0/2 · - | 1.17× | 0.06 | sim | lejepa: nao 0.56 | 0.20 · 0.29 | 0.36 · -0.007 | 1.61× | sim | f5:0.469 > f15:0.465 > f45:0.478 |

### aid

| modelo (busca/cand/fid) | estrato · params | κ val GA | init | AE: alvo/sched | AE pico ini (pior bloco): x pico/x0 · blocos>2× · recupera em | AE junção (decoder→bloco) | AE fim/x0 | AE converge | SSL método: pico ini/x0 | SSL oscil. cv·rev | SSL fim/x0 · incl. final | salto na promoção (máx) | SSL converge | κ na cadeia |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid/c00020/f5 (tabela) | huge · 2.14M | 0.845 | spifil | feature/joint | **GRANDE** 4.9× · 1/4 · 2 passos | - | 0.14 | sim | ae+sigreg: nao 0.37 | 0.08 · 0.00 | 0.71 · - | - | sim | f5:0.845 |
| hybrid/c00043/f45 | medium · 1.26M | 0.844 | spifil | masked/decoder_first | **GRANDE** 994.7× · 3/3 · 6 passos | 1.01× | 0.28 | sim | ae+sigreg: nao 0.38 | 0.43 · 0.29 | 0.21 · -0.009 | 3.13× | sim | f5:0.837 > f15:0.841 > f45:0.844 |
| hybrid/c00043/f15 | medium · 1.26M | 0.841 | spifil | masked/decoder_first | **GRANDE** 994.7× · 3/3 · 6 passos | 1.01× | 0.28 | sim | ae+sigreg: nao 0.38 | 0.21 · 0.18 | 0.29 · -0.041 | 1.34× | sim | f5:0.837 > f15:0.841 |
| hybrid/c00043/f5 (tabela) | medium · 1.26M | 0.837 | spifil | masked/decoder_first | **GRANDE** 994.7× · 3/3 · 6 passos | 1.01× | 0.28 | sim | ae+sigreg: nao 0.38 | 0.06 · 0.00 | 0.66 · - | - | sim | f5:0.837 |
| hybrid/c00038/f45 | small · 0.50M | 0.837 | spifil | feature/decoder_first | nao 0.9× · 0/5 · - | 0.99× | 0.08 | sim | ae: nao 0.43 | 3.85 · 0.32 | 0.05 · -0.006 | 13.17× | sim | f5:0.830 > f15:0.832 > f45:0.837 |
| hybrid/c00038/f15 (tabela) | small · 0.50M | 0.832 | spifil | feature/decoder_first | nao 0.9× · 0/5 · - | 0.99× | 0.08 | sim | ae: nao 0.43 | 3.61 · 0.18 | 0.07 · -0.025 | 6.14× | sim | f5:0.830 > f15:0.832 |
| hybrid/c00025/f45 | tiny · 0.21M | 0.820 | spifil | masked/joint | nao 0.8× · 0/3 · - | - | 0.12 | sim | simclr: nao 0.79 | 0.04 · 0.49 | 0.64 · -0.005 | 1.14× | sim | f5:0.819 > f15:0.820 > f45:0.820 |
| hybrid/c00025/f15 (tabela) | tiny · 0.21M | 0.820 | spifil | masked/joint | nao 0.8× · 0/3 · - | - | 0.12 | sim | simclr: nao 0.79 | 0.03 · 0.36 | 0.78 · -0.004 | 1.04× | sim | f5:0.819 > f15:0.820 |
| hybrid_fullepoch_1555/c00001/f5 | tiny · 0.02M | 0.733 | spifil | pixel/decoder_first | nao 1.0× · 0/2 · - | 1.22× | 0.02 | sim | lejepa: nao 0.50 | 0.06 · 0.00 | 0.71 · - | - | sim | f5:0.733 |
| lemonade/c00001/f15 (tabela) | tiny · 0.15M | 0.706 | spifil | pixel/decoder_first | nao 0.5× · 0/3 · - | 0.96× | 0.01 | sim | lejepa: nao 0.53 | 2.14 · 0.55 | 0.09 · -0.003 | 1.05× | sim | f5:0.702 > f15:0.706 |
| hybrid_C01_2049/c00001/f15 | tiny · 0.02M | 0.668 | spifil | pixel/decoder_first | nao 1.0× · 0/2 · - | 0.98× | 0.12 | sim | lejepa: nao 0.51 | 0.10 · 0.18 | 0.62 · -0.028 | 1.18× | sim | f5:0.668 > f15:0.668 |

### coconut_trees

| modelo (busca/cand/fid) | estrato · params | κ val GA | init | AE: alvo/sched | AE pico ini (pior bloco): x pico/x0 · blocos>2× · recupera em | AE junção (decoder→bloco) | AE fim/x0 | AE converge | SSL método: pico ini/x0 | SSL oscil. cv·rev | SSL fim/x0 · incl. final | salto na promoção (máx) | SSL converge | κ na cadeia |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lemonade/c00003/f200 (tabela) | medium · 1.34M | 0.795 | spifil | pixel/decoder_first | **GRANDE** 5.4× · 3/6 · 2 passos | 1.48× | 0.03 | sim | lejepa: nao 0.80 | 0.03 · 0.64 | 0.48 · -0.000 | 0.98× | sim | f5:0.788 > f15:0.779 > f45:0.782 > f100:0.783 > f200:0.795 |
| hybrid/c00007/f5 (tabela) | medium · 1.48M | 0.790 | spifil | pixel/decoder_first | **GRANDE** 27.2× · 2/4 · 12 passos | 1.22× | 0.17 | sim | lejepa: nao 0.57 | 0.07 · 0.00 | 0.84 · - | - | sim | f5:0.790 |
| hybrid/c00004/f100 (tabela) | huge · 2.30M | 0.790 | spifil | pixel/joint | **GRANDE** 1538.0× · 2/4 · 14 passos | - | 0.09 | sim | lejepa: moderado 1.23 | 0.59 · 0.42 | 0.10 · -0.002 | 1.06× | sim | f5:0.783 > f15:0.783 > f45:0.783 > f100:0.790 |
| lemonade/c00004/f100 (tabela) | huge · 2.22M | 0.787 | spifil | pixel/decoder_first | **GRANDE** 6.6× · 2/6 · 2 passos | 1.13× | 0.03 | sim | lejepa: nao 0.84 | 0.02 · 0.49 | 0.55 · -0.001 | 0.98× | sim | f5:0.781 > f15:0.782 > f45:0.774 > f100:0.787 |
| hybrid/c00014/f5 (tabela) | small · 0.34M | 0.778 | spifil | pixel/decoder_first | nao 0.9× · 0/3 · - | 1.00× | 0.02 | sim | lejepa: nao 0.99 | 0.00 · 0.00 | 0.99 · - | - | sim | f5:0.778 |
| hybrid/c00012/f5 | over · 3.54M | 0.768 | spifil | pixel/joint | **GRANDE** 344.1× · 1/4 · 2 passos | - | 0.02 | sim | lejepa: nao 0.69 | 0.67 · 0.00 | 0.37 · - | - | sim | f5:0.768 |
| lemonade/c00001/f45 (tabela) | tiny · 0.20M | 0.736 | spifil | pixel/decoder_first | nao 0.4× · 0/3 · - | 1.01× | 0.01 | sim | lejepa: nao 0.33 | 2.84 · 0.54 | 0.03 · -0.008 | 1.33× | sim | f5:0.725 > f15:0.724 > f45:0.736 |
| hybrid/c00009/f5 (tabela) | tiny · 0.08M | 0.714 | spifil | pixel/joint | **GRANDE** 19.3× · 1/2 · 2 passos | - | 0.04 | sim | lejepa: nao 0.53 | 0.52 · 0.00 | 0.37 · - | - | sim | f5:0.714 |
| lemonade/c00002/f15 (tabela) | small · 0.59M | 0.663 | spifil | pixel/joint | **GRANDE** 116.4× · 2/5 · 9 passos | - | 0.06 | sim | lejepa: nao 0.57 | 0.12 · 0.18 | 0.61 · -0.012 | 1.22× | sim | f5:0.651 > f15:0.663 |