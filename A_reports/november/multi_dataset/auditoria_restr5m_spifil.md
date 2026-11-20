# Auditoria: os modelos ≤ 5 M ("leis da literatura") são SPiFiL puro?

**Resposta curta: sim.** Os 11 YAMLs `experiments/autoencoder/disc_md_restr5m_*_jaci.yaml` montam todos a mesma
rede: só Conv2d + ReLU + MaxPool, com cada filtro recortado de uma janela da imagem. Os nomes EfficientNet,
ConvNeXt, RegNet e dWr servem só para escolher a forma da rede (largura, profundidade, kernel, pool e tamanho
da imagem). Nenhum peso, bloco ou código dessas redes entra.

Auditoria feita em 11/11/2026, só leitura: nada foi lançado nem editado. Código no athena e artefatos no jaci.

## Veredito por afirmação

| # | Afirmação | Veredito | Evidência |
|---|---|---|---|
| 1 | Os filtros saem de janelas da imagem (fps / fpsw@mapa), sem backprop e sem rótulo; whitening entre camadas, saliência quando configurada, GeM + PCA-whitening + SVM linear com TTA | **Confirmada** | ver §1 |
| 2 | Só a forma muda: profundidade, larguras (`lei: W`), kernel por camada (`lei.k`), pools e resolução | **Confirmada** | ver §2 |
| 3 | EfficientNet, ConvNeXt, RegNet e dWr são só a regra de escala | **Confirmada** | ver §2 e §3 |
| 4 | Nenhum peso, camada ou bloco dessas redes; `random_z` = aleatório da mesma forma | **Confirmada** (com uma nuance no controle, ver §4) | ver §3 e §4 |

## 1. De onde vem cada filtro

- **Candidatos.** `patch_pool` recorta uma janela k×k em volta de cada semente (o medoide de cada superpixel DISF),
  no mapa da camada anterior (`eval/l1_sem_rotulo.py:113-120`; `eval/spf_grade.py:1524-1541`). As sementes vêm do
  cache `fontes`, um por sorteio (`eval/spf_grade.py:629-643`, montado em `sources`, `:516-553`).
- **Escolha.** `fps` é o FPS do repositório (`eval/l1_sem_rotulo.py:184-189`, ligado em `eval/spf_grade.py:1005`).
  `fpsw` é o mesmo FPS com a distância multiplicada pela saliência (`eval/spf_grade.py:825-833`).
  O mapa de saliência é o `gmr`, um método clássico de ranqueamento em grafo, sem rede neural
  (`saliency/batch.py:77`), aplicado como `soft(s, 0.5, 1)` (`eval/spf_grade.py:381`).
- **Do patch para o peso.** `weight = u / stdev`, `bias = -(weight·mean)` (`eval/spf_grade.py:1557-1567`).
  Da camada 2 em diante, o whitening do mapa é embutido nesse mesmo peso (`:1525`, `:1569-1570`, `folded` em `:1391-1394`).
  Na camada 1 o mapa é só z-scorado (`between and n` só vale para n ≥ 1).
- **Sem rótulo.** As 500 imagens são sorteadas do treino sem olhar a classe (`bank_pool`, `:1743-1751`,
  sem `por_classe`). As máscaras de `MASK_DIR` só preenchem os campos de oráculo e de diagnóstico (`obj`, `m_obj`);
  `fps` e `fpsw@mapa_*` nunca as leem.
- **Sem backprop.** O único `torch` usado é forward e álgebra. `spifil_rows`/`grow_layer` (`:1458-1476`) só rodam
  para braços `spifil*`, que estes YAMLs não usam.
- **Leitura.** As leituras são `gem_pw` e `gem_tta_pw`, o GeM médio nas 8 vistas rot90×flip (`:1598-1600`;
  `eval/svm.py:331-332`). Depois vem `pw_whiten`, PCA-whitening com shrinkage + L2 (`eval/svm.py:369-377`),
  e o SVM com `kernel="linear"` (`core/svm.py:183`), em `score_full` (`eval/spf_grade.py:1047-1052`).

**Prova no jaci (CPU, efêmero).** Reconstruí a camada 1 de `effnet_s104` para Eggs split1 a partir do cache
`fontes_filtros/restr5m_effnet_s104/Eggs_split1_d42.pt`: 500 imagens LAB de 3×104×104.
- Os 160 filtros têm cosseno de 1,0000 com a janela z-scorada da semente de origem. As janelas vêm de
  123 imagens (fps) e de 121 (fpsw).
- As estatísticas dos kernels batem com a linha `layer=1` do CSV do run:
  `fps`: cos_mean 0,2125 / erank 46,84; `fpsw`: 0,2389 / 44,60, iguais aos CSVs `_sh1` e `_sh0`.
  O `random_z` do mesmo CSV dá 0,092 / 70,0.
- O encoder montado pelo caminho oficial (`grown_arch` → `write_weights` → `build`) é
  `Conv2d(3,160,k5,pad2) → ReLU → MaxPool2d(3, stride 2)`, com 12.160 parâmetros.
  Os tipos de módulo são só `Conv2d`, `ReLU`, `MaxPool2d` e os contêineres.

## 2. O que muda entre as arquiteturas

- **Larguras.** `lei: W` lê as larguras literais "a-b-c" (`eval/spf_grade.py:1417-1418`). Cada lei vira uma célula,
  com o pool cortado na profundidade (`:1765-1769`) e o plano em `:1736`.
- **Kernel por camada.** `lei.k` vem de `lei_kernels` (`:1431-1434`) e chega em `LayerSpec(kernel_size=ks[n])` (`:1522`).
- **Bloco.** O bloco é sempre o de `build_encoder_from_arch`: Conv2d, depois ReLU, depois MaxPool 3 se stride > 1
  (`flim/arch.py:104-112`, `flim/spifil.py:701-728`).
  O `arch` de partida do FLIM tem as camadas apagadas (`eval/spf_grade.py:1497`).
  `write_weights` só copia `conv1..conv(n-1)` da pasta temporária desta própria cadeia (`flim/spifil.py:688-691`),
  então nenhum peso FLIM antigo entra.
- **Ativação e normalização.** Não há `norm`, `ativacao`, `atencao` nem `encdeep` nos YAMLs. Os sufixos ficam vazios
  (`tails`, `:1699-1708`), e a troca de ReLU por GELU/LeakyReLU (`:1584-1586`) nunca dispara.
- **Resolução.** `l1_sizes` vira `image_size`: a imagem inteira passa por letterbox, sem corte
  (`core/data/transforms.py:33-40`, `eval/l1_fast.py:73-76`). As fontes de patch também saem nesse tamanho:
  o cache do s104 tem 104×104. O SVM e os filtros veem a mesma resolução.
- **ConvNeXt.** O "1:1:3:1" virou apenas pools `2-2-1-1-2-1` e larguras C-2C-4C-4C-4C-8C, com k3.
  Não há depthwise, gargalo invertido, LayerNorm nem GELU.
- **EfficientNet e dWr.** Só largura × profundidade × resolução, com k5. Não há MBConv, SE nem swish.

## 3. Redes pré-treinadas: o que existe no repo e o que é alcançado

O grep por `torchvision.models|timm|from_pretrained|hub.load|load_state_dict|efficientnet|convnext|regnet` acha
`benchmark/*` (convnextv2, dinov2, aimv2, lista), `core/blocks/timm_encoder.py`, `methods/*`, `src/*`,
`eval/svm_variants/*`, `eval/unified_eval.py` e `scripts/spifil_resnet_graft.py`.
**Nenhum desses arquivos está no caminho destes runs.** O que aparece nos módulos que `spf_grade` importa:

- `flim/encoder.py:34` faz `import timm` no topo, mas `Encoder.from_timm` (`:126`) só roda se for chamado.
  Estes runs usam só `from_flim` (`flim/build.py:56`).
- `flim/spifil.py:1152` carrega `resnet18(IMAGENET1K_V1)` dentro de `graft()` (`:1133`). `spf_grade` não chama `graft`.
- `eval/spf_grade.py:595` importa `benchmark.lista` dentro de `add_ijepa`. Isso só acontece com a chave `amostra.ijepa`
  (`:550`), ausente aqui. `add_dino` também está fechado, pela chave `dino`, ausente.
- O `torch.load` de `eval/spf_grade.py:636` lê só o cache `fontes`: imagens LAB + sementes.
  `eval/svm.py:633` (`_ckpt_epoch`) e `growth.py:175` (`_assert_frozen`) não são chamados.

## 4. O controle `random_z`

`random_z = init.normal_` (`eval/spf_grade.py:1027-1028`) sorteia uma direção por filtro, com semente fixa (`control`, `:1031-1041`).
Essa direção passa pela mesma normalização L2, pela mesma dobra z-score e pelo mesmo whitening entre camadas
dos braços SPiFiL (`:1557-1570`). As formas são idênticas: os CSVs dão o mesmo `plan`, `widths` e `params` nos dois braços.

**Nuance:** o `random_z` é "direção aleatória no espaço normalizado do patch", não um `nn.init` cru sobre o Conv2d.
É a régua mais justa, mas o texto deve dizer isso, e não "pesos aleatórios" sem qualificação.

## 5. Desvios e ressalvas encontrados

Nenhum desvio de construção: nada pré-treinado, nada depthwise, ReLU em toda camada e resolução aplicada igual
para filtros e SVM. Ressalvas de registro:

1. **`kernel` no CSV.** Nas leis com `lei.k = 3-3-3-5-5-5` a coluna `kernel` diz 3, porque é o kernel da célula.
   O kernel real por camada só aparece em `lei_args`/`rf`: rf 205 contra 125.
2. **`params` no CSV não conta bias.** `conv_cost` (`:1437-1450`) conta só pesos. O bias soma menos de 0,1 %
   (tabela abaixo).
3. **Braço por família.** `*_obj` roda só `fpsw@mapa_w0.5_1`; `*_cena` roda só `fps`, com o bloco de saliência
   presente no YAML mas sem uso pelo braço; `effnet_*` e `dwr_*` rodam os dois.
4. **Whitening.** Começa na camada 2; a camada 1 é só z-score. Cada leitura também sai na variante `_b`,
   só com L2 e sem PCA-whitening (`:1626-1629`).
5. **Runs que faltam.** `effnet_s128`, `effnet_s160` e `dwr_s184` ainda não têm CSV nem `fontes` no jaci;
   para eles a auditoria vale só pelo código.
6. **Erros.** 7 células morreram por CUDA OOM (k3 com 408×4, `216×6 k3-3-3-5-5-5`, `144×6 k...` e
   k5 88×6 fpsw em Eggs s1). Não afetam a natureza do modelo.

## 6. Forma × parâmetros

A entrada é LAB, com 3 canais. Conv "same", MaxPool 3 com stride p e sem padding.
A coluna "params (CSV)" é a mesma de `conv_cost` e bate com o CSV do jaci nos casos conferidos: 982.008,
1.970.856, 4.492.080, 1.000.512 etc.

| Arquitetura | Prof. | Larguras | Kernel | Pools | px | Params (CSV, sem bias) | Params + bias |
|---|---|---|---|---|---|---|---|
| k3 uniforme (VGG) | 6 | 152×6 | 3 | 2-2-2-2-1-1 | 200 | 1.043.784 | 1.044.696 |
| k3 WRN | 4 | 192×4 | 3 | 2-2-2-2 | 200 | 1.000.512 | 1.001.280 |
| k3 RegNet | 6 | 24-48-104-208-208-208 | 3 | 2-2-2-2-1-1 | 200 | 1.029.384 | 1.030.184 |
| k3 ResNet | 6 | 40-80-152-152-152-312 | 3 | 2-2-2-2-1-1 | 200 | 982.008 | 982.896 |
| k3 RF | 6 | 104×6 | 3-3-3-5-5-5 | 2-2-2-2-1-1 | 200 | 1.008.696 | 1.009.320 |
| k3 uniforme | 6 | 208×6 | 3 | 2-2-2-2-1-1 | 200 | 1.952.496 | 1.953.744 |
| k3 WRN | 4 | 272×4 | 3 | 2-2-2-2 | 200 | 2.004.912 | 2.006.000 |
| k3 RegNet | 6 | 40-72-144-288-288-288 | 3 | 2-2-2-2-1-1 | 200 | 1.986.552 | 1.987.672 |
| k3 ResNet | 6 | 56-112-216-216-216-440 | 3 | 2-2-2-2-1-1 | 200 | 1.970.856 | 1.972.112 |
| k3 RF | 6 | 144×6 | 3-3-3-5-5-5 | 2-2-2-2-1-1 | 200 | 1.932.336 | 1.933.200 |
| k3 uniforme | 6 | 312×6 | 3 | 2-2-2-2-1-1 | 200 | 4.388.904 | 4.390.776 |
| k3 WRN | 4 | 408×4 | 3 | 2-2-2-2 | 200 | 4.505.544 | 4.507.176 |
| k3 RegNet | 6 | 56-112-216-432-432-432 | 3 | 2-2-2-2-1-1 | 200 | 4.474.728 | 4.476.408 |
| k3 ResNet | 6 | 80-168-328-328-328-656 | 3 | 2-2-2-2-1-1 | 200 | 4.492.080 | 4.493.968 |
| k3 RF | 6 | 216×6 | 3-3-3-5-5-5 | 2-2-2-2-1-1 | 200 | 4.344.840 | 4.346.136 |
| ConvNeXt 1:1:3:1 | 6 | 40-80-152-152-152-312 | 3 | 2-2-1-1-2-1 | 200 | 982.008 | 982.896 |
| ConvNeXt 1:1:3:1 | 6 | 56-112-216-216-216-440 | 3 | 2-2-1-1-2-1 | 200 | 1.970.856 | 1.972.112 |
| ConvNeXt 1:1:3:1 | 6 | 80-168-328-328-328-656 | 3 | 2-2-1-1-2-1 | 200 | 4.492.080 | 4.493.968 |
| k5 (MobileNet α) | 6 | 88×6 | 5 | 2-2-2-2-1-1 | 200 | 974.600 | 975.128 |
| k5 (MobileNet α) | 6 | 128×6 | 5 | 2-2-2-2-1-1 | 200 | 2.057.600 | 2.058.368 |
| k5 (MobileNet α) | 6 | 192×6 | 5 | 2-2-2-2-1-1 | 200 | 4.622.400 | 4.623.552 |
| EfficientNet s104 | 3 | 160×3 | 5 | 2-2-2 | 104 | 1.292.000 | 1.292.480 |
| EfficientNet s128 | 3 | 192×3 | 5 | 2-2-2 | 128 | 1.857.600 | 1.858.176 |
| EfficientNet s160 | 4 | 224×4 | 5 | 2-2-2-2 | 160 | 3.780.000 | 3.780.896 |
| dWr s176 | 5 | 96×5 | 5 | 2-2-2-2-1 | 176 | 928.800 | 929.280 |
| dWr s184 | 5 | 144×5 | 5 | 2-2-2-2-1 | 184 | 2.084.400 | 2.085.120 |

Os orçamentos de ~1 M, ~2 M e ~4,5 M se confirmam. As exceções: EfficientNet s104 tem 1,29 M e s160 tem 3,78 M,
e k5 192×6 passa um pouco, com 4,62 M. Todos ficam ≤ 5 M.
