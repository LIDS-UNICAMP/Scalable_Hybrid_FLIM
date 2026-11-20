# Mecanismos globais sem treino para o SPiFiL: o que diz a literatura (11/11/2026)

Duas buscas, ambas com citações; os itens marcados como "incerto" não foram conferidos.

## Por que a nossa atenção falhou (hipótese da literatura, ainda a medir)
1. **Virou identidade.** Depois do whitening, ‖x‖² ≈ 256. A nota do token consigo mesmo (√d = 16) domina o softmax, então A ≈ I. Teste: média de A_ii. É o efeito do β na rede de Hopfield moderna (Ramsauer et al., 2021).
2. **O GeM desfaz a mistura.** Misturar tokens com pesos que somam 1 e depois tirar a média global quase não muda o resultado. Para ligar pontos distantes é preciso guardar **co-ocorrência**, não só misturar.
3. **Os features são pós-ReLU.** Todo produto escalar é positivo, então a atenção fica difusa ou saturada.

## Ranking para o SPiFiL
| # | mecanismo | fonte | sem treino? | por que ajuda |
|---|---|---|---|---|
| 1 | Covariância/bilinear com **raiz de matriz** (Newton-Schulz), após PCA 256→60 (1830-d) | Li et al., CVPR 2018 (iSQRT-COV); Lin & Maji, BMVC 2017 | sim | mede quais filtros disparam juntos na imagem inteira (estatística de cena e textura que o GeM descarta) |
| 2 | MSCP: covariância empilhando L3–L6 | He et al., IEEE TGRS 2018 (cenas aéreas, CNN congelada + SVM) | sim | é o caso mais parecido com o nosso |
| 3 | Self-self attention (cosseno, temperatura, diagonal mascarada, 2–3 iterações), na construção | SCLIP (ECCV 2024), ClearCLIP (ECCV 2024), GEM (CVPR 2024) | sim | corrige a falha 1 |
| 4 | Gram deslocado (F × F deslocado por δ = 8, 16, 32) | Berger & Memisevic, ICLR 2017 (síntese; ganho em classificação é incerto) | sim | é o único que codifica "o que está a distância d de quê" |
| 5 | Fisher vector sobre as convs (GMM sem rótulo) | Cimpoi et al., CVPR 2015 (MIT Indoor 81%) | sim | é clássico em cenas, mas é da família do VLAD, que já perdeu |
| — | NMF de baixo posto (Hamburger) | Geng et al., ICLR 2021 | o NMF sim | é um contexto global que combina com features ≥ 0 |

## Descartados
- **MOP-CNN:** é um VLAD multiescala, e o VLAD já perdeu.
- **R-MAC e CroW:** foram feitos para busca de imagens, não para classificação.
- **Scattering:** é parecido com os filtros aleatórios.
- **PoolFormer:** na prática é só um blur.
- **GFNet:** os filtros são aprendidos.
- **PAMR:** o alcance cresce devagar.

Nenhum artigo testa diretamente filtros fixos ou aleatórios mais um misturador global em cenas ou flowers. Não dá para afirmar que isso fecha a lacuna com o DINOv2.
Os mecanismos 1 a 4 foram passados ao estudo `relacoes_globais.md`, nessa ordem de prioridade.
