# G3: pilotos da rodada L4/L5

## Geometria, sem saliência (`encdeep_geo_off`, athena, END 23:03 UTC rc=0; aid e remote_sensing, split 1, seed 42, 200 px, fps; κ de validação)

| base | braço | cat até L3 | cat até L4 | cat até L5 |
|---|---|---|---|---|
| aid | G0 z0 / hebb | 0,750 | 0,755 / 0,755 | 0,770 / 0,762 |
| aid | G1 z0 / hebb | 0,750 | 0,766 / 0,766 | 0,775 / **0,784** |
| aid | G2 z0 / hebb | 0,750 | 0,766 / 0,766 | 0,775 / 0,768 |
| remote_sensing | G0 z0 / hebb | 0,485 | 0,465 / 0,472 | 0,451 / 0,445 |
| remote_sensing | G1 z0 / hebb | 0,485 | **0,493** / 0,486 | 0,459 / 0,460 |
| remote_sensing | G2 z0 / hebb | 0,485 | 0,493 / 0,486 | 0,473 / 0,480 |

- Geometria: G1 (sem redução depois de L4) ≥ G0 nas duas bases (aid L5 +0,005 a +0,022; rs L4 +0,014 a +0,028). A dilatação (G2) não acrescenta nada em aid; em rs, só reduz a perda de L5.
- Mecanismo: o SoftHebb fica igual à atualização zero, dentro de ±0,01, exceto em aid G1 L5 (+0,009).
- Profundidade: aid ainda sobe até L5; em remote_sensing, L5 piora em todas as geometrias (o melhor é L3 ou L4). Isso é evidência para a regra de parada (`depth-stopping-rule`).
- Dimensão controlada (`cat_eq256`): as diferenças encolhem; parte do ganho de `cat` com mais camadas vem da dimensão.
- Decisão: G1 vira a geometria dos pilotos de mecanismo (já usada no `encdeep_mech`).

## Mecanismos, sem saliência (`encdeep_mech_off`, jaci, END 23:32 UTC rc=0; aid e remote_sensing, 200 px, G1, split 1, seed 42; Δ contra a atualização zero `z0`)
- **L5 `cat`** (z0: aid 0,769; rs 0,522):
  - hebb: +0,006 / −0,023
  - hebb~nw: +0,008 / −0,033
  - clapp: −0,001 / −0,014
  - clapp2: −0,005 / −0,014
  - sfa: −0,016 / −0,044
  - sfa0: −0,016 / −0,037
  - z0~nw: +0,010 / −0,020
  - aleatório: aid 0,69; rs 0,44 a 0,47
- **L5 isolada `L`** (z0: 0,629 / 0,367):
  - hebb: +0,035 / +0,062
  - clapp: +0,004 / +0,057
  - clapp2: +0,025 / +0,044
  - sfa: −0,032 / −0,021
  - **z0~nw (sem aprender nada, só sem whitening em L4/L5): +0,035 / +0,057**
- Leitura (n = 1):
  - Em `cat`, nenhum mecanismo aprendido supera a atualização zero; o SFA perde.
  - Na L5 isolada, o ganho do hebb e do clapp é igual ao de tirar o whitening sem aprender (`z0~nw`). O efeito é da normalização, não da plasticidade.
  - O SPiFiL em L4/L5 sem atualização fica cerca de +0,07 acima do aleatório puro na mesma geometria.
- Decisão provisória: até aqui, nenhum mecanismo (H, C, S) mostra ganho do encoder pelo critério da §12. Falta flowers e coconut (`encdeep_mech_on`) e o teste a 320 px (`encdeep_mech320_off`, rodando). Pergunta nova: o whitening entre camadas ajuda até L3 e atrapalha a leitura isolada de L4/L5?

## Geometria, com saliência (`encdeep_geo_on`, athena, END 23:37 UTC rc=0; flowers e coconut_trees, 200 px, receita, split 1, seed 42)
- coconut_trees:
  - Na leitura isolada (L), cada camada dá cerca de 0,76, de L3 a L5, em qualquer geometria.
  - Na concatenação (`cat`), o κ cai a cada camada: 0,741 (L2) → 0,720 → 0,66 → 0,60 a 0,65. Confirma que, nessa base, empilhar camadas no descritor atrapalha. A profundidade não traz informação nova, e a regra de parada deveria parar em L2 ou L3.
- flowers:
  - Na `cat`, o κ sobe com a profundidade: 0,576 → 0,585 → 0,61 a 0,64 → 0,62 a 0,64. Aqui a melhor geometria é G0 (L5 0,641 a 0,644); G1 e G2 ficam em 0,617 a 0,629.
- Mecanismo: o hebb fica igual à atualização zero (±0,02) nas duas bases.
- Conclusão parcial: a geometria ideal depende da base (G1 em aid e remote_sensing, G0 em flowers), e a profundidade útil também (coconut para cedo; aid e flowers continuam até L5). Isso reforça a necessidade de uma regra de parada e de geometria guiada pelos dados, em vez de uma escolha fixa.

## Confirmação ~nw, sem saliência (`encdeep_nw_off`, athena, END 01:27 UTC de 09/11; aid e rs, 3 splits × 2 sorteios, G1, 200 px, L5)
Diferença pareada contra o z0 (média ± desvio, n = 6):
- aid: `cat` z0 0,758 → z0~nw **+0,020 ± 0,006**; L +0,036 ± 0,020. hebb~nw igual ao z0~nw (+0,019 / +0,035).
- remote_sensing: `cat` +0,004 ± 0,037; L 0,000 ± 0,019. Sem efeito.
- O aleatório na mesma geometria: aid 0,679 (`cat`); rs 0,494.
Decisão: **em aid, tirar o whitening em L4/L5 é um ganho confirmado** (+0,02 em `cat`, positivo nos 6 pares). Em remote_sensing, é neutro. O hebb não acrescenta nada ao ~nw. Falta flowers e coconut (`encdeep_nw_on`, jaci).

## Mecanismos a 320 px, sem saliência (`encdeep_mech320_off`, jaci, END 01:31 UTC de 09/11; aid e rs, G1, split 1, seed 42)
Conferido: image_size 320 em todas as linhas; mapas 159-79-39-39-39 (a 200 px, 99-49-24-24-24); trunk_hash diferente do de 200 px. O z0 `cat` igual até a 3ª casa é coincidência (aid 0,769024 contra 0,769012).
- L5 `cat`, 320 contra 200 px: aid quase sem mudança (+0,000 a +0,009); remote_sensing hebb +0,029, clapp2 +0,035 (0,543, o melhor), sfa +0,052. O aleatório também sobe (+0,03 a +0,04).
- L5 isolada: z0 +0,027 (aid) e +0,076 (rs).
- Leitura: a 320 px os mecanismos aprendidos ganham do z0 em remote_sensing (clapp2 +0,021), mas boa parte do ganho vem da resolução, já que o aleatório sobe junto. Sinal exploratório (n = 1).
- Correção de uma afirmação anterior: na geometria G1 (3 pools de 2), o mapa de L4/L5 é 24×24 a 200 px e 39×39 a 320 px. O 11×11 e 19×19 citados antes valem para a receita de 4 pools (G0).
