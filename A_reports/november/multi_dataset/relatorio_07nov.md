# SPiFiL sem backprop: o que sabemos (07/11, 22:10 UTC)

Fontes: Athena (`/exps/moliveira/results`, 5.960 leituras em L3 `cat`) e Jaci (resumos da sessão `spifil_explore` às 21:55 UTC, parciais). Só validação. Δ = κ do método menos o melhor aleatório da mesma base, split e largura. Objetivo do usuário: o SPiFiL detecta o objeto e classifica, dentro do modelo, sem máscara; o aleatório é só a régua.

## Resposta curta

O melhor resultado até agora é o whitening entre camadas (Jaci, parcial): com ele, `fps@combo` passa do melhor aleatório em todas as 6 bases medidas, inclusive `aid` e `flowers`, que antes perdiam. Sem ele, o SPiFiL ganha nos parasitos e empata ou perde em `flowers` e `aid`. Nenhum método passou ainda no critério formal (3 splits completos).

## Funciona

| ideia | evidência | ressalva |
|---|---|---|
| Whitening entre camadas (`spfcombo_w`, Jaci) | `fps@combo` com 256 filtros, L3: Eggs_subgroup 0,821 → 0,905; Larvae_subgroup 0,613 → 0,828; Protozoan_subgroup 0,813 → 0,862; aid 0,714 → 0,748; flowers 0,588 → 0,637. Em L6: 0,80 a 0,95 nos parasitos, aid 0,701 (64), flowers 0,659 (256, leitura `cat_b`) | 1 ou 2 splits; o aleatório piora com esse whitening, então comparar com o melhor aleatório de qualquer pipeline |
| Filtros diferentes entre si (`fps`) | Athena: +0,071 (64) e +0,072 (256) em 7 bases, pior −0,017; passou para `r3` | Não medido em Eggs e Protozoan |
| Saliência como peso (`fpsw@mapa_peso`) | +0,162 (64) e +0,133 (256) em 5 bases; contra `fps`: +0,043 [+0,019; +0,067] com 64 | Larvae puxa; sem aid e helminth; com 256 o ganho sobre `fps` não é significativo |
| Concatenar camadas e ir fundo | `spfdeep_esp` L8 `cat`: `fps` +0,235 (Larvae_subgroup 64), +0,098 (Protozoan_subgroup 64), `fps@sift` +0,097 (Eggs_subgroup 64) | Lendo só a última camada, cai |
| Pontos SIFT | `crus@sift` +0,087 (64, 7 bases); `fps@sift` entre os 5 do `r3`; detecção 36% contra 25% nas larvas | Pequena melhora de detecção |

## Não funciona

| ideia | evidência |
|---|---|
| ZCA ou PCA nos filtros | `zca2` −0,109 (64) e −0,126 (256), 0 a 1 de 7 bases |
| Agrupamento por K-means | empata com patch sorteado e perde de `fps` (−0,028 com 64; −0,044 com 256); cluster puxa patch liso (53% contra 2%) |
| k automático (`km_sil`, `hdbscan`) | k fica em 8; `hdbscan` −0,25 |
| Só fundo | `crus@fundo` −0,074 |
| DoG como filtro, Gauss antes do recorte, várias escalas | S4 −0,024; Gauss e várias escalas empatam com o patch cru |
| `spifil` original | +0,037 (64), pior −0,072; escolhe por classe e usa poucas imagens |
| Combo de tudo sem whitening entre camadas | +0,017, não ganha do ingrediente sozinho |
| Soma de camadas | Δ maior só porque o aleatório piora |

## Detecção (máscara só para medir)

- Padrão do SPiFiL: 25% a 49% dos pontos dentro do objeto, 2,4 a 4 vezes o acaso.
- Saliência com corte: 70% a 98% dentro, mas com 9 a 23 pontos por imagem; o foco dos superpixels (400 SLIC) leva 75% a 86% em Larvae_subgroup.
- Mapas: `gmr` IoU 0,781 e fusão B 0,795 na amostra; fora da amostra, em Larvae_subgroup, B 0,657 contra `gmr` 0,643. Nas 3 bases recortadas (objeto enche a imagem), `gmr` ganha.
- Quanto falta para detectar (`spf_hz` no Jaci contra `spf_h` no Athena, 6 bases, 1 a 2 splits): no fundo real, `spifil` e `crus` ganham +0,071 do aleatório; se soubessem onde está o objeto (fundo zerado), ganhariam +0,21. O SPiFiL percorre cerca de 1/3 do caminho, e saber a posição daria +0,14 de κ. Larvae chega a 0,6 a 0,8 do caminho; Eggs e Eggs_subgroup ficam perto de 0 ou abaixo. Conta: (κ real − aleatório real) / (κ zerado − aleatório real). Os números de Larvae com 256 filtros variam muito entre splits.

## Por base

Parasitos: o SPiFiL ganha, mais em larvas e protozoários. `flowers` e `aid`: perdem ou empatam, exceto com whitening entre camadas. `helminth_larvae`, `protozoan_cysts` e `remote_sensing` só agora entram (`r5`, multiescala `a2`/`b2`).

## Em andamento

Athena: `spf_h_raw`, `ms_ctl2`, `spfdeep_cedo`, `spfcombo_a`, agrupamento `cl_ctl` → `cl_a`, teste do SIFT como mapa de objeto.
Jaci: `spfdeep_esp`, `spfcombo_w`, `spf_hz`, multiescala `a2` → `b2`, saliência como peso `salw_a`, mapas da fusão Top-K; `r3` → `r4` → `r5` depois do rsync dos CSVs de `r1` e `r2`.

## Próximo passo sugerido

Juntar o que funciona numa receita só e medir com 3 splits em todas as bases: `fps` (ou `fps@sift`), peso da saliência, whitening entre camadas, camadas concatenadas, 256 filtros.
