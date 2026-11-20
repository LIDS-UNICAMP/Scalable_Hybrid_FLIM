# Decisões sobre o SPiFiL sem backprop (07/11, 22:50 UTC)

Base: `relatorio_07nov.md`, `sol_auditoria.md`, `saliencia.md` e resumos do Jaci (parciais). Só validação. Régua: melhor aleatório de qualquer pipeline, na mesma base, split e largura. Objetivo: detectar e classificar dentro do modelo, sem máscara.

## Receita atual (o que usamos)

| peça | escolha | por quê |
|---|---|---|
| superpixels | 50, DISF | fixado pelo usuário; custa 0,029 na média, até 0,075 em Protozoan e 0,069 em Eggs (1 camada) |
| pontos | medoide; SIFT como alternativa (`fps@sift`) | SIFT acha o objeto um pouco melhor (36% contra 25% nas larvas) |
| seleção | `fps` (filtros diferentes entre si) | +0,071 em 7 bases, pior −0,017; mais estável que os outros |
| saliência | `gmr` como peso, não máscara | `fpsw@mapa_peso` +0,043 sobre `fps` com 64 filtros; máscara rígida deixa poucos pontos |
| camadas | 3 a 6, concatenadas (`cat`) | ler só a última camada cai com a profundidade |
| leitura | GeM + PCA-whitening + SVM linear | protocolo único; sem TTA na grade |

## Whitening entre camadas: confirmado em 6 bases (07/11, 23:35 UTC)

`spfcombo_w` terminou no Jaci com 3 splits. Mesmo braço (`fps@combo`, leitura `cat`, 256 filtros, L3), sem o whitening entre camadas (`spfcombo_a`, Athena) contra com ele: Eggs_subgroup 0,821 → 0,873; Larvae_subgroup 0,613 → 0,769; Protozoan_subgroup 0,813 → 0,859; aid 0,714 → 0,732; flowers 0,588 → 0,643; helminth_eggs 0,931 → 0,944. Com 64 filtros, aid empata e flowers cai 0,011.
Contra o melhor aleatório de qualquer configuração, em L6 com 256 filtros: Eggs_subgroup +0,095, Larvae_subgroup +0,22, Protozoan_subgroup +0,10, aid +0,041, flowers +0,006 (+0,05 sem whitening na leitura), helminth_eggs +0,029. Positivo nas 5 famílias medidas; em flowers a folga é pequena. A configuração vem de umas 8 opções (camada, largura, leitura), então há algum otimismo.
Decisão: entra na receita. Falta medir nas outras 6 bases (`disc_md_spfcombo_w2_jaci`).
Correção: `cat_b` é a leitura sem PCA-whitening, com o whitening entre camadas ligado; não mede o efeito do whitening entre camadas.

## Descartado

ZCA ou PCA nos filtros; K-means, HDBSCAN e k automático; banco só de fundo; DoG, Gauss antes do recorte e várias escalas; combo de tudo sem whitening; soma de camadas; densidade SIFT como mapa de objeto (perde do `gmr` nas 9 bases com máscara); fusão Top-K (ganho pequeno, parada pelo usuário); `spifil` original como método sem rótulo (escolhe por classe: fica no regime B, separado).

## Em aberto, rodando

`r3` → `r5`; whitening entre camadas com 3 splits; saliência do peso suave à máscara rígida; agrupamento por grafo (comparar com o melhor aleatório antigo, porque o Z-score por patch derruba o aleatório em 0,197); multiescala nas 12 bases; Rodada 1 do prompt novo (normalizações, parte negativa das ativações, diversidade pelas ativações, pooling, fundo trocado, Scattering, SoftHebb, PCANet).

## Regras de leitura

- 5 famílias independentes: ovos, larvas, protozoários, aid + remote_sensing (455 imagens em comum), flowers. Ganho em pelo menos 3 de 5, com magnitude e intervalo.
- Subgroup e base completa são a mesma imagem: não contam como provas separadas.
- Detecção medida só nos braços SPiFiL; hoje o SPiFiL faz cerca de 1/3 do caminho até o teto de fundo zerado.
- Aleatório congelado antes dos métodos; teste nunca entra em escolha.

## Distância do objetivo

O melhor SPiFiL sem backprop fica de 0,11 a 0,35 de κ abaixo do DINOv2. A próxima etapa, depois da receita fechada, é o backprop.

## Atualização (08/11, 01:00 UTC)

- Receita candidata: `fps` (ou `fps@sift`) + saliência `gmr` (peso suave ou foco dos superpixels) + whitening entre camadas + camadas concatenadas até L6 + 256 filtros. A união está em teste (`sol_wsal`, Jaci).
- Saliência como peso ou foco (`salw_a`, parcial): +0,10 a +0,14 sobre `fps` nas larvas, +0,07 a +0,11 em ovos e protozoários, +0,05 em flowers; nada em aid.
- Detecção: com 3 splits, o SPiFiL faz cerca de 38% do caminho até o teto de fundo zerado; ovos perto de 0. A detecção vem do mapa de saliência, não dos filtros sozinhos.
- Mais filtros (512, 1024) não resolvem flowers.
- Encoders alternativos (12 bases, 3 splits): SPiFiL sem whitening entre camadas ganha de Scattering e SoftHebb em 6 de 9 bases; perde em Eggs (0,661 contra 0,742) e helminth_eggs (0,926 contra 0,957). Scattering é forte nas bases de fundo preto (0,93 a 0,98). Em remote_sensing ninguém passa do aleatório.
- Multiescala nas bases completas (parcial): igual ao geodésico; S4 sempre o pior.
- Ponto fraco: ovos (Eggs). Próximo a testar ali: SoftHebb partindo do SPiFiL e o `spfcombo_w2`.

## Whitening entre camadas nas 6 bases que faltavam (08/11, 02:45 UTC)

`spfcombo_w2` (Jaci, 3 splits, completo). Melhor de `fps@combo`/`cos@combo` em L6, 256 filtros, leitura `cat`, contra o melhor aleatório de qualquer experimento na mesma base:
- Eggs 0,845 (antes, melhor SPiFiL 0,661; aleatório 0,681; SoftHebb 0,742; Scattering 0,739).
- Larvae 0,785 (0,814 sem whitening na leitura); aleatório 0,533.
- Protozoan 0,849; aleatório 0,651; antes 0,761.
- protozoan_cysts 0,920; aleatório 0,800; Scattering 0,926.
- helminth_larvae 0,966; aleatório 0,961; Scattering 0,980.
- remote_sensing 0,506; aleatório 0,504 (empate).
Com o `spfcombo_w`, o whitening entre camadas passa do aleatório em 11 das 12 bases; remote_sensing empata. Resolve o ponto fraco dos ovos (Eggs +0,16). A configuração é a melhor de 4 (L3/L6 × cat/cat_b), então há algum otimismo.

## Agrupamento C3 a C9 (`cl_a`, Athena, 3 splits; 08/11, 04:40 UTC)

Com o Z-score do próprio patch, que vale para o job inteiro, o aleatório do próprio experimento cai muito (Eggs_subgroup 0,324 contra 0,656 no protocolo comum). Contra esse aleatório, os braços de agrupamento ganham; contra o aleatório comum (`ms_ctl`) e o geodésico comum, todos perdem: de −0,09 a −0,12 na média, L3 `cat`, 64 filtros.

| base | melhor agrupamento | braço | aleatório comum | geodésico comum |
|---|---|---|---|---|
| Eggs_subgroup | 0,482 | gr_m@mapa_otsu | 0,656 | 0,682 |
| Larvae_subgroup | 0,438 | km_w@sift | 0,319 | 0,585 |
| Protozoan_subgroup | 0,522 | bovw_w@sift | 0,652 | 0,738 |
| aid | 0,506 | km_w_m@mapa_otsu | 0,627 | 0,578 |
| flowers | 0,479 | hib_m@sift | 0,536 | 0,537 |

Decisão: descartados nesta forma o agrupamento por grafo (Louvain), o híbrido SIFT + patch, o K-means com 400 superpixels SLIC e o Z-score do próprio patch. Os dois efeitos (Z-score por patch e agrupamento) não foram separados: um teste de agrupamento no Z-score comum responderia isso, se valer o custo.

## Diversidade pelas respostas dos filtros (`sol_p3`, Jaci, 3 splits; 08/11, 05:16 UTC)

6 bases, L3 `cat`, sem whitening entre camadas. O melhor geral é `fpsw@mapa_peso` (topo ou empate em 7 de 12 células, custo igual ao `fps`). F3 (baixa correlação), F5 e F2 ganham do `fps` em aid (+0,02 a +0,03) e Eggs_subgroup; F6 é instável (+0,145 em Protozoan_subgroup 64, cai em Larvae_subgroup e aid); K-medoids não ganha do `fps` e custa 3,5 vezes mais; em flowers nenhum passa do aleatório com clareza.
Decisão: diversidade funcional não substitui o `fps`; F3 fica como alternativa barata para aid. CSVs no jaci ficam em `/dados/home/moliveira/exps/results_sol/filter_diversity/`, não em `results/`.

## Leituras (`sol_p5`, Athena, 3 splits; 08/11, 06:45 UTC)

`fpsw@mapa_w0_1`, média das 6 bases, κ absoluto em L3. Concatenado: `cat` 0,770 (256) e 0,710 (64); `cat_ms` (três escalas da imagem) 0,783 e 0,739; GeM com p fixo 2 ou 4 empata com o GeM atual; GAP e GMP pioram (0,65 a 0,72); SPP piora e passa de 1300 dimensões. Só a última camada: covariância (`L_cov`, 1176 dimensões) sobe de 0,706 para 0,755 (256) e de 0,655 para 0,762 (64).
O Δ contra o aleatório no GMP (+0,24 a +0,32) é grande só porque o aleatório desaba com máximo; não é ganho do método.
Decisão: a leitura multiescala (`cat_ms`) é o único ganho barato (+0,013 a +0,029, 219 dimensões) e entra como candidata; a covariância ajuda quando só a última camada é lida, mas custa 5 vezes a dimensão.

## N6: Z-score por patch + whitening entre camadas (`sol_p1_n6`, Jaci, 3 splits; 08/11, 07:28 UTC)

O Δ contra o aleatório cresce (+0,08 a +0,38) só porque o aleatório desaba com essa normalização. O κ absoluto dos mesmos braços cai em todas as células contra o protocolo comum (`sol_p3`, mesma leitura L3 `cat`): Eggs_subgroup 64 `fps` 0,694 → 0,473 e `fpsw` 0,755 → 0,484; Protozoan_subgroup 256 `fps` 0,819 → 0,644; aid 256 `fps` 0,705 → 0,609; helminth_eggs 256 `fps` 0,929 → 0,874. Nenhum λ (0,001; 0,01; 0,1) domina.
Decisão: o Z-score por patch sai de vez, também combinado com whitening entre camadas. O whitening entre camadas segue com a normalização comum (`spfcombo_w`, `spfcombo_w2`).

## Saliência do peso suave à máscara rígida (`salw_a`, Jaci, 3 splits; 08/11, 08:02 UTC)

6 bases, 26 braços, L3 `cat`, sem whitening entre camadas. Todos os 26 ganham do `fps` puro em média (+0,015 a +0,034). Maior ganho: nas larvas (Larvae_subgroup 64: `fpsw@mapa_w0.5_1` 0,726 contra `fps` 0,582); quase nulo em aid 256 e flowers 256.
- Peso na seleção (`fpsw@mapa_w*`): maiores ganhos médios, menos estável (vence o `fps` em 8 a 9 de 12 células).
- Foco dos superpixels (`fps@foco_w*`): ganho um pouco menor, vence em 11 de 12 células; mantém cerca de 50 pontos por imagem.
- Corte t entre 0,25 e 0,5 é o melhor; a força g quase não importa; máscara rígida (`otsu`) é das piores.
Decisão para a receita única: `fps@foco_w0.25_4` (o mais estável) ou `fpsw@mapa_w0.5_1` (o maior ganho médio); a escolha final sai da tabela de arrependimento com o `sol_wsal`.

## Whitening entre camadas + saliência (`sol_wsal`, Jaci, 12 bases, 3 splits; 08/11, 09:43 UTC)

72 jobs ok. `fps@combo` em L6 bate com o `spfcombo_w` até a terceira casa nas 12 células em comum. Com whitening entre camadas, todas as 24 células (12 bases × 64/256) ficam acima do melhor aleatório em L3 e L6; remote_sensing empata. A saliência soma ao whitening: +0,02 a +0,025 sobre o `fps` em média (mapa_w0_1 vence o `fps` em 18 de 24 células em L6; foco_w0_1 em 20). Grande nas larvas (Larvae_subgroup 64: 0,648 → 0,824; Larvae 256: 0,744 → 0,855), quase nulo em aid e Protozoan_subgroup 256. O dicionário SIFT (`combo`) quase não soma com whitening (+0,003). Melhores cortes aqui: t = 0 ou 0,5, g = 1.
Melhor por base em L6: Eggs 0,866; Eggs_subgroup 0,893; helminth_eggs 0,958; Larvae 0,855; Larvae_subgroup 0,885; helminth_larvae 0,972; Protozoan 0,863; Protozoan_subgroup 0,906; protozoan_cysts 0,913 a 0,921; aid 0,775; flowers 0,690; remote_sensing 0,508.
Próximo: tabela de arrependimento (receita única), pedida ao Jaci.

## RECEITA ÚNICA (08/11, 10:00 UTC)

Tabela de arrependimento sobre o `sol_wsal` (104 receitas fixas, 12 bases, 5 famílias, só validação): pelo minimax, a receita é **`fpsw@mapa_w0.5_1`, 6 camadas concatenadas (L6 `cat`), 256 filtros, whitening entre camadas, saliência `gmr` como peso com corte 0,5, 50 superpixels DISF, leitura GeM + PCA-whitening + SVM**.
- Pior arrependimento entre as famílias: −0,029 (ovos −0,008, larvas −0,024, protozoários −0,022, aid+remote_sensing −0,029, flowers −0,027).
- Contra o melhor aleatório: +0,199 ovos, +0,290 larvas, +0,210 protozoários, +0,028 aid+remote_sensing, +0,067 flowers. Ganha nas 5 famílias.
- Família de fora: em 4 das 5 rodadas a escolha cai na mesma receita (3 com a mesma leitura); arrependimento na família de fora de −0,008 a −0,047, pior ao tirar as larvas.
- As 10 melhores receitas são todas L6 com 256 filtros. A melhor receita-controle fica em 73º lugar.
Ressalva: arrependimento e máximo medidos na mesma validação; otimista para todas as receitas.

## Normalizações N2, N3 e N7 (`sol_p1`, Jaci, 3 splits; 08/11, 10:00 UTC)

Contra a normalização atual (N0), nos métodos, L3 `cat`: global (N2) −0,089 (vence em 4 de 36 células), por canal (N3) −0,003 (12 de 36, empate), mediana e MAD (N7) −0,236 (0 de 36). O N2 derruba as larvas (−0,29); o N7 desaba em todas as bases.
Decisão: a normalização atual fica. Junto com o N6, a normalização não é onde está o ganho.

## Multiescala nas 7 bases restantes (`ms_a2`, Jaci, 3 splits × 5 sorteios; 08/11, 11:31 UTC)

Média das 7 bases (64 filtros, L3 `cat`): S8 0,714; S5 0,710; geodésico sem escala 0,704; S3 0,698; S7 0,697; S6 0,687; S1 0,686; S2 0,686; S4 0,623. Junto com a Rodada A, a multiescala fica no empate com o geodésico (S8 e S5 +0,006 a +0,010); S4 (mapa DoG como filtro) é o pior em todas as bases. Decisão: multiescala fora da receita. Rodada B (`ms_b2`) em andamento.

## Receita com e sem saliência (`sol_wsal`, L6 `cat`, 3 splits; 08/11, 11:50 UTC)

Receita (`fpsw@mapa_w0.5_1`) − `fps` sem saliência: aid −0,010 (64) e −0,014 (256); remote_sensing +0,006 e −0,015; flowers +0,022 e +0,020. Larvas +0,03 a +0,14; ovos +0,006 a +0,033; protozoários +0,038 a +0,049 com 64 filtros, mas −0,002 a −0,016 com 256 (Protozoan, Protozoan_subgroup, protozoan_cysts). Onde a saliência não ajuda, `fps@combo` sem saliência fica igual ou um pouco acima do `fps`.
Leitura: a saliência ajuda quando há objeto isolado e o mapa o acha; em aid atrapalha um pouco, em remote_sensing é ruído. Candidata a regra sem rótulo: ligar a saliência só quando o mapa `gmr` é concentrado (medida de concentração já existe em `saliency/fusion.py`), validada deixando uma família de fora.

## Tamanho da imagem × arquitetura (Jaci, só leitura; 08/11, 13:00 UTC)

Fontes: `sol_wsal` (12 bases, 64/256, L2 a L6), `spfdeep_esp` (até L8), `r4` (kernel 3/5/7, largura 32 a 256, entrada 128/200, 5 bases), coconut_trees do athena (1 split).
- Largura: 256 vence em 12 de 12 bases do jaci, inclusive Protozoan (nativo 64×64, ganho +0,118). Só coconut_trees prefere 64 (−0,118 para 256, 1 split). Não há relação entre tamanho nativo e largura (ρ = +0,41, p = 0,17, puxado só pelo coconut).
- Profundidade: o ganho de L6 sobre L3 acompanha o número de imagens de treino (ρ = +0,67, p = 0,02, n = 12). Com pouco treino (remote_sensing 682, Larvae_subgroup 519, helminth_larvae) o melhor fica em L3 ou L4; Protozoan (26 mil) e Eggs (8,9 mil) ganham +0,06 a +0,08 com L6.
- Kernel 3 e entrada 200 ganham ou empatam em 4 e 5 de 5 bases do `r4`; a largura cresce de forma monotônica nas 5.
- Proposta de teste: coconut_trees e Protozoan com larguras 32/64/128/256 lado a lado, para separar tamanho da imagem de número de classes.
- Correção (08/11, 13:05 UTC): o `sol_wsal` tem 3 splits × 1 sorteio (seed 42), não 2 sorteios. Os κ e a tabela de arrependimento não mudam; só o n de cada célula é menor.

## SoftHebb Fila 1 completa (`sol_hebb1`, athena, END 13:08 UTC rc=0; 12 bases, L3, médias de splits e leituras)

Regimes: A = filtros SPiFiL sem SoftHebb; C = SPiFiL refinado pelo SoftHebb; D = SoftHebb partindo do aleatório.
`fps`: A ≥ C em 9 das 12 bases (C ganha só em aid +0,005, flowers +0,004, remote_sensing +0,013). Larvae −0,035 e Larvae_subgroup −0,016 com C. `crus@geodesic`: C ganha em 8 das 12, mas parte de um nível mais baixo e fica abaixo do `fps` A em 9 das 12. D (aleatório + SoftHebb) fica abaixo de `fps` A em 9 das 12 (exceções: aid, remote_sensing, Eggs_subgroup empatado).
Veredito: o SoftHebb não melhora o melhor SPiFiL; confirma o piloto. Filas 3 a 6 ficam sem prioridade.
- Correção (08/11, 14:00 UTC) da seção SoftHebb Fila 1: os regimes são A = SPiFiL com 0 passos (só calibração das estatísticas de BN), C = SoftHebb (início random, fps ou crus; 30, 90 ou 300 passos) e D = aleatório com 0 passos. Não existe "D = aleatório + SoftHebb": essa comparação é C com início aleatório. O veredito (o SoftHebb não melhora o fps fixo) não muda. Com fps em 256: A 0,732; C30 0,708; C90 0,703; C300 0,704; C restrito (bola de raio 0,5·|w0|) 0,721.

## HAM10000 (jaci, `disc_md_ham_jaci`, parcial em 08/11 14:03 UTC; 256 filtros, `cat`, só validação)
κ / F1 macro / acurácia balanceada, L6: receita .460 / .518 / .485 (n=4); fps sem saliência .474 / .532 / .489 (n=6); melhor aleatório (escolhido no sorteio 42, reportado no 43) .333 / .375 / .341. L3: receita .454, fps .459, aleatório .427.
Receita − fps pareado: L6 −0,001 ± 0,008; L3 +0,008 ± 0,012. O SPiFiL ganha do aleatório por +0,13 em L6 (o aleatório cai de L3 para L6; o SPiFiL sobe). A saliência ainda não acrescenta κ, embora concentre a ativação na lesão. Faltam os jobs do sorteio 44 (previsão: 14:45 a 15:00 UTC).

## Polaridade (`sol_p2`, athena, END 14:12 UTC rc=0; 6 bases × 3 splits, L3 `cat`, κ de validação)
- `abs` (|R|): perde em quase tudo (fps 256: Larvae_subgroup .698 → .500; Protozoan_subgroup .813 → .710).
- `leaky` 0,1: empata com o fps; único sinal é Larvae_subgroup com `fpsw@mapa_peso` em 64 (.617 → .657).
- `crelu` (R+ e R−, dobra os canais): fps 256 → 512 canais ganha +0,007 a +0,028 (Protozoan_subgroup .813 → .841; aid .704 → .724), mas o aleatório com `crelu` também sobe, e em aid e flowers fica acima (.742 e .644): é ganho por dimensão.
- `crelu_h` (mesmo orçamento, metade das direções): igual ou abaixo do fps.
Veredito: guardar a parte negativa não melhora o SPiFiL com o mesmo orçamento; linha encerrada. O ReLU atual fica.
- HAM10000 final (jaci, rc=0 às 15:34 UTC; 3 splits × 3 sorteios, 256 filtros, `cat`): L6 receita 0,461 ± 0,009 e fps 0,468 ± 0,021 (receita − fps pareado −0,007 ± 0,026); o melhor aleatório (escolhido no sorteio 42, medido no 43 e 44) fica em 0,340. L3: fps 0,457, aleatório 0,419. Empate entre receita e fps; o SPiFiL fica +0,12 acima do aleatório em L6.

## Análise de dados: textura e filtros por classe (09/11, 00:50 UTC; só diagnóstico, rótulos fora do encoder)
- Equivalência de patches (`../analise_textura/analise_textura.md`): o patch "só se parecer com ele mesmo" quase não acontece (0,1 a 1,3%). Mas em remote_sensing, aid e flowers o vizinho mais próximo costuma ser de outra classe: a razão d_mesma/d_outra é de 1,11 a 1,19, contra 0,92 a 0,98 nas larvas e nos ovos. Essa razão acompanha a distância até o DINOv2 (ρ = 0,77, 6 bases).
- Filtros fiéis, trocados e genéricos (`../analise_textura/kernels_classe/kernels_classe.md`): no banco L1 real da receita, os filtros fiéis são rs 9%, aid 2% e flowers 17%. Em todas as 6 bases, o FPS pega MENOS filtros fiéis que um sorteio dos mesmos candidatos. Classes de textura lisa (Desert) dominam a maioria dos filtros; aid tem 27 de 30 classes sem nenhum filtro fiel. Em rs e aid, as classes que não dominam filtros têm recall de 0,27 e 0,52, contra 0,57 e 0,79. Mesmo assim, os trocados são úteis (tirá-los custa tanto quanto tirar filtros sorteados), exceto em flowers (tirá-los ajuda +0,032).
- Hipótese de ação (proposta do coordenador, ainda não testada): limitar, sem rótulo, quantos filtros cada imagem ou grupo de imagens parecidas pode fornecer, para as texturas lisas não monopolizarem o banco.
