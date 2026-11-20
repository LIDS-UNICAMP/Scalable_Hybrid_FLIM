# SPiFiL sem backprop: relatório de 07 e 08/11/2026

Tudo medido em validação; o teste nunca entra em escolha. Régua: aleatório puro (sem nenhuma técnica) e DINOv2. Horas em UTC (relógio do athena; o jaci está 3 h atrás).
Detalhes em `decisoes_spifil.md` e `results_log.md` (nesta pasta) e em `/exps/moliveira/reports/rodada2/{plano,auditoria,pool_foco,leis_p0_p1,hebb_4_1_auditoria}.md`.

## 1. A receita até agora

| peça | escolha | evidência |
|---|---|---|
| pontos dos filtros | `fps` | outros pontos (geodésico, centroide, SIFT, multiescala) mudam menos de 0,015 |
| saliência | peso pelo mapa `gmr`, corte 0,5 (`fpsw@mapa_w0.5_1`) | ajuda em larvas, ovos e flowers; em aid e remote_sensing, sem ganho → if/else do usuário |
| whitening entre camadas | sim | maior ganho isolado; vence o aleatório em 11 de 12 bases |
| profundidade | 6 camadas, concatenadas | com pouco treino (< ~1000 imagens), 3 a 4 bastam (ρ = 0,67 entre ganho L6−L3 e nº de imagens de treino) |
| filtros | 256 por camada | 256 vence em 12 de 12 bases; kernel 3 a 5 |
| leitura | GeM + PCA-whitening + SVM | multiescala com dimensão igual ganha +0,016 (passou o critério) |
| ativação | ReLU | polaridade (abs, leaky, crelu) não ganha com o mesmo orçamento; GELU em teste |

Receita única (minimax em 5 famílias): pior arrependimento −0,029. Ganho sobre o melhor aleatório: ovos +0,199, larvas +0,290, protozoários +0,210, aid + remote_sensing +0,028, flowers +0,067.

## 2. Melhor SPiFiL por base (`sol_wsal`, 3 splits; o "melhor" é escolhido na própria validação, é otimista)

| base | melhor κ | receita | fps sem saliência | melhor aleatório | DINOv2 |
|---|---|---|---|---|---|
| helminth_larvae | 0,980 | 0,948 | 0,954 | 0,931 | 0,994 |
| helminth_eggs | 0,960 | 0,957 | 0,953 | 0,833 | 0,987 |
| protozoan_cysts | 0,921 | 0,908 | 0,921 | 0,797 | 0,966 |
| Protozoan_subgroup | 0,906 | 0,893 | 0,895 | 0,687 | 0,945 |
| Eggs_subgroup | 0,893 | 0,890 | 0,873 | 0,721 | 0,989 |
| Larvae_subgroup | 0,887 | 0,883 | 0,782 | 0,443 | 0,981 |
| Eggs | 0,866 | 0,866 | 0,844 | 0,630 | 0,980 |
| Protozoan | 0,863 | 0,834 | 0,850 | 0,592 | 0,939 |
| Larvae | 0,855 | 0,855 | 0,744 | 0,473 | 0,968 |
| aid | 0,775 | 0,759 | 0,773 | 0,677 | 0,970 |
| flowers | 0,690 | 0,681 | 0,662 | 0,584 | 0,973 |
| remote_sensing | 0,535 | 0,507 | 0,511 | 0,508 | 0,708 |
| coconut_trees (L, 1 split) | 0,792 | — | — | 0,511 | — |
| HAM10000 (3×3) | 0,468 (fps) | 0,461 | 0,468 | 0,340 | — |

## 3. Linha do tempo

### 07/11
- SPiFiL passa a ser objetivo próprio: detectar e classificar dentro do modelo, sem backprop e sem máscara; o aleatório é só régua.
- Mapas de saliência e detecção (gmr, fusões), galeria de ativações por base, estudo de superpixels por base.
- Início das grades SOL (P1 a P6, ablações T1 a T18), agrupamento C3 a C9, multiescala (Rodadas A e B), SoftHebb Fila 1 e Fila 2 (pilotos).
- Jaci entra como segunda máquina.
- Teto de GPU no athena: 15 GB nossos por GPU.

### 08/11, madrugada e manhã
- Whitening entre camadas nas 12 bases (`spfcombo_w2`, `sol_wsal`): confirmado.
- Saliência como peso ou foco (`salw_a`, 26 variantes): todas ganham do fps (+0,015 a +0,034).
- Tabela de arrependimento → receita única; deixando uma família de fora, a mesma receita sai em 4 de 5 rodadas.
- Descartados: Z-score do patch, normalizações N2, N3, N6 e N7, agrupamento C3 a C9, ZCA e PCA nos filtros, multiescala dos pontos, densidade SIFT, fusão Top-K, GAP, GMP e SPP, mais de 256 filtros.
- Bases novas: coconut_trees e food101 (50%). Incidentes: GPU acima de 15 GB no athena (imagens por classe sem teto), load de 160 às 02:00.

### 08/11, tarde
- **Saliência com e sem**: sem saliência, aid e remote_sensing ficam iguais ou melhores; o if/else fica com o usuário.
- **Tamanho da imagem × arquitetura**: imagem pequena NÃO pede rede estreita (Protozoan, 64 px nativos, ganha +0,118 com 256 filtros); a profundidade útil acompanha o nº de imagens de treino.
- **Pastas do athena reorganizadas**: `fontes_filtros/`, `pilotos/`, `analises/`, `recebido_jaci/`, `codigo_congelado/`, `temporarios_sistema/`; `tmp` virou atalho. Guia: `/exps/moliveira/LEIAME_pastas.md`.
- **Saliência gaussiana**: peso por gaussiana ajustada ao mapa (k = 1, 2, 3), em vez de corte; rodando no jaci (`disc_md_gauss_jaci`).
- **Rodada 2: auditoria** (o que já existia):
  - Métricas: `val_kappa` = Cohen; `val_f1` = F1 macro; `val_acc` = acurácia balanceada.
  - Os 3 splits são reembaralhamentos (treinos com ~70% em comum): a família é a unidade independente.
  - 971 linhas duplicadas de shards (com valores iguais).
- **Diversidade funcional (B0 a B3)**: nenhuma variante melhora o fps com o mesmo orçamento (diferenças −0,004 a +0,001). Encerrada.
- **SoftHebb Fila 1 (implementação antiga)**: piora o fps (256: 0,732 → 0,704 com 300 passos). A auditoria contra o código dos autores achou 3 diferenças: competição seletiva demais (inv_temp 4 contra 1 e 0,65), teto do passo por filtro em vez de por tensor e lr igual nas camadas. O efeito vem dos pesos, não da calibração.
- **Polaridade (`sol_p2`)**: o ganho do crelu vem da dimensão dobrada; com o mesmo orçamento, empata. Encerrada.
- Regra do usuário: **o aleatório fica sempre sozinho** (régua pura).
- **HAM10000**: receita e fps empatam (0,46 e 0,47), +0,12 sobre o aleatório; a saliência concentra a energia na lesão, mas não muda o κ.
- **SSH athena→jaci**: o Claude passa a sincronizar sozinho; contador em `/exps/moliveira/queues/rsync_contador.log`.
- **food101 pausada** (base grande demais): 16 experimentos guardados.

### 08/11, tarde e noite
- **SoftHebb com a receita dos autores: CONFIRMADO** (`sol_hebb3_conf`, 5 bases × 3 splits × 5 sorteios). Com 256 filtros, FPS + SoftHebb − FPS fixo (κ, `cat`): aid +0,064, Larvae +0,052, flowers +0,036, Eggs +0,026, remote_sensing +0,026. F1 e acurácia balanceada sobem junto. Com 64 filtros, só aid e remote_sensing ganham. Limite: rede de 3 camadas, sem whitening e sem saliência.
- **Leitura multiescala com dimensão igual (`leitura_a2`)**: passou o critério (+0,02 a +0,03 em 4 de 6 bases); a maior parte do ganho é informação de escala. GeM p=2 encerrado.
- **Leis de construção** (cada camada com a sua largura; leis A geométrica, B por posição, C por orçamento; código em `law_channels`): 17 arquiteturas.
  - aid e flowers: a melhor é A 512-512-256-256-128-128, que empata com a uniforme 256×6 (0,786 contra 0,786; 0,644 contra 0,656) com 1,5 vez os parâmetros. Ordem estável entre sorteios em aid (ρ = 0,94).
  - remote_sensing: todas as leis ficam entre 0,47 e 0,49, mas a ordem entre sorteios é ruído (ρ = −0,07) e o aleatório sobe junto.
  - Larvas, ovos e protozoários: ainda rodando no jaci e no athena.
- **Foco novo do usuário**: aid, flowers, remote_sensing e coconut_trees; parasitos só para derivar leis. Meta: flowers e remote_sensing acima de 0,70.
- **VLAD (leitura clássica de cenas)**: ganha do GeM da última camada (+0,04), mas não supera o `cat` em nenhuma base. Encerrado como leitura principal.
- **coconut_trees**: a última camada sozinha (0,79) é muito melhor que o `cat` (0,63 com 256). O problema era a concatenação, não a largura. 128 px = 200 px.
- Filas automáticas: `fila_athena` (lança quando o nosso máximo por GPU ≤ 10,8 GB) e `fila3` no jaci (trava por item, corte do mais novo primeiro).

## 4. O que está rodando e na fila (08/11, 20:30 UTC)
- **athena**: leis dos protozoários (sorteios 42 e 43), GELU, resolução 320 px (aid, remote_sensing), controle aleatório do SoftHebb de 3 camadas, coconut nos experimentos restantes. Fila automática: SoftHebb de 3 camadas com saliência, SoftHebb de 4 camadas (sem e com saliência), aleatório de 4 camadas, 448 px. Depois, benchmark de 12 modelos.
- **jaci**: imagens por classe (7 bases), saliência gaussiana, teste de largura (coconut × Protozoan), leis de Larvae, Eggs e das bases menores (sorteios 42 e 43), SoftHebb de 3 camadas sem saliência, r4 → r5. Na fila3: leis das bases menores no sorteio 43, leis + multiescala (sem e com saliência). Depois, benchmark de 6 modelos.
- **Parados de propósito**: food101 (16 experimentos) e SoftHebb Fila 2 (19 de 36 prontos).

## 5. Encerrado, confirmado, em aberto
- **Confirmado**: whitening entre camadas; saliência nas bases com objeto; fps; 256 filtros; SoftHebb com a receita dos autores (rede de 3 camadas, 256); multiescala com dimensão igual.
- **Encerrado**: normalizações novas; agrupamento; ZCA e PCA nos filtros; multiescala dos pontos; densidade SIFT; fusão Top-K; diversidade funcional; polaridade; GAP, GMP, SPP e VLAD como leitura principal; SoftHebb antigo; GeM p=2; mais de 256 filtros por camada.
- **Em aberto**: GELU; saliência gaussiana; imagens por classe; resolução 320 e 448 px; SoftHebb nas 4 bases de foco e dentro da receita completa; leis nos parasitos (para derivar a regra); leitura dependente da base (última camada contra `cat`); benchmark.
