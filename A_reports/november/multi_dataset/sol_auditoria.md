# SOL, Rodada 0: auditoria (07/11, 22:45 UTC)

Só leitura. Nada foi lançado nem editado. CSVs: `/exps/moliveira/results_sol/audit/T1_datasets.csv` e `T2_auditoria.csv` (58 linhas, com arquivo:linha de cada mecanismo).

## Resposta curta

Metade do pedido já tem dado, mas quase tudo com 1 a 2 splits ou em 2 a 7 bases. Funcionam: FPS, concatenar camadas e o whitening entre camadas (Jaci, parcial). Não funcionam: ZCA nos filtros e Z-score por patch (o aleatório cai −0,20 com ele). Faltam P2 quase inteiro, F2–F6, pooling na grade e os 3 encoders alternativos.

## Fontes e avisos

- O grafo (`graphify-out` de 06/11) não indexa `spf_grade`, `filter_diversity`, `scale_space` nem `saliency/`: `graphify query` não trouxe nada útil. Li os arquivos direto e usei grep nos nomes.
- "Jaci, parcial" = números da parte "Funciona" de `relatorio_07nov.md`: `spfcombo_w`, `spfdeep_esp`, `spf_hz`.
- Três números são contas desta auditoria sobre CSVs do Athena (marcados em T2): N1 contra N0 no aleatório, `rz2` − `random_z` e `zca2` − `crus`.

## Sutilezas que mudam a leitura

1. **A grade não usa TTA.** `READOUT = "gem_pw"` (`eval/spf_grade.py:128`) vira `gem` em `MODES` (`eval/l1_amostras.py:62`): GeM p=3, PCA-whitening com shrinkage 0,1 e L2 (`eval/svm.py:365-373`), SVC linear. TTA (`gem_tta_pw`) só existe na lei de superpixels e nos modelos treinados.
2. **O GeM já corta o negativo** (`clamp(min=1e-6)`, `eval/svm.py:360`). Sem ReLU, o negativo continua sumindo na leitura. Para testar polaridade, a leitura também tem de mudar (A5).
3. **`entre_camadas` = ZCA dos canais + o z-score de sempre.** Da L2 em diante, mede média e ZCA C×C no pixel central das sementes medoide (shrinkage 0,1). Os candidatos passam pelo mesmo ZCA, depois vem o z-score N0. O u escolhido é dobrado na conv sobre o mapa cru: ReLU e pool continuam no mapa cru. O kernel não é branqueado. Sai também a leitura `_b`, sem `pw_whiten` (`eval/spf_grade.py:893-913`, `:973-981`, `:1001-1002`, `:1023-1026`). Não roda com `spifil` (`:947`).
4. **O Z-score por patch vale para todos os braços do job**, controles inclusive (`PatchCtx`, `:665-690`; `ctx.rez` nos controles, `:990-991`), menos `spifil` (`:948`). Ele também tira a média do kernel por banda e zera o bias (`:997-998`): o filtro perde a cor média. Os YAMLs `cl_*` ainda trocam DISF 50 por SLIC 400.
5. **`activation` de `filter_diversity.py` já é F3** (gulosa de baixa correlação nas ativações, `flim/filter_diversity.py:87-95`). Não é F2 (FPS na ativação). Ninguém importa o arquivo: nunca rodou.
6. **`cat`** junta o GeM das camadas, da última para trás, até ficar abaixo de 2000-d, e branqueia junto (`:1029-1037`). **`soma`** soma os GeM com L2 de cada camada; exige larguras iguais (`:1038-1039`).
7. **Já existe R+ e R− com orçamento igual, mas só no aleatório.** `rz2` e `whrand2` usam +u e −u com metade das direções (`:726-727`, `:866-870`). Não faz diferença: `rz2` − `random_z` = −0,001 em 12 pares. Para FPS não existe, e o FPS mede `|cos|`, ou seja, trata u e −u como iguais.
8. **N0 já é um z-score por dimensão** (posição × banda). Por canal (N3) e global (N2) não existem.

## 1. Datasets (T1)

Contagens de `organized/<base>/`, splits `splits/split{1,2,3}.json`. Fração de primeiro plano: `area_frac` dos `_diag.csv` a 200 px; nas bases `helminth_*` e `protozoan_cysts`, 50 máscaras na resolução original.

| base | imagens | classes | menor / maior classe | tamanho | canais | máscara | primeiro plano | split 1 treino/val/teste |
|---|---|---|---|---|---|---|---|---|
| Eggs (raw) | 12.691 | 9 | 83 / 9.815 | variável, 96 a 1360 px | RGB | `raw/label` | 0,198 | 8.885 / 1.903 / 1.903 |
| Protozoan (raw) | 37.376 | 7 | 309 / 28.528 | variável, 32 a 512 px | RGB | `raw/label` | 0,121 | 26.162 / 5.607 / 5.607 |
| Larvae (raw) | 1.598 | 2 | 247 / 1.351 | variável, 112 a 2736 px | RGB | `raw/label` | 0,060 | 1.118 / 240 / 240 |
| Eggs_subgroup | 3.974 | 9 | 48 / 2.579 | 200×200 | RGB | `masks` | 0,202 | 2.782 / 593 / 599 |
| Protozoan_subgroup | 7.919 | 7 | 18 / 4.457 | 200×200 | RGB | `masks` | 0,120 | 5.532 / 1.190 / 1.197 |
| Larvae_subgroup | 741 | 2 | 137 / 604 | 200×200 | RGB | `masks` | 0,063 | 519 / 107 / 115 |
| helminth_eggs | 5.112 | 9 | 80 / 3.344 | 200×200, fundo zerado | RGB | `label` | 0,031 | 3.580 / 766 / 766 |
| helminth_larvae | 3.514 | 2 | 446 / 3.068 | 200×200, fundo zerado | RGB | `label` | 0,020 | 2.460 / 527 / 527 |
| protozoan_cysts | 9.568 | 7 | 78 / 5.716 | 200×200, fundo zerado | RGB | `label` | 0,034 | 6.698 / 1.435 / 1.435 |
| aid | 10.000 | 30 | 220 / 420 | 600×600 | RGB | não | - | 6.996 / 1.502 / 1.502 |
| remote_sensing | 1.000 | 14 | 20 / 173 | 640×640 | RGB | não | - | 682 / 159 / 159 |
| flowers | 2.746 | 5 | 495 / 646 | variável, moda 500×333 | RGB | não | - | 1.922 / 412 / 412 |

Os 3 subgrupos e remote_sensing mudam de tamanho entre splits; as outras bases não. Os splits são reamostragens, não folds.

### Sobreposição

Hash do arquivo e hash dos pixels a 64×64: 0 de 200 iguais em todos os pares. Por isso entrou um terceiro teste: recorte pela máscara, 32×32 cinza, vizinho mais próximo com correlação ≥ 0,98.

| par | o que achou | leitura |
|---|---|---|
| subgrupo × base completa (ovos, protozoários, larvas) | 195, 198 e 181 de 200; o vizinho é o alvo de `images_raw_links` em 200 de 200 | subgrupo = imagens raw em letterbox 200×200 |
| `helminth_*`, `protozoan_cysts` × raw, pre e subgrupo | 0 a 2 de 200 | sem pixel em comum achado |
| remote_sensing × aid | 455 de 1.000 (454 com ≥ 0,995); aid contra si mesmo: 0 de 500 | remote_sensing contém aid: vazamento entre as duas |
| flowers × todas | 0 de 200 | independente |
| raw × pre (Eggs) | 0; só 1.555 nomes em comum | numerações diferentes |
| internas | flowers 3, Eggs 1, aid 1 | duplicatas dentro da base |

Nome de arquivo não serve: todas usam `CCCCCC_NNNNNNNN`, e as colisões são coincidência.

**Conflito em aberto.** `mapa_ids.csv` dos subgrupos tem a coluna `nome_hoje`, que aponta para um arquivo de `helminth_*` ou `protozoan_cysts`. Testei 40 pares por base, só nos pixels dentro da máscara: correlação mediana 0,40 / 0,27 / 0,02 no par do mapa, contra 0,21 / 0,11 / 0,01 num par sorteado da mesma base. Nenhum par passa de 0,9. O mapa liga as bases, mas os pixels não confirmam. Talvez seja a mesma lâmina, refotografada ou segmentada de novo.

### Famílias independentes

- Pelo pixel: 8 famílias. Ovos (Eggs + Eggs_subgroup), protozoários, larvas, helminth_eggs, helminth_larvae, protozoan_cysts, aid + remote_sensing, flowers.
- Conservador, que sugiro para o critério "≥ 3 de 5 famílias": **5 famílias**. Ovos (Eggs, Eggs_subgroup, helminth_eggs), protozoários (Protozoan, Protozoan_subgroup, protozoan_cysts), larvas (Larvae, Larvae_subgroup, helminth_larvae), aid + remote_sensing, flowers. Mesma taxonomia, mesmo laboratório, e um mapa de ids que liga as bases.
- Piloto de 6 bases: Eggs_subgroup, Protozoan_subgroup, Larvae_subgroup, aid, flowers e helminth_eggs. helminth_eggs entra como contraste de fundo zerado. remote_sensing não entra como família à parte.

## 2. Auditoria por mecanismo (T2)

Protocolo comum, salvo aviso: 3 camadas, kernel 5, pool 2, ReLU, 200 px, 50 superpixels DISF, 500 imagens, 50% dos dados, leitura `gem_pw` L3 `cat`. Δκ é contra o melhor aleatório, só na validação. As colunas completas (bases, arquitetura, sementes, comparabilidade) estão no CSV.

### P1 normalização

| id | mecanismo | identificador | estado | resultado | código | próximo |
|---|---|---|---|---|---|---|
| N0 | z-score atual | todos os YAMLs disc_md_spf_*, ms_*, spfdeep_*, spfcombo_* | concluido | referencia de todos os deltas | eval/spf_grade.py:976-977, eval/spf_grade.py:999-1000 | reusar |
| N1 | Z-score por patch | disc_md_cl_ctl (concluido), disc_md_cl_a (ativo, sem CSV) | parcial | aleatorio (xavier, random_z) com N1 menos o mesmo aleatorio com N0: -0,197 (30 pares, 5 bases), de -0,10 (flowers) a -0,33 (Eggs_subgroup) | eval/spf_grade.py:659-690 (band_z, PatchCtx), eval/spf_grade.py:985-998, assert eval/spf_grade.py:948 | completar: cl_a da o SPiFiL; falta fps e crus@geodesic sob N1 com DISF 50 |
| N2 | Z-score global | - | inexistente | - | ausente | fazer: chave amostra.norm |
| N3 | Z-score por canal | - | inexistente | - | ausente (por dimensao: eval/spf_grade.py:976) | fazer: mesma chave amostra.norm |
| N4 | whitening dos filtros | zca2, zca2h, zca2@*, pca, pca2, whrand, whrand2, zca_km2 em disc_md_spf_r1/r2, spf_h, spf_h_raw | concluido | zca2 -0,109 (64) e -0,126 (256), 0 a 1 de 7 bases; zca2 - crus -0,115 (12 pares, 2 bases, calculo desta auditoria) | eval/spf_grade.py:730-745, Ctx.white eval/spf_grade.py:646-651 | reusar |
| N5 | whitening entre camadas | disc_md_spfcombo_w_jaci | parcial (Jaci, parcial) | fps@combo 256 L3: Eggs_subgroup 0,821->0,905; Larvae_subgroup 0,613->0,828; Protozoan_subgroup 0,813->0,862; aid 0,714->0,748; flowers 0,588->0,637; passa do melhor aleatorio nas 6 bases | eval/spf_grade.py:893-913, eval/spf_grade.py:973-981, eval/spf_grade.py:1001-1002, _b eval/spf_grade.py:1023-1026, assert eval/spf_grade.py:947 | completar: 3 splits, fps e crus@geodesic, aleatorio com e sem |
| N6 | Z-score por patch + whitening entre camadas | - | inexistente | - | eval/spf_grade.py:946-998 | fazer: so YAML (agrupamento + entre_camadas), conferir num check |
| N7 | mediana/MAD | - | inexistente | - | ausente | fazer: amostra.norm: mad |

### P2 polaridade

| id | mecanismo | identificador | estado | resultado | código | próximo |
|---|---|---|---|---|---|---|
| A0 | ReLU | todas as grades | concluido | referencia | flim/arch.py:107-108, flim/spifil.py:722, eval/svm.py:360 | reusar |
| A1 | R+ e R- (orcamento igual) | rz2, whrand2, zca2, pca2 (controles e ZCA/PCA) | parcial (so aleatorio e ZCA) | rz2 - random_z -0,001; whrand2 - whrand -0,009 (12 pares cada) | eval/spf_grade.py:726-727, eval/spf_grade.py:866-870, fps eval/l1_sem_rotulo.py:184-189 | fazer: fps2/crus2 = signs(metodo(largura/2)), 1 linha em METHODS |
| A2 | /R/ | - | inexistente | - | ausente | fazer: chave amostra.ativacao (troca o ReLU do encoder) |
| A3 | Leaky fixo | ga_search/spifil_ext.py (caminho antigo do GA) | inexistente na grade | - | ga_search/spifil_ext.py:31-46; ausente em flim/arch.py | fazer: mesma chave amostra.ativacao |
| A4 | sinal e magnitude | - | inexistente | - | ausente | fazer: amostra.ativacao, com largura/2 |
| A5 | agregacao separada de R+ e R- | - | inexistente | - | ausente; clamp em eval/svm.py:360 | fazer: leitura nova sobre o mapa antes do ReLU |

### P3 diversidade funcional

| id | mecanismo | identificador | estado | resultado | código | próximo |
|---|---|---|---|---|---|---|
| cand | candidatos crus, geodesico, centroide, SIFT filtrado, objeto, combo | crus, crus@geodesic, crus@centroid, crus@sift_nms, crus@obj, fps@combo: disc_md_spf_r1/r2, spf_h, ms_b, spfcombo_a | concluido | ranking (celula escolhida em 2 splits): crus@obj +0,126, crus@geodesic +0,097, crus@sift_nms +0,097; ms_b em 5 bases: geodesico +0,066, centroide +0,077, SIFT +0,060 | eval/spf_grade.py:24-29, eval/spf_grade.py:476-512 | reusar |
| sel-ale | selecao por patch: aleatoria (F0) | crus | concluido | crus +0,113 (7 bases, 0 boas) | eval/l1_sem_rotulo.py:180-181 | reusar |
| sel-fps | selecao por patch: FPS (F1) | fps, fps@* | concluido | +0,071 (64) e +0,072 (256) em 7 bases, pior -0,017; ms_b: +0,087, Delta>0 em 5/5 | eval/l1_sem_rotulo.py:184-189 | reusar |
| sel-kmed | selecao por patch: K-medoids | kmedoids_visual (nao ligado); km_w e o k-means com representante real | parcial | km_w - fps: -0,028 [-0,049; -0,007] (64), -0,044 [-0,072; -0,016] (256); K-medoids proprio: sem dado | flim/filter_diversity.py:55-68, flim/filter_diversity.py:82-86; km_w eval/spf_grade.py:839, eval/spf_grade.py:812-819 | fazer: ligar kmedoids_visual na grade |
| sel-cos | selecao por patch: cosseno | cos, cos@* | concluido | cos +0,108 (5 bases, 1 boa) | eval/l1_sem_rotulo.py:228-233 | reusar |
| sel-corr | selecao por patch: correlacao | - | inexistente | - | ausente | fazer junto com N1 (cos sobre patch em z proprio = correlacao) |
| sel-dpp | selecao por patch: DPP | dpp em disc_md_amostra_stack_flowers (l1_sem_rotulo) | parcial | Delta contra um so aleatorio (rand): +0,028 (64, L1), +0,031 (64, L3), +0,044 (128, L3), +0,024 (256, L3) | eval/l1_sem_rotulo.py:207-221 | fazer: 1 linha em METHODS da grade (rowwise(dpp)) |
| F2 | FPS na ativacao | - | inexistente | - | ausente; peca: flim/filter_diversity.py:43-52 | fazer: greedy(corr, n, first, max) em filter_diversity e ligar na grade |
| F3 | baixa correlacao de ativacao | activation (filter_diversity, D3) | inexistente como experimento (codigo pronto, ninguem importa) | - | flim/filter_diversity.py:87-95, flim/filter_diversity.py:98-125 | completar: adaptador Ctx -> PatchSet e ligar em METHODS |
| F4 | cobertura | - | inexistente | - | ausente | fazer |
| F5 | visual + funcional | - | inexistente | - | ausente | fazer: soma das duas distancias em filter_diversity |
| F6 | redundancia minima + estabilidade multiescala | - | inexistente | - | ausente; pecas: eval/spf_fundo.py:347, flim/scale_space.py | fazer (depois de F3) |
| dd | eliminar copias (spifil_dd, cos_prune) | spifil_dd, cos_prune | concluido (regime B) | spifil_dd +0,027 (2 bases) | flim/kernel_redundancy.py:100-180 | reusar |

### P4 saliência e fundo

| id | mecanismo | identificador | estado | resultado | código | próximo |
|---|---|---|---|---|---|---|
| mapas | 11 mapas classicos | saliency/batch.py, experiments/saliency/lote.yaml | concluido | gmr escolhido sem rotulo: Q 0,725, estabilidade 0,898, 0,33 s; IoU 0,781 (180 imagens) | saliency/maps_a.py:212-219, saliency/maps_b.py:261-267 | reusar |
| S0 | sem atencao | todos os bracos sem @mapa | concluido | referencia | - | reusar |
| S1 | melhor mapa (gmr fixo) | fps@mapa_*, cos@mapa_*, crus@nms_*, fpsw@mapa_peso: disc_md_spf_sal_fl, spf_sal_par | concluido | fpsw@mapa_peso +0,174 (ranking); contra fps: +0,043 [+0,019; +0,067] (64), +0,015 [-0,013; +0,043] (256); ganho so nas larvas | eval/spf_grade.py:31-37, fps_weighted eval/spf_grade.py:703-711 | completar: aid, helminth, remote_sensing; salw_a no Jaci |
| S2 | media dos mapas | D (simple_mean) | parcial (so IoU) | AP 0,939, IoU 0,751 | saliency/fusion.py:225-227 | fazer: cache de D e saliency.cache |
| S3 | fusao ponderada Top-K | B (top_k + weighted_mean), topk_lote, topk_jaci | parcial | IoU 0,657 contra gmr 0,643 (432 imagens); em helminth_* perde do gmr | saliency/fusion.py:201-213, :233-235, saliency/topk_lote.py | completar: B via saliency.cache na grade |
| S4 | fusao regional | C (regional_fusion) | parcial (so IoU) | IoU 0,774 | saliency/fusion.py:311-369 | fazer: cache de C |
| S5 | adaptativa (melhor mapa por imagem) | A (best_single) | parcial (so IoU) | IoU 0,787; estabilidade cai a 0,62 em remote_sensing | saliency/fusion.py:229-231 | fazer: cache de A |
| S6 | estabilidade entre escalas | scale_consistency e _stability so entram no Q | inexistente como braco | - | saliency/fusion.py:133-140, :160-168 | fazer: peso = estabilidade do mapa |
| fundo-a | onde as sementes caem | _diag.csv de toda grade | concluido | medoide 25% a 49% no objeto (2,4 a 4x o acaso); saliencia 70% a 98%, mas so 9 a 23 pontos | eval/spf_grade.py:530-550 | reusar |
| fundo-b | oraculo: banco so objeto / so fundo | spifil@m_obj, crus@m_fundo...: disc_md_spf_h | concluido | so fundo ate -0,13; so objeto ate +0,066, nenhum IC acima de 0 | eval/spf_grade.py:41-44 docstring, eval/spf_grade.py:186-195 | reusar |
| fundo-c | fundo zerado (teto de oraculo) | disc_md_spf_hz_jaci | parcial (Jaci, parcial) | fundo real +0,071 sobre aleatorio; zerado +0,21; SPiFiL anda ~1/3 do caminho | eval/spf_grade.py:552-560, eval/spf_grade.py:1095-1099 | completar |
| fundo-d | cor, textura, periferia, iluminacao, troca de fundo | eval/spf_fundo.py F1-F4 | inexistente como resultado (codigo pronto, sem YAML nem CSV) | - | eval/spf_fundo.py:73, :281-398 | completar: YAML e rodar |
| fundo-e | resolucao | law_size_superpixels (L1) | concluido (so L1) | kappa medio 0,643 (64 px), 0,645 (128), 0,569 (640) | eval/l1_fast.py | completar em L3 |

### P5 pooling

| id | mecanismo | identificador | estado | resultado | código | próximo |
|---|---|---|---|---|---|---|
| R0 | leitura atual | READOUT=gem_pw | concluido | referencia | eval/spf_grade.py:128, eval/spf_grade.py:1015, eval/l1_amostras.py:62, eval/svm.py:359-361, eval/svm.py:365-373 | reusar |
| R1 | GAP | avgpool2d (fora da grade) | parcial (outro regime) | com backprop, L4: avg 0,880/0,917/0,837 contra GeM 0,899/0,910/0,875 (eggs/larvae/protozoan); na L3 de 64 canais GeM perde do avg | eval/svm.py:362 | fazer: leitura na grade |
| R2 | GMP | - | inexistente | - | ausente | fazer: modo novo em _encode_pooled |
| R3 | GeM | = R0 | concluido | - | eval/svm.py:359-361 | reusar |
| R4 | GeM com p por validacao | - | inexistente | - | ausente | fazer: grade de p, escolha em 2 splits e medida no 3o |
| R5 | SPP | spp (fora da grade) | parcial (outro regime) | spp L3 0,878/0,872/0,829 | eval/svm.py:353-355 | fazer: leitura na grade, so 64 ou com PCA ate 2000 |
| R6 | regional | gem_fg, avgpool2d_fg (fora da grade) | inexistente na grade | - | eval/svm.py:339-350 | fazer: leitura na grade |
| R7 | concatenacao de camadas | cat | concluido | spfdeep_cedo 64: fps cat L8 +0,094 contra L +0,035; lendo so a ultima camada, cai | eval/spf_grade.py:1029-1037 | reusar |
| R8 | multiescala (entrada) | gem_tta e so rotacao/flip; nao ha piramide de entrada | inexistente | - | ausente; TTA em eval/svm.py:330-332 | fazer: GeM sobre 2 ou 3 escalas da imagem |
| R9 | segunda ordem / covariancia | - | inexistente | - | ausente; eval/svm.py:713, :763 | fazer: covariancia de canais reduzidos por PCA (ex. 48 -> 1176-d) |
| R10 | combinacoes | soma, _b, cat_b | parcial | soma: Delta maior so porque o aleatorio piora; combo sem whitening entre camadas +0,017 | eval/spf_grade.py:1038-1039 | reusar |

### P6 encoders alternativos

| id | mecanismo | identificador | estado | resultado | código | próximo |
|---|---|---|---|---|---|---|
| scat | Wavelet Scattering | - | inexistente | - | ausente | fazer: modulo novo (instalar kymatio) |
| hebb | SoftHebb (regime C) | - | inexistente | - | ausente | fazer: modulo novo |
| rcnn | Random CNN (regime D) | xavier, he, orth, tnorm, random_z, whrand, rz2 | concluido | orth e o melhor em 1548 celulas, random_z 1083, he 844 | eval/spf_grade.py:855-872 | reusar |
| pcanet | PCANet | pca, pca2 (filtros = PCA dos patches) | parcial | pca -0,060, pca2 -0,025 (ranking); Coates e Ng (zca_km2) +0,015 | eval/spf_grade.py:738-745, zca_km2 eval/spf_grade.py:827-831 | completar se viavel |

### Profundidade e largura

| id | mecanismo | identificador | estado | resultado | código | próximo |
|---|---|---|---|---|---|---|
| prof-cedo | profundidade 1-8 x largura 64/256 | disc_md_spfdeep_cedo | concluido | fps@sift 4 de 6 bases boas, +0,088 (bom); fps 64 cat sobe ate L8 (+0,094); fps 256 cat pico L3 (+0,052) | eval/spf_grade.py:916-918 | reusar |
| prof-esp | profundidade 1-8, pools 2-1-2-1... | disc_md_spfdeep_esp_jaci | parcial (Jaci, parcial) | L8 cat: fps +0,235 (Larvae_subgroup 64), +0,098 (Protozoan_subgroup 64); fps@sift +0,097 (Eggs_subgroup 64) | eval/spf_grade.py:916-918 | completar |
| combo-a | combo 3-6 camadas | disc_md_spfcombo_a | concluido (aid 2 splits) | fps@combo_sc 4 boas +0,123; fps@combo 3 boas +0,103; fps@geodesic +0,143 (2 boas) | eval/spf_grade.py:375-392 | reusar |
## 3. O que falta, com a menor implementação e o custo do piloto

Piloto: 6 bases (uma por família), 3 splits, 2 sorteios, 64 e 256 filtros, 3 camadas. São 36 unidades (base, split, sorteio).

Base do custo: medianas dos CSVs `spf_r1`, `spf_r2`, `ms_ctl`, `ms_a` (seleção + extração + SVM, 3 camadas, por célula):

| célula | 64 filtros | 256 filtros | 36 unidades, 64 + 256 |
|---|---|---|---|
| braço SPiFiL | 56 s | 119 s | 1,75 h |
| controle aleatório | 20 s | 71 s | 0,9 h |
| fontes (DISF, pontos), uma vez por sorteio | ~250 s | - | 2,5 h, reusável com `amostra.fontes` |

Horas em série. Com 8 jobs ao mesmo tempo (4 GPUs, `max_per_gpu: 2`), divida por 6 a 8.
Pacote típico de um mecanismo: 2 braços (`crus@geodesic`, `fps`) e 2 controles (`xavier`, `random_z`) = 5,3 h, mais as fontes.

Bases do piloto: Eggs_subgroup, Protozoan_subgroup, Larvae_subgroup, aid, flowers, helminth_eggs.

Controles prontos para reusar quando o mecanismo não mexe no aleatório (P3, P4):
- 64 filtros: 5 sorteios, nas 12 bases (`ms_ctl`, `ms_ctl2`).
- 256 filtros: só o sorteio 42, mas exatamente nas 6 bases do piloto. É a L3 de `spfdeep_cedo`, que é a mesma rede de 3 camadas com pool 2.
- Falta só o controle de 256 filtros no sorteio 43, cerca de 0,45 h.

| mecanismo | menor implementação | onde | custo do piloto (série) |
|---|---|---|---|
| N1 completar | YAML com `agrupamento` e DISF 50, braços `crus@geodesic`, `fps` | só YAML | 5,3 h |
| N2, N3, N7 | `amostra.norm: global / canal / mad`: troca `mean, stdev` em `chain` (~6 linhas) | `spf_grade`, chave nova | 5,3 h cada; 16 h as três |
| N5 completar | YAML Athena de `spfcombo_w` com 3 splits, mais `fps` e `crus@geodesic`, aleatório com e sem | só YAML | ~7 h |
| N6 (N1 + N5) | YAML com `agrupamento` + `entre_camadas`; check de 1 célula antes | só YAML | 5,3 h |
| A1 para SPiFiL | `fps2`, `crus2` = `signs(metodo(largura/2))`, 1 linha cada em `METHODS` | `spf_grade` | 3,5 h + 1,8 h (`rz2`, `random_z`) |
| A2, A3 | `amostra.ativacao: abs / leaky`: troca o `nn.ReLU` do encoder depois de `build` (~6 linhas) | `spf_grade`, chave nova | 5,3 h cada (o aleatório refaz) |
| A4 | pede 2 saídas por canal: muda `ch_in` em `flim/arch.py`. Sugestão: tratar como A1 + A2 | mexe fora da grade | só se A1 ou A2 ganhar |
| A5 | leitura sobre o mapa antes do ReLU: GeM(ReLU(x)) e GeM(ReLU(−x)) juntos, 2C-d | `spf_grade`, chave `amostra.leituras` | ~3 h (refaz extração) |
| DPP na grade | `"dpp": rowwise(dpp)` em `METHODS` | `spf_grade`, 1 linha | 1,75 h |
| K-medoids, F3 | adaptador `Ctx` → `select_filters` de `flim/filter_diversity.py` (~15 linhas); F3 já existe como `activation` | `spf_grade` + importação | 2 a 2,5 h cada (a correlação de ativação pede os mapas) |
| F2 | `greedy(corr, n, first, torch.maximum)`, 1 linha em `filter_diversity` | `filter_diversity` + adaptador | 2,5 h |
| F4, F5 | cobertura (k-centro na ativação) e soma de distâncias visual + funcional, ~10 linhas cada | `filter_diversity` | 2,5 h cada |
| F6 | F3 + estabilidade por filtro de `spf_fundo.f3_estabilidade` | `filter_diversity`, depois de F3 | ~4 h |
| S2, S4, S5 | cache de D, C e A no formato de `saliency.cache` (estender `topk_lote`) | `saliency/`; a grade já lê | cache ~60 h-CPU (37 s por imagem, ~6.000 imagens), ~1 h com 64 workers; depois 3,5 h por mapa |
| S3 | B já tem cache no Jaci; só YAML com `saliency.cache` | só YAML | 3,5 h |
| S6 | peso = estabilidade do mapa entre escalas (`fusion._stability`) | `saliency/` + chave em `saliency.pesos` | ~4 h |
| fundo F1-F4 | YAML de `eval/spf_fundo.py` com os bancos salvos | só YAML | ~2 h (só SVM limpo + perturbações) |
| R1, R5, R6, TTA | `amostra.leituras` no lugar da constante `READOUT`, mais entradas em `MODES` | `spf_grade` + `eval/l1_amostras.py` | +30% por leitura sobre um pacote: ~1,6 h cada |
| R2, R4 | modos `gmp` e `gem_p<p>` em `_encode_pooled` (arquivo compartilhado: só acrescentar) | `eval/svm.py` | ~1,6 h cada; R4 escolhe p em 2 splits e mede no 3º |
| R8 | GeM em 2 ou 3 escalas da imagem, concatenado | `spf_grade` | ~3 h |
| R9 | covariância de canais depois de PCA para 48 canais (1.176-d) | `spf_grade` | ~2 h |
| Scattering | módulo novo `eval/` que entrega o vetor para `score_full`; instalar `kymatio` | módulo novo | ~2 h de GPU; J=3, L=8, ordem 2 dá ~651-d |
| SoftHebb (C) | módulo novo, aprendizado local por camada | módulo novo | ~3 h |
| PCANet completo | hashing binário e histograma; o histograma passa de 2000-d sem pooling | módulo novo, "se viável" | ~3 h |
| profundidade 3-8, finalistas | YAML de `spfdeep_cedo` com 3 splits, 2 sorteios, só os finalistas | só YAML | ~3 h por braço (8 camadas) |
