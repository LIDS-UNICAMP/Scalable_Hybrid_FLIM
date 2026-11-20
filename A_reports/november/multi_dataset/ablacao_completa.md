# SPiFiL sem backprop: índice das ablações (até 08/11/2026, 20:30 UTC)

Fontes: `decisoes_spifil.md`, `results_log.md`, `fila.md`, `relatorio_07nov.md`, `relatorio_08nov.md`, `relatorio_07_08nov.md`, `sol_auditoria.md`, `agrupamento_auditoria.md`, `saliencia.md`, `spifil_vs_aleatorio.md` e `multiescala.md` (nesta pasta); `/exps/moliveira/reports/rodada2/{plano,auditoria,pool_foco,leis_p0_p1,hebb_4_1_auditoria}.md`; `/exps/moliveira/reports/prompt_*_resumo.md`; `/exps/moliveira/recebido_jaci/exps/reports/`; comentários do topo de `experiments/autoencoder/disc_md_*.yaml`. Nenhum número foi calculado aqui: só copiado das fontes.

## Regras e leitura

- **Só validação.** O teste nunca entra em escolha. (Única exceção: `l1fast`, lido no teste a pedido do usuário, só relato.)
- **Régua:** aleatório puro (`random_z`, `xavier`, `orth`, `tnorm`, `he`), sem nenhuma técnica, mesma arquitetura e mesma leitura. "Melhor aleatório" = o melhor deles na mesma base, split e largura. Referência de cima: DINOv2 giant.
- **κ** = kappa de Cohen (`val_kappa`). **F1** = F1 macro (`val_f1`). **acc** = acurácia balanceada (`val_acc`).
- **Splits se sobrepõem:** os 3 splits são reembaralhamentos (treinos com ~70% em comum). A unidade independente é a **família**: ovos, larvas, protozoários, aid + remote_sensing (455 imagens em comum), flowers. Subgroup e base completa contam como a mesma prova.
- Protocolo comum (salvo aviso): 200 px, 50 superpixels DISF, 50% dos dados, 500 imagens para o banco, kernel 5, ReLU, leitura GeM + PCA-whitening + SVM linear, sem TTA. `L3 cat` = 3 camadas lidas juntas; `L` = só a última.
- Abreviações de base: Esub/Psub/Lsub = Eggs/Protozoan/Larvae_subgroup; heggs/hlarv = helminth_eggs/larvae; pcysts = protozoan_cysts; rs = remote_sensing; coco = coconut_trees. "s42" = sorteio 42.

**Receita atual (08/11):** `fpsw@mapa_w0.5_1` (fps + peso da saliência `gmr`, corte 0,5), 6 camadas concatenadas, 256 filtros, whitening entre camadas, GeM + PCA-whitening + SVM. Sem objeto definido (aid, rs): `fps` sem saliência.

---

Detalhe de cada experimento em [`../ablacao/`](../ablacao/); cópias dos relatórios de fora do repositório em [`../ablacao/fontes/`](../ablacao/fontes/). A coluna de valores traz só κ (e F1 quando há); o resto está no detalhe.

## 1. Whitening entre camadas

Fontes: [`decisoes_spifil.md`](decisoes_spifil.md), [`relatorio_07nov.md`](relatorio_07nov.md), [`sol_auditoria.md`](sol_auditoria.md)

| experimento | bases | κ (e F1) | estado | decisão |
|---|---|---|---|---|
| [`spfcombo_w` (Jaci)](../ablacao/01_spfcombo_w.md) | Esub, Lsub, Psub, aid, flowers, heggs | 256, L3, sem → com: Esub 0,821→0,873; Lsub 0,613→0,769; Psub 0,813→0,859; aid 0,714→0,732; flowers 0,588→0,643; heggs 0,931→0,944. Contra melhor aleatório (L6, 256): Esub +0,095, Lsub +0,22, Psub +0,10, aid +0,041, flowers +0,006 | concluído | entra na receita |
| [`spfcombo_w2` (Jaci)](../ablacao/01_spfcombo_w2.md) | Eggs, Larvae, Protozoan, pcysts, hlarv, rs | L6 256: Eggs 0,845 (aleat. 0,681); Larvae 0,785 (0,533); Protozoan 0,849 (0,651); pcysts 0,920 (0,800); hlarv 0,966 (0,961); rs 0,506 (0,504) | concluído | passa do aleatório em 11 de 12 bases; rs empata; resolve Eggs (+0,16) |
| [`sol_wsal` (Athena)](../ablacao/01_sol_wsal.md) | 12 bases | 24 de 24 células acima do melhor aleatório em L3 e L6 (rs empata); `fps@combo` igual ao `spfcombo_w` até a 3ª casa; dicionário SIFT soma só +0,003 | concluído | base da receita única |
| [correção `cat_b`](../ablacao/01_correcao_cat_b.md) | — | `cat_b` não mede o whitening entre camadas | — | nota de leitura |

## 2. Saliência

Fontes: [`saliencia.md`](saliencia.md), [`decisoes_spifil.md`](decisoes_spifil.md), [`spifil_vs_aleatorio.md`](spifil_vs_aleatorio.md), [`fila.md`](fila.md)

| experimento | bases | κ (e F1) | estado | decisão |
|---|---|---|---|---|
| [Fase 1 (`saliency/batch.py`)](../ablacao/02_fase_1.md) | 9 bases | `gmr`: Q 0,725, estab. 0,898, 0,33 s, IoU 0,781; A: Q 0,737, 37 s | concluído | `gmr` fixo |
| [`spf_sal_fl`](../ablacao/02_spf_sal_fl.md) | flowers | 6 de 120 células passam, todas 64; `fpsw@mapa_peso` L2 cat Δ +0,053 [+0,031; +0,075]; `fps` sem mapa já dá +0,047 | concluído | mérito pequeno |
| [`spf_sal_par`](../ablacao/02_spf_sal_par.md) | Psub, Esub, Lsub, Larvae | ganho do mapa: Lsub +0,064; Larvae +0,041; Esub +0,013; Psub −0,001; máscara e peso empatam (+0,022 a +0,028) | concluído | só para larvas (na época) |
| [`fpsw@mapa_peso` (ranking)](../ablacao/02_fpsw_mapa_peso.md) | 5 bases | +0,174 (ranking); vs `fps`: +0,043 [+0,019; +0,067] (64), +0,015 [−0,013; +0,043] (256) | concluído | candidato |
| [Fusão Top-K (B)](../ablacao/02_fusao_top_k.md) | Lsub | IoU 0,657 vs 0,643; em heggs/pcysts perde | parado pelo usuário | descartado |
| [SIFT como mapa](../ablacao/02_sift_como_mapa.md) | 9 bases | perde do `gmr` em IoU nas 9; AP em 7–8; revocação 0,32 vs 0,78 | concluído | descartado (SIFT só para pontos) |
| [`salw_a` (Jaci)](../ablacao/02_salw_a.md) | flowers, Psub, Esub, Lsub, aid, Larvae | todos ganham do `fps` (+0,015 a +0,034); Lsub 64: 0,726 vs 0,582; nada em aid/flowers 256; foco vence em 11/12 células, peso em 8–9/12; t 0,25–0,5 melhor; otsu entre os piores | concluído | `fps@foco_w0.25_4` ou `fpsw@mapa_w0.5_1` |
| [`sol_wsal` (saliência)](../ablacao/02_sol_wsal.md) | 12 bases | +0,02 a +0,025 sobre `fps`; Lsub 64: 0,648→0,824; Larvae 256: 0,744→0,855; t = 0 ou 0,5, g = 1 | concluído | soma |
| [Tabela de arrependimento](../ablacao/02_tabela_de_arrependimento.md) | 12 bases | `fpsw@mapa_w0.5_1` L6 256: pior arrependimento −0,029; vs aleatório: ovos +0,199, larvas +0,290, protoz. +0,210, aid+rs +0,028, flowers +0,067; família de fora: mesma receita em 4 de 5 | concluído | receita única |
| [Com × sem (if/else)](../ablacao/02_com_sem.md) | 12 bases | aid −0,010/−0,014; rs +0,006/−0,015; flowers +0,022/+0,020; larvas +0,03 a +0,14; ovos +0,006 a +0,033; protoz. +0,04 com 64, −0,002 a −0,016 com 256 | concluído | regra do usuário: sem saliência em aid e rs |
| [`gauss` (Jaci)](../ablacao/02_gauss.md) | 12 bases | rodando | rodando | — |

## 3. Pontos (sementes)

Fontes: [`multiescala.md`](multiescala.md), [`spifil_vs_aleatorio.md`](spifil_vs_aleatorio.md), [`results_log.md`](results_log.md), [`decisoes_spifil.md`](decisoes_spifil.md)

| experimento | bases | κ (e F1) | estado | decisão |
|---|---|---|---|---|
| [`spf_r1` (candidatos)](../ablacao/03_spf_r1_candidatos.md) | flowers, Psub | ranking: `crus@obj` +0,126; `@centroid` +0,105; `@geodesic` +0,097; `@sift_nms` +0,097; `@sift_sp` +0,108 | concluído | diferenças pequenas |
| [`ms_b`](../ablacao/03_ms_b.md) | flowers, Psub, Esub, Lsub, aid | média Δ: centroide +0,077; SIFT +0,060; objeto +0,085; FPS +0,087 (5 de 5 bases > 0) | concluído | FPS o mais estável |
| [`ms_b2` (Jaci)](../ablacao/03_ms_b2.md) | Eggs, Protozoan, Larvae, pcysts, rs, heggs, hlarv | empata com o geodésico | concluído | fora da receita |
| [`fps@sift`](../ablacao/03_fps_sift.md) | várias | `crus@sift` +0,087 (64, 7 bases); detecção 36% vs 25% nas larvas; entre os 5 do `r3` | concluído | alternativa ao medoide |
| [Sementes dentro/fora](../ablacao/03_sementes_dentro_fora.md) | 9 bases | medoide 25–49% no objeto (2,4–4× o acaso); com corte de saliência 70–98%, mas 9–23 pontos por imagem | concluído | diagnóstico |
| [`ms_a`](../ablacao/03_ms_a.md) | flowers, Psub, Esub, Lsub, aid | média Δ: S8 +0,077; S5 +0,080; geodésico +0,066; S4 −0,024 (pior); Gauss = cru | concluído | empate com geodésico |
| [`ms_a2` (Jaci)](../ablacao/03_ms_a2.md) | Eggs, Protozoan, Larvae, pcysts, rs, heggs, hlarv | κ médio: S8 0,714; S5 0,710; geodésico 0,704; S4 0,623 | concluído | multiescala fora |
| [Superpixels L1](../ablacao/03_superpixels_l1.md) | 7 bases | heggs 0,744 (800) vs 0,842 (50); pcysts 0,589 vs 0,743; 50 fixado pelo usuário (custa 0,029 na média) | parcial (28 de 117) | 50 DISF |

## 4. Normalizações

Fontes: [`sol_auditoria.md`](sol_auditoria.md), [`decisoes_spifil.md`](decisoes_spifil.md), [`prompt_sol_resumo.md`](../ablacao/fontes/prompt_sol_resumo.md)

| experimento | bases | κ (e F1) | estado | decisão |
|---|---|---|---|---|
| [N0](../ablacao/04_n0.md) | — | — | concluído | fica |
| [N1 (`cl_ctl`)](../ablacao/04_n1.md) | 5 bases | aleatório com N1 − com N0: −0,197 (−0,10 flowers a −0,33 Esub) | concluído | descartado |
| [`sol_p1`](../ablacao/04_sol_p1.md) | Esub, Psub, Lsub, aid, flowers, heggs | vs N0: N2 −0,089 (4/36 células); N3 −0,003 (12/36); N7 −0,236 (0/36); N2 derruba larvas −0,29 | concluído | N0 fica |
| [`sol_p1_n6` (Jaci)](../ablacao/04_sol_p1_n6.md) | idem | κ cai em todas as células: Esub 64 `fps` 0,694→0,473; Psub 256 0,819→0,644; aid 256 0,705→0,609; heggs 256 0,929→0,874 | concluído | descartado |

## 5. Agrupamento

Fontes: [`agrupamento_auditoria.md`](agrupamento_auditoria.md), [`decisoes_spifil.md`](decisoes_spifil.md), [`prompt_clustering_resumo.md`](../ablacao/fontes/prompt_clustering_resumo.md)

| experimento | bases | κ (e F1) | estado | decisão |
|---|---|---|---|---|
| [`spf_r1`](../ablacao/05_spf_r1_kmeans.md) | flowers, Psub | empata com patch sorteado (+0,004); vs `fps`: −0,028 (64), −0,044 (256); 53% de patches planos vs 2% | concluído | descartado |
| [`spf_r1`](../ablacao/05_spf_r1_k_automatico.md) | flowers, Psub | `km_sil` fica em k = 8 (+0,073 ranking); `hdbscan` −0,135 | concluído | descartado |
| [`spf_r1`](../ablacao/05_spf_r1_bovw_sift.md) | flowers, Psub | `fpsd@sift` − `fps@sift` = −0,031 (256); `bovw_w` +0,046 ranking | concluído | descritor não ajuda |
| [`cl_a`](../ablacao/05_cl_a.md) | flowers, Psub, Esub, Lsub, aid | melhor: Esub 0,482, Lsub 0,438, Psub 0,522, aid 0,506, flowers 0,479; vs aleatório comum e geodésico: −0,09 a −0,12 | concluído | descartado nesta forma |
| [(não testado)](../ablacao/05_agrupamento_zscore_comum.md) | — | — | proposto | se valer o custo |

## 6. Diversidade de filtros e seleção

Fontes: [`sol_auditoria.md`](sol_auditoria.md), [`rodada2_sol_p3.md`](../ablacao/fontes/rodada2_sol_p3.md), [`auditoria.md`](../ablacao/fontes/auditoria.md), [`auditoria_rodada2_jaci.md`](../ablacao/fontes/auditoria_rodada2_jaci.md), [`results_log.md`](results_log.md)

| experimento | bases | κ (e F1) | estado | decisão |
|---|---|---|---|---|
| [`spf_r1`](../ablacao/06_spf_r1_fps.md) | flowers, Psub (+ `r3`) | +0,071 (64) e +0,072 (256) em 7 bases, pior −0,017 | concluído | entra na receita |
| [`spf_r1`](../ablacao/06_spf_r1_cos_qr_crus.md) | idem | `cos` +0,108; `qr` +0,061; `crus` +0,113 (0 bases boas) | concluído | `fps` é mais estável |
| [`amostra_stack_flowers`](../ablacao/06_amostra_stack_flowers.md) | flowers | +0,028 (64 L1); +0,031 (64 L3); +0,044 (128 L3); +0,024 (256 L3) | parcial | não levado à grade |
| [`spf_r1`](../ablacao/06_spf_r1_spifil_original.md) | flowers, Psub | `spifil` +0,099 ranking, +0,037 (64), pior −0,072; `spifil_dd` +0,027 | concluído | fora (usa rótulo) |
| [`sol_p3` (Jaci)](../ablacao/06_sol_p3.md) | Esub, Lsub, Psub, aid, flowers, heggs | f − fps em 72 células: f2 −0,001; f3 +0,001; f4 −0,023; f5 −0,004; f6 −0,051. F1: f3 +0,007, f5 +0,004. aid f3 +0,017/+0,028. K-medoids não ganha e custa 3,5× | concluído | não substitui `fps`; f3 sinal em aid |
| [K1 dedup (treino com backprop)](../ablacao/06_k1_dedup.md) | pcysts, heggs, hlarv | pcysts +0,118; heggs +0,028; hlarv −0,010 | concluído | regra candidata (fora do regime sem backprop) |

## 7. PCA/ZCA nos filtros

Fontes: [`sol_auditoria.md`](sol_auditoria.md), [`spifil_vs_aleatorio.md`](spifil_vs_aleatorio.md)

| experimento | bases | κ (e F1) | estado | decisão |
|---|---|---|---|---|
| [`spf_r1`/`r2`, `spf_h`](../ablacao/07_zca2.md) | 2 a 7 bases | `zca2` −0,109 (64) e −0,126 (256), 0–1 de 7 bases; `zca2` − `crus` −0,115 | concluído | descartado |
| [`spf_r1`](../ablacao/07_spf_r1_pca.md) | flowers, Psub | −0,060; −0,025; −0,030 | concluído | descartado |
| [`spf_r1`](../ablacao/07_spf_r1_zca_km2.md) | flowers, Psub | +0,015 | concluído | descartado |

## 8. Leituras

Fontes: [`decisoes_spifil.md`](decisoes_spifil.md), [`plano.md`](../ablacao/fontes/plano.md), [`pool_foco.md`](../ablacao/fontes/pool_foco.md), [`auditoria.md`](../ablacao/fontes/auditoria.md)

| experimento | bases | κ (e F1) | estado | decisão |
|---|---|---|---|---|
| [`spfdeep_*`](../ablacao/08_spfdeep.md) | 6 bases | `fps` 64 `cat` L8 +0,094 vs `L` +0,035; só a última cai com a profundidade | concluído | `cat` na receita |
| [`spfcombo_a`](../ablacao/08_spfcombo_a.md) | 6 bases | Δ maior só porque o aleatório piora | concluído | descartado |
| [`sol_p5`](../ablacao/08_sol_p5.md) | Esub, Psub, Lsub, aid, flowers, heggs | média 6 bases (`fpsw@mapa_w0_1`): `cat` 0,770 (256)/0,710 (64); `cat_ms` 0,783/0,739; GeM p fixo empata; GAP/GMP 0,65–0,72; SPP piora (>1300-d); `L_cov` 0,706→0,755 (256), 0,655→0,762 (64) | concluído | `cat_ms` candidata; GAP, GMP, SPP fora |
| [`leitura_a2`](../ablacao/08_leitura_a2.md) | Esub, Lsub, Psub, aid, flowers, rs | `cat_ms_eq` − `cat`, fps 256: +0,009, −0,013, +0,029, +0,020, −0,002, +0,030 (4/6 bases, 3/5 famílias); fps 64: 5/6 bases; A1 ±0,03 sem direção | concluído | **passa o critério** (escala); GeM p2 encerrado |
| [`vlad_off`](../ablacao/08_vlad_off.md) | aid, rs | aid: `cat` 0,776, vlad32 0,669, L 0,630; rs: vlad32 0,462, `cat` 0,456 | concluído | ganha do GeM da última, não do `cat` |
| [`vlad_on`](../ablacao/08_vlad_on.md) | flowers, coco | coco: L 0,788, vlad32 0,689, `cat` 0,630; flowers: `cat` 0,630, vlad 0,604 | concluído | VLAD encerrado |
| [`L` vs `cat` em coco (`vlad_on`, `pf_res128_on`)](../ablacao/08_l_vs_cat_em_coco.md) | coco | 128 px: L 0,792 (256) vs `cat` 0,639; aleatório L 0,511 | concluído | em aberto: leitura por base |
| [A5 (`sol_p2`, leitura `pol`)](../ablacao/08_a5.md) | 6 bases | — (não separado no relatório) | concluído | ver polaridade |
| [`cov` com `cat`](../ablacao/08_cov_com_cat.md) | — | — | não existe | — |

## 9. Largura, profundidade, kernel e tamanho da entrada

Fontes: [`decisoes_spifil.md`](decisoes_spifil.md), [`spifil_vs_aleatorio.md`](spifil_vs_aleatorio.md), [`pool_foco.md`](../ablacao/fontes/pool_foco.md), [`results_log.md`](results_log.md), [`law_size_superpixels.md`](law_size_superpixels.md)

| experimento | bases | κ (e F1) | estado | decisão |
|---|---|---|---|---|
| [`spf_r1`](../ablacao/09_spf_r1_largura.md) | flowers, Psub | receita provisória `fps` L3: +0,064 (256), +0,056 (64) | concluído | sobreviventes → `r2` |
| [`spf_r2`](../ablacao/09_spf_r2.md) | flowers, Psub | mais filtros não resolvem flowers | concluído | >256 encerrado |
| [`spf_r3` (Jaci)](../ablacao/09_spf_r3.md) | flowers, Psub, heggs, hlarv, aid | `fps@sift` entre os 5 | concluído | → `r4` |
| [`spf_r4` (Jaci)](../ablacao/09_spf_r4.md) | mesmas 5 | kernel 3 e entrada 200 ganham ou empatam em 4 e 5 de 5; largura cresce monotônica | parcial (58/60) | kernel 3–5 |
| [`spf_r5` (Jaci)](../ablacao/09_spf_r5.md) | Esub, Lsub, Larvae, Eggs, Protozoan, pcysts, rs | rodando | rodando/na fila | — |
| [`spfdeep_cedo`](../ablacao/09_spfdeep_cedo.md) | Psub, flowers, Esub, Lsub, heggs, aid | `fps@sift` 4/6 boas (+0,088); `fps` 64 `cat` sobe até L8 (+0,094); `fps` 256 `cat` pico L3 (+0,052) | concluído | ir fundo com `cat` |
| [`spfdeep_esp` (Jaci)](../ablacao/09_spfdeep_esp.md) | idem | L8 `cat`: `fps` +0,235 (Lsub 64), +0,098 (Psub 64); `fps@sift` +0,097 (Esub 64) | concluído | idem |
| [Tamanho × arquitetura (só leitura)](../ablacao/09_tamanho_arquitetura.md) | 12 + coco | 256 vence em 12 de 12 (Protozoan, 64 px, +0,118); ρ tamanho × largura +0,41 (p 0,17); ganho L6−L3 vs nº de treino ρ +0,67 (p 0,02) | concluído | 256 e L6; pouco treino → L3–L4 |
| [`largura` (Jaci)](../ablacao/09_largura.md) | coco, Protozoan | rodando | rodando | — |
| [coco 64 vs 256](../ablacao/09_coco_64_vs_256.md) | coco | 0,785 (64) vs 0,667 (256); explicado pelo `cat`, não pela largura | concluído | ver tema 8 |
| [`pf_res128_on`](../ablacao/09_pf_res128_on.md) | coco | igual a 200 px (L 0,792 vs 0,788) | concluído | 128 = 200 |
| [`pf_res320_off`](../ablacao/09_pf_res320_off.md) | aid, rs | aid 0,801/0,794 vs 0,786 (+0,015); rs 0,533/0,486 vs 0,432 (+0,054 a +0,10); aleatório 0,692–0,700 e 0,449–0,481 | concluído (piloto) | promissor; falta 3 splits |
| [`pf_res448_off`](../ablacao/09_pf_res448_off.md) | aid, rs | — | na fila (Athena) | — |
| [`pf_res320_conf` (Jaci)](../ablacao/09_pf_res320_conf_jaci.md) | aid, rs | — (splits 2 e 3 × s42/43, 8 jobs) | na fila (Jaci, na frente) | confirma ou não o piloto de 320 px |
| [Lei tamanho × superpixels (L1)](../ablacao/09_lei_tamanho_superpixels.md) | 12 | log2 n* = 4,01 + 0,18 log2 S, R² 0,013 | concluído | sem lei única |
| [`l1fast`](../ablacao/09_l1fast.md) | várias | κ médio 0,643 (64 px), 0,645 (128), 0,569 (640) | concluído | só relato |

## 10. Leis de construção (largura por camada)

Fontes: [`leis_p0_p1.md`](../ablacao/fontes/leis_p0_p1.md), [`plano.md`](../ablacao/fontes/plano.md), [`relatorio_07_08nov.md`](relatorio_07_08nov.md), [`fila.md`](fila.md)

| experimento | bases | κ (e F1) | estado | decisão |
|---|---|---|---|---|
| [`leis_p1` (Athena)](../ablacao/10_leis_p1.md) | aid, flowers, rs | aid: A 512 r0,71 0,788 vs âncora 0,786 (1,5× params); flowers 0,659 vs 0,656; rs: as 17 passam a âncora 0,432 (melhor C 0,492), mas o aleatório sobe junto | concluído | sem ganho com mesmo orçamento |
| [`leis_p1_s43`](../ablacao/10_leis_p1_s43.md) | aid, flowers, rs | Spearman 42 vs 43: aid 0,94; flowers 0,64; rs −0,07; A 512 r0,71 em 1º (aid 0,786, flowers 0,644) | concluído | ordem estável só em aid |
| [`leis_p1_jaci`](../ablacao/10_leis_p1_jaci.md) | Larvae, Eggs | rodando | rodando (Jaci) | — |
| [`leis_p1b_jaci`](../ablacao/10_leis_p1b_jaci.md) | Lsub, Esub, heggs, hlarv | `cat`: receita ≥ `fps` (+0,007 a +0,032), +0,15 a +0,39 sobre aleatório; leis ±0,03; L: A r0,71 melhor em 3/4 (Esub 0,865 vs 0,739 com r0,5) | concluído | começar largo, reduzir devagar |
| [`leis_p1b` (Athena)](../ablacao/10_leis_p1b.md) | Protozoan, Psub, pcysts | rodando | rodando (Athena) | — |
| [`leis_p1c_s43` (Athena)](../ablacao/10_leis_p1c_s43.md) | Protozoan, Psub, pcysts | rodando | rodando (Athena) | — |
| [`leis_p1b_s43_jaci`, `leis_p1_s43_jaci`](../ablacao/10_leis_p1b_s43_jaci_leis_p1_s43_jaci.md) | Lsub, Esub, heggs, hlarv; Larvae, Eggs | — | na fila (Jaci, fila3) / rodando | — |
| [`pf_leis_off`](../ablacao/10_pf_leis_off.md) | aid, rs | — | na fila (Jaci, fila3) | — |
| [`pf_leis_on`](../ablacao/10_pf_leis_on.md) | flowers, coco | — | na fila (Jaci, fila3) | — |
| [Lei D (por dados)](../ablacao/10_lei_d.md) | — | — | proposto | — |

## 11. Polaridade e ativação

Fontes: [`decisoes_spifil.md`](decisoes_spifil.md), [`sol_auditoria.md`](sol_auditoria.md)

| experimento | bases | κ (e F1) | estado | decisão |
|---|---|---|---|---|
| [`rz2`, `whrand2`](../ablacao/11_rz2_whrand2.md) | — | `rz2` − `random_z` −0,001; `whrand2` − `whrand` −0,009 | concluído | sem efeito |
| [`sol_p2` `abs`](../ablacao/11_sol_p2_abs.md) | Esub, Psub, Lsub, aid, flowers, heggs | perde: `fps` 256 Lsub 0,698→0,500; Psub 0,813→0,710 | concluído | descartado |
| [`sol_p2` `leaky` 0,1](../ablacao/11_sol_p2_leaky_0_1.md) | idem | empata; só Lsub 64 `fpsw` 0,617→0,657 | concluído | descartado |
| [`sol_p2` `crelu`](../ablacao/11_sol_p2_crelu.md) | idem | `fps` 256→512 canais +0,007 a +0,028 (Psub 0,813→0,841; aid 0,704→0,724), mas aleatório também sobe (aid 0,742, flowers 0,644) | concluído | ganho por dimensão |
| [`sol_p2` `crelu_h`](../ablacao/11_sol_p2_crelu_h.md) | idem | igual ou abaixo do `fps` | concluído | ReLU fica |
| [`gelu` (Athena)](../ablacao/11_gelu.md) | Larvae, Eggs, Protozoan, aid, flowers, rs | rodando | rodando | — |

## 12. SoftHebb

Fontes: [`decisoes_spifil.md`](decisoes_spifil.md), [`results_log.md`](results_log.md), [`hebb_4_1_auditoria.md`](../ablacao/fontes/hebb_4_1_auditoria.md), [`plano.md`](../ablacao/fontes/plano.md), [`pool_foco.md`](../ablacao/fontes/pool_foco.md), [`prompt_hebb_resumo.md`](../ablacao/fontes/prompt_hebb_resumo.md)

| experimento | bases | κ (e F1) | estado | decisão |
|---|---|---|---|---|
| [`sol_p6` (braço SoftHebb)](../ablacao/12_sol_p6.md) | 12 bases | Eggs 0,742; Esub 0,803; Psub 0,774; aid 0,690; flowers 0,590 | concluído | ver tema 13 |
| [`sol_hebb1` (Fila 1, E00–E03)](../ablacao/12_sol_hebb1.md) | 12 bases | `fps` 256 L3: A 0,732; C30 0,708; C90 0,703; C300 0,704; restrito 0,721; A ≥ C em 9 de 12 | concluído | não melhora (implementação antiga) |
| [`sol_hebb2_piloto` (Fila 2)](../ablacao/12_sol_hebb2_piloto.md) | flowers, aid, rs, Esub, Lsub, Psub | banco fica redundante (vizinho 0,77–0,84); E04 random 64 L3 0,519→0,597 (puxado por Lsub); E05 quase não age | concluído | não resolve |
| [`sol_hebb2` (Jaci)](../ablacao/12_sol_hebb2.md) | 12 bases | — | pausado (19 de 36) | sem prioridade |
| [Auditoria 4.1](../ablacao/12_auditoria_4_1.md) | — | regra igual (5,6e-17); diferenças: inv_temp 4 vs 1/0,65, passo por neurônio vs por tensor, lr igual nas camadas, norma inicial √2 vs ~25; efeito vem do peso, não da BN (26% vs 1,6%) | concluído | receita dos autores → `hebb3` |
| [`sol_hebb3` (piloto)](../ablacao/12_sol_hebb3.md) | Eggs, Larvae, aid, flowers, rs | H05−H01 `cat`: 64 κ 0,585→0,625, F1 0,663→0,690, acc 0,661→0,694; 256 κ 0,642→0,673, F1 0,699→0,721, acc 0,690→0,711; Larvae 256 −0,057; H03 ≈ H01 | concluído | passa em `gem_pw` → confirmação |
| [`sol_hebb3_conf`](../ablacao/12_sol_hebb3_conf.md) | Eggs, Larvae, aid, flowers, rs | 256 `cat`: Eggs +0,026, Larvae +0,052, aid +0,064, flowers +0,036, rs +0,026; F1 +0,021 a +0,060; acc +0,020 a +0,059; 64: só aid +0,073 e rs +0,053 | concluído | **confirmado** (256, 3 camadas, sem whitening nem saliência) |
| [`pf_hebb_d3_off`](../ablacao/12_pf_hebb_d3_off.md) | aid, rs | — | rodando (Jaci) | — |
| [`pf_hebb_d3_on`](../ablacao/12_pf_hebb_d3_on.md) | flowers, coco | — | na fila (Athena) | — |
| [`pf_hebb_d4_off`, `pf_hebb_d4_on`](../ablacao/12_pf_hebb_d4_off_pf_hebb_d4_on.md) | aid, rs; flowers, coco | — | na fila (Athena) | — |
| [`pf_hebb_rand_d3`](../ablacao/12_pf_hebb_rand_d3.md) | aid, rs, flowers, coco | — | rodando (Athena) | — |
| [`pf_hebb_rand_d4`](../ablacao/12_pf_hebb_rand_d4.md) | idem | — | na fila (Athena) | — |
| [SoftHebb na receita completa](../ablacao/12_softhebb_na_receita_completa.md) | — | — | proposto | — |

## 13. Encoders alternativos

Fontes: [`results_log.md`](results_log.md), [`auditoria.md`](../ablacao/fontes/auditoria.md), [`sol_auditoria.md`](sol_auditoria.md)

| experimento | bases | κ (e F1) | estado | decisão |
|---|---|---|---|---|
| [`sol_p6` Scattering](../ablacao/13_sol_p6_scattering.md) | 12 bases | Eggs 0,739; heggs 0,957; hlarv 0,980; pcysts 0,926; Larvae 0,704; aid 0,616; flowers 0,555; rs 0,421 | concluído | forte em fundo preto; SPiFiL ganha em 6 de 9 |
| [`sol_p6` PCANet](../ablacao/13_sol_p6_pcanet.md) | 12 bases | Lsub 0,586; Esub 0,752; rs 0,450; abaixo nas demais | concluído | descartado |
| [D2 FPS + Scattering](../ablacao/13_d2_fps_scattering.md) | — | — | não existe | proposto |

## 14. Fundo e detecção

Fontes: [`results_log.md`](results_log.md), [`spifil_vs_aleatorio.md`](spifil_vs_aleatorio.md), [`sol_auditoria.md`](sol_auditoria.md), [`prompt_multiescala_resumo.md`](../ablacao/fontes/prompt_multiescala_resumo.md)

| experimento | bases | κ (e F1) | estado | decisão |
|---|---|---|---|---|
| [`spf_h` (+ `spf_h3`)](../ablacao/14_spf_h.md) | Psub, Esub, Lsub, Larvae | só fundo até −0,13 (IC < 0 em 7 de 8); só objeto até +0,066, nenhum IC > 0 | concluído | fundo atrapalha; objeto sozinho não basta |
| [`spf_h_raw`](../ablacao/14_spf_h_raw.md) | Eggs, Protozoan | — (entra nas tabelas de `spf_h`) | concluído | — |
| [`crus@fundo`](../ablacao/14_crus_fundo.md) | 7 bases | −0,058 (ranking); −0,074 | concluído | descartado |
| [`spf_hz` (Jaci)](../ablacao/14_spf_hz.md) | Psub, Esub, Lsub, Larvae, Eggs, Protozoan | `crus` +0,086 (teria +0,226); `spifil` +0,082 (+0,216): ~38% do caminho; larvas 46–67%; ovos 1–18% | concluído | detectar daria +0,14 de κ |
| [Detecção 12 bases (`spf_deteccao`)](../ablacao/14_deteccao_12_bases.md) | 9 com máscara | `fpsw@mapa_peso` acende no objeto em 6 de 6 bases de fundo real (1,3–3,4× a área) | concluído (sinal) | saliência guia a detecção |
| [P4 fundo F1–F4 (`spf_fundo`)](../ablacao/14_p4_fundo_f1f4.md) | 9 bases | κ retido, cor sólida: `fpsw` 0,99, `fps` 0,83, `random_z` 0,59; textura: 0,64 / 0,37 / 0,29; energia/área: 2,11 / 1,07 / 0,83 | concluído (2 splits) | `fpsw@mapa_peso` o mais robusto |
| [HAM10000 (saliência na lesão)](../ablacao/14_ham10000.md) | — | saliência concentra energia na lesão, κ não muda | concluído | — |

## 15. Imagens por classe

Fontes: [`results_log.md`](results_log.md), [`fila.md`](fila.md)

| experimento | bases | κ (e F1) | estado | decisão |
|---|---|---|---|---|
| [`amostra_l1_*`, `amostra_stack_*`, `amostra_banco_*`, `amostra_imb_*`](../ablacao/15_amostra_l1_amostra_stack_amostra_banco_amostra_imb.md) | aid, flowers, hlarv | "mais imagens por classe" ficou de fora (07/11) | concluído | — |
| [`porclasse_piloto`](../ablacao/15_porclasse_piloto.md) | flowers | B: 0,647/0,626/0,659/0,689/0,656; A: 0,619/0,653/0,652/0,680/0,677; controle 0,601–0,610 | concluído | B e A trocam de lugar; SPiFiL +0,04 a +0,09 sobre controle |
| [`porclasse` (Athena)](../ablacao/15_porclasse.md) | 8 bases | — (8 de 24 jobs) | parado (sem teto de GPU) | movido para o Jaci |
| [`porclasse_jaci`](../ablacao/15_porclasse_jaci.md) | flowers, Lsub, rs, Esub, Psub, aid, coco | — | rodando (Jaci) | — |
| [`porclasse_food_jaci`](../ablacao/15_porclasse_food_jaci.md) | food101 | — | pausado (food101) | — |

## 16. Bases novas

Fontes: [`fila.md`](fila.md), [`decisoes_spifil.md`](decisoes_spifil.md), [`relatorio_07_08nov.md`](relatorio_07_08nov.md), [`pool_foco.md`](../ablacao/fontes/pool_foco.md)

| experimento | bases | κ (e F1) | estado | decisão |
|---|---|---|---|---|
| [`ext_piloto`](../ablacao/16_ext_piloto.md) | coco, food101 | coco: 0,740 vs 0,675; food101: 0,17 vs 0,15, 3 h 23 min por job | concluído | coco entra; food101 caro |
| [`*_ext` (18 YAMLs)](../ablacao/16_ext.md) | coco | coco L6: 64 0,785 vs 256 0,667; melhor 0,792 (L, 1 split); aleatório 0,511 | parcial | ver temas 8 e 9 |
| [`*_extf_jaci`](../ablacao/16_extf_jaci.md) | food101 | — | pausado (16 experimentos guardados) | base grande demais |
| [`disc_md_ham_jaci`](../ablacao/16_disc_md_ham_jaci.md) | HAM10000 | L6 receita κ 0,461 ± 0,009, `fps` 0,468 ± 0,021 (pareado −0,007 ± 0,026); melhor aleatório 0,340; parcial: F1 0,518/0,532, acc 0,485/0,489 | concluído | empate; SPiFiL +0,12 sobre aleatório |

## 17. Benchmark

Fontes: [`relatorio_07_08nov.md`](relatorio_07_08nov.md), [`results_log.md`](results_log.md)

| experimento | bases | κ (e F1) | estado | decisão |
|---|---|---|---|---|
| [Benchmark de modelos prontos](../ablacao/17_benchmark_de_modelos_prontos.md) | — | DINOv2 já medido (ex.: flowers 0,973, rs 0,708) | na fila (por último) | — |
| [Benchmark da fase 1 (4 modelos)](../ablacao/17_benchmark_fase1_4_modelos.md) | 12 bases | DINOv2 giant 0,939 (Protozoan) a 0,994 (hlarv), rs 0,708; AIMv2 3B aid 0,981, flowers 0,978 | concluído (DINOv3 pendente de login) | régua de cima |

A lista dos 18 modelos (12 no Athena, 6 no Jaci) e a agenda estão no detail do benchmark.

## 18. Fase com backprop (05–07/11, pausada)

Fontes: [`results_log.md`](results_log.md), [`fila.md`](fila.md), [`pausa_treinos.md`](pausa_treinos.md), [`CONTEXT.md`](CONTEXT.md), `/exps/moliveira/reports/multi_dataset/{plano,agentes}/`. Treinos com backprop (runner `growth`), parados em 07/11 12:00 UTC porque o SPiFiL não saía na frente do aleatório. Retomada: `pausa_treinos.md`.

| experimento | bases | κ (e F1) | estado | decisão |
|---|---|---|---|---|
| [Estágio 1: X1, X2, X3](../ablacao/18_estagio1_x1_x2_x3.md) | 12 bases | ver Superpixels L1 (28 de 117) | pausado | 50 superpixels (usuário) |
| [`res_*` (ablação de tamanho)](../ablacao/18_res_ablacao_tamanho.md) | várias | — | pausado | — |
| [`sc_*`, `scbig_*` (lei de escala)](../ablacao/18_sc_lei_de_escala.md) | 6 famílias | réplicas w1 heggs L3: média 0,870, desvio 0,016 | parado (piloto) | fora da lei |
| [`base_*` (linha de base)](../ablacao/18_base_linha_de_base.md) | 12 bases | — | pausado | — |
| [`law_f0` (ruído)](../ablacao/18_law_f0_ruido.md) | flowers, heggs, hlarv | desvio entre sementes 0,026 / 0,004 / 0,016 | concluído | decisão pendente |
| [`law_f1` degrau 1 (perdas)](../ablacao/18_law_f1_degrau1_perdas.md) | flowers, heggs, hlarv | UR-JEPA 0,01 +0,024 sobre só reconstrução (0,796) | concluído | 5 perdas → degrau 2 |
| [`law_f1d2` e `law_ctl_rand`](../ablacao/18_law_f1d2_degrau2_e_controle.md) | flowers, heggs, hlarv, Eggs | aleatório − SPiFiL em B0 +0,010 (erro 0,009) | parcial | motivo da pausa |


---

## O que está rodando ou na fila (08/11, 20:30 UTC)

| máquina | rodando | na fila |
|---|---|---|
| Athena | `leis_p1b` e `leis_p1c_s43` (protozoários, s42 e s43); `gelu`; `pf_res320_off` (já com resultado do piloto); `pf_hebb_rand_d3`; coco nos `_ext` restantes | fila automática: `pf_hebb_d3_on`, `pf_hebb_d4_off`, `pf_hebb_d4_on`, `pf_hebb_rand_d4`, `pf_res448_off`; depois benchmark (12 modelos) |
| Jaci | `porclasse_jaci` (7 bases); `gauss`; `largura` (coco × Protozoan); `leis_p1_jaci` e bases menores (s42 e s43); `pf_hebb_d3_off`; `spf_r4` → `spf_r5` | fila3: leis das bases menores s43; `pf_leis_off`, `pf_leis_on`; depois benchmark (6 modelos) |
| Parados de propósito | food101 (16 experimentos), `sol_hebb2` (19 de 36), `porclasse` do Athena | — |

## Veredito

**Confirmado**
- Whitening entre camadas (maior ganho; 11 de 12 bases acima do aleatório).
- Saliência `gmr` como peso ou foco, nas bases com objeto (larvas, ovos, flowers).
- `fps` para escolher filtros.
- 6 camadas concatenadas (`cat`), 256 filtros.
- Leitura multiescala com dimensão igual (`cat_ms_eq`, passou no limite).
- SoftHebb com a receita dos autores (rede de 3 camadas, 256 filtros).
- Receita única `fpsw@mapa_w0.5_1` (pior arrependimento −0,029).

**Descartado**
- Normalizações N1, N2, N3, N6, N7.
- Agrupamento (K-means, HDBSCAN, k automático, grafo, híbrido, BoVW).
- ZCA e PCA nos filtros; PCANet.
- Multiescala dos pontos (S1–S8) e DoG como filtro.
- Densidade SIFT e fusão Top-K como mapa.
- Diversidade funcional (f2–f6, K-medoids).
- Polaridade (abs, leaky, crelu com mesmo orçamento).
- GAP, GMP, SPP, VLAD, soma de camadas, GeM p2.
- Mais de 256 filtros; SoftHebb antigo; homeostase; banco só de fundo.

**Em aberto**
- GELU; saliência gaussiana; imagens por classe.
- Resolução 320 px (piloto bom em aid e rs) e 448 px.
- Leitura por base (última camada vs `cat`, caso coco).
- Leis nos parasitos e com `cat_ms_eq`; lei D.
- SoftHebb em 512, em 4 camadas, com saliência e dentro da receita completa.
- `r5`; largura coco × Protozoan; FPS + Scattering; agrupamento com z-score comum.
- Benchmark de 12 + 6 modelos.

---

## Atualizações depois das 20:30 UTC (08/11)

- `pf_hebb_rand_d3` terminou no Athena às 21:02, sem falhas; ainda sem tabela, porque `pf_hebb_d3_off` (Jaci, 1 de 2 jobs) e `pf_hebb_d3_on` (Athena) estão rodando.
- `sol_hebb2` (SoftHebb Fila 2) foi relançado no Jaci às 21:11: pula os 19 jobs prontos e roda os 17 que faltam.
- `pf_res320_conf` (Jaci) está na frente da fila: 8 jobs, splits 2 e 3, entra quando a GPU baixar de 77 GB.
- O sorteio 43 das leis das bases menores entrou às 20:29 e foi derrubado às 20:43 (GPU do Jaci a 95 GB); voltou ao começo da fila.
- Andamento sem número: gaussiana 20 de 72; imagens por classe 6 de 21; largura 2 de 12 (20:42).
- Jaci: o disco `/` (437 GB) está 100% cheio desde as 21:13 e cresce ~1,4 GB por hora. Não parece ser nosso. Os experimentos seguem.

## Regras e decisões do usuário

Fontes: mensagens do usuário na conversa (05–08/11), [`CONTEXT.md`](CONTEXT.md), [`fila.md`](fila.md), [`relatorio_07_08nov.md`](relatorio_07_08nov.md), [`plano.md`](../ablacao/fontes/plano.md), [`pool_foco.md`](../ablacao/fontes/pool_foco.md). A data é a da mensagem.

**Máquinas e recursos**
- **Athena, GPU:** 15 GB nossos por GPU, nas 4 GPUs (07/11 23:06). O uso dos outros usuários não entra na conta: se alguém usa 8 GB, o máximo da placa fica em 23 GB (08/11 18:27). Picos curtos acima disso são tolerados. Nunca quebrar o job de outro usuário. Antes disso valeram 40 GB (05/11), a GPU inteira sem quebrar o outro usuário (06/11) e 35 GB (07/11 01:03). A máquina não deve ficar ociosa (08/11 18:27).
- **Jaci:** usar tudo, "sem restrição" (07/11 21:52). Só se deixa 4 GB de GPU para o root, RAM livre ≥ 15 GB e load ≤ 115. Rodar no Jaci primeiro; no Athena, só o que não cabe lá (08/11 01:34). Um vigia no Jaci mantém CPU e GPU perto do máximo (08/11 18:27).
- **Disco:** tudo vai para `/exps/moliveira` (pesos, resultados, artifacts); nada no `/` (07/11 14:41).
- **rsync:** o Claude sincroniza sozinho pelo SSH athena→jaci, cuidando do espaço; cada rsync é contado em `/exps/moliveira/queues/rsync_contador.log`, com o comando (08/11 14:48 e 16:45).

**Como os experimentos são feitos**
- **Experimento novo = YAML** no padrão do repositório (`experiments/autoencoder/disc_md_*.yaml`, gêmeo `_jaci`), lançado por `experiments.ray.launch`. Nunca flag de argparse.
- **Filas automáticas:** pode lançar sem perguntar (05/11); não deixar nada esperando se houver espaço (07/11 21:05); a fila puxa o próximo sozinha.
- **Benchmark por último**, nas duas máquinas, depois da fila do SPiFiL (08/11 18:15). Pesos oficiais do Hugging Face (06/11 01:24).
- **Teste nunca entra em escolha:** só treino e validação. Única exceção: o relato no teste que o usuário pediu (`l1fast`, 06/11 04:13).
- **Máscaras só para avaliar:** nada de máscara no SPiFiL, nos superpixels, na loss ou na augmentation (05/11 22:28). Usar a pasta `raw` (05/11 22:26).
- **Pacote SPiFiL intocado:** importar a função e trabalhar em cima dela (06/11 04:17).
- **Não interromper jobs em execução; não duplicar; não sobrescrever** (prompts SOL, rodada 2 e leis).

**Direção científica**
- **Pausa do backprop:** pausar os treinos (SSL, lei de escala) até o SPiFiL sair na frente do aleatório; trabalhar só no SPiFiL sem backprop (07/11 12:00).
- **O SPiFiL detecta e classifica dentro do modelo**, sem backprop (07/11 21:51). O aleatório só inicializa e não detecta nada (07/11 21:53).
- **O aleatório fica sempre sozinho:** é régua pura, sem nenhuma técnica. A comparação é sempre SPiFiL melhorado contra o aleatório puro e contra o DINOv2 (08/11 13:16). Por isso saíram H02 e H04 da rodada 2.
- **Um modelo para todas as bases:** receita padrão ou regra sem rótulo, julgada por arrependimento (08/11 01:22).
- **Regra if/else da saliência:** sem saliência em aid e remote_sensing; com saliência em flowers e coconut_trees (08/11 11:37 e 18:15).
- **Foco (08/11 18:15):** aid, flowers, remote_sensing e coconut_trees. As larvas e os parasitos já estão bons: servem só para derivar leis. Meta: passar de 0,70 em flowers e remote_sensing; o DINOv2 chega a ~0,97 e ~0,71. Pool de experimentos sem backprop, antes do benchmark.
- **Leis de construção são a prioridade** (08/11 15:27): larguras diferentes por camada, sem grade uniforme nova.
- **Superpixels:** 50 por imagem, fixado pelo usuário.
- **Mais de uma imagem por classe** nas bases variáveis: 10 a 15 imagens, sem passar do total de treino (08/11 01:50). É o "próximo passo" (08/11 11:32).
- **Bases:** coconut_trees e food101 entram em todos os experimentos a 50% (08/11 01:28). A **food101 está pausada** com 16 experimentos guardados, porque é grande demais (08/11 15:27). dermamnist (28 px) e fer2013 ficaram de fora; a HAM10000 entrou.
- **Fusão Top-K:** parar, apagar e seguir com o `gmr` (07/11 22:32).
- **GELU:** testar no melhor modelo (08/11 16:52).
- **Saliência gaussiana:** pedido do usuário (08/11 13:11). Se não coubesse no SPiFiL sem backprop, ficaria guardada para o treino.
- **Pergunta respondida "pode passar"** (08/11 16:50): deixar terminar o 2º sorteio dos protozoários, mesmo acima de 15 GB.

## Pedidos ainda não atendidos

### Decisões que esperam o usuário
- **`r4` travado no Jaci** (desde ~18h30 no último job, `flowers` split 1): matar só esse processo (recomendado) ou deixar. Enquanto isso, o `r5` não começa ([`09_spf_r4.md`](../ablacao/09_spf_r4.md)).
- **`aimv2_large_native` no benchmark:** rodar em 224 px, no tamanho original ou pular.
- **Disco `/` do Jaci cheio:** avisar o administrador; autorizar mover `/tmp/ray` (1,2 GB, sessões de 16/10 a 02/11) para `/dados/home/moliveira/exps/tmp/ray_antigo`.
- **DINOv3:** falta o login (gated). Também pendentes da fase 1: PCA dos patches do SPiFiL (`layer_bank`) e Eggs a 896 px.
- **`law_f0`:** a decisão sobre o ruído de flowers (0,026, acima de 0,02).

### Pedidos em andamento, ainda sem resultado
- Saliência gaussiana (`gauss`); GELU; imagens por classe (`porclasse_jaci`); largura coco × Protozoan; `r5`; 320 px com 3 splits (`pf_res320_conf`) e 448 px; leis nos protozoários, em Larvae/Eggs e o sorteio 43 das bases menores; SoftHebb das 4 bases de foco (`pf_hebb_*`); `pf_leis_off`/`on`; benchmark de 18 modelos.
- Auditoria do padrão do repositório e plano de refatoração (08/11 21:12): em andamento, só leitura.
- Lei de montagem do SPiFiL (tipo ResNet/EfficientNet/LeJEPA) e lei tamanho do modelo × tamanho da imagem (06/11): parcial. A lei dos superpixels saiu negativa (sem lei única); as leis P1 estão em andamento.
- Família B0–B3 por lei de escala, no estilo EfficientNet (07/11 01:06), com SSL e as perdas do primeiro prompt: pausada junto com o backprop.

### Prompt das leis de construção (08/11 15:27): P0–P7

| item | estado | onde |
|---|---|---|
| P0 auditoria e canais por camada | concluído | [`leis_p0_p1.md`](../ablacao/fontes/leis_p0_p1.md) |
| P1 leis × profundidade (A, B, C; 17 arquiteturas) | parcial: concluído em aid, flowers e rs (s42 e s43) e nas 4 bases menores (s42); rodando nos protozoários (Athena) e em Larvae/Eggs (Jaci); s43 das bases menores na fila | tema 10 |
| P2 orçamento semelhante | parcial: as 3 leis C entraram na grade P1; falta a comparação com orçamento igual | `leis_p1` |
| P3 lei × resolução × heterogeneidade | parcial: `pf_leis_off`/`on` na fila (só as 4 bases de foco); a heterogeneidade (seção 5: propriedades sem rótulo que preveem a largura) não começou | tema 10 |
| P4 kernel e pooling | não iniciado com leis (o `r4` testou kernel 3/5/7 só com largura uniforme) | — |
| P5 confirmação dos finalistas | não iniciado | — |
| P6 SoftHebb nas arquiteturas das leis | não iniciado (os `pf_hebb_*` usam a rede de 3 a 4 camadas do `alt_encoders`) | — |
| P7 leitura e complementaridade | na fila: `cat_ms_eq` dentro de `pf_leis_off`/`on` | tema 8 |
| Lei D (crescimento pelos dados) | proposto: só a interface (`NotImplementedError`) | [`10_lei_d.md`](../ablacao/10_lei_d.md) |

### Rodada 2 (08/11): seções 1–7, com a seção 4 trocada pelo usuário (H00–H38)

| item | estado | onde |
|---|---|---|
| 1 auditoria | concluído | [`auditoria.md`](../ablacao/fontes/auditoria.md), [`auditoria_rodada2_jaci.md`](../ablacao/fontes/auditoria_rodada2_jaci.md) |
| 2 A0 GeM p3, A1 GeM p2, A2 multiescala | concluído (`leitura_a2`): A2 passa; A1 encerrada | tema 8 |
| 3 B0–B3 diversidade funcional | concluído (`sol_p3` reaproveitado): nenhuma melhora o `fps` | tema 6 |
| 4.1 validar a implementação | concluído (auditoria 4.1) | tema 12 |
| H00 aleatório fixo | concluído no `sol_hebb1`; `pf_hebb_rand_d3` terminou (sem tabela), `_d4` na fila | tema 12 |
| H01, H03, H05 | concluído (`sol_hebb3`, `sol_hebb3_conf`); extensão 512/L4/TTA/saliência em `pf_hebb_d3_*`, `d4_*`: rodando ou na fila | tema 12 |
| H02, H04 (aleatório + técnica) | retirados pela regra do aleatório sozinho | — |
| H06 geodésico + SoftHebb | reaproveitado do `sol_hebb1` | tema 12 |
| H07–H12 taxa e temperatura | não iniciado | — |
| H13–H18 por camada | não iniciado | — |
| H19–H24 normalização e restrições | não iniciado | — |
| H25–H29 amostragem | não iniciado | — |
| H30 homeostase | reaproveita o `sol_hebb2` (relançado 21:11, 19 de 36 prontos) | tema 12 |
| H31–H35 inativos, raros, redundantes, substituição | não iniciado (a recuperação de filtros não existe no código) | — |
| H36–H38 banco parcialmente fixo, paralelo, subconjunto | não iniciado | — |
| 4.9 leitura dos finalistas | parcial (`gem_pw`, `cat_gem_pw`, `gem_tta_pw`) | tema 12 |
| 4.10 etapas A–E | A feita; B a E não iniciadas | — |
| 5 Scattering: sozinho | reaproveitado do `sol_p6` | tema 13 |
| 5 D2 FPS + Scattering | não iniciado (não existe) | [`13_d2_fps_scattering.md`](../ablacao/13_d2_fps_scattering.md) |
| 6 triagem e confirmação (3 splits × 5 sorteios) | feita só no SoftHebb (`sol_hebb3_conf`); a leitura teve 3 × 2; o resto não | — |
| 7 escala arquitetural | sem item próprio executado; o prompt das leis (P0–P7) trata o tema | tema 10 |

### Prompt SoftHebb E00–E20 (07/11 23:25)

| item | estado |
|---|---|
| E00–E03 (Fila 1) | concluído (`sol_hebb1`): não melhora com a implementação antiga |
| E04–E05 (Fila 2, homeostase) | piloto concluído; 12 bases no Jaci relançado (19 de 36 prontos) |
| E06–E09 (Fila 3, diagnóstico e proteção) | não iniciado; perdeu prioridade (08/11 13:09) |
| E10–E13 (Fila 4, robustez e recuperação) | não iniciado; perdeu prioridade |
| E14–E17 (Fila 5, combinações) | não iniciado; perdeu prioridade |
| E18–E20 (Fila 6, políticas) | não iniciado; perdeu prioridade |

### Prompt SOL (07/11 22:15): rodadas e as 20 perguntas

Rodadas: 0 auditoria concluída ([`sol_auditoria.md`](sol_auditoria.md)); 1 mecanismos P1–P5 concluída (temas 4, 6, 8, 11 e 2); 2 encoders P6 concluída (tema 13); 3 combinações parcial (só whitening + saliência, `sol_wsal`); 4 escala parcial (`spfdeep_*`, `r4`, leis); 5 confirmação com 5 sementes só no SoftHebb; 6 análise: sem relatório final com tabelas T1–T18.

| pergunta | estado |
|---|---|
| 1 patches > aleatório? | respondida: sim, com a receita (11 de 12 bases; tema 1) |
| 2 FPS no espaço de ativação | parcial (f2 empata com o `fps`; tema 6) |
| 3 diversidade funcional > visual? | respondida: não (tema 6) |
| 4 normalização entre camadas reduz redundância? | sem medida direta |
| 5 whitening melhora redes profundas? | respondida: sim (tema 1) |
| 6 ativações negativas? | respondida: não, com o mesmo orçamento (tema 11) |
| 7 efeito da profundidade | respondida (tema 9) |
| 8 `cat` > última camada? | respondida: sim, exceto coco (tema 8) |
| 9 pooling | respondida: GeM fica (tema 8) |
| 10 saliência ajuda? | respondida: sim, nas bases com objeto (tema 2) |
| 11 dependência do fundo | respondida (tema 14) |
| 12 maior ganho isolado | respondida: whitening entre camadas |
| 13 ganhos complementares | parcial (whitening + saliência somam) |
| 14 ganho com filtros controlados | parcial (`crelu`, leis C) |
| 15 estabilidade entre sementes | parcial (só `sol_hebb3_conf` com 5 sorteios) |
| 16 generalização entre famílias | parcial (família deixada de fora na tabela de arrependimento) |
| 17 custo de cada melhoria | não respondida (sem custo por leitura) |
| 18 SoftHebb e Scattering competitivos? | parcial (tema 12 e 13) |
| 19 melhor compromisso | parcial (receita única) |
| 20 contribuição científica nova? | não respondida |

### Prompt do agrupamento (07/11 20:23): C0–C9 e as 10 perguntas
- C0–C2: avaliados. C3–C9 (grafo Louvain, SIFT + cluster, híbrido, união): concluídos no `cl_a`, todos descartados nesta forma. K-means: avaliado em 2 bases.
- Faltam: Spectral, Leiden e Affinity Propagation (não implementados); agrupamento com z-score comum, que separaria os dois efeitos ([`05_agrupamento_zscore_comum.md`](../ablacao/05_agrupamento_zscore_comum.md)); agrupamento em profundidade (4, 5, 6 e 8 camadas), pergunta 10 sem dado.

### Prompt da multiescala (07/11 15:12)
- Rodadas A (S1–S8) e B (localização): concluídas (`ms_a`, `ms_a2`, `ms_b`, `ms_b2`). Rodada C (diversidade): coberta pelo `fps` e pelo `sol_p3`. Rodada D (4, 5, 6 e 8 camadas, 64 e 256): coberta pelo `spfdeep_*`. Fundo F1–F4: concluído com 2 splits.

### Ideias levantadas, ainda sem experimento
- VLAD concatenado às camadas (`cat` + VLAD) com a mesma dimensão final.
- Leitura que depende da base (última camada contra `cat`), por regra sem rótulo, por exemplo pela resolução nativa.
- SoftHebb dentro da receita completa (6 camadas, whitening, saliência) e nas leis finalistas.
- f3 (baixa redundância) em aid, como sinal exploratório.

## Diagnóstico de dados (09/11)
| análise | bases | resultado principal | detalhe |
|---|---|---|---|
| Equivalência de patches | rs, aid, flowers, larvas, ovos | vizinho de outra classe em rs, aid e flowers; prevê a distância até o DINOv2 (ρ = 0,77) | [analise_textura.md](../analise_textura/analise_textura.md) |
| Filtros fiéis / trocados | as mesmas 6 | FPS escolhe menos filtros fiéis que o sorteio; Desert domina; os trocados ainda são úteis | [kernels_classe.md](../analise_textura/kernels_classe/kernels_classe.md) |
| Patches constantes (helminth_eggs) | helminth_eggs, Eggs_subgroup, Larvae | 51% de janelas pretas; a regra nova pega, o teste antes e depois está rodando | [patches_constantes.md](../analise_textura/patches_constantes.md) |
| Decisão E0 a E7 | aid, rs, flowers, coconut | 320 px, multiescala e só a última camada no coconut | [decisao.md](../decisao_E0_E7/decisao.md) |
