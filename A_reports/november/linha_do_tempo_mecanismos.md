# Linha do tempo dos mecanismos testados no SPiFiL (07 a 09/11/2026)

Montada a partir de `relatorio_07_08nov.md`, `multi_dataset/decisoes_spifil.md`, `multi_dataset/ablacao_completa.md` e `encdeep/G3_pilotos.md`. Efeito sempre medido contra o SPiFiL de referência do momento, em κ de validação. ✅ melhorou · ❌ piorou · ➖ empatou · ❔ inconclusivo (interrompido).

| # | dia | mecanismo | efeito |
|---|---|---|---|
| 1 | 07/11 | Número de superpixels (12 a 1600) | ➖ depende da base; 50 ficou como padrão |
| 2 | 07/11 | Mapa de saliência `gmr` para detecção | ✅ 1,3 a 3,4× mais energia dentro do objeto |
| 3 | 07/11 | Mapa de densidade SIFT | ❌ descartado |
| 4 | 07/11 | Fusão Top-K de mapas | ❌ descartado |
| 5 | 07/11 | Multiescala nos pontos dos filtros (Rodada A) | ➖ igual ao geodésico |
| 6 | 07/11 | Agrupamento dos filtros (C3 a C9) | ❌ |
| 7 | 07/11 | Normalizações novas (Z-score do patch, N2, N3, N6, N7) | ❌ |
| 8 | 07/11 | PCA/ZCA nos filtros | ❌ |
| 9 | 07/11 | SoftHebb antigo (Fila 1) | ❌ 0,732 → 0,704 |
| 10 | 07/11 | Homeostase no SoftHebb (Fila 2) | ➖ sem ganho |
| 11 | 08/11 | Whitening entre camadas | ✅ maior ganho; vence o aleatório em 11 de 12 bases |
| 12 | 08/11 | Saliência como peso/foco | ✅ +0,015 a +0,034 (larvas até +0,11) |
| 13 | 08/11 | Saliência desligada em aid e remote_sensing | ✅ igual ou melhor ali (if/else do usuário) |
| 14 | 08/11 | Leituras GAP, GMP e SPP | ❌ (SPP −0,05) |
| 15 | 08/11 | Mais de 256 filtros por camada | ➖ |
| 16 | 08/11 | Multiescala nos pontos (Rodada B) | ➖ |
| 17 | 08/11 | Encoders alternativos (Scattering, PCANet) | ❌ piores que o SPiFiL |
| 18 | 08/11 | Diversidade funcional (B1 a B3) | ➖ |
| 19 | 08/11 | Polaridade (abs, crelu) | ❌/➖ abs piora; crelu só pela dimensão dobrada |
| 20 | 08/11 | GeM p=2 | ➖ |
| 21 | 08/11 | SoftHebb na receita dos autores (rede de 3 camadas) | ✅ +0,026 a +0,064 |
| 22 | 08/11 | Leitura multiescala com a mesma dimensão | ✅ +0,016 |
| 23 | 08/11 | Leis de construção (512 → reduzir devagar) | ➖ empata, com 1,5× os parâmetros |
| 24 | 08/11 | Leitura VLAD | ❌ perde para a concatenação |
| 25 | 08/11 | coconut_trees: só a última camada | ✅ 0,63 → 0,79 |
| 26 | 08/11 | L4/L5 com SoftHebb, CLAPP e SFA | ➖/❌ empatam; SFA piora |
| 27 | 08/11 | Sem whitening em L4/L5 | ✅ aid +0,020; rs neutro |
| 28 | 08/11 | Saliência gaussiana | ❔ parou em 44/72 |
| 29 | 08/11 | GELU | ❌ completo em 10/11 (36 células): GELU vence 10 de 36 splits, Δ de −0,037 a +0,013 (dentro de 1 dp); manter ReLU. jaci `results/svm_disc_md_gelu_*` |
| 30 | 09/11 | Resolução 320 px (aid, remote_sensing) | ✅ +0,011 e +0,036 |
| 31 | 09/11 | Resolução abaixo de 200 px | ❌/➖ |
| 32 | 09/11 | Resolução maior em bases com objeto | ❌ Larvae −0,05, flowers −0,03 |
| 33 | 09/11 | Regra do fundo preto (helminth_eggs) | ❔ parou em 8/12 |
| 34 | 09/11 | Receita com backprop | ❔ parcial: L2 melhora sobre L1, L3 piora; Larvae 1% (12 imagens) com κ negativo |
| 35 | 10/11 | TTA na receita (8 vistas rot90 × flip; `multi_dataset/tta_receita.md`) | ✅ aid +0,073, remote_sensing +0,047; ➖ flowers; parasitos rodando no jaci |
| 36 | 10/11 | Kernels nos tokens de alta norma do DINOv2 ("registers"; `analise_textura/dino_registers_pontos.png`) | ❌ igual ao sorteio em aid e rs; flowers −0,08 contra o fps (pontos no fundo liso); só o split 1 |
| 37 | 10/11 | Atenção sem parâmetros (sa, espacial, SimAM, non-local, multiescala; `atencao/triagem.md`) | ❌ 3 splits: −0,015 a +0,007 na receita; só no remote_sensing soma com 320+TTA (+0,026) |
| 38 | 10/11 | Combinação TTA + 320 px (+ atenção) | ✅ remote_sensing +0,079 (3/3); aid: o TTA sozinho é o melhor (+0,073); flowers ➖ |
| 39 | 10/11 | Backprop layerwise antigo (`curvas/investigacao_backprop.md`) | ❌ nunca partiu da receita (grow com 1 imagem por classe, sem saliência e sem whitening); substituído pela receita inteira + autoencoder (jaci) |
| 40 | 10/11 | Receita inteira + autoencoder (reconstrução) | ❌ Eggs: κ 0,77 → 0,53 em 10 épocas enquanto a recon melhora; trocado por LeJEPA |
| 41 | 11/11 | Leituras globais da literatura: covariância com raiz (iSQRT), MSCP L3–L6, Gram deslocado, Fisher vector, self-self attention (`atencao/relacoes_globais.md`) | ✅ só cat+MSCP no remote_sensing (+0,055, 3/3); ❌ o resto e a flowers; com 320 px + TTA não passa de 0,571 |
| 42 | 11/11 | 320 px + saliência + atenção nas bases ruins (`results/svm_disc_md_atencao_sal320_*`, 3 splits) | ✅ flowers: fpsw 320 cat 0,689 / + TTA + SimAM no mapa final 0,695, contra 0,660 a 200 px + TTA (+0,035); ➖ HAM 0,530 (contra 0,515 a 200 px); ➖ remote_sensing 0,552 (abaixo do 0,571 de 320 + espacial@L5 + TTA) |
| 43 | 11/11 | Imagens por classe para escolher os filtros (porclasse, jaci) | ➖ escolher por classe não ganha do sorteio sem classe; 5 a 10 imagens por classe chegam a ~0,01 do padrão de 500 (Protozoan, aid, remote_sensing); exceções: Larvae (ainda sobe em k=30) e flowers |
