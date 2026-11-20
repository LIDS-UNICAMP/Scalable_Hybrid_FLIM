# Log de resultados: leis em 12 datasets

Atualizado em 07/11/2026, 07:40 UTC. Regra: decisoes so pela validacao. Numeros de teste abaixo foram pedidos pelo usuario e sao so relato.
Fontes: `results/benchmark/*.csv`, `results/svm_disc_md_eval_*.csv`, `/exps/moliveira/queues/logs/`. Relatorios dos agentes em `agentes/`.

## Estado
- Treinos: 458 bracos prontos de ~1.300 do Estagio 1, sem falha. Modelo completo: linha de base lancada, sem resultado.
- Pesos: `/exps/moliveira/spifil_growth/disc_md_*`.

## Benchmark, kappa de validacao a 50%, media de 3 splits
| Dataset | Pico | I-JEPA g/16 | DINOv2 giant | AIMv2 3B |
|---|---|---|---|---|
| Eggs | 0,932 | 0,970 | 0,980 | 0,983 |
| Eggs_subgroup | 0,924 | 0,963 | 0,989 | 0,980 |
| Larvae | 0,876 | 0,850 | 0,968 | 0,941 |
| Larvae_subgroup | 0,834 | 0,818 | 0,981 | 0,981 |
| Protozoan | 0,823 | 0,921 | 0,939 | 0,953 |
| Protozoan_subgroup | 0,859 | 0,926 | 0,945 | 0,953 |
| aid | 0,864 | 0,903 | 0,970 | 0,981 |
| flowers | 0,888 | 0,937 | 0,973 | 0,978 |
| helminth_eggs | 0,952 | 0,972 | 0,987 | parcial |
| helminth_larvae | 0,966 | 0,965 | 0,994 | parcial |
| protozoan_cysts | 0,898 | 0,925 | 0,966 | parcial |
| remote_sensing | 0,601 | 0,603 | 0,708 | 0,688 |

DINOv3 pendente de login. Pico oficial no HF foi ajustado com rotulo ImageNet.

## Superpixels, camada 1, SVM de validacao, 200 px (parcial, 28 de 117)
| Dataset | 800 sp, 32 filtros | 800 sp, 64 filtros | 50 sp, 64 filtros |
|---|---|---|---|
| helminth_larvae | 0,938 | - | 0,932 |
| helminth_eggs | 0,744 | - | 0,842 |
| protozoan_cysts | 0,589 | - | 0,743 |
| Eggs_subgroup | 0,499 | 0,580 | - |
| Protozoan_subgroup | 0,446 | - | - |
| Larvae_subgroup | 0,087 | 0,427 | - |
| Protozoan | 0,068 | 0,086 (1 split) | - |

Sem lei ainda. Camada 1 ve 7 px.

## Teste da camada 1 (pedido do usuario, so relato)
| Dataset | 800 sp, 32 f, 200 px | 50 sp, 64 f, 200 px | 50 sp, 64 f, 320 px |
|---|---|---|---|
| helminth_larvae | 0,896 | 0,907 | - |
| helminth_eggs | 0,735 | 0,830 | - |
| protozoan_cysts | 0,575 | 0,749 | - |
| Eggs_subgroup | 0,497 | 0,674 | - |
| Larvae_subgroup | 0,105 | 0,656 | - |
| Protozoan_subgroup | 0,458 | 0,646 | - |
| flowers | - | - | 0,504 |
| Eggs (2 splits) | - | - | 0,496 |
| Protozoan | - | 0,292 | - |

## Kernels (K1, treino real, validacao)
| Dataset | Antes | Sem copias | Diferenca |
|---|---|---|---|
| protozoan_cysts | 0,743 | 0,861 | +0,118 |
| helminth_eggs | 0,842 | 0,870 | +0,028 |
| helminth_larvae (ipc 3) | 0,932 | 0,921 | -0,010 |

Regra candidata: copias acima de 10% do banco: tirar e repor. Detalhes em `agentes/L_kernels.md`.

## F1 degrau 1: qual perda (07/11, SVM de validacao, gem_tta_pw, L4)
B0, 200 px, 50 superpixels, 50%, split 1, semente 42, 15 epocas por estagio. Uma corrida por celula. Delta e a media dos 3 datasets contra so reconstrucao. CSV: `results/svm_disc_md_law_eval_f1.csv`.

| Perda | Lambda | k flowers | k helminth_eggs | k helminth_larvae | k medio | Delta k | F1 medio | Delta F1 |
|---|---|---|---|---|---|---|---|---|
| UR-JEPA | 0,01 | 0,574 | 0,947 | 0,940 | 0,820 | +0,024 | 0,860 | +0,020 |
| Strong-SIGReg | 0,001 | 0,559 | 0,952 | 0,932 | 0,814 | +0,018 | 0,855 | +0,015 |
| Barlow | 0,001 | 0,535 | 0,945 | 0,957 | 0,812 | +0,016 | 0,850 | +0,010 |
| VICRegL | 0,0001 | 0,528 | 0,950 | 0,956 | 0,811 | +0,015 | 0,851 | +0,011 |
| VICReg | 0,01 | 0,538 | 0,940 | 0,949 | 0,809 | +0,012 | 0,849 | +0,009 |
| CorInfoMax | 0,0001 | 0,525 | 0,935 | 0,957 | 0,806 | +0,009 | 0,848 | +0,007 |
| Strong-SIGReg | 0,01 | 0,535 | 0,940 | 0,930 | 0,802 | +0,005 | 0,843 | +0,003 |
| Barlow | 0,01 | 0,498 | 0,938 | 0,966 | 0,800 | +0,004 | 0,841 | +0,000 |
| Weak-SIGReg | 0,01 | 0,531 | 0,932 | 0,930 | 0,798 | +0,001 | 0,841 | +0,000 |
| Weak-SIGReg | 0,001 | 0,523 | 0,940 | 0,929 | 0,797 | +0,001 | 0,841 | +0,000 |
| So reconstrucao | - | 0,532 | 0,935 | 0,923 | 0,796 | +0,000 | 0,841 | +0,000 |
| CorInfoMax | 0,001 | 0,491 | 0,923 | 0,974 | 0,796 | -0,000 | 0,834 | -0,006 |
| UR-JEPA | 0,001 | 0,501 | 0,935 | 0,947 | 0,795 | -0,002 | 0,840 | -0,000 |
| VICReg | 0,001 | 0,492 | 0,923 | 0,958 | 0,791 | -0,006 | 0,834 | -0,007 |
| VICRegL | 0,001 | 0,471 | 0,920 | 0,940 | 0,777 | -0,020 | 0,825 | -0,015 |

Leitura: os 5 primeiros vao para o degrau 2. O melhor ganha +0,024, mas a diferenca entre dois bracos, pelo ruido de F0, tem desvio de ate ~0,015 na media dos 3 datasets (cota por cima: os bracos dividem a semente 42): a ordem do topo ainda nao esta resolvida. Por dataset a regua e 0,005 em helminth_eggs, 0,022 em helminth_larvae e 0,037 em flowers. So em helminth_eggs o ganho passa da regua com folga. O checkpoint saiu na ultima ou penultima epoca em todos: `val_recon_loss` ainda caia.

## F0: ruido entre sementes (07/11, SVM de validacao)
B0, Weak-SIGReg 0,01, 60 epocas por estagio. CSV: `results/svm_disc_md_law_eval_f0.csv`.

| Dataset | Split | Semente 42 | 43 | 44 | Desvio |
|---|---|---|---|---|---|
| flowers | 1 | 0,547 | 0,493 | 0,461 | 0,043 |
| flowers | 2 | 0,504 | 0,486 | 0,475 | 0,015 |
| flowers | 3 | 0,583 | 0,583 | 0,583 | 0,000 |
| helminth_eggs | 1 | 0,936 | 0,929 | 0,935 | 0,004 |
| helminth_eggs | 2 | 0,937 | 0,945 | 0,936 | 0,005 |
| helminth_eggs | 3 | 0,941 | 0,943 | 0,943 | 0,001 |
| helminth_larvae | 1 | 0,931 | 0,974 | 0,965 | 0,023 |
| helminth_larvae | 2 | 0,957 | 0,948 | 0,965 | 0,009 |
| helminth_larvae | 3 | 0,974 | 0,966 | 0,949 | 0,013 |

| Dataset | k medio | Desvio entre sementes | Desvio entre splits |
|---|---|---|---|
| flowers | 0,524 | 0,026 | 0,051 |
| helminth_eggs | 0,938 | 0,004 | 0,005 |
| helminth_larvae | 0,959 | 0,016 | 0,004 |

Regra de PLANO_FINAL: agrupado 0,018, na faixa de 0,01 a 0,02 (2 sementes so no top k e em B1 de F3). Mas a regra nao diz se o desvio e agrupado ou por dataset: flowers sozinho da 0,026, acima de 0,02, onde o plano manda pausar e tornar o sorteio de imagens um fator. Quem puxa e a celula flowers split 1 (0,043). Sao 3 sementes por celula: a estimativa e grosseira. Decisao pendente.
Mesma celula com 15 e 60 epocas (Weak-SIGReg 0,01, semente 42, split 1): flowers 0,531 e 0,547; helminth_eggs 0,932 e 0,936; helminth_larvae 0,930 e 0,931. Quadruplicar as epocas mexeu pouco.
flowers fica em 0,52, contra 0,89 a 0,98 do benchmark a 50%.

## Controle aleatorio em B0 (07/11, SVM de validacao)
Mesma forma e largura real, Weak-SIGReg 0,01, 60 epocas, semente 42, 3 splits. CSV: `results/svm_disc_md_law_eval_ctl.csv`.

| Dataset | SPiFiL s42 | Aleatorio s42 | Aleatorio - SPiFiL, splits 1/2/3 | Media | SPiFiL, 3 sementes |
|---|---|---|---|---|---|
| flowers | 0,545 | 0,548 | -0,037 / +0,042 / +0,006 | +0,004 | 0,524 |
| helminth_eggs | 0,938 | 0,938 | -0,001 / +0,005 / -0,003 | +0,001 | 0,938 |
| helminth_larvae | 0,954 | 0,980 | +0,052 / +0,018 / +0,009 | +0,026 | 0,959 |

Agrupado: aleatorio +0,010 sobre SPiFiL, erro-padrao 0,009, 9 pares. HS1 pede o contrario (SPiFiL +0,01): em B0 nao passa. Em helminth_larvae os 3 aleatorios (0,975 a 0,983) ficam acima dos 9 SPiFiL (0,931 a 0,974). Ressalva: a cabeca e quase toda sorteada nos dois bracos (SPiFiL so em ~97 de 1280 filtros em helminth_larvae, 250 em flowers, ~168 em helminth_eggs), entao a diferenca vem do tronco e o par mede pouco do SPiFiL. Sementes 43 e 44 do aleatorio entram no lote seguinte.

## Replicas da grade sc (07/11, SVM de validacao, gem_tta_pw)
helminth_eggs, split 1, 50%, L3, w1: 6 treinos iguais ate a L3, so muda a GPU e a epoca que a sonda escolheu.
| Run | Epoca do checkpoint | Kappa |
|---|---|---|
| c1280 | 52 | 0,899 |
| d6 | 52 | 0,876 |
| d5 | 59 | 0,869 |
| d4 | 32 | 0,861 |
| c640 | 27 | 0,859 |
| c2560 | 53 | 0,857 |

Media 0,870, desvio 0,016, faixa 0,041. O ruido aqui mistura GPU e escolha de epoca pela sonda: nao separa os dois. F0 mede de novo sem sonda. CSV: `results/svm_disc_md_law_eval_replicas_l3.csv`.

## Incidentes
- Teto de 40 GB estourou 3 vezes em 05-06/11 (42,2; 41,2; 43,6 GB). Depois: teto duro por processo. Gestor liberou a GPU inteira em 06/11, preservando o outro usuario; limite adotado 46 GB por placa.
- Bug da sonda (float16 no treino, float32 na validacao), corrigido em 06/11 05:20. `epoch_metrics` e `best_kappa` anteriores de helminth_larvae, helminth_eggs e protozoan_cysts nao sao confiaveis; SVM nao afetado.
- I-JEPA antigo recebeu LAB, nao RGB. Remedido.
- Cache de imagens: ligado em 212 YAMLs ainda nao iniciados; tensor igual bit a bit (`agentes/CACHE.md`).
- Lei tamanho x superpixels, L1 sem treino, teste a pedido: sem lei unica; log2 n* = 4,01 + 0,18 log2 S, R2 0,013; 2 classes pedem n ~100, resto n = 12 (`law_size_superpixels.md`).

## Pendente do usuario
PCA das patches do SPiFiL (`layer_bank`), Eggs a 896 px, login do DINOv3.

## 07/11: SPiFiL contra aleatório, sem backprop

Grade pareada e hipótese do fundo em `spifil_vs_aleatorio.md`, com tabela automática. Sessões `disc_md_spf_*`, logs `/exps/moliveira/queues/logs/spf.*`.
- 07/11 Saliência, fase 1: 270 imagens, 9 bases; mapa escolhido sem label: gmr (Q 0,725, estabilidade 0,898, 0,33 s). Relatório saliencia.md. Fase 2 em disc_md_sal_q, resultado em spifil_vs_aleatorio.md.
- 07/11 Multiescala: `flim/scale_space.py` e braços S1 a S8 na grade; auditoria em multiescala_auditoria.md; critério, tabela pareada e matriz em multiescala.md. Fila `disc_md_ms_q`: aleatório congelado, A, B, depois as outras 7 bases. Logs `/exps/moliveira/queues/logs/ms.*`.

## r1 flowers split 1 sh3 (07/11, 17:45 UTC)

O job travou duas vezes no mesmo ajuste de SVM: controle `whrand`, plano 6-9-18, depois da camada 1. A primeira cópia ficou 4 h 50 min sem gravar nada; a segunda, 1 h 18 min nesse ajuste. Parei as duas para liberar `r2`. O CSV ficou com 270 linhas e sem o `whrand` das camadas 2 e 3 desse plano; só esse controle, nesse split, ficou faltando.

## Conclusões parciais do SPiFiL sem backprop (07/11, 19:45 UTC)

Só validação, 50%. κ "justo": célula escolhida em 2 splits e medida no 3º, para método e para aleatório. Oráculos fora. Sem veredito: nenhum método passou no critério.

| base | SPiFiL | aleatório | ganho | melhor receita | DINOv2 |
|---|---|---|---|---|---|
| Larvae_subgroup | 0,680 | 0,402 | +0,278 | crus + saliência | 0,981 |
| Larvae | 0,678 | 0,518 | +0,160 | crus + saliência | 0,968 |
| Protozoan_subgroup | 0,833 | 0,763 | +0,069 | fps | 0,945 |
| Eggs_subgroup | 0,801 | 0,760 | +0,041 | fps | 0,989 |
| Protozoan | 0,665 | 0,645 | +0,020 | crus@geodesic | 0,939 |
| aid | 0,643 | 0,625 | +0,018 | fps@geodesic | 0,970 |
| flowers | 0,620 | 0,616 | +0,004 | fps@sift | 0,973 |
| Eggs | 0,661 | 0,679 | −0,018 | crus@obj | 0,980 |

helminth_eggs: só 2 splits, sem seleção justa, cerca de 0,94 contra 0,92. Sem dado nesta grade: helminth_larvae, protozoan_cysts, remote_sensing.

Receita fixa provisória: `fps`, 3 camadas concatenadas, whitening só na leitura. Δ médio contra o melhor aleatório: +0,064 com 256 filtros (7 bases, pior −0,017) e +0,056 com 64 (7 bases, pior −0,003). Não medido em Eggs e Protozoan.

Fica de fora: ZCA, PCA e HDBSCAN nos filtros, DoG, Gauss antes do recorte, várias escalas, combo de tudo, soma de camadas, mais imagens por classe. Saliência só como opção para larvas. Abertos: rodada B, S5, whitening entre camadas, mais de 4 camadas, 256 contra 64.

## Pontos dentro e fora do objeto (07/11, 20:30 UTC)

`sementes_dentro_fora.png`: 6 bases com máscara, fundo real, 200 px, 50 superpixels, média dos splits. Máscara só mede. Padrão do SPiFiL (medoide): 25% a 49% dentro, com o objeto ocupando de 6% a 20% da imagem. Saliência sobe para 70% a 98%, mas tirando pontos de fora: fica com 9 a 23 pontos por imagem, e os de dentro até caem um pouco (ex.: Eggs_subgroup, 25 para 22). Saliência não foi medida em Eggs e Protozoan.

## Objetivo fixado pelo usuário (07/11, 21:55 UTC)

O SPiFiL sem backprop tem de detectar o objeto e depois classificá-lo, as duas coisas dentro do modelo, sem máscara. Ele já classifica bem quando sabe onde está o objeto e cai quando não sabe. Daqui em diante, cada braço é lido por duas medidas: κ de validação e detecção (fração de pontos e de energia de ativação dentro do objeto, medida só como diagnóstico).

## Detecção e ativação do SPiFiL sem backprop, 12 bases (07/11, 23:30 UTC; corrigida 23:45)

Figuras em `/exps/moliveira/results_sol/figures/deteccao/` (`<base>_deteccao.png`, `<base>_ativacao.png`, `<base>_fracoes.csv`), feitas por `eval/spf_deteccao.py`. Split 1, 50%, 200 px, 50 superpixels, 64 filtros, 3 camadas, sorteio 42, bancos de 500 imagens; 4 imagens de treino em intervalos iguais da lista. A máscara não escolhe nada: só desenha o contorno e mede. **Só 4 imagens por base: é sinal para olhar, não resultado.**

Pontos: % dos pontos dentro do objeto. Área = % da grade 200×200 que é objeto (o que daria um ponto jogado ao acaso).

| base | área (grade) | SIFT todos | sift_nms | medoide | obj |
|---|---|---|---|---|---|
| Eggs | 26 | 41 | 36 | 38 | 43 |
| Eggs_subgroup | 36 | 53 | 54 | 48 | 31 |
| helminth_eggs | 2 | 93 | 100 | 21 | 32 |
| Larvae | 2 | 48 | 41 | 16 | 21 |
| Larvae_subgroup | 6 | 36 | 31 | 24 | 33 |
| helminth_larvae | 2 | 100 | 100 | 15 | 30 |
| Protozoan | 12 | 55 | 46 | 32 | 43 |
| Protozoan_subgroup | 22 | 66 | 61 | 44 | 54 |
| protozoan_cysts | 5 | 93 | 80 | 48 | 57 |

Energia na camada 3: % da energia dentro do objeto. Área = % da região medida que é objeto. Região: sem 8 px de borda e sem as faixas de `content_box`; em helminth_eggs, helminth_larvae e protozoan_cysts só sem os 8 px de borda, sem `content_box` (fundo já preto; coluna `regiao` do CSV e título da figura dizem isso). Acima da área, o banco acende mais no objeto que no fundo.

| base | área (medida) | fps | fpsw@mapa_peso | crus@sift | km_w | zca2 | spifil | aleatório (régua) |
|---|---|---|---|---|---|---|---|---|
| Eggs | 34 | 35 | 53 | 26 | 30 | 38 | 44 | 31 |
| Eggs_subgroup | 47 | 45 | 67 | 41 | 48 | 60 | 53 | 45 |
| helminth_eggs | 2 | 15 | 32 | 60 | 32 | 27 | 2 | 11 |
| Larvae | 21 | 24 | 28 | 23 | 23 | 25 | 27 | 19 |
| Larvae_subgroup | 13 | 18 | 30 | 21 | 20 | 25 | 13 | 18 |
| helminth_larvae | 2 | 15 | 37 | 46 | 29 | 17 | 9 | 11 |
| Protozoan | 16 | 21 | 55 | 40 | 24 | 31 | 27 | 21 |
| Protozoan_subgroup | 30 | 35 | 68 | 36 | 27 | 38 | 26 | 29 |
| protozoan_cysts | 6 | 25 | 44 | 73 | 46 | 30 | 7 | 19 |

Leitura: `fpsw@mapa_peso` é o único braço que acende no objeto em todas as 6 bases com fundo real (de 1,3× a 3,4× a área); `spifil` padrão fica perto do aleatório, menos em Eggs. Nas 3 bases de fundo preto, todo braço acende no objeto bem acima da área, `crus@sift` mais (46% a 73% contra 2% a 6%); `spifil` não: fica na área (2% a 9%), porque acende o fundo preto chapado inteiro (o viés do filtro dá energia onde a imagem é constante; ver as figuras). Nos pontos, SIFT cai mais no objeto que o medoide; `obj` sobe o medoide em quase todas.

Ressalvas: (1) helminth_eggs, helminth_larvae e protozoan_cysts têm o fundo já apagado em preto; a primeira rodada usou `content_box`, que lê o preto como faixa e recorta só o objeto (energia ~100% contra área ~100%). Refeitas sem `content_box`. A área de 40% a 91% de `saliencia.md` é a mesma coisa: medida dentro do recorte de `content_box`. Na imagem inteira o objeto ocupa de 1% a 12% (a máscara é exatamente o que não é preto), então a área de 2% a 6% está certa; a tabela de pontos nunca usou `content_box` e não mudou. (2) Em Larvae raw, 2 das 4 imagens são tiras de 16 a 23 px: ali a borda de 8 px apagaria tudo e não foi cortada (o título da imagem avisa). (3) Na camada 3 o brilho do preenchimento da convolução ainda aparece logo dentro dos 8 px de borda. (4) flowers, aid e remote_sensing não têm máscara: só figuras.

## Quanto falta para detectar, 3 splits (08/11, 00:05 UTC)

`spf_hz` (fundo zerado, Jaci) contra `spf_h` (fundo real, Athena), 6 bases com máscara, L3 `cat`, média dos 3 splits. Ganho = κ do braço − melhor aleatório no fundo real. Teto = κ do braço com fundo zerado − melhor aleatório no fundo real.
- Média: `crus` ganha +0,086 e teria +0,226; `spifil` +0,082 e teria +0,216. O SPiFiL faz cerca de 38% do caminho; saber onde está o objeto daria +0,14 de κ.
- Larvas: 46% a 67% do caminho (Larvae com 256 filtros passa de 100%, mas varia muito entre splits). Protozoários: 24% a 50% (Protozoan_subgroup com 64 filtros e `spifil`: 1%). Ovos: 1% a 18% com 64 filtros e negativo com 256.
- O aleatório com fundo zerado sobe 0,231 e fica no nível do SPiFiL zerado: saber a posição ajuda os dois quase igual.

## Encoders alternativos, 12 bases, 3 splits (08/11, 00:30 UTC)

`disc_md_sol_p6` (Athena), 864 linhas ok. κ de validação; configuração escolhida em 2 splits e medida no 3º. Aleatório = `ms_ctl`/`ms_ctl2`, só 64 filtros, L3 `cat`.

| base | Scattering | SoftHebb | PCANet | aleatório 64 | melhor SPiFiL (Athena, sem whitening entre camadas) |
|---|---|---|---|---|---|
| Eggs | 0,739 | 0,742 | 0,651 | 0,582 | 0,661 |
| Eggs_subgroup | 0,782 | 0,803 | 0,752 | 0,656 | 0,834 |
| Larvae | 0,704 | 0,514 | 0,485 | 0,370 | 0,678 |
| Larvae_subgroup | 0,492 | 0,491 | 0,586 | 0,293 | 0,680 |
| Protozoan | 0,664 | 0,738 | 0,451 | 0,508 | 0,761 |
| Protozoan_subgroup | 0,730 | 0,774 | 0,548 | 0,652 | 0,830 |
| aid | 0,616 | 0,690 | 0,570 | 0,625 | 0,719 |
| flowers | 0,555 | 0,590 | 0,540 | 0,525 | 0,611 |
| helminth_eggs | 0,957 | 0,908 | 0,899 | 0,860 | 0,926 |
| helminth_larvae | 0,980 | 0,940 | 0,862 | 0,923 | sem dado |
| protozoan_cysts | 0,926 | 0,846 | 0,798 | 0,798 | sem dado |
| remote_sensing | 0,421 | 0,447 | 0,450 | 0,470 | sem dado |

SPiFiL ganha de Scattering e SoftHebb em 6 das 9 bases medidas; perde em Eggs (0,661 contra 0,742) e helminth_eggs (0,926 contra 0,957), e Scattering passa em Larvae (0,704 contra 0,678). SoftHebb é regime C (atualização local). Scattering não tem parâmetros. O aleatório aqui só tem 64 filtros; SoftHebb usa 64 ou 256.

## P4, avaliação do fundo: F1 a F4 de `eval/spf_fundo.py` em escala (08/11, 02:30 UTC)

**Resposta curta:** os bancos SPiFiL e a saliência põem mais energia no objeto do que o acaso e mudam menos sob transformação do que os bancos aleatórios. Mesmo assim, trocar o fundo derruba o κ de todos eles. É só uma leitura com 2 splits, não uma conclusão.

O que rodou: 9 bases, splits 1 e 2, 64 e 256 filtros, 3 camadas. Bancos: `spifil`, `fpsw@mapa_peso`, `crus@geodesic`, `fps` e, como régua, `random_z` e `xavier`. São 216 bancos, todos na largura pedida, escolhidos com 500 imagens de treino no sorteio 42. A sonda usa 400 imagens de treino e 200 de validação, com SVM linear treinado no treino limpo e avaliado na validação perturbada. O teste não entra. A máscara só mede e só gera o fundo substituído: nenhum banco a usa para escolher filtro. flowers, aid e remote_sensing não têm máscara, então nelas só entram estabilidade e periferia.

Mediana na camada 3 (6 bases com máscara × 2 larguras):

| banco | energia/área do objeto | κ retido, fundo de cor sólida | κ retido, textura de outra imagem | parte da mudança vinda do fundo (área do fundo 0,86) | mudança da leitura nas 5 transformações (9 bases) |
|---|---|---|---|---|---|
| fpsw@mapa_peso | 2,11 | 0,99 | 0,64 | 0,77 | 0,85 |
| spifil | 1,62 | 0,82 | 0,38 | 0,82 | 0,82 |
| fps | 1,07 | 0,83 | 0,37 | 0,86 | 0,94 |
| crus@geodesic | 0,94 | 0,84 | 0,42 | 0,84 | 0,76 |
| xavier | 0,92 | 0,57 | 0,31 | 0,90 | 1,13 |
| random_z | 0,83 | 0,59 | 0,29 | 0,91 | 1,13 |

- Energia/área igual a 1 é o acaso. O aleatório fica abaixo de 1, ou seja, olha mais para o fundo. `fpsw@mapa_peso` e `spifil` olham mais para o objeto: venceram `random_z` em 24 de 24 e em 23 de 24 pares (base, largura, split).
- Com fundo de cor sólida ou com textura de outra imagem, os bancos SPiFiL guardam mais κ do que o aleatório, mas não em todos os pares: `fpsw` venceu em 20 e 21 de 24, `spifil` em 16 e 15 de 24. A textura de outra imagem custa mais da metade do κ em quase todos os bancos.
- Ruído forte (σ = 40) ou fundo preto no lugar do fundo zera o κ de todos os bancos. Essas duas trocas tiram a imagem da distribuição do treino e não separam um banco do outro.
- Mexer só no objeto (14% da área) causa de 56% a 73% da mudança total (mediana por banco). Os bancos leem o objeto, e o fundo pesa mais ou menos na proporção da área que ocupa: um pouco abaixo dela nos bancos SPiFiL e um pouco acima nos aleatórios.
- Estabilidade, sem máscara, nas 9 bases: em translação, rotação, escala, iluminação e ruído, os 4 bancos SPiFiL mudaram menos do que `random_z` nos 36 pares de 36. Na periferia (anel externo de 20%, sem máscara, que **não** é troca de fundo), o κ retido fica entre 0,55 e 0,63 nos SPiFiL e entre 0,40 e 0,41 nos aleatórios.
- O κ limpo daqui (de 0,13 a 0,68) não é o κ da grade: são só 400 imagens de treino e só a camada 3. Ele serve para comparar perturbações, não bancos.

Arquivos: `/exps/moliveira/results_sol/tables/T7_fundo.csv` e `.md`; CSVs brutos em `/exps/moliveira/results_sol/saliency/fundo/csv/<base>_split<k>_w<W>/`; figuras 7 a 10 e 15 de `spf_figuras` e o resumo `resumo_T7.png` em `/exps/moliveira/results_sol/saliency/fundo/figs/`; bancos em `.../fundo/bancos/`.

## Banco por classe: piloto (08/11, 02:24 UTC)

flowers, split 1, 64 filtros, L6 com whitening entre camadas, só validação; κ = melhor linha (camada/leitura) do melhor braço SPiFiL. B = k imagens de treino por classe (regime B); A = mesmo total sorteado sem classe; controle = melhor de orth/xavier/tnorm/random_z.

| k | imagens | B | A | melhor controle |
|---|---|---|---|---|
| 1 | 5 | 0,647 | 0,619 | 0,607 |
| 5 | 25 | 0,626 | 0,653 | 0,601 |
| 10 | 50 | 0,659 | 0,652 | 0,610 |
| 15 | 75 | 0,689 | 0,680 | 0,601 |
| 30 | 150 | 0,656 | 0,677 | 0,604 |

Um split só: diferenças de ±0,03 estão dentro do ruído. Leitura: o SPiFiL passa o controle em todo k (+0,04 a +0,09); B e A trocam de lugar, sem ganho claro de escolher por classe. 26 min de parede. A grade completa (`disc_md_porclasse`, 8 bases, 3 splits, 64 e 256) começou às 02:24; resumo com `python <scratchpad>/pc_sum.py disc_md_porclasse` (de `results/`).

## SoftHebb, Fila 1 contra Fila 2: piloto (08/11, 03:20 UTC)

6 bases (flowers, aid, remote_sensing, Eggs_subgroup, Larvae_subgroup, Protozoan_subgroup), split 1, sorteio 42, só validação. Mesma rede (BN + conv + triângulo + max-pool, 3 camadas), mesmos lotes e mesmo SVM (`gem_pw`, `pw_whiten`). Média das 6 bases, camada 3. "p0" = banco inicial sem SoftHebb (com 0 passos); F1 = SoftHebb livre, 300 passos; E04 = viés de competição pela frequência de vitória (ganho 0,5, teto ±2); E05 = reforço só para kernels sem vitória em 2 janelas seguidas de 30 passos.

| banco | largura | braço | κ L3 | κ cat | cos. c/ inicial | vizinho mais próximo | posto efetivo | corr. funcional (vizinho) | quase nunca vencem (L3 / L1) |
|---|---|---|---|---|---|---|---|---|---|
| random | 64 | p0 | 0,519 | 0,634 | 1 | 0,06 | 63,7 | 0,47 | 1,8 / 31,8 |
| random | 64 | F1 | 0,525 | 0,608 | 0,02 | 0,78 | 27,9 | 0,89 | 3,2 / 37,7 |
| random | 64 | E04 | 0,597 | 0,604 | 0,02 | 0,80 | 26,8 | 0,91 | 1,8 / 26,7 |
| random | 64 | E05 | 0,525 | 0,606 | 0,02 | 0,78 | 28,1 | 0,89 | 2,7 / 37,7 |
| random | 256 | p0 | 0,609 | 0,686 | 1 | 0,04 | 254,7 | 0,52 | 40,8 / 155,2 |
| random | 256 | F1 | 0,604 | 0,644 | 0,02 | 0,83 | 78,8 | 0,94 | 18,0 / 206,5 |
| random | 256 | E04 | 0,605 | 0,645 | 0,01 | 0,85 | 71,3 | 0,94 | 7,0 / 155,8 |
| random | 256 | E05 | 0,601 | 0,651 | 0,00 | 0,84 | 75,3 | 0,94 | 13,0 / 206,5 |
| fps | 64 | p0 | 0,566 | 0,652 | 1 | 0,29 | 55,7 | 0,69 | 9,7 / 31,3 |
| fps | 64 | F1 | 0,609 | 0,641 | 0,50 | 0,77 | 28,1 | 0,90 | 0,7 / 26,3 |
| fps | 64 | E04 | 0,594 | 0,622 | 0,49 | 0,79 | 27,1 | 0,90 | 0,5 / 17,2 |
| fps | 64 | E05 | 0,609 | 0,641 | 0,50 | 0,77 | 28,1 | 0,90 | 0,7 / 26,3 |
| fps | 256 | p0 | 0,620 | 0,677 | 1 | 0,46 | 174,6 | 0,82 | 74,5 / 138,7 |
| fps | 256 | F1 | 0,597 | 0,664 | 0,53 | 0,84 | 71,6 | 0,93 | 2,0 / 169,3 |
| fps | 256 | E04 | 0,590 | 0,650 | 0,53 | 0,85 | 70,2 | 0,94 | 2,2 / 140,8 |
| fps | 256 | E05 | 0,598 | 0,664 | 0,53 | 0,84 | 71,5 | 0,94 | 2,0 / 169,3 |

- O SoftHebb livre (F1) reescreve o banco e o deixa redundante: o vizinho mais próximo sobe para 0,77–0,84 e o posto efetivo cai pela metade ou mais. O κ não acompanha de forma consistente: na camada 3 oscila ±0,04 em torno do p0, e no `cat` fica igual ou abaixo.
- E04 espalha as vitórias (menos kernels quase sem vitória na camada 1: 207 → 156 em random 256, 169 → 141 em fps 256), mas **não** reduz a redundância: o vizinho mais próximo e a correlação funcional ficam iguais ou sobem. O κ não muda, com uma exceção: random 64 na L3 (+0,07). Sobe em 5 de 6 bases, mas a média é puxada por Larvae_subgroup (+0,31, só 107 imagens de validação); sem ela o ganho é +0,03, e no `cat` não aparece.
- E05 quase não age: com janelas de 30 passos, só 0 a 9 kernels por camada chegam a receber reforço, e o resultado sai igual ao F1, linha por linha nas configurações em que nenhum foi reforçado.
- Leitura: a redundância vem da regra de aprendizado (todos convergem para as mesmas direções), não da falta de vitória. Homeostase sobre a frequência não resolve isso; o que falta atacar é a redundância por resposta (Fila 3, E08) ou a restrição ao inicial (E03 da Fila 1 já mostra posto 123–129 contra 72–77 do livre em fps 256).
- Um split e um sorteio: nenhuma diferença acima é conclusão. A grade completa (`disc_md_sol_hebb2`, 12 bases, 3 splits, 2 sorteios) espera numa sessão tmux própria até a `disc_md_sol_hebb1` terminar, a carga ficar abaixo de 110 e nenhuma GPU ter 12 GB nossos.

Arquivos: `/exps/moliveira/results_sol/alternative_encoders/svm_disc_md_sol_hebb{1,2}_piloto_*.csv`; kernels e vitórias por camada em `.../alternative_encoders/hebb/<job>/`; figuras da Fila 1 em `.../alternative_encoders/figs/`.

## Resumo de 08/11 (até 12:40 UTC)

Detalhes e números em `decisoes_spifil.md`; receita em `relatorio_08nov.md`.
- `spfcombo_w2` (whitening entre camadas nas 6 bases restantes, 3 splits): passa do aleatório em Eggs (0,845), Larvae, Protozoan, protozoan_cysts, helminth_larvae; remote_sensing empata.
- `sol_wsal` (whitening + saliência, 12 bases, 3 splits): todas as 24 células acima do aleatório, menos remote_sensing; saliência soma +0,02 a +0,025.
- Tabela de arrependimento: receita única `fpsw@mapa_w0.5_1`, L6 `cat`, 256 filtros, whitening entre camadas; pior arrependimento −0,029 entre 5 famílias.
- Com e sem saliência: sem saliência é igual ou melhor em aid e remote_sensing; com saliência ganha em flowers, larvas e ovos.
- `sol_p1` e `sol_p1_n6`: nenhuma normalização nova ganha da atual.
- `sol_p3`: diversidade pelas respostas não supera o `fps`.
- `sol_p5`: só a leitura multiescala (`cat_ms`) ganha (+0,013 a +0,029).
- `cl_a`: agrupamento com Z-score do patch perde do protocolo comum.
- `ms_a2`, `ms_b2`: multiescala e ponto de recorte empatam com o geodésico; S4 é o pior.
- `salw_a`: os 26 jeitos de usar a saliência ganham do `fps` (+0,015 a +0,034); foco com corte 0,25 é o mais estável.
- SoftHebb Filas 1 e 2 (piloto): não melhoram o SPiFiL; o banco fica mais repetitivo; a homeostase não corrige.
- Testes de fundo (P4): `fpsw@mapa_peso` é o mais robusto (κ retido 0,99 com fundo de cor sólida).
- coconut_trees (athena, parcial): ganha do aleatório; 64 filtros melhor que 256 (0,785 contra 0,667, L6, split 1).
- food101: piloto ok, 3 h 23 min por job; κ baixo (0,17 contra 0,15).
- Imagens por classe: parado no athena (sem teto de GPU), relançado no jaci às 12:10 com teto, 8 bases.
- Incidentes: GPU acima de 15 GB no athena durante a noite (imagens por classe sem teto e testes do SoftHebb); carga 160 às 02:00 (teste sem limite de threads e 8 sessões das bases novas); a sessão de food101 no athena foi barrada pela permissão.
