# Relações globais sem parâmetros: remote_sensing e flowers (11/11/2026)

**Resposta curta:** só uma coisa ajuda, e só no remote_sensing. É juntar ao `cat` da receita a covariância de várias camadas (`cat+mscp`): κ 0,492 → 0,547 (+0,055), ganho nos 3 de 3 splits, +0,060 acima do random_z puro. Na flowers nada melhora a receita: o melhor resultado empata.

## O que foi testado, em palavras simples
A receita olha cada pedacinho da imagem e depois tira uma média. Os mecanismos abaixo tentam fazer a rede ver "o que aparece junto com o quê", mesmo longe na imagem. Nenhum aprende peso, nenhum usa rótulo e nenhum usa backprop.

| mecanismo | o que faz (dia a dia) | onde entra | fonte |
|---|---|---|---|
| `cov60` | anota quais canais acendem juntos na L6: uma tabela 60×60 de "anda junto" | leitura da L6 | iSQRT-COV / MPN-COV, Li et al., CVPR 2018; Lin & Maji, BMVC 2017 |
| `mscp` | a mesma tabela, com 15 canais de cada uma das L3, L4, L5 e L6 empilhados na grade da L6 | leitura da L3 à L6 | MSCP, He et al., IEEE TGRS 2018 |
| `gram` | "o que está a d casas de quê": canal × canal entre um ponto e o vizinho a 1, 2 e 4 casas | leitura da L6 | Gram deslocado, Berger & Memisevic, ICLR 2017 |
| `fisher` | descreve cada ponto pela distância até 16 "tipos de ponto" (GMM), primeira e segunda ordem | leitura da L6 | Fisher vector, Cimpoi et al., CVPR 2015 |
| `ss` | cada ponto vira a média dos pontos parecidos com ele no mapa todo (cosseno), sem contar ele mesmo; a camada seguinte é construída sobre esse mapa | construção (modo b) em L3, L4 ou L5 | self-self: SCLIP (ECCV 2024), ClearCLIP (ECCV 2024), GEM (CVPR 2024) |
| `cat+cov60`, `cat+mscp` | **soma, não troca:** o `cat` da receita mais o bloco de covariância, lado a lado | leitura | idem |

Detalhes que mudam números:
- **Covariância:** PCA dos canais ajustada no treino sem rótulo (256 → 60 no `cov60`, 15 por camada no `mscp`). Raiz quadrada de matriz por Newton-Schulz, 5 iterações (não log-Euclidiano), depois sinal·raiz e L2. Fica com 1830 dimensões.
- **`gram`:** 24 PCs, 3 deslocamentos, 1728 dimensões. Os deslocamentos são 1, 2 e 4 casas da grade 11×11 da L6, cerca de 18, 36 e 73 px. O pedido era 8, 16 e 32 px, mas 8 px cabem em menos de uma casa (cada casa tem ~18 px).
- **`fisher`:** 32 PCs, GMM diagonal com 16 gaussianas ajustada em 50 mil pixels de treino, 1024 dimensões.
- **`ss`:** a temperatura é escolhida por imagem, sem rótulo, numa grade de 16 valores. Fica a que deixa a entropia média da atenção na metade do máximo. Diagonal mascarada, 2 passos, sem resíduo.
- **`cat+…`:** cada bloco passa por PCA até metade da dimensão (768 + 915) e depois pelo `pw_whiten` do próprio SVM (com shrinkage, mais L2). O SVM faz o `pw_whiten` de sempre no vetor junto. No remote_sensing o treino tem 344 imagens, então a dimensão efetiva fica em 344.
- **random_z:** fica sem mecanismo nenhum (régua pura). Ele não recebe as leituras de 2ª ordem, não ganha braço `ss` e não ganha TTA.

## Diagnóstico pedido: a `sa` antiga virava identidade?
Média de A_ii (o peso que cada ponto dá a si mesmo) da `sa` atual, medida em 16 imagens de treino do remote_sensing, split 1, nos encoders salvos:

| camada | receita fps: A_ii | random_z: A_ii |
|---|---|---|
| L3 (24×24, N=576) | 0.521 | 0.889 |
| L4 (11×11) | 0.650 | 0.968 |
| L5 (11×11) | 0.665 | 0.997 |
| L6 (11×11) | 0.555 | 1.000 |

No random_z a `sa` é praticamente identidade. Na receita, metade a dois terços do peso fica no próprio ponto, e com α = 0,5 a mistura real com os outros fica em torno de 15–25%. Isso explica boa parte do "neutro" da triagem anterior. Por isso a `ss` mascara a diagonal.

## Tabela (3 splits, 50%, 200 px, só validação)
Receita: fps no remote_sensing, fpsw@mapa_w0.5_1 na flowers. Δ é contra o `cat` da receita no mesmo encoder e no mesmo split. A régua é o random_z puro (`cat`).
"κ − random_z" é o κ da variante menos o κ do random_z puro, no mesmo split.
ms/img é o `extract_s` da linha da L6 dividido por (treino + validação). Nas leituras ele cobre todas as leituras da L6 juntas; no `ss` cobre a extração do braço inteiro. GPU MiB é o pico do braço (`max_memory_allocated`).
As linhas `cat+…` vêm de `svm_disc_md_relglob_somar_*`; o resto vem de `svm_disc_md_relglob_*`. A receita e o random_z saem iguais nos dois CSVs (κ 0,492 nos dois).

| base | posição | mecanismo | κ | Δκ vs receita | F1 | ΔF1 | splits com ganho | κ − random_z | ms/img | GPU MiB | CSV:linha (split 1, 2, 3) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| remote_sensing | L1–L6 | receita (cat GeM) | 0.492 | +0.000 | 0.459 | +0.000 | 0/3 | +0.005 | 72.0 | 796 | svm_disc_md_relglob_remote_sensing_s0200_split1_pct50_sh1.csv:22, svm_disc_md_relglob_remote_sensing_s0200_split2_pct50_sh1.csv:22, svm_disc_md_relglob_remote_sensing_s0200_split3_pct50_sh1.csv:22 |
| remote_sensing | L1–L6 | random_z puro (régua) | 0.487 | | 0.465 | | | | | | svm_disc_md_relglob_remote_sensing_s0200_split1_pct50_sh0.csv:22, svm_disc_md_relglob_remote_sensing_s0200_split2_pct50_sh0.csv:22, svm_disc_md_relglob_remote_sensing_s0200_split3_pct50_sh0.csv:22 |
| remote_sensing | L6 | só GeM da L6 (referência) | 0.383 | -0.109 | 0.375 | -0.084 | 0/3 | -0.104 | 72.0 | 796 | svm_disc_md_relglob_remote_sensing_s0200_split1_pct50_sh1.csv:20, svm_disc_md_relglob_remote_sensing_s0200_split2_pct50_sh1.csv:20, svm_disc_md_relglob_remote_sensing_s0200_split3_pct50_sh1.csv:20 |
| remote_sensing | L6, leitura | cov60 (iSQRT-COV) | 0.440 | -0.052 | 0.426 | -0.033 | 1/3 | -0.046 | 72.0 | 796 | svm_disc_md_relglob_remote_sensing_s0200_split1_pct50_sh1.csv:24, svm_disc_md_relglob_remote_sensing_s0200_split2_pct50_sh1.csv:24, svm_disc_md_relglob_remote_sensing_s0200_split3_pct50_sh1.csv:24 |
| remote_sensing | L3–L6, leitura | mscp (MSCP) | 0.518 | +0.026 | 0.489 | +0.030 | 3/3 | +0.031 | 72.0 | 796 | svm_disc_md_relglob_remote_sensing_s0200_split1_pct50_sh1.csv:26, svm_disc_md_relglob_remote_sensing_s0200_split2_pct50_sh1.csv:26, svm_disc_md_relglob_remote_sensing_s0200_split3_pct50_sh1.csv:26 |
| remote_sensing | L6, leitura | gram (Gram deslocado) | 0.390 | -0.101 | 0.393 | -0.066 | 0/3 | -0.096 | 72.0 | 796 | svm_disc_md_relglob_remote_sensing_s0200_split1_pct50_sh1.csv:28, svm_disc_md_relglob_remote_sensing_s0200_split2_pct50_sh1.csv:28, svm_disc_md_relglob_remote_sensing_s0200_split3_pct50_sh1.csv:28 |
| remote_sensing | L6, leitura | fisher (Fisher vector) | 0.415 | -0.077 | 0.386 | -0.073 | 0/3 | -0.072 | 72.0 | 796 | svm_disc_md_relglob_remote_sensing_s0200_split1_pct50_sh1.csv:30, svm_disc_md_relglob_remote_sensing_s0200_split2_pct50_sh1.csv:30, svm_disc_md_relglob_remote_sensing_s0200_split3_pct50_sh1.csv:30 |
| remote_sensing | L1–L6 + L6 | cat+cov60 | 0.481 | -0.011 | 0.444 | -0.015 | 1/3 | -0.006 | 57.2 | 796 | svm_disc_md_relglob_somar_remote_sensing_s0200_split1_pct50_sh1.csv:26, svm_disc_md_relglob_somar_remote_sensing_s0200_split2_pct50_sh1.csv:26, svm_disc_md_relglob_somar_remote_sensing_s0200_split3_pct50_sh1.csv:26 |
| remote_sensing | L1–L6 + L3–L6 | cat+mscp | 0.547 | +0.055 | 0.503 | +0.044 | 3/3 | +0.060 | 57.2 | 796 | svm_disc_md_relglob_somar_remote_sensing_s0200_split1_pct50_sh1.csv:30, svm_disc_md_relglob_somar_remote_sensing_s0200_split2_pct50_sh1.csv:30, svm_disc_md_relglob_somar_remote_sensing_s0200_split3_pct50_sh1.csv:30 |
| remote_sensing | L3, modo (b) | ss (self-self) | 0.493 | +0.002 | 0.475 | +0.016 | 1/3 | +0.007 | 69.4 | 796 | svm_disc_md_relglob_remote_sensing_s0200_split1_pct50_sh2.csv:22, svm_disc_md_relglob_remote_sensing_s0200_split2_pct50_sh2.csv:22, svm_disc_md_relglob_remote_sensing_s0200_split3_pct50_sh2.csv:22 |
| remote_sensing | L4, modo (b) | ss (self-self) | 0.490 | -0.001 | 0.456 | -0.003 | 2/3 | +0.004 | 69.5 | 796 | svm_disc_md_relglob_remote_sensing_s0200_split1_pct50_sh0.csv:44, svm_disc_md_relglob_remote_sensing_s0200_split2_pct50_sh0.csv:44, svm_disc_md_relglob_remote_sensing_s0200_split3_pct50_sh0.csv:44 |
| remote_sensing | L5, modo (b) | ss (self-self) | 0.501 | +0.009 | 0.468 | +0.009 | 2/3 | +0.014 | 66.8 | 796 | svm_disc_md_relglob_remote_sensing_s0200_split1_pct50_sh1.csv:52, svm_disc_md_relglob_remote_sensing_s0200_split2_pct50_sh1.csv:52, svm_disc_md_relglob_remote_sensing_s0200_split3_pct50_sh1.csv:52 |
| flowers | L1–L6 | receita (cat GeM) | 0.654 | +0.000 | 0.725 | +0.000 | 0/3 | +0.104 | 51.6 | 862 | svm_disc_md_relglob_flowers_s0200_split1_pct50_sh2.csv:52, svm_disc_md_relglob_flowers_s0200_split2_pct50_sh2.csv:52, svm_disc_md_relglob_flowers_s0200_split3_pct50_sh2.csv:52 |
| flowers | L1–L6 | random_z puro (régua) | 0.550 | | 0.642 | | | | | | svm_disc_md_relglob_flowers_s0200_split1_pct50_sh0.csv:22, svm_disc_md_relglob_flowers_s0200_split2_pct50_sh0.csv:22, svm_disc_md_relglob_flowers_s0200_split3_pct50_sh0.csv:22 |
| flowers | L6 | só GeM da L6 (referência) | 0.608 | -0.045 | 0.688 | -0.037 | 0/3 | +0.058 | 51.6 | 862 | svm_disc_md_relglob_flowers_s0200_split1_pct50_sh2.csv:50, svm_disc_md_relglob_flowers_s0200_split2_pct50_sh2.csv:50, svm_disc_md_relglob_flowers_s0200_split3_pct50_sh2.csv:50 |
| flowers | L6, leitura | cov60 (iSQRT-COV) | 0.578 | -0.076 | 0.669 | -0.056 | 0/3 | +0.028 | 51.6 | 862 | svm_disc_md_relglob_flowers_s0200_split1_pct50_sh2.csv:54, svm_disc_md_relglob_flowers_s0200_split2_pct50_sh2.csv:54, svm_disc_md_relglob_flowers_s0200_split3_pct50_sh2.csv:54 |
| flowers | L3–L6, leitura | mscp (MSCP) | 0.594 | -0.060 | 0.680 | -0.045 | 0/3 | +0.044 | 51.6 | 862 | svm_disc_md_relglob_flowers_s0200_split1_pct50_sh2.csv:56, svm_disc_md_relglob_flowers_s0200_split2_pct50_sh2.csv:56, svm_disc_md_relglob_flowers_s0200_split3_pct50_sh2.csv:56 |
| flowers | L6, leitura | gram (Gram deslocado) | 0.451 | -0.202 | 0.564 | -0.161 | 0/3 | -0.099 | 51.6 | 862 | svm_disc_md_relglob_flowers_s0200_split1_pct50_sh2.csv:58, svm_disc_md_relglob_flowers_s0200_split2_pct50_sh2.csv:58, svm_disc_md_relglob_flowers_s0200_split3_pct50_sh2.csv:58 |
| flowers | L6, leitura | fisher (Fisher vector) | 0.377 | -0.277 | 0.504 | -0.221 | 0/3 | -0.174 | 51.6 | 862 | svm_disc_md_relglob_flowers_s0200_split1_pct50_sh2.csv:60, svm_disc_md_relglob_flowers_s0200_split2_pct50_sh2.csv:60, svm_disc_md_relglob_flowers_s0200_split3_pct50_sh2.csv:60 |
| flowers | L1–L6 + L6 | cat+cov60 | 0.652 | -0.001 | 0.728 | +0.003 | 2/3 | +0.102 | 56.7 | 862 | svm_disc_md_relglob_somar_flowers_s0200_split1_pct50_sh0.csv:48, svm_disc_md_relglob_somar_flowers_s0200_split2_pct50_sh0.csv:48, svm_disc_md_relglob_somar_flowers_s0200_split3_pct50_sh0.csv:48 |
| flowers | L1–L6 + L3–L6 | cat+mscp | 0.643 | -0.010 | 0.723 | -0.002 | 1/3 | +0.093 | 56.7 | 862 | svm_disc_md_relglob_somar_flowers_s0200_split1_pct50_sh0.csv:52, svm_disc_md_relglob_somar_flowers_s0200_split2_pct50_sh0.csv:52, svm_disc_md_relglob_somar_flowers_s0200_split3_pct50_sh0.csv:52 |
| flowers | L3, modo (b) | ss (self-self) | 0.644 | -0.010 | 0.718 | -0.007 | 0/3 | +0.094 | 73.6 | 862 | svm_disc_md_relglob_flowers_s0200_split1_pct50_sh0.csv:74, svm_disc_md_relglob_flowers_s0200_split2_pct50_sh0.csv:74, svm_disc_md_relglob_flowers_s0200_split3_pct50_sh0.csv:74 |
| flowers | L4, modo (b) | ss (self-self) | 0.654 | +0.000 | 0.724 | -0.001 | 2/3 | +0.104 | 80.8 | 862 | svm_disc_md_relglob_flowers_s0200_split1_pct50_sh1.csv:82, svm_disc_md_relglob_flowers_s0200_split2_pct50_sh1.csv:82, svm_disc_md_relglob_flowers_s0200_split3_pct50_sh1.csv:82 |
| flowers | L5, modo (b) | ss (self-self) | 0.660 | +0.006 | 0.730 | +0.005 | 1/3 | +0.110 | 59.7 | 862 | svm_disc_md_relglob_flowers_s0200_split1_pct50_sh2.csv:82, svm_disc_md_relglob_flowers_s0200_split2_pct50_sh2.csv:82, svm_disc_md_relglob_flowers_s0200_split3_pct50_sh2.csv:82 |

Splits do vencedor (remote_sensing, κ, `cat` → `cat+mscp`; random_z puro): split 1 0,425 → 0,480 (0,447), split 2 0,577 → 0,626 (0,562), split 3 0,473 → 0,534 (0,451).

## Leitura
- **remote_sensing:** `mscp` sozinho já passa a receita (+0,026, 3/3). Somado ao `cat`, sobe para +0,055 (3/3). A covariância de uma camada só (`cov60`) perde para o `cat` e não soma. O ganho vem de juntar camadas: o que acende junto em L3 e em L6 ao mesmo tempo.
- **flowers:** nada passa a receita. `cat+cov60` empata (−0,001, 2/3), `cat+mscp` dá −0,010, e as leituras sozinhas perdem de −0,06 a −0,28.
- **`gram` e `fisher`:** perdem nas duas bases (até −0,28). Sozinhos, com um mapa 11×11, são ruidosos demais para 344 ou 963 imagens de treino.
- **`ss` (modo b):** neutro. O melhor é L5 (+0,009 no remote_sensing, +0,006 na flowers), com 1 a 2 splits de 3: está dentro do ruído. A figura mostra o porquê: o `ss` alisa o mapa da L5 numa mancha só, e a L6 construída em cima dele muda de forma, mas não fica mais informativa.
- Por que o `cat+mscp` não ajuda a flowers: lá a receita já está +0,10 acima do random_z, e o que falta é o objeto (a saliência), não a relação entre pontos distantes.

## Figura
`mapas_relacoes_globais.png` mostra o `ss` em L5 (o melhor mecanismo que muda o mapa; as leituras não mudam mapa nenhum). São 3 imagens de validação de cada base, no split 1. Colunas: a imagem (canal L), a L5 da receita, a L5 com `ss`, a L6 da receita e a L6 do braço `~ab_ss_5` (construída sobre o mapa atendido). Cada mapa é a média dos canais.

## Código (opt-in; saídas antigas idênticas)
`eval/spf_grade.py`:
- 109: import da `GaussianMixture`
- 1073-1121: `_gc`, `_ss` (com `SS_TAUS`), `Dilated`, e `fourier`/`gc`/`ss` no `ATTN`
- 1124-1131: `attach`
- 1161-1199: `isqrt_cov`, `shifted_gram`, `fisher`, `SECOND` e `MSCP_LAYERS`
- 1217: parâmetro `tta` de `readouts`
- 1241-1278: PCA de pixels, `second` e GMM
- 1290-1312: as leituras `SECOND` e a TTA das leituras (`amostra.leituras_tta`)
- 1423-1424: o rótulo `~ab_<mec>_<pos>`
- 1522-1523: `attach` no modo (b)
- 1529-1531: o random_z sem mecanismo nem TTA
- 1537: o modo (a) só com `modo: a`
- 1568, 1589, 1598-1603: `seconds`, `cats` e `cat+<r>` (`amostra.somar`)
- 1641-1642: braços do modo (b) em `tails`
- 1667-1668: o random_z fora do modo (b)
- 1716, 1720: contagem esperada
- 1739, 1814: os gêmeos sem o rótulo `ab_`

`experiments/ray/schema.py` 244-251: `modo: b` e os mecanismos novos.

Identidade: rodei uma célula pequena (aid, largura 16, 2 camadas, fps + random_z) com o código antigo e com o novo, sem as opções novas. Sem `leituras`, 12/12 linhas e 35/35 de diagnóstico são iguais fora das colunas de tempo; com `leituras: [gem, cov]`, 24/24 e 35/35. Os imports de `eval.spf_grade`, `eval.alt_encoders` e `eval.encdeep` passam antes e depois.

## YAMLs
- `experiments/autoencoder/disc_md_relglob.yaml`: as 4 leituras + `ss` em L3/L4/L5. 18 jobs, rc 0, de 01:03 a 01:45.
- `experiments/autoencoder/disc_md_relglob_somar.yaml`: `cat+cov60` e `cat+mscp`. 12 jobs, rc 0, de 01:54 a 02:19. A primeira tentativa fazia whitening por componente sem shrinkage e caiu para κ≈0; parei e refiz.
- `experiments/autoencoder/disc_md_relglob_320tta.yaml`: remote_sensing a 320 px + TTA + `cat+mscp` (seção abaixo).
- `experiments/autoencoder/disc_md_relglob_somar_bases_jaci.yaml`: o mesmo protocolo do `somar` em aid, ham10000 e coconut_trees, para o jaci (`GPU_CAP_MIB=4200`). Dry-run com 18 jobs; ainda não rodou. Serve para ver se o ganho do `mscp` é uma lei de cena ou um caso isolado.

## Não feito
- aid e ham10000, a pedido (ficam para depois).
- `fourier`, `gc` e `dilat`: o código está pronto (`atencao.modo: b`), mas não rodaram; passaram para depois destes 5.
- NMF (Hamburger): não rodou.
- Retomada: com `leituras` de 2ª ordem, o random_z grava menos linhas que o `expected` global. Se um run for morto e relançado, o random_z é refeito (custo de um braço).

## Último teste: remote_sensing a 320 px + TTA + mscp (pedido: "chegamos perto de 0,60?")
**Resposta curta: não.** O melhor resultado novo é `mscp` com TTA a 320 px, κ 0,553. A melhor combinação anterior (320 + espacial@L5 + TTA, κ 0,571) continua na frente, e 0,60 fica longe.

Neste teste a TTA passa pela leitura nova: o `cat` e o `mscp` são a média das 8 vistas (`amostra.leituras_tta`). Para conferir, o `cat` com TTA deste run sai igual ao `cat_tta` do combo320, linha por linha (0,536/0,595/0,504). Receita fps sem saliência, como no combo320. O random_z fica sem TTA e sem mecanismo. Δ é contra a receita a 200 px (`cat`), no mesmo split.

| variante | splits | κ | Δκ vs receita 200 | F1 | splits com ganho | κ − random_z 200 puro | κ por split | CSV:linha (split 1, 2, 3) |
|---|---|---|---|---|---|---|---|---|
| receita 200 px `cat` | 3 | 0.492 | +0.000 | 0.459 | – | +0.005 | 0.425/0.577/0.473 | svm_disc_md_relglob_somar_remote_sensing_s0200_split{1,2,3}_pct50_sh1.csv:22 |
| random_z 200 px puro | 3 | 0.487 | −0.005 | 0.465 | – | – | 0.447/0.562/0.451 | svm_disc_md_relglob_somar_remote_sensing_s0200_split{1,2,3}_pct50_sh0.csv:22 |
| random_z 320 px puro | 3 | 0.491 | −0.000 | 0.459 | – | +0.005 | 0.481/0.529/0.465 | svm_disc_md_relglob_320tta_remote_sensing_s0320_split{1,2,3}_pct50_sh0.csv:22 |
| 200 px `cat+mscp` | 3 | 0.547 | +0.055 | 0.503 | 3/3 | +0.060 | 0.480/0.626/0.534 | svm_disc_md_relglob_somar_remote_sensing_s0200_split{1,2,3}_pct50_sh1.csv:30 |
| 320 px + TTA `cat` | 3 | 0.545 | +0.053 | 0.512 | 3/3 | +0.058 | 0.536/0.595/0.504 | svm_disc_md_relglob_320tta_remote_sensing_s0320_split{1,2,3}_pct50_sh1.csv:22 |
| 320 px + TTA `mscp` | 3 | **0.553** | +0.062 | 0.535 | 3/3 | +0.067 | 0.520/0.593/0.547 | svm_disc_md_relglob_320tta_remote_sensing_s0320_split{1,2,3}_pct50_sh1.csv:24 |
| 320 px + TTA `cat+mscp` | 3 | 0.536 | +0.044 | 0.509 | 3/3 | +0.049 | 0.507/0.592/0.509 | svm_disc_md_relglob_320tta_remote_sensing_s0320_split{1,2,3}_pct50_sh1.csv:26 |
| 320 px + espacial@L5 + TTA (melhor anterior) | 3 | **0.571** | +0.079 | 0.537 | 3/3 | +0.084 | 0.570/0.617/0.526 | svm_disc_md_atencao_combo320_remote_sensing_s0320_split{1,2,3}_pct50_sh1.csv:44 |

Leitura:
- A 200 px, o `mscp` soma bem com o `cat`. Já a 320 px com TTA, o `cat+mscp` fica abaixo do `mscp` sozinho. Com 344 imagens de treino, o vetor junto cabe em no máximo 344 dimensões, e somar blocos passa a custar mais do que ajuda.
- O `mscp` com TTA a 320 px (0,553) quase empata com o `cat` com TTA (0,545), mas fica abaixo do espacial@L5 com TTA (0,571). Nenhuma combinação chega a 0,60. O melhor split chega a 0,626 (split 2, `cat+mscp` a 200 px), mas a média não.
- Para chegar a 0,60, o próximo candidato seria espacial@L5 + `mscp` com TTA, que hoje não roda junto: o modo (a) da atenção fica desligado quando há `leituras`.

YAML: `experiments/autoencoder/disc_md_relglob_320tta.yaml`. 6 jobs, rc 0, de 02:21 a 02:41, `GPU_CAP_MIB` 5000.
