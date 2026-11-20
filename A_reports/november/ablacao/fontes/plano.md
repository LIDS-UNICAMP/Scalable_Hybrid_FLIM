# Rodada 2 do SPiFiL sem backprop: plano (08/11/2026)

Restrições: sem backprop, sem pré-treino, sem rótulo nem máscara na construção, seleção ou atualização dos filtros; filtros e estatísticas só com dados de treino; teste nunca entra em escolha; os regimes ficam separados (fixo não supervisionado, aprendizado local, aleatório). Não sobrescrever resultados, não duplicar jobs, não interromper processos.

Seções: 1 auditoria; 2 leitura (A0 GeM p=3, A1 GeM p=2, A2 multiescala); 3 diversidade funcional (B0 a B3); 4 SoftHebb amplo (substituída pelo usuário, ver abaixo); 5 Scattering (D0 a D2); 6 triagem e confirmação; 7 escala arquitetural.
Bases da triagem: uma de parasitos com ganho do FPS, uma de parasitos variável, aid, flowers, remote_sensing. Bases completas e subgrupos não são evidências independentes. Confirmação: 3 splits × pelo menos 5 sementes.

## Seção 4: investigação ampla do SoftHebb
- 4.1 Validar a implementação contra os artigos e o código oficial (Moraitis et al. 2022, *SoftHebb: Bayesian inference in unsupervised Hebbian soft WTA networks*; Journé et al. 2023, *Hebbian deep learning without feedback*). Verificações: atualização contra referência explícita; lr = 0 preserva pesos; separar pesos de estatísticas; valores finitos e reprodutibilidade; encoder congelado na extração. Rotular (a) referência, (b) adaptação atual, (c) variantes novas.
- 4.2 H00 a H06: aleatório e FPS fixos, com e sem calibração das estatísticas, com SoftHebb, e geodésico + SoftHebb (reaproveitado). Reescala de norma do FPS só como variante separada.
- 4.3 H07 a H12: taxa (constante, decrescente, aquecimento + decaimento) e temperatura (suave, seletiva, recozida). Monitorar entropia, vitórias, filtros efetivos, norma da atualização, deriva, redundância, cobertura; duração em atualizações, imagens, patches, passagens e tempo; checkpoints fixos de antemão.
- 4.4 H13 a H18: por camada (só a 1ª, 1ª fixa, só a última, sequencial, simultânea, congelamento progressivo), com orçamentos por apresentações, atualizações e tempo.
- 4.5 H19 a H24: normalização oficial, calibrada e congelada, atualizada; limite de deslocamento, restrição angular, magnitude relativa; uma restrição por vez.
- 4.6 H25 a H29: amostragem uniforme, limite por imagem, diversa, regiões mal cobertas, mistura de comuns e raros.
- 4.7 H30 a H35: homeostase (reaproveita hebb2), inativos, proteção de raros, redundantes, substituição por regiões mal cobertas, aceitação em dados sem rótulo separados do treino.
- 4.8 H36 a H38: banco parcialmente fixo, bancos paralelos, subconjunto escolhido; comparar com FPS do mesmo tamanho total.
- 4.9 Leitura dos finalistas: última camada, concatenação, leitura de custo comparável; por camada e por classe; estabilidade.
- 4.10 Etapas A (implementação e calibração), B (taxa, temperatura, duração), C (camada e normalização), D (amostragem, proteção, banco parcial), E (combinações e confirmação). Sem produto cartesiano.
Entrega: tabela hipótese → variante → controle → orçamento → resultado → incerteza → decisão, e as 10 perguntas.

## Regra do usuário (08/11): o aleatório fica sozinho
O aleatório é régua pura: nenhum braço "aleatório + técnica". Saem do plano H02 (aleatório + calibração) e H04 (aleatório + SoftHebb); em D (Scattering) e nas leituras, o aleatório entra só como controle puro no mesmo protocolo de avaliação. Toda comparação é SPiFiL melhorado contra o aleatório puro e contra o DINOv2.

## Etapa 1 pronta (08/11, ~14:40 UTC): pilotos no jaci
Mesmo split (1) e sorteio (42) em todas as comparações pareadas; κ, F1 macro e acurácia balanceada; escolha só na validação.
- `disc_md_sol_hebb3_jaci.yaml`, 5 jobs: H01 (fps fixo, 1 lote de calibração), H03 (fps só calibração), H05 (fps + SoftHebb na receita dos autores: inv_temp 1 e 0,65, lr 0,08 e 0,005, delta_norm tensor, 300 atualizações por camada; cerca de 1,2 passada). Hipótese: a receita dos autores reduz a deriva e não perde do fps fixo. Avanço: H05 − H01 ≥ 0 em pelo menos 3 de 5 bases e na média, sem perder mais de 0,02 em nenhuma família. H06 (geodésico + SoftHebb) e o aleatório fixo vêm do `sol_hebb1`.
- `disc_md_leitura_a2_jaci.yaml`, 18 jobs: A0 (GeM p3), A1 (GeM p2), A2 (`ms`, `cat_ms` e as versões `_eq` com a mesma dimensão, reduzidas pela PCA de treino). Hipótese: a multiescala ganha por informação de escala, não por dimensão. Avanço: `cat_ms_eq` − `cat` > 0 em pelo menos 4 de 6 bases, e em pelo menos 3 das 5 famílias.
- Código: `eval/alt_encoders.py` (chaves novas em `amostra.hebb_init`, desligadas por padrão, iguais bit a bit ao código antigo) e `eval/spf_grade.py` (`amostra.eq`, `pca_fit`).

## Resultado do piloto `sol_hebb3` (athena, 14:55 a 15:03 UTC; 5 bases, split 1, sorteio 42, L3)
κ de validação, H05 (SoftHebb na receita dos autores) − H01 (fps fixo):
- `gem_pw` (última camada): 64 filtros +0,031, +0,039, +0,085, −0,003, +0,031; 256 filtros +0,058, +0,072, +0,079, +0,079, +0,048 (Eggs, Larvae, aid, flowers, remote_sensing).
- `cat_gem_pw`: 64 filtros +0,045, +0,018, +0,076, +0,026, +0,035; 256 filtros +0,045, **−0,057 (Larvae)**, +0,069, +0,019, +0,076.
- Médias (`cat`): 64 filtros κ 0,585 → 0,625, F1 0,663 → 0,690, acurácia balanceada 0,661 → 0,694; 256 filtros κ 0,642 → 0,673, F1 0,699 → 0,721, acurácia balanceada 0,690 → 0,711.
- H03 (só calibração, 300 lotes) ≈ H01: médias −0,017 e −0,002. O ganho vem da atualização, não da calibração.
- Critério fixado antes: passa em `gem_pw` (≥ 3 de 5 bases e média ≥ 0, nenhuma perda > 0,02). Em `cat` com 256, Larvae perde 0,057: falha numa célula. Com n = 1 split × 1 sorteio, é sinal exploratório.
- Diferença para o `sol_hebb1` (que perdia): competição mais suave (inv_temp 1 e 0,65, contra 4), teto do passo por tensor e lr por camada.
- Confirmação lançada às 15:04 UTC: `disc_md_sol_hebb3_conf.yaml`, 3 splits × 5 sorteios, mesmos braços, GPUs 0 e 1 com teto de 4,2 GB por processo.
- Ressalvas: é a rede de 3 camadas do `alt_encoders`, não a receita de 6 camadas com whitening; o diagnóstico de deriva (cos_init, erank) não foi gravado nesses braços.

## Resultado da leitura A0, A1, A2 (`leitura_a2`, athena, END 17:15 UTC rc=0; 6 bases × 3 splits × 2 sorteios; diferença pareada no κ contra `cat`)
- A2 controlado (`cat_ms_eq`, PCA de treino até a dimensão de `cat`, mediana 219):
  - fps 256: Eggs_sub +0,009, Larvae_sub −0,013, Protozoan_sub +0,029, aid +0,020, flowers −0,002, remote_sensing +0,030 (4 de 6 bases; famílias: ovos, protozoários e aid + rs sobem, larvas e flowers não, 3 de 5).
  - fps 64: +0,063, +0,020, +0,001, +0,025, −0,008, +0,022 (5 de 6; 4 de 5 famílias).
  - **Critério fixado antes: PASSA** (≥ 4 de 6 bases e ≥ 3 de 5 famílias), no limite em 256.
- A2 completo (`cat_ms`, 576-d): só um pouco acima do `_eq` (fps 256: +0,012, −0,013, +0,037, +0,025, −0,004, +0,030). A maior parte do ganho vem da informação de escala, não da dimensão.
- A1 (GeM p=2 contra p=3): ±0,03, sem direção consistente no fps; encerrada.
- Aleatório puro com as mesmas leituras: também ganha com `cat_ms_eq` em parte das bases (aid +0,006, rs +0,010, Eggs_sub +0,028 em 256), mas menos que o fps em aid, remote_sensing e Protozoan_sub.
- Custo: `extract_s` é o mesmo para todas as leituras (as três escalas saem de uma passada e o tempo é medido por job, não por leitura); a PCA do `_eq` custa pouco. Medição de custo por leitura: pendente.
- Decisão: `cat_ms_eq` avança como leitura candidata para a confirmação com a receita completa (L6, whitening entre camadas), junto com o SoftHebb e as leis.

## Confirmação do SoftHebb (`sol_hebb3_conf`, athena, END 17:33 UTC rc=0; 5 bases × 3 splits × 5 sorteios, 1800 linhas ok, L3; diferença pareada H05 − H01, n = 15 por célula)
`cat_gem_pw`, κ (fração de pares em que H05 vence):
- 256 filtros: Eggs +0,026 (87%), Larvae +0,052 (67%, desvio 0,093), aid +0,064 (100%), flowers +0,036 (87%), remote_sensing +0,026 (73%).
- 64 filtros: Eggs +0,003 (47%), Larvae +0,007 (60%), aid +0,073 (100%), flowers +0,014 (73%), remote_sensing +0,053 (87%).
- F1 macro e acurácia balanceada seguem o mesmo sinal: com 256, +0,021 a +0,060 de F1 e +0,020 a +0,059 de acurácia em todas as bases.
- `gem_pw` (só a última camada), 256: +0,041 / +0,037 / +0,078 / +0,032 / +0,027, todas positivas.
- H03 (só calibração) − H01: entre −0,024 e +0,016; o efeito é da atualização dos pesos, não da calibração.
**Decisão: CONFIRMADO com 256 filtros, nas 5 bases e nas 5 famílias** (o critério pedia ≥ 3 de 5 bases e média ≥ 0, sem perda > 0,02 em família). Com 64 filtros, ganho claro só em aid e remote_sensing.
Limites: rede de 3 camadas do `alt_encoders` (não a receita de 6 camadas com whitening e saliência); os splits se sobrepõem, então a família é a unidade; Larvae varia muito.
Próximo passo condicionado: FPS + SoftHebb (receita dos autores) dentro da receita SPiFiL completa e nas leis finalistas (P6).

## Leis P1, athena, sorteio 42 (`leis_p1`, END 19:20 UTC rc=0; split 1; `cat` na última camada; âncora uniforme 256×6 do `sol_wsal`, mesmo split e seed)
- aid (fps, âncora 0,786): melhor A 512-512-256-256-128-128 0,788 (12,7M parâmetros, 1,5 vez a âncora de 8,2M). Leis magras perdem de 0,06 a 0,09. O aleatório na mesma arquitetura fica entre 0,64 e 0,71.
- flowers (receita, âncora 0,656): melhor a mesma A 512 r0,71, 0,659. Todas as outras ficam abaixo da âncora.
- remote_sensing (fps, âncora 0,432): **todas as 17 leis passam a âncora**, de +0,006 a +0,060; a melhor é C gradual 512-256-256-128-64-32, 0,492 com 6,0M. Mas o aleatório na mesma arquitetura também sobe (0,42 a 0,50) e às vezes passa o fps: o ganho é da arquitetura, não do SPiFiL.
Leitura (1 split, 1 sorteio, exploratória):
- aid e flowers: com o mesmo orçamento, nenhuma lei bate a uniforme; só a A com 1,5 vez os parâmetros empata.
- remote_sensing: redes que começam com 512 e afunilam ajudam todos os métodos, inclusive o aleatório.
- A regra "começar largo (512) e reduzir devagar (r ≈ 0,71)" é a melhor nas 3 bases, mas não supera a uniforme com orçamento igual.
- Falta o sorteio 43 e a comparação com orçamento igual (C).
- Sorteio 43 (`leis_p1_s43`, END 19:25 UTC): concordância da ordem das 17 leis entre os sorteios 42 e 43 (Spearman): aid 0,94; flowers 0,64; remote_sensing −0,07.
  - aid: a ordem é estável. A 512 r0,71 d6 fica em primeiro nos dois sorteios (0,786).
  - flowers: A 512 r0,71 d6 também fica em primeiro na média (0,644), mas com mais ruído.
  - remote_sensing: a ordem é ruído; as leis ficam entre 0,47 e 0,49 e o aleatório perto disso. A diferença contra a âncora (0,432, só o sorteio 42) precisa de confirmação.

## Leis P1, bases menores no jaci (`leis_p1b_jaci`, END 20:59 UTC rc=0; split 1, seed 42; tabela em `/dados/home/moliveira/exps/reports/leis_p1b_jaci_por_lei.txt` no jaci)
- `cat`: a receita fica ≥ fps nas 4 bases (mediana +0,007 a +0,032) e muito acima do aleatório (+0,15 a +0,39). As diferenças entre leis são de ±0,03, menores que o ruído esperado entre sorteios.
- L (última camada): a lei A com r = 0,71 (largura cai devagar) é a melhor em 3 de 4 bases. Exemplo: Eggs_subgroup, receita w256 d4, 0,865 com r = 0,71 contra 0,739 com r = 0,5. Cortar pela metade a cada camada prejudica a leitura da última camada.
- Regra provisória das leis (parasitos + aid + flowers): começar largo (256 a 512) e reduzir devagar (metade a cada 2 camadas). Com `cat`, a forma importa pouco; com L, ela importa muito.
