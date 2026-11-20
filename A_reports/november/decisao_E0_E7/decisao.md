# Decisão E0–E7 nas 4 bases de foco (09/11)

Só validação. Δκ = diferença **pareada** (mesmo split e sorteio) contra a âncora da **própria família** (`fps` ou receita), nunca entre famílias. n = splits × sorteios. Splits se sobrepõem (~70%): n = 6 não são 6 provas independentes.

## As 8 perguntas

**E0 "Qual é a referência?"** Receita `sol_wsal`, L6 `cat`, 256, whitening entre camadas, 200 px (fps em aid/rs, `fpsw@mapa_w0.5_1` em flowers/coco).
- flowers **0,681** (n=3) · coconut_trees **0,625** (n=1) · aid **0,773** (n=3) · remote_sensing **0,493** (n=3).
- Melhor aleatório: 0,550 / 0,580 / 0,638 / 0,487. DINOv2: 0,973 / — / 0,970 / 0,708.
- Em palavras: o SPiFiL ganha do aleatório em todas, mas em remote_sensing só por um fio (+0,006).

**E1 "Falta campo de visão?"** (`cat_ms_eq` − `cat`, mesma dimensão, `leitura_a2`, fps, L3, 256)
- flowers **Não** (−0,002, n=6, 3/6) · coconut_trees **Não testado** → `disc_md_e07_on.yaml` · aid **Sim** (+0,020, n=6, 6/6) · remote_sensing **Sim** (+0,030, n=6, 6/6).
- Em palavras: olhar a imagem também "de longe" ajuda as fotos aéreas; nas flores o ganho do `cat_ms` sem igualar a dimensão (+0,019 com fpsw, `sol_p5`) some quando a dimensão é igual, então era só tamanho do vetor.

**E2 "Os filtros são redundantes?"** (f2–f6 − `fps`, `sol_p3`, L3 `cat`, 256)
- flowers **Não** (melhor f5 +0,001, n=6) · coconut_trees **Inconclusivo** (f2 −0,025, n=1, só L1 64, job parcial) · aid **Sim, fraco** (f3 +0,028, n=6, 6/6; f5 +0,020; mas f4 −0,035 e f6 −0,092) · remote_sensing **Não** (análise L1 nova: S1–S3 − fps = −0,076 a 256, n=1; a 64 o S1 dá +0,078 — instável).
- Em palavras: trocar o fps por filtros "de respostas diferentes" só ajuda em aid, e é o melhor de 5 variantes (algum otimismo).

**E3 "A localização espacial está sendo perdida?"** (`L_spp` − `L`, `sol_p5`, L3, 256)
- flowers **Não** (−0,050, n=6, 0/6; `cat_spp` −0,079) · coconut_trees **Não testado** → `disc_md_e07_on.yaml` · aid **Não** (−0,054, n=6, 0/6) · remote_sensing **Não testado** → `disc_md_e07_off.yaml`.
- Em palavras: guardar "onde" cada coisa está piora; o vetor fica grande demais (>1300-d) para o SVM.

**E4 "A seleção local é insuficiente?"** (contexto nos pontos − receita, `sol_wsal`, L6 `cat`, 256)
- flowers **Não** (foco_w0.5_1 +0,009, n=3) · coconut_trees **Inconclusivo** (`fps@combo` +0,042, n=1; mas em `salw_a`, L3, foco −0,010, n=6, 2/6) · aid **Não** (+0,002, n=3) · remote_sensing **Inconclusivo** (`fps@combo` +0,013 ± 0,030, n=3, 2/3).
- Pontos multiescala (`ms_a`/`ms_a2`, 64, L3, vs geodésico): S8 +0,012 flowers, +0,006 aid, +0,008 rs (n=15) — empate.
- Em palavras: escolher os recortes com mais contexto não muda quase nada.

**E5 "A representação perde diversidade?"** (com − sem whitening entre camadas; `spfcombo_w` − `spfcombo_a`, `fps@combo`, L6 `cat`, 256)
- flowers **Sim** (+0,075, n=3, 3/3) · coconut_trees **Não** (+0,002, n=1) · aid **Sim** (+0,055, n=3, 3/3) · remote_sensing **Inconclusivo** (encdeep, n=1: tirar o whitening em L4/L5 custa −0,020 no `cat` L5, mas **ganha +0,057 na L5 isolada**; em aid +0,035 na L5 isolada).
- Em palavras: o whitening entre camadas já está na receita e é o que mais rendeu; nas camadas fundas, lidas sozinhas, ele parece atrapalhar.

**E6 "Os mecanismos são complementares?"** (multiescala + SPP)
- **Não testado** nas 4. E não cabe em YAML: em `eval/spf_grade.py` cada item de `leituras` (`ms`, `spp`, …) vira uma leitura **separada**; não existe "SPP em várias escalas". Mudança mínima proposta (não feita): em `readouts`, um ramo `ms_spp` igual ao `ms` trocando `gem` por `POOLS["spp"]` (≈3 linhas). Dado que SPP perde −0,05 sozinho, a prioridade é baixa.

**E7 "A combinação supera as melhorias isoladas?"** (multiescala + diversidade + SPP)
- **Não testado** nas 4. A parte sem SPP (multiescala + f3/f5) cabe e está em `disc_md_e07_{off,on}.yaml` (f3/f5 lidos em `cat_ms_eq`). A versão com SPP precisa da mesma mudança do E6.

## Gráficos

- `heatmap_delta.png`: Δκ por base × E1–E7, com n; hachurado = não testado.
- `barras_kappa.png`: κ absoluto de E0–E7, melhor aleatório e DINOv2. As barras E1–E5 são da própria família (L3 ou L6): leia o Δ, não a altura.

Contexto (fora de E1–E7):
| fato | aid | remote_sensing | flowers | coconut_trees |
|---|---|---|---|---|
| resolução, split 1, fps `cat`: 200 / 224 / 320 px | 0,786 / 0,784 / 0,798 | 0,432 / 0,474 / 0,510 | — | — |
| 320 px confirmado, 3 splits (Δ) | +0,011 | +0,036 | — | — |
| leitura `L` − `cat`, receita L6 (`sol_wsal`) | −0,146 (n=3) | −0,077 (n=3) | −0,067 (n=3) | **+0,169** (n=1; 0,788 vs 0,630 no `vlad_on`, n=2) |
| SoftHebb (rede pequena, L3, split 1): aleatório / fps / fps+SoftHebb | 0,724 / 0,704 / 0,759 | 0,491 / 0,451 / 0,498 | 0,623 / 0,581 / 0,629 | 0,677 / 0,726 / 0,719 |

SoftHebb: tudo abaixo da receita (E0); dentro da receita (encdeep) empata com a atualização zero.

## O que fazer agora

1. **coconut_trees: ler só a última camada (`L`).** +0,169 (n=1) e +0,158 (n=2, `vlad_on`): é o maior número da tabela. Falta confirmar em 3 splits e achar uma regra sem rótulo para quando usar `L`.
2. **aid e remote_sensing: subir para 320 px.** +0,011 e +0,036, positivo nos 3 splits. Já confirmado; só adotar.
3. **aid e remote_sensing: adotar `cat_ms_eq`.** +0,020 e +0,030, 6/6 nas duas. Próximo passo: medir junto com 320 px (os dois mexem no campo de visão; podem se sobrepor).
4. **Rodar `disc_md_e07_off` e `disc_md_e07_on`** (8 jobs): preenche E1 e E3 em coco, E3 em rs e testa multiescala + f3/f5 (E7 sem SPP) nas 4.
5. **Parar** SPP (−0,05), contexto nos pontos (|Δ| ≤ 0,013) e diversidade funcional fora de aid. E6/E7 com SPP só se o item 4 mostrar que SPP ajuda em rs ou coco.

## Células NÃO testadas

- E1: coconut_trees → `disc_md_e07_on.yaml`.
- E3: coconut_trees → `disc_md_e07_on.yaml`; remote_sensing → `disc_md_e07_off.yaml`.
- E6: as 4 bases — precisa do ramo `ms_spp` em `spf_grade.readouts` (não escrito).
- E7: as 4 bases — só a parte multiescala + diversidade está nos YAMLs `e07`; com SPP, idem E6.
- Fracas (n=1): E2 coco, E2 rs (só L1), E4 coco, E5 coco, E5 rs.

## YAMLs novos (dry-run ok, 0 erros no preflight)

| YAML | bases | braços | leituras | jobs |
|---|---|---|---|---|
| `experiments/autoencoder/disc_md_e07_off.yaml` (+ `_jaci`) | aid, remote_sensing | `fps`, f3, f5; controles `random_z`, `orth` | `gem`, `ms`, `spp`, `eq: true` | 4 (2 bases × 2 shards) |
| `experiments/autoencoder/disc_md_e07_on.yaml` (+ `_jaci`) | flowers, coconut_trees | `fpsw@mapa_w0.5_1`, f3, f5; idem | idem | 4 |

Split 1, sorteios 42 e 43, 256 filtros, L6, `entre_camadas`, 200 px. f3/f5 não levam pontos de saliência (F2–F6 são métodos à parte em `spf_grade`). Lançar: `python -m experiments.ray.launch experiments/autoencoder/disc_md_e07_off.yaml` (e `_on`).

Fontes: CSVs em `/exps/moliveira/results{,_sol}/` e `/exps/moliveira/recebido_jaci/exps/results{,_sol}/` (linhas de shard sem duplicatas), `encdeep/G3_pilotos.md`, `analise_textura/selecao_diversidade/tabelas.md`, `/exps/moliveira/reports/rodada2/pool_foco.md`. Script de leitura e números brutos: scratchpad `e07.py` / `e07.json`.
