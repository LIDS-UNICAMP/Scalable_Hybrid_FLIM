# A1c — Generalização (item E) e diferenças entre domínios (análise quantitativa 8)

_Agente 1c, 16–17/11/2026. Só leitura, CPU com nice, nenhum arquivo de produção alterado. Dados: `A1c_data.csv` (formato longo: `secao, base, split_json, metrica, valor, extra, fonte`) e `A1c_dup_pairs.csv` (um par por linha: imagens, splits, rótulos, nome raw, md5, corr, dHash, nível, veredito visual). Scripts em `A1c_scripts/` e montagens de evidência em `A1c_evidencias/`. Legenda das afirmações: **[M]** medido, **[C]** correlação, **[H]** hipótese._

## 0. Achados críticos

1. **[M] Há vazamento de conteúdo entre splits nas bases de parasitas: o mesmo objeto físico aparece em duas imagens.** O mesmo ovo ou larva foi recortado duas vezes, com enquadramento levemente deslocado, e cada recorte caiu num split. O md5 não pega, e a hash da imagem raw inteira também não. Aparece no pre (objeto alinhado e centrado) e foi confirmado olhando as imagens raw correspondentes (`A1c_evidencias/eggs_raw_of_pre.png`, `larvae_raw_of_pre.png`). Pares confirmados visualmente no `split1_modehb`:
   - eggs: 7 test↔train, 5 val↔train e 1 val↔test (inclui 1 cópia byte a byte val↔train, raw `000007_00008946`=`000007_00009157`);
   - larvae: 2 test↔train e 4 val↔train.

   Esses números são **limites inferiores**: só o topo dos candidatos foi auditado, e o recall não é conhecido. No impacto direto no κ de teste o efeito é pequeno: ≥0,37 % do teste de eggs e ≥0,83 % do teste de larvae. **O problema é estrutural:** não existe id de lâmina/campo, então o split por imagem não consegue agrupar recapturas. No protozoan pre a hash 32×32 não discrimina nada (1,45 M pares "quase-idênticos"; 0/8 confirmados), então a questão fica **indeterminada** ali.
2. **[M] HAM10000.**
   - No `split1_modehb`, train e val não compartilham nenhuma lesão (0 lesões, 0 imagens).
   - O test herdado não está limpo: **907 lesões** do train e **181** do val estão também no test. Isso dá 1091 + 221 = **1312 imagens de test (26,2 %)** cuja lesão também está no train ou no val.
   - O vazamento pesa mais nas classes raras. Fração do test com lesão no train: MEL 41 %, DF 42 %, BCC 32 %, contra NV 16 %.
   - No `split1` original, val ≡ train (5007 imagens, 4284 lesões), e 1088 lesões / 1312 imagens do test estão no train. Nos splits 2 e 3 o padrão é o mesmo (1365 e 1315 imagens).
   - Há ainda **1 duplicata de imagem com `lesion_id` diferente**: ISIC_0025226 (train, IL_1299355) = ISIC_0030074 (test, IL_8593443). É a mesma lesão, com o mesmo pelo (`A1c_evidencias/ham.png`). Agrupar por `lesion_id` não basta.
3. **[M] O benchmark dos FMs no HAM usa outro protocolo.**
   - Os FMs treinam o SVM em **5007 imagens (o nosso train+val)**, e não nas nossas 4129.
   - O "κ val" deles é 1,000 porque val == train.
   - O test é o mesmo, mas o vazamento por lesão é maior para os FMs (1312 imagens de test com lesão no conjunto de treino deles, contra 1091 para nós).
   - A comparação nosso × FM no HAM não é igual (mais dados e mais vazamento a favor dos FMs).
4. **[M] A comparação "nosso pre × FM raw" é enviesada *a favor do nosso*.** Os FMs em pre (fundo preto) já existem no athena (`benchmark/modehb_pre`, 3 de 5 FMs) e são **melhores** que em raw em todos os 9 casos (+0,010 a +0,085). Exemplos: protozoan RN50 0,822→0,907 e CNXv2p 0,853→0,921; larvae I-JEPA 0,919→0,968. Os Δ "⚠" de §0/§4.3/§4.7/§4.9 de `all_results_today.md` **subestimam** a distância.
5. **[M] Há incoerência entre a régua da seleção e a régua do reporte.** O "sel-val" escolhe o candidato pelo κ de val da **fitness**: `readout: cov_head`, `scaler: standard`, `max_train: 4000`, melhor α do WiSE (`search_v3*.yaml:145-146`, `pnas_cs.yaml:145-146`). Mas reporta o κ de teste **ORIG** (head 1280, sem scaler, train inteiro). A seleção é feita numa régua e o número sai de outra. **[H]** Isso explica casos como `larvae_raw` sel-val 0,188 (val COV 0,915) contra máx-teste 0,495.
6. **[M] Splits limpos.** Não há imagem repetida por nome em nenhum split das 12 bases, e todos os sha256 e contagens batem com os YAMLs. Duplicatas de conteúdo só aparecem em poucos casos (tabela §3):
   - aid: 1 par idêntico train↔test (`000013_00000117`/`000013_00000119`);
   - flowers: 4 pares entre splits no split1, sendo 1 cópia byte a byte com **rótulo diferente** (test rose `000003_00000104` = train tulip `000005_00000207`);
   - remote_sensing e coconut_trees: nenhum.

## 1. Método

- **Splits auditados**: os 12 YAMLs de `ga_search/config/modehb/datasets/*.yaml` → `split_json` (sha256 conferido), mais `split1/2/3.json` (protocolo das médias n=3 dos FMs) e o `split1.json` original do HAM.
- **Por arquivo**: interseção de nomes entre train, val e test, e existência dos arquivos.
- **Por conteúdo** (todas as imagens de cada pasta, todos × todos):
  - md5 dos bytes;
  - miniatura cinza 32×32 (BOX) com aHash 1024 bits, dHash 32×31 = 992 bits e correlação de Pearson da miniatura;
  - busca nas 8 transformações diedrais (rot90×flip); no food101 e no protozoan pre, só identidade;
  - níveis: **EXATA** (md5), **T1** (corr ≥ 0,98 e dHash ≤ 50), **T2** (corr ≥ 0,95 e dHash ≤ 100).
- **Calibração visual** (montagens em `A1c_evidencias/`):
  - flowers e aid: T1/T2 = 100 % duplicatas reais (recolorações e recompressões);
  - HAM: T1/T2 = mesma lesão, exceto 3 falsos positivos de vinheta e 1 duplicata real com `lesion_id` diferente;
  - raw de parasitas: corr alta com dHash > 120 = impurezas redondas diferentes, nenhuma duplicata;
  - **pre de parasitas: T1/T2 têm precisão baixa.** Os objetos são alinhados e a escala é normalizada (largura ≈ 180 px), então ovos da mesma espécie se parecem. Amostra aleatória de T2 train↔test do eggs: 0/16 iguais, 2 incertos. Por isso as contagens T2 do pre **não** são duplicatas: no pre só vale o veredito visual (`veredito_visual` em `A1c_dup_pairs.csv`; 33 pares de eggs, 27 de larvae e 8 de protozoan auditados).
- **Tentativas de verificador automático para o pre**:
  - correlação RGB 64×64 nas máscaras;
  - template matching (raw, normal e passa-alta);
  - hash do recorte do objeto.

  Nenhuma separou "mesmo objeto" de "mesma espécie" sem falsos positivos, e foram descartadas. Os escores ficam em `A1c_scripts/` para reprodução.
- **Limitações:**
  - recall desconhecido para recapturas com mudança de foco ou segmentação;
  - não há id de lâmina, campo ou paciente nos parasitas;
  - a recaptura também pode ocorrer entre imagens raw com prefixos finos diferentes (ex.: eggs raw `000009_00001467` ↔ `000013_00005688`, os dois "impurities").

## 2. Integridade dos splits usados nas buscas (`secao=integridade_split`, `dominio`)

| base | split_json | train/val/test | nomes em 2 splits | sha256 = YAML | arquivos ausentes |
|---|---|---|---|---|---|
| aid | split1.json | 6996/1502/1502 | 0 | sim | 0 |
| coconut_trees | split1.json | 9511/2038/2038 | 0 | sim | 0 |
| flowers | split1.json | 1922/412/412 | 0 | sim | 0 |
| remote_sensing | split1.json (stratified_group, 568 grupos) | 682/159/159 | 0 | sim | 0 |
| ham10000 | split1_modehb.json | 4129/878/5008 | 0 | sim | 0 |
| eggs / eggs_raw | split1_modehb(_raw).json | 8885/1903/1903 | 0 | sim | 0 |
| larvae / larvae_raw | split1_modehb(_raw).json | 1118/240/240 | 0 | sim | 0 |
| protozoan / protozoan_raw | split1_modehb(_raw).json | 26159/5607/5606 | 0 | sim | 0 |
| food101 (não buscado) | split1.json | 70700/15150/15150 | 0 | sim | 0 |

**Parasitas pre × raw [M]:**
- `split1_modehb.json` (nomes pre), `split1_modehb_mapping.csv`, `split1.json` (nomes raw) e `split1_modehb_raw.json` (symlinks `raw_modehb` → `raw/orig`) formam **a mesma partição**:
  - a coluna split do mapping bate 100 % com `split1_modehb.json`;
  - o realpath de cada nome do `_raw.json` cai no mesmo split do `split1.json`;
  - o prefixo do nome pre é igual ao rótulo do mapping, com 0 divergências.
- Status do mapping:
  - Eggs: 12 688 ok, 3 `weak_geometry`;
  - Larvae: 1597 ok, 1 `weak_geometry`;
  - Protozoan: 37 340 ok, 32 `weak_geometry`, 4 `unmapped_no_pre_file` (3 de train e 1 de test, excluídos).
- **O README do Protozoan está desatualizado.** O `split1_modehb_README.md` dele cita 3 pares de `weak_geometry`; o CSV tem 32.
- Isso confirma `ga_search/modehb/reports/external_seed_leakage.md`: não há vazamento por nome entre as sementes externas e o nosso test. **Mas aquele relatório só checou nomes e md5**, e as recapturas do §0.1 passam pelos dois.

## 3. Duplicatas de conteúdo entre splits (split usado na busca; `secao=duplicatas_entre_splits`)

| base | train↔val | train↔test | val↔test | observação |
|---|---|---|---|---|
| aid | 0 | 1 T1 (idêntica) | 0 | `000013_00000117`(train)=`000013_00000119`(test); split3 tem mais 1 train↔val |
| coconut_trees | 0 | 0 | 0 | — |
| remote_sensing | 0 | 0 | 0 | split agrupado por cena (568 grupos) |
| flowers | 1 T2 | 1 EXATA (**classes 3≠5**) + 1 T1 | 1 T1 | 2746 imagens com 3 pares md5 internos; split2/3 também têm 2–4 pares entre splits |
| ham10000 (split1_modehb) | 0 | 3 T1 + 14 T2 (13 test), todos da mesma lesão exceto 1 duplicata real com lesion_id diferente (+3 falsos positivos) | 4 T2 (mesma lesão) | o problema principal é a lesão (§4) |
| eggs (pre, auditado) | 5 mesmo objeto (1 byte a byte) | **7 mesmo objeto** | 1 | todos com o mesmo rótulo nos dois lados (classes 6, 7 e 9); raw: só o byte a byte foi achado |
| larvae (pre, auditado) | 4 mesmo objeto | **2 mesmo objeto** | 0 | 19 de 27 candidatos T1/T2 eram larvas diferentes |
| protozoan (pre) | indeterminado | indeterminado | indeterminado | hash 32×32 sem poder discriminante (objetos pequenos, 90,6 % de pixels pretos) |
| *_raw (hash da imagem inteira) | eggs 1 EXATA | 0 | 0 | as recapturas não são vistas na imagem raw inteira |
| food101 (fora da busca) | 3 EXATA + 28 T1 | 2 EXATA + 21 T1 | 4 T1 | EXATA com classe diferente em 2 pares |

Lista completa e rastreável dos pares: `A1c_dup_pairs.csv`. Os níveis T2 do pre (451 pares no eggs e 1308 no larvae) **não** são duplicatas; ficam listados só para auditoria futura.

## 4. HAM10000 por `lesion_id` (`secao=ham_lesoes`; metadata `organized/ham10000/images/isic2018/metadata.csv`, 10 015 imagens, 7470 lesões, até 6 imagens por lesão)

| split | lesões train/val/test | lesões train∩val | train∩test | val∩test | imagens de test com lesão no train | idem train∪val |
|---|---|---|---|---|---|---|
| split1_modehb (busca) | 3529/755/4274 | **0** | 907 | 181 | 1091 (21,8 %) | 1312 (26,2 %) |
| split1 original (FMs; sementes externas) | 4284/4284/4274 (val≡train) | 4284 | 1088 | 1088 | 1312 (26,2 %) | 1312 |
| split2 | 4300/4300/4305 (val≡train) | 4300 | 1135 | 1135 | 1365 (27,3 %) | 1365 |
| split3 | 4280/4280/4280 (val≡train) | 4280 | 1090 | 1090 | 1315 (26,3 %) | 1315 |

- No total, **1088 lesões** aparecem em mais de um split no `split1_modehb` e 4284 no `split1`.
- O test do HAM é 50 % da base (5008 imagens), e não 15 %.

**Consequências:**
- [M] A seleção por val no `split1_modehb` é limpa no nível de lesão.
- [H] O κ de teste do HAM é otimista para todos, mais para quem treina o SVM em 5007 imagens (FMs) e mais nas classes MEL/DF/BCC.
- [M] Os FMs no HAM têm val ≡ train, então o κ val deles (1,000) é inválido.

**Recomendação para o `evaluation_contract.md`:** criar um `split1_lesion.json` (70/15/15 agrupado por lesion_id + correção do par ISIC_0025226/0030074), reavaliar os FMs e os nossos checkpoints nele e manter o `split1_modehb` só como histórico.

## 5. Comparações de `ga_search/modehb/reports/all_results_today.md` que misturam protocolos incompatíveis

| # | onde (linha) | o que é comparado | incompatibilidade | direção provável do viés |
|---|---|---|---|---|
| 1 | §0 linhas 20, 24, 26 (eggs/larvae/protozoan); §4.3, §4.7, §4.9; §9 heatmaps linhas 8419, 8422, 8423 | nosso **pre** × FM **raw** | domínio de imagem | a favor do nosso: FM pre > FM raw em 9/9 (+0,010…+0,085; `secao=FM_pre_vs_raw`). Só AIMv2/DINOv2 pre ainda faltam |
| 2 | §0 linha 33 ("Comparação justa nas bases de parasitas") | nosso **COV** (scaler + 2ª ordem) × FM **ORIG** | régua | a favor do nosso ("só na leitura COV o nosso passa RN50/CNXv2p em protozoan_raw") |
| 3 | §0 coluna "melhor FM média ± std n=3" ao lado do nosso n=1; §0 linha 34 ("contra 0,655–0,753 dos FMs no split 1") | split1 × média de 3 splits | o 0,655 é a **média** do CNXv2p; o valor no split1 é 0,682 | a faixa citada mistura as duas coisas |
| 4 | §3.1 linha 226 (SPiFiL híbrido de outubro "empatou/superou o I-JEPA") e a tabela principal de `A_reports/october/report_halloween_october.md` | GTP (TTA×8 + whitening) × I-JEPA 1 vista; **pct 50**, média de 3 splits do split antigo `helminth_*` (outra partição, não o split1_modehb); I-JEPA "IJ" da destilação | régua + split + pct + modelo | a favor do nosso; não serve de referência para o GA |
| 5 | §6.1 conclusões (linha 8072): "FMs AIMv2 0,746, DINOv2 0,719…" ao lado dos nossos números no split 1 | média de 3 splits (FMs, discovery GTP n=3) × split1 n=1 | split | indefinida (variação do RS entre splits de ±0,03) |
| 6 | §0 colunas "máx-teste" e "máx-COV"; linha 31 (coconut c00001 0,813 "passa os 5 FMs"; ham c00001 0,534); §9 heatmaps (linhas 8417–8424) | **melhor retrospectivo no teste** usado como "nosso" | seleção pelo teste (oráculo) | a favor do nosso. Nos heatmaps, **nenhum** dos 8 "nossos" é o candidato sel-val do §0 |
| 7 | §9 heatmaps linhas 8417 (aid c00001 f45) e 8420 (flowers c00001 f200 κ 0,649); §0 coluna máx-teste do flowers (H-MODEHB-C01) | checkpoints de buscas **C=0,1 / kw 0,75** (`*_C01_*`) apresentados como "nosso" ao lado de C=100 | fitness C=0,1 × C=100 | indefinida; viola R1 na seleção |
| 8 | §4 tabelas do GA: as linhas `-C01` trazem "val κ (fitness)" na mesma coluna das C=100 | quality C=0,1 (kw 0,75: 0,75κ + 0,25F1) × κ C=100 | régua da fitness | os valores de val não são comparáveis entre as linhas |
| 9 | §0 "sel-val" (todas as bases) | seleção por val **COV + scaler + max_train 4000 + WiSE**, reporte ORIG (head, sem scaler, train inteiro) | régua seleção ≠ régua reporte | contra o nosso (seleção desalinhada) |
| 10 | §4.6 linha 514 (FMs no ham10000) × GA ham | FM: SVM em 5007 (train+val do split1, val≡train, κ val 1,000); nosso: 4129 | dados de treino e vazamento por lesão (1312 × 1091 imagens) | a favor dos FMs |
| 11 | §4 FMs (todas as bases) | FMs em 448 (AIMv2) e 518 px (DINOv2) × nosso 224 px; pre nativo 200 px, coconut 90 px, larvae_raw mediana 968×864 px | resolução de entrada | protocolo nativo de cada modelo; registrar, não corrigir |
| 12 | §6.1 tabela "régua igual" | ORIG = head 1280 (nossos), 2048 (H-LEM-C01 c00098) ou GAP 312 (SPF-P) | dimensão do embedding | — |

Linhas do §0 que são comparações limpas: `*_raw` sel-val × FM em `benchmark/modehb_raw` (mesmo split, mesmas imagens, ORIG, n=1), aid/flowers/RS/coconut sel-val × coluna FM split1, e ham com a ressalva do #10.

## 6. Domínios e κ (análise 8; `secao=dominio`, `kappa_vs_FM`, `FM_pre_vs_raw`, `correlacao_exploratoria`)

| base | grupo | classes | n train | desbal. max/min (train) | menor classe no test | resolução em disco (mediana) | % px pretos | κ sel-val | κ máx-teste | melhor FM split1 | razão sel-val/FM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| coconut_trees | cena aérea, tile 2 classes | 2 | 9511 | 3,1 | 498 | 90×90 | 0 | 0,772 | 0,813 | 0,792 (CNXv2p) | **0,97** |
| eggs | objeto segmentado | 9 | 8885 | **116** | 12 | 200×200 (fixo) | 56 | 0,827 | 0,846 | 0,988 raw / 0,988 pre (I-JEPA) | 0,84 |
| aid | cena aérea | 30 | 6996 | 1,9 | 33 | 600×600 | 0 | 0,719 | 0,719 | 0,979 | 0,73 |
| protozoan | objeto segmentado | 7 | 26159 | 92 | 46 | 200×200 | 91 | 0,706 | 0,798 | 0,968 raw / 0,948 pre (I-JEPA*) | 0,73 |
| larvae | objeto segmentado | 2 | 1118 | 5,5 | 37 | 200×200 | 86 | 0,693 | 0,858 | 1,000 raw / 0,968 pre (I-JEPA*) | 0,69 |
| ham10000 | dermatoscopia | 7 | 4129 | 59 | 53 | 600×450 | 0 | 0,501 | 0,534 | 0,725 | 0,69 |
| eggs_raw | objeto + fundo | 9 | 8885 | 116 | 12 | 384×384 (96–1360) | 0 | 0,684 | 0,684 | 0,988 | 0,69 |
| remote_sensing | cena aérea | 14 | 682 | 8,0 | **1** | 640×640 | 0 | 0,494 | 0,633 | 0,753 | 0,66 |
| flowers | objeto natural | 5 | 1922 | 1,3 | 74 | 320×240 (143–1024) | 3 | 0,594 | 0,649 | 0,979 | **0,61** |
| protozoan_raw | objeto + fundo | 7 | 26159 | 92 | 46 | 104×96 (32–384) | 0 | 0,452 | 0,452 | 0,968 | 0,47 |
| larvae_raw | objeto + fundo | 2 | 1118 | 5,5 | 37 | 968×864 (112–2736) | 0 | **0,188** | 0,495 | 1,000 | **0,19** |

\* AIMv2 e DINOv2 em pre ainda pendentes no athena (só RN50, CNXv2p e I-JEPA rodaram); o melhor FM em pre pode ser maior.

**Medido [M]:**
- A distância ao FM é **mínima em coconut_trees** (2 classes, tiles de 90 px, em que o FM não ganha com resolução nem semântica) e **máxima em flowers** (objeto semântico, ImageNet-like) e **larvae_raw**.
- Para o nosso, **raw é sempre pior que pre**: eggs −0,14, larvae −0,51, protozoan −0,25 no sel-val. Para os FMs, raw também é pior que pre, mas por muito menos (−0,01 a −0,09). O fundo de microscopia prejudica muito mais o nosso encoder que o FM.
- O desbalanceamento é extremo nos parasitas (impurities = 76–77 % do train em eggs/protozoan) e no HAM (NV = 67 %). O κ pune a classe majoritária, mas a menor classe do test tem 12 imagens (eggs) e 1 imagem (RS): o κ por base com n=1 tem incerteza grande. Erro-padrão binomial da acurácia ≈ 0,039 no RS (159 imagens de test) e ≈ 0,02 no larvae (240).

**Correlação [C]:** com Spearman nas 11 bases, o gap FM − sel-val não se relaciona com n_train (ρ −0,17), n_classes (−0,32), desbalanceamento (−0,12) ou resolução (+0,19); p > 0,3 em todos. Com n=11 e 3 pares de bases que compartilham os dados (pre/raw), a análise não tem poder: nenhuma conclusão.

**Hipóteses [H]**, a testar pelos agentes 2/3 com a régua única e o split limpo:
- **(a) Resolução.** O larvae_raw reduz imagens de ~900–2700 px para 224 px, e a larva fina vira poucos pixels. O nosso encoder pequeno (sem escala) perde, e o FM a 448/518 px não. Controle: larvae_raw com recorte pela máscara × redução simples.
- **(b) Fundo.** O fundo de microscopia (raw) tem textura e detritos que o SPiFiL label-free amostra como "patches informativos". Controle: SPiFiL com patches só dentro da máscara × em toda a imagem, no raw.
- **(c) Semântica.** Em flowers, aid e RS (cenas e objetos semânticos) o gap vem da semântica do pré-treino em larga escala. Em coconut e HAM (textura, cor e baixa resolução) o gap é menor. Aqui textura/dermatoscopia e cena aérea de baixa resolução são os domínios "favoráveis".
- **(d) Seleção.** A diferença sel-val × máx-teste é grande em larvae (0,165), larvae_raw (0,307), RS (0,139) e protozoan (0,092): a seleção por val COV escolhe mal para a régua ORIG (§5 #9). Em larvae e RS ainda soma o val pequeno (240 e 159 imagens).

## 7. Recomendações para o contrato de avaliação (insumos ao `evaluation_contract.md`)

1. Usar uma régua só para seleção e reporte. Se o reporte é ORIG, a fitness de val tem de ser ORIG também (ou reportar as duas, sempre pareadas).
2. No HAM, usar o split agrupado por lesão em train/val/test, mais a exclusão do par ISIC_0025226/0030074. Refazer o benchmark dos FMs com o **mesmo** train (4129), e não com 5007.
3. Nos parasitas:
   - comparar sempre pre × FM-pre e raw × FM-raw (completar AIMv2/DINOv2 em `modehb_pre`);
   - registrar que existem recapturas do mesmo objeto entre splits (lista em `A1c_dup_pairs.csv`, `veredito_visual=mesmo`);
   - reavaliar o κ de teste excluindo essas imagens como análise de sensibilidade;
   - pedir a proveniência (lâmina/campo) para um split agrupado.
4. flowers e aid: excluir do test (ou mover para o train) os 4 + 1 pares entre splits do split1, inclusive a cópia byte a byte com rótulo diferente.
5. Reportar sempre n=1 (split1) contra o valor split1 do FM; média ± std só quando os dois lados tiverem n=3. Nunca usar o máx-teste nem checkpoints C=0,1 como "nosso".
