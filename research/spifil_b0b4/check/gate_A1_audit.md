# CHECK: auditoria do gate A1 (diagnóstico do Agente 1)

> Agente CHECK, 17/11/2026. A auditoria rodou na Jaci, só em CPU (`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 15`). Não toquei em processos, na fila nem na árvore viva, e não criei nada em `gates/`.
>
> **Objeto da auditoria:** `research/spifil_b0b4/agent1/`, ou seja: `diagnosis.md`, `seed_registry.csv`, `experiment_matrix.csv`, `research_hypotheses.md`, `evaluation_contract.md`, `parts/` (A1a–A1d, com CSVs e scripts) e `synth_scripts/`.
>
> **Scripts e saídas do CHECK** ficam em `/tmp/claude-1014/-dados-home-moliveira-Scalable-Hybrid-FLIM/05a95f35-17a2-4976-a2bd-9c4d1e4ff9eb/scratchpad/checkA1/`:
> - `rho.py`: ρ e arrependimento, a partir dos jsonl de teste e dos `search_state`;
> - `wise.py`: WiSE e fidelidade, a partir dos `search_state` vivos;
> - `orig.py` → `orig_results.jsonl`: κ ORIG em CPU;
> - `bands.py`: re-extração das faixas B0–B4 do `search_state` vivo;
> - `mac.py`: contagem independente de MAC;
> - `reg_repro.csv`: reexecução do `seed_registry.py`.

## Veredito: **APROVADO COM CONDIÇÕES**

**O diagnóstico se sustenta.**
- Os 5 números centrais foram reproduzidos de forma independente a partir dos artefatos brutos.
- O κ ORIG recalculado em CPU bate com o do A1b nos 3 checkpoints testados (0,4940 / 0,0000 / 0,3219), e o rank efetivo também.
- As unidades de FLOPs conferem, e o vazamento HAM (1312 imagens, 26,2%) também.
- A conclusão principal se mantém: a busca seleciona por COV e reporta por ORIG, e a cabeça é o gargalo.

**Os entregáveis operacionais ainda não estão prontos para congelar.**
- O `seed_registry.csv` está velho: 15 células divergem do `search_state` atual, e as mudanças do CHECK anterior só foram anotadas, não aplicadas.
- O registro não é reprodutível pelo próprio script.
- O contrato de avaliação contradiz R2/R7 no acesso ao teste.
- Há rótulos de evidência a corrigir e um critério de hipótese tautológico (H6).

O Agente 2 pode começar a implementação e as ablações sem treino. As condições **C1–C3** valem antes do X00 e antes de congelar qualquer semente. **C4** vale antes de o Agente 4 ligar a fitness nova. As demais são de documentação e valem antes do gate A2.

### Condições

| # | Bloqueia | Condição |
|---|---|---|
| **C1** | X00 / congelamento de sementes | **Re-extrair o `seed_registry.csv` do `search_state` atual**, com um script só e entradas congeladas (md5 de cada `search_state` e jsonl no log). As mudanças do CHECK anterior entraram só como texto em `obs`: o registro ainda traz ham B2 = c00023 (0,5765) em vez de c00043 (0,612), larvae_raw B1 = c00022 (0,8631) em vez de c00026 (0,9154, 349k), e flowers B3 = `lacuna=sim`. Além disso, há mais 12 células divergentes, a maioria porque as buscas migradas do athena (eggs, eggs_raw, coconut, flowers LEMONADE) agora têm `search_state` completo na Jaci e o registro usou só as "linhas testadas" (ver item 4.3). A exclusão de coconut c00012 (`failed`) está correta. |
| **C2** | X00 | **Registro reprodutível e sem mistura de sub-réguas na seleção.** Rodar `synth_scripts/seed_registry.py` hoje não gera o arquivo entregue: larvae B0 ORIG passa de c00001 (κ 0,000) para c00021 (0,8095), porque o `agent6/results/E0_val_jaci_cpu.jsonl` cresceu. Pedidos: congelar as entradas por cópia + md5; separar o argmax ORIG **por sub-régua** (hoje `sort_values('_k')` mistura ORIG_full, ORIG_sub4000, ORIG_subamostra e o legado da pipeline antiga na mesma faixa); marcar como inutilizável a semente ORIG larvae B0 c00001 (κ 0, degenerada); corrigir o caminho de RS B3 ORIG (`artifacts/modehb_strat/...` não existe, o checkpoint está em `artifacts/modehb_strat_prefix_2130/...`, então o "ausente_na_jaci" é falso). |
| **C3** | X00 | **Aplicar a regra de desempate do próprio contrato.** O contrato (§2.3) diz que \|Δκ\| < MDE é empate prático, resolvido por menos FLOPs, depois menos params, depois menor fidelidade numérica. O registro escolhe o κ máximo estrito e desempata por fidelidade antes de FLOPs. Exemplos: RS B2 tem 7 candidatos a ≤ 0,01 do melhor; eggs B0, 6; larvae B0, 6. Re-gerar com a regra do contrato ou declarar a divergência. |
| **C4** | fitness nova (Agente 4) e todo job com teste | **Reconciliar o §4 do contrato com R2/R7.** R2 manda rodar o "SVM original no TESTE para todo job do GA (todas as fidelidades)", e R7 manda levar o resultado direto à tabela de ranking. O monitor já faz isso (`test_svm_C100.jsonl` → `HYBRID_PROGRESS.md` / `ranking_ga_C100.csv`). O contrato, ao contrário, exige um runner que **recusa** `--split test` sem manifesto e "uma leitura de teste por manifesto". Isso contradiz uma regra do usuário, que tem precedência. **Correção:** o teste por job continua (R2), num canal só de escrita que o controller, o seletor, o WiSE e a promoção nunca leem. A trava fica sobre o **uso** do teste (`never_feeds_fitness`, verificada por teste unitário no seletor), não sobre o cálculo. O manifesto congelado continua obrigatório para o número de **comparação final** (nosso × FM). |
| C5 | gate A2 (doc) | **Corrigir rótulos de evidência no `diagnosis.md`** (lista no item 3). Os principais: §1 "seleção quase aleatória para ORIG" marcado CAUSAL → MEDIDO (código) + CORRELAÇÃO; §6 "a vantagem dos tiny no teste ORIG tem outra causa: o ORIG não escala" → HIPÓTESE; §2 "a cabeça cumpre ≥1280 só no nome; dimensão efetiva ~3" → HIPÓTESE (o participation ratio não mede separabilidade, e o próprio dado mostra ORIG+PW−ORIG = +0,009 na mediana); §7 "impacto direto no κ de teste ≥ 0,37%" é a **fração de imagens** de teste (7/1903), não impacto no κ. |
| C6 | X03 / H6 | **H6 tem critério tautológico.** "Rank efetivo ≥ 20" é atingido **por construção** pelo braço whitening (PCA-whiten iguala os autovalores e leva o participation ratio ao posto numérico). O critério tem de ser só o κ ORIG-val (Δ ≥ MDE, IC > 0). O rank efetivo fica como diagnóstico. O mesmo vale para o contrato §1: o rank efetivo é **reportado**, nunca usado como gate de "alta dimensão". Registrar também que os dados existentes já pesam contra H6 (PW: +0,009 mediano; scaler: +0,034). |
| C7 | X00, X02, X03, X09 | **ORIG reprodutível entre CPU e GPU.** A busca e o monitor embedam com AMP em CUDA (`run_search.py:488`). O X00 e o A1b usam FP32 em CPU. Com embeddings mal condicionados (até 3,9·10⁷ iterações do libsvm, κ=0 em larvae c00001), o κ pode mudar de lado. **Correção:** antes do X00, medir ORIG-val em GPU-AMP e em CPU-FP32 nos mesmos 5 checkpoints e declarar a tolerância. A régua oficial fixa um dos dois. |
| C8 | X00 / fitness | **Validar ORIG_sub4000 contra ORIG_full** onde eles diferem (HAM 4129, eggs 8885, aid 6996, coconut 9511, protozoan 26159), com ρ e Δ em ≥ 10 checkpoints por dataset, antes de usar sub4000 como fitness. Nos datasets com train < 4000 (RS, larvae, flowers), sub4000 = full e o rótulo deve dizer isso. |
| C9 | fitness (Agente 4) | **O contador de FLOPs só vê Conv2d/Linear** (`run_search.py:322-360`; conferi: c00009 RS 875 445 760 MAC e c00009 eggs 490 688 352 MAC, idênticos ao registro). Pooling `moments`/GeM, covariância, sqrtm, normalizações e atenção fixa somam **zero**. O §6 do contrato exige contá-los, mas não há contador para isso. Sem ele, o objetivo FLOPs fica cego exatamente para os operadores que X02/X03 e o Agente 6 vão introduzir. **Correção:** implementar e testar um contador para essas operações antes de qualquer fitness que as inclua. |
| C10 | relatório de teste | **X17: regra de troca do número oficial depende do resultado do teste.** "Se a versão limpa inverter a ordem, o número oficial passa a ser o limpo" é uma escolha de reporte condicionada ao teste. **Correção:** pré-registrar agora qual número é oficial (proposta: HAM `test_lesion_clean` oficial e `test_full` ao lado; parasitas/flowers/aid com as duas colunas sempre), sem depender do resultado. |
| C11 | X00 / X09 (doc) | **Custos subestimados nos datasets grandes.** O ORIG no teste custa 1360 s por checkpoint em eggs e 1794 s em protozoan_raw (`orig_seconds` em `test_svm_C100.jsonl`). O X00 (~400 ckpts, ~1/3 em eggs/protozoan pre+raw) fica perto de 60–70 CPU-h só de SVM, e não 40. O X09 (×5 α) passa de 200 CPU-h se incluir esses datasets. Re-estimar, ou restringir X09 a RS/larvae/HAM/flowers. |
| C12 | gate A2 (doc) | **Declarar os riscos não declarados do item 7**, principalmente o otimismo do argmax sobre a mesma validação (n_val = 159 em RS) e o fato de o teste das sementes já ter sido visto (R2). |
| C13 | — (decisão do usuário) | **A migração do HAM para `split1_lesion.json`** (contrato §7) contradiz "splits imutáveis" (§5.1) e quebra a comparabilidade com o histórico e com os FMs. Só com decisão explícita do usuário (R4: push + opções). Até lá, `test_lesion_clean` como coluna. |

---

## Evidência por item

### 1. Os 5 números centrais, recalculados dos brutos

| número | relatório | CHECK (fonte bruta) | situação |
|---|---|---|---|
| ρ(val COV, teste ORIG), mediana de 19 grupos | 0,09 (IQR −0,41 a 0,57; negativo em 9/19) | **0,091** (IQR −0,41 a 0,57; 9/19 negativos) | reproduzido |
| ρ(val COV, teste COV) | 0,80 | **0,801** | reproduzido |
| arrependimento ORIG | mediana 0,047, máx 0,307; coincide em 4/19 | **0,047; 0,307** (hybrid larvae_raw); 4/19 | reproduzido |
| rank efetivo do embedding 1280 | 3,4 (top-1 51%) | **3,44** (top-1 51,4%; GAP 1,79; COV 13,0) | reproduzido do CSV; os 2 checkpoints que recalculei batem (ver item 2) |
| WiSE α=1 − α=0 | +0,003 (n=656) | **+0,0028** no snapshot (n=657); **+0,0028** no `search_state` vivo (n=924, IQR −0,014 a +0,019, 55% positivos) | reproduzido e estável |
| ρ(f5, f200) | 0,985 (64 linhagens) | **0,985** com as 63 linhagens f5→f200 do `A1b_lineages.csv`; 0,992 só com C=100 (52); **0,993** no vivo (72) | reproduzido |

Fontes do CHECK:
- ρ e arrependimento (`rho.py`): os dois `test_svm_C100.jsonl` (Jaci + bundle), deduplicados pela linha mais recente, com n ≥ 3 por (run, dataset). O `val_kappa` logado bate com o `fitness_kappa` do `search_state` em 241 de 242 casos.
- WiSE e fidelidade (`wise.py`): os `search_state.json` vivos (hybrid, LEMONADE e pnas_cs do bundle).

**Ressalvas que o relatório não declara** (nenhuma derruba o número):
- **4 dos 19 grupos são buscas C=0,1** (`*_C01_final`). Nelas o val é COV com C=0,1, e o contrato proíbe misturar C. Só com C=100 (15 grupos): ρ ORIG 0,091, ρ COV 0,829, arrependimento 0,041 / 0,307. Robusto.
- **Pseudo-replicação:** o mesmo candidato em várias fidelidades conta como pontos distintos. Deduplicando por candidato (maior fidelidade), ρ ORIG cai para **−0,05** (8/15 negativos), o que reforça a conclusão.
- **Grupos de n = 3–4** dão ρ = ±1,0 (lemonade eggs n=3, ρ COV = 1,0). A mediana é robusta, mas cada grupo isolado não é.
- **ρ(f5, f200) = 0,985 está inflado pela variância entre datasets e estratos.** Cada grupo tem quase só 4 linhagens (1 representante de calibração por estrato, tiny…huge, com κ muito diferentes). Dentro do grupo, a mediana é 1,0 com n=4. Centrando por grupo, o ρ agregado ainda é 0,954. A conclusão "não reordena" vale para candidatos distantes. **Não está testada** entre quase-empatados perto do corte de promoção, que é onde a reordenação importa. O X11 deve medir isso.
- 11 das 63 linhagens do 0,985 são C=0,1 (o texto diz "régua COV" sem separar).

### 2. κ ORIG recalculado em CPU (A1b_embed_diag)

Código próprio (`orig.py`):
- dados pelo `progressive_data.load_data` oficial (`search_v3.yaml`, transform de teste, sem shuffle);
- `z = head(encoder(x))` em `eval()`/`no_grad`, FP32;
- `core.svm.fit_svm(Z, y, C=100, scaler=False)`, treino inteiro → val inteira;
- `sklearn.cohen_kappa_score`;
- participation ratio em treino+val.

| checkpoint (md5) | A1b κ ORIG / rank ef. | CHECK κ ORIG / rank ef. | libsvm |
|---|---|---|---|
| RS hybrid c00001 f200 (`7e4de30d…`) | 0,4940 / 3,308 | **0,4940 / 3,308** | fit_status 0, 16 367 it. |
| larvae hybrid c00001 f200 (`bc601404…`) | 0,0000 / 1,149 (top-1 93%) | **0,0000 / 1,149 (top-1 93,2%)** | fit_status 0 |
| larvae_raw hybrid c00003 f200 (`0c0fdd7a…`) | 0,3219 / 2,526 | **0,3219 / 2,526** | fit_status 0, 2,1·10⁶ it. |
| extra: RS legado `modehb_spifil` c00306 f100 (`7c46734a…`; registro ORIG_sub4000_legado 0,5719, valor da busca em GPU) | — | **0,5714** (dim 2560) | fit_status 0 |

- Os 3 checkpoints do diagnóstico reproduzem **exatamente**. A régua ORIG do A1b é a ORIG do contrato: sem scaler, C=100, train inteiro em RS, larvae e larvae_raw. Em protozoan e ham o A1b usou subamostra (treino ≤ 2000), e isso está declarado.
- **O ORIG legado de RS é de fato a régua ORIG.** Com train 682 < 4000, sub4000 = full. A diferença de 0,0005 em relação ao valor gravado (GPU-AMP) é a primeira evidência de que CPU e GPU diferem pouco, mas diferem (C7).

### 3. Rotulagem (MEDIDO / CORRELAÇÃO / HIPÓTESE / CAUSAL)

Em geral, a rotulagem é cuidadosa: as hipóteses estão marcadas, e o CAUSAL é restrito a intervenções reais (blend WiSE, mesma linhagem, leitura do mesmo encoder). Problemas encontrados:

| local | texto | rótulo dado | rótulo correto / motivo |
|---|---|---|---|
| diagnosis §1 | "A seleção é coerente dentro da própria régua (ρ 0,80) e quase aleatória para ORIG" | CAUSAL, por construção | **MEDIDO (código) + CORRELAÇÃO.** Que a fitness usa COV é fato do código. O ρ ≈ 0,09 é observacional. |
| diagnosis §2 | "A cabeça atual cumpre '≥ 1280' só no nome: a dimensão efetiva é ~3" | sem rótulo (consequência) | **HIPÓTESE.** O participation ratio depende de escala e não mede separabilidade linear. O próprio A1b mostra ORIG+PCA-whiten − ORIG = **+0,009** e ORIG+scaler − ORIG = +0,034 na mediana: isotropizar quase não muda o κ fora dos casos degenerados. O gargalo é a informação (2ª ordem, blocos), não a anisotropia. |
| diagnosis §2 | "2ª ordem × melhor 1ª ordem por bloco: +0,107" | CAUSAL | **CAUSAL na leitura, mas confundido.** "COV sem head" usa **todos os blocos**, e "melhor GAP" usa um bloco escolhido na própria validação. Comparações casadas que recalculei do CSV: 2ª − 1ª ordem só no último bloco = **+0,072** [−0,08; +0,56]; multiescala 2ª − multiescala 1ª = **+0,133**. A conclusão se mantém, mas o número citado deveria ser o casado. A dimensão das features também difere (COV PCA-64 por bloco ≫ C), e isso é um confundidor de capacidade a declarar. |
| diagnosis §4 | "α=0 … sem tendência com estrato, params, profundidade ou fidelidade (OLS, p > 0,16)" | MEDIDO | **CORRELAÇÃO** (ausência de associação observacional). |
| diagnosis §5 | ρ(f5, f_final) = 0,985 | CAUSAL (mesma linhagem) | Aceitável como intervenção (mais épocas), mas **inflado entre grupos** (item 1) e misturando C=0,1. |
| diagnosis §6 | "A vantagem dos tiny no teste ORIG tem outra causa: o ORIG não escala com params" | MEDIDO + inferência | **HIPÓTESE.** Afirmação causal sem intervenção. |
| diagnosis §7 | "Impacto direto no κ de teste: ≥ 0,37% (eggs) e ≥ 0,83% (larvae)" | MEDIDO | **O número está certo, mas a grandeza está errada:** 7/1903 e 2/240 são a fração de imagens de teste afetadas, não o impacto no κ. |
| diagnosis §3 | "ORIG piora em 5 de 11"; ρ = −0,26 | CAUSAL, n=11 | Rótulo ok, mas sem faixa de ruído. RS c00004 (−0,036) está abaixo do EP de ≈ 0,04 com n_val=159. ρ com n=11 não sustenta nada. |

Não encontrei afirmação causal sobre arquitetura (profundidade, inception, largura) sem o rótulo CORRELAÇÃO. As de init (§8b) estão como HIPÓTESE ou MEDIDO (código).

### 4. `seed_registry.csv`

1. **Caminhos e md5.**
   - 76 linhas têm checkpoint; 74 existem. As 2 ausentes:
     - RS B1 COV `artifacts/pnas_cs/…/c00042` (só no athena; declarado);
     - RS B3 ORIG `artifacts/modehb_strat/…/c00016`, cujo caminho está **errado**: o arquivo existe em `artifacts/modehb_strat_prefix_2130/remote_sensing/state/seed42/candidates/c00016/f5/encoder.pt` (C2).
   - O md5 bate em **8 de 8** linhas sorteadas (seed 7; das duas fontes: Agente 6 e "calculado_agora").
2. **Nenhum número de teste entrou na seleção: OK.**
   - No `seed_registry.py`, `test` só aparece na flag `teste_historico_existe`.
   - A ordenação COV é por `val_kappa`, `fidelity` e `flops`; a ORIG é por `_k` de validação.
3. **Mas o pool de 7 linhas era enviesado.** Essas linhas (`ok(athena_test_only)`: flowers B0, eggs B3, eggs_raw B0–B4, coconut B0–B3) vieram de um pool restrito aos jobs **que tinham teste** (`athena:reports/test_svm_C100.jsonl`). Agora que esses `search_state` estão na Jaci, o pool completo muda o vencedor. Exemplos: eggs_raw B0 vai de c00001 (0,527) para c00025 (**0,886**); eggs_raw B1 de 0,753 para 0,880; coconut B1 de 0,763 para 0,778. Não há vazamento de valor de teste, mas a seleção estava errada (C1).
4. **Mudanças do CHECK anterior:** só anotadas em `obs` (POS-SNAPSHOT), não aplicadas.
   - ham B2 continua c00023;
   - larvae_raw B1 continua c00022;
   - flowers B3 continua lacuna;
   - coconut c00012 foi excluído corretamente (`failed` confirmado no `search_state`).
   - Diferenças do registro para o `search_state` vivo, no total (`bands.py`): ham B2, larvae_raw B0/B1/B2, flowers B2/B3, eggs B0–B3, eggs_raw B0–B2, coconut B1/B2 e protozoan B0 (0,7725 → 0,867).
   - larvae_raw B4 é lacuna no registro, mas existe c00024 (3,49M, 0,863) no vivo.
5. **Unidades de FLOPs: OK.**
   - `flops_MAC` é MAC de Conv2d+Linear. A contagem independente bate exatamente em 2 checkpoints (`mac.py`), e `flops_2MAC = 2 × flops_MAC`.
   - Ressalva: as linhas legadas (spifil_v1) vêm de outra pipeline. RS B4 c00099 tem 90,5 GMAC (largura 252 sem pooling cedo), e esse valor não é comparável entre pipelines no mesmo Pareto.
6. **Linhas legadas no ORIG** (`modehb_spifil`/`strat`, pipeline antiga, kw 0,75).
   - `embed_dim` é ≥ 1280 (1536/2560, conferido no `search_state`).
   - Mas são encoders de outra receita (sem AE/protect atuais). Como "semente" para a pipeline nova, são outra população e precisam ficar marcados como tal.

### 5. `evaluation_contract.md`

| ponto | situação |
|---|---|
| R1 (fit_svm, C=100, sem scaler) | **OK** (§1, §10.2) |
| R2 (SVM original no teste de **todo** job) | **CONFLITO** (§4.1, §4.3) → C4 |
| R3 (fitness = κ + FLOPs) | **OK** (§2.1). A unidade 2·MAC é monotônica com MAC, então não muda o Pareto. A fitness passa a ser ORIG-val, coerente com MISSION:66 |
| R11 (cabeça ≥ 1280) | **OK** (§1, §10.5). Todos os braços do X02/X03 mantêm ≥ 1280 |
| Vazamentos | **OK**, com C10 e C13. Reconferi o HAM: 0 lesões train∩val; **1312/5008 = 26,2%** das imagens de teste com lesão em train∪val (`metadata.csv` + `split1_modehb.json`) |
| FMs no mesmo domínio | **OK** (§8): pre×pre, raw×raw, mesmo train 4129 no HAM, sem COV/GTP contra ORIG de FM |
| Bootstrap e seeds | **OK** (§5.2–5.3): bootstrap pareado B = 10 000, hierárquico com seeds, Holm, ≥ 3 seeds, n=1 rotulado. O MDE do X01 é razoável |
| Armadilha `svm_on_demand --scaler standard` | **OK e confirmada** (`svm_on_demand.py:212`, `default="standard"`). O `--readout` default já é `head` e o `--config` default é `search.yaml`, com seção `data` idêntica à do `search_v3.yaml` (sem efeito hoje). Recomendo o contrato exigir também `--config` explícito |
| FLOPs de operadores sem parâmetros | §6 exige contá-los, mas o contador não existe → C9 |
| Rank efetivo obrigatório | aceitável como coluna, não como critério → C6 |
| ORIG_sub4000 na busca | aceitável e declarado, mas falta a validação contra full → C8 |

### 6. `experiment_matrix.csv` e `research_hypotheses.md`

- **Controle e critério de rejeição:** todas as 18 hipóteses (H1–H18) têm controle explícito e critério fixado antes de rodar. Os X00/X01/X18 são entregáveis, sem hipótese. O único critério defeituoso é o do H6 (C6).
- **O X00 vem antes:** P0, sem dependências. X01, X02, X03, X09 e X11 dependem dele.
- **O teste não decide nada operacional**, mas há três pontos a declarar:
  - (a) H1/X12 avaliam a política de fitness pelo arrependimento **no teste**. É aceitável porque a política já está fixada pela MISSION, e o X12 só confirma. Tem de estar escrito que o resultado do X12 não muda a fitness.
  - (b) A "auditoria prévia" do H1 no X00 compara ρ(ORIG-val, teste) com ρ(COV-val, teste) e pode "enfraquecer H1". Idem: só relatório.
  - (c) O X17 condiciona o número oficial ao resultado do teste (C10).
- **Custos:** plausíveis na ordem de grandeza, mas subestimados nos datasets grandes (C11). As estimativas de GPU-h por job (f5 0,06–0,18; f45 0,12–0,78) batem com `A1a_data.gpu_hours`.
- **Cobertura da MISSION:** as 7 hipóteses pedidas estão cobertas. "Preservação por aprendizagem local" aparece só como AE × noAE (H12/X07), sem medir preservação da representação (similaridade antes/depois). Sugiro acrescentar a métrica ao X07.

### 7. Riscos não declarados

1. **Snapshot vivo.** Os `search_state` continuam mudando (ham, protozoan e larvae_raw ganharam vencedores novos depois das 22:05). Qualquer "melhor por faixa" precisa de manifesto com md5 do `search_state` e timestamp (C1).
2. **Otimismo do argmax na mesma validação.** O X00 re-pontua ~400 checkpoints e escolhe o máximo ORIG-val por faixa, com n_val = 159 (RS) e 240 (larvae). Com dezenas de candidatos e EP ≈ 0,04, o vencedor carrega um viés positivo da ordem de 1–2 EP. O contrato trata o WiSE com metades A/B (X09), mas não a seleção das sementes. **Sugestão:** confirmar os finalistas por faixa em metades A/B fixas, ou registrar o otimismo esperado.
3. **O teste das sementes já foi visto.** Por R2, todo job já tem ORIG-teste (`test_svm_C100.jsonl`, 265 linhas), e os agentes leram rankings de teste ao desenhar o diagnóstico. O "teste só no fim" não é um holdout virgem para esses checkpoints: o relatório final tem de dizer isso, e a comparação com FM tem de usar o manifesto congelado.
4. **Amostra do diagnóstico do embedding.** São 17 checkpoints: 13 de calibração (c00001–c00004, f200), 6 de RS, nenhum de eggs, flowers, aid ou coconut, e 1 de busca C=0,1 (c00098). Rank efetivo 3,4, "+0,107" e "o head não acrescenta nada" devem ser enunciados para essa amostra, não para "o embedding 1280".
5. **CPU × GPU no ORIG** (C7) e **sub4000 × full** (C8).
6. **FLOPs cego a operadores sem parâmetros** (C9). O GA tende a preferir operadores caros e "grátis".
7. **WiSE re-pontuável?** O X09 assume que todo checkpoint guarda `anchor` e `ssl_state`. O A1b só conseguiu a0/a1 em parte dos 17. É preciso conferir a cobertura antes de orçar.

## Resumo para o orquestrador

- **Gate A1 → Agente 2: APROVADO COM CONDIÇÕES.** O Agente 2 pode implementar e rodar ablações sem depender das sementes congeladas.
- **Antes do X00 e de congelar sementes:** C1, C2, C3, C7 e C8.
- **Antes da fitness nova do Agente 4:** C4 e C9.
- **Documentação, antes do gate A2:** C5, C6, C10, C11 e C12.
- **Pergunta ao usuário:** C13.
