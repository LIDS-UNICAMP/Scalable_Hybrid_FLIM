# Agente 1 — Hipóteses de pesquisa (falsificáveis)

_17/11/2026. Cada hipótese traz a afirmação, a evidência que a motivou (`arquivo:linha`, relativo a `research/spifil_b0b4/`), o teste (id em `experiment_matrix.csv`) e o **critério de rejeição**, fixado antes de rodar._

## Regras comuns (do `evaluation_contract.md`)

- **Régua de decisão:** κ **ORIG-val** (`head(encoder(x))` ≥ 1280, `core.svm.fit_svm` C=100, sem scaler, train inteiro).
- **Teste:** só no relatório final, com a escolha congelada.
- **Unidade de análise:** Δ pareado (mesmas imagens de validação) por seed de treino.
- **IC:** bootstrap pareado por imagem, B = 10 000, IC 95%. Correção de Holm entre braços do mesmo experimento.
- **Critério:** sempre o κ ORIG-val. Rank efetivo, redundância e erank são diagnósticos de mecanismo e nunca decidem (C6).
- **Seeds:** ≥ 3 por braço, salvo indicação. Datasets prioritários: RS, Flowers, HAM10000. Depois eggs, larvae, protozoan, com raw e pre separados. aid e coconut são controles.
- **"Ganho"** = média sobre seeds de Δ ≥ MDE do dataset **e** IC 95% > 0. O MDE é medido no X01. Até lá vale o piso de 0,02 (0,03 em RS/larvae, que têm n_val 159/240).
- **"Sem efeito"** = IC 95% contido em [−MDE, +MDE].
- **Inconclusivo** é o caso restante, e não conta como confirmação.

---

## Bloco R — Régua e seleção

### H1. Selecionar por ORIG-val reduz o arrependimento no teste ORIG
- **Afirmação:** com a régua oficial ORIG, a fitness ORIG-val escolhe candidatos melhores no teste ORIG do que a fitness COV-val.
- **Evidência:** ρ(val COV, teste ORIG) = 0,09, contra 0,80 dentro do COV; arrependimento mediano 0,047 (A1b:28-31, 310-323; A1a:85).
- **Teste:** X12. Mini-buscas pareadas (mesmo orçamento e seed de busca), fitness ORIG-val × COV-val, 3 seeds de busca, RS/Flowers/HAM. A escolha é congelada por faixa B0–B4 e o teste ORIG é lido uma vez no fim.
- **Rejeição:** o arrependimento ORIG (melhor ORIG do pool − ORIG do escolhido) da seleção ORIG-val **não** é menor que o da COV-val em ≥ 60% das células dataset × faixa, ou o Δ mediano fica abaixo de 0,01. Auditoria prévia sem treino (X00): se ρ(ORIG-val, ORIG-teste) ≤ ρ(COV-val, ORIG-teste) + 0,1 nos checkpoints já testados, H1 fica enfraquecida. Isso é auditoria, nunca seleção.
- **Nota (CHECK §6a/b):** a política de fitness ORIG-val já está fixada pela MISSION (MISSION:66) e por R2/R3. O X12 só **confirma**: o resultado dele não muda a fitness, e a leitura do teste no X12 não alimenta nenhuma seleção de sementes.

### H2. A ordem dos candidatos em f5 se mantém em ORIG-val
- **Afirmação:** como em COV (ρ f5 × f200 = 0,985), a ordem em ORIG-val de f5 já é a final.
- **Evidência:** A1b:131-139. Em COV a ordem se mantém, mas o ORIG é sensível ao head e ao último bloco, que o SSL altera (A1b:241-250).
- **Teste:** X11. Re-pontuar em ORIG-val as 64 linhagens existentes (f5 … f200), sem treino.
- **Rejeição:** ρ Spearman(ORIG-val f5, ORIG-val f_final) < 0,8 em ≥ 2 de 3 datasets prioritários. Se for rejeitada, a escada de fidelidade se justifica para ORIG.

### H3. A escolha do α do WiSE é ruído de validação
- **Afirmação:** κ(α*) escolhido em uma metade da validação não supera, na outra metade, um α fixo (0 ou 1) por mais que o MDE.
- **Evidência:** otimismo da escolha +0,016 ≈ ganho do WiSE +0,011 a +0,022; DP entre α vizinhos de 0,008 a 0,078 (A1b:123-129).
- **Teste:** X09. Validação dividida em 2 metades estratificadas e fixas (A/B, seed registrada): escolhe-se α em A e mede-se em B, e o inverso. Régua ORIG-val. Sem treino novo (os 5 α já existem por job).
- **Rejeição:** κ_B(α*_A) − max(κ_B(α=0), κ_B(α=1)) ≥ MDE com IC > 0 em ≥ 2 de 3 datasets. Nesse caso o WiSE carrega sinal.

### H4. A promoção protegida desperdiça orçamento
- **Afirmação:** trocar `min_share = 0,25` por um limiar global (não promover abaixo do quantil 50% global de κ ORIG-val) não reduz o HV final κ × FLOPs e aumenta o κ dos promovidos.
- **Evidência:** tiny promovidos com κ mediano 0,414 contra 0,565 dos huge; 3–6 huge acima do corte ficam de fora (A1b:142-156).
- **Teste:** X13. Mini-busca pareada com 3 seeds de busca, RS + Flowers.
- **Rejeição:** HV_final(limiar) < HV_final(protegida) com IC 95% bootstrap entre seeds abaixo de 0, **ou** a frente B0 perde ≥ 1 ponto não dominado em ≥ 2 de 3 seeds.

## Bloco E — Embedding e cabeça ≥ 1280

### H5. Uma cabeça 1280 autocontida com 2ª ordem recupera a maior parte do gap COV − ORIG
- **Afirmação:** nos mesmos encoders, uma cabeça autocontida (covariância PCA-k por bloco, normalizada por raiz de matriz, projetada para 1280, ou compact bilinear) dá κ ORIG-val ≥ ORIG atual + 50% do gap (COV − ORIG) do encoder.
- **Evidência:** 2ª − 1ª ordem casada no último bloco +0,072 [−0,08; +0,56] e multiescala 2ª − 1ª +0,133 (`check/gate_A1_audit.md:102`); o head atual não acrescenta nada (A1b:204-217, 264-272; A1d:217-228). Confundidor: a dimensão da leitura de 2ª ordem é maior.
- **Teste:** X02. 3 encoders (tiny, medium, huge) por dataset × {atual; GAP-multiescala→1280; cov-PCA-k→1280; cov + multiescala→1280}. Primeiro sem treino da cabeça (estatísticas e PCA ajustadas só no treino), depois com a cabeça treinada. 3 seeds de cabeça. Custo dentro do orçamento Bk.
- **Rejeição:** recuperação mediana < 50% do gap **ou** Δ ORIG-val < 0,05 sobre a cabeça atual em ≥ 2 de 3 datasets prioritários.

### H6. A normalização da saída dentro da cabeça melhora o κ ORIG
- **Afirmação:** signed-sqrt + L2, ou whitening ajustado no treino, dentro da cabeça eleva o κ ORIG-val em ≥ MDE, sem scaler no SVM.
- **Evidência a favor:** rank efetivo 3,4; 51% da variância num eixo; larvae c00001 vai de 0,000 a 0,785 com PCA-whiten (A1b:219-231).
- **Evidência contra (já existente):** na mediana dos 17 checkpoints, ORIG+PCA-whiten − ORIG = **+0,009** e ORIG+scaler − ORIG = +0,034 (A1b:211-212). A prioridade de H6 é menor que a de H5/H7.
- **Rank efetivo é só diagnóstico (C6).** O braço whitening iguala os autovalores e leva o participation ratio ao posto numérico **por construção**. Por isso o rank não pode ser critério: é reportado e nunca decide.
- **Teste:** X03, sem treino. Mesmos checkpoints do X02.
- **Rejeição:** Δκ ORIG-val < MDE (IC 95% inclui 0) em ≥ 2 de 3 datasets. Também é rejeitada se melhorar só os casos degenerados (κ < 0,2) e piorar os demais. O rank efetivo não entra no critério.

### H7. Os blocos intermediários carregam mais sinal que o último
- **Afirmação:** concatenar estatísticas de blocos intermediários na cabeça (multiescala) melhora o ORIG-val em relação a ler só o último bloco, e mais nos modelos com ≥ 4 blocos.
- **Evidência:** o último bloco é pior que um intermediário em 10 de 11 (A1b:233-239); multiescala de 2ª ordem +0,049 (A1b:217); Belilovsky +1,2 pp (A1d:118).
- **Teste:** braço multiescala de X02 × braço só-último, com interação com a profundidade.
- **Rejeição:** Δ(multiescala − último) < MDE em ≥ 2 de 3 datasets **e** sem interação positiva com nº de blocos ≥ 4.

## Bloco I — Inicialização

### H8. Na pipeline atual, a init SPiFiL supera a aleatória
- **Afirmação:** na mesma topologia efetiva (gêmeo pela `effective_spec`), SPiFiL-0 ≥ He-aleatório + MDE em ORIG-val, com `protect=freeze` e com `full`.
- **Evidência:** o único controle é antigo e favorece o aleatório em 6 de 8 (Δ −0,023, p = 0,20, confundido pela largura) (A1a:291-306). Hoje não há gêmeo (A1a:306).
- **Teste:** X04. RS, Flowers, HAM, protozoan; profundidade 3 e 5; 3 seeds.
- **Rejeição:** IC 95% de Δ(SPiFiL − aleatório) ≤ 0 em ≥ 2 de 4 datasets. Nesse caso a hipótese central do SPiFiL tem de ser revista; o resultado negativo é registrado e não descartado.

### H9. Candidatos na grade da feature corrigem as camadas profundas
- **Afirmação:** amostrar as sementes na grade da feature, sem reescala de superpixels (modo `spifil_in_feature` ou denso), com N distinto ≥ 5·D, reduz a fração de filtros com par cos > 0,95 (~40%) e melhora o κ ORIG-val dos modelos com ≥ 4 blocos.
- **Evidência:** 10–14 sementes distintas por imagem na grade 7 dos parasitas; N_ef/D < 1 em 57% (A1a:219-243); o orçamento conta duplicatas (A1a:211).
- **Teste:** X05. Mesma arquitetura com 4–6 blocos; RS, Flowers, protozoan, larvae; 3 seeds.
- **Rejeição:** Δκ ORIG-val < MDE (IC inclui 0) nos modelos com ≥ 4 blocos em ≥ 2 de 4 datasets. A queda da redundância é reportada como verificação do mecanismo, não como critério.

### H10. Protótipos com whitening (SPiFiL-1) superam o DiversitySelector
- **Afirmação:** k-means++ com ZCA (ou PCA) sobre patches centrados, com o mesmo K, eleva o rank efetivo dos filtros e o κ ORIG-val.
- **Evidência:** redundância de ~40% já no SPiFiL puro (A1a:268); Coates (whitening crucial para k-means) e PCANet (remoção da média obrigatória) (A1d:34, 69).
- **Teste:** X06. SPiFiL-0 × SPiFiL-1, mesmo K e arquitetura; RS, Flowers, HAM; 3 seeds.
- **Rejeição:** Δκ ORIG-val < MDE (IC inclui 0) em ≥ 2 de 3. O rank efetivo dos filtros/ativações é reportado só como verificação do mecanismo, nunca como critério (C6).

### H11. Os rótulos não são necessários na init
- **Afirmação:** trocar o Fisher supervisionado por um escore label-free (`DistanceSumScorer` ou SPiFiL-1) muda o κ ORIG-val em menos que o MDE.
- **Evidência:** a init atual é supervisionada (A1a:207-214); ipc está confundido com largura (A1a:245).
- **Teste:** X06b. Mesmo desenho do X06.
- **Rejeição:** Δ(supervisionado − label-free) ≥ MDE com IC > 0 em ≥ 2 de 3. Nesse caso os rótulos ajudam e isso precisa ser declarado.

## Bloco L — Aprendizagem

### H12. Com o regime da busca, o AE local é neutro
- **Afirmação:** com 2 épocas por camada, AE (pixel ou masked) − noAE dá IC contido em ±MDE. Com 15 épocas e alvo pixel/masked, Δ ≥ MDE.
- **Evidência:** joint × decoder_first −0,001; Fase 0 +0,010 (n=4); alvo feature é o pior (A1b:64-86).
- **Teste:** X07. {noae, pixel, masked, feature} × {2, 15 ép/camada}; RS + Flowers; 3 seeds. Épocas de SSL iguais entre braços, para remover o confundidor da Fase 0. Mede também a **preservação da representação**: CKA linear e similaridade de cosseno das ativações por bloco antes e depois do AE e do SSL, em imagens de treino (diagnóstico pedido pela MISSION:38, "preservação por aprendizagem local").
- **Rejeição** (da parte "15 ép ajuda"): Δ(AE15 − noAE) < MDE em ≥ 2 de 2 datasets.

### H13. O LeJEPA + SIGReg prejudica o ORIG em parasitas pre, e a máscara de objeto nas vistas remove o prejuízo
- **Afirmação:** em larvae e protozoan (pre), o SSL longo reduz o κ ORIG-val em relação ao α=0. Com vistas restritas ao objeto (máscara), Δ ≥ 0.
- **Evidência:** ORIG piora em 5 de 11 de α=0 ao α salvo (A1b:241-250); sementes larvae −0,12 a −0,34 e protozoan nunca > L0 (A1b:101-107; FILA:245-247).
- **Teste:** X08. Mesmo L0, SSL {sem, LeJEPA, LeJEPA + máscara}, checkpoint escolhido por ORIG-val (nunca pela perda); larvae, protozoan, aid (controle positivo); 3 seeds.
- **Rejeição:** em parasitas pre, Δ(LeJEPA − sem SSL) ≥ 0 com IC > −MDE (o SSL não prejudica), **ou** Δ(máscara − LeJEPA) < MDE (a máscara não ajuda).

### H14. Herdar o prefixo é melhor que reinicializar
- **Afirmação:** no crescimento Bk → Bk+1 com morfismo (Net2Net), herdar os pesos do prefixo dá κ ORIG-val ≥ reinicializar a mesma arquitetura final com SPiFiL + AE.
- **Evidência:** Δκ das mutações é negativo inclusive com preservação de função no PNAS-CS (A1a:162-168); `widen_block` só preserva em blocos planos (FILA:172).
- **Teste:** X14. RS + Flowers; 5 pares de crescimento; 3 seeds.
- **Rejeição:** Δ(herança − reinício) < 0 com IC < 0 em ≥ 1 dataset, ou IC inclui 0 nos dois. No segundo caso não há vantagem demonstrável da herança.

## Bloco A — Arquitetura e escala

### H15. Com a init corrigida, a penalidade de profundidade some, e o ótimo depende do domínio
- **Afirmação:** com o mesmo orçamento de params e largura compensada, a profundidade ótima em ORIG-val é 2–3 blocos em parasitas pre e 4–6 em Flowers/RS. Com H9 aplicada, o coeficiente por bloco deixa de ser negativo.
- **Evidência:** −0,20 z por bloco; ótimo em 3–4 pools (A1a:92-133); PCANet/Coates/TinyNet (A1d:236, 241).
- **Teste:** X10. Profundidade {2, 3, 4, 6, 8} × init {SPiFiL-0, melhor de X05/X06}; RS, Flowers, protozoan, larvae; 3 seeds.
- **Rejeição:** o argmax da profundidade é igual entre parasitas e Flowers/RS (sem efeito de domínio) **ou** o coeficiente por bloco segue < 0 com IC < 0 depois da init corrigida.

### H16. O κ ORIG satura antes de 5,5M
- **Afirmação:** na curva κ ORIG-val × params (B0 … B4, melhor por faixa, cabeça corrigida), Δ(B4 − B2) < MDE nos datasets prioritários.
- **Evidência:** saturação em B0/B1 em 6 de 11 (COV); B4 vazio (A1a:176-191); ORIG +0,013/década (A1b:169-174).
- **Teste:** X15. Crescimento B0 → B4 com o melhor mecanismo dos Agentes 2/3; 3 seeds.
- **Rejeição:** Δ(B4 − B2) ≥ MDE com IC > 0 em ≥ 2 de 3 datasets.

### H17. Crescer em profundidade primeiro é melhor nos portes pequenos
- **Afirmação:** em B0–B1, no mesmo FLOPs (2·MAC), crescer em profundidade ≥ crescer em largura. Em B2+, compound > dimensão única.
- **Evidência:** TinyNet e EfficientNet (A1d:159, 210-211); medição nossa contrária: largura correlaciona com κ COV (A1a:83).
- **Teste:** X16. RS, Flowers; 3 seeds.
- **Rejeição:** largura primeiro > profundidade primeiro com IC > 0 em B0–B1 nos 2 datasets.

## Bloco V — Vazamento (sensibilidade; não muda a seleção)

### H18. O vazamento conhecido não altera rankings, só o κ absoluto
- **Afirmação:** excluindo recapturas e duplicatas (eggs, larvae, flowers, aid) e restringindo o HAM às imagens de teste com lesão fora de train∪val, a **ordem** dos finalistas congelados (e a nossa posição em relação ao FM do mesmo domínio) não muda. O κ absoluto do HAM cai mais nas classes MEL/DF/BCC.
- **Evidência:** A1c:7-17, 26-28, 98-115; CHECK:146-154.
- **Teste:** X17. Pós-congelamento, sobre as predições salvas; sem treino.
- **Rejeição:** alguma inversão de ordem entre finalistas cujo Δ original era > MDE, **ou** mudança de sinal de (nosso − FM) em algum dataset. Isso é registrado como resultado ("o vazamento afeta o ranking").
- **O número oficial NÃO depende deste teste (C10).** Ele já está pré-registrado no `evaluation_contract.md` §7.1: a coluna limpa é a oficial em eggs, larvae, flowers e aid; no HAM o oficial é `test_full` com `test_lesion_clean` ao lado (decisão do usuário, 17/11).
