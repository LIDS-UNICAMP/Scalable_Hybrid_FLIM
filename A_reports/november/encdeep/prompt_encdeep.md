# Prompt para continuar a fila: aprender L4 e L5 do SPiFiL sem backpropagation e sem rótulos

Continue o trabalho na sessão que já gerencia a fila de experimentos. Implemente, verifique e acrescente à fila somente os experimentos faltantes deste prompt, preservando os jobs em andamento. O objetivo é melhorar o **encoder SPiFiL**, principalmente da quarta camada em diante, para produzir uma representação separável pelo **mesmo SVM linear**. Não resolva o problema aumentando a capacidade do classificador.
As decisões atuais do usuário são: encoder sem rótulos, sem backpropagation e sem máscaras; incorporar mecanismos da literatura no próprio encoder; manter o SVM linear como único classificador e sonda; investigar como montar e aprender L4 e L5. Não executar o FP original com rótulos, nem escondê-los em pseudo-rótulos ou seleção por classe.

## 1. Contratos que não podem mudar

1. Apenas imagens do treino do split ajustam filtros, estatísticas, normalizações, projeções, alvos e pares do encoder. O dataloader de aprendizado não entrega `y` ao mecanismo. Rótulos de treino entram somente no SVM; os de validação entram somente na avaliação e seleção dos experimentos.
2. Não selecionar imagens, patches, negativos ou kernels por classe. Não usar máscaras de segmentação, mapas derivados de rótulos, teachers pré-treinados por backprop, previsões do SVM ou pseudo-rótulos como alvos do encoder.

4. SVM **linear**, configuração congelada por comparação: mesmo C, fração rotulada, kernel, 

O SVM é supervisionado, mas o encoder não. Preditores laterais de CLAPP comparam vistas e não produzem classes; são mecanismos autossupervisionados temporários, não classificadores adicionais.

## 2. Decomposição atômica, resultados e dependências

As unidades são: descobrir estado e contratos; montar suporte espacial de L4/L5; adaptar aprendizado competitivo; implementar plasticidade preditiva local; implementar alternativa algébrica de invariância; integrar os módulos; avaliar pilotos; confirmar finalistas. Cada módulo tem dono exclusivo. Integração, avaliação comum e inserção na fila pertencem ao coordenador.

```text
G0 (1) seq.  ┌────────────────────────────────────────┐
Estado real  │ A0                                     │
             │ Estado e contratos                     │
             └────────────────────────────────────────┘
                                ↓ contratos
G1 (3) ║     ┌────────────────────────────────────────┐
Sem escrita  │ H1          C1          S1             │
compartilhada│ Hebb        Predição    Invariância    │
             └────────────────────────────────────────┘
                                ↓ por módulo pronto
G2 (1) seq.  ┌────────────────────────────────────────┐
Dono comum   │ A0                                     │
             │ Integração                             │
             └────────────────────────────────────────┘
                                ↓ checks aprovados
G3 (2) ║     ┌────────────────────────────────────────┐
Fila vigente │ E1                      D1             │
             │ Pilotos                 Diagnóstico    │
             └────────────────────────────────────────┘
                                ↓ evidência pareada
G4 (1) seq.  ┌────────────────────────────────────────┐
Sem teste    │ A0                                     │
na seleção   │ Confirmação e relato                   │
             └────────────────────────────────────────┘
Legenda: ║ concorrência numa chamada; seq. execução sequencial.
Ciclo: falha em G2 retorna ao dono de G1; falha de piloto retorna
ao módulo correspondente, sem bloquear os módulos independentes.
```

| Grupo | Agente | Saída e propriedade | Dependência |
|---|---|---|---|
| G0, sequencial | A0, coordenador | Estado real, contratos, arquivos de cada dono e protocolo de inserção | Nenhuma |
| G1, concorrente | H1 | SoftHebb em L4/L5, whitening e contratos da geometria | Contrato do tronco de G0 |
| G1, concorrente | C1 | CLAPP manual e extensão espacial compacta | Mesmo contrato de G0 |
| G1, concorrente | S1 | SFA sobre respostas SPiFiL e regressão de consistência opcional | Mesmo contrato de G0 |
| G2, sequencial por integração | A0 | Schema, dispatcher, extrator, avaliação comum, YAMLs e fila | Cada módulo aprovado, sem esperar os demais |
| G3, concorrente | E1 e D1 | E1 acompanha pilotos; D1 analisa erros e métricas de camadas | Artefatos disponíveis de cada braço |
| G4, sequencial por finalista | A0 | Confirmação, comparação e próximos jobs faltantes | Evidência do respectivo piloto |

Usar no máximo três workers de desenvolvimento, além do coordenador, ou menos se o ambiente permitir menos agentes. Concorrência de agentes não autoriza concorrência adicional de GPU. A0 define arquivos reais antes do despacho; H1/C1/S1 não editam simultaneamente `eval/spf_grade.py`, schema, entrada principal, SVM ou arquivos de fila. Se for preciso criar módulos, os caminhos abaixo são sugestões, não arquivos comprovadamente existentes: `flim/local_plasticity.py`, `flim/local_predictive.py`, `flim/slow_features.py`. Preferir os módulos já presentes.

## 3. Primeiro descubra o estado atual, sem parar a fila

Leia AGENTS.md, CLAUDE.md e INDEX.md aplicáveis. Se houver `graphify-out/graph.json` e CLI funcional, consultar o grafo antes da busca textual; depois ler os detalhes que o grafo não trouxer. Não reconstruir o grafo por rotina.

Os caminhos seguintes são referências dos relatórios de 08/11/2026, não uma afirmação do estado atual:

| Referência | O que conferir |
|---|---|
| `/dados/home/moliveira/Scalable_Hybrid_FLIM_discovery`, branch `discovery` | Worktree, branch e alterações dos agentes atualmente em uso |
| `/dados/home/moliveira/miniforge3/envs/scalable_FLIM` | Python absoluto, versão, dependências e ambiente efetivo |
| `/exps/moliveira/queues/`, `fila_athena`, `fila3` | Gerenciadores ativos, locks, manifests, tetos e inserção segura |
| `/exps/moliveira/LEIAME_pastas.md`, `/exps/moliveira/reports/rodada2/` | Organização e resultados posteriores ao snapshot |
| `experiments/autoencoder/`, `python -m experiments.ray.launch` | Schema, despacho e convenção reais da sessão |
| `eval/spf_grade.py`, `eval/svm.py`, `eval/l1_amostras.py`, `flim/arch.py` | Chamadores, implementação corrigida e contratos de extração |
| SPiFiL editable externo | Apenas leitura; não modificar o pacote |

As referências antigas a Athena-only e tetos de 40/46 GB foram seguidas por uso de Athena e Jaci e outros limites. Reutilize apenas máquinas já autorizadas na sessão atual e os tetos vigentes. Não aumentar recursos por inferência a partir deste texto. Confirmar `LD_LIBRARY_PATH`, `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO` e demais variáveis realmente necessárias sem substituir configurações correntes arbitrariamente.

Não matar, suspender, reiniciar ou alterar processos em andamento, seus ou de terceiros. Não criar uma fila paralela. Para cada nova célula, procurar primeiro nos concluídos, em execução, esperando e interrompidos. `skip: state` e nome de YAML não bastam para identificar trabalho pronto. Um job existente compatível vira dependência, não duplicata.

Conferir especialmente SoftHebb L3/L4, controles, resoluções 320/448 e `leitura_a2`, já descritos como prontos ou enfileirados. Não retomar Food101 nem acrescentar benchmark fundacional pesado. Reservar prefixo livre como `disc_md_encdeep_*`. Preflight também usa diretório novo, pois o dry-run histórico podia escrever `jobs/*.yaml`.

## 4. Ponto de partida científico e pergunta principal

O snapshot mais recente relata: whitening entre camadas como maior ganho; SoftHebb corrigido positivo em L3/256, sem whitening/saliência; leitura multiescala com dimensão igual positiva; coconut com última camada aproximadamente 0,79 contra concatenação aproximadamente 0,63, ainda piloto de um split. `sol_wsal` tinha três splits e um sorteio, não cinco sorteios. Não extrapolar esses resultados para L4/L5.

Priorizar `aid`, `flowers`, `remote_sensing` e `coconut_trees`, a 50% na definição já usada pela fila. Parasitos servem depois como contraste para uma regra que se generalize; não reabrir automaticamente as 12 bases. Meta aspiracional registrada: flowers e remote_sensing acima de 0,70, sem prometer esse valor.

Pergunta central: **L4 e L5 aprendem relações entre padrões de L3 e conservam informação útil para a separação linear, ou apenas acumulam compressão, redundância e invariâncias inadequadas?**

O mecanismo procurado tem três partes: composição espacial de detectores anteriores; estabilidade a variações da mesma imagem; preservação de diversidade e informação. Um campo receptivo maior, maior posto efetivo ou mais profundidade não comprovam abstração semântica. A evidência principal continua sendo a melhora fora do fitting com o SVM linear fixo.

## 5. Como montar L4 e L5 a partir do SPiFiL

1. Recuperar o checkpoint L3 compatível já existente. Congelar L1-L3, inclusive estatísticas que possam mudar durante o novo fitting. Registrar sua configuração e hash; não escolher outro tronco para favorecer um mecanismo.
2. Extrair, pelo caminho atual da grade, patches das features L3 de imagens de treino. Preservar o contrato de projeção das sementes para a grade de features. Não aplicar DISF diretamente a 256 canais sem implementação comprovada. Verificar bordas, posições repetidas, largura real e kernels únicos.
3. Inicializar L4 por FPS/SPiFiL sobre essas features, preferencialmente com kernel 3×3 e 256 filtros, se o orçamento real permitir. Um filtro L4 passa a combinar padrões de L3 em posições vizinhas, em vez de apenas recortar RGB novamente.
4. Aprender L4 com um dos mecanismos abaixo. Congelar sua versão escolhida usando validação, com estado zero elegível. Calibrar a entrada de L5 somente com imagens de treino passadas pela L4 efetivamente escolhida.
5. Inicializar L5 por FPS/SPiFiL sobre essa nova representação L4 e aprender L5. Não reutilizar banco/whitening calculado sobre outra L4. Salvar L3, L4 e L5 como etapas identificáveis.
6. Manter largura e número de kernels comparáveis entre mecanismos. Se não houver candidatos únicos suficientes, registrar a largura real e resolver a causa com o contrato existente; não inserir canais aleatórios sem identificação.

### 5.1 Contexto espacial: três arquiteturas, sem busca combinatória

Reutilizar o operador de contexto de [Yu e Koltun, ICLR 2016, §2-3](https://arxiv.org/abs/1511.07122): dilatação amplia suporte espacial sem acrescentar coeficientes ao kernel. O artigo usa backprop; aqui se transfere apenas o operador e se aprendem os pesos por SPiFiL/plasticidade local.

| ID | Geometria | Pergunta |
|---|---|---|
| G0 | Continuação L4/L5 com pooling vigente | Referência geométrica |
| G1 | G0, retirando somente a redução espacial após L4 | L5 recebe uma grade mais rica? |
| G2 | G1, com dilatação 2 na convolução L5 de kernel 3×3 | Contexto maior acrescenta informação? |

Calcular H×W, stride acumulado e campo receptivo de cada operação, incluindo pooling. Não presumir que todas as bases chegam à mesma grade. A dilatação 2 de um kernel 3×3 cobre suporte efetivo 5×5 com os mesmos nove coeficientes, mas não equivale a um kernel denso 5×5. Registrar cobertura e possível efeito de grade esparsa.

**Extração e aprendizado precisam usar a dilatação real.** Os candidatos SPiFiL, N0/ZCA e patches Hebbianos de G2 usam os mesmos offsets, padding e stride da convolução dilatada. Não treinar patches contíguos e mudar apenas a inferência para `dilation=2`.

Não combinar essas três geometrias com todos os mecanismos de início. Primeiro comparar G0/G1/G2 com o mesmo SoftHebb local e a mesma leitura. Usar a geometria escolhida nos candidatos novos. Manter controles de zero atualização com geometria idêntica.

## 6. H: aprender composições por competição Hebbiana em L4/L5

Fonte: [SoftHebb, ICLR 2023](https://arxiv.org/abs/2209.11883) e [implementação dos autores](https://github.com/NeuromorphicComputing/SoftHebb/blob/main/demo.py). Aproveitar a implementação corrigida da sessão, sem reintroduzir competição excessiva, normalização por filtro no lugar de tensor ou taxas iguais entre camadas. Não copiar a cabeça neural do demo.

O mecanismo seleciona respostas competitivas e atualiza filtros por atividade local. Em L4/L5, o estímulo é uma composição de features anteriores. A novidade a medir é aprender essas composições profundas, em vez de apenas empilhar filtros estáticos. A aplicação ao SPiFiL profundo, com seu orçamento e whitening, é adaptação a testar.

Usar aprendizagem stagewise para o ensaio profundo: tronco anterior congelado, estatísticas de entrada calibradas no treino, atualização da camada atual, congelamento e depois construção da seguinte. Identificar essa agenda como adaptação, pois o demo canônico pode atualizar várias camadas no mesmo forward.

Comparar H0/H1/H2/H3 no mesmo backbone e agenda: zero atualização sem/com whitening, SoftHebb sem/com whitening. Os quatro braços usam a MESMA ativação, pooling, bias e calibração. Não comparar ReLU no controle com Triangle no braço aprendido e atribuir a diferença apenas ao Hebb. Preservar a receita ReLU antiga como referência histórica adicional.

### 6.1 Coordenadas corretas do whitening

Se o patch cru coluna é p, a transformação efetiva congelada é v=A(p−m), e a resposta é u=Qv+b, a convolução dobrada usa K=QA e b_cru=b−QAm. A deve compor ZCA e N0 na ordem real do código.

Aprender Q no espaço v, e não atualizar K como se ele ainda estivesse nesse espaço. Manter A/m fixos durante o fitting da camada. Só recalibrar a entrada da próxima depois de congelar a anterior. Verificar equivalência numérica do forward explícito e dobrado, inclusive padding e bordas. Não alterar A ao final sem compensar os pesos.

Se o fatorial equivalente já estiver concluído ou rodando em L3/L4, reutilizá-lo. O novo trabalho central é sua extensão controlada para L5; não repetir a confirmação L3 apenas para gerar mais linhas.

## 7. C: plasticidade preditiva local para aprender L4/L5 mais úteis

### 7.1 Por que este artigo é pertinente à profundidade

[CLAPP, NeurIPS 2021, equações 3 e 6-8 e experimentos visuais](https://proceedings.neurips.cc/paper/2021/hash/feade1d2047977cd0cefdafc40175a99-Abstract.html), investiga hierarquias profundas usando atividade local, contexto preditivo e modulação por pares. Ele oferece um sinal de aprendizado para camadas altas além de representar apenas a distribuição instantânea de patches.

**Atenção à implementação:** o [README dos autores](https://github.com/EPFL-LCN/pub-illing2021-neurips) informa que seu loop usa autograd para emular as regras. Não importar esse treinamento. O arquivo [compare_updates.py](https://raw.githubusercontent.com/EPFL-LCN/pub-illing2021-neurips/master/vision/CLAPPVision/vision/compare_updates.py) contém o cálculo explícito que deve orientar nossa implementação manual. Não usar o checkpoint publicado: ele não pertence ao aprendizado SPiFiL desta experiência.

### 7.2 Pares sem classes e regra manual

Construir duas vistas da mesma imagem de treino para um par positivo; usar outra identidade de imagem para o negativo. O sinal q∈{+1,−1} indica a origem do par, não um rótulo de categoria. Não filtrar negativos por classe. Documentar que imagens diferentes podem pertencer à mesma classe, sem resolver isso consultando os rótulos.

Começar com geometria preservada e variação fotométrica leve. Para crops ou escala, usar grande sobreposição e correspondência conhecida; não afirmar que dois crops sem conteúdo comum sejam o mesmo padrão. Não eliminar cor em flowers como transformação padrão, nem usar masks/recortes semânticos.

Na mesma camada convolucional, sejam X_a/X_b mapas de entrada, a_a=Conv(W,X_a)+b, a_b=Conv(W,X_b)+b, F_a=ReLU(a_a), F_b=ReLU(a_b). U é o operador linear de average pooling com vetorização interna. Na versão simples C0, sua saída é explicitamente 1×1 por canal. Definir c=U F_a, z=U F_b, P como preditor lateral e R como via recíproca. Para um par, com x_a,p e x_b,p indicando os patches que entram na posição p:

```text
score = z^T P c
g = q * indicador(q * score < 1)
m_b = U^T(P c) * indicador(a_b > 0)
m_a = U^T(R z) * indicador(a_a > 0)
ΔW[j,:] = η g soma_p(m_b[j,p] x_b,p^T + m_a[j,p] x_a,p^T)
Δb[j] = η g soma_p(m_b[j,p] + m_a[j,p])
ΔP = η_p g z c^T
ΔR = η_p g c z^T
```

Os produtos `*` dos gates são elemento a elemento. Uᵀ distribui a modulação às posições do mapa pelo operador linear transposto explícito. Acumular a convolução com patches/unfold e média sobre os pares do lote. U já inclui a média espacial: não dividir novamente pelo número de posições sem identificar essa mudança de escala. Calcular todos os incrementos com o mesmo snapshot, antes de alterar W/P/R. Conferir índices, sinais e escalas contra as equações publicadas e o cálculo explícito dos autores. Não omitir o segundo termo quando as vistas compartilham W. Nesta seção W tem orientação C_out×d; na seção algébrica seguinte usa-se sua transposta d×C_out para patches linha.

Começar pela variante simétrica com R=Pᵀ, mantida por construção, para reduzir parâmetros e facilitar verificação. Armazenar apenas P, aplicar ΔP uma vez e definir R por transposição; não somar ΔRᵀ novamente a P. Aprender apenas a convolução L4 e seu preditor; congelar L4, descartar o preditor de treino, inicializar L5 e repetir. Não retropropagar erro de L5 para L4 ou L3.

A adaptação SPiFiL precisa declarar onde o sinal local é calculado em relação ao pooling. No piloto, calcular no mapa conv+ReLU anterior ao pooling da próxima camada, com referência estática na mesma arquitetura. Isso é uma adaptação operacional, não reprodução literal de todo o experimento original.

### 7.3 Extensão espacial informada por CLAPP++

O trabalho [Can Local Learning Match Self-Supervised Backpropagation?, 2026, §3.3 e equação 7](https://arxiv.org/html/2601.21683v2), com aceitação ICML 2026 informada no [registro dos autores](https://arxiv.org/abs/2601.21683), aponta a dependência espacial do pretexto como um fator relevante. Não copiar seu loop com stop-gradient/autograd nem apresentar a versão abaixo como reprodução integral de CLAPP++.

Após a versão manual simples passar, testar um score lateral espacial compacto:

```text
c = vec(AdaptiveAvgPool(output_size=(2,2), F_a))
z = vec(AdaptiveAvgPool(output_size=(2,2), F_b))
score = z^T P c
```

Manter o sinal de pares e a regra de modulação. C0 tem saída global 1×1; C1 tem saída espacial 2×2, com todo o restante idêntico. Não confundir `output_size=(2,2)` com kernel de pooling 2. Distribuir explicitamente cada componente da modulação às posições de sua célula, respeitando área real dos bins e somando contribuições quando houver sobreposição. Implementar U/Uᵀ por álgebra explícita, sem autodiff. Não incluir feedback da última camada nesta primeira extensão.

Com C=256, o vetor interno tem 1.024 entradas e P tem 1.048.576 coeficientes. Uma camada ativa de cada vez limita os parâmetros temporários. Reportar o custo do preditor além do encoder exportado. Se H ou W não comportar 2×2, o braço não testa dependência espacial e precisa ser identificado como tal. O `vec` interno não é flatten entregue ao SVM.

Pergunta: a modulação por relações espaciais melhora L4/L5 sobre CLAPP com agregação global, sob a mesma inferência? Não chamar ganho de concatenação ou de maior dimensão de ganho dos kernels profundos.

## 8. S: aprender estabilidade sem apagar informação nas camadas profundas

Fontes: [DL-SFA, CVPR 2014, §3.1-3.2](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Sun_DL-SFA_Deeply-Learned_Slow_2014_CVPR_paper.pdf), aprende uma hierarquia de filtros lentos por álgebra linear; [Bio-SFA, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/ab73f542b6d60c4de151800b8abc0a6c-Abstract.html), fornece uma formulação de plasticidade local; [information-preserving SFA, Machine Learning 2020, §1 e §4](https://link.springer.com/article/10.1007/s10994-019-05860-9), explica por que descartar cedo informação pode prejudicar níveis superiores. Neste último, usar apenas o princípio e a variante sem supervisão, não grafos construídos por rótulos.

Nossa adaptação substitui continuidade temporal por duas vistas da MESMA imagem. Isso aprende estabilidade às transformações escolhidas; não prova que estamos recuperando categorias semânticas. Manter o SVM linear, sem copiar o classificador χ² usado na avaliação do DL-SFA.

### 8.1 Ajuste sobre respostas dos próprios filtros SPiFiL

Inicializar L4 com SPiFiL e congelar um snapshot. Coletar suas preativações locais correspondentes u_a/u_b de treino, antes da ReLU. Trabalhar inicialmente nas 256 respostas desses filtros, evitando um Gram de todos os elementos dos patches 3×3×256.

Estimar média μ, covariância C0 das duas vistas centradas e CΔ=E[(u_a−u_b)(u_a−u_b)ᵀ]. Resolver CΔ v=δ C0 v, com v_iᵀ C0 v_j=δ_ij. As menores razões δ indicam direções mais estáveis, sujeitas a variância e descorrelação. Implementar por eigendecomposição do subespaço de C0 com variância positiva e depois diagonalização da covariância de diferenças branqueada. Excluir direções numericamente nulas por tolerância previamente declarada.

Não alegar variância unitária apenas porque se adicionou ε: medir a covariância real da saída no treino. Uma invariância perfeita obtida por colapso não satisfaz o objetivo.

### 8.2 Preservar uma parte da informação para L5

O piloto usa largura total 256 dividida em 128 canais lentos e 128 canais reconstrutivos do resíduo, sem grade de proporções. Com respostas centradas X:

```text
S = X R_slow
P_rec = solução estável de mínimos quadrados de S P_rec ≈ X
E = X − S P_rec
V_res = 128 componentes principais de E
R_all = [R_slow, (I − R_slow P_rec) V_res]
```

Essa construção conserva informação não explicada pelas direções lentas. PCA aqui serve ao resíduo das respostas e não reabre o ensaio de PCA/ZCA dos kernels que já perdeu. Se o posto real não comportar 128+128, não fabricar componentes: registrar a limitação e reduzir a largura explicitamente, com controle pareado da mesma largura.

Fixar a escala relativa antes da ReLU: calibrar cada bloco centrado para RMS unitário usando apenas treino, com piso numérico predefinido. Aplicar a mesma convenção nos controles correspondentes; medir e reportar os fatores. Incorporar os dois fatores escalares nas respectivas colunas de R_all antes de dobrar a transformação. Uma parte não pode dominar apenas por ter amplitude maior. Essa calibração compacta é parte da adaptação; não afirmar reprodução exata do escalonamento iSFA. Fixar também a convenção de sinal dos autovetores para reprodução entre runs.

Com a convenção de patches linha, a preativação inicial é U=A_patch W0+b0. Dobrar a transformação na própria convolução:

```text
W_novo = W0 R_all
b_novo = (b0 − μ) R_all
```

Depois aplicar ReLU e o pooling estabelecido e construir L5 sobre essa nova L4. Não colocar essa mistura apenas no vetor final: uma rotação linear invertível antes do SVM não cria nova separabilidade linear. O mecanismo precisa ocorrer antes da não linearidade interna e da composição seguinte. A garantia de estabilidade vale para o espaço ajustado, não automaticamente para a saída pós-ReLU.

Comparar: SPiFiL estático; SFA com mesma largura factível; SFA com preservação residual. O braço slow-only com todas as direções pode ser apenas uma mudança de base; identificar esse limite. Só aplicar novamente em L5 se o primeiro estágio mantiver informação e apresentar sinal útil no piloto.

## 9. Como aproveitar Forward Projection sem violar o encoder sem rótulos

O [Forward Projection original, Nature Communications 2026, suplemento §1.2](https://arxiv.org/html/2501.16476v4#S1.SS2) ajusta convoluções por regressão fechada com alvos de entradas E rótulos. **Não executar essa versão nesta rodada.** Não chamar `sign(AQ)` isolado de mecanismo comprovado para abstração.

Se houver orçamento após H/C/S, preparar um único candidato F-U: **regressão analítica de consistência ancorada no SPiFiL**, uma hipótese nossa inspirada no solver, sem evidência publicada de ganho nesta forma.

Em L4, usar pares de patches correspondentes A/B de duas vistas fotométricas leves, W0 do snapshot SPiFiL e alvo sem bias congelado T0=(A W0+B W0)/2. Manter b0 fixo: as predições completas são A W+b0 e os alvos T0+b0. Assim, o bias cancela no fitting abaixo. Ajustar a própria convolução, sem cabeça nova:

```text
min_W (||A W−T0||² + ||B W−T0||²)/(2M) + λ ||W−W0||²
G = (AᵀA + BᵀB)/(2M)
C = (AᵀT0 + BᵀT0)/(2M)
W = solve(G + λI, C + λW0)
```

Para esse T0 pré-ativação, equivale a W=W0−(1/4) solve(G+λI, D W0), com D=(A−B)ᵀ(A−B)/M. Isso explicita o efeito: atenuar direções instáveis entre vistas, sem inventar um alvo de classe.

Acumular G/C em streaming, float64, e resolver por Cholesky/solve, sem inversa explícita. Para d=3×3×256, só o Gram float64 ocupa 40,5 MiB; contabilizar workspace, patches e tempo do solve. Fixar inicialmente λ=0,1·tr(G)/d, como escolha nossa, sem varrer várias escalas. Se G for degenerado, registrar e usar a política numérica predefinida, sem ajuste guiado pelo teste.

Fazer um ajuste por camada com teacher próprio congelado, sem atualizar recursivamente o alvo pelo student. A âncora não garante ausência de colapso: medir variância, canais mortos e norma relativa da atualização. Refinar L4 antes da ReLU e então inicializar L5 sobre sua saída. Menor erro de consistência sem melhora do SVM não prova avanço. F-U é candidato secundário; não atrasar C/S para implementá-lo.

## 10. Multiescala: separar descritor melhor de kernels melhores

### 10.1 Reutilizar a leitura multiescala existente

Localizar no código/manifesto as escalas, pooling e dimensão de `leitura_a2`. Reusar o braço existente ou esperar sua conclusão. Se os metadados não permitirem reprodução, declarar uma nova receita, por exemplo {128, 200, 320}, fixada antes de olhar os novos resultados; não afirmar que é a mesma experiência.

Gerar cada escala da imagem nativa com o preprocessing vigente. Todas passam pelo mesmo encoder, sem pesos independentes por escala. Para a primeira comparação, manter as camadas lidas e GeM p=3 constantes. Formar o descritor como concatenação das leituras por escala, seguida de PCA/whitening e L2 ajustados apenas no treino. Dimensão final igual à referência, limitada pelo posto e abaixo de 2.000.

### 10.2 Aprender os próprios filtros L4/L5 em múltiplas escalas

Reutilizar o mecanismo profundo vencedor, começando por SoftHebb, com pesos compartilhados. Alternar as escalas de treino de forma balanceada por imagem e normalizar a contribuição das posições: escala maior não deve dominar só por produzir mais patches. Fixar total de imagens, atualizações e limite de patches plásticos; se a versão densa exigir amostragem, aplicá-la também ao controle monoescala e documentar a adaptação.

| ID | Aprendizado dos filtros | Inferência | Comparação principal |
|---|---|---|---|
| M0 | Uma escala | Uma escala | Referência |
| M1 | Uma escala | Multiescala | Benefício do descritor |
| M2 | Multiescala | Uma escala | M2−M0 mede o efeito nos filtros |
| M3 | Multiescala | Multiescala | M3−M1 mede o efeito nos filtros com a mesma leitura |

M0/M1 compartilham checkpoint, assim como M2/M3. Não reaprender quatro encoders para esse 2×2. Pesos compartilhados e exposição a escalas não garantem invariância; medir estabilidade entre vistas. Se M1 ganhar e M2 não, o avanço foi da leitura, sem prova de kernels mais abstratos.

Não reabrir multiescala dos pontos, DoG ou novas fusões de saliência. Esta rodada trata do aprendizado dos mapas profundos e de seu descritor exportado.

## 11. Investigar erros de remote_sensing para decidir mudanças no encoder

Executar essa análise apenas com treino e validação e sem bloquear mecanismos independentes. Não mudar rótulos, classes ou splits para aumentar κ.

1. Com o SVM linear oficial, guardar ID, rótulo, predição e recorrência do erro entre sementes para as leituras isoladas L3/L4/L5. Guardar scores com sua semântica real: no SVM OVO, não subtrair colunas como se fossem probabilidades por classe.
2. Construir matriz de confusão, recall por classe e comparação pareada: erros persistentes; erros resolvidos por L4; resolvidos por L5; erros introduzidos pela profundidade. Usar todos os erros de validação no cálculo e uma galeria pequena predefinida para inspeção.
3. Comparar imagem nativa, preprocessing efetivo e vistas de resolução/escala já autorizadas. Verificar perda de detalhe, letterbox/crop, recorte de contexto e cache reduzido. Não aplicar zoom ao cache de 200 px e alegar recuperação de informação.
4. Fazer intervenções sem máscaras: transformações geométricas leves conhecidas, pequenas mudanças fotométricas e, se necessário, dividir a imagem em blocos fixos e permutar sua ordem. Aplicar a mesma intervenção a acertos e erros. Uma permutação forte é fora da distribuição e não prova sozinha dependência de relações espaciais.
5. Produzir uma tabela hipótese → evidência → experimento do encoder, sem selecionar amostras do treino por categoria para fabricar o mecanismo.

| Hipótese | Evidência procurada | Comparação que retorna ao encoder |
|---|---|---|
| Falta detalhe | Erros caem com vistas nativas maiores, sob leitura igual | M2/M0 e resolução nativa já enfileirada |
| Falta contexto espacial | Ganho com G1/G2 ou modulação espacial, além do efeito de resolução | G0/G1/G2 e C simples/espacial |
| Instabilidade a aparência/escala | Mesma imagem muda muito de representação sob transformação leve | C, S ou M2 sob inferência fixa |
| Informação perdida pela profundidade | L5 piora L4 e desaparecem padrões preservados no tronco | G1, preservação residual em S e estado zero |
| Sobreajuste do mecanismo | Separabilidade de treino sobe, validação não | Reduzir passos ou selecionar zero, sem mudar SVM |
| Diferença na semântica do alvo | Casos visualmente ambíguos e discordâncias entre taxonomias | Relatar limite da tarefa; não aumentar arquitetura automaticamente |

Nos relatórios posteriores, os supostos 568 grupos de remote_sensing foram identificados como colisões de números de arquivo, sem representar aquisição. Não usá-los como grupos de bootstrap. Os rótulos derivam da classe mais frequente de caixas de detecção e não necessariamente da classe de cena. As 89 discordâncias de nomes entre 453 pares com AID são alerta de semântica, não 89 erros comprovados nem teto de κ=0,73.

Em coconut, procurar proveniência real dos dois mosaicos e coordenadas existentes para uma confirmação espacial separada. Não inventar coordenadas a partir de nomes. Se não for possível, declarar a limitação e preservar os splits legados. Não usar imagens correspondentes de AID e remote_sensing em lados diferentes de uma confirmação supostamente independente.

## 12. Como provar contribuição de L4 e L5

Fixar por base uma dimensão comum de descritor, no máximo 256 no diagnóstico inicial e limitada pelo posto real do treino. PCA/whitening/L2 fazem parte do contrato constante e são estimados só no treino. Não procurar nova normalização por mecanismo. Nas comparações com seleção interna, refazer todas as estatísticas no treino interno correspondente.

Avaliar com o MESMO SVM linear:

- L3 isolada, como referência do tronco congelado.
- L4 isolada antes e depois do mecanismo.
- L5 isolada com sua referência estática e sua versão aprendida.
- Concatenação vigente, apenas como resultado adicional, na mesma dimensão final.
- Remoção de L4/L5 do descritor concatenado, como ablação de contribuição.

Guardar κ de Cohen, F1 macro, acurácia balanceada e recall por classe. Separar seleção de checkpoint de medição do ganho: usar a mesma regra e orçamento de escolha em todos os braços, incluindo zero. Melhorar apenas o melhor de muitos checkpoints não equivale a melhorar a receita congelada.

Diagnósticos sem rótulos: espectro das ativações, posto efetivo, canais mortos, correlação funcional, filtros únicos, estabilidade entre vistas da mesma imagem e distinção entre imagens diferentes. Não confundir estabilidade constante com boa representação. Diagnósticos com classes, se necessários para explicar erros, acontecem após congelar o encoder e não realimentam suas atualizações.

Para interpretar os kernels profundos, guardar exemplares de treino que mais ativam cada filtro L4/L5 e desenhar apenas sua região receptiva calculada pela arquitetura. Não usar segmentação, síntese por gradiente ou rótulos para encontrar esses exemplares. Registrar se o filtro combina padrões espaciais distintos, se repete outro filtro ou responde a padding/fundo; as figuras são diagnóstico, não comprovação isolada de abstração.

Classificar cada conclusão:

- **Ganho do encoder:** filtros internos mudaram e o SVM fixo melhorou com a mesma inferência.
- **Ganho do descritor:** pooling/vistas melhoraram, sem evidência de melhora dos filtros.
- **Ganho arquitetural:** geometria mudou; comparar com mecanismo e leitura fixos.
- **Sem contribuição demonstrada:** só melhora loss, estabilidade ou concatenação que depende do tronco.

Não usar aumento do campo receptivo, imagem de ativação ou posto como prova isolada de semântica. Sem rótulos no encoder, a alegação defensável é que composição e invariâncias produziram features mais úteis à separação linear fora do fitting.

## 13. Ordem do lote e regras para evitar explosão de experimentos

| Prioridade | Família | Despacho mínimo | Dependência e reutilização |
|---|---|---|---|
| P0 | Auditoria e controles | Conferir contratos e artefatos existentes | Não ocupa GPU por rotina |
| P1 | H profundo e geometria | L4/L5, G0/G1/G2, mesmo mecanismo; células faltantes | Reusar H×whitening L3/L4 prontos ou pendentes |
| P1 | C manual | L4, depois L5; controles estáticos pareados | Implementação manual aprovada; pode avançar sem esperar S |
| P1 | S com preservação | L4, depois L5 quando justificável | Mesmo tronco e geometria; pode avançar sem esperar C |
| P2 | C espacial compacto | Apenas após C manual válido | Mesmo orçamento de inferência; um preditor ativo |
| P2 | M0-M3 | Um mecanismo profundo selecionado | Reusar leitura multiescala existente |
| P3 | F-U | Um ajuste analítico em L4 e controle | Não executar FP supervisionado; não atrasar P1 |
| P4 | Confirmação | No máximo dois mecanismos novos finalistas, além da referência | Três splits e cinco sorteios, completando só células faltantes |

Piloto inicial: um split e um sorteio predefinidos, nas quatro bases de foco, com orçamento herdado do experimento local compatível. Isso verifica implementação e sinal, não comprova ganho. Usar como checkpoints Hebbianos os já estabelecidos na sessão, incluindo zero; para CLAPP definir antecipadamente um orçamento curto em imagens/pares/updates e medir custo antes de ampliar. Não equiparar 300 passos de Hebb a 300 passos de CLAPP sem registrar lote, vistas e posições.

Antes da confirmação, ampliar os sobreviventes para os três splits e pelo menos outro sorteio. Promover no máximo dois mecanismos por evidência pareada de separabilidade útil e custo aceitável, sem limiar universal inventado. Se nenhum apresentar sinal, concluir os diagnósticos e reportar o resultado negativo; não multiplicar automaticamente hiperparâmetros. A melhor arquitetura ou mecanismo pode depender da base, mas toda regra adaptativa precisa ser explícita e selecionada sem teste.

Não cruzar todos os mecanismos, resoluções, ativações, larguras, saliências e leituras. Primeiro mecanismo causal, depois combinação de finalistas. Manter saliência desligada em AID e remote_sensing; em flowers usar a condição vigente nos pares correspondentes e só estudar sua interação após o mecanismo principal. Coconut não recebe saliência nova por suposição.

## 14. Checks necessários antes de um piloto entrar na fila

Checks pequenos em CPU, no ambiente exato, sem criar uma infraestrutura de testes paralela:

1. Opção desligada preserva saída, contratos e parâmetros da referência. Nova configuração é aceita pelo schema e efetivamente despachada.
2. Encoder aprende sem acesso a labels/masks e com autograd desabilitado. Bloquear chamadas de backward/grad no caminho experimental. Verificar a origem das atualizações manuais, não apenas procurar `.grad`.
3. L1-L3 não mudam; apenas a camada e os auxiliares autorizados são atualizados. Inferência não atualiza nenhum peso ou estatística.
4. Shapes, padding, dilatação e extração de patches concordam. Whitening explícito e dobrado concordam inclusive nas bordas.
5. CLAPP: sinais positivo/negativo, hinge inativo sem atualização, ambos os termos de pesos compartilhados e adjunto explícito do pooling corretos. Usar exemplos numéricos minúsculos e equações manuais; não executar loop de referência com backprop como substituto do mecanismo.
6. SFA: resíduos dos autoproblemas e restrições medidos, posto real respeitado, transformação dobrada numericamente equivalente antes da ReLU.
7. F-U: solução streaming concorda com batch pequeno, residual do sistema correto, alvo congelado e bias consistente.
8. Exportação remove auxiliares de treino de C, preserva a CNN e entrega o mesmo contrato ao extrator/SVM. Nenhum ramo recebe classes na inferência.
9. Contar parâmetros de implantação e parâmetros auxiliares, memória de fitting e número de vistas. Medir piloto antes de extrapolar custo. Não inventar GPU-h.

Com falha, devolver o caso mínimo ao dono do mecanismo; continuar os demais módulos independentes. Não marcar implementação como entregue apenas por haver YAML.

## 15. Inserção concreta na fila existente

A0 mantém um manifesto novo no formato já usado pela sessão, com: ID, família, base, split, fração, seed, hash do tronco, arquitetura real, camada ajustada, init, whitening, saliência, agenda, resolução/vistas, passos/pares/patches, leitura, dimensão, SVM fixo, regime sem rótulos, paths e dependências.

A identidade inclui o conjunto real de imagens usadas pelo encoder e pelo SVM; não confundir treino completo, fração rotulada de 50% e banco de imagens dos filtros. Distinguir comparação pareada, histórico e controles não equivalentes.

Gerar apenas YAMLs faltantes em `experiments/autoencoder/` se esse continuar sendo o caminho vigente. Chaves citadas neste prompt como nomes científicos não são flags existentes: conferir schema e implementar o mínimo antes de escrever YAML válido. Não acrescentar YAML que o dispatcher ignore silenciosamente.

Inserir no final da fila ou na prioridade já autorizada da sessão, usando o protocolo e lock atuais. Não sobrescrever lista consumida por outro processo. Não alterar o código que jobs em andamento ainda possam importar: isolar módulos/configurações novos e verificar o padrão de importação antes de mudar arquivos compartilhados.

O SVM roda depois da extração, em CPU, serializado pelo lock existente e com `nice -n 10`, sem criar outro processo de avaliação concorrente. Limitar também threads BLAS/OpenMP e memória do solver segundo os limites atuais. Não alterar limites de recursos dos jobs já existentes.

Se a fila não tiver inserção segura identificável, preparar todo o lote no formato real, registrar o ponto concreto que impede a inserção e continuar o trabalho independente. Não declarar enfileirado o que apenas foi escrito. Não criar outro scheduler como contorno.

## 16. Entregas e critério científico final

Entregar, nos diretórios de relatórios vigentes da sessão:

1. Estado real antes da inserção e tabela do que foi reutilizado, preparado e efetivamente enfileirado.
2. Mapa artigo → mecanismo → alteração dentro de L4/L5 → código → ablação → resultado, distinguindo reprodução, adaptação e hipótese nova.
3. Tabela por camada L3/L4/L5 com SVM idêntico, referência estática, estado zero e versão aprendida; dimensão e custo juntos.
4. Diagnóstico de remote_sensing que retorne decisões sobre contexto, escala, informação e profundidade, sem corrigir rótulos automaticamente.
5. Resultado dos checks e limites de recursos medidos; estados separados de implementado, smoke, piloto, confirmação e falha.
6. Regra de montagem provisória: como inicializar L4 a partir de L3, como aprendê-la, quando conservar resolução, como inicializar L5 após a nova L4 e quando outra camada deixa de ajudar. Não ajustar uma lei universal a partir de quatro pontos ou de L1.
7. Um comando curto usando o gerenciador existente para o usuário ver jobs rodando e esperando, sem criar daemon de monitoramento adicional.

Os três splits são reamostragens; cinco seeds não criam cinco novos datasets. Reportar diferenças pareadas e sua variabilidade sem tratar 3×5 como 15 amostras independentes. Usar grupos reais de origem se disponíveis, não os falsos grupos de remote_sensing. Teste fica fora da seleção; não abrir teste novo nesta rodada sem a etapa final já autorizada na sessão.

O sucesso procurado é que as camadas profundas acrescentem features úteis à separação linear, sob encoder sem rótulos e sem backprop. Melhorar apenas o pretexto, trocar a sonda, usar supervisão oculta ou esconder L4/L5 fracas na concatenação não satisfaz esse objetivo.

## 17. Referências e limites de transferência

| Fonte primária | Mecanismo utilizado | O que este prompt não presume |
|---|---|---|
| [SoftHebb, ICLR 2023](https://arxiv.org/abs/2209.11883) | Competição e atualização local dos filtros | Que hiperparâmetros e largura do paper se transferem diretamente a SPiFiL |
| [CLAPP, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/feade1d2047977cd0cefdafc40175a99-Abstract.html) | Plasticidade preditiva local para hierarquias | Que usar o loop com backward dos autores cumpra nossa restrição |
| [CLAPP++, 2026, §3.3](https://arxiv.org/html/2601.21683v2) | Modulação espacial compacta | Reprodução integral, feedback da última camada ou equivalência com DINO |
| [DL-SFA, CVPR 2014](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Sun_DL-SFA_Deeply-Learned_Slow_2014_CVPR_paper.pdf) | Aprender filtros invariantes por solução algébrica em hierarquia | Que vídeo e augmentações de imagens estáticas tenham os mesmos resultados |
| [Bio-SFA, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/ab73f542b6d60c4de151800b8abc0a6c-Abstract.html) | Invariância com restrições e plasticidade local | Que seja necessário reproduzir todo o circuito online |
| [Information-preserving SFA, Machine Learning 2020](https://link.springer.com/article/10.1007/s10994-019-05860-9) | Preservação de informação residual para camadas posteriores | Uso de sua variante supervisionada ou dos grafos por classe |
| [Dilated Convolutions, ICLR 2016](https://arxiv.org/abs/1511.07122) | Agregar contexto preservando resolução | Que o treinamento original seja sem backprop |
| [Forward Projection, Nature Communications 2026](https://www.nature.com/articles/s41467-026-69161-1), [suplemento](https://arxiv.org/html/2501.16476v4#S1.SS2) | Solver convolucional fechado, inspirando F-U | Que o algoritmo supervisionado original esteja permitido ou que F-U já tenha ganho demonstrado |

Conferir as versões e trechos indicados antes de implementar. Se uma implementação de referência usar autograd, transcrever e verificar a regra explícita ou deixar o braço como não implementado; não flexibilizar a restrição para dizer que reproduziu o artigo. As equações e contratos deste prompt permitem continuar mesmo que os relatórios históricos não estejam copiados na mesma pasta. Usar fontes e caminhos disponíveis na sessão, sem depender da pasta Downloads do computador de origem.


< coloque na fila na frente dos benchmarks. para serem os proximos a serem rodados.