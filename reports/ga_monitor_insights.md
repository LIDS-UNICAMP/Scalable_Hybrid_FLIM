# Monitoramento das buscas GA com W&B

**Situação em 2026-10-26, 09:47 UTC:** eggs terminou com exit 0 após quatro
gerações; W&B confirmou `finished`, sequência final sincronizada e cinco
artefatos `nas-logs`. Protozoan segue `running`, na quarta geração, com uploads
verificados.
Melhores J completos: protozoan 0,10381153 (profundidade 4); eggs 0,29512415
(profundidade 4), cada um avaliado em seu
próprio dataset, sem comparação de superioridade entre eles.
Os SVMs complementares terminaram em segundos, atingindo o
limite de 10.000 iterações; seus resultados têm convergência limitada.

O monitor lê as execuções `results/nas_protozoan_50_wandb` e
`results/nas_eggs_50_wandb`. O coordenador inicia o daemon após o relançamento.
Esta revisão não altera pesos, configurações, processos ou seleção.

Relatórios automáticos ficam em `reports/ga_monitor_latest.md` e
`results/ga_monitor/status.json`; somente o daemon escreve nesses dois destinos.
Este arquivo registra interpretações do agente revisor em separado.

## Critérios de interpretação

- Menor J válido determina a seleção. Penalidades de indivíduos inviáveis não
  devem aparecer como uma arquitetura válida vencedora.
- L1 menor indica reconstrução melhor, mas pode coexistir com DBI maior,
  cobertura insuficiente ou J sem melhora. Não tratar L1 como fitness.
- Comparar DBI, Hoyer, cobertura e DBCV separadamente. Se a cobertura cair abaixo
  do limite, uma boa DBI nas poucas amostras agrupadas não torna o candidato
  elegível. Hoyer mede embeddings, não pesos esparsos.
- Contar arquiteturas únicas por genoma solicitado, sem duplicar elites,
  resultados retidos e cache. A população de uma geração pode conter cópias.
- Alto descarte por MACs/parâmetros sugere que a amostragem está gastando
  avaliações fora do orçamento. Isso é diagnóstico; não muda o espaço durante
  uma comparação já iniciada.
- CPU/GPU ociosas durante uma etapa serial não provam travamento. Examinar os
  timers de HDBSCAN, DBCV, inicialização e SVM. Os runs anteriores mostraram que
  o SVM complementar pode dominar o tempo sem participar de J.
- Sequência remota menor que a local exige observar a tolerância de 180 s.
  Verificação online depende de resposta da API; existência de arquivos locais
  ou uma URL, isoladamente, não comprova upload.
- Depois de três falhas consecutivas da API, registrar falha de verificação sem
  inventar sincronização. No término, conferir artefatos `nas-logs`.

## Situação inicial

Registro anterior ao início do acompanhamento: a implementação estava preparada,
mas os novos runs ainda não haviam sido inspecionados. A confirmação de upload e
os resultados posteriores estão nas entradas cronológicas abaixo.

## 2026-10-26 — primeiro acompanhamento dos runs online

URLs persistidas pelo logger: [protozoan](https://wandb.ai/ophira-ai/progressive_nas_ga/runs/bb6b6236d6cc)
e [eggs](https://wandb.ai/ophira-ai/progressive_nas_ga/runs/76e29b73043e).
Nesta primeira leitura, a confirmação remota pelo daemon ainda está pendente.

Eggs, primeira profundidade, fase A, épocas 1→4: L1 de treino caiu de
0,464502 para 0,004979; L1 de validação passou de 0,007173 a 0,005480.
A cobertura permaneceu insuficiente: 57,70%→61,06%, abaixo dos 80% exigidos.
J ficou na penalidade 1e12 em todas essas épocas. DBI passou de 0,91168 a
0,85027 e Hoyer de 0,20705 a 0,20753; esses movimentos não tornam a avaliação
válida. Há aprendizado de reconstrução, sem evidência de agrupamento elegível
nesta profundidade. Não extrapolar para as próximas profundidades.

Protozoan ainda não havia completado épocas na leitura inicial. O primeiro
indivíduo foi rejeitado pelo limite de parâmetros/MACs. Os dados de um dataset
não servem de controle experimental para comparar J com o outro.

### 09:03 UTC — confirmação remota e primeiro checkpoint elegível

O daemon confirmou ambos os runs pela API do W&B em estado `running`, sem erros
consecutivos. A defasagem era de quatro eventos em ambos, com dados remotos de
aproximadamente 6–11 s: dentro da tolerância de 180 s. Ainda não havia artefato
`nas-logs`, esperado antes do término da primeira geração.

Protozoan completou cinco épocas da fase A na primeira profundidade de um
candidato. A quinta época atingiu cobertura 83,28%, J=0,36715203,
DBI=0,57600059 e Hoyer=0,21712790. É um checkpoint intermediário elegível,
não um indivíduo completo nem um vencedor da busca. A fase B começou em seguida.
Eggs ainda apresentava cobertura insuficiente no candidato acompanhado, e o
primeiro indivíduo completo foi inviável por custo. A taxa inicial de descarte
é informativa, mas a população ainda está incompleta.

### 09:06 UTC — refinamento e limite do SVM confirmado

Protozoan, profundidade 2, fase B, primeira época: J=0,30526686,
DBI=0,43535519, cobertura=83,53% e Hoyer=0,20706086. A melhora em J é
intermediária e interna ao dataset; ainda não constitui resultado final de busca.
Eggs, profundidade 2, fase A, ainda tem cobertura de 63,25% e avaliação inválida,
apesar de DBI=0,52670 entre as amostras agrupadas. O limite de cobertura impede
que a exclusão de muitas amostras gere uma seleção favorável indevida.

Os eventos `svm_end` confirmam `max_iter=10000`, protocolo linear/C=100/ovo com
standardização. O último SVM protozoan levou 6,74 s completos, dos quais 1,861 s
no fit; os recentes de eggs levaram 3,13–5,81 s completos. Todos registraram
`svm_fit_status=1` e `svm_n_iter_max=10000`: encerraram pelo limite de iterações,
sem comprovar convergência. Isso elimina o bloqueio de horas da sonda anterior,
mas seus kappas complementares precisam manter essa ressalva. J não usa SVM.

### 09:08 UTC — checkpoint selecionado por J, não pela última época

Protozoan encerrou a profundidade 2 selecionando fase B, época 2:
J=0,29526544, DBI=0,41496084, cobertura=80,393% e L1 de treino=0,012727.
A última época B5 tinha L1 ligeiramente menor (0,012532), mas J maior
(0,38179359). O checkpoint correto pelo critério arquitetural foi preservado
para o crescimento à profundidade 3. A cobertura selecionada fica próxima do
limite e merece atenção; não alterar esse limite durante a execução.
Os dois uploads seguem verificados, com defasagem inferior a 20 s nesta leitura.

### 09:10 UTC — primeira avaliação elegível em eggs

Eggs alcançou um checkpoint intermediário válido na profundidade 4, fase A,
época 5: J=0,29512415, DBI=0,41469525 e cobertura=95,47% (ruído=4,53%).
As profundidades anteriores estavam abaixo da cobertura mínima. Ainda faltam
refinamento e término do indivíduo; não contabilizar este checkpoint como
arquitetura completa vencedora. Em protozoan, profundidade 3/B2, J=0,32340147
não superou o checkpoint da profundidade 2, reforçando que maior profundidade
não implica fitness melhor. As confirmações online continuam recentes.

## Política de dados reforçada pelo usuário

Usar SOMENTE treino e validação. Por enquanto é proibido carregar imagens ou
rótulos do conjunto de teste para avaliação, embeddings, SVM, seleção de
arquiteturas/checkpoints ou diagnósticos. O encerramento da busca não autoriza
avaliação final em teste; isso exigirá autorização futura explícita. O monitor
acompanha somente artefatos, métricas e sondas produzidos com treino/validação.

### 09:12 UTC — primeiro indivíduo completo elegível

Os eventos locais de eggs registram o término da primeira geração e uma
arquitetura completa válida de profundidade 4, J=0,29512415 e cobertura=95,47%,
selecionando A5. Entradas repetidas na geração seguinte retêm o mesmo genoma;
não são novas arquiteturas descobertas. A confirmação remota dos artefatos dessa
primeira geração será acompanhada no próximo ciclo do daemon.
Protozoan selecionou profundidade 3/A1 com J=0,28593114 e cobertura=84,57%,
avançando à profundidade 4. Essas métricas são de validação, nunca de teste.

### 09:13 UTC — artefato de geração confirmado

O daemon confirmou um artefato `nas-logs` remoto em eggs, com sequência W&B
apenas um evento atrás da local. A primeira geração teve população 6, um
indivíduo válido e cinco inviáveis. O desvio de J igual a zero nessa geração
reflete somente uma avaliação válida; não demonstra convergência da população.
O monitor deduplicou seis arquiteturas únicas e preservou J=0,29512415 como
melhor fitness válido. Protozoan segue na primeira geração, com cobertura
intermediária de 78,47% em 4/A3, abaixo do limite; não antecipar seu resultado.

### 09:16 UTC — primeira geração protozoan e identidade do fitness

Protozoan completou a primeira geração com um indivíduo válido e cinco inviáveis.
O melhor indivíduo completo, profundidade 4, tem J=0,30655706 e cobertura=82,27%,
selecionando B4. O J intermediário da profundidade 3 (0,28593114) não é o fitness
final desse genoma de profundidade 4. A API confirmou o primeiro artefato
`nas-logs` protozoan; ambos os datasets agora têm upload de geração comprovado.
Eggs mantém J=0,29512415 como melhor resultado completo de sua própria busca.
Nenhuma dessas observações usa conjunto de teste.

### 09:20 UTC — segunda geração e melhora protozoan

Protozoan melhorou seu J de 0,30655706 para 0,27162284 (redução de 11,40% dentro
da mesma busca), com arquitetura completa de dois blocos: DBI=0,36898335,
cobertura=85,87%, Hoyer=0,20749134 e DBCV=0,29499227. O checkpoint escolhido
foi A1. Custo medido: 29.854 parâmetros de inferência, 307,44 milhões de MACs e
latência média de 0,638 ms, batch 1, float32, GPU A6000. Isso demonstra melhora
do critério de validação com uma arquitetura mais rasa; não comprova generalização
fora desse split e não utiliza dados de teste.

Eggs manteve o melhor J=0,29512415. A segunda geração tinha três entradas válidas
e três inválidas, mas incluía elites/cópias. Na contagem única havia dez
arquiteturas: duas válidas, sete rejeitadas por custo e uma por cobertura.
Protozoan também tinha dez únicas, duas válidas e oito rejeitadas por custo.
O daemon já confirmou duas versões de artefatos em eggs; o novo término de
geração protozoan aguarda a próxima leitura remota. A alta rejeição por custo
permanece um diagnóstico do espaço inicial, sem modificar o experimento ativo.

### 09:32 UTC — melhora por agrupamento apesar de Hoyer menor

Protozoan encontrou indivíduo completo de quatro blocos com J=0,19421768,
redução de 28,50% sobre o melhor anterior da mesma busca. DBI=0,23726853,
cobertura=98,70% (ruído=1,30%) e DBCV=0,43921110. Hoyer diminuiu de 0,20749
para 0,19582; portanto, o ganho veio da qualidade/cobertura do agrupamento,
sem depender de uma esparsidade maior dominando J. Seleção: fase B, época 2.
O custo aumentou para 39.710 parâmetros e 405,44 milhões de MACs, ainda dentro
do limite de 500 milhões; latência média 0,838 ms nas condições registradas.

Eggs completou três gerações, com três artefatos remotos confirmados e melhor
J ainda 0,29512415. Das quinze arquiteturas únicas concluídas, três são válidas,
onze foram rejeitadas por custo e uma por cobertura. A busca continua porque
a paciência configurada é três gerações sem melhora, e o estado atual registra
duas. Esses resultados continuam restritos a treino/validação do split 1.

### 09:45 UTC — término de eggs confirmado local e remotamente

Eggs terminou quatro gerações, parando após três sem melhora. Foram dezenove
arquiteturas únicas concluídas: três válidas, doze inviáveis por custo e quatro
inválidas por cobertura. Não contar cópias de elites como avaliações únicas.
O melhor foi encontrado na primeira geração e permaneceu: quatro blocos,
checkpoint A5, J=0,2951241522, DBI=0,4146952488, Hoyer=0,2093229827,
cobertura=95,4652%, dois clusters e DBCV=0,2878434517.

Custo final: 50.888 parâmetros de inferência, 257,31 milhões de MACs, latência
média de 0,55593 ms e checkpoint de 3.777.690 bytes. O kappa SVM complementar
foi 0,28803, com status de ajuste 1/limite de 10.000 iterações; não comprova
convergência da sonda. Resultado estruturado: `results/nas_eggs_50_wandb/result.json`.

O daemon confirmou W&B `finished`, sequência remota 2286 igual à local e cinco
artefatos `nas-logs` (gerações e término). A busca finalizou corretamente, mas
não há evidência de melhora genética além do candidato inicial vencedor nessa
execução. Todos os valores são de validação. Nenhuma avaliação em teste foi
executada ou autorizada. O monitor continua acompanhando protozoan.

### 09:47 UTC — novo melhor J com tradeoffs explícitos

Protozoan encontrou novo indivíduo completo de quatro blocos, selecionando B1:
J=0,10381153, redução de 46,55% frente ao melhor anterior de 0,19421768.
DBI=0,11249509, Hoyer=0,22590426, cobertura=90,34%, dois clusters e
DBCV=0,30030677. O custo foi 43.280 parâmetros, 232,82 milhões de MACs e
latência média de 0,60314 ms.

O fitness e o custo em MACs melhoraram, mas a cobertura caiu de 98,70% para
90,34% e DBCV caiu de 0,43921 para 0,30031. Ambos continuam registrados:
J não mede DBCV, e cobertura acima de 80% é uma restrição, não um objetivo
contínuo desse escalar. Portanto, não afirmar que todos os aspectos melhoraram.
A busca continua; a API confirma o run ativo e upload recente. São observações
de validação, sem acesso ao conjunto de teste.
