# GA: métricas, configuração, curvas e comparação — 28/10/2026

## Comparação com o relatório de outubro

Fontes: `A_reports/october/report_results_only_spifil_kappa.md` e seu relatório complementar `report_results_only_spifil.md`. O complementar declara explicitamente: teste, SVM oficial no encoder congelado, média e desvio dos três splits.

Os resultados atuais do GA são de validação, split 1, resolução 128. A validação fixa corresponde a 20% do pool nativo de treino; as porcentagens são aplicadas aos 80% restantes. Não foi executado teste do GA nesta análise.

| Dataset / porcentagem | SPiFiL: teste, média 3 splits | SPiFiL + backprop: teste, média 3 splits | GA: validação, split 1 |
|---|---:|---:|---:|
| Eggs 1% | 0,1278 | 0,2748 | 0,2238 |
| Eggs 5% | 0,5852 | 0,6749 | 0,1275 |
| Eggs 25% | 0,8050 | 0,8698 | 0,1775 |
| Larvae 1% | 0,0284 | 0,3485 | 0,3265 |
| Larvae 5% | 0,5712 | 0,5971 | 0,3625 |
| Larvae 25% | 0,8049 | 0,8318 | 0,1996 |
| Protozoan 1% | 0,1447 | 0,4734 | 0,2681 |

Todos os sete valores atuais são numericamente menores que SPiFiL + backprop; isso não prova piora no teste. Em 1%, os três superam numericamente o SPiFiL puro. Seis das sete SVMs atuais atingiram 10.000 iterações; somente Larvae 1% convergiu.

Outras diferenças: relatório antigo em 200 pixels e representações flatten de até 27.648 dimensões; GA atual em 128 pixels e projeção seguida de GAP, com 1280 dimensões. A tabela antiga também apresenta variantes avgpool, cujos resultados devem ser distinguidos de flatten. O GA seleciona por J, não por kappa. Não há comparação pareada suficiente para atribuir o resultado à inicialização SPiFiL isoladamente.

## Métricas

| Métrica | Interpretação |
|---|---|
| J | Fitness minimizado: 0,996 × DBI/(1+DBI) + 0,004 × (1−Hoyer). Não é acurácia. |
| DBI | Dispersão dentro dos clusters em relação à separação entre eles. Menor é melhor. Calculado sem o ruído −1. |
| DBI normalizado | DBI/(1+DBI), entre 0 e 1 para DBI finito não negativo. |
| Hoyer | Esparsidade do embedding: 0 é denso e 1 é esparso, nas condições matemáticas aplicáveis. Registra média e desvio. Não mede esparsidade dos pesos. Vetores nulos/não finitos são inválidos. |
| Cobertura | Fração das amostras que pertencem a clusters. Mínimo 80%. É uma restrição; acima do limite, não há recompensa direta por aumentá-la. |
| DBCV | Validade dos clusters baseada em densidade, normalmente entre −1 e 1. Maior é melhor. Complementar; pode ficar inválida sem substituir J. |
| Número de clusters | Descoberto pelo HDBSCAN; não recebe o número de classes como meta prévia. |
| L1 | Erro absoluto médio da reconstrução. É a perda usada nos gradientes. Menor L1 não garante menor J nem maior kappa. |
| Kappa SVM | Concordância com os rótulos descontando o acaso: 1 perfeito, 0 equivalente ao acaso, negativo pior que o acaso. |
| Acurácia SVM | Neste repositório é balanceada, média por classe; não é a fração bruta de acertos. |
| F1 macro | Média do F1 por classe. Maior é melhor. |
| Parâmetros / MACs | Tamanho e custo de inferência do encoder e projeção, excluindo o decoder. MACs contabilizam Conv2d/Linear, não todas as operações. |
| Latência | Medição em FP32, batch 1, cinco aquecimentos e vinte repetições. A carga compartilhada da GPU influencia o valor. |

## Critério e parâmetros ativos

Uma melhoria genética exige J estritamente menor; a tolerância atual é zero. A validação permanece fixa dentro de cada busca. Cobertura abaixo de 80%, menos de dois clusters ou valores inválidos recebem penalidade 1e12. Os limites são dois milhões de parâmetros e 500 milhões de MACs.

| Grupo | Configuração |
|---|---|
| GA | População 6; até 5 gerações; elitismo 1; torneio com 3 candidatos; dois pais por cruzamento; probabilidade de mutação 0,8; paciência 3 gerações. |
| Operadores | Cruzamento por cortes entre blocos; mutação altera um parâmetro, insere ou remove bloco. Sem herança de pesos entre indivíduos. |
| Fitness do indivíduo | Profundidade completa de seu genoma, usando o checkpoint de menor J nessa profundidade. |
| Treino | Até 1000 épocas em cada fase A/B; parada após 50 épocas sem melhorar J válido na fase; min_delta=0. Épocas inválidas contam como ausência de melhoria. |
| Otimização | AdamW; LR 0,001 na fase A e 0,0001 na B; batch 16; seed 42; perda L1; SSIM desligado. |
| Arquitetura | 1–10 blocos; convoluções convencionais 3×3/5×5; conexões sequenciais; pooling stride 1/2; BN opcional; ReLU/ReLU6/SiLU. |
| Dropout | 0; 0,05; 0,1; 0,2; 0,3. |
| Canais | Eggs: 9/18/27/36; Larvae: 4/8/12/16; Protozoan: 7/14/21/28. Canais efetivamente obtidos pelo SPiFiL são registrados. |
| Embedding | Projeção independente 1×1 → 1280 em cada profundidade; GAP gera o vetor. O decoder recebe o mapa espacial. |
| Grade externa | Resoluções 128/160/200/256; splits 1/2/3; porcentagens 1/5/25/50/75/100. Resolução não é gene do GA. |
| SPiFiL | Uma imagem de treino por classe; SLIC 450; modo baseado em imagem; sementes projetadas para as ativações; orçamento N>D. |
| HDBSCAN | min_cluster_size=15; min_samples=5 incluindo o próprio ponto; distância euclidiana; seleção eom; embeddings sem StandardScaler para agrupamento. |
| SVM | Linear; C=100; StandardScaler ajustado no treino; max_iter=10000. Avaliação complementar na validação, sem participar da seleção. |

## Curvas A → B

A consulta online ao W&B confirmou 143 pontos de Eggs 5%, 321 de Larvae 25% e 182 de Protozoan 1%, sem divergências dos valores locais. As fases foram identificadas pelos eventos locais e associadas às sequências do W&B. Evidência: `verified_curves.json`.

Backpropagation já está ativo na fase A. A transição para B descongela o prefixo; ela não representa o início de todo o backpropagation. Não há nestas curvas uma avaliação do embedding SPiFiL puro antes do primeiro gradiente. Projeção e decoder também usam inicialização auxiliar convencional.

| Modelo vencedor | Melhor J na fase A | Melhor J na fase B | Checkpoint escolhido |
|---|---:|---:|---|
| Eggs 5%, profundidade 7 | 0,221193 | 0,228918 | A, época 30 |
| Larvae 25%, profundidade 4 | 0,319300 | 0,115881 | B, época 220 |
| Protozoan 1%, profundidade 4 | 0,206279 | 0,205244 | B, época 28 |

No Protozoan, imediatamente após A → B, L1 passou de 0,024560 para 0,036501 (+48,6%), e J de 0,208267 para 0,231306 (+11,1%); depois houve recuperação. Em Larvae, B oscilou, mas produziu menor J. Em Eggs, a cobertura frequentemente ficou abaixo de 80%, e o melhor checkpoint permaneceu em A.

Picos de J=1e12 representam penalidades de inviabilidade; não são uma medição contínua de explosão de gradientes. O gráfico de J válido exclui essas penalidades e deve ser lido junto com cobertura.

Oscilações após descongelamento são compatíveis com adaptação dos pesos e estatísticas de BN, mas a causa não foi isolada. Não existe aqui um controle com mesma arquitetura, dados e sementes comparando SPiFiL e inicialização aleatória. Portanto, estas curvas não provam que SPiFiL seja ruim nem que seja superior. Os gráficos estão em `phase_A_B_curves.png`.
