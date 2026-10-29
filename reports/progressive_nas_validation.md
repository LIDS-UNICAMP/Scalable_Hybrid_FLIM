# Validação do NAS progressivo — gpt_sol_G_A

Implementação realizada com coordenador e três agentes especializados, com
reconhecimento somente leitura, contratos antes das edições, propriedade
exclusiva dos arquivos e revisão independente. O quarto papel de implementação
foi dividido entre coordenador e agente liberado após o modelo, respeitando o
limite de quatro agentes simultâneos. Não houve edição concorrente de arquivos.

## Resultado executado

- **29 testes passaram**, em 27,777 s, na suíte conjunta final.
- Check legado de crescimento: **ALL CHECKS PASSED**.
- Check de equivalência de métricas e SVM: **ALL CHECKS PASSED**.
- Launcher Ray: preflight e dry-run passaram, produzindo um comando real para
  o braço NAS. O agendamento distribuído completo não foi executado.
- Treinamento sintético: duas profundidades, fases A/B em ambas, SPiFiL real.
- Busca sintética: população dois, duas gerações completas, dois genomas únicos
  avaliados. Operadores de crossover/mutação e elitismo também foram exercitados
  pelos testes; a geração seguinte pode repetir genomas e aproveitar cache.
- Interrupção após época salva, retomada, save/load, exportação de embeddings,
  cache efetivo e checkpoint ausente foram verificados com SPiFiL real.
- Smoke de dez blocos verificou projeções/embeddings/reconstrução; esse smoke usa
  convoluções locais para testar formatos, não treinamento SPiFiL de dez níveis.

O log da suíte final está em [tests.txt](progressive_nas_validation_logs/tests.txt);
os demais logs estão em [progressive_nas_validation_logs](progressive_nas_validation_logs/).
O registro estruturado com métricas, custos, filtros e caminhos está em
[progressive_nas_validation.json](progressive_nas_validation.json).

## Configuração medida

`ga_search/config/progressive_nas_synthetic.yaml`, seed 23, CPU, batch de treino
dois, quatro imagens de treino e dezesseis de validação, imagens/máscaras
sintéticas 200×200, duas classes. Superpixels SLIC na imagem, 100 por imagem,
prefixo em eval e extração dos filtros nas ativações. Foram usados os quatro
doadores de treino, registrados pelos IDs `0_train_0000`, `1_train_0001`,
`0_train_0002`, `1_train_0003`. Nenhuma amostra de validação/teste foi doadora.
Essa fixture usa todos os quatro doadores; a seleção híbrida existente é usada
no dataset real.

Treino: uma época A e uma B por profundidade, LR 0,001/0,0001, L1, auxiliares He,
embedding 1280, HDBSCAN `min_cluster_size=3`, `min_samples=2` incluindo a própria
amostra, cobertura mínima 0,8, lambda 0,004. Limites: dois milhões de parâmetros
de inferência e 500 milhões de MACs. Nenhum peso externo foi carregado.

Ambiente de validação: Python 3.11, PyTorch 2.2.2, NumPy 1.26.4, SPiFiL 1.0.0
editable, sklearn 1.8.0 e hdbscan 0.8.44. `pytest` não estava instalado;
os testes usam unittest e os checks executáveis existentes. hdbscan foi
instalado em `/tmp/flim_nas_deps`, sem alterar a env principal. A suíte também
foi exercitada sem o pacote: HDBSCAN sklearn funciona, DBCV fica indisponível e
o teste específico é ignorado explicitamente.

O sandbox não expôs CUDA. Uma demonstração adicional, executada com acesso
ao diretório externo `artifacts`, detectou uma RTX A6000 e terminou na GPU;
seus resultados ficam em `artifacts/progressive_nas_validation/train`.
As medições abaixo fixaram CPU para comparação e reprodutibilidade; CUDA pode
ter operações não determinísticas. Os candidatos não treinaram em paralelo
na mesma GPU.

## Métricas sintéticas

| Medida | Genoma configurado, duas profundidades | Melhor J na pequena busca |
| --- | ---: | ---: |
| J, lambda 0,004 | 0,1787823180 | 0,0042698794 |
| DBI sem ruído | 0,2140404909 | 0,0010851496 |
| Hoyer média | 0,2041772416 | 0,2024394966 |
| Cobertura | 0,9375 | 1,0 |
| Clusters descobertos | 3 | 2 |
| DBCV | 0,8539127876 | 0,9995818346 |
| L1 da validação no checkpoint escolhido | 2,1333407462 | 0,7079077438 |
| Parâmetros de inferência | 6.676 | 9.486 |
| MACs por imagem | 18.560.000 | 33.360.000 |
| Latência média CPU, ms | 4,3411 | 5,1417 |

São dados sintéticos separáveis por construção e somente duas épocas por
profundidade. Esses valores comprovam execução e registro, sem demonstrar
superioridade ou generalização em imagens reais. A reconstrução ainda é pouco
refinada; o checkpoint deste experimento é escolhido por J, não por menor L1.

Genoma configurado: Conv3×3, quatro canais, BN, ReLU, pool2 em ambos os blocos.
Melhor genoma da busca: Conv5×5/quatro canais/ReLU6/sem BN/pool2, seguido de
Conv3×3/seis canais/ReLU/sem BN/pool2. As duas projeções têm 1280 canais.

MACs foram contados por hooks Conv2d/Linear, um multiply-accumulate por termo
do kernel. Não são FLOPs; bias, BN, ativação, pooling e interpolação não entram
na conta. Inferência contém somente encoder/projeção corrente/GAP. A latência
usa float32, batch um, uma thread CPU, um warmup e três repetições. É uma
medição de smoke; as amostras e o hardware reportado pelo ambiente (`x86_64`)
estão no JSON. Os checkpoints completos têm aproximadamente 393/453 kB,
incluindo os auxiliares e o otimizador (393.102/453.510 bytes no JSON).

SPiFiL ajustou 400 patches por profundidade. No genoma configurado, D foi
27 e 36; os tempos medidos foram aproximadamente 0,468 e 0,417 s. No melhor
genoma da busca, D foi 75 e 36. Tempo e pico de memória SPiFiL são separados
da inferência. `process_peak_rss_kib` é pico do processo, incluindo seu baseline,
e não uma medida isolada do incremento de RAM da fase.

Os mesmos embeddings foram reavaliados com lambda 0,04; os valores exatos de J
estão em `lambda_recalculation` no JSON. Isso é uma recomputação de fitness,
não um segundo treinamento ou uma nova busca independente.

## Erros encontrados e corrigidos

1. Primeira execução integrada encontrou `phase` duplicado na composição do
   registro de treino; corrigido e executado novamente até completar o pipeline.
2. Revisão independente encontrou skip indevido com checkpoint de cache ausente,
   ausência de validação de hash e histórico/proveniência de outro pedido.
   Corrigidos; testes executam cache para pedidos 4/5 canais que produzem quatro
   canais reais e remoção do checkpoint de origem.
3. Política `trunc_normal` central não cobria convoluções auxiliares. O modo NAS
   agora aceita somente He/Xavier, sem declarar uma inicialização não aplicada.
4. Equivalência de `min_samples` entre os backends HDBSCAN foi corrigida,
   inclusive quando o parâmetro é default; ambos foram testados.
5. Check legado usava imports/caminhos anteriores à refatoração e checkpoint
   artificial com shapes incorretos. O próprio check foi adaptado às APIs
   atuais, mantendo o código de produção legado intacto; passou integralmente.
6. A sonda complementar foi ligada ao `fit_svm` e às métricas centrais, com
   remapeamento explícito dos IDs das classes e diagnósticos do solver.
7. A validação de identidade na retomada foi antecipada para evitar gravações
   antes de rejeitar uma configuração incompatível. Fixtures já existentes não
   são sobrescritas; arquivos alterados invalidam o experimento. Um teste
   verifica que imagens e manifesto permanecem preservados nesses casos.

## Limitações e próximos experimentos

- Dados reais estão disponíveis: o loader foi verificado em protozoan-cysts,
  split 1, 5%, com 235 imagens de treino, 4.547 de validação e sete doadores
  selecionados pelas regras existentes. **Não foi executado treinamento/busca
  completo com esse dataset real.**
- Não foi executado treinamento real de dez profundidades nem uma busca grande.
- Operações sem contrato SPiFiL validado e outros números de canais de entrada
  são rejeitados. Não há fallback aleatório nem pesos externos.
- NSGA-II e herança de pesos entre indivíduos não estão habilitados. Os
  operadores e o avaliador estão separados para permitir extensão posterior.
- DBCV pode apresentar overflow em 1280 dimensões; valores não finitos são
  registrados como inválidos. Nesta execução os valores selecionados foram
  finitos, embora a biblioteca tenha emitido warnings de overflow intermediário.
- A restauração exata foi testada em CPU. O ambiente GPU adicional não recebeu
  certificação de determinismo bit a bit.

Comandos de treino, busca, retomada e inferência estão no
[guia executável](../ga_search/PROGRESSIVE_NAS.md).

## Contribuições

- Coordenador: branch, contratos, genética, integração de CLI/schema/runner,
  logging, cache, seleção de checkpoint, documentação e validação final.
- A1/B1: adapter SPiFiL, testes reais de ajuste, loader integrado e correção do
  check legado.
- A2/B2/B4: modelo/treinador/otimizador/checkpoints, testes de modelo/genética,
  custos e revisão independente D2, inclusive confirmação das correções.
- A3/B3/D1: métricas/HDBSCAN/DBCV, extração, testes de métricas e pipeline
  integrado com interrupção/retomada/cache.
