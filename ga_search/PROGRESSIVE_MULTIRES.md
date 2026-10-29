# Busca progressiva em múltiplas resoluções

A grade reutiliza o launcher Ray e o runner `growth` existentes. São **144 buscas**: Eggs e Larvae × porcentagens 1, 5, 25, 50, 75 e 100 × splits 1, 2 e 3 × resoluções 128, 160, 200 e 256. Cada busca mantém população 6, até 5 gerações, paciência 3 e no máximo 10 blocos.

A resolução varia entre configurações; ela não é um gene mutado pelo GA. Cada execução usa uma resolução fixa, reconstrói essa resolução e produz embeddings de 1280 dimensões. O batch passou de 2 para 16. Isso altera a otimização e a Batch Normalization, portanto as novas buscas usam diretórios próprios.

| Parâmetro arquitetural | Eggs | Larvae |
|---|---|---|
| Canais solicitados por bloco | 9, 18, 27, 36 | 4, 8, 12, 16 |
| Dropout | 0; 0,05; 0,1; 0,2; 0,3 | igual |
| Kernel | 3×3 ou 5×5 | igual |
| Pooling stride | 1 ou 2 | igual |
| Ativação | ReLU, ReLU6 ou SiLU | igual |
| Batch Normalization | ligada ou desligada | igual |

O espaço continua limitado às convoluções convencionais sequenciais com suporte SPiFiL validado. Os canais efetivos retornados pelo inicializador são registrados e podem diferir dos solicitados. Projeções e decoder usam a política auxiliar convencional. O limite de inferência permanece em 500 milhões de MACs e 2 milhões de parâmetros; aumentar a resolução pode aumentar a proporção de candidatos rejeitados.

## Dados: somente treino e validação

O protocolo `train_pool_stratified_fixed_validation_v1` lê somente a lista `train` do descritor nativo de 100% de cada split. Separa, por classe, 20% dessa lista para validação fixa com semente `42 + split`. As porcentagens indicam subconjuntos aninhados dos 80% restantes, usando arredondamento para baixo e pelo menos uma imagem por classe. A validação fica fixa entre porcentagens e resoluções do mesmo split. Os descritores originais não são alterados.

Assim, **100% significa todo o novo pool de treinamento**, depois da reserva de validação. Este protocolo difere das buscas anteriores de 50%; seus números não constituem uma comparação controlada com aqueles resultados. Nenhuma amostra do conjunto de teste é usada para filtros, treinamento, embeddings, SVM, métricas ou seleção. Nenhuma avaliação de teste será executada automaticamente no encerramento.

Eggs mantém um doador SPiFiL por classe. Larvae usa `one_per_class: false` e `n_images: 100`, limitados às imagens de treino disponíveis, com seleção uniforme determinística existente. Isso amplia o orçamento de patches em relação aos dois doadores do modo por classe; falhas de orçamento continuam sendo registradas como inviabilidade, sem pesos aleatórios de fallback. IDs efetivos e identidades dos dados ficam registrados por execução.

## Preparar, executar e retomar

Na raiz do repositório e no ambiente Python com as dependências instaladas:

```bash
python -m ga_search.prepare_multires_grid
python -m experiments.ray.launch ga_search/config/progressive_multires/ray_eggs_r128.yaml
python -m experiments.ray.launch ga_search/config/progressive_multires/ray_larvae_r128.yaml
```

O preparador gera oito pares de YAMLs e 36 descritores treino/validação em `ga_search/config/progressive_multires/`, acompanhados de `manifest.json` com os hashes. Ele recusa sobrescrever arquivos preparados cujo conteúdo seja diferente. Cada YAML Ray expande 18 combinações de split e porcentagem. Substitua `r128` por `r160`, `r200` ou `r256` para as outras resoluções.

Os scripts operacionais `results/nas_multires/launch_eggs.sh` e `results/nas_multires/launch_larvae.sh` percorrem as quatro resoluções sequencialmente. Para retomar, execute novamente o mesmo script: o launcher usa `skip: state`, e o subprocesso progressivo recebe `--resume`. A retomada exige identidade compatível de configuração, código, dados e checkpoints. Não altere configurações de buscas já iniciadas; crie outra saída para mudar o protocolo.

As saídas ficam em `results/nas_multires/helminth-eggs/r<resolução>/` e `results/nas_multires/helminth-larvae/r<resolução>/`, separadas por split e porcentagem. Métricas, eventos do GA, estados de busca e checkpoints seguem o formato progressivo existente. O W&B é habilitado nos YAMLs progressivos; `wandb.enabled: false` no YAML Ray evita uma execução adicional do launcher e não desliga a telemetria de cada busca.

## Recursos

Há dois processos de busca simultâneos: Eggs na GPU física 2 e Larvae na 3, com um processo por GPU. As GPUs 0 e 1 têm outros usuários/processos e não são incluídas. Cada busca recebe 16 threads de CPU, 16 workers para preload e 16 para DBCV. O runner Ray chama `ga_search.progressive_runtime`, que aplica esses limites após os imports do CLI e registra consumo e tempos por etapa.

Batch16 aumenta o trabalho por chamada CUDA. O uso da GPU ainda oscila durante HDBSCAN e outras etapas de CPU. Criar 96 threads por processo não garante aceleração e causaria competição entre as buscas; o consumo real deve ser acompanhado nos arquivos `runtime_resources.jsonl`, nos tempos por etapa e no W&B.

## Resultados finais das buscas anteriores, 50%, split 1

| Métrica de validação | Protozoan | Eggs |
|---|---:|---:|
| Gerações concluídas | 5 | 4 |
| Fitness J, menor é melhor | 0,103812 | 0,295124 |
| DBI | 0,112495 | 0,414695 |
| Cobertura | 90,34% | 95,47% |
| Hoyer | 0,225904 | 0,209323 |
| DBCV | 0,300307 | 0,287843 |
| Blocos | 4 | 4 |
| Canais efetivos | 28, 14, 28, 14 | 27, 9, 9, 27 |
| Parâmetros de inferência | 43.280 | 50.888 |
| MACs por imagem | 232,82 milhões | 257,31 milhões |
| SVM: acurácia | 18,66% | 74,51% |
| SVM: kappa | 0,0980 | 0,2880 |

Os dois SVMs atingiram o teto de 10.000 iterações (`fit_status=1`); são avaliações complementares sem convergência garantida e não selecionam arquiteturas. O J baixo mede a estrutura do agrupamento, sem comprovar correspondência dos grupos às classes semânticas. Não houve avaliação de teste.
