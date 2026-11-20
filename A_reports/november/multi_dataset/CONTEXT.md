# Contexto comum dos agentes: leis para um encoder SPiFiL leve em 12 datasets

Objetivo: continuar a descoberta "hybrid SPiFiL" em 12 datasets, para achar uma receita (SSL + downstream)
e uma lei que montem uma CNN leve (<= 5M parametros), inicializada pelo SPiFiL, com kappa >= melhor modelo
do benchmark (I-JEPA, ConvNeXt V2 Pico, DINOv3, DINOv2, AIMv2). Primeiro a 50% dos dados, 3 splits; depois {1,5,25,75,100}%.
Problema fundamental: relacao entre tamanho da imagem, numero de superpixels e numero de superpixels selecionados;
ponto representativo (medoid, geodesic, centroid) e combinacao dos tres; eliminar kernels redundantes ANTES de inicializar o modelo.

## Locais
- Maquina: so Athena. NADA em Jaci (nao mexer, nao lancar, nao sincronizar).
- Worktree de trabalho: /dados/home/moliveira/Scalable_Hybrid_FLIM_discovery, branch discovery, hash 0de1838. Nunca a main.
- Dados: /dados/home/moliveira/datasets/organized/
- Pesos novos: /exps/moliveira/ . Pesos antigos: /dados/home/moliveira/artifacts/spifil_growth/disc_* (so leitura).
- Env: /dados/home/moliveira/miniforge3/envs/scalable_FLIM (python absoluto). Sempre exportar
  LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib:$LD_LIBRARY_PATH RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
- SPiFiL: pacote editable `spifil` em /dados/home/moliveira/SPiFiL. NAO PODE SER MODIFICADO.
- Relatorios dos agentes: <scratchpad>/p0/<ID>.md, com scratchpad =
  /tmp/claude-1033/-dados-home-moliveira-Scalable-Hybrid-FLIM/990a33ea-4327-4228-ac99-2136fb213621/scratchpad

## Regras
- Busca no codigo: graphify primeiro, dentro do worktree (`graphify query "..."`, `graphify explain "<no>"`, `graphify affected`, `graphify path`).
  O grafo do worktree foi atualizado agora. So use grep depois que o grafo nao responder, e diga em uma linha o que ele nao respondeu.
- Toda afirmacao vem com arquivo:linha ou com a saida do comando. Nao invente: se nao achou, diga que nao achou.
- Modelo: so CNN, sem transformer, SPiFiL inicializa toda camada convolucional nova, backprop sempre junto, sem fine-tuning com rotulo, <= 5M parametros.
- Avaliacao: escolha so pela validacao; teste so para reportar. Sem flatten, leitura < 2000-d. SVM so depois do treino, sob flock, com nice -n 10.
- GPU: no maximo 40 GB por GPU, somando todos os processos do moliveira. Processo de outra pessoa nunca e parado.
- Experimentos: so YAML em experiments/autoencoder/, lancado com `python -m experiments.ray.launch`. Prefixo novo disc_md_*.
- Nunca apagar ou sobrescrever CSV ou checkpoint registrado. Nunca criar arquivo .sh. Nunca commitar.
- Armadilhas: `skip: state` nao pula pontos prontos do growth; flock nao segue ordem de chegada; pgrep devolve o PID do servidor tmux; `_pw` sem shrinkage colapsa.

## Os 12 datasets (dizeres do usuario, a conferir)
Eggs 12.691 img, 9 (ou 8, 16) classes, pre/ 200x200, raw/ variavel | Protozoan 37.376 (pre/: 37.372), 7 (ou 6, 12), pre/ 200x200, raw/ variavel |
Larvae 1.598, 2, pre/ 200x200, raw/ variavel | aid 10.000, 30, 600x600 | Protozoan_subgroup 7.919, 7, 200x200 | Eggs_subgroup 3.974, 9, 200x200 |
flowers 2.746, 5, variavel | protozoan_cysts 9.568, 7, 200x200 | remote_sensing 1.000, 14, 640x640 | helminth_eggs 5.112, 9, 200x200 |
helminth_larvae 3.514, 2, 200x200 | Larvae_subgroup 741, 2, 200x200

## Fase P0
So leitura. Nao edite nada no repositorio, no SPiFiL ou nos dados. O unico arquivo que voce escreve e o seu relatorio em <scratchpad>/p0/<ID>.md.
Na resposta final devolva um resumo curto (ate 25 linhas) e o caminho do relatorio.

## Pedidos do usuario durante a sessao
- Lancar sem perguntar: a tabela de D1 e mostrada, mas nao se espera liberacao. Continua parando nos casos que o prompt manda perguntar.
- O usuario so quer ver a fila: dar um comando de uma linha que mostre rodando e esperando.
- M1 tambem monitora metricas por epoca de TREINO E VALIDACAO (loss e kappa nos dois), por dataset e por familia, para descobrir a melhor epoca de parada. Teste nunca entra nisso.
- Decisao do usuario: usar raw/ em Eggs, Protozoan e Larvae (substitui a leitura padrao "pre/ na receita"). Os splits e manifests dessas bases valem para raw/orig. pre/ nao entra.
- Pendente com o usuario: DINOv3 oficial (gated) ou copia timm/. Ate responder, nao baixar DINOv3.
- Decisao do usuario: NAO usar mascara em nada: nem raw/label, nem em SPiFiL/superpixels, nem na loss, nem em augmentation. Cai "aderencia a borda" de X1 e "troca de fundo guiada por mascara". Tamanho do objeto, se preciso, nao vem de mascara no pipeline do modelo.

## Fase P1 (substitui a regra "so leitura" da fase P0)
Escreva SO no worktree /dados/home/moliveira/Scalable_Hybrid_FLIM_discovery, e SO nos arquivos do seu destino. Nunca a main, nunca o SPiFiL, nunca os dados. Sem GPU. Sem commit.
Varios agentes editam o worktree ao mesmo tempo: nao rode git stash/checkout/reset, nao formate arquivo alheio.
Relatorios de P0, com arquivo:linha: <scratchpad>/p0/R1.md ... R9.md. Leia os que seu brief cita. Seu relatorio: <scratchpad>/p1/<ID>.md.
Estilo ponytail: reuse o que existe, escreva o minimo, nada de abstracao ou config alem do pedido. Sem comentario desnecessario.
Todo .py novo comeca com o banner de 17 linhas, copiado byte-exato: escreva sem banner e prefixe com `head -17 config.py`. Nunca digite o banner.
Tudo novo fica desligado por padrao: com a chave desligada, sai o mesmo resultado de hoje.
So o dono de arquivo comum edita: A0 edita o que os 12 datasets exigem; L e Y criam arquivos novos e NAO registram nada no loop, isso e de J1.
Checks: rode com o python absoluto do env e LD_LIBRARY_PATH, em CPU. Deixe o check junto do modulo (bloco `if __name__ == "__main__"`), sem criar pasta de testes.
Resposta final: ate 15 linhas, com caminho dos arquivos, saida do check e o que J1 precisa fazer para ligar.

## Fase S1 + J1
- Relatorios de P1: <scratchpad>/p1/*.md . Relatorios: <scratchpad>/s1/<ID>.md .
- Disco: /dados tem 57 GB e / tem 7,3 GB. Tudo grande vai para /exps. Nunca gravar pesos ou caches em /dados ou /tmp.
- Ambiente: impuro_grid e rand_grid ainda rodam (moliveira) e crispim usa GPU: nao tocar.
- Decisao do usuario: todos os modelos do benchmark vem do Hugging Face, pesos OFICIAIS (repos dos autores: facebook/, apple/). Espelho timm/ nao. DINOv3 = facebook/, gated: precisa do login do usuario.

## Objetivo refinado pelo usuario (06/11)
- Aceite: kappa igual ou maior que o melhor do benchmark; chegar muito perto ja satisfaz (leitura: ate 0,01 abaixo; reportar a diferenca real).
- O produto e uma LEI DE MONTAGEM, no estilo ResNet / EfficientNet / LeJEPA-SIGReg: regra de como escalar profundidade, largura e resolucao e qual objetivo usar. NAO e uma config unica igual para todos os datasets.
- Explorar ideias classicas para extrair insights e montar modelos cada vez maiores, medindo a curva.
- Eixo proprio: relacao SPiFiL x tamanho de entrada, procurando tamanhos otimos para o SPiFiL (grade de resolucao, nao um tamanho fixo por dataset).
- Usuario (06/11): a regra do SPiFiL existe porque ele define bem o inicializador; tudo gira em torno do SPiFiL hibrido (inicializacao SPiFiL + backprop). A lei de montagem e uma lei do SPiFiL hibrido: toda regra de escala deve dizer como o SPiFiL inicializa o modelo maior.
- Usuario (06/11): gestor liberou usar toda a GPU, sem quebrar o job de outro usuario. Novo limite: placa inteira <= 46.000 MiB de 49.140 (crispim ~4,9 GB, ja visto com ~18 GB), moliveira ~41 GB; processos do crispim preservados.
- Usuario (06/11): entregas finais: (1) lei de tamanho do modelo com SPiFiL x tamanho de imagem; (2) escolha dos kernels SPiFiL: quais ajudam e quais atrapalham.
- Bug da sonda corrigido em 06/11 ~05:25 (float16 no treino, float32 na validacao): epoch_metrics e best_kappa de helminth_larvae, helminth_eggs, protozoan_cysts anteriores a isso sao invalidos. SVM oficial nao e afetado. Decidir sempre por SVM de validacao.
