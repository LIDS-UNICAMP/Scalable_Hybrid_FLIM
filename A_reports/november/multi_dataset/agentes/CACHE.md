# Cache em disco das imagens redimensionadas

Resposta curta: pronto e ligavel por YAML; o dataloader fica 22x mais rapido em Larvae e 7x em Eggs a 640 px, com tensor igual bit a bit.

## Ganho medido (640 px, no_crop, 8 workers, CPU, nice 10, maquina com load 80-110)
| dataset | sem cache | com cache | ganho |
|---|---|---|---|
| Larvae (400 img, 3 passadas) | 5,5 frio; 12,2 quente img/s | 268-298 img/s | 22-24x quente |
| Eggs (1600 img, 3 passadas) | 64-67 img/s | 449-485 img/s | 7x |
Job de eval real, Larvae 128 px, 800 imagens: 165 s sem cache, 9 s com cache. SVM stubado nessa medida.
Nao medi o tempo de um braco inteiro: o ganho do braco depende de quanto da GPU era espera.

## Como funciona
- O .npy guarda o uint8 [3,N,N] que build_test produz antes do ToDtype float: mesma decodificacao ift_lab, mesmo resize/letterbox.
- Sem mudar transform: no tamanho final, Resize, pad e CenterCrop nao fazem nada. Geo/strong aug, TrainAug e Normalize seguem depois.
- Formato .npy e nao PNG: disco sobra e nao ha descompressao.
- Arquivo ausente ou pasta ausente: leitura normal (aviso no log). YAML com helminth_eggs + Larvae funciona, so Larvae acelera.
- ParasiteDataModule recusa a chave se nao for multi_view, V_train = V_eval = 1, sem multicrop, ift_lab, com parasite_name: RandomResizedCrop parte do original, e ali o cache mudaria o resultado.

## Ligar
runner_args do YAML growth e do YAML eval:
  image_cache_dir: /exps/moliveira/image_cache
O eval tem que repetir tambem image_size e no_crop. Gerar mais: python tools/build_image_cache.py /exps/moliveira/image_cache 4 1 Eggs:96,128
(raiz, processos, no_crop 0/1, dataset:tamanhos; ja se poe em nice 10 e pula o que existe.)

## Arquivos
tools/build_image_cache.py (novo), core/data/parasite_dataset.py, core/data/parasite_data_module.py, eval/svm.py, experiments/ray/runners/eval.py, experiments/ray/runners/growth.py, experiments/ray/schema.py. Sem commit.

## Fora do cache
- grow/SPiFiL: flim.spifil.Cropped le RGB pelo SpifilDataset (preenche com 0) e encoder_features le ift_lab direto. Nao cobri: sao poucas imagens (2 em Larvae com one_per_class) e o Cropped e outro conteudo, RGB e nao LAB.
- eval/growth_stages.py CLI antigo e metodos com V_train > 1.
- aid, flowers, helminth_eggs: sem cache gerado (nao pedido). aid so entrou na prova.

## Prova
- 32 combinacoes (Eggs, Larvae, Protozoan, aid x 96, 200, 448, 640 x no_crop sim/nao), 1328 comparacoes, 0 diferentes: torch.equal e bytes iguais, dtype, shape e rotulo. aid inclui PNGs chamados .jpg e o jpg corrompido 000022_00000228.
- Treino real pelo comando do growth com a chave, Larvae 128, 1 epoca, GPU 2 com gpucap 8000: rc=0, 25 s, config.yaml traz a chave.
- Eval pelo _run_svm_job, mesmo ckpt: hash das features igual com e sem cache (treino 5b0f48c0b29b7339, val e974315a7a3f184e). So o fit do SVM foi stubado, porque svm.lock estava com a fila.
- O primeiro launch completo do e2e falhou no treino por falta de .git na copia (pyrootutils); artefato da copia. Rodei o comando do estagio direto. Nao ha launch completo com a chave do inicio ao fim.

## Sem quebrar as filas
- Aplicado com arquivo temporario + py_compile + mv, dataset antes do module. Worktree conferido igual a copia base antes de aplicar.
- 37 YAMLs (34 das filas + eval s0640 + exemplo + growth_disc_bp01w64c1280): dry-run com preflight identico antes, na copia editada e no worktree, 0 diferencas. 36 rc=0; 1 rc=1 igual nos dois (eval s0640: 0 ckpts ainda).
- Filas: 0 linhas rc != 0 nos q*.status depois de aplicar.

## Disco
50,8 GB em /exps/moliveira/image_cache, gerado em 15,6 min com 4 processos em nice 10. 236.836 arquivos.
Eggs 96,128,200,320,448,640 = 28 GB | Larvae 128,200,320,448,640,896,1280 = 15 GB | Protozoan 64,96,128,200 = 7,8 GB. So modo nocrop. Contagens completas: 12691, 1598, 37376 por pasta; 0 .tmp.

## Atencao
Os YAMLs em fila nao tem a chave e nao foram editados: res_lote1/res_lote2/res_hold rodam sem cache ate alguem por a chave num YAML ainda nao iniciado.
