# Crescimento camada por camada com AE, SimCLR e VICReg: plano D1 (13/11/2026)

## Decisões do Mateus (13/11 ~01:40)
- K0: tudo derrubado nos dois hosts. Registros: athena `/exps/moliveira/queues/logs/k0_derrubados_2026-11-13_0121.md`; jaci `/dados/home/moliveira/exps/queues/logs/k0_derrubados_jaci_20261112.md` (trava `queues/ssl_segura`).
- Athena: GPUs de índice 2 e 3 (as "3 e 4" do prompt).
- Rede 2 cresce as 6 camadas; SVM depois de cada uma.
- Checkpoint: melhor κ em validação (exceção à regra de 10/11, só neste experimento).
- Filtros de cada camada nova: spf_grade (FPS ou random_z + SoftHebb) sobre as features já treinadas.
- Augmentation: aguardando o arquivo que o Mateus vai enviar.

## O que existe (P0)
- Crescimento `layerwise` com congelamento e decoder novo por camada: `experiments/ray/runners/growth.py:431-445` (discovery). Falta a fase descongelada por camada.
- SimCLR (NT-Xent) já liga no lugar do decoder: `runner_args.lejepa.objective: simclr` (`methods/autoencoder/autoencoder_flim_module.py:625-658`). Nunca rodou dentro do laço de crescimento.
- VICReg só existe como regularizador do LeJEPA (`losses/vicreg.py:122`, pesos 0/1/1). Falta `objective: vicreg` com expansor 8192³ e pesos 25/25/1.
- FPS, random_z, SoftHebb, whitening por canal, GeM e TTA D4: só no `eval/spf_grade.py`. A ponte para o crescimento é `salvar_encoder` → `init_flim_dir`, hoje só para a rede inteira.
- Faltam: GeM por região 4×4 e 2×2 (redes 1 e 4), bloco Inception (rede 4), TTA com 1/4/8 vistas configurável.
- Modelos grandes: 5 modelos completos nas 6 bases × 50/75/100 × 3 splits, val e test (`discovery/results/benchmark/*.csv`).

## Código novo (P1), um arquivo por agente
| id | o quê | onde |
|---|---|---|
| I1 | YAML da rede 2 (D=2048), FPS e random_z | `experiments/autoencoder/` |
| I5 | fase descongelada por camada + recriar `ssl_head` a cada camada | `growth.py` |
| I6 | `objective: vicreg` com expansor | `autoencoder_flim_module.py` |
| I8 | gêmeos aleatórios (`random_layer_classic`) | YAMLs |
| IX | spf_grade gera só a camada k sobre o encoder treinado | `eval/spf_grade.py` |
| I2–I4, I7, I9 | redes 1, 3, 4, Inception/residual, augmentation e TTA | depois da rede 2 |

## Grade do primeiro bloco
Rede 2 × {FPS, random_z} × {AE, SimCLR, VICReg} × {SPiFiL, aleatório} × 3 bases de fundo preto × 3 splits × 50%.
- 81 runs (54 SPiFiL + 27 aleatórios; o gêmeo aleatório é o mesmo para FPS e random_z), cada um com 6 camadas e 2 fases por camada.
- O custo por run é medido em Q1. A estimativa grosseira é de 1 a 3 h por run.
- Cabem cerca de 7 runs ao mesmo tempo: 3 no jaci e 2 em cada uma das GPUs 2 e 3 do athena.
- Isso dá cerca de 1 a 2 dias por porcentagem.
- Ordem: aleatório → AE → SimCLR → VICReg; FPS antes de random_z; 50 → 100 → 75.

## Augmentation (de improve_my_experiments.md, enviado pelo Mateus em 13/11; usado só como referência)
- Entra: flip H/V e rot90 (sem interpolação). Talvez: brilho/contraste leve só no canal L. Recorte: NÃO (Mateus, 13/11).
- Fica fora: matiz/saturação, blur, solarize, cinza, cutout, shear/elástica.
- Só no treino com backprop (entrada = alvo no AE; vistas no SimCLR/VICReg). spf_grade e SVM veem a imagem original.
- Máscara = pixels não pretos da própria imagem (fundo preto), a mesma geometria aplicada nos dois.

## Contrato entre os agentes de P1 (chaves e interfaces)
Repositório: discovery (`/dados/home/moliveira/Scalable_Hybrid_FLIM_discovery`). Tudo opt-in: sem a chave, o caminho antigo fica idêntico.
- **IX `eval/spf_grade.py`**: modo "uma camada". Recebe um encoder FLIM já treinado (diretório no formato de `amostra.salvar_encoder`) e o índice k. Monta só a camada k (canais, kernel, pool dados), com `selecao: fps|random_z` e `softhebb` como no spf_grade, sobre as features das camadas 1..k-1. Grava o encoder 1..k no mesmo formato FLIM num diretório de saída. Expõe uma função Python importável `montar_camada(encoder_dir, k, cfg: dict, out_dir) -> out_dir`, além da linha de comando.
- **I5 `experiments/ray/runners/growth.py`** + `experiments/ray/schema.py`: novas chaves de `runner_args`:
  - `layerwise_spf: {selecao, softhebb, n_imagens, ...}`: antes de treinar L{k}, exporta o encoder do ckpt L{k-1} para FLIM, chama `montar_camada` e carrega o resultado como `init_flim_dir`. Sem isso, usa o `grow` atual.
  - `unfreeze_phase: true`: depois de `L{k}` (congelado), roda `L{k}_ft` com tudo solto, partindo do ckpt de `L{k}`, e usa o ckpt resultante como base da camada seguinte.
  - `ckpt_by: val_kappa`: em cada fase, escolhe o ckpt pelo melhor κ de validação (best_kappa.ckpt).
  - O gêmeo aleatório (`random_layer_classic: true`) não chama o `montar_camada`.
- **I6 `methods/autoencoder/autoencoder_flim_module.py`**:
  - `lejepa.objective: vicreg` com expansor (`expander: "8192-8192-8192"`, Linear+BN+ReLU, como o projetor do Barlow) e pesos `vicreg: {inv: 25, var: 25, cov: 1}`.
  - `parasite_aug: {flip: true, rot90: true, bc: 0.0}` (sem recorte) (desligado por padrão). No AE, entrada = alvo aumentado. No SimCLR/VICReg, ele substitui as vistas `_ssl_view` (sem cinza, blur ou solarize).
  - Ao carregar `init_ckpt`, ignora `ssl_head`/expansor de largura diferente (recria por camada).

## Decisões do Mateus, 2ª rodada (13/11)
- Recorte com vazamento: NÃO entra.
- Augmentation: flip + rot90 nas 3 bases (ovos inclusive).
- Primeira rodada do Apêndice C entra como experimento à parte, sem backprop: baseline, aug só no banco FPS, aug só no treino do SVM (`_aug8` existe em eval/svm.py), aug nos dois. O crescimento com backprop usa aug só no treino.
- SoftHebb: desligado na 1ª rodada, igual dentro do par FPS/random_z.
- Tabela final: checkpoint pelo κ de validação, número reportado no teste.
- Antes de rodar no jaci: comparar e mesclar eval/spf_grade.py com a cópia do jaci.

## Balanço CPU × GPU (pedido do Mateus, 13/11)
- jaci: 128 núcleos livres, 237 GB de RAM, 1 GPU com ~85 GB livres → 3 runs; cada um com `num_workers` 8 e `OMP_NUM_THREADS`/`MKL_NUM_THREADS` 16 (~3×24 = 72 núcleos); sobram ~50 para o SVM e o spf_grade.
- athena: 96 núcleos, mas carga ~29 de outros usuários → usar ~56; GPUs 2 e 3 com 2 runs cada → cada run com `num_workers` 4 e threads 8 (~48 núcleos); o SVM roda um de cada vez com 8 threads.
- SVM (CPU pesado): fila única por host, um de cada vez, enquanto a GPU treina o próximo run.
- Q1 mede o uso da GPU (nvidia-smi dmon) e a carga da CPU com 1 e 2 runs por GPU; se a GPU ficar < 80% ocupada esperando dados, sobe `num_workers`; se a CPU saturar, desce.

## Épocas (pedido do Mateus, 13/11)
- Cada fase (L{k} congelada e L{k}_ft solta): até 1000 épocas, early stopping com paciência 50 no κ de validação (o mesmo critério do checkpoint).

## Mudança do Mateus (13/11 ~02:00) — VALE SOBRE TUDO ACIMA
- NADA aleatório: sem gêmeos aleatórios (`random_layer_classic`) e sem `random_z` (que é filtro sorteado `init.normal_`, spf_grade.py:1028). Toda camada nova é inicializada pelo SPiFiL (FPS).
- Primeiro método: growth (autoencoder), depois SimCLR, depois VICReg.
- Grade da rede 2: 3 métodos × 3 bases × 3 splits = 27 runs por porcentagem; 81 no total (50 → 100 → 75).
- Não lançar nada até o Mateus mandar.

## Bases (Mateus, 13/11 ~02:10): todas as 13
- Fundo preto: helminth_eggs, helminth_larvae, protozoan_cysts. Fundo real: Eggs, Larvae, Protozoan (avaliam pelo worktree impuro? conferir o data root). Outras: aid, remote_sensing, flowers, coconut_trees, ham10000, food101.
- Ordem (sem preferência do Mateus; segue "50% primeiro"): porcentagem (50 → 100 → 75) > método (growth/AE → SimCLR → VICReg) > base > split.
- 13 bases × 3 splits × 3 métodos = 117 por porcentagem, 351 no total (menos onde a base tiver menos splits).
- Augmentation por base: flip + rot90 em todas, menos flowers (só flip horizontal); `bc` só nas 3 de fundo preto (a máscara "pixel não preto" não vale nas outras).

## Bases, versão final (Mateus, 13/11 ~02:15) — SUBSTITUI a seção anterior
- SEM fundo preto (o Mateus já sabe o resultado). São 12 bases ao todo, não 13 (erro de conta meu); ficam 9.
- Ordem: primeiro as 6 não parasitas (aid, remote_sensing, flowers, coconut_trees, ham10000, food101) e depois os 3 parasitos de fundo real (Eggs, Larvae, Protozoan).
- Dentro de cada grupo: porcentagem (50 → 100 → 75) > método (growth/AE → SimCLR → VICReg) > base > split.
- 9 bases × 3 splits × 3 métodos = 81 por porcentagem, 243 no total (menos onde a base tiver menos splits; Protozoan de fundo real pode ter 1 só).
- Augmentation: flip + rot90, menos flowers (só flip horizontal). `bc` desligado em todas (nenhuma tem fundo preto).
- Conferir em Q1 que o growth do discovery lê os dados de FUNDO REAL de Eggs/Larvae/Protozoan (no main, o impuro só existe no worktree impuro).

## SoftHebb (Mateus, 13/11): LIGADO em todas as camadas — porque ficou bom em algumas bases (ex.: Larvae 0,679 → 0,732 na rede de 3 camadas, 10/11)
- Cada camada nova: FPS escolhe os filtros → SoftHebb ajusta → backprop (congelado, depois solto). GeM global e TTA 8 vistas na avaliação; whitening por canal entre camadas como na receita.
- Testar o SoftHebb dentro de `montar_camada` (ainda não testado) e o kernel 1×1 antes de lançar.

## Augmentation sempre ligada (Mateus, 13/11)
- `parasite_aug` ligado em TODOS os experimentos de treino: rede 2, redes 1/3/4, Inception/residual, growth/SimCLR/VICReg (flip + rot90; flowers só flip H).
- Único lugar sem augmentation: o caso "baseline" do estudo de 4 casos (bloco 6), que existe justamente para medir o efeito dela.

## Prioridade nova (Mateus, 13/11 ~03:45) — VALE SOBRE A ORDEM ACIMA
- Foco: 50% e 100% nas 9 bases, nas 4 arquiteturas do Apêndice A (rede 2, 1, 3, 4). 75% só depois. Inception/residual (item 2) e o estudo de augmentation (item 6) depois disso.
- Assim que cada run termina: SVM (val e test, gem+TTA+PW; leitura da arquitetura) com o máximo de núcleos livres, e o resultado vai para o Mateus.
- GPUs nossas (athena 2, 3; jaci 0) sempre ocupadas: 1 monitor por GPU puxa o próximo item da fila quando a GPU tem folga; num_workers calculado a cada lançamento pelos núcleos livres.
- As cadeias g2_athena/g2_jaci terminam no bloco atual (g2_ae p50, bases do 1º grupo): os YAMLs foram renomeados para g2q_*.yaml, então as iterações seguintes da cadeia falham na hora (rc≠0 no g2.status) sem rodar nada. Daqui em diante quem lança são os monitores.
- Rede 1: canais 32-48-64-128 (C=128 → 128×16 = 2048), k 5-5-5-1, pools 2-2-2-1, leitura GeM por região 4×4.
- Rede 3: m=64 → 64-128-192-320-512-832 (soma 2048), k 5-5-3-1-1-1, pools 2-2-2-2-1-1, leitura GeM em cada camada + cat.
- Rede 4: L1-L3 5-5-3 (32-64-128); L4 Inception 1×1|3×3|5×5 (64 por ramo) + pool; L5 Inception (96 por ramo); L6 1×1 → C=512 (512×4 = 2048); leitura GeM 2×2.

## Whitening (Mateus, 13/11 ~04:05)
- SVM: a camada avaliada entra com a largura inteira (L6 = 2048), whitening do mesmo tamanho, sem o corte do PCA para n_treino − 1 (o gem_tta_pw cortava a L6 do remote_sensing para 344). Novo: whitening com encolhimento que mantém as D dimensões (sufixo novo nos YAMLs de SVM).
- Modelo: whitening por canal entre as camadas no forward do encoder (`whiten_between: true`), o mesmo do spf_grade, em todas as redes.
- Fila segura (g2.hold) até os dois entrarem.
