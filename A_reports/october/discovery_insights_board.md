# Quadro de insights — discovery híbrido SPiFiL

Só se acrescenta texto aqui, nunca se edita ou apaga: `cat >> ... <<'EOF'` com um bloco por insight. Cabeçalho: `## <AAAA-MM-DD HH:MM> [agente] <título>`.
Cada bloco traz: o que rodou (YAML, work_dir), métricas (κ val por camada × dataset, κ teste quando houver), a conclusão e o que o próximo agente deve tentar ou evitar.
Agentes: **A0** = orquestrador (frente 1: baseline R1 e dilation R2), **A1** = frente da loss/objetivo, **A2** = frente da convolução (largura/kernel).

## 2026-10-30 00:55 [A0] R1 baseline, SVM em validação (flatten)
- R1 = layerwise 3, 5x5, 24/32/48, DISF/50, BCE, 200 épocas (growth_disc_r1.yaml).
- κ val: eggs L1 0.891 / L2 0.892 / L3 0.898 (2 splits). larvae split1: L1 0.796 / L2 0.822.
- A val_recon_loss trava em ~0.468 com ~60 épocas: a BCE para de ensinar cedo.
- Alvo (I-JEPA, teste): eggs 0.9397, larvae 0.9433, protozoan 0.8636. Faltam ~4 pontos em eggs.

## 2026-10-30 01:05 [A1] a BCE "parada" em ~0.468 é o piso de entropia, não um platô de aprendizado
- Com alvo contínuo em [0,1], o mínimo da BCE é a entropia do próprio alvo, perto de 0.47 para LAB. O código admite isso em autoencoder_flim_module.py:~650 ("BCE floor ... target's own entropy"). Ver 0.468 parado não prova que o modelo parou de aprender. Prova só que a BCE é uma régua ruim para isso.
- Candidato A1-1, `growth_disc_a1_mask50.yaml`: R1 mais recon mascarado, com 50% dos patches 20x20 escondidos (preenchidos pela média da imagem) e a BCE contando só neles. A chave nova `model.init_args.mask_ratio`/`mask_patch` fica desligada por default.
  - Hipótese: preencher o buraco obriga o encoder a carregar contexto e morfologia. Copiar pixel não basta. É o mesmo motivo de o I-JEPA ganhar: ele prevê o que está escondido.
- tmux `disc_a1`, work_dir `.../spifil_growth/disc_a1_mask50`. CSVs `results/svm_disc_a1_mask50{,_val}.csv`.

## 2026-10-30 01:23 [A2] lancei disc_a2_w64 (largura)
- growth_disc_a2_w64.yaml: R1 com 32/48/64 filtros (antes 24/32/48), o resto igual. Sessão tmux disc_a2_w64, saída em artifacts/spifil_growth/disc_a2_w64.
- Hipótese: mais sementes por camada dão ao SPiFiL mais padrões de cor e textura, e a L3 chega a 36.864 dimensões no flatten.
- Depois vêm o kernel 7 na L1 e o kernel 3 em L2–L3.

## 2026-10-30 02:00 [A3] mudança de abordagem: SPiFiL montado no repo, pacote intocado
- Correção do usuário: não editar o pacote spifil. Removi o worktree que tinha criado no SPiFiL, sem mudanças; a branch discovery_a3 foi apagada.
- Montei a minha versão em flim/spifil_a3.py. Ela usa o `grow` de flim/spifil.py e troca só, e só durante o grow, `spifil.learner.build_filters` e `GrowOneLayer`. Liga com `runner_args.grow_impl: a3`; o default é o comportamento atual.
- Eixo A3-1 (popz): o z-score que vira peso e bias do filtro sai de TODOS os patches da camada anterior, não só dos ~24 escolhidos. A docstring do builder promete "z-score the dataset per feature", mas o código usa `selected`, e 24 amostras para 75+ features dão um desvio que é ruído. Hipótese: filtros centrados na média real dos dados, com o ReLU cortando em torno dela, dão ativações mais informativas e menos achatadas (R6: magnitude achatada).
- Teste em CPU: com pool = selecionados, a saída é idêntica ao original.
- Run: growth_disc_a3_popz.yaml (R1 + grow_impl a3), tmux disc_a3, saída em artifacts/spifil_growth/disc_a3_popz.

## 2026-10-30 [A0] REGRA NOVA DO USUÁRIO: modelo ≤ 1M de parâmetros
- O encoder, mais a projetora ou preditora se houver, tem no máximo 1M de parâmetros. A projetora ou preditora nunca pode ser maior que o encoder.
- O decoder não entra na conta.
- O R1 tem ~59k parâmetros no encoder. Cada candidato informa o seu nº de parâmetros no bloco do quadro.

## 2026-10-30 01:30 [A1] parâmetros (regra nova: encoder + projetora ≤ 1M; decoder não conta)
- A1-1 mask50 tem o mesmo encoder do R1: 3 convs 5x5 3→24→32→48 = 59.504 parâmetros. Não tem projetora nem preditora. Está dentro da regra.

## 2026-10-30 01:30 [A2] parâmetros do encoder (teto de 1M: encoder + projetora; o decoder não conta)
- Contagem por conv = cin·cout·k² + cout:
  - R1 24/32/48 k5: 59504
  - disc_a2_w64 32/48/64 k5: 117744
  - 48/64/96 k5: 234208
  - 64/128/192 k5: 824384
- Na largura, todos ficam bem abaixo de 1M. O SVM recebe o flatten direto, sem projetora; se uma for criada, ela não pode ser maior que o encoder.

## 2026-10-30 [A3] teto de 1M de parâmetros: o A3-1 cumpre
- O A3-1 (popz) não muda nº de filtros nem de sementes: 24/32/48 filtros 5x5, iguais ao R1. Só muda a estatística do z-score.
- Encoder: 1.824 + 19.232 + 38.448 = **59.504 parâmetros** (conv + bias). O state_dict do ckpt R1 L3 soma 119.008 em `model.encoder`, mas é a mesma conv contada duas vezes (armadilha conhecida: encoder duplicado no state_dict). Não existe projetora no módulo; o decoder (49.105) não conta.
- Folga contra o teto: ~17x. Mesmo dobrando a largura, o A3 fica bem abaixo de 1M.

## 2026-10-30 [A0] REGRAS NOVAS: profundidade livre e teto de 5M
- **Profundidade:** não se limitem a 3 camadas. Basta passar `layerwise: N` com `layer_channels`, `layer_kernels`, `layer_pools` e `layer_dilations` de N itens. Sem código novo.
- **Teto:** 5M de parâmetros (encoder mais projetora). Projetora ≤ encoder. O decoder não conta.
- **R2 do A0:** vai ter 8 camadas (24/32/48/48/64×4; pool 2 nas L1–L4, sem pool nas L5–L8 com dilation 2,2,4,4; ~256k parâmetros). Sobe quando o R1 terminar.

## 2026-10-30 01:35 [A2] teto de 5M: escada de largura planejada
- Ordem dos candidatos de largura com kernel 5, um por vez, sempre em cima do vencedor anterior:
  - w64 32/48/64: 117744
  - w96 48/64/96: 234208
  - w192 64/128/192: 824384
  - w256 96/192/256: 1697344
- Kernel 7 na L1 depois do melhor de largura. A profundidade fica com o A0 (R2), e combino com ela quando os dois tiverem resultado.
- Risco: com o flatten, a L1 já tem C·99·99 dimensões, então o SVM fica mais lento à medida que a largura cresce.

## 2026-10-30 [A3] trava N≤D com mais camadas: ela já é violada a partir da L2
- Log do R1 (logs/disc_r1_tmux.log, linhas `[grow] N= D=`):
  - L1: N=400, D=75. Só aqui N > D.
  - L2: N=400, D=600 (24 canais × 5×5).
  - L3: N=400, D=800.
  - Larvae (2 classes) usa N=100 (2 imagens × 50 superpixels): D=600 e 800 com N=100.
  - O R1 passa porque `budget_check: false`. O Mahalanobis do score usa shrinkage e não está bem condicionado. É provável que isso pese mais no larvae, onde a L2/L3 ganha pouco.
- Em camadas mais fundas fica pior: com k5, a L4 de 48 canais tem D=1200. O N continua 400, porque as sementes só são projetadas, não aumentam.
- Na minha montagem (popz), média e desvio são por feature (diagonal): 400 amostras por feature continuam sãs para qualquer D. A dobra dos filtros não quebra. O que fica fraco é o score/ranking (covariância cheia) do pacote.
- Sugestão para quem for crescer mais camadas: subir `n_images` e `n_superpixels` nas camadas fundas, para N chegar a pelo menos 2·D (L4 k5 com 48 canais pede ~2400 sementes). Outra saída é kernel 3 nas camadas fundas: D=48·9=432, já abaixo de N=400×… com ~10 imagens/classe. Vai ser o meu próximo eixo (A3-2: mais sementes), depois do resultado do popz.
- Parâmetros: com o teto de 5M há muita folga. O A3-1 usa 59.504.

## 2026-10-30 [A0] REGRA QUE PREVALECE: o SPiFiL tem que contribuir
- Todo candidato inicializa as camadas com os kernels do SPiFiL, nunca aleatórios.
- O melhor candidato de cada frente ganha um controle `random_layer_classic: true`, que o A0 roda. O SPiFiL tem de vencer esse controle.
- Informe no bloco a similaridade de cosseno média entre os pesos de cada Lk no best ckpt e os do Lk_grow. Se os filtros se afastarem dos do SPiFiL, o ganho não conta.

## 2026-10-30 [A4] Regras de treinamento (1ª análise: 23 estágios disc_* + 15 históricos, eggs/larvae a 50%)
Fontes: ckpts best_kappa em artifacts/spifil_growth/disc_*/…/checkpoints e curvas `probe/val_recon_loss` pelo W&B (projeto flim-ssl). Os scripts estão no scratchpad a4/.
1. **O "best" no fim é efeito do scheduler, não aprendizado.**
   - O best sai nas épocas 193–199 de 200 (disc_r1, 21 de 22 estágios) e nas 964–990 de 1000 (layerwise5*).
   - O early stop nunca dispara (wait_count 0 em todos os ckpts, patience 50).
   - Com cosine até max_epochs, a lr vai a 0 e a loss cai ~1e-4 até o fim.
2. **A loss chega a 1e-3 do mínimo cedo:** épocas 14–26 na L1, 8–11 na L2, 6–9 na L3, 8–11 nas L4–L5 (histórico).
   - De 1000 para 200 épocas, o mínimo muda só 1,6e-4 (L1 eggs: 0.46743 → 0.46759).
   - Camadas mais fundas convergem mais rápido.
   - **Proposta:** max_epochs 60, cosine até 60, ~3x mais barato. Opcional: 60/40/30 para L1/L2/L3+. Antes de adotar, falta uma checagem: κ val de um ckpt de 60 épocas contra o de 200 numa célula.
3. **A loss de reconstrução não prevê o κ.**
   - Em eggs, a L3 tem loss maior que a L1 (0.4678 contra 0.4675), mas κ val maior (0.898 contra 0.891).
   - Entre splits, a loss varia 1e-4 e o κ varia 2 pontos, sem correlação.
   - O piso depende do dataset: eggs 0.4675, larvae 0.4649.
   - **Regra:** a loss só serve para detectar NaN ou divergência. O ranking é sempre pelo SVM.
4. **Largura, BN, aug e popz não mudam o piso da BCE.**
   - O piso fica em 0.4674–0.4679: w64 (A2) 0.4677, popz (A3) 0.4679, bn 0.46743, aug 0.46743, R1 0.46759.
   - A BN só atrasa a convergência: v10 0.4857 contra 0.4722; 1e-3 na época 43 contra 32.
   - O efeito desses eixos, se houver, só aparece no SVM.
   - A máscara (A1) tem piso 0.4794. Não é comparável, porque a loss só conta nos patches escondidos, e converge mais devagar (1e-3 nas épocas 37–42).
5. **lr, wd e warmup nunca variaram:** 5e-4, 5e-2 e warmup 10 em todos os runs (configs/model/autoencoder/flim/helminth-eggs/split_1.yaml:8-9). Não há regra a tirar; é um eixo inexplorado.
6. **Custo.**
   - Sozinho na GPU: eggs 7,7–8,7 s/época, larvae 6–6,5 s/época.
   - Com ~8 runs na GPU 3: 19–23 s/época, 2,5–3x mais lento. O gargalo é a CPU (load 111 em 96 núcleos).
   - SVM flatten: ~4,4 min por job, 2 concorrentes (10 jobs em 22 min).
7. **Contagem de parâmetros:** o state_dict tem o encoder duplicado, então a soma bruta dá o dobro. Real: L1 1.824, L3 ~59,5k.
8. **Pendente:** a regra de profundidade (em que Lk o κ val para de subir e se a época do best muda) depende do R2 com 8 camadas e do novo teto de 5M de parâmetros.

## 2026-10-30 01:50 [A2] o SPiFiL contribui? cos(pesos best, kernels do Lk_grow)
- Formato: o `conv{k}-kernels.npy` do grow vem como (k·k·cin, cout), na ordem **HWC**. Comparar na ordem CHW dá cos≈0, um falso negativo.
- disc_r1, cos médio por filtro:

  | dataset | L1 | L2 | L3 |
  |---|---|---|---|
  | eggs split1 | 0.998 | 0.489 | 0.596 |
  | larvae split1 | 0.999 | 0.731 | 0.777 |

  - A norma ‖W‖/‖G‖ fica entre 0.91 e 1.13.
- Leitura: na L1 o filtro continua sendo o do SPiFiL (o decoder quase não mexe). Na L2 e na L3 cerca de metade a 3/4 da direção ainda é do SPiFiL; ele contribui, e a reconstrução ajusta.
- Filtros repetidos (pares com cos > 0.95 dentro do grow):
  - R1 L1 eggs: 3, 5 e 7 pares de 24 filtros (splits 1–3). Larvae: 0 pares.
  - w64 L1 eggs: 7 e 3 pares de 32.
  - Na L1 o SPiFiL já gera repetição em eggs (3 canais × 25 = 75 dimensões, poucas para tantas sementes). Alargar a L1 tende a só duplicar filtros; a largura extra rende mais na L2/L3, onde há 600–800 dimensões.
- Próximo candidato de largura: crescer só L2/L3, ex. 24/64/96. A L1 continua com 24.

## 2026-10-30 02:10 [A1] cosseno entre os pesos SPiFiL (Lk_grow) e o best ckpt (scratchpad/cos_spifil.py <família>)
- Usei a média, por filtro, do cosseno entre conv{k}-kernels.npy e o state_dict `model.encoder.conv{k}.0.weight`.
- **R1 (BCE):**

  | | L1 | L2 | L3 |
  |---|---|---|---|
  | eggs | 0.996–0.998 | 0.31–0.54 | 0.42–0.60 |
  | larvae | 0.998–0.999 | 0.47–0.73 | 0.76–0.78 |

  - A L1 quase não sai do SPiFiL. A L2 e a L3 treinadas se afastam MUITO dos kernels SPiFiL (cos ~0.3–0.6 em eggs), ou seja o backprop reescreve as camadas novas. Isso pesa para a regra "o SPiFiL tem que contribuir". Sugiro reduzir o lr da camada nova, ou usar o warmup com ela congelada, para mantê-la perto do SPiFiL.
- **A1 mask50:** até agora L1 0.999 (eggs split1 e split2), igual ao R1. Faltam L2 e L3.

## 2026-10-30 [A3] o SPiFiL contribui? Cosseno médio por filtro entre o best ckpt Lk e o Lk_grow
- Método: em cada filtro, cosseno entre `model.encoder.conv{k}.0.weight` do best ckpt do estágio Lk e os kernels de `Lk_grow/conv{k}-kernels.npy`, convertidos com `flim.weights.get_weights`. A tabela mostra a média sobre os filtros. 1 = o treino manteve o kernel do SPiFiL; ~0 = o treino reescreveu o kernel.

| run | L1 | L2 | L3 |
|---|---|---|---|
| r1 eggs s1 | 0.998 | 0.489 | 0.596 |
| r1 eggs s2 | 0.998 | 0.538 | 0.483 |
| r1 eggs s3 | 0.996 | 0.314 | 0.417 |
| r1 larvae s1 | 0.999 | 0.731 | 0.777 |
| r1 larvae s2 | 0.999 | 0.674 | 0.763 |
| r1 larvae s3 | 0.998 | 0.465 | — |
| a3_popz eggs s1 | 1.000 | — | — |
| a3_popz eggs s2 | 0.999 | — | — |

- Leitura:
  - Na L1 o kernel do SPiFiL fica praticamente intacto: o backprop quase não o move. Ali o SPiFiL contribui de fato.
  - Na L2 e na L3 o treino reescreve metade ou mais dos kernels (cosseno de 0,3 a 0,6 em eggs). Nas camadas fundas o SPiFiL vira só um ponto de partida.
  - Bate com a trava N≤D furada (N=400 contra D=600/800, e N=100 no larvae): poucas sementes para patches grandes dão kernels fracos, e o decoder puxa os pesos para longe.
- Consequência para a regra "o SPiFiL tem que contribuir": a alavanca é a L2+.
  - O A3-2 vai aumentar as sementes nas camadas fundas, não os filtros: subir `n_images`/`n_superpixels` até N ≥ D.
  - Meta: cosseno na L2/L3 subir e κ val subir junto. Se só o cosseno subir e o κ cair, o SPiFiL está "segurando" sem ajudar.
- Para o A0 e o A2: com `layerwise_backprop` vale medir este mesmo cosseno, para ver se o backprop conjunto apaga o SPiFiL também na L1.
- Não mexo em growth.py nem em schema.py enquanto o A2 edita.

## 2026-10-30 [A4] O SPiFiL contribui? Deriva dos pesos e controle aleatório
Script: scratchpad a4/drift.py. Compara a conv da camada Lk no best ckpt contra `Lk_grow/conv{k}-kernels.npy`. O layout foi conferido: a L1 dá cos ~0,998.
1. **Deriva por camada:** cosseno médio por filtro e mudança relativa ‖W−W0‖/‖W0‖, nos 22 estágios disc_*.
   - **L1 fica quase igual ao SPiFiL:** cos 0,996–0,9997, mudança de 5–13%. Vale no R1, mask50, w64 e popz.
   - **L2 é sobrescrita:** cos 0,31–0,73 (mínimo por filtro 0,05–0,25), mudança de 0,62–1,61x a norma.
   - **L3 é sobrescrita:** cos 0,42–0,78, mudança de 0,33–1,32x.
   - O larvae deriva menos que o eggs: L2 0,47–0,73 contra 0,31–0,54.
   - **Leitura:** do jeito que está (lr 5e-4, 200 épocas, só a camada nova treina), o SPiFiL só sobrevive na L1. Nas L2–L3 ele vira só uma inicialização.
2. **Deriva contra κ val:** Pearson(cos, κ val) = −0,35, com n=10 (só o R1 tem κ val por enquanto).
   - Isso é confundido pela camada: L2/L3 têm κ maior e cos menor.
   - Dentro da mesma camada não há relação. Eggs L2: cos 0,49 → κ 0,907; 0,54 → 0,881; 0,31 → 0,889.
   - Não há evidência de que afastar-se do SPiFiL ajude ou atrapalhe. Precisa de mais pontos (A1/A2/A3 com SVM val).
3. **Controle pareado no histórico** (results/eval_growth_stages_g5_{random,head}_pct{50,5}.csv; growth_g5_random.yaml:1-3,40).
   - g5_random troca os valores do SPiFiL por aleatórios preservando a norma filtro a filtro. Difere de g5_head também em `impurities`. Os estágios round*_head usam head_finetune supervisionado e foram ignorados; comparo só os estágios de autoencoder.
   - **pct50, stage2 (κ teste, 3 splits), aleatório × SPiFiL:**
     - eggs: 0,9081 × 0,9062
     - larvae: 0,859 × 0,853
     - protozoan: 0,8595 × 0,8584
     - O aleatório empata ou passa por menos de 0,006.
   - **pct5, stage2:** eggs 0,6252 × 0,6664, ou seja, o SPiFiL ganha por +0,041. Larvae e protozoan dão valores idênticos, o que sugere que a troca não aconteceu ali (a conferir).
   - **Regra:** com 50% dos dados, o valor dos pesos SPiFiL da camada nova ≈ aleatório com a mesma norma. O SPiFiL só pesa com pouco dado.
4. **Sugestão para o SPiFiL contribuir** (A0 decide):
   - (a) lr menor para a conv nova, ou congelar o peso SPiFiL e treinar só o bias e o decoder por algumas épocas;
   - (b) penalidade L2-SP, λ‖W−W0‖², puxando a conv para o SPiFiL;
   - (c) medir κ val do Lk_grow puro (sem treino) contra o Lk treinado, para separar a contribuição do SPiFiL da contribuição do backprop. Custa só SVM.

## 2026-10-30 [A0] REGRA REVISTA: o SPiFiL pode ser só o inicializador
- A deriva dos filtros é permitida e não invalida ganho.
- A contribuição do SPiFiL é provada por ESTABILIDADE contra o controle aleatório (`random_layer_classic`, mesma arquitetura, ≥3 seeds): variância do κ entre seeds e splits, épocas até estabilizar e oscilação da curva.
- O A0 roda o controle do R1.

## 2026-10-30 02:15 [A2] nova frente: SPiFiL + backprop no layerwise (chave layerwise_backprop)
- Código pronto e smoke OK. `runner_args.layerwise_backprop: s` faz conv1..conv{k-1} treinarem com lr·s junto com a camada nova; sem a chave, nada muda.
- Opcional: `overrides: model.init_args.new_layer_lr_scale` (default 1.0) aplica uma escala de lr só na camada nova.
- Candidato disc_a2_bp01: R1 + layerwise_backprop 0.1, com 59.504 parâmetros (os mesmos do R1). Começa sozinho quando o w64 acabar.
- A escada de largura para cima foi cancelada; o w64 termina e entra na lei de largura.
- Hipótese: o SPiFiL dá um bom ponto de partida e o backprop leve deixa L1–L(k-1) se ajustarem às features de reconstrução da Lk. É o ganho do grid4 backprop (o melhor em protozoan), agora sem deixar de crescer camada a camada.

## [A0] Achado: o backprop puxa a L2 para o fundo (2026-10-30)
- Figura: `A_reports/october/figs/disc_r1_eggs_s1_L2_act_raw_vs_trained.png` (eggs s1, 4 classes da val).
- SPiFiL cru: a L2 ativa só no ovo; o fundo fica ~0. Após o backprop, o fundo sobe na média dos canais e o canal 2 vira um detector de fundo (acende fora, apaga o ovo). Cos médio dos mapas cru vs treinado: 0,78.
- Hipótese (SPiFiL): os seeds do SPiFiL vêm do objeto. A BCE de recon pesa cada pixel por igual e a maior parte da imagem é fundo, então o gradiente gasta filtros para redesenhar o fundo. Não é overfitting clássico (memorizar o treino); é um desvio do objetivo. Os filtros vão para onde está a loss, não para onde está a classe.
- Teste com uma mudança: loss ponderada pelo objeto (máscara de foreground, sem rótulo de classe), ou recon só dentro do objeto. Métrica: κ do SVM na val e a fração de ativação fora do objeto.

## 2026-10-30 [A3] máscara de objeto sem anotação vinda do SPiFiL cru (para o A1 ponderar a loss)
- Teste com a L1 do grow (R1, split 1, 20 imagens de treino), filtro a filtro contra as máscaras de data/:
  - Alguns filtros marcam o fundo quase perfeitamente (AUC ≈ 0) e outros o objeto (AUC até 0,82).
  - A média de todos os filtros dá AUC 0,20 (invertida): por isso o "SPiFiL olha o objeto" só vale filtro a filtro.
- Critério sem anotação: o objeto cobre 2–4% da imagem, então os filtros que disparam em mais pixels são os de fundo. Objeto = 1 − média dos 25% mais densos.
  - AUC: eggs 0.936, larvae 0.998, protozoan 1.000.
  - Pela "esparsidade" direta (média dos mais esparsos) fica pior: 0.78, 0.89, 0.89.
- Pronto em `flim.spifil_a3.object_map(feats_L1) -> (B,1,H,W)`. Avisei o A1.
- Ligação com o achado do A0 (o backprop puxa a L2 para o fundo): ponderar a reconstrução por esta máscara faz o decoder cobrar o objeto. Isso deve segurar a L2/L3 no objeto, como o SPiFiL cru faz.

## 2026-10-30 03:55 [A1] base: ativação dentro vs fora do objeto; candidato A1-2 fg01 pronto (não lançado)
- **Definição** (para o A2 usar a mesma). Para 200 imagens de val, split 1:
  - ReLU(Lk), média nos canais; máscara de masks/ por nearest até o tamanho do mapa.
  - `fora` = soma da ativação fora do objeto / soma total.
  - `razão` = (densidade dentro) / (densidade fora).
  - Script: scratchpad/bg_frac.py `<família> <dataset> <k>`.
- **As imagens já vêm com o fundo de cor constante.** A máscara sem rótulo |LAB − mediana da borda| > 0.04 tem IoU 1.000 com masks/ (eggs; o objeto ocupa 3,2% da área). Dá para usar o foreground sem anotação e sem o object_map do A3.
- **R1 eggs:**

  | camada | fora (cru → treinado) | razão (cru → treinado) |
  |---|---|---|
  | L2 | 0.961 → 0.968 | 1.27× → 1.03× |
  | L3 | 0.963 → 0.973 | 1.17× → 0.84× |

  - O treino com BCE plena apaga o contraste objeto/fundo que o SPiFiL trazia. Na L3 a ativação passa a ser MAIOR no fundo. Isso confirma o achado do A0 em número.
- **Candidato A1-2 `growth_disc_a1_fg01.yaml`:** R1 com `model.init_args.recon_bg_weight: 0.1` (BCE com peso 0.1 no fundo e 1 no objeto). Default 1.0 = comportamento de hoje.
  - Hipótese SPiFiL: os seeds vêm do objeto. Tirar o fundo da loss deve manter a razão > 1 e o cosseno com o grow mais alto, e o κ val deve subir. Mesmo encoder do R1: 59.504 parâmetros.
  - Dry-run com 9 pontos, 0 erros. Ainda não lançado: espera o A1-1 mask50 terminar (um experimento por agente) ou o A0 decidir trocar a ordem.
- **A1-1 mask50, às 03:54:** só eggs s1 L1 e s2 L1–L2. A GPU 3 está com ~43 GB usados por 4 experimentos e está lenta. A val_recon_loss mascarada é ruidosa, então o early stop não corta e cada estágio vai até 200 épocas.

## 2026-10-30 04:05 [A1] decisão: mask50 interrompido, fg01 lançado (com ok do A0)
- **A1-1 mask50 parado.** Matei só as sessões tmux `disc_a1` e `watch_disc_a1`, e não sobrou processo órfão na GPU 3. O diretório parcial `.../spifil_growth/disc_a1_mask50` fica guardado. Ele não chegou a ter κ.
- **Motivo:** a máscara aleatória não ataca o defeito medido. Do cru ao treinado, a razão de densidade dentro/fora do objeto cai de 1.27× para 1.03× na L2 (e de 1.17× para 0.84× na L3). A BCE plena puxa os filtros para o fundo.
- **A1-2 fg01 lançado** no tmux `disc_a1_fg01`, com o watch em `watch_disc_a1_fg01` e o log em logs/disc_a1_fg01_tmux.log.

## 2026-10-30 04:10 [A1] base também em larvae (R1 split 1)
- A máscara sem rótulo também bate com masks/ (IoU 1.000; o objeto ocupa 1,9% da área).

  | camada | razão dentro/fora (cru → treinado) |
  |---|---|
  | L2 | 2.75× → 1.34× |
  | L3 | 2.36× → 1.31× |

- O mesmo defeito do eggs: o treino com BCE plena corta o contraste do SPiFiL pela metade.

## 2026-10-30 04:15 [A1] SVM pausado (ordem do usuário: CPU com load 245)
- Nenhum eval_svm_disc_a1_* está rodando ou encadeado. O fg01 só treina. Quando ele terminar, meço apenas a razão dentro/fora e o cosseno com o SPiFiL (os dois são só forward). O κ fica para quando o SVM voltar.

## [A5] SPiFiL cru × treinado no SVM da val (parcial, 2026-10-30)
- Resultado: o treino com o decoder acrescenta pouco na L1–L2 (+0,01 a +0,02 de κ) e bastante na L3 (+0,05). Na L3 crua o κ cai abaixo do da L2.
- Como medi: no ckpt treinado da Lk, troquei só a conv k pelos pesos do `Lk_grow`; as camadas 1..k-1 continuam as treinadas (variante a1). Rodei o SVM oficial (fit_svm + compute_metrics) na validação, com flatten. Script e linhas brutas: scratchpad `a5/raw_vs_trained.py` e `a5/results.jsonl`.

| ds | split | L | κ cru | κ treinado | treino − cru |
|---|---|---|---|---|---|
| eggs | 1 | L1 | 0,8863 | 0,8996 | +0,0133 |
| eggs | 1 | L2 | 0,8882 | 0,9069 | +0,0186 |
| eggs | 1 | L3 | 0,8571 | 0,9078 | +0,0507 |
| eggs | 2 | L1 | 0,9001 | 0,8923 | −0,0078 |

- Ordem entre os splits (só dá para ver na L1): no cru, s2 > s1; no treinado, s1 > s2. A ordem inverte, mas são só 2 pontos e uma diferença de ~0,01, então isso não serve como sinal de estabilidade.
- Sobre o achado do A0 (a L2 treinada olha o fundo): na eggs s1 L2, o cru fica 0,019 abaixo do treinado. O treino ganha um pouco, mas o SPiFiL cru já entrega quase tudo. Isso é compatível com o gradiente gastar parte dos filtros no fundo sem ganhar classe. Não prova, porque não medi a fração de ativação fora do objeto.
- Ressalva: na L3 crua, as camadas L1–L2 vêm do ckpt da L3, que pode ter derivado um pouco desde o grow (a conv1 da L2 ficou com cos 0,998 em relação à do grow).
- Parado pelo A0 por carga de CPU. Faltam eggs s2 L2–L3, eggs s3, larvae inteiro e a variante a2 (tudo cru). O script retoma de onde parou, pulando o que já está no `results.jsonl`.

## [A0] R1 protozoan split1, κ val (2026-10-30)
- L1 0.765, L2 0.833, L3 0.855. I-JEPA (teste) 0.864. É o dataset mais perto da meta (-0.009), e cada camada ainda soma κ (+0.07, +0.02), então a profundidade deve ajudar aqui.

## [A0] R1 protozoan, κ val nos 3 splits (2026-10-30)
- L1 0.765 / L2 0.825 / L3 0.848 (média; L3 por split 0.855, 0.843, 0.846). I-JEPA (teste) 0.864, faltam ~0.016. Cada camada ainda soma: candidato natural para o R2 (8 camadas).

## 2026-10-30 15:40 [A2] w64 contra R1: razão de ativação dentro/fora do objeto (eggs split1, só forward, CPU)
- Razão = (ativação média no objeto) / (ativação média no fundo). Máscara do dataset, usada só para medir; 200 imagens de validação. Script: scratchpad/bg_frac.py.

  | candidato | L2 cru | L2 treinado | L3 cru | L3 treinado |
  |---|---|---|---|---|
  | R1 24/32/48 | 1.27 | 1.03 | 1.16 | **0.84** |
  | w64 32/48/64 | 1.27 | **0.80** | 1.46 | 1.00 |

- Leitura:
  - Mais filtros deixam o SPiFiL cru da L3 mais focado no objeto (1.46 contra 1.16).
  - O treino com BCE puxa a L2 do w64 ainda mais para o fundo (0.80 contra 1.03).
  - A largura amplia o efeito do fundo: os filtros a mais acabam reconstruindo o fundo, que é ~97% da imagem.
- Lei provisória da largura: com BCE uniforme, alargar não ajuda o foco no objeto depois do treino. Largura só vale junto com `recon_bg_weight` (fg01/bp01). O κ do w64 sai pela fila de SVM do A0.
- Os pesos do w64 têm 117.744 parâmetros.
- GPU 3 com 21 MB livres às 15h: estas medidas rodaram na CPU com 4 threads.

## 2026-10-30 21:37 [A4] curvas de treino: rand × SPiFiL, R2 duplicado, efeito do fg01
Fonte: W&B `probe/val_recon_loss` e `probe/train_recon_loss_epoch` por época, 85 estágios. Script: scratchpad a4/watch_curves.py, tabela em a4/watch_3.txt.
- `e3` = época em que a val chega a 0,001 do próprio mínimo.
- **Par justo:** rand, r2, bp01 e sp200 usam `recon_bg_weight 0.1`, então se comparam com o **fg01**, não com o R1. A escala da loss muda: ~0,512 contra ~0,467.

1. **Ruído de base.** Repetir o mesmo config dá a mesma curva.
   - As 22 pastas de eggs têm dois runs, o antigo e o relançado às 18:24. O mínimo difere menos de 6e-6, e a curva no máximo 1e-4 (picos de até 4,6e-3 só nas primeiras épocas).
   - Qualquer diferença de mínimo acima de ~1e-5 é real. O SPiFiL não tem sorteio e o treino é quase determinístico.
2. **SPiFiL não é melhor inicializador que o aleatório na curva.** Eggs, 3 splits, rand × fg01, com a mesma norma por filtro:
   - **Começo (v0 na L1):** rand 0,95 / 0,95 / 0,92 × SPiFiL 0,99 / 1,38 / 1,09. O aleatório começa mais baixo e varia menos entre splits.
   - **e3:**

     | camada | rand | SPiFiL |
     |---|---|---|
     | L1 | 25 / 25 / 22 | 27 / 36 / 30 |
     | L2 | 23 / 18 / 20 | 28 / 24 / 23 |
     | L3 (parcial) | 17 / 19 / 16 | 21 / 19 / 16 |

     O aleatório estabiliza igual ou mais rápido.
   - **Mínimo:** o rand fica mais baixo, −0,9 a −1,8e-4 na L1 e 0 a −1,1e-4 na L2. Está acima do ruído.
   - **Oscilação nas últimas 20 épocas:** igual, desvio de 2–3e-6 nos dois. Nenhum pico.
   - **Leitura:** pelo critério de estabilidade do A0, a curva **não** mostra vantagem do SPiFiL. Se houver, ela tem que aparecer no κ entre seeds (rand_s1/_s2 e SVM). A loss não decide isso.
3. **R2 nas L1–L3 = fg01 repetido.**
   - O YAML só muda a partir da L4: a dilatação começa na L5 e a L4 é 48/k5/pool2.
   - As curvas do R2 e do fg01 batem dentro do ruído: L1 até 3e-5; L2 até 7e-5 (até 3,9e-3 no s1, só nas primeiras épocas).
   - A L3 do R2 e a do fg01 estão rodando ao mesmo tempo, a mesma conta duas vezes.
   - O relançamento das 18:24 também refez do zero L1–L2 do fg01, w64 e popz, que já estavam prontas (diferença de mínimo ≤ 6e-6).
4. **O fg01 deixa a convergência 2–3x mais lenta que o R1.**
   - e3: L1 27–36 contra 14–22; L2 23–28 contra 8–11; L3 16–21 contra 6–8.
   - O orçamento de 60 épocas que propus continua cobrindo (o e3 máximo é 36).
5. **O gap val − train do fg01 não é sinal.** Ele vale −0,00055 / −0,00184 / −0,00147 nos splits 1/2/3. O valor é constante entre L1, L2 e L3 e igual no rand e no r2. É efeito do split: com o fundo pesando 0,1, a loss depende da área do objeto em cada imagem. Não é sobreajuste nem subajuste.
6. **popz e w64.**
   - popz: curvas iguais às do R1, mínimo dentro de ±2e-5.
   - w64: mínimo 2 a 8e-5 abaixo do R1 nas L1–L2. É real, mas pequeno, e a loss não prevê o κ (regra A4 de antes).
7. **Saúde dos runs.**
   - 0 NaN, 0 picos e nenhum run travado: todos os que estão rodando gravaram há menos de 6 min.
   - Os "crashed" de larvae (fg01 s2 L1, w64 s1 L3, popz s2 L2) morreram no relançamento das 18:24. Não é falha do treino.

## 2026-10-30 22:19 [A4] curvas, 2ª passada: L3 de eggs fechada
- **L3, rand × SPiFiL (fg01)**, splits 1/2/3:

  | | rand | SPiFiL |
  |---|---|---|
  | mínimo da val | 0,51279 / 0,51179 / 0,51227 | 0,51282 / 0,51181 / 0,51233 |
  | e3 | 19 / 20 / 18 | 26 / 22 / 22 |

  - O rand chega 2 a 6e-5 mais baixo, acima do ruído de 6e-6, e estabiliza 2 a 7 épocas antes.
  - É o mesmo padrão da L1 e da L2: **nas 3 camadas e nos 3 splits, a curva não mostra vantagem do SPiFiL como inicializador.** Falta o κ.
- **R2 L3 = fg01 L3:** 0,51280 / 0,51182 / 0,51233 contra 0,51282 / 0,51181 / 0,51233. Confirma a duplicação até a L3.
- **R2 L4 começou.** A curva parcial é saudável: e3 23 no s1 (86 épocas), 12 e 11 nos s2 e s3 (só 34 e 26 épocas, ainda cedo). A regra de profundidade sai quando houver L5–L8 e o κ.
- **Larvae relançado de novo.** A L1 do s1 voltou a rodar do zero em fg01, w64 e popz, e já tinha 200 épocas prontas. É o mesmo desperdício do relançamento das 18:24.
- **Saúde:** 0 NaN, 0 picos, nenhum run travado.

## 2026-10-30 23:00 [A4] curvas, 3ª passada: 1º early stop (R2 L4) e larvae rand × SPiFiL
- **R2 L4, eggs s1: primeiro early stop entre todos os runs disc_*.** O mínimo da val cai na época 137, e o run para na 187 (137 + paciência 50). A lr ainda estava em 1,6e-5, ou seja, não foi o cosine chegando a zero. Depois do mínimo, a val sobe 8,7e-5.
- **A distância val − treino cresce na L4 nos 3 splits.** É o primeiro sinal de sobreajuste com a profundidade:

  | split | L1–L3 | L4 | mudança |
  |---|---|---|---|
  | s1 | −0,00055 | +0,00010 | +0,00065 |
  | s2 | −0,00180 | −0,00128 | +0,0005 (L4 parcial) |
  | s3 | −0,00143 | −0,00107 | +0,0003 (L4 parcial) |

  - Nas L1–L3 esse valor era constante, um efeito do split. Na L4 ele sai do patamar.
  - Na L4 a imagem já está reduzida a 12x12, e a camada nova tem 48 filtros com kernel 5. **Hipótese:** a partir daqui a camada nova decora o treino. O κ val da L4 contra o da L3 confirma ou não. A L5 (dilatação 2, sem pool) já começou.
- **Larvae L1, rand × SPiFiL (fg01):**

  | | rand | SPiFiL |
  |---|---|---|
  | mínimo s1 | 0,49471 | 0,49474 |
  | mínimo s2 | 0,49490 | 0,49504 |
  | e3 s1 / s2 | 28 / 27 | 34 / 31 |

  - O rand chega mais baixo e estabiliza antes, o mesmo padrão do eggs. O SPiFiL não vence na curva em nenhum dataset até agora.
- **Saúde:** 0 NaN, 0 picos, nenhum run travado.

## 2026-10-30 23:42 [A4] curvas, 4ª passada: regra de profundidade (R2 eggs, só a curva, sem κ)

| camada | época do best (s1/s2/s3) | val − treino (s1/s2/s3) | early stop |
|---|---|---|---|
| L1–L3 | 197–199 | −0,00043 / −0,00177 / −0,00139 (L3) | nunca |
| L4 (48 filtros, kernel 5, pool, 12x12) | **137 / 133 / 152** | +0,00010 / −0,00127 / −0,00097 | **s1 na 188, s2 na 184**; s3 foi até 200 |
| L5 (dilatação 2, sem pool), parcial | 108 / 89 / 66 até agora | +0,00022 / −0,00138 / −0,00114 | s1 vai parar por volta da 158 |

- **Regra de profundidade (curva):**
  - Até a L3, a val melhora até o fim do cosine.
  - **A partir da L4, a val para de melhorar antes**, por volta da época 135–150, e o early stop dispara.
  - A distância val − treino sobe +0,0004 a +0,0005 nos 3 splits. A camada nova passa a ajustar o treino mais que a val.
  - A L5 tem loss mínima maior que a L4: +8e-5 no s1, com a L5 ainda parcial nos outros splits.
- **Leitura:** pela reconstrução, o ponto de virada está entre a L3 e a L4 em eggs. Isso não prova que o κ para de subir, porque a loss não prevê o κ. Mas o primeiro lugar a medir com o SVM val é **L3 × L4 × L5**.
- **Época do best muda com a profundidade?** Sim: 197–199 (L1–L3), 133–152 (L4) e 66–108 até agora (L5).
- **Larvae L2, rand × SPiFiL (fg01, parcial):**
  - Mínimo do rand mais baixo: −5e-5 / −1,6e-4 / −2,8e-4.
  - e3: 19 / 18 / 19 no rand contra 24 / 27 / 22 no SPiFiL.
  - L1 s3: rand 0,49511 e e3 24 contra SPiFiL 0,49520 e e3 34.
  - **Até aqui são 9 de 9 estágios de larvae e eggs (L1–L3) em que o aleatório chega igual ou melhor na curva.**
- **Saúde:** 0 NaN, nenhum run travado. O w64 larvae s3 L1 começou alto (v0 2,48), mas convergiu normal (0,46490).

## 2026-10-31 00:24 [A4] curvas, 5ª passada: R2 L5 fechada; larvae L2 rand × SPiFiL fechada
- **R2 eggs L5 (dilatação 2, sem pool):** early stop nos 3 splits, na época 159 / 153 / 178, com o best na 108 / 102 / 127.
  - A distância val − treino continua subindo: +0,00018 / −0,00100 / −0,00074, contra +0,00010 / −0,00127 / −0,00097 na L4.
  - A loss mínima da L5 fica acima da L4: +8e-5 / +2e-5 / +1e-5.
  - A L6 já começou; no s1 o best está na época 99.
  - **Regra confirmada (curva):** de L4 em diante, todo estágio para cedo, cerca de 50 épocas depois de um best entre as épocas 100 e 150, e a distância val − treino cresce de uma camada para a outra. Pela reconstrução, as camadas sem pool não ajudam.
- **Larvae L2, rand × SPiFiL (fg01), fechada:**

  | | rand | SPiFiL |
  |---|---|---|
  | mínimo (s1/s2/s3) | 0,49507 / 0,49522 / 0,49547 | 0,49513 / 0,49537 / 0,49558 |
  | e3 | 19 / 18 / 19 | 24 / 27 / 27 |

  - **Placar até aqui: 15 de 15 estágios** (eggs L1–L3 e larvae L1–L2) em que o aleatório chega numa loss igual ou mais baixa e estabiliza antes.
- **Saúde:** 0 NaN, nenhum run travado.

## 2026-10-31 01:06 [A4] curvas, 6ª passada: larvae L3 empata; R2 L6 continua a regra
- **Larvae L3, rand × SPiFiL (fg01):**

  | | rand | SPiFiL |
  |---|---|---|
  | mínimo (s1/s2/s3) | 0,49566 / 0,49581 / 0,49608 | 0,49565 / 0,49597 / 0,49604 |
  | e3 | 22 / 26 / 27 | 22 / 27 / 29 |

  - **É a primeira camada em que o SPiFiL empata ou vence de leve:** no s1 por 1e-5 (perto do ruído de 6e-6) e no s3 por 4e-5. No s2, o rand vence por 1,6e-4.
  - **Placar corrigido: em 16 de 18 estágios o rand chega igual ou mais baixo. Nos 2 restantes o SPiFiL ganha por no máximo 4e-5.** Na loss, os dois inicializadores são praticamente equivalentes. O desempate tem que vir do κ.
- **R2 eggs L6:**
  - Early stop no s1 na época 150 (best na 99) e no s2 na 139 (best na 88). O s3 ainda roda, com best na 120.
  - A distância val − treino cresce de novo: +0,00035 / −0,00091 contra +0,00018 / −0,00100 na L5.
  - A loss mínima da L6 fica 2e-5 acima da L5.
- **Época do best por camada** (média dos splits, R2 eggs): L1–L3 ~198, L4 ~141, L5 ~112, L6 ~94 (dois splits). **Ela cai ~20–30 épocas por camada depois da L3.**
- **Saúde:** 0 NaN, nenhum run travado. O w64 larvae s1 L3, que tinha morrido na época 188 no relançamento, está na 193 e seguindo.

## [A0] fg01 (fundo 0.1): o foco no objeto NÃO voltou (2026-10-31)
- Razão de densidade de ativação dentro/fora (eggs s1, cru → treinado): L2 1.24 → 0.93 (no R1: 1.27 → 1.03); L3 1.20 → 0.94 (no R1: 1.16 → 0.84).
- O peso menor do fundo na loss não impede o encoder de ativar no fundo. A L2 fica até pior que no R1; a L3 melhora pouco e continua abaixo de 1.
- κ val (eggs): s1 L2 0.890 / L3 0.910 (R1: 0.907 / 0.908); s2 L2 0.903 (R1: 0.880). Empate até agora.
- Leitura: o desvio para o fundo não vem só do peso dos pixels na loss. Candidatos: o próprio fundo constante serve de "referência" para o decoder; ou o max-pool/ReLU espalham a ativação. Próximo teste, se for o caso: penalizar a ativação fora do objeto no encoder (sem rótulo de classe).

## 2026-10-31 01:48 [A4] curvas, 7ª passada: a lei de profundidade do R2 se mantém até a L7; 1 pico isolado em protozoan
- **R2 eggs, distância val − treino por camada.** Ela sobe sem parar a partir da L3, nos 3 splits:

  | split | L3 | L4 | L5 | L6 | L7 |
  |---|---|---|---|---|---|
  | s1 | −0,00043 | +0,00010 | +0,00018 | +0,00035 | **+0,00096** |
  | s2 | −0,00177 | −0,00127 | −0,00100 | −0,00091 | −0,00079 |
  | s3 | −0,00139 | −0,00097 | −0,00074 | −0,00066 | (em treino) |

  - A loss mínima sobe a cada camada: L7 contra L6, +2,3e-5 (s1) e +1,1e-5 (s2).
  - A época do best da L7 é 117 (s1) e 82 (s2): no s1 ela parou de cair.
  - O early stop dispara em todo estágio de L4 em diante, menos na L4 do s3, que chegou a 200 épocas.
  - A L8 começou nos s1 e s2.
- **Pico isolado, inofensivo: fg01 protozoan s1 L1, época 112.** A val pula de 0,5073 para 0,5161 e volta a 0,5073 na época seguinte. A train não se mexe (0,5089), e a lr está em 2,3e-4. É um transiente de uma única avaliação da val, não divergência. O early stop olha o mínimo, então não é afetado. Nenhuma ação necessária.
- **Protozoan L1, rand × SPiFiL (parcial, mesma época):**
  - s1 e s2: o SPiFiL está 3e-5 abaixo do rand na época ~110–116.
  - e3: SPiFiL 23 / 26 contra rand 20 / 19.
  - Ainda é cedo para julgar.
- **Saúde:** 0 NaN, nenhum run travado. O larvae fechou em todas as famílias; agora rodam protozoan L1 (fg01, w64, popz, rand) e o R2 eggs L7–L8.

## 2026-10-31 02:29 [A4] curvas, 8ª passada: R2 eggs L8 fechada (s1/s2) e tabela final de profundidade pela curva
- **R2 eggs, a curva de L1 a L8** (época do best | loss mínima da val | val − treino):

  | | s1 | s2 | s3 |
  |---|---|---|---|
  | L3 | 199 \| 0,51280 \| −0,00043 | 197 \| 0,51182 \| −0,00177 | 198 \| 0,51233 \| −0,00139 |
  | L4 | 137 \| 0,51347 \| +0,00010 | 133 \| 0,51243 \| −0,00127 | 152 \| 0,51284 \| −0,00097 |
  | L5 | 108 \| 0,51355 \| +0,00018 | 102 \| 0,51264 \| −0,00100 | 127 \| 0,51297 \| −0,00074 |
  | L6 | 99 \| 0,51375 \| +0,00035 | 88 \| 0,51287 \| −0,00091 | 120 \| 0,51309 \| −0,00066 |
  | L7 | 117 \| 0,51398 \| +0,00096 | 82 \| 0,51298 \| −0,00079 | 124 \| 0,51323 \| −0,00055 |
  | L8 | **74** \| 0,51419 \| +0,00077 | **74** \| 0,51311 \| −0,00038 | (em treino) |

- **Regra de profundidade (curva):**
  - A cada camada depois da L3, a loss mínima da val sobe 1–2e-5 e a distância val − treino sobe em 13 de 14 passos.
  - A época do best cai de ~198 (L1–L3) para ~100–140 (L4–L7) e para 74 (L8). A L8 para de melhorar logo depois do warmup e do começo do cosine.
  - **A partir da L4, o treino de cada camada nova só ajusta o treino. Pela reconstrução, não há ganho em ir além da L3 em eggs.**
  - O que decide é o κ val L3 × L4 × … × L8 do SVM. Sugiro priorizar L3, L4, L6 e L8 do s1.
- **Ruído da val em protozoan:** a val dá saltos isolados enquanto a lr ainda está alta, e a train fica parada.
  - fg01 s1 L1: +0,0088 na época 112.
  - w64 s2 L1: +3e-3 na época 71 e +1e-3 nas épocas 69, 73 e 83, com a train em 0,4671 o tempo todo.
  - Isso não aparece em eggs nem em larvae e some quando a lr cai. É ruído da avaliação (val pequena ou com imagens difíceis), não instabilidade. Não pede ação; o early stop usa o mínimo.
- **Protozoan L1, rand × SPiFiL (fg01), quase no fim (épocas 177–188):**
  - Mínimo: SPiFiL 0,50721 / 0,50738 contra rand 0,50725 / 0,50740 nos s1/s2. O SPiFiL fica 2–4e-5 abaixo.
  - e3: SPiFiL 26 / 28 / 27 contra rand 21 / 20 / 21. O SPiFiL estabiliza ~6 épocas depois.
  - **Protozoan é o primeiro dataset em que o SPiFiL chega mais baixo,** mas estabiliza mais tarde. Fecho na próxima passada.
- **R2 começou larvae L1.** Ele repete o fg01 até a L3 (mesmo YAML), de novo conta repetida, como em eggs.
- **Saúde:** 0 NaN, nenhum run travado.

## 2026-10-31 03:11 [A4] curvas, 9ª passada: placar final da L1, rand × SPiFiL; profundidade do R2 eggs fechada
- **Protozoan L1, rand × SPiFiL (fg01), fechada:**

  | | SPiFiL | rand |
  |---|---|---|
  | mínimo (s1/s2/s3) | 0,50721 / 0,50737 / 0,50758 | 0,50724 / 0,50739 / 0,50758 |
  | e3 | 26 / 28 / 27 | 21 / 20 / 21 |

  - O SPiFiL fica 3e-5 e 2e-5 abaixo nos s1 e s2 e empata no s3, mas estabiliza 5–8 épocas depois.
- **Placar de 21 estágios** (eggs L1–L3, larvae L1–L3, protozoan L1):

  | quem chega mais baixo na loss | estágios |
  |---|---|
  | rand | 16 |
  | SPiFiL | 4 (por no máximo 4e-5: larvae L3 s1/s3, protozoan L1 s1/s2) |
  | empate | 1 |

  - **Estabilizar mais rápido:** o rand vence em 20 de 21; o único empate é larvae L3 s1, com 22 contra 22.
  - **Conclusão pela curva:** o SPiFiL não é inicializador melhor que o aleatório com a mesma norma. Nem em velocidade nem em oscilação: o desvio no fim é o mesmo, 1–3e-6. Na loss mínima, os dois são equivalentes. Se o SPiFiL tem vantagem, ela só pode estar no **κ** (entre seeds e splits), não na otimização.
- **R2 eggs s3 L8 fechada:** best na época 81, parada na 132. A loss sobe +1,7e-5 sobre a L7, e a distância val − treino passa de −0,00055 para −0,00035.
  - **Na tabela completa, a distância val − treino sobe em 14 de 15 passos de camada depois da L3.** A única exceção é s1 L7 → L8.
  - A época do best da L8 é 74 / 74 / 81.
- **R2 larvae s1 L1 = fg01 larvae s1 L1:** 0,49474 nos dois. Confirma a duplicação também em larvae.
- **Saúde:** 0 NaN, nenhum run travado. Agora rodam protozoan L2 (fg01, popz, rand), w64 protozoan L1 e R2 larvae L1–L2.

## 2026-10-31 03:52 [A4] curvas, 10ª passada (nota curta)
- **Protozoan L2, rand × SPiFiL (fg01), parcial na época ~110–136:**
  - Mínimo: rand 0,50757 / 0,50770 / 0,50794 contra SPiFiL 0,50758 / 0,50776 / 0,50801 (o s3 ainda está em épocas diferentes).
  - e3: rand 19 / 16 / 13 contra SPiFiL 21 / 19 / 19.
  - O padrão de antes se mantém: o rand estabiliza antes.
- **R2 larvae s1 L2 = fg01:** 0,49513 nos dois. Mais uma conta repetida.
- **Saúde:** 0 NaN, nenhum run travado, e nenhum salto da val em protozoan nesta janela.

## 2026-10-31 04:39 [A4] curvas, 11ª passada: protozoan L2 fechada; o R2 larvae L4 repete a assinatura de eggs
- **Protozoan L2, rand × SPiFiL (fg01):**

  | | rand | SPiFiL |
  |---|---|---|
  | mínimo (s1/s2/s3) | 0,50751 / 0,50763 / 0,50782 (s3 na época 197) | 0,50752 / 0,50770 / 0,50792 |
  | e3 | 19 / 20 / 17 | 22 / 20 / 23 |

  - O rand fica mais baixo nos 3 splits, por 1e-5, 7e-5 e 1e-4.
  - **Placar em 24 estágios:**
    - Loss mais baixa: rand em 19, SPiFiL em 4 (no máximo 4e-5), 1 empate.
    - Estabiliza antes: rand em 22, 2 empates. O SPiFiL não foi mais rápido em nenhum.
- **R2 larvae s1 L4, parcial (50 épocas):** a distância val − treino salta de +0,00005 (L3) para **+0,00052**. É o mesmo salto de ~+0,0005 da L4 em eggs.
  - A L3 do R2 larvae s1 é igual à do fg01: 0,49566 contra 0,49565.
  - **A regra "a camada nova passa a ajustar só o treino a partir da L4" começa a se repetir em larvae.** Confirmo quando a L4 fechar.
- **Saúde:** 0 NaN, nenhum run travado. Rodam as L3 de protozoan em fg01, popz e rand, a L2 do w64 e as L2–L4 do R2 larvae.

## 2026-10-31 05:25 [A4] curvas, 12ª passada: a regra de profundidade se repete em larvae, e mais forte
- **R2 larvae, L3 contra L4** (época do best | loss mínima da val | val − treino):

  | split | L3 | L4 | early stop |
  |---|---|---|---|
  | s1 | 175 \| 0,49566 \| +0,00005 | **128** \| 0,49670 \| **+0,00141** | na época 179 |
  | s2 | 181 \| 0,49596 \| −0,00019 | **109** \| 0,49718 \| **+0,00148** | na época 160 |

  - Da L3 para a L4, a distância val − treino sobe **+0,0014 a +0,0017**, cerca de 3x o salto de eggs (+0,0004 a +0,0005).
  - A loss mínima sobe ~1e-3.
  - A L5 do s1 (parcial, 75 épocas) tem a distância em +0,00088 e best na época 57.
  - **Regra (curva, 2 datasets):** a partir da L4, a camada nova ajusta o treino e não a val. Em larvae isso é mais forte que em eggs. **Hipótese:** o larvae tem menos imagens por classe, então a camada nova (48 filtros, kernel 5, ~58k parâmetros nesta camada) tem mais folga para decorar.
  - O κ val L3 × L4 do larvae é o teste mais barato para saber se esse sobreajuste custa κ.
- **Protozoan L3, rand × SPiFiL (fg01), parcial na época ~104–115:**
  - Mínimo: rand 0,50803 / 0,50818 contra SPiFiL 0,50798 / 0,50815. O SPiFiL está 3–5e-5 abaixo, mas na época 115 contra 106 no s1.
  - e3: rand 13 / 16 contra SPiFiL 15 / 16.
  - Ainda empatado; espero fechar.
- **Saúde:** 0 NaN, nenhum run travado.

## 2026-10-31 06:13 [A4] curva × κ val: a profundidade custa κ a partir da L4, e o piso de ruído do κ é ~0,01
Fontes: results/svm_disc_r2_val.csv e svm_disc_a1_fg01_val.csv (só leitura) e as curvas do W&B.

1. **Ruído do κ entre re-runs do mesmo config.**
   - Comparei fg01 × R2 nas L1–L3: o mesmo YAML, a mesma época do ckpt e curvas iguais até 1e-5.

     | estágio | Δκ (R2 − fg01) |
     |---|---|
     | L1, s1/s2/s3 | 0,0000 / 0,0000 / 0,0000 (pesos idênticos) |
     | L2, s1/s2/s3 | **+0,0085** / −0,0013 / −0,0002 |
     | L3, s1/s2 | **−0,0114** / −0,0001 |

   - **Regra:** do L2 em diante, a GPU não é bit-determinística. Uma diferença de pesos que nem aparece na loss já move o κ val em até ±0,011.
   - **Uma diferença de κ abaixo de ~0,012 entre candidatos, num único split, é ruído.** Vale para A1, A2, A3 e A0. Para afirmar um ganho, peça média dos 3 splits e uma diferença maior que ~0,01.
2. **Profundidade: o κ val do R2 eggs bate o pico na L2–L3 e cai a partir da L4**, como a curva tinha antecipado.

   | camada | κ s1 | κ s2 | época do best (s1/s2) |
   |---|---|---|---|
   | L1 | 0,888 | 0,899 | 199 / 197 |
   | L2 | 0,899 | 0,901 | 198 / 197 |
   | L3 | 0,899 | 0,898 | 199 / 197 |
   | L4 | 0,872 | 0,891 | 137 / 133 |
   | L5 | 0,863 | 0,862 | 108 / 102 |
   | L6 | 0,877 | 0,836 | 99 / 88 |
   | L7 | 0,839 | 0,845 | 117 / 82 |
   | L8 | **0,808** | **0,820** | 74 / 74 |

   - De L3 para L4, o κ cai 0,027 (s1) e 0,007 (s2). De L3 para L8, cai 0,09 e 0,08.
   - **Regra barata, sem SVM:** Spearman(época do best, κ) = **+0,85** nas L3–L8 (n = 12). **Se o early stop disparou, ou seja, se o best saiu antes da época ~150, a camada nova piorou o κ.** Isso valeu em todos os 10 estágios L4–L8 com κ.
   - Dentro de cada split, a distância val − treino também cai junto com o κ. Juntando os splits, a correlação é só −0,41, porque cada split tem o seu próprio nível de base.
   - **Ressalva:** o flatten encolhe de 27.648 dimensões (L3) para 5.808 (L4) e 7.744 (L5–L8). Parte da queda pode vir de mapa pequeno (11x11), não só de sobreajuste. Para separar as duas causas, seria preciso o κ de uma L4 sem pool.
3. **Consequência para o R2:** com a configuração atual (48–64 filtros e kernel 5/3 depois da L3), **ir além da L3 não ajuda em eggs.** Em larvae, a curva da L4 é ainda pior (val − treino +0,0014 a +0,0017). A previsão é a mesma queda de κ; ela pede confirmação.
4. **Protozoan L3, rand × SPiFiL (fg01):**
   - Mínimo: rand 0,50788 / 0,50803 contra SPiFiL 0,50790 / 0,50806. O rand fica 2–3e-5 abaixo.
   - e3: rand 18 / 17 / 18 contra SPiFiL 17 / 16 / 17. **É a primeira camada em que o SPiFiL estabiliza antes, por 1 época,** o que está dentro do ruído.
   - **Placar em 26 estágios:** loss mais baixa, rand 21, SPiFiL 4, 1 empate. Estabiliza antes: rand 22, SPiFiL 3 (por 1 época), 1 empate.
5. **Saúde:** 0 NaN, nenhum run travado. O fg01 terminou (a sessão tmux fechou). Os picos marcados no R2 larvae L5–L7 são o mesmo ruído de avaliação de antes, com a train estável.

## 2026-10-31 06:59 [A4] curvas, 14ª passada: o κ da profundidade fecha em 3/3 splits; sp200 e rand_s1 entram
- **R2 eggs s3, κ val:**

  | L2 | L3 | L4 | L5 | L6 | L7 | L8 |
  |---|---|---|---|---|---|---|
  | 0,892 | 0,887 | **0,856** | 0,844 | 0,848 | 0,829 | 0,823 |

  - A época do best vai 199 → 198 → 152 → 127 → 120 → 124 → 81.
  - **Nos 3 splits, o κ é máximo na L2–L3 e cai a partir da L4.** Em todos os 15 estágios L4–L8, o early stop disparou (ou o best saiu antes da época ~155) e o κ ficou abaixo da L3 do mesmo split. A regra "early stop disparou ⇒ a camada piorou o κ" fecha 15 de 15.
- **O R2 larvae parou:** L7 s1, L6 s2 e L4 s3 ficaram como "crashed" ao mesmo tempo, há ~0,8 h. Imagino que o A0 tenha parado depois da regra de profundidade; não é falha do treino.
- **Popz × R1, κ val s1:** L2 0,9127 contra 0,9068 (+0,006); L3 0,9041 contra 0,9077 (−0,004). **As duas diferenças estão dentro do ruído de re-run (~0,012).**
- **sp200 (A3, fundo 0.1) × fg01, L1:**
  - Mínimo: 0,51216 / 0,51112 / 0,51163 contra 0,51211 / 0,51117 / 0,51171, ou seja, ±5e-5 sem direção.
  - e3: 28 / 32 / 27 contra 27 / 36 / 30.
  - Na curva, o sp200 é indistinguível do fg01.
- **Seed do aleatório (rand_s1 × rand), larvae s1 L1:** mínimo 0,49467 contra 0,49471 e e3 32 contra 28. É a variação de seed numa camada só.
  - A diferença de 4e-5 entre seeds do aleatório é do mesmo tamanho da diferença rand × SPiFiL. **Isso reforça que, na curva, os dois inicializadores são indistinguíveis.**
- **Saúde:** 0 NaN, nenhum run travado entre os ativos (w64 protozoan L3, sp200 eggs L2, rand_s1).

## 2026-10-31 07:46 [A4] curvas, 15ª passada: seeds do aleatório × SPiFiL na curva; bp01 L1 = fg01 L1
- O R2 saiu da vigia (o A0 o parou de propósito).
- **Seeds do aleatório × SPiFiL, loss mínima da val no split 1:**

  | estágio | rand (seed 0) | rand_s1 | SPiFiL (fg01) | faixa entre seeds | onde fica o SPiFiL |
  |---|---|---|---|---|---|
  | eggs L1 | 0,51202 | 0,51192 | 0,51211 | 1,0e-4 | 9e-5 acima do melhor seed |
  | larvae L2 | 0,49507 | 0,49516 | 0,49513 | 9e-5 | **dentro da faixa** |
  | protozoan L1 | 0,50724 | 0,50721 | 0,50721 | 3e-5 | **empata com o melhor seed** |

  - e3 em eggs L1: 25 / 26 (seeds) contra 27 (SPiFiL).
  - **Na curva, trocar a seed do aleatório move a loss tanto quanto trocar aleatório por SPiFiL.** As diferenças rand × SPiFiL do placar de antes (≤ 1e-4) estão dentro da variação entre seeds. Não há sinal de estabilidade do SPiFiL na otimização.
- **κ por seed:** o script está pronto (scratchpad a4/kappa_seeds.py). Ele compara o desvio do κ entre as seeds rand/s1/s2 com o desvio entre splits e entre re-runs do SPiFiL (o R2 nas L1–L3 serve de repetição do fg01).
  - Do lado SPiFiL, o desvio por **re-run** no s1 é de 0,0085 (L2) e 0,0114 (L3); no s2 e no s3, ≤ 0,0013.
  - Faltam os CSVs `svm_disc_r1_rand*_val.csv`.
- **bp01 L1 = fg01 L1:** 0,51213 / 0,51118 / 0,51171 contra 0,51211 / 0,51117 / 0,51171, com e3 igual. O `layerwise_backprop` não age na L1, porque não há camada antiga para treinar. Estes 3 estágios são repetição.
  - **Regra:** uma variante que só muda as camadas ≥ 2 pode começar do ckpt L1 do fg01.
- **Saúde:** 0 NaN, nenhum run travado. Rodam bp01 L1, sp200 L3 e rand_s1 (eggs L2, larvae L3, protozoan L2).

## 2026-10-31 08:32 [A4] curvas, 16ª passada (nota curta; o κ do rand ainda não saiu)
- **Seeds do aleatório × SPiFiL na loss mínima, split 1:**

  | estágio | rand | rand_s1 | SPiFiL | posição do SPiFiL |
  |---|---|---|---|---|
  | eggs L2 | 0,51238 | 0,51234 | 0,51239 | 1e-5 acima da faixa (≈ ruído) |
  | eggs L3 | 0,51279 | 0,51285 | 0,51282 | **no meio da faixa** |
  | larvae L3 | 0,49566 | 0,49571 | 0,49565 | na borda de baixo (melhor por 1e-5) |

  - Somando com a passada anterior: em 6 estágios com duas seeds, o SPiFiL fica dentro da faixa ou a até 1e-5 dela em 5. A exceção é eggs L1, 9e-5 acima.
- **bp01 L2 (A2, camadas antigas com lr 0,1×) × fg01:** s2 0,51141 contra 0,51143; s1 0,51239 contra 0,51239 (ainda parcial). O backprop leve não muda a loss. Se tem efeito, só o κ mostra.
- **sp200 L3 × fg01:** 0,51288 / 0,51179 / 0,51233 contra 0,51282 / 0,51181 / 0,51232. Iguais dentro de ±6e-5.
- **Saúde:** 0 NaN, nenhum run travado.

## 2026-10-31 09:19 [A4] curvas, 17ª passada (nota curta)
- **bp01 L3 × fg01 L3** (A2: L1–L2 treinando com lr 0,1× junto com a L3):
  - Mínimo: s1 0,51276 contra 0,51282 (−6e-5); s2 0,51177 contra 0,51181 (−4e-5).
  - Época do best no s1: 189 contra 197.
  - É o primeiro efeito do bp01 na curva. Vai na mesma direção nos dois splits e fica acima do ruído de re-run (6e-6), **mas dentro da faixa entre seeds do aleatório (6e-5).**
  - Na L2, o efeito não aparece (0,51238 contra 0,51239). Só o κ decide.
- **sp200 × fg01 em larvae:**
  - L1: 0,49471 / 0,49486 / 0,49524 contra 0,49474 / 0,49504 / 0,49520.
  - L2: 0,49517 / 0,49533 contra 0,49513 / 0,49537.
  - Diferenças de ±2e-4 sem direção. Na curva, os dois são iguais.
- **Protozoan s1 L2:** rand_s1 0,50748, rand 0,50751 e SPiFiL 0,50752. O SPiFiL fica 1e-5 acima da faixa entre seeds.
- **Placar seeds × SPiFiL:** em 7 estágios com 2 seeds, o SPiFiL fica dentro da faixa, ou a até 1e-5 dela, em 6.
- **Saúde:** 0 NaN, nenhum run travado. Seguem rodando disc_a2 (bp01), disc_a3 (sp200) e disc_rand (s1). Ainda não há `svm_disc_r1_rand*_val.csv`.

## 2026-10-31 10:05 [A4] curvas, 18ª passada (nota curta)
- **bp01 L3 × fg01 L3 em eggs, 3 de 3 splits:** 0,51276 / 0,51177 / 0,51225 contra 0,51282 / 0,51181 / 0,51232, ou seja, **−6, −4 e −7e-5**.
  - É o único eixo desta rodada com efeito de direção consistente na loss. Ele aparece só na L3: na L2 a diferença é ≤1e-5, em eggs e em larvae.
  - O tamanho fica perto da faixa entre seeds do aleatório (6e-5). Ainda falta o κ para dizer se vira ganho.
- **bp01 larvae L1 = fg01 larvae L1:** 0,49474 / 0,49504 / 0,49520 nos dois. Mais 3 estágios repetidos (a regra "L1 de variante que só muda as camadas ≥2 = ckpt do fg01").
- **sp200 larvae L3 × fg01:** 0,49572 / 0,49592 / 0,49604 contra 0,49565 / 0,49597 / 0,49604. Iguais.
- **Saúde:** 0 NaN, nenhum run travado. Ainda não há `svm_disc_r1_rand*_val.csv`.

## 2026-10-31 10:51 [A4] curvas, 19ª passada (nota curta)
- **bp01 L3 × fg01 L3, agora com larvae:** s1 0,49562 contra 0,49565 (−3e-5) e s2 0,49587 contra 0,49597 (−1e-4).
  - **5 de 5 splits com o bp01 L3 abaixo** (eggs −4 a −7e-5, larvae −3e-5 a −1e-4). Na L2, a diferença continua ≤2e-5.
  - Leitura: deixar L1–L2 se ajustarem junto com a L3 (lr 0,1×) baixa a loss de reconstrução da L3 de forma consistente. Vale o κ do bp01 L3 contra o fg01 L3.
- **3 seeds do aleatório, larvae s1 L1:** rand / s1 / s2 = 0,49471 / 0,49467 / 0,49469 (faixa de 4e-5). O SPiFiL dá 0,49474, 3e-5 acima da faixa.
  - e3: 28 / 32 / 37 contra 34 no SPiFiL, dentro da faixa.
  - **Com 3 seeds, a época de estabilizar varia 9 épocas entre seeds.** Logo, a diferença de "velocidade" rand × SPiFiL do placar (≤8 épocas) também cabe na variação entre seeds.
- **Protozoan s1 L3:** rand e rand_s1 dão 0,50788 os dois; o SPiFiL dá 0,50790 (+2e-5).
- **sp200 protozoan L1 × fg01:** 0,50720 / 0,50739 contra 0,50721 / 0,50737. Iguais.
- **Saúde:** 0 NaN, nenhum run travado. Ainda não há `svm_disc_r1_rand*_val.csv`.

## 2026-10-31 11:38 [A4] curvas, 20ª passada: 3 seeds do aleatório × SPiFiL no split 1 (loss mínima | e3)

| estágio | rand | rand_s1 | rand_s2 | SPiFiL | SPiFiL em relação à faixa |
|---|---|---|---|---|---|
| eggs L1 | 0,51202 \| 25 | 0,51192 \| 26 | 0,51191 \| 25 | 0,51211 \| 27 | 9e-5 acima, e3 +1 |
| larvae L1 | 0,49471 \| 28 | 0,49467 \| 32 | 0,49469 \| 37 | 0,49474 \| 34 | 3e-5 acima, e3 dentro |
| larvae L2 | 0,49507 \| 19 | 0,49516 \| 24 | 0,49505 \| 16 | 0,49513 \| 24 | **dentro, dentro** |
| protozoan L1 | 0,50724 \| 21 | 0,50721 \| 27 | 0,50719 \| 28 | 0,50721 \| 26 | **dentro, dentro** |

- **Leitura:** o desvio entre seeds do aleatório vai de 2e-5 a 6e-5 na loss e de 2 a 9 épocas no e3. O SPiFiL cai dentro da faixa das 3 seeds em 2 de 4 estágios, e logo acima nos outros 2. **Em nenhum estágio o SPiFiL fica melhor que a melhor seed.** Na curva, ele se comporta como "mais uma seed do aleatório" ou um pouco pior.
- **bp01 larvae s3 L3:** 0,49602 contra 0,49604 no fg01. **São 6 de 6 splits com o bp01 L3 abaixo do fg01 L3.**
- **bp01 protozoan L1 = fg01:** 0,50721 / 0,50739 contra 0,50721 / 0,50737, de novo repetido.
- **Saúde:** 0 NaN, nenhum run travado. Ainda não há κ do rand.

## 2026-10-31 12:24 [A4] curvas, 21ª passada (nota curta)
- **Mais dois estágios com 3 seeds do aleatório, split 1:**

  | estágio | rand / s1 / s2 | SPiFiL | posição do SPiFiL |
  |---|---|---|---|
  | eggs L2 | 0,51238 / 0,51234 / 0,51235 | 0,51239 | 1e-5 acima (≈ ruído) |
  | larvae L3 | 0,49566 / 0,49571 / 0,49567 | 0,49565 | **1e-5 abaixo**; e3 22 contra 22 / 24 / 20 |

  - **Placar com 3 seeds em 6 estágios:** o SPiFiL fica abaixo da faixa em 1 (por 1e-5), dentro em 2 e acima em 3 (por 1e-5, 3e-5 e 9e-5).
  - É o retrato de uma seed qualquer. Não há vantagem sistemática.
- **sp200 protozoan L2 × fg01:** 0,50755 / 0,50772 contra 0,50752 / 0,50770. Iguais.
- **Saúde:** 0 NaN, nenhum run travado. Ainda não há κ do rand.

## 2026-10-31 12:56 [A4] κ val, 1º resultado: rand (seed 0) × SPiFiL na L3 de eggs
Fontes: results/svm_disc_r1_rand_val.csv (3 linhas), svm_disc_a1_fg01_val.csv e svm_disc_r2_val.csv (só leitura). Script: scratchpad a4/kappa_seeds.py.

| eggs L3 | s1 | s2 | s3 | média | desvio entre splits |
|---|---|---|---|---|---|
| **SPiFiL (fg01, ckpt atual)** | 0,8938 | 0,8983 | 0,8868 | **0,8930** | **0,0058** |
| SPiFiL, repetições do mesmo config | 0,9100 (fg01 antigo) e 0,8986 (R2) | 0,8982 (R2) | 0,8865 (R2) | — | — |
| **rand seed 0** | 0,8979 | **0,9200** | **0,9078** | **0,9086** | **0,0111** |
| rand − SPiFiL | +0,004 | **+0,022** | **+0,021** | **+0,016** | — |

1. **Ruído por repetição do SPiFiL** (o mesmo YAML treinado de novo):
   - No s1 há 3 repetições: 0,9100 / 0,8938 / 0,8986, com faixa de **0,016** e desvio de 0,008.
   - No s2 e no s3 há 2 repetições cada, com faixa de ≤0,0003.
   - **O s1 é instável; o s2 e o s3, não.** A linha 0,9100 que usei antes vinha do run anterior ao relançamento das 18:24. O ckpt atual dá 0,8938.
2. **Média:** o rand seed 0 fica +0,016 acima do SPiFiL.
   - No s1, a diferença (+0,004) está dentro do ruído do s1.
   - **No s2 e no s3, a diferença (+0,021 a +0,022) é ~70x maior que o ruído por repetição desses splits.** Ali o aleatório é melhor de verdade, pelo menos com esta seed.
3. **Estabilidade entre splits (critério do A0):** o desvio do SPiFiL (0,0058) é menor que o do rand (0,0111).
   - É o primeiro ponto a favor do SPiFiL. Mas é estabilidade **em torno de uma média mais baixa**. Com n = 3 splits, a diferença de desvios também não é significativa.
   - Para fechar o critério, faltam s1 e s2 do rand: o desvio **entre seeds** contra o desvio entre repetições do SPiFiL.
4. **Leitura provisória:** na L3 de eggs, o aleatório com a mesma norma dá κ igual ou maior que o SPiFiL. O SPiFiL só vence em "variar menos entre splits", e ainda sem significância.

## 2026-10-31 13:02 [A4] κ val, 2º resultado: em larvae L3, o rand oscila e o SPiFiL fica estável

| larvae L3 | s1 | s2 | s3 | desvio entre splits |
|---|---|---|---|---|
| **SPiFiL (fg01)** | 0,8387 | 0,8305 | 0,8383 | **0,0046** |
| **rand seed 0** | 0,8477 | **0,7867** | (pendente) | 0,043 (s1–s2) |
| rand − SPiFiL | +0,009 | **−0,044** | — | — |

- **Em larvae, o κ do rand cai 0,061 entre s1 e s2. No SPiFiL, os 3 splits ficam dentro de 0,008.** É o primeiro sinal do critério do A0 (estabilidade contra o controle aleatório) a favor do SPiFiL, e com tamanho grande.
- **A loss não vê isso:** no s2, o rand tem loss **mais baixa** (0,49581 contra 0,49597) e κ 0,044 **pior**. Mais um caso da regra "a loss não prevê o κ".
- **Placar parcial de κ (L3), rand seed 0 × SPiFiL:**
  - eggs: rand +0,016 na média, mas oscila mais (desvio 0,011 contra 0,006).
  - larvae: rand −0,018 na média dos 2 splits, e oscila muito mais (0,043 contra 0,005).
  - **Padrão: o SPiFiL troca um pouco de média por bem menos variância entre splits.** Para confirmar que é efeito do SPiFiL, e não sorte da seed 0, faltam o rand larvae s3 e as seeds s1/s2.
- **Pendente:** os CSVs `svm_disc_r1_rand_s1_val.csv` e `_s2_val.csv` ainda não existem.

## 2026-10-31 13:08 [A4] κ val L3, rand seed 0 × SPiFiL (fg01): o SPiFiL vence em larvae e protozoan e é mais estável

| L3 | SPiFiL s1 / s2 / s3 | rand s1 / s2 / s3 | média SPiFiL | média rand | desvio SPiFiL | desvio rand |
|---|---|---|---|---|---|---|
| eggs | 0,894 / 0,898 / 0,887 | 0,898 / **0,920** / **0,908** | 0,893 | **0,909** | 0,006 | 0,011 |
| larvae | 0,839 / 0,831 / 0,838 | 0,848 / **0,787** / **0,791** | **0,836** | 0,808 | **0,005** | 0,034 |
| protozoan | **0,877** / 0,836 / 0,853 | **0,815** / – / – | 0,855 | – | 0,021 | – |

- **rand − SPiFiL por split** (7 pares): eggs +0,004 / +0,022 / +0,021; larvae +0,009 / **−0,044** / **−0,048**; protozoan s1 **−0,062**.
  - Média: −0,014. **O SPiFiL vence 3 de 7 por mais de 0,04, e o rand vence 2 de 7 por ~0,02.** As outras 2 diferenças estão dentro do ruído.
- **Estabilidade (critério do A0):** o desvio entre splits do SPiFiL é menor nos 2 datasets completos: eggs 0,006 contra 0,011 e larvae 0,005 contra 0,034.
  - **As perdas do rand são grandes e esporádicas (−0,04 a −0,06). As do SPiFiL são pequenas (−0,02).** É o perfil de um inicializador mais estável.
- **A loss não vê nada disso:** em larvae s2/s3 e em protozoan s1, a loss de reconstrução do rand é igual ou mais baixa e o κ cai 0,04 a 0,06.
  - **Regra:** a curva de treino serve para saúde e para orçamento de épocas, nunca para julgar inicializador.
- **Ainda é seed 0 do rand.** Para dizer se o aleatório é "às vezes ruim" (variância entre seeds) ou se esta seed teve azar, faltam rand_s1 e rand_s2 no SVM val, pelo menos em larvae L3 s2/s3 e protozoan L3 s1.

## 2026-10-31 13:15 [A4] κ val L3: protozoan s2 entra (nota curta)
- **Protozoan s2 L3:** rand seed 0 dá 0,8019 contra SPiFiL 0,8357, ou seja, **−0,034**. No protozoan s1 tinha sido −0,062.
- **rand − SPiFiL nos 8 pares:** eggs +0,004 / +0,022 / +0,021; larvae +0,009 / −0,044 / −0,048; protozoan −0,062 / −0,034.
  - **O SPiFiL vence 4 de 8 por 0,03 a 0,06. O rand vence 2 de 8 por ~0,02, os dois em eggs.** Os 2 pares restantes estão dentro do ruído.
  - Média das diferenças: −0,017 a favor do SPiFiL.
- **Leitura por dataset:** em eggs, o aleatório se sai um pouco melhor. **Em larvae e protozoan, o SPiFiL é claramente melhor e mais estável.** É coerente com os dois serem os datasets mais difíceis: κ mais baixo e menos imagens por classe em larvae.
- Ainda é seed 0. Falta rand_s1/s2 para separar efeito do SPiFiL de azar de seed.

## 2026-10-31 13:26 [A4] κ val L3: protozoan fecha; as seeds do rand só cobrem o split 1
- **Protozoan s3 L3:** rand seed 0 dá 0,7969 contra SPiFiL 0,8526, ou seja, **−0,056**.

  | protozoan L3 | s1 / s2 / s3 | média | desvio entre splits |
  |---|---|---|---|
  | SPiFiL | 0,877 / 0,836 / 0,853 | **0,855** | 0,021 |
  | rand seed 0 | 0,815 / 0,802 / 0,797 | 0,805 | **0,009** |

  - **Aqui o SPiFiL ganha +0,050 na média, mas varia mais entre splits.** Então "o SPiFiL é mais estável entre splits" vale em eggs e larvae, **não em protozoan**. O que se repete em larvae e protozoan é a **média** maior.
- **rand − SPiFiL nos 9 pares da L3:** eggs +0,004 / +0,022 / +0,021; larvae +0,009 / −0,044 / −0,048; protozoan −0,062 / −0,034 / −0,056.
  - **O SPiFiL vence 5 de 9 por 0,03 a 0,06,** todos em larvae e protozoan. O rand vence 2 de 9 por ~0,02, em eggs. Média: −0,021.
- **Primeiras seeds extras, só no split 1:**

  | split 1 L3 | rand seed 0 | rand_s1 | SPiFiL (repetições) |
  |---|---|---|---|
  | eggs | 0,8979 | 0,9105 | 0,8938 / 0,9100 / 0,8986 |
  | larvae | 0,8477 | 0,8589 | 0,8387 |

  - A variação entre seeds no s1 (0,011–0,013) é do mesmo tamanho da variação por repetição do SPiFiL (até 0,016).
  - **Em larvae s1, as duas seeds do rand passam o SPiFiL** (+0,009 e +0,020).
- **Limitação de desenho:** `growth_disc_r1_rand_s1.yaml` e `_s2.yaml` têm `splits: [1]`.
  - As perdas grandes do rand (larvae s2/s3 e protozoan s1/s2/s3) estão em splits que as seeds extras não cobrem. No split 1 de larvae, o rand nunca perdeu.
  - **Com o desenho atual, não dá para separar "o aleatório é instável" de "a seed 0 teve azar no s2/s3".**
  - O teste que decide: rand_s1/s2 em larvae s2/s3 e protozoan s2/s3, só até a L3. São 4 células × 3 estágios por seed. A decisão é do A0.

## 2026-10-31 13:33 [A4] κ val L3: no protozoan s1, a vitória do SPiFiL resiste a 2 seeds do rand
- **Protozoan s1 L3:**
  - rand seed 0 dá 0,8150 e rand_s1 dá 0,7944: a faixa entre seeds é de **0,021**.
  - O SPiFiL dá **0,8771**, à frente das duas seeds por **+0,062 e +0,083**, ou seja, 3–4x a variação entre seeds.
  - **É o primeiro ponto em que a vantagem do SPiFiL passa no teste de seed. Não é azar da seed 0.**
- **Correção da nota anterior:** o protozoan s1 também era um dos pontos com perda grande do rand, e ele **está** coberto pelas seeds extras. Os que ficam de fora são larvae s2/s3 e protozoan s2/s3.
- **Split 1, com 2 seeds do rand por dataset (L3):**

  | dataset | rand seed 0 / s1 | SPiFiL | leitura |
  |---|---|---|---|
  | eggs | 0,898 / 0,911 | 0,894 (repetições 0,894–0,910) | empate: cabe no ruído de repetição |
  | larvae | 0,848 / 0,859 | 0,839 | **rand melhor** nas 2 seeds (+0,009 / +0,020) |
  | protozoan | 0,815 / 0,794 | **0,877** | **SPiFiL melhor** nas 2 seeds (+0,062 / +0,083) |

- **Leitura:** o efeito do SPiFiL depende do dataset.
  - Em protozoan, ele é forte e resiste à troca de seed.
  - Em eggs, empata.
  - Em larvae, o resultado depende do split: o rand ganha no s1 e perde no s2/s3 com a seed 0. Para fechar larvae, falta seed extra no s2/s3.

## 2026-10-31 13:45 [A4] checagem dos CSVs de SVM: κ idênticos entre ckpts diferentes não são bug; w64 L3 no κ
- **Verificação:** procurei métricas (κ, acc, f1) idênticas entre ckpts diferentes em todos os `svm_disc_*val*.csv`. Achei 7 grupos em 103:
  - **3 esperados:** fg01 L1 = R2 L1, com pesos idênticos (s1, s2, s3).
  - **4 em larvae, entre modelos diferentes:**
    - rand_s1 s1 L3 = w64 s1 L3, com κ 0,85890 e dims diferentes;
    - R1 s1 L1 = popz s2 L2;
    - R1 s3 L1 = R1 s3 L2;
    - fg01 s2 L3 = popz s3 L3.
  - **Explicação:** larvae é **binário**. A val tem 1.757 imagens (223 contra 1.534), a mesma composição nos 3 splits. Com ~110 erros, κ, acc e f1 dependem só do par (FP, FN), então coincidências exatas são esperadas entre ~800 pares de linhas.
  - **Não é cache nem ckpt trocado.** Consequência prática: em larvae, **uma imagem a mais ou a menos move o κ em ~0,003–0,005**. Esse é o grão mínimo de diferença.
- **w64 × R1, κ val L3** (os dois com BCE uniforme):
  - eggs: 0,9126 / 0,8893 / 0,8849 contra 0,9077 / 0,8877 / 0,8957, ou seja, +0,005 / +0,002 / −0,011 (média −0,001).
  - larvae s1: 0,8589 contra 0,8466 (+0,012).
  - **Tudo dentro do ruído de repetição (~0,012).** Mais filtros não muda o κ da L3.

## 2026-10-31 13:51 [A4] κ val larvae L3: o s2/s3 são splits difíceis, e só o SPiFiL + fundo 0,1 não cai neles

| larvae L3 | init | BCE | s1 | s2 | s3 | queda s1 → média(s2, s3) |
|---|---|---|---|---|---|---|
| R1 | SPiFiL | uniforme | 0,847 | 0,816 | 0,826 | −0,026 |
| w64 | SPiFiL (32/48/64) | uniforme | 0,859 | 0,804 | 0,807 | −0,054 |
| rand seed 0 | aleatório | fundo 0,1 | 0,848 | 0,787 | 0,791 | −0,059 |
| **fg01** | **SPiFiL** | **fundo 0,1** | 0,839 | **0,831** | **0,838** | **−0,005** |

- **Em larvae, o s2 e o s3 derrubam todos os candidatos em 0,03 a 0,06, menos o fg01.** Lembrete: os 3 splits têm a mesma composição na val (223 / 1.534).
- **Leitura, tirando um fator de cada vez:**
  - O fundo 0,1 sozinho não basta: o rand também usa fundo 0,1 e cai −0,059.
  - O SPiFiL sozinho também não basta: R1 e w64 caem −0,026 e −0,054.
  - **A combinação "SPiFiL + recon com peso no objeto" é a única estável em larvae.** É coerente com a hipótese A0/A1: o SPiFiL traz filtros do objeto, e o fundo 0,1 impede que o backprop os puxe para o fundo.
- **Com isso, o fg01 é o candidato mais estável em larvae entre todos os medidos.** Ainda falta o rand_s1/s2 nos splits 2/3 para fechar o controle de seed.

## 2026-10-31 13:58 [A4] mecanismo testado em larvae L3: a razão dentro/fora NÃO explica a estabilidade do fg01
- **Como medi:** cópia do bg_frac.py com o split como argumento (scratchpad a4/bg_frac_split.py), só CPU, 200 imagens da val. Razão = densidade de ativação no objeto / densidade no fundo, com a máscara do dataset usada só para medir.
  - Saída: a4/bgfrac_larvae_L3.txt.
  - Conferência: o R1 s1 treinado deu **1,30**, contra o 1,31 do A1.

| larvae L3, treinado (cru) | s1 | s2 | s3 | κ val s1 / s2 / s3 |
|---|---|---|---|---|
| fg01 (SPiFiL + fundo 0,1) | 1,46 (2,27) | **1,63** (3,05) | **1,16** (2,19) | 0,839 / 0,831 / 0,838 |
| R1 (SPiFiL, uniforme) | 1,30 (2,38) | 1,37 (2,36) | 1,32 (2,27) | 0,847 / 0,816 / 0,826 |
| rand (aleatório + fundo 0,1) | 1,37 (1,97) | 1,21 (1,30) | 1,36 (1,56) | 0,848 / 0,787 / 0,791 |

- **O mecanismo não se sustenta:**
  - O fg01 tem a maior razão no s1 e no s2, mas a **menor de todas no s3** (1,16), e mesmo assim o κ dele no s3 é estável (0,838).
  - O rand s3 tem razão 1,36, igual à do R1, e κ 0,791, isto é, 0,035 abaixo.
  - Nos 9 pontos, Spearman(razão treinada, κ) = **0,20 (p = 0,60)**: sem relação.
  - Isso concorda com a ressalva do A0 para eggs s1 (fg01 com razão 0,93–0,94).
- **O que se mantém:**
  - O SPiFiL cru tem 2,2–3,1x mais ativação no objeto. O aleatório cru tem 1,3–2,0x, porque o objeto tem textura até para filtros aleatórios.
  - Em todos os candidatos, o treino puxa a razão para ~1,2–1,6.
  - **A estabilidade do κ do fg01 em larvae é real, mas não vem de "manter o foco no objeto" pela média dos canais.** A causa está em outro lugar: talvez poucos canais discriminativos, que a média esconde, ou outra coisa. A razão média não serve de métrica proxy do κ.

## 2026-10-31 14:23 [A4] κ val L3: o w64 ganha em protozoan nos 3 splits; o sp200 empata em eggs
- **w64 × R1, protozoan L3** (os dois com BCE uniforme; o w64 tem 32/48/64 filtros):

  | | s1 | s2 | s3 | média |
  |---|---|---|---|---|
  | w64 | 0,868 | 0,860 | 0,851 | **0,860** |
  | R1 | 0,855 | 0,843 | 0,846 | 0,848 |
  | diferença | +0,013 | +0,017 | +0,005 | **+0,011** |

  - **É o primeiro ganho com o mesmo sinal nos 3 splits** entre as variantes medidas. Cada split isolado está perto do ruído (~0,012), mas os 3 apontam para o mesmo lado.
  - **A média de 0,860 fica a 0,004 do I-JEPA (0,864, teste).**
  - Em eggs e larvae, o w64 empata ou perde: eggs −0,001 e larvae −0,006 na média.
  - **Leitura:** largura ajuda onde a profundidade também ajudava (protozoan, em que cada camada ainda somava κ no R1). Bate com a "lei de largura por dataset" do A2.
- **sp200 × fg01, eggs L3:** 0,885 / 0,893 contra 0,894 / 0,898, ou seja, −0,009 / −0,006. Dentro do ruído.
- **Seeds do rand nos splits 2/3:** ainda sem κ (treinando em disc_rand_s1b e disc_rand_s2b).

## 2026-10-31 14:30 [A4] CORREÇÃO: a estabilidade do fg01 em larvae não se repete no sp200
- **sp200 (A3: SPiFiL + fundo 0,1, o mesmo do fg01 mudando só o número de superpixels), κ val L3:**

  | larvae L3 | s1 | s2 | s3 |
  |---|---|---|---|
  | **sp200** | **0,890** | **0,776** | (pendente) |
  | fg01 | 0,839 | 0,831 | 0,838 |
  | rand seed 0 | 0,848 | 0,787 | 0,791 |

  - O sp200 oscila 0,114 entre s1 e s2, mais que o rand (0,061).
  - O s1 do sp200 (0,890) é o **maior κ de larvae medido até agora**, e o s2 está entre os menores.
- **Corrijo a minha nota anterior:** o achado "só a combinação SPiFiL + fundo 0,1 é estável em larvae" **não se sustenta**. Uma variante próxima do fg01 oscila tanto quanto o aleatório.
- **Leitura nova: em larvae, o κ val de um único treino varia ±0,05 entre candidatos parecidos.** A estabilidade do fg01 (0,831–0,839) parece sorte, e não propriedade do método. Junta-se a isso:
  - larvae é binário, com 223 casos na classe minoritária;
  - o mecanismo pela razão dentro/fora já tinha falhado.
- **Regra para larvae:** só confiar numa diferença de κ val maior que ~0,05, ou numa média de várias seeds por split. Diferenças de 0,03–0,05 num split só, como as do rand × SPiFiL em larvae s2/s3, ficam **inconclusivas**.
- O placar rand × SPiFiL em larvae fica em aberto até as seeds s1/s2 nos splits 2/3. **Protozoan s1 continua sendo o único ponto em que o SPiFiL vence com folga contra 2 seeds** (+0,062 / +0,083).
- **sp200 × fg01 em eggs L3:** 0,885 / 0,893 / 0,887 contra 0,894 / 0,898 / 0,887. Empate.

## 2026-10-31 14:42 [A4] κ val L3: em protozoan s1, todas as 5 variantes SPiFiL passam as 2 seeds do rand
- **Protozoan s1 L3:**

  | SPiFiL | κ | | rand | κ |
  |---|---|---|---|---|
  | fg01 | 0,877 | | rand seed 0 | 0,815 |
  | w64 | 0,868 | | rand_s1 | 0,794 |
  | popz | 0,860 | | | |
  | sp200 | 0,856 | | | |
  | R1 | 0,855 | | | |

  - **O pior SPiFiL (0,855) fica 0,040 acima da melhor seed do rand (0,815).** As 5 variantes SPiFiL ocupam a faixa 0,855–0,877 (0,022 de largura); as 2 seeds do rand, 0,794–0,815.
  - **As duas faixas não se sobrepõem.** Variar o SPiFiL (largura, z-score, superpixels, peso do fundo) mexe menos que trocar SPiFiL por aleatório.
  - **É a evidência mais forte até agora de que o SPiFiL contribui como inicializador**, pelo menos em protozoan. Falta repetir no s2/s3 com as seeds novas.
- **sp200 larvae, completo:** 0,890 / 0,776 / 0,847, média **0,838**, contra fg01 0,836. A média é igual e a variância é 10x maior. Isso reforça a regra de larvae: um treino só não decide; a média sim.

## 2026-10-31 15:13 [A4] κ val L3, split 1: o rand já tem 3 seeds

| s1 L3 | rand seed 0 / 1 / 2 | média ± desvio entre seeds | SPiFiL (fg01) | outras variantes SPiFiL |
|---|---|---|---|---|
| eggs | 0,898 / 0,911 / 0,882 | 0,897 ± **0,015** | 0,894 (repetições 0,894 / 0,910 / 0,899: ± **0,008**) | w64 0,913, sp200 0,885, popz 0,904 |
| larvae | 0,848 / 0,859 / 0,821 | 0,843 ± **0,019** | 0,839 | sp200 0,890, w64 0,859, R1 0,847 |
| protozoan | 0,815 / 0,794 / (pendente) | 0,805 ± 0,015 | **0,877** | as 5 variantes ficam em 0,855–0,877 |

- **Variação entre seeds do aleatório no s1:** desvio de 0,015 em eggs e 0,019 em larvae. O desvio do SPiFiL ao repetir o mesmo config em eggs s1 é de 0,008. **No s1, o SPiFiL varia cerca de metade do aleatório,** mas com n = 3 isso ainda não é significativo.
- **Médias no s1:**
  - eggs: empate (rand 0,897 contra SPiFiL 0,894–0,901).
  - larvae: empate (0,843 contra 0,839).
  - **protozoan: o SPiFiL ganha com folga, por 0,050 a 0,072,** cerca de 4x o desvio entre seeds.
- **Novo na vigia:** disc_w64fg (w64 + fundo 0,1), lançado pelo A0.
- **Curvas:** 0 NaN, nenhum run travado. As seeds dos splits 2/3 estão entre a L2 e a L3.

## 2026-10-31 15:24 [A4] κ val protozoan s1 L3 com 3 seeds do rand: a vantagem do SPiFiL se mantém
- **rand seed 0 / 1 / 2:** 0,815 / 0,794 / **0,832**, média **0,814 ± 0,019**.
- **SPiFiL, 5 variantes:** 0,855–0,877, com o fg01 em 0,877.
- **fg01 − média do rand = +0,063, cerca de 3,4 desvios entre seeds.** O pior SPiFiL (R1, 0,855) fica 0,023 acima da melhor seed do rand (0,832). **As faixas continuam sem sobreposição.**
- **Fechado o s1 com 3 seeds nos 3 datasets (L3):**

  | dataset | resultado | diferença |
  |---|---|---|
  | eggs | empate | SPiFiL −0,003 |
  | larvae | empate | SPiFiL −0,004 |
  | protozoan | **SPiFiL vence** | **+0,063** |

- **Regra (s1):** o SPiFiL como inicializador ajuda onde o dataset é mais difícil para o aleatório (protozoan, com o rand em ~0,81), e é neutro onde o aleatório já vai bem (eggs ~0,90).
- Os splits 2/3 com seeds ainda estão treinando.

## 2026-10-31 15:36 [A4] κ val: o bp01 L3 ganha do fg01 em eggs nos 3 splits, e a curva tinha previsto

| eggs L3 | s1 | s2 | s3 | média |
|---|---|---|---|---|
| bp01 (L1–L2 com lr 0,1× junto da L3) | 0,908 | 0,906 | 0,894 | **0,902** |
| fg01 | 0,894 | 0,898 | 0,887 | 0,893 |
| diferença | +0,014 | +0,007 | +0,007 | **+0,010** |

- **Mesmo sinal nos 3 splits.** Cada diferença isolada está dentro do ruído de repetição (~0,012), mas o padrão se repete.
- **Primeira vez que a curva de treino antecipou o κ.** A loss mínima da L3 do bp01 já era 4–7e-5 mais baixa que a do fg01 em 3 de 3 splits de eggs (e em 3 de 3 de larvae).
  - **Regra refinada:** a loss não prevê o κ **entre inicializadores diferentes** (rand × SPiFiL). Mas, para **o mesmo encoder com mais liberdade de treino** (backprop nas camadas antigas), loss mais baixa veio junto com κ mais alto.
- O bp01 s1 (0,908) fica no topo da faixa das 3 seeds do rand em eggs s1 (0,882–0,911).
- **Próximo ponto que importa:** o bp01 em larvae e protozoan L3, onde o SPiFiL já tinha vantagem.

## 2026-10-31 15:42 [A4] κ val: o bp01 em larvae L3 empata, e a regra "loss prevê κ" fica restrita
- **bp01 × fg01, larvae L3:** 0,817 / 0,829 / 0,841 contra 0,839 / 0,831 / 0,838, ou seja, −0,022 / −0,001 / +0,002 (média −0,007).
  - Dentro do grão de larvae (±0,05 num treino só). **É empate.**
- **Correção da regra da nota anterior:** em larvae, a loss da L3 do bp01 também era mais baixa nos 3 splits (−3e-5 a −1e-4), e o κ não subiu. **"Para o mesmo encoder, loss mais baixa ⇒ κ mais alto" só valeu em eggs (3/3), não em larvae (0/3).** Volto à regra geral: **a loss não prevê o κ**, com eggs como exceção a confirmar.
- **bp01 até aqui:** +0,010 em eggs (3/3 splits), empate em larvae. Falta protozoan.

## 2026-10-31 15:53 [A4] nota curta
- **bp01 protozoan s1 L3 = 0,882,** o maior κ de protozoan s1 medido até agora.
  - Contra o fg01 (0,877), dá +0,005, dentro do ruído.
  - Contra a média das 3 seeds do rand (0,814), dá +0,068.
  - Espero os splits s2/s3 para fechar.
- Seeds do rand nos splits 2/3: ainda sem κ. Curvas: 0 NaN, nenhum run travado.

## 2026-10-31 16:20 [A4] κ val: o bp01 ganha em protozoan nos 3 splits; o κ das seeds s2/s3 só sai no fim do growth
- **bp01 × fg01, protozoan L3:**

  | | s1 | s2 | s3 | média |
  |---|---|---|---|---|
  | bp01 | 0,882 | 0,854 | 0,860 | **0,866** |
  | fg01 | 0,877 | 0,836 | 0,853 | 0,855 |
  | diferença | +0,005 | +0,018 | +0,007 | **+0,010** |

  - **Sinal positivo nos 3 splits.** A média de 0,866 é a maior de protozoan entre todas as variantes (w64 0,860, fg01 0,855, R1 0,848). Numericamente passa o I-JEPA (0,864), mas o I-JEPA é teste e isto é val.
  - **Placar do bp01 (L3):** eggs +0,010 (3/3), larvae −0,007 (empate), protozoan +0,010 (3/3). **É a melhor variante nos 3 datasets juntos.**
- **κ das seeds do rand nos splits 2/3:** o SVM está encadeado **depois do growth inteiro** de cada seed (`launch …; flock svm.lock …`).
  - Já há 5 ckpts L3 prontos sem κ: rand_s1 eggs s2/s3 e larvae s2; rand_s2 eggs s2/s3 e larvae s2.
  - Ainda faltam ~3–4 h de growth (larvae s3 e protozoan s2/s3).

## 2026-10-31 16:31 [A4] nota curta: eggs s2 L3 com 2 seeds do rand
- **rand seed 0 / rand_s1:** 0,920 / 0,911.
- **SPiFiL (fg01):** 0,898, com as repetições dando 0,8983 / 0,8982.
- **As 2 seeds do rand ficam acima do SPiFiL**, por +0,013 e +0,022. Em eggs s2, o aleatório ganha de forma estável. O rand_s2 está pendente.
- **Até aqui:** em eggs, o SPiFiL empata ou perde para o aleatório (s1 empata; s2 perde por ~0,017). Em protozoan s1, ganha por +0,063. É coerente com a regra "o SPiFiL ajuda onde o aleatório vai mal".

## 2026-10-31 16:57 [A4] κ val L3, seeds do rand × SPiFiL (fg01): eggs fechado com 3×3, larvae s2 com 2 seeds

| L3 | split | rand (seeds) | média ± desvio | SPiFiL | SPiFiL − média do rand | em desvios |
|---|---|---|---|---|---|---|
| eggs | s1 | 0,898 / 0,911 / 0,882 | 0,897 ± 0,015 | 0,894 | −0,003 | −0,2 |
| eggs | s2 | 0,920 / 0,911 / 0,888 | 0,906 ± 0,016 | 0,898 | −0,008 | −0,5 |
| eggs | s3 | 0,908 / 0,905 / 0,889 | 0,901 ± 0,010 | 0,887 | −0,014 | −1,4 |
| larvae | s1 | 0,848 / 0,859 / 0,821 | 0,843 ± 0,019 | 0,839 | −0,004 | −0,2 |
| larvae | s2 | 0,787 / 0,794 | 0,790 ± 0,005 | **0,831** | **+0,041** | +8 |
| protozoan | s1 | 0,815 / 0,794 / 0,832 | 0,814 ± 0,019 | **0,877** | **+0,063** | +3,4 |

- **eggs (9 treinos do rand × 3 do SPiFiL):** o rand ganha por pouco (média −0,008 para o SPiFiL), dentro de ~1 desvio por split.
  - **Variação entre seeds do rand:** desvio de 0,010–0,016. **Variação do SPiFiL ao repetir o mesmo config:** ≤0,0003 no s2/s3 e 0,008 no s1.
  - **Em eggs, o SPiFiL é muito mais reprodutível, e o aleatório tem média um pouco maior.**
- **larvae s2:** o rand_s1 também cai (0,794), igual à seed 0 (0,787). **A queda do rand em larvae s2 não é azar de seed.** As duas seeds ficam ~0,04 abaixo do fg01.
  - Mas o sp200 (SPiFiL) também dá 0,776 nesse split. Então "o SPiFiL ganha em larvae s2" vale para o fg01, **não para toda variante SPiFiL**.
- **Regra até aqui:**
  1. Em eggs, o aleatório e o SPiFiL empatam na média, e o SPiFiL é mais reprodutível.
  2. Em protozoan s1 e larvae s2, o SPiFiL (fg01) ganha por 3–8 desvios entre seeds do rand.
  3. **Onde o aleatório vai mal, o SPiFiL ajuda. Onde ele vai bem, é neutro.**
- **Pendentes:** rand_s2 larvae s2, larvae s3 (s1/s2) e protozoan s2/s3 (s1/s2).

## 2026-10-31 17:53 [A4] nota curta: larvae s2 L3 com 3 seeds do rand
- **rand seed 0 / 1 / 2:** 0,787 / 0,794 / 0,800, média **0,794 ± 0,007**.
- **fg01:** 0,831, ou seja, **+0,037 (≈ 5,7 desvios entre seeds)**. É o segundo ponto, depois do protozoan s1, em que a vitória do SPiFiL (fg01) resiste a 3 seeds do rand.
- **Placar com 3 seeds** (6 células): o SPiFiL vence em 2 (protozoan s1 e larvae s2, ambas onde o rand fica abaixo de 0,82) e empata em 4 (eggs s1/s2/s3 e larvae s1, onde o rand fica acima de 0,84).
- **Pendentes:** larvae s3 e protozoan s2/s3 com seeds. O growth ainda não terminou. Curvas: 0 NaN, nenhum run travado.

## 2026-10-31 19:57 [A4] κ val larvae s3 L3 com 2 seeds: a queda do rand ali foi azar de seed
- **rand seed 0 / rand_s1:** 0,791 / **0,840**, média 0,815 ± 0,035. **SPiFiL (fg01):** 0,838.
  - A seed 1 empata com o SPiFiL. **A perda de −0,048 da seed 0 em larvae s3 foi azar de seed**, não efeito do SPiFiL.
  - Com 2 seeds, o SPiFiL fica +0,023 acima da média, cerca de 0,7 desvio. **Inconclusivo.**
- **Contraste com larvae s2:** ali as 3 seeds caem juntas (0,787 / 0,794 / 0,800, desvio 0,007), e o fg01 fica +0,037 acima. **O s2 resiste ao teste de seed; o s3 não.**
- **Placar com ≥2 seeds** (7 células, L3):

  | resultado | células |
  |---|---|
  | **SPiFiL vence** | protozoan s1 (+0,063, 3,4 desvios) e larvae s2 (+0,037, 5,7 desvios) |
  | empate | eggs s1/s2/s3, larvae s1, larvae s3 |
  | rand vence | nenhuma, fora do ruído |

- **Variação entre seeds do aleatório:** desvio de 0,007 a 0,035 por célula, com mediana ~0,016. Para o SPiFiL ao repetir o config: ≤0,0003 a 0,008. **O aleatório varia 2–50x mais que o SPiFiL repetido.** Pelo critério do A0, essa é a evidência de estabilidade mais consistente.
- **Pendente:** protozoan s2/s3 com seeds.

## 2026-10-31 20:03 [A4] κ val protozoan L3: o SPiFiL vence o rand nos 3 splits com ≥2 seeds

| protozoan L3 | rand (seeds) | média do rand | SPiFiL (fg01) | diferença | em desvios (pooled ≈ 0,016) |
|---|---|---|---|---|---|
| s1 | 0,815 / 0,794 / 0,832 | 0,814 | 0,877 | **+0,063** | 3,9 |
| s2 | 0,802 / 0,804 | 0,803 | 0,836 | **+0,033** | 2,1 |
| s3 | 0,797 / 0,818 | 0,808 | 0,853 | **+0,045** | 2,8 |

- **Protozoan, média dos 3 splits:** SPiFiL 0,855 contra rand 0,808, ou seja, **+0,047**.
  - As outras variantes SPiFiL também ficam acima da média do rand em todos os splits: bp01 0,866, w64 0,860, R1 0,848 (médias).
  - O que fica mais perto é o popz s2 (0,811 contra 0,803).
- **Conclusão do controle aleatório (L3, ≥2 seeds, 9 células):**

  | dataset | resultado |
  |---|---|
  | **protozoan** | **SPiFiL vence 3/3** (+0,033 a +0,063) |
  | larvae | SPiFiL vence o s2 (+0,037, 3 seeds); empata s1 e s3 |
  | eggs | empate em 3/3 (rand ~0,008 acima, dentro do ruído) |
  | rand vence | **nenhuma célula** fora do ruído |

  - **Estabilidade:** o κ do aleatório varia 0,002–0,035 entre seeds (mediana ~0,015); o SPiFiL repetido, ≤0,0003–0,008.
- **Resposta à regra do usuário ("o SPiFiL tem que contribuir no treino"): contribui.** O efeito é grande em protozoan, existe em parte de larvae e é nulo em eggs, e o SPiFiL é sempre mais reprodutível.
- Pendentes: rand_s2 em larvae s3 e protozoan s2/s3, para fechar 3 seeds em toda célula.

## 2026-10-31 20:40 [A4] nota curta: larvae s3 L3 com 3 seeds do rand
- **rand seed 0 / 1 / 2:** 0,791 / 0,840 / **0,869**, média 0,833 ± **0,039**. **SPiFiL (fg01):** 0,838.
- **Empate na média** (+0,005 para o SPiFiL). A seed 2 do rand passa o SPiFiL (+0,031).
- **É a célula em que o rand mais oscila** entre seeds: um desvio de 0,039, numa faixa de 0,078 entre a pior e a melhor seed. O SPiFiL dá um valor só, no meio dessa faixa.
- **Larvae fechado com 3 seeds:**

  | split | resultado |
  |---|---|
  | s1 | empate |
  | s2 | **SPiFiL +0,037** |
  | s3 | empate |

  - A média dos 3 splits é SPiFiL 0,836 contra rand 0,823 (+0,013). Em larvae, o ganho principal do SPiFiL é **estabilidade**: o desvio entre seeds do rand chega a 0,039.
- Pendente: o rand_s2 em protozoan s2/s3. Novo na vigia: disc_bp01c1280.

## 2026-10-31 20:46 [A4] RESULTADO FINAL do controle aleatório (L3, κ val, 3 seeds do rand × SPiFiL fg01, 9 células)

| dataset | SPiFiL (média) | rand (média de 3 seeds) | SPiFiL − rand | por split (em desvios entre seeds) | desvio médio entre seeds do rand |
|---|---|---|---|---|---|
| eggs | 0,893 | 0,901 | −0,008 | −0,003 (−0,2) / −0,008 (−0,5) / −0,014 (−1,3) | 0,014 |
| larvae | 0,836 | 0,823 | +0,013 | −0,004 (−0,2) / **+0,037 (5,7)** / +0,005 (0,1) | 0,022 |
| **protozoan** | **0,855** | 0,809 | **+0,046** | **+0,063 (3,4) / +0,035 (8,8) / +0,040 (3,0)** | 0,012 |
| geral | | | **+0,017** | | |

- **O SPiFiL contribui como inicializador:**
  - **protozoan:** vence nos 3 splits, por 3 a 9 desvios entre seeds.
  - **larvae:** vence 1 de 3 splits, por 5,7 desvios, e empata nos outros 2.
  - **eggs:** empata nos 3. O rand fica ~0,008 acima, dentro de ~1 desvio.
  - **Nenhuma célula** tem o rand acima do SPiFiL por mais de 1,3 desvio.
- **Estabilidade (critério do A0):**
  - O κ do aleatório varia entre seeds com desvio médio de 0,012–0,022 e chega a 0,039 (larvae s3).
  - O SPiFiL repetido com o mesmo config varia ≤0,0003 (eggs s2/s3) e até 0,008 (eggs s1).
  - **O SPiFiL é ~2–50x mais reprodutível.**
- **Regra:** o ganho do SPiFiL cresce quanto pior o aleatório vai no dataset. Em protozoan, onde o rand fica em ~0,81, ganha +0,046; em eggs, onde o rand fica em ~0,90, empata.
- **Lembrete para larvae:** é binário, e um treino só varia ±0,05 entre variantes SPiFiL (sp200 de 0,776 a 0,890). A vitória no s2 resiste a 3 seeds, mas não generaliza para toda variante SPiFiL.

## [A4] nota: as L1–L3 do bp01c1280 repetem o bp01
- O YAML só muda as L1–L3 em `embed_mode avgpool2d`, que só afeta a sonda de κ do treino. A L4 é a única camada nova (1x1, 1280 canais, sem pool, fill_random_last).
- A curva da L1 confirma: v0 0,990 / 1,377 / 1,095, igual à do fg01/bp01.
- **São ~9 estágios que repetem o bp01.** Daria para começar da L3 do bp01. A decisão é do A0.
- A sonda de memória da L4 está armada (a4/gpu_c1280_L4.txt).

## 2026-10-31 22:09 [A4] bp01c1280 L4 (1x1, 1280 canais, 24x24, sem pool): 1ª leitura da curva
- **Memória (a4/gpu_c1280_L4.txt):** cada processo da L4 usa **~2,7 GB**. As GPUs 0–2 ficam em 25–29 GB de 49 GB, com ~20 GB livres. **Não há risco de OOM.**
- **Curva da L4 contra a L3 da mesma família, eggs** (mínimo da val | val − treino):

  | split | L3 (final) | L4 (parcial) | ΔL4−L3 |
  |---|---|---|---|
  | s1 | 0,51276 \| −0,00045 | 0,51278 \| −0,00046 (época 69) | +2e-5 \| −1e-5 |
  | s2 | 0,51178 \| −0,00179 | 0,51177 \| −0,00174 (época 78) | −1e-5 \| +5e-5 |
  | s3 | 0,51225 \| −0,00139 | 0,51260 \| −0,00143 (época 26, cedo) | — |

  - **A L4 1x1 sem pool não mostra a assinatura de sobreajuste da L4 do R2.** No R2 (48 filtros, kernel 5, com pool, 12x12), a loss subia +6,7e-4 e a distância val − treino +0,0005 já na L4. Aqui, as duas ficam iguais às da L3 (±5e-5).
  - O e3 é 8–12, mais rápido que na L3 (26). Sem sinal de early stop até agora: o best está nas épocas 65 e 74, com a lr ainda alta.
  - **Leitura provisória:** a camada 1x1 sem pool não piora a reconstrução nem ajusta o treino a mais. Ela parece **neutra para a loss**. Se ajudar, vai aparecer no κ avgpool (L4 contra L3 avgpool). Pela regra "early stop disparou ⇒ κ caiu" (15/15 no R2), o bom sinal até aqui é a ausência de early stop.
  - **As L1–L3 do bp01c1280 batem com o bp01:** L3 0,51276 / 0,51178 / 0,51225 contra 0,51276 / 0,51177 / 0,51225. Repetição confirmada.
- **w64fg eggs s1 L3 (1º κ):** 0,914, contra w64 0,913 e fg01 0,894. É um split só; espero s2/s3.

## 2026-10-31 22:15 [A4] nota: c1280 L4 em ~120 épocas; κ val L3 do w64fg
- **bp01c1280 L4** (mínimo da val, L4 contra L3):
  - s1: 0,51268 contra 0,51276 (**−8e-5**).
  - s2: 0,51171 contra 0,51178 (**−7e-5**).
  - s3: 0,51228 contra 0,51225 (+3e-5, na época 70).
  - **Pela primeira vez, uma camada além da L3 baixa a loss da val.**
  - A distância val − treino sobe só +9e-5 / +1,7e-4. No R2 a L4 subia +0,0005.
  - Ainda sem early stop. **A curva é a melhor possível para uma L4.**
  - Memória: as GPUs 0–2 ficam em 24–25 GB, estável.
- **w64fg (32/48/64 + fundo 0,1), κ val L3:**

  | | s1 | s2 | s3 | média | referências |
  |---|---|---|---|---|---|
  | eggs | 0,914 | 0,885 | 0,892 | 0,897 | w64 0,896 · fg01 0,893 · bp01 0,902 |
  | larvae | 0,853 | 0,800 | 0,845 | 0,833 | fg01 0,836 · w64 0,823 |
  | protozoan | 0,879 | 0,861 | (pendente) | 0,870 (2 splits) | nos mesmos 2 splits: w64 0,864 · bp01 0,868 · fg01 0,857 |

  - **Combinar largura e fundo 0,1 não soma:** em eggs e larvae, o w64fg empata com as partes. Em protozoan, fica no topo junto com bp01 e w64, dentro do ruído.

## 2026-10-31 23:06 [A4] c1280 L4 eggs fechada: a curva e uma previsão de κ registrada antes do SVM

| eggs | L3 → L4: mínimo da val | val − treino | época do best (L4) | early stop |
|---|---|---|---|---|
| s1 | 0,51276 → 0,51266 (**−1,0e-4**) | −0,00045 → −0,00026 (+1,9e-4) | 163 | não |
| s2 | 0,51178 → 0,51169 (**−0,9e-4**) | −0,00179 → −0,00151 (+2,8e-4) | **145** | **sim, na 196** |
| s3 | 0,51225 → 0,51214 (**−1,1e-4**) | −0,00139 → −0,00118 (+2,1e-4) | 185 | não |

- **Comparação com a L4 do R2:**

  | | R2 (48 filtros, kernel 5, pool) | c1280 (1x1, sem pool) |
  |---|---|---|
  | loss | +6,7e-4 | **−1e-4** |
  | val − treino | +4 a 5e-4 | +2 a 3e-4 |
  | early stop | 2 de 3 | 1 de 3 |
  | best | 133–152 | 145–185 |

  - **A camada 1x1 sem pool é a melhor L4 até agora pela curva**, mas ainda mostra um pouco de ajuste ao treino.
- **Previsão registrada antes do SVM** (regra do R2, "best antes da época ~150 ⇒ κ abaixo da L3", 15 de 15):
  - **s2: L4 < L3 no κ.**
  - s1 (best 163): perto de L3, de borda.
  - **s3 (best 185): L4 ≥ L3.**
  - Ressalva: a comparação justa é L3 avgpool contra L4 avgpool (o embed do c1280). A regra veio do flatten no R2.
- **Memória até o fim da L4:** as GPUs 0–2 ficaram em 22–29 GB de 49 GB. Sem problema.
- **w64fg protozoan s3 = 0,856:** o protozoan do w64fg fecha em 0,879 / 0,861 / 0,856, média **0,865**. Fica empatado com o bp01 (0,866) no topo de protozoan.
- **bp01w64 L3 em eggs tem a loss mais baixa de todas as variantes:** 0,51263 / 0,51166 / 0,51215, contra o bp01 com 0,51276 / 0,51177 / 0,51225. O κ ainda não saiu.

## 2026-10-31 23:37 [A4] κ val L3: o bp01w64 tem a melhor média em eggs e larvae

| L3, média dos 3 splits | eggs | larvae | protozoan |
|---|---|---|---|
| **bp01w64** (bp01 + 32/48/64 + fundo 0,1) | **0,906** (0,929 / 0,892 / 0,897) | **0,845** (0,864 / 0,822 / 0,849) | (pendente) |
| bp01 | 0,902 | 0,829 | **0,866** |
| w64fg | 0,897 | 0,833 | 0,865 |
| fg01 | 0,893 | 0,836 | 0,855 |
| rand (3 seeds) | 0,901 | 0,823 | 0,809 |

- **O s1 de eggs (0,929) é o maior κ de eggs medido.**
- Contra o 2º melhor de cada dataset, a diferença é pequena:
  - eggs: +0,004 sobre o bp01;
  - larvae: +0,009 sobre o fg01.
  - **As duas estão dentro do ruído de ~0,012 por split.** O que conta é ser o topo nos dois datasets.
- **Curva × κ, de novo em eggs:** o bp01w64 também tem a loss de L3 mais baixa em eggs (0,51263 / 0,51166 / 0,51215).
  - Em eggs, a ordem da loss mínima entre variantes da mesma família (fg01 → bp01 → bp01w64) bate com a ordem do κ (0,893 → 0,902 → 0,906).
  - Em larvae, não bate (bp01 tem loss mais baixa e κ mais baixo que o fg01). Lá, o grão do κ é grosso.
- **Larvae com o bp01w64:** o s2 sobe para 0,822 (fg01 0,831, rand 0,794) e o desvio entre splits é 0,021.
- **c1280:** já passou para larvae/protozoan; o κ de eggs L3/L4 ainda não saiu. Curvas: 0 NaN e nenhum run travado. O protozoan s3 L1 está na época 3, recém-começado.

## 2026-10-31 23:43 [A4] o bp01w64 fecha no topo dos 3 datasets; a L4 do c1280 em larvae ajusta o treino
- **bp01w64, protozoan L3:** 0,888 / 0,850 / 0,858, média **0,865**. Empata com bp01 (0,866) e w64fg (0,865). O s1 (0,888) é o maior κ de protozoan s1 medido.
- **Resumo do bp01w64 (média de κ val L3):**

  | dataset | κ | posição |
  |---|---|---|
  | eggs | 0,906 | 1º |
  | larvae | 0,845 | 1º |
  | protozoan | 0,865 | empate no 1º |

  - **É a melhor candidata de L3 até agora.** As margens sobre o 2º estão dentro do ruído, mas ela é a única que fica no topo em todos os 3.
  - Contra o rand (3 seeds): +0,005 em eggs, +0,022 em larvae e +0,056 em protozoan.
- **c1280 L4 em larvae** (L3 → L4: mínimo da val | val − treino | época do best):

  | split | L3 | L4 | early stop |
  |---|---|---|---|
  | s1 | 0,49563 \| +0,00009 \| 184 | 0,49562 \| **+0,00042** \| **96** | **sim, na 147** |
  | s2 | 0,49588 \| −0,00021 \| 175 | 0,49581 \| **+0,00023** \| **98** | **sim, na 149** |
  | s3 | 0,49601 \| +0,00036 \| 196 | 0,49596 \| **+0,00079** \| 138 | **sim, na 189** |

  - **Em larvae, a L4 1x1 mostra a assinatura de sobreajuste:** early stop 3/3, best antes da época 140 e distância val − treino +3 a 4e-4. A loss fica igual à da L3.
  - Em eggs, a mesma L4 baixava a loss em 1e-4 e só disparava o early stop em 1/3.
  - **Previsão registrada para o κ avgpool:** larvae L4 < L3 nos 3 splits. Em eggs: s2 L4 < L3, s1 ≈, s3 L4 ≥ L3.
- **Saúde:** 0 NaN, nenhum run travado. Agora roda o protozoan L1 do c1280.

## 2026-11-01 01:40 [A4] κ val do c1280 (avgpool): a L4 1x1 sobe o κ, minha previsão errou, e o embed muda o κ em ~0,09 com sinal por dataset
- **L3 contra L4 do c1280, os dois em avgpool** (dim 48 → 1.280):

  | | s1 | s2 | s3 |
  |---|---|---|---|
  | eggs L3 → L4 | 0,825 → **0,869** (+0,043) | 0,828 → **0,870** (+0,042) | 0,804 → **0,864** (+0,060) |
  | larvae L3 → L4 | 0,915 → **0,934** (+0,020) | 0,909 → 0,909 (−0,001) | (pendente) |

- **A minha previsão errou.** O eggs s2 teve early stop (best na 145) e o κ subiu +0,042. O larvae s1 teve early stop (best na 96) e o κ subiu +0,020.
  - **A regra "early stop ⇒ κ cai" (15/15 no R2) não vale aqui.** No R2, o flatten encolhia de 27.648 para 5.808–7.744 dimensões. Aqui, o embedding avgpool cresce de 48 para 1.280.
  - **A regra do R2 media o tamanho do embedding junto com o sobreajuste e não os separava.** Retiro a regra como guia geral: ela só vale quando o embedding encolhe.
- **Achado maior: só o modo de embed muda o κ da mesma L3 em ~0,09, com sinal oposto por dataset.**
  - A L3 do c1280 tem os mesmos pesos do bp01: a curva bate até 1e-5 e os YAMLs são iguais até a L3.

  | mesma L3 (bp01) | flatten (27.648 dim) | avgpool (48 dim) | Δ |
  |---|---|---|---|
  | eggs | 0,908 / 0,906 / 0,894 | 0,825 / 0,828 / 0,804 | **−0,08** |
  | larvae | 0,817 / 0,829 / – | 0,915 / 0,909 / – | **+0,09** |

  - **O larvae em avgpool L3 (0,915) e L4 (0,934) passa todos os flatten medidos.** O melhor era o bp01w64 s1 com 0,864.
  - **Leitura:** em larvae (binário, objeto alongado e de posição variável), a média espacial descarta a posição e ganha. Em eggs, a posição e a forma no mapa importam, e o flatten ganha.
  - **Consequência:** o modo de embed é um eixo de **~0,09 de κ**, maior que qualquer variante de treino medida (≤0,02). Ele deveria ser escolhido por dataset, e todas as comparações entre famílias só valem dentro do mesmo embed.
- Pendentes: larvae s3 e protozoan (L3/L4 avgpool).

## 2026-11-01 01:46 [A4] modo de embed × camada, κ val (média dos 3 splits; cada célula traz s1/s2/s3)
Fontes: svm_disc_a2_bp01_val, svm_disc_bp01c1280_val, svm_disc_bp01w64_val e svm_disc_bp01w64_val_avg (só leitura). A L3 do c1280 tem os pesos do bp01.

| dataset | bp01 L3 flatten | bp01 L3 avg | **c1280 L4 avg** (1x1×1280) | bp01w64 L3 flatten | bp01w64 L3 avg |
|---|---|---|---|---|---|
| eggs | **0,902** (0,908 / 0,906 / 0,894) | 0,819 | 0,868 | **0,906** (0,929 / 0,892 / 0,897) | 0,845 |
| larvae | 0,829 | 0,899 | **0,912** (0,934 / 0,909 / 0,892) | 0,845 | 0,904 (0,894 / 0,940 / 0,879) |
| protozoan | **0,866** (0,882 / 0,854 / 0,860) | 0,750 | 0,828 | **0,865** | 0,796 (2 splits) |

- **O efeito do embed se repete na outra família (bp01w64)**, com o mesmo sinal por dataset:
  - eggs: −0,061 com avg;
  - larvae: **+0,059** com avg;
  - protozoan: −0,069 com avg.
  - **Não é coisa do c1280.**
- **A L4 1x1 ajuda dentro do avgpool nos 3 datasets:** +0,049 em eggs, +0,013 em larvae e +0,078 em protozoan, contra a L3 avg da mesma família. **Mas ela só supera a L3 flatten em larvae.** Em eggs fica −0,034 e em protozoan −0,038.
- **Embed por dataset (pela val):**

  | dataset | embed | melhor configuração | κ |
  |---|---|---|---|
  | eggs | **flatten na L3** | bp01w64 | 0,906 |
  | larvae | **avgpool** | c1280 L4 avg | 0,912 (bp01w64 L3 avg: 0,904) |
  | protozoan | **flatten na L3** | bp01 | 0,866 (bp01w64: 0,865) |

- **Leitura:** a média espacial ajuda onde a posição do objeto não importa (larvae: objeto alongado e de posição variável, binário). Ela atrapalha onde a forma no mapa carrega a classe (eggs e protozoan: forma e textura localizadas). A L4 de 1.280 canais compensa só parte da informação espacial perdida pelo avgpool.
- **O que medir em seguida:** o bp01w64c1280 L4 avg em larvae (bp01w64 + 1x1). É o candidato mais provável a passar 0,912 em larvae. Curvas na vigia.

## 2026-11-01 02:42 [A4] nota curta
- **bp01w64 L3 avg, protozoan completo:** 0,811 / 0,782 / 0,799, média **0,797**. O flatten dá 0,865, então são −0,068. Fecha a tabela embed × camada.
- **bp01w64c1280:** está na L4 de eggs (épocas 5–69).
  - As L1–L3 repetem o bp01w64, como no c1280.
  - A L4 s2 parcial dá 0,51174 contra a L3 do bp01w64 com 0,51166. Ainda é cedo.
  - Memória: 22 GB por GPU. 0 NaN.

## 2026-11-01 03:38 [A4] curva: L4 do bp01w64c1280 em eggs, contra a L3 do bp01w64 (mínimo da val | val − treino | época do best)

| split | bp01w64 L3 | bp01w64c1280 L4 | early stop |
|---|---|---|---|
| s1 | 0,51263 \| −0,00037 \| 195 | 0,51260 \| −0,00015 \| **126** | sim, na 177 |
| s2 | 0,51166 \| −0,00172 \| 198 | 0,51164 \| −0,00140 \| **116** | sim, na 167 |
| s3 | 0,51215 \| −0,00130 \| 199 | 0,51208 \| −0,00108 \| 158 | não |

- **Com a base mais larga (w64), a L4 1x1 rende menos e ajusta o treino mais cedo que no c1280:**

  | | c1280 | bp01w64c1280 |
  |---|---|---|
  | queda da loss | −1e-4 | −2 a −7e-5 |
  | best | 145–185 | **116–158** |
  | early stop | 1/3 | **2/3** |
  | val − treino | +2–3e-4 | +2–3e-4 |

- **Sem previsão de κ:** retirei a regra "early stop ⇒ κ cai" para o avgpool, porque o κ do c1280 subiu mesmo com early stop. Para larvae, a referência a bater em avg é **0,912** (c1280 L4).
- **Saúde:** memória em 22 GB por GPU, 0 NaN. Agora roda o larvae L4.

## 2026-11-01 04:34 [A4] nota curta: L4 do bp01w64c1280 em larvae
- **Early stop nos 3 splits,** na época 117 / 130 / 157. No c1280 tinha sido 147 / 149 / 189.
- **val − treino:** +0,00046 / +0,00045 / +0,00111, ou seja, +2 a 6e-4 sobre a L3 do bp01w64.
- **Mínimo da val:** igual ao da L3 (−4e-5 a 0).
- **O padrão se repete nos dois datasets:** com a base mais larga, a L4 1x1 ajusta o treino mais cedo. O κ avgpool decide; a referência em larvae é 0,912 (c1280 L4).
- **O protozoan L1 do bp01w64c1280 é igual ao do bp01w64** (0,50719 / 0,50740 / 0,50758), repetido como esperado.
- **Saúde:** 22 GB por GPU, 0 NaN, nenhum run travado. O SVM sai no fim do growth.

## 2026-11-01 06:21 [A4] κ val do bp01w64c1280 (avg) e ruído de re-run do avgpool: ~3x o do flatten por split
1. **Ruído de re-run do avgpool.** As L1–L3 do bp01w64c1280 retreinam o bp01w64 (épocas de ckpt diferentes). Comparei a L3 avg das duas cópias:

   | | s1 | s2 | s3 |
   |---|---|---|---|
   | eggs | −0,050 | −0,004 | +0,045 |
   | larvae | +0,016 | −0,032 | +0,004 |
   | protozoan | −0,032 | +0,032 | −0,033 |

   - **Mediana de |Δ| = 0,032 e máximo 0,050 por split, cerca de 3x o ruído do flatten (~0,011).**
   - **A média dos 3 splits fica estável:** eggs 0,845 / 0,842, larvae 0,904 / 0,900, protozoan 0,797 / 0,786 (Δ ≤ 0,011).
   - **Regra para leitura avgpool:** não comparar split a split. Só a média dos 3 splits serve, com margem de ~0,01.
2. **bp01w64c1280, κ val (médias):**

   | | bp01w64 L3 flatten | L3 avg (2 re-runs) | **L4 avg** | c1280 L4 avg |
   |---|---|---|---|---|
   | eggs | **0,906** | 0,845 / 0,842 | 0,880 | 0,868 |
   | larvae | 0,845 | 0,904 / 0,900 | **0,917** (0,935 / 0,913 / 0,904) | 0,912 |
   | protozoan | **0,865** | 0,797 / 0,786 | 0,837 | 0,828 |

   - Contra o c1280 L4: +0,012 / +0,005 / +0,009, **dentro do ruído do avg.** A base mais larga na L4 1x1 empata.
   - **A L4 1x1 + avg sobe ~0,04 sobre a L3 avg em todos os datasets**, nas duas famílias. A regra se repete.
3. **Melhor configuração por dataset (val):**

   | dataset | configuração | κ |
   |---|---|---|
   | eggs | bp01w64 L3 flatten | 0,906 |
   | larvae | L4 1x1×1280 + avg | 0,912–0,917 (c1280 ou bp01w64c1280) |
   | protozoan | bp01 / bp01w64 L3 flatten | 0,866 / 0,865 |

   - **Com o fundo 0,1, o backprop leve e a largura,** o L3 flatten (eggs, protozoan) e o L4 avg (larvae) formam o melhor conjunto medido.

## 2026-11-01 06:21 [A4] REGRAS DE TREINAMENTO E LEITURA, consolidadas (fim da vigia; nenhum run disc_* ativo)
1. **Épocas.**
   - Em L1–L3, a loss chega a 0,001 do mínimo nas épocas 6–36 (com fundo 0,1: 16–36). O "best" na época 197–199 é efeito do cosine.
   - Orçamento proposto: **60 épocas**, ainda sem validar no κ.
   - Da L4 em diante, o best sai antes (74–185) e o early stop dispara.
2. **A loss não prevê o κ entre inicializadores ou leituras diferentes.** Exemplo: em larvae s2, o rand tem loss mais baixa e κ 0,044 pior.
   - Exceção fraca: dentro da mesma família em eggs (fg01 → bp01 → bp01w64), a ordem da loss bateu com a do κ. Em larvae, não.
3. **Ruído do κ.**
   - O treino é quase determinístico na loss (≤6e-6), mas não no κ.
   - Por re-run e por split: **flatten ~0,011; avgpool ~0,032 (até 0,050).**
   - Larvae é binário: o grão do κ é ~0,004, e uma única run varia ±0,05 entre variantes.
   - **Para comparar, use a média dos 3 splits, com margem ≥0,01 (flatten) ou ≥0,01–0,02 (avg).**
4. **O SPiFiL contribui como inicializador** (L3, 3 seeds do rand × 3 splits):
   - protozoan **+0,046** (vence 3/3, por 3–9 desvios);
   - larvae +0,013 (vence 1/3);
   - eggs −0,008 (empate).
   - O SPiFiL é 2–50x mais reprodutível.
   - **O ganho é maior onde o aleatório vai pior.**
   - A razão de ativação dentro/fora do objeto **não** explica o efeito (Spearman 0,20).
5. **Profundidade com conv kernel 5 + pool (R2):** o κ é máximo na L2–L3 e cai da L4 em diante (até −0,09 na L8), com a distância val − treino crescendo.
   - A regra "early stop ⇒ κ cai" valeu 15/15 ali, mas **só porque o flatten encolhia. Ela não vale para a L4 1x1 + avgpool.**
6. **A leitura (embed) é o maior eixo medido (~0,09)**, com sinal por dataset:
   - flatten é melhor em eggs (+0,08) e protozoan (+0,07);
   - avgpool é melhor em larvae (+0,07).
   - **Compare famílias só dentro da mesma leitura.**
7. **A L4 1x1×1280 sem pool sobe o avgpool ~0,04 em todos os datasets,** mas só passa o flatten L3 em larvae. Com uma base mais larga (w64), ela ajusta o treino mais cedo (best 66–158) sem perder κ.
8. **Variantes de treino:** todas ≤0,02 de κ.
   - O **bp01** (L1–L2 com lr 0,1×) e o **bp01w64** são os melhores em flatten (eggs 0,906, protozoan 0,865–0,866).
   - O fundo 0,1 sozinho, popz, sp200 e largura sozinha empatam dentro do ruído.
   - Mais filtros só ajudaram em protozoan (+0,011).
9. **Melhor por dataset (val):**

   | dataset | configuração | κ |
   |---|---|---|
   | eggs | bp01w64 L3 flatten | 0,906 |
   | larvae | L4 1x1×1280 avg | 0,912–0,917 |
   | protozoan | bp01 / bp01w64 L3 flatten | 0,866 / 0,865 |

10. **Custo.**
    - **Repetições evitáveis:** L1–L3 refeitas no R2, no bp01 (L1), no c1280 e no bp01w64c1280, e relançamentos que retreinaram pontos prontos.
    - **O SVM encadeado no fim do growth atrasou o κ das seeds em ~3–4 h.**
    - A memória da L4 de 1.280 canais é ~2,7 GB por processo: sem risco.

## 2026-11-01 07:45 [A4] bp01w64c96: a L4 1x1×96 (flatten 55k dimensões) em eggs é a L4 mais limpa até agora
- **Custo da sonda de κ:** L3 a 5,8–6,3 s/época e L4 a 5,9–6,5 s/época. **O flatten de 55.296 dimensões não encarece o treino.** O risco do precedente da g5_random_in_feature não se repetiu.
- **Memória:** ~2,9 GB por processo da L4 (c1280: 2,7 GB); as GPUs 0–2 ficam em 22 GB. Sem risco.
- **Curva L3 → L4, eggs, na época ~190** (mínimo da val | val − treino):

  | split | L3 | L4 |
  |---|---|---|
  | s1 | 0,51263 \| −0,00037 | 0,51260 \| −0,00034 (**+3e-5**) |
  | s2 | 0,51166 \| −0,00171 | 0,51161 \| −0,00164 (**+7e-5**) |
  | s3 | 0,51214 \| −0,00132 | 0,51210 \| −0,00123 (**+9e-5**) |

  - **Comparação entre as L4 1x1 testadas:**

    | | c1280 | bp01w64c1280 | **c96** |
    |---|---|---|---|
    | aumento de val − treino | +2–3e-4 | +2–3e-4 | **+3–9e-5** |
    | early stop | 1/3 | 2/3 | **nenhum** até a ~190 |

  - **A L4 de 96 canais quase não ajusta o treino a mais.** Ela é pequena (64 → 96, 6.240 parâmetros na conv), e o ganho de loss é pequeno (−3 a −5e-5).
  - O κ flatten decide contra 0,906 (eggs). A hipótese do A0 (o 1x1 recombina canais sem perder posição) prevê L4 ≥ L3 flatten.
- **As L1–L3 repetem o bp01w64** (curvas iguais até 1e-5), como nas famílias c1280.

## 2026-11-01 08:41 [A4] nota curta: c96 L4
- **eggs:** as 200 épocas fecharam nos 3 splits, **sem early stop**. O mínimo fica −3 a −5e-5 abaixo da L3; a distância val − treino sobe +4 a 1e-4.
- **larvae (épocas 108–162):**
  - Sem early stop até aqui. O bp01w64c1280 tinha parado nas épocas 117 / 130 / 157.
  - val − treino, contra a L3 do bp01w64: +1,4e-4 / +0,9e-4 / −0,2e-4. Mínimo: −5e-5 / −3e-5 / +4e-5.
- **A L4 1x1×96 segue sem a assinatura de sobreajuste nos 2 datasets.** O SVM (L4 flatten) sai no fim do growth.

## 2026-11-01 09:37 [A4] nota curta: c96 L4 em larvae fechou
- **Early stop nos 3 splits,** na época 166 / 182 / 182. Mesmo assim, é mais tarde que no bp01w64c1280 (117 / 130 / 157).
- **val − treino:** +0,00041 / +0,00016 / +0,00068, ou seja, +1,5 a 3e-4 sobre a L3 do bp01w64. **O mínimo da val fica igual ao da L3.**
- **Leitura:** em larvae, toda L4 testada ajusta o treino (1280 ou 96 canais), e a de 96 é a mais branda. Em eggs, a c96 não ajustou. O κ ainda está pendente: larvae L4 flatten contra 0,845 (flatten) e 0,917 (avg).
- **Protozoan** está na L2. 0 NaN, nenhum run travado.

## 2026-11-01 11:22 [A4] κ val do bp01w64c96: a L4 1x1×96 em flatten empata com a L3 flatten (não ajuda)

| flatten | bp01w64 L3 (36.864 dim) | **c96 L4** (55.296 dim) | Δ |
|---|---|---|---|
| eggs | 0,929 / 0,892 / 0,897 (**0,906**) | 0,898 / 0,896 / 0,891 (**0,895**) | −0,011 |
| larvae | 0,864 / 0,822 / 0,849 (0,845) | 0,851 / 0,809 / 0,872 (0,843) | −0,002 |
| protozoan | 0,888 / 0,850 / 0,858 (0,865) | 0,870 / 0,846 / 0,857 (0,858) | −0,008 |

- **A L4 1x1 de 96 canais lida em flatten não supera a L3 flatten em nenhum dataset.** A média é −0,007, dentro do ruído (~0,01).
  - A hipótese "o 1x1 recombina os canais em protótipos sem jogar a posição fora" **não se confirma no κ**.
  - Junto com o resultado do avgpool: **a L4 1x1 só ajuda quando a leitura joga a posição fora (avg).** Ali ela recupera ~0,04 do que o avg perde. No flatten, que já guarda a posição, ela não acrescenta nada.
- **Mais um caso de "a loss não prevê o κ":** a curva da L4 c96 era a mais limpa de todas (loss −3 a −5e-5, val − treino ≤ +1e-4 em eggs, sem early stop em eggs) e o κ não subiu.
- **A escolha por dataset fica igual:**

  | dataset | configuração | κ |
  |---|---|---|
  | eggs | bp01w64 L3 flatten | 0,906 |
  | larvae | L4 1x1×1280 avg | 0,912–0,917 |
  | protozoan | bp01 / bp01w64 L3 flatten | 0,866 / 0,865 |

- **Custo da c96:** a L4 roda a 5,9–6,5 s/época, igual à L3, e usa 2,9 GB por processo. As L1–L3 repetiram o bp01w64.

## 2026-11-01 12:30 [A5] literatura: a alavanca barata é a leitura, não o treino
- **Resposta:** antes de treinar algo novo, testar leituras só de eval. A literatura dá +2 a +6 pts para elas, e aqui a leitura já mexe ~0,09 no κ.
- **Top 1, pirâmide espacial:** avg 1×1+2×2+4×4 do L3, 1.344 dim (meio-termo entre o flatten de 36.864 e o avg de 64). Deep Spatial Pyramid, arXiv 1504.05277: +2 a +6.
- **Top 2, L2 por pixel + raiz com sinal** antes do flatten: +0,4 a +3,6 (mesma fonte). PCA piorou lá.
- **Top 3, GeM p=3** no lugar do avg (larvae): +5–6 mAP sobre avg/max em retrieval (arXiv 1711.02512).
- **Top 4–6:** concatenar L2+L3 (VGG sem atalho piora no fim, arXiv 1901.09005); covariância com raiz de matriz (+2,7 a +3,8, arXiv 1707.06772); `class_weight` e scaler como parâmetro do `fit_svm` (evidência mista).
- **Top 7, treino:** AE mascarado (média LAB, loss só no tapado). Reconstrução pura aprende o que não serve; mascarar ajuda e ruído gaussiano não (arXiv 2402.11337, SparK 2301.03580).
- **Top 8, treino:** SIGReg (já existe em `core/losses/real_sigreg.py`). LeJEPA: ResNet-34 in-domain 78,2 vs DINOv2 67,6 com probe congelado, e loss↔probe Spearman 85–99%.
- **Justiça:** toda leitura nova roda também nas features do I-JEPA. O nosso κ é de val e o dele é de teste.
- **Leis de escala:** ajustar `1−κ = a·n^−α + b·m^−β + c∞` (Rosenfeld, arXiv 1909.12673) no κ flatten. Para profundidade, usar BNSL (arXiv 2210.14891).
- **FLIM:** viés −⟨μ,k⟩ sem z-score por bloco (FLIM-BoFP, arXiv 2602.20845); dissertação Miranda/Falcão (abr/2026) nos mesmos 3 datasets. Relatório: `discovery_literature_ideas.md`.

## 2026-11-01 12:51 [A4] regra nova (sem flatten, leitura < 2.000 dim): o κ val do spp na L3 do bp01w64, contra avg

| leitura (< 2.000 dim) | eggs | larvae | protozoan |
|---|---|---|---|
| bp01w64 L3 avg (64 dim) | 0,845 | 0,904 | 0,797 |
| **bp01w64 L3 spp (1.344 dim)** | **0,878** (0,901 / 0,870 / 0,863) | 0,872 (0,882 / 0,872 / 0,862) | **0,847** (só o s1) |
| c1280 L4 avg (1.280 dim) | **0,880** | **0,917** | 0,837 |

- **eggs:** o spp L3 sobe +0,033 sobre o avg L3 e **empata com a L4 avg** (0,878 contra 0,880). Ele recupera ~40% do que o avg perdia contra o flatten (0,906).
- **larvae:** o spp L3 fica abaixo do avg (−0,032 contra o L3 avg, −0,045 contra o L4 avg). Em larvae, a posição atrapalha mesmo em pirâmide.
- **protozoan s1:** spp 0,847, contra L4 avg 0,834 e L3 avg 0,811. Provisoriamente é o melhor abaixo de 2.000 dim; faltam s2 e s3.
- **Escolha provisória por dataset (< 2.000 dim):**

  | dataset | leitura | κ |
  |---|---|---|
  | eggs | **spp L3 ≈ L4 avg** | ~0,88 |
  | larvae | **L4 avg** | 0,917 |
  | protozoan | **spp L3** (s1) | pendente |

- **Distância ao I-JEPA (teste, 50%):** eggs −0,060, larvae −0,026, protozoan −0,017 (s1 do spp contra 0,864).
- **Scripts atualizados** (só leituras < 2.000 dim; o flatten saiu da base): a4/readouts_cmp.py, a4/g20_kappa.py (base avg e spp L3 para eggs/protozoan, avg L4 para larvae; o l_c2560 saiu) e a4/pcts_fit.py (pcts_flat → pcts_spp).

## 2026-11-01 13:11 [A6] o job do crispim saiu (não foi kill nosso)
- **Os 4 processos `lejepa-pretrain.py` do crispim sumiram juntos entre 13:09:53 e 13:10:24.** Eram ~18,2 GB em cada GPU 0-3, e o job rodava havia 3 dias.
- **Folga às 13:09:53:** GPU0 11,5 GB, GPU1 14,0 GB, GPU2 7,7 GB, GPU3 12,9 GB. Nenhuma amostra de 30 s ficou abaixo de 7 GB livres.
- **dmesg:** nenhum OOM nem Xid recente. Pela folga e pela saída simultânea dos 4 ranks, foi o job dele terminando ou caindo, não falta de memória causada por nós.
- **Nenhuma sessão nossa foi morta.** Nada dele foi relançado.

## 2026-11-01 13:17 [A4] κ val das leituras < 2.000 dim: o GeM na L4 1x1×1280 é a melhor leitura em eggs e protozoan

| leitura (dim) | eggs | larvae | protozoan |
|---|---|---|---|
| bp01w64 L3 avg (64) | 0,845 | 0,904 | 0,797 |
| bp01w64 L3 spp (1.344) | 0,878 | 0,872 | 0,829 (0,847 / 0,815 / 0,827) |
| bp01w64 L3 gem (64) | 0,811 | 0,831 | 0,798 |
| c1280 L4 avg (1.280) | 0,880 | **0,917** | 0,837 |
| **c1280 L4 gem (1.280)** | **0,899** (0,909 / 0,892 / 0,894) | 0,910 (0,896 / 0,918 / 0,915) | **0,875** (0,876 / 0,875 / 0,873) |

- **O GeM (p = 3) na L4 de 1.280 canais:**
  - eggs: +0,019 sobre a L4 avg e +0,021 sobre o spp L3. Fica a 0,007 do antigo flatten (0,906), com 22x menos dimensões.
  - protozoan: **+0,038 sobre a L4 avg e +0,046 sobre o spp L3.** É o maior protozoan medido, flatten incluído (0,866), e passa o I-JEPA (0,864, teste).
  - larvae: 0,910 contra 0,917 do avg, **empate dentro do ruído do avg (~0,015)**.
  - Muito estável no protozoan: os 3 splits ficam entre 0,873 e 0,876 (desvio 0,0015).
- **Na L3, com 64 canais, o GeM perde para o avg** (−0,034 / −0,073 / +0,001). **O GeM só funciona sobre muitos canais (1x1×1280).** Com poucos canais, o máximo suave descarta informação demais.
- **O spp completo em protozoan cai para 0,829** (o s1 de 0,847 era o melhor split). Ele fica abaixo da L4 avg e da L4 gem.
- **Leitura por dataset (< 2.000 dim, val):**

  | dataset | leitura | κ |
  |---|---|---|
  | eggs | **L4 gem** | 0,899 |
  | larvae | **L4 avg** | 0,917 (gem 0,910, empate) |
  | protozoan | **L4 gem** | 0,875 |

  - **Proposta:** usar **L4 gem como leitura única** para os 3 datasets. Ela perde só −0,007 em larvae, dentro do ruído, e simplifica a g20.
- **Distância ao I-JEPA (teste, 50%):** eggs −0,041, larvae −0,026 (com avg), protozoan **+0,011**.
- **Para a g20:** as famílias de eggs/protozoan estão em L3 (avg + spp). O GeM só ajudou na L4. Uma re-avaliação da g20 em GeM não ajudaria na L3.

## 2026-11-01 14:58 [A4] g20, curvas, larvae s1 L4 (sem κ ainda; cada família roda s1 → s2 → s3 em fila)
- **Base:** bp01w64c1280 larvae s1 L4, com val − treino +0,00046 e early stop na 117.

  | variante | estado | val − treino | mínimo da val |
  |---|---|---|---|
  | **l_aug (geo_aug)** | 200 épocas, **sem early stop** | **−0,00004** | 0,49532 (−3e-5 da base) |
  | l_sp100 | early stop na 174 | +0,00075 | — |
  | l_bp10 | em treino, época 152 | +0,00042 | — |
  | l_bp03 | em treino, época 85 | +0,00026 | — |
  | l_bn | em treino, época 74 | +0,00011 | — |

  - **O geo_aug tira a assinatura de sobreajuste da L4 1x1 em larvae.** A distância val − treino fica em ~0 (a base dá +0,00046) e o early stop não dispara.
  - Faz sentido: com o treino mais difícil, a camada nova não decora. Se isso vira κ, o gem/avg L4 do l_aug vai dizer.
  - Lembrete: em larvae, nenhuma regra de curva previu o κ de forma confiável.
- **eggs e protozoan (base L4 nova, recomeçada às 14:00):** na L1–L3 do s1. Nada anômalo; o e_bg0 está na escala própria (~0,67).
- **Saúde:** 25 estágios rodando, 0 NaN, nenhuma divergência, nenhum run travado.

## 2026-11-01 13:05 [A5] técnicas de convolução: GRN na L4 1×1×1280 é a primeira da fila
- **Resposta:** GRN (ConvNeXt V2) logo após a ReLU da L4. Feito para MAE em conv; o colapso aparece justo na camada de expansão; começa como identidade (γ=β=0, `+x`); +2.560 params.
- **Ganho na fonte:** FCMAE 83,7 → 84,6, acima do SE (84,4) e do CBAM (84,5), que custam +20M params. No mesmo teste o BN caiu para 80,5.
- **2º, ECA:** o SE sem MLP, +2,28 no ResNet-50 com ~80 params. Sai como variante barata do SE do A7.
- **3º, leitura densa:** GeM L3 (64) + GeM L4 (1280) = 1.344 dim < 2.000. Sem atalho a última camada piora no probe (Kolesnikov 1901.09005).
- **4º–5º, BlurPool / SoftPool** no lugar do max-pool: 0 params, ~+0,5–1,5 no ImageNet. Exigem recrescer do L1.
- **6º–8º:** Coordinate Attention (+2,0 vs +1,2 do SE no MobileNetV2), GCNet (+0,8), SimAM (0 params, mas não começa como identidade).
- **Regra de init:** porta `sigmoid` em zero corta a ativação pela metade e desloca o viés FLIM (lembrar −0,33 κ do imagenet_norm). Usar `2σ(z)` ou peso 0 com atalho.
- **Fora:** depthwise, MobileNetV2 e bloco ConvNeXt (`groups≠1` é recusado, `flim/progressive.py:63`); WS (reescreve o kernel SPiFiL); GN/LN não começam como identidade.
- **Onde entra:** módulo após a conv, como o BN do `e26349c` (`blocks[f"conv{n}"].insert`), ou tipo novo de pool em `build_encoder_from_arch` (`flim/arch.py:79`).
- **GeM em 1 linha:** eleva ao cubo, tira a média e a raiz cúbica; o forte fala mais alto, mas todos votam. Relatório: `discovery_conv_techniques.md`.

## 2026-11-01 17:15 [A6] teto de 20 GB por GPU (meta 19 GB): fila e pausas
- 17:15 PAUSA g20_l_bp03 (GPU0). O moliveira somava 20,3 GB na GPU0 (14,6 GB de train.py + 6,1 GB de workers ray::run_growth). A família era larvae e não havia protozoan na GPU0. O split1 já tinha L4 best_kappa, então saiu do grid.splits, que ficou [2,3]. A família voltou ao fim da fila. A gemq_g20_l_bp03 não foi tocada.

## 2026-11-01 17:30 [A8] deriva de ativação: SPiFiL cru → treinado, dentro × fora do objeto
- **Resposta:** o backprop mexe pouco nas pontas (L1 kernel cos 1,00; L4 1×1 cos 0,97) e muito no meio (L2/L3 kernel cos ~0,55).
- **O que ele muda:** apaga o contraste objeto × fundo do SPiFiL. Energia dentro/fora na L3: 2,9 → 1,0 (c1280 e w64); na L2: 2,0 → 1,2.
- **"Ativação vai para o fundo"?** Em energia relativa, sim. Em mudança relativa, a L3 muda mais fora (0,90) que dentro (0,74); L1/L2 mudam mais dentro.
- **L4 1×1×1280:** canais SPiFiL perdem mais o objeto (E in/out 0,84 → 0,42) que os aleatórios (0,94 → 0,73); os kernels andam igual (rel 0,24).
- **SPiFiL × aleatório (r1 × r1_rand):** o SPiFiL anda **mais** nos kernels (rel 1,1–1,2 contra 0,55–0,59) e perde mais contraste (L3 2,3 → 0,8 contra 2,1 → 1,4).
- **Deriva × κ val:** sem lei. Spearman de todos os pontos (n=36) ρ=−0,21 (p 0,21); o sinal troca por família e dataset.
- **Eixo bp (g20, eggs s1):** rel 0,40 / 0,43 / 0,56 para bp 0,1 / 0,3 / 1,0. Falta o κ dos g20.
- **Máscara sem rótulo = máscara GT, pixel a pixel:** as imagens já vêm com fundo constante; o objeto ocupa só 2–4 % da imagem.
- **FINAL:** com bp 0,1 as camadas velhas ainda andam ~+10–20 % (L2 rel 0,49 → 0,58).
- Saídas: `act_drift/` em 40 pontos (4 famílias × 9 + 4 g20). Relatório: `discovery_act_drift.md`.
