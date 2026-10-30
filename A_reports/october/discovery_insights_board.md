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
