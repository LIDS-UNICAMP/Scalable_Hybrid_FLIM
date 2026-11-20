# SimCLR / SimCLRv2 — regras de implementação fiéis e auditoria do nosso `simclr`

## Referência

- **SimCLR (v1)** — T. Chen, S. Kornblith, M. Norouzi, G. Hinton. *A Simple Framework for Contrastive Learning of Visual Representations*. ICML 2020. arXiv:2002.05709 (lido em ar5iv.labs.arxiv.org/html/2002.05709).
- **SimCLRv2** — T. Chen, S. Kornblith, K. Swersky, M. Norouzi, G. Hinton. *Big Self-Supervised Models are Strong Semi-Supervised Learners*. NeurIPS 2020. arXiv:2006.10029 (lido em ar5iv.labs.arxiv.org/html/2006.10029).
- **Código oficial** — github.com/google-research/simclr (branch `master`), arquivos lidos: `run.py` (flags), `data_util.py` (augmentations), `model.py`, `model_util.py` (projetor, schedule, weight decay), `objective.py` (loss), `lars_optimizer.py`, `tf2/model.py`, `README.md` (comandos CIFAR-10 / ImageNet).
- **Nossa implementação auditada** — `ga_search/modehb/ssl_trainers.py` (`nt_xent` L52, `AugConfig` L77, `make_views` L88, `_make_projector` L222, `train_global_ssl` L261, `build_opt` L322–336, ramo simclr L375–387), chamada em `ga_search/modehb/run_search.py` L712–717, hiperparâmetros em `ga_search/config/modehb/search_v3_seeded.yaml` L157–165.

Convenção: "§" = seção do artigo; "Tab."/"Fig." = tabela/figura; "App." = apêndice; arquivos de código com nome do arquivo.

## Resumo

SimCLR = duas vistas fortemente aumentadas da mesma imagem → encoder `f` → representação `h` → projetor MLP `g` → `z`; loss NT-Xent (cross-entropy sobre similaridades de cosseno / τ, com as outras 2N−2 vistas do batch como negativos). O que faz a receita funcionar, segundo os próprios ablations: (1) **composição crop + color distortion** (§3, Fig. 5); (2) **projetor não linear e avaliação ANTES dele** (§4.2, Fig. 8, Tab. 3); (3) **NT-Xent com τ ajustado e ℓ2-norm** (§5.1, Tab. 4/5); (4) **batch grande e/ou treino longo** (§5.2, Fig. 9, App. B.1); (5) **LARS + warmup + cosine** (§2.2, §2.3). O v2 acrescenta projetor de 3 camadas, fine-tuning a partir da 1ª camada do projetor e fila MoCo — ganhos pequenos ou nulos em CIFAR (v2 App. G, nota 5).

A nossa versão mantém o núcleo (NT-Xent correta, 2 vistas, projetor descartado, avaliação em `h`), mas diverge em quase todo o resto: batch 64, AdamW com lr fixo sem warmup/cosine, augmentation fraca e sem cor/grayscale/blur, projetor em gargalo (1280→128→64), weight decay efetivo zero e τ=0.2 fixo no código. Isso não é "SimCLR" no sentido do artigo; é um contrastivo de batch pequeno com augmentations fracas.

## Receita oficial (tabela)

| Item | SimCLR v1 (artigo) | SimCLRv2 (artigo) | Código oficial (`run.py`/README) | Fonte |
|---|---|---|---|---|
| Loss | NT-Xent, cos-sim / τ, 2N−2 negativos | idem (Eq. 1) | `add_contrastive_loss`: ℓ2-norm (`hidden_norm=True`), `logits_aa/bb` com −1e9 na diagonal, `loss = loss_a + loss_b` (soma das duas direções) | v1 §2.1 Eq. 1; `objective.py` |
| τ | 0.1 (melhor na Tab. 5: 0.05→59.7, **0.1→64.4**, 0.5→60.7, 1→58.0) | não declarado no texto do pré-treino | `temperature=0.1`; README CIFAR `0.5` | v1 Tab. 5; v1 App. B.9; `run.py` |
| ℓ2-norm | essencial (sem norm, τ=10/100 → 57.2/57.0) | — | `hidden_norm=True` | v1 Tab. 5 |
| Encoder `h` | ResNet-50, `h` = saída do GAP (2048-d) | ResNet-152 3×+SK | `resnet_depth=50` | v1 §2.3 |
| Projetor `g` | MLP 2 camadas: `W2·ReLU(W1·h)`, saída 128-d | MLP 3 camadas; largura do meio = largura da entrada | `proj_head_mode=nonlinear`, `num_proj_layers=3`, `proj_out_dim=128`; camadas do meio: Linear(d→d)+BN+bias+ReLU; última: Linear(d→128) **sem bias, com BN (center=False), sem ReLU**; init N(0, 0.01) | v1 §2.1/§2.3; v2 §2 item 2; `model_util.py` / `tf2/model.py` |
| Onde avaliar | em `h` (antes do projetor): `h` > `g(h)` por >10% | fine-tune a partir da 1ª camada do projetor (1%/10% rótulos); avaliação linear na entrada do projetor (§3.3) | `ft_proj_selector=0` (= descarta projetor) | v1 §4.2, Fig. 8, Tab. 3; v2 §3.1, §3.3 |
| Dim. de saída | indiferente com projetor não linear | — | 128 | v1 §4.2 Fig. 8 |
| Vistas | 2 por imagem, concatenadas no eixo do batch (2N) | idem | `num_transforms=2`, `tf.concat(features_list, 0)` | `model.py` |
| Random resized crop | área U[0.08, 1.0], aspecto [3/4, 4/3], resize 224 | igual ao v1 | `crop_and_resize(area_range=(0.08,1.0))` | v1 App. A; `data_util.py` |
| Flip | horizontal, p=0.5 (remover: 64.5→63.4) | igual | `random_flip_left_right` | v1 App. A |
| Color jitter | força s: brilho/contraste/saturação 0.8·s, matiz 0.2·s, aplicado com **p=0.8**, ordem aleatória; s=1 | igual | `color_jitter_strength=1.0`; `random_apply(p=0.8)`, `random_order=True`; v2 usa brilho multiplicativo | v1 App. A, Tab. 1; `data_util.py` |
| Grayscale | p=0.2 (independente do jitter) | igual | `random_apply(to_grayscale, p=0.2)` | v1 App. A; `data_util.py` |
| Gaussian blur | p=0.5, σ∈U[0.1, 2.0], kernel = 10% do lado (63.2→64.5) | igual | `use_blur=True`, `batch_random_blur(p=0.5)`, `kernel=height//10`; só no pré-treino | v1 App. A, Tab. 1; `model.py`/`data_util.py` |
| Força de cor | SSL precisa de cor forte: s=1/8→59.6, 1/4→61.0, 1/2→62.6, 1→63.2, 1+blur→64.5 | — | — | v1 §3.2 Tab. 1 |
| Batch | 4096 padrão (256–8192 estudados); 8192 → 16382 negativos | 4096 | `train_batch_size=512` (default do código); README ImageNet 4096, CIFAR 512 | v1 §2.2, §2.3, Fig. 9; v2 §3.1 |
| Otimizador | LARS (para todos os batches) | LARS, momentum 0.9 | `optimizer=lars`, momentum 0.9, `eeta=0.001`, classic momentum, sem Nesterov; exclui `batch_normalization`, `bias` do wd e da adaptação | v1 §2.2; v2 §3.1; `lars_optimizer.py`, `model_util.py` |
| LR | 4.8 = **0.3 × BS/256** (linear); alternativa 0.075·√BS (App. B.1) | 0.1·√BS = 6.4 | `learning_rate=0.3`, `learning_rate_scaling=linear`; README ImageNet `0.075` + `sqrt`; CIFAR `1.0` linear (BS 512 → 2.0) | v1 §2.3, App. B.1; v2 §3.1; README |
| Warmup | linear, 10 épocas | linear, 5% das épocas | `warmup_epochs=10` (convertido em passos) | v1 §2.3; v2 §3.1; `model_util.py` |
| Decaimento | cosine sem restart (até 0) | cosine | `cosine_decay(global_step − warmup, total − warmup)` | v1 §2.3; `model_util.py` |
| Weight decay | **1e-6** | **1e-4** | `weight_decay=1e-4`; BN e bias excluídos | v1 §2.3; v2 §3.1; `run.py`; `model_util.py` |
| Épocas | 100 (ablations); até 800/1000 nos melhores | 800 | `train_epochs=100`; README CIFAR 1000 | v1 §2.3, Fig. 9; v2 §3.1; README |
| BN | Global BN (stats sobre todos os devices; evita "information leakage") | global BN | `global_bn=True` | v1 §2.2 |
| Avaliação linear | classificador linear sobre a base congelada; Nesterov 0.9, lr 1.6 = 0.1×BS/256, 90 épocas, crop+flip; teste: resize 256 + center crop 224 | linear na entrada do projetor | README CIFAR: momentum, lr 0.1, wd 0, 100 épocas, BS 512, sem warmup | v1 §2.3, App. B.5/B.6; v2 §3.3 |
| Avaliação linear em transferência (datasets pequenos) | regressão logística multinomial com ℓ2, **L-BFGS**, **sem augmentation**, center crop 224, regularização escolhida em validação numa grade de 45 valores log-espaçados | — | — | v1 App. B.8 |
| CIFAR-10 | stem 3×3 stride 1, sem max-pool; crop Inception + flip, resize 32; **cor s=0.5, sem blur**; varredura lr {0.5,1.0,1.5}, τ {0.1,0.5,1.0}, BS 256–4096; τ=0.5 ótimo para >300 épocas, em qualquer batch; 94.0% linear (BS 1024) | 3 camadas no projetor e EMA não ajudaram em CIFAR | README: ResNet-18, BS 512, 1000 épocas, lr 1.0, wd 1e-4, τ 0.5, `use_blur=False`, `color_jitter_strength=0.5` (~91%) | v1 App. B.9, Fig. B.7/B.8; v2 App. G nota 5; README |
| Batch pequeno | com poucas épocas, batch grande tem vantagem significativa; com treino longo o gap encolhe/desaparece. BS 256, 100 ép.: 57.5 (lr linear) vs **62.8 (lr √)** | fila MoCo (64K, EMA 0.999): ~+1% (marginal porque o batch já dá muitos negativos) | README: "tuned for large batches … suboptimal with smaller batch"; "sqrt scaling … train better when smaller batch size is used" | v1 §5.2 Fig. 9, App. B.1 (Tab. B.1); v2 §2 item 3, App. D; README |

## Regras de implementação (checklist)

**Loss**
- [ ] NT-Xent: concatena `z1,z2` (2N×D), ℓ2-normaliza, `sim = z zᵀ / τ`, mascara a diagonal (−∞ ou −1e9), alvo de `i` é `i±N`; cross-entropy sobre os 2N−1 logits restantes (v1 Eq. 1; `objective.py`).
- [ ] τ configurável, nunca hard-coded; padrão 0.1 (ImageNet) / 0.5 (CIFAR, treino >300 épocas) (v1 Tab. 5, App. B.9).
- [ ] Usar as duas direções (a→b e b→a). O oficial **soma** (`loss_a + loss_b`); a média é equivalente a menos de um fator 2 (irrelevante para LARS/Adam, relevante só se houver termos somados à loss com pesos fixos).

**Projetor e avaliação**
- [ ] Projetor MLP não linear: `Linear(d→d)+BN+ReLU` (× L−1 camadas) + `Linear(d→128, sem bias)+BN(sem center)`; L=2 (v1) ou 3 (v2) (v1 §2.3; `model_util.py`).
- [ ] Largura oculta = dimensão de entrada `d` (sem gargalo) (v2 §2 nota 2; `mid_dim = hiddens.shape[-1]`).
- [ ] Projetor recebe as **2N** vistas numa única chamada (BN com estatística conjunta das duas vistas) (`model.py`: concat no eixo do batch).
- [ ] Avaliação **sempre em `h`**, antes do projetor; projetor descartado após o pré-treino (v1 §4.2, Tab. 3; `ft_proj_selector=0`).

**Augmentations (por vista, independentes)**
- [ ] Random resized crop: área U[0.08, 1], aspecto log-U[3/4, 4/3], resize para o tamanho de entrada (v1 App. A).
- [ ] Flip horizontal p=0.5.
- [ ] Color jitter com p=0.8, ordem aleatória: brilho 0.8s, contraste 0.8s, saturação 0.8s, matiz 0.2s; s=1 (ImageNet) / 0.5 (CIFAR).
- [ ] Grayscale p=0.2.
- [ ] Gaussian blur p=0.5, σ∈U[0.1, 2.0], kernel = 10% do lado (só quando a resolução justifica — removido em CIFAR 32 px).
- [ ] Composição obrigatória crop **+** cor: nenhuma transformação isolada basta; sem cor a rede usa o histograma de cor como atalho (v1 §3, Fig. 5, Fig. 6).

**Otimização**
- [ ] Batch o maior possível; reportar o nº de negativos (2B−2).
- [ ] LARS (momentum 0.9, η=0.001), BN e bias fora do weight decay e da adaptação por camada (`lars_optimizer.py`, `model_util.py`).
- [ ] LR de pico = 0.3·B/256 (linear) ou 0.075·√B (√; preferível para batch pequeno, App. B.1).
- [ ] Warmup linear de 10 épocas (v1) ou 5% do treino (v2), em **passos**; depois cosine até 0.
- [ ] Weight decay 1e-6 (v1) ou 1e-4 (v2 / código / CIFAR), efetivamente aplicado.
- [ ] Épocas = passadas reais pelo dataset; 100 é o mínimo dos ablations, 800–1000 o regime dos melhores resultados.
- [ ] BN global (em 1 GPU isto é automático desde que as 2 vistas passem juntas).

**Avaliação linear**
- [ ] Encoder congelado, modo eval, features de `h` com transformação de teste (resize + center crop, sem augmentation).
- [ ] Protocolo de transferência para datasets pequenos: logística ℓ2 + L-BFGS, regularização escolhida na validação (v1 App. B.8).

## Divergências

Legenda de prioridade: **P1** = muda o método/resultado; **P2** = relevante; **P3** = detalhe.

| # | Item | Artigo / código oficial | Nosso (`ssl_trainers.py` + `run_search.py` + `search_v3_*.yaml`) | Ação para ficar fiel |
|---|---|---|---|---|
| 1 | **Batch / negativos** (P1) | 4096 (8190 negativos); 256 é o menor estudado e já perde ~7 p.p. sem compensação (v1 Fig. 9, Tab. B.1) | `batch_size: 64` "FIXO" → 126 negativos por âncora; nada compensa (sem lr √, sem τ ajustado, sem fila) | Batch ≥ 256 (medir memória); se não couber, 128 + lr √ + treino longo; nunca usar acumulação de gradiente como substituto (não aumenta negativos). Config própria para o SimCLR fiel, sem mexer no `batch_size` FIXO da busca. |
| 2 | **Otimizador e schedule** (P1) | LARS, pico 0.3·B/256 (ou 0.075·√B), warmup 10 ép., cosine até 0 (v1 §2.3) | `AdamW` com 2 grupos de lr **fixos** (`lr_old=1e-4`, `lr_new=1e-3`) × `conv_lr_scale`; sem warmup, sem cosine (`build_opt` L322–336; docstring "Learning rates are fixed") | Implementar LARS (η=0.001, momentum 0.9, exclusão BN/bias) + `LambdaLR` warmup linear → cosine em passos; manter os multiplicadores por grupo (proteção SPiFiL) **sobre** o lr de pico. |
| 3 | **Augmentations** (P1) | crop área [0.08,1]; flip-H; jitter s=1 (0.8/0.8/0.8/0.2) com p=0.8 e ordem aleatória; grayscale p=0.2; blur p=0.5 σ∈[0.1,2] k=10% (v1 App. A) | `AugConfig` (L77): crop área **[0.3,1]**; flips H+V; brilho aditivo ±0.2 e contraste ±0.2 aplicados **sempre** (p=1), `channel_jitter` ±0.05; **sem saturação, sem matiz, sem grayscale, sem blur**; ordem fixa | Adicionar `simclr_aug` (abaixo, em LAB), com `crop_scale=(0.08,1)`, jitter s com p=0.8 e ordem aleatória, grayscale p=0.2, blur p=0.5. Manter flip vertical como adaptação de microscopia. |
| 4 | **Projetor** (P1) | `d→d` (BN, ReLU) `→128` (sem bias, BN); 2 camadas (v1) / 3 (v2); 2N vistas numa chamada | `_make_projector(d, hidden=128, out=64)` (L222): `1280→128` (gargalo 10×) `→64`, sem BN na saída, com bias; `proj(hs[0]), proj(hs[1])` **separados** (L378) → BN com estatística por vista | `_make_projector(d, hidden=d, out=128, n_layers=2)` com `BatchNorm1d(128, affine=True)` sem bias na última camada (center=False ≈ `bias=False` no BN); chamar `proj(torch.cat(hs))` e depois `split`. |
| 5 | **Weight decay** (P2) | 1e-6 (v1) / 1e-4 (v2, código, CIFAR); BN e bias excluídos | `train_global_ssl(weight_decay=0.0)` e o `run_search.py` L712–717 **não repassa** `training.weight_decay: 1e-4` do YAML → wd efetivo **0**; sem exclusão de BN/bias | Repassar `weight_decay` do config; grupo separado com wd=0 para `bias`/`BatchNorm*`. |
| 6 | **Temperatura** (P2) | 0.1 (ImageNet), 0.5 (CIFAR >300 ép.) | `tau=0.2` default de `train_global_ssl`/`nt_xent`, não exposto no config nem no genoma | Expor `tau` no YAML; 0.5 para o regime de dataset pequeno e treino longo; varrer {0.1, 0.2, 0.5}. |
| 7 | **Épocas / "época"** (P2) | épocas = passadas pelo dataset; 100–1000 | `ssl_epochs` = fidelidade do DEHB; 1 "época" = ≤2048 imagens sorteadas (`max_images_per_epoch`), ou seja 32 passos com B=64 | Medir orçamento em **passadas reais** e passos; warmup/cosine calculados sobre `total_steps`; alvo ≥300 passadas (regime em que τ=0.5 é estável, App. B.9). |
| 8 | **Avaliação linear** (P2 — divergência aceita) | logística ℓ2 L-BFGS, C escolhido em validação, sem aug, center crop (v1 App. B.8) | SVM linear `C=100` fixo (régua do repo), às vezes com readout `cov_head` + `StandardScaler` (`svm_fitness`) | **Não mudar** (regra do usuário: SVM original C=100). Garantir que a avaliação é em `h` (já é) e com transformação sem augmentation. Se quiser comparar com o artigo, logística L-BFGS só como diagnóstico paralelo, nunca no lugar do κ do SVM C=100. |
| 9 | **Gene `projector`** (P3) | — | Para `ssl='simclr'` o projetor **sempre** existe (L327); o gene `projector=True` só muda onde o regularizador atua (`reg_in=[z1,z2]`, L384) | Documentar; não há "SimCLR sem projetor" na nossa busca (o artigo mostra que sem projetor perde >10%, §4.2). |
| 10 | **Escala da loss** (P3) | `loss_a + loss_b` (soma) | `F.cross_entropy` média sobre 2N âncoras = (loss_a+loss_b)/2 | Indiferente com LARS/AdamW; só documentar, porque muda o peso relativo de `sigreg_lambda·R` quando há regularizador somado. |
| 11 | **Extras fora do SimCLR** (P3 — divergências deliberadas) | — | proteção dos kernels SPiFiL (`conv_lr_scale`, `ElementProtect`), WiSE-FT pós-treino (`wise_alphas`), regularizador SIGReg opcional, AMP bf16 | Manter (princípio Hybrid SPiFiL: o backprop soma, não desfaz), mas rotular o método como "SimCLR + proteção SPiFiL", não como SimCLR puro. |

Pontos que já estão **conformes**: NT-Xent correta (ℓ2-norm, diagonal mascarada com −∞, alvo `i±N`, 2N−2 negativos — L52–61); exatamente 2 vistas (L294); vistas independentes por imagem; projetor descartado ao fim (`extra.clear()`, L474); avaliação no embedding da cabeça `h` (antes do projetor); encoder + cabeça numa só passada com as 2N vistas concatenadas (L370).

## Adaptação

Caso alvo: CNN rasa (3–5 blocos conv 5×5) inicializada por SPiFiL, cabeça `EmbeddingHead` (1×1 conv + pooling → `h` com d ≥ 1280), imagens 224 px em LAB `[0,1]` (LABNorm2 do pyift), datasets de 1k–10k imagens, uma GPU de 48 GB.

**Regra geral de adaptação:** *mudar só o que o artigo mostra ser dependente de escala (batch, lr, τ, épocas, blur por resolução); manter tudo o que é estrutural (NT-Xent, 2 vistas, projetor não linear sem gargalo, avaliação em `h`, composição crop+cor); toda adaptação de domínio (flip vertical, cor em LAB, proteção SPiFiL) é declarada e ablada contra a versão fiel.*

Regras concretas:

1. **Batch.** Maior potência de 2 que caiba em ~80% da memória livre, medida com `torch.cuda.max_memory_allocated` em 1 passo com B ∈ {128, 256, 512}. Alvo B=256 (2B=512 vistas de 224 px numa CNN rasa). Piso B=128. Sem acumulação de gradiente. Se ficar em B ≤ 128, considerar fila de negativos (MoCo, v2 §2 item 3) como extensão opcional: o v2 mede só ~1% com batch 4096, e o ganho com batch pequeno é uma hipótese a testar.
2. **LR.** LARS (η=0.001, momentum 0.9) com pico `lr = 0.075·√B` (B=128 → 0.85; B=256 → 1.2), porque com batch pequeno a escala √ ganha da linear (v1 Tab. B.1: 62.8 vs 57.5 em B=256). Proteção SPiFiL: multiplicar o pico do grupo de convs herdados por `conv_lr_scale` (ou `lr_old/lr_new`). Com LARS, cada passo muda os pesos numa fração `lr·η` de ‖w‖, então essa proteção continua sendo um teto relativo.
3. **Schedule.** Warmup linear em 5–10% dos passos totais (v2: 5%; v1: 10 de 100 épocas), no mínimo 1 passada. Depois cosine até 0. Tudo calculado em passos a partir de `total_steps = passadas × |D| / B`. O conceito de "época de 2048 imagens" não entra no schedule.
4. **Épocas.** Contar passadas reais pelo dataset: mínimo 300 (regime da App. B.9), alvo 800–1000 (README CIFAR = 1000; v2 = 800). Com B=256, 1k imagens × 1000 passadas ≈ 3,9k passos e 10k imagens × 1000 ≈ 39k passos. Como o artigo não estuda datasets de 1k imagens, monitorar κ de validação para detectar sobreajuste do SSL.
5. **τ.** 0.5 (regime CIFAR: dataset pequeno, treino longo, estável em qualquer batch, App. B.9). Varredura {0.1, 0.2, 0.5}.
6. **Weight decay.** 1e-4 (código, v2, CIFAR), efetivamente repassado, com BN/bias excluídos.
7. **Projetor.** 2 camadas (o v2 App. G diz que 3 camadas não ajudaram em CIFAR): `Linear(d,d)+BN+ReLU → Linear(d,128,bias=False)+BN`, numa única chamada com as 2N vistas. Avaliação em `h`.
8. **Augmentations em LAB.** Implementar sobre o tensor LAB `[0,1]`, já que o loader `ift_lab` não passa por RGB:
   - crop área U[0.08, 1], aspecto log-U[3/4, 4/3]. Se objetos pequenos (ovos/cistos) saírem do crop com frequência, subir o piso de área usando `scripts/measure_object_coverage.py` (ou `src/transforms/object_crop.py`) e declarar a mudança. O nosso 0.3 só fica como ablação;
   - flip H p=0.5 + flip V p=0.5 (microscopia não tem orientação canônica; é adaptação declarada). Rot90 é opcional;
   - jitter com p=0.8, ordem aleatória: brilho = `L·U[1−0.8s, 1+0.8s]` (multiplicativo, como `impl=simclrv2`); contraste = escala de L em torno da média de L; saturação = escala de (a,b) em torno do **cromático neutro** por U[1−0.8s, 1+0.8s]; matiz = rotação de (a,b) em torno do neutro por ângulo U[−0.2s, 0.2s]·2π;
   - grayscale p=0.2: (a,b) ← cromático neutro. O valor neutro tem de ser **medido** (passar um pixel cinza pelo `LABNorm2` do pyift), não suposto 0.5;
   - blur gaussiano p=0.5, σ∈U[0.1, 2.0] px, kernel ímpar ≈ 10% do lado (23 px em 224). O CIFAR remove o blur por causa da resolução de 32 px, não por ser um dataset pequeno, então em 224 px a versão fiel mantém o blur;
   - força s: o fiel é s=1 em 224 px. Como cor pode ser informativa da classe em parasitas (coloração), rodar s ∈ {0.5, 1} e escolher pelo κ de validação. s=0.5 é o valor da receita de dataset pequeno (CIFAR).
9. **Avaliação.** Manter a régua do repo (SVM linear C=100 sobre `h`, transformação de teste sem augmentation; regra do usuário). Para comparar com o artigo, logística ℓ2 L-BFGS com C escolhido na validação, só como diagnóstico paralelo.
10. **Extras do Hybrid SPiFiL** (proteção de kernels, WiSE-FT, SIGReg): ficam, mas como camada sobre o SimCLR fiel. Ablação mínima: `SimCLR-fiel` × `SimCLR-fiel + proteção` × `SimCLR-fiel + proteção + WiSE`.

Configuração-alvo resumida (B=256): `tau=0.5`, LARS momentum 0.9 η=0.001, `lr_peak=1.2` (√), `warmup=5–10% dos passos`, cosine→0, `wd=1e-4` (sem BN/bias), projetor `d→d→128` com BN, 2 vistas, crop [0.08,1], flip H+V, jitter s∈{0.5,1} p=0.8, grayscale p=0.2, blur p=0.5, ≥300 passadas (alvo 800–1000).

## Testes de conformidade

Testes unitários sugeridos para `tests/` (sem GPU, exceto o 10):

1. **NT-Xent = oficial.** Portar `add_contrastive_loss` (com `LARGE_NUM=1e9`, `loss_a + loss_b`) para torch e verificar `nt_xent(z1,z2,τ) ≈ 0.5·oficial` com tolerância de 1e-5, para τ ∈ {0.1, 0.5} e N ∈ {4, 64}.
2. **Valor de referência.** Com `z` aleatório em dimensão alta, `nt_xent ≈ log(2N−1)`; com `z2=z1` e linhas ortogonais, loss → 0 quando τ → 0.
3. **τ propagado.** O `tau` do YAML chega a `nt_xent` (monkeypatch registrando o argumento).
4. **Projetor.** As camadas são `Linear(d,d)`, `BN`, `ReLU`, `Linear(d,128,bias=False)`, `BN`; hidden == d; saída 128. Um hook mostra **uma** chamada com 2B linhas por passo.
5. **Avaliação antes do projetor.** Após `train_global_ssl`, `extra` está vazio, e o embedding passado ao SVM é `head(encoder(x))` em `eval()` (shape d, não 128).
6. **Estatística das augmentations** (10⁴ amostras com seed): área do crop ~U[0.08,1] (KS p>0.01); log-aspecto ~U[log 3/4, log 4/3]; fração de flips H ≈ 0.5; fração com jitter ≈ 0.8; fração grayscale ≈ 0.2 (todos os pixels (a,b) == neutro); fração com blur ≈ 0.5, σ ∈ [0.1, 2.0], kernel = 10% do lado; fatores de brilho/contraste/saturação dentro de [1−0.8s, 1+0.8s] e matiz em ±0.2s; ordem do jitter variando entre amostras.
7. **Schedule.** `lr(0)=0` (ou pequeno), máximo exatamente no fim do warmup e igual a `0.075·√B` (ou `0.3·B/256`) × multiplicador do grupo, monótono cosine até ≈0 no último passo; a razão entre grupos é constante ao longo do treino.
8. **Weight decay.** Os grupos com `bias`/`BatchNorm` têm `weight_decay=0`; os demais têm o valor do YAML (pega o bug atual do wd não repassado).
9. **LARS.** Num parâmetro de brinquedo, um passo bate com o cálculo manual `trust = η·‖w‖/‖g + wd·w‖` (1.0 se uma das normas for 0); BN/bias sem adaptação.
10. **Smoke** (GPU, ~200 passos, 1 dataset/split): loss cai abaixo de `log(2B−1)`, `ssl/pos_sim` sobe, `ssl/neg_sim` fica perto de 0, sem passos não finitos, pico de memória registrado e < 80% da GPU.
11. **Negativos.** O log registra `2B−2` e B ≥ 128 no modo fiel (assert na config `simclr_faithful`).
12. **Determinismo.** Mesma seed → mesmas vistas e mesma loss nos 5 primeiros passos.
