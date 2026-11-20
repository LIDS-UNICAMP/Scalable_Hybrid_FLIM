# BYOL: regras de implementação fiéis ao artigo

> Fontes lidas: fonte LaTeX do arXiv:2006.07733 (v3: `method.tex`, `experiments.tex`, `ablation.tex`, `appendix.tex`, tabelas) e o código oficial `google-deepmind/deepmind-research/byol` (JAX/Haiku: `byol_experiment.py`, `configs/byol.py`, `utils/{augmentations,schedules,optimizers,helpers,networks}.py`). Nosso código: `ga_search/modehb/ssl_trainers.py`.
>
> Seções: §3.1 descrição, §3.2 intuições, §3.3 implementação, §4 experimentos, §5 ablações. Apêndices: A algoritmo, B augmentações (Tab. 6), C avaliação ImageNet, D transferência, E Places365, F ablações extras, G batch menor, H detalhes da Eq. 3, I predictor quase ótimo, J hiperparâmetros e pseudo-código. Quando a numeração das tabelas do apêndice é ambígua no LaTeX, cito o rótulo ou a legenda.
>
> Legenda: **[FIEL]**, **[ADAPT]**, **[HIP]**, como em `dino.md`.

## Referência

- Grill, Strub, Altché, Tallec, Richemond, Buchatskaya, Doersch, Pires, Guo, Azar, Piot, Kavukcuoglu, Munos, Valko. *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning*. NeurIPS 2020. arXiv:2006.07733.
- Código oficial: https://github.com/google-deepmind/deepmind-research/tree/master/byol

## Resumo

Duas redes. A **online** (θ) tem encoder f, projetor g e **predictor q**. A **target** (ξ) tem f e g, sem predictor, e é a **EMA** da online. A online prevê a projeção da target sobre a **outra** vista. A perda é o MSE entre vetores ℓ2-normalizados (= 2 − 2·cos) e é simetrizada. Não há negativos (§3.1). O colapso é evitado pela assimetria predictor + stop-grad + EMA. A hipótese dos autores é que a EMA mantém o predictor quase ótimo (§3.2; App. I). No fim, **só o encoder online f_θ** é mantido e y = f_θ(x) é a representação (§3.1, legenda da Fig. 2).

**CNN e poucos dados (o que o artigo mostra):**
- O BYOL é nativo de **ResNet**: 74,3 % linear no ImageNet com ResNet-50, 1000 ép. (Tab. 1).
- **Batch:** cai pouco de 4096 até 256 (72,5 → 71,8). Cai de verdade em 128 (69,6) e em 64 (59,7). Os autores atribuem a queda **ao BN do encoder** (§5 "Batch size", nota de rodapé; App. F, tabela "Influence of the batch size"). Isso importa para o nosso bs 64.
- **Augmentações:** o BYOL é bem mais robusto que o SimCLR a retirar cor. Sem cor: −9,1 contra −22,2. Só crop: 59,4 contra 40,3 (§5, Fig. 3b; App. F, tabela "Ablation on image transformations"). É relevante para microscopia, onde a cor carrega diagnóstico e o jitter forte pode ser nocivo.
- **Multi-crop não funciona direto no BYOL** (relato do DINO, App. E e Tab. 14 linha 10: 71,4 → 64,8, inclusive em ResNet-50).
- **Nenhum pré-treino em dataset pequeno.** O que existe é transferência ImageNet → 12 datasets pequenos (Food, CIFAR, Flowers, Pets, DTD…) por **regressão logística ℓ2 sobre features congeladas** (Tab. 3; App. D.2). Esse é o protocolo mais próximo do nosso SVM linear.

## Receita oficial (tabela)

| Item | Valor (ResNet-50, 1000 ép.) | Fonte |
|---|---|---|
| Perda | L = ‖q̄(z_θ) − z̄′_ξ‖² = 2 − 2·⟨q, z′⟩/(‖q‖‖z′‖), por amostra | §3.1 Eq. 2; `helpers.regression_loss` |
| Simetrização | L_BYOL = L(v → v′) + L̃(v′ → v), com v′ na online e v na target; média sobre o batch | §3.1; `byol_experiment.loss_fn` (soma das duas e depois `mean`) |
| Stop-grad | sobre z′_ξ (gradiente só em θ) | §3.1, Fig. 2 |
| Target EMA | ξ ← τξ + (1−τ)θ após cada passo | Eq. 1 |
| Schedule de τ | τ = 1 − (1 − τ_base)·(cos(πk/K) + 1)/2, com **τ_base = 0,996** → 1 | §3.3; `schedules.target_ema` |
| Presets de τ_base por orçamento | 40 ép.: 0,97; 100: 0,99; 300: 0,99; 1000: 0,996 | `configs/byol.py` (`_EMA_PRESETS`) |
| Encoder | ResNet-50 v1 pós-ativação; y = saída do avg-pool final (2048) | §3.3 |
| Projetor g | Linear(2048 → **4096**) → **BN** → ReLU → Linear(4096 → **256**), sem BN na saída | §3.3 |
| Predictor q | mesma arquitetura: Linear(256 → 4096) → BN → ReLU → Linear(4096 → 256), só na online | §3.3; `predictor_hidden_size=4096` |
| BN | decay 0,9, eps 1e-5, **sincronizado entre réplicas**. A target roda com `is_training=True` (estatísticas do batch) | `configs/byol.py` (`bn_config`); `loss_fn` |
| Augmentações T / T′ | RRC área [0,08, 1], razão log-uniforme [3/4, 4/3], **bicúbica** 224; flip 0,5; jitter de cor p = 0,8 (brilho 0,4, contraste 0,4, saturação 0,2, hue 0,1, **ordem aleatória**); grayscale p = 0,2 (luma 0,2989/0,5870/0,1140); blur p = **1,0 / 0,1** (kernel 23 × 23 para 224, isto é, lado/10; σ ∈ [0,1, 2,0]); solarize p = **0 / 0,2** (x se x < 0,5, senão 1 − x); normalização ImageNet após as augmentações | App. B, Tab. 6; `augmentations.augment_config` |
| Otimizador | **LARS**, momentum 0,9, η (trust) = 1e-3. **Viés e BN fora da adaptação LARS e do wd** | §3.3; `optimizers.exclude_bias_and_norm` |
| lr | base **0,2** × bs/256; warm-up linear de **10 ép.**; cosseno até **0** (sem `min_lr`) | §3.3; `schedules.learning_schedule` |
| Presets de lr | 40 / 100 ép.: 0,45; 300: 0,3; 1000: 0,2 | `configs/byol.py` (`_LR_PRESETS`) |
| Weight decay | **1,5e-6** (1000 ép.); 1e-6 para ≤ 300 ép.; constante | §3.3; `_WD_PRESETS` |
| Batch | **4096** (512 núcleos TPU v3) | §3.3 |
| Batch 512 | lr base **0,4**, τ_base **0,9995**, wd 1,5e-6, 1000 ép. → 73,7 % | App. G |
| Épocas | 1000 (ablações: 300 com lr 0,3, wd 1e-6, τ_base 0,99) | §3.3; §5 |
| Monitoramento | classificador linear online sobre stop-grad(y), só para log | `loss_fn` (`classif_loss`) |
| Linear (ImageNet) | encoder e BN congelados; SGD Nesterov, momentum 0,9, **80 ép.**, bs 1024, sem wd; lr ∈ {0,4, 0,3, 0,2, 0,1, 0,05} escolhido num val local de 10 009 imagens; treino com RRC + flip, teste com resize 256 + crop central 224 | App. C.1 |
| Linear (transferência) | regressão logística multinomial ℓ2 com **L-BFGS** em features congeladas, **sem augmentação**; λ em 45 valores log de 1e-6 a 1e5 escolhido na validação; retreino em treino + val | App. D.2 |
| k-NN | **não usado** no artigo do BYOL | — |

Ablações que justificam as regras (300 ép.; §5 e App. F): target constante e aleatória: 18,8; τ_base 0,999 / **0,99** / 0,9: 69,8 / **72,5** / 68,4; τ = 0 (stop-grad da online): 0,3 (Tab. 5a). Sem predictor e sem negativos: 0,2 (Tab. 5b). Profundidade 2 do projetor e do predictor é a melhor (App. F, "Projector and predictor depth"). Dimensão de projeção de 64 a 512 fica estável, com 72,2–72,6 (App. F). lr base 0,3 é o ótimo a 300 ép., e a faixa 0,2–0,4 é robusta (App. F). wd de 5e-7 a 1e-6 é o ótimo; com wd = 0 o treino fica instável (App. F). Sem EMA, com τ = 0, funciona se o lr do predictor for 2–10× o do resto, chegando a 66,6 (App. I).

## Regras de implementação (checklist)

- [ ] **[FIEL]** Online = encoder + `EmbeddingHead` (y de 1280 d, a régua do SVM) + projetor g + predictor q. Target = `deepcopy`(encoder, head, g), **sem q**, com `requires_grad_(False)`, inicializada idêntica.
- [ ] **[FIEL]** Duas vistas por imagem com **distribuições diferentes**: T (blur p = 1, solarize p = 0) e T′ (blur p = 0,1, solarize p = 0,2). Não usar a mesma `AugConfig` para as duas.
- [ ] **[FIEL]** Perda por amostra: `2 - 2 * cos(q(z1), sg(z2'))` mais o termo trocado, com média sobre o batch. As normas são feitas **dentro** da perda; q e z não são normalizados na arquitetura.
- [ ] **[FIEL]** O predictor existe só na online. O gradiente nunca passa pela target.
- [ ] **[FIEL]** A target roda em **modo train** (BN com estatísticas do batch), como o `is_training=True` oficial, sob `no_grad`.
- [ ] **[FIEL]** EMA **depois** de `opt.step()`, por passo, com τ_k = 1 − (1 − τ_base)(cos(πk/K) + 1)/2 e K = total de passos. A EMA cobre encoder, head e projetor.
- [ ] **[FIEL]** BN no projetor e no predictor (camada escondida); nenhum BN ou ReLU na última Linear.
- [ ] **[FIEL]** lr: warm-up linear de 0 até base·bs/256 em 10 ép.; depois cosseno até 0. wd constante. Viés e BN (todos os 1-D) fora do wd **e** fora da adaptação LARS.
- [ ] **[FIEL]** Com gradiente acumulado (bs efetivo = N × bs), atualizar a target **uma vez a cada N passos**, depois da online (§5 "Batch size").
- [ ] **[FIEL]** No fim, devolver o **encoder online** (não a target). Descartar g e q.
- [ ] **[ADAPT]** Logar por época: ‖y‖, a variância por dimensão de z̄ (colapso → 0), cos(q, z′) médio e τ_k.

## Divergências e o que falta

**Entre o artigo e o código oficial:** são coerentes nos pontos centrais. Notas:
1. O lr e o τ_base dependem do orçamento de épocas (presets no `configs/byol.py`). O texto principal cita só 0,2 / 0,996 (1000 ép.), e as ablações usam 0,3 / 0,99 (300 ép.).
2. O blur do código usa kernel ≈ lado/10 (`blur_divider=10`). O texto fala em 23 × 23 para 224, o que é consistente.
3. O texto não menciona o classificador linear online de monitoramento, que é só log (stop-grad) e não altera o encoder.
4. O BN da target em modo train é um detalhe que aparece só no código.

**Entre o nosso `ssl_trainers.py` e o BYOL:**
1. Não há `byol` em `_SSL` (l. 144) nem no gene `ssl`.
2. Não há rede target/EMA nem predictor. O `_make_projector` (l. 222) é Linear(d, 128) → BN → ReLU → Linear(128, 64); o do BYOL é 4096 → 256.
3. `make_views` (l. 88) aplica a **mesma** distribuição às duas vistas. Faltam a assimetria T/T′, blur, solarização, saturação/hue, grayscale e a ordem aleatória do jitter. A RRC usa escala (0,3, 1), contra (0,08, 1) no BYOL.
4. AdamW com lr **fixo** e wd único em tudo (l. 322–336), contra LARS com warm-up + cosseno e exclusão de viés/BN.
5. Batch **64** (`search.yaml:93`). É justamente o ponto em que o BYOL cai para 59,7 (App. F) por causa do BN.
6. `extra.clear()` devolve a online. Isso **está correto** para o BYOL, ao contrário do DINO e do I-JEPA.
7. Avaliação: o SVM C = 100 do projeto (R1) é o análogo direto da regressão logística ℓ2 sobre features congeladas (App. D.2), com a diferença de que o C é fixo e não varrido. Manter R1.

## Adaptação

### Pseudo-código em `train_global_ssl` (ramo `ssl == "byol"`)

```python
# kwargs opt-in: ssl="byol", reg="none", schedule="official"|"fixed",
#   byol_cfg=dict(proj_hidden=4096, proj_out=256, pred_hidden=4096, tau_base=0.996,
#                 aug_T=AugT, aug_Tp=AugTprime, opt="lars", base_lr=0.2, wd=1.5e-6,
#                 warmup_ep=10, accum=1)

def build_opt_byol(x0):
    d = head(encoder(x0[:2])).shape[1]                                     # 1280 = y
    extra["byol_proj"] = MLP(d, 4096, 256, bn=True).to(device)             # [FIEL]
    extra["byol_pred"] = MLP(256, 4096, 256, bn=True).to(device)           # [FIEL] só online
    T = {"enc": deepcopy(encoder), "head": deepcopy(head), "proj": deepcopy(extra["byol_proj"])}
    for m in T.values(): m.train(); [p.requires_grad_(False) for p in m.parameters()]
    groups = split_param_groups(encoder, head, extra.values(), new_names, lr_old, lr_new)
    groups = split_decay(groups)                       # viés/1-D: wd=0 e lars_exclude=True [FIEL]
    groups = _scale_conv_groups(groups, encoder, conv_lr_scale)            # proteção SPiFiL [ADAPT]
    for g in groups: g["lr_mult"] = g["lr"] / lr_new
    opt = LARS(groups, momentum=0.9, trust_coef=1e-3)  # [FIEL] (AdamW = braço [ADAPT])
    K = epochs * ipe // accum                          # passos de otimização
    lr_s = warmup_cosine(base_lr * (B * accum) / 256, warmup=warmup_ep * ipe // accum, total=K, final=0.0)
    tau_s = lambda k: 1 - (1 - tau_base) * (math.cos(math.pi * k / K) + 1) / 2

def byol_side(xa, xb):                                 # perda L(xa -> xb)
    with amp():
        q = extra["byol_pred"](extra["byol_proj"](head(encoder(xa))))
    with torch.no_grad(), amp():
        zt = T["proj"](T["head"](T["enc"](xb)))        # target em modo train
    with fp32():
        return (2 - 2 * F.cosine_similarity(q.float(), zt.float(), dim=-1)).mean()

k = 0
for ep in range(epochs):
    for bi, x in enumerate(loader):                    # drop_last=True (BN)
        v1, v2 = aug(x, AugT), aug(x, AugTprime)       # T: blur 1,0 / solar 0 ; T': blur 0,1 / solar 0,2
        loss = byol_side(v1, v2) + byol_side(v2, v1)   # simetrizada [FIEL]
        (loss / accum).backward()
        if (bi + 1) % accum: continue                  # acumulação: a target só anda a cada N [FIEL §5]
        for g in opt.param_groups: g["lr"] = lr_s[k] * g["lr_mult"]
        opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            tau = tau_s(k)
            for po, pt in zip(chain(encoder, head, extra["byol_proj"]).parameters(),
                              chain(T["enc"], T["head"], T["proj"]).parameters()):
                pt.mul_(tau).add_((1 - tau) * po.detach())
        k += 1
        log(loss, tau, lr, std_per_dim(F.normalize(zt)), cos_mean)

extra.clear()                                          # fica a ONLINE f_θ [FIEL]
```

### Valores sugeridos (CNN rasa SPiFiL, 224 px, 1–10 mil imagens, GPU de 48 GB)

| Item | Fiel | Adaptação proposta | Marca |
|---|---|---|---|
| Representação y | avg-pool do ResNet-50 (2048) | saída da `EmbeddingHead` (≥ 1280) | [ADAPT] |
| Projetor / predictor | 4096 → 256 com BN | igual. Se a memória do BN apertar, 2048 → 256 (proj 64–512 é robusto, App. F) | [FIEL] / [ADAPT] |
| Batch | 4096 (512 no App. G) | **≥ 256** reais, ou 64 × `accum = 4` com a target atualizada a cada 4 passos. Mas **o BN do encoder continua vendo 64**: a queda do App. F vem do BN, e a acumulação não resolve isso. Preferir bs real 256 (cabe em 48 GB) | [FIEL] / [ADAPT] |
| BN no encoder SPiFiL | BN sincronizado | se o genoma usar BN/ZCA/DBN com bs pequeno, considerar GroupNorm como braço identificado | [HIP] |
| Otimizador | LARS, lr base 0,2–0,45 conforme o orçamento, wd 1e-6–1,5e-6, warm-up 10, cosseno → 0 | LARS fiel. AdamW como braço, mantendo `lr_old`/`lr_new`/`conv_lr_scale` (Hybrid SPiFiL: o SSL soma ao SPiFiL, nunca o desfaz) | [FIEL] / [ADAPT] |
| τ_base | 0,996 (1000 ép.); 0,99 (100–300); 0,97 (40) | escolher pelo **orçamento de passos**: com poucos milhares de passos, começar em **0,99** (preset 100–300 ép.) e testar 0,97 e 0,996 na validação | [FIEL]-preset / [HIP] |
| Augmentações | RGB, Tab. 6 | LAB [0, 1]: as mesmas operações reescritas em LAB (ver `dino.md`). Como o BYOL tolera tirar a cor (App. F), propor um braço **"cor fraca"** para microscopia: jitter só em L, sem hue | [ADAPT] / [HIP] |
| RRC | (0,08, 1) | fiel (0,08, 1). Em imagens com objeto pequeno, testar o crop condicionado ao objeto (`src/transforms/object_crop.py`) como braço | [FIEL] / [HIP] |
| Multi-crop | não usar | não usar (DINO App. E) | [FIEL] |
| Épocas | 300–1000 | por passos (≥ 3–5 mil), como no DINO | [HIP] |
| Avaliação | linear (ImageNet), log-reg ℓ2 (transferência) | **SVM C = 100** (R1); κ de teste só no fim | [ADAPT] |

## Testes de conformidade

Colocar em `tests/modehb/test_ssl_byol.py`.

1. **Perda:** `2 - 2*cos(q, z)` é igual a `((q/‖q‖ - z/‖z‖)**2).sum(-1)` em 1e-6, e a perda simetrizada é a soma dos dois lados.
2. **Stop-grad:** depois de `backward`, todo parâmetro da target tem `grad is None`, e o predictor tem grad ≠ 0.
3. **Predictor só online:** a target não tem módulo predictor. Um hook conta exatamente 2 forwards do predictor por passo.
4. **τ:** `tau_s(0) == tau_base`, `tau_s(K) == 1`, monótono, e `tau_s(K/2) == 1 - (1-tau_base)/2`.
5. **EMA:** após 1 passo, `ξ1 == τ0·ξ0 + (1−τ0)·θ1` em 1e-6. Com `accum = 4`, ξ não muda nos 3 primeiros micro-passos.
6. **lr:** `lr_s[0] == 0`, pico = base·bs/256 no fim do warm-up, `lr_s[-1] ≈ 0`.
7. **LARS/wd:** parâmetros de viés e 1-D com `weight_decay == 0` e sem adaptação de confiança (`lars_exclude`).
8. **Assimetria de augmentação:** com seed fixa e N grande, a fração de vistas borradas fica ≈ 1,0 em T e ≈ 0,1 em T′, e a de solarizadas ≈ 0 em T e ≈ 0,2 em T′.
9. **Target em modo train:** `T["enc"].training is True` durante o passo, e as estatísticas correntes do BN da target mudam.
10. **Saída = online:** os pesos devolvidos são os da online.
11. **Ablação de sanidade (toy):** sem predictor e com τ = 0, a variância por dimensão de z̄ → 0 (colapso, Tab. 5b). Com predictor e τ_base = 0,99, ela não colapsa.
12. **Compatibilidade:** `schedule="fixed"` mantém os testes atuais de `test_ssl_trainers.py`.
