# DINO: regras de implementação fiéis ao artigo

> Fontes lidas: fonte LaTeX do arXiv:2104.14294 (v2) e o código oficial `facebookresearch/dino` (`main_dino.py`, `utils.py`, `vision_transformer.py::DINOHead`, `eval_knn.py`, `eval_linear.py`, README). Também os `args.txt` dos checkpoints publicados (ViT-S/16 800 ep e ResNet-50). Nosso código: `ga_search/modehb/ssl_trainers.py` (494 linhas, lido inteiro). Seções e tabelas seguem a numeração do PDF do arXiv: Tab. 1 = arquiteturas, Tab. 2 = SOTA, Tab. 6 = transferência, Tab. 7 = componentes, Tab. 8 = computação, Tab. 9 = batch, Tab. 13–15 = apêndice B. Os apêndices vão de A a H.
>
> Legenda: **[FIEL]** reproduz o artigo ou o código oficial; **[ADAPT]** adaptação ao nosso caso; **[HIP]** hipótese sem evidência no artigo.

## Referência

- Caron, Touvron, Misra, Jégou, Mairal, Bojanowski, Joulin. *Emerging Properties in Self-Supervised Vision Transformers*. ICCV 2021. arXiv:2104.14294.
- Código oficial: https://github.com/facebookresearch/dino (`main_dino.py`).
- Receitas publicadas: `dino_deitsmall16_pretrain/args.txt` (ViT-S/16, 800 ep) e `dino_resnet50_pretrain/args.txt` (ResNet-50).

## Resumo

DINO é **autodestilação sem rótulos**. A rede aluna g_θs = h∘f (backbone f + cabeça de projeção h) aprende a casar a distribuição softmax de uma professora g_θt de mesma arquitetura. A professora é a **EMA** dos pesos da aluna (§3.1). O colapso é evitado só com **centering** (subtrai da saída da professora um centro c mantido por EMA) mais **sharpening** (temperatura baixa τt na professora), sem predictor nem perda contrastiva (§3.1, §5.3). O **multi-crop** é central: 2 vistas globais (224²) passam pela professora; todas as vistas, globais e locais (96²), passam pela aluna. A perda é a entropia cruzada entre toda vista global da professora e toda **outra** vista da aluna (Eq. 3).

A representação avaliada é a saída do backbone f, não a da cabeça h (§3.1, "Network architecture"). Na avaliação oficial usa-se a **professora** (`eval_linear.py --checkpoint_key teacher`), que supera a aluna durante todo o treino, também com ResNet-50 (§5.2, figura teacher × student; App. D, "The teacher outperforms the student").

**CNN e poucos dados (o que o artigo mostra):**
- O DINO funciona com ResNet-50: 75,3 linear e 67,5 k-NN, empatado com SwAV/DCv2 (Tab. 2). A vantagem sobre BYOL e SwAV na mesma CNN é pequena: 74,5 contra 72,7 (BYOL) no linear e 65,6 contra 65,4 no k-NN (Tab. 13). O ganho grande do DINO aparece só com ViT (§4.1, App. B).
- A receita da CNN é **outra**: LARS, lr 0,3, wd 1e-6 constante, crops globais (0,14, 1), BN na cabeça (args.txt RN50; README "ResNet-50 and other convnets").
- Batch pequeno funciona: 128 → 57,9 e 1024 → 59,9 no k-NN (100 ep, sem multi-crop; Tab. 9, §5.5). O artigo cita 35,2 % com batch 8 após 50 épocas.
- **Nenhum pré-treino em dataset pequeno.** Os resultados em Flowers, Cars e CIFAR são fine-tuning após pré-treino no ImageNet (Tab. 6). Para 1–10 mil imagens não há evidência no artigo [HIP].

## Receita oficial (tabela)

| Item | ViT-S/16 (artigo §3.2 + args.txt 800 ep) | ResNet-50 (args.txt oficial; README) | Fonte |
|---|---|---|---|
| Perda | H(Pt, Ps) = −Σ Pt log Ps, soma sobre pares (vista global t, vista s ≠ t), média sobre os termos | igual | §3.1 Eq. 2–3; `DINOLoss.forward` |
| Ps | softmax(g_s(x)/τs), **τs = 0,1** | igual | §3.2; `student_temp=0.1` |
| Pt | softmax((g_t(x) − c)/τt), stop-grad | igual | Alg. 1; `DINOLoss` |
| τt | warm-up **linear 0,04 → 0,07 nas 30 primeiras épocas**, depois 0,07 | 0,04 → 0,07 em **50** épocas | §3.2; App. D "Sharpening"; args.txt |
| Centro c | c ← m·c + (1−m)·média_batch(g_t(x_global)), **m = 0,9**; atualizado depois da perda, com a média sobre as 2 globais | igual | Eq. 4; App. D "Online centering" (m = 0,999 colapsa) |
| Professora | θt ← λθt + (1−λ)θs; λ em **cosseno 0,996 → 1** por iteração | igual | §3.1 "Teacher network"; `cosine_scheduler(0.996, 1, …)` |
| Init da professora | cópia exata da aluna; `requires_grad=False` | igual | `teacher.load_state_dict(student.state_dict())` |
| Cabeça h | MLP de 3 camadas (2048, GELU) → bottleneck 256 → **ℓ2-norm** → Linear com weight-norm, sem viés, saída **K = 65536** | **BN na cabeça** (`use_bn_in_head=true`), K = 60000 | §3.1; App. C; `DINOHead` |
| `norm_last_layer` | `false` no ViT-S 800 ep (o README diz que só é seguro em vit_small); padrão `true` | `true` | args.txt; README |
| `freeze_last_layer` | zera os grads de `last_layer` na época 0 | igual | `cancel_gradients_last_layer` |
| Predictor | **não há** (com predictor: 75,6 contra 76,1) | — | §3.1; Tab. 7 linha 6 |
| Multi-crop | 2 × 224² escala (0,25, 1) + **10** × 96² escala (0,05, 0,25) (args 800 ep). No código, o padrão é (0,4, 1) / 8 × (0,05, 0,4); o ótimo do artigo é s ≈ 0,3 | 2 × 224² **(0,14, 1)** + **6** × 96² **(0,05, 0,14)** | App. E; args.txt; Tab. 8 |
| Augmentações | as do BYOL: flip 0,5; ColorJitter(0,4, 0,4, 0,2, 0,1) p = 0,8; grayscale p = 0,2. Global 1: blur p = 1,0. Global 2: blur p = 0,1 + solarize p = 0,2. Locais: blur p = 0,5. Blur com raio σ ∈ [0,1, 2,0]. RandomResizedCrop **bicúbico**. Normalização ImageNet | igual | §3.2; `DataAugmentationDINO` |
| Otimizador | **AdamW** | **LARS** (lr 0,3) ou, no README, SGD lr 0,03 | §3.2; args.txt; README |
| lr | 0,0005 × bs/256; warm-up linear de 10 ép.; depois cosseno até `min_lr` 1e-5 (1e-6 por padrão) | 0,3 × bs/256 (bs 4080), warm-up de 10 ép., `min_lr` 0,0048 | §3.2; `cosine_scheduler` |
| Weight decay | **cosseno 0,04 → 0,4**; viés e parâmetros 1-D sem wd | **1e-6 constante** (README SGD: 1e-4) | §3.2; `get_params_groups`; args.txt |
| Batch | 1024 (64 × 16 GPUs) | 4080 (51 × 80) | §3.2; args.txt |
| Épocas | 100 / 300 / 800 (k-NN 70,9 / 72,8 / 74,5) | 800 | App. D "Longer training" |
| Clip de gradiente | 3,0 no padrão do código; **0** nos dois args publicados | 0 | `--clip_grad`; args.txt |
| Precisão | fp16 + GradScaler (padrão); os publicados usam fp32 | fp32 | `--use_fp16`; args.txt |
| Drop-path | 0,1 (aluna) | — | `--drop_path_rate` |
| Avaliação k-NN | features **ℓ2-normalizadas** da professora; voto ponderado α = exp(sim/0,07); **k = 20** | igual | §3.2; App. F.1; `eval_knn.py` |
| Avaliação linear | backbone congelado, cabeça removida; SGD, bs 1024, 100 ép., sem wd, varredura de lr; RRC + flip no treino, crop central no teste. ViT-S concatena o [CLS] das 4 últimas camadas; convnets usam GAP | GAP do último mapa | App. F.2 |

Ablações que justificam as regras: sem momentum, colapso (0,1; Tab. 7 linha 2; §5.2, tabela "Teacher": student copy e previous iter = 0,1); MSE no lugar de CE: 62,4 (Tab. 7 linha 5); sem multi-crop: 72,5 contra 76,1 (Tab. 14); sem centering ou sem sharpening, KL → 0, ou seja, colapso (§5.3, figura "Collapse study"); τt fixo em 0,08 colapsa (App. D); cabeça profunda sem o gargalo ℓ2 colapsa (App. C).

## Regras de implementação (checklist)

- [ ] **[FIEL]** Aluna = encoder + `EmbeddingHead` (o f do DINO, que dá a régua de 1280 d do SVM) + `DINOHead`. Professora = `deepcopy` dos três, com `requires_grad_(False)`, inicializada idêntica à aluna.
- [ ] **[FIEL]** Gradiente só na aluna. A professora roda sob `torch.no_grad()` e recebe **apenas as 2 vistas globais**.
- [ ] **[FIEL]** Forward da aluna **por resolução** (globais juntas e locais juntas), como o `MultiCropWrapper`. As saídas são concatenadas antes da cabeça.
- [ ] **[FIEL]** Perda: para cada iq ∈ {0, 1} (globais da professora) e cada v ≠ iq (todas as vistas da aluna), somar `(-q * log_softmax(s_v/0.1)).sum(-1).mean()` e dividir pelo nº de termos, 2·(V−1) com V = 2 + n_local.
- [ ] **[FIEL]** `q = softmax((t − center)/τt)`, com stop-grad. O centro é atualizado **depois** de calcular a perda, com a média de `t` sobre as 2 globais (todo o batch global, como no all-reduce), m = 0,9. O centro é um buffer e vai no checkpoint.
- [ ] **[FIEL]** τt por **época**: `linspace(0.04, τt_final, warmup_ep)` e depois constante. τs = 0,1.
- [ ] **[FIEL]** Schedules **por iteração** (`it = ep·ipe + bi`): lr com warm-up linear (de 0) e cosseno até `min_lr`; wd em cosseno, aplicado **só ao grupo regularizado**; momentum em cosseno de 0,996 até 1. Depois de `opt.step()`: `pt.mul_(m).add_((1-m)*ps)` sobre `zip(student.parameters(), teacher.parameters())`, na mesma ordem.
- [ ] **[FIEL]** Grupos do otimizador: viés e parâmetros 1-D (BN, `p_raw` do GeM etc.) com `weight_decay=0`; o resto, com o wd do schedule.
- [ ] **[FIEL]** `DINOHead`: Linear(d, 2048) → [BN se CNN] → GELU → Linear(2048, 2048) → [BN] → GELU → Linear(2048, 256) → `F.normalize` → `weight_norm(Linear(256, K, bias=False))` com `weight_g = 1`, congelado se `norm_last_layer`. Init trunc_normal(std = 0,02), viés 0.
- [ ] **[FIEL]** `freeze_last_layer = 1`: na época 0, `p.grad = None` para `last_layer` antes do `step`.
- [ ] **[FIEL]** Augmentações assimétricas nas globais: blur p = 1,0 na global 1; blur p = 0,1 + solarize p = 0,2 na global 2; blur p = 0,5 nas locais.
- [ ] **[FIEL]** BN da professora em modo **train**: o código oficial nunca chama `teacher.eval()`, e as estatísticas correntes da professora vêm do próprio forward.
- [ ] **[FIEL]** No fim, **devolver os pesos da professora** (encoder + head) como modelo avaliado, como em `checkpoint_key=teacher`. A `DINOHead` é descartada.
- [ ] **[FIEL]** `drop_last=True` no loader (o centro e o BN supõem um batch cheio).
- [ ] **[FIEL]** Perda e softmax em fp32 (nosso padrão `amp_dtype` já força isso).
- [ ] **[ADAPT]** Diagnóstico de colapso por época: entropia de Pt e KL(Pt‖Ps) (§5.3). Alerta se KL → 0 ou se a entropia → 0 ou ln K.

## Divergências e o que falta

**Entre o artigo e o código oficial** (registrar qual dos dois seguimos):
1. τt: o artigo e o args 800 ep usam 0,04 → 0,07 em 30 ép.; o padrão do código é 0,04 fixo (`warmup_teacher_temp_epochs=0`), e o README chama 0,07 de "boosting".
2. Escalas do multi-crop: o artigo usa s ≈ 0,3 (App. E: (s, 1) e (0,05, s)); o padrão do código é 0,4; o ViT-S publicado, 0,25; o RN50, 0,14. Nº de locais: 6 (App. E), 8 (padrão do código), 10 (args 800 ep).
3. Clip de gradiente: 3,0 no padrão do código, 0 nos runs publicados.
4. `norm_last_layer`: padrão `true`; `false` no ViT-S publicado.
5. A Eq. 4 diz que c é "adicionado como viés"; no código, c é **subtraído** antes do softmax (é a mesma coisa com sinal trocado).

**Entre o nosso `ssl_trainers.py` e o DINO** (o que falta implementar):
1. `_SSL = ("simclr", "lejepa", "ae")` (l. 144): não existe `dino` nem em `method_name` nem no gene `ssl` (`modehb.py` `SSLS`).
2. Sem professora EMA, sem centro, sem schedule de τt nem de momentum. Todo o código atual é aluna-só.
3. `make_views` (l. 88–138) gera V vistas de **um único tamanho** (`out_size`) com a mesma `AugConfig`. Faltam crops locais 96², augmentação por vista (global 1, global 2, local), blur gaussiano, solarização, saturação/hue e grayscale. A RRC usa `affine_grid` bilinear com `padding_mode="reflection"`, e o oficial é RRC bicúbica sem padding.
4. Otimizador (l. 322–336): AdamW com **lr fixo** em dois grupos (`lr_old` = 1e-4, `lr_new` = 1e-3; `search.yaml:94-95`), sem warm-up, sem cosseno, com o mesmo wd em todos os parâmetros (viés e BN inclusos), sem schedule de wd. O DINO exige os três schedules e a exclusão de viés e norm.
5. Batch 64 fixo (`search.yaml:93`, "nunca autoajustado"). O DINO usa 1024 (ViT) e 4080 (RN50); a Tab. 9 só desce até 128.
6. `extra.clear()` (l. 474) devolve a **aluna**. O DINO avalia a **professora**.
7. Sem `drop_last`, sem clip específico (há `grad_clip` opcional) e sem `freeze_last_layer`.
8. A avaliação do projeto é o SVM linear C = 100 (regra R1). O DINO usa k-NN (k = 20, T = 0,07) e linear com SGD. Não trocar: o k-NN pode entrar **só como diagnóstico**, nunca como fitness e nunca no teste durante a seleção.

## Adaptação

### Pseudo-código em `train_global_ssl` (ramo `ssl == "dino"`)

```python
# ---- assinatura: novos kwargs opt-in; o padrão mantém o comportamento atual (testes atuais intactos) ----
# train_global_ssl(..., ssl="dino", reg="none", schedule="official" | "fixed",
#                  dino_cfg=dict(K=65536, hidden=2048, bottleneck=256, nlayers=3, use_bn_head=True,
#                                norm_last_layer=True, freeze_last_layer=1,
#                                tau_s=0.1, tau_t0=0.04, tau_t=0.07, tau_t_warmup_ep=30,
#                                center_m=0.9, m_base=0.996,
#                                n_local=6, global_scale=(0.14, 1.0), local_scale=(0.05, 0.14),
#                                global_size=224, local_size=96),
#                  opt="adamw" | "lars", base_lr, min_lr, warmup_ep=10, wd=(w0, w1))

def build_opt_dino(x0):                                   # substitui build_opt para "dino"
    d = head(encoder(x0[:2])).shape[1]                    # 1280 = f do DINO (régua do SVM)
    extra["dino_head"] = DINOHead(d, K, use_bn=use_bn_head, norm_last_layer=nll).to(device)    # [FIEL]
    T = {"enc": deepcopy(encoder), "head": deepcopy(head), "proj": deepcopy(extra["dino_head"])}  # [FIEL]
    for m in T.values(): m.train(); [p.requires_grad_(False) for p in m.parameters()]
    center = torch.zeros(1, K, device=device)             # buffer [FIEL]
    groups = split_param_groups(encoder, head, extra.values(), new_names, lr_old, lr_new)       # nosso
    groups = split_decay(groups)                          # viés/1-D -> wd=0; g["regularized"] [FIEL]
    groups = _scale_conv_groups(groups, encoder, conv_lr_scale)                                  # proteção SPiFiL [ADAPT]
    for g in groups: g["lr_mult"] = g["lr"] / lr_new      # preserva as razões old/new/spifil [ADAPT]
    opt = AdamW(groups) if opt == "adamw" else LARS(groups)   # LARS = utils.LARS do DINO (RN50) [FIEL]
    ipe = len(loader)                                     # drop_last=True no loader [FIEL]
    lr_s = cosine_scheduler(base_lr * B / 256, min_lr, epochs, ipe, warmup_epochs=warmup_ep)    # [FIEL]
    wd_s = cosine_scheduler(w0, w1, epochs, ipe)                                                  # [FIEL]
    m_s  = cosine_scheduler(m_base, 1.0, epochs, ipe)                                             # [FIEL]
    tt_s = np.r_[np.linspace(tau_t0, tau_t, tau_t_warmup_ep), np.full(epochs - tau_t_warmup_ep, tau_t)]

for ep in range(epochs):
    for bi, x in enumerate(loader):
        it = ep * ipe + bi
        for g in opt.param_groups:
            g["lr"] = lr_s[it] * g["lr_mult"]             # schedule "official"; com "fixed", não mexer
            if g["regularized"]: g["weight_decay"] = wd_s[it]
        G = [aug(x, global_cfg_1), aug(x, global_cfg_2)]                      # 224², blur 1,0 | blur 0,1 + solar 0,2
        L = [aug(x, local_cfg) for _ in range(n_local)]                       # 96², blur 0,5
        with torch.no_grad(), amp():
            t_out = T["proj"](T["head"](T["enc"](torch.cat(G))))              # [2B, K], só globais
        with amp():
            h_g = head(encoder(torch.cat(G)))                                 # forward por resolução
            h_l = head(encoder(torch.cat(L)))
            s_out = extra["dino_head"](torch.cat([h_g, h_l]))                 # [(2+n_local)B, K]
        with fp32():
            loss = dino_loss(s_out.float(), t_out.float(), center, tau_s, tt_s[ep], ncrops=2 + n_local)
            center.mul_(center_m).add_((1 - center_m) * t_out.float().mean(0, keepdim=True))   # depois da perda
        opt.zero_grad(set_to_none=True); loss.backward()
        if clip: clip_grad_norm_(student_params, clip)                        # 0 nos runs publicados
        if ep < freeze_last_layer: extra["dino_head"].last_layer.weight_v.grad = None
        opt.step()
        with torch.no_grad():                                                 # EMA por iteração [FIEL]
            m = m_s[it]
            for ps, pt in zip(chain(encoder, head, extra["dino_head"]).parameters(),
                              chain(T["enc"], T["head"], T["proj"]).parameters()):
                pt.mul_(m).add_((1 - m) * ps.detach())
        log(loss, entropy(Pt), kl(Pt, Ps), center.norm(), m, tt_s[ep], lr, wd)   # diagnóstico §5.3

# fim: o modelo avaliado é a PROFESSORA [FIEL]
encoder.load_state_dict(T["enc"].state_dict()); head.load_state_dict(T["head"].state_dict())
extra.clear()

def dino_loss(s, t, center, tau_s, tau_t, ncrops):        # = DINOLoss.forward oficial
    s = (s / tau_s).chunk(ncrops)
    q = F.softmax((t - center) / tau_t, dim=-1).detach().chunk(2)
    tot, n = 0.0, 0
    for iq, qq in enumerate(q):
        for v, sv in enumerate(s):
            if v == iq: continue
            tot += (-qq * F.log_softmax(sv, -1)).sum(-1).mean(); n += 1
    return tot / n
```

Observações do nosso caso:
- Com a proteção SPiFiL (`conv_lr_scale = 0`, conv congelada), a EMA de parâmetros congelados é a identidade. Nada a corrigir.
- O WiSE (`run_search.py:721-725`) faz o snapshot depois de `train_global_ssl`. Como devolvemos a professora, o WiSE mistura âncora e professora sem mudar nada lá.

### Valores sugeridos (CNN rasa SPiFiL, 224 px, 1–10 mil imagens, GPU de 48 GB)

| Item | Fiel (referência) | Adaptação proposta | Marca |
|---|---|---|---|
| Backbone f | ResNet-50 / ViT | encoder SPiFiL B0–B4 + `EmbeddingHead` (≥ 1280 d, obrigatório pela MISSION) | [ADAPT] |
| Cabeça | 3 camadas, 2048, bottleneck 256, BN (CNN) | igual; **K = 4096** (65536 → 4096 custa só 0,4 de k-NN, App. C) por ter 1–10 mil imagens | [ADAPT], baseado em App. C |
| Crops | RN50: 2 × 224 (0,14, 1) + 6 × 96 (0,05, 0,14) | igual. Se o objeto (ovo/larva/cisto) ocupar pouco da imagem, testar crops locais **condicionados ao objeto** (`src/transforms/object_crop.py`) como braço separado | [FIEL] / [HIP] |
| Cor | ColorJitter RGB (0,4, 0,4, 0,2, 0,1), grayscale 0,2, solarize | entrada em **LAB [0, 1]**: brilho e contraste em L; saturação = escala de (a, b) em torno de 0,5; hue = rotação de (a, b); grayscale = (a, b) → 0,5; solarize só em L. Alternativa: LAB → RGB → jitter → LAB | [ADAPT] |
| Blur | σ ∈ [0,1, 2], p = 1 / 0,1 / 0,5 | igual, em GPU (conv gaussiana agrupada por amostra, kernel ≈ 10 % do lado) | [FIEL] |
| Otimizador | RN50: LARS lr 0,3 × bs/256, wd 1e-6, `min_lr` 0,0048 | **LARS como fiel-CNN**; AdamW (receita ViT: 5e-4 × bs/256, wd 0,04 → 0,4) como braço, mantendo as razões `lr_old`/`lr_new` e a proteção SPiFiL | [FIEL] / [ADAPT] |
| Batch | 1024 / 4080 | **256** (cabe com folga em 48 GB numa CNN rasa a 224 + 6 × 96). Com 1 mil imagens: bs 128 e `drop_last` | [ADAPT], Tab. 9 |
| Épocas | 100–800 | fixar por **passos** (≥ 3–5 mil), porque 1 mil imagens com bs 256 dão só 3 passos por época. Warm-up do lr = 10 ép. ou ≥ 5 % dos passos, o que for maior. Warm-up de τt = 30 ép. ou 10–30 % do treino | [HIP] |
| Momentum m | cosseno 0,996 → 1 por passo | 0,996 fiel. Com poucos passos totais, testar 0,99 (o BYOL baixa τ_base em treinos curtos: 40 ép. → 0,97, `configs/byol.py`) | [FIEL] / [HIP] |
| Avaliação | k-NN k = 20 e linear | **SVM C = 100** (R1) na professora; k-NN k = 20, T = 0,07 só como diagnóstico | [ADAPT] (regra do projeto) |

Riscos específicos: (i) BN dentro do encoder SPiFiL com bs pequeno e crops de resoluções diferentes (o forward por resolução isola as estatísticas); (ii) ZCA/DBN (`ops_norm_act.py`) na professora atualizam os buffers no próprio forward, o que é aceitável mas deve ficar registrado; (iii) com fundo preto nos parasitas "pre", crops locais de 5–14 % caem no fundo e geram alvos triviais.

## Testes de conformidade

Colocar em `tests/modehb/test_ssl_dino.py`, em CPU e com TinyEncoder/TinyHead como em `test_ssl_trainers.py`.

1. **Perda = oficial:** com tensores aleatórios, `dino_loss` bate em 1e-6 com uma cópia literal de `DINOLoss.forward` (sem all-reduce). O nº de termos é 2·(V−1).
2. **Pares excluídos:** zerar a contribuição de (global i, aluna i) não muda a perda; trocar a ordem das locais não muda a perda.
3. **Centro:** após um passo, `center == 0.9*c0 + 0.1*t.mean(0)`. O centro não muda a perda do passo corrente.
4. **Schedules:** `m_s[0] == 0.996`, `m_s[-1] ≈ 1`, monótono. `tt_s[0] == 0.04`, `tt_s[29] == 0.07`, `tt_s[30:] == 0.07`. `lr_s[0] == 0`, pico em `it = warmup·ipe`, final ≈ `min_lr`. `wd_s` em cosseno de w0 a w1. `len(*) == epochs·ipe`.
5. **EMA:** depois de 1 passo, `θt == m·θt0 + (1−m)·θs1` em 1e-6. A professora tem `grad is None` em todos os parâmetros.
6. **Só as globais na professora:** hook de contagem no `forward` da professora → batch 2B.
7. **Weight decay:** nenhum parâmetro de viés ou 1-D em grupo com wd > 0. Os grupos regularizados recebem `wd_s[it]`.
8. **freeze_last_layer:** na época 0, `last_layer.weight_v` não muda; na época 1, muda.
9. **Saída = professora:** os pesos de `encoder`/`head` devolvidos são iguais aos da professora (não aos da aluna).
10. **Anticolapso:** num toy de 2 classes (≈ 200 imagens), 20 épocas com centering + sharpening → KL(Pt‖Ps) > 0 e entropia de Pt fora de {0, ln K}. Sem centering, a entropia → 0; com τt = 0,08 fixo, a perda → ln K (reproduz §5.3 e App. D qualitativamente).
11. **Determinismo:** a mesma seed dá a mesma perda nos 3 primeiros passos.
12. **Compatibilidade:** `schedule="fixed"` reproduz exatamente os lrs fixos (o teste atual `test_train_uses_two_fixed_lrs` continua passando).
