# I-JEPA: regras de implementação fiéis ao artigo

> Fontes lidas: fonte LaTeX do arXiv:2301.08243 (v3, `main.tex`, incluindo o Apêndice A "Implementation Details" e o C "Additional Ablations") e o código oficial `facebookresearch/ijepa` (`src/train.py`, `src/helper.py`, `src/masks/multiblock.py`, `src/models/vision_transformer.py`, `src/transforms.py`, `src/utils/schedulers.py`, `configs/*.yaml`). Nosso código: `ga_search/modehb/ssl_trainers.py` e `src/losses/lejepa_loss.py`.
>
> Seções: §3 método, §5 classificação, §7 escalabilidade, §9 ablações. Tabelas: 1 linear IN-1k, 2 IN-1 %, 3 transferência, 5 escala de dados e modelo, 6 máscara, 7 alvo pixel × representação. Apêndice: A implementação; C ablações (Tab. 8–14).
>
> Legenda: **[FIEL]**, **[ADAPT]**, **[HIP]**, como em `dino.md`.
>
> **Por que importa aqui:** o LeJEPA, nosso `lejepa` = invariância multivista + RealSIGReg (`src/losses/lejepa_loss.py`), é da família JEPA, mas **não** tem target EMA, predictor nem máscara. O I-JEPA é o JEPA "clássico" com os três. Implementá-lo fielmente separa o que vem do *princípio JEPA* (prever no espaço de representação) do que vem do *SIGReg*.

## Referência

- Assran, Duval, Misra, Bojanowski, Vincent, Rabbat, LeCun, Ballas. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. CVPR 2023. arXiv:2301.08243.
- Código oficial: https://github.com/facebookresearch/ijepa (config de referência `configs/in1k_vith14_ep300.yaml`).
- Adaptações a CNN citadas **só pelo resumo** (não lidas em detalhe; tratar como [HIP]): CNN-JEPA (Kalapos & Gyires-Tóth, arXiv:2408.07514), com encoder CNN esparso para entrada mascarada, predictor totalmente convolucional com conv separável em profundidade e 73,3 % linear no ImageNet-100 com ResNet-50; SparK (Tian et al., arXiv:2301.03580), modelagem mascarada esparsa e hierárquica para CNN.

## Resumo

A partir de **um bloco de contexto**, prever as **representações** (não os pixels) de **M = 4 blocos-alvo** da mesma imagem (§3).
- **Alvos:** a imagem inteira passa pelo *target encoder* f_θ̄ (a EMA do *context encoder*), e os blocos-alvo são recortados **na saída** dele, não na entrada (§3 "Targets"; Tab. 11: saída 67,3 contra entrada 56,1).
- **Contexto:** um bloco grande (escala 0,85–1,0, razão 1) do qual se **remove a sobreposição** com os alvos (§3 "Context").
- **Predictor:** um ViT estreito recebe os tokens do contexto mais *mask tokens* (um vetor compartilhado + embedding posicional) nas posições do alvo i, e é aplicado M vezes (§3 "Prediction").
- **Perda:** L2 média entre predição e alvo por patch (§3 "Loss").
- **Augmentação:** praticamente nenhuma: só RandomResizedCrop (0,3, 1), sem flip, sem cor e sem blur (config oficial; §1).
- **Avaliação:** feita no **target encoder**, com average-pool dos tokens de saída (App. A.1 "Architectures").

**CNN e poucos dados (o que o artigo mostra):**
- **Só ViT.** Não há resultado com CNN no artigo. Para CNN, a evidência vem de terceiros (CNN-JEPA, SparK) e conta como [HIP].
- **Não há pré-treino em dataset pequeno.** O artigo mostra que o I-JEPA **ganha com mais dados** (IN-1k → IN-22k, §7, Tab. 5), o que é um sinal de risco para 1–10 mil imagens.
- O que joga a favor do nosso caso: o I-JEPA **não depende de augmentação manual** de cor (§1, §7), e a cor é diagnóstica em microscopia. Ele também converge em menos iterações que o MAE (§7, "≈5× fewer iterations").

## Receita oficial (tabela)

| Item | Artigo | Código oficial (`in1k_vith14_ep300.yaml` + `src/`) | Fonte |
|---|---|---|---|
| Perda | (1/M) Σ_i Σ_{j∈B_i} ‖ŝ_yj − s_yj‖²₂ | `F.smooth_l1_loss(z, h)` (média), com **`h = F.layer_norm(h, (D,))`** no alvo | §3 "Loss"; `train.py` l. 295–311 |
| Alvo | saída do target encoder, mascarada **depois** do forward | `h = target_encoder(imgs)` → layer_norm → `apply_masks(h, masks_pred)` | §3; Tab. 11 |
| EMA | momentum 0,996 → 1,0, **linear** ao longo do treino | gerador linear por iteração: `ema[0] + i·(ema[1]−ema[0])/(ipe·epochs·ipe_scale)` | App. A.1; `train.py` l. 228 |
| Init do target | idêntico ao context encoder | `copy.deepcopy(encoder)`, `requires_grad=False` | App. A.1 |
| Context encoder | ViT; processa **só** os patches visíveis do contexto | `encoder(imgs, masks_enc)` (pos-embed e depois `apply_masks`) | §3 |
| Predictor | ViT estreito: dim **384**, nº de cabeças = o do encoder, profundidade **12** (L/16, H/16, H/14) ou **16** (G/16) | `VisionTransformerPredictor`: Linear(D → 384), mask token (trunc_normal 0,02), pos-emb **sin-cos 2D fixo**, blocos, LayerNorm, Linear(384 → D); concatena contexto + mask tokens e lê só a parte dos alvos | App. A.1; `vision_transformer.py` |
| Nº de alvos M | 4 | `num_pred_masks: 4` | §3; Tab. 10 (1 → 9,0; 4 → 54,2) |
| Escala do alvo | (0,15, 0,2) | `pred_mask_scale: [0.15, 0.2]` | Fig. 4; Tab. 8 |
| Razão de aspecto do alvo | (0,75, 1,5) | `aspect_ratio: [0.75, 1.5]` | §3 |
| Contexto | 1 bloco, escala (0,85, 1,0), razão 1; remove a sobreposição com os 4 alvos | `enc_mask_scale: [0.85, 1.0]`, `num_enc_masks: 1`, `allow_overlap: false` | §3; Tab. 9 |
| Amostragem das máscaras | independente por imagem; tamanhos iguais no mesmo GPU (para formar o batch) | o tamanho (h, w) do bloco é sorteado **uma vez por batch**, e o mesmo `_rand` serve para escala **e** razão; posição por imagem; `min_keep = 10`; as máscaras são **truncadas** ao menor comprimento do batch; se o contexto ficar vazio, relaxa as restrições após 20 tentativas | App. A.1 "Masking"; `multiblock.MaskCollator` |
| Augmentação | sem augmentações manuais de vista | RRC(224, escala **(0,3, 1,0)**), sem flip, cor ou blur; normalização ImageNet | `make_transforms`; config |
| Otimizador | AdamW (context encoder + predictor) | 4 grupos: viés e 1-D com `weight_decay = 0` (`WD_exclude`) | App. A.1; `helper.init_opt` |
| lr | warm-up **linear de 1e-4 a 1e-3 em 15 ép.**, depois cosseno até 1e-6 | `start_lr` **2e-4**, `lr` 1e-3, warm-up **40 ép.** (3 ép. no IN-22k), `final_lr` 1e-6 | App. A.1; config |
| Weight decay | "linearly increased" de **0,04 a 0,4** | `CosineWDSchedule` 0,04 → 0,4 (**cosseno**) | App. A.1; `schedulers.py` |
| Batch | 2048 | 128 × 16 GPUs | App. A.1; config |
| Épocas | ViT-H/14: 300; ViT-B/16 e L/16: 600 (Tab. 1) | 300 | Tab. 1 |
| Precisão | — | bf16 (`use_bfloat16: true`) | config |
| `ipe_scale` | — | 1,0 nas configs (o padrão da função é 1,25, o que estende os schedules) | config; `helper.init_opt` |
| Representação avaliada | target encoder, average-pool da saída (sem [cls]) | — | App. A.1 |
| Linear | VISSL: avg-pool da última camada ou concatenação das 4 últimas; cabeça linear com/sem BN (o melhor dos dois); RRC + flip. ImageNet: LARS, bs 16384, 50 ép., lr ÷ 10 a cada 15 ép., lr ∈ {0,01, 0,05, 0,001}, wd ∈ {5e-4, 0} | — | App. A.2 |
| Low-shot | IN-1 %: fine-tuning de 50 ép. (AdamW, layer decay 0,75, label smoothing 0,1) | — | App. A.2 |
| k-NN | **não usado** | — | — |

Ablações que justificam as regras (ViT-B/16, 300 ép., linear em IN-1 %, salvo indicação): **multi-block** 54,2, contra rasterized 15,5, block 20,2 e random 17,6 (Tab. 6). Alvo em pixels: 40,7 contra 66,9 com a saída do target encoder (ViT-L, Tab. 7). Alvo pequeno (0,075, 0,2): 19,2; alvo grande (0,2, 0,3): 33,6 (Tab. 8). Contexto (0,4, 1): 31,2 (Tab. 9). Predictor de profundidade 6: 64,0 contra 66,9 com 12 (Tab. 12). Gargalo de largura 384: 70,7 contra 68,4 com 1024 (Tab. 14). wd fixo em 0,05 melhora o fine-tuning, mas piora o linear (Tab. 13).

## Regras de implementação (checklist)

- [ ] **[FIEL]** Três módulos: context encoder (o nosso encoder + head), target encoder (`deepcopy`, `requires_grad_(False)`) e predictor (só no treino; descartado no fim; **fora do orçamento Bk** e registrado à parte, conforme MISSION).
- [ ] **[FIEL]** Um único view por imagem: RRC(224, escala (0,3, 1)), sem flip, cor ou blur.
- [ ] **[FIEL]** Máscaras na **grade de saída** (posições do mapa de features; no ViT, um patch por posição). Para cada batch: sortear um tamanho de alvo (escala (0,15, 0,2) e razão (0,75, 1,5) com o **mesmo** número aleatório, como no código) e um tamanho de contexto (escala (0,85, 1), razão 1). Para cada imagem: 4 alvos (podem se sobrepor entre si) e 1 contexto = bloco ∖ ∪alvos. `min_keep = 10`. Truncar todas as máscaras ao menor comprimento do batch.
- [ ] **[FIEL]** Alvo: `with no_grad(): h = target(x)` com a imagem **inteira**, depois `layer_norm` no eixo dos canais (código) e então indexar as posições de B_i.
- [ ] **[FIEL]** O context encoder **não vê** pixels fora do contexto (no ViT, os tokens são removidos). Ver em "Adaptação" a emulação para CNN.
- [ ] **[FIEL]** Predictor: projeção para a largura estreita (384) + pos-emb 2D sin-cos **fixo** nas posições de contexto; mask token compartilhado (aprendível) + pos-emb nas posições do alvo i; M chamadas (ou batch × M); saída projetada de volta para D.
- [ ] **[FIEL]** Perda: `smooth_l1(pred, target)` (código) **ou** L2 média (artigo). Registrar qual foi usada. Recomendo o código como "fiel-código" e o L2 como braço.
- [ ] **[FIEL]** Schedules por iteração: lr com warm-up linear `start_lr → lr` e cosseno até `final_lr`; wd em cosseno 0,04 → 0,4 (código; o artigo diz linear); momentum **linear** 0,996 → 1. Viés e 1-D sem wd.
- [ ] **[FIEL]** EMA depois do `step`, sobre os parâmetros do context encoder → target.
- [ ] **[FIEL]** No fim, **devolver o target encoder** como modelo avaliado; a representação global é o pooling da saída.
- [ ] **[ADAPT]** Logar a fração média de contexto (o artigo mede "avg ratio" ≈ 0,25 no multi-block, Tab. 6), a perda, a variância por canal dos alvos (colapso → 0) e o momentum.

## Divergências e o que falta

**Entre o artigo e o código oficial** (importantes):
1. **Perda:** o artigo usa L2; o código usa **smooth-L1 com layer-norm no alvo**. O artigo não menciona o layer-norm.
2. **Warm-up do lr:** o artigo diz 1e-4 → 1e-3 em 15 ép.; o código usa 2e-4 → 1e-3 em 40 ép. (IN-1k) ou 3 ép. (IN-22k).
3. **wd:** "linearly increased" no artigo; cosseno no código (`CosineWDSchedule`).
4. **Máscaras:** o artigo diz "independentemente por imagem". No código, só a **posição** é por imagem; o **tamanho** é por batch, e escala e razão do alvo são correlacionadas (o mesmo `_rand`).
5. O artigo não cita a augmentação RRC (0,3, 1); ela aparece no código.

**Entre o nosso código e o I-JEPA:**
1. Não há `ijepa` em `_SSL` (l. 144). O `lejepa` atual é multivista (V = 4) com invariância à média e RealSIGReg. Não usa máscara, predictor nem EMA.
2. O encoder é **CNN** (`encoder(x) -> [N, C, h, w]`) e não tem como "remover tokens". É preciso emular a entrada esparsa.
3. A `EmbeddingHead` faz pooling **dentro** do `forward`. O I-JEPA precisa do mapa **antes** do pooling: usar `head.proj(feat)` (o conv 1×1 → 1280 canais por posição), que é o análogo exato dos tokens de saída do ViT. A representação global = `head.pool(...)` corresponde ao "average pool da saída do target" do artigo.
4. `make_views` (l. 88) faz V vistas com flips e jitter. O I-JEPA usa **uma** vista, só com RRC.
5. AdamW com lr fixo e wd único (l. 322–336), contra schedules de lr/wd/momentum e exclusão de viés e 1-D.
6. `extra.clear()` (l. 474) devolve o context encoder. O I-JEPA avalia o **target encoder**.
7. Batch 64 contra 2048. O I-JEPA não tem negativos, mas o artigo não testa batch pequeno [HIP].
8. Avaliação: o artigo usa linear (VISSL/LARS) e low-shot. Nós usamos SVM C = 100 (R1). Não há k-NN no artigo.

## Adaptação

### Mascaramento para CNN (o ponto não trivial)

No ViT, o contexto é literalmente o subconjunto de tokens visíveis. Numa CNN, o campo receptivo mistura os vizinhos, então zerar a entrada cria bordas artificiais e deixa vazar informação dos alvos para as posições de contexto próximas. As opções, da mais fiel ao espírito para a mais simples:

- **(A) Esparso emulado [ADAPT, recomendado]**, no espírito de SparK/CNN-JEPA: `x_ctx = x * up(M_ctx)`, e depois de **cada** bloco conv/pool reaplicar `feat *= down(M_ctx)` (forward hooks nos `ConvBlock` de `ops_norm_act.py`). As estatísticas de BN devem ser calculadas só nas posições visíveis (BN mascarado). Sem isso, os zeros contaminam a média e a variância.
- **(B) Entrada mascarada simples [ADAPT, baseline]:** `x_ctx = x * up(M_ctx) + (1 − up(M_ctx)) * fill`, com `fill` = média por canal da imagem (LAB ≈ (μ_L, 0,5, 0,5)). É barata, mas vaza informação pela borda. Medir o vazamento (ver os testes).
- O target encoder sempre vê a imagem **inteira** sem máscara [FIEL].

Grade: a resolução do mapa final do encoder depende do genoma (B0 raso com 3 pools de stride 2: 224 → 28 × 28; mais fundo: 14 × 14 ou 7 × 7). As escalas (0,15, 0,2) e (0,85, 1) são **frações da grade** e se aplicam a qualquer tamanho [FIEL]. Ajustar `min_keep` proporcionalmente à grade, pois 10 de 256 equivale a ~4 % [ADAPT].

### Pseudo-código em `train_global_ssl` (ramo `ssl == "ijepa"`)

```python
# kwargs opt-in: ssl="ijepa", reg="none", schedule="official"|"fixed",
#   ijepa_cfg=dict(M=4, tgt_scale=(0.15,0.2), tgt_ar=(0.75,1.5), ctx_scale=(0.85,1.0),
#                  min_keep_frac=10/256, loss="smooth_l1"|"l2", target_ln=True,
#                  pred_width=384, pred_depth=6|12, pred_type="vit"|"dwsep",
#                  ctx_mask="sparse"|"input", crop_scale=(0.3,1.0),
#                  lr=(2e-4, 1e-3, 1e-6), warmup_ep=40, wd=(0.04, 0.4), ema=(0.996, 1.0))

def build_opt_ijepa(x0):
    with torch.no_grad():
        Z = head.proj(encoder(x0[:2]))                         # [2, D=1280, h, w] tokens de saída
    D, gh, gw = Z.shape[1:]
    extra["pred"] = Predictor(D, width=384, depth=pred_depth, grid=(gh, gw),       # [FIEL]: ViT estreito
                              kind=pred_type).to(device)       # "dwsep" = CNN-JEPA [HIP]
    T = {"enc": deepcopy(encoder), "head": deepcopy(head)}     # [FIEL]
    for m in T.values(): m.train(); [p.requires_grad_(False) for p in m.parameters()]
    collator = MultiBlockMask(grid=(gh, gw), **mask_cfg)       # porte de multiblock.MaskCollator
    groups = split_decay(split_param_groups(encoder, head, [extra["pred"]], new_names, lr_old, lr_new))
    groups = _scale_conv_groups(groups, encoder, conv_lr_scale)          # proteção SPiFiL [ADAPT]
    for g in groups: g["lr_mult"] = g["lr"] / lr_new
    opt = AdamW(groups)                                                   # [FIEL]
    lr_s = warmup_cosine(start=2e-4, ref=1e-3, final=1e-6, warmup=warmup_ep*ipe, total=epochs*ipe)
    wd_s = cosine(0.04, 0.4, epochs*ipe)                                  # código; artigo: linear
    m_s  = np.linspace(0.996, 1.0, epochs*ipe + 1)                        # linear [FIEL]

for ep in range(epochs):
    for bi, x in enumerate(loader):
        it = ep*ipe + bi; set_lr_wd(opt, lr_s[it], wd_s[it])
        x = rrc(x, scale=(0.3, 1.0), size=224)                 # única augmentação [FIEL]
        ctx_idx, tgt_idx = collator(B)                         # [B, Nc] e M × [B, Nt], índices na grade
        with torch.no_grad(), amp():
            Ht = T["head"].proj(T["enc"](x)).flatten(2).transpose(1, 2)        # [B, h*w, D]
            Ht = F.layer_norm(Ht.float(), (D,))                                # código [FIEL]
            tgts = [gather(Ht, t) for t in tgt_idx]                            # M × [B, Nt, D]
        with amp():
            Fc = encoder_masked(encoder, x, ctx_idx, mode=ctx_mask)            # (A) esparso | (B) entrada
            Zc = gather(head.proj(Fc).flatten(2).transpose(1, 2), ctx_idx)     # [B, Nc, D]
            preds = [extra["pred"](Zc, ctx_idx, t) for t in tgt_idx]           # M chamadas [FIEL]
        with fp32():
            loss = mean([F.smooth_l1_loss(p.float(), t) if loss_type == "smooth_l1"
                         else (p.float() - t).pow(2).sum(-1).mean() for p, t in zip(preds, tgts)])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():                                                   # EMA [FIEL]
            m = m_s[it]
            for ps, pt in zip(chain(encoder, head).parameters(), chain(T["enc"], T["head"]).parameters()):
                pt.mul_(m).add_((1 - m) * ps.detach())
        log(loss, len(ctx_idx[0]) / (gh*gw), Ht.var(dim=(0, 1)).mean(), m, lr, wd)

# fim: avaliar o TARGET encoder [FIEL]; pooling = head.pool (análogo ao avg-pool do artigo)
encoder.load_state_dict(T["enc"].state_dict()); head.load_state_dict(T["head"].state_dict())
extra.clear()
```

`Predictor(kind="vit")` é o porte direto de `VisionTransformerPredictor` sobre os tokens da grade: com h·w ≤ 784 e largura 384, cabe em 48 GB com bs 256. `kind="dwsep"` é uma pilha conv separável 3×3 sobre o mapa (contexto preenchido + mask tokens), como no CNN-JEPA [HIP].

### Valores sugeridos (CNN rasa SPiFiL, 224 px, 1–10 mil imagens, GPU de 48 GB)

| Item | Fiel | Adaptação proposta | Marca |
|---|---|---|---|
| Tokens | saída do ViT (patch 14/16) | `head.proj(encoder(x))`: 1280 canais por posição da grade final | [ADAPT] (análogo direto) |
| Representação global | avg-pool da saída do target | `head.pool` (gap / GeM / moments, conforme o gene) do target | [FIEL]-análogo |
| Máscaras | (0,15, 0,2) × 4; contexto (0,85, 1) | iguais, em frações da grade | [FIEL] |
| Contexto CNN | remoção de tokens | (A) esparso emulado + BN mascarado; (B) entrada preenchida como baseline | [ADAPT] |
| Predictor | ViT 384, profundidade 12 | ViT 384, profundidade **6** (encoder raso; a Tab. 12 mostra que o 12 ganha num ViT-L e deve ser testado) ou dw-sep | [ADAPT] / [HIP] |
| Perda | smooth-L1 + LN (código) / L2 (artigo) | smooth-L1 + LN como principal; L2 como braço | [FIEL] |
| Batch | 2048 | **256** | [ADAPT] |
| lr | 1e-3 com bs 2048 (AdamW) | 1e-3 fiel. O artigo não dá regra de escala, então testar também 1e-3 × 256/2048 ≈ 1,25e-4. Manter as razões `lr_old`/`lr_new` e `conv_lr_scale` | [FIEL] / [HIP] |
| Épocas | 300–600 | por passos (≥ 5 mil). O I-JEPA converge rápido, mas **gosta de dados** (Tab. 5) | [HIP] |
| Augmentação | só RRC (0,3, 1) | só RRC (0,3, 1), em LAB e sem cor (é o ponto forte para microscopia). O crop condicionado ao objeto (`object_crop.py`) entra como braço | [FIEL] / [HIP] |
| Híbrido com LeJEPA | — | braço identificado `ijepa+sigreg`: SIGReg sobre `head.pool` do contexto. **Não** chamar de I-JEPA nem de LeJEPA | [HIP] |
| Avaliação | linear / low-shot | **SVM C = 100** (R1) no target; κ de teste só no fim | [ADAPT] |

## Testes de conformidade

Colocar em `tests/modehb/test_ssl_ijepa.py`.

1. **Máscaras (estatística, 10 mil sorteios em grade 16 × 16 e 28 × 28):** a área de cada alvo, em fração da grade, fica em [0,15, 0,2] (até o arredondamento de `int`/`round` do código); a razão em [0,75, 1,5]; o contexto tem razão 1 e área-bloco em [0,85, 1]; **contexto ∩ alvos = ∅**; `len(ctx) ≥ min_keep`; M = 4; todas as máscaras do batch com o mesmo comprimento; a fração média de contexto fica perto de ~0,25 na grade 16 × 16 (Tab. 6, "avg ratio").
2. **Paridade com o oficial:** com seed fixa, `MultiBlockMask` reproduz exatamente os índices de `src/masks/multiblock.MaskCollator` (copiado para os testes) na grade 14 × 14.
3. **Alvo:** o target recebe a imagem inteira (hook de shape), o alvo sai de `layer_norm` (média ≈ 0, var ≈ 1 por token) e é indexado **depois** do forward.
4. **Vazamento do contexto (CNN):** perturbar pixels **só dentro** dos blocos-alvo. No modo (A), as features de contexto devem ficar **inalteradas** (diferença ≤ 1e-6). No modo (B), reportar a fração de posições de contexto alteradas; esse é o vazamento a registrar.
5. **Predictor:** M chamadas por passo; a saída tem shape [B, N_t, D]; o pos-emb sin-cos é fixo (`requires_grad=False`); o mask token é aprendível.
6. **Perda:** `smooth_l1` bate com `F.smooth_l1_loss` oficial; a opção L2 bate com (1/M) Σ Σ ‖·‖² (normalizada pelo nº de posições).
7. **Schedules:** `lr_s[0] ≈ start_lr`, pico `lr` ao fim do warm-up, final ≈ `final_lr`; `wd_s` em cosseno 0,04 → 0,4; `m_s` linear com `m_s[0] = 0,996` e `m_s[-1] = 1`.
8. **EMA e stop-grad:** o target sem grad; após 1 passo, `θ̄1 = m0·θ̄0 + (1−m0)·θ1`.
9. **Saída = target:** os pesos devolvidos são os do target encoder.
10. **Sem augmentação de cor:** a vista é RRC pura (comparar com `F.interpolate` do crop; sem flip).
11. **Anticolapso (toy):** a variância por canal de `Ht` não vai a 0 em 20 épocas. A remoção da EMA (m = 0) deve degradar (sanidade, sem número de referência no artigo).
12. **Compatibilidade:** `schedule="fixed"` mantém os testes atuais de `test_ssl_trainers.py`.
