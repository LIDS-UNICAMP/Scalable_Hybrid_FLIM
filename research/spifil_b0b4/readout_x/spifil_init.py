# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC — Institute of Computing            ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀   Computer Science Department              ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · IC · 2026                    ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝
"""spifil_init.py -- variante "so-SPiFiL" de cada modelo testado pelo run_all_untested.py: a MESMA
arquitetura do checkpoint, so com o init SPiFiL/FLIM que a propria fonte usa, sem nenhum treino
(sem AE, SoftHebb, SSL, WiSE, backprop). Cada funcao roda com sys.path[0] = raiz do codigo da fonte.

  modehb/PNAS/a2  Evaluator._grow do run_search (camada a camada, mesmos doadores e cfg spifil do
                  config_resolved.yaml), larguras EFETIVAS do treinado, ae_off, softhebb off;
                  cabeca = EmbeddingHead semeada como no _evaluate.
  growth          genoma final inteiro, SPiFiL camada a camada (init_method da cadeia, spifil0), ver abaixo.
  hybrid (ssl)    AutoEncoderFlimModule(**hparams) sem load_state_dict: o __init__ le os kernels FLIM.
"""
def spifil_init_modehb(ckpt, device):
    """Mesmo genoma do candidato (modehb / PNAS / a2), so com o init SPiFiL do proprio Evaluator._grow
    (camada a camada, mesmos doadores e cfg spifil do config_resolved.yaml), sem AE, SoftHebb, SSL
    nem WiSE. Cabeca = EmbeddingHead semeada como no _evaluate. -> (encoder, head, n_params)."""
    import copy, tempfile
    from pathlib import Path
    import torch, yaml
    from ga_search.modehb.run_search import Evaluator, _NullRun
    from ga_search.modehb.pooling_head import EmbeddingHead
    ckpt = Path(ckpt)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfgp = next(d / "config_resolved.yaml" for d in ckpt.parents if (d / "config_resolved.yaml").is_file())
    cfg = yaml.safe_load(cfgp.read_text())
    cfg["device"] = str(device)
    cfg["eval"] = {**cfg.get("eval", {}), "fitness_mode": "spifil_init_only"}   # sem cache da SVM
    cfg["wandb"] = {"enabled": False}
    g = copy.deepcopy(ck["genome"])
    g["init"] = "spifil"
    for L in g["layers"]:
        L["softhebb"] = False                       # SoftHebb e aprendizado: fora
    eff = ck.get("effective_specs") or []           # mesma arquitetura: larguras EFETIVAS do treinado
    for i, e in enumerate(eff[: g["depth"]]):
        g["layers"][i]["channels"] = int(e["channels"])
    if any((e or {}).get("builder") == "spifil_ops" for e in eff):   # blocos SpifilOpsBlock (alocacao exata)
        cfg["spifil"] = {**cfg.get("spifil", {}), "spifil_ops": True}
    n = lambda *ms: sum(p.numel() for m in ms for p in m.parameters())
    from ga_search.modehb import spifil_ops as SO
    fit = SO.fit_ops_block
    with tempfile.TemporaryDirectory() as tmp:
        ev = Evaluator(cfg, tmp)
        for exact in (False, True):     # larguras fora do alocador (PNAS fp-widen: 72 com 14 classes):
            if exact:                   # fit_ops_block(exact_width=True) = ExactBudgetAllocator, o mesmo
                if not cfg["spifil"].get("spifil_ops"):   # que o crescimento fp/external do PNAS usa
                    break
                SO.fit_ops_block = lambda *a, **k: fit(*a, **{**k, "exact_width": True})
            try:
                enc, _specs, _logs = ev._grow(g, {}, 0, int(g["seed"]), _NullRun(), ae_off=True)
            finally:
                SO.fit_ops_block = fit
            torch.manual_seed(int(g["seed"]))
            head = EmbeddingHead(ev._out_channels(enc), int(g["embed_dim"]), g["pooling"]).to(device)
            if n(enc, head) == n(ck["encoder"], ck["head"]):
                break
    enc.spifil_exact_width = exact
    return enc.eval(), head.eval(), (n(enc, head), n(ck["encoder"], ck["head"]))


def spifil_init_hybrid(ckpt, device):
    """Copiado do agente (validado em 3 ckpts; 66 de 74 com params/shapes iguais, rand_* recusados).
    `load_from_checkpoint` = cls(**hparams) + load_state_dict; aqui so o primeiro passo, hparams
    filtrados pela assinatura (como o Lightning) e init_ckpt="" para nao puxar pesos treinados."""
    import inspect
    import torch
    from methods.autoencoder.autoencoder_flim_module import AutoEncoderFlimModule
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False, mmap=True)
    hp = ck["hyper_parameters"]
    if "random_z" in str(hp.get("flim_weights_path")):
        raise ValueError(f"{ckpt}: init aleatorio por desenho (random_z), nao tem versao SPiFiL")
    sig = inspect.signature(AutoEncoderFlimModule.__init__).parameters
    module = AutoEncoderFlimModule(**{**{k: v for k, v in hp.items() if k in sig}, "init_ckpt": ""})
    enc = module.model.encoder.to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    enc_sd = {v.untyped_storage().data_ptr(): v for k, v in ck["state_dict"].items() if k.startswith("model.encoder.")}
    n1 = sum(v.numel() for v in enc_sd.values())     # o state_dict repete o encoder (aliases): conta storage unico
    return enc, (sum(p.numel() for p in enc.parameters()), n1)


def _spifil_init_growth(ckpt, device):
    """Do agente (validado: v2 RS B3g0c02 e v3 flowers B2g1c00, params/chaves/arch_hash iguais, 0 passos
    de treino). Genoma final inteiro, SPiFiL camada a camada com o init_method da cadeia (spifil0 ->
    spifil_ops.fit_ops_block, mesma chamada do growth_chain.NewFilterInit); linhas/blocos que o SPiFiL
    nao preenche ficam no init do codigo (semeado), como no NewFilterInit; ZCA = Evaluator._fit_zca;
    doadores/ipc/n_superpixels/spifil cfg do growth_state.json (+ genes do no na v3)."""
    import json, time
    from pathlib import Path
    from types import SimpleNamespace
    import torch
    from torch import nn
    from ga_search.abstract_init.base import InitContext, TrainData, get_initializer
    from ga_search.backbones import build as BB
    from ga_search.backbones.checkpoint import load_backbone
    from ga_search.modehb import run_search as RS
    from ga_search.modehb import spifil_ops as SO
    from ga_search.modehb.ops_norm_act import InceptionConv
    from ga_search.progressive_data import ReconstructionDataset, _real_samples

    ckpt = Path(ckpt).resolve()
    res = json.loads((ckpt.parent / "result.json").read_text())
    root = next(p for p in ckpt.parents if p.name == "nodes").parent
    full = json.loads((root / "growth_state.json").read_text())["config_full"]
    run, g = full["run"], dict(full["growth"])
    if res.get("genes"):                                   # v3: genes do nó (ipc, n_superpixels)
        from ga_search.hmodehb.growth_chain import apply_genes
        g = apply_genes(g, res["genes"])
    _, spec = load_backbone(ckpt, "cpu", expected_sha256=res.get("checkpoint_sha256"))
    seed, dev = int(res.get("seed") or run["seed"]), torch.device(device)

    # doadores do NewFilterInit: Evaluator._load_data -> build_spifil_pool -> spifil_subset(k)
    d = {**run["data"], "seed": run["data"].get("seed", run["seed"])}
    train, _, selected = _real_samples(d)
    k_max = max(int(k) for k in (run.get("search") or {}).get("spifil_images_choices") or RS.SPIFIL_IMAGES_CHOICES)
    pool = RS.build_spifil_pool(selected, train, k_max, int(d["seed"]))
    k = min(int(g.get("spifil_images_per_class", 4)), k_max)
    donors = RS.spifil_subset(pool, k)
    td = TrainData(donors=donors, donor_pool=pool, split_key="train", n_classes=len(pool),
                   image_size=int(d.get("image_size", 224)), n_images=len(donors))
    init = get_initializer(g.get("init_method", "spifil0"), **(g.get("init_config") or {}))
    scfg = {**(run.get("spifil") or {}), "device": str(dev), "image_size": int(d.get("image_size", 224))}
    scfg.pop("in_channels", None)
    if (g.get("v3") or {}).get("spifil_n_superpixels"):
        scfg["n_superpixels"] = int(g["v3"]["spifil_n_superpixels"])

    # ZCA: mesmo loader do Evaluator (treino, embaralhado com gerador semeado, batch_size do run)
    zloader = torch.utils.data.DataLoader(
        ReconstructionDataset(train, int(d.get("image_size", 224)), bool(d.get("imagenet_norm", False))),
        batch_size=int(d.get("batch_size", 64)), shuffle=True, generator=torch.Generator().manual_seed(int(run["seed"])))
    zself = SimpleNamespace(train_loader=zloader, device=dev, cfg={"train": {
        "zca_fit_batches": int((run.get("train") or {}).get("zca_fit_batches", 8))}})

    torch.manual_seed(seed)                                # tudo que não é SPiFiL: init do código, semeado
    model = BB.Backbone(spec).to(dev).eval()
    model.spec.cost = spec.cost
    layers, log = spec.layers(), []

    def fit(j, pre, kind, kk, c, cin, act, exact=False):
        ops = SO.to_ops_spec({"conv": kind, "kernel": kk, "channels": c, "activation": act, "norm": "none",
                              "pool": "none", "residual": False, "dropout": 0.0}, cin)
        cfg = {**scfg, "seed": seed + j}
        if exact:
            blk, _ = SO.fit_ops_block(pre, donors, ops, cfg, exact_width=True)
            return blk.state_dict()
        ctx = InitContext(seed=seed + j, device=dev, layer_idx=j, prefix_id=f"spifil_only:{spec.hash12}:{j}",
                          config=dict(init.config), spifil_config=cfg, genome=None)
        w, b, _ = init.fit(pre, td, ops, ctx)
        return {**w, **b}

    @torch.no_grad()
    def put(conv, sd, key):
        W, B = sd[f"{key}.weight"], sd.get(f"{key}.bias")
        n = min(int(W.shape[0]), int(conv.weight.shape[0]))
        conv.weight[:n] = W[:n].to(conv.weight)
        if conv.bias is not None and B is not None:
            conv.bias[:n] = B[:n].to(conv.bias)
        return n

    for j, (L, blk) in enumerate(zip(layers, model.blocks)):
        t0 = time.perf_counter()
        cin = BB.block_in_channels(spec, j) if hasattr(BB, "block_in_channels") else \
            int(spec.input.in_channels if j == 0 else layers[j - 1].channels)
        prefix = nn.Sequential(*model.blocks[:j]) if j else None
        conv = blk.conv
        parts = []                                         # (nome, módulo, closure do fit, chave)
        if isinstance(conv, InceptionConv):
            for name, kb, cb in conv.branch_spec:
                if name == "pool":
                    pre = nn.Sequential(*model.blocks[:j], nn.MaxPool2d(3, 1, 1))
                    parts.append((f"inception.{name}", [(conv.branches[name].proj, "conv")],
                                  lambda pre=pre, cb=cb: fit(j, pre, "pointwise", 1, int(cb), cin, L.activation)))
                else:
                    kind = "conv" if int(kb) > 1 else "pointwise"
                    parts.append((f"inception.{name}", [(conv.branches[name], "conv")],
                                  lambda kind=kind, kb=kb, cb=cb: fit(j, prefix, kind, int(kb), int(cb), cin, L.activation)))
        elif isinstance(conv, nn.Sequential):              # dwsep (SpifilOpsBlock ou ConvBlock)
            parts.append(("dwsep", [(conv[0], "conv.depthwise"), (conv[1], "conv.pointwise")],
                          lambda: fit(j, prefix, "dwsep", int(L.kernel), int(L.channels), cin, L.activation)))
        else:
            kind = "depthwise" if conv.groups > 1 else ("conv" if conv.kernel_size[0] > 1 else "pointwise")
            parts.append((kind, [(conv, "conv")],
                          lambda: fit(j, prefix, kind, int(conv.kernel_size[0]), int(conv.out_channels), cin,
                                      L.activation)))
        if isinstance(getattr(blk, "skip", None), nn.Conv2d):
            parts.append(("skip_project", [(blk.skip, "conv")],
                          lambda: fit(j, prefix, "pointwise", 1, int(blk.skip.out_channels), cin, L.activation,
                                      exact=True)))
        for name, mods, run_fit in parts:
            rec = {"block": j, "part": name, "rows": int(mods[-1][0].weight.shape[0]), "n_spifil": 0, "fallback": None}
            try:
                sd = run_fit()
                rec["n_spifil"] = [put(m, sd, key) for m, key in mods][-1]   # dwsep: conta o pointwise
            except Exception as exc:                       # InfeasibleCandidate, N<=D... -> init do código
                rec["fallback"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            log.append(rec)
        model.eval()
        RS.Evaluator._fit_zca(zself, prefix if prefix is not None else nn.Identity(), blk)
        model.eval()
        print(f"[spifil_init_growth] bloco {j} ({L.conv} k{L.kernel} {cin}->{L.channels}, norm {L.norm}) "
              f"{time.perf_counter() - t0:.1f}s {[(r['part'], r['n_spifil'], r['rows'], r['fallback']) for r in log if r['block'] == j]}",
              flush=True)

    for p in model.parameters():                           # o Trunk do SPiFiL congela o prefixo; devolve como o load_backbone
        p.requires_grad_(True)
    model.spifil_init_log = log
    model.spifil_init_config = {"init_method": init.name, "spifil_cfg": scfg, "ipc": k, "n_donors": len(donors),
                                "n_classes": len(pool), "seed": seed, "zca_fit_batches": zself.cfg["train"]["zca_fit_batches"]}
    return model.eval()


def spifil_init_growth(ckpt, device):
    from ga_search.backbones.checkpoint import load_backbone
    model = _spifil_init_growth(ckpt, device)
    trained, _ = load_backbone(ckpt, "cpu")
    n = lambda m: sum(p.numel() for p in m.parameters())
    return model, (n(model), n(trained))
