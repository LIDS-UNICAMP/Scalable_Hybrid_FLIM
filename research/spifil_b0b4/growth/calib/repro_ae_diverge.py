"""Reprodução CPU da divergência do AE local nos blocos alargados (v3.1, RS, fid 1-5, batch 64).

Uso: python repro_ae.py <case> <out.json> [--gset k=v ...] [--fid N]
case: rs306_w455 | tiny_chain
Grava curvas por passo (ae/recon_loss) por bloco e diagnósticos da escala.
"""
import argparse, json, math, sys, time, copy
from pathlib import Path
import torch
from torch import nn

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from ga_search.hmodehb import growth_chain as G
from ga_search.modehb.run_search import split_epochs

ap = argparse.ArgumentParser()
ap.add_argument("case")
ap.add_argument("out")
ap.add_argument("--gset", action="append", default=[])
ap.add_argument("--fid", type=int, default=1)
ap.add_argument("--diag", action="store_true")
ap.add_argument("--train-n", type=int, default=None)
a = ap.parse_args()

CK = {"rs306": "artifacts/modehb_spifil/remote_sensing/state/seed42/candidates/c00306/f100/encoder.pt",
      "tiny": "artifacts/modehb_hybrid/remote_sensing/state/seed42/candidates/c00001/f200/encoder.pt"}

out_dir = Path(a.out).with_suffix("")
cfg = G.build_cfg("remote_sensing", 42, ["device=cpu", "wandb.enabled=false"], out_dir)
g = G.load_growth_config(ROOT / "ga_search/config/hmodehb_b0b4/growth_v3.yaml", a.gset)
t0 = time.time()
gev = G.GrowthEvaluator(cfg, out_dir / "eval", a.train_n, None, v3=g.get("v3"))
print("eval pronto", time.time() - t0, len(gev.ev.svm_train_y), flush=True)

if a.case == "rs306_w455":
    parent, _ = G.backbone_from_production(CK["rs306"], "remote_sensing", 224)
    ops = [{"op": "widen", "widths": {"0": 455, "1": 455}, "seed": 7}]
elif a.case == "tiny_chain":
    parent, _ = G.backbone_from_production(CK["tiny"], "remote_sensing", 224)
    ops = None
    for th3 in (0.9, 0.8, 0.7, 0.6, 0.5):
        o, why = G.plan_with_family(parent, "B1", ("widen_all", None), [0.5, 0.5, th3, 0.5], 3, g)
        if o:
            ops = o
            break
    assert ops, why
else:
    raise SystemExit(a.case)
print("ops", json.dumps(ops), flush=True)
seed = 11
nfi = G.NewFilterInit(gev, g["init_method"], g.get("init_config") or {}, int(g.get("spifil_images_per_class", 4)), seed)
res = G.realize_plan(parent, ops, nfi, g)
print("filho", res.child.spec.cost.params_total, [L.channels for L in res.child.spec.layers()],
      "nfi", json.dumps(nfi.log, default=str)[:600], flush=True)

# ---- mesmo laço do train_child (só a fase AE)
from ga_search.modehb.external_seed import ElementProtect
from ga_search.modehb.local_ae import train_local_ae
from ga_search.modehb import modehb as M
from ga_search.modehb.run_search import _amp_kw

ev, dev = gev.ev, gev.device
tr = gev.cfg["train"]
mem = G._mem(g)
model = res.child.to(dev)
masks = {k: v.to(dev) for k, v in res.masks.items()}
scale = G.PROTECT_SCALE[g["protect"]]
ae_ep, _ = split_epochs(a.fid, tr.get("ae_ssl_split", [1, 2]))
new_blocks = [j for j in range(len(model.blocks))
              if any(not bool(masks[f"blocks.{j}.{k}"].all()) for k, _ in model.blocks[j].named_parameters())]


class Rec:
    def __init__(self):
        self.rows = []

    def log(self, d, step=None):
        self.rows.append({"step": step, **{k: v for k, v in d.items() if isinstance(v, (int, float))}})

    def alert(self, *a, **k):
        print("ALERT", a, flush=True)


@torch.no_grad()
def diag_block(model, j, x):
    h = x
    for b in model.blocks[:j]:
        h = b(h)
    blk = model.blocks[j]
    mk = {k: v for k, v in masks.items() if k.startswith(f"blocks.{j}.")}
    z = blk(h)
    out_mask = None
    for n, p in blk.named_parameters():
        m = mk.get(f"blocks.{j}.{n}")
        if m is not None and p.dim() == 4 and p.shape[0] == z.shape[1]:
            out_mask = m.reshape(m.shape[0], -1).all(1)   # True = canal herdado
    d = {"in_abs_mean": float(h.abs().mean()), "out_abs_mean": float(z.abs().mean()),
         "out_std": float(z.std()), "C_out": int(z.shape[1])}
    if out_mask is not None and (~out_mask).any():
        zc = z.transpose(0, 1).reshape(z.shape[1], -1)
        d["old_ch_rms"] = float(zc[out_mask].pow(2).mean().sqrt())
        d["new_ch_rms"] = float(zc[~out_mask].pow(2).mean().sqrt())
        d["n_new"] = int((~out_mask).sum())
    wn = {}
    for n, p in blk.named_parameters():
        m = mk.get(f"blocks.{j}.{n}")
        if p.dim() >= 2 and m is not None:
            rows = p.reshape(p.shape[0], -1)
            mr = m.reshape(m.shape[0], -1).all(1)
            wn[n] = {"old_row_norm": float(rows[mr].norm(dim=1).mean()) if mr.any() else None,
                     "new_row_norm": float(rows[~mr].norm(dim=1).mean()) if (~mr).any() else None,
                     "fan_in": int(rows.shape[1])}
    d["w"] = wn
    return d


xs = ev.svm_train_x[:64].to(dev).float()
out = {"case": a.case, "ops": ops, "fid": a.fid, "ae_epochs": ae_ep, "gset": a.gset, "blocks": []}
torch.manual_seed(seed)
for j in new_blocks:
    for p in model.parameters():
        p.grad = None
    prefix = nn.Sequential(*model.blocks[:j]).to(dev).eval()
    prefix.out_channels = G._block_in(model.spec, j)
    blk = model.blocks[j]
    pre = diag_block(model, j, xs)
    rec = Rec()
    KW = dict(_amp_kw(ev))
    if hasattr(G, "ae_extra_kw"):
        KW.update(G.ae_extra_kw(g))
    with ElementProtect(blk, G._masks_under(masks, f"blocks.{j}."), scale):
        ae = train_local_ae(prefix, blk, ev.train_loader, g.get("ae_target", "pixel"), ae_ep,
                            float(g.get("lr_ae", tr.get("lr_ae", 1e-3))), dev, rec, seed + j,
                            conv_lr_scale=1.0, decoder_warmup_frac=M.AE_WARMUP_FRAC[g.get("ae_schedule", "joint")],
                            progress=f"[ae] bloco {j}", **KW)
    post = diag_block(model, j, xs)
    curve = [r["ae/recon_loss"] for r in rec.rows if "ae/recon_loss" in r]
    gn = [{k: v for k, v in r.items() if k.startswith("ae/grad_norm")} for r in rec.rows[:3]]
    ae_s = {k: v for k, v in ae.items() if isinstance(v, (int, float, str, bool))}
    print(f"bloco {j}: curva={['%.4g' % c for c in curve]} ratio={ae_s.get('recon_ratio')}", flush=True)
    out["blocks"].append({"block": j, "curve": curve, "ae": ae_s, "pre": pre, "post": post, "grad_first": gn})
Path(a.out).write_text(json.dumps(out, indent=1, default=str))
print("ok", time.time() - t0)
