"""Summary estilo PyTorch (forward real com hooks) dos melhores modelos por base (escolha pela VAL; κ teste só informativo)."""
import sys, torch, torch.nn as nn
sys.path.insert(0, '/dados/home/moliveira/Scalable_Hybrid_FLIM')
torch.set_grad_enabled(False)
A = '/dados/home/moliveira/Scalable_Hybrid_FLIM/artifacts/'
MODELS = [
 ('COCONUT — GA híbrido c00001 f200 (κ teste 0,813, cabeça atual; passa os 5 FMs)', A+'modehb_hybrid/coconut_trees/state/seed42/candidates/c00001/f200/encoder.pt'),
 ('COCONUT — GA LEMONADE c00003 f200 (melhor pela val no ga_best; κ teste 0,806 com 2ª ordem + TTA8)', A+'modehb_lemonade/coconut_trees/state/seed42/candidates/c00003/f200/encoder.pt'),
 ('AID — GA híbrido c00043 f5 (κ teste 0,825 com 2ª ordem + TTA8)', A+'modehb_hybrid/aid/state/seed42/candidates/c00043/f5/encoder.pt'),
 ('HAM10000 — GA híbrido c00034 f45 (κ teste 0,615 com 2ª ordem + TTA8; passa 3 FMs)', A+'modehb_hybrid/ham10000/state/seed42/candidates/c00034/f45/encoder.pt'),
 ('FLOWERS — B0 SPiFiL c00003 (seed01; κ teste 0,796 com 2ª ordem)', A+'modehb_spifil/flowers/state/seed42/candidates/c00003/f15/encoder.pt'),
 ('FLOWERS — GA híbrido c00002 f200 (melhor do GA no teste: 0,729 com 2ª ordem)', A+'modehb_hybrid/flowers/state/seed42/candidates/c00002/f200/encoder.pt'),
 ('REMOTE_SENSING — GA híbrido C01 c00010 f200 (melhor do GA: κ teste 0,625 com 2ª ordem)', A+'modehb_hybrid_C01_final/remote_sensing/state/seed42/candidates/c00010/f200/encoder.pt'),
]
def label(m):
    t = type(m).__name__
    if isinstance(m, nn.Conv2d):
        t = f"Conv2d k{m.kernel_size[0]} {m.in_channels}→{m.out_channels}" + (f" g{m.groups}" if m.groups > 1 else "")
    elif isinstance(m, nn.Linear):
        t = f"Linear {m.in_features}→{m.out_features}"
    return t
def summary(path, title, out):
    d = torch.load(path, map_location='cpu', weights_only=False)
    enc, head = d['encoder'], d['head']
    model = nn.ModuleDict({'encoder': enc, 'head': head})
    rows, hooks = [], []
    def mk(name):
        def h(mod, inp, o):
            x = inp[0] if isinstance(inp, tuple) and inp else inp
            o = o[0] if isinstance(o, (tuple, list)) else o
            rows.append((name, label(mod), tuple(x.shape) if torch.is_tensor(x) else None,
                         tuple(o.shape) if torch.is_tensor(o) else None,
                         sum(p.numel() for p in mod.parameters(recurse=False))))
        return h
    for name, m in model.named_modules():
        if name and len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(mk(name)))
    model.eval()
    feat = enc(torch.zeros(1, 3, 224, 224)); emb = head(feat)
    for h in hooks: h.remove()
    W = 112
    p = lambda s='': out.write(s + '\n')
    p('=' * W); p(title); p(path.replace(A, 'artifacts/')); p('=' * W)
    p(f"{'Layer (nome)':44s} {'Tipo':26s} {'Saída':22s} {'Param #':>10s}  Operação"); p('-' * W)
    for name, t, ish, osh, n in rows:
        op = f"não muda shape: {t}" if ish and osh and ish == osh else ''
        p(f"{name:44s} {t:26s} {str(list(osh[1:])) if osh else '-':22s} {n:>10,}  {op}")
    tot = sum(x.numel() for x in model.parameters()); e = sum(x.numel() for x in enc.parameters())
    p('-' * W)
    p(f"Total params: {tot:,} | encoder {e:,} | cabeça {tot-e:,} | entrada 1×3×224×224 | mapa final {list(feat.shape[1:])} | embedding {list(emb.shape[1:])}")
    g = d.get('genome') or {}
    gl = {k: g[k] for k in ('init', 'ssl', 'reg', 'pooling', 'embed_dim', 'protect', 'ae_target', 'projector') if k in g}
    if gl: p(f"Genes globais: {gl}")
    p(f"Leitura 2ª ordem (sem treino, fora do modelo): μ+σ+√Cov do mapa final, PCA k=88 (treino) → 4096")
    p()
if __name__ == '__main__':
    import io
    buf = io.StringIO()
    for t, pth in MODELS:
        try: summary(pth, t, buf)
        except Exception as ex: buf.write(f"{t}: ERRO {ex}\n\n")
    s = buf.getvalue(); print(s)
    open('/dados/home/moliveira/Scalable_Hybrid_FLIM/research/spifil_b0b4/growth/summaries/torch_summary_melhores_20261118.txt', 'w').write(s)
