"""CPU inspection of SPiFiL filter banks: best-by-validation checkpoint per dataset x stratum (fitness C100_kw1_cov_head, Jaci).
For each Conv2d of the encoder, for the three stored states (anchor = SPiFiL+AE before SSL; ssl_state; final = WiSE mix):
effective rank, rank99, pairwise |cos|, norms, near-null filters, drift anchor->final.
Activation stats (dead channels, activation erank) on 16 SPiFiL donor images (train split) through the FINAL encoder.
Read-only; inference only (no training)."""
import glob, json, os, sys, math
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
torch.set_num_threads(4)
os.chdir('/dados/home/moliveira/Scalable_Hybrid_FLIM'); sys.path.insert(0, '.')
from spifil.data import Sample
from flim.spifil import encoder_features

P = 'research/spifil_b0b4/agent1/parts'
d = pd.read_csv(P + '/A1a_data.csv', low_memory=False)
c = d[(d.status == 'ok') & d.fitness_sig.str.startswith('C100_kw1_cov') & (d.host == 'jaci') & d.checkpoint.notna()]
c = c[c.role.fillna('search') == 'search']
pick = c.sort_values(['val_kappa', 'fidelity'], ascending=False).groupby(['dataset', 'stratum']).head(1)
# + protect=freeze best per dataset (anchor conv == SPiFiL init for freshly fitted layers)
fr = c[c.protect == 'freeze'].sort_values('val_kappa', ascending=False).groupby('dataset').head(1)
pick = pd.concat([pick.assign(why='best_val_stratum'), fr.assign(why='best_val_freeze')]).drop_duplicates(['run', 'dataset', 'cand_id', 'fidelity'])
print(len(pick), 'checkpoints')

def bank_stats(W):
    W = W.detach().double()
    C = W.shape[0]; F = W.reshape(C, -1); Dd = F.shape[1]
    n = F.norm(dim=1)
    med = n.median().clamp_min(1e-12)
    s = torch.linalg.svdvals(F)
    p = s / s.sum().clamp_min(1e-12); p = p[p > 0]
    er = float(torch.exp(-(p * p.log()).sum()))
    e = (s ** 2).cumsum(0) / (s ** 2).sum().clamp_min(1e-12)
    r99 = int((e < 0.99).sum()) + 1
    out = dict(C=C, D=Dd, erank=er, erank_ratio=er / min(C, Dd), rank99_ratio=r99 / min(C, Dd),
               norm_mean=float(n.mean()), norm_cv=float(n.std() / n.mean().clamp_min(1e-12)) if C > 1 else np.nan,
               frac_near_null=float((n < 1e-2 * med).double().mean()), frac_zero=float((n < 1e-10).double().mean()))
    if C > 1:
        Fn = F / n.clamp_min(1e-12)[:, None]
        cs = (Fn @ Fn.T).abs(); iu = torch.triu_indices(C, C, 1)
        v = cs[iu[0], iu[1]]
        out.update(mean_abs_cos=float(v.mean()), frac_cos_gt09=float((v > 0.9).double().mean()), max_abs_cos=float(v.max()))
        # filters with a near-duplicate partner
        cs.fill_diagonal_(0); out['frac_with_dup_abs'] = float((cs.max(1).values > 0.95).double().mean())
        sg = Fn @ Fn.T; sg.fill_diagonal_(-1); out['frac_with_dup'] = float((sg.max(1).values > 0.95).double().mean())
        out['frac_with_dup09'] = float((sg.max(1).values > 0.9).double().mean())
    return out

rows, act_rows = [], []
for r in pick.itertuples():
    ck_path = r.checkpoint
    if not Path(ck_path).exists():
        alt = glob.glob(f"{r.run}*/{r.dataset}/state/seed42/candidates/{r.cand_id}/f{int(r.fidelity)}/encoder.pt")
        if not alt: print('missing', ck_path); continue
        ck_path = alt[0]
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    enc = ck['encoder']
    states = {'final': {k: v for k, v in enc.state_dict().items()}}
    for nm in ('anchor', 'ssl_state'):
        sd = ck.get(nm) or {}
        if isinstance(sd, dict) and 'encoder' in sd: states[nm] = sd['encoder']
    origins = [((L.get('prov') or {}).get('origin')) for L in []]
    res = json.load(open(Path(ck_path).parent / 'result.json')) if (Path(ck_path).parent / 'result.json').exists() else {}
    lay_orig = [(L.get('prov') or {}).get('origin') for L in (res.get('layers') or [])]
    convs = [(n, m) for n, m in enc.named_modules() if isinstance(m, torch.nn.Conv2d)]
    for name, m in convs:
        blk = int(name.split('.')[0])
        key = name + '.weight'
        base = dict(dataset=r.dataset, stratum=r.stratum, why=r.why, run=r.run, cand=r.cand_id, fid=r.fidelity, val_kappa=r.val_kappa,
                    test_orig=r.test_kappa_orig, protect=r.protect, wise_alpha=r.wise_alpha, ipc=r.spifil_ipc, block=blk, conv=name,
                    groups=m.groups, k=m.kernel_size[0], origin=lay_orig[blk] if blk < len(lay_orig) else None, checkpoint=ck_path)
        for st, sd in states.items():
            if key not in sd: continue
            W = sd[key]
            if m.groups > 1:   # depthwise: treat each filter as (1,k,k)
                W = W.reshape(W.shape[0], -1)
            rows.append(dict(base, state=st, **bank_stats(W)))
        if 'anchor' in states and key in states['anchor']:
            A = states['anchor'][key].double().reshape(m.out_channels, -1); Fw = states['final'][key].double().reshape(m.out_channels, -1)
            cosd = torch.nn.functional.cosine_similarity(A, Fw, dim=1)
            fin_row = [x for x in rows if x['conv'] == name and x['cand'] == r.cand_id and x['fid'] == r.fidelity and x['state'] == 'final' and x['run'] == r.run][-1]
            fin_row['drift_cos_anchor_final'] = float(cosd.mean())
            fin_row['drift_rel_norm'] = float((Fw - A).norm() / A.norm().clamp_min(1e-12))
            if 'ssl_state' in states and key in states['ssl_state']:
                S_ = states['ssl_state'][key].double().reshape(m.out_channels, -1)
                fin_row['drift_rel_norm_anchor_ssl'] = float((S_ - A).norm() / A.norm().clamp_min(1e-12))
    # activations on donors
    cfg = yaml.safe_load(open(f'ga_search/config/modehb/datasets/{r.dataset}.yaml'))
    idir = cfg['images_dir'].format(root=cfg['root'])
    ids = res.get('spifil_sample_ids') or []
    by = {}
    for n in ids: by.setdefault(n.split('_')[0], []).append(n)
    names = [n for k in sorted(by) for n in by[k][:max(1, 16 // max(1, len(by)))]][:16]
    samples = []
    for n in names:
        ip = next(iter(glob.glob(f'{idir}/{n}.*')), None)
        if ip: samples.append(Sample(name=n, image_path=Path(ip), mask_path=None, label=int(n.split('_')[0])))
    if samples:
        enc.eval()
        with torch.no_grad():
            xs = []
            from core.data.loaders import ift_lab_loader
            from core.data.transforms import build_test
            tf = build_test(224, imagenet_norm=False)
            x = torch.stack([tf(torch.from_numpy(np.ascontiguousarray(ift_lab_loader(s.image_path))).permute(2, 0, 1).float()) for s in samples])
            h = x
            for bi, blk in enumerate(enc):
                h = blk(h)
                f = h.permute(0, 2, 3, 1).reshape(-1, h.shape[1]).double()
                sd_ = f.std(0); mx = f.abs().max(0).values
                dead = (sd_ < 1e-6 * sd_.max().clamp_min(1e-12)) | (mx < 1e-8)
                fc = f - f.mean(0)
                sv = torch.linalg.svdvals(fc[:: max(1, fc.shape[0] // 20000)])
                p = sv / sv.sum().clamp_min(1e-12); p = p[p > 0]
                er = float(torch.exp(-(p * p.log()).sum()))
                z = fc / sd_.clamp_min(1e-12)
                corr = (z.T @ z) / z.shape[0]
                iu = torch.triu_indices(h.shape[1], h.shape[1], 1)
                act_rows.append(dict(dataset=r.dataset, stratum=r.stratum, why=r.why, cand=r.cand_id, fid=r.fidelity, block=bi, C=h.shape[1],
                                     H=h.shape[-1], frac_dead=float(dead.double().mean()), act_erank=er, act_erank_ratio=er / h.shape[1],
                                     mean_abs_corr=float(corr[iu[0], iu[1]].abs().mean()) if h.shape[1] > 1 else np.nan,
                                     frac_sparse_relu=float((f <= 0).double().mean()), n_img=len(samples)))
    print(r.dataset, r.stratum, r.cand_id, r.fidelity, flush=True)
pd.DataFrame(rows).to_csv(P + '/A1a_filter_stats.csv', index=False)
pd.DataFrame(act_rows).to_csv(P + '/A1a_activation_stats.csv', index=False)
print('done', len(rows), len(act_rows))
