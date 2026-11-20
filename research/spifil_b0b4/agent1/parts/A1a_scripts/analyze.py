"""A1a quantitative analyses 1,3,5,6 + architecture items. Read-only on A1a_data.csv / A1a_layers.csv."""
import json, math, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = '/dados/home/moliveira/Scalable_Hybrid_FLIM/research/spifil_b0b4/agent1/parts'
FIG = P + '/figs'
OUT = open('/tmp/claude-1014/-dados-home-moliveira-Scalable-Hybrid-FLIM/05a95f35-17a2-4976-a2bd-9c4d1e4ff9eb/scratchpad/a1a/tables.md', 'w')
def w(*a): print(*a, file=OUT)
def md(df, floatfmt=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == 'f': df[c] = df[c].map(lambda v: '' if pd.isna(v) else f'{v:.{floatfmt}f}')
    return df.to_markdown(index=False)

PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
plt.rcParams.update({'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'axes.edgecolor': '#52514e',
                     'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False})

d = pd.read_csv(P + '/A1a_data.csv', low_memory=False)
ok = d[d.status.astype(str).str.startswith('ok') & d.val_kappa.notna()].copy()
ok['role'] = ok.role.fillna('search')
ok = ok[ok.role == 'search']
ok['sig_full'] = ok.fitness_sig
ok['regua_fit'] = np.where(ok.fitness_sig.str.contains('cov_head'), 'cov_head+scaler', 'head(sem scaler)')
ok['grp'] = ok.kind + '|' + ok.fitness_sig_short
CUR = ok[ok.fitness_sig.str.startswith('C100_kw1_cov_head')]          # current fitness (C=100, kw=1, cov_head)
# percentile of val kappa inside (host, run, dataset, sig, fidelity)
ok['kpct'] = ok.groupby(['host', 'run', 'dataset', 'fitness_sig', 'fidelity']).val_kappa.rank(pct=True)
ok['kz'] = ok.groupby(['host', 'run', 'dataset', 'fitness_sig', 'fidelity']).val_kappa.transform(lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1))
CUR = ok[ok.fitness_sig.str.startswith('C100_kw1_cov_head')]
ORDER = ['remote_sensing', 'ham10000', 'flowers', 'aid', 'coconut_trees', 'eggs', 'eggs_raw', 'larvae', 'larvae_raw', 'protozoan', 'protozoan_raw']

# ---------------- 0. counts
w('## T0 contagem de jobs ok (role=search) por dataset × busca × assinatura de fitness × fidelidade')
t = ok.groupby(['dataset', 'kind', 'fitness_sig', 'host']).agg(n=('val_kappa', 'size'), n_f5=('fidelity', lambda s: (s == 5).sum()),
        n_fge45=('fidelity', lambda s: (s >= 45).sum()), n_test=('test_kappa_orig', 'count'), kval_max=('val_kappa', 'max'),
        params_max=('params', 'max'), depth_max=('depth', 'max')).reset_index()
w(md(t)); w()
fail = d[d.status == 'failed']
w('## T0b jobs falhos por run (motivo truncado)')
fail2 = fail.assign(motivo=fail.error.fillna('').str.slice(0, 60)).groupby(['run', 'dataset', 'motivo']).size().reset_index(name='n').sort_values('n', ascending=False).head(25)
w(md(fail2)); w()
w('## T0c tipos de op por run (jobs ok)')
w(pd.crosstab(ok.run.str.replace('artifacts/', '') + '/' + ok.dataset, ok.op_type.fillna('none')).reset_index().to_markdown(index=False)); w()

# ---------------- 1. spearman
def sp(g, c):
    x = g[[c, 'val_kappa']].dropna()
    if len(x) < 6 or x[c].nunique() < 2: return np.nan
    return spearmanr(x[c], x.val_kappa)[0]
def partial_sp(g, c, ctrl='params'):
    x = g[[c, ctrl, 'val_kappa']].dropna()
    if len(x) < 8 or x[c].nunique() < 2: return np.nan
    r = x.rank()
    def res(a, b):
        A = np.c_[np.ones(len(r)), r[b]]; return r[a] - A @ np.linalg.lstsq(A, r[a], rcond=None)[0]
    return np.corrcoef(res(c, ctrl), res('val_kappa', ctrl))[0, 1]
rows = []
for (ds, k, s), g in ok[ok.fidelity == 5].groupby(['dataset', 'kind', 'fitness_sig']):
    if len(g) < 12: continue
    rows.append(dict(dataset=ds, busca=k, fitness=s, n=len(g), **{f'ρ_{c}': sp(g, c) for c in ['params', 'flops', 'depth', 'max_width', 'n_pools', 'spifil_ipc']},
                     ρ_depth_parcial_params=partial_sp(g, 'depth'), ρ_ipc_parcial_params=partial_sp(g, 'spifil_ipc')))
T1 = pd.DataFrame(rows); T1['ds_o'] = T1.dataset.map({k: i for i, k in enumerate(ORDER)}); T1 = T1.sort_values(['ds_o', 'busca']).drop(columns='ds_o')
w('## T1 Spearman ρ(κ_val, X) na fidelidade 5 (n≥12), por dataset × busca × assinatura de fitness')
w(md(T1, 2)); w()
# pooled within-group: kz vs log params by readout type
for lab, sub in [('cov_head (atual)', ok[(ok.fidelity == 5) & ok.fitness_sig.str.contains('cov_head')]),
                 ('head sem scaler (legado)', ok[(ok.fidelity == 5) & ~ok.fitness_sig.str.contains('cov_head')])]:
    w(f'- agregado f5 {lab}: n={len(sub)}, ρ(percentil κ, params)={spearmanr(sub.kpct, sub.params)[0]:.2f}, ρ(percentil κ, depth)={spearmanr(sub.kpct, sub.depth)[0]:.2f}, ρ(percentil κ, n_pools)={spearmanr(sub.kpct, sub.n_pools)[0]:.2f}, ρ(percentil, max_width)={spearmanr(sub.kpct, sub.max_width)[0]:.2f}')
w()

# per-candidate max fidelity, CUR only
top = CUR.sort_values('fidelity').groupby(['host', 'run', 'dataset', 'cand_id']).tail(1)
rows = []
for ds, g in top.groupby('dataset'):
    if len(g) < 8: continue
    rows.append(dict(dataset=ds, n=len(g), **{f'ρ_{c}': sp(g, c) for c in ['params', 'flops', 'depth', 'max_width', 'n_pools']}))
w('## T1b Spearman com a fidelidade MAIS ALTA de cada candidato (fitness atual C100_kw1_cov_head; mistura fidelidades)')
w(md(pd.DataFrame(rows), 2)); w()

# F1: kappa vs params grid
dss = [x for x in ORDER if x in set(CUR.dataset)]
fig, axs = plt.subplots(3, 4, figsize=(15, 10.5)); axs = axs.ravel()
kinds = ['hybrid', 'lemonade', 'pnas_cs']
for i, ds in enumerate(dss):
    ax = axs[i]; g = CUR[CUR.dataset == ds]
    for kk, kind in enumerate(kinds):
        gg = g[g.kind == kind]
        if gg.empty: continue
        f5 = gg[gg.fidelity == 5]; hi = gg[gg.fidelity > 5]
        ax.scatter(f5.params, f5.val_kappa, s=14, color=PAL[kk], alpha=0.55, edgecolor='none', label=f'{kind} f5')
        ax.scatter(hi.params, hi.val_kappa, s=30, color=PAL[kk], marker='D', edgecolor='#fcfcfb', linewidth=0.8, label=f'{kind} f≥15')
    for b in [3e5, 8e5, 1.5e6, 3e6]: ax.axvline(b, color='#c3c2b7', lw=0.8, ls='--')
    ax.set_xscale('log'); ax.set_title(ds); ax.set_xlabel('params (encoder + proj. 1×1)'); ax.set_ylabel('κ val (fitness cov_head)')
for j in range(len(dss), len(axs)): axs[j].axis('off')
h, l = axs[dss.index('remote_sensing')].get_legend_handles_labels()
axs[-1].legend(h, l, loc='center', frameon=False, title='busca / fidelidade')
fig.suptitle('κ de validação × parâmetros (fitness atual C=100, kw=1, cov_head). Linhas tracejadas: limites B0..B3', fontsize=11)
fig.tight_layout(); fig.savefig(FIG + '/F1_kval_vs_params.png', dpi=130); plt.close(fig)

# F1b: legacy head readout vs cov readout on RS (same dataset), f5
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
for ax, (lab, sub) in zip(axs, [('fitness head sem scaler (v3/strat/spifil_v1, C100 kw0.75)', ok[(ok.dataset == 'remote_sensing') & ~ok.fitness_sig.str.contains('cov_head') & (ok.fidelity == 5)]),
                               ('fitness cov_head + scaler (hybrid/lemonade/pnas, C100 kw1)', ok[(ok.dataset == 'remote_sensing') & ok.fitness_sig.str.startswith('C100_kw1_cov') & (ok.fidelity == 5)])]):
    for kk, (kind, gg) in enumerate(sub.groupby('kind')):
        ax.scatter(gg.params, gg.val_kappa, s=14, color=PAL[kk], alpha=0.6, edgecolor='none', label=kind)
    ax.set_xscale('log'); ax.set_title(lab, fontsize=9); ax.set_xlabel('params'); ax.set_ylabel('κ val f5')
    ax.legend(frameon=False, fontsize=8)
    r = spearmanr(sub.params, sub.val_kappa)[0]; ax.text(0.02, 0.95, f'ρ={r:.2f}, n={len(sub)}', transform=ax.transAxes, va='top')
fig.suptitle('remote_sensing: a relação tamanho→κ depende da régua da fitness', fontsize=10)
fig.tight_layout(); fig.savefig(FIG + '/F1b_rs_readout_params.png', dpi=130); plt.close(fig)

# ---------------- spatial reduction
f5c = CUR[CUR.fidelity == 5]
w('## T2 redução espacial: n_pools / resolução final × κ (f5, fitness atual; percentil de κ dentro de cada busca×dataset)')
t = f5c.groupby('n_pools').agg(n=('kpct', 'size'), pct_mediano=('kpct', 'median'), kz_medio=('kz', 'mean'), res_final_med=('final_res', 'median'),
                               depth_med=('depth', 'median'), params_med=('params', 'median')).reset_index()
w(md(t, 2)); w()
t = f5c.groupby('depth').agg(n=('kpct', 'size'), pct_mediano=('kpct', 'median'), kz_medio=('kz', 'mean'), pools_med=('n_pools', 'median'), params_med=('params', 'median')).reset_index()
w('## T2b profundidade × κ (f5, fitness atual)'); w(md(t, 2)); w()
# per dataset pools table
t = f5c.pivot_table(index='dataset', columns='n_pools', values='kz', aggfunc='mean').round(2)
w('## T2c z-score médio de κ por n_pools e dataset (f5, fitness atual)'); w(t.reset_index().to_markdown(index=False)); w()
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, col, lab in zip(axs, ['depth', 'n_pools', 'final_res'], ['profundidade (blocos)', 'nº de pools', 'resolução final (px)']):
    vals = sorted(f5c[col].dropna().unique())
    data = [f5c[f5c[col] == v].kz.values for v in vals]
    bp = ax.boxplot(data, tick_labels=[str(int(v)) for v in vals], patch_artist=True, widths=0.55, medianprops=dict(color='#0b0b0b'))
    for b in bp['boxes']: b.set_facecolor('#86b6ef'); b.set_edgecolor('#52514e')
    for i, v in enumerate(vals): ax.text(i + 1, ax.get_ylim()[0] if False else -3.3, f'n={len(data[i])}', ha='center', fontsize=7, color='#52514e')
    ax.set_xlabel(lab); ax.set_ylabel('z(κ val) dentro de busca×dataset'); ax.set_ylim(-3.5, 2.8)
fig.suptitle('Arquitetura × κ val na fidelidade 5 (fitness atual, todos os datasets agregados por z-score)', fontsize=10)
fig.tight_layout(); fig.savefig(FIG + '/F2_depth_pools_res.png', dpi=130); plt.close(fig)

# ---------------- operators
ok['n_layers_ops'] = ok[['n_conv', 'n_dwsep', 'n_inception']].sum(axis=1)
for c in ['conv', 'dwsep', 'inception']:
    ok[f'frac_{c}'] = ok[f'n_{c}'] / ok.depth
f5c = ok[ok.fitness_sig.str.startswith('C100_kw1_cov_head') & (ok.fidelity == 5)].copy()
def dom(r):
    if r.n_inherited and r.n_inherited > 0: return 'misto/herdado'
    fr = {c: r[f'frac_{c}'] for c in ['conv', 'dwsep', 'inception']}
    m = max(fr, key=fr.get)
    return m if fr[m] > 0.5 else 'misto'
f5c['op_dom'] = f5c.apply(dom, axis=1)
t = f5c.groupby('op_dom').agg(n=('kz', 'size'), kz_medio=('kz', 'mean'), pct_mediano=('kpct', 'median'), params_med=('params', 'median'), flops_med=('flops', 'median')).reset_index()
w('## T3 operador dominante (>50% dos blocos) × κ (f5, fitness atual)'); w(md(t, 2)); w()
# OLS with dataset/run fixed effects through within-group z already; regress kz on features
X = f5c[['params', 'depth', 'n_pools', 'frac_dwsep', 'frac_inception', 'n_residual']].copy()
X['log10_params'] = np.log10(X.pop('params'))
X = X.dropna(); y = f5c.loc[X.index, 'kz']
A = np.c_[np.ones(len(X)), X.values]
beta = np.linalg.lstsq(A, y.values, rcond=None)[0]
rng = np.random.default_rng(0); B = []
for _ in range(1000):
    i = rng.integers(0, len(X), len(X)); B.append(np.linalg.lstsq(A[i], y.values[i], rcond=None)[0])
B = np.array(B); lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
w('## T3b OLS z(κ val) ~ log10(params) + depth + n_pools + frac_dwsep + frac_inception + n_residual (f5, fitness atual; IC 95% bootstrap)')
w(md(pd.DataFrame({'termo': ['intercepto'] + list(X.columns), 'coef': beta, 'ic_lo': lo, 'ic_hi': hi}), 3)); w(f'n={len(X)}'); w()
# per-layer op fraction overall in population vs in top decile
t = f5c.assign(top=f5c.kpct >= 0.9).groupby('top')[['frac_conv', 'frac_dwsep', 'frac_inception', 'n_residual', 'depth', 'n_pools']].mean().reset_index()
w('## T3c composição média: top-10% × resto (f5, fitness atual)'); w(md(t, 2)); w()
fig, ax = plt.subplots(figsize=(6, 3.6))
cats = ['conv', 'dwsep', 'inception', 'misto']
data = [f5c[f5c.op_dom == c].kz.values for c in cats]
bp = ax.boxplot(data, tick_labels=[f'{c}\n(n={len(x)})' for c, x in zip(cats, data)], patch_artist=True, widths=0.5, medianprops=dict(color='#0b0b0b'))
for b in bp['boxes']: b.set_facecolor('#86b6ef'); b.set_edgecolor('#52514e')
ax.set_ylabel('z(κ val) dentro de busca×dataset'); ax.set_title('Operador dominante × κ val (f5, fitness atual)', fontsize=10)
fig.tight_layout(); fig.savefig(FIG + '/F3_operators.png', dpi=130); plt.close(fig)

# ---------------- mutations / parent-child
idx = {}
for r in ok.itertuples():
    idx.setdefault((r.host, r.run, r.dataset, r.fitness_sig, r.cand_id), {})[float(r.fidelity)] = r
pairs = []
for r in ok.itertuples():
    par = r.op_parent if isinstance(r.op_parent, str) else None
    if not par: continue
    P_ = idx.get((r.host, r.run, r.dataset, r.fitness_sig, par))
    if not P_: continue
    f = float(r.fidelity)
    same = P_.get(f)
    pf = float(r.op_parent_fid) if not pd.isna(r.op_parent_fid) else None
    inh = P_.get(pf) if pf else None
    le = max([x for x in P_ if x <= f], default=None)
    ref = same if same is not None else (inh if inh is not None else (P_[le] if le else None))
    if ref is None: continue
    lab = r.op_morph if isinstance(r.op_morph, str) else r.op_type
    if r.op_type == 'grow': lab = 'grow(P_k→P_k+1)'
    pairs.append(dict(dataset=r.dataset, busca=r.kind, fitness=r.fitness_sig_short, op_type=r.op_type, op=lab, child=r.cand_id, parent=par,
                      fid=f, parent_fid_ref=ref.fidelity, same_fid=same is not None, k_child=r.val_kappa, k_parent=ref.val_kappa,
                      dk=r.val_kappa - ref.val_kappa, d_params=r.params - ref.params, d_depth=r.depth - ref.depth,
                      p_child=r.params, p_parent=ref.params, layer_map=r.layer_map, gpu_h=r.gpu_hours, n_reuse=r.inherit_n_blocks))
PR = pd.DataFrame(pairs)
PR.to_csv(P + '/A1a_pairs_parent_child.csv', index=False)
# promotions
prom = []
for key, fids in idx.items():
    fs = sorted(fids)
    for a, b in zip(fs[:-1], fs[1:]):
        prom.append(dict(dataset=key[2], busca=fids[b].kind, fitness=fids[b].fitness_sig_short, f_lo=a, f_hi=b, dk=fids[b].val_kappa - fids[a].val_kappa, params=fids[b].params))
PM = pd.DataFrame(prom)
def summ(g):
    return pd.Series(dict(n=len(g), dk_mediana=g.dk.median(), dk_media=g.dk.mean(), frac_melhora=(g.dk > 0).mean(),
                          dparams_med=g.d_params.median() if 'd_params' in g else np.nan,
                          dk_por_100k=(g.dk / (g.d_params.abs() / 1e5)).replace([np.inf, -np.inf], np.nan).median() if 'd_params' in g else np.nan,
                          frac_mesma_fid=g.same_fid.mean() if 'same_fid' in g else np.nan))
w('## T5 eficiência das mutações: Δκ_val = κ(filho) − κ(pai), pai na MESMA fidelidade quando existe, senão na fidelidade herdada (≤ filho)')
t = PR.groupby(['busca', 'fitness', 'op']).apply(summ).reset_index()
w(md(t, 3)); w()
t = PR.groupby(['op']).apply(summ).reset_index()
w('## T5b agregado por operador (todas as buscas/assinaturas; só remote_sensing tem pares)'); w(md(t, 3)); w()
w('## T5c datasets com pares pai→filho'); w(PR.groupby(['dataset', 'busca']).size().reset_index(name='n').to_markdown(index=False)); w()
t = PM.groupby(['f_lo', 'f_hi']).agg(n=('dk', 'size'), dk_mediana=('dk', 'median'), frac_melhora=('dk', lambda s: (s > 0).mean())).reset_index()
w('## T5d promoções (mesmo candidato, fidelidade f_lo → f_hi), todas as buscas'); w(md(t, 3)); w()
t = PM.groupby(['dataset']).agg(n=('dk', 'size'), dk_mediana=('dk', 'median'), frac_melhora=('dk', lambda s: (s > 0).mean())).reset_index()
w(md(t, 3)); w()
fig, ax = plt.subplots(figsize=(8, 3.8))
ops = PR.groupby('op').dk.median().sort_values().index.tolist()
for i, o in enumerate(ops):
    g = PR[PR.op == o]; xj = i + (np.random.default_rng(i).random(len(g)) - 0.5) * 0.35
    ax.scatter(xj, g.dk, s=16, color=np.where(g.same_fid, PAL[0], PAL[1]), alpha=0.75, edgecolor='none')
    ax.plot([i - 0.3, i + 0.3], [g.dk.median()] * 2, color='#0b0b0b', lw=2)
ax.axhline(0, color='#52514e', lw=0.8)
ax.set_xticks(range(len(ops))); ax.set_xticklabels([f'{o}\n(n={(PR.op == o).sum()})' for o in ops], fontsize=8)
ax.set_ylabel('Δκ val (filho − pai)')
pass
ax.set_title('Eficiência das mutações/morfismos (remote_sensing; LEMONADE + PNAS-CS)', fontsize=10)
fig.tight_layout(); fig.savefig(FIG + '/F4_mutation_dk.png', dpi=130); plt.close(fig)

# marginal per block
MB = PR[PR.d_depth > 0].copy()
MB['dk_por_bloco'] = MB.dk / MB.d_depth
w('## T4 benefício marginal por bloco: pares pai→filho com profundidade maior (deepen / add_layer / grow)')
w(md(MB[['busca', 'fitness', 'op', 'child', 'parent', 'fid', 'parent_fid_ref', 'k_parent', 'k_child', 'dk', 'd_depth', 'dk_por_bloco', 'p_parent', 'p_child', 'layer_map']].sort_values(['op', 'dk']), 3)); w()
w(f'- resumo: n={len(MB)}, Δκ/bloco mediana={MB.dk_por_bloco.median():.4f}, média={MB.dk_por_bloco.mean():.4f}, fração>0={(MB.dk > 0).mean():.2f}; mesma fidelidade: n={MB.same_fid.sum()}, mediana={MB[MB.same_fid].dk_por_bloco.median():.4f}')
nb = PR[PR.d_depth == 0]
w(f'- controle (mesma profundidade: widen/kernel/residual/pooling): n={len(nb)}, Δκ mediana={nb.dk.median():.4f}, fração>0={(nb.dk > 0).mean():.2f}')
w()
# PNAS families
pn = ok[(ok.kind == 'pnas_cs')]
fam = pn[pn.op_family.notna()].groupby('op_family').agg(n=('cand_id', 'nunique'), levels=('stratum', lambda s: ','.join(sorted(set(s)))), kmax=('val_kappa', 'max'),
                                                         pmin=('params', 'min'), pmax=('params', 'max')).reset_index().sort_values('kmax', ascending=False).head(15)
w('## T4b PNAS-CS: famílias (mesma base, níveis P0..P4)'); w(md(fam, 3)); w()
# same-family P-level curve: best per (family, stratum)
fb = pn[pn.op_family.notna()].groupby(['op_family', 'stratum']).val_kappa.max().unstack()
w(fb.round(3).reset_index().to_markdown(index=False)); w()

# ---------------- 6. best per budget
BUD = [('B0', 3e5), ('B1', 8e5), ('B2', 1.5e6), ('B3', 3e6), ('B4', 5.5e6)]
rows = []
for (ds, sig), g in ok.groupby(['dataset', 'fitness_sig']):
    if not sig.startswith('C100_kw1_cov'): continue
    for b, cap in BUD:
        gg = g[g.params <= cap]
        if gg.empty:
            rows.append(dict(dataset=ds, B=b, n_cand=0)); continue
        best = gg.sort_values(['val_kappa', 'fidelity'], ascending=False).iloc[0]
        band = g[(g.params <= cap) & (g.params > dict(BUD).get(f'B{int(b[1]) - 1}', 0))] if b != 'B0' else gg
        rows.append(dict(dataset=ds, B=b, n_cand=gg.cand_id.nunique(), n_na_faixa=band.cand_id.nunique(), busca=best.kind, run=best.run.replace('artifacts/', ''), host=best.host,
                         cand=best.cand_id, fid=best.fidelity, params=int(best.params), flops_G=best.flops / 1e9 if not pd.isna(best.flops) else np.nan,
                         depth=best.depth, widths=best.widths, pools=best.n_pools, k_val=best.val_kappa, k_test_ORIG=best.test_kappa_orig, k_test_COV=best.test_kappa_cov,
                         veio_de_faixa_menor=(b != 'B0' and best.params <= dict(BUD)[f'B{int(b[1]) - 1}']), checkpoint=best.checkpoint))
BB = pd.DataFrame(rows)
BB['ds_o'] = BB.dataset.map({k: i for i, k in enumerate(ORDER)}); BB = BB.sort_values(['ds_o', 'B']).drop(columns='ds_o')
BB.to_csv(P + '/A1a_best_by_budget.csv', index=False)
w('## T6 melhor checkpoint por orçamento (≤ cap, cumulativo) selecionado por κ VALIDAÇÃO (fitness atual C100_kw1_cov_head); teste só informativo')
w(md(BB.drop(columns=['checkpoint']), 3)); w()
# legacy/archived signatures: best per budget too (separate)
rows = []
for (ds, sig), g in ok.groupby(['dataset', 'fitness_sig']):
    if sig.startswith('C100_kw1_cov'): continue
    for b, cap in BUD:
        gg = g[g.params <= cap]
        if gg.empty: continue
        best = gg.sort_values(['val_kappa', 'fidelity'], ascending=False).iloc[0]
        rows.append(dict(dataset=ds, fitness=sig, B=b, n=gg.cand_id.nunique(), cand=best.cand_id, run=best.run.replace('artifacts/', ''), fid=best.fidelity, params=int(best.params), k_val=best.val_kappa, k_test_ORIG=best.test_kappa_orig))
w('## T6b idem para assinaturas antigas (NÃO comparáveis com T6: outra régua/C)'); w(md(pd.DataFrame(rows), 3)); w()
fig, ax = plt.subplots(figsize=(8, 4.2))
for i, ds in enumerate([x for x in ORDER if x in set(BB.dataset)]):
    g = BB[BB.dataset == ds]
    col = PAL[i % 8]; ls = '-' if i < 8 else '--'
    ax.plot(range(5), g.k_val.values, marker='o', ms=5, lw=2, color=col, ls=ls, label=ds)
ax.set_xticks(range(5)); ax.set_xticklabels(['B0 ≤300k', 'B1 ≤800k', 'B2 ≤1,5M', 'B3 ≤3M', 'B4 ≤5,5M'])
ax.set_ylabel('melhor κ val (cumulativo ≤ cap)'); ax.legend(frameon=False, fontsize=7, ncol=2, loc='lower right')
ax.set_title('Melhor κ de validação por orçamento (fitness atual; seleção só por validação)', fontsize=10)
fig.tight_layout(); fig.savefig(FIG + '/F5_best_by_budget.png', dpi=130); plt.close(fig)

# small dominate big? compare band max
rows = []
for ds, g in CUR.groupby('dataset'):
    band = pd.cut(g.params, [0, 3e5, 8e5, 1.5e6, 3e6, 5.5e6, 1e9], labels=['B0', 'B1', 'B2', 'B3', 'B4', '>B4'])
    m = g.groupby(band, observed=False).val_kappa.max()
    f5m = g[g.fidelity == 5].groupby(band[g.fidelity == 5], observed=False).val_kappa.median()
    rows.append(dict(dataset=ds, **{f'max_{k}': v for k, v in m.items()}, **{f'medf5_{k}': v for k, v in f5m.items()}))
w('## T6c máximo e mediana(f5) de κ val por FAIXA (não cumulativo), fitness atual'); w(md(pd.DataFrame(rows), 3)); w()

# val vs test relationship among CUR tested
tt = ok[ok.test_kappa_orig.notna()]
rows = []
for (ds, sig), g in tt.groupby(['dataset', 'fitness_sig_short']):
    if len(g) < 4: continue
    rows.append(dict(dataset=ds, fitness=sig, n=len(g), ρ_val_testORIG=spearmanr(g.val_kappa, g.test_kappa_orig)[0],
                     ρ_val_testCOV=spearmanr(g.val_kappa, g.test_kappa_cov)[0] if g.test_kappa_cov.notna().sum() >= 4 else np.nan,
                     ρ_params_testORIG=spearmanr(g.params, g.test_kappa_orig)[0]))
w('## T7 (contexto) ρ entre κ val (fitness) e κ teste ORIG / COV nas linhas testadas'); w(md(pd.DataFrame(rows), 2)); w()
OUT.close()
print('ok')
