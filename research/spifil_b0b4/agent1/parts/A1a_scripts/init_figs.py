import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
P = '/dados/home/moliveira/Scalable_Hybrid_FLIM/research/spifil_b0b4/agent1/parts'; FIG = P + '/figs'
OUT = open('/tmp/claude-1014/-dados-home-moliveira-Scalable-Hybrid-FLIM/05a95f35-17a2-4976-a2bd-9c4d1e4ff9eb/scratchpad/a1a/tables_init.md', 'w')
def w(*a): print(*a, file=OUT)
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948', '#52514e']
plt.rcParams.update({'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'axes.grid': True, 'grid.color': '#e6e5e0',
                     'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
S = pd.read_csv(P + '/A1a_seed_distinct.csv')
G = [224, 112, 56, 28, 14, 7, 4]
m = S.groupby('dataset')[[f'distinct_{g}' for g in G] + ['mask_coverage']].mean()
w('## TB1 sementes DISTINTAS por imagem após reescala para a grade da camada (100 superpixels SLIC/imagem; medido)')
w(m.round(1).reset_index().to_markdown(index=False)); w()
fig, ax = plt.subplots(figsize=(7, 4))
order = ['remote_sensing', 'ham10000', 'flowers', 'aid', 'eggs', 'larvae', 'protozoan', 'larvae_raw', 'protozoan_raw']
for i, ds in enumerate(order):
    if ds not in m.index: continue
    ax.plot(range(len(G)), m.loc[ds, [f'distinct_{g}' for g in G]].values, marker='o', ms=4, lw=2, color=PAL[i], ls='-' if i < 4 else '--', label=f"{ds} (máscara {m.loc[ds,'mask_coverage']*100:.0f}%)")
ax.set_xticks(range(len(G))); ax.set_xticklabels([f'{g}×{g}' for g in G])
ax.set_xlabel('grade de entrada da camada (224 → após k pools)'); ax.set_ylabel('posições distintas por imagem (de 100 sementes)')
ax.legend(frameon=False, fontsize=7); ax.set_title('Candidatos SPiFiL efetivos por imagem colapsam nas camadas profundas', fontsize=10)
fig.tight_layout(); fig.savefig(FIG + '/F6_seed_distinct_per_grid.png', dpi=130); plt.close(fig)

L = pd.read_csv(P + '/A1a_layers.csv', low_memory=False)
L = L[(L.origin == 'spifil') & L.fitness_sig.str.startswith('C100_kw1_cov')].copy()
def neff(r):
    g = int(r.in_res); col = f'distinct_{g}' if g in G else None
    if r.dataset not in m.index or col is None: return np.nan
    return r.n_images * m.loc[r.dataset, col]
L['N_eff'] = L.apply(neff, axis=1)
L['ND_nominal'] = L.patch_count / L.patch_dim; L['ND_eff'] = L.N_eff / L.patch_dim
L['grupo'] = np.where(L.dataset.isin(['eggs', 'larvae', 'protozoan', 'larvae_raw', 'protozoan_raw', 'eggs_raw']), 'parasitas (máscara)', 'natural (sem máscara)')
t = L.groupby(['grupo', 'in_res']).agg(n=('patch_count', 'size'), N_nominal=('patch_count', 'median'), N_eff=('N_eff', 'median'), D=('patch_dim', 'median'),
        ND_nominal=('ND_nominal', 'median'), ND_eff=('ND_eff', 'median'), frac_ND_eff_lt5=('ND_eff', lambda s: (s < 5).mean()),
        frac_ND_eff_lt1=('ND_eff', lambda s: (s <= 1).mean())).reset_index().sort_values(['grupo', 'in_res'], ascending=[True, False])
w('## TB2 orçamento de covariância por camada: N nominal (com duplicatas) × N efetivo (posições distintas) e D = C_in·k² (fitness atual, camadas SPiFiL)')
w(t.round(2).to_markdown(index=False)); w()
L['shrink'] = L.act_ch / L.req_ch
t = L.groupby('dataset').agg(n_camadas=('shrink', 'size'), ipc_mediano=('ipc', 'median'), n_imgs_mediano=('n_images', 'median'),
        shrink_mediano=('shrink', 'median'), shrink_p10=('shrink', lambda s: s.quantile(.1)), frac_shrink_lt_09=('shrink', lambda s: (s < 0.9).mean()),
        frac_exact_small=('exact_small', lambda s: s.fillna(False).astype(bool).mean())).reset_index()
w('## TB3 largura efetiva/requisitada pelo alocador por classe e rota exact_small (fitness atual)'); w(t.round(3).to_markdown(index=False)); w()
L.to_csv(P + '/A1a_layers_neff.csv', index=False)
fig, ax = plt.subplots(figsize=(7, 4))
for i, (gname, gg) in enumerate(L.groupby('grupo')):
    q = gg.groupby('in_res').ND_eff.median().sort_index(ascending=False)
    qn = gg.groupby('in_res').ND_nominal.median().sort_index(ascending=False)
    ax.plot([str(x) for x in q.index], q.values, marker='o', lw=2, color=PAL[i], label=f'{gname}: N efetivo/D')
    ax.plot([str(x) for x in qn.index], qn.values, marker='o', lw=1.2, ls=':', color=PAL[i], label=f'{gname}: N nominal/D')
ax.set_yscale('log'); ax.axhline(1, color='#e34948', lw=1); ax.text(0.02, 1.15, 'N = D (limite do budget_check)', color='#e34948', fontsize=7, transform=ax.get_yaxis_transform())
ax.set_xlabel('grade de entrada da camada'); ax.set_ylabel('N/D (mediana)'); ax.legend(frameon=False, fontsize=7)
ax.set_title('Amostras por dimensão do patch: o budget_check conta sementes duplicadas', fontsize=10)
fig.tight_layout(); fig.savefig(FIG + '/F8_ND_eff_per_layer.png', dpi=130); plt.close(fig)

F = pd.read_csv(P + '/A1a_filter_stats.csv'); A = pd.read_csv(P + '/A1a_activation_stats.csv')
fin = F[(F.groups == 1) & (F.state == 'final')]
an = F[(F.groups == 1) & (F.state == 'anchor') & (F.protect == 'freeze') & (F.origin == 'spifil')]
cols = ['C', 'D', 'erank_ratio', 'rank99_ratio', 'mean_abs_cos', 'frac_cos_gt09', 'frac_with_dup', 'frac_with_dup09', 'frac_near_null']
w('## TB4 estatísticas do banco de filtros por bloco (checkpoint final; 38 checkpoints = melhor κ val por dataset×estrato + melhor freeze; convs densas)')
w(fin.groupby('block')[cols].median().round(3).reset_index().to_markdown(index=False)); w(f'n convs = {len(fin)}'); w()
w('## TB4b idem, só âncoras de candidatos protect=freeze em camadas de origem spifil (= filtros SPiFiL puros, sem AE/SSL nos pesos conv)')
w(an.groupby('block')[cols].median().round(3).reset_index().to_markdown(index=False)); w(f'n convs = {len(an)}'); w()
w('## TB4c por kernel (final)'); w(fin.groupby('k')[cols].median().round(3).reset_index().to_markdown(index=False)); w()
w('## TB5 deriva dos pesos conv: âncora (SPiFiL+AE) → SSL e → final (WiSE), ‖ΔW‖/‖W‖ mediana')
w(fin.groupby('protect')[['drift_rel_norm_anchor_ssl', 'drift_rel_norm', 'drift_cos_anchor_final']].agg(['median', 'max']).round(4).reset_index().to_markdown(index=False)); w()
w('## TB6 ativações (16 imagens doadoras de treino, encoder final): canais mortos, rank efetivo e |corr| entre canais por bloco')
w(A.groupby('block')[['C', 'H', 'frac_dead', 'act_erank_ratio', 'mean_abs_corr']].median().round(3).reset_index().to_markdown(index=False)); w()
w(A[A.frac_dead > 0][['dataset', 'stratum', 'cand', 'block', 'C', 'H', 'frac_dead']].round(3).to_markdown(index=False)); w()
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, (col, lab, src) in zip(axs, [('erank_ratio', 'rank efetivo / min(C,D) (pesos)', fin), ('frac_with_dup09', 'fração de filtros com par cos>0,9', fin), ('act_erank_ratio', 'rank efetivo / C (ativações)', A)]):
    bl = sorted(src.block.unique())
    data = [src[src.block == b][col].dropna().values for b in bl]
    bp = ax.boxplot(data, tick_labels=[str(b) for b in bl], patch_artist=True, widths=0.5, medianprops=dict(color='#0b0b0b'))
    for b in bp['boxes']: b.set_facecolor('#86b6ef'); b.set_edgecolor('#52514e')
    ax.set_xlabel('bloco'); ax.set_ylabel(lab)
fig.suptitle('Diversidade dos filtros SPiFiL por profundidade (38 checkpoints, melhor κ val por dataset×estrato)', fontsize=10)
fig.tight_layout(); fig.savefig(FIG + '/F7_filter_diversity_by_block.png', dpi=130); plt.close(fig)
OUT.close(); print('ok')
