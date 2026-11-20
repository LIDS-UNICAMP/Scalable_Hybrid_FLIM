"""A1a extraction: one row per job (all search_state.json, Jaci + athena PNAS-CS) + athena-only tested rows.
Read-only. Output: research/spifil_b0b4/agent1/parts/A1a_data.csv and A1a_layers.csv"""
import csv, glob, json, math, os, collections
from pathlib import Path

REPO = Path('/dados/home/moliveira/Scalable_Hybrid_FLIM')
ATH = Path('/dados/home/moliveira/exps/queues/relatorios_athena/all_results_bundle/athena')
OUT = REPO / 'research/spifil_b0b4/agent1/parts'
os.chdir(REPO)

def kind_of(run):
    n = Path(run).name
    if 'pnas_cs' in n: return 'pnas_cs'
    if n.startswith('modehb_lemonade'): return 'lemonade'
    if n.startswith('modehb_hybrid'): return 'hybrid'
    if n == 'modehb_spifil': return 'spifil_v1'
    if n.startswith('modehb_v3'): return 'v3'
    if n.startswith('modehb_strat'): return 'strat'
    return n

def resolved_sig(state_path):
    cfgp = Path(state_path).parent / 'config_resolved.yaml'
    if cfgp.exists():
        import yaml
        c = yaml.safe_load(cfgp.read_text())
        sv = (c.get('eval') or {}).get('svm_fitness') or {}
    else:
        sv = {}
    return sv

def sig_str(sv, run):
    if not sv:
        if 'pnas_cs' in str(run):   # pnas_cs.yaml svm_fitness (lido do yaml do repo)
            sv = {'C': 100, 'kappa_weight': 1.0, 'readout': 'cov_head', 'scaler': 'standard', 'max_train': 4000}
        else:
            return 'unknown'
    return f"C{float(sv.get('C',100)):g}_kw{float(sv.get('kappa_weight',1.0)):g}_{sv.get('readout') or 'head'}_{sv.get('scaler','none')}_mt{sv.get('max_train')}"

def short_sig(sv_s):
    # monitor_report signature (C, kw) for joining with test jsonl
    p = sv_s.split('_')
    return f"{p[0]}_{p[1]}" if len(p) > 1 else sv_s

def spatial_after(specs, size):
    h = size; res = []
    for s in specs:
        ps = int(s.get('pool_stride', 1) or 1)
        pool = s.get('pool', 'none')
        if ps > 1 or pool in ('max', 'avg'):
            if s.get('pool_padding') == 0:
                h = (h - int(s.get('pool_size', 3))) // 2 + 1
            else:
                h = math.ceil(h / 2)
        res.append(h)
    return res

def conv_kind(s):
    if s.get('origin') == 'inherited': return 'inherited'
    return s.get('conv') or s.get('op') or '?'

rows, layer_rows = [], []
states = sorted(glob.glob('artifacts/modehb_*/*/state/seed*/search_state.json'))
states.append(str(ATH / 'pnas_cs/search_state.json'))
for sp in states:
    s = json.load(open(sp))
    if str(sp).startswith(str(ATH)):
        host, run, ds = 'athena', 'artifacts/pnas_cs', s['config'].get('pnas_cs', {}).get('dataset', 'remote_sensing')
    else:
        host = 'jaci'; parts = Path(sp).parts; run = '/'.join(parts[:2]); ds = parts[2]
    sv = resolved_sig(sp) if host == 'jaci' else {}
    sig = sig_str(sv, run)
    size = int(s['config'].get('image_size', 224))
    for jn, j in enumerate(s['jobs']):
        r = j.get('result') or {}
        g = j.get('genome') or {}
        op = j.get('op') or {}
        specs = r.get('effective_specs') or []
        chans = [int(x.get('channels', 0) or 0) for x in specs]
        sp_after = spatial_after(specs, size) if specs else []
        convs = [conv_kind(x) for x in specs]
        layers = r.get('layers') or []
        provs = [(L.get('prov') or {}) for L in layers]
        origins = [p.get('origin', '?') for p in provs]
        req = [p.get('requested_channels') for p in provs]
        act = [p.get('actual_channels') for p in provs]
        inh = j.get('inherit') or {}
        row = dict(
            host=host, run=run, kind=kind_of(run), dataset=ds, fitness_sig=sig, fitness_sig_short=short_sig(sig),
            job=jn, cand_id=j.get('cand_id'), fidelity=j.get('fidelity'), status=j.get('status'), role=j.get('role'),
            stratum=j.get('stratum'), generation=j.get('generation'), parent_id=j.get('parent_id'),
            twin_of=j.get('twin_of'), init=g.get('init'),
            op_type=op.get('type'), op_morph=op.get('morphism') or op.get('mutation') or op.get('variant'),
            op_parent=op.get('parent') or inh.get('parent_cand_id'), op_parent_fid=inh.get('parent_fidelity'),
            op_layer=op.get('layer', op.get('position')), op_from=op.get('from', op.get('from_level')),
            op_to=op.get('to', op.get('to_level')), op_family=op.get('family'), op_origin=op.get('origin'),
            layer_map=json.dumps(inh.get('layer_map')) if inh.get('layer_map') is not None else '',
            inherit_n_blocks=inh.get('n_blocks'),
            params=r.get('params', j.get('params_est')), flops=r.get('flops'), latency_ms=r.get('latency_ms'),
            depth=len(specs) if specs else g.get('depth'),
            widths='-'.join(map(str, chans)), max_width=max(chans) if chans else None,
            final_width=chans[-1] if chans else None, mean_width=(sum(chans)/len(chans)) if chans else None,
            n_pools=sum(1 for a, b in zip([size]+sp_after[:-1], sp_after) if b < a) if sp_after else None,
            final_res=sp_after[-1] if sp_after else None,
            n_conv=convs.count('conv'), n_dwsep=convs.count('dwsep'), n_inception=convs.count('inception'),
            n_depthwise=convs.count('depthwise'), n_pointwise=convs.count('pointwise'), n_inherited=convs.count('inherited'),
            n_residual=sum(1 for x in specs if x.get('residual')),
            kernels='-'.join(str(x.get('kernel')) for x in specs),
            head_pooling=g.get('pooling'), ssl=g.get('ssl'), reg=g.get('reg'), protect=g.get('protect'),
            ae_schedule=g.get('ae_schedule'), ae_target=g.get('ae_target'),
            spifil_ipc=r.get('spifil_images_per_class', g.get('spifil_images_per_class')),
            spifil_n_images=r.get('spifil_n_images'),
            n_spifil_layers=origins.count('spifil'), n_inherited_layers=origins.count('inherited'),
            width_shrink=sum((a or 0) for a in act if a) / max(1, sum((q or 0) for q in req if q)) if any(req) else None,
            val_kappa=r.get('fitness_kappa'), val_f1=r.get('fitness_macro_f1'), quality=r.get('quality'),
            wise_alpha=r.get('wise_alpha'),
            wise_k0=(r.get('wise_kappas') or {}).get('0.0'), wise_k1=(r.get('wise_kappas') or {}).get('1.0'),
            collapse_dim=(r.get('collapse') or {}).get('dimensional'), emb_erank=(r.get('metrics') or {}).get('effective_rank'),
            ssl_epochs=r.get('ssl_epochs'), ae_epochs=r.get('ae_epochs_per_layer'), gpu_hours=r.get('gpu_hours'),
            checkpoint=r.get('checkpoint'), error=str(r.get('error', ''))[:160] if j.get('status') != 'ok' else '',
            source=str(sp),
        )
        rows.append(row)
        if j.get('status') == 'ok':
            hin = size
            for i, (L, p) in enumerate(zip(layers, provs)):
                e = specs[i] if i < len(specs) else {}
                ae = L.get('ae') or {}
                layer_rows.append(dict(host=host, run=run, kind=row['kind'], dataset=ds, fitness_sig=sig, cand_id=j.get('cand_id'),
                    fidelity=j.get('fidelity'), job=jn, layer=i, origin=p.get('origin'), conv=e.get('conv'), kernel=e.get('kernel'),
                    in_ch=p.get('in_channels', e.get('in_channels')), req_ch=p.get('requested_channels'), act_ch=p.get('actual_channels'),
                    in_res=hin, out_res=sp_after[i] if i < len(sp_after) else None,
                    patch_count=p.get('patch_count'), patch_dim=p.get('patch_dimension'), n_images=p.get('n_images'),
                    ipc=p.get('images_per_class'), exact_small=e.get('exact_small_width'),
                    repairs='|'.join(p.get('spifil_repairs') or []), superpixel_mode=p.get('superpixel_mode'),
                    fit_s=p.get('fit_seconds'), ae_erank_ratio=ae.get('effective_rank_ratio'), ae_recon_ratio=ae.get('recon_ratio'),
                    ae_stability=ae.get('stability'), val_kappa=row['val_kappa'], params=row['params'], depth=row['depth'], stratum=row['stratum']))
                hin = sp_after[i] if i < len(sp_after) else hin

# ---- tests (both files); merge per key, require val_kappa match for the join
tests = {}
def add_tests(path, host):
    for line in open(path):
        t = json.loads(line)
        if 'test_kappa' not in t and 'cov_kappa' not in t: continue
        k = (host, t['run'], t['dataset'], t['cand_id'], float(t['fidelity']), t.get('fitness_sig') or 'C100_kw1')
        cur = tests.setdefault(k, {})
        for f in ('test_kappa', 'test_f1', 'cov_kappa', 'cov_f1', 'val_kappa', 'params', 'flops', 'stratum', 'kind', 'op'):
            if t.get(f) is not None: cur[f] = t[f]
        cur['_src'] = f"{host}:{path}"
add_tests('ga_search/modehb/reports/test_svm_C100.jsonl', 'jaci')
add_tests(str(ATH / 'reports/test_svm_C100.jsonl'), 'athena')

used = set()
for row in rows:
    row.update(test_kappa_orig=None, test_f1_orig=None, test_kappa_cov=None, test_src='')
    if row['status'] != 'ok': continue
    for host in ('jaci', 'athena'):
        k = (host, row['run'], row['dataset'], row['cand_id'], float(row['fidelity']), row['fitness_sig_short'])
        t = tests.get(k)
        if t and t.get('val_kappa') is not None and row['val_kappa'] is not None and abs(t['val_kappa'] - row['val_kappa']) < 1e-6:
            if not row['test_src']:
                row.update(test_kappa_orig=t.get('test_kappa'), test_f1_orig=t.get('test_f1'), test_kappa_cov=t.get('cov_kappa'), test_src=t['_src'])
            elif row['test_kappa_cov'] is None and t.get('cov_kappa') is not None:
                row['test_kappa_cov'] = t.get('cov_kappa')
            used.add(k)

# athena-only tested rows (search_state not in the bundle): keep them, marked
ath_shapes = {}
for line in open(ATH / 'shapes_athena.jsonl'):
    r = json.loads(line); ath_shapes[r['key']] = r
for k, t in tests.items():
    if k in used or k[0] != 'athena': continue
    host, run, ds, cand, fid, fs = k
    key = f"{t.get('kind','hybrid')}|{ds}|{fs}|{cand}|{fid:g}"
    sh = ath_shapes.get(key, {})
    blocks = sh.get('blocos') or []
    rows.append(dict(host='athena', run=run, kind=kind_of(run), dataset=ds,
        fitness_sig=('C0.1_kw0.75_cov_head_standard_mt4000' if fs.startswith('C0.1') else 'C100_kw1_cov_head_standard_mt4000'),
        fitness_sig_short=fs, cand_id=cand, fidelity=fid, status='ok(athena_test_only)', stratum=t.get('stratum'),
        op_type=(t.get('op') or {}).get('type'), params=t.get('params'), flops=t.get('flops'),
        depth=len(blocks) if blocks else None, val_kappa=t.get('val_kappa'),
        test_kappa_orig=t.get('test_kappa'), test_f1_orig=t.get('test_f1'), test_kappa_cov=t.get('cov_kappa'),
        test_src=t['_src'], checkpoint=sh.get('checkpoint', ''), source='athena:reports/test_svm_C100.jsonl'))

cols = []
for r in rows:
    for c in r:
        if c not in cols: cols.append(c)
with open(OUT / 'A1a_data.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]
with open(OUT / 'A1a_layers.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(layer_rows[0])); w.writeheader(); [w.writerow(r) for r in layer_rows]
print(len(rows), 'rows;', sum(1 for r in rows if r.get('test_kappa_orig') is not None), 'with test;', len(layer_rows), 'layer rows')
print('unused test keys:', len([k for k in tests if k not in used]), 'jaci unused:', [k for k in tests if k not in used and k[0]=='jaci'][:12])
