"""A1c: caracterização dos domínios das 12 bases das buscas (configs ga_search/config/modehb/datasets)."""
import glob, hashlib, json, math, os, sys
from collections import Counter
import numpy as np
import yaml

REPO = '/dados/home/moliveira/Scalable_Hybrid_FLIM'
S = os.path.dirname(os.path.abspath(__file__))
FEAT = {'aid': 'aid', 'coconut_trees': 'coconut_trees', 'flowers': 'flowers', 'food101': 'food101',
        'ham10000': 'ham10000', 'remote_sensing': 'remote_sensing', 'eggs': 'Eggs_pre', 'eggs_raw': 'Eggs_rawm',
        'larvae': 'Larvae_pre', 'larvae_raw': 'Larvae_raw', 'protozoan': 'Protozoan_pre', 'protozoan_raw': 'Protozoan_rawm'}
out = {}
for y in sorted(glob.glob(f'{REPO}/ga_search/config/modehb/datasets/*.yaml')):
    c = yaml.safe_load(open(y))
    root = c['root']
    sj = c['split_json'].format(root=root, split=c['split'])
    sha = hashlib.sha256(open(sj, 'rb').read()).hexdigest()
    s = json.load(open(sj))
    f = np.load(f'{S}/feat/{FEAT[c["name"]]}.npz')
    idx = {n: i for i, n in enumerate(f['names'])}
    r = {'yaml': os.path.relpath(y, REPO), 'split_json': sj, 'sha_ok': sha == c['split_sha256'],
         'n_classes': c['n_classes'], 'image_dir': c['images_dir'].format(root=root)}
    allnames = []
    for k in ('train', 'validation', 'test'):
        v = s[k]
        cnt = Counter(int(n[:6]) for n in v)
        r[f'n_{k}'] = len(v)
        r[f'ok_count_{k}'] = len(v) == c['expected_counts'][k]
        r[f'classes_{k}'] = dict(sorted(cnt.items()))
        allnames += v
        miss = [n for n in v if n not in idx]
        r[f'missing_files_{k}'] = len(miss)
    tr = Counter(int(n[:6]) for n in s['train'])
    p = np.array(list(tr.values()), float) / sum(tr.values())
    r['imbalance_max_min_train'] = round(max(tr.values()) / min(tr.values()), 2)
    r['norm_entropy_train'] = round(float(-(p * np.log(p)).sum() / math.log(len(p))), 3) if len(p) > 1 else 1.0
    r['majority_frac_train'] = round(float(p.max()), 3)
    r['min_class_train'] = int(min(tr.values()))
    r['min_class_test'] = int(min(Counter(int(n[:6]) for n in s['test']).values()))
    ii = [idx[n] for n in allnames if n in idx]
    w, h = f['w'][ii], f['h'][ii]
    r['w_min_med_max'] = [int(w.min()), int(np.median(w)), int(w.max())]
    r['h_min_med_max'] = [int(h.min()), int(np.median(h)), int(h.max())]
    r['frac_black_px_mean'] = round(float(f['black'][ii].mean()), 3)
    r['thumb_gray_mean'] = round(float(f['thumb'][ii].mean()), 1)
    r['n_unique_sizes'] = int(len(set(zip(w.tolist(), h.tolist()))))
    out[c['name']] = r
json.dump(out, open(f'{S}/domains.json', 'w'), indent=1)
for k, r in out.items():
    print(k, {kk: vv for kk, vv in r.items() if not kk.startswith('classes_')})
