"""Measured: distinct SPiFiL seed positions per image after rescale to each layer grid (same code path as
flim.spifil.build_layer0: Cropped(224, mask) -> SLIC(100) -> Medoids -> rescale). CPU only, read-only."""
import json, glob, sys, os
from pathlib import Path
import numpy as np, torch, yaml
os.chdir('/dados/home/moliveira/Scalable_Hybrid_FLIM'); sys.path.insert(0, '.')
from spifil import preparation
from spifil.color import LabNorm
from spifil.superpixels.slic import SLIC
from spifil.superpixels.centers import Medoids
from spifil.data import Sample
from flim.spifil import Cropped, rescale

DS = ['remote_sensing', 'ham10000', 'flowers', 'aid', 'eggs', 'larvae', 'protozoan', 'larvae_raw', 'protozoan_raw']
STATES = {'remote_sensing': 'modehb_lemonade', 'ham10000': 'modehb_hybrid', 'flowers': 'modehb_hybrid', 'aid': 'modehb_hybrid',
          'eggs': 'modehb_lemonade', 'larvae': 'modehb_hybrid', 'protozoan': 'modehb_hybrid', 'larvae_raw': 'modehb_hybrid', 'protozoan_raw': 'modehb_hybrid'}
GRIDS = [224, 112, 56, 28, 14, 7, 4]
out = []
for ds in DS:
    cfg = yaml.safe_load(open(f'ga_search/config/modehb/datasets/{ds}.yaml'))
    root = cfg['root']; idir = cfg['images_dir'].format(root=root); mdir = (cfg.get('masks_dir') or '') and cfg['masks_dir'].format(root=root)
    st = json.load(open(f'artifacts/{STATES[ds]}/{ds}/state/seed42/search_state.json'))
    ids = None
    for j in st['jobs']:
        r = j.get('result') or {}
        if r.get('spifil_sample_ids') and (r.get('spifil_images_per_class') or 0) >= 4:
            ids = r['spifil_sample_ids']; ipc = r['spifil_images_per_class']; break
    # first 3 per class (nested subsets: first k per class are shared by all ipc>=k)
    by = {}
    for n in ids: by.setdefault(int(n.split('_')[0]), []).append(n)
    names = [n for c in sorted(by) for n in by[c][:3]]
    samples = []
    for n in names:
        ip = next(iter(glob.glob(f'{idir}/{n}.*')), None)
        mp = next(iter(glob.glob(f'{mdir}/{n}.*')), None) if mdir else None
        if ip: samples.append(Sample(name=n, image_path=Path(ip), mask_path=Path(mp) if mp else None, label=int(n.split('_')[0])))
    data = Cropped(samples, 224, use_mask=True)
    for im, s in zip(preparation.prepare(data, superpixels=SLIC(), seed_extractor=Medoids(), n_superpixels=100, color=LabNorm()), samples):
        mask = data.load_mask(samples.index(s))
        cov = float((mask > 0).mean()) if mask is not None else 1.0
        row = dict(dataset=ds, name=s.name, mask=mask is not None, mask_coverage=cov, n_seeds=len(im.seeds.coords))
        for g in GRIDS:
            sd = rescale(im.seeds, (g, g))
            row[f'distinct_{g}'] = int(torch.unique(sd.coords, dim=0).shape[0])
        out.append(row)
    print(ds, len(samples), flush=True)
import pandas as pd
df = pd.DataFrame(out)
df.to_csv('research/spifil_b0b4/agent1/parts/A1a_seed_distinct.csv', index=False)
print(df.groupby('dataset').mean(numeric_only=True).round(2).to_string())
