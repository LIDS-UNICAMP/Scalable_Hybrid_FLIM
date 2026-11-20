"""A1c: consolida pares duplicados × splits -> CSVs rastreáveis.
Níveis (sobre a miniatura cinza 32x32 da imagem inteira):
  EXATA: md5 dos bytes igual
  T1: corr>=0.98 e dHash<=50 (de 992 bits)    -- quase-idêntica
  T2: corr>=0.95 e dHash<=100                  -- quase-duplicata provável (auditar)
Nas bases pre (fundo preto) os níveis T1/T2 têm muitos falsos positivos (objetos alinhados parecidos);
lá vale o veredito visual (AUDIT) quando houver."""
import csv, json, os, sys
from collections import Counter, defaultdict
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
O = '/dados/home/moliveira/datasets/organized'
OUT = sys.argv[1]
TAGS = {  # tag -> (Dataset dir, split jsons, domain)
    'aid': ('aid', ['split1.json', 'split2.json', 'split3.json'], 'natural'),
    'coconut_trees': ('coconut_trees', ['split1.json', 'split2.json', 'split3.json'], 'natural'),
    'flowers': ('flowers', ['split1.json', 'split2.json', 'split3.json'], 'natural'),
    'remote_sensing': ('remote_sensing', ['split1.json', 'split2.json', 'split3.json'], 'natural'),
    'food101': ('food101', ['split1.json', 'split2.json', 'split3.json'], 'natural'),
    'ham10000': ('ham10000', ['split1_modehb.json', 'split1.json', 'split2.json', 'split3.json'], 'natural'),
    'Eggs_raw': ('Eggs', ['split1.json', 'split2.json', 'split3.json'], 'raw'),
    'Larvae_raw': ('Larvae', ['split1.json', 'split2.json', 'split3.json'], 'raw'),
    'Protozoan_raw': ('Protozoan', ['split1.json', 'split2.json', 'split3.json'], 'raw'),
    'Eggs_pre': ('Eggs', ['split1_modehb.json', 'split2.json', 'split3.json'], 'pre'),
    'Larvae_pre': ('Larvae', ['split1_modehb.json', 'split2.json', 'split3.json'], 'pre'),
    'Protozoan_pre': ('Protozoan', ['split1_modehb.json', 'split2.json', 'split3.json'], 'pre'),
}
AUDIT = json.load(open(f'{S}/audit.json'))  # {tag: {"a|b": "mesmo"/"diferente"}}
ham_meta = {r['isic_id']: r['lesion_id'] for r in csv.DictReader(open(f'{O}/ham10000/images/isic2018/metadata.csv'))}
iid = lambda n: n.split('_', 1)[1].rsplit('.', 1)[0]

data, prs = [], []
for tag, (D, sjs, dom) in TAGS.items():
    if not os.path.exists(f'{S}/pairs/{tag}.npz'):
        print('skip', tag); continue
    f = dict(np.load(f'{S}/feat/{tag}.npz')); p = dict(np.load(f'{S}/pairs/{tag}.npz'))
    names = f['names']
    mp = {}
    lab = {}
    if dom in ('pre', 'raw'):
        for r in csv.DictReader(open(f'{O}/{D}/splits/split1_modehb_mapping.csv')):
            if r['new_name_pre']:
                mp[r['new_name_pre']] = r['old_name_raw']
            lab[r['old_name_raw']] = int(r['label'])
    def raw_of(n):
        return mp.get(n, n) if dom == 'pre' else n
    def label(n):
        if dom == 'raw':
            return lab.get(n, -1)
        return int(n[:6])
    tier = np.full(len(p['i']), '', dtype=object)
    c, dh = p['corr'], p['dham']
    tier[(c >= 0.95) & (dh <= 100)] = 'T2'
    tier[(c >= 0.98) & (dh <= 50)] = 'T1'
    tier[p['same_md5']] = 'EXATA'
    aud = AUDIT.get(tag, {})
    for sj in sjs:
        s = json.load(open(f'{O}/{D}/splits/{sj}'))
        sp = {}
        for k, v in s.items():
            for n in v:
                sp.setdefault(n, k)  # ham split1: val==train -> fica 'train'
        def split_of(n):
            if dom == 'pre' and sj != 'split1_modehb.json':
                return sp.get(raw_of(n))
            return sp.get(n)
        cnt = defaultdict(lambda: [0, set(), 0])
        for k in range(len(p['i'])):
            a, b = names[p['i'][k]], names[p['j'][k]]
            sa, sb = split_of(a), split_of(b)
            if sa is None or sb is None or sa == sb:
                continue
            key = '|'.join(sorted([a, b]))
            verdict = aud.get(key, '')
            tiers = [tier[k]] if tier[k] else []
            if verdict == 'mesmo':
                tiers.append('AUDIT_mesmo')
            if not tiers:
                continue
            pair = tuple(sorted([sa, sb], key=['train', 'validation', 'test'].index))
            for t in tiers:
                cc = cnt[(t, pair)]
                cc[0] += 1
                later = b if split_of(b) == pair[1] else a
                cc[1].add(later)
                cc[2] += int(label(a) != label(b))
            if sj in ('split1_modehb.json',) or (dom != 'pre' and sj == 'split1.json') or (tag == 'ham10000' and sj == 'split1_modehb.json'):
                row = dict(tag=tag, split_json=sj, img_a=a, split_a=sa, label_a=label(a), img_b=b, split_b=sb, label_b=label(b),
                           raw_a=raw_of(a) if dom == 'pre' else '', raw_b=raw_of(b) if dom == 'pre' else '',
                           md5_igual=bool(p['same_md5'][k]), corr32=round(float(c[k]), 4), dhash=int(dh[k]), ahash=int(p['aham'][k]),
                           transf_diedral=int(p['t'][k]), nivel=tier[k], veredito_visual=verdict)
                if tag == 'ham10000':
                    la, lb = ham_meta[iid(a)], ham_meta[iid(b)]
                    row['nota'] = f'lesion {la} / {lb}' + (' (mesma lesão)' if la == lb else ' (lesões DIFERENTES)')
                prs.append(row)
        for (t, pair), (npairs, imgs, diffc) in sorted(cnt.items()):
            data.append(dict(secao='duplicatas_entre_splits', base=tag, split_json=sj, metrica=f'{t}:{pair[0]}-{pair[1]}',
                             valor=npairs, extra=f'imagens_distintas_em_{pair[1]}={len(imgs)}; pares_classe_diferente={diffc}',
                             fonte=f'scratchpad/A1c/pairs/{tag}.npz'))
        if not cnt:
            data.append(dict(secao='duplicatas_entre_splits', base=tag, split_json=sj, metrica='nenhum_par_EXATA/T1/T2/AUDIT', valor=0, extra='', fonte=f'scratchpad/A1c/pairs/{tag}.npz'))
    # pares dentro do dataset (independente de split)
    for t in ('EXATA', 'T1', 'T2'):
        data.append(dict(secao='duplicatas_no_dataset', base=tag, split_json='(todas as imagens)', metrica=f'pares_{t}',
                         valor=int((tier == t).sum()), extra=f'N_imagens={len(names)}', fonte=f'scratchpad/A1c/pairs/{tag}.npz'))
    if aud:
        v = Counter(aud.values())
        data.append(dict(secao='auditoria_visual', base=tag, split_json='split1_modehb.json', metrica='pares_auditados',
                         valor=len(aud), extra=f'mesmo={v.get("mesmo",0)}; diferente={v.get("diferente",0)}', fonte='montagens scratchpad/A1c/mont'))

keys = ['secao', 'base', 'split_json', 'metrica', 'valor', 'extra', 'fonte']
with open(f'{OUT}/A1c_data_dups.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(data)
pk = ['tag', 'split_json', 'img_a', 'split_a', 'label_a', 'img_b', 'split_b', 'label_b', 'raw_a', 'raw_b', 'md5_igual', 'corr32', 'dhash',
      'ahash', 'transf_diedral', 'nivel', 'veredito_visual', 'nota']
with open(f'{OUT}/A1c_dup_pairs.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, pk, restval=''); w.writeheader(); w.writerows(prs)
print(len(data), len(prs))
