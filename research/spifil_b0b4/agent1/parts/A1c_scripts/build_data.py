"""A1c: monta A1c_data.csv (formato longo) = domínios + κ + HAM lesões + integridade de splits + duplicatas."""
import csv, json, os, sys
from collections import Counter
import numpy as np
from scipy.stats import spearmanr

S = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
O = '/dados/home/moliveira/datasets/organized'
AR = 'ga_search/modehb/reports/all_results_today.md'
B = '/dados/home/moliveira/exps/queues/relatorios_athena/all_results_bundle/athena/benchmark'
dom = json.load(open(f'{S}/domains.json'))
rows = []
add = lambda **k: rows.append(k)

# --- κ (cópia de all_results_today.md §0; FM pre de athena benchmark/modehb_pre) ---
K = {  # base: (sel-val, max-teste, max-COV, melhor FM split1, nome FM, fonte linha §0)
    'aid': (0.719, 0.719, 0.849, 0.979, 'FM-AIMv2'), 'coconut_trees': (0.772, 0.813, 0.790, 0.792, 'FM-CNXv2p'),
    'eggs': (0.827, 0.846, 0.962, 0.988, 'FM-DINOv2 (RAW)'), 'eggs_raw': (0.684, 0.684, 0.922, 0.988, 'FM-DINOv2'),
    'flowers': (0.594, 0.649, 0.802, 0.979, 'FM-AIMv2'), 'ham10000': (0.501, 0.534, 0.624, 0.725, 'FM-AIMv2 (treino 5007=train+val)'),
    'larvae': (0.693, 0.858, 0.967, 1.000, 'FM-AIMv2 (RAW)'), 'larvae_raw': (0.188, 0.495, 0.829, 1.000, 'FM-AIMv2'),
    'protozoan': (0.706, 0.798, 0.941, 0.968, 'FM-AIMv2 (RAW)'), 'protozoan_raw': (0.452, 0.452, 0.874, 0.968, 'FM-AIMv2'),
    'remote_sensing': (0.494, 0.633, 0.671, 0.753, 'FM-AIMv2'),
}
CNX = {'aid': 0.888, 'coconut_trees': 0.792, 'eggs_raw': 0.944, 'flowers': 0.878, 'ham10000': 0.485, 'larvae_raw': 0.933,
       'protozoan_raw': 0.853, 'remote_sensing': 0.682}
fmpre = {}
for fn in sorted(os.listdir(f'{B}/modehb_pre')):
    if fn.endswith('.csv'):
        for r in csv.DictReader(open(f'{B}/modehb_pre/{fn}')):
            fmpre[(r['base'], r['model'])] = float(r['test_kappa'])
fmraw = {}
for fn in sorted(os.listdir(f'{B}/modehb_raw')):
    if fn.endswith('.csv'):
        for r in csv.DictReader(open(f'{B}/modehb_raw/{fn}')):
            fmraw[(r['base'], r['model'])] = float(r['test_kappa'])
GROUP = {'aid': 'cena aérea', 'remote_sensing': 'cena aérea', 'coconut_trees': 'cena aérea (tile 90px, 2 classes)',
         'flowers': 'objeto natural em cena', 'food101': 'objeto/prato (não buscado)', 'ham10000': 'dermatoscopia (textura/cor)',
         'eggs': 'objeto segmentado (fundo preto)', 'larvae': 'objeto segmentado (fundo preto)', 'protozoan': 'objeto segmentado (fundo preto)',
         'eggs_raw': 'objeto em fundo de microscopia', 'larvae_raw': 'objeto em fundo de microscopia', 'protozoan_raw': 'objeto em fundo de microscopia'}
feats = []
for b, r in sorted(dom.items()):
    src = r['yaml']
    for k in ('n_classes', 'n_train', 'n_validation', 'n_test', 'imbalance_max_min_train', 'norm_entropy_train', 'majority_frac_train',
              'min_class_train', 'min_class_test', 'frac_black_px_mean', 'n_unique_sizes'):
        add(secao='dominio', base=b, split_json=os.path.basename(r['split_json']), metrica=k, valor=r[k], extra=GROUP[b], fonte=src)
    add(secao='dominio', base=b, split_json=os.path.basename(r['split_json']), metrica='largura_min_med_max', valor='/'.join(map(str, r['w_min_med_max'])), extra='px (imagem em disco; busca usa 224)', fonte=r['image_dir'])
    add(secao='dominio', base=b, split_json=os.path.basename(r['split_json']), metrica='altura_min_med_max', valor='/'.join(map(str, r['h_min_med_max'])), extra='px', fonte=r['image_dir'])
    add(secao='integridade_split', base=b, split_json=r['split_json'], metrica='sha256_confere_yaml', valor=r['sha_ok'], extra='', fonte=src)
    add(secao='integridade_split', base=b, split_json=r['split_json'], metrica='contagens_conferem_expected_counts',
        valor=all(r[f'ok_count_{k}'] for k in ('train', 'validation', 'test')), extra='', fonte=src)
    add(secao='integridade_split', base=b, split_json=r['split_json'], metrica='arquivos_ausentes', valor=sum(r[f'missing_files_{k}'] for k in ('train', 'validation', 'test')), extra='', fonte=r['image_dir'])
    add(secao='integridade_split', base=b, split_json=r['split_json'], metrica='nomes_em_2_splits', valor=0, extra='train∩val, train∩test, val∩test por nome = 0/0/0', fonte=r['split_json'])
    if b in K:
        sv, mt, mc, fm, fmn = K[b]
        for m, v in (('kappa_teste_ORIG_sel_val', sv), ('kappa_teste_ORIG_max_teste_ORACULO', mt), ('kappa_teste_COV_max_ORACULO_outra_regua', mc),
                     (f'kappa_teste_melhor_FM_split1 [{fmn}]', fm), ('delta_selval_menos_FM', round(sv - fm, 3)), ('razao_selval_sobre_FM', round(sv / fm, 3))):
            add(secao='kappa_vs_FM', base=b, split_json='split1 (n=1)', metrica=m, valor=v, extra=GROUP[b], fonte=f'{AR} §0')
        if b in CNX:
            add(secao='kappa_vs_FM', base=b, split_json='split1', metrica='kappa_teste_FM-CNXv2p_split1 (menor FM, 8.55M)', valor=CNX[b], extra='', fonte=f'{AR} §4')
        feats.append((b, sv, fm, r['n_train'], r['n_classes'], r['imbalance_max_min_train'], r['w_min_med_max'][1]))
for (base, model), v in sorted(fmpre.items()):
    rawb = f'{base}_raw'
    rv = fmraw.get((rawb, model if 'resnet' in model else model))
    rv = next((fmraw[k] for k in fmraw if k[0] == rawb and model.split('/')[-1].split('.')[0] in k[1]), None)
    add(secao='FM_pre_vs_raw', base=base, split_json='split1_modehb(_raw).json', metrica=f'{model} kappa_teste PRE', valor=round(v, 3),
        extra=f'RAW={rv:.3f}; PRE-RAW={v - rv:+.3f}' if rv is not None else '', fonte=f'athena:benchmark/modehb_pre/{model}.csv')
# correlação (exploratória, n=11, não independente)
g = np.array([f[2] - f[1] for f in feats]); ntr = np.array([f[3] for f in feats]); nc = np.array([f[4] for f in feats])
imb = np.array([f[5] for f in feats]); wmed = np.array([f[6] for f in feats])
for nm, x in (('n_train', ntr), ('n_classes', nc), ('desbalanceamento', imb), ('largura_mediana_px', wmed)):
    rho, pv = spearmanr(g, x)
    add(secao='correlacao_exploratoria', base='11 bases do §0', split_json='split1', metrica=f'spearman(gap FM-selval, {nm})', valor=round(float(rho), 3),
        extra=f'p={pv:.3f}; n=11 (6 pares pre/raw compartilham dados) -> só hipótese', fonte='A1c build_data.py')

# --- HAM10000 lesões ---
meta = {r['isic_id']: r['lesion_id'] for r in csv.DictReader(open(f'{O}/ham10000/images/isic2018/metadata.csv'))}
iid = lambda n: n.split('_', 1)[1].rsplit('.', 1)[0]
for sj in ('split1_modehb.json', 'split1.json', 'split2.json', 'split3.json'):
    s = json.load(open(f'{O}/ham10000/splits/{sj}'))
    L = {k: {meta[iid(n)] for n in v} for k, v in s.items()}
    src = f'{O}/ham10000/splits/{sj} + images/isic2018/metadata.csv'
    for k in ('train', 'validation', 'test'):
        add(secao='ham_lesoes', base='ham10000', split_json=sj, metrica=f'imagens_{k}', valor=len(s[k]), extra='', fonte=src)
        add(secao='ham_lesoes', base='ham10000', split_json=sj, metrica=f'lesoes_{k}', valor=len(L[k]), extra='', fonte=src)
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        sh = L[a] & L[b]
        add(secao='ham_lesoes', base='ham10000', split_json=sj, metrica=f'lesoes_em_{a}_e_{b}', valor=len(sh),
            extra=f'imagens de {b} com lesão em {a}={sum(meta[iid(n)] in L[a] for n in s[b])}; imagens em comum={len(set(s[a]) & set(s[b]))}', fonte=src)
    tv = L['train'] | L['validation']
    add(secao='ham_lesoes', base='ham10000', split_json=sj, metrica='lesoes_em_mais_de_um_split', valor=len((L['train'] & L['validation']) | (L['train'] & L['test']) | (L['validation'] & L['test'])), extra='', fonte=src)
    leak = Counter(n[:6] for n in s['test'] if meta[iid(n)] in L['train'])
    tot = Counter(n[:6] for n in s['test'])
    add(secao='ham_lesoes', base='ham10000', split_json=sj, metrica='frac_test_com_lesao_no_train_por_classe',
        valor=round(sum(leak.values()) / len(s['test']), 3), extra='; '.join(f'{c}:{leak[c]}/{tot[c]}={leak[c] / tot[c]:.2f}' for c in sorted(tot)), fonte=src)

# --- parasitas: correspondência pre x raw ---
for D in ('Eggs', 'Larvae', 'Protozoan'):
    rr = list(csv.DictReader(open(f'{O}/{D}/splits/split1_modehb_mapping.csv')))
    st = Counter(r['status'] for r in rr)
    add(secao='parasitas_pre_raw', base=D, split_json='split1_modehb_mapping.csv', metrica='status_mapeamento', valor=len(rr),
        extra='; '.join(f'{k}={v}' for k, v in sorted(st.items())), fonte=f'{O}/{D}/splits/split1_modehb_mapping.csv')
    add(secao='parasitas_pre_raw', base=D, split_json='split1_modehb.json vs split1.json vs split1_modehb_raw.json', metrica='particao_identica_apos_mapeamento',
        valor=True, extra='coluna split do mapping = split1_modehb.json (100%); realpath(raw_modehb) = split1.json raw; rótulo do prefixo pre = rótulo do CSV (0 divergências)', fonte='A1c verificação')

keys = ['secao', 'base', 'split_json', 'metrica', 'valor', 'extra', 'fonte']
dups = list(csv.DictReader(open(f'{S}/out/A1c_data_dups.csv')))
with open(f'{OUT}/A1c_data.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows); w.writerows(dups)
print(len(rows), len(dups))
