"""A1c: verificação "mesmo objeto" para pares de parasitas candidatos no pre, usando as imagens RAW
(fundo original, resolução nativa): template matching (TM_CCOEFF_NORMED, cinza) de um recorte central
(60% do menor lado) de cada imagem dentro da outra, nas 8 transformações diedrais do recorte; score = máximo.
Uso: tmatch.py <Dataset> <feat_pre.npz> <pairs_pre.npz> <out.npz> <corr_min> <dham_max>"""
import csv, os, sys
from multiprocessing import Pool
import cv2
import numpy as np

O = '/dados/home/moliveira/datasets/organized'


def score(args):
    pa, pb = args
    A = cv2.imread(pa, cv2.IMREAD_GRAYSCALE)
    B = cv2.imread(pb, cv2.IMREAD_GRAYSCALE)
    m = max(A.shape + B.shape)
    if m > 256:  # mesmo fator nas duas (mesma magnificação do microscópio)
        fct = 256.0 / m
        A = cv2.resize(A, None, fx=fct, fy=fct, interpolation=cv2.INTER_AREA)
        B = cv2.resize(B, None, fx=fct, fy=fct, interpolation=cv2.INTER_AREA)
    best = -1.0
    for X, Y in ((A, B), (B, A)):
        s = int(0.6 * min(X.shape + Y.shape))
        if s < 16:
            continue
        cy, cx = X.shape[0] // 2, X.shape[1] // 2
        T0 = X[cy - s // 2: cy - s // 2 + s, cx - s // 2: cx - s // 2 + s]
        for t in range(8):
            T = np.rot90(T0, t % 4)
            T = np.ascontiguousarray(T[:, ::-1] if t >= 4 else T)
            if T.shape[0] > Y.shape[0] or T.shape[1] > Y.shape[1]:
                continue
            r = cv2.matchTemplate(Y, T, cv2.TM_CCOEFF_NORMED)
            best = max(best, float(r.max()))
    return best


def main(D, feat, pairs, out, cmin, dmax):
    cv2.setNumThreads(1)
    f = np.load(feat)
    p = np.load(pairs)
    pre2raw = {r['new_name_pre']: r['old_name_raw'] for r in csv.DictReader(open(f'{O}/{D}/splits/split1_modehb_mapping.csv')) if r['new_name_pre']}
    sel = np.where((p['corr'] >= float(cmin)) & (p['dham'] <= int(dmax)))[0]
    names = f['names']
    jobs = [(f'{O}/{D}/raw/orig/{pre2raw[names[p["i"][k]]]}', f'{O}/{D}/raw/orig/{pre2raw[names[p["j"][k]]]}') for k in sel]
    with Pool(24) as pool:
        sc = pool.map(score, jobs, chunksize=16)
    np.savez_compressed(out, sel=sel, tm=np.array(sc, np.float32))
    print(D, len(sel), 'tm>=0.9:', int((np.array(sc) >= 0.9).sum()), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:7])
