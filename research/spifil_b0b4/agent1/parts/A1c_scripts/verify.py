"""A1c: verificação dos pares candidatos em resolução 64x64 RGB (mesma transformação diedral t do par).
Mede IoU das máscaras (pixel não-preto, max canal > 8; só faz sentido no pre) e correlação de Pearson RGB
(no pre: na união das máscaras; nas demais: imagem toda). Uso: verify.py <imgdir> <feat.npz> <pairs.npz> <out.npz>"""
import os, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image

R = 64


def load(p):
    im = Image.open(p)
    if im.format == 'JPEG':
        im.draft('RGB', (2 * R, 2 * R))
    return np.asarray(im.convert('RGB').resize((R, R), Image.BOX), dtype=np.uint8)


def dihedral1(x, t):
    x = np.rot90(x, k=t % 4, axes=(0, 1))
    return x[:, ::-1] if t >= 4 else x


def main(d, feat, pairs, out):
    f = np.load(feat)
    p = np.load(pairs)
    names = f['names']
    need = np.unique(np.concatenate([p['i'], p['j']])) if len(p['i']) else np.zeros(0, int)
    with Pool(16) as pool:
        ims = pool.map(load, [os.path.join(d, names[k]) for k in need], chunksize=64)
    pos = {k: n for n, k in enumerate(need)}
    iou = np.zeros(len(p['i']), np.float32)
    cor = np.zeros(len(p['i']), np.float32)
    for n, (a, b, t) in enumerate(zip(p['i'], p['j'], p['t'])):
        A = dihedral1(ims[pos[a]], int(t)).astype(np.float32)
        B = ims[pos[b]].astype(np.float32)
        ma, mb = A.max(2) > 8, B.max(2) > 8
        u = ma | mb
        iou[n] = (ma & mb).sum() / max(1, u.sum())
        if u.mean() > 0.98:  # imagem sem fundo preto: correlação na imagem toda
            u = np.ones_like(u)
        x, y = A[u].ravel(), B[u].ravel()
        x, y = x - x.mean(), y - y.mean()
        cor[n] = float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-6))
    np.savez_compressed(out, iou=iou, cor64=cor)
    print(pairs, len(cor), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:5])
