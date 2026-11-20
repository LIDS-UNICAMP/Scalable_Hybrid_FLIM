"""A1c: busca de "mesmo objeto" no pre (fundo preto) com o objeto recortado pela caixa do não-preto:
miniatura cinza 32x32 do recorte + tamanho da caixa (px). Par candidato: |Δw|,|Δh| <= 3 px,
corr(recorte 32x32) >= CMIN e dHash(recorte) <= DMAX. Uso: crop_pairs.py <imgdir> <out.npz> <CMIN> <DMAX>"""
import os, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits


def feat(p):
    im = np.asarray(Image.open(p).convert('RGB'))
    m = im.max(2) > 8
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return 0, 0, np.zeros((32, 32), np.uint8), (0, 0, 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    c = Image.fromarray(im[y0:y1, x0:x1])
    g = np.asarray(c.convert('L').resize((32, 32), Image.BOX), np.uint8)
    col = im[m].mean(0)
    return x1 - x0, y1 - y0, g, tuple(col)


def main(d, out, cmin, dmax):
    names = sorted(os.listdir(d))
    with Pool(20) as pool:
        R = pool.map(feat, [os.path.join(d, n) for n in names], chunksize=64)
    W = np.array([r[0] for r in R]); H = np.array([r[1] for r in R]); G = np.stack([r[2] for r in R]).astype(np.float32)
    COL = np.array([r[3] for r in R], np.float32)
    z = G.reshape(len(G), -1); z = z - z.mean(1, keepdims=True); z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-6
    dbits = (G[:, :, 1:] > G[:, :, :-1]).reshape(len(G), -1).astype(np.float32); nd = dbits.sum(1)
    out_i, out_j, out_c, out_d = [], [], [], []
    N = len(G)
    with threadpool_limits(8):
        for s in range(0, N, 1024):
            e = min(N, s + 1024)
            C = z[s:e] @ z.T
            ok = (C >= float(cmin)) & (np.abs(W[s:e, None] - W[None]) <= 3) & (np.abs(H[s:e, None] - H[None]) <= 3)
            ii, jj = np.nonzero(ok); ii += s
            k = ii < jj; ii, jj = ii[k], jj[k]
            dh = nd[ii] + nd[jj] - 2 * np.einsum('ij,ij->i', dbits[ii], dbits[jj])
            k = dh <= int(dmax)
            out_i.append(ii[k]); out_j.append(jj[k]); out_c.append(C[ii[k] - s, jj[k]]); out_d.append(dh[k])
    i, j = np.concatenate(out_i), np.concatenate(out_j)
    np.savez_compressed(out, names=np.array(names), i=i, j=j, corr=np.concatenate(out_c), dham=np.concatenate(out_d),
                        W=W, H=H, col=COL, coldiff=np.linalg.norm(COL[i] - COL[j], axis=1))
    print(out, N, 'pairs', len(i), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:5])
