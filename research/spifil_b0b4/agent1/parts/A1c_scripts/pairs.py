"""A1c: busca de pares (quase-)duplicados dentro de uma base (todos × todos), por:
 - md5 dos bytes (exata);
 - dHash 32x31 (992 bits) e aHash 32x32 (1024 bits) da miniatura cinza 32x32;
 - correlação de Pearson da miniatura 32x32 (verificação);
 - opcional: as 8 transformações diedrais (rot90 x flip) do lado "consulta".
Grava pares com corr >= CORR_MIN e dHash <= DH_MAX (frouxo; os níveis são aplicados depois).
Uso: pairs.py <feat.npz> <out_pairs.npz> <dihedral 0/1>"""
import sys
import numpy as np
from threadpoolctl import threadpool_limits

import os
CORR_MIN, DH_MAX = float(os.environ.get("CORR_MIN", 0.90)), int(os.environ.get("DH_MAX", 150))


def bits(th):
    th = th.astype(np.float32)
    a = (th > th.mean(axis=(1, 2), keepdims=True)).reshape(len(th), -1).astype(np.float32)
    d = (th[:, :, 1:] > th[:, :, :-1]).reshape(len(th), -1).astype(np.float32)
    z = th.reshape(len(th), -1) - th.reshape(len(th), -1).mean(1, keepdims=True)
    z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-6)
    return a, d, z.astype(np.float32)


def dihedral(th, t):
    x = np.rot90(th, k=t % 4, axes=(1, 2))
    return x[:, :, ::-1] if t >= 4 else x


def main(feat, out, use_dih):
    f = np.load(feat)
    th = f['thumb']
    N = len(th)
    A, Dd, Z = bits(th)
    na, nd = A.sum(1), Dd.sum(1)
    flat = th.reshape(N, -1).std(1) < 1.0  # miniaturas constantes (sem conteúdo) -> marcadas
    rows = []
    chunk = 512 if N > 50000 else 1024
    ts = range(8) if int(use_dih) else [0]
    with threadpool_limits(8):
        for t in ts:
            At, Dt, Zt = bits(dihedral(th, t)) if t else (A, Dd, Z)
            nat, ndt = At.sum(1), Dt.sum(1)
            for s in range(0, N, chunk):
                e = min(N, s + chunk)
                C = Zt[s:e] @ Z.T
                Dh = ndt[s:e, None] + nd[None, :] - 2 * (Dt[s:e] @ Dd.T)
                ii, jj = np.nonzero((C >= CORR_MIN) & (Dh <= DH_MAX))
                ii = ii + s
                keep = ii < jj if t == 0 else ii != jj
                ii, jj = ii[keep], jj[keep]
                if len(ii):
                    ah = nat[ii] + na[jj] - 2 * np.einsum('ij,ij->i', At[ii], A[jj])
                    rows.append(np.stack([ii, jj, np.full_like(ii, t), C[ii - s, jj], Dh[ii - s, jj], ah], 1))
    P = np.concatenate(rows) if rows else np.zeros((0, 6))
    # para diedrais, mantém só o melhor t por par não-ordenado e descarta os já achados com t=0
    if len(P):
        lo, hi = np.minimum(P[:, 0], P[:, 1]), np.maximum(P[:, 0], P[:, 1])
        P[:, 0], P[:, 1] = lo, hi
        order = np.lexsort((-P[:, 3], P[:, 1], P[:, 0]))
        P = P[order]
        key = P[:, 0] * N + P[:, 1]
        _, first = np.unique(key, return_index=True)
        P = P[first]
    md5 = f['md5']
    same_md5 = md5[P[:, 0].astype(int)] == md5[P[:, 1].astype(int)] if len(P) else np.zeros(0, bool)
    # pares md5 idênticos que não passaram no filtro (ex.: miniatura constante)
    from collections import defaultdict
    g = defaultdict(list)
    for i, m in enumerate(md5):
        g[m].append(i)
    extra = []
    have = set(map(tuple, P[:, :2].astype(int).tolist())) if len(P) else set()
    for v in g.values():
        for a in range(len(v)):
            for b in range(a + 1, len(v)):
                if (v[a], v[b]) not in have:
                    extra.append([v[a], v[b], 0, 1.0, 0, 0])
    if extra:
        P = np.concatenate([P, np.array(extra, dtype=P.dtype)])
        same_md5 = np.concatenate([same_md5, np.ones(len(extra), bool)])
    np.savez_compressed(out, i=P[:, 0].astype(np.int64), j=P[:, 1].astype(np.int64), t=P[:, 2].astype(np.int8),
                        corr=P[:, 3].astype(np.float32), dham=P[:, 4].astype(np.int16), aham=P[:, 5].astype(np.int16),
                        same_md5=same_md5, flat=flat)
    print(feat, N, 'pairs', len(P), 'md5dup', int(same_md5.sum()), 'flat', int(flat.sum()), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:4])
