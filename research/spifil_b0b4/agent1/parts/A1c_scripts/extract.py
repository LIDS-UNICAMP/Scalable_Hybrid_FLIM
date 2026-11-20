"""A1c: extrai, por diretório de imagens, md5 dos bytes, (w,h,mode), miniatura cinza 32x32 (área),
fração de pixels pretos (<8 em todos os canais) e cor média. Só leitura. Uso: extract.py <tag> <dir> <out.npz>"""
import hashlib, os, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image


def feat(path):
    b = open(path, 'rb').read()
    md5 = hashlib.md5(b).hexdigest()
    im = Image.open(path)
    w, h = im.size
    mode = im.mode
    if im.format == 'JPEG':
        im.draft('RGB', (128, 128))
    im = im.convert('RGB')
    small = np.asarray(im.resize((64, 64), Image.BILINEAR, reducing_gap=2.0)).astype(np.float32)
    g = im.convert('L').resize((32, 32), Image.BOX)
    thumb = np.asarray(g, dtype=np.uint8)
    black = float((small.max(axis=2) < 8).mean())
    mean_rgb = small.reshape(-1, 3).mean(0)
    return md5, w, h, mode, thumb, black, mean_rgb, len(b)


def main(tag, d, out):
    names = sorted(f for f in os.listdir(d) if not f.startswith('.'))
    paths = [os.path.join(d, n) for n in names]
    with Pool(int(os.environ.get('NPROC', 16))) as p:
        res = p.map(feat, paths, chunksize=64)
    np.savez_compressed(out, names=np.array(names), md5=np.array([r[0] for r in res]),
                        w=np.array([r[1] for r in res]), h=np.array([r[2] for r in res]),
                        mode=np.array([r[3] for r in res]), thumb=np.stack([r[4] for r in res]),
                        black=np.array([r[5] for r in res]), mean_rgb=np.stack([r[6] for r in res]),
                        nbytes=np.array([r[7] for r in res]),
                        realpaths=np.array([os.path.realpath(p) for p in paths]))
    print(tag, len(names), 'ok', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:4])
