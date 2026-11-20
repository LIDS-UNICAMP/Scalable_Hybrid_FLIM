"""montage.py <imgdir> <out.png> name1,name2 name1,name2 ... -> pares lado a lado (128px)"""
import sys, os
from PIL import Image, ImageDraw
d, out, pairs = sys.argv[1], sys.argv[2], [p.split(',') for p in sys.argv[3:]]
S = 128
M = Image.new('RGB', (2 * S + 10, len(pairs) * (S + 14)), 'white')
dr = ImageDraw.Draw(M)
for r, (a, b) in enumerate(pairs):
    for c, n in enumerate((a, b)):
        im = Image.open(os.path.join(d, n)).convert('RGB').resize((S, S))
        M.paste(im, (c * (S + 10), r * (S + 14) + 12))
    dr.text((2, r * (S + 14)), f'{a} | {b}'[:60], fill='black')
M.save(out)
