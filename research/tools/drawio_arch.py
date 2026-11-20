"""drawio_arch.py -- desenha em Draw.io a arquitetura REAL (forward 1x3x224x224 com hooks) de modelos modehb (encoder.pt).
Uma aba (diagram) por modelo, nome "<dataset> – <sigla>". Cores por mecanismo; espaçamento fixo em grade.
Uso: python drawio_arch.py out.drawio "aba|caminho/encoder.pt|subtítulo" ...
"""
import sys, html, torch, torch.nn as nn
sys.path.insert(0, '/dados/home/moliveira/Scalable_Hybrid_FLIM')
torch.set_grad_enabled(False)
COL = {  # padrão de draw_io/coconut_c00001_tiny.drawio (athena)
 'conv': ('#fff2cc', '#d6b656'), 'inception': ('#fff2cc', '#d6b656'), 'depthwise': ('#fff2cc', '#d6b656'),
 'pointwise': ('#fff2cc', '#d6b656'), 'skip': ('#fff2cc', '#d6b656'), 'head': ('#fff2cc', '#d6b656'),
 'bn': ('#dae8fc', '#6c8ebf'), 'whiten': ('#dae8fc', '#6c8ebf'), 'readout': ('#dae8fc', '#6c8ebf'),
 'act': ('#f8cecc', '#b85450'), 'pool': ('#eeeeee', '#999999'), 'input': ('#eeeeee', '#999999'), 'out': ('#b9e0a5', '#5c9e42')}
LEG = [('conv', 'Convolução (SPiFiL; cabeça 1×1 aleatória)'), ('bn', 'Normalização / whitening / covariância'), ('act', 'Ativação'), ('pool', 'Sem peso (entrada, concat, pooling)'), ('out', 'Saída / SVM')]
def kind(name, m):
    t = type(m).__name__
    if 'skip' in name: return 'skip'
    if name.startswith('head'): return 'head'
    if isinstance(m, nn.Conv2d):
        if 'branches' in name: return 'inception'
        if m.groups > 1: return 'depthwise'
        if 'pointwise' in name: return 'pointwise'
        return 'conv'
    if 'Decorrelated' in t or 'ZCA' in t or 'Whiten' in t: return 'whiten'
    if 'BatchNorm' in t: return 'bn'
    if 'Pool' in t: return 'pool'
    if t in ('ReLU', 'GELU', 'SiLU', 'SELU', 'ReLU6', 'LeakyReLU', 'PReLU', 'Tanh'): return 'act'
    return None
def label(m):
    t = type(m).__name__
    if isinstance(m, nn.Conv2d):
        return f"Conv{m.kernel_size[0]}×{m.kernel_size[1]} {m.in_channels}→{m.out_channels}" + (f" g{m.groups}" if m.groups > 1 else "")
    if 'Decorrelated' in t: return 'DBN'
    if 'ZCA' in t: return 'ZCA'
    if 'BatchNorm' in t: return 'BatchNorm'
    if 'Pool' in t and hasattr(m, 'kernel_size'):
        k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
        st = m.stride if isinstance(m.stride, int) else (m.stride[0] if m.stride else k)
        return f"{'MaxPool' if 'Max' in t else 'AvgPool'} {k}×{k} s{st}"
    if 'Pool' in t: return t
    return t
KIND = 'modehb'
def trace(path):
    d = {}
    if KIND == 'growth':
        from ga_search.backbones.checkpoint import load_backbone
        m, spec = load_backbone(path, 'cpu')
        enc, head = nn.Sequential(*list(m.blocks)), m.head
    elif KIND == 'hybrid':
        from eval.growth_stages import _encoder
        e_, _h, _ep = _encoder({'ckpt': str(path)}, 'cpu')
        enc = nn.Sequential(*[getattr(e_, f'conv{n}') for n in range(1, getattr(e_, 'n_layers', 3) + 1)])
        head = nn.Identity()
    else:
        d = torch.load(path, map_location='cpu', weights_only=False)
        enc, head = d['encoder'], d['head']
    model = nn.ModuleDict({'encoder': enc, 'head': head}); rows = []
    def mk(name):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            rows.append((name, mod, tuple(o.shape[1:]), sum(p.numel() for p in mod.parameters(recurse=False))))
        return h
    hs = [m.register_forward_hook(mk(n)) for n, m in model.named_modules() if n and not list(m.children())]
    model.eval(); f = enc(torch.zeros(1, 3, 224, 224)); head(f)
    for h in hs: h.remove()
    rows = [r for r in rows if kind(r[0], r[1])]
    tot = sum(p.numel() for p in model.parameters()); e = sum(p.numel() for p in enc.parameters())
    gg = dict(d.get('genome') or {}); gg['_fidelity'] = d.get('fidelity')
    return rows, tot, e, list(f.shape[1:]), gg
W, H, GX, GY, X0 = 220, 56, 40, 26, 40
INIT_STYLE = {'spifil': ('#2e7d32', 3, False, 'SPiFiL'), 'random': ('#c62828', 2, True, 'aleatória'),
              'partial': ('#ef6c00', 2, True, 'parcial'), 'zero': ('#6a1b9a', 2, True, 'zero')}
LEG_INIT = [('spifil', 'Conv inicializada pelo SPiFiL'), ('random', 'Conv aleatória (PyTorch)'),
            ('partial', 'Conv parcial (canais novos sem SPiFiL)'), ('zero', 'Conv iniciada em zero (morfismo)')]
def init_info(path):
    """block -> (status, nota). modehb/hybrid: encoder SPiFiL, cabeça aleatória. growth: growth_spec.json/result.json."""
    import json
    from pathlib import Path
    info = {'head': ('random', 'cabeça 1×1 aleatória')}
    if KIND != 'growth':
        return info, 'spifil'
    pp = Path(path).parent
    try:
        gs = json.loads((pp / 'growth_spec.json').read_text()); rs = json.loads((pp / 'result.json').read_text())
    except Exception:
        return info, 'spifil'
    fb = {int(x['layer']): x for x in rs.get('new_filter_init', []) if x.get('n_new')}
    for L in gs.get('inheritance', {}).get('per_layer', []):
        ci = int(L['child_index']); n_new = int(L.get('n_new_elements') or 0); ni = (L.get('new_init') or '').lower()
        if L.get('origin') == 'new' and 'zero' in ni:
            info[str(ci)] = ('zero', f"bloco novo iniciado em zero ({ni})")
        elif ci in fb and 'pytorch' in (fb[ci].get('fallback') or '').lower():
            info[str(ci)] = ('partial', f"{fb[ci]['n_new']} filtros novos com init PyTorch")
        elif n_new and 'zero' in ni:
            info[str(ci)] = ('partial', f"{n_new} pesos novos em zero")
        elif n_new and ni and 'spifil' not in ni:
            info[str(ci)] = ('partial', f"{n_new} pesos novos ({ni})")
    h = gs.get('inheritance', {}).get('head')
    if h: info['head'] = ('partial' if 'zero' in h else 'random', f"cabeça: {h}")
    return info, 'spifil'
def page(name, path, sub, proc=None):
    rows, tot, e, fmap, g = trace(path)
    iinfo, idefault = init_info(path)
    cells, nid = [], [2]
    def box(x, y, w, h, text, k, extra='', init=None):
        i = nid[0]; nid[0] += 1; fc, sc = COL[k]; sw = 1
        if init:
            sc, sw, dsh, _ = INIT_STYLE[init]; extra += f"strokeWidth={sw};" + ("dashed=1;" if dsh else "")
        cells.append(f'<mxCell id="{i}" value="{html.escape(text)}" style="rounded=1;html=1;whiteSpace=wrap;fontFamily=Helvetica;fontSize=15;fillColor={fc};strokeColor={sc};{extra}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i
    def edge(a, b, dashed=False, style='straight', pts=None):
        i = nid[0]; nid[0] += 1
        st = {'straight': 'edgeStyle=none;', 'down': 'edgeStyle=none;exitX=0.5;exitY=1;entryX=0.5;entryY=0;',
              'skip_out': 'edgeStyle=orthogonalEdgeStyle;exitX=1;exitY=0.75;entryX=0.5;entryY=0;',
              'skip_in': 'edgeStyle=orthogonalEdgeStyle;exitX=0.5;exitY=1;entryX=1;entryY=0.25;'}[style]
        geo = '<mxGeometry relative="1" as="geometry"/>'
        if pts:
            geo = '<mxGeometry relative="1" as="geometry"><Array as="points">' + ''.join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts) + '</Array></mxGeometry>'
        cells.append(f'<mxCell id="{i}" style="{st}rounded=0;html=1;endArrow=block;endFill=1;strokeColor={"#b85450" if dashed else "#222222"};strokeWidth=1.5;fontSize=10;{"dashed=1;" if dashed else ""}" edge="1" parent="1" source="{a}" target="{b}">{geo}</mxCell>')
    def edge_xy(a, b, ex, ey, nx, ny, dashed=False):
        i = nid[0]; nid[0] += 1
        cells.append(f'<mxCell id="{i}" style="edgeStyle=none;exitX={ex};exitY={ey};entryX={nx};entryY={ny};rounded=0;html=1;endArrow=block;endFill=1;strokeColor=#222222;strokeWidth=1.2;fontSize=10;" edge="1" parent="1" source="{a}" target="{b}"><mxGeometry relative="1" as="geometry"/></mxCell>')
    gl = ', '.join(f"{k}={g[k]}" for k in ('init', 'ssl', 'reg', 'pooling', 'protect', 'ae_target') if k in g)
    CX = X0 + W + GX                       # coluna central
    RX = X0 + 3 * W + 2 * GX + 30           # coluna do residual: à direita de TUDO (inclusive dos ramos Inception)
    box(X0, 10, 4 * W + 2 * GX + 60, 60, f"<b><font style='font-size:20px'>{name}</font></b><br><font color='#444444'>{sub} · params {tot:,} (encoder {e:,} · cabeça {tot-e:,}) · {gl}</font>", 'input', 'align=left;spacingLeft=8;fillColor=none;strokeColor=none;fontSize=12;')
    y = 10 + 60 + GY
    prev = box(CX, y, W, H, 'Entrada 3×224×224', 'input'); y += H + GY
    lane = [0]
    blocks = {}
    for n, m, sh, p in rows:
        b = n.split('.')[1] if n.startswith('encoder.') else 'head'
        blocks.setdefault(b, []).append((n, m, sh, p))
    def conv_init(b, k):
        if k not in ('conv', 'inception', 'depthwise', 'pointwise', 'skip', 'head'): return None, ''
        st, note = iinfo.get(b, (idefault, ''))
        return st, note
    for b, items in blocks.items():
        if b == 'head': continue
        y0 = y; block_in = prev; bp = sum(it[3] for it in items)
        y += 32
        incep = [it for it in items if kind(it[0], it[1]) == 'inception' or ('branches' in it[0] and 'pool' in it[0])]
        rest = [it for it in items if it not in incep and kind(it[0], it[1]) != 'skip']
        skip = [it for it in items if kind(it[0], it[1]) == 'skip']
        st_b, note_b = iinfo.get(b, (idefault, ''))
        if incep:
            # um ramo por coluna (b1, b3, b5, pool...); módulos do mesmo ramo empilhados (ex.: pool -> proj 1x1)
            import re as _re
            cols = {}
            for it in incep:
                key = _re.search(r"branches\.([^.]+)", it[0]); cols.setdefault(key.group(1) if key else it[0], []).append(it)
            n = len(cols); span = 3 * W + 2 * GX; gap = 18; bw = min(W, (span - (n - 1) * gap) / n); x0 = X0 + (span - (n * bw + (n - 1) * gap)) / 2
            depth = max(len(v) for v in cols.values()); bh = H + 24
            last_ids = []
            for j, (ck, mods) in enumerate(cols.items()):
                prev_c = None
                for d_, (nn_, m, sh, p) in enumerate(mods):
                    k = kind(nn_, m) or 'pool'; ini, _ = conv_init(b, k)
                    tag = f"<br><i>{INIT_STYLE[ini][3]}</i>" if ini else ''
                    yy = y + d_ * (bh + GY) + (depth - len(mods)) * (bh + GY)   # alinha a base dos ramos
                    i_ = box(x0 + j * (bw + gap), yy, bw, bh, f"{label(m)}<br>{sh[0]}×{sh[1]}×{sh[2]}{tag}", k, init=ini)
                    if prev_c is None:
                        edge_xy(prev, i_, (j + 0.5) / n, 1, 0.5, 0)
                    else:
                        edge(prev_c, i_, style='down')
                    prev_c = i_
                last_ids.append(prev_c)
            y += depth * (bh + GY) + 10
            cat = box(CX, y, W, H, f"Concat → {sum(r[2][0] for r in incep if isinstance(r[1], nn.Conv2d) and 'pool.pool' not in r[0])} canais", 'pool')
            for j, i_ in enumerate(last_ids): edge_xy(i_, cat, 0.5, 1, (j + 0.5) / n, 0)
            prev = cat; y += H + GY
        for nn_, m, sh, p in rest:
            k = kind(nn_, m); ini, _ = conv_init(b, k)
            tag = f"<br><i>{INIT_STYLE[ini][3]}</i>" if ini else ''
            i_ = box(CX, y, W, H + (24 if ini else 0), f"{label(m)} · {sh[0]}×{sh[1]}×{sh[2]}" + (f"<br>{p:,} params".replace(',', '.') if p else '') + tag, k, init=ini)
            edge(prev, i_, style='down'); prev = i_; y += H + (24 if ini else 0) + GY
        for nn_, m, sh, p in skip:
            ini, _ = conv_init(b, 'skip')
            lane[0] += 1; rx = RX + ((lane[0] + 1) % 2) * 60   # residuais consecutivos em faixas alternadas: as linhas não se sobrepõem
            s = box(rx, (y0 + y) / 2 - H / 2, W, H + 12, f"Residual {label(m)}<br>{sh[0]}×{sh[1]}×{sh[2]}<br><i>{INIT_STYLE[ini][3] if ini else ''}</i>", 'skip', init=ini)
            edge(block_in, s, True, 'skip_out'); edge(s, prev, True, 'skip_in')
        title = f"Block {int(b)+1} · {bp:,} params".replace(',', '.') + (f" · {note_b}" if note_b else '')
        i = nid[0]; nid[0] += 1
        cells.insert(0, f'<mxCell id="{i}" value="{html.escape(title)}" style="rounded=1;arcSize=3;html=1;whiteSpace=wrap;fillColor=#fafafa;strokeColor=#cccccc;verticalAlign=top;align=left;spacingLeft=8;fontSize=15;fontStyle=1;fontColor=#555555;" vertex="1" parent="1"><mxGeometry x="{X0-10}" y="{y0-8}" width="{3*W+2*GX+W+120}" height="{y-y0+2}" as="geometry"/></mxCell>')
        y += 10
    for nn_, m, sh, p in blocks.get('head', []):
        if p == 0: continue   # sem cabeça treinável (ex.: AE de outubro: o mapa da última camada é lido direto)
        ini, note = iinfo.get('head', ('random', ''))
        i_ = box(CX, y, W, H + 32, f"Cabeça {label(m)} · {sh[0]}×{sh[1]}×{sh[2]}<br>{p:,} params".replace(',', '.') + f"<br><i>{INIT_STYLE[ini][3]}</i>", 'head', init=ini)
        edge(prev, i_, style='down'); prev = i_; y += H + 32 + GY
    r = box(CX - 30, y, W + 60, H + 34, f"Leitura 2ª ordem (μ+σ+√Cov, PCA k=88 → 4096) + TTA<br>sobre o mapa {fmap[0]}×{fmap[1]}×{fmap[2]} (sem treino)", 'readout')
    edge(prev, r, True, 'down'); y += H + 34 + GY
    svm = box(CX, y, W, H, "SVM linear C=100 (teste)", 'out'); edge(r, svm, style='down'); y += H + GY
    lx = X0 + 4 * W + 2 * GX + 170; ly = 90
    for j, (k, t) in enumerate(LEG):
        box(lx, ly + j * 42, 300, 34, t, k, 'fontSize=13;')
    for j, (ini, t) in enumerate(LEG_INIT):
        box(lx, ly + (len(LEG) + j) * 42 + 16, 300, 34, t, 'conv', 'fontSize=13;', init=ini)
    if proc is None:
        ssl = {'lejepa': 'LeJEPA', 'simclr': 'SimCLR', 'ae': 'autoencoder global'}.get(g.get('ssl'), g.get('ssl'))
        reg = {'sigreg': '+ SIGReg', 'weak_sigreg': '+ SIGReg fraco', 'none': ''}.get(g.get('reg'), '')
        tgt = {'pixel': 'pixels', 'masked': 'imagem mascarada', 'feature': 'feature'}.get(g.get('ae_target'), g.get('ae_target'))
        prot = {'full': 'taxa normal', 'lowlr': 'taxa reduzida', 'freeze': 'congelados'}.get(g.get('protect'), g.get('protect'))
        fid = g.get('_fidelity')
        proc = (f"<b>Procedimento: AE + {ssl}</b><br>1) SPiFiL inicializa cada bloco ({g.get('spifil_images_per_class', '?')} img/classe)"
                f"<br>2) Autoencoder local por bloco (alvo: {tgt}); bloco congelado em seguida"
                f"<br>3) SSL global no modelo inteiro: {ssl} {reg} ({int(fid) if fid else '?'} épocas; filtros SPiFiL com {prot})"
                f"<br>4) Leitura 2ª ordem + TTA → SVM linear C=100")
    py = ly + (len(LEG) + len(LEG_INIT)) * 42 + 44
    box(lx, py, 320, 300, proc, 'out', 'align=left;spacingLeft=6;verticalAlign=top;fontSize=13;')
    body = ''.join(cells)
    return f'<diagram name="{html.escape(name)}"><mxGraphModel dx="1200" dy="900" grid="1" gridSize="10" page="1" pageScale="1" pageWidth="{int(lx)+360}" pageHeight="{int(y)+80}"><root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root></mxGraphModel></diagram>'
if __name__ == '__main__':
    # uso: drawio_arch.py [--kind modehb|growth|hybrid] [--code-root DIR] [--frag] out spec...
    a = sys.argv[1:]
    while a and a[0].startswith('--'):
        if a[0] == '--kind': KIND = a[1]; a = a[2:]
        elif a[0] == '--code-root': sys.path.insert(0, a[1]); a = a[2:]
        elif a[0] == '--frag': FRAG = True; a = a[1:]
    out = a[0]; pages = []
    for spec in a[1:]:
        parts = spec.split('|')
        pages.append(page(parts[0], parts[1], parts[2], parts[3] if len(parts) > 3 else None))
    if 'FRAG' in globals(): open(out, 'w').write(''.join(pages))
    else: open(out, 'w').write('<mxfile host="drawio">' + ''.join(pages) + '</mxfile>')
    print(f"{out}: {len(pages)} abas")
