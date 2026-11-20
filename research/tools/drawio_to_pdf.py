"""drawio_to_pdf.py -- exporta para UM PDF (uma página por aba) os .drawio gerados por drawio_arch.py,
desenhando caixas, textos e setas com matplotlib (sem precisar do draw.io instalado).
Uso: python drawio_to_pdf.py saida.pdf a.drawio b.drawio ..."""
import sys, re, html, textwrap
import xml.etree.ElementTree as ET
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
def sty(s):
    return dict(kv.split("=", 1) for kv in (s or "").split(";") if "=" in kv)
def txt(v):
    v = html.unescape(v or "")
    v = re.sub(r"<br\s*/?>", "\n", v)
    bold = "<b>" in v
    return re.sub(r"<[^>]+>", "", v), bold
def anchor(g, ex, ey):
    x, y, w, h = g
    return x + w * ex, y + h * ey
def draw_page(d, pdf):
    m = d.find("mxGraphModel"); W = float(m.get("pageWidth")); H = float(m.get("pageHeight"))
    cells = list(d.iter("mxCell")); geo = {}
    fig = plt.figure(figsize=(W / 72, H / 72)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    for c in cells:
        if c.get("vertex") != "1": continue
        g = c.find("mxGeometry"); x, y, w, h = (float(g.get(k)) for k in ("x", "y", "width", "height")); geo[c.get("id")] = (x, y, w, h)
        s = sty(c.get("style")); fc = s.get("fillColor", "#ffffff"); sc = s.get("strokeColor", "#000000")
        if fc != "none" or sc != "none":
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%g" % (3 if s.get("arcSize") else 6),
                         fc=("none" if fc == "none" else fc), ec=("none" if sc == "none" else sc), lw=float(s.get("strokeWidth", 1)),
                         ls="--" if s.get("dashed") == "1" else "-", zorder=1 if s.get("arcSize") else 2))
        t, bold = txt(c.get("value")); fs = float(s.get("fontSize", 11)) * 0.92
        if s.get("arcSize"):   # moldura de bloco: título no topo
            ax.text(x + 8, y + 4, t, ha="left", va="top", fontsize=fs, color=s.get("fontColor", "#555"), fontweight="bold", zorder=3)
        elif s.get("align") == "left" and s.get("verticalAlign") == "top":
            wc = max(10, int((w - 12) / (fs * 0.62)))
            t = "\n".join(textwrap.fill(line, wc) for line in t.split("\n"))
            ax.text(x + 6, y + 6, t, ha="left", va="top", fontsize=fs, zorder=3, linespacing=1.35)
        elif s.get("align") == "left":
            lines = t.split("\n")
            ax.text(x + 8, y + 6, lines[0], ha="left", va="top", fontsize=15, fontweight="bold", zorder=3)
            ax.text(x + 8, y + 30, "\n".join(lines[1:]), ha="left", va="top", fontsize=9, color="#444444", zorder=3, wrap=True)
        else:
            wc = max(8, int((w - 6) / (fs * 0.62)))
            t = "\n".join(textwrap.fill(line, wc) for line in t.split("\n"))
            ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
                    color=s.get("fontColor", "#000000"), zorder=3, fontstyle="normal")
    for c in cells:
        if c.get("edge") != "1": continue
        a, b = geo.get(c.get("source")), geo.get(c.get("target"))
        if not a or not b: continue
        s = sty(c.get("style")); col = s.get("strokeColor", "#222"); ls = "--" if s.get("dashed") == "1" else "-"
        if "exitX" in s:
            p0 = anchor(a, float(s["exitX"]), float(s["exitY"])); p1 = anchor(b, float(s["entryX"]), float(s["entryY"]))
        else:   # reta entre os centros, cortada nas bordas
            p0 = (a[0] + a[2] / 2, a[1] + a[3]); p1 = (b[0] + b[2] / 2, b[1])
        if s.get("edgeStyle") == "orthogonalEdgeStyle":
            if float(s.get("exitX", .5)) == 1.0:   # sai pela direita, desce até o alvo
                path = [p0, (p1[0], p0[1]), p1]
            else:                                # sai por baixo, vai até a direita do alvo
                path = [p0, (p0[0], p1[1]), p1]
            for q0, q1 in zip(path[:-2], path[1:-1]):
                ax.plot([q0[0], q1[0]], [q0[1], q1[1]], color=col, lw=1.3, ls=ls, zorder=1.5)
            p0 = path[-2]
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, color=col, lw=1.3, ls=ls, zorder=1.5))
    pdf.savefig(fig)
    import os
    if os.environ.get("PNG_DIR"):
        fig.savefig(os.path.join(os.environ["PNG_DIR"], re.sub(r"[^A-Za-z0-9_]+", "_", d.get("name")) + ".png"), dpi=60)
    plt.close(fig)
if __name__ == "__main__":
    out = sys.argv[1]; n = 0
    with PdfPages(out) as pdf:
        for f in sys.argv[2:]:
            for d in ET.parse(f).getroot().iter("diagram"):
                draw_page(d, pdf); n += 1
    print(f"{out}: {n} páginas")
