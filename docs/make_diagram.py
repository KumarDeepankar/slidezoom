"""Build docs/slidezoom.html (the architecture diagram) and render it to docs/slidezoom.png.

    python docs/make_diagram.py            # needs Google Chrome; Pillow trims the bottom margin
"""
import subprocess
from html import escape
from pathlib import Path

import numpy as np

DOCS = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

STYLE = {  # fill, stroke, dash, title colour
    "plain": ("#ffffff", "#cfcfd6", None, "#1d1d1f"),
    "frozen": ("#efedf3", "#a9a5b8", "4 3", "#1d1d1f"),
    "trained": ("#e4f2ef", "#2e8b7a", None, "#1d6b5e"),
    "text": ("#fbe8d5", "#e2a66e", None, "#1d1d1f"),
    "zoom": ("#eee8fb", "#7c5cd6", None, "#4f35a8"),
}
ARROW = {"grey": "#6e6e78", "teal": "#2e8b7a", "purple": "#7c5cd6"}


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.items = w, h, []

    def box(self, x, y, w, h, title, sub=None, kind="plain", size=12, sub_size=9.5, bold=True, rx=6):
        fill, stroke, dash, color = STYLE[kind]
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="1.3"{d}/>')
        cy = y + h / 2 + (-(sub_size * 0.6) if sub else 0) + size * 0.36
        self.text(x + w / 2, cy, title, size=size, color=color, weight=600 if bold else 400)
        if sub:
            for i, line in enumerate(sub.split("\n")):
                self.text(x + w / 2, cy + size * 0.55 + sub_size * 1.05 * (i + 1) - 1, line, size=sub_size, color="#6e6e78")

    def text(self, x, y, s, size=10, color="#3a3a40", anchor="middle", weight=400, italic=False, mono=False):
        fam = ' font-family="SF Mono, Menlo, monospace"' if mono else ""
        st = ' font-style="italic"' if italic else ""
        self.items.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
                          f'font-weight="{weight}"{fam}{st}>{escape(s)}</text>')

    def line(self, pts, color="grey", dash=None, head=True, width=1.4):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#a-{color})"' if head else ""
        p = " ".join(f"{x},{y}" for x, y in pts)
        self.items.append(f'<polyline points="{p}" fill="none" stroke="{ARROW[color]}" stroke-width="{width}"{d}{m}/>')

    def path(self, d, color="purple", dash="5 3", head=True, width=1.4):
        m = f' marker-end="url(#a-{color})"' if head else ""
        self.items.append(f'<path d="{d}" fill="none" stroke="{ARROW[color]}" stroke-width="{width}" '
                          f'stroke-dasharray="{dash}"{m}/>')

    def raw(self, s):
        self.items.append(s)

    def render(self):
        defs = "".join(f'<marker id="a-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                       f'orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{c}"/></marker>'
                       for k, c in ARROW.items())
        return (f'<svg viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
                f'font-family="-apple-system, Helvetica Neue, Arial, sans-serif"><defs>{defs}</defs>'
                + "".join(self.items) + "</svg>")


def matrix(s: Svg, x, y, name, note, kind="trained", circle=False):
    fill, stroke, _, color = STYLE[kind]
    if circle:
        s.raw(f'<circle cx="{x + 22}" cy="{y + 22}" r="13" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
        s.text(x + 22, y + 26, name, size=11, color=color, weight=600)
    else:
        s.raw(f'<rect x="{x}" y="{y}" width="44" height="44" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
        for i in range(1, 4):
            s.raw(f'<line x1="{x + 11 * i}" y1="{y}" x2="{x + 11 * i}" y2="{y + 44}" stroke="{stroke}" stroke-opacity=".25"/>')
            s.raw(f'<line x1="{x}" y1="{y + 11 * i}" x2="{x + 44}" y2="{y + 11 * i}" stroke="{stroke}" stroke-opacity=".25"/>')
        s.raw(f'<rect x="{x + 9}" y="{y + 15}" width="26" height="14" fill="{fill}"/>')
        s.text(x + 22, y + 26, name, size=10, color=color, weight=600)
    for i, line in enumerate(note.split("\n")):
        s.text(x + 22, y + 60 + 12 * i, line, size=9, color="#6e6e78")


# ---------------------------------------------------------------------------------------------- 1
def architecture():
    s = Svg(640, 430)
    s.box(10, 20, 105, 46, "Whole slide", "H&E · ~100,000 px")
    s.box(140, 20, 105, 46, "Tile encoder", "Phikon · tile → 768", kind="frozen")
    s.box(270, 20, 105, 46, "Tile vectors", "all N tiles × 768")
    s.box(400, 20, 130, 46, "Hierarchy", "regions · sub-regions · tiles", kind="zoom")
    for x in (115, 245, 375):
        s.line([(x + 2, 43), (x + 23, 43)])
    s.box(400, 100, 130, 46, "Projection", "W_proj + level embedding", kind="trained")
    s.line([(465, 66), (465, 98)])
    s.line([(465, 146), (465, 208)], color="teal")
    s.text(472, 168, "3 slide memories:", size=9, color="#1d6b5e", anchor="start")
    s.text(472, 180, "regions · sub-regions · tiles", size=9, color="#1d6b5e", anchor="start")
    s.box(10, 212, 130, 56, "Question", '"Is there tumour?"', kind="text")
    s.box(385, 210, 160, 62, "", kind="frozen")
    s.text(465, 229, "GPT-2 · frozen", size=12, color="#1d1d1f", weight=600)
    s.box(395, 238, 140, 24, "4 zoom cross-attn layers", kind="zoom", size=9.5)
    s.box(570, 212, 65, 56, "Answer", '"yes"', kind="text")
    s.line([(142, 240), (383, 240)])
    s.text(262, 233, "T words × 768", size=9, color="#6e6e78")
    s.line([(547, 240), (568, 240)])
    s.raw('<line x1="10" y1="292" x2="630" y2="292" stroke="#e3e3e8"/>')
    s.text(10, 315, "Trained matrices", size=12, color="#1d1d1f", anchor="start", weight=600)
    s.text(120, 315, "each square is 768 × 768 numbers unless noted, starts random, learned by us",
           size=9.5, color="#6e6e78", anchor="start")
    s.text(10, 336, "Once per slide", size=9, color="#6e6e78", anchor="start")
    s.text(175, 336, "Inside each of the 4 zoom cross-attention layers", size=9, color="#6e6e78", anchor="start")
    matrix(s, 10, 344, "W_proj", "tile → word\nspace")
    matrix(s, 85, 344, "E_lvl", "3 × 768: which\nlevel am I?")
    for i, (n, note) in enumerate([("W_q", "word → query"), ("W_k", "cell → key"),
                                   ("W_v", "cell → value"), ("W_o", "back into\nthe word")]):
        matrix(s, 175 + i * 68, 344, n, note)
    matrix(s, 447, 344, "g", "gate, starts 0", circle=True)
    matrix(s, 517, 344, "β", "zoom, starts 0", kind="zoom", circle=True)
    s.text(612, 370, "× 4", size=11, color="#1d6b5e", weight=600)
    return s.render()


# ---------------------------------------------------------------------------------------------- 2
def end_to_end():
    s = Svg(640, 470)
    s.text(110, 14, "SLIDE · ONCE PER SLIDE", size=8.5, color="#8a8a94")
    s.text(435, 14, "GPT-2 · 12 BLOCKS, FROZEN", size=8.5, color="#8a8a94")
    left = [("Whole slide", "H&E · ~100,000 px wide", "plain"),
            ("Tissue tiles", "~4,000 tiles of 256 px (128 µm)", "plain"),
            ("Phikon ViT · frozen", "each tile → 768 numbers, saved to disk", "frozen"),
            ("Build hierarchy", "group tiles by grid · mean-pool", "zoom"),
            ("Projection W_proj + E_lvl", "768 × 768 · trained", "trained")]
    y = 24
    for i, (t, sub, k) in enumerate(left):
        s.box(10, y, 200, 38, t, sub, kind=k, size=10.5, sub_size=8.5)
        if i:
            s.line([(110, y - 12), (110, y - 2)])
        y += 50
    mems = [("Region memory", "R × 768 · 32×32 tiles (~4 mm) · R ≈ 10–30"),
            ("Sub-region memory", "S × 768 · 8×8 tiles (~1 mm) · S ≈ 100–300"),
            ("Tile memory", "N × 768 · every tile · N ≈ 4,000")]
    mem_y = [300, 352, 404]
    s.line([(110, y - 12), (110, 298)])
    s.text(116, y + 2, "once per question", size=8.5, color="#6e6e78", anchor="start")
    for (t, sub), my in zip(mems, mem_y):
        s.box(10, my, 200, 40, t, sub, kind="trained", size=10.5, sub_size=8.2)
    s.line([(110, 342), (110, 350)], head=False)
    s.line([(110, 394), (110, 402)], head=False)

    right = [("Question", '"Question: Is there tumour? Answer:"', "text"),
             ("Tokens + position embeddings", "T word pieces → T × 768", "frozen"),
             ("GPT-2 blocks 1–3", None, "frozen"),
             ("Zoom cross-attn 1 · reads regions", None, "zoom"),
             ("GPT-2 blocks 4–6", None, "frozen"),
             ("Zoom cross-attn 2 · reads sub-regions", None, "zoom"),
             ("GPT-2 blocks 7–9", None, "frozen"),
             ("Zoom cross-attn 3 · reads tiles", None, "zoom"),
             ("GPT-2 blocks 10–12", None, "frozen"),
             ("Zoom cross-attn 4 · reads tiles", None, "zoom"),
             ("LayerNorm + LM head", "last row → 50,257 word scores → pick the top word", "frozen"),
             ("Answer, one word at a time", '"yes", then (end) → stop', "text")]
    ys, y = [], 24
    for i, (t, sub, k) in enumerate(right):
        h = 32 if sub else 24
        s.box(295, y, 280, h, t, sub, kind=k, size=10, sub_size=8.2)
        if i:
            s.line([(435, y - 9), (435, y - 1)])
        ys.append((y, h))
        y += h + 10
    zoom_rows = [3, 5, 7, 9]
    # memory -> cross-attention lanes
    lanes = [(0, 3, 236), (1, 5, 250), (2, 7, 264), (2, 9, 264)]
    for m, r, x in lanes:
        my = mem_y[m] + 20
        ry = ys[r][0] + ys[r][1] / 2
        s.line([(210, my), (x, my), (x, ry), (293, ry)], color="teal")
    # zoom prior between consecutive cross-attention blocks
    for a, b in zip(zoom_rows, zoom_rows[1:]):
        ya, yb = ys[a][0] + 12, ys[b][0] + 12
        s.path(f"M575,{ya} C598,{ya} 598,{yb} 577,{yb}")
    mid = (ys[3][0] + ys[9][0]) / 2 + 12
    s.raw(f'<text x="610" y="{mid}" font-size="8.5" fill="#4f35a8" font-weight="600" text-anchor="middle" '
          f'transform="rotate(-90 610 {mid})">zoom prior: β · log a of the block above</text>')
    # generation loop
    last = ys[-1][0] + 12
    s.line([(575, last), (628, last), (628, ys[1][0] + 16), (577, ys[1][0] + 16)], dash="3 3")
    s.raw(f'<text x="636" y="{ys[11][0] - 30}" font-size="8.5" fill="#6e6e78" text-anchor="start" '
          f'transform="rotate(-90 636 {ys[11][0] - 30})">add the new word, run again</text>')
    return s.render()


# ---------------------------------------------------------------------------------------------- 3
def hierarchy_panel():
    """The same toy slide at the three levels, each coarse cell filled with the mean colour of its tiles."""
    s = Svg(640, 250)
    cols, rows = 64, 40
    W, gap = 186, 41
    px = W / cols
    yy, xx = np.mgrid[0:rows, 0:cols]
    tissue = (((xx - 20) / 17) ** 2 + ((yy - 21) / 15) ** 2 < 1) | (((xx - 46) / 14) ** 2 + ((yy - 17) / 12) ** 2 < 1)
    pink, lilac, dark = np.array([236, 170, 205]), np.array([176, 140, 214]), np.array([96, 52, 150])
    t = np.clip(xx / cols + 0.25 * np.sin(yy / 6), 0, 1)[..., None]
    col = pink * (1 - t) + lilac * t
    focus = np.exp(-(((xx - 44) ** 2 + (yy - 12) ** 2) / 14.0))[..., None]  # a small dark focus
    col = col * (1 - focus) + dark * focus
    col = np.clip(col + np.random.default_rng(3).normal(0, 9, col.shape), 0, 255)

    def cell_means(size):
        out = np.zeros_like(col)
        for r0 in range(0, rows, size):
            for c0 in range(0, cols, size):
                m = tissue[r0:r0 + size, c0:c0 + size]
                if m.any():
                    out[r0:r0 + size, c0:c0 + size] = col[r0:r0 + size, c0:c0 + size][m].mean(0)
        return out

    def count(size):
        return sum(tissue[r:r + size, c:c + size].any() for r in range(0, rows, size) for c in range(0, cols, size))

    top = 52
    reg, sub = (32, 0, 32, 32), (40, 8, 8, 8)  # highlighted region and sub-region: (col, row, w, h) in tiles
    panels = [(32, "Level 0 · regions", "4 mm · average of its tiles", "#7c5cd6"),
              (8, "Level 1 · sub-regions", "1 mm · average of its tiles", "#2e8b7a"),
              (1, "Level 2 · tiles", "128 µm · one Phikon vector each", "#6e6e78")]
    for k, (size, title, sub_t, color) in enumerate(panels):
        ox = k * (W + gap)
        fill = col if size == 1 else cell_means(size)
        for r in range(rows):
            for c in range(cols):
                if tissue[r, c]:
                    rgb = ",".join(str(int(v)) for v in fill[r, c])
                    s.raw(f'<rect x="{ox + c * px:.2f}" y="{top + r * px:.2f}" width="{px + .05:.2f}" height="{px + .05:.2f}" fill="rgb({rgb})"/>')
        if size > 1:  # cell borders, only where there is tissue
            for r0 in range(0, rows, size):
                for c0 in range(0, cols, size):
                    if tissue[r0:r0 + size, c0:c0 + size].any():
                        s.raw(f'<rect x="{ox + c0 * px:.2f}" y="{top + r0 * px:.2f}" width="{min(size, cols - c0) * px:.2f}" '
                              f'height="{min(size, rows - r0) * px:.2f}" fill="none" stroke="#fff" stroke-width="1.2"/>')
        s.text(ox + W / 2, 16, title, size=11, color=STYLE["zoom"][3] if k == 0 else STYLE["trained"][3] if k == 1 else "#1d1d1f", weight=600)
        s.text(ox + W / 2, 30, sub_t, size=9, color="#6e6e78")
        n = count(size) if size > 1 else int(tissue.sum())
        s.text(ox + W / 2, top + rows * px + 24, f"{n:,} tokens", size=12, color="#1d1d1f", weight=600)

        def outline(box, stroke, w, dash=None):
            c0, r0, bw, bh = box
            bh = min(bh, rows - r0)
            d = f' stroke-dasharray="{dash}"' if dash else ""
            s.raw(f'<rect x="{ox + c0 * px:.2f}" y="{top + r0 * px:.2f}" width="{bw * px:.2f}" height="{bh * px:.2f}" '
                  f'fill="none" stroke="{stroke}" stroke-width="{w}"{d}/>')
        if k == 0:
            outline(reg, "#7c5cd6", 2.4)
        if k == 1:
            outline(reg, "#7c5cd6", 1.6, "4 3")
            outline(sub, "#2e8b7a", 2.4)
        if k == 2:
            outline(sub, "#2e8b7a", 2.0)
        if k < 2:
            ay = top + rows * px / 2
            s.line([(ox + W + 6, ay), (ox + W + gap - 6, ay)], color="purple")
            s.text(ox + W + gap / 2, ay - 6, "zoom in", size=8.5, color="#4f35a8")
    s.text(320, 228, "Outlines: one zoom path. The model picks a region (purple), then a sub-region inside it (green), "
           "then reads its 64 tiles.", size=9.5, color="#3a3a40")
    return s.render()


# ---------------------------------------------------------------------------------------------- 4
def prior_panel():
    s = Svg(640, 350)
    s.box(10, 20, 140, 40, "Previous block's attention", "a_{l-1} · QT × N_coarse", kind="zoom", size=10, sub_size=8.5)
    s.box(180, 20, 110, 40, "Copy to children", "gather by ancestor", kind="plain", size=10, sub_size=8.5)
    s.box(320, 20, 110, 40, "Prior", "QT × N_fine", kind="plain", size=10, sub_size=8.5)
    s.box(460, 20, 110, 40, "β_l · log(prior)", "β starts at 0", kind="zoom", size=10, sub_size=8.5)
    for x in (150, 290, 430):
        s.line([(x + 2, 40), (x + 28, 40)], color="purple")
    s.box(10, 100, 100, 36, "Words", "T × 768", kind="text", size=10, sub_size=8.5)
    s.box(10, 180, 100, 36, "Memory, level l", "N_fine × 768", kind="trained", size=10, sub_size=8.5)
    s.box(140, 104, 50, 28, "W_q", kind="trained", size=9.5)
    s.box(140, 166, 50, 28, "W_k", kind="trained", size=9.5)
    s.box(140, 206, 50, 28, "W_v", kind="trained", size=9.5)
    s.line([(110, 118), (138, 118)])
    s.line([(110, 198), (124, 198), (124, 180), (138, 180)])
    s.line([(124, 198), (124, 220), (138, 220)])
    s.box(230, 130, 130, 40, "Scores", "q · kᵀ / √d · QT × N_fine", kind="plain", size=10, sub_size=8.5)
    s.line([(190, 118), (210, 118), (210, 142), (228, 142)])
    s.line([(190, 180), (210, 180), (210, 158), (228, 158)])
    s.raw('<circle cx="515" cy="150" r="11" fill="#fff" stroke="#7c5cd6" stroke-width="1.4"/>')
    s.text(515, 154, "+", size=13, color="#4f35a8", weight=600)
    s.line([(360, 150), (502, 150)])
    s.line([(515, 60), (515, 137)], color="purple")
    s.box(440, 190, 150, 40, "Mask + softmax", "padding · top-k at inference", kind="plain", size=10, sub_size=8.5)
    s.line([(515, 161), (515, 188)])
    s.box(250, 245, 150, 40, "Read", "attention · values → W_o", kind="trained", size=10, sub_size=8.5)
    s.line([(190, 220), (230, 220), (230, 258), (248, 258)])
    s.line([(515, 230), (515, 265), (402, 265)])
    s.box(10, 300, 190, 36, "words + tanh(g) · W_o(read)", "→ next GPT-2 block", kind="text", size=9.5, sub_size=8.5)
    s.line([(325, 285), (325, 318), (202, 318)])
    s.path("M590,210 L628,210 L628,312", dash="5 3")
    s.text(622, 328, "attention, mean over heads", size=8.5, color="#4f35a8", anchor="end")
    s.text(622, 340, "→ prior of the next block", size=8.5, color="#4f35a8", anchor="end")
    return s.render()


# ---------------------------------------------------------------------------------------------- 5
def beta_panel():
    s = Svg(640, 150)
    s.box(10, 50, 150, 44, "Words ask the cells", "q · kᵀ / √d", kind="plain", size=10, sub_size=9)
    s.box(200, 50, 170, 44, "Where I looked one level up", "β · log a_{l-1}(parent)", kind="zoom", size=10, sub_size=9)
    s.raw('<circle cx="400" cy="72" r="11" fill="#fff" stroke="#6e6e78" stroke-width="1.3"/>')
    s.text(400, 76, "+", size=13, color="#3a3a40", weight=600)
    s.box(440, 50, 190, 44, "softmax → attention on cells", kind="plain", size=10)
    s.line([(160, 72), (180, 72), (180, 30), (400, 30), (400, 59)])
    s.line([(370, 72), (387, 72)], color="purple")
    s.line([(411, 72), (438, 72)])
    for i, (b, t) in enumerate([("β = 0", "ignore the coarse view (multi-scale Flamingo, the start)"),
                                ("β = 1", "attention = P(parent) × e^score / Z: a child is read only if its parent was"),
                                ("β > 1", "sharper: commit to the regions the last block chose")]):
        s.text(10, 118 + 13 * i, b, size=9.5, color="#4f35a8", anchor="start", weight=600, mono=True)
        s.text(58, 118 + 13 * i, t, size=9.5, color="#3a3a40", anchor="start")
    return s.render()


# ---------------------------------------------------------------------------------------------- 6
def topk_panel():
    s = Svg(640, 220)
    ox, oy = 10, 30

    def grid(x, y, n, cell, lit, color, label):
        for r in range(n):
            for c in range(n):
                on = (r, c) in lit
                s.raw(f'<rect x="{x + c * cell}" y="{y + r * cell}" width="{cell - 2}" height="{cell - 2}" rx="2" '
                      f'fill="{color if on else "#efedf3"}" fill-opacity="{0.85 if on else 1}"/>')
        s.text(x + n * cell / 2, y + n * cell + 16, label, size=9.5, color="#3a3a40", weight=600)

    grid(ox, oy, 2, 60, {(0, 0), (1, 1)}, "#7c5cd6", "regions: keep top 2")
    grid(ox + 170, oy, 8, 15, {(0, 1), (1, 1), (1, 2), (2, 0), (4, 5), (5, 5), (6, 4), (6, 6)} |
         set(), "#2e8b7a", "sub-regions: top 8 inside them")
    for r in range(8):
        for c in range(8):
            if not ((r < 4 and c < 4) or (r >= 4 and c >= 4)):
                s.raw(f'<rect x="{ox + 170 + c * 15}" y="{oy + r * 15}" width="13" height="13" rx="2" fill="#fff" '
                      f'stroke="#e3e3e8" stroke-dasharray="2 2"/>')
    s.line([(ox + 128, oy + 60), (ox + 160, oy + 60)], color="purple")
    s.line([(ox + 298, oy + 60), (ox + 330, oy + 60)], color="purple")
    s.box(ox + 340, oy + 20, 140, 80, "≤ 8 × 64 = 512 tiles", "scored per text token,\nout of the whole slide",
          kind="trained", size=11, sub_size=9)
    s.text(ox + 500, oy + 44, "measured with --topk 2,8:", size=9.5, color="#6e6e78", anchor="start")
    s.text(ox + 500, oy + 64, "383 of 4,797 tiles", size=13, color="#1d6b5e", anchor="start", weight=600)
    s.text(ox + 500, oy + 82, "≈ 8% (same with β = 0)", size=9.5, color="#6e6e78", anchor="start")
    s.text(ox, oy + 170, "Training reads every cell (soft prior, fully differentiable). Pruning is switched on only at "
           "inference, per text token,", size=9.5, color="#3a3a40", anchor="start")
    s.text(ox, oy + 184, "so different questions can zoom into different parts of the same slide.",
           size=9.5, color="#3a3a40", anchor="start")
    return s.render()


CSS = """
:root{--ink:#1d1d1f;--muted:#6e6e78;--line:#e3e3e8}
*{box-sizing:border-box}
body{margin:0;background:#fff;color:var(--ink);font-family:-apple-system,"Helvetica Neue",Arial,sans-serif;width:2190px}
main{padding:28px 36px 30px}
.grid{display:grid;grid-template-columns:repeat(3,690px);gap:0 34px}
header{display:flex;align-items:baseline;gap:22px;margin-bottom:4px}
h1{font-size:30px;margin:0;letter-spacing:-.5px}
.lede{color:var(--muted);margin:0;font-size:14px}
h2{font-size:16px;margin:22px 0 10px}
.card{border:1px solid var(--line);border-radius:12px;padding:20px 18px 16px}
.card p{font-size:12.5px;line-height:1.6;color:#3a3a40;margin:12px 4px 2px}
code{font-family:"SF Mono",Menlo,monospace;font-size:11.5px;background:#f4f4f6;padding:1px 4px;border-radius:4px}
.legend{display:flex;gap:18px;font-size:11.5px;color:#3a3a40;margin:10px 4px 0}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:13px;height:13px;border-radius:3px;display:inline-block;border:1.3px solid}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th,td{text-align:left;padding:8px 6px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:500;font-size:11.5px}
td code{white-space:nowrap}
ol{font-size:13px;line-height:1.65;padding-left:20px;margin:0}
ol li{margin-bottom:6px}
.new{color:#4f35a8;font-weight:600}
"""


def section(title, body, card=True):
    inner = f'<div class="card">{body}</div>' if card else body
    return f"<section><h2>{title}</h2>{inner}</section>"


def page():
    legend = ('<div class="legend">'
              '<span><i class="sw" style="background:#efedf3;border-color:#a9a5b8;border-style:dashed"></i>frozen, used as is</span>'
              '<span><i class="sw" style="background:#e4f2ef;border-color:#2e8b7a"></i>trained by us (~10M parameters)</span>'
              '<span><i class="sw" style="background:#eee8fb;border-color:#7c5cd6"></i>new in SlideZoom</span>'
              '<span><i class="sw" style="background:#fbe8d5;border-color:#e2a66e"></i>text</span></div>')
    col1 = [
        section("Architecture", architecture() + legend),
        section("From tiles to a hierarchy", hierarchy_panel() + """
<p>The same slide at three levels. A region or sub-region token is simply the <b>average of the tiles inside
it</b>, so every level covers the whole slide, just in less detail: few tokens to scan, many to read closely. Built
from the tile positions saved with the features, so no new feature extraction is needed.</p>"""),
    ]
    col2 = [
        section("End to end", end_to_end() + """
<p>The slide is turned into three memories once per question. The first cross-attention reads <b>regions</b>, the second
<b>sub-regions</b>, the last two read <b>tiles</b>, so magnification follows GPT-2's depth. The <span class="new">purple</span>
arrows are the zoom prior: what each word attended to in one block biases where it looks in the next.</p>"""),
        section("Why β starts at 0", beta_panel() + """
<p>Like the gate <code>g</code>, the zoom strength <code>β</code> is one learned number per layer that starts at 0. At
the start the model is an ordinary multi-scale Flamingo model (baseline B2), so any zooming it does is learned because
it lowers the loss. Plotting β over training shows whether the model chooses to zoom; in the first runs it stayed below 0.25."""),
        section("Results on unseen patients (42 slides, seed 0)", """<table>
<tr><th>Model</th><th>Organ</th><th>Subtype</th><th>Test NLL</th><th>Tiles / token</th></tr>
<tr><td>B0 flat, 512 random tiles</td><td>93%</td><td>74%</td><td>0.098</td><td>512</td></tr>
<tr><td>B1 flat, all tiles</td><td>98%</td><td>76%</td><td>0.089</td><td>4,797</td></tr>
<tr><td>B2 multiscale, β = 0</td><td>95%</td><td>76%</td><td>0.104</td><td>4,797</td></tr>
<tr><td class="new">SlideZoom</td><td>95%</td><td>76%</td><td>0.103</td><td>4,797</td></tr>
<tr><td>SlideZoom, β forced to 0</td><td>95%</td><td>76%</td><td>0.104</td><td>4,797</td></tr>
<tr><td>B2 / SlideZoom, top-k 2,8</td><td>93%</td><td>74%</td><td>0.110 / 0.109</td><td class="new">383 (8%)</td></tr>
</table>
<p><b>The learned zoom prior has no effect here:</b> with β set to 0 no answer changes, and B2 matches SlideZoom on
every slide. No paired difference is significant. Pruning to 8% of the tiles works, but the hierarchy does it,
not β. These questions can be answered from almost any part of the slide; a focal-evidence task is next.</p>"""),
    ]
    col1.append(section("Needle test: finding a small patch", """<table>
<tr><th>Patch</th><th>B1 flat</th><th>B2 multi</th><th>Zoom β₀=1</th><th>Attention on patch, zoom ÷ β off</th></tr>
<tr><td>20%</td><td>0.73</td><td>0.72</td><td>0.70</td><td>—</td></tr>
<tr><td>1%</td><td>0.61</td><td>0.61</td><td>0.59</td><td class="new">×2.05 [1.58, 2.58]</td></tr>
<tr><td>0.5%</td><td>0.72</td><td>0.71</td><td>0.63</td><td class="new">×2.01 [1.49, 2.70]</td></tr>
<tr><td>0.1%</td><td>0.65</td><td>0.66</td><td>0.66</td><td>×0.80 (n.s.)</td></tr>
<tr><th colspan="5">Cross-organ patch (easy to recognise)</th></tr>
<tr><td>0.1%</td><td>0.72</td><td>0.74</td><td>0.77</td><td>B0 (512 tiles): 0.66</td></tr>
</table>
<p>A patch of another subtype is implanted into a held-out slide; chance is 50%. <b>Zoom finds the patch but the
answer doesn't use it:</b> at 0.5–1% the zoom prior doubles the attention on the patch, yet accuracy doesn't
move. With an easy cross-organ patch, reading every tile beats random sampling at 0.1%; the zoom adds nothing
measurable on top.</p>"""))
    col3 = [
        section("Where the zoom prior works", prior_panel() + """
<p>The usual Flamingo block (queries from words, keys and values from the slide, gated residual) plus one new term:
the previous block's attention over coarse cells, averaged over heads, is copied to each fine cell's children and added
to the scores as <code>β · log a</code>. It is computed per text token, so different questions can zoom to different
places.</p>"""),
        section("Reading a whole slide on a budget", topk_panel()),
        section("Models compared", """<table>
<tr><th>Model</th><th>What it is</th><th>Rules out</th></tr>
<tr><td>B0 SlideTalk</td><td>flat cross-attention, 512 random tiles</td><td>reference</td></tr>
<tr><td>B1 flat, all tiles</td><td>flat cross-attention over every tile</td><td>"it's just more tiles"</td></tr>
<tr><td>B2 multiscale</td><td>same hierarchy, β fixed at 0</td><td>"it's multi-scale tokens, not the zoom"</td></tr>
<tr><td>B3 budget</td><td>zoom with <code>--topk 2,8</code> (≤ 512 tiles)</td><td>"it's more compute"</td></tr>
<tr><td class="new">SlideZoom</td><td>hierarchy + learned zoom prior</td><td>—</td></tr>
</table>"""),
    ]
    col1.append(section("How it works", """<ol>
<li>The slide is cut into 128 µm tiles; a frozen encoder turns each into 768 numbers (shared with SlideTalk).</li>
<li>Tiles are grouped into 1 mm sub-regions and 4 mm regions: three memories of the slide.</li>
<li>GPT-2 reads the question. Its four cross-attention layers look at regions, then sub-regions, then tiles, and
each layer's attention steers the next.</li>
<li>GPT-2 writes the answer. Only the projection and cross-attention layers are trained (~10M parameters,
about 12 minutes on a MacBook).</li></ol>""", card=False))
    col3.append(section("Findings so far", """<ol>
<li><b>The zoom makes the model look at small evidence</b> (1.6–2× more attention on a 0.5–1% patch).</li>
<li><b>But it doesn't improve the answers</b> (same accuracy as the identical model with the zoom off).</li>
<li><b>Reading every tile beats 512 random tiles on tiny evidence</b> (0.72–0.77 vs 0.66 at a 0.1% patch).</li>
<li><b>On whole-slide questions, every model scores the same.</b></li>
<li><b>The models can't recognise lung subtypes in a patch</b> (chance even at 20% of the slide).</li>
<li><b>Picking the top regions by their average loses small evidence.</b></li>
<li><b>The zoom has to start switched on</b> (started at 0, it never turns on for an easy patch).</li>
</ol>""", card=False))
    cols = "".join(f'<div class="col">{"".join(c)}</div>' for c in (col1, col2, col3))
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>SlideZoom</title><style>{CSS}</style></head>
<body><main>
<header><h1>SlideZoom</h1>
<p class="lede">A frozen language model reads a whole-slide image coarse to fine, the way a pathologist scans and zooms.</p></header>
<div class="grid">{cols}</div>
</main></body></html>"""


def main():
    html = DOCS / "slidezoom.html"
    png = DOCS / "slidezoom.png"
    html.write_text(page())
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    "--window-size=2190,2400", f"--screenshot={png}", html.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from PIL import Image, ImageChops
    im = Image.open(png).convert("RGB")
    bbox = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
    im.crop((0, 0, im.width, min(im.height, bbox[3] + 60))).save(png)
    print(f"wrote {html} and {png} ({im.width} px wide)")


if __name__ == "__main__":
    main()
