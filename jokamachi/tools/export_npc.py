# Export NPC portraits (+ expression face patches and stand-alone expression images) as a JS data block.
#   NPC_IMG[id]  = [w, h, lx, ly, rx, ry, cutBottom, b64]      eyes in image px
#   NPC_FACE[id] = {expr: [x, y, w, h, b64]}                    patch laid over NPC_IMG[id]
#   NPC_ALT[id]  = {expr: [w, h, lx, ly, rx, ry, cutBottom, b64]} whole replacement image
import json, os, sys, base64, io, numpy as np
from PIL import Image
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(__file__))
B = '/tmp/claude-0/-home-user-New-games/e10e8c9b-88e9-5bdd-8465-a555008c7672/scratchpad/'
OUT = B + 'final/npc/'; os.makedirs(OUT, exist_ok=True)
info = json.load(open(B + 'npc/info.json')); eyes = json.load(open(B + 'npc/eyes.json'))
report = json.load(open(B + 'npc/faces/report.json'))
STORY = ['me', 'lord', 'mikado', 'genba', 'sanzoku', 'zoku', 'saeki', 'gen', 'shino', 'yami']
SKIP = {'x_chaya'}
Q, AQ = 74, 70
def webp(im, name):
    p = OUT + name + '.webp'; im.save(p, quality=Q, method=6, alpha_quality=AQ); return base64.b64encode(open(p, 'rb').read()).decode(), os.path.getsize(p)
img, face, alt, total = {}, {}, {}, 0
scale = {}
for k in sorted(info):
    if k in SKIP or k.startswith('t_'): continue
    im = Image.open(B + f'npc/{k}.png').convert('RGBA'); W, H = im.size
    maxd = 780 if k in STORY else 540
    s = min(1.0, maxd / H, maxd / W); scale[k] = s
    if s < 1: im = im.resize((round(W * s), round(H * s)), Image.LANCZOS)
    xl, xr, ye = eyes[k]; w, h = im.size
    b, sz = webp(im, k); total += sz
    img[k] = [w, h, round(xl / 100 * w, 1), round(ye / 100 * h, 1), round(xr / 100 * w, 1), round(ye / 100 * h, 1), int(info[k]['cutBottom']), b]
    print(k, (w, h), sz // 1024, 'KB')
for k, d in report.items():
    for ex, v in d.items():
        if not isinstance(v, dict) or 'box' not in v: continue
        p = Image.open(B + f'npc/faces/{k}_{ex}.png').convert('RGBA'); x, y, w, h = v['box']; s = scale[k]
        a = np.asarray(p)[..., 3]; ys, xs = np.where(a > 2); bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        p = p.crop((bx0, by0, bx1, by1)); X, Y = (x + bx0) * s, (y + by0) * s
        if s < 1: p = p.resize((max(1, round(p.width * s)), max(1, round(p.height * s))), Image.LANCZOS)
        b, sz = webp(p, f'{k}_{ex}'); total += sz
        face.setdefault(k, {})[ex] = [round(X, 1), round(Y, 1), p.width, p.height, b]; print(' face', k, ex, sz // 1024, 'KB')
# mikado: stand-alone expression images, aligned to the base portrait by the robe (lower body)
from npc_faces import search
base = Image.open(B + 'npc/mikado.png').convert('RGBA'); bw, bh = base.size
bl = Image.new('RGBA', base.size, 'white'); bl.alpha_composite(base); BG = np.asarray(bl.convert('L')).astype(float) / 255
xl, xr, ye = eyes['mikado']; E = [(xl / 100 * bw, ye / 100 * bh), (xr / 100 * bw, ye / 100 * bh)]
for ex in ['happy', 'angry', 'surprise']:
    t = Image.open(B + f'npc/t_mikado_{ex}.png').convert('RGBA'); s0 = bw / t.width
    tt = t.resize((bw, round(t.height * s0)), Image.LANCZOS); tl = Image.new('RGBA', tt.size, 'white'); tl.alpha_composite(tt)
    T = np.asarray(tl.convert('L')).astype(float) / 255
    hh = min(T.shape[0], BG.shape[0]); Tm = np.ones_like(BG); Tm[:hh] = T[:hh]
    mask = np.zeros_like(BG); mask[int(bh * 0.55):int(bh * 0.95)] = 1
    sc, (s1, th, dx, dy) = search(BG, Tm, mask, 40, scales=np.arange(0.95, 1.051, 0.01), rots=[0.0])
    # base point p -> tile-resized point c + (p + d - c)/s1 ; -> tile point / s0
    c = (bw / 2, bh / 2)
    te = [((c[0] + (px + dx - c[0]) / s1) / s0, (c[1] + (py + dy - c[1]) / s1) / s0) for px, py in E]
    s = min(1.0, 640 / t.height, 640 / t.width)
    if s < 1: t = t.resize((round(t.width * s), round(t.height * s)), Image.LANCZOS)
    b, sz = webp(t, f'mikado_{ex}'); total += sz
    alt.setdefault('mikado', {})[ex] = [t.width, t.height, round(te[0][0] * s, 1), round(te[0][1] * s, 1), round(te[1][0] * s, 1), round(te[1][1] * s, 1), int(info['t_mikado_' + ex]['cutBottom']), b]
    print(' alt mikado', ex, round(sc, 3), (s1, dx, dy), sz // 1024, 'KB')
print('total', total // 1024, 'KB')
def js(o): return json.dumps(o, ensure_ascii=False, separators=(',', ':'))
lines = ['<script>', '/* townsfolk & story NPC portraits: cut-out illustrations, WebP base64 (sources in art/npc/). eyes in image px */']
lines.append('const NPC_IMG = {'); lines += [f'{k}:{js(v)},' for k, v in img.items()]; lines.append('};')
lines.append('const NPC_FACE = {'); lines += [f'{k}:{js(v)},' for k, v in face.items()]; lines.append('};')
lines.append('const NPC_ALT = {'); lines += [f'{k}:{js(v)},' for k, v in alt.items()]; lines.append('};')
lines.append('</script>')
open(B + 'final/npc_b64.js', 'w').write('\n'.join(lines) + '\n')
