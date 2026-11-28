# Build face patches: map each (x4 upscaled) expression tile onto the full-body sprite's face.
# chain: sprite P -> ref tile p (inverse similarity from landmarks) -> expression tile q (registration) -> x4 pixels.
import json, sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi

LM = {  # landmarks: [eyeL, eyeR, chin, mouth] in ref-tile px and in sprite px
  'koharu': {'ref': 'normal', 't': [(91.8, 141.3), (145.5, 133.8), (125, 189.5), (122.5, 171)], 's': [(350, 277), (452, 262), (415, 370), (407, 335)]},
  'yukino': {'ref': 'normal', 't': [(103, 118), (163.75, 116.25), (136.25, 186.25), (135, 159.5)], 's': [(335, 180), (412, 180), (380, 267), (378, 235)]},
  'chizuru': {'ref': 'normal', 't': [(103.75, 105), (158, 103.75), (131.25, 167.5), (132.5, 143)], 's': [(372, 212), (462, 206), (424, 306), (422, 267)]},
  'kaede': {'ref': 'u_normal', 't': [(72.8, 108.9), (116.6, 106.7), (100.6, 150), (95, 135.6)], 's': [(235, 260), (327, 260), (282, 352), (281, 324)]},
}
def similarity(src, dst):
    # least squares z' = a z + b with complex numbers (rotation+uniform scale+translation)
    s = np.array([complex(*p) for p in src]); d = np.array([complex(*p) for p in dst])
    sm, dm = s.mean(), d.mean(); a = ((d - dm) * np.conj(s - sm)).sum() / (np.abs(s - sm) ** 2).sum()
    return a, dm - a * sm
def build(k, n, reg, shape, out_prefix, color_fix=True):
    L = LM[k]; a, b = similarity(L['t'], L['s'])
    sp = Image.open(f'art/{k}.png').convert('RGBA'); S = np.asarray(sp).astype(np.float64) / 255
    tile = np.asarray(Image.open(f'diff/tiles/{k}_{n}_x4.png').convert('RGB')).astype(np.float64) / 255
    ref = Image.open(f'diff/tiles/{k}_{L["ref"]}.png'); c1 = np.array([ref.width / 2, ref.height / 2])
    s_, th, dx, dy = reg[k][n][:4]
    t = np.deg2rad(th); Mi = np.linalg.inv(np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) * s_)
    # patch frame in sprite px
    (lx, ly), (rx, ry) = L['s'][0], L['s'][1]
    u = np.hypot(rx - lx, ry - ly); ang = np.arctan2(ry - ly, rx - lx); ex, ey = (lx + rx) / 2, (ly + ry) / 2
    top, bot, half, fe = shape['top'] * u, shape['bot'] * u, shape['half'] * u, shape['feather'] * u
    x0, x1 = int(ex - half - fe - 4), int(ex + half + fe + 4); y0, y1 = int(ey - top - fe - 4), int(ey + bot + fe + 4)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    # face-local coords (rotate by -ang about eye midpoint)
    lxp = (xx - ex) * np.cos(-ang) - (yy - ey) * np.sin(-ang); lyp = (xx - ex) * np.sin(-ang) + (yy - ey) * np.cos(-ang)
    # super-ellipse mask: top half uses 'top', bottom half 'bot'
    ryy = np.where(lyp < 0, top, bot); hb = shape.get('half_bot', shape['half']) * u
    # width eases from 'half' at the eye line to 'half_bot' at the bottom edge
    hx = np.where(lyp < 0, half, half + (hb - half) * np.clip(lyp / bot, 0, 1))
    e = (np.abs(lxp / hx) ** shape.get('p', 2) + np.abs(lyp / ryy) ** shape.get('p', 2)) ** (1 / shape.get('p', 2))
    inside = (1 - e) * np.minimum(hx, ryy)
    alpha = np.clip(inside / fe, 0, 1); alpha = alpha * alpha * (3 - 2 * alpha)
    # sprite -> ref tile -> expression tile
    if shape.get('affine'):
        # direct least-squares affine from expression-tile landmarks to sprite landmarks (for turned heads)
        src = np.array([q for q, _ in shape['affine']], float); dst = np.array([d for _, d in shape['affine']], float)
        A = np.linalg.lstsq(np.c_[dst, np.ones(len(dst))], src, rcond=None)[0]   # sprite -> tile
        qx = A[0, 0] * xx + A[1, 0] * yy + A[2, 0]; qy = A[0, 1] * xx + A[1, 1] * yy + A[2, 1]
    else:
        P = xx + 1j * yy; p = (P - b) / a
        px, py = p.real, p.imag
        v = np.stack([px + dx - c1[0], py + dy - c1[1]])
        qx = c1[0] + Mi[0, 0] * v[0] + Mi[0, 1] * v[1]; qy = c1[1] + Mi[1, 0] * v[0] + Mi[1, 1] * v[1]
    X4 = qx * 4 - 0.5 * 0 ; Y4 = qy * 4
    rgb = np.dstack([ndi.map_coordinates(tile[..., ch], [Y4, X4], order=3, mode='nearest') for ch in range(3)]).clip(0, 1)
    base = S[y0:y1, x0:x1, :3]
    if shape.get('nohair'):
        # near the sides, never paint the expression tile's dark hair over the sprite's skin
        edge = np.abs(lxp) > shape['nohair'] * u
        hairy = (rgb.mean(2) < shape.get('hair_lum', 0.36)) & (base.mean(2) > 0.55)
        kill = ndi.binary_dilation(edge & hairy, iterations=shape.get('grow', 3))
        alpha = ndi.gaussian_filter(np.where(kill, 0, alpha), 1.2)
    if color_fix:
        # match skin tone: compare in the feather ring where both are face pixels
        ring = (alpha > 0.15) & (alpha < 0.9)
        lum_b = base.mean(2); lum_p = rgb.mean(2)
        skin = ring & (lum_b > 0.72) & (lum_p > 0.72) & (base[..., 0] > base[..., 2])
        if skin.sum() > 50:
            g = base[skin].mean(0) / np.maximum(rgb[skin].mean(0), 1e-3)
            rgb = np.clip(rgb * g, 0, 1)
    out = np.dstack([rgb, alpha])
    Image.fromarray((out * 255 + 0.5).astype(np.uint8), 'RGBA').save(f'{out_prefix}.png')
    comp = S.copy(); comp[y0:y1, x0:x1, :3] = base * (1 - alpha[..., None]) + rgb * alpha[..., None]
    return (x0, y0, x1 - x0, y1 - y0), comp
if __name__ == '__main__':
    reg = json.load(open('diff/reg.json'))
    k = sys.argv[1]; shape = json.loads(sys.argv[2]); names = sys.argv[3:]
    import os; os.makedirs('diff/patch', exist_ok=True)
    over = shape.pop('per', {})
    for n in names:
        box, comp = build(k, n, reg, {**shape, **over.get(n, {})}, f'diff/patch/{k}_{n}')
        Image.fromarray((comp * 255 + 0.5).astype(np.uint8), 'RGBA').save(f'diff/patch/{k}_{n}_comp.png')
        print(k, n, box)
