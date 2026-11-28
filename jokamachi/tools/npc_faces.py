# Expression face patches for story NPCs: register the sheet's reference tile to the individual portrait,
# register each expression tile to the reference tile, then feather the expression's face onto the portrait.
import sys, os, json, numpy as np, onnxruntime as ort
from PIL import Image
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(__file__))
from register import grads, warp, fcorr

B = '/tmp/claude-0/-home-user-New-games/e10e8c9b-88e9-5bdd-8465-a555008c7672/scratchpad/'
ESR = None
def upscale(rgb):
    global ESR
    if ESR is None: ESR = ort.InferenceSession(B + 'models/anime6b.onnx', providers=['CPUExecutionProvider'])
    a = np.asarray(rgb).astype(np.float32) / 255
    y = ESR.run(None, {'input': a.transpose(2, 0, 1)[None]})[0][0]
    return np.clip(y.transpose(1, 2, 0), 0, 1)
def gray(a): return a[..., :3].mean(2) if a.ndim == 3 else a
def sim_from(p, q):  # similarity mapping points p -> q (complex least squares)
    s = np.array([complex(*v) for v in p]); d = np.array([complex(*v) for v in q])
    sm, dm = s.mean(), d.mean(); a = ((d - dm) * np.conj(s - sm)).sum() / (np.abs(s - sm) ** 2).sum()
    return a, dm - a * sm
def apply_inv_sim(a, b, x, y):  # given q = a p + b, return p for q = (x, y)
    z = ((x + 1j * y) - b) / a; return z.real, z.imag
def warp_sim(img, a, b, shape):  # output(q) = img(p) where q = a p + b
    h, w = shape; yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    px, py = apply_inv_sim(a, b, xx, yy)
    return ndi.map_coordinates(img, [py, px], order=1, mode='nearest')
def search(ref, tgt, mask, rng, scales=np.arange(0.94, 1.061, 0.01), rots=np.arange(-4, 4.01, 1.0)):
    rx, ry = grads(ref); rx, ry = rx * mask, ry * mask; den_r = np.sqrt((rx ** 2 + ry ** 2).sum())
    h, w = ref.shape; c = (w / 2, h / 2); best = (-1, None)
    for s in scales:
        for th in rots:
            tw = warp(tgt, s, th, c); tx, ty = grads(tw)
            ncc = (fcorr(rx, tx) + fcorr(ry, ty)) / (den_r * np.sqrt(np.maximum(fcorr(mask, tx ** 2 + ty ** 2), 1e-9)))
            ncc = np.fft.fftshift(ncc); cy, cx = h // 2, w // 2; win = ncc[cy - rng:cy + rng + 1, cx - rng:cx + rng + 1]
            y, x = np.unravel_index(win.argmax(), win.shape)
            if win[y, x] > best[0]: best = (float(win[y, x]), (float(s), float(th), int(x - rng), int(y - rng)))
    return best
def res_map(params, c, x, y):  # ref point -> warped-target point, per register's convention
    s, th, dx, dy = params; t = np.deg2rad(th); ct, st = np.cos(t), np.sin(t)
    u, v = x + dx - c[0], y + dy - c[1]
    # inverse rotation/scale (R^-1 / s)
    return c[0] + (ct * u + st * v) / s, c[1] + (-st * u + ct * v) / s
def face_mask(shape, eL, eR, top, bot, half, p=2.2, half_bot=None):
    h, w = shape; yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ex, ey = (eL[0] + eR[0]) / 2, (eL[1] + eR[1]) / 2; u = np.hypot(eR[0] - eL[0], eR[1] - eL[1]); ang = np.arctan2(eR[1] - eL[1], eR[0] - eL[0])
    lx = (xx - ex) * np.cos(-ang) - (yy - ey) * np.sin(-ang); ly = (xx - ex) * np.sin(-ang) + (yy - ey) * np.cos(-ang)
    ryy = np.where(ly < 0, top * u, bot * u); hb = (half_bot if half_bot is not None else half) * u
    hx = np.where(ly < 0, half * u, half * u + (hb - half * u) * np.clip(ly / (bot * u), 0, 1))
    e = (np.abs(lx / hx) ** p + np.abs(ly / ryy) ** p) ** (1 / p)
    return e, np.minimum(hx, ryy)
if __name__ == '__main__':
    spec = json.load(open(sys.argv[1])); out_dir = B + 'npc/faces/'; os.makedirs(out_dir, exist_ok=True)
    eyes_pct = json.load(open(B + 'npc/eyes.json')); report = {}
    for key, d in spec.items():
        ind = np.asarray(Image.open(B + f'npc/{key}.png').convert('RGBA')).astype(np.float64) / 255
        H, W = ind.shape[:2]; xl, xr, ye = eyes_pct[key]
        eL, eR = (xl / 100 * W, ye / 100 * H), (xr / 100 * W, ye / 100 * H)
        if 'eyes_px' in d: eL, eR = tuple(d['eyes_px'][0]), tuple(d['eyes_px'][1])
        u_ind = np.hypot(eR[0] - eL[0], eR[1] - eL[1]); sw = 110 / u_ind
        indw = ind[..., :3] * ind[..., 3:] + (1 - ind[..., 3:])  # on white
        Iw = np.asarray(Image.fromarray((gray(indw) * 255).astype(np.uint8)).resize((round(W * sw), round(H * sw)), Image.LANCZOS)).astype(np.float64) / 255
        sheet = Image.open(B + d['sheet']).convert('RGB')
        def tile(box): return np.asarray(sheet.crop(tuple(box))).astype(np.float64) / 255
        Tn = tile(d['ref']); tn_e = d['ref_eyes']
        # 1) reference tile -> individual (working frame): eyes first, then refine on the hair/outline
        a0, b0 = sim_from(tn_e, [(eL[0] * sw, eL[1] * sw), (eR[0] * sw, eR[1] * sw)])
        Tw = warp_sim(gray(Tn), a0, b0, Iw.shape)
        e, _ = face_mask(Iw.shape, (eL[0] * sw, eL[1] * sw), (eR[0] * sw, eR[1] * sw), 0.45, 1.0, 0.75)
        m = (e > 1.0).astype(float); m *= (Iw < 0.97) | ndi.binary_dilation(Iw < 0.97, iterations=4)
        sc, p1 = search(Iw, Tw, m, 16); c1 = (Iw.shape[1] / 2, Iw.shape[0] / 2)
        report[key] = {'ref_ncc': round(sc, 3), 'p1': p1}; print(key, 'ref->ind', round(sc, 3), p1, flush=True)
        for ex, box in d['ex'].items():
            Tx = tile(box)
            # 2) expression tile -> reference tile (tile frame), hair/outline only
            hh, ww = Tn.shape[:2]
            if Tx.shape[:2] != (hh, ww): Tx = np.asarray(Image.fromarray((Tx * 255).astype(np.uint8)).resize((ww, hh))).astype(np.float64) / 255
            et, _ = face_mask((hh, ww), tn_e[0], tn_e[1], 0.5, 1.15, 0.85)
            mt = (et > 1.0).astype(float)
            sc2, p2 = search(gray(Tn), gray(Tx), mt, 30); c2 = (ww / 2, hh / 2)
            print(' ', ex, 'tile->ref', round(sc2, 3), p2, flush=True); report[key][ex] = {'ncc': round(sc2, 3), 'p2': p2}
            # 3) patch: individual px -> working -> (residual) -> warped ref tile -> ref tile -> expression tile -> x4 upscale
            P = d.get('patch', {}); top, bot, half = P.get('top', 0.55), P.get('bot', 1.0), P.get('half', 0.9); fe = P.get('feather', 0.16)
            e2, rad = face_mask((H, W), eL, eR, top, bot, half, P.get('p', 2.2), P.get('half_bot'))
            u = u_ind; inside = (1 - e2) * rad; alpha = np.clip(inside / (fe * u), 0, 1); alpha = alpha * alpha * (3 - 2 * alpha)
            ys, xs = np.where(alpha > 0); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
            wx, wy = xx * sw, yy * sw
            qx, qy = res_map(p1, c1, wx, wy)            # point in the eye-warped ref tile (working frame)
            rx_, ry_ = apply_inv_sim(a0, b0, qx, qy)     # point in the ref tile
            if d.get('vstretch'):                        # the sheet's lower face is shorter than the portrait's
                eyt = (tn_e[0][1] + tn_e[1][1]) / 2; k = d['vstretch']
                ry_ = np.where(ry_ > eyt, eyt + (ry_ - eyt) / k, ry_)
            tx_, ty_ = res_map(p2, c2, rx_, ry_)         # point in the expression tile
            if 'up' not in d: d['up'] = {}
            if ex not in d['up']: d['up'][ex] = upscale(Image.fromarray((Tx * 255).astype(np.uint8)))
            U = d['up'][ex]; k = U.shape[1] / Tx.shape[1]
            rgb = np.dstack([ndi.map_coordinates(U[..., ch], [ty_ * k, tx_ * k], order=3, mode='nearest') for ch in range(3)]).clip(0, 1)
            al = alpha[y0:y1, x0:x1] * ind[y0:y1, x0:x1, 3]
            vb = d.get('valid', [0, 0, Tx.shape[1], Tx.shape[0]]); fz = 14.0
            vmask = np.clip(np.minimum.reduce([tx_ - vb[0], ty_ - vb[1], vb[2] - tx_, vb[3] - ty_]) / fz, 0, 1)
            al = ndi.gaussian_filter(al * vmask * vmask * (3 - 2 * vmask), 1.0)
            base = ind[y0:y1, x0:x1, :3]
            ring = (al > 0.15) & (al < 0.85); lum_b, lum_p = base.mean(2), rgb.mean(2)
            skin = ring & (lum_b > 0.55) & (lum_p > 0.55) & (base[..., 0] > base[..., 2])
            if skin.sum() > 40 and P.get('color', True):
                g = base[skin].mean(0) / np.maximum(rgb[skin].mean(0), 1e-3); rgb = np.clip(rgb * np.clip(g, 0.85, 1.15), 0, 1)
            patch = np.dstack([rgb, al]); im = Image.fromarray((patch * 255 + 0.5).astype(np.uint8), 'RGBA')
            im.save(out_dir + f'{key}_{ex}.png')
            comp = ind.copy(); comp[y0:y1, x0:x1, :3] = base * (1 - al[..., None]) + rgb * al[..., None]
            Image.fromarray((comp * 255 + 0.5).astype(np.uint8), 'RGBA').save(out_dir + f'{key}_{ex}_comp.png')
            report[key][ex]['box'] = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
        d.pop('up', None)
    json.dump(report, open(out_dir + 'report.json', 'w'), indent=1, default=float)
