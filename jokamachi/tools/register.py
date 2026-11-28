# Register every expression tile onto its heroine's reference tile using hair + face outline only.
# Similarity search (scale, rotation) x FFT masked NCC (translation) on gradient images at 2x tile scale.
import json, sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi

S = 2  # work at 2x tile scale (from the x4 upscales)
def load(k, n):
    im = Image.open(f'diff/tiles/{k}_{n}_x4.png').convert('L')
    im = im.resize((im.width * S // 4, im.height * S // 4), Image.LANCZOS)
    return np.asarray(im).astype(np.float64) / 255
def grads(a):
    a = ndi.gaussian_filter(a, 1.0)
    return ndi.sobel(a, 1), ndi.sobel(a, 0)
def warp(a, s, th, c):
    # rotate/scale about centre c (x, y): output(p) = input(c + R^-1 (p - c) / s)
    t = np.deg2rad(th); R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    Mi = np.linalg.inv(R * s)                              # maps out->in in (x,y)
    cy, cx = c[1], c[0]
    M = np.array([[Mi[1, 1], Mi[1, 0]], [Mi[0, 1], Mi[0, 0]]])  # (row, col) order
    off = np.array([cy, cx]) - M @ np.array([cy, cx])
    return ndi.affine_transform(a, M, offset=off, order=1, mode='nearest')
def fcorr(a, b):  # cross-correlation, result[d] = sum a(x) b(x + d), same size, circular
    return np.fft.ifft2(np.conj(np.fft.fft2(a)) * np.fft.fft2(b)).real
def register(ref, tgt, mask, rng=28):
    rx, ry = grads(ref); rx, ry = rx * mask, ry * mask
    den_r = np.sqrt((rx ** 2 + ry ** 2).sum())
    h, w = ref.shape; c = (w / 2, h / 2)
    best = (-1, None)
    for s in np.arange(0.92, 1.081, 0.01):
        for th in np.arange(-4, 4.01, 1.0):
            tw = warp(tgt, s, th, c); tx, ty = grads(tw)
            num = fcorr(rx, tx) + fcorr(ry, ty)
            den_t = np.sqrt(np.maximum(fcorr(mask, tx ** 2 + ty ** 2), 1e-9))
            ncc = num / (den_r * den_t)
            ncc = np.fft.fftshift(ncc); cy, cx = h // 2, w // 2
            win = ncc[cy - rng:cy + rng + 1, cx - rng:cx + rng + 1]
            y, x = np.unravel_index(win.argmax(), win.shape)
            if win[y, x] > best[0]: best = (win[y, x], (float(s), float(th), x - rng, y - rng))
    return best
if __name__ == '__main__':
    spec = json.load(open(sys.argv[1]))   # {heroine: {"ref": name, "names": [...], "hole": [cx, cy, rx, ry] in tile px}}
    out = {}
    for k, d in spec.items():
        ref = load(k, d['ref']); h, w = ref.shape
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy, ax, ay = [v * S for v in d['hole']]
        mask = (((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 > 1).astype(float)
        mask[:6 * S] = 0; mask[-6 * S:] = 0; mask[:, :6 * S] = 0; mask[:, -6 * S:] = 0
        out[k] = {d['ref']: [1.0, 0.0, 0.0, 0.0, 1.0]}
        for n in d['names']:
            tgt = load(k, n)
            if tgt.shape != ref.shape: tgt = np.asarray(Image.fromarray((tgt * 255).astype(np.uint8)).resize((w, h))).astype(float) / 255
            sc, (s, th, dx, dy) = register(ref, tgt, mask)
            # dx, dy: shift (in 2x px) such that ref(x) ~ warped_tgt(x + d)
            out[k][n] = [s, th, dx / S, dy / S, round(float(sc), 3)]
            print(k, n, out[k][n], flush=True)
    json.dump(out, open(sys.argv[2], 'w'), indent=1)
