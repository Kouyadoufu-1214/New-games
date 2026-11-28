# Clean the model mask, decontaminate edge colours, crop to the figure and scale to a common figure height.
import sys, json, numpy as np
from PIL import Image
from scipy import ndimage as ndi
TARGET_H = int(sys.argv[1]) if len(sys.argv) > 1 else 1650
info = {}
for name in ['koharu', 'chizuru', 'yukino', 'kaede']:
    src = Image.open(f'art/{name}_src.webp').convert('RGB'); C = np.asarray(src).astype(np.float32) / 255
    a = np.asarray(Image.open(f'art/{name}_mask.png')).astype(np.float32) / 255
    # keep only the big connected blobs (drops stray petals / flowers)
    lab, n = ndi.label(a > 0.25)
    sizes = ndi.sum(np.ones_like(a), lab, range(1, n + 1))
    keep = np.zeros(n + 1, bool); keep[1:] = sizes > sizes.max() * 0.02
    blob = keep[lab]
    near = ndi.binary_dilation(blob, iterations=6)
    a = np.where(near, a, 0)
    # crisp up the soft model edge a little
    a = np.clip((a - 0.06) / 0.88, 0, 1)
    # background colour: median of clearly-background pixels, per row (bg has a gentle vertical gradient)
    bgm = a < 0.02
    B = np.zeros_like(C)
    rows = np.arange(C.shape[0])
    glob = np.median(C[bgm], axis=0)
    for y in range(C.shape[0]):
        m = bgm[y]
        B[y] = np.median(C[y][m], axis=0) if m.sum() > 50 else glob
    am = a[..., None]
    F = np.where(am > 0.03, np.clip((C - (1 - am) * B) / np.maximum(am, 1e-3), 0, 1), C)
    F = np.where(am > 0.97, C, F)
    rgba = np.dstack([F, a]); rgba = (rgba * 255 + 0.5).astype(np.uint8)
    ys, xs = np.where(a > 0.04)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = 12
    y0 = max(0, y0 - pad); x0 = max(0, x0 - pad); x1 = min(a.shape[1], x1 + pad); y1b = min(a.shape[0], y1 + pad)
    im = Image.fromarray(rgba, 'RGBA').crop((x0, y0, x1, y1b))
    s = TARGET_H / (y1 - y0)
    out = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    out.save(f'art/{name}.png')
    out.save(f'art/{name}.webp', quality=86, method=6, alpha_quality=90)
    import os
    info[name] = {'w': out.width, 'h': out.height, 'scale': s, 'crop': [int(x0), int(y0)], 'cutBottom': bool(y1 >= a.shape[0] - 2), 'kb': round(os.path.getsize(f'art/{name}.webp') / 1024)}
    print(name, info[name])
json.dump(info, open('art/info.json', 'w'), indent=1)
