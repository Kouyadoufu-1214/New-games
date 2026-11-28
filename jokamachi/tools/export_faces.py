# export face patches (cropped to their alpha) as WebP + a JS data block HERO_FACE
import json, sys, os, base64, numpy as np
from PIL import Image
sys.path.insert(0, 'scripts'); from compose import build
reg = json.load(open('diff/reg.json'))
SETS = {'koharu': ['happy', 'blush', 'angry', 'sad', 'surprise'], 'yukino': ['happy', 'blush', 'angry', 'sad', 'surprise'],
        'chizuru': ['happy', 'blush', 'angry', 'sad', 'surprise'], 'kaede': ['m_normal', 'm_happy', 'm_blush', 'm_angry', 'm_sad', 'm_surprise', 'u_happy']}
q = int(sys.argv[1]) if len(sys.argv) > 1 else 80
os.makedirs('final/faces', exist_ok=True)
data, total = {}, 0
for k, names in SETS.items():
    shape = json.load(open(f'diff/shape_{k}.json')); over = shape.pop('per', {})
    data[k] = {}
    for n in names:
        (x0, y0, w, h), _ = build(k, n, reg, {**shape, **over.get(n, {})}, f'diff/patch/{k}_{n}')
        im = Image.open(f'diff/patch/{k}_{n}.png'); a = np.asarray(im)[..., 3]
        ys, xs = np.where(a > 2); bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        im = im.crop((bx0, by0, bx1, by1)); p = f'final/faces/{k}_{n}.webp'
        im.save(p, quality=q, method=6, alpha_quality=80); sz = os.path.getsize(p); total += sz
        data[k][n] = [int(x0 + bx0), int(y0 + by0), int(bx1 - bx0), int(by1 - by0), base64.b64encode(open(p, 'rb').read()).decode()]
        print(k, n, data[k][n][:4], sz // 1024, 'KB')
print('total', total // 1024, 'KB')
lines = []
for k, d in data.items():
    lines.append(f'{k}: {{')
    for n, (x, y, w, h, b) in d.items(): lines.append(f"{n}: [{x}, {y}, {w}, {h}, '{b}'],")
    lines.append('},')
open('final/hero_face_b64.js', 'w').write('<script>\n/* heroine expression faces: patches laid over the sprite face, [x, y, w, h, WebP base64] in sprite pixels (sources in art/faces/) */\nconst HERO_FACE = {\n' + '\n'.join(lines) + '\n};\n</script>\n')
